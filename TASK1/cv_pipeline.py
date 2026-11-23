"""
cv_pipeline.py - Task 1: routes CVs into the sandbox taxonomy and builds S3 envelopes.

Usage:
    python cv_pipeline.py [synthetic_cvs.json] [sandbox_taxonomy.json] [outputs/parsed_cvs.json] [--llm] [--s3]

    Set CV_PIPELINE_S3_BUCKET to enable S3 output; CV_PIPELINE_S3_PREFIX defaults to "cv-pipeline".
    S3 credentials use the standard AWS credential chain.
Stages:
    1. Domain gate: score domains and job categories, choose primary and secondary.
    2. Scoped extraction: match taxonomy skills only in the locked domains, report
       out-of-scope matches separately, collect skill-like phrases the taxonomy lacks
       (rules first, then an optional LLM pass whose output is validated against the CV text),
       and send everything to other_skills_pending_approval for admin review.
    3. Envelope: wrap each CV in the S3 JSON shape. user_verified_profile starts as a copy,
       and diff_log records user edits so they can be exported as training pairs.

Each CV is processed on its own, so one bad CV goes to failed_cvs.json without stopping the batch.
"""

import hashlib
import json
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from importlib import import_module
from typing import Any, Callable, Literal, Optional, get_args
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, model_validator


Domain = Literal["Information Technology", "Marketing", "Consulting", "Finance"]
DOMAINS = get_args(Domain)

# Alternate spellings that resolve to a taxonomy synonym
ALIASES = {
    "TAM/SAM/SOM": "TAM SAM SOM Estimation",
    "Five Whys": "Five Whys Method",
}

# Job-title phrases that point to a taxonomy job title (each hit is worth 2 points)
TITLE_CUES = {
    "head of growth": "Growth Marketing Manager",
    "management consultant": "Strategy Consultant",
    "financial analyst": "Financial Planning Analyst",
    "data & backend engineer": "Data Engineer",
    "data engineer": "Data Engineer",
    "backend engineer": "Back End Engineer",
}

# Labels for lines that list skills, such as "Tools used: ..."
SKILL_LIST_LABELS = r"(tools used|core competencies|competencies|skills|tools|technologies|stack)"

# Lowercase tool names that the capitalization rules cannot detect
KNOWN_TOOLS = ["pandas", "numpy", "scikit-learn"]

# Acronyms that are roles, regions, or formats rather than skills
NOISE_ACRONYMS = {"CEO", "CFO", "CTO", "COO", "VP", "APAC", "EU", "UK", "US", "USA", "UAE", "MBA", "PDF", "JSON"}

# CamelCase tool names such as FastAPI, PostgreSQL, LangGraph
CAMEL_TOKEN = re.compile(r"(?<![A-Za-z0-9&])[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*(?![A-Za-z0-9&])")
# Acronyms such as SQL, SHAP, ML, B2B, and FP&A (single letters like R are ignored)
ACRONYM_TOKEN = re.compile(r"(?<![A-Za-z0-9&])[A-Z][A-Z0-9]+(?:&[A-Z]+)?(?![A-Za-z0-9&])")

LLM_PROMPT = (
    "Extract the technical and business skill phrases from the CV below.\n"
    "Rules:\n"
    "- Copy each phrase exactly as it appears in the CV.\n"
    "- Do not include job titles, companies, locations, dates, degrees, names, or certifications.\n"
    "- Do not invent phrases that are not in the CV.\n"
    "Return only a JSON array of strings, with no other text."
)


# Data models

class TaxonomyRow(BaseModel):
    """One row of the sandbox taxonomy, using the JSON key names."""
    model_config = ConfigDict(populate_by_name=True)
    domain: Domain = Field(alias="Domain")
    job_category: str = Field(alias="Job Category")
    job_title: str = Field(alias="Job Title")
    core_skill: str = Field(alias="Core Skill")
    sub_skill: str = Field(alias="Sub Skill")
    synonym: str = Field(alias="Synonym")


class SkillPath(BaseModel):
    """A CV skill mapped to its full taxonomy path."""
    domain: str
    job_category: str
    job_title: str
    core_skill: str
    sub_skill: str
    synonym: str
    matched_text: str  # the exact words found in the CV


class Stage1Routing(BaseModel):
    """The domain and job category chosen for a CV. Values are checked against the taxonomy later."""
    primary_domain: Domain
    primary_job_category: str
    secondary_domain: Optional[Domain] = None
    secondary_job_category: Optional[str] = None
    domain_scores: dict[str, int]
    tiebreak_applied: bool = False

    @model_validator(mode="after")
    def check_secondary(self):
        if (self.secondary_domain is None) != (self.secondary_job_category is None):
            raise ValueError("secondary domain and category must be set together")
        if self.secondary_domain == self.primary_domain:
            raise ValueError("secondary domain must differ from primary domain")
        return self


class ExtractedProfile(BaseModel):
    """Skills found in a CV, plus anything that needs admin review and extractor metadata."""
    skills: list[SkillPath] = Field(default_factory=list)
    other_skills_pending_approval: list[str] = Field(default_factory=list)
    out_of_scope_domain_matches: list[SkillPath] = Field(default_factory=list)
    extractor: str = "rules"            # "rules" or "rules+llm"
    llm_status: str = "not_run"         # "not_run", "ok", or "failed: <reason>"
    llm_rejected: list[dict] = Field(default_factory=list)  # phrases the validator refused, with reasons


class CVEnvelope(BaseModel):
    """The S3-ready record for one CV."""
    cv_id: str
    raw_text: str
    stage1_routing: Stage1Routing
    model_extracted_profile: ExtractedProfile
    user_verified_profile: ExtractedProfile  # starts as a copy; user edits go here
    diff_log: list[dict] = Field(default_factory=list)


# Text and matching helpers

def normalize(line: str) -> str:
    """Collapse runs of whitespace inside one line."""
    return re.sub(r"\s+", " ", line).strip()


def build_text(raw_text: str) -> tuple[str, list[str]]:
    """Clean each line and join them with newlines, so positions map back to lines."""
    lines = [normalize(line) for line in raw_text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines), lines


def find_terms(text: str, terms: dict, taken: list) -> list:
    """
    Find whole-phrase matches for each term, longest first.
    A character can belong to only one match. A letter, digit, or '&' next to a match
    blocks it, so "Go" does not match inside "Google" and "R" does not match "R&D".
    Returns (value, matched_text, position) tuples in text order.
    """
    hits = []
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9&]){re.escape(term)}(?![A-Za-z0-9&])", re.IGNORECASE
        )
        for match in pattern.finditer(text):
            span = range(match.start(), match.end())
            if any(taken[i] for i in span):
                continue
            for i in span:
                taken[i] = True
            hits.append((terms[term], match.group(0), match.start()))
    return sorted(hits, key=lambda h: h[2])


def build_term_index(taxonomy: list) -> dict:
    """Map every synonym and alias to its taxonomy row."""
    by_synonym = {row.synonym: row for row in taxonomy}
    terms = dict(by_synonym)
    for alias, target in ALIASES.items():
        terms[alias] = by_synonym[target]
    return terms


def validate_taxonomy(taxonomy: list) -> None:
    """Reject incomplete or ambiguous taxonomies before routing any CVs."""
    domains = {row.domain for row in taxonomy}
    if domains != set(DOMAINS):
        missing = sorted(set(DOMAINS) - domains)
        unexpected = sorted(domains - set(DOMAINS))
        raise ValueError(f"Taxonomy must contain exactly the allowed domains; missing={missing}, unexpected={unexpected}")

    synonym_counts = Counter(row.synonym.casefold() for row in taxonomy)
    duplicates = sorted(term for term, count in synonym_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Taxonomy synonyms must be unique (case-insensitive): {duplicates}")

    by_synonym = {row.synonym for row in taxonomy}
    missing_alias_targets = sorted(set(ALIASES.values()) - by_synonym)
    if missing_alias_targets:
        raise ValueError(f"Alias targets are missing from the taxonomy: {missing_alias_targets}")

    titles = {row.job_title for row in taxonomy}
    missing_cue_titles = sorted(set(TITLE_CUES.values()) - titles)
    if missing_cue_titles:
        raise ValueError(f"Title-cue targets are missing from the taxonomy: {missing_cue_titles}")


def build_cue_index(taxonomy: list) -> dict:
    """Map each title cue to a representative taxonomy row for that job title."""
    first_row_for_title = {}
    for row in taxonomy:
        first_row_for_title.setdefault(row.job_title, row)
    return {cue: first_row_for_title[title] for cue, title in TITLE_CUES.items()}


def tally(syn_hits: list, cue_hits: list, group_of) -> tuple[dict, dict]:
    """Each unique sub-skill is worth 1 point and each title cue is worth 2 points."""
    scores, first, seen = defaultdict(int), {}, set()
    for row, _, pos in syn_hits:
        group = group_of(row)
        if (group, row.sub_skill) not in seen:
            seen.add((group, row.sub_skill))
            scores[group] += 1
        first[group] = min(first.get(group, pos), pos)
    for row, _, pos in cue_hits:
        group = group_of(row)
        scores[group] += 2
        first[group] = min(first.get(group, pos), pos)
    return dict(scores), first


def rank(scores: dict, first: dict) -> list:
    """Highest score first. Ties go to whichever is mentioned earliest in the CV."""
    return sorted(scores, key=lambda g: (-scores[g], first[g]))


def best_category(domain: str, syn_hits: list, cue_hits: list) -> str:
    syn = [h for h in syn_hits if h[0].domain == domain]
    cues = [h for h in cue_hits if h[0].domain == domain]
    scores, first = tally(syn, cues, lambda r: r.job_category)
    return rank(scores, first)[0]


def skill_from_row(row: TaxonomyRow, matched: str) -> SkillPath:
    return SkillPath(
        domain=row.domain,
        job_category=row.job_category,
        job_title=row.job_title,
        core_skill=row.core_skill,
        sub_skill=row.sub_skill,
        synonym=row.synonym,
        matched_text=matched,
    )


def unique_skills(hits: list) -> list:
    """Keep one entry per (domain, sub-skill), so "TAM/SAM/SOM" and "TAM SAM SOM" count once."""
    out, seen = [], set()
    for row, matched, _ in hits:
        key = (row.domain, row.sub_skill)
        if key not in seen:
            seen.add(key)
            out.append(skill_from_row(row, matched))
    return out


# Stage 1: domain gate

def stage1_route(text: str, taxonomy: list) -> Stage1Routing:
    """Choose the primary and secondary domain and job category. Domains come from the taxonomy file."""
    validate_taxonomy(taxonomy)
    domains = list(DOMAINS)
    syn_hits = find_terms(text, build_term_index(taxonomy), [False] * len(text))
    cue_hits = find_terms(text, build_cue_index(taxonomy), [False] * len(text))

    dom_scores, dom_first = tally(syn_hits, cue_hits, lambda r: r.domain)
    ranked = rank(dom_scores, dom_first)
    if not ranked:
        raise ValueError("No taxonomy evidence found in this CV")

    primary = ranked[0]
    secondary = None
    # A second domain is a hybrid signal if it has at least half the primary's score
    if len(ranked) > 1 and dom_scores[ranked[1]] >= 0.5 * dom_scores[primary]:
        secondary = ranked[1]
    tie = len(ranked) > 1 and dom_scores[ranked[0]] == dom_scores[ranked[1]]

    return Stage1Routing(
        primary_domain=primary,
        primary_job_category=best_category(primary, syn_hits, cue_hits),
        secondary_domain=secondary,
        secondary_job_category=best_category(secondary, syn_hits, cue_hits) if secondary else None,
        domain_scores={d: dom_scores.get(d, 0) for d in domains},
        tiebreak_applied=tie,
    )


# Stage 2: scoped extraction, quarantine, pending detection, optional LLM pass

def analyze_lines(lines: list) -> tuple[list, list]:
    """
    Find parts of the CV that are not skills: section headers, contact and company lines,
    education, project titles, and labels. Also pulls items out of skill-list lines.
    Returns (ignored_spans, list_pieces) with offsets into "\\n".join(lines).
    """
    ignored, pieces = [], []
    offset, in_education = 0, False
    for idx, line in enumerate(lines):
        start, end = offset, offset + len(line)
        offset = end + 1

        if re.fullmatch(r"[A-Z][A-Z &/]+:?", line):  # section header
            in_education = line.upper().startswith("EDUCATION")
            ignored.append((start, end))
            continue
        if re.match(r"(?i)edu\b", line):
            in_education = True
        if idx == 0 or "|" in line or "@" in line or in_education:  # name, contact, company, education
            ignored.append((start, end))
            continue

        label = re.match(r"^[A-Z][A-Z &/]+:", line)
        if label:
            ignored.append((start, start + label.end()))
        project = re.match(r"^[^:]+\(\d{4}\):", line)
        if project:
            ignored.append((start, start + project.end()))

        skill_list = re.match(rf"^\s*(?:[-•]\s*)?{SKILL_LIST_LABELS}\s*:\s*(.+)$", line, re.IGNORECASE)
        if skill_list:
            group_base = start + skill_list.start(2)
            for piece in re.finditer(r"[^,()]+", skill_list.group(2)):
                raw = piece.group(0)
                phrase = raw.strip().rstrip(".")
                if not phrase:
                    continue
                lead = len(raw) - len(raw.lstrip())
                s = group_base + piece.start() + lead
                pieces.append((phrase, s, s + len(phrase)))
    return ignored, pieces


def call_llm_extractor(text: str, backend: Callable[[str], str]) -> list:
    """Ask the model for phrases. Raises if the reply is not a JSON array of strings."""
    raw = backend(f"{LLM_PROMPT}\n\nCV:\n{text}")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    data = json.loads(cleaned)
    if isinstance(data, dict) and len(data) == 1 and isinstance(next(iter(data.values())), list):
        data = next(iter(data.values()))  # some backends wrap the array in an object
    if not isinstance(data, list):
        raise ValueError("LLM output is not a JSON array")
    return data


def validate_llm_phrases(phrases: list, text: str, blocked: list) -> tuple[list, list]:
    """
    Accept a model phrase only if it appears in the CV text, is a sensible length,
    and does not overlap a taxonomy match or a non-skill section. Anything else is rejected with a reason.
    """
    accepted, rejected = [], []
    for item in phrases:
        if not isinstance(item, str):
            rejected.append({"phrase": repr(item), "reason": "not a string"})
            continue
        phrase = normalize(item)
        if not 2 <= len(phrase) <= 60:
            rejected.append({"phrase": phrase, "reason": "length outside 2-60 characters"})
            continue
        if phrase.upper() in NOISE_ACRONYMS:
            rejected.append({"phrase": phrase, "reason": "role, region, or format, not a skill"})
            continue
        pattern = re.compile(
            r"(?<![A-Za-z0-9&])" + r"\s+".join(re.escape(p) for p in phrase.split(" ")) + r"(?![A-Za-z0-9&])",
            re.IGNORECASE,
        )
        spans = [m.span() for m in pattern.finditer(text)]
        if not spans:
            rejected.append({"phrase": phrase, "reason": "not found in CV text (possible hallucination)"})
            continue
        free = [(s, e) for s, e in spans if not any(blocked[s:e])]
        if not free:
            rejected.append({"phrase": phrase, "reason": "overlaps a taxonomy match or a non-skill section"})
            continue
        s, e = free[0]
        accepted.append(text[s:e])
        for i in range(s, e):  # claim the span so the same text is not accepted twice
            blocked[i] = True
    return accepted, rejected


def merge_unique(first: list, second: list) -> list:
    """Combine two lists, keeping the first spelling of each item (case-insensitive)."""
    out, seen = [], set()
    for item in first + second:
        if item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


def stage2_extract(
    text: str,
    lines: list,
    taxonomy: list,
    locked: set,
    backend: Optional[Callable[[str], str]] = None,
) -> ExtractedProfile:
    """Match taxonomy skills in locked domains, quarantine the rest, and collect pending phrases."""
    terms = build_term_index(taxonomy)
    in_terms = {t: r for t, r in terms.items() if r.domain in locked}
    out_terms = {t: r for t, r in terms.items() if r.domain not in locked}

    taken = [False] * len(text)
    in_hits = find_terms(text, in_terms, taken)
    out_hits = find_terms(text, out_terms, taken)

    blocked = list(taken)
    ignored, pieces = analyze_lines(lines)
    for s, e in ignored:
        for i in range(s, e):
            blocked[i] = True

    # Rule-based pending detection
    candidates = []  # (position, phrase)
    for phrase, s, e in pieces:
        if not any(blocked[s:e]):
            candidates.append((s, phrase))
    for pattern in (CAMEL_TOKEN, ACRONYM_TOKEN):
        for m in pattern.finditer(text):
            if m.group(0) in NOISE_ACRONYMS or any(blocked[m.start():m.end()]):
                continue
            candidates.append((m.start(), m.group(0)))
    for tool in KNOWN_TOOLS:
        pattern = re.compile(rf"(?<![A-Za-z0-9&]){re.escape(tool)}(?![A-Za-z0-9&])", re.IGNORECASE)
        for m in pattern.finditer(text):
            if not any(blocked[m.start():m.end()]):
                candidates.append((m.start(), m.group(0)))
    candidates.sort()
    rule_pending = merge_unique([phrase for _, phrase in candidates], [])

    # Optional LLM pass for free-text phrases. Its output is only accepted after validation.
    pending, llm_status, llm_rejected, extractor = rule_pending, "not_run", [], "rules"
    if backend is not None:
        try:
            proposed = call_llm_extractor(text, backend)
        except Exception as exc:  # network down, bad JSON, or any other backend failure
            raise RuntimeError(f"LLM extraction failed: {type(exc).__name__}: {exc}") from exc
        accepted, llm_rejected = validate_llm_phrases(proposed, text, blocked)
        pending = merge_unique(rule_pending, accepted)
        llm_status, extractor = "ok", "rules+llm"

    return ExtractedProfile(
        skills=unique_skills(in_hits),
        other_skills_pending_approval=pending,
        out_of_scope_domain_matches=unique_skills(out_hits),
        extractor=extractor,
        llm_status=llm_status,
        llm_rejected=llm_rejected,
    )


# Validation: every output must trace back to the taxonomy

def assert_routing_valid(routing: Stage1Routing, taxonomy: list) -> None:
    valid = {(r.domain, r.job_category) for r in taxonomy}
    if (routing.primary_domain, routing.primary_job_category) not in valid:
        raise ValueError(f"Primary route not in taxonomy: {routing.primary_domain} / {routing.primary_job_category}")
    if routing.secondary_domain and (routing.secondary_domain, routing.secondary_job_category) not in valid:
        raise ValueError(f"Secondary route not in taxonomy: {routing.secondary_domain} / {routing.secondary_job_category}")


def assert_in_taxonomy(profile: ExtractedProfile, taxonomy: list) -> None:
    valid = {(r.domain, r.job_category, r.job_title, r.core_skill, r.sub_skill, r.synonym) for r in taxonomy}
    for s in profile.skills + profile.out_of_scope_domain_matches:
        key = (s.domain, s.job_category, s.job_title, s.core_skill, s.sub_skill, s.synonym)
        if key not in valid:
            raise ValueError(f"Path not in taxonomy: {key}")


def assert_pending_from_cv(profile: ExtractedProfile, text: str) -> None:
    """Pending phrases must be present in the source CV, never invented by extraction."""
    for phrase in profile.other_skills_pending_approval:
        normalized = normalize(phrase)
        pattern = re.compile(
            r"(?<![A-Za-z0-9&])"
            + r"\s+".join(re.escape(part) for part in normalized.split(" "))
            + r"(?![A-Za-z0-9&])",
            re.IGNORECASE,
        )
        if not pattern.search(text):
            raise ValueError(f"Pending skill is not present in the CV: {phrase!r}")


# Stage 3: envelope, diff log, and training pairs

def process_cv(cv: dict, taxonomy: list, backend: Optional[Callable[[str], str]] = None) -> CVEnvelope:
    """Run all three stages on one CV and return its envelope."""
    text, lines = build_text(cv["raw_text"])
    routing = stage1_route(text, taxonomy)

    locked = {routing.primary_domain}
    if routing.secondary_domain:
        locked.add(routing.secondary_domain)
    profile = stage2_extract(text, lines, taxonomy, locked, backend)

    assert_routing_valid(routing, taxonomy)
    assert_in_taxonomy(profile, taxonomy)
    assert_pending_from_cv(profile, text)

    return CVEnvelope(
        cv_id=cv["cv_id"],
        raw_text=cv["raw_text"],
        stage1_routing=routing,
        model_extracted_profile=profile,
        user_verified_profile=profile.model_copy(deep=True),
    )


def process_batch(cvs: list, taxonomy: list, backend=None) -> tuple[list, list]:
    """Process every CV. A failing CV is recorded and does not stop the others."""
    envelopes, failures = [], []
    for cv in cvs:
        try:
            envelopes.append(process_cv(cv, taxonomy, backend))
        except Exception as exc:
            failures.append({"cv_id": cv.get("cv_id", "UNKNOWN"), "error": f"{type(exc).__name__}: {exc}"})
    return envelopes, failures


def diff_profiles(model: ExtractedProfile, user: ExtractedProfile) -> list[dict]:
    """
    List every change between what the model extracted and what the user confirmed.
    Each change is {"field", "op": "added" | "removed", "item"}. Output is sorted, so it is stable.
    """
    changes = []

    def compare_skills(field_name: str, m_list: list, u_list: list) -> None:
        key = lambda s: (
            s.domain,
            s.job_category,
            s.job_title,
            s.core_skill,
            s.sub_skill,
            s.synonym,
            s.matched_text,
        )
        m = {key(s): s for s in m_list}
        u = {key(s): s for s in u_list}
        for k in sorted(set(m) - set(u)):
            changes.append({"field": field_name, "op": "removed", "item": m[k].model_dump()})
        for k in sorted(set(u) - set(m)):
            changes.append({"field": field_name, "op": "added", "item": u[k].model_dump()})

    compare_skills("skills", model.skills, user.skills)
    compare_skills("out_of_scope_domain_matches", model.out_of_scope_domain_matches, user.out_of_scope_domain_matches)

    m_pending = {p: p for p in model.other_skills_pending_approval}
    u_pending = {p: p for p in user.other_skills_pending_approval}
    for k in sorted(set(m_pending) - set(u_pending)):
        changes.append({"field": "other_skills_pending_approval", "op": "removed", "item": m_pending[k]})
    for k in sorted(set(u_pending) - set(m_pending)):
        changes.append({"field": "other_skills_pending_approval", "op": "added", "item": u_pending[k]})
    return changes


def attach_diff(envelope: CVEnvelope) -> CVEnvelope:
    """Refresh diff_log from the current model and user profiles."""
    envelope.diff_log = diff_profiles(envelope.model_extracted_profile, envelope.user_verified_profile)
    return envelope


def export_training_pairs(envelope: CVEnvelope) -> list[dict]:
    """A training pair exists only when a user actually changed something."""
    if not envelope.diff_log:
        return []
    return [{
        "cv_id": envelope.cv_id,
        "input_text": envelope.raw_text,
        "model_output": envelope.model_extracted_profile.model_dump(),
        "user_output": envelope.user_verified_profile.model_dump(),
        "changes": envelope.diff_log,
    }]


def update_user_verification(
    envelope: CVEnvelope,
    user_profile: ExtractedProfile | dict,
    taxonomy: list,
) -> CVEnvelope:
    """Apply a user-approved profile, validate its taxonomy paths, and refresh its diff."""
    verified = (
        user_profile
        if isinstance(user_profile, ExtractedProfile)
        else ExtractedProfile.model_validate(user_profile)
    )
    assert_in_taxonomy(verified, taxonomy)
    envelope.user_verified_profile = verified.model_copy(deep=True)
    return attach_diff(envelope)



class S3FlywheelStore:
    """Persist CV envelopes and immutable training pairs in S3."""

    def __init__(
        self,
        bucket: str,
        taxonomy: list,
        prefix: str = "cv-pipeline",
        client: Any = None,
    ) -> None:
        if not bucket.strip():
            raise ValueError("An S3 bucket name is required")
        validate_taxonomy(taxonomy)
        self.bucket = bucket
        self.taxonomy = taxonomy
        self.prefix = prefix.strip("/")
        if client is None:
            try:
                boto3 = import_module("boto3")
            except ImportError as exc:
                raise RuntimeError("S3 support requires boto3; install it before using --s3") from exc
            client = boto3.client("s3")
        self.client = client

    def _key(self, suffix: str) -> str:
        return f"{self.prefix}/{suffix}" if self.prefix else suffix

    def put_envelope(self, envelope: CVEnvelope) -> None:
        assert_routing_valid(envelope.stage1_routing, self.taxonomy)
        assert_in_taxonomy(envelope.model_extracted_profile, self.taxonomy)
        assert_in_taxonomy(envelope.user_verified_profile, self.taxonomy)
        key = self._key(f"cvs/{quote(envelope.cv_id, safe='')}.json")
        body = json.dumps(envelope.model_dump(), ensure_ascii=False, indent=2).encode("utf-8")
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )

    def load_envelope(self, cv_id: str) -> CVEnvelope:
        key = self._key(f"cvs/{quote(cv_id, safe='')}.json")
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"].read()
        return CVEnvelope.model_validate(json.loads(body.decode("utf-8")))

    def put_training_pair(self, pair: dict) -> str:
        serialized = json.dumps(pair, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        cv_id = quote(str(pair["cv_id"]), safe="")
        key = self._key(f"training-pairs/{cv_id}/{digest}.json")
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(pair, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )
        return key

    def record_user_verification(self, cv_id: str, user_profile: ExtractedProfile | dict) -> CVEnvelope:
        """Persist a Bizyl review and create an idempotent S3 training-pair object."""
        envelope = self.load_envelope(cv_id)
        envelope = update_user_verification(envelope, user_profile, self.taxonomy)
        pairs = export_training_pairs(envelope)
        if pairs:
            self.put_training_pair(pairs[0])
        self.put_envelope(envelope)
        return envelope



# Backend

def ollama_backend(prompt: str) -> str:
    """Send the prompt to a local Ollama server and return the reply text."""
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
    body = json.dumps({
        "model": os.environ.get("OLLAMA_MODEL", "llama3.2"),
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"]



# Self-checks, run on every execution

def run_collision_checks() -> None:
    """Short tokens must not match inside longer words."""
    cases = [
        ("Google Ads Smart Bidding", ["Go"], []),
        ("Led R&D partner due diligence", ["R"], []),
        ("Used R for statistics", ["R"], ["R"]),
        ("Built Django services", ["Django"], ["Django"]),
    ]
    for text, terms, expected in cases:
        hits = find_terms(text, {t: t for t in terms}, [False] * len(text))
        got = [name for name, _, _ in hits]
        assert got == expected, f"collision check failed for {text!r}: got {got}"


def run_gap_checks(taxonomy: list) -> None:
    """Generic pending detection and noise filters."""
    sample = (
        "JANE DOE | Pune | jane@example.com\n"
        "Tools used: FastAPI, Pydantic.\n"
        "Led R&D reviews and CFO briefings. Built SQL reports for ML teams."
    )
    text, lines = build_text(sample)
    profile = stage2_extract(text, lines, taxonomy, locked={"Information Technology"})
    names = {p.lower() for p in profile.other_skills_pending_approval}
    assert {"fastapi", "pydantic", "sql", "ml"} <= names, f"missing pending items: {names}"
    assert not ({"r&d", "cfo", "jane doe", "tools used"} & names), f"noise leaked: {names}"



# Command line

if __name__ == "__main__":
    use_llm = "--llm" in sys.argv
    use_s3 = "--s3" in sys.argv
    args = [a for a in sys.argv[1:] if a not in {"--llm", "--s3"}]
    cvs_path = args[0] if len(args) > 0 else "synthetic_cvs.json"
    tax_path = args[1] if len(args) > 1 else "sandbox_taxonomy.json"
    out_path = args[2] if len(args) > 2 else "outputs/parsed_cvs.json"

    run_collision_checks()
    with open(tax_path, encoding="utf-8") as f:
        taxonomy = [TaxonomyRow.model_validate(r) for r in json.load(f)]
    validate_taxonomy(taxonomy)
    run_gap_checks(taxonomy)

    with open(cvs_path, encoding="utf-8") as f:
        cvs = json.load(f)

    backend = ollama_backend if use_llm else None
    envelopes, failures = process_batch(cvs, taxonomy, backend)

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in envelopes], f, indent=2, ensure_ascii=False)
    failed_path = os.path.join(out_dir, "failed_cvs.json")
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)

    if use_s3:
        store = S3FlywheelStore(
            bucket=os.environ.get("CV_PIPELINE_S3_BUCKET", ""),
            prefix=os.environ.get("CV_PIPELINE_S3_PREFIX", "cv-pipeline"),
            taxonomy=taxonomy,
        )
        for env in envelopes:
            store.put_envelope(env)
        print(f"Uploaded {len(envelopes)} envelopes to s3://{store.bucket}/{store.prefix}/cvs/")

    for env in envelopes:
        r = env.stage1_routing
        p = env.model_extracted_profile
        print(
            f"{env.cv_id}: primary={r.primary_domain} / {r.primary_job_category}"
            f" | secondary={r.secondary_domain}"
            f" | skills={len(p.skills)}"
            f" | pending={len(p.other_skills_pending_approval)}"
            f" | out_of_scope={len(p.out_of_scope_domain_matches)}"
            f" | extractor={p.extractor} ({p.llm_status})"
        )
    print(f"Failed CVs: {len(failures)}")
    print(f"Wrote {out_path} and {failed_path}")