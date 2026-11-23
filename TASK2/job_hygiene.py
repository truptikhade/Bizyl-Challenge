"""
job_hygiene.py - deterministic cleaning for Bizyl job board records.

Usage:
    python job_hygiene.py [dirty_jobs.json] [outputs/cleaned_jobs.json]

Stages:
    1. Text cleaning: decode HTML entities and normalize whitespace.
    2. Activity check: detect closure phrases and quarantine inactive listings.
    3. Filter mapping: normalize employment type, remote scope, domain, and seniority.
    4. Validation: verify mapped values against the frontend taxonomy.
    5. Output: write cleaned JSON records and report active/inactive totals.
"""

import html
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional


# The allowed values the Bizyl filters accept
EMPLOYMENT_TYPES = {"Full-time", "Part-time", "Contract", "Internship"}
REMOTE_SCOPES = {"Remote (Global)", "Remote (Restricted)", "Hybrid", "On-site"}
DOMAINS = {"Information Technology", "Marketing", "Consulting", "Finance"}
SENIORITY_LEVELS = {"Intern/Entry", "Mid-Level", "Senior", "Lead/Executive"}


# Patterns for clean dirty job records

# Phrases that mean the job is closed or the page is gone
INACTIVE_PATTERNS = [
    r"no longer available",       # job was taken down or expired
    r"position has been closed",  # hiring is finished
    r"has been filled",           # someone was hired
    r"no longer accepting",       # applications are shut
    r"404 not found",             # the page itself is missing
    r"requisition .{0,40}does not exist",  # the job ID is no longer valid
]

# Maps messy hint text to our employment types
EMPLOYMENT_MAP = {
    "full time": "Full-time",
    "permanent": "Full-time",  # we assume permanent means full-time
    "part time": "Part-time",
    "contract": "Contract",
    "contractor": "Contract",
    "internship": "Internship",
    "intern": "Internship",
}

# Phrases in a job snippet that limit who can apply
RESTRICTION_PATTERNS = [
    r"must reside in [^.]+",       # must live in a certain country
    r"eligible to work in [^.]+",  # must have work rights in a country
    r"only (?:in|for|open to) [^.]+",  # limited to a region or group
]

# Words that point to each domain
DOMAIN_KEYWORDS = {
    "Information Technology": [
        "backend", "back end", "engineer", "engineering", "python", "django",
        "pyspark", "data pipeline", "ml", "machine learning", "nlp",
    ],
    "Marketing": [
        "growth", "paid search", "paid social", "meta ads", "marketing", "ads",
        "acquisition", "audience", "audiences", "d2c", "conversions api",
    ],
    "Consulting": [
        "strategy consultant", "strategy", "market sizing", "tam/sam/som",
        "porter", "consulting", "consultant",
    ],
    "Finance": [
        "fp&a", "financial", "financial models", "dcf", "wacc", "3-statement",
        "run-rate", "unit economics", "valuation", "finance",
    ],
}

# Title patterns for seniority levels
SENIOR_TITLE = re.compile(r"\bsenior\b")
INTERN_TITLE = re.compile(r"\bintern")
LEAD_TITLE = re.compile(r"\b(lead|head|director|vp|chief)\b")

# Patterns that pull years of experience from the snippet
YEARS_RANGE = re.compile(r"(\d+)\s*(?:-|–|to)\s*(\d+)\s*years?")
YEARS_MIN = re.compile(r"(\d+)\+?\s*years?")



# The shape of one cleaned job record
@dataclass
class CleanedJob:
    job_id: str
    title: str
    raw_location: str
    clean_snippet: str
    is_active: bool
    inactive_reason: Optional[str]
    employment_type: Optional[str]
    remote_scope: Optional[str]
    mapped_domain: Optional[str]
    seniority: Optional[str]
    flags: list = field(default_factory=list)


# Cleaning steps

def clean_text(text: str) -> str:
    """Decode HTML entities and collapse extra spaces."""
    text = html.unescape(text or "")  # turns &amp; into &
    return re.sub(r"\s+", " ", text).strip()  # removes repeated spaces

def detect_inactive(snippet: str) -> Optional[str]:
    """Return the closure phrase found, or None if the job looks open."""
    low = snippet.lower()  # match without worrying about case
    for pattern in INACTIVE_PATTERNS:
        match = re.search(pattern, low)
        if match:
            return match.group(0)
    return None

def map_employment(hint: Optional[str], flags: list) -> Optional[str]:
    """Convert the employment hint into one of our four types."""
    key = re.sub(r"[\s_-]+", " ", (hint or "").lower()).strip()  # tidy the text first
    result = EMPLOYMENT_MAP.get(key)
    if key == "permanent":
        flags.append("employment_assumed: 'Permanent' mapped to Full-time")  # record our guess
    if result is None:
        flags.append(f"employment_unmapped: '{hint}'")  # we could not map it
    return result

def map_remote_scope(location: str, snippet: str, flags: list) -> Optional[str]:
    """Work out the remote scope from the location, then double-check the snippet."""
    loc = location.lower()
    if "on-site" in loc or "onsite" in loc:
        scope = "On-site"
    elif "hybrid" in loc:
        scope = "Hybrid"
    elif "remote" in loc:
        is_global = re.search(r"\b(anywhere|global|worldwide)\b", loc)
        scope = "Remote (Global)" if is_global else "Remote (Restricted)"
    else:
        flags.append(f"scope_unparsed: '{location}'")  # no clear location clue
        return None

    # If the location says global but the snippet restricts applicants, downgrade it
    if scope == "Remote (Global)":
        for pattern in RESTRICTION_PATTERNS:
            match = re.search(pattern, snippet, flags=re.IGNORECASE)
            if match:
                flags.append(
                    f"contradiction: location says global but snippet says '{match.group(0)}'"
                )
                scope = "Remote (Restricted)"
                break
    return scope

def map_domain(title: str, snippet: str, flags: list) -> Optional[str]:
    """Pick the domain with the most matching keywords, or leave it empty if unclear."""
    text = f"{title} {snippet}".lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(
            1
            for kw in keywords
            # match whole words only, so "ml" does not match inside "html"
            if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", text)
        )
    best = max(scores.values())
    winners = [d for d, s in scores.items() if s == best]  # domains tied for the top score
    if best == 0 or len(winners) > 1:
        flags.append(f"domain_ambiguous: scores={scores}")  # too unclear to pick one
        return None
    return winners[0]

def map_seniority(title: str, snippet: str, flags: list) -> str:
    """Use the title first, then the years of experience, then a default."""
    t = title.lower()
    if SENIOR_TITLE.search(t):
        return "Senior"
    if INTERN_TITLE.search(t):
        return "Intern/Entry"
    if LEAD_TITLE.search(t):
        return "Lead/Executive"

    s = snippet.lower()
    range_match = YEARS_RANGE.search(s)  # for example "2-4 years"
    min_match = YEARS_MIN.search(s)      # for example "5+ years"
    if range_match:
        years = int(range_match.group(1))
    elif min_match:
        years = int(min_match.group(1))
    else:
        flags.append("seniority_defaulted: no title or experience cue, set to Mid-Level")
        return "Mid-Level"

    if years >= 5:
        return "Senior"
    if years >= 2:
        return "Mid-Level"
    return "Intern/Entry"



# Runs all the steps on one record

def clean_job(record: dict) -> CleanedJob:
    """Clean one job record and return the result."""
    flags: list = []
    title = clean_text(record.get("title", ""))
    location = clean_text(record.get("raw_location", ""))
    snippet = clean_text(record.get("raw_page_snippet", ""))

    closure = detect_inactive(snippet)  # check whether the job is closed
    job = CleanedJob(
        job_id=record["job_id"],
        title=title,
        raw_location=location,
        clean_snippet=snippet,
        is_active=closure is None,
        inactive_reason=f"snippet matched '{closure}'" if closure else None,
        employment_type=map_employment(record.get("employment_hint"), flags),
        remote_scope=None,
        mapped_domain=None,
        seniority=None,
    )

    # Only open jobs get the filter values
    if job.is_active:
        job.remote_scope = map_remote_scope(location, snippet, flags)
        job.mapped_domain = map_domain(title, snippet, flags)
        job.seniority = map_seniority(title, snippet, flags)
    else:
        flags.append(f"listing_inactive: {job.inactive_reason}; UI mapping skipped")

    # Stop the program if any value falls outside the allowed list
    assert job.employment_type in EMPLOYMENT_TYPES | {None}, job.employment_type
    assert job.remote_scope in REMOTE_SCOPES | {None}, job.remote_scope
    assert job.mapped_domain in DOMAINS | {None}, job.mapped_domain
    assert job.seniority in SENIORITY_LEVELS | {None}, job.seniority

    job.flags = flags
    return job

def run(records: list) -> list:
    """Clean a list of records."""
    return [asdict(clean_job(r)) for r in records]



if __name__ == "__main__":
    # Read the input and output paths from the command line, or use defaults
    src = sys.argv[1] if len(sys.argv) > 1 else "dirty_jobs.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "outputs/cleaned_jobs.json"

    with open(src, encoding="utf-8") as f:
        records = json.load(f)

    cleaned = run(records)

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)  # create the output folder if needed
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    active = sum(1 for j in cleaned if j["is_active"])
    print(f"Processed {len(cleaned)} records: {active} active, {len(cleaned) - active} inactive")
    print(f"Wrote {dst}")