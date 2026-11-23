"""Tests for cv_pipeline.py. Run with: python -m unittest -v test_cv_pipeline"""

import json
import io
import unittest

import cv_pipeline as cp


with open("sandbox_taxonomy.json", encoding="utf-8") as f:
    TAXONOMY = [cp.TaxonomyRow.model_validate(r) for r in json.load(f)]
with open("synthetic_cvs.json", encoding="utf-8") as f:
    CVS = {cv["cv_id"]: cv for cv in json.load(f)}

CV1, CV2, CV3 = CVS["CV_001_CLEAN_TECH"], CVS["CV_002_COMPLEX_HYBRID"], CVS["CV_003_NOISY_SWITCHER"]


def route(raw: str):
    text, _ = cp.build_text(raw)
    return cp.stage1_route(text, TAXONOMY)


def extract(raw: str, locked: set, backend=None):
    text, lines = cp.build_text(raw)
    return cp.stage2_extract(text, lines, TAXONOMY, locked, backend)


def stub_backend(reply: str):
    return lambda prompt: reply


def failing_backend(prompt: str) -> str:
    raise ConnectionRefusedError("connection refused")


class FakeS3Client:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


class TestStage1Routing(unittest.TestCase):
    def test_cv001_is_single_domain_it(self):
        r = cp.stage1_route(cp.build_text(CV1["raw_text"])[0], TAXONOMY)
        self.assertEqual(r.primary_domain, "Information Technology")
        self.assertEqual(r.primary_job_category, "Data and Artificial Intelligence")
        self.assertIsNone(r.secondary_domain)

    def test_cv002_is_hybrid_marketing_consulting(self):
        r = cp.stage1_route(cp.build_text(CV2["raw_text"])[0], TAXONOMY)
        self.assertEqual(r.primary_domain, "Marketing")
        self.assertEqual(r.secondary_domain, "Consulting")
        self.assertEqual(r.secondary_job_category, "Strategy and Corporate Advisory")
        self.assertTrue(r.tiebreak_applied)

    def test_cv003_career_switcher_finance_primary_it_secondary(self):
        r = cp.stage1_route(cp.build_text(CV3["raw_text"])[0], TAXONOMY)
        self.assertEqual(r.primary_domain, "Finance")
        self.assertEqual(r.secondary_domain, "Information Technology")

    def test_secondary_at_exact_half_is_kept(self):
        # IT has 4 sub-skills and Marketing has 2, which is exactly half
        r = route("PySpark Pipelines, dbt Transformations, XGBoost, Dead Letter Queues. "
                  "Google Ads Smart Bidding, Meta Conversions API.")
        self.assertEqual(r.secondary_domain, "Marketing")

    def test_secondary_below_half_is_dropped(self):
        # IT has 4 sub-skills and Marketing has 1, which is below half
        r = route("PySpark Pipelines, dbt Transformations, XGBoost, Dead Letter Queues. "
                  "Google Ads Smart Bidding.")
        self.assertIsNone(r.secondary_domain)

    def test_tie_goes_to_earlier_mention(self):
        a = route("Google Ads Smart Bidding. PySpark Pipelines.")
        b = route("PySpark Pipelines. Google Ads Smart Bidding.")
        self.assertEqual(a.primary_domain, "Marketing")
        self.assertEqual(b.primary_domain, "Information Technology")
        self.assertTrue(a.tiebreak_applied and b.tiebreak_applied)

    def test_no_evidence_raises(self):
        with self.assertRaises(ValueError):
            route("Hobbies: reading, cooking, hiking.")

    def test_domain_values_are_pydantic_constrained(self):
        row = TAXONOMY[0].model_dump(by_alias=True)
        row["Domain"] = "Healthcare"
        with self.assertRaises(ValueError):
            cp.TaxonomyRow.model_validate(row)

    def test_taxonomy_must_have_every_allowed_domain(self):
        it_only = [row for row in TAXONOMY if row.domain == "Information Technology"]
        with self.assertRaisesRegex(ValueError, "exactly the allowed domains"):
            cp.validate_taxonomy(it_only)

    def test_duplicate_synonyms_are_rejected(self):
        duplicated = TAXONOMY + [
            TAXONOMY[0].model_copy(update={"synonym": TAXONOMY[0].synonym.upper()})
        ]
        with self.assertRaisesRegex(ValueError, "synonyms must be unique"):
            cp.validate_taxonomy(duplicated)


class TestStage2Extraction(unittest.TestCase):
    def test_cv001_skills_exact_set(self):
        r = cp.stage1_route(cp.build_text(CV1["raw_text"])[0], TAXONOMY)
        p = extract(CV1["raw_text"], {r.primary_domain})
        self.assertEqual(sorted(s.sub_skill for s in p.skills), sorted([
            "Batch Data Pipeline Development", "Extract Transform Load Automation",
            "Data Pipeline Error Handling", "Python Server Development", "Object Relational Mapping",
            "SQL Query Optimization", "Virtual Private Cloud Design", "Terraform Infrastructure Provisioning",
            "Gradient Boosting Algorithm Tuning", "API Data Ingestion Programming",
        ]))

    def test_out_of_scope_finance_is_quarantined_not_mapped(self):
        p = extract(CV2["raw_text"], {"Marketing", "Consulting"})
        self.assertNotIn("Finance", {s.domain for s in p.skills})
        self.assertEqual({s.matched_text for s in p.out_of_scope_domain_matches},
                         {"Cash Flow Breakeven Timeline", "Monthly Run Rate Analysis"})

    def test_google_ads_alone_does_not_map_to_go(self):
        p = extract("Google Ads reporting for the team.", {"Information Technology"})
        self.assertEqual(p.skills, [])

    def test_case_and_extra_spaces_still_resolve(self):
        p = extract("Built pyspark    pipelines for ingestion.", {"Information Technology"})
        self.assertEqual(len(p.skills), 1)
        self.assertEqual(p.skills[0].sub_skill, "Batch Data Pipeline Development")
        self.assertEqual(p.skills[0].matched_text, "pyspark pipelines")

    def test_headers_and_name_are_not_pending(self):
        r = cp.stage1_route(cp.build_text(CV1["raw_text"])[0], TAXONOMY)
        p = extract(CV1["raw_text"], {r.primary_domain})
        names = {x.lower() for x in p.other_skills_pending_approval}
        for noise in ("experience", "summary", "projects", "education", "arjun mehta"):
            self.assertNotIn(noise, names)

    def test_rule_pending_items_for_cv001(self):
        p = extract(CV1["raw_text"], {"Information Technology"})
        names = {x.lower() for x in p.other_skills_pending_approval}
        self.assertTrue({"fastapi", "pydantic", "redis", "docker", "postgresql", "ml"} <= names)

    def test_pending_candidates_must_exist_in_cv(self):
        profile = cp.ExtractedProfile(other_skills_pending_approval=["InventedSkill"])
        with self.assertRaisesRegex(ValueError, "not present in the CV"):
            cp.assert_pending_from_cv(profile, "Used Python for data processing.")


class TestLLMExtractor(unittest.TestCase):
    def test_rules_only_reports_not_run(self):
        p = extract(CV1["raw_text"], {"Information Technology"})
        self.assertEqual(p.extractor, "rules")
        self.assertEqual(p.llm_status, "not_run")

    def test_valid_phrases_accepted_and_invalid_rejected(self):
        reply = json.dumps([
            "regex validation",          # real phrase in CV_001, should be accepted
            "Kubernetes",                # not in the CV, hallucinated
            "Django",                    # already a taxonomy match, overlap
            "x",                         # too short
            "A" * 70,                    # too long
            42,                          # not a string
            "FastAPI",                   # real, but already a rule pending item; merge dedupes it
        ])
        p = extract(CV1["raw_text"], {"Information Technology"}, stub_backend(reply))
        self.assertEqual(p.llm_status, "ok")
        self.assertEqual(p.extractor, "rules+llm")
        self.assertIn("regex validation", p.other_skills_pending_approval)
        self.assertNotIn("Kubernetes", p.other_skills_pending_approval)
        self.assertEqual(sum(1 for x in p.other_skills_pending_approval if x.lower() == "fastapi"), 1)
        reasons = {r["phrase"]: r["reason"] for r in p.llm_rejected}
        self.assertIn("not found", reasons["Kubernetes"])
        self.assertIn("overlaps", reasons["Django"])
        self.assertIn("length", reasons["x"])

    def test_llm_recovers_free_text_gaps_in_all_three_cvs(self):
        gaps = {
            "CV_001_CLEAN_TECH": "regex validation",
            "CV_002_COMPLEX_HYBRID": "cohort payback models",
            "CV_003_NOISY_SWITCHER": "variance checks",
        }
        for cv_id, phrase in gaps.items():
            raw = CVS[cv_id]["raw_text"]
            r = cp.stage1_route(cp.build_text(raw)[0], TAXONOMY)
            locked = {r.primary_domain} | ({r.secondary_domain} if r.secondary_domain else set())
            p = extract(raw, locked, stub_backend(json.dumps([phrase])))
            self.assertIn(phrase, p.other_skills_pending_approval, cv_id)

    def test_backend_failure_fails_the_extraction_instead_of_returning_partial_results(self):
        with self.assertRaisesRegex(RuntimeError, "LLM extraction failed"):
            extract(CV1["raw_text"], {"Information Technology"}, failing_backend)

    def test_non_json_reply_fails_the_extraction(self):
        with self.assertRaisesRegex(RuntimeError, "LLM extraction failed"):
            extract(CV1["raw_text"], {"Information Technology"}, stub_backend("here are some skills"))

    def test_wrapped_array_is_unwrapped(self):
        reply = json.dumps({"skills": ["regex validation"]})
        p = extract(CV1["raw_text"], {"Information Technology"}, stub_backend(reply))
        self.assertIn("regex validation", p.other_skills_pending_approval)


class TestFlywheel(unittest.TestCase):
    def setUp(self):
        r = cp.stage1_route(cp.build_text(CV1["raw_text"])[0], TAXONOMY)
        self.env = cp.process_cv(CV1, TAXONOMY)
        self.assertEqual(r.primary_domain, "Information Technology")

    def test_user_profile_starts_as_independent_copy(self):
        self.env.user_verified_profile.skills.pop()
        self.assertEqual(len(self.env.model_extracted_profile.skills), 10)

    def test_no_edits_means_empty_diff_and_no_training_pair(self):
        cp.attach_diff(self.env)
        self.assertEqual(self.env.diff_log, [])
        self.assertEqual(cp.export_training_pairs(self.env), [])

    def test_user_removal_and_addition_are_logged(self):
        removed = self.env.user_verified_profile.skills.pop(0)
        self.env.user_verified_profile.other_skills_pending_approval.remove("Redis")
        self.env.user_verified_profile.other_skills_pending_approval.append("Kafka")
        cp.attach_diff(self.env)
        ops = {(c["field"], c["op"]) for c in self.env.diff_log}
        self.assertIn(("skills", "removed"), ops)
        self.assertIn(("other_skills_pending_approval", "removed"), ops)
        self.assertIn(("other_skills_pending_approval", "added"), ops)
        removed_item = next(c for c in self.env.diff_log if c["field"] == "skills")
        self.assertEqual(removed_item["item"]["sub_skill"], removed.sub_skill)

    def test_changed_skill_path_details_are_logged(self):
        verified = self.env.user_verified_profile.model_copy(deep=True)
        verified.skills[0] = verified.skills[0].model_copy(update={"matched_text": "pyspark pipelines"})
        updated = cp.update_user_verification(self.env, verified, TAXONOMY)
        self.assertEqual(
            {(change["field"], change["op"]) for change in updated.diff_log},
            {("skills", "added"), ("skills", "removed")},
        )

    def test_training_pair_contains_input_and_both_outputs(self):
        self.env.user_verified_profile.other_skills_pending_approval.append("Kafka")
        cp.attach_diff(self.env)
        pairs = cp.export_training_pairs(self.env)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["input_text"], CV1["raw_text"])
        self.assertIn("Kafka", pairs[0]["user_output"]["other_skills_pending_approval"])
        self.assertNotIn("Kafka", pairs[0]["model_output"]["other_skills_pending_approval"])

    def test_s3_user_review_persists_envelope_and_idempotent_training_pair(self):
        client = FakeS3Client()
        store = cp.S3FlywheelStore("test-bucket", TAXONOMY, "test-prefix", client)
        store.put_envelope(self.env)
        verified = self.env.user_verified_profile.model_copy(deep=True)
        verified.other_skills_pending_approval.append("Kafka")

        updated = store.record_user_verification(CV1["cv_id"], verified.model_dump())
        self.assertIn("Kafka", updated.user_verified_profile.other_skills_pending_approval)
        self.assertTrue(updated.diff_log)
        pair_keys = [key for _, key in client.objects if "/training-pairs/" in key]
        self.assertEqual(len(pair_keys), 1)

        store.record_user_verification(CV1["cv_id"], verified)
        pair_keys = [key for _, key in client.objects if "/training-pairs/" in key]
        self.assertEqual(len(pair_keys), 1)
        saved = store.load_envelope(CV1["cv_id"])
        self.assertIn("Kafka", saved.user_verified_profile.other_skills_pending_approval)


class TestBatchIsolation(unittest.TestCase):
    def test_bad_cv_does_not_stop_the_batch(self):
        cvs = list(CVS.values()) + [{"cv_id": "BAD_EMPTY", "raw_text": "Hobbies: reading."}]
        envelopes, failures = cp.process_batch(cvs, TAXONOMY)
        self.assertEqual(len(envelopes), 3)
        self.assertEqual([f["cv_id"] for f in failures], ["BAD_EMPTY"])


if __name__ == "__main__":
    unittest.main()
