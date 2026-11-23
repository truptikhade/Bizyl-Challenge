# Bizyl Backend and Applied NLP Challenge

Submission for the three-part challenge: CV routing and skill mapping (Task 1), job board cleaning (Task 2), and a profile snapshot audit (Task 3).

## Files

| File                                                       | What it is                                          |
| ---------------------------------------------------------- | --------------------------------------------------- |
| cv_pipeline.py                                             | Task 1 pipeline                                     |
| test_cv_pipeline.py                                        | Task 1 unit tests                                   |
| outputs/parsed_cvs.json                                    | Task 1 output, one envelope per CV                  |
| outputs/failed_cvs.json                                    | Task 1 failures (empty for the three sample CVs)    |
| job_hygiene.py                                             | Task 2 cleaning script                              |
| outputs/cleaned_jobs.json                                  | Task 2 output for the six job records               |
| job_hygiene_llm.py, test_llm_stub.py                       | Optional LLM version of Task 2, kept for comparison |
| sandbox_taxonomy.json, synthetic_cvs.json, dirty_jobs.json | Input data from the appendices                      |

## Setup

Python 3.10 or newer. Task 1 needs Pydantic:

    pip install pydantic

Task 2 uses only the standard library. The S3 option needs boto3 and AWS credentials, which are optional.

## Running

Task 1:

    python cv_pipeline.py synthetic_cvs.json sandbox_taxonomy.json outputs/parsed_cvs.json

The pipeline runs its own checks at startup: taxonomy validation, collision checks, and noise checks. A broken taxonomy stops the run before any CV is read.

Tests:

    python -m unittest -v test_cv_pipeline

Task 2:

    python job_hygiene.py dirty_jobs.json outputs/cleaned_jobs.json

## Task 1

### Taxonomy validation

Before routing, `validate_taxonomy` checks that the taxonomy contains exactly the four allowed domains, that synonyms are unique when case is ignored, and that every alias target and title-cue target exists in the taxonomy. The allowed domains are declared once, in the `Domain` type. The taxonomy file has to match that list.

### Stage 1: domain gate

Each CV is scored against the taxonomy. A unique sub-skill match is worth 1 point. A job-title cue such as "head of growth" or "financial analyst" is worth 2 points. Domains and categories use the same scoring.

- The primary domain is the highest scorer. A tie goes to the domain mentioned first in the CV, and the tie is recorded in `tiebreak_applied`.
- The secondary domain is the runner-up, but only if its score is at least half the primary score. This is how the hybrid CV (CV_002) gets two domains.
- The chosen domain and category must exist as a pair in the taxonomy, or the CV fails.

### Stage 2: extraction

Domain drift. Only taxonomy rows in the primary and secondary domains are used to map skills. Matches in any other domain go to `out_of_scope_domain_matches`. They are kept for review and never mapped. In CV_002, the two Finance phrases ("Cash Flow Breakeven Timeline", "Monthly Run Rate Analysis") are reported this way.

Token collisions. Matching is by whole phrase. A letter, digit, or ampersand next to a match blocks it, so "Go" does not match inside "Google", and "R" does not match inside "R&D". Longer phrases are matched first, and each character can belong to one match. The collision cases are checked on every run.

Synonyms and aliases. "PySpark Pipelines" resolves to its full taxonomy path. "TAM/SAM/SOM" and "Five Whys" are handled through an alias table. Matching ignores case and extra spaces.

Pending review. Skill-like phrases the taxonomy does not cover go to `other_skills_pending_approval` and are never forced onto a taxonomy node. Three rule-based passes find them:

- items on skill-list lines, such as "Tools used: FastAPI, Pydantic"
- CamelCase and acronym tokens, such as PostgreSQL, SHAP, and FP&A
- a short list of lowercase tools such as pandas

Header lines, the name line, contact and company lines, education, project titles, and labels are excluded. Roles and regions such as CFO and APAC are filtered out.

### Stage 3: envelope and feedback loop

Each CV is written as an envelope with `cv_id`, `raw_text`, `stage1_routing`, `model_extracted_profile`, `user_verified_profile`, and `diff_log`. The verified profile starts as a copy of the model profile.

When a user edits the profile, `update_user_verification` checks the edited paths against the taxonomy, saves them, and recomputes `diff_log`. `diff_profiles` lists each added and removed skill or pending phrase. A training pair is created only when the diff is not empty, so unedited CVs add nothing.

S3 storage (`S3FlywheelStore`) uses two key patterns:

- `cvs/<cv_id>.json` holds the envelope. A new upload replaces the old one.
- `training-pairs/<cv_id>/<sha256>.json` holds a training pair. The key is a hash of the content, so writing the same pair twice creates one object.

`record_user_verification` loads an envelope, applies the user's edits, writes the pair, and saves the envelope. The `--s3` flag only uploads envelopes. Training pairs are written when a review is recorded through `record_user_verification`.

### Results (rules only, sample CVs)

| CV           | Primary                                                   | Secondary                                                 | Mapped skills | Pending | Out of scope |
| ------------ | --------------------------------------------------------- | --------------------------------------------------------- | ------------- | ------- | ------------ |
| CV_001 Arjun | Information Technology / Data and Artificial Intelligence | none                                                      | 10            | 6       | 0            |
| CV_002 Priya | Marketing / Growth and Performance Media                  | Consulting / Strategy and Corporate Advisory              | 8             | 9       | 2            |
| CV_003 Rohan | Finance / Corporate Finance and Strategic Planning        | Information Technology / Data and Artificial Intelligence | 6             | 5       | 0            |

Rohan's CV ties Finance and IT at 4 points. Finance is primary because its evidence appears first, which fits a career change. The current direction is IT, so a reviewer could reasonably prefer the other order. The tie is recorded in the output.

### Tests

The suite has 30 tests (`python -m unittest -v test_cv_pipeline`). They cover:

- routing for all three sample CVs
- the secondary threshold at exactly half and just below
- tie-breaking by first mention
- taxonomy validation: allowed domains, duplicate synonyms
- exact skill sets for CV_001
- out-of-scope quarantine and collision guards
- case and spacing variants
- pending-phrase presence checks
- LLM acceptance and rejection (using a stub), and LLM failures failing the CV
- the diff log, including changed skill paths
- training pairs
- S3 round trip with a fake client, including that a repeated review does not create a second training-pair object
- batch isolation for a bad CV

## Task 2

`job_hygiene.py` applies rules to each record:

1. HTML entities are decoded and whitespace is normalized.
2. Closure phrases in the snippet ("no longer available", "has been filled", "404 Not Found", "requisition ... does not exist") set `is_active` to false and record `inactive_reason`. Inactive listings skip UI mapping and get a `listing_inactive` flag.
3. Employment hints are normalized to the four enum values.
4. Remote scope comes from the location: on-site, hybrid, remote global, or remote restricted.
5. If the location says global but the snippet restricts applicants ("must reside in ..."), the scope becomes Remote (Restricted) and the record is flagged. This is the JB_101 case.
6. The domain is the one with the most whole-word keyword matches. A tie or no matches leaves it empty and flags the record.
7. Seniority comes from title cues first, then years of experience, then a default that is flagged.
8. Every value is checked against its enum before output.

Decisions to review:

| Record         | Decision                              | Flag                |
| -------------- | ------------------------------------- | ------------------- |
| JB_101         | Remote Anywhere restricted by snippet | contradiction       |
| JB_102, JB_105 | Closed listings                       | listing_inactive    |
| JB_103, JB_107 | No seniority cue, default Mid-Level   | seniority_defaulted |
| JB_104         | "Permanent" treated as Full-time      | employment_assumed  |

## Task 3: profile snapshot audit

I compared my profile snapshot with my CV.

1. Matplotlib is filed under Journalism and News Reporting (0%), with the domain "Media and Communication". My CV lists it under AI and ML and in two projects.
   Fix: accept a match only when the same role or project has supporting evidence, such as Python or Pandas, and send everything else to review.

2. The header and domain weights say Fullstack Developer, Software Engineer 78%, and Full Stack 22%. The CV headline is AI Engineer with Machine Learning and Computer Vision, and the ML work is not reflected.
   Fix: weight domains by skills, projects, and the current role, not by job titles, and reuse the scoring in cv_pipeline.py.

3. The generated summary repeats "Java Enterprise Development" and uses taxonomy labels such as "CPP High Performance Computing" instead of the CV's wording.
   Fix: build the stack line from de-duplicated matched text, and keep the candidate's own summary as the primary text.

4. Four of eight job categories show 0% and 0 years, and the profile lists Achievements and Personality Traits that the CV does not contain.
   Fix: create a category only when a skill matches it, and show Achievements and Personality Traits only when the source text has them. In the frontend, hide zero-experience categories by default.
