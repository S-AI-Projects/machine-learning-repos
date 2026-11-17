# R43-T001 — Schema Validator: Text Summarization Input Contract

**Task type:** Dataset contract
**Student:** RUHAN JANGIR (E26B07F0583)
**Branch:** `student/e26b07f0583/R43-T001`

> **⚠️ Synthetic Fixture Notice**
> All records in this check are **synthetic fixtures for pipeline verification only**.
> They are NOT drawn from any real summarization dataset and results do NOT constitute
> evidence of real-world or production summarization performance.

## Purpose

Implements a schema validator for summarization input records with fields:
- `record_id` — unique non-empty string identifier
- `source_text` — string (may be empty)
- `expected_first_sentence` — explicitly labelled first-sentence baseline (NOT an LLM quality claim)

## Synthetic Fixture

Three fictional records used for validation:

| record_id | source_text | expected_first_sentence |
|-----------|-------------|------------------------|
| r001 | "Ada wrote notes. Bo reviewed them." | "Ada wrote notes." |
| r002 | "One sentence." | "One sentence." |
| r003 | "" | "" |

## Data Contract

- `record_id`: non-empty `str`, unique within a batch
- `source_text`: `str` (empty string allowed)
- `expected_first_sentence`: `str` — deterministic first-sentence extraction baseline (rule-based, not LLM)

## Failure Modes Covered

| Failure | Detected |
|---------|----------|
| Missing `record_id` | ✅ |
| Missing `source_text` | ✅ |
| Missing `expected_first_sentence` | ✅ |
| Non-string `record_id` (e.g. int) | ✅ |
| Non-string `source_text` (e.g. int) | ✅ |
| Duplicate `record_id` | ✅ |

## Running the Checks

```bash
python3 -m unittest discover -s projects/algorithms/Text_Summarization_Model_Langchain_LLAMA2/course-checks/r43-t001 -p "test_*.py"
```

All tests are **offline, CPU-only, and require no external datasets or credentials**.

## Limits

- The first-sentence extractor is a simple regex rule; it is NOT an LLM.
- Results on the synthetic fixture do NOT predict performance on any real corpus.
- No real dataset, API key, or GPU is used.

## Environment

- Python 3 (standard library only — no third-party packages required)
