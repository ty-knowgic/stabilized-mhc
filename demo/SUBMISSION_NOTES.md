# Submission Notes

## Overview
This change delivers an audit-ready Algebraic mHC reachability kill switch demo that structurally enforces forbidden primitives as hard-zero routing columns, with deterministic ordering, CLI-driven input, and a regression test plus audit artifacts.

## Files to INCLUDE in main

- `demo/constrained_routing_demo.py`
  - Implements the CLI-driven demo and hard-zero masking with explicit diagnostics.
  - Provides deterministic primitive ordering and auditable output for review.
- `tests/test_forbidden_primitives_zero.py`
  - Regression guard verifying hard-zero forbidden columns and normalization targets.
  - Shares the same primitive-loading path as the demo.
- `demo/README_MHC_DEMO.md`
  - Documents how to run the demo and the exact guarantees (and non-guarantees).
- `demo/AUDIT_SUMMARY.md`
  - Captures the executed commands and key numeric results for audit traceability.
- `demo/SUBMISSION_NOTES.md`
  - Submission classification and rationale for included/excluded files.

## Files to EXCLUDE from main

- `README.md` (modified)
  - Rationale: unrelated Colab reproducibility note; not required for the demo/test/audit story.
  - Handling: keep as local-only (no git action in this submission).
- `benchmarks/colab_benchmark.py` (untracked)
  - Rationale: unrelated benchmark helper; not required for the demo/test/audit story.
  - Handling: keep as local-only (no git action in this submission).
- `notebooks/` (untracked)
  - Rationale: unrelated notebook; not required for the demo/test/audit story.
  - Handling: keep as local-only (no git action in this submission).
