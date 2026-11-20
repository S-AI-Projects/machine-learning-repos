# Verification Evidence: R43-T049

## Command Run
`python3 -m unittest discover -s 'projects/detection/Exoplanet Detection/course-checks/r43-t049' -p "test_*.py"`

## Results
- Total Tests: 6
- Passed: 6
- Failed: 0
- Exit Code: 0

## Test Output
```
test_explicit_fallback_mapping (test_pipeline.TestInputPerturbationRobustness.test_explicit_fallback_mapping) ... ok
test_fixture_exact_dimensions_and_labels (test_pipeline.TestInputPerturbationRobustness.test_fixture_exact_dimensions_and_labels) ... ok
test_record_order_permutation_invariance_with_aligned_ids (test_pipeline.TestInputPerturbationRobustness.test_record_order_permutation_invariance_with_aligned_ids) ... ok
test_synthetic_verification_label (test_pipeline.TestInputPerturbationRobustness.test_synthetic_verification_label) ... ok
test_unsupported_input_missing_field_rejected (test_pipeline.TestInputPerturbationRobustness.test_unsupported_input_missing_field_rejected) ... ok
test_unsupported_input_missing_id_rejected (test_pipeline.TestInputPerturbationRobustness.test_unsupported_input_missing_id_rejected) ... ok

----------------------------------------------------------------------
Ran 6 tests in 0.001s

OK
```

## Tool Versions
- Python: 3.14.7
- Environment: offline, CPU-only
