# Verification Evidence: R43-T047

## Command Run
`python3 -m unittest discover -s 'projects/prediction/Retail Price Prediction/course-checks/r43-t047' -p "test_*.py"`

## Results
- Total Tests: 6
- Passed: 6
- Failed: 0
- Exit Code: 0

## Test Output
```
test_contradictory_settings_rejected (test_pipeline.TestReproducibleConfiguration.test_contradictory_settings_rejected) ... ok
test_environment_versions_recorded (test_pipeline.TestReproducibleConfiguration.test_environment_versions_recorded) ... ok
test_incompatible_input_shape_rejected (test_pipeline.TestReproducibleConfiguration.test_incompatible_input_shape_rejected) ... ok
test_reloaded_config_reconstructs_identical_splits_and_predictions (test_pipeline.TestReproducibleConfiguration.test_reloaded_config_reconstructs_identical_splits_and_predictions) ... ok
test_synthetic_verification_label (test_pipeline.TestReproducibleConfiguration.test_synthetic_verification_label) ... ok
test_unknown_field_rejected (test_pipeline.TestReproducibleConfiguration.test_unknown_field_rejected) ... ok

----------------------------------------------------------------------
Ran 6 tests in 0.001s

OK
```

## Tool Versions
- Python: 3.14.7
- Environment: offline, CPU-only
