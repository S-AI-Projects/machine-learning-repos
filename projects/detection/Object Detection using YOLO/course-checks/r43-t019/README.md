# R43-T019: Record-order robustness check

The pinned YOLO example consumes images and model detections; time order is not
applicable. This bounded classification-interface fixture uses eight synthetic
rows with stable record and group IDs, fixed seed 19, and explicit stratified
train, validation, and test memberships. It is not data from the named object
detection project and makes no detection or real-world performance claim.

The tested property is row-order invariance: reversing input record order keeps
predictions identical after alignment by stable `record_id`. Records missing a
feature or containing an unsupported field are rejected. This narrow check does
not claim broad adversarial robustness.

Run from the repository root:

```bash
python3 -m unittest discover -s 'projects/detection/Object Detection using YOLO/course-checks/r43-t019' -p 'test_*.py' -v
```

The check uses only Python's standard library and does not load YOLO weights,
images, or external datasets.
