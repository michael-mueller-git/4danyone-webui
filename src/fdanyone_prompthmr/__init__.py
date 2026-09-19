"""PromptHMR-Vid motion stage (SMPL-X video HMR, stable world trajectory).

Wraps the vendored PromptHMR checkout (``PROMPTHMR_ROOT``) and adapts its
camera-frame SMPL-X output into 4DAnyone's ``MotionResult`` contract. Runs in
the main torch 2.8 environment: PromptHMR's compiled world-video extras
(detectron2 / SAM2 / DROID-SLAM / Metric3D / pytorch3d) are not needed for the
single-person, static-camera path used here.
"""
