# src/smplx — Mocap Unification

Converts mocap data of any origin into a unified SMPL-X representation
(159-d canonical pose: trans + global_orient + body_pose + both hand
poses). See `doc/Mocap_Unification_Design.docx` and
`doc/SMPLX_Fitting_Methods.docx` for the full design rationale — this file
documents what is actually implemented, tested, and shipped.

## Requirements

- `pip install -r requirements.txt` (numpy, scipy, torch, smplx, ...)
- An official SMPL-X body model file (`SMPLX_{NEUTRAL,FEMALE,MALE}.npz`),
  registration-gated at https://smpl-x.is.tue.mpg.de and **not** included in
  this repo. Point to it with:

  ```bash
  export LEONIDAS_SMPLX_MODEL=/path/to/SMPLX_FEMALE.npz
  ```

  Without it, Tier 0 (pure parameter conversion) still works fully; any
  fitting (Tier 1+, `fitting/`, `regressor.py`) raises a clear
  `SMPLXModelNotFoundError` rather than failing silently or faking a result.

## What's implemented and tested

| Piece | File(s) | Status |
|---|---|---|
| Canonical schema, part masks, quality scorecard | `schema.py`, `quality.py` | Done, unit-tested |
| 159-d axis-angle <-> 315-d 6D-rotation model features | `features.py` | Done, round-trip tested incl. near-singular rotations |
| fps resampling (SLERP), up-axis, units | `ops.py` | Done, unit-tested |
| Tier 0: SMPL/SMPL-H/SMPL-X params -> canonical | `adapters/params.py` | Done, tested against real AMASS sample data |
| Correspondence layouts (COCO-17, OpenPose-25, MediaPipe-33, H36M-17) | `layouts/` | Done, unit-tested |
| Method 1 — Similarity alignment (Umeyama/Kabsch) | `fitting/align.py` | Done, exact-recovery tested |
| Real SMPL-X body model (forward kinematics, batched, differentiable) | `fitting/body.py` | Done, requires the model file |
| Method 2 — Staged optimization solver | `fitting/solver.py` | Done, end-to-end synthetic-recovery tested against the real body model |
| Method 3 — Learned-regressor interface + 2D reprojection refinement | `regressor.py` | Refinement step is real and tested; the network itself (`NullRegressor`) is a documented stub — see below |

## Known, honest limitations

- **No VPoser.** The pose prior (`fitting/priors.py::GaussianJointPrior`) is
  a per-joint L2 + anatomical-limit substitute, not the VPoser latent-space
  prior the design doc specifies — no VPoser checkpoint was available in
  this environment. It implements the same `PosePrior` interface, so a real
  `VPoserPrior` can replace it without touching `solver.py`.
- **No trained image regressor.** `NullRegressor` raises
  `NotImplementedError` rather than fabricating a pose — running Method 3's
  network step requires downloading a SMPLer-X/NLF/OSX checkpoint, which
  this environment cannot do on your behalf. `refine_from_2d` (the
  reprojection-refinement half of Method 3) is real and tested; it just
  needs a `RegressorEstimate` to start from.
- **Hands are never fit by the Tier-1 skeleton solver.** The built-in
  layouts supply at most a few sparse fingertip points per hand — not
  enough to identify a 45-dof hand pose without overfitting noise. Hand
  pose is left at rest and `part_mask` marks it unobserved, honestly,
  rather than presenting a fabricated hand fit. A layout with real
  per-finger-joint correspondences can add a hand stage.
- **Face is never fit anywhere in this package.** It is out of scope of
  the 159-d canonical layout by design (see the main design doc, section
  2.1).

## Quick usage

```python
from src.smplx.adapters.params import ingest_params   # Tier 0
from src.smplx.adapters.skeleton3d import ingest_skeleton3d  # Tier 1
from src.smplx.fitting.body import SMPLXBody

# Tier 0: already-parametric mocap (AMASS, Motion-X, ...)
motion = ingest_params(raw_npz_dict)  # -> CanonicalMotion

# Tier 1: 3D joints of unknown scale/orientation (BVH, MediaPipe world
# landmarks, a custom skeleton) -> fit to SMPL-X
body = SMPLXBody()  # finds the model file via LEONIDAS_SMPLX_MODEL
motion, quality = ingest_skeleton3d(points, fps=30.0, body=body, layout="mediapipe_33")
print(quality.verdict)  # GOOD | DEGRADED | REJECT
```

## Streamlit viewer app (`app/`)

A two-tab visual UI for exactly this pipeline: upload a `.npz` (SMPL/SMPL-H/
SMPL-X params), `.c3d` (optical mocap markers), or a video, and compare it
raw against a unified SMPL-X fit.

```bash
export LEONIDAS_SMPLX_MODEL=/path/to/SMPLX_FEMALE.npz
streamlit run app/streamlit_app.py
```

- **Tab 1 — Raw Mocap**: the data exactly as the source format measured it.
  `.npz` params are shown as a skeleton (direct forward kinematics, no
  fitting). `.c3d` markers are shown as a raw point cloud. `.bvh` files are
  shown as a skeleton using the file's own bone hierarchy. Video is run
  through MediaPipe Pose to extract real 3D world-landmark mocap, shown as
  a skeleton. Unobserved points are drawn in red, never hidden.
- **Tab 2 — SMPL-X Fit**: the same clip unified onto the real SMPL-X mesh
  and rendered with `pyrender` (a genuine lit, shaded 3D render, orbit
  controls in the sidebar), plus the `FitQuality` verdict (GOOD/DEGRADED/
  REJECT) and the fitting notes (e.g. how many C3D markers or BVH joints
  matched a known naming convention, or that hands are shown at rest).

`.npz` params need no fitting (Tier 0, pure conversion). `.c3d` markers fit
via `adapters/markers.py`'s `VICON_50` vertex-correspondence layout (built
from real marker->vertex ids extracted from an actual AMASS/SOMA capture
file — see `layouts/builtin.py`), matching whichever of its 50 markers the
upload actually has by label (namespace-prefixed labels like `male2:LFHD`
are recognized too). `.bvh` files are parsed and forward-kinematics'd
ourselves (`adapters/bvh.py`; the `bvh` package only exposes raw channels,
not world positions), auto-detecting both the file's up-axis and its
unit scale — BVH has no standard for either, and the unit heuristic is
careful to use only the per-frame vertical span, not the full bounding-box
diagonal, so a clip where the character travels a long distance
horizontally (a run, a traveling skip) doesn't get mistaken for a giant
body. Joint names are matched against a CMU/Mixamo-style alias table
(`mixamorig:` and similar namespace prefixes are stripped automatically).
Video fits via MediaPipe's real 3D world landmarks through the Tier-1
solver (`mediapipe_33` layout) — this sidesteps needing the (unavailable,
see above) trained regressor checkpoint entirely, since MediaPipe already
provides genuine metric 3D pose.

C3D and BVH fits are only as good as how well the uploaded file's marker/
joint placements match the specific protocol our correspondence tables were
derived from — a file using the same marker *names* but a different lab's
placement convention can still land in `DEGRADED`/`REJECT` even though it
loads and fits without error; that's `FitQuality` doing its job, not a bug.

Tested via Streamlit's `AppTest` harness (`tests/test_streamlit_app.py`)
against real sample files, plus `tests/test_app_pipeline.py` for the
upload-processing logic itself, and confirmed to boot as a live server with
no startup errors.

## Running the tests

```bash
python -m pytest tests/ -q
```

Run from the repo root (not from inside `src/`) so `from src.smplx import
...` resolves correctly and does not collide with the installed `smplx` pip
package. Model-dependent tests skip automatically if no SMPL-X file is
found; the Tier-0 real-data regression tests skip if the GPSM-1 sample data
path isn't present on the machine.
