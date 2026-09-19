# Leonidas

A foundation model for human motion, pretrained on large-scale mocap data spanning sports, combat, and tactical movement, for general-purpose motion understanding and generation.

## Motion data unification

Mocap data comes in many shapes: different file formats, skeletons, units, and axis conventions. A foundation model can only learn from all of it if it is first converted into one common representation. This repo does that conversion, targeting the [SMPL-X](https://smpl-x.is.tue.mpg.de) body model.

Every input ends up as a **159-number pose per frame** (root translation, global orientation, 21 body joints, both hands), plus a per-part mask that records what was actually observed versus filled in. Missing data is never silently guessed.

| Input | How it is unified |
|---|---|
| `.npz` (SMPL / SMPL-H / SMPL-X params) | Direct conversion, no fitting |
| `.c3d` (optical markers) | Marker-to-body-vertex matching, then fitted to SMPL-X |
| `.bvh` (skeleton animation) | Forward kinematics, joint matching, then fitted to SMPL-X |
| Video | MediaPipe 3D landmarks, then fitted to SMPL-X (work in progress, accuracy is limited on occluded or side-on footage) |

Fitting is a closed-form similarity alignment (scale, rotation, translation) followed by a staged, coarse-to-fine optimization. Each result gets a quality verdict: `GOOD`, `DEGRADED`, or `REJECT`.

Code lives in `src/smplx/`. See [src/smplx/README.md](src/smplx/README.md) for details and known limitations.

## Testing app

A Streamlit viewer in `app/` for trying the pipeline on your own files. Upload a mocap file and compare two views:

1. **Raw Mocap**: the data as the source measured it (skeleton, marker cloud, or video landmarks), full clip, playable.
2. **SMPL-X Fit**: the same clip unified onto the SMPL-X mesh in an interactive 3D viewport (drag to orbit, scroll to zoom, right-drag to pan, static ground), with the fit quality verdict.

## Setup

Requires Python 3.8+ (developed on 3.8).

```bash
git clone https://github.com/MorningStarTM/Leonidas.git
cd Leonidas

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
```

### SMPL-X model file

The SMPL-X body model is registration-gated and is not included in this repo. Register and download it from https://smpl-x.is.tue.mpg.de, then either:

- place it at `src/model/SMPLX_FEMALE.npz` (this folder is gitignored), or
- point to it with an environment variable:

```bash
# Windows PowerShell:  $env:LEONIDAS_SMPLX_MODEL = "C:\path\to\SMPLX_FEMALE.npz"
export LEONIDAS_SMPLX_MODEL=/path/to/SMPLX_FEMALE.npz
```

Without the model file, parameter conversion (`.npz`) still works, but any fitting raises a clear "model not found" error.

## Run the app

From the repo root:

```bash
streamlit run app/streamlit_app.py
```

Then upload a `.npz`, `.c3d`, `.bvh`, or video file from the sidebar.

## Run the tests

Always run from the repo root (not from inside `src/`), so the local `smplx` package does not collide with the installed one.

```bash
# Quick check (fast tests)
python -m pytest tests/test_schema.py tests/test_ops.py tests/test_features.py tests/test_align.py tests/test_client_side_playback.py -q

# Full suite (about 10 minutes, includes fitting on real data)
python -m pytest tests/ -q
```

Notes:

- Tests that need the SMPL-X model skip automatically if it is not found.
- Some tests use sample mocap and video files at local paths on the author's machine and skip if those files are missing.



## License

See [LICENSE](LICENSE). The SMPL-X model is licensed separately by its authors.
