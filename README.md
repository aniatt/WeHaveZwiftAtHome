# WeHaveZwiftAtHome

Real-time rendering of cycling climbs via 3D Gaussian Splatting, with camera pose driven by smart trainer output (e.g., Wahoo Kickr Core) for an interactive indoor riding experience.

## Features

- **Ride iconic climbs from your trainer.** Pre-trained 3DGS models of popular routes are available out of the box -- just download, launch the viewer, and ride.
- **Interactive viewer.** Browser-based renderer with WASD keyboard controls (smart trainer integration coming soon).
- **Build your own routes.** Capture video of any climb, run the preprocessing and training pipeline, and add it to your local library.
- **Smart trainer support (planned).** Connect a Bluetooth FTMS trainer (Wahoo Kickr Core, etc.) to control speed via power output and feel gradient changes as resistance.

## Quick Start

### 1. Pick a route

Pre-trained models for select routes will be published as releases. Download the `.ply` file for the climb you want to ride:

| Route | Location | Distance | Avg Grade | Model |
|-------|----------|----------|-----------|-------|
| *Coming soon* | -- | -- | -- | -- |

Alternatively, [train your own model](#build-your-own-route) from video footage of any climb.

### 2. Launch the viewer

```bash
cd viewer
npm install
npm run dev
```

Open the local URL in your browser, load the `.ply` model file, and you're on the road.

### 3. Ride

| Key | Action |
|-----|--------|
| W / S | Accelerate / brake |
| A / D | Steer left / right |
| Mouse | Look around |
| Shift | Sprint |

## Architecture

```
Route Video
    |
    v
ffmpeg (extract frames)
    |
    v
COLMAP (Structure-from-Motion --> camera poses)
    |
    v
gsplat (train 3DGS model)
    |
    v
.ply model file  ---------->  Web Viewer (Three.js + GaussianSplats3D)
                                    ^
                                    |
                              Wahoo Kickr Core (Bluetooth FTMS) [planned]
```

## Build Your Own Route

For contributors or anyone who wants to ride a climb that isn't in the pre-trained library, the full pipeline is described below. This requires an NVIDIA GPU with CUDA support.

### Prerequisites

System dependencies:

- **ffmpeg** -- frame extraction from video
- **COLMAP** -- Structure-from-Motion for camera pose estimation
- **CUDA toolkit** -- required by gsplat for GPU-accelerated training

Python dependencies:

```bash
pip install gsplat torch numpy viser Pillow
```

### Hardware Requirements

- **Training:** NVIDIA GPU with CUDA support (RTX 3060+ recommended, 8GB+ VRAM)
- **Viewing:** Any modern GPU with WebGL support (integrated graphics OK for small scenes)
- **Data capture:** GoPro, phone, or any camera capable of 1080p+ video

### Step 1: Capture video

Mount a camera (GoPro, phone) on your handlebars or helmet and ride the climb. Tips for best results:

- Steady, forward-facing camera with minimal vibration
- Consistent lighting (avoid tunnels and heavy shadows)
- Slow riding speed helps -- more frames with overlapping views
- 1080p minimum resolution; 4K preferred but increases processing time
- Start small: a 1--2 minute segment (~200--500m of road) is ideal for a first attempt

Store raw video in `data/raw/`.

### Step 2: Extract frames

```bash
ffmpeg -i data/raw/climb.mp4 -vf "fps=2" data/frames/frame_%05d.jpg
```

`fps=2` extracts 2 frames per second. Tune based on riding speed -- slower riding or higher fps yields more frame overlap. A 2-minute clip at fps=2 produces roughly 240 frames.

### Step 3: Run COLMAP

```bash
colmap automatic_reconstructor \
  --workspace_path data/colmap \
  --image_path data/frames
```

This outputs camera intrinsics, extrinsics (poses), and a sparse 3D point cloud. GLOMAP is an alternative for faster processing on large scenes.

Validate the reconstruction by checking that the camera trajectory follows a sensible path along the road. Visualize in the COLMAP GUI or with rerun.io.

### Step 4: Train the 3DGS model

Train with [gsplat](https://github.com/nerfstudio-project/gsplat):

```bash
python -m gsplat.examples.simple_trainer default \
  --data_dir data/colmap \
  --data_factor 4 \
  --result_dir results/climb_v1
```

- `--data_factor 4` downsamples images by 4x for faster iteration
- Training includes a live Viser browser viewer for real-time monitoring
- Typical training: 7,000--30,000 iterations (roughly 10--60 min depending on GPU)

The output is a `.ply` file containing all Gaussian parameters (position, covariance, color, opacity). Check quality by rendering novel views along and slightly off the captured trajectory, and iterate on parameters as needed.

### Step 5: Load in viewer

Copy the `.ply` file into the viewer and ride your route.

## Smart Trainer Integration (Planned)

The end-state goal is to replace keyboard input with real cycling data from a Bluetooth FTMS trainer.

**Connection:** The Wahoo Kickr Core (and similar trainers) broadcast power, speed, and cadence over Bluetooth FTMS at 1 Hz (up to 10 Hz in race mode). The viewer connects via the Web Bluetooth API.

**Input mapping:**

- **Power (watts)** --> forward speed along the route
- **Cadence (rpm)** --> UI display / optional pedaling animation
- **Steering** --> future: Wahoo steering accessory or phone-based lean detection

**Resistance simulation:**

- Road gradient is derived from elevation changes in the COLMAP camera trajectory
- Grade resistance commands are sent back to the trainer via the FTMS control point
- The rider feels the climb steepen in real time

**HUD overlay:** Power, speed, cadence, gradient, and distance displayed in the viewer.

## Project Structure

```
WeHaveZwiftAtHome/
  README.md
  pyproject.toml
  .gitignore
  scripts/
    preprocess.sh       # ffmpeg + COLMAP pipeline
    train.py            # gsplat training wrapper
  viewer/
    index.html          # Web viewer entry point
    main.js             # 3DGS viewer + WASD controls
    package.json        # JS dependencies
  data/                 # (gitignored) raw video, frames, COLMAP output
    raw/
    frames/
    colmap/
  results/              # (gitignored) trained models
```

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Project setup and environment | Not started |
| 1 | Data capture (video of a cycling climb) | Not started |
| 2 | Preprocessing (frame extraction + COLMAP SfM) | Not started |
| 3 | Train 3DGS model with gsplat | Not started |
| 4 | Interactive MVP viewer with WASD controls | Not started |
| 5 | Pre-trained route library | Not started |
| 6 | Smart trainer integration (Bluetooth FTMS) | Planned |

## References

- [3D Gaussian Splatting (original paper)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [gsplat](https://github.com/nerfstudio-project/gsplat) -- training library
- [COLMAP](https://colmap.github.io/) -- Structure-from-Motion
- [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) -- Three.js web viewer
- [antimatter15/splat](https://github.com/antimatter15/splat) -- lightweight WebGL viewer
- [Web Bluetooth API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Bluetooth_API) -- FTMS trainer connection
