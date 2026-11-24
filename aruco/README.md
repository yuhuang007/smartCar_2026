# ArUco Smart Car Localization System

## Overview

Camera mounted at **4m height** looking down at a **4m x 5m** ground area.
Detects ArUco reference markers on the ground and a marker on the smart car
to compute real-time position (x, z) and heading.

## Hardware Requirements

| Device | Notes |
|--------|-------|
| USB camera | Wide-angle recommended, 720p+ |
| Camera mount | Fixed at 4m height, lens pointing straight down |
| Printer | Laser printer for crisp black/white markers |
| Cardboard | To keep markers flat on the ground |
| Smart car | Flat surface on top for ArUco marker |

## Software Installation

```powershell
cd E:\zhoumian\smartCar\aruco
pip install -r requirements.txt
```

Verify:
```powershell
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

## Step-by-Step Usage

### Step 1: Generate ArUco Markers

```powershell
python generate_markers.py
```

Output in `markers/` folder:

| File | Purpose | Print Size |
|------|---------|------------|
| marker_0_pure.png | Ground Top-Left (0, 0) | 25cm x 25cm |
| marker_1_pure.png | Ground Top-Right (4m, 0) | 25cm x 25cm |
| marker_2_pure.png | Ground Bottom-Left (0, 5m) | 25cm x 25cm |
| marker_3_pure.png | Ground Bottom-Right (4m, 5m) | 25cm x 25cm |
| marker_10_pure.png | Car marker | 15cm x 15cm |

> Use the `_pure.png` files (no labels) for better detection.

### Step 2: Print and Place Markers

1. Print with laser printer, ensure high contrast
2. Mount each marker on cardboard to keep flat
3. Place ground markers at the 4 corners of your 4m x 5m area
4. Stick car marker on top center of the smart car

```
Placement (top view):

  Z(5m)
  ^
  [marker_0]----------[marker_1]
  |                        |
  |     Your map area       |    4m (X axis)
  |     4m x 5m             |
  [marker_2]----------[marker_3]
                        --> X(4m)
```

### Step 3: Map Calibration

```powershell
python map_calibrate.py
```

1. Camera window opens, ensure all 4 ground markers are visible
2. Press **SPACE** to sample (need all 4 markers detected)
3. After 5 samples, calibration saves automatically
4. Press **Q** to exit

> Do NOT move the camera after calibration!

### Step 4: Real-time Tracking

```powershell
python track.py
```

- **Left panel**: Camera view with detected markers and car coordinates
- **Right panel**: Top-down map with red triangle = car position/heading

| Key | Action |
|-----|--------|
| Q | Quit |
| R | Clear trail |
| S | Save screenshot |

## Configuration

Edit `config.py` to adjust:

- `CAMERA_INDEX`: 0 or 1 (try both to find your USB camera)
- `MAP_WIDTH / MAP_HEIGHT`: Map dimensions in meters
- `CAMERA_HEIGHT_M`: Camera height (for reference only)
- `GROUND_MARKER_SIZE`: Ground marker print size in meters
- `CAR_MARKER_SIZE`: Car marker print size in meters
- `GROUND_MARKER_POSITIONS`: World coordinates of each ground marker

## Troubleshooting

### Markers not detected
- Ensure good, even lighting (avoid glare and shadows)
- Print markers clearly with sharp edges
- Increase print size if needed
- Check camera focus

### Wrong camera opens
- Change `CAMERA_INDEX` in config.py (try 0, 1, 2)

### Coordinates don't match reality
- Re-run `map_calibrate.py` after any config change
- Ensure markers are placed at the exact positions defined in config
- Markers must be flat on the ground, no bending

### "No module named cv2"
```powershell
pip install opencv-contrib-python
```

## Project Files

```
aruco/
  config.py             # Configuration (map size, camera, marker IDs)
  generate_markers.py   # Generate ArUco marker images
  camera_calibrate.py   # (Optional) Camera intrinsic calibration
  map_calibrate.py      # Map calibration (pixel -> world mapping)
  track.py              # Real-time tracking main program
  requirements.txt      # Python dependencies
  homography.npz        # Calibration result (auto-generated)
  markers/              # Generated marker images
```