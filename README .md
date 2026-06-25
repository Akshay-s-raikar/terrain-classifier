# Terrain Classifier for Robot Navigation

Real-time terrain segmentation and traversability scoring using OpenCV and a standard RGB camera. Built for ground robots and autonomous drone landing systems.

---

## What it does

- Opens a live camera feed and extracts the ground ROI (lower field of view)
- Classifies terrain type in real time — grass, concrete, gravel, sand, mud, obstacle
- Computes a traversability score (0–100) based on weighted terrain coverage
- Draws a safe zone overlay highlighting traversable regions
- Displays a HUD with terrain type, coverage %, safe zone %, FPS, and navigation status
- Logs all terrain data with timestamps to CSV for analysis
- Includes a separate HSV calibration tool to tune profiles for your camera and lighting

---

## Why this matters

Most robot navigation systems assume flat, uniform ground. In practice, robots encounter mixed terrain — grass patches, gravel paths, concrete edges, mud. This tool gives a robot a simple, fast, camera-only answer to: **"is this surface safe to drive or land on?"**

No GPU required. Runs on Raspberry Pi, Jetson Nano, or any Linux embedded system.

---

## Demo

| Output | Description |
|---|---|
| Green overlay | Safe to traverse |
| Blue overlay | Caution — low confidence |
| Red overlay | Do not traverse |

---

## Terrain profiles

| Terrain | Traversable | Base score |
|---|---|---|
| Concrete | Yes | 95 |
| Grass | Yes | 80 |
| Gravel | Yes | 70 |
| Sand | Yes | 60 |
| Mud | No | 25 |
| Obstacle (dark) | No | 0 |

---

## How to run

```bash
git clone https://github.com/Akshay-s-raikar/terrain-classifier.git
cd terrain-classifier
pip install -r requirements.txt
python src/terrain_classifier.py
```

**Keybinds:**

| Key | Action |
|---|---|
| Q | Quit |
| D | Toggle debug mask windows |
| S | Save screenshot |

---

## Calibrate for your environment

HSV ranges vary by camera sensor and lighting. Run the calibration tool first:

```bash
python src/calibrate_hsv.py
```

Point the camera at each surface type, tune the sliders until only that surface is highlighted in the mask, then press **S** to print the HSV range. Copy those values into `TERRAIN_PROFILES` in `terrain_classifier.py`.

---

## Config options

In `terrain_classifier.py`, top of file:

```python
CAMERA_INDEX = 0       # change to 1 for external USB camera
LOG_ENABLED  = True    # set False to disable CSV logging
SHOW_DEBUG   = False   # set True to show HSV mask windows on startup
```

---

## Output log format

Logs are saved to `logs/terrain_YYYYMMDD_HHMMSS.csv`:

```
timestamp, terrain, traversability_score, dominant_coverage_pct, safe_zone_area_pct
2026-06-24T10:22:01, grass, 78.4, 62.3, 55.1
```

---

## Folder structure

```
terrain-classifier/
├── src/
│   ├── terrain_classifier.py   # main classifier
│   └── calibrate_hsv.py        # HSV tuning tool
├── logs/                        # CSV output (auto-created)
├── assets/                      # screenshots
├── requirements.txt
└── README.md
```

---

## Roadmap

- [ ] Add texture-based classification (LBP features) for better accuracy
- [ ] ROS 2 node wrapper for Nav2 integration
- [ ] Depth camera support (Intel RealSense) for 3D safe zone mapping
- [ ] Export safe zone as costmap for move_base

---

## Related projects

- [PID Visualizer](https://github.com/Akshay-s-raikar/pid-visualizer) — flight controller tuning tool
- Custom STM32 Flight Controller — in progress
- Arena Mission Stack (IRoC-U 2026) — full autonomous drone system

---

## Author

Akshay Shailesh Raikar
CSE Student, Jain College of Engineering and Research, Belagavi
[LinkedIn](https://www.linkedin.com/in/akshay-s-raikar) · [GitHub](https://github.com/Akshay-s-raikar)
