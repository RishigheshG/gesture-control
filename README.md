# Gesture Control App — macOS Hand Gesture Volume & Brightness Control

A background macOS tool that uses webcam-based hand-landmark tracking (MediaPipe) to control system volume and brightness through pinch-and-drag gestures. Self-directed portfolio project, not tied to a specific job-posting skill gap.

## Phase 1 status: core loop working

This repo currently implements **Phase 1** of a 4-phase plan: proving the camera → landmark → system-action pipeline end to end using plain geometric thresholds, no machine learning involved yet.

**What works:**
- Real-time hand landmark tracking via MediaPipe Hands
- Pinch detection (thumb + index → volume, thumb + middle → brightness) via Euclidean distance thresholds
- Anchor-relative dragging: displacement is measured from wherever the pinch started, not from a fixed screen position or the previous frame, so the gesture works comfortably regardless of hand position or camera placement
- Real system volume control via `osascript`
- Real system brightness control via a Silicon-compatible `brightness` CLI fork
- Debounce and dead-zone logic so a held pinch doesn't spam system calls or drift on its own from natural hand tremor

**What's deliberately not here yet (later phases):**
- No trained ML gesture classifier, this is pure geometric distance thresholding
- No idle class or majority-vote smoothing (Phase 2)
- Only two gestures (volume, brightness); the full gesture set (media controls, Mission Control, kill-switch, cursor control) is Phase 3
- Not yet packaged as a background menu-bar app (Phase 4)

## Project structure
gesture-control/
├── hand_tracking_test.py   # Standalone diagnostic script: camera + landmark overlay only.
│                           # Kept intentionally to isolate the camera/MediaPipe pipeline
│                           # from gesture logic when debugging, not leftover clutter.
├── gesture_detector.py     # Pinch detection (is_pinching, is_pinching_middle) and
│                           # anchor-relative drag displacement (get_drag_displacement)
├── actions.py              # System-level volume/brightness read + write (osascript, brightness CLI)
├── main.py                 # The actual application: camera loop + detection + actions, wired together
├── .gitignore
└── README.md

## Setup

Requires Python 3.9–3.12 (MediaPipe does not yet support 3.13+).

```bash
brew install python@3.11
python3.11 -m venv venv
source venv/bin/activate
pip install opencv-python mediapipe==0.10.21
```

> **Why mediapipe is pinned:** Google removed the classic `mediapipe.solutions` API in mediapipe 0.10.31 and later. `0.10.21` is confirmed to still have it.

### Brightness on Apple Silicon

The standard `brew install brightness` CLI fails on Apple Silicon's built-in display with an IOKit error (`error -536870201`). This project uses a Silicon-native fork instead:

```bash
git clone https://github.com/s-age/brightness.git
cd brightness
swift build -c release
rm /opt/homebrew/bin/brightness
cp .build/release/brightness /opt/homebrew/bin/brightness
```

External monitors generally don't support software brightness control at all, a hardware limitation, not specific to this tool. This only controls the Mac's built-in panel.

## Running it

```bash
python main.py
```

- **Pinch thumb + index, then move up/down** to change volume
- **Pinch thumb + middle, then move up/down** to change brightness
- Press `q` to quit

## How the thresholds were picked

Not guessed, measured: printing the live thumb-index distance showed resting-hand values around 0.13–0.23 and pinched values around 0.023–0.038 (MediaPipe's coordinates are normalized 0–1, not pixels). The pinch threshold (`0.07`) sits with margin on both sides of that gap rather than at either edge. Drag `DEAD_ZONE` (`0.02`), `COOLDOWN` (`0.15`s), and step sizes (`STEP_VOLUME=3`, `STEP_BRIGHTNESS=0.05`) were tuned by feel against real usage, not copied from a tutorial.

## Design notes

- **Anchor-relative dragging, not frame-to-frame:** early testing showed that comparing each frame's pinch position to the *previous* frame tied the gesture's usable range to wherever the hand happened to be in the camera's field of view, degrading badly near the frame's edges (MediaPipe extrapolates rather than tracks there). Anchoring to wherever the pinch actually starts keeps the whole gesture inside a small, reliable, comfortable region regardless of camera placement.
- **Reads real system state before acting:** `get_volume()`/`get_brightness()` read the actual current value at startup so the first gesture doesn't snap the system to a guessed baseline.

## Next: Phase 2

Replacing the geometric thresholds with a trained MLP gesture classifier, adding an explicit idle class, and majority-vote smoothing across frames.