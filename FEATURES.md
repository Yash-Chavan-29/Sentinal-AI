# Sentinel-AI: Rebuild Architecture & 10 Core Features

## Project Overview
Sentinel-AI is a next-generation, real-time AI surveillance and threat-monitoring platform focused exclusively on live camera streams (Webcam, Mobile IP Camera, RTSP). All legacy video upload and redundant fire detection modules have been phased out to deliver high-performance, real-time altercation and violence detection.

---

## 5 Features to Minimize Backlogs & Technical Debt

### 1. Zero-Lag Asynchronous Frame Processing Pipeline
- **Problem Solved:** Legacy sequential processing caused video lag, dropped frames, and memory bloat on live camera feeds.
- **Solution:** Multi-threaded producer-consumer architecture using OpenCV capture threads, bounded queues (`Queue(maxsize=2)`), and frame-dropping logic so AI inference never blocks the real-time video display.

### 2. PyTorch 2.x Safe Globals & Robust 5D Tensor Preprocessing
- **Problem Solved:** `torch.load` failed on PyTorch 2.6+ due to strict safe global requirements (`numpy.dtypes.Float64DType`), and tensor normalization threw non-singleton dimension mismatches `(16 != 3)`.
- **Solution:** Native PyTorch safe globals registration for NumPy types with weights-only compatibility, paired with a bulletproof `(1, C, T, H, W)` temporal tensor formatter.

### 3. ByteTrack Precision Person Tracking & Minimalist HUD
- **Problem Solved:** Bounding boxes flickered or disappeared when people crossed paths, and oversized fonts cluttered the frame.
- **Solution:** Integrated ByteTrack/BoT-SORT tracking with Exponential Moving Average (EMA) box smoothing, ultra-crisp 1.5px adaptive bounding borders, and micro confidence tags.

### 4. Dual-Stage Spatial-Temporal False-Positive Filter
- **Problem Solved:** Fast single-person movement (running, jumping, waving) triggered false fight alarms.
- **Solution:** Dual-stage validation: an altercation alarm only fires when multiple overlapping/proximate person bounding boxes are detected alongside sustained high X3D action confidence across consecutive temporal windows.

### 5. Resilient Camera Auto-Reconnect Engine
- **Problem Solved:** Mobile camera feeds (DroidCam/IP Webcam) disconnect intermittently over Wi-Fi, freezing or crashing the app.
- **Solution:** Automatic background heartbeat with exponential backoff reconnect logic that maintains UI state and resumes streaming without manual reload.

---

## 5 Features to Upgrade the Surveillance Website & UI

### 6. Modern Glassmorphic Dark-Mode Surveillance Dashboard
- **Feature:** Tactical command-center interface featuring high-contrast dark theme, live status pills, camera metadata overlays, and responsive telemetry panels.

### 7. Universal Camera Source Switcher (Webcam / IP Cam / RTSP)
- **Feature:** Single-click switching between local USB/webcams and remote mobile IP cameras (HTTP/RTSP/DroidCam stream URLs) with live connection validation.

### 8. Live Incident Alert Feed & Automatic Snapshot Evidence Vault
- **Feature:** Whenever an altercation is verified, Sentinel-AI instantly logs the event with timestamp, confidence, and captures a high-resolution annotated screenshot saved into an evidence gallery.

### 9. Real-Time Telemetry & Threat Probability HUD
- **Feature:** Live telemetry showing real-time processing FPS, model inference latency (ms), active person count, and an interactive colored gauge tracking threat severity (Green: Normal, Yellow: Suspicious, Red: Altercation).

### 10. Configurable Audio Siren & Visual Flash Alarm System
- **Feature:** Client-side audible alarm siren with toggle control, complemented by a pulsing red HUD perimeter warning whenever high-threat altercations are detected.
