
import sys
import argparse
from pathlib import Path

import torch

# Make the official model files importable
ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "fight_engine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

try:
    from fight_engine.model_loader import ViolenceModelLoader
    from fight_engine.temporal_analyzer import analyze_video_timeline
except ImportError:
    from model_loader import ViolenceModelLoader
    from temporal_analyzer import analyze_video_timeline


def main():
    parser = argparse.ArgumentParser(
        description="AI violence detection from an MP4 video"
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=str(ROOT / "videos" / "sample_fixed.mp4"),
        help="Path to an MP4 video (default: videos/sample_fixed.mp4)"
    )
    args = parser.parse_args()

    video_path = Path(args.video)

    if not video_path.is_file():
        print(f"ERROR: Video not found: {video_path}")
        return

    device = torch.device("cpu")

    print("\nLoading X3D-M violence detection model...")
    loader = ViolenceModelLoader(device=device, max_cached=1)
    model, spec = loader.load("final_x3d_realtime")

    print("Model loaded successfully!")
    print(f"Model: {spec.display_name}")
    print(f"Device: {device}")
    print("\nAnalyzing video...\n")

    result = analyze_video_timeline(
        str(video_path),
        model,
        device,
        backbone=spec.backbone,
        num_frames=spec.num_frames,
        frame_size=spec.frame_size,
        threshold=spec.threshold,
        num_windows=6,
        overlap=0.5,
        consecutive_required=2,
    )

    print("\n========== DETECTION RESULT ==========")
    print(f"Video: {video_path.name}")
    print(f"Duration: {result.duration_sec:.2f} seconds")
    print(f"Status: {result.state} {result.state_emoji}")
    print(
        f"Violence probability: "
        f"{result.aggregated_violent_prob * 100:.2f}%"
    )
    print(
        f"Non-violence probability: "
        f"{result.aggregated_non_violent_prob * 100:.2f}%"
    )
    print(f"Consecutive warning windows: {result.consecutive_count}")

    print("\n========== WINDOW SCORES ==========")
    for window in result.windows:
        score = window.violent_prob * 100
        print(
            f"{window.start_sec:.1f}s - {window.end_sec:.1f}s"
            f" | Violence: {score:.2f}%"
        )

    print(
        f"\nInference time: "
        f"{result.inference_ms_total / 1000:.2f} seconds"
    )
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
