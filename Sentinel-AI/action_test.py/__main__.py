"""
Sentinel AI - Action Recognition Module Entry Point
"""

import sys
import argparse
from pathlib import Path

# Add current folder to path
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

def main():
    parser = argparse.ArgumentParser(description="Sentinel AI - Action Recognition Module")
    parser.add_argument(
        "--mode",
        choices=["check", "model"],
        default="model",
        help="Run environment check ('check') or action recognition model ('model')"
    )
    args, unknown = parser.parse_known_args()

    if args.mode == "check":
        import action_test
    else:
        import video_model

if __name__ == "__main__":
    main()

