
from pathlib import Path
import shutil

APP_PATH = Path("fight_engine/app.py")

if not APP_PATH.exists():
    raise FileNotFoundError(
        "app.py not found. Run this script from the Sentinel-AI folder."
    )

source = APP_PATH.read_text(encoding="utf-8")

if "LIVE_CAMERA_INTEGRATION" in source:
    print("Live camera integration already exists.")
    raise SystemExit(0)

# 1. Add live camera import
import_anchor = "from video_utils import ensure_browser_playable"

if import_anchor not in source:
    raise RuntimeError("Could not find the video_utils import.")

source = source.replace(
    import_anchor,
    import_anchor + "\nfrom live_camera import live_camera_predictions",
    1,
)

# 2. Add live prediction function
function_anchor = (
    "# ═════════════════════════ category dropdown "
    "════════════════════════"
)

if function_anchor not in source:
    raise RuntimeError("Could not find the category dropdown section.")

live_function = '''
# LIVE_CAMERA_INTEGRATION

def predict_live(stream_url, model_display_name):
    """Run continuous violence detection from an IP camera."""

    if not stream_url or not stream_url.strip():
        raise gr.Error("Please enter your camera stream URL.")

    name = _resolve_model_name(model_display_name)

    try:
        model, spec = LOADER.load(name)

        yield from live_camera_predictions(
            stream_url.strip(),
            model,
            LOADER.device,
            backbone=spec.backbone,
            num_frames=spec.num_frames,
            frame_size=spec.frame_size,
            threshold=spec.threshold,
        )

    except Exception as exc:
        logger.exception("Live camera failed.")
        raise gr.Error(f"Live camera error: {exc}") from exc


'''

source = source.replace(
    function_anchor,
    live_function + function_anchor,
    1,
)

# 3. Add live camera interface
ui_anchor = '            with gr.Column(elem_classes=["svd-left-col"]):'

if ui_anchor not in source:
    raise RuntimeError("Could not find the left dashboard column.")

live_ui = '''            with gr.Column(elem_classes=["svd-left-col"]):

                # LIVE CAMERA CONTROLS
                with gr.Accordion("Live Camera Monitoring", open=True):
                    live_url = gr.Textbox(
                        label="Android IP Webcam URL",
                        placeholder="http://PHONE_IP:8080/video",
                    )

                    live_preview = gr.Image(
                        label="Live Camera Feed",
                        type="numpy",
                        height=360,
                    )

                    live_risk = gr.Textbox(
                        label="Current Violence Risk",
                        interactive=False,
                    )

                    live_confidence = gr.Number(
                        label="Violence Probability (0-1)",
                        precision=3,
                        interactive=False,
                    )

                    live_status = gr.Textbox(
                        label="Monitoring Status",
                        value="Not started",
                        interactive=False,
                    )

                    with gr.Row():
                        live_start = gr.Button(
                            "Start Monitoring",
                            variant="primary",
                        )

                        live_stop = gr.Button(
                            "Stop Monitoring",
                            variant="stop",
                        )

'''

source = source.replace(ui_anchor, live_ui, 1)

# 4. Add event wiring immediately before return demo
wiring_anchor = "    return demo"

if wiring_anchor not in source:
    raise RuntimeError("Could not find return demo.")

live_wiring = '''        # LIVE CAMERA EVENT WIRING
        live_event = live_start.click(
            fn=predict_live,
            inputs=[live_url, model_dropdown],
            outputs=[
                live_preview,
                live_risk,
                live_confidence,
                live_status,
            ],
            show_progress="hidden",
        )

        live_stop.click(
            fn=lambda: "Stopped",
            inputs=None,
            outputs=live_status,
            cancels=[live_event],
            show_progress="hidden",
        )

'''

source = source.replace(
    wiring_anchor,
    live_wiring + wiring_anchor,
    1,
)

# 5. Back up original app.py
backup_path = APP_PATH.with_name("app.py.before_live.bak")

if not backup_path.exists():
    shutil.copy2(APP_PATH, backup_path)

# 6. Save updated app.py
APP_PATH.write_text(source, encoding="utf-8")

print("SUCCESS: Live camera integration added!")
print(f"Updated file: {APP_PATH}")
print(f"Backup file: {backup_path}")
print("Next: Run the Python syntax check.")
