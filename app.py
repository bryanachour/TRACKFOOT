import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

try:
    import torch
    DEFAULT_DEVICE = "0" if torch.cuda.is_available() else "cpu"
except Exception:
    DEFAULT_DEVICE = "cpu"

import config as C
from src.ingest import resolve_input
from src.pipeline import PipelineOptions, run


UPLOAD_DIR = C.ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


CUSTOM_CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
#hero {
  background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%);
  border-radius: 20px;
  padding: 28px 36px;
  margin-bottom: 18px;
  color: #f0fdf4;
  box-shadow: 0 12px 28px -12px rgba(6, 95, 70, 0.4);
}
#hero h1 { color: #f0fdf4 !important; font-size: 2.1rem !important; margin: 0 !important; letter-spacing: -0.02em; }
#hero p { color: #a7f3d0 !important; margin: 6px 0 0 0 !important; font-size: 0.98rem; }
.stat-card {
  background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary);
  border-radius: 14px;
  padding: 16px 20px;
  text-align: center;
}
.stat-card .v { font-size: 1.9rem; font-weight: 700; color: #059669; line-height: 1.1; }
.stat-card .k { font-size: 0.82rem; color: var(--body-text-color-subdued); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.04em; }
.gr-button-primary {
  background: linear-gradient(135deg, #059669, #047857) !important;
  border: none !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em;
}
.gr-button-primary:hover { filter: brightness(1.08); transform: translateY(-1px); }
footer { display: none !important; }
"""


def _stats_html(n_players: int, n_frames: int, n_ball: int, fps: float, took_s: float) -> str:
    cards = [
        (str(n_players), "Joueurs trackés"),
        (str(n_frames), "Frames traitées"),
        (str(n_ball), "Détections ballon"),
        (f"{fps:.1f}", "FPS source"),
        (f"{took_s:.1f}s", "Temps total"),
    ]
    html = '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:14px;">'
    for v, k in cards:
        html += f'<div class="stat-card"><div class="v">{v}</div><div class="k">{k}</div></div>'
    html += "</div>"
    return html


def _empty_stats() -> str:
    return _stats_html(0, 0, 0, 0.0, 0.0)


def process(
    file_path: Optional[str],
    url: str,
    stride: int,
    device: str,
    save_stacked: bool,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Optional[str], Optional[str], Optional[str], List[str], str, Optional[str], str]:
    if not file_path and not (url and url.strip()):
        return None, None, None, [], _empty_stats(), None, "⚠️ Upload un fichier mp4 ou colle une URL."

    started = datetime.now()
    try:
        progress(0.02, desc="téléchargement / copie source")
        src = resolve_input(file_path, url, UPLOAD_DIR)

        run_dir = C.OUTPUT_DIR / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        opts = PipelineOptions(
            source=src,
            output_dir=run_dir,
            device=device or None,
            stride=max(int(stride), 1),
            save_annotated=True,
            save_tactical=True,
            save_stacked=save_stacked,
        )

        def cb(cur: int, total: int, msg: str) -> None:
            frac = cur / max(total, 1)
            progress(min(max(frac, 0.0), 1.0), desc=msg)

        result = run(opts, progress_cb=cb)
    except Exception as e:
        return None, None, None, [], _empty_stats(), None, f"❌ Erreur : {type(e).__name__} — {e}"

    took = (datetime.now() - started).total_seconds()
    stats = _stats_html(result.n_players, result.n_frames, result.n_ball_points, result.fps, took)

    gallery = [str(p) for p in result.heatmap_files]

    return (
        str(result.annotated) if result.annotated else None,
        str(result.tactical) if result.tactical else None,
        str(result.stacked) if result.stacked else None,
        gallery,
        stats,
        str(result.trajectories_json) if result.trajectories_json else None,
        f"✅ Analyse terminée en {took:.1f}s — {result.n_frames} frames, {result.n_players} joueurs trackés.",
    )


def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="TRACKFOOT") as demo:
        gr.HTML(
            """
            <div id="hero">
              <h1>🏟️ TRACKFOOT</h1>
              <p>Détection · Tracking · Vue tactique 2D · Heatmaps — pour vidéos football amateur</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(
                    label="📁 Upload vidéo",
                    file_types=["video", ".mp4", ".mov", ".mkv", ".avi"],
                    type="filepath",
                )
            with gr.Column(scale=1):
                url_in = gr.Textbox(
                    label="🔗 ou URL vidéo",
                    placeholder="https://...mp4, lien Veo public, YouTube, ...",
                    lines=1,
                )
                gr.Markdown(
                    "<span style='font-size:0.82rem;color:#64748b'>Géré via yt-dlp. Pour un Veo privé, télécharge le mp4 puis upload.</span>"
                )

        with gr.Row():
            stride = gr.Slider(1, 5, value=1, step=1, label="Stride (1 = toutes les frames)")
            device = gr.Textbox(value=DEFAULT_DEVICE, label="Device", scale=1)
            save_stacked = gr.Checkbox(value=True, label="Vidéo combinée (caméra + tactique)")

        run_btn = gr.Button("▶ Lancer l'analyse", variant="primary", size="lg")
        status = gr.Markdown("")

        stats_html = gr.HTML(_empty_stats())

        with gr.Tabs():
            with gr.Tab("🎥 Caméra annotée"):
                cam_video = gr.Video(label=None, interactive=False)
            with gr.Tab("🗺️ Vue tactique"):
                tac_video = gr.Video(label=None, interactive=False)
            with gr.Tab("🎬 Combiné"):
                stk_video = gr.Video(label=None, interactive=False)
            with gr.Tab("🔥 Heatmaps"):
                heatmaps = gr.Gallery(
                    label=None,
                    columns=3,
                    rows=2,
                    height=520,
                    object_fit="contain",
                    show_label=False,
                )
            with gr.Tab("📊 Trajectoires JSON"):
                json_file = gr.File(label="Télécharger trajectoires.json", interactive=False)

        run_btn.click(
            fn=process,
            inputs=[file_in, url_in, stride, device, save_stacked],
            outputs=[cam_video, tac_video, stk_video, heatmaps, stats_html, json_file, status],
        )

        gr.Markdown(
            "<div style='text-align:center;margin-top:16px;color:#94a3b8;font-size:0.85rem'>"
            "YOLOv8 · ByteTrack · OpenCV homography · supervision — "
            "<a href='https://github.com/roboflow/sports' style='color:#10b981'>roboflow/sports</a></div>"
        )

    return demo


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="public link via Gradio (Colab/serveur distant)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    app = build_app()
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
    )
