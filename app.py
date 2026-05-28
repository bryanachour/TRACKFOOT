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
from src.ingest import resolve_input, hint_for_url
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


def _players_table(summary: Optional[dict]) -> str:
    if not summary or not summary.get("players"):
        return "<p style='color:#94a3b8'>Pas encore de données — lance une analyse.</p>"
    rows = ""
    for p in summary["players"][:30]:
        team = p.get("team")
        if team == 0:
            badge = "<span style='background:#EF4444;color:white;padding:2px 8px;border-radius:6px;font-size:0.75rem'>A</span>"
        elif team == 1:
            badge = "<span style='background:#3B82F6;color:white;padding:2px 8px;border-radius:6px;font-size:0.75rem'>B</span>"
        else:
            badge = "<span style='background:#94A3B8;color:white;padding:2px 8px;border-radius:6px;font-size:0.75rem'>—</span>"
        poss_pct = f"{p['possession_ratio'] * 100:.1f}%"
        rows += (
            f"<tr>"
            f"<td style='padding:8px 12px;font-weight:600'>#{p['tracker_id']}</td>"
            f"<td style='padding:8px 12px'>{badge}</td>"
            f"<td style='padding:8px 12px;text-align:right'>{p['distance_m']} m</td>"
            f"<td style='padding:8px 12px;text-align:right'>{p['speed_avg_kmh']} km/h</td>"
            f"<td style='padding:8px 12px;text-align:right'>{p['speed_max_kmh']} km/h</td>"
            f"<td style='padding:8px 12px;text-align:right'>{poss_pct}</td>"
            f"<td style='padding:8px 12px;text-align:right;color:#94a3b8'>{p['n_frames']}</td>"
            f"</tr>"
        )
    return (
        "<div style='overflow:auto;border-radius:12px;border:1px solid var(--border-color-primary)'>"
        "<table style='width:100%;border-collapse:collapse;font-size:0.9rem'>"
        "<thead style='background:var(--block-background-fill);position:sticky;top:0'>"
        "<tr>"
        "<th style='padding:10px 12px;text-align:left'>ID</th>"
        "<th style='padding:10px 12px;text-align:left'>Équipe</th>"
        "<th style='padding:10px 12px;text-align:right'>Distance</th>"
        "<th style='padding:10px 12px;text-align:right'>V moy</th>"
        "<th style='padding:10px 12px;text-align:right'>V max</th>"
        "<th style='padding:10px 12px;text-align:right'>Possession</th>"
        "<th style='padding:10px 12px;text-align:right'>Frames</th>"
        "</tr></thead><tbody>"
        f"{rows}"
        "</tbody></table></div>"
        "<p style='font-size:0.78rem;color:#94a3b8;margin-top:8px'>Trié par distance parcourue. Possession = % des frames avec ballon détecté où ce joueur est le plus proche (rayon 2.5m).</p>"
    )


def process(
    file_path: Optional[str],
    url: str,
    stride: int,
    device: str,
    save_stacked: bool,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Optional[str], Optional[str], Optional[str], List[str], str, Optional[str], str]:
    if not file_path and not (url and url.strip()):
        return None, None, None, [], _empty_stats(), None, None, _players_table(None), "⚠️ Upload un fichier mp4 ou colle une URL."

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
        return None, None, None, [], _empty_stats(), None, None, _players_table(None), f"❌ Erreur : {type(e).__name__} — {e}"

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
        str(result.stats_json) if result.stats_json else None,
        _players_table(result.stats_summary),
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
                    placeholder="https://app.veo.co/matches/...  ou  https://...mp4  ou  YouTube",
                    lines=1,
                )
                url_hint = gr.Markdown(
                    "<span style='font-size:0.82rem;color:#64748b'>Veo, mp4 direct, YouTube — géré nativement par yt-dlp.</span>"
                )

                def _on_url_change(u: str) -> str:
                    hint = hint_for_url(u or "")
                    if hint:
                        return f"<span style='font-size:0.82rem'>{hint}</span>"
                    return "<span style='font-size:0.82rem;color:#64748b'>Veo, mp4 direct, YouTube — géré nativement par yt-dlp.</span>"

                url_in.change(fn=_on_url_change, inputs=url_in, outputs=url_hint)

        with gr.Row():
            stride = gr.Slider(1, 15, value=5, step=1, label="Stride (5 = 1 frame sur 5 — recommandé)")
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
            with gr.Tab("📈 Stats joueurs"):
                players_table = gr.HTML(_players_table(None))
                with gr.Row():
                    stats_file = gr.File(label="stats.json", interactive=False)
                    json_file = gr.File(label="trajectoires.json", interactive=False)

        run_btn.click(
            fn=process,
            inputs=[file_in, url_in, stride, device, save_stacked],
            outputs=[cam_video, tac_video, stk_video, heatmaps, stats_html, json_file, stats_file, players_table, status],
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
