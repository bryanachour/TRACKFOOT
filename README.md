# TRACKFOOT

Pipeline Computer Vision pour vidéos de football amateur : détection (YOLOv8) + tracking (ByteTrack) + projection homographique vers vue tactique 2D + heatmaps + export JSON des trajectoires.

## Structure

```
TRACKFOOT/
├── app.py                        # front Gradio (upload + URL + résultats)
├── main.py                       # CLI entrypoint
├── config.py                     # chemins, seuils, dimensions terrain
├── requirements.txt
├── src/
│   ├── detection.py              # YOLOv8 inference (joueurs + ballon)
│   ├── tracking.py               # ByteTrack via supervision
│   ├── homography.py             # estimation H + lissage EMA
│   ├── pitch.py                  # SoccerPitchConfiguration + rendu terrain
│   ├── visualization.py          # annotators + vue tactique
│   ├── trajectories.py           # store JSON + heatmaps
│   ├── ingest.py                 # upload / URL (yt-dlp)
│   └── pipeline.py               # orchestrateur
├── scripts/
│   └── download_weights.py       # téléchargement poids Roboflow
├── notebooks/
│   └── trackfoot_colab.ipynb     # notebook Colab clé-en-main
├── weights/                      # poids YOLOv8 (.pt) — gitignored
└── output/                       # vidéos + JSON + heatmaps générés
```

## Modèles utilisés (Roboflow Universe)

| Tâche                  | Modèle Roboflow                          | Classes / sortie                    |
|------------------------|------------------------------------------|-------------------------------------|
| Détection joueurs      | `football-players-detection-3zvbc/v11`   | `ball, goalkeeper, player, referee` |
| Détection terrain      | `football-field-detection-f07vi/v14`     | 32 keypoints (cf. `src/pitch.py`)   |
| Détection ballon (HR)  | `football-ball-detection-rejhg/v2`       | `ball` (modèle spécialisé)          |

L'ordre des 32 keypoints du modèle terrain doit correspondre à `SoccerPitchConfiguration.vertices`. Si vous entraînez votre propre modèle keypoints, calez-vous sur l'ordre défini dans [src/pitch.py](src/pitch.py).

## Front Gradio — la voie principale

Une interface web avec upload de mp4, champ URL (yt-dlp), barre de progression live et résultats en onglets (caméra annotée, vue tactique, combiné, heatmaps, JSON).

```bash
python app.py            # local → http://localhost:7860
python app.py --share    # tunnel public gradio.live (Colab / serveur distant)
```

Dans Colab : exécute la cellule `!python app.py --share` du notebook. Une URL `https://xxxx.gradio.live` apparaît dans la sortie, partageable ~72h.

**Inputs** : upload mp4/mov/mkv/avi *ou* URL (lien direct mp4, YouTube, lien Veo public). Veo privé → télécharge mp4 manuellement puis upload.

## Démarrage rapide — Colab

Ouvre [notebooks/trackfoot_colab.ipynb](notebooks/trackfoot_colab.ipynb) (`Runtime > T4 GPU`), exécute les cellules : install → download weights → `app.py --share`.

## Mode CLI (sans front)

```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY=xxxx                      # clé gratuite sur app.roboflow.com
python scripts/download_weights.py
python main.py --source path/to/match.mp4 --device 0
```

Sorties dans `output/run_YYYYMMDD_HHMMSS/` :
- `annotated.mp4` — vidéo annotée (bbox ellipses + IDs + traces)
- `tactical.mp4` — vue tactique 2D vue du dessus
- `stacked.mp4` — caméra + tactique empilées
- `trajectories.json` — `{players: {id: [{frame,x,y}]}, ball: [...]}`
- `heatmaps/player_NNN.png` + `heatmaps/ball.png`

## Options CLI

```bash
python main.py --source video.mp4 \
  --output output/ \
  --device 0 \              # cuda:0 ; "cpu" pour fallback
  --stride 1 \              # >1 pour sous-échantillonner
  --no-stacked              # désactiver une sortie
```

## Architecture du pipeline

```
frame ─► YOLOv8 player ─► split(player/keeper/referee/ball)
           │                  │
           │                  └─► ByteTrack ─► tracker_id persistant
           │
       YOLOv8 pitch ─► 32 keypoints ─► findHomography (RANSAC) ─► EMA smoothing
                                            │
       bbox bottom-centre ─► perspectiveTransform ─► (x,y) en cm sur terrain 105×68m
                                            │
                                            ├─► TrajectoryStore (JSON)
                                            ├─► tactical view (cv2)
                                            └─► heatmap (gaussian KDE)
```

## Détails techniques

**Détection ballon** — le ballon est petit (~20px à 1080p depuis tribune). Le `BallDetector` utilise `sv.InferenceSlicer` (slicing 640×640 avec overlap) pour gagner ~10–15 pts de recall sur ce cas. Désactivable en supprimant `football-ball-detection.pt`.

**Tracking** — ByteTrack via `supervision`. Paramètres dans [config.py](config.py) : `TRACKER_TRACK_BUFFER=60` (≈2s de tolérance occlusion), `TRACKER_MATCH_THRESH=0.85`. Pour des occlusions plus longues / changements de plan brusques, basculer sur BoT-SORT (remplacer dans [src/tracking.py](src/tracking.py)).

**Homographie** — keypoints détectés `>` 4 → `cv2.findHomography(... RANSAC)`. Au moins 6 inliers requis sinon on conserve la dernière `H` valide. EMA (`α=0.85`) lisse les sauts inter-frames. Point projeté = milieu-bas de la bbox (approximation pieds du joueur).

**Vue tactique** — `SoccerPitchConfiguration` en cm (12000×7000), rendu à `1050×680px` par défaut. Joueurs colorés par `tracker_id` via la palette `supervision`.

**Heatmaps** — accumulation sur grille pixel + `gaussian_filter(σ=12)`, blending alpha sur le terrain.

## Limites V1

- Pas de classification d'équipe — tous les joueurs partagent la même palette indexée par ID. Pour distinguer les équipes : ajouter un clustering couleur des maillots (HSV + k-means k=2) sur le crop bbox.
- Pas de ré-identification après occultation longue (>60 frames) — l'ID peut changer.
- L'homographie suppose un terrain plat visible avec ≥4 keypoints détectés. Sur des plans serrés (gros plan caméra) la projection sera instable.

## Pistes V2

- Team classification (k-means HSV sur les jersey crops)
- Possession ballon (joueur le plus proche du ballon projeté en coord terrain)
- Détection événements (passes, tirs) à partir des trajectoires
- Re-ID avec embedding visuel (OSNet, ReID-Strong-Baseline) pour gérer les longues occultations
- Inférence ONNX/TensorRT pour atteindre les 30 fps temps réel

## Références

- Roboflow Sports : https://github.com/roboflow/sports
- Ultralytics YOLOv8 : https://github.com/ultralytics/ultralytics
- ByteTrack : https://github.com/ifzhang/ByteTrack
- BoT-SORT : https://github.com/NirAharon/BoT-SORT
- SoccerNet : https://github.com/SoccerNet
