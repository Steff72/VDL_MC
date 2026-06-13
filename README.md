# VDL Mini-Challenge: Image Captioning mit Attention

Dieses Repository enthält die Abgabe zur VDL Mini-Challenge für Image Captioning auf Flickr8k. Das finale Notebook vergleicht eine robuste CNN-LSTM-Baseline mit einem CNN-LSTM-Attention-Modell und evaluiert das beste Modell zusätzlich mit Beam Search, CLIPScore und Bootstrap-Konfidenzintervallen.

## Projektüberblick

Ziel ist es, für Bilder aus Flickr8k automatisch Captions zu generieren. Der Hauptvergleich umfasst:

- **Baseline-V2:** ResNet18 frozen + LSTM-Decoder, Bildinformation über `h0/c0` und initiales Image-Token.
- **Attention-Modell:** räumliche CNN-Features + additive Soft-Attention.
- **Bestes Modell:** Attention mit ResNet50 frozen, regularisiert und final auf Train+Val trainiert.
- **Decoding:** Greedy und Beam Search.
- **Evaluation:** Validation/Test Loss, BLEU-1 bis BLEU-4, CLIPScore, qualitative Beispiele, Attention-Heatmaps und Bootstrap-CIs.

Bewusst nicht Teil des Hauptpfads sind Transformer, Unfreezing, Optuna-Läufe und Beam Search während des Trainings.

## Repository-Struktur

```text
.
├── notebooks/
│   └── VDL_Image_Captioning_MiniChallenge_clean.ipynb
├── src/
│   └── vdl_helpers.py
├── data/
│   └── README.md
├── checkpoints/
├── outputs/
├── requirements.txt
└── _quarto.yml
```

Das zentrale Artefakt ist:

```text
notebooks/VDL_Image_Captioning_MiniChallenge_clean.ipynb
```

Der Modellcode bleibt bewusst im Notebook. `src/vdl_helpers.py` enthält nur generische Hilfsfunktionen.

## Setup

Empfohlene Umgebung:

- Python 3.9+
- PyTorch mit MPS-Unterstützung auf Apple Silicon, falls verfügbar
- CPU-Fallback ist implementiert

Installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Das Notebook nutzt lokale Checkpoints und Outputs. Lange Trainingsläufe sind standardmässig deaktiviert.

## Daten

Flickr8k wird nicht mitversioniert. Die Daten müssen lokal unter `data/` abgelegt werden.

Erwartetes Layout:

```text
data/
  Flickr8k/
    Images/
      ...
    captions.txt
```

Weitere Hinweise stehen in [data/README.md](data/README.md).

## Notebook ausführen

Notebook öffnen:

```bash
jupyter notebook notebooks/VDL_Image_Captioning_MiniChallenge_clean.ipynb
```

Trainingsflags im Notebook sind für die Abgabe auf `False` gesetzt:

- `RUN_OVERFIT_TESTS = False`
- `RUN_FULL_TRAINING = False`
- `RUN_OPTIMIZED_TRAINING = False`
- `RUN_ATTENTION_RESNET50 = False`
- `RUN_FINAL_TRAINVAL_TRAINING = False`

Dadurch werden vorhandene Artefakte geladen und keine langen Trainingsläufe automatisch gestartet.

## PDF/HTML mit Quarto

HTML-Render aus dem Projektroot:

```bash
quarto render notebooks/VDL_Image_Captioning_MiniChallenge_clean.ipynb --to html
```

PDF-Render, falls eine LaTeX-Installation verfügbar ist:

```bash
quarto render notebooks/VDL_Image_Captioning_MiniChallenge_clean.ipynb --to pdf
```

## Wichtigste Resultate

Validierungsvergleich der zentralen Modelle:

| Run | best val_loss | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---:|---:|---:|---:|---:|
| Baseline-V2 opt, ResNet18 frozen | 2.7338 | 0.5357 | 0.3588 | 0.2344 | 0.1541 |
| Attention opt, ResNet18 frozen | 2.7137 | 0.5631 | 0.3811 | 0.2526 | 0.1678 |
| Attention opt, ResNet50 frozen | 2.7062 | 0.5915 | 0.4153 | 0.2796 | 0.1842 |

Finale unabhängige Testevaluation des Train+Val-Modells:

| Decoding | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CLIPScore |
|---|---:|---:|---:|---:|---:|
| Greedy | 0.5735 | 0.3976 | 0.2685 | 0.1796 | 27.4567 |
| Beam Search, beam=5 | 0.6526 | 0.4787 | 0.3430 | 0.2447 | 27.6688 |

Bootstrap-Intervalle auf dem Testset:

| Decoding | Metric | Point | 95% CI |
|---|---:|---:|---:|
| Greedy | BLEU-4 | 0.1796 | [0.1675, 0.1923] |
| Beam Search | BLEU-4 | 0.2447 | [0.2287, 0.2613] |
| Greedy | CLIPScore | 27.4567 | [27.1965, 27.7290] |
| Beam Search | CLIPScore | 27.6688 | [27.3914, 27.9375] |

Interpretation: Beam Search verbessert BLEU-4 auf dem Testset stabil. Der CLIPScore-Punktwert ist ebenfalls höher, die Intervalle überlappen aber, daher wird dieser Unterschied vorsichtiger interpretiert.

## Zentrale Artefakte

Wichtige Outputs:

```text
outputs/results_summary.csv
outputs/final_test_summary.csv
outputs/final_test_bootstrap_ci.json
outputs/final_test_bootstrap_ci.csv
outputs/attention_resnet50_final_test_metrics.json
outputs/attention_resnet50_final_test_predictions.json
outputs/attention_resnet50_final_test_examples.json
```

Bestes finales Modell:

```text
checkpoints/attention_resnet50_final_trainval/
  attention-resnet50-final-trainval-epoch=10-train_loss=2.175.ckpt
```

Geprüfte Heatmaps:

```text
outputs/attention_resnet50_final_trainval_heatmaps_checked/
```

## Methodische Hinweise

- Das Testset wurde erst nach Modell-, Regularisierungs- und Decoding-Auswahl verwendet.
- Overfitting-Tests dienen nur als technischer Plausibilitätstest für Modell, Loss und Datenfluss.
- BLEU, CLIPScore und Loss messen unterschiedliche Aspekte und werden gemeinsam mit qualitativen Beispielen interpretiert.
- Bootstrap-CIs quantifizieren Unsicherheit durch die konkrete Testbild-Auswahl, nicht Trainingsvarianz durch andere Seeds.
- W&B wurde bewusst nicht aktiv genutzt; Metriken, Beispiele und Checkpoints werden lokal gespeichert.

## Disclaimer

Dieses Projekt wurde mit Unterstützung von ChatGPT und OpenAI Codex umgesetzt, insbesondere beim Debugging komplexerer Code-Abschnitte.

Das fachliche Fundament für die Modellarchitekturen, Trainingsmethoden und Evaluationsansätze basiert auf den Kursen:

- VDL Deep Dive von Martin Melchior
- Deep Learning Specialization von Andrew Ng
- PyTorch for Deep Learning von Laurence Moroney

Diese Weiterbildungen bildeten die Grundlage für das technische Verständnis und die Umsetzung aller Kernideen in diesem Projekt.
