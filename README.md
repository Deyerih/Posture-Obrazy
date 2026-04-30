# Posture Recognition (Realtime Camera)

Projekt uczy model poziomow postawy (`5`, `6` albo `7`) na podstawie danych z folderu `obrazy`, a nastepnie pokazuje poziom postawy na zywo z kamery.

## 1) Instalacja

```bash
python -m pip install -r requirements.txt
```

## Szybka kontrola jakosci

```bash
pytest -q
```

Smoke-testy sprawdzaja kluczowe obliczenia i obsluge granicznych przypadkow (np. puste wejscia, brak obrazow, niepoprawne rozszerzenia).

## 2) Trenowanie modelu poziomow

Domyslnie: `7` poziomow.

```bash
python train_posture_levels.py --dataset-dir obrazy --output-dir model --levels 7
```

Mozliwe poziomy: `5`, `6`, `7`.

## 3) Uruchomienie kamery

```bash
python live_posture_camera.py --model-path model/posture_level_model.joblib --camera-id 0
```

## 4) Predykcja dla obrazow testowych

Jedno zdjecie:

```bash
python predict_posture_images.py --input test_images/sample.jpg --model-path model/posture_level_model.joblib
```

Caly folder zdjec:

```bash
python predict_posture_images.py --input test_images --model-path model/posture_level_model.joblib
```

Z zapisem obrazow z naniesionym wynikiem:

```bash
python predict_posture_images.py --input test_images --save-visuals --output-dir outputs
```

## 5) Benchmark wielu modeli (YOLO + MediaPipe)

Porownanie kilku modeli na tym samym zbiorze obrazow z logiem wynikow (`CSV + JSON`):

```bash
python benchmark_pose_models.py \
  --input test_images \
  --model-path model/posture_level_model.joblib \
  --engines yolo,mediapipe \
  --yolo-models yolo11n-pose.pt,yolo11s-pose.pt,yolo11m-pose.pt \
  --mediapipe-complexities 0,1,2 \
  --log-dir benchmark_logs
```

Po uruchomieniu powstana pliki:

- `benchmark_logs/<timestamp>_summary.csv` - agregaty per model (latency, detection rate, valid posture rate, errors)
- `benchmark_logs/<timestamp>_details.json` - szczegolowe rekordy per obraz i model

Najwazniejsze pola w logu szczegolowym:

- `engine`, `model_name`, `image_path`
- `detected_pose` (`0` lub `1`)
- `inference_ms`
- `posture_level` (`null`, jesli brak stabilnej detekcji)
- `error` (tekst bledu, jesli wystapil)

W oknie kamery zobaczysz:

- `Poziom postawy: X/Y (...)` - aktualna ocena postawy
- kolor od czerwonego (slabo) do zielonego (dobrze)
- `Q` aby zakonczyc

## Jak to dziala

1. Z etykiet YOLO-Pose (`obrazy/labels`) pobierane sa 4 punkty kluczowe.
2. Liczone sa cechy biomechaniczne (odchylenia i katy od pionu).
3. `KMeans` dzieli probki na `N` poziomow postawy.
4. W czasie rzeczywistym YOLO-Pose wykrywa punkty ciala z kamery.
5. Cechy sa klasyfikowane do jednego z poziomow i od razu wyswietlane.

## Uwagi runners.append(

                MediaPipePoseRunner(
                    model_path=args.mediapipe_model_path,
                    complexity=int(complexity_str),
                    min_detection_confidence=args.mediapipe_min_detection_conf,
                    min_tracking_confidence=args.mediapipe_min_tracking_conf,
                )
            )

- Jesli masz wiecej niz jedna kamere, zmien `--camera-id` (np. `1`).
- Jesli detekcja chwilowo znika, ustaw sie profilem do kamery.
- Przy pierwszym uruchomieniu kamera moze chwile startowac, bo pobiera sie model YOLO-Pose.
