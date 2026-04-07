# Posture Recognition (Realtime Camera)

Projekt uczy model poziomow postawy (`5`, `6` albo `7`) na podstawie danych z folderu `obrazy`, a nastepnie pokazuje poziom postawy na zywo z kamery.

## 1) Instalacja

```bash
python -m pip install -r requirements.txt
```

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

## Uwagi

- Jesli masz wiecej niz jedna kamere, zmien `--camera-id` (np. `1`).
- Jesli detekcja chwilowo znika, ustaw sie profilem do kamery.
- Przy pierwszym uruchomieniu kamera moze chwile startowac, bo pobiera sie model YOLO-Pose.
