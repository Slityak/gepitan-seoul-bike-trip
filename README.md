# Gépi Tanulás Beadandó — Seoul Bike Trip Duration Prediction

Regressziós feladat: kerékpáros utak időtartamának előrejelzése a szöuli kerékpármegosztó rendszer adatai alapján.

**Dataset:** [Seoul Bike Trip Duration Prediction (Kaggle)](https://www.kaggle.com/datasets/saurabhshahane/seoul-bike-trip-duration-prediction)

## Setup

### Lokális (VS Code / Jupyter)

```bash
git clone https://github.com/Slityak/gepitan-seoul-bike-trip.git
cd gepitan-beadando

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt

# Jupyter notebook output strip-elés (commit előtt)
nbstripout --install
```

### Google Colab

Minden notebook tetején van egy "Open in Colab" cella, ami klónozza a repót és mountolja a Drive-ot. Csak nyisd meg a notebookot Colabban, és futtasd az első cellát.

> **Drive setup:** A megosztott `GepiTan_Beadando/` mappának a Drive gyökerében kell lennie, az alábbi struktúrával:
> ```
> GepiTan_Beadando/
> └── data/
>     ├── splits/
>     └── processed/
> ```

## Adat-verziók

| Verzió | Tartalom                                                      |
|--------|---------------------------------------------------------------|
| **v0** | ~200k soros random sample, alap tisztítás, train/test split   |
| **v1** | Teljes 9,6M sor, feature engineering kész, encoding, skálázás |
| **v2** | Opcionális finomítás visszajelzések alapján                   |

## Workflow

1. `git pull` minden munkakezdés előtt
2. Saját branch a nagyobb változtatásokhoz: `git checkout -b decision-tree-tuning`
3. Kód a `src/` alá megy (importálható), notebook csak hívja
4. Eredmények a `results/metrics.csv`-be (közös, append-elhető formátum)
5. Push, PR a main-re

## Reprodukálhatóság

- Random seed: **42** mindenhol (lásd `src/config.py`)
- Csomagverziók fixálva: `requirements.txt`
- Az adat NEM kerül a repóba (lásd `.gitignore`), Drive-on él
