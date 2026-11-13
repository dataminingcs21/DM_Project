import joblib
import numpy as np
from functools import lru_cache
from pathlib import Path

# On remonte de 3 niveaux : ml_loader.py -> detector/ -> ids_project/ -> ROOT
ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / "cache"

# Liste EXACTE des 13 classes après la fusion (ordre alphabétique/index du modèle)
MULTICLASS_NAMES = [
    'BENIGN', 'FTP-Patator', 'SSH-Patator', 'DoS slowloris',
    'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye', 'Heartbleed',
    'Web Attack', 'Infiltration', 'Bot', 'DDoS', 'PortScan'
]

@lru_cache(maxsize=1)
def load_binary_model():
    return joblib.load(MODELS_DIR / "binary_rf.joblib")

@lru_cache(maxsize=1)
def load_multiclass_model():
    return joblib.load(MODELS_DIR / "multiclass_rf.joblib")

@lru_cache(maxsize=1)
def load_scaler():
    return joblib.load(CACHE_DIR / "scaler.joblib")

@lru_cache(maxsize=1)
def load_feature_names():
    return joblib.load(CACHE_DIR / "feature_names.joblib")

def predict_single(features: list[float], task: str) -> dict:
    """
    Prend une liste de 66 float, applique le scaler, et prédit.
    Retourne un dictionnaire formaté pour l'interface.
    """
    task_name = (task or "").strip().lower()
    if task_name not in ("binary", "multiclass"):
        raise ValueError("Task must be 'binary' or 'multiclass'")

    if len(features) != 66:
        raise ValueError(f"Expected 66 features, got {len(features)}")

    # 1. Préparation des données (DataFrame pour préserver les noms de features et éviter les warnings)
    import pandas as pd
    feature_names = list(load_feature_names())
    X_df = pd.DataFrame([features], columns=feature_names, dtype=np.float32)
    X_scaled = load_scaler().transform(X_df)

    # 2. Prédiction
    if task_name == "binary":
        model = load_binary_model()
        proba = model.predict_proba(X_scaled)[0]
        pred_idx = int(model.predict(X_scaled)[0])
        pred_label = "ATTACK" if pred_idx == 1 else "BENIGN"
        prob_val = float(proba[pred_idx])

        # Top classes pour affichage unifié
        top_binary = sorted(
            [
                {"class": "ATTACK", "probability": float(proba[1]), "percent": round(float(proba[1]) * 100, 2)},
                {"class": "BENIGN", "probability": float(proba[0]), "percent": round(float(proba[0]) * 100, 2)},
            ],
            key=lambda x: x["probability"],
            reverse=True,
        )

        return {
            "task": "binary",
            "prediction": pred_label,
            "probability": prob_val,
            "percent": round(prob_val * 100, 2),
            "probabilities": {
                "BENIGN": float(proba[0]),
                "ATTACK": float(proba[1]),
            },
            "probabilities_pct": {
                "BENIGN": round(float(proba[0]) * 100, 2),
                "ATTACK": round(float(proba[1]) * 100, 2),
            },
            "top3": top_binary,
        }

    else:  # multiclass
        model = load_multiclass_model()
        proba = model.predict_proba(X_scaled)[0]

        # Récupérer le top 3
        top3_idx = np.argsort(proba)[-3:][::-1]  # Descendant
        top3 = [
            {
                "class": MULTICLASS_NAMES[i],
                "probability": float(proba[i]),
                "percent": round(float(proba[i]) * 100, 2),
            }
            for i in top3_idx
        ]

        return {
            "task": "multiclass",
            "prediction": top3[0]["class"],
            "probability": top3[0]["probability"],
            "percent": top3[0]["percent"],
            "top3": top3,
        }
