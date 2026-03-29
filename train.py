# train.py — version générique tous projets
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from config import MODEL_PATH, PROJECT_NAME
from queries import get_data_ml
from mon_outillage import (
    pretraitement_data,
    cleaner_outlier,
    pipeline_nettoyage_modele,
    compare_modele,
    evaluer_classification,
    valider_stabilite
)
from mlflow_tracker import tracker_mlflow


# ══════════════════════════════════════════════
#   CONFIGURATION — seule partie à modifier
#   pour chaque nouveau projet
# ══════════════════════════════════════════════

CONFIG = {
    # Données
    "table"        : "passagers",   # ex: "clients", "ventes", "employes"
    "target"       : "survived",    # ex: "churn", "prix", "score"
    "drop_cols"    : [],            # ex: ["nom", "id", "ticket"]
    "outlier_cols" : ["fare", "age"],# ex: ["salaire", "montant"]

    # Split
    "test_size"    : 0.2,
    "random_state" : 42,

    # Modèle par défaut
    "modele"       : RandomForestClassifier,
    "params"       : {
        "n_estimators": 200,
        "max_depth"   : 4,
        "random_state": 42
    }
}


# ══════════════════════════════════════════════
#   PIPELINE — ne jamais modifier cette partie
# ══════════════════════════════════════════════

def run_training(config=CONFIG):

    print("\n" + "="*50)
    print(f"  TRAINING — {PROJECT_NAME}")
    print("="*50)

    # ── 1. Charger depuis PostgreSQL ─────────────
    print("\n📥 [1/8] Chargement des données...")
    X, y = get_data_ml(
        table     = config["table"],
        target    = config["target"],
        drop_cols = config["drop_cols"]
    )

    # ── 2. Nettoyage ─────────────────────────────
    print("\n🧹 [2/8] Nettoyage...")
    df = pretraitement_data(X.copy())

    for col in config["outlier_cols"]:
        if col in df.columns:
            avant = len(df)
            df    = cleaner_outlier(df, col)
            print(f"   {col} : {avant - len(df)} outliers supprimés")

    # Aligner y avec df après suppression outliers
    y = y.loc[df.index]

    # ── 3. Split train/test ───────────────────────
    print("\n✂️  [3/8] Split train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size    = config["test_size"],
        random_state = config["random_state"]
    )
    print(f"   Train : {len(X_train)} | Test : {len(X_test)}")

    # ── 4. Comparaison des modèles ────────────────
    print("\n🏆 [4/8] Comparaison des modèles...")
    pipeline_base = pipeline_nettoyage_modele(X_train)
    resultats     = compare_modele(X_train, X_test, y_train, y_test, pipeline_base)
    print(resultats.to_string(index=False))

    # ── 5. Entraînement du modèle choisi ─────────
    print(f"\n🚀 [5/8] Entraînement {config['modele'].__name__}...")
    modele   = config["modele"](**config["params"])
    pipeline = pipeline_nettoyage_modele(X_train)
    pipeline.set_params(regressor=modele)

    # ── 6. MLflow tracking ────────────────────────
    print("\n📊 [6/8] Tracking MLflow...")
    modele_final, y_pred = tracker_mlflow(
        model   = pipeline,
        params  = config["params"],
        X_train = X_train,
        X_test  = X_test,
        y_train = y_train,
        y_test  = y_test
    )

    # ── 7. Evaluation ─────────────────────────────
    print("\n📈 [7/8] Evaluation...")
    evaluer_classification(y_test, y_pred, config["modele"].__name__)

    # ── 8. Sauvegarde ─────────────────────────────
    print(f"\n💾 [8/8] Sauvegarde → {MODEL_PATH}")
    joblib.dump(modele_final, MODEL_PATH)
    print(f"✅ Modèle sauvegardé : {MODEL_PATH}")

    print("\n" + "="*50)
    print("  TRAINING TERMINÉ !")
    print("="*50 + "\n")

    return modele_final


if __name__ == "__main__":
    run_training()