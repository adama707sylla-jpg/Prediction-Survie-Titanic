# queries.py — version générique tous projets
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from config import DATABASE_URL

engine = create_engine(DATABASE_URL)

# ══════════════════════════════════════════════
#   FONCTIONS GÉNÉRIQUES — marchent sur
#   n'importe quelle table, n'importe quel projet
# ══════════════════════════════════════════════

def get_data_ml(table, target=None, drop_cols=None, dropna=True):

    query = f"SELECT * FROM {table}"
    df = pd.read_sql(query, engine)

    # Supprimer les colonnes inutiles
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Supprimer les lignes vides
    if dropna:
        avant = len(df)
        df = df.dropna()
        print(f"⚠️  {avant - len(df)} lignes supprimées (valeurs manquantes)")

    # Séparer features et cible
    if target and target in df.columns:
        X = df.drop(columns=[target])
        y = df[target]
        print(f"✅ {len(df)} lignes | {len(X.columns)} features | cible : {target}")
        return X, y

    print(f"✅ {len(df)} lignes chargées depuis '{table}'")
    return df


def get_data_quality(table):
   
    # Récupérer les colonnes de la table automatiquement
    inspector = inspect(engine)
    colonnes  = [col["name"] for col in inspector.get_columns(table)]

    total_query = f"SELECT COUNT(*) AS total FROM {table}"
    total       = pd.read_sql(total_query, engine)["total"][0]

    print(f"\n📊 Rapport qualité — table : '{table}'")
    print(f"   Total lignes : {total}")
    print(f"\n{'Colonne':<25} {'Manquants':>10} {'%':>8} {'Statut':>10}")
    print("-" * 58)

    rapport = []
    for col in colonnes:
        query  = f"SELECT COUNT(*) - COUNT({col}) AS manquants FROM {table}"
        manq   = pd.read_sql(query, engine)["manquants"][0]
        pct    = round(manq / total * 100, 1) if total > 0 else 0
        statut = "🔴 critique" if pct > 60 else "⚠️  attention" if pct > 20 else "✅ ok"
        print(f"{col:<25} {manq:>10} {pct:>7}% {statut:>10}")
        rapport.append({"colonne": col, "manquants": manq, "pct": pct})

    return pd.DataFrame(rapport)


def get_stats_groupe(table, groupe, cible=None):
    """
    Statistiques par groupe sur n'importe quelle table.
    
    Paramètres :
        table  : nom de la table
        groupe : colonne de groupement   ex: "pclass", "region", "categorie"
        cible  : colonne à analyser      ex: "survived", "churn", "prix"
    
    Exemple :
        get_stats_groupe("passagers", groupe="pclass", cible="survived")
        get_stats_groupe("ventes",    groupe="region", cible="montant")
    """
    if cible:
        query = f"""
            SELECT
                {groupe},
                COUNT(*)                              AS total,
                ROUND(AVG({cible})::numeric, 4)       AS moyenne_{cible},
                ROUND(MIN({cible})::numeric, 2)       AS min_{cible},
                ROUND(MAX({cible})::numeric, 2)       AS max_{cible}
            FROM {table}
            GROUP BY {groupe}
            ORDER BY {groupe}
        """
    else:
        query = f"""
            SELECT {groupe}, COUNT(*) AS total
            FROM {table}
            GROUP BY {groupe}
            ORDER BY total DESC
        """

    df = pd.read_sql(query, engine)
    print(f"\n📊 Stats '{table}' groupé par '{groupe}' :")
    print(df.to_string(index=False))
    return df


def get_anomalies(table, regles=None):
    """
    Détecte les anomalies selon des règles personnalisées.
    
    Paramètres :
        table  : nom de la table
        regles : liste de conditions SQL  ex: ["age < 0", "fare < 0"]
    
    Exemple :
        get_anomalies("passagers", regles=["age < 0", "age > 120", "fare < 0"])
        get_anomalies("clients",   regles=["age < 18", "solde < 0"])
    """
    if not regles:
        print("⚠️  Aucune règle définie — retour table complète")
        return pd.read_sql(f"SELECT * FROM {table} LIMIT 10", engine)

    conditions = " OR ".join(regles)
    query      = f"SELECT * FROM {table} WHERE {conditions}"
    df         = pd.read_sql(query, engine)

    print(f"⚠️  {len(df)} anomalie(s) dans '{table}'")
    if len(df) > 0:
        print(df.head())
    return df


def get_outliers_iqr(table, colonne):
    """
    Détecte les outliers via la méthode IQR (interquartile range).
    Fonctionne sur n'importe quelle colonne numérique.
    
    Exemple :
        get_outliers_iqr("passagers", "fare")
        get_outliers_iqr("ventes",    "montant")
        get_outliers_iqr("employes",  "salaire")
    """
    query = f"""
        WITH stats AS (
            SELECT
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {colonne}) AS q1,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {colonne}) AS q3
            FROM {table}
        )
        SELECT t.*,
               ROUND(t.{colonne}::numeric, 2) AS valeur_analysee
        FROM {table} t, stats s
        WHERE t.{colonne} > s.q3 + 1.5 * (s.q3 - s.q1)
           OR t.{colonne} < s.q1 - 1.5 * (s.q3 - s.q1)
        ORDER BY t.{colonne} DESC
    """
    df = pd.read_sql(query, engine)
    print(f"💰 {len(df)} outlier(s) sur '{colonne}' dans '{table}'")
    return df


def run_query(sql):
    """
    Exécute n'importe quelle requête SQL personnalisée.
    Pour les cas spécifiques non couverts par les fonctions ci-dessus.
    
    Exemple :
        df = run_query("SELECT * FROM passagers WHERE pclass = 1")
        df = run_query("SELECT COUNT(*) FROM ventes WHERE mois = '2024-01'")
    """
    df = pd.read_sql(sql, engine)
    print(f"✅ {len(df)} lignes retournées")
    return df


# ══════════════════════════════════════════════
#   MAIN — test sur le projet Titanic
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("\n=== TEST QUERIES.PY — Projet Titanic ===\n")

    # 1. Charger données ML
    X, y = get_data_ml(
        table     = "passagers",
        target    = "survived",
        drop_cols = ["name", "ticket", "cabin"]
    )

    # 2. Qualité des données
    get_data_quality("passagers")

    # 3. Stats par groupe
    get_stats_groupe("passagers", groupe="pclass", cible="survived")

    # 4. Anomalies
    get_anomalies("passagers", regles=["age < 0", "age > 120", "fare < 0"])

    # 5. Outliers
    get_outliers_iqr("passagers", "fare")

    # 6. Requête libre
    df = run_query("SELECT pclass, COUNT(*) as total FROM passagers GROUP BY pclass")
    print(df)

    print("\n✅ queries.py OK — prêt pour tous projets !")