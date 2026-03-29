import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:1234@localhost:5432/titanic_db")

# Charger et voir les colonnes
df = pd.read_csv("train.csv")
print("Colonnes disponibles :", df.columns.tolist())


colonnes = {
    "Survived": "survived",
    "Pclass":   "pclass",
    "Sex":      "sex",
    "Age":      "age",
    "SibSp":    "sibsp",
    "Parch":    "parch",
    "Fare":     "fare",
    "Embarked": "embarked"
}

# Garder seulement les colonnes qui existent
cols_dispo = [c for c in colonnes.keys() if c in df.columns]
df = df[cols_dispo].rename(columns=colonnes)

print(f"Lignes a importer : {len(df)}")
df.to_sql("passagers", engine, if_exists="append", index=False)
print(f"✅ {len(df)} passagers importes dans titanic_db !")