"""
utils.py
Fonctions de chargement et de transformation des données électorales
Présidentielle 2022 - premier tour
"""

import pandas as pd

URL_DATA = "https://www.data.gouv.fr/fr/datasets/r/182268fc-2103-4bcb-a850-6cf90b02a9eb"


def load_data(url: str = URL_DATA) -> pd.DataFrame:
    """
    Charge les données brutes depuis data.gouv.fr.

    - `code_commune` est lu comme chaîne pour préserver les zéros initiaux
      (ex. '001', '028'), car il s'agit d'un identifiant administratif.
    - `low_memory=False` stabilise l'inférence des types à l'import
      et évite ici le DtypeWarning observé.
    """
    df = pd.read_csv(url, dtype={"code_commune": str}, low_memory=False)
    return df


def build_code_commune(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un code commune complet en concaténant `code_departement`
    et `code_commune` sur 3 caractères.

    Exemples :
    - dept='92', commune='049' -> '92049'
    - dept='971', commune='001' -> '971001'

    Dans les données chargées, `code_departement` est déjà correctement
    formaté (ex. '01', '2A', '971', 'fr_etranger'), donc aucun `zfill`
    n'est appliqué dessus.

    Pour les Français de l'étranger (`code_departement` commençant par 'fr_'),
    `code_commune` est laissé inchangé.
    """
    df = df.copy()

    masque_etranger = df["code_departement"].astype(str).str.startswith("fr_")

    df.loc[~masque_etranger, "code_commune"] = (
        df.loc[~masque_etranger, "code_departement"].astype(str)
        + df.loc[~masque_etranger, "code_commune"].astype(str).str.zfill(3)
    )

    return df


def build_candidat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée la colonne `candidat` = prenom + ' ' + nom.

    Dans ce jeu de données, les lignes non-candidats (abstentions, blancs, nuls)
    ont `prenom` manquant. La concaténation laisse donc `candidat` à NaN
    pour ces lignes, ce qui permet de les exclure naturellement des agrégations
    par candidat tout en les conservant dans le DataFrame source.
    """
    df = df.copy()
    df["candidat"] = df["prenom"].str.cat(df["nom"], sep=" ")
    return df


def compute_scores_nationaux(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le score national de chaque candidat.

    Le dénominateur correspond à la somme des voix des candidats uniquement.
    Les lignes où `candidat` est manquant (abstentions, blancs, nuls) sont
    exclues par `groupby(..., dropna=True)`.

    Retourne un DataFrame trié par voix décroissant :
        candidat | voix | score_national (%)
    """
    scores = (
        df.groupby("candidat", dropna=True)["voix"]
        .sum()
        .reset_index()
        .sort_values("voix", ascending=False)
        .reset_index(drop=True)
    )

    total_exprimes = scores["voix"].sum()
    scores["score_national"] = (scores["voix"] / total_exprimes * 100).round(2)

    return scores


def compute_scores_departements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule, pour chaque département, le nombre de voix et le score (%)
    de chaque candidat.

    Le dénominateur est calculé sur les lignes candidats uniquement.
    Les lignes où `candidat` est manquant (abstentions, blancs, nuls)
    sont exclues par `groupby(..., dropna=True)`.

    Retourne :
        code_departement | candidat | votes_departement | score_departement (%)
    """
    scores = (
        df.groupby(["code_departement", "candidat"], dropna=True)["voix"]
        .sum()
        .reset_index()
        .rename(columns={"voix": "votes_departement"})
    )

    total_par_dept = (
        scores.groupby("code_departement")["votes_departement"]
        .sum()
        .rename("total_dept")
    )

    scores = scores.merge(total_par_dept, on="code_departement")
    scores["score_departement"] = (
        scores["votes_departement"] / scores["total_dept"] * 100
    ).round(2)
    scores = scores.drop(columns="total_dept")

    return scores


def build_score_departements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne les scores départementaux avec les scores nationaux
    et calcule la surreprésentation.

    Surreprésentation (%) =
        ((score_departement - score_national) / score_national) * 100

    Exemple :
        score_departement = 30
        score_national = 15
        -> surrepresentation = +100

    Retourne :
        code_departement | candidat | votes_departement | score_departement |
        votes_national   | score_national | surrepresentation
    """
    scores_nat = compute_scores_nationaux(df)
    scores_dept = compute_scores_departements(df)

    merged = scores_dept.merge(
        scores_nat.rename(columns={"voix": "votes_national"}),
        on="candidat",
        how="left",
    )

    merged["surrepresentation"] = (
        (merged["score_departement"] - merged["score_national"])
        / merged["score_national"]
        * 100
    ).round(2)

    return merged