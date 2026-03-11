"""
╔══════════════════════════════════════════════════════════════════╗
║            train_column_model.py                                 ║
║   Entraîne un vrai classifieur sur TES noms de colonnes Excel    ║
║   et le sauvegarde dans column_classifier.pkl                    ║
╚══════════════════════════════════════════════════════════════════╝

PIPELINE :
  1. Scan de tous tes fichiers Excel → collecte les noms de colonnes
  2. Génération d'un fichier de labellisation CSV (à remplir une fois)
  3. Entraînement : embeddings (MiniLM figé) + classifieur sklearn
  4. Sauvegarde du modèle (.pkl) + rapport de performance

Usage :
  # Étape 1 – Extraire les colonnes de tes fichiers Excel
  python train_column_model.py --extract data/*.xlsx

  # Étape 2 – Ouvre column_labels.csv, remplis la colonne "label"
  #            (ex: id_client, nom_client, date, montant, quantite, autre)

  # Étape 3 – Entraîner le modèle
  python train_column_model.py --train

  # Tester le modèle entraîné
  python train_column_model.py --predict "réf acheteur" "Date Opé" "Clt ID"
"""

import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from excel_ai import detect_headers_upgrade
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Imports optionnels ──────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_OK = True
except ImportError:
    SENTENCE_OK = False
    print("[ERREUR] sentence-transformers manquant : pip install sentence-transformers")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[ERREUR] scikit-learn manquant : pip install scikit-learn")

# ── Constantes ─────────────────────────────────────────────────────
LABELS_CSV       = "column_labels.csv"
MODEL_PKL        = "column_classifier.pkl"
ENCODER_PKL      = "label_encoder.pkl"
EMBEDDER_NAME    = "all-MiniLM-L6-v2"

# ── Labels canoniques suggérés ──────────────────────────────────────
CANONICAL_LABELS = [
    "id_client", "nom_client", "prenom", "date", "montant",
    "quantite", "statut", "adresse", "telephone", "email",
    "id_commande", "id_produit", "description", "categorie", "autre",
]

# ── Données d'augmentation intégrées ──────────────────────────────
# Ces exemples s'ajoutent aux vraies colonnes pour enrichir l'entraînement
AUGMENTATION_DATA: List[Tuple[str, str]] = [
    # id_client
    ("id client",         "id_client"), ("identifiant client", "id_client"),
    ("customer id",       "id_client"), ("client id",          "id_client"),
    ("num client",        "id_client"), ("no client",          "id_client"),
    ("clientid",          "id_client"), ("client_id",          "id_client"),
    ("id_clt",            "id_client"), ("ref client",         "id_client"),
    ("code client",       "id_client"), ("clt_id",             "id_client"),
    ("réf acheteur",      "id_client"), ("acheteur id",        "id_client"),
    # nom_client
    ("nom client",        "nom_client"), ("customer name",     "nom_client"),
    ("raison sociale",    "nom_client"), ("société",           "nom_client"),
    ("entreprise",        "nom_client"), ("client name",       "nom_client"),
    ("nom_clt",           "nom_client"), ("libelle client",    "nom_client"),
    ("nom",               "nom_client"), ("name",              "nom_client"),
    # prenom
    ("prénom",            "prenom"),     ("first name",        "prenom"),
    ("firstname",         "prenom"),     ("given name",        "prenom"),
    # date
    ("date opération",    "date"),       ("date transaction",  "date"),
    ("date facture",      "date"),       ("invoice date",      "date"),
    ("date_op",           "date"),       ("date mouvement",    "date"),
    ("date valeur",       "date"),       ("date saisie",       "date"),
    ("created_at",        "date"),       ("Date Opé",          "date"),
    ("date opé",          "date"),       ("date mvt",          "date"),
    # montant
    ("montant",           "montant"),    ("amount",            "montant"),
    ("total",             "montant"),    ("valeur",            "montant"),
    ("prix",              "montant"),    ("price",             "montant"),
    ("débit",             "montant"),    ("crédit",            "montant"),
    ("solde",             "montant"),    ("balance",           "montant"),
    ("net",               "montant"),    ("brut",              "montant"),
    ("montant ht",        "montant"),    ("montant ttc",       "montant"),
    # quantite
    ("quantité",          "quantite"),   ("quantity",          "quantite"),
    ("qty",               "quantite"),   ("qté",               "quantite"),
    ("nb",                "quantite"),   ("nombre",            "quantite"),
    ("volume",            "quantite"),   ("units",             "quantite"),
    # statut
    ("statut",            "statut"),     ("status",            "statut"),
    ("état",              "statut"),     ("state",             "statut"),
    ("flag",              "statut"),
    # adresse
    ("adresse",           "adresse"),    ("address",           "adresse"),
    ("rue",               "adresse"),    ("street",            "adresse"),
    ("ville",             "adresse"),    ("city",              "adresse"),
    ("code postal",       "adresse"),    ("cp",                "adresse"),
    # telephone
    ("téléphone",         "telephone"),  ("phone",             "telephone"),
    ("tel",               "telephone"),  ("mobile",            "telephone"),
    ("gsm",               "telephone"),
    # email
    ("email",             "email"),      ("mail",              "email"),
    ("e-mail",            "email"),      ("courriel",          "email"),
    # id_commande
    ("id commande",       "id_commande"), ("order id",         "id_commande"),
    ("num commande",      "id_commande"), ("ref commande",     "id_commande"),
    ("commande id",       "id_commande"), ("no commande",      "id_commande"),
    # id_produit
    ("id produit",        "id_produit"), ("product id",        "id_produit"),
    ("ref produit",       "id_produit"), ("sku",               "id_produit"),
    ("code article",      "id_produit"), ("article id",        "id_produit"),
    # description
    ("description",       "description"), ("libellé",          "description"),
    ("designation",       "description"), ("desc",             "description"),
    ("label",             "description"), ("commentaire",      "description"),
    # categorie
    ("catégorie",         "categorie"),  ("category",          "categorie"),
    ("type",              "categorie"),  ("famille",           "categorie"),
    ("sous-catégorie",    "categorie"),
]


# ══════════════════════════════════════════════════════════════════
#  ÉTAPE 1 : Extraction des noms de colonnes depuis les Excels
# ══════════════════════════════════════════════════════════════════

def extract_columns_from_excels(excel_paths: List[str]) -> pd.DataFrame:
    """
    Scanne tous les fichiers Excel et extrait les noms de colonnes
    en utilisant le détecteur ML de header.
    """

    seen = {}

    for path in excel_paths:

        p = Path(path)

        if not p.exists():
            print(f"✘ Introuvable : {path}")
            continue

        try:

            print(f"\n📄 Scan : {p.name}")

            # 1️⃣ détecter les headers avec ton modèle ML
            headers = detect_headers_upgrade(str(p))

            if not headers:
                print("   ⚠ Aucun header détecté")
                continue

            # 2️⃣ lire chaque sheet avec la bonne ligne d'en-tête
            for sheet, info in headers.items():

                    confidence = info["confidence"]

                    if confidence < 0.4:
                        print(f"   ⚠ {sheet} confidence faible ({confidence:.2f})")
                        continue

                    # lignes header détectées
                    header_rows = info.get("header_rows", [])
                    reconstructed_cols = info.get("columns")

                    header_row = header_rows[0] - 1  # pandas index

                    df = pd.read_excel(
                        p,
                        sheet_name=sheet,
                        header=header_row,
                        engine="openpyxl"
                    )

                    # si on a reconstruit le header → remplacer les colonnes
                    if reconstructed_cols:

                        reconstructed_cols = [
                            str(c).strip() if c not in (None, "", "nan") else f"col_{i}"
                            for i, c in enumerate(reconstructed_cols)
                        ]

                        if len(reconstructed_cols) == len(df.columns):
                            df.columns = reconstructed_cols

                    for col in df.columns:

                        col_clean = str(col).strip()

                        # éviter colonnes vides ou Unnamed
                        if (
                            not col_clean
                            or col_clean.lower().startswith("unnamed")
                            or col_clean == "nan"
                        ):
                            continue

                        if col_clean not in seen:
                            seen[col_clean] = (p.name, sheet)

                    print(
                        f"   ✔ {sheet} header détecté ligne {header_row+1} "
                        f"(conf {confidence:.2f})"
                    )

        except Exception as e:
            print(f"✘ {path} : {e}")

    if not seen:
        print("[ERREUR] Aucune colonne extraite.")
        return pd.DataFrame()

    rows = [
        {
            "column_name": col,
            "source_file": src[0],
            "source_sheet": src[1],
            "label": "",
        }
        for col, src in seen.items()
    ]

    df_out = pd.DataFrame(rows)

    # suggestion automatique
    df_out["label"] = df_out["column_name"].apply(_suggest_label)

    df_out.to_csv(LABELS_CSV, index=False, encoding="utf-8-sig")

    print(f"\n✔ {len(df_out)} colonnes uniques extraites → {LABELS_CSV}")

    return df_out


def _suggest_label(col_name: str) -> str:
    """Suggestion automatique par matching sur les données d'augmentation."""
    col_low = col_name.lower().strip()

    # Chercher dans les augmentations
    best_label, best_score = "autre", 0
    for col_aug, label in AUGMENTATION_DATA:
        score = _simple_similarity(col_low, col_aug.lower())
        if score > best_score:
            best_score = score
            best_label = label

    return best_label if best_score > 0.55 else "autre"


def _simple_similarity(a: str, b: str) -> float:
    """Similarité simple sans dépendances."""
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.8
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    inter = words_a & words_b
    return len(inter) / max(len(words_a), len(words_b))


# ══════════════════════════════════════════════════════════════════
#  ÉTAPE 2 : Entraînement du classifieur
# ══════════════════════════════════════════════════════════════════

def train_model(labels_csv: str = LABELS_CSV) -> bool:
    """
    Entraîne le classifieur sklearn sur les colonnes labellisées
    + les données d'augmentation intégrées.

    Sauvegarde :
      - column_classifier.pkl  → le modèle SVM entraîné
      - label_encoder.pkl      → encodeur label↔entier
    """
    if not SENTENCE_OK or not SKLEARN_OK:
        print("[ERREUR] Dépendances manquantes.")
        return False

    if not Path(labels_csv).exists():
        print(f"[ERREUR] {labels_csv} introuvable. Lance d'abord --extract.")
        return False

    df = pd.read_csv(labels_csv, encoding="utf-8-sig")
    df = df[df["label"].notna() & (df["label"].str.strip() != "")]
    if len(df) < 5:
        print(f"[ERREUR] Trop peu de lignes labellisées ({len(df)}). "
              f"Remplis la colonne 'label' dans {labels_csv}.")
        return False

    print(f"\n[TRAIN] {len(df)} colonnes labellisées chargées depuis {labels_csv}")

    # ── Combinaison données réelles + augmentation ──────────────
    texts  = list(df["column_name"].astype(str))
    labels = list(df["label"].astype(str))

    for text_aug, label_aug in AUGMENTATION_DATA:
        texts.append(text_aug)
        labels.append(label_aug)

    print(f"[TRAIN] {len(texts)} exemples total (réels + augmentation)")

    # ── Vérification des classes ─────────────────────────────────
    from collections import Counter
    label_counts = Counter(labels)
    rare_labels  = [l for l, c in label_counts.items() if c < 3]
    if rare_labels:
        print(f"[WARN] Labels avec < 3 exemples : {rare_labels}")
        print("       La cross-validation sera ignorée pour ces classes.")

    # ── Embeddings (modèle figé, rapide) ────────────────────────
    print(f"\n[TRAIN] Chargement du modèle {EMBEDDER_NAME}…")
    embedder = SentenceTransformer(EMBEDDER_NAME)
    print(f"[TRAIN] Calcul des embeddings pour {len(texts)} textes…")
    X = embedder.encode(texts, show_progress_bar=True, batch_size=64)

    # ── Encodage des labels ─────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    print(f"[TRAIN] Classes : {list(le.classes_)}")

    # ── Entraînement SVM (robuste sur peu de données) ────────────
    print("\n[TRAIN] Entraînement du classifieur SVM…")
    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    clf.fit(X, y)

    # ── Cross-validation ─────────────────────────────────────────
    min_class_count = min(label_counts.values())
    n_splits = min(5, min_class_count)
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        print(f"\n[TRAIN] Cross-validation ({n_splits} folds) :")
        print(f"        Accuracy moyenne : {scores.mean():.3f} ± {scores.std():.3f}")
    else:
        print("[WARN] Pas assez d'exemples par classe pour la cross-validation.")

    # ── Rapport complet ──────────────────────────────────────────
    y_pred = clf.predict(X)
    print("\n[TRAIN] Rapport sur les données d'entraînement :")
    print(classification_report(y, y_pred, target_names=le.classes_))

    # ── Sauvegarde ────────────────────────────────────────────────
    model_bundle = {
        "classifier": clf,
        "embedder_name": EMBEDDER_NAME,
        "label_encoder": le,
        "n_train": len(texts),
        "classes": list(le.classes_),
    }
    joblib.dump(model_bundle, MODEL_PKL)
    print(f"\n[TRAIN] ✔ Modèle sauvegardé → {MODEL_PKL}")
    print(f"         Classes : {list(le.classes_)}")
    print(f"         Exemples d'entraînement : {len(texts)}")
    return True


# ══════════════════════════════════════════════════════════════════
#  PRÉDICTION (chargement du modèle sauvegardé)
# ══════════════════════════════════════════════════════════════════

class ColumnClassifier:
    """
    Classifieur de noms de colonnes.
    Utilise le modèle entraîné pour prédire le type sémantique d'une colonne.

    Intégration dans smart_excel_finder.py :
      clf = ColumnClassifier()
      label, confidence = clf.predict("Réf. Acheteur")
      # → ("id_client", 0.94)
    """

    def __init__(self, model_path: str = MODEL_PKL):
        if not SKLEARN_OK or not SENTENCE_OK:
            raise ImportError("scikit-learn ou sentence-transformers manquant.")
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Modèle {model_path!r} introuvable. Lance d'abord --train."
            )
        self._bundle = joblib.load(model_path)
        self._embedder = SentenceTransformer(self._bundle["embedder_name"])
        self._clf = self._bundle["classifier"]
        self._le  = self._bundle["label_encoder"]
        print(f"[ColumnClassifier] Modèle chargé : {model_path}")
        print(f"  Classes : {self._bundle['classes']}")

    def predict(self, col_name: str) -> Tuple[str, float]:
        """
        Prédit le label sémantique d'un nom de colonne.

        Returns:
            (label_str, confidence_0_to_1)
        """
        emb  = self._embedder.encode([col_name])
        pred = self._clf.predict(emb)[0]
        prob = self._clf.predict_proba(emb)[0].max()
        return self._le.inverse_transform([pred])[0], float(prob)

    def predict_batch(self, col_names: List[str]) -> List[Tuple[str, float]]:
        """Prédit en batch (plus rapide que predict() en boucle)."""
        embs  = self._embedder.encode(col_names, batch_size=64)
        preds = self._clf.predict(embs)
        probs = self._clf.predict_proba(embs).max(axis=1)
        labels = self._le.inverse_transform(preds)
        return list(zip(labels, probs.tolist()))

    def score_column_for_hint(self, col_name: str, hint: str) -> float:
        """
        Retourne un score 0–100 indiquant à quel point col_name correspond au hint.
        Utilisé pour intégration dans smart_excel_finder.py
        """
        predicted_label, confidence = self.predict(col_name)
        hint_clean = hint.lower().replace(" ", "_").replace("-", "_")

        if predicted_label == hint_clean:
            return confidence * 100  # ex: 94.0

        # Essayer aussi les variantes du hint
        hint_parts = set(hint_clean.split("_"))
        label_parts = set(predicted_label.split("_"))
        overlap = len(hint_parts & label_parts) / max(len(hint_parts), len(label_parts))
        return overlap * confidence * 60


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        prog="train_column_model",
        description="Entraîne un classifieur sur tes propres noms de colonnes Excel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow complet :
  1) python train_column_model.py --extract data/*.xlsx
     → Génère column_labels.csv avec suggestions automatiques

  2) Ouvre column_labels.csv dans Excel ou un éditeur
     → Vérifie / corrige la colonne "label"
     → Labels disponibles : id_client, nom_client, prenom, date, montant,
                            quantite, statut, adresse, telephone, email,
                            id_commande, id_produit, description, categorie, autre

  3) python train_column_model.py --train
     → Entraîne le modèle → column_classifier.pkl

  4) python train_column_model.py --predict "Réf Acheteur" "Date Opé" "Clt ID"
     → Teste le modèle entraîné

  5) Le modèle est automatiquement utilisé par smart_excel_finder.py
        """,
    )
    p.add_argument("--extract", nargs="+", metavar="EXCEL",
                   help="Extraire les colonnes des fichiers Excel fournis")
    p.add_argument("--train",   action="store_true",
                   help=f"Entraîner le modèle depuis {LABELS_CSV}")
    p.add_argument("--predict", nargs="+", metavar="COL",
                   help="Tester le modèle entraîné sur des noms de colonnes")
    p.add_argument("--labels-csv", default=LABELS_CSV,
                   help=f"Chemin vers le CSV de labellisation (défaut: {LABELS_CSV})")
    p.add_argument("--model",   default=MODEL_PKL,
                   help=f"Chemin du modèle sauvegardé (défaut: {MODEL_PKL})")
    return p


def main():
    args = build_parser().parse_args()

    if args.extract:

        excel_files = []

        for path in args.extract:
            p = Path(path)

            # si c'est un dossier → scan récursif
            if p.is_dir():
                excel_files.extend(p.rglob("*.xlsx"))
                excel_files.extend(p.rglob("*.xlsm"))

            # si c'est un fichier Excel direct
            elif p.suffix.lower() in [".xlsx", ".xlsm"]:
                excel_files.append(p)

            # si wildcard ou autre
            else:
                excel_files.extend(Path().glob(path))

        excel_files = [str(f) for f in excel_files]

        if not excel_files:
            print("[ERREUR] Aucun fichier Excel trouvé.")
            return

        print(f"\nExtraction depuis {len(excel_files)} fichier(s)…")
        extract_columns_from_excels(excel_files)

    elif args.train:
        train_model(labels_csv=args.labels_csv)

    elif args.predict:
        clf = ColumnClassifier(model_path=args.model)
        print(f"\nPrédictions :")
        print(f"{'Colonne':<30} {'Label prédit':<20} {'Confiance'}")
        print("─" * 65)
        results = clf.predict_batch(args.predict)
        for col, (label, conf) in zip(args.predict, results):
            bar = "█" * int(conf * 20)
            print(f"{col:<30} {label:<20} {conf:.0%}  {bar}")

    else:
        build_parser().print_help()


if __name__ == "__main__":
    main()
