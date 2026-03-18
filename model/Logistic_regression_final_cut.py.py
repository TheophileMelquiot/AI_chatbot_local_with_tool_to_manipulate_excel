import argparse
import warnings
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import joblib
import re
import random
import string

from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from excel_ai import detect_headers_upgrade
from typing import List, Tuple
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import top_k_accuracy_score


warnings.filterwarnings("ignore")

LABELS_CSV = "column_labels_corrected.csv"
MODEL_PKL = "column_classifier.pkl"

EMBEDDER_NAME = "models/all-MiniLM-L6-v2"
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


# ==========================================
# STATISTIQUES DE COLONNE
# ==========================================

def column_stats(series):

    values = series.dropna().astype(str)

    if len(values) == 0:
        return [0,0,0,0,0]

    numeric = values.str.match(r"^-?\d+(\.\d+)?$")
    dates = values.str.match(r"\d{4}-\d{2}-\d{2}")

    percent_numeric = numeric.mean()
    percent_dates = dates.mean()
    percent_text = 1 - percent_numeric
    avg_len = values.str.len().mean()
    unique_ratio = values.nunique() / len(values)

    return [
        percent_numeric,
        percent_dates,
        percent_text,
        avg_len,
        unique_ratio
    ]

def is_numeric_column(series):

    values = series.dropna().astype(str)

    if len(values) == 0:
        return 0

    numeric = values.str.match(r"^-?\d+(\.\d+)?$")

    return 1 if numeric.mean() > 0.8 else 0


def is_date_column(series):

    values = series.dropna().astype(str)

    if len(values) == 0:
        return 0

    date_patterns = (
        values.str.match(r"\d{4}-\d{2}-\d{2}") |
        values.str.match(r"\d{2}/\d{2}/\d{4}") |
        values.str.match(r"\d{2}-\d{2}-\d{4}")
    )

    return 1 if date_patterns.mean() > 0.6 else 0


def is_id_like(series):

    values = series.dropna().astype(str)

    if len(values) == 0:
        return 0

    avg_len = values.str.len().mean()
    unique_ratio = values.nunique() / len(values)

    if unique_ratio > 0.9 and avg_len < 15:
        return 1

    return 0


def contains_currency(series):

    values = series.dropna().astype(str)

    if len(values) == 0:
        return 0

    currency = values.str.contains(r"€|\$|£|eur|usd", case=False)

    return 1 if currency.mean() > 0.3 else 0

# ==========================================
# EXTRACTION VALEURS COLONNE
# ==========================================

def is_valid_column_name(name):

    name = str(name).strip()

    if len(name) < 2:
        return False

    if len(name) > 60:
        return False

    if name.count(" ") > 6:
        return False

    if "\n" in name:
        return False

    if name.lower().startswith(("date de l", "exemple", "l'onglet")):
        return False

    return True

def extract_column_samples(df, col, n=5, max_cell_chars=60, max_total_chars=200):

    values = df[col].dropna().astype(str)

    if len(values) == 0:
        return ""


    samples = []

    total_len = 0

    for v in values:

        v = v.strip().replace("\n", " ")

        # limiter taille cellule
        v = v[:max_cell_chars]

        samples.append(v)

        total_len += len(v)

        if len(samples) >= n:
            break

        if total_len >= max_total_chars:
            break

    return " ".join(samples)

def _suggest_label(col_name: str):

    col_low = col_name.lower().strip()

    best_label = "autre"
    best_score = 0

    for col_aug, label in AUGMENTATION_DATA:

        score = _simple_similarity(col_low, col_aug.lower())

        if score > best_score:
            best_score = score
            best_label = label

    if best_score > 0.55:
        return best_label

    return "autre"


def _simple_similarity(a: str, b: str):

    if a == b:
        return 1.0

    if a in b or b in a:
        return 0.8

    words_a = set(a.split())
    words_b = set(b.split())

    if not words_a or not words_b:
        return 0

    inter = words_a & words_b

    return len(inter) / max(len(words_a), len(words_b))

# ==========================================
# EXTRACTION DATASET EXCEL
# ==========================================

def extract_columns_from_excels(excel_paths: List[str]) -> pd.DataFrame:

    rows = []

    for path in excel_paths:

        p = Path(path)

        if not p.exists():
            print(f"✘ Introuvable : {path}")
            continue

        try:

            print(f"\n📄 Scan : {p.name}")

            headers = detect_headers_upgrade(str(p))

            if not headers:
                print("   ⚠ Aucun header détecté")
                continue

            for sheet, info in headers.items():

                confidence = info["confidence"]

                if confidence < 0.4:
                    print(f"   ⚠ {sheet} confidence faible ({confidence:.2f})")
                    continue

                header_rows = info.get("header_rows", [])
                reconstructed_cols = info.get("columns")

                header_row = header_rows[0] - 1

                df = pd.read_excel(
                    p,
                    sheet_name=sheet,
                    header=header_row,
                    engine="openpyxl"
                )

                # reconstruction header multi-lignes
                if reconstructed_cols:

                    reconstructed_cols = [
                        str(c).strip() if c not in (None,"","nan")
                        else f"col_{i}"
                        for i,c in enumerate(reconstructed_cols)
                    ]

                    if len(reconstructed_cols) == len(df.columns):
                        df.columns = reconstructed_cols


                for col in df.columns:

                    col_clean = str(col).strip()

                    if not is_valid_column_name(col_clean):
                        continue

                    if df[col].dropna().shape[0] < 3:
                        continue
                
                    unique_ratio = df[col].nunique() / df[col].dropna().shape[0]

                    if unique_ratio < 0.05:
                        continue
                    
                    if df[col].astype(str).str.len().max() > 200:
                        continue
                    
                    # colonne quasi constante
                    if df[col].nunique() <= 1:
                        continue

                    if (
                        not col_clean
                        or col_clean.lower().startswith("unnamed")
                        or col_clean.lower() == "nan"
                    ):
                        continue

                        # 🔎 supprimer colonnes texte trop longues
                    if df[col].astype(str).str.len().mean() > 120:
                        continue


                    # ===== EXTRACTION VALEURS =====

                    samples = extract_column_samples(df, col)

                    # ===== STATS =====

                    stats = column_stats(df[col])
                    numeric_flag = is_numeric_column(df[col])
                    date_flag = is_date_column(df[col])
                    id_flag = is_id_like(df[col])
                    currency_flag = contains_currency(df[col])


                    rows.append({


                        "column_name": col_clean,
                        "sample_values": samples,

                        "percent_numeric": stats[0],
                        "percent_dates": stats[1],
                        "percent_text": stats[2],
                        "avg_len": stats[3],
                        "unique_ratio": stats[4],

                        "is_numeric_column": numeric_flag,
                        "is_date_column": date_flag,
                        "is_id_like": id_flag,
                        "contains_currency": currency_flag,

                        "source_file": p.name,
                        "source_sheet": sheet,

                        "label": ""
                    })

                print(
                    f"   ✔ {sheet} header détecté ligne {header_row+1} "
                    f"(conf {confidence:.2f})"
                )

        except Exception as e:

            print(f"✘ {path} : {e}")


    if not rows:
        print("[ERREUR] Aucune colonne extraite")
        return pd.DataFrame()


    df_out = pd.DataFrame(rows)

    df_out.drop_duplicates(
        subset=["column_name","sample_values"],
        inplace=True
    )
    # suggestion automatique
    df_out["label"] = df_out["column_name"].apply(_suggest_label)

    df_out.to_csv(
        LABELS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n✔ Dataset créé → {LABELS_CSV}")
    print(f"Colonnes extraites : {len(df_out)}")

    return df_out

# ==========================================
# BUILD FEATURE VECTOR
# ==========================================

def build_features(df, embedder):

    headers = df["column_name"].astype(str).tolist()
    values = df["sample_values"].astype(str).tolist()

    # embeddings classiques
    emb_header = embedder.encode(headers)
    emb_values = embedder.encode(values)

    # header + values
    combined = [
        f"{h} | {v}"
        for h, v in zip(headers, values)
    ]

    emb_combined = embedder.encode(combined)

    # =============================
    # CONTEXTE DES COLONNES VOISINES
    # =============================

    context_texts = []

    for i in range(len(headers)):

        left = headers[i-1] if i > 0 else ""
        right = headers[i+1] if i < len(headers)-1 else ""

        context = f"{left} | {headers[i]} | {right}"

        context_texts.append(context)

    emb_context = embedder.encode(context_texts)

    # =============================
    # FEATURES STATISTIQUES
    # =============================

    stats = df[
        [
            "percent_numeric",
            "percent_dates",
            "percent_text",
            "avg_len",
            "unique_ratio",
            "is_numeric_column",
            "is_date_column",
            "is_id_like",
            "contains_currency"
        ]
    ].values

    # =============================
    # CONCATENATION FINALE
    # =============================

    X = np.concatenate(
        [
            emb_header,
            emb_values,
            emb_combined,
            emb_context,   # ← nouvelle feature
            stats
        ],
        axis=1
    )

    return X

# ==========================================
# Sanity metrics for the models 
# ==========================================

def block_permutation_importance(clf, X, y):

    baseline = accuracy_score(y, clf.predict(X))

    blocks = {
        "header": (0,384),
        "values": (384,768),
        "combined": (768,1152),
        "context": (1152,1536),
        "stats": (1536,1545)
    }

    results = {}

    for name,(start,end) in blocks.items():

        X_perm = X.copy()

        idx = np.random.permutation(len(X_perm))

        X_perm[:,start:end] = X_perm[idx,start:end]

        score = accuracy_score(y, clf.predict(X_perm))

        results[name] = baseline - score

    return results


def header_ablation_test(df, embedder, clf, le):

    print("\n🔎 HEADER ABLATION TEST")

    df_test = df.copy()

    # supprimer header
    df_test["column_name"] = ""

    X_test = build_features(df_test, embedder)

    X_test = StandardScaler().fit_transform(X_test)

    y_true = le.transform(df_test["label"])

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_true, y_pred)

    print("Accuracy without header :", acc)


def random_header_test(df, embedder, clf, le):

    print("\n🔎 RANDOM HEADER TEST")

    df_rand = df.copy()

    def random_header():
        return "col_" + "".join(random.choices(string.ascii_lowercase, k=5))

    # remplacer tous les headers
    df_rand["column_name"] = [random_header() for _ in range(len(df_rand))]

    # reconstruire les features
    X_rand = build_features(df_rand, embedder)

    scaler = StandardScaler()
    X_rand = scaler.fit_transform(X_rand)

    y_true = le.transform(df_rand["label"])

    y_pred = clf.predict(X_rand)

    acc = accuracy_score(y_true, y_pred)

    print("Accuracy with random headers :", acc)

# ==========================================
# TRAIN MODEL
# ==========================================


def train_model():

    if not Path(LABELS_CSV).exists():

        print("❌ column_labels.csv introuvable")

        return

    df = pd.read_csv(LABELS_CSV, sep=";")

    df = df[df["label"].notna()]
    df = df[df["label"].str.strip() != ""]

    print("Colonnes labellisées :", len(df))

    #loading the embedder 
    embedder = SentenceTransformer(EMBEDDER_NAME)

    #X becoming an matrice of features 
    X = build_features(df, embedder)

    #transforming the label into vector, creating our target vector 
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    #Creation of the K-fold Stratified, five slipt, mixing the data before the division and autorising the reproductibility 
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #Accuracy of each fold 
    acc_scores = []

    #All the prediction of each split 
    all_X_test = []
    all_y_test = []
    all_y_pred = []
    all_y_proba = []

    #initialysing the fold 
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        #separation of the train/ test 
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        #Normalisation of the features, avoiding certains features to over throne only due their range of value being larger without having the most importance weight to the model, and creating bias 
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        #Creation of news sample for undermining classes better to use that, than recalculatng the weight in this parameter cause the under present class would still dont have a lot of representation 
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        #Creation of the modele, 2000 authorising much iteration and -1 working on all the available CPU 
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)

        #model predict y with x 
        clf.fit(X_train, y_train)

        #return the predicted label 
        y_pred = clf.predict(X_test)

        #prediction for each classes 
        y_proba = clf.predict_proba(X_test)

        #accuracy calcul 
        acc = accuracy_score(y_test, y_pred)

        acc_scores.append(acc)

        print(f"Fold {fold+1} accuracy :", acc)

        # sauvegarder toutes les prédictions
        all_y_test.extend(y_test)
        all_X_test.extend(X_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)


    print("\nMean accuracy :", np.mean(acc_scores))
    print("Std :", np.std(acc_scores))

    #initializing the final model to train on full dataset 
    print("\nTraining final model on full dataset...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1
    )

    # entraîner sur tout le dataset
    clf.fit(X_scaled, y)

    print("\nConfusion matrix")

    #transformation in numpy array for the metrics 
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)
    all_X_test = np.array(all_X_test)

    #creation of the figure 
    fig, ax = plt.subplots(figsize=(12,10))

    #construction of the confusion matrice 
    cm_display = ConfusionMatrixDisplay.from_predictions(
        all_y_test,
        all_y_pred,
        display_labels=le.classes_,
        cmap="viridis",
        normalize="true",
        values_format=".2f",
        xticks_rotation=45,
        ax=ax
    )

    #ploting it 
    plt.title("Normalized Confusion Matrix (Cross Validation)")
    plt.tight_layout()
    plt.show()

    #is the real class is the top 3 predictions 
    top3 = top_k_accuracy_score(all_y_test, all_y_proba, k=3)
    print("Top-3 accuracy :", top3)

    #complete rapport 
    print("\nClassification report")
    print(
        classification_report(
            all_y_test,
            all_y_pred,
            target_names=le.classes_
        )
    )
    
    #showing what the code is really using to predict by permuting features block 
    print("\n🔎 BLOCK FEATURE IMPORTANCE")
    block_imp = block_permutation_importance(
        clf,
        all_X_test,
        all_y_test
    )
    for k,v in block_imp.items():
        print(f"{k:15s} : {v:.4f}")

    model_bundle = {
        "classifier": clf,
        "label_encoder": le,
        "embedder_name": EMBEDDER_NAME,
        "scaler" : scaler
    }

    print("\n🔎 SANITY CHECK : LABEL SHUFFLE TEST")

    #randomizing the relation between x and y, to test for any data leakage 
    y_shuffled = np.random.permutation(y)

    #split classical, 80% train 20% test 
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X,
        y_shuffled,
        test_size=0.2,
        random_state=42
    )

    #training the model on random labels, learing noise 
    clf_test = LogisticRegression(max_iter=2000)
    clf_test.fit(X_train_s, y_train_s)

    #prediction 
    y_pred_shuffle = clf_test.predict(X_test_s)

    # if superior >0.3, their data leakage. Meaning that your model learn hidden information to do prediction or that the train/ test is not correctly separated  
    print("Shuffle accuracy :", accuracy_score(y_test_s, y_pred_shuffle))

    # ================================
    # RANDOM HEADER TEST
    # ================================

    random_header_test(df, embedder, clf, le)

    model_bundle = {
        "classifier": clf,
        "label_encoder": le,
        "embedder_name": EMBEDDER_NAME,
        "scaler" : scaler
    }

    joblib.dump(model_bundle, MODEL_PKL)
    header_ablation_test(df, embedder, clf, le)

    
    # binarisation des labels pour multiclass, because the ROC is normally use on binary problem but we have a lot of different class thus one label against the rest 
    y_test_bin = label_binarize(
        all_y_test,
        classes=np.arange(len(le.classes_))
    )
    y_proba = np.array(all_y_proba)

    #ROC calcul 
    roc_auc = roc_auc_score(
        y_test_bin,
        y_proba,
        multi_class="ovr",
        average="macro"
    )

    print("\nROC AUC (macro OVR) :", roc_auc)
    plt.figure()

    for i, class_name in enumerate(le.classes_):

        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.2f})")

    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Multiclass")
    plt.legend()
    plt.show()

    joblib.dump(model_bundle, MODEL_PKL)

    print("\n✔ Model saved :", MODEL_PKL)

# ==========================================
# PREDICTOR
# ==========================================

class ColumnClassifier:

    def __init__(self, model_path=MODEL_PKL):

        bundle = joblib.load(model_path)

        self.clf = bundle["classifier"]
        self.le = bundle["label_encoder"]
        self.embedder = SentenceTransformer(bundle["embedder_name"])
        self.scaler = bundle["scaler"]


    def predict(self, header, sample_values="", left_header="", right_header=""):

        # -----------------------------
        # rebuild stats exactly like training
        # -----------------------------

        values_list = sample_values.split()
        series = pd.Series(values_list)

        stats = column_stats(series)

        percent_numeric = stats[0]
        percent_dates = stats[1]
        percent_text = stats[2]
        avg_len = stats[3]
        unique_ratio = stats[4]

        is_numeric = is_numeric_column(series)
        is_date = is_date_column(series)
        is_id = is_id_like(series)
        currency = contains_currency(series)

        # -----------------------------
        # recreate dataframe used by build_features
        # -----------------------------

        df = pd.DataFrame([{
            "column_name": header,
            "sample_values": sample_values,

            "percent_numeric": percent_numeric,
            "percent_dates": percent_dates,
            "percent_text": percent_text,
            "avg_len": avg_len,
            "unique_ratio": unique_ratio,

            "is_numeric_column": is_numeric,
            "is_date_column": is_date,
            "is_id_like": is_id,
            "contains_currency": currency
        }])

        # -----------------------------
        # build features EXACTLY like training
        # -----------------------------

        X = build_features(df, self.embedder)

        # -----------------------------
        # apply scaler used in training
        # -----------------------------

        X = self.scaler.transform(X)

        # -----------------------------
        # prediction
        # -----------------------------

        pred = self.clf.predict(X)[0]
        prob = self.clf.predict_proba(X).max()

        label = self.le.inverse_transform([pred])[0]

        return label, prob
# ==========================================
# CLI
# ==========================================
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

    parser = argparse.ArgumentParser()

    parser.add_argument("--extract", nargs="+")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", nargs="+")

    args = parser.parse_args()

    if args.extract:

        excel_files = []

        for path in args.extract:

            p = Path(path)

            # si dossier
            if p.is_dir():

                excel_files.extend(p.rglob("*.xlsx"))
                excel_files.extend(p.rglob("*.xlsm"))

            # si fichier direct
            elif p.suffix.lower() in [".xlsx",".xlsm"]:

                excel_files.append(p)

        excel_files = [str(f) for f in excel_files]

        if not excel_files:

            print("❌ Aucun fichier Excel trouvé")
            return

        print(f"📂 {len(excel_files)} fichiers Excel trouvés")

        extract_columns_from_excels(excel_files)

    elif args.train:

        train_model()

    elif args.predict:

        clf = ColumnClassifier()

        for col in args.predict:

            label, conf = clf.predict(col)

            print(col, "→", label, conf)


if __name__ == "__main__":

    main()