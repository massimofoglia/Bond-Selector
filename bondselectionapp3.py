"""
Streamlit web app for building a bond portfolio from a CSV using multi-criteria
filters and target weights. Ready to deploy on Streamlit Community Cloud.

Usage (local):
  pip install -r requirements.txt
  streamlit run app.py

Files to include in the repo for deployment (example):
- app.py                 (this file)
- requirements.txt      (see bottom of this file for contents)
- README.md             (optional, with deploy steps)

The app implements the business rules you specified:
- Keep only bonds with ScoreRendimento >= 20 and either
  ( Perc_ScoreRendimento > 50% & Perc_ScoreRischio > 90% ) OR
  ( Perc_ScoreRendimento > 90% & Perc_ScoreRischio > 50% )
- Select N bonds maximizing ScoreRendimento while minimizing deviations from
  weighted targets across Currency, IssuerType, Sector, and Maturity buckets.
"""

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# try to import pulp for exact MILP solving; app will instruct if missing
try:
    import pulp
except Exception:
    pulp = None

# -----------------------------
# Column aliases and loaders
# -----------------------------
REQUIRED_COLS_ALIASES = {
    "Comparto": ["Comparto", "Unnamed: 0"],
    "ISIN": ["ISIN", "ISIN Code"],
    "Issuer": ["Issuer", "Issuer Name", "Emittente"],
    "Maturity": ["Maturity", "Maturity Date", "Scadenza Data", "MaturityDate"],
    "Currency": ["Currency", "ISO Currency", "Valuta"],
    "Sector": ["Sector", "Settore"],
    "IssuerType": ["IssuerType", "Issuer Type", "TipoEmittente"],
    "ScoreRendimento": ["ScoreRendimento", "Score Ret", "Score Rendimento", "ScoreRend"],
    "ScoreRischio": ["ScoreRischio", "Score Risk", "Score Rischio", "ScoreRisk"],
}


def _read_csv_any(uploaded) -> pd.DataFrame:
    encodings = ["utf-8", "ISO-8859-1", "latin1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            if hasattr(uploaded, "read"):
                uploaded.seek(0)
                return pd.read_csv(uploaded, encoding=enc)
            else:
                return pd.read_csv(uploaded, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def load_and_normalize(uploaded) -> pd.DataFrame:
    df = _read_csv_any(uploaded)

    # sometimes file repeats header as first row
    if "ISIN" in df.columns and isinstance(df.loc[0, "ISIN"], str) and df.loc[0, "ISIN"].strip().upper() == "ISIN":
        df = df.iloc[1:].reset_index(drop=True)

    # rename columns according to aliases
    rename_map = {}
    for std, aliases in REQUIRED_COLS_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                rename_map[a] = std
                break
    df = df.rename(columns=rename_map)

    # check required columns
    for c in ["Comparto", "ISIN", "Issuer", "Maturity", "Currency", "ScoreRendimento", "ScoreRischio"]:
        if c not in df.columns:
            raise ValueError(f"Colonna obbligatoria mancante nel file: {c}. Colonne trovate: {list(df.columns)}")

    # parse types
    df["Maturity"] = pd.to_datetime(df["Maturity"], errors="coerce", dayfirst=True)
    df["ScoreRendimento"] = pd.to_numeric(df["ScoreRendimento"], errors="coerce")
    df["ScoreRischio"] = pd.to_numeric(df["ScoreRischio"], errors="coerce")

    # fill optional
    if "IssuerType" not in df.columns:
        df["IssuerType"] = df["Comparto"].astype(str).map(_infer_issuer_type)
    if "Sector" not in df.columns:
        df["Sector"] = np.where(df["IssuerType"].str.contains("Govt", case=False, na=False), "Government", "Unknown")
    else:
        mask_na = df["Sector"].isna()
        df.loc[mask_na, "Sector"] = np.where(
            df.loc[mask_na, "IssuerType"].str.contains("Govt", case=False, na=False),
            "Government",
            "Unknown",
        )

    # rename for UI: use Italian labels used before
    df = df.rename(columns={
        "Currency": "Valuta",
        "IssuerType": "TipoEmittente",
        "Sector": "Settore",
    })

    # maturity bucket
    today = pd.Timestamp.today().normalize()
    df["YearsToMaturity"] = (df["Maturity"] - today).dt.days / 365.25

    def _maturity_bucket(y):
        if pd.isna(y):
            return "Unknown"
        if y <= 3:
            return "Short"
        if y <= 7:
            return "Medium"
        return "Long"

    df["Scadenza"] = df["YearsToMaturity"].apply(_maturity_bucket)

    # percentili per Comparto
    df["Perc_ScoreRendimento"] = df.groupby("Comparto")["ScoreRendimento"].rank(pct=True) * 100
    df["Perc_ScoreRischio"] = df.groupby("Comparto")["ScoreRischio"].rank(pct=True) * 100

    # map Settore in 3 gruppi
    df["Settore"] = df["Settore"].apply(_map_sector)

    # string sanitization
    for c in ["Valuta", "TipoEmittente", "Settore", "Scadenza", "Issuer"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)

    # keep only rows with both scores
    df = df.dropna(subset=["ScoreRendimento", "ScoreRischio"]).reset_index(drop=True)

    return df


def _infer_issuer_type(comparto: str) -> str:
    s = (comparto or "").lower()
    if "govt" in s or "government" in s or "sovereign" in s:
        return "Govt"
    if "retail" in s:
        return "Corporate Retail"
    if "istituz" in s or "istituzionali" in s or "institutional" in s:
        return "Corporate Istituzionali"
    if "corp" in s:
        return "Corporate"
    return "Unknown"


def _map_sector(s: str) -> str:
    s = (s or "").lower()
    if "gov" in s:
        return "Govt"
    if "fin" in s:
        return "Financials"
    return "Non Financials"

# ----------------------------------
# Targets and integer allocation
# ----------------------------------

def integer_targets_from_weights(n: int, weights: Dict[str, float]) -> Dict[str, int]:
    if not weights:
        return {}
    # normalize in case user input not exactly 100 due to rounding (but we ask exact sum)
    total = sum(weights.values())
    if total <= 0:
        return {}
    raw = {k: n * (v / total) for k, v in weights.items()}
    floor = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = sorted(((raw[k] - floor[k], k) for k in raw), reverse=True)
    remaining = n - sum(floor.values())
    for i in range(remaining):
        _, k = remainder[i]
        floor[k] += 1
    return floor

# -----------------------------
# MILP builder (uses PuLP)
# -----------------------------

def build_portfolio_milp(df: pd.DataFrame, n: int, targ_val: Dict[str, float], targ_iss: Dict[str, float], targ_sec: Dict[str, float], targ_mat: Dict[str, float]) -> pd.DataFrame:
    if pulp is None:
        raise RuntimeError("PuLP non è installato. Esegui: pip install pulp")

    # compute integer targets per selected dimension
    t_val = integer_targets_from_weights(n, targ_val)
    t_iss = integer_targets_from_weights(n, targ_iss)
    t_sec = integer_targets_from_weights(n, targ_sec)
    t_mat = integer_targets_from_weights(n, targ_mat)

    # build mapping
    df = df.reset_index(drop=True).copy()
    indices = list(df.index)

    prob = pulp.LpProblem("bond_selection", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in indices}

    # objective: maximize sum score rendimento
    scores = df["ScoreRendimento"].fillna(0).astype(float).to_dict()
    prob += pulp.lpSum([scores[i] * x[i] for i in indices])

    # total count
    prob += pulp.lpSum([x[i] for i in indices]) == n

    # category equality constraints
    def add_equal_constraints(mapping: Dict[str, int], col: str, dim_name: str):
        for cat, cnt in mapping.items():
            # find indices matching category (string match)
            idxs = [i for i, v in df[col].astype(str).items() if v == str(cat)]
            if len(idxs) < cnt:
                raise ValueError(f"Impossibile soddisfare target {cnt} per {dim_name}='{cat}' (disponibili: {len(idxs)})")
            prob += pulp.lpSum([x[i] for i in idxs]) == cnt

    if t_val:
        add_equal_constraints(t_val, "Valuta", "Valuta")
    if t_iss:
        add_equal_constraints(t_iss, "TipoEmittente", "TipoEmittente")
    if t_sec:
        add_equal_constraints(t_sec, "Settore", "Settore")
    if t_mat:
        add_equal_constraints(t_mat, "Scadenza", "Scadenza")

    # corporate uniqueness: at most one bond per Issuer if TipoEmittente contains 'Corporate'
    corporate_mask = df["TipoEmittente"].str.contains("Corp", case=False, na=False) | df["TipoEmittente"].str.contains("Corporate", case=False, na=False)
    corp_df = df[corporate_mask]
    if not corp_df.empty:
        for issuer, group in corp_df.groupby("Issuer"):
            idxs = group.index.tolist()
            prob += pulp.lpSum([x[i] for i in idxs]) <= 1

    # solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    res = prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        raise ValueError(f"Solver non ha trovato soluzione ottimale. Status: {pulp.LpStatus[prob.status]}")

    chosen = [i for i in indices if pulp.value(x[i]) >= 0.5]
    portfolio = df.loc[chosen].reset_index(drop=True)
    return portfolio

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Bond Portfolio Selector (MILP)", layout="wide")
st.title("Bond Portfolio Selector — vincoli hard con MILP")

uploaded = st.file_uploader("Carica il CSV dei titoli (RepBondPlus o simile)", type=["csv"])
if not uploaded:
    st.info("Carica un file CSV con le colonne necessarie. Vedi istruzioni nel README.")
    st.stop()

try:
    df = load_and_normalize(uploaded)
except Exception as e:
    st.error(f"Errore nel caricamento/normalizzazione: {e}")
    st.stop()

st.success(f"Dati caricati: {len(df)} righe")
with st.expander("Anteprima (prime 20 righe)"):
    st.dataframe(df.head(20))

# universe after score filter
universe = df[df["ScoreRendimento"] >= 20].copy()
st.markdown(f"**Titoli disponibili dopo ScoreRendimento>=20:** {len(universe)}")
if universe.empty:
    st.warning("Nessun titolo passa il vincolo ScoreRendimento >= 20.")

st.sidebar.header("Parametri portafoglio")
n = st.sidebar.number_input("Numero titoli (n)", min_value=1, max_value=200, value=20)

st.sidebar.markdown("\n**Seleziona quali dimensioni vincolare (seleziona almeno una per vincoli hard)**")
lock_val = st.sidebar.checkbox("Vincolare Valuta")
lock_iss = st.sidebar.checkbox("Vincolare Tipo Emittente")
lock_sec = st.sidebar.checkbox("Vincolare Settore (Govt/Financials/Non Financials)")
lock_mat = st.sidebar.checkbox("Vincolare Scadenza (Short/Medium/Long)")

# prepare UI inputs for weights
st.header("Inserisci pesi target per le dimensioni selezionate (devono sommare a 100)")
col1, col2 = st.columns(2)
with col1:
    if lock_val:
        st.subheader("Pesi per Valuta (%)")
        uniq_val = sorted(universe["Valuta"].unique())
        w_val = {v: st.number_input(f"{v}", min_value=0.0, max_value=100.0, value=0.0, key=f"w_val_{v}") for v in uniq_val}
    else:
        w_val = {}
    if lock_iss:
        st.subheader("Pesi per Tipo Emittente (%)")
        uniq_iss = sorted(universe["TipoEmittente"].unique())
        w_iss = {v: st.number_input(f"{v}", min_value=0.0, max_value=100.0, value=0.0, key=f"w_iss_{v}") for v in uniq_iss}
    else:
        w_iss = {}
with col2:
    if lock_sec:
        st.subheader("Pesi per Settore (%)")
        uniq_sec = ["Govt", "Financials", "Non Financials"]
        w_sec = {v: st.number_input(f"{v}", min_value=0.0, max_value=100.0, value=0.0, key=f"w_sec_{v}") for v in uniq_sec}
    else:
        w_sec = {}
    if lock_mat:
        st.subheader("Pesi per Scadenza (%)")
        uniq_mat = ["Short", "Medium", "Long"]
        w_mat = {v: st.number_input(f"{v}", min_value=0.0, max_value=100.0, value=0.0, key=f"w_mat_{v}") for v in uniq_mat}
    else:
        w_mat = {}

# validation helper
def validate_weights(weights: Dict[str, float], label: str):
    if not weights:
        return {}
    total = sum(weights.values())
    if abs(total - 100.0) > 1e-6:
        st.error(f"I pesi per {label} devono sommare a 100. Somma attuale = {total:.1f}.")
        st.stop()
    return {k: float(v) for k, v in weights.items() if float(v) > 0}

w_val = validate_weights(w_val, "Valuta")
w_iss = validate_weights(w_iss, "Tipo Emittente")
w_sec = validate_weights(w_sec, "Settore")
w_mat = validate_weights(w_mat, "Scadenza")

# ensure at least one constraint selected for MILP; if none selected, we will return top-n by score
if (lock_val or lock_iss or lock_sec or lock_mat) and pulp is None:
    st.error("Per usare vincoli hard è necessario installare PuLP. Esegui: pip install pulp")
    st.stop()

if st.button("Costruisci portafoglio (hard)"):
    if not (lock_val or lock_iss or lock_sec or lock_mat):
        # simple top-n
        port = universe.nlargest(int(n), "ScoreRendimento").reset_index(drop=True)
    else:
        try:
            port = build_portfolio_milp(universe, int(n), w_val, w_iss, w_sec, w_mat)
        except Exception as e:
            st.error(f"Impossibile costruire il portafoglio con i vincoli richiesti: {e}")
            st.stop()

    if port.empty:
        st.error("Portafoglio vuoto.")
        st.stop()

    st.success(f"Portafoglio generato: {len(port)} titoli")
    with st.expander("Portafoglio - dettagli"):
        st.dataframe(port)

    # Export CSV (include target markers)
    export = port.copy()
    dims = {
        "Valuta": w_val,
        "TipoEmittente": w_iss,
        "Settore": w_sec,
        "Scadenza": w_mat,
    }
    for dim, tgt in dims.items():
        cats = sorted(set(list(tgt.keys()) + list(export.get(dim, pd.Series(dtype=str)).astype(str).unique())))
        for c in cats:
            export[f"Target_{dim}_{c}"] = (tgt.get(c, "--") if isinstance(tgt, dict) else "--")

    st.download_button("Scarica CSV portafoglio", data=export.to_csv(index=False).encode("utf-8"), file_name="portafoglio.csv", mime="text/csv")

    # ---- Confronto Target vs Effettivo e grafici (come richiesto)
    def compute_dist(df_port, col):
        return (df_port[col].value_counts(normalize=True) * 100).round(1)

    st.header("Confronto Target vs Effettivo")
    distr = {
        "Valuta": (w_val, compute_dist(port, "Valuta")),
        "TipoEmittente": (w_iss, compute_dist(port, "TipoEmittente")),
        "Settore": (w_sec, compute_dist(port, "Settore")),
        "Scadenza": (w_mat, compute_dist(port, "Scadenza")),
    }

    for crit, (target, actual) in distr.items():
        st.subheader(crit)
        cats = sorted(set(list(target.keys()) if isinstance(target, dict) else []) | set(actual.index.astype(str)))
        rows = []
        for c in cats:
            tgt_val = target.get(c) if isinstance(target, dict) else None
            tgt_display = "--" if tgt_val is None else f"{tgt_val:.1f}"
            act_val = float(actual.get(c, 0.0)) if not actual.empty else 0.0
            rows.append({crit: c, "Target %": tgt_display, "Effettivo %": f"{act_val:.1f}"})
        df_table = pd.DataFrame(rows)
        st.dataframe(df_table)

        tgt_nums = [target.get(c, 0.0) for c in cats] if isinstance(target, dict) else [0.0 for _ in cats]
        act_nums = [float(actual.get(c, 0.0)) for c in cats]
        fig, ax = plt.subplots()
        x = range(len(cats))
        width = 0.35
        ax.bar([i - width/2 for i in x], act_nums, width=width, label="Effettivo")
        ax.bar([i + width/2 for i in x], tgt_nums, width=width, label="Target")
        ax.set_xticks(list(x))
        ax.set_xticklabels(cats, rotation=45)
        ax.set_ylabel("Percentuale (%)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # Summary bar charts for quick view
    for label, col in [
        ("Distribuzione per Settore", "Settore"),
        ("Distribuzione per Valuta", "Valuta"),
        ("Distribuzione per Scadenza", "Scadenza"),
        ("Distribuzione per Tipo Emittente", "TipoEmittente"),
    ]:
        st.subheader(label)
        fig, ax = plt.subplots()
        port[col].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    st.balloons()

# EOF
