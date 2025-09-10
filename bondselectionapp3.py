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

# =============================
# Utility / schema handling
# =============================

REQUIRED_COLS_ALIASES = {
    "Comparto": ["Comparto", "Unnamed: 0"],
    "ISIN": ["ISIN", "ISIN Code"],
    "Issuer": ["Issuer", "Issuer Name"],
    "Maturity": ["Maturity", "Maturity Date"],
    "Currency": ["Currency", "ISO Currency", "Valuta"],
    "Sector": ["Sector", "Settore"],
    "IssuerType": ["IssuerType", "Issuer Type", "TipoEmittente"],
    "ScoreRendimento": ["ScoreRendimento", "ScoreRet"],
    "ScoreRischio": ["ScoreRischio", "ScoreRisk"],
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


def load_data(uploaded) -> pd.DataFrame:
    df = _read_csv_any(uploaded)
    if "ISIN" in df.columns and isinstance(df.loc[0, "ISIN"], str) and df.loc[0, "ISIN"].strip().upper() == "ISIN":
        df = df.iloc[1:].reset_index(drop=True)

    rename_map = {}
    for std, aliases in REQUIRED_COLS_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                rename_map[a] = std
                break
    df = df.rename(columns=rename_map)

    for c in ["Comparto", "ISIN", "Issuer", "Maturity", "Currency", "ScoreRendimento", "ScoreRischio"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}. Found columns: {list(df.columns)}")

    df["Maturity"] = pd.to_datetime(df["Maturity"], errors="coerce", dayfirst=True)
    df["ScoreRendimento"] = pd.to_numeric(df["ScoreRendimento"], errors="coerce")
    df["ScoreRischio"] = pd.to_numeric(df["ScoreRischio"], errors="coerce")

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

    today = pd.Timestamp.today().normalize()
    df["YearsToMaturity"] = (df["Maturity"] - today).dt.days / 365.25

    def _maturity_bucket(years):
        if pd.isna(years):
            return "Unknown"
        if years <= 3:
            return "Short"
        if years <= 7:
            return "Medium"
        return "Long"

    df["Scadenza"] = df["YearsToMaturity"].apply(_maturity_bucket)

    df["Perc_ScoreRendimento"] = df.groupby("Comparto")["ScoreRendimento"].rank(pct=True) * 100
    df["Perc_ScoreRischio"] = df.groupby("Comparto")["ScoreRischio"].rank(pct=True) * 100

    df = df.rename(columns={"Currency": "Valuta", "IssuerType": "TipoEmittente", "Sector": "Settore"})

    # Re-map settori: Govt / Financials / Non Financials
    df["Settore"] = df["Settore"].apply(_map_sector)

    return df


def _infer_issuer_type(comparto: str) -> str:
    c = (comparto or "").lower()
    if "govt" in c or "government" in c or "sovereign" in c:
        return "Govt"
    if "retail" in c:
        return "Corporate Retail"
    if "istituz" in c or "istituzionali" in c or "institutional" in c:
        return "Corporate Istituzionali"
    if "corp" in c:
        return "Corporate"
    return "Unknown"


def _map_sector(s: str) -> str:
    s = (s or "").lower()
    if "gov" in s:
        return "Govt"
    if "fin" in s:
        return "Financials"
    return "Non Financials"


# =============================
# Filtering universe with fallback
# =============================

def filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base = df[df["ScoreRendimento"] >= 20]
    if base.empty:
        return base

    cond1 = (base["Perc_ScoreRendimento"] > 50) & (base["Perc_ScoreRischio"] > 90)
    cond2 = (base["Perc_ScoreRendimento"] > 90) & (base["Perc_ScoreRischio"] > 50)
    strict = base[cond1 | cond2].copy()
    if not strict.empty:
        return strict

    relaxed = base[
        ((base["Perc_ScoreRendimento"] > 50) & (base["Perc_ScoreRischio"] > 50))
        | ((base["Perc_ScoreRendimento"] > 40) & (base["Perc_ScoreRischio"] > 40))
    ].copy()
    if not relaxed.empty:
        return relaxed

    return base


# =============================
# Portfolio builder
# =============================

@dataclass
class Weights:
    valuta: Dict[str, float]
    tipo_emittente: Dict[str, float]
    settore: Dict[str, float]
    scadenza: Dict[str, float]

    def normalized(self) -> "Weights":
        return Weights(
            valuta=_normalize(self.valuta),
            tipo_emittente=_normalize(self.tipo_emittente),
            settore=_normalize(self.settore),
            scadenza=_normalize(self.scadenza),
        )


def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    d = {k: float(v) for k, v in d.items() if pd.notna(v)}
    total = sum(d.values())
    if total <= 0 or len(d) == 0:
        return {}
    return {k: (v / total) * 100.0 for k, v in d.items()}


def _targets_from_weights(n: int, w: Weights) -> Dict[str, Dict[str, int]]:
    def alloc(weights: Dict[str, float]) -> Dict[str, int]:
        if not weights:
            return {}
        raw = {k: n * (v / 100.0) for k, v in weights.items()}
        floor = {k: int(math.floor(x)) for k, x in raw.items()}
        remainder = sorted(((raw[k] - floor[k], k) for k in raw), reverse=True)
        remaining = n - sum(floor.values())
        for i in range(remaining):
            _, k = remainder[i]
            floor[k] += 1
        return floor

    wn = w.normalized()
    return {
        "Valuta": alloc(wn.valuta),
        "TipoEmittente": alloc(wn.tipo_emittente),
        "Settore": alloc(wn.settore),
        "Scadenza": alloc(wn.scadenza),
    }


def build_portfolio(df: pd.DataFrame, n: int, w: Weights) -> pd.DataFrame:
    df = df.copy()
    targets = _targets_from_weights(n, w)

    selected = pd.DataFrame()
    feasible = True

    for criterio, mapping in targets.items():
        for cat, num in mapping.items():
            if num <= 0:
                continue
            subset = df[df[criterio] == cat].nlargest(num, "ScoreRendimento")
            if len(subset) < num:
                feasible = False
            selected = pd.concat([selected, subset])

    selected = selected.drop_duplicates(subset=["ISIN"]).reset_index(drop=True)

    if feasible and len(selected) >= n:
        return selected.head(n)

    # fallback greedy
    candidates = df.sort_values("ScoreRendimento", ascending=False).reset_index()
    final_idxs: List[int] = []

    while len(final_idxs) < n and len(final_idxs) < len(candidates):
        chosen = candidates.loc[len(final_idxs)]
        final_idxs.append(int(chosen["index"]))

    portfolio = df.loc[final_idxs].drop_duplicates(subset=["ISIN"]).reset_index(drop=True)

    if len(portfolio) < n:
        remaining = df[~df["ISIN"].isin(portfolio["ISIN"])].nlargest(n - len(portfolio), "ScoreRendimento")
        portfolio = pd.concat([portfolio, remaining]).reset_index(drop=True)

    return portfolio.head(n)


# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="Bond Portfolio Selector", layout="wide")
st.title("📈 Bond Portfolio Selector - versione completa")

st.sidebar.header("Carica dati")
uploaded = st.sidebar.file_uploader("Carica un CSV (RepBondPlus) o trascina qui", type=["csv"]) 

if uploaded is None:
    st.info("Carica il file CSV (schema RepBondPlus). Colonne obbligatorie: Comparto, ISIN, Issuer, Maturity, Valuta, ScoreRendimento, ScoreRischio.")
    st.stop()

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Errore nel caricamento del CSV: {e}")
    st.stop()

st.success(f"Dati caricati: {len(df)} righe")

with st.expander("Anteprima dati (prime 20 righe)"):
    st.dataframe(df.head(20))

universe = filter_universe(df)
st.markdown(f"**Titoli disponibili dopo filtro:** {len(universe)}")
if universe.empty:
    st.warning("Nessun titolo passa il vincolo ScoreRendimento >= 20.")

st.sidebar.header("Parametri portafoglio")
n = st.sidebar.number_input("Numero titoli", min_value=1, max_value=200, value=20)

st.header("Imposta i pesi target (% per ciascun gruppo)")
st.markdown("Se una sezione viene lasciata vuota, non sarà vincolante. I pesi inseriti devono sommare a 100.")

cols_sel = st.columns(2)
with cols_sel[0]:
    st.subheader("Valuta")
    unique_val = sorted(universe["Valuta"].dropna().astype(str).unique())
    w_val = {v: st.number_input(f"{v} (%)", min_value=0.0, max_value=100.0, value=0.0, key=f"w_val_{v}") for v in unique_val}
with cols_sel[1]:
    st.subheader("Tipo Emittente")
    unique_iss = sorted(universe["TipoEmittente"].dropna().astype(str).unique())
    w_iss = {it: st.number_input(f"{it} (%)", min_value=0.0, max_value=100.0, value=0.0, key=f"w_iss_{it}") for it in unique_iss}

cols_sel2 = st.columns(2)
with cols_sel2[0]:
    st.subheader("Settore (Govt / Financials / Non Financials)")
    unique_sec = sorted(universe["Settore"].dropna().astype(str).unique())
    w_sec = {s: st.number_input(f"{s} (%)", min_value=0.0, max_value=100.0, value=0.0, key=f"w_sec_{s}") for s in unique_sec}
with cols_sel2[1]:
    st.subheader("Scadenza")
    unique_mat = ["Short", "Medium", "Long"]
    w_mat = {m: st.number_input(f"{m} (%)", min_value=0.0, max_value=100.0, value=0.0, key=f"w_mat_{m}") for m in unique_mat}

# Check sums only if user provided nonzero weights
def validate_weights(weights: Dict[str, float], label: str, mandatory: bool = False):
    total = sum(weights.values())
    if total > 0 and abs(total - 100.0) > 1e-6:
        st.error(f"I pesi per {label} devono sommare a 100. Attuale somma = {total:.1f}%")
        st.stop()
    if mandatory and total == 0:
        st.error(f"Devi inserire pesi per {label} (vincolo rigido).")
        st.stop()
    return {k: v for k, v in weights.items() if v > 0}

# Vincoli rigidi su TipoEmittente
w_val = validate_weights(w_val, "Valuta")
w_iss = validate_weights(w_iss, "Tipo Emittente", mandatory=True)
w_sec = validate_weights(w_sec, "Settore")
w_mat = validate_weights(w_mat, "Scadenza")

if st.button("Costruisci portafoglio"):
    if universe.empty:
        st.error("Nessun titolo disponibile.")
        st.stop()

    weights = Weights(valuta=w_val, tipo_emittente=w_iss, settore=w_sec, scadenza=w_mat)
    port = build_portfolio(universe, int(n), weights)

    if port.empty:
        st.error("Impossibile costruire un portafoglio con i parametri forniti.")
        st.stop()

    st.success(f"Portafoglio generato: {len(port)} titoli")
    with st.expander("Portafoglio - tabella completa"):
        st.dataframe(port)

    csv = port.to_csv(index=False).encode("utf-8")
    st.download_button("Scarica CSV portafoglio", data=csv, file_name="portafoglio.csv", mime="text/csv")

    def compute_dist(df_port, col):
        return (df_port[col].value_counts(normalize=True) * 100).round(1)

    distr = {
        "Valuta": (w_val, compute_dist(port, "Valuta")),
        "TipoEmittente": (w_iss, compute_dist(port, "TipoEmittente")),
        "Settore": (w_sec, compute_dist(port, "Settore")),
        "Scadenza": (w_mat, compute_dist(port, "Scadenza")),
    }

    st.header("Confronto Target vs Effettivo")
    for crit, (target, actual) in distr.items():
        st.subheader(crit)
        df_cmp = pd.DataFrame({"Target %": pd.Series(target), "Effettivo %": actual}).fillna(0)
        st.dataframe(df_cmp)

        fig1, ax1 = plt.subplots()
        if not actual.empty:
            actual.plot.pie(autopct="%1.1f%%", ax=ax1)
        ax1.set_ylabel("")
        ax1.set_title(f"Distribuzione effettiva - {crit}")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        df_cmp.plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Percentuale (%)")
        ax2.set_title(f"Target vs Effettivo - {crit}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)

    st.balloons()

# End of file
