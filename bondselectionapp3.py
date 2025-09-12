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
    "ScoreRendimento": ["ScoreRendimento", "ScoreRet", "Score Rendimento"],
    "ScoreRischio": ["ScoreRischio", "ScoreRisk", "Score Rischio"],
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

    # sometimes exported files include an extra header row as data
    if "ISIN" in df.columns and isinstance(df.loc[0, "ISIN"], str) and df.loc[0, "ISIN"].strip().upper() == "ISIN":
        df = df.iloc[1:].reset_index(drop=True)

    # unify column names using aliases
    rename_map = {}
    for std, aliases in REQUIRED_COLS_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                rename_map[a] = std
                break
    df = df.rename(columns=rename_map)

    # required columns check
    for c in ["Comparto", "ISIN", "Issuer", "Maturity", "Currency", "ScoreRendimento", "ScoreRischio"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}. Found columns: {list(df.columns)}")

    # parse types
    df["Maturity"] = pd.to_datetime(df["Maturity"], errors="coerce", dayfirst=True)
    df["ScoreRendimento"] = pd.to_numeric(df["ScoreRendimento"], errors="coerce")
    df["ScoreRischio"] = pd.to_numeric(df["ScoreRischio"], errors="coerce")

    # infer optional columns
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

    # maturity buckets
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

    # percentiles inside Comparto
    df["Perc_ScoreRendimento"] = df.groupby("Comparto")["ScoreRendimento"].rank(pct=True) * 100
    df["Perc_ScoreRischio"] = df.groupby("Comparto")["ScoreRischio"].rank(pct=True) * 100

    # standard column names for UI
    df = df.rename(columns={"Currency": "Valuta", "IssuerType": "TipoEmittente", "Sector": "Settore"})

    # remap Settore into 3 groups
    df["Settore"] = df["Settore"].apply(_map_sector)

    # ensure string categories
    for c in ["Valuta", "TipoEmittente", "Settore", "Scadenza"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)

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


def _build_portfolio_soft(df: pd.DataFrame, n: int, targets: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """Greedy soft fallback: pick up to n bonds minimizing target overruns, tie-break by ScoreRendimento."""
    counts = {crit: {k: 0 for k in v} for crit, v in targets.items()}
    candidates = df.sort_values("ScoreRendimento", ascending=False).reset_index(drop=True)

    selected_rows: List[pd.Series] = []
    selected_isins: set = set()

    while len(selected_rows) < n and not candidates.empty:
        best_idx = None
        best_pen = float("inf")
        best_score = -float("inf")
        for idx, row in candidates.iterrows():
            isin = row.get("ISIN")
            if isin in selected_isins:
                continue
            pen = 0.0
            for crit, mapping in targets.items():
                if not mapping:
                    continue
                key = row.get(crit)
                cur = counts[crit].get(key, 0)
                tgt = mapping.get(key, 0)
                if tgt > 0:
                    pen += max(0, (cur + 1) - tgt)
            score = float(row.get("ScoreRendimento") or 0)
            if pen < best_pen or (abs(pen - best_pen) < 1e-12 and score > best_score):
                best_pen = pen
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        chosen = candidates.loc[best_idx]
        selected_rows.append(chosen)
        selected_isins.add(chosen.get("ISIN"))
        for crit in counts:
            key = chosen.get(crit)
            counts[crit][key] = counts[crit].get(key, 0) + 1
        candidates = candidates.drop(best_idx).reset_index(drop=True)

    df_selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    if len(df_selected) < n:
        remaining = df[~df["ISIN"].isin(df_selected.get("ISIN", pd.Series(dtype=object)))].nlargest(n - len(df_selected), "ScoreRendimento")
        df_selected = pd.concat([df_selected, remaining]).reset_index(drop=True)
    return df_selected.head(n)


def build_portfolio(df: pd.DataFrame, n: int, w: Weights) -> pd.DataFrame:
    """Try to satisfy hard targets for all dimensions using a greedy coverage algorithm; fall back to soft if infeasible."""
    df = df.copy()
    # Ensure category columns exist and are strings
    for col in ["Valuta", "TipoEmittente", "Settore", "Scadenza"]:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str)

    targets = _targets_from_weights(n, w)

    # If no targets provided at all, just return top-n
    if not any(targets.values()):
        return df.nlargest(n, "ScoreRendimento").reset_index(drop=True)

    # prepare remaining needs (mutable)
    needs = {crit: dict(mapping) for crit, mapping in targets.items()}

    candidates = df.sort_values("ScoreRendimento", ascending=False).reset_index(drop=True)
    selected = []
    selected_isins = set()

    dims = ["TipoEmittente", "Valuta", "Settore", "Scadenza"]

    # Greedy: pick candidate that covers the largest number of still-needed categories
    while len(selected) < n and not candidates.empty:
        best_idx = None
        best_cover = -1
        best_score = -1.0
        for idx, row in candidates.iterrows():
            isin = row["ISIN"]
            if isin in selected_isins:
                continue
            cover = 0
            for dim in dims:
                dim_needs = needs.get(dim, {})
                if dim_needs and dim_needs.get(row[dim], 0) > 0:
                    cover += 1
            score = float(row.get("ScoreRendimento") or 0)
            if cover > best_cover or (cover == best_cover and score > best_score):
                best_cover = cover
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        # if best_cover==0 means no remaining candidate helps satisfy any outstanding need -> infeasible
        if best_cover == 0:
            return _build_portfolio_soft(df, n, targets)
        chosen = candidates.loc[best_idx]
        selected.append(chosen)
        selected_isins.add(chosen["ISIN"])
        # decrement needs
        for dim in dims:
            if needs.get(dim) and needs[dim].get(chosen[dim], 0) > 0:
                needs[dim][chosen[dim]] -= 1
        # remove chosen from candidates
        candidates = candidates.drop(best_idx).reset_index(drop=True)

    # after selection, check if all needs satisfied
    for dim, mapping in needs.items():
        if any(v > 0 for v in mapping.values()):
            return _build_portfolio_soft(df, n, targets)

    # if selected < n (all needs satisfied), fill with highest scoring remaining
    if len(selected) < n:
        remaining = df[~df["ISIN"].isin(selected_isins)].nlargest(n - len(selected), "ScoreRendimento")
        for _, r in remaining.iterrows():
            selected.append(r)

    portfolio = pd.DataFrame(selected).drop_duplicates(subset=["ISIN"]).reset_index(drop=True)
    # ensure exactly n
    if len(portfolio) > n:
        portfolio = portfolio.nlargest(n, "ScoreRendimento").reset_index(drop=True)
    if len(portfolio) < n:
        # pad with top remaining
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
st.markdown("I pesi devono essere inseriti per tutte le sezioni e sommare a 100 (vincoli hard). Se i vincoli non sono realizzabili il sistema userà il fallback soft.)")

cols_sel = st.columns(2)
with cols_sel[0]:
    st.subheader("Valuta")
    unique_val = sorted(universe["Valuta"].dropna().astype(str).unique())
    w_val = {v: st.number_input(f"{v} (%)", min_value=0.0, max_value=100.0, value=0.0, key=f"w_val_{v}") for v in unique_val}
with cols_sel[1]:
    st.subheader("Tipo Emittente")
    unique_iss = sorted(universe["TipoEmittente"].dropna().astype(str).unique())
    # ensure standard issuer types exist in UI
    mandatory_issuers = [it for it in ["Govt", "Corporate Retail", "Corporate Istituzionali"] if it in unique_iss]
    if not mandatory_issuers:
        # fallback to detected ones
        mandatory_issuers = unique_iss
    w_iss = {it: st.number_input(f"{it} (%)", min_value=0.0, max_value=100.0, value=0.0, key=f"w_iss_{it}") for it in mandatory_issuers}

cols_sel2 = st.columns(2)
with cols_sel2[0]:
    st.subheader("Settore (Govt / Financials / Non Financials)")
    unique_sec = ["Govt", "Financials", "Non Financials"]
    w_sec = {s: st.number_input(f"{s} (%)", min_value=0.0, max_value=100.0, value=0.0, key=f"w_sec_{s}") for s in unique_sec}
with cols_sel2[1]:
    st.subheader("Scadenza")
    unique_mat = ["Short", "Medium", "Long"]
    w_mat = {m: st.number_input(f"{m} (%)", min_value=0.0, max_value=100.0, value=0.0, key=f"w_mat_{m}") for m in unique_mat}


def validate_weights(weights: Dict[str, float], label: str, mandatory: bool = True):
    total = sum(weights.values())
    if total <= 0:
        st.error(f"Devi inserire pesi per {label} (obbligatorio). Somma attuale = {total}.")
        st.stop()
    if abs(total - 100.0) > 1e-6:
        st.error(f"I pesi per {label} devono sommare a 100. Attuale somma = {total:.1f}%")
        st.stop()
    return {k: v for k, v in weights.items() if v > 0}

# Validate all weights (all hard)
w_val = validate_weights(w_val, "Valuta", mandatory=True)
w_iss = validate_weights(w_iss, "Tipo Emittente", mandatory=True)
w_sec = validate_weights(w_sec, "Settore", mandatory=True)
w_mat = validate_weights(w_mat, "Scadenza", mandatory=True)

if st.button("Costruisci portafoglio"):
    if universe.empty:
        st.error("Nessun titolo disponibile.")
        st.stop()

    weights = Weights(valuta=w_val, tipo_emittente=w_iss, settore=w_sec, scadenza=w_mat)

    try:
        port = build_portfolio(universe, int(n), weights)
    except Exception as e:
        st.error(f"Errore nella costruzione del portafoglio: {e}")
        st.stop()

    if port is None or port.empty:
        st.error("Impossibile costruire un portafoglio con i parametri forniti.")
        st.stop()

    st.success(f"Portafoglio generato: {len(port)} titoli")
    with st.expander("Portafoglio - tabella completa"):
        st.dataframe(port)

    # Prepare CSV export with target markers ("--" when not present)
    export = port.copy()
    dims = {
        "Valuta": w_val,
        "TipoEmittente": w_iss,
        "Settore": w_sec,
        "Scadenza": w_mat,
    }
    for dim, tgt in dims.items():
        cats = sorted(set(list(tgt.keys()) + list(export[dim].astype(str).unique())))
        for c in cats:
            export[f"Target_{dim}_{c}"] = (tgt.get(c, "--") if isinstance(tgt, dict) else "--")

    csv = export.to_csv(index=False).encode("utf-8")
    st.download_button("Scarica CSV portafoglio", data=csv, file_name="portafoglio.csv", mime="text/csv")

    # ---- Confronto Target vs Effettivo ----
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
        cats = sorted(set(list(target.keys()) + list(actual.index.astype(str))))
        rows = []
        for c in cats:
            tgt_val = target.get(c) if isinstance(target, dict) else None
            tgt_display = "--" if tgt_val is None else f"{tgt_val:.1f}"
            act_val = float(actual.get(c, 0.0)) if not actual.empty else 0.0
            rows.append({crit: c, "Target %": tgt_display, "Effettivo %": f"{act_val:.1f}"})
        df_table = pd.DataFrame(rows)
        st.dataframe(df_table)

        tgt_nums = [target.get(c, 0.0) for c in cats]
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

    st.balloons()

# End of file
