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

from __future__ import annotations
import io
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# =============================
# Data Loading & Preparation
# =============================

REQUIRED_COLS_ALIASES = {
    "Comparto": ["Comparto", "Unnamed: 0"],
    "ISIN": ["ISIN", "ISIN Code"],
    "Issuer": ["Issuer", "Issuer Name"],
    "Maturity": ["Maturity", "Maturity Date"],
    "Coupon": ["Coupon", "Coupon Rate"],
    "Currency": ["Currency", "ISO Currency", "Valuta"],
    "Sector": ["Sector", "Settore"],
    "IssuerType": ["IssuerType", "Issuer Type", "TipoEmittente"],
    "ScoreRendimento": ["ScoreRendimento", "ScoreRet"],
    "ScoreRischio": ["ScoreRischio", "ScoreRisk"],
}


@st.cache_data(show_spinner=False)
def _read_csv_any(file) -> pd.DataFrame:
    # file can be a path or a BytesIO/StringIO from Streamlit uploader
    encodings = ["utf-8", "ISO-8859-1", "latin1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(file, encoding=enc)
        except Exception as e:
            last_err = e
            if hasattr(file, "seek"):
                file.seek(0)
    # re-raise last error if all failed
    raise last_err  # type: ignore[misc]


def load_data(uploaded) -> pd.DataFrame:
    df = _read_csv_any(uploaded)

    # If first data row duplicates headers, drop it
    if "ISIN" in df.columns and isinstance(df.loc[0, "ISIN"], str) and df.loc[0, "ISIN"].strip().upper() == "ISIN":
        df = df.iloc[1:].reset_index(drop=True)

    # Unify schema
    rename_map = {}
    for std_name, aliases in REQUIRED_COLS_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                rename_map[a] = std_name
                break
    df = df.rename(columns=rename_map)

    # Mandatory columns
    for col in ["Comparto", "ISIN", "Issuer", "Maturity", "Currency", "ScoreRendimento", "ScoreRischio"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Found: {list(df.columns)}")

    # Parse & types
    df["Maturity"] = pd.to_datetime(df["Maturity"], errors="coerce", dayfirst=True)
    df["ScoreRendimento"] = pd.to_numeric(df["ScoreRendimento"], errors="coerce")
    df["ScoreRischio"] = pd.to_numeric(df["ScoreRischio"], errors="coerce")

    # Optional columns fallback
    if "IssuerType" not in df.columns:
        df["IssuerType"] = df["Comparto"].astype(str).map(_infer_issuer_type)

    if "Sector" not in df.columns:
        # if missing entirely, create with Government/Unknown based on IssuerType
        df["Sector"] = np.where(df["IssuerType"].str.contains("Govt", case=False, na=False), "Government", "Unknown")
    else:
        mask_na = df["Sector"].isna()
        df.loc[mask_na, "Sector"] = np.where(
            df.loc[mask_na, "IssuerType"].str.contains("Govt", case=False, na=False),
            "Government",
            "Unknown",
        )

    # Derived features
    today = pd.Timestamp.today().normalize()
    df["YearsToMaturity"] = (df["Maturity"] - today).dt.days / 365.25
    df["MaturityBucket"] = df["YearsToMaturity"].apply(_maturity_bucket)

    # Percentiles within Comparto
    df["Perc_ScoreRendimento"] = df.groupby("Comparto")["ScoreRendimento"].rank(pct=True) * 100
    df["Perc_ScoreRischio"] = df.groupby("Comparto")["ScoreRischio"].rank(pct=True) * 100

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


def _maturity_bucket(years: Optional[float]) -> str:
    if years is None or (isinstance(years, float) and math.isnan(years)):
        return "Unknown"
    if years <= 3:
        return "Short"
    if years <= 7:
        return "Medium"
    return "Long"

# =============================
# Filters & Portfolio Builder
# =============================

def filtro_titoli(df: pd.DataFrame) -> pd.DataFrame:
    cond1 = df["ScoreRendimento"] >= 20
    cond2 = (df["Perc_ScoreRendimento"] > 50) & (df["Perc_ScoreRischio"] > 90)
    cond3 = (df["Perc_ScoreRendimento"] > 90) & (df["Perc_ScoreRischio"] > 50)
    return df[cond1 & (cond2 | cond3)].copy()


@dataclass
class Weights:
    currency: Dict[str, float]
    issuer_type: Dict[str, float]
    sector: Dict[str, float]
    maturity: Dict[str, float]

    def normalized(self) -> "Weights":
        return Weights(
            currency=_normalize(self.currency),
            issuer_type=_normalize(self.issuer_type),
            sector=_normalize(self.sector),
            maturity=_normalize(self.maturity),
        )


def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    d = {k: float(v) for k, v in d.items() if pd.notna(v)}
    total = sum(d.values())
    if total <= 0:
        return {k: 100.0 / len(d) for k in d} if d else {}
    return {k: (v / total) * 100.0 for k, v in d.items()}


def targets_from_weights(n: int, w: Weights) -> Dict[str, Dict[str, int]]:
    def alloc(weights: Dict[str, float]) -> Dict[str, int]:
        raw = {k: n * (v / 100.0) for k, v in weights.items()}
        floor = {k: int(math.floor(x)) for k, x in raw.items()}
        remainder = sorted(((raw[k] - floor[k], k) for k in raw), reverse=True)
        remaining = n - sum(floor.values())
        for i in range(remaining):
            _, k = remainder[i]
            floor[k] += 1
        return floor

    w = w.normalized()
    return {
        "currency": alloc(w.currency),
        "issuer_type": alloc(w.issuer_type),
        "sector": alloc(w.sector),
        "maturity": alloc(w.maturity),
    }


def build_portfolio(df: pd.DataFrame, n: int, w: Weights) -> pd.DataFrame:
    w = w.normalized()
    targets = targets_from_weights(n, w)

    counts = {k: {kk: 0 for kk in v} for k, v in targets.items()}
    selected = []
    candidates = df.sort_values(["ScoreRendimento"], ascending=[False]).reset_index(drop=True)

    def penalty(row) -> float:
        cur = str(row.get("Currency", "Unknown"))
        it = str(row.get("IssuerType", "Unknown"))
        sec = str(row.get("Sector", "Unknown"))
        mat = str(row.get("MaturityBucket", "Unknown"))
        pen = 0.0
        pen += _over_penalty(counts["currency"], targets["currency"], cur)
        pen += _over_penalty(counts["issuer_type"], targets["issuer_type"], it)
        pen += _over_penalty(counts["sector"], targets["sector"], sec)
        pen += _over_penalty(counts["maturity"], targets["maturity"], mat)
        # prefer combos aligned with larger weights
        weight_prod = (
            (w.currency.get(cur, 0) + 1e-9)
            * (w.issuer_type.get(it, 0) + 1e-9)
            * (w.sector.get(sec, 0) + 1e-9)
            * (w.maturity.get(mat, 0) + 1e-9)
        )
        return pen - 1e-6 * weight_prod

    for _ in range(n):
        best_i, best_pen, best_ret = None, float("inf"), -float("inf")
        for i, row in candidates.iterrows():
            if i in selected:
                continue
            p = penalty(row)
            r = float(row.get("ScoreRendimento", 0))
            if p < best_pen or (abs(p - best_pen) < 1e-12 and r > best_ret):
                best_i, best_pen, best_ret = i, p, r
        if best_i is None:
            break
        row = candidates.loc[best_i]
        selected.append(best_i)
        # update counts
        counts["currency"][str(row["Currency"])]= counts["currency"].get(str(row["Currency"]),0)+1
        counts["issuer_type"][str(row["IssuerType"])]= counts["issuer_type"].get(str(row["IssuerType"]),0)+1
        counts["sector"][str(row["Sector"])]= counts["sector"].get(str(row["Sector"]),0)+1
        counts["maturity"][str(row["MaturityBucket"])]= counts["maturity"].get(str(row["MaturityBucket"]),0)+1

    # fill if needed
    if len(selected) < n:
        for i in candidates.index:
            if i not in selected:
                selected.append(i)
                if len(selected) >= n:
                    break

    return candidates.loc[selected].reset_index(drop=True)


def _over_penalty(counts: Dict[str, int], targets: Dict[str, int], key: str) -> float:
    cur = counts.get(key, 0)
    tgt = targets.get(key, 0)
    return max(0, (cur + 1) - tgt)

# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="Bond Portfolio Selector", layout="wide")
st.title("📈 Bond Portfolio Selector")

st.sidebar.header("1) Carica dati")
uploaded = st.sidebar.file_uploader("CSV dei titoli", type=["csv"])  # expects RepBondPlus_20250825.csv-like schema

col_main, col_side = st.columns([3, 2], vertical_alignment="top")

with col_main:
    if uploaded is None:
        st.info("Carica un CSV per iniziare. Le colonne richieste sono: Comparto, ISIN, Issuer, Maturity, Currency, ScoreRendimento, ScoreRischio. Colonne opzionali: Sector, IssuerType.")
    else:
        try:
            df = load_data(uploaded)
        except Exception as e:
            st.error(f"Errore nel parsing del CSV: {e}")
            st.stop()

        st.success(f"File caricato: {len(df)} righe")
        with st.expander("Anteprima dati", expanded=False):
            st.dataframe(df.head(20))

        # Apply score filter
        df_filt = filtro_titoli(df)
        st.subheader("Titoli filtrati per Score")
        st.caption("Regola: ScoreRendimento >= 20 e (PercRet>50% & PercRischio>90% oppure PercRet>90% & PercRischio>50%)")
        st.dataframe(
            df_filt[[
                "Comparto","ISIN","Issuer","Currency","IssuerType","Sector","Maturity",
                "ScoreRendimento","ScoreRischio","Perc_ScoreRendimento","Perc_ScoreRischio","MaturityBucket"
            ]]
        )

        if df_filt.empty:
            st.error("Nessun titolo soddisfa i criteri. Verifica il dataset o i punteggi.")
            st.stop()

        # Controls
        st.subheader("Impostazioni portafoglio")
        n_sel = st.slider("Numero titoli nel portafoglio", 5, 100, 20, step=1)

        # Build weight inputs from available values
        def weight_inputs(label: str, values: List[str]) -> Dict[str, float]:
            st.markdown(f"**{label}**")
            cols = st.columns(3)
            weights: Dict[str, float] = {}
            for i, val in enumerate(sorted(values)):
                with cols[i % 3]:
                    weights[val] = st.number_input(f"{val}", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"w_{label}_{val}")
            return weights

        st.markdown("### Pesi target (somma 100 ciascuno; se ≠100 verranno normalizzati)")
        w_cur = weight_inputs("Valuta", sorted(df_filt["Currency"].dropna().astype(str).unique()))
        w_iss = weight_inputs("Tipo Emittente", sorted(df_filt["IssuerType"].dropna().astype(str).unique()))
        w_sec = weight_inputs("Settore", sorted(df_filt["Sector"].dropna().astype(str).unique()))
        w_mat = weight_inputs("Scadenza (bucket)", ["Short","Medium","Long"])  # buckets already computed

        st.caption("Suggerimento: imposta solo le categorie rilevanti (>0). Le altre verranno ignorate.")

        # Clean zeros
        w_cur = {k:v for k,v in w_cur.items() if v>0}
        w_iss = {k:v for k,v in w_iss.items() if v>0}
        w_sec = {k:v for k,v in w_sec.items() if v>0}
        w_mat = {k:v for k,v in w_mat.items() if v>0}

        # If any group left empty, auto-fill equal weights from available values
        def ensure_or_equal(weights: Dict[str, float], universe: List[str]) -> Dict[str, float]:
            if weights:
                return weights
            if not universe:
                return {}
            eq = 100.0 / len(universe)
            return {k: eq for k in universe}

        w_cur = ensure_or_equal(w_cur, sorted(df_filt["Currency"].dropna().astype(str).unique()))
        w_iss = ensure_or_equal(w_iss, sorted(df_filt["IssuerType"].dropna().astype(str).unique()))
        w_sec = ensure_or_equal(w_sec, sorted(df_filt["Sector"].dropna().astype(str).unique()))
        w_mat = ensure_or_equal(w_mat, ["Short","Medium","Long"])  # always offer three buckets

        weights = Weights(currency=w_cur, issuer_type=w_iss, sector=w_sec, maturity=w_mat)

        # Action
        if st.button("🧮 Costruisci portafoglio", type="primary"):
            portfolio = build_portfolio(df_filt, n_sel, weights)

            st.subheader("📊 Portafoglio selezionato")
            st.dataframe(portfolio[[
                "Comparto","ISIN","Issuer","Currency","IssuerType","Sector","Maturity",
                "ScoreRendimento","ScoreRischio","Perc_ScoreRendimento","Perc_ScoreRischio","MaturityBucket"
            ]])

            # Download
            csv = portfolio.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Scarica CSV", data=csv, file_name="portafoglio.csv", mime="text/csv")

            # Distributions
            st.markdown("### Distribuzioni effettive (% numero titoli)")
            colA, colB, colC, colD = st.columns(4)
            for col, series, title in [
                (colA, portfolio["Currency"], "Valuta"),
                (colB, portfolio["IssuerType"], "Tipo Emittente"),
                (colC, portfolio["Sector"], "Settore"),
                (colD, portfolio["MaturityBucket"], "Scadenza"),
            ]:
                with col:
                    if series.empty:
                        st.write("—")
                    else:
                        share = (series.value_counts(normalize=True) * 100).round(1)
                        st.bar_chart(share)

with col_side:
    st.header("Guida rapida")
    st.markdown(
        """
        **Passi**
        1. Carica il CSV dei titoli (schema RepBondPlus).
        2. Verifica i titoli filtrati secondo gli Score richiesti.
        3. Imposta i pesi target (Valuta / Emittente / Settore / Scadenza).
        4. Premi **Costruisci portafoglio** per vedere i risultati e scaricare il CSV.
        
        **Note**
        - Se i pesi non sommano a 100, verranno normalizzati automaticamente.
        - Le scadenze sono bucketizzate in **Short (≤3y)**, **Medium (3–7y)**, **Long (>7y)**.
        - La selezione è greedy con penalità quando si supera il target.
        """
    )

# =============================
# requirements.txt content (place in a separate file when deploying)
# =============================
# streamlit>=1.35
# pandas>=2.2
# numpy>=1.26
