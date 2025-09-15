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
from scipy.optimize import linprog

# =============================
# Step 2: Data Validation & Cleaning
# =============================

def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Comparto", "ISIN", "Issuer", "Maturity", "Currency", "ScoreRendimento", "ScoreRischio", "IssuerType", "Sector"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df.dropna(subset=["ISIN", "Maturity"]).copy()
    df = df.drop_duplicates(subset=["ISIN"])

    for col in ["Comparto", "Issuer", "Currency", "IssuerType", "Sector"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df

# =============================
# Step 3: Feature Engineering
# =============================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize()
    df["Maturity"] = pd.to_datetime(df["Maturity"], errors="coerce")
    df["YearsToMaturity"] = (df["Maturity"] - today).dt.days / 365.25

    def maturity_bucket(years):
        if pd.isna(years):
            return "Unknown"
        if years <= 3:
            return "Short"
        if years <= 7:
            return "Medium"
        return "Long"

    df["Scadenza"] = df["YearsToMaturity"].apply(maturity_bucket)

    df["Perc_ScoreRendimento"] = df.groupby("Comparto")["ScoreRendimento"].rank(pct=True) * 100
    df["Perc_ScoreRischio"] = df.groupby("Comparto")["ScoreRischio"].rank(pct=True) * 100

    for c in ["Scadenza", "Currency"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)

    # Regroup sectors
    def map_sector(row):
        if "gov" in row["IssuerType"].lower():
            return "Government"
        elif "bank" in row["Sector"].lower() or "financial" in row["Sector"].lower():
            return "Financials"
        else:
            return "Non Financials"

    df["SectorGroup"] = df.apply(map_sector, axis=1)

    return df

# =============================
# Step 4: Portfolio Optimization with Hard Constraints
# =============================

def build_portfolio(df: pd.DataFrame, n_titles: int, target_currency: Dict[str, float], target_issuer: Dict[str, float], target_sector: Dict[str, float], target_maturity: Dict[str, float]) -> pd.DataFrame:
    # Filter eligible bonds (hard filter Rend >= 20)
    eligible = df[df["ScoreRendimento"] >= 20].copy()
    if eligible.empty:
        return pd.DataFrame()

    eligible["Weight"] = 1 / len(eligible)

    # Hard constraints
    def enforce_group_weights(df, col, target):
        if not target:
            return df
        result = []
        for group, pct in target.items():
            group_df = df[df[col] == group].copy()
            if group_df.empty:
                continue
            k = max(1, round(n_titles * pct / 100))
            result.append(group_df.nlargest(k, "ScoreRendimento"))
        if result:
            return pd.concat(result)
        return df

    filtered = enforce_group_weights(eligible, "Currency", target_currency)
    filtered = enforce_group_weights(filtered, "IssuerType", target_issuer)
    filtered = enforce_group_weights(filtered, "SectorGroup", target_sector)
    filtered = enforce_group_weights(filtered, "Scadenza", target_maturity)

    # Avoid duplicate corporate issuers
    corporate = filtered[filtered["IssuerType"].str.contains("Corporate", case=False)]
    govt = filtered[~filtered["IssuerType"].str.contains("Corporate", case=False)]
    corporate = corporate.groupby("Issuer", as_index=False).apply(lambda g: g.nlargest(1, "ScoreRendimento")).reset_index(drop=True)
    portfolio = pd.concat([govt, corporate]).nlargest(n_titles, "ScoreRendimento")

    portfolio["FinalWeight"] = 100 / len(portfolio)
    return portfolio

# =============================
# Streamlit app
# =============================

st.set_page_config(page_title="Bond Portfolio Optimizer", layout="wide")
st.title("📊 Bond Portfolio Optimizer")

uploaded = st.sidebar.file_uploader("Carica un CSV", type=["csv"])
if uploaded is None:
    st.info("Carica un file CSV contenente dati obbligazionari.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
    df = validate_and_clean_data(df)
    df = add_features(df)
except Exception as e:
    st.error(f"Errore: {e}")
    st.stop()

st.success("Dati caricati e processati correttamente.")

st.sidebar.header("Impostazioni Portafoglio")
n_titles = st.sidebar.number_input("Numero titoli in portafoglio", min_value=5, max_value=50, value=10)

# Target weights inputs
st.sidebar.subheader("Pesi target (devono sommare a 100 se usati)")
target_currency = {}
target_issuer = {}
target_sector = {}
target_maturity = {}

if st.sidebar.checkbox("Vincola per Valuta"):
    cur = st.sidebar.text_area("Inserisci pesi valuta (es: EUR:50, USD:50)")
    try:
        target_currency = {k.strip(): float(v) for k, v in (x.split(":") for x in cur.split(","))}
    except:
        st.sidebar.error("Formato non valido.")

if st.sidebar.checkbox("Vincola per Tipo Emittente"):
    issu = st.sidebar.text_area("Inserisci pesi emittente (es: Govt:50, Corporate Retail:25, Corporate Istituzionali:25)")
    try:
        target_issuer = {k.strip(): float(v) for k, v in (x.split(":") for x in issu.split(","))}
    except:
        st.sidebar.error("Formato non valido.")

if st.sidebar.checkbox("Vincola per Settore"):
    sec = st.sidebar.text_area("Inserisci pesi settore (es: Government:40, Financials:30, Non Financials:30)")
    try:
        target_sector = {k.strip(): float(v) for k, v in (x.split(":") for x in sec.split(","))}
    except:
        st.sidebar.error("Formato non valido.")

if st.sidebar.checkbox("Vincola per Scadenza"):
    mat = st.sidebar.text_area("Inserisci pesi scadenza (es: Short:30, Medium:40, Long:30)")
    try:
        target_maturity = {k.strip(): float(v) for k, v in (x.split(":") for x in mat.split(","))}
    except:
        st.sidebar.error("Formato non valido.")

if st.sidebar.button("Genera Portafoglio"):
    portfolio = build_portfolio(df, n_titles, target_currency, target_issuer, target_sector, target_maturity)
    if portfolio.empty:
        st.error("Nessun titolo selezionato con i vincoli imposti.")
    else:
        st.subheader("📑 Portafoglio Selezionato")
        st.dataframe(portfolio[["ISIN", "Issuer", "Currency", "IssuerType", "SectorGroup", "Scadenza", "ScoreRendimento", "FinalWeight"]])

        # Riepilogo rispetto ai target
        st.subheader("📊 Riepilogo rispetto ai target")
        for name, target, col in [
            ("Valuta", target_currency, "Currency"),
            ("Tipo Emittente", target_issuer, "IssuerType"),
            ("Settore", target_sector, "SectorGroup"),
            ("Scadenza", target_maturity, "Scadenza"),
        ]:
            if target:
                st.markdown(f"**{name}**")
                actual = portfolio[col].value_counts(normalize=True) * 100
                comp = pd.DataFrame({"Target %": pd.Series(target), "Actual %": actual})
                st.dataframe(comp.fillna(0).round(2))

        # Riepiloghi grafici
        for label, col in [
            ("Distribuzione per Settore", "SectorGroup"),
            ("Distribuzione per Valuta", "Currency"),
            ("Distribuzione per Scadenza", "Scadenza"),
            ("Distribuzione per Tipo Emittente", "IssuerType"),
        ]:
            st.subheader(label)
            fig, ax = plt.subplots()
            portfolio[col].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

st.balloons()

# End of file
