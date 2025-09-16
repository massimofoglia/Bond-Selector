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
# Debug helper
# -----------------------------
def debug_log(msg):
    st.session_state.setdefault("debug_msgs", []).append(msg)
    print(msg)

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

def load_and_normalize(uploaded_file):
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    colmap = {}
    for canon, aliases in REQUIRED_COLS_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                colmap[a] = canon
                break
    df = df.rename(columns=colmap)
    missing = [c for c in REQUIRED_COLS_ALIASES if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel file: {missing}")
    return df

# -----------------------------
# Funzione di costruzione portafoglio MILP
# -----------------------------
def build_portfolio_milp(universe, n, w_val, w_iss, w_sec, w_mat):
    if pulp is None:
        raise RuntimeError("PuLP non installato, impossibile risolvere MILP.")

    debug_log("Inizio definizione problema MILP")
    prob = pulp.LpProblem("PortfolioSelection", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", universe.index, lowBound=0, upBound=1, cat="Binary")
    debug_log(f"Definite {len(x)} variabili binarie")

    prob += pulp.lpSum(universe.loc[i, "ScoreRendimento"] * x[i] for i in universe.index), "TotalReturn"
    prob += pulp.lpSum(x[i] for i in universe.index) == n, "Cardinality"

    if w_val:
        for cur, pct in w_val.items():
            prob += pulp.lpSum(x[i] for i in universe.index if universe.loc[i, "Currency"] == cur) == int(pct/100*n), f"Valuta_{cur}"

    if w_iss:
        for typ, pct in w_iss.items():
            prob += pulp.lpSum(x[i] for i in universe.index if universe.loc[i, "IssuerType"] == typ) == int(pct/100*n), f"IssuerType_{typ}"

    if w_sec:
        for sec, pct in w_sec.items():
            prob += pulp.lpSum(x[i] for i in universe.index if universe.loc[i, "Sector"] == sec) == int(pct/100*n), f"Sector_{sec}"

    if w_mat:
        for mat, pct in w_mat.items():
            prob += pulp.lpSum(x[i] for i in universe.index if universe.loc[i, "Maturity"] == mat) == int(pct/100*n), f"Maturity_{mat}"

    # Vincolo: non più di un bond corporate per emittente
    for issuer in universe["Issuer"].unique():
        sub = universe[(universe["Issuer"] == issuer) & (universe["IssuerType"] == "Corporate")]
        if not sub.empty:
            prob += pulp.lpSum(x[i] for i in sub.index) <= 1, f"IssuerCorp_{issuer}"

    debug_log("Avvio solver MILP")
    status = prob.solve(pulp.PULP_CBC_CMD(msg=True))
    debug_log(f"Solver terminato con status: {pulp.LpStatus[status]}")

    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Soluzione non ottimale: {pulp.LpStatus[status]}")

    chosen_idx = [i for i in universe.index if pulp.value(x[i]) > 0.5]
    return universe.loc[chosen_idx]

# -----------------------------
# Funzioni di riepilogo
# -----------------------------
def show_summary(df, col, title):
    st.subheader(f"Distribuzione per {title}")
    counts = df[col].value_counts(normalize=True) * 100
    st.write(counts.round(2).astype(str) + "%")
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("%")
    st.pyplot(fig)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Bond Portfolio Selection App")

uploaded = st.file_uploader("Carica file CSV/XLSX")
if uploaded:
    try:
        df = load_and_normalize(uploaded)
        st.success(f"File caricato: {len(df)} righe")
    except Exception as e:
        st.error(f"Errore caricamento file: {e}")
        st.stop()

    n = st.number_input("Numero di titoli nel portafoglio", min_value=1, max_value=100, value=10)

    w_val = {}
    if st.checkbox("Vincola per Valuta"):
        for c in df["Currency"].unique():
            w_val[c] = st.number_input(f"Peso {c} (%)", min_value=0, max_value=100, value=0)
        if sum(w_val.values()) != 100:
            st.error("I pesi valuta devono sommare a 100")
            st.stop()

    w_iss = {}
    if st.checkbox("Vincola per Tipo Emittente"):
        for t in df["IssuerType"].unique():
            w_iss[t] = st.number_input(f"Peso {t} (%)", min_value=0, max_value=100, value=0)
        if sum(w_iss.values()) != 100:
            st.error("I pesi tipo emittente devono sommare a 100")
            st.stop()

    w_sec = {}
    if st.checkbox("Vincola per Settore"):
        for s in df["Sector"].unique():
            w_sec[s] = st.number_input(f"Peso {s} (%)", min_value=0, max_value=100, value=0)
        if sum(w_sec.values()) != 100:
            st.error("I pesi settore devono sommare a 100")
            st.stop()

    w_mat = {}
    if st.checkbox("Vincola per Maturity"):
        for m in df["Maturity"].unique():
            w_mat[m] = st.number_input(f"Peso {m} (%)", min_value=0, max_value=100, value=0)
        if sum(w_mat.values()) != 100:
            st.error("I pesi maturity devono sommare a 100")
            st.stop()

    if st.button("Costruisci portafoglio"):
        try:
            port = build_portfolio_milp(df, int(n), w_val, w_iss, w_sec, w_mat)
            if port.empty:
                st.error("Portafoglio vuoto.")
            else:
                st.success(f"Portafoglio generato: {len(port)} titoli")
                st.dataframe(port)

                # Riepiloghi
                show_summary(port, "Currency", "Valuta")
                show_summary(port, "IssuerType", "Tipo Emittente")
                show_summary(port, "Sector", "Settore")
                show_summary(port, "Maturity", "Maturity")
                show_summary(port, "Issuer", "Emittente")
        except Exception as e:
            st.error(f"Errore costruzione portafoglio: {e}")

    if "debug_msgs" in st.session_state:
        with st.expander("Debug log"):
            st.text("\n".join(st.session_state["debug_msgs"]))

# EOF
