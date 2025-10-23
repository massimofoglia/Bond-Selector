import io
import math
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Prova a importare pulp; l'app indicherà se manca
try:
    import pulp
except ImportError:
    pulp = None

# -----------------------------
# Alias delle colonne e loader
# -----------------------------
REQUIRED_COLS_ALIASES = {
    "Comparto": ["Comparto", "Unnamed: 0"],
    "ISIN": ["ISIN", "ISIN Code"],
    "Issuer": ["Issuer", "Issuer Name", "Emittente"],
    "Maturity": ["Maturity", "Maturity Date", "Scadenza Data", "MaturityDate"],
    "Currency": ["Currency", "ISO Currency", "Valuta"],
    "Sector": ["Sector", "Settore"],
    "IssuerType": ["IssuerType", "Issuer Type", "TipoEmittente"],
    "ExchangeName": ["ExchangeName", "Exchange Name", "Mercato"],
    "ScoreRendimento": ["ScoreRendimento", "Score Ret", "Score Rendimento", "ScoreRet"],
    "ScoreRischio": ["ScoreRischio", "Score Risk", "Score Rischio", "ScoreRisk"],
    "MarketPrice": ["MarketPrice", "Market Price", "Prezzo Mercato", "PrezzoDirty", "Ask Price"],
    "AccruedInterest": ["AccruedInterest", "Accrued Interest", "Rateo"],
    "DenominationMinimum": ["DenominationMinimum", "Denomination Minimum", "Lotto Minimo"],
    "DenominationIncrement": ["DenominationIncrement", "Denomination Increment", "Lotto Incremento"],
    "ModifiedDuration": ["Modified Duration", "DURATION", "Mod Dur"],
    "AskYield": ["Ask Yield", "YIELD Ask", "AskYld"],
}

def _read_data(uploaded) -> pd.DataFrame:
    """
    Legge un file CSV o TXT, gestendo diversi encoding e il BOM in modo robusto.
    """
    file_name = uploaded.name
    separator = '\t' if file_name.lower().endswith('.txt') else ','
    encodings = ["utf-8-sig", "utf-8", "ISO-8859-1", "latin1", "cp1252"]

    for enc in encodings:
        try:
            uploaded.seek(0)
            return pd.read_csv(uploaded, sep=separator, encoding=enc, engine='python', skipinitialspace=True)
        except Exception:
            continue
    raise ValueError("Impossibile leggere il file. Prova a salvarlo nuovamente in formato UTF-8.")

def clean_col_name(col_name: str) -> str:
    """ Pulisce il nome di una colonna per il matching."""
    if not isinstance(col_name, str):
        return ""
    # Rimuove spazi iniziali/finali, converte in minuscolo
    cleaned = col_name.strip().lower()
    # Rimuovi caratteri non alfanumerici eccetto spazi interni (per alias come "ask price")
    #cleaned = re.sub(r'[^a-z0-9\s]+', '', cleaned)
    #cleaned = re.sub(r'\s+', '', cleaned) # Rimuovi spazi interni solo se necessario
    return cleaned


def load_and_normalize(uploaded) -> pd.DataFrame:
    df = _read_data(uploaded)
    original_columns = list(df.columns)

    df.dropna(how='all', inplace=True)
    df.dropna(how='all', axis=1, inplace=True)

    df_columns_cleaned_for_check = {clean_col_name(col): col for col in df.columns}
    isin_cleaned = 'isin'
    if isin_cleaned in df_columns_cleaned_for_check and len(df) > 0:
        original_isin_col_name = df_columns_cleaned_for_check[isin_cleaned]
        first_isin_val = df.iloc[0][original_isin_col_name]
        if isinstance(first_isin_val, str) and first_isin_val.strip().upper() == 'ISIN':
            df = df.iloc[1:].reset_index(drop=True)

    alias_to_standard_map = {}
    for standard_name, aliases in REQUIRED_COLS_ALIASES.items():
        for alias in aliases:
            cleaned_alias = clean_col_name(alias)
            alias_to_standard_map[cleaned_alias] = standard_name

    rename_dict = {}
    standard_name_assignments = {}
    column_mapping_details = {}

    for original_col in df.columns:
        cleaned_col = clean_col_name(original_col)
        target_standard_name = alias_to_standard_map.get(cleaned_col)
        # Diagnostica: mappa anche se non c'è target
        column_mapping_details[original_col] = {'cleaned': cleaned_col, 'target': target_standard_name or 'N/A'}


        if target_standard_name:
            if target_standard_name in standard_name_assignments and standard_name_assignments[target_standard_name] != original_col:
                 raise ValueError(
                     f"**ERRORE: Conflitto nella mappatura delle colonne!**\n\n"
                     f"Sia la colonna originale '{standard_name_assignments[target_standard_name]}' che la colonna '{original_col}' "
                     f"corrispondono entrambe al nome standard richiesto '{target_standard_name}'.\n"
                     f"Verifica le intestazioni nel tuo file."
                 )
            # Rinnova solo se il nome standard non è GIA' presente o se la colonna corrente è quella giusta
            if target_standard_name not in df.columns or clean_col_name(original_col) == clean_col_name(target_standard_name):
                 if target_standard_name not in standard_name_assignments.values():
                     rename_dict[original_col] = target_standard_name
                     standard_name_assignments[target_standard_name] = original_col


    try:
        df_renamed = df.rename(columns=rename_dict)
        renamed_columns = list(df_renamed.columns)
    except Exception as e:
        raise RuntimeError(f"Errore durante l'applicazione della rinomina: {e}")

    required_standard_names = ["ISIN", "Issuer", "Maturity", "Currency", "ExchangeName", "ScoreRendimento", "ScoreRischio",
                               "MarketPrice", "AccruedInterest", "DenominationMinimum", "DenominationIncrement",
                               "ModifiedDuration", "AskYield"]
    current_columns_set = set(df_renamed.columns)
    missing_after_rename = [c for c in required_standard_names if c not in current_columns_set]

    if missing_after_rename:
        # Mostra più dettagli nella diagnostica
        found_cleaned = {clean_col_name(c):c for c in original_columns}
        alias_cleaned = {clean_col_name(a): a for std, alias_list in REQUIRED_COLS_ALIASES.items() for a in alias_list}
        raise ValueError(
            f"**ERRORE CRITICO: Colonne obbligatorie mancanti DOPO il tentativo di rinomina.**\n\n"
            f"**Colonne originali lette:**\n`{original_columns}`\n\n"
            #f"**Dettagli mappatura (Originale -> Pulito -> Target):**\n`{column_mapping_details}`\n\n"
            f"**Mappatura applicata (Originale -> Standard):**\n`{rename_dict}`\n\n"
            f"**Colonne dopo rinomina:**\n`{renamed_columns}`\n\n"
            f"**Colonne richieste ma MANCANTI:**\n`{', '.join(missing_after_rename)}`\n\n"
            f"**Verifica la mappatura e i nomi nel file originale.** Assicurati che per ogni colonna mancante, esista una colonna nel file originale che corrisponda (ignorando maiuscole/minuscole e spazi) a uno degli alias previsti."
        )

    df = df_renamed

    df["Maturity"] = pd.to_datetime(df["Maturity"], errors="coerce", dayfirst=True)
    numeric_cols = ["ScoreRendimento", "ScoreRischio", "MarketPrice", "AccruedInterest",
                    "DenominationMinimum", "DenominationIncrement", "ModifiedDuration", "AskYield"]
    for col in numeric_cols:
        try:
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            else:
                 raise ValueError(f"Colonna '{col}' mancante.")
        except Exception as e:
            raise ValueError(f"Errore nella conversione numerica della colonna '{col}': {e}.")

    critical_cols_check = required_standard_names + ["Maturity"]
    rows_before_dropna = len(df)
    df.dropna(subset=critical_cols_check, inplace=True)
    rows_after_dropna = len(df)
    if rows_before_dropna > rows_after_dropna:
        st.warning(f"{rows_before_dropna - rows_after_dropna} righe rimosse per valori mancanti nelle colonne critiche.")

    if "IssuerType" not in df.columns: df["IssuerType"] = df.get("Comparto", pd.Series(dtype=str)).astype(str).map(_infer_issuer_type)
    if "Sector" not in df.columns: df["Sector"] = np.where(df["IssuerType"].str.contains("Govt", case=False, na=False), "Government", "Unknown")

    df = df.rename(columns={"Currency": "Valuta", "IssuerType": "TipoEmittente", "Sector": "Settore"})

    today = pd.Timestamp.today().normalize()
    if pd.api.types.is_datetime64_any_dtype(df["Maturity"]):
         df["YearsToMaturity"] = (df["Maturity"] - today).dt.days / 365.25
    else:
         df["YearsToMaturity"] = pd.NA

    df["Scadenza"] = df["YearsToMaturity"].apply(lambda y: "Short" if y <= 3 else ("Medium" if y <= 7 else "Long") if pd.notna(y) else "Unknown")
    if "Settore" in df.columns:
        df["Settore"] = df["Settore"].apply(_map_sector)

    for c in ["Valuta", "TipoEmittente", "Settore", "Scadenza", "Issuer", "ExchangeName"]:
        if c in df.columns: df[c] = df[c].fillna("Unknown").astype(str)

    return df.reset_index(drop=True)

def _infer_issuer_type(comparto: str) -> str:
    # ... (invariato)
    s = str(comparto).lower()
    if any(k in s for k in ["govt", "government", "sovereign"]): return "Govt"
    if "retail" in s: return "Corporate Retail"
    if any(k in s for k in ["istituz", "institutional"]): return "Corporate Istituzionali"
    if "corp" in s: return "Corporate"
    return "Unknown"

def _map_sector(s: str) -> str:
    # ... (invariato)
    s_lower = str(s).lower()
    financial_keywords = [
        'fin', 'banking', 'insurance', 'leasing', 'securit',
        'other financial', 'real estate', 'asset management'
    ]
    if 'gov' in s_lower:
        return "Govt"
    if any(keyword in s_lower for keyword in financial_keywords):
        return "Financials"
    return "Non Financials"

def integer_targets_from_weights(n: int, weights: Dict[str, float]) -> Dict[str, int]:
    # ... (invariato)
    if not weights or n <= 0: return {}
    total = sum(weights.values())
    if total <= 0: return {}
    normalized_weights = {k: v / total for k, v in weights.items()}
    raw_counts = {k: n * w for k, w in normalized_weights.items()}
    floor_counts = {k: int(v) for k, v in raw_counts.items()}
    remainders = {k: raw_counts[k] - floor_counts[k] for k in raw_counts}
    sorted_remainders = sorted(remainders.items(), key=lambda item: item[1], reverse=True)
    remaining_n = max(0, n - sum(floor_counts.values()))
    for i in range(remaining_n):
        if not sorted_remainders: break
        key_to_increment = sorted_remainders[i % len(sorted_remainders)][0]
        floor_counts[key_to_increment] += 1
    return floor_counts

def _solve_and_get_status(prob, solver):
    # ... (invariato)
    try:
        prob.solve(solver)
        return prob.status
    except Exception as e:
        raise RuntimeError(f"Il risolutore MILP ha generato un errore critico: {e}")

def build_portfolio_milp(df: pd.DataFrame, n: int, targ_val, targ_iss, targ_sec, targ_mat, weighting_scheme: str) -> pd.DataFrame:
    # ... (invariato - l'ottimizzazione seleziona solo i titoli, non alloca capitale)
    if pulp is None: raise RuntimeError("La libreria PuLP non è installata.")
    if n <= 0: raise ValueError("Il numero di titoli (n) deve essere maggiore di 0.")
    if df.empty: raise ValueError("L'universo investibile fornito è vuoto.")

    prob = pulp.LpProblem("bond_selection", pulp.LpMaximize)
    df_copy = df.reset_index(drop=True)
    indices = list(df_copy.index)
    x = {i: pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in indices}

    scores = df_copy["ScoreRendimento"].fillna(0).to_dict()
    weights = {i: 1.0 for i in indices}
    if weighting_scheme == "Risk Weighted":
        risk_scores = df_copy["ScoreRischio"].fillna(0)
        mean_risk = risk_scores.mean()
        if pd.notna(mean_risk) and mean_risk > 0:
             weights = {i: min(2.0, risk_scores.get(i, 0) / mean_risk) for i in indices}

    prob += pulp.lpSum(scores[i] * weights[i] * x[i] for i in indices)
    prob += pulp.lpSum(x[i] for i in indices) == n

    targets = {"Valuta": targ_val, "TipoEmittente": targ_iss, "Settore": targ_sec, "Scadenza": targ_mat}
    for col_ui, weights_map in targets.items():
        if not weights_map: continue
        integer_counts = integer_targets_from_weights(n, weights_map)
        for cat, count in integer_counts.items():
            cat_indices = df_copy[df_copy[col_ui] == str(cat)].index.tolist()
            if cat_indices:
                prob += pulp.lpSum(x[i] for i in cat_indices) == count, f"Constraint_{col_ui}_{cat}"
            elif count > 0:
                 raise ValueError(f"Vincolo impossibile: {count} titoli per {col_ui}='{cat}', ma nessuno disponibile.")

    corp_issuers = df_copy[df_copy["TipoEmittente"].str.contains("Corp", case=False, na=False)]["Issuer"].unique()
    for issuer in corp_issuers:
        issuer_indices = df_copy[df_copy["Issuer"] == issuer].index.tolist()
        if len(issuer_indices) > 1:
            prob += pulp.lpSum(x[i] for i in issuer_indices) <= 1, f"Unique_Corp_{issuer}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status_code = _solve_and_get_status(prob, solver)
    status = pulp.LpStatus[status_code]

    if status == "Optimal":
        chosen_indices = [i for i in indices if x[i].varValue is not None and x[i].varValue > 0.5]
        if len(chosen_indices) != n:
             st.warning(f"Solver ottimale ma ha selezionato {len(chosen_indices)} titoli invece di {n}.")
        return df_copy.loc[chosen_indices].reset_index(drop=True)
    else:
        # ... (gestione errore invariata)
        details = ""
        if status == "Infeasible":
             try:
                  infeasible_constraints = [name for name, c in prob.constraints.items() if not c.valid(0)]
                  details = f" Vincoli potenzialmente problematici: {', '.join(infeasible_constraints[:5])}"
             except Exception:
                  details = " Impossibile ottenere dettagli sui vincoli."
        raise ValueError(f"Il risolutore non ha trovato una soluzione ottimale. Stato: '{status}'.{details}")


def calculate_capital_allocation(portfolio: pd.DataFrame, total_capital: float, weighting_scheme: str, n: int) -> pd.DataFrame:
    if portfolio.empty or total_capital <= 0 or n <= 0:
        return portfolio.assign(**{"Valore Nominale (€)": 0, "Controvalore di Mercato (€)": 0, "Peso (%)": 0})

    # --- CALCOLO PESI TARGET CON CAP ---
    if weighting_scheme == 'Risk Weighted':
        total_risk = portfolio['ScoreRischio'].sum()
        raw_weights = portfolio['ScoreRischio'] / total_risk if total_risk > 0 else pd.Series([1/n] * n, index=portfolio.index)
    else: # Equally Weighted
        raw_weights = pd.Series([1/n] * n, index=portfolio.index)

    step = 1 / (4 * n)
    rounded_weights = (raw_weights / step).round() * step

    normalized_weights = pd.Series([0.0]*len(portfolio), index=portfolio.index) # Inizializza a zero
    if rounded_weights.sum() > 0:
        normalized_weights = rounded_weights / rounded_weights.sum()
    elif n > 0: # Fallback a EW se somma arrotondata è 0
        normalized_weights = pd.Series([1/n] * n, index=portfolio.index)

    # Applica il CAP massimo del 2.5x EW
    max_weight_limit = (1 / n) * 2.5
    capped_weights = normalized_weights.copy()
    
    # Identifica chi supera e quanto
    exceeding_mask = capped_weights > max_weight_limit
    total_excess = (capped_weights[exceeding_mask] - max_weight_limit).sum()
    
    # Applica il cap
    capped_weights[exceeding_mask] = max_weight_limit
    
    # Redistribuisci l'eccesso a chi è sotto il cap
    below_cap_mask = ~exceeding_mask
    sum_below_cap = capped_weights[below_cap_mask].sum()

    if total_excess > 0 and sum_below_cap > 0:
         # Redistribuisci proporzionalmente al peso corrente sotto il cap
         capped_weights[below_cap_mask] += capped_weights[below_cap_mask] * (total_excess / sum_below_cap)
         # Potrebbe essere necessario ri-normalizzare leggermente a causa di arrotondamenti
         capped_weights = capped_weights / capped_weights.sum()
    elif total_excess > 0 and sum_below_cap == 0: # Tutti erano sopra il cap
         capped_weights = pd.Series([1/n] * n, index=portfolio.index) # Fallback EW
         
    portfolio['target_weight'] = capped_weights # Usa i pesi cappati come target
    portfolio['target_capital'] = portfolio['target_weight'] * total_capital
    max_capital_per_bond = max_weight_limit * total_capital # Limite massimo di capitale per titolo
    # ---- FINE CALCOLO PESI TARGET CON CAP ----


    portfolio = portfolio.assign(**{"Valore Nominale (€)": 0.0, "Controvalore di Mercato (€)": 0.0})
    capital_allocated = 0
    min_nominal_floor = 1000 # CORREZIONE 2

    # Check preliminare fattibilità lotto minimo rispetto al capitale totale
    min_total_investment = 0
    for idx, row in portfolio.iterrows():
         min_nominal_data = row['DenominationMinimum'] if pd.notna(row['DenominationMinimum']) else min_nominal_floor
         effective_min_nominal = max(min_nominal_floor, min_nominal_data)
         market_price = row['MarketPrice'] if pd.notna(row['MarketPrice']) else 100
         accrued = row['AccruedInterest'] if pd.notna(row['AccruedInterest']) else 0
         market_value_per_unit = (market_price + accrued) / 100
         min_total_investment += effective_min_nominal * market_value_per_unit
    if min_total_investment > total_capital:
         raise ValueError(f"Capitale ({total_capital:,.0f}€) insufficiente per investimento minimo richiesto ({min_total_investment:,.0f}€).")

    # Allocazione iniziale minima
    for idx, row in portfolio.iterrows():
        min_nominal_data = row['DenominationMinimum'] if pd.notna(row['DenominationMinimum']) else min_nominal_floor
        effective_min_nominal = max(min_nominal_floor, min_nominal_data)
        market_price = row['MarketPrice'] if pd.notna(row['MarketPrice']) else 100
        accrued = row['AccruedInterest'] if pd.notna(row['AccruedInterest']) else 0
        market_value_per_unit = (market_price + accrued) / 100
        min_market_value = effective_min_nominal * market_value_per_unit

        portfolio.loc[idx, 'Valore Nominale (€)'] = effective_min_nominal
        portfolio.loc[idx, 'Controvalore di Mercato (€)'] = min_market_value
        capital_allocated += min_market_value

    capital_remaining = max(0, total_capital - capital_allocated)

    # Allocazione incrementale con controllo CAP SUL CAPITALE EFFETTIVO
    if capital_remaining > 0:
        # Ordina per differenza CAPITALE ASSOLUTA (target - current)
        portfolio['capital_diff_abs'] = np.maximum(0, portfolio['target_capital'] - portfolio['Controvalore di Mercato (€)'])
        portfolio.sort_values(by='capital_diff_abs', ascending=False, inplace=True)

        can_allocate_more = True
        while capital_remaining > 1 and can_allocate_more: # Tolleranza 1€
            allocated_in_iteration = False
            indices_to_iterate = portfolio.index.tolist()

            for idx in indices_to_iterate:
                 row = portfolio.loc[idx]
                 increment_nominal_data = row['DenominationIncrement'] if pd.notna(row['DenominationIncrement']) and row['DenominationIncrement'] > 0 else 1000
                 effective_increment_nominal = max(100, increment_nominal_data)

                 market_price = row['MarketPrice'] if pd.notna(row['MarketPrice']) else 100
                 accrued = row['AccruedInterest'] if pd.notna(row['AccruedInterest']) else 0
                 market_value_per_unit = (market_price + accrued) / 100
                 increment_market_value = effective_increment_nominal * market_value_per_unit

                 if increment_market_value <= 0: continue

                 # Controlli:
                 # 1. C'è abbastanza capitale rimanente?
                 # 2. L'incremento non farebbe superare il CAPITALE MASSIMO per questo bond?
                 current_capital_pos = portfolio.loc[idx, 'Controvalore di Mercato (€)']
                 potential_capital_pos = current_capital_pos + increment_market_value

                 can_afford = capital_remaining >= increment_market_value
                 # Controlla se il NUOVO controvalore supera il limite massimo consentito
                 within_max_cap = potential_capital_pos <= max_capital_per_bond

                 if can_afford and within_max_cap:
                      portfolio.loc[idx, 'Valore Nominale (€)'] += effective_increment_nominal
                      portfolio.loc[idx, 'Controvalore di Mercato (€)'] += increment_market_value
                      capital_remaining -= increment_market_value
                      allocated_in_iteration = True
                      # Ricalcola diff per il riordino nel prossimo ciclo while
                      portfolio.loc[idx, 'capital_diff_abs'] = np.maximum(0, portfolio.loc[idx, 'target_capital'] - portfolio.loc[idx, 'Controvalore di Mercato (€)'])


            # Riordina DOPO aver tentato di allocare su tutti i titoli
            portfolio.sort_values(by='capital_diff_abs', ascending=False, inplace=True)

            if not allocated_in_iteration:
                 can_allocate_more = False # Stop se nessun incremento è stato possibile

    total_market_value = portfolio['Controvalore di Mercato (€)'].sum()
    if total_market_value > 0:
        portfolio['Peso (%)'] = (portfolio['Controvalore di Mercato (€)'] / total_market_value) * 100
    else:
         portfolio['Peso (%)'] = 0.0

    return portfolio.drop(columns=['target_weight', 'target_capital', 'capital_diff_abs'], errors='ignore')


# --- Interfaccia Streamlit (principalmente invariata, ma con titoli aggiunti) ---
st.set_page_config(page_title="Bond Portfolio Selector", layout="wide")
st.title("Bond Portfolio Selector — Ottimizzazione con Vincoli")

uploaded = st.file_uploader("Carica il file dei titoli (CSV o TXT)", type=["csv", "txt"])

if uploaded:
    try:
        with st.spinner("Caricamento e normalizzazione dati..."):
            df = load_and_normalize(uploaded)
        st.success(f"File caricato: {len(df)} titoli validi.")

        st.sidebar.header("Parametri Portafoglio")
        total_capital = st.sidebar.number_input("Valore Portafoglio (€)", min_value=1000, value=100000, step=1000)
        n = st.sidebar.number_input("Numero di titoli (n)", min_value=1, max_value=len(df), value=min(10, len(df)))

        st.sidebar.markdown("---")
        st.sidebar.subheader("Filtro per Mercato")
        market_filter = st.sidebar.radio( "Mercati:", ("Qualsiasi", "Solo Italiani"), key="market_filter")

        st.sidebar.markdown("---")
        st.sidebar.subheader("Schema Ponderazione Target")
        weighting_scheme = st.sidebar.radio( "Criterio Pesi:", ("Equally Weighted", "Risk Weighted"), key="weighting_scheme",
            help="EW: peso target uguale. RW: peso target proporzionale allo ScoreRischio (con cap 2x EW).")

        st.sidebar.markdown("---")
        st.sidebar.subheader("Vincoli Selezione (Numero Titoli)")
        use_val = st.sidebar.checkbox("Valuta")
        use_iss = st.sidebar.checkbox("Tipo Emittente")
        use_sec = st.sidebar.checkbox("Settore")
        use_mat = st.sidebar.checkbox("Scadenza")

        base_universe = df[df["ScoreRendimento"] >= 20].copy()

        if market_filter == "Solo Italiani":
            italian_exchanges = ['borsa italiana', 'euro tlx', 'hi-mtf']
            if 'ExchangeName' in base_universe.columns:
                 universe = base_universe[base_universe['ExchangeName'].str.lower().str.contains('|'.join(italian_exchanges), na=False)]
            else:
                 st.error("Colonna 'ExchangeName' non trovata per filtro mercato.")
                 universe = pd.DataFrame()
        else:
            universe = base_universe.copy()
            
        # ---- NUOVO: Filtro preliminare per lotti minimi troppo grandi ----
        if n > 0 and total_capital > 0 and not universe.empty:
             max_weight_limit_filter = (1 / n) * 2.5
             max_capital_per_bond_filter = max_weight_limit_filter * total_capital
             min_nominal_floor_filter = 1000
             
             # Calcola il costo del lotto minimo effettivo (>=1000)
             universe['effective_min_nominal'] = np.maximum(min_nominal_floor_filter, universe['DenominationMinimum'].fillna(min_nominal_floor_filter))
             universe['min_investment_cost'] = universe['effective_min_nominal'] * (universe['MarketPrice'].fillna(100) + universe['AccruedInterest'].fillna(0)) / 100
             
             # Filtra i titoli il cui costo minimo supera il cap
             oversized_lots_mask = universe['min_investment_cost'] > max_capital_per_bond_filter
             if oversized_lots_mask.any():
                  n_excluded = oversized_lots_mask.sum()
                  st.warning(f"{n_excluded} titoli esclusi dall'universo perché il loro lotto minimo (min {min_nominal_floor_filter}) supera il limite di peso massimo consentito ({max_weight_limit_filter*100:.1f}% del capitale).")
                  universe = universe[~oversized_lots_mask]
             # Rimuovi colonne temporanee
             universe = universe.drop(columns=['effective_min_nominal', 'min_investment_cost'], errors='ignore')
        # ---- FINE FILTRO LOTTI MINIMI ----


        st.info(f"Universo investibile (dopo filtri): {len(universe)} titoli.")

        w_val, w_iss, w_sec, w_mat = {}, {}, {}, {}

        def get_weights_from_user_ui(title, col_name_ui, df_source, universe_current, key_prefix):
            st.subheader(title) # Titolo aggiunto
            options = sorted(df_source[col_name_ui].unique())
            valid_options_in_universe = sorted(universe_current[col_name_ui].unique()) if col_name_ui in universe_current else []
            valid_options_to_show = valid_options_in_universe or options # Mostra tutte se universo filtrato è vuoto
            if not valid_options_to_show: return {}
            return {opt: st.number_input(f"{opt} (%)", 0.0, 100.0, 0.0, key=f"{key_prefix}_{opt}") for opt in valid_options_to_show}

        cols = st.columns(2)
        with cols[0]:
            if use_val: w_val = get_weights_from_user_ui("Pesi per Valuta", "Valuta", df, universe, "val")
            if use_iss: w_iss = get_weights_from_user_ui("Pesi per Tipo Emittente", "TipoEmittente", df, universe, "iss")
        with cols[1]:
            if use_sec:
                st.subheader("Pesi per Settore") # Titolo aggiunto
                w_sec = {opt: st.number_input(f"{opt} (%)", 0.0, 100.0, 0.0, key=f"sec_{opt}") for opt in ["Govt", "Financials", "Non Financials"]}
            if use_mat:
                st.subheader("Pesi per Scadenza") # Titolo aggiunto
                w_mat = {opt: st.number_input(f"{opt} (%)", 0.0, 100.0, 0.0, key=f"mat_{opt}") for opt in ["Short", "Medium", "Long"]}

        def validate_weights(weights, label):
            if not weights: return True, {}
            total = sum(weights.values())
            if abs(total - 100.0) > 0.01:
                st.error(f"I pesi per '{label}' devono sommare a 100. Somma: {total:.1f}")
                return False, {}
            return True, {k: v for k, v in weights.items() if v > 0}

        is_valid_val, w_val = validate_weights(w_val, "Valuta")
        is_valid_iss, w_iss = validate_weights(w_iss, "Tipo Emittente")
        is_valid_sec, w_sec = validate_weights(w_sec, "Settore")
        is_valid_mat, w_mat = validate_weights(w_mat, "Scadenza")
        is_valid = all([is_valid_val, is_valid_iss, is_valid_sec, is_valid_mat])

        if st.button("Costruisci Portafoglio"):
            if not is_valid: st.stop()
            if universe.empty:
                st.error("Universo investibile vuoto.")
                st.stop()

            if n > len(universe):
                st.warning(f"N titoli ({n}) > disponibili ({len(universe)}). Seleziono {len(universe)}.")
                n_effettivo = len(universe)
            else:
                 n_effettivo = n

            if n_effettivo <= 0:
                 st.error("N titoli deve essere > 0.")
                 st.stop()

            with st.spinner("Ottimizzazione e allocazione..."):
                try:
                    portfolio_selection = build_portfolio_milp(universe, n_effettivo, w_val, w_iss, w_sec, w_mat, weighting_scheme)
                    portfolio = calculate_capital_allocation(portfolio_selection, total_capital, weighting_scheme, n_effettivo)
                    st.success("Portafoglio generato!")

                    # CORREZIONE 1: Indice da 1
                    portfolio_display = portfolio.copy()
                    portfolio_display.index = np.arange(1, len(portfolio_display) + 1)

                    cols_order = ['Peso (%)', 'Valore Nominale (€)', 'Controvalore di Mercato (€)'] + [c for c in portfolio.columns if c not in ['Peso (%)', 'Valore Nominale (€)', 'Controvalore di Mercato (€)']]
                    portfolio_display = portfolio_display[cols_order]

                    for col in ['Valore Nominale (€)', 'Controvalore di Mercato (€)']:
                         portfolio_display[col] = pd.to_numeric(portfolio[col], errors='coerce').fillna(0).map('{:,.0f}'.format)
                    portfolio_display['Peso (%)'] = pd.to_numeric(portfolio['Peso (%)'], errors='coerce').fillna(0).map('{:.2f}%'.format)

                    st.dataframe(portfolio_display)

                    st.subheader("Riepilogo Allocazione")
                    invested_capital = portfolio['Controvalore di Mercato (€)'].sum()
                    remaining_capital = total_capital - invested_capital
                    if remaining_capital > max(1, total_capital * 0.001):
                        st.metric("Capitale Investito", f"€ {invested_capital:,.2f}", f"€ {remaining_capital:,.2f} non allocato (causa lotti/cap)")
                    else:
                         st.metric("Capitale Investito", f"€ {invested_capital:,.2f}", f"€ {remaining_capital:,.2f} non allocato")

                    # Grafico Scatter Duration/Yield
                    st.subheader("Posizionamento Rischio/Rendimento")
                    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
                    if 'ModifiedDuration' in universe.columns and 'AskYield' in universe.columns and \
                       'ModifiedDuration' in portfolio.columns and 'AskYield' in portfolio.columns:
                        weights_num = portfolio['Peso (%)'] / 100
                        portfolio_avg_duration = (portfolio['ModifiedDuration'].fillna(0) * weights_num).sum() if not weights_num.empty else 0
                        portfolio_avg_yield = (portfolio['AskYield'].fillna(0) * weights_num).sum() if not weights_num.empty else 0

                        ax_scatter.scatter(universe['ModifiedDuration'], universe['AskYield'], alpha=0.2, label='Universo', s=30, color='grey')
                        ax_scatter.scatter(portfolio['ModifiedDuration'], portfolio['AskYield'], color='red', label='Portafoglio', s=50)
                        if len(portfolio) <= 15:
                             for i, txt in enumerate(portfolio['ISIN']):
                                  ax_scatter.annotate(txt, (portfolio['ModifiedDuration'].iloc[i], portfolio['AskYield'].iloc[i]), fontsize=7, alpha=0.8)
                        if portfolio_avg_duration > 0 or portfolio_avg_yield > 0:
                           ax_scatter.scatter(portfolio_avg_duration, portfolio_avg_yield, color='blue', marker='*', s=250, label='Media Portafoglio', zorder=10)
                        ax_scatter.set_xlabel('Modified Duration')
                        ax_scatter.set_ylabel('Ask Yield (%)')
                        ax_scatter.set_title('Rischio (Duration) vs Rendimento (Yield)')
                        ax_scatter.legend()
                        ax_scatter.grid(True, linestyle='--', alpha=0.6)
                        st.pyplot(fig_scatter)
                    else: st.warning("Colonne Duration/Yield mancanti per grafico scatter.")

                    st.subheader("Distribuzione Pesi")
                    fig_pie, ax_pie = plt.subplots()
                    plot_weights = portfolio['Peso (%)'] if 'Peso (%)' in portfolio.columns and portfolio['Peso (%)'].sum() > 0 else [1]*len(portfolio)
                    ax_pie.pie(plot_weights, labels=portfolio['ISIN'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
                    ax_pie.axis('equal')
                    st.pyplot(fig_pie)

                    csv = portfolio[cols_order].to_csv(index=False).encode('utf-8')
                    st.download_button("Scarica Portafoglio (CSV)", data=csv, file_name="portafoglio_ottimizzato.csv")

                    # Grafici Confronto Target vs Effettivo (basati su pesi effettivi)
                    if any([w_val, w_iss, w_sec, w_mat]):
                        st.header("Confronto Target Vincoli vs Pesi Effettivi")
                        compute_weight_dist = lambda df_port, col: df_port.groupby(col)['Peso (%)'].sum() if col in df_port and 'Peso (%)' in df_port else pd.Series(dtype=float)

                        distr_weights = {"Valuta": (w_val, compute_weight_dist(portfolio, "Valuta")),
                                        "TipoEmittente": (w_iss, compute_weight_dist(portfolio, "TipoEmittente")),
                                        "Settore": (w_sec, compute_weight_dist(portfolio, "Settore")),
                                        "Scadenza": (w_mat, compute_weight_dist(portfolio, "Scadenza"))}

                        for crit_std, (target_num_perc, actual_weight_perc) in distr_weights.items():
                            if not target_num_perc: continue
                            crit_ui = crit_std.replace("TipoEmittente", "Tipo Emittente") # Nome UI corretto
                            st.subheader(f"Distribuzione Pesi per {crit_ui}")

                            cats = sorted(set(list(target_num_perc.keys())) | set(actual_weight_perc.index.astype(str)))
                            rows = []
                            for c in cats:
                                rows.append({crit_ui: c, "Target (Input) %": target_num_perc.get(c, 0.0), "Effettivo (Capitale) %": float(actual_weight_perc.get(c, 0.0)) })
                            df_table = pd.DataFrame(rows).round(1)
                            st.dataframe(df_table.set_index(crit_ui))

                            fig_bar, ax_bar = plt.subplots()
                            x = np.arange(len(cats))
                            width = 0.35
                            ax_bar.bar(x - width/2, df_table["Effettivo (Capitale) %"], width, label="Effettivo (Capitale %)")
                            ax_bar.bar(x + width/2, df_table["Target (Input) %"], width, label="Target (Input %)")
                            ax_bar.set_ylabel("%")
                            ax_bar.set_title(f'Confronto Pesi per {crit_ui}')
                            ax_bar.set_xticks(x)
                            ax_bar.set_xticklabels(cats, rotation=45, ha="right")
                            ax_bar.legend()
                            plt.tight_layout()
                            st.pyplot(fig_bar)


                    st.balloons()

                except (ValueError, RuntimeError) as e:
                    st.error(f"Impossibile costruire il portafoglio: {e}")
                except Exception as e:
                    st.error(f"Errore inatteso:")
                    st.exception(e)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_num = exc_tb.tb_lineno if exc_tb else 'N/A'
        tb_details = traceback.format_exception(exc_type, exc_obj, exc_tb)
        error_message = (
            f"**Errore nel caricamento/normalizzazione:**\n\n"
            f"**Tipo:** `{exc_type.__name__}`\n"
            f"**Messaggio:** `{str(e)}`\n"
            f"**Riga:** `{line_num}`\n\n"
            f"**Traceback:**\n```\n{''.join(tb_details)}\n```\n\n"
            "**Verifica il formato file e mappatura colonne.**"
        )
        st.error(error_message)
