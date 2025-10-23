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
    # Opzionale: rimuovere caratteri non alfanumerici se necessario, ma partiamo più leggeri
    # cleaned = re.sub(r'[^a-z0-9]+', '', cleaned)
    return cleaned

def load_and_normalize(uploaded) -> pd.DataFrame:
    df = _read_data(uploaded)
    original_columns = list(df.columns) # Per diagnostica

    df.dropna(how='all', inplace=True)
    df.dropna(how='all', axis=1, inplace=True)

    # Rimuove header duplicato (controllando nome colonna pulito)
    df_columns_cleaned_for_check = {clean_col_name(col): col for col in df.columns}
    isin_cleaned = 'isin' # clean_col_name('ISIN')
    if isin_cleaned in df_columns_cleaned_for_check and len(df) > 0:
        original_isin_col_name = df_columns_cleaned_for_check[isin_cleaned]
        if isinstance(df.iloc[0][original_isin_col_name], str) and df.iloc[0][original_isin_col_name].strip().upper() == 'ISIN':
            df = df.iloc[1:].reset_index(drop=True)

    # Costruisce mappa alias pulito -> nome standard
    alias_to_standard_map = {}
    for standard_name, aliases in REQUIRED_COLS_ALIASES.items():
        for alias in aliases:
            cleaned_alias = clean_col_name(alias)
            alias_to_standard_map[cleaned_alias] = standard_name

    # Crea dizionario di rinomina (originale -> standard) con controllo conflitti
    rename_dict = {}
    standard_name_assignments = {} # standard -> originale
    column_mapping_details = {} # originale -> pulito -> standard (per diagnostica)

    for original_col in df.columns:
        cleaned_col = clean_col_name(original_col)
        target_standard_name = alias_to_standard_map.get(cleaned_col)
        column_mapping_details[original_col] = {'cleaned': cleaned_col, 'target': target_standard_name}

        if target_standard_name:
            # Controllo conflitti: questo nome standard è già stato assegnato a un'altra colonna?
            if target_standard_name in standard_name_assignments and standard_name_assignments[target_standard_name] != original_col:
                 raise ValueError(
                     f"**ERRORE: Conflitto nella mappatura delle colonne!**\n\n"
                     f"Sia la colonna originale '{standard_name_assignments[target_standard_name]}' che la colonna '{original_col}' "
                     f"(pulite rispettivamente come '{clean_col_name(standard_name_assignments[target_standard_name])}' e '{cleaned_col}') "
                     f"corrispondono entrambe al nome standard richiesto '{target_standard_name}'.\n"
                     f"Verifica le intestazioni nel tuo file per eliminare duplicati o alias ambigui."
                 )

            # Se non c'è conflitto o è la stessa colonna, assegna (se non già fatto)
            if target_standard_name not in rename_dict.values():
                 rename_dict[original_col] = target_standard_name
                 standard_name_assignments[target_standard_name] = original_col

    # Applica la rinomina
    try:
        df_renamed = df.rename(columns=rename_dict)
        renamed_columns = list(df_renamed.columns)
    except Exception as e:
        raise RuntimeError(f"Errore durante l'applicazione della rinomina delle colonne: {e}")


    # --- CONTROLLO ESPLICITO DOPO LA RINOMINA ---
    required = ["ISIN", "Issuer", "Maturity", "Currency", "ExchangeName", "ScoreRendimento", "ScoreRischio",
                "MarketPrice", "AccruedInterest", "DenominationMinimum", "DenominationIncrement"]
    current_columns_set = set(df_renamed.columns)
    missing_after_rename = [c for c in required if c not in current_columns_set]

    if missing_after_rename:
        raise ValueError(
            f"**ERRORE CRITICO: Colonne mancanti DOPO il tentativo di rinomina.**\n\n"
            f"**Colonne originali lette:**\n`{original_columns}`\n\n"
            f"**Dettagli mappatura (Originale -> Pulito -> Target Standard):**\n`{column_mapping_details}`\n\n"
            f"**Mappatura effettivamente applicata (Originale -> Standard):**\n`{rename_dict}`\n\n"
            f"**Colonne risultanti dopo rinomina:**\n`{renamed_columns}`\n\n"
            f"**Colonne richieste ma ancora mancanti:**\n`{', '.join(missing_after_rename)}`\n\n"
            f"**Controlla la mappatura e i nomi nel file originale. L'errore indica che la colonna '{missing_after_rename[0]}' non è stata trovata o mappata correttamente.**"
        )
    # --- FINE CONTROLLO ESPLICITO ---

    df = df_renamed # Prosegui con il dataframe rinominato

    # --- Conversione Tipi e Calcoli Successivi ---
    df["Maturity"] = pd.to_datetime(df["Maturity"], errors="coerce", dayfirst=True)
    for col in ["ScoreRendimento", "ScoreRischio", "MarketPrice", "AccruedInterest", "DenominationMinimum", "DenominationIncrement"]:
        try:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        except Exception as e:
            raise ValueError(f"Errore nella conversione numerica della colonna '{col}': {e}. Controlla i dati.")

    if "IssuerType" not in df.columns: df["IssuerType"] = df.get("Comparto", pd.Series(dtype=str)).astype(str).map(_infer_issuer_type)
    if "Sector" not in df.columns: df["Sector"] = np.where(df["IssuerType"].str.contains("Govt", case=False, na=False), "Government", "Unknown")

    df = df.rename(columns={"Currency": "Valuta", "IssuerType": "TipoEmittente", "Sector": "Settore"})

    today = pd.Timestamp.today().normalize()
    df["YearsToMaturity"] = (df["Maturity"] - today).dt.days / 365.25
    df["Scadenza"] = df["YearsToMaturity"].apply(lambda y: "Short" if y <= 3 else ("Medium" if y <= 7 else "Long") if pd.notna(y) else "Unknown")
    df["Settore"] = df["Settore"].apply(_map_sector)

    for c in ["Valuta", "TipoEmittente", "Settore", "Scadenza", "Issuer", "ExchangeName"]:
        if c in df.columns: df[c] = df[c].fillna("Unknown").astype(str)

    critical_numeric_cols = ["MarketPrice", "AccruedInterest", "DenominationMinimum", "DenominationIncrement", "ScoreRendimento", "ScoreRischio"]
    rows_with_nulls = df[critical_numeric_cols].isnull().any(axis=1)
    if rows_with_nulls.any():
         st.warning(f"Attenzione: {rows_with_nulls.sum()} righe contengono valori mancanti o non validi in colonne numeriche essenziali e verranno escluse.")

    # Il dropna ora dovrebbe funzionare perché abbiamo verificato la presenza delle colonne
    return df.dropna(subset=required).reset_index(drop=True)


# --- Il resto del codice rimane invariato ---

def _infer_issuer_type(comparto: str) -> str:
    s = str(comparto).lower()
    if any(k in s for k in ["govt", "government", "sovereign"]): return "Govt"
    if "retail" in s: return "Corporate Retail"
    if any(k in s for k in ["istituz", "institutional"]): return "Corporate Istituzionali"
    if "corp" in s: return "Corporate"
    return "Unknown"

def _map_sector(s: str) -> str:
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
    if not weights: return {}
    total = sum(weights.values())
    if total <= 0: return {}
    raw = {k: n * (v / total) for k, v in weights.items()}
    floor = {k: int(v) for k, v in raw.items()}
    remainder = sorted([(raw[k] - floor[k], k) for k in raw], reverse=True)
    remaining = n - sum(floor.values())
    for i in range(remaining):
        _, k = remainder[i % len(remainder)]
        floor[k] += 1
    return floor

def _solve_and_get_status(prob, solver):
    try:
        prob.solve(solver)
        return prob.status
    except Exception as e:
        raise RuntimeError(f"Il risolutore MILP ha generato un errore critico: {e}")

def build_portfolio_milp(df: pd.DataFrame, n: int, targ_val, targ_iss, targ_sec, targ_mat, weighting_scheme: str) -> pd.DataFrame:
    if pulp is None: raise RuntimeError("La libreria PuLP non è installata. Esegui: pip install pulp")

    prob = pulp.LpProblem("bond_selection", pulp.LpMaximize)
    df_copy = df.reset_index(drop=True)
    indices = list(df_copy.index)
    x = {i: pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in indices}

    scores = df_copy["ScoreRendimento"].fillna(0).to_dict()
    weights = {i: 1.0 for i in indices}
    if weighting_scheme == "Risk Weighted":
        risk_scores = df_copy["ScoreRischio"].fillna(0)
        avg_risk_score = risk_scores.mean()
        if avg_risk_score > 0: weights = {i: min(2.0, risk_scores.get(i, 0) / avg_risk_score) for i in indices}

    prob += pulp.lpSum(scores[i] * weights[i] * x[i] for i in indices)
    prob += pulp.lpSum(x[i] for i in indices) == n

    targets = {"Valuta": targ_val, "TipoEmittente": targ_iss, "Settore": targ_sec, "Scadenza": targ_mat}
    for col, weights_map in targets.items():
        if not weights_map: continue
        for cat, count in integer_targets_from_weights(n, weights_map).items():
            prob += pulp.lpSum(x[i] for i, v in df_copy[col].items() if v == str(cat)) == count

    corp_issuers = df_copy[df_copy["TipoEmittente"].str.contains("Corp", case=False, na=False)]["Issuer"].unique()
    for issuer in corp_issuers:
        prob += pulp.lpSum(x[i] for i in df_copy[df_copy["Issuer"] == issuer].index) <= 1

    solver = pulp.PULP_CBC_CMD(msg=False)
    status_code = _solve_and_get_status(prob, solver)

    status = pulp.LpStatus[status_code]
    if status == "Optimal":
        chosen_indices = [i for i in indices if x[i].varValue is not None and x[i].varValue > 0.5]
        return df_copy.loc[chosen_indices].reset_index(drop=True)
    else:
        raise ValueError(f"Il risolutore non ha trovato una soluzione ottimale. Stato: '{status}'. Questo indica che i vincoli sono in conflitto.")

def calculate_capital_allocation(portfolio: pd.DataFrame, total_capital: float, weighting_scheme: str, n: int) -> pd.DataFrame:
    if portfolio.empty or total_capital <= 0:
        return portfolio.assign(**{"Valore Nominale (€)": 0, "Controvalore di Mercato (€)": 0, "Peso (%)": 0})

    if weighting_scheme == 'Risk Weighted':
        total_risk = portfolio['ScoreRischio'].sum()
        # Evita divisione per zero se tutti i rischi sono 0
        raw_weights = portfolio['ScoreRischio'] / total_risk if total_risk > 0 else pd.Series([1/n] * n, index=portfolio.index)
    else: # Equally Weighted
        raw_weights = pd.Series([1/n] * n, index=portfolio.index)

    step = 1 / (4 * n) if n > 0 else 0.01 # Evita divisione per zero se n=0
    rounded_weights = (raw_weights / step).round() * step
    # Gestisce il caso in cui l'arrotondamento porti a somma zero
    if rounded_weights.sum() == 0 and n > 0 :
         normalized_weights = pd.Series([1/n] * n, index=portfolio.index) # Fallback a EW
    elif rounded_weights.sum() > 0:
         normalized_weights = rounded_weights / rounded_weights.sum()
    else: # Caso n=0 o somma pesi arrotondati = 0
         normalized_weights = pd.Series([0.0] * len(portfolio), index=portfolio.index) # Pesi a zero se n=0

    portfolio['target_weight'] = normalized_weights
    portfolio['target_capital'] = portfolio['target_weight'] * total_capital

    portfolio = portfolio.assign(**{"Valore Nominale (€)": 0.0, "Controvalore di Mercato (€)": 0.0})

    capital_allocated = 0
    for idx, row in portfolio.iterrows():
        min_nominal = row['DenominationMinimum']
        # Assicurati che MarketPrice e AccruedInterest siano numerici e non NaN
        market_price = row['MarketPrice'] if pd.notna(row['MarketPrice']) else 0
        accrued = row['AccruedInterest'] if pd.notna(row['AccruedInterest']) else 0
        market_value_per_unit = (market_price + accrued) / 100
        min_market_value = min_nominal * market_value_per_unit

        portfolio.loc[idx, 'Valore Nominale (€)'] = min_nominal
        portfolio.loc[idx, 'Controvalore di Mercato (€)'] = min_market_value
        capital_allocated += min_market_value

    capital_remaining = total_capital - capital_allocated
    if capital_remaining < 0:
        st.warning(f"Il capitale ({total_capital:,.0f}€) potrebbe essere insufficiente per l'investimento minimo ({capital_allocated:,.0f}€). Allocazione basata solo sul minimo.")
        capital_remaining = 0 # Non possiamo allocare oltre se già in negativo

    # Allocazione incrementale solo se c'è capitale rimanente positivo
    if capital_remaining > 0:
        # Ordina per differenza di capitale per allocare prima dove manca di più
        portfolio.sort_values(by='target_capital', ascending=False, inplace=True)
        
        for _ in range(n * 100): # Limita il numero di iterazioni per sicurezza
            portfolio['current_capital'] = portfolio['Controvalore di Mercato (€)']
            portfolio['capital_diff'] = portfolio['target_capital'] - portfolio['current_capital']
            
            # Trova il titolo più sotto-allocato che può ancora ricevere capitale
            eligible_for_increment = portfolio['capital_diff'] > 0
            if not eligible_for_increment.any(): break # Nessun titolo necessita più capitale

            most_underfunded_idx = portfolio[eligible_for_increment]['capital_diff'].idxmax()

            row = portfolio.loc[most_underfunded_idx]
            increment_nominal = row['DenominationIncrement']
            market_price = row['MarketPrice'] if pd.notna(row['MarketPrice']) else 0
            accrued = row['AccruedInterest'] if pd.notna(row['AccruedInterest']) else 0
            market_value_per_unit = (market_price + accrued) / 100
            increment_market_value = increment_nominal * market_value_per_unit

            if increment_market_value <= 0: # Evita loop infiniti se l'incremento ha costo zero
                 portfolio.loc[most_underfunded_idx, 'capital_diff'] = -1 # Escludi questo titolo da future allocazioni
                 continue

            if capital_remaining >= increment_market_value:
                portfolio.loc[most_underfunded_idx, 'Valore Nominale (€)'] += increment_nominal
                portfolio.loc[most_underfunded_idx, 'Controvalore di Mercato (€)'] += increment_market_value
                capital_remaining -= increment_market_value
            else:
                 # Se non basta per questo titolo, prova il successivo più sotto-allocato
                 portfolio.loc[most_underfunded_idx, 'capital_diff'] = -1 # Marca come non più allocabile
                 if (portfolio['capital_diff'] > 0).sum() == 0: # Se nessun altro titolo può ricevere, esci
                      break

    total_market_value = portfolio['Controvalore di Mercato (€)'].sum()
    if total_market_value > 0:
        portfolio['Peso (%)'] = (portfolio['Controvalore di Mercato (€)'] / total_market_value) * 100
    else:
         portfolio['Peso (%)'] = 0.0


    return portfolio.drop(columns=['target_weight', 'target_capital', 'current_capital', 'capital_diff'], errors='ignore')


# --- Interfaccia Streamlit (invariata dalla versione precedente) ---
st.set_page_config(page_title="Bond Portfolio Selector", layout="wide")
st.title("Bond Portfolio Selector — Ottimizzazione con Vincoli")

uploaded = st.file_uploader("Carica il file dei titoli (CSV o TXT)", type=["csv", "txt"])

if uploaded:
    try:
        df = load_and_normalize(uploaded)
        st.success(f"File caricato e processato: {len(df)} titoli validi.")

        st.sidebar.header("Parametri Portafoglio")
        total_capital = st.sidebar.number_input("Valore Portafoglio (€)", min_value=1000, value=100000, step=1000)
        n = st.sidebar.number_input("Numero di titoli (n)", min_value=1, max_value=len(df), value=min(10, len(df)))

        st.sidebar.markdown("---")
        st.sidebar.subheader("Filtro per Mercato")
        market_filter = st.sidebar.radio(
            "Scegli su quali mercati cercare i titoli:",
            ("Qualsiasi Mercato", "Solo Mercati Italiani"),
            key="market_filter"
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("Schema di Ponderazione")
        weighting_scheme = st.sidebar.radio(
            "Scegli il criterio per i pesi target:",
            ("Equally Weighted", "Risk Weighted"),
            key="weighting_scheme",
            help="**Equally Weighted**: I titoli hanno lo stesso peso target. **Risk Weighted**: I titoli hanno un peso target proporzionale al loro ScoreRischio."
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("Vincoli di Selezione (per numero di titoli)")
        use_val = st.sidebar.checkbox("Vincola per Valuta")
        use_iss = st.sidebar.checkbox("Vincola per Tipo Emittente")
        use_sec = st.sidebar.checkbox("Vincola per Settore")
        use_mat = st.sidebar.checkbox("Vincola per Scadenza")
        
        # Filtra l'universo in base ai filtri
        base_universe = df[df["ScoreRendimento"] >= 20].copy()
        
        if market_filter == "Solo Mercati Italiani":
            italian_exchanges = ['borsa italiana', 'euro tlx', 'hi-mtf']
            universe = base_universe[base_universe['ExchangeName'].str.lower().str.contains('|'.join(italian_exchanges), na=False)]
        else:
            universe = base_universe.copy()
            
        st.info(f"Universo investibile (dopo filtri): {len(universe)} titoli.")

        w_val, w_iss, w_sec, w_mat = {}, {}, {}, {}

        def get_weights_from_user(title, options, key_prefix):
            st.subheader(title)
            return {opt: st.number_input(f"{opt} (%)", 0.0, 100.0, 0.0, key=f"{key_prefix}_{opt}") for opt in options}

        cols = st.columns(2)
        with cols[0]:
            if use_val: w_val = get_weights_from_user("Pesi per Valuta", sorted(universe["Valuta"].unique()), "val")
            if use_iss: w_iss = get_weights_from_user("Pesi per Tipo Emittente", sorted(universe["TipoEmittente"].unique()), "iss")
        with cols[1]:
            if use_sec: w_sec = get_weights_from_user("Pesi per Settore", ["Govt", "Financials", "Non Financials"], "sec")
            if use_mat: w_mat = get_weights_from_user("Pesi per Scadenza", ["Short", "Medium", "Long"], "mat")

        def validate_weights(weights, label):
            if not weights: return True, {}
            total = sum(weights.values())
            if abs(total - 100.0) > 1e-6:
                st.error(f"I pesi per '{label}' devono sommare a 100. Somma attuale: {total:.1f}")
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
                st.error("L'universo investibile è vuoto. Prova a modificare i filtri.")
                st.stop()
            if n > len(universe):
                st.warning(f"Il numero di titoli richiesti ({n}) è maggiore dei titoli disponibili nell'universo ({len(universe)}). Verranno selezionati {len(universe)} titoli.")
                n = len(universe)


            with st.spinner("Ottimizzazione e allocazione in corso..."):
                try:
                    portfolio_selection = build_portfolio_milp(universe, n, w_val, w_iss, w_sec, w_mat, weighting_scheme)
                    portfolio = calculate_capital_allocation(portfolio_selection, total_capital, weighting_scheme, n)
                    st.success("Portafoglio generato con successo!")

                    cols_order = ['Peso (%)', 'Valore Nominale (€)', 'Controvalore di Mercato (€)'] + [c for c in portfolio.columns if c not in ['Peso (%)', 'Valore Nominale (€)', 'Controvalore di Mercato (€)']]
                    portfolio_display = portfolio[cols_order].copy()
                    for col in ['Valore Nominale (€)', 'Controvalore di Mercato (€)']:
                        portfolio_display[col] = portfolio_display[col].map('{:,.0f}'.format)
                    portfolio_display['Peso (%)'] = portfolio_display['Peso (%)'].map('{:.2f}%'.format)

                    st.dataframe(portfolio_display)

                    st.subheader("Riepilogo Allocazione Capitale")
                    invested_capital = portfolio['Controvalore di Mercato (€)'].sum()
                    st.metric("Capitale Investito", f"€ {invested_capital:,.2f}", f"€ {total_capital - invested_capital:,.2f} non allocato")

                    st.subheader("Distribuzione Pesi del Portafoglio")
                    fig_pie, ax_pie = plt.subplots()
                    # Gestisce il caso di pesi a zero per evitare errori nel grafico
                    plot_weights = portfolio['Peso (%)'] if portfolio['Peso (%)'].sum() > 0 else [1]*len(portfolio) 
                    ax_pie.pie(plot_weights, labels=portfolio['ISIN'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
                    ax_pie.axis('equal')
                    st.pyplot(fig_pie)

                    csv = portfolio.to_csv(index=False).encode('utf-8')
                    st.download_button("Scarica Portafoglio (CSV)", data=csv, file_name="portafoglio_ottimizzato.csv")

                    if any([w_val, w_iss, w_sec, w_mat]):
                        st.header("Confronto Target vs Effettivo (per numero titoli)")
                        compute_dist_count = lambda df_port, col: (df_port[col].value_counts(normalize=True) * 100) if not df_port.empty else pd.Series()

                        distr_count = {"Valuta": (w_val, compute_dist_count(portfolio, "Valuta")), "TipoEmittente": (w_iss, compute_dist_count(portfolio, "TipoEmittente")),
                                    "Settore": (w_sec, compute_dist_count(portfolio, "Settore")), "Scadenza": (w_mat, compute_dist_count(portfolio, "Scadenza"))}

                        for crit, (target, actual) in distr_count.items():
                            if not target: continue
                            st.subheader(f"Distribuzione per {crit}")
                            cats = sorted(set(list(target.keys())) | set(actual.index.astype(str)))
                            rows = []
                            for c in cats: rows.append({crit: c, "Target (num. titoli) %": target.get(c, 0.0), "Effettivo (num. titoli) %": float(actual.get(c, 0.0))})
                            df_table = pd.DataFrame(rows)
                            st.dataframe(df_table)

                    st.balloons()

                except (ValueError, RuntimeError) as e:
                    st.error(f"Impossibile costruire il portafoglio: {e}")
                except Exception as e:
                    st.error(f"Si è verificato un errore inatteso: {e}")

    except Exception as e:
        # --- BLOCCO DI ERRORE CON DIAGNOSTICA DETTAGLIATA ---
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_num = exc_tb.tb_lineno if exc_tb else 'N/A'
        tb_details = traceback.format_exception(exc_type, exc_obj, exc_tb)
        
        error_message = (
            f"**Si è verificato un errore nel caricamento o nella normalizzazione del file:**\n\n"
            f"**Tipo di errore:** `{exc_type.__name__}`\n"
            f"**Messaggio:** `{str(e)}`\n"
            f"**Punto critico nel codice (approssimativo):** Riga `{line_num}`\n\n"
            f"**Stack Trace (dettagli tecnici):**\n```\n{''.join(tb_details)}\n```\n\n"
            "**Causa probabile:** Potrebbe esserci un'incongruenza nel formato del file (caratteri speciali, separatori non corretti, BOM imprevisto) "
            "o un problema nella mappatura dei nomi delle colonne. Controlla il messaggio di errore specifico (`ValueError` conterrà dettagli sulle colonne) per identificare il problema."
        )
        st.error(error_message)
        # --- FINE BLOCCO ---
