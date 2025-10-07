import io
import math
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
    "ScoreRendimento": ["ScoreRendimento", "Score Ret", "Score Rendimento", "ScoreRet"],
    "ScoreRischio": ["ScoreRischio", "Score Risk", "Score Rischio", "ScoreRisk"],
}

def _read_csv_any(uploaded) -> pd.DataFrame:
    encodings = ["utf-8", "ISO-8859-1", "latin1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def load_and_normalize(uploaded) -> pd.DataFrame:
    df = _read_csv_any(uploaded)
    if "ISIN" in df.columns and isinstance(df.loc[0, "ISIN"], str) and df.loc[0, "ISIN"].strip().upper() == "ISIN":
        df = df.iloc[1:].reset_index(drop=True)

    rename_map = {a: std for std, aliases in REQUIRED_COLS_ALIASES.items() for a in aliases if a in df.columns}
    df = df.rename(columns=rename_map)

    required = ["Comparto", "ISIN", "Issuer", "Maturity", "Currency", "ScoreRendimento", "ScoreRischio"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"Colonne obbligatorie mancanti. Trovate: {list(df.columns)}, Richieste: {required}")

    df["Maturity"] = pd.to_datetime(df["Maturity"], errors="coerce", dayfirst=True)
    for col in ["ScoreRendimento", "ScoreRischio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "IssuerType" not in df.columns:
        df["IssuerType"] = df["Comparto"].astype(str).map(_infer_issuer_type)
    if "Sector" not in df.columns:
        df["Sector"] = np.where(df["IssuerType"].str.contains("Govt", case=False, na=False), "Government", "Unknown")

    df = df.rename(columns={"Currency": "Valuta", "IssuerType": "TipoEmittente", "Sector": "Settore"})

    today = pd.Timestamp.today().normalize()
    df["YearsToMaturity"] = (df["Maturity"] - today).dt.days / 365.25
    df["Scadenza"] = df["YearsToMaturity"].apply(lambda y: "Short" if y <= 3 else ("Medium" if y <= 7 else "Long") if pd.notna(y) else "Unknown")
    df["Settore"] = df["Settore"].apply(_map_sector)

    for c in ["Valuta", "TipoEmittente", "Settore", "Scadenza", "Issuer"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)

    return df.dropna(subset=["ScoreRendimento", "ScoreRischio"]).reset_index(drop=True)

def _infer_issuer_type(comparto: str) -> str:
    s = str(comparto).lower()
    if any(k in s for k in ["govt", "government", "sovereign"]): return "Govt"
    if "retail" in s: return "Corporate Retail"
    if any(k in s for k in ["istituz", "institutional"]): return "Corporate Istituzionali"
    if "corp" in s: return "Corporate"
    return "Unknown"

def _map_sector(s: str) -> str:
    s = str(s).lower()
    if "gov" in s: return "Govt"
    if "fin" in s: return "Financials"
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

def precheck_targets(df: pd.DataFrame, n: int, **targets) -> List[Dict[str, object]]:
    target_map = {
        "Valuta": targets.get("targ_val", {}),
        "TipoEmittente": targets.get("targ_iss", {}),
        "Settore": targets.get("targ_sec", {}),
        "Scadenza": targets.get("targ_mat", {}),
    }
    shortages = []
    for col, weights in target_map.items():
        if not weights: continue
        int_targets = integer_targets_from_weights(n, weights)
        for cat, req in int_targets.items():
            mask = df[col] == str(cat)
            sub = df[mask]
            if sub.empty:
                shortages.append({"dim": col, "category": cat, "required": req, "available": 0, "max_possible": 0})
                continue
            
            is_corp = sub["TipoEmittente"].str.contains("Corp", case=False, na=False)
            gov_count = (~is_corp).sum()
            unique_corp_issuers = sub[is_corp]["Issuer"].nunique()
            max_possible = gov_count + unique_corp_issuers
            if req > max_possible:
                shortages.append({"dim": col, "category": cat, "required": req, "available": len(sub), "max_possible": max_possible})
    return shortages

def _solve_and_get_status(prob, solver):
    """
    Funzione isolata per eseguire il solve e prevenire l'errore 'UnboundLocalError'.
    """
    try:
        prob.solve(solver)
        return prob.status
    except Exception as e:
        raise RuntimeError(f"Il risolutore MILP ha generato un errore critico: {e}")

def build_portfolio_milp(df: pd.DataFrame, n: int, targ_val, targ_iss, targ_sec, targ_mat, weighting_scheme: str) -> pd.DataFrame:
    if pulp is None:
        raise RuntimeError("La libreria PuLP non è installata. Esegui: pip install pulp")

    shortages = precheck_targets(df, n, targ_val=targ_val, targ_iss=targ_iss, targ_sec=targ_sec, targ_mat=targ_mat)
    if shortages:
        details = "; ".join([f"{s['dim']} '{s['category']}': richiesti {s['required']}, max ottenibili {s['max_possible']}" for s in shortages])
        raise ValueError(f"Vincoli impossibili da soddisfare. Dettagli: {details}")

    prob = pulp.LpProblem("bond_selection", pulp.LpMaximize)
    df_copy = df.reset_index(drop=True)
    indices = list(df_copy.index)
    x = {i: pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in indices}
    
    # --- LOGICA DI PONDERAZIONE ---
    scores = df_copy["ScoreRendimento"].fillna(0).to_dict()
    weights = {i: 1.0 for i in indices}  # Default: Equally Weighted

    if weighting_scheme == "Risk Weighted":
        risk_scores = df_copy["ScoreRischio"].fillna(0)
        avg_risk_score = risk_scores.mean()
        if avg_risk_score > 0:
            # Calcola i pesi basati sul rischio, con un limite massimo (cap) di 2
            weights = {i: min(2.0, risk_scores.get(i, 0) / avg_risk_score) for i in indices}

    # Funzione obiettivo modificata per includere i pesi
    prob += pulp.lpSum(scores[i] * weights[i] * x[i] for i in indices)
    # --- FINE LOGICA DI PONDERAZIONE ---
    
    prob += pulp.lpSum(x[i] for i in indices) == n

    targets = {
        "Valuta": targ_val, "TipoEmittente": targ_iss,
        "Settore": targ_sec, "Scadenza": targ_mat
    }
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

st.set_page_config(page_title="Bond Portfolio Selector", layout="wide")
st.title("Bond Portfolio Selector — Ottimizzazione con Vincoli")

uploaded = st.file_uploader("Carica il CSV dei titoli", type=["csv"])

if uploaded:
    try:
        df = load_and_normalize(uploaded)
        st.success(f"File caricato e processato: {len(df)} titoli validi.")
        
        st.sidebar.header("Parametri Portafoglio")
        n = st.sidebar.number_input("Numero di titoli (n)", min_value=1, max_value=len(df), value=min(20, len(df)))

        # --- NUOVO SELETTORE PER LA PONDERAZIONE ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Schema di Ponderazione Obiettivo")
        weighting_scheme = st.sidebar.radio(
            "Scegli come ponderare i titoli nella selezione:",
            ("Equally Weighted", "Risk Weighted"),
            key="weighting_scheme",
            help="**Equally Weighted**: Massimizza solo lo ScoreRendimento. **Risk Weighted**: Massimizza una combinazione di ScoreRendimento e ScoreRischio (favorendo i titoli con ScoreRischio più alto)."
        )
        # --- FINE NUOVO SELETTORE ---

        st.sidebar.markdown("---")
        st.sidebar.subheader("Vincoli di costruzione")
        use_val = st.sidebar.checkbox("Vincola per Valuta")
        use_iss = st.sidebar.checkbox("Vincola per Tipo Emittente")
        use_sec = st.sidebar.checkbox("Vincola per Settore")
        use_mat = st.sidebar.checkbox("Vincola per Scadenza")

        universe = df[df["ScoreRendimento"] >= 20].copy()
        st.info(f"Universo investibile (Score Rendimento >= 20): {len(universe)} titoli.")

        w_val, w_iss, w_sec, w_mat = {}, {}, {}, {}
        
        def get_weights_from_user(title, options, key_prefix):
            st.subheader(title)
            weights = {opt: st.number_input(f"{opt} (%)", 0.0, 100.0, 0.0, key=f"{key_prefix}_{opt}") for opt in options}
            return weights

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

        is_valid = True
        is_valid_val, w_val = validate_weights(w_val, "Valuta")
        is_valid_iss, w_iss = validate_weights(w_iss, "Tipo Emittente")
        is_valid_sec, w_sec = validate_weights(w_sec, "Settore")
        is_valid_mat, w_mat = validate_weights(w_mat, "Scadenza")
        is_valid = all([is_valid_val, is_valid_iss, is_valid_sec, is_valid_mat])

        if st.button("Costruisci Portafoglio"):
            if not is_valid:
                st.stop()
            
            with st.spinner("Ottimizzazione in corso..."):
                try:
                    has_constraints = any([w_val, w_iss, w_sec, w_mat])
                    if not has_constraints and weighting_scheme == "Equally Weighted":
                        st.info("Nessun vincolo specificato e schema Equally Weighted. Seleziono i migliori N titoli per Score Rendimento.")
                        portfolio = universe.nlargest(n, "ScoreRendimento").reset_index(drop=True)
                    else:
                        portfolio = build_portfolio_milp(universe, n, w_val, w_iss, w_sec, w_mat, weighting_scheme)

                    st.success("Portafoglio generato con successo!")
                    st.dataframe(portfolio)
                    
                    csv = portfolio.to_csv(index=False).encode('utf-8')
                    st.download_button("Scarica Portafoglio (CSV)", data=csv, file_name="portafoglio_ottimizzato.csv")
                    
                    # --- SEZIONE GRAFICI E CONFRONTO ---
                    if has_constraints:
                        st.header("Confronto Target vs Effettivo")
                        
                        def compute_dist(df_port, col):
                            return (df_port[col].value_counts(normalize=True) * 100).round(1)

                        distr = {
                            "Valuta": (w_val, compute_dist(portfolio, "Valuta")),
                            "TipoEmittente": (w_iss, compute_dist(portfolio, "TipoEmittente")),
                            "Settore": (w_sec, compute_dist(portfolio, "Settore")),
                            "Scadenza": (w_mat, compute_dist(portfolio, "Scadenza")),
                        }

                        for crit, (target, actual) in distr.items():
                            if not target: continue
                            st.subheader(crit)
                            cats = sorted(set(list(target.keys())) | set(actual.index.astype(str)))
                            rows = []
                            for c in cats:
                                tgt_val = target.get(c, 0.0)
                                act_val = float(actual.get(c, 0.0))
                                rows.append({crit: c, "Target %": f"{tgt_val:.1f}", "Effettivo %": f"{act_val:.1f}"})
                            df_table = pd.DataFrame(rows)
                            st.dataframe(df_table)

                            # Grafico a barre
                            fig, ax = plt.subplots()
                            x = np.arange(len(cats))
                            width = 0.35
                            ax.bar(x - width/2, [float(r["Effettivo %"]) for r in rows], width, label="Effettivo")
                            ax.bar(x + width/2, [float(r["Target %"]) for r in rows], width, label="Target")
                            ax.set_ylabel("Percentuale (%)")
                            ax.set_title(f'Confronto per {crit}')
                            ax.set_xticks(x)
                            ax.set_xticklabels(cats, rotation=45, ha="right")
                            ax.legend()
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    st.balloons()

                except (ValueError, RuntimeError) as e:
                    st.error(f"Impossibile costruire il portafoglio: {e}")
                except Exception as e:
                    st.error(f"Si è verificato un errore inatteso: {e}")

    except Exception as e:
        st.error(f"Errore nel caricamento del file: {e}")
