import os
import warnings
from datetime import date

import pandas as pd
import streamlit as st

import charts
from analysis import detect_crises, episode_list
from config import (CONVENTIONAL_PRESETS, CREDIT_FILE, CRISIS_STD, CUSTOM, DEFAULT_SAMPLE_START, EM_ISLAMIC, GPR_FILE,
                    ISLAMIC_PRESETS, MERGE_GAP, MIN_DAYS, RF_PRESETS, STLFSI_FILE)
from data import build_dataset, build_stress
from metrics import annualise, is_sig, risk_table, sig_phrase, stars
from models import capm, crisis_capm

warnings.filterwarnings("ignore")

CRISIS_RULE = (f"Crisis day = the **previous day's** value above its **average + {CRISIS_STD:g} standard "
               f"deviation** over the sample. Episodes shorter than {MIN_DAYS} days are dropped; episodes "
               f"separated by {MERGE_GAP} calm days or fewer are merged.")


def method(what, source):
    return f"\n**How the {what} index is built:**\n1. Source: {source}\n2. {CRISIS_RULE}\n"


VIX_METHOD = method("market stress", "**CBOE Volatility Index (^VIX)**: the market's standard measure of general "
                    "stress.")
STLFSI_METHOD = method("banking stress", "**St. Louis Fed Financial Stress Index** (FRED STLFSI4). It combines 18 "
                       "weekly financial market series (interest rates, yield spreads and other indicators) into one "
                       "number. **0 = normal financial-market stress**; above 0 = above-average stress.")
OVX_METHOD = method("energy stress", "**CBOE Crude Oil Volatility Index (^OVX)**: expected 30-day volatility of crude "
                    "oil, from options on the United States Oil Fund.")
VXN_METHOD = method("technology stress", "**CBOE Nasdaq-100 Volatility Index (^VXN)**: expected 30-day volatility of "
                    "the technology-heavy Nasdaq-100.")
GPR_METHOD = method("geopolitical stress", "**Geopolitical Risk index** (Caldara & Iacoviello), daily, 7-day average.")
CREDIT_METHOD = method("credit stress", "**Moody's Baa corporate bond yield minus the 10-year Treasury yield** (FRED "
                       "BAA10Y), in percentage points. It rises when investors fear defaults by indebted companies.")
RATES_METHOD = method("interest-rate stress", "**ICE BofA MOVE Index (^MOVE)**: expected volatility of US Treasury "
                      "yields, the 'VIX of bonds'.")
EM_METHOD = method("emerging-market stress", "**CBOE Emerging Markets ETF Volatility Index (^VXEEM)**: expected "
                   "volatility of the EEM emerging-markets ETF. Run for the emerging-market pair only.")

SECTIONS = [
    ("Market", "2. Market stress (VIX)", VIX_METHOD, "VIX", True),
    ("Banking", "3. Banking stress", STLFSI_METHOD, "St. Louis Fed Financial Stress Index", True),
    ("Energy", "4. Energy stress (OVX)", OVX_METHOD, "Crude Oil Volatility Index (OVX)", True),
    ("Technology", "5. Technology stress (VXN)", VXN_METHOD, "Nasdaq-100 Volatility Index (VXN)", True),
    ("Geopolitical", "6. Geopolitical stress", GPR_METHOD, "Geopolitical Risk index", True),
    ("Credit", "7. Credit stress (Baa spread)", CREDIT_METHOD, "Baa minus 10Y Treasury spread (pp)", True),
    ("Rates", "8. Interest-rate stress (MOVE)", RATES_METHOD, "MOVE index", True),
    ("EM", "9. Emerging-market stress (VXEEM)", EM_METHOD, "EM ETF Volatility Index (VXEEM)", True),
]


def _pick(label, presets, default_custom):
    choice = presets[st.sidebar.selectbox(label, list(presets))]
    if choice == CUSTOM:
        choice = st.sidebar.text_input(f"{label}: ticker", default_custom).strip()
    return choice


def sidebar() -> dict:
    st.sidebar.header("1. Assets")
    isl = _pick("Islamic index / ETF", ISLAMIC_PRESETS, "SPUS")
    conv = _pick("Conventional index / ETF", CONVENTIONAL_PRESETS, "SPY")

    st.sidebar.header("2. Risk-free rate")
    rf = RF_PRESETS[st.sidebar.selectbox("Risk-free rate", list(RF_PRESETS))]

    st.sidebar.header("3. Sample window")
    start = st.sidebar.date_input("Start", date(*DEFAULT_SAMPLE_START), min_value=date(1990, 1, 1),
                                  format="DD/MM/YYYY")
    end = st.sidebar.date_input("End", date.today(), format="DD/MM/YYYY")

    folder = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(folder, GPR_FILE)
    gpr = local if os.path.exists(local) else st.sidebar.file_uploader(
        "Daily GPR file (CSV: date, GPRD, GPRD_MA7)", type=["csv"])
    fred = {"Banking": os.path.join(folder, STLFSI_FILE),
            "Credit": os.path.join(folder, CREDIT_FILE)}

    if st.sidebar.button("Run", type="primary", width="stretch"):
        st.session_state["run"] = True
    return {"islamic": isl, "conv": conv, "rf": rf, "start": start, "end": end, "gpr": gpr, "fred": fred}


def _pfmt(p) -> str:
    if p is None or pd.isna(p):
        return ""
    return " (p < 0.001)" if p < 0.001 else f" (p = {p:.3f})"


def _pcell(value, p, fmt):
    """Value + stars + p-value, e.g. '0.859*** (p < 0.001)'."""
    return f"{value:{fmt}}{stars(p)}{_pfmt(p)}"


def results_table(base, cr, n_full) -> pd.DataFrame:
    rows = [{"Window": "Full sample", "Days": n_full,
             "Alpha (ann. %)": _pcell(annualise(base["alpha"]), base["alpha_p"], "+.2f"),
             "Beta": f"{base['beta']:.3f}"}]
    if cr.get("error"):
        rows.append({"Window": "Crisis days", "Days": cr["n_crisis"], "Alpha (ann. %)": cr["error"], "Beta": "—"})
    else:
        rows.append({"Window": "Crisis days", "Days": cr["n_crisis"],
                     "Alpha (ann. %)": _pcell(annualise(cr["alpha_crisis"]), cr["alpha_crisis_p"], "+.2f"),
                     "Beta": _pcell(cr["beta_crisis"], cr["beta_change_p"], ".3f")})
    return pd.DataFrame(rows).set_index("Window")


def answer(name, cr, isl, conv) -> tuple:
    """(short verdict, explanation). Better = significant positive crisis alpha or significantly lower crisis beta."""
    if cr.get("error"):
        return "—", f"Not enough crisis days to test {name.lower()} stress."
    ac, pa = annualise(cr["alpha_crisis"]), cr["alpha_crisis_p"]
    bc, bl, pb = cr["beta_crisis"], cr["beta_calm"], cr["beta_change_p"]
    earns_more, earns_less = ac > 0 and is_sig(pa), ac < 0 and is_sig(pa)
    falls_less, falls_more = bc < bl and is_sig(pb), bc > bl and is_sig(pb)

    parts = [f"During {name.lower()} crises, {isl}'s alpha is **{ac:+.2f}% a year** ({sig_phrase(pa)})",
             f"and its beta is **{bc:.3f}** versus {bl:.3f} on calm days ({sig_phrase(pb)})."]

    better_how = [w for w, ok in (("higher alpha", earns_more), ("lower risk", falls_less)) if ok]
    worse_how = [w for w, ok in (("lower alpha", earns_less), ("higher risk", falls_more)) if ok]

    if better_how and worse_how:
        verdict = f"Mixed ({', '.join(better_how)}; {', '.join(worse_how)})"
        concl = (f"**Mixed:** under {name.lower()} stress, {isl} shows {' and '.join(better_how)} "
                 f"but also {' and '.join(worse_how)} than {conv}.")
    elif better_how:
        verdict = f"✅ Islamic performs better during crisis ({' and '.join(better_how)})"
        concl = (f"**Islamic performs better during {name.lower()} crises:** {isl} shows "
                 f"{' and '.join(better_how)} compared with {conv}.")
    elif worse_how:
        verdict = f"❌ Islamic performs worse during crisis ({' and '.join(worse_how)})"
        concl = (f"**Islamic performs worse during {name.lower()} crises:** {isl} shows "
                 f"{' and '.join(worse_how)} compared with {conv}.")
    else:
        verdict = "No significant difference"
        concl = f"**No significant difference:** under {name.lower()} stress, {isl} behaves like {conv}."
    return verdict, " ".join(parts) + " " + concl


def stress_section(name, title, method_text, label, own_axis, stress, parts, combined, y, x, base, isl, conv):
    st.header(title)
    s = stress[name]
    st.markdown(method_text)

    v = s.dropna()
    threshold = float(v.mean() + CRISIS_STD * v.std(ddof=1))
    d, episodes = detect_crises(s, threshold)
    d = d.where(s.shift(1).notna())   # days before the stress series starts are excluded, not counted as calm
    st.markdown(f"**{label}:** available from {v.index[0]:%d/%m/%Y}, average **{v.mean():.2f}**, standard deviation "
                f"{v.std(ddof=1):.2f}, maximum {v.max():.2f}. **Crisis threshold = {v.mean():.2f} + "
                f"{v.std(ddof=1):.2f} = {threshold:.2f}.**")

    if name == "Geopolitical":
        st.plotly_chart(charts.gpr_chart(parts[name], threshold), width="stretch")
    else:
        st.plotly_chart(charts.single_index_chart(s, threshold, label, label, zero_line=(name == "Banking")),
                        width="stretch")
    st.plotly_chart(charts.stress_vs_funds_chart(s, combined, isl, conv, label, threshold, episodes, own_axis),
                    width="stretch")

    st.subheader(f"{name} crises detected by the data")
    st.markdown(f"Crisis day = previous day's {label} above **{threshold:.2f}**. "
                f"**{len(episodes)} episodes**, {int(d.sum())} crisis days.")
    ep = episode_list(combined, s, episodes)
    if not ep.empty:
        st.dataframe(ep.set_index("Episode").style.format(
            {"Peak stress": "{:.2f}", "Islamic return (%)": "{:+.2f}", "Conventional return (%)": "{:+.2f}",
             "Difference (pp)": "{:+.2f}"}), width="stretch")

    st.subheader(f"Answer: does the Islamic index do better under {name.lower()} stress?")
    cr = crisis_capm(y, x, d)
    st.dataframe(results_table(base, cr, base["n_obs"]), width="stretch")
    calm = f" Calm-day beta = {cr['beta_calm']:.3f}." if not cr.get("error") else ""
    st.caption("CAPM, OLS with Newey-West errors. *** p<0.01, ** p<0.05, * p<0.10. "
               "Alpha p-values test whether alpha differs from 0. "
               "The crisis beta p-value tests whether crisis beta differs from the calm-day beta." + calm)
    verdict, text = answer(name, cr, isl, conv)
    st.markdown(text)
    row = {"stress": name, "measure": label, "threshold": threshold, "episodes": len(episodes),
           "crisis_days": int(d.sum()),
           "alpha_calm_ann_pct": annualise(cr.get("alpha_calm")),
           "alpha_crisis_ann_pct": annualise(cr.get("alpha_crisis")), "alpha_crisis_p": cr.get("alpha_crisis_p"),
           "alpha_change_p": cr.get("alpha_change_p"),
           "beta_calm": cr.get("beta_calm"), "beta_crisis": cr.get("beta_crisis"),
           "beta_change_p": cr.get("beta_change_p"), "answer": verdict}
    return row, d


def main():
    st.set_page_config(page_title="Islamic vs Conventional under stress", layout="wide")
    st.title("Does the Islamic index do better under stress?")
    cfg = sidebar()
    if not st.session_state.get("run"):
        st.info("Choose the assets, risk-free rate and sample in the sidebar, then click **Run**.")
        return
    if cfg["start"] >= cfg["end"]:
        st.error("Start must be before end.")
        return

    isl, conv = cfg["islamic"], cfg["conv"]
    with st.spinner("Downloading data..."):
        combined, rf, err = build_dataset(isl, conv, cfg["rf"], cfg["start"], cfg["end"])
        if err:
            st.error(err)
            return
        stress, parts, notes = build_stress(combined.index, cfg["start"], cfg["end"], cfg["gpr"], cfg["fred"])
    first = combined.index[0].date()
    if (first - cfg["start"]).days > 10:
        st.info(f"Requested start {cfg['start']:%d/%m/%Y}, but the sample effectively starts on "
                f"**{first:%d/%m/%Y}**: that is the first date where both {isl} and {conv} have prices.")
    for n in notes:
        st.warning(n)

    y, x = combined["islamic"] - rf, combined["conventional"] - rf
    base = capm(y, x)

    # 1. The two funds
    st.header("1. The two funds")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.daily_returns_chart(combined, isl, conv), width="stretch")
    c2.plotly_chart(charts.volatility_chart(combined, isl, conv), width="stretch")
    st.subheader("Risk-adjusted metrics (full sample)")
    st.dataframe(risk_table(combined["islamic"], combined["conventional"], rf, isl, conv).style.format("{:.3f}"),
                 width="stretch")

    # 2-9. Stress sections
    summary, dummies = [], {}
    for name, title, method_text, label, own_axis in SECTIONS:
        if name not in stress.columns or stress[name].dropna().empty:
            continue
        if name == "EM" and isl not in EM_ISLAMIC:
            continue
        row, d = stress_section(name, title, method_text, label, own_axis, stress, parts, combined, y, x, base,
                                isl, conv)
        summary.append(row)
        dummies[name] = d

    # Summary
    if summary:
        st.header("Summary")
        st.dataframe(pd.DataFrame(summary).set_index("stress")[["measure", "threshold", "episodes", "crisis_days",
                                                                 "answer"]]
                     .rename(columns={"measure": "Crisis measure", "threshold": "Threshold", "episodes": "Episodes",
                                      "crisis_days": "Crisis days", "answer": "Does the Islamic index do better?"})
                     .style.format({"Threshold": "{:.2f}"}), width="stretch")

    # Download
    st.header("Download")
    daily = pd.DataFrame(index=combined.index)
    daily.index.name = "date"
    daily[f"{isl}_log_return"] = combined["islamic"]
    daily[f"{conv}_log_return"] = combined["conventional"]
    daily["rf_daily"] = rf
    daily = daily.join(stress.add_suffix("_stress"))
    for n, d in dummies.items():
        daily[f"{n}_crisis"] = d.astype("Int64")
    c1, c2 = st.columns(2)
    c1.download_button("Daily data (CSV)", daily.to_csv().encode("utf-8"),
                       file_name=f"{isl}_vs_{conv}_daily.csv", mime="text/csv", width="stretch")
    c2.download_button("Results (CSV)", pd.DataFrame(summary).to_csv(index=False).encode("utf-8"),
                       file_name=f"{isl}_vs_{conv}_results.csv", mime="text/csv", width="stretch")


if __name__ == "__main__":
    main()
