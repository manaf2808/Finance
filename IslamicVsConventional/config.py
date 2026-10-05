"""Settings and asset lists."""

TRADING_DAYS = 252
HAC_LAGS = 5                          # Newey-West lag
DEFAULT_SAMPLE_START = (2000, 8, 28)

# Stress measures
STRESS_WINDOW = 21                    # rolling window (trading days) for fund volatility charts
VIX_TICKER = "^VIX"                   # market stress: CBOE VIX (Yahoo)
ENERGY_TICKER = "^OVX"                # energy stress: CBOE Crude Oil Volatility Index (Yahoo)
TECH_TICKER = "^VXN"                  # technology stress: CBOE Nasdaq-100 Volatility Index (Yahoo)
RATES_TICKER = "^MOVE"                # interest-rate stress: ICE BofA MOVE Index (Yahoo)
EM_TICKER = "^VXEEM"                  # emerging-market stress: CBOE EM ETF Volatility Index (Yahoo)
EM_ISLAMIC = {"ISDE.L"}               # EM stress is only run for these Islamic tickers
STLFSI_ID, STLFSI_FILE = "STLFSI4", "STLFSI4.csv"      # banking stress (FRED, weekly), local file
CREDIT_ID, CREDIT_FILE = "BAA10Y", "BAA10Y.csv"        # credit stress (FRED, daily), local file
GPR_FILE = "data_gpr_export.csv"       # daily Geopolitical Risk index (Caldara & Iacoviello), local file
GPR_COLUMN = "GPRD_MA7"               # 7-day moving average of the daily GPR

CUSTOM = "__custom__"

# The matched thesis assets are listed first.
ISLAMIC_PRESETS = {
    "S&P 500 Sharia ETF (SPUS) — US": "SPUS",
    "iShares MSCI World Islamic UCITS (ISWD.L) — World, London": "ISWD.L",
    "iShares MSCI EM Islamic UCITS (ISDE.L) — Emerging, London": "ISDE.L",
    "iShares MSCI USA Islamic UCITS USD (ISDU.L) — US, London, since 2007": "ISDU.L",
    "Wahed FTSE USA Shariah ETF (HLAL) — US": "HLAL",
    "Wahed Dow Jones Islamic World ETF (UMMA)": "UMMA",
    "Invesco Dow Jones Islamic Global Developed (IGDA)": "IGDA",
    "Custom ticker...": CUSTOM,
}

CONVENTIONAL_PRESETS = {
    "S&P 500 ETF (SPY) — US": "SPY",
    "iShares MSCI World UCITS Dist (IWRD.L) — World, London": "IWRD.L",
    "iShares MSCI EM UCITS Dist (IEEM.L) — Emerging, London": "IEEM.L",
    "iShares Core S&P 500 UCITS USD Dist (IDUS.L) — US, London, since 2002": "IDUS.L",
    "iShares MSCI USA UCITS USD Acc (CSUS.L) — US, London, since 2010": "CSUS.L",
    "S&P 500 Index (^GSPC)": "^GSPC",
    "MSCI World ETF (URTH) — New York": "URTH",
    "MSCI Emerging Markets ETF (EEM) — New York": "EEM",
    "MSCI ACWI ETF (ACWI)": "ACWI",
    "Custom ticker...": CUSTOM,
}

RF_PRESETS = {
    "13-week US T-Bill (^IRX)": "^IRX",
    "None (0% risk-free rate)": None,
}

# Crisis detection (same rule for every stress type)
# Crisis day = previous day's stress above its sample average + CRISIS_STD standard deviations
CRISIS_STD = 1.0
MIN_DAYS = 5                          # minimum crisis episode length (trading days)
MERGE_GAP = 5                         # episodes separated by <= N calm days are merged
