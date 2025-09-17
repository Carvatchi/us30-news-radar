import os, requests, pandas as pd, streamlit as st, plotly.express as px
from datetime import datetime, timedelta, timezone
from math import exp

# =========================
#   CONFIG & PAGE
# =========================
st.set_page_config(page_title="US30 / DIA News Radar", layout="wide")
st.title("📈 US30 / DIA News Radar – Dashboard")

# =========================
#   API KEYS (hardcoded + UI)
# =========================
# NewsAPI (hardcoded – nu mai cere în UI)
NEWSAPI_KEY = "a3b28961a34841c98c8f2b95643ee3c1"

with st.sidebar:
    st.header("🔐 Chei & Provider Calendar")
    cal_provider = st.radio("Provider calendar", ["TradingEconomics", "FinancialModelingPrep"], index=0)
    # TradingEconomics – default guest:guest (demo)
    TE_KEY = st.text_input("TradingEconomics Key", value="guest:guest", type="password")
    # FMP – poți pune cheia ta sau 'demo'
    FMP_KEY = st.text_input("FMP Key (opțional)", value="", type="password", placeholder="ex: demo sau cheia ta")

if not NEWSAPI_KEY:
    st.error("❌ Lipsă NewsAPI Key – verifică codul.")
    st.stop()

# =========================
#   LEXICON + DOW + SCORING V2
# =========================
KEYWORDS = {
    "fed_policy": [
        "FOMC","rate decision","rate hike","rate cut","policy meeting","press conference",
        "dot plot","SEP","Powell","FOMC minutes","blackout period",
        "balance sheet","quantitative tightening","QT","QE","IOER","reverse repo","RRP",
        "financial conditions","Fed funds futures"
    ],
    "inflation_labor": [
        "CPI","Core CPI","PCE","Core PCE","supercore","inflation expectations","disinflation","sticky inflation",
        "Nonfarm payrolls","NFP","unemployment rate","average hourly earnings","AHE",
        "jobless claims","JOLTS","ECI"
    ],
    "growth_activity": [
        "GDP","retail sales","ISM Manufacturing","ISM Services","S&P Global PMI","PMI",
        "durable goods","industrial production","capacity utilization",
        "housing starts","building permits","new home sales","existing home sales",
        "consumer confidence","Michigan sentiment","UMich"
    ],
    "rates_credit_liquidity": [
        "Treasury yields","2-year","10-year","curve steepening","yield curve inversion",
        "credit spreads","high yield","CDX","repo stress","financial conditions index",
        "auction tails","liquidity"
    ],
    "fx_commodities": [
        "DXY","US dollar","WTI","Brent","gasoline","EIA inventories","OPEC","gold","copper","natural gas"
    ],
    "dow_micro": [
        "earnings","guidance","preannouncement","downgrade","upgrade","dividend","buyback",
        "stock split","antitrust","recall","accident","strike","production halt","production restart","investigation",
        "FAA","FDA","NHTSA"
    ],
    "crisis_recession": [
        "recession","hard landing","soft landing",
        "bank failure","bankruptcy","default","sovereign default","liquidity crisis","credit crunch",
        "sovereign downgrade","IMF bailout","capital controls"
    ],
    "global_geopolitics": [
        "war","conflict","escalation","ceasefire","missile","strike","troop buildup","border clashes",
        "sanctions","tariffs","trade deal","summit","peace talks","OPEC+","production cut",
        "pipeline attack","shipping disruption","Suez Canal","Red Sea","Strait of Hormuz",
        "ECB decision","BOE decision","BOJ intervention","PBOC stimulus","currency devaluation"
    ],
}
NEGATIVE_FILTERS = [
    "Fed Cup","animal feed","powell river","fashion summit","trade show","peace concert",
    "movie","series premiere","concert","festival","coach","tournament","game week","celebrity",
    "video game","console war","smartphone leak"
]
DOW_COMPONENTS = {
    "GS":"Goldman Sachs","MSFT":"Microsoft","CAT":"Caterpillar","HD":"Home Depot","SHW":"Sherwin-Williams",
    "V":"Visa","UNH":"UnitedHealth","AXP":"American Express","JPM":"JPMorgan","MCD":"McDonald's",
    "AMGN":"Amgen","TRV":"Travelers","IBM":"IBM","CRM":"Salesforce","AAPL":"Apple","AMZN":"Amazon",
    "BA":"Boeing","HON":"Honeywell","JNJ":"Johnson & Johnson","NVDA":"Nvidia","CVX":"Chevron",
    "PG":"Procter & Gamble","MMM":"3M","DIS":"Disney","WMT":"Walmart","MRK":"Merck",
    "NKE":"Nike","CSCO":"Cisco","KO":"Coca-Cola","VZ":"Verizon"
}
DOW_NAMES = set([*DOW_COMPONENTS.values(), *DOW_COMPONENTS.keys()])
CATEGORY_WEIGHTS = {
    "fed_policy": 2.6, "inflation_labor": 2.2, "growth_activity": 1.6, "rates_credit_liquidity": 1.7,
    "fx_commodities": 1.5, "dow_micro": 1.5, "crisis_recession": 3.0, "global_geopolitics": 2.3,
}
SOURCE_WEIGHTS = {
    "reuters": 1.00, "bloomberg": 1.00, "wsj": 0.95, "financial times": 0.95,
    "cnbc": 0.90, "ap news": 0.90, "associated press": 0.90,
    "investing.com": 0.85, "marketwatch": 0.85, "yahoo": 0.80,
}

def normalize_source(name: str) -> float:
    if not name: return 0.85
    n = name.lower()
    for key, w in SOURCE_WEIGHTS.items():
        if key in n: return w
    return 0.85

def contains_any(text: str, terms) -> bool:
    if not text: return False
    t = text.lower()
    return any(term.lower() in t for term in terms)

def categorize_strict(title: str, desc: str):
    blob = f"{title or ''} {desc or ''}"
    if contains_any(blob, NEGATIVE_FILTERS):
        return []
    cats = [cat for cat, words in KEYWORDS.items() if contains_any(blob, words)]
    if contains_any(blob, DOW_NAMES) and "dow_micro" not in cats:
        cats.append("dow_micro")
    return cats

BULLISH_TERMS = {
    "rate cut","cooling inflation","disinflation","soft landing",
    "beat expectations","earnings beat","upgrade","buyback","stimulus",
    "ceasefire","peace deal","tariffs lifted","sanctions lifted"
}
BEARISH_TERMS = {
    "rate hike","hot inflation","sticky inflation","recession","hard landing",
    "miss expectations","earnings miss","downgrade",
    "escalation","war","conflict","sanctions","tariffs","default",
    "bank failure","liquidity crisis","credit crunch","shutdown"
}
def direction_sign(text: str) -> int:
    t = (text or "").lower()
    pos = any(p in t for p in BULLISH_TERMS)
    neg = any(n in t for n in BEARISH_TERMS)
    if pos and not neg: return +1
    if neg and not pos: return -1
    return 0

def intensity(text: str) -> float:
    t = (text or "").lower()
    score = 1.0
    for strong in ["surge","plunge","soar","collapse","shock","crash","soaring","spiking"]:
        if strong in t: score += 0.3
    for mild in ["edges","slight","modest","muted"]:
        if mild in t: score -= 0.1
    return max(0.7, min(1.6, score))

def recency_decay(pub_dt, now_dt, tau_hours=12.0) -> float:
    try:
        dt_hours = max(0.0, (now_dt - pub_dt).total_seconds()/3600.0)
    except Exception:
        return 1.0
    return exp(-dt_hours / tau_hours)

def score_article_v2(article, now_dt):
    title = article.get("title") or ""
    desc  = article.get("description") or ""
    source = (article.get("source") or {}).get("name","")
    cats = categorize_strict(title, desc)
    if not cats: return None
    text = f"{title} {desc}"
    sign = direction_sign(text)
    inten = intensity(text)
    w_cat = sum(CATEGORY_WEIGHTS.get(c,1.0) for c in cats)/len(cats)
    w_src = normalize_source(source)
    pub = pd.to_datetime(article.get("publishedAt"), errors="coerce", utc=True)
    decay = recency_decay(pub.to_pydatetime() if pub is not None else now_dt, now_dt)
    raw = sign * inten * w_cat * w_src * decay
    HEAVY = {"GS","CAT","HD","SHW"}
    if contains_any(text, HEAVY):
        raw *= 1.15
    return {"_score": round(raw, 4), "_cats": cats, "_source_w": w_src}

# =========================
#   SIDEBAR – FILTRE & ALERTS
# =========================
with st.sidebar:
    st.header("⚙️ Setări & Filtre")
    period = st.selectbox("Perioadă", ["1 zi","3 zile","7 zile"], index=0)
    days = {"1 zi":1,"3 zile":3,"7 zile":7}[period]
    pages = st.slider("Pagini știri (x100/articole)", 1, 5, 3)
    only_hi_impact = st.checkbox("Doar Macro / Fed / Crize (impact mare)")
    only_dow = st.checkbox("Doar companii Dow (DIA/US30)")
    cat_filter = st.multiselect("Categorii incluse", list(KEYWORDS.keys()), default=list(KEYWORDS.keys()))
    src_whitelist = st.text_input("Surse permise (virgule, opțional)", placeholder="ex: reuters,bloomberg,wsj")
    st.subheader("🔔 Alerte (bias)")
    bear_thr = st.slider("Prag Bearish", -1.0, 0.0, -0.8, 0.05)
    bull_thr = st.slider("Prag Bullish", 0.0, 1.0, 0.8, 0.05)
    refresh = st.button("🔄 Refresh")

if only_hi_impact:
    cat_filter = [c for c in cat_filter if c in ["fed_policy","inflation_labor","crisis_recession","global_geopolitics"]]

def build_query(only_dow: bool) -> str:
    base = '("United States" OR US OR USA OR "Federal Reserve" OR "Dow Jones")'
    if only_dow:
        comps = " OR ".join(f'"{name}"' for name in DOW_COMPONENTS.values())
        return f"({base}) AND ({comps})"
    return base

def fetch_news(fr_iso: str, to_iso: str, pages: int, sources: str|None):
    url = "https://newsapi.org/v2/everything"
    all_rows = []
    headers = {"Authorization": NEWSAPI_KEY}
    for page in range(1, pages+1):
        params = {
            "q": build_query(only_dow),
            "language": "en", "from": fr_iso, "to": to_iso,
            "pageSize": 100, "page": page, "sortBy":"publishedAt"
        }
        if sources:
            params["sources"] = sources
        r = requests.get(url, headers=headers, params=params, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"NewsAPI error {r.status_code}: {r.text[:200]}")
        arts = r.json().get("articles", [])
        if not arts: break
        for a in arts:
            all_rows.append({
                "publishedAt": a.get("publishedAt"),
                "source": (a.get("source") or {}).get("name"),
                "title": a.get("title"),
                "description": a.get("description"),
                "url": a.get("url")
            })
        if len(arts) < 100: break
    df = pd.DataFrame(all_rows)
    if df.empty: return df
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce")
    return df

@st.cache_data(ttl=600, show_spinner=False)
def load_news(days:int, pages:int, sources:str|None, active_cats:list):
    now = datetime.now(timezone.utc)
    fr = now - timedelta(days=days)
    df = fetch_news(fr.isoformat(), now.isoformat(), pages, sources)
    if df.empty: return df
    rows = []
    for _, a in df.iterrows():
        art = {"publishedAt": a["publishedAt"], "source": {"name": a["source"]}, "title": a["title"], "description": a["description"], "url": a["url"]}
        scored = score_article_v2(art, now)
        if scored is None: 
            continue
        if not any(c in active_cats for c in scored["_cats"]):
            continue
        art.update(scored)
        rows.append(art)
    if not rows: 
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["bias"] = out["_score"].apply(lambda s: "Bullish" if s>0.1 else ("Bearish" if s<-0.1 else "Mixed"))
    return out

def impact_arrow(delta):
    if delta is None: return "≈"
    if delta > 0.1:  return "⬆️"
    if delta < -0.1: return "⬇️"
    return "≈"

@st.cache_data(ttl=900, show_spinner=False)
def load_te_calendar(start_date: datetime, end_date: datetime, key: str):
    if not key: return pd.DataFrame()
    base = "https://api.tradingeconomics.com/calendar"
    d1, d2 = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    params = {"country":"united states", "d1": d1, "d2": d2, "importance":"2,3", "c": key}
    r = requests.get(base, params=params, timeout=20)
    if r.status_code != 200: return pd.DataFrame()
    data = r.json() or []
    rows = []
    for it in data:
        rows.append({
            "datetime": it.get("Date"),
            "event": it.get("Event"),
            "actual": it.get("Actual"),
            "forecast": it.get("Forecast"),
            "previous": it.get("Previous"),
            "importance": it.get("Importance"),
            "country": it.get("Country"),
            "category": it.get("Category")
        })
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    def to_num(x):
        try:
            if x is None: return None
            return float(str(x).replace("%","").replace(",","").strip())
        except: return None
    df["actual_n"] = df["actual"].apply(to_num)
    df["forecast_n"] = df["forecast"].apply(to_num)
    df["delta"] = df.apply(lambda r: None if r["actual_n"] is None or r["forecast_n"] is None else (r["actual_n"] - r["forecast_n"]), axis=1)
    df["arrow"] = df["delta"].apply(impact_arrow)
    imp_map = {"High":"🔴 High","Medium":"🟠 Medium","Low":"🟡 Low"}
    df["imp_lbl"] = df["importance"].map(imp_map).fillna(df["importance"])
    return df.sort_values("datetime")

@st.cache_data(ttl=900, show_spinner=False)
def load_fmp_calendar(start_date: datetime, end_date: datetime, api_key: str):
    if not api_key: return pd.DataFrame()
    base = "https://financialmodelingprep.com/api/v3/economic_calendar"
    d1, d2 = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    params = {"from": d1, "to": d2, "apikey": api_key}
    try:
        r = requests.get(base, params=params, timeout=20)
        if r.status_code != 200: return pd.DataFrame()
        data = r.json() or []
    except Exception:
        return pd.DataFrame()
    rows = []
    for it in data:
        rows.append({
            "datetime": pd.to_datetime(it.get("date"), errors="coerce"),
            "event": it.get("event"),
            "actual": it.get("actual"),
            "forecast": it.get("previous"),
            "previous": it.get("previous"),
            "importance": it.get("impact") or "",
            "country": it.get("country") or "United States",
            "category": it.get("category") or "",
        })
    df = pd.DataFrame(rows)
    if df.empty: return df
    def to_num(x):
        try:
            if x is None: return None
            return float(str(x).replace("%","").replace(",","").strip())
        except: return None
    df["actual_n"] = df["actual"].apply(to_num)
    df["forecast_n"] = df["forecast"].apply(to_num)
    df["delta"] = df.apply(lambda r: None if r["actual_n"] is None or r["forecast_n"] is None else (r["actual_n"] - r["forecast_n"]), axis=1)
    df["arrow"] = df["delta"].apply(impact_arrow)
    imp_map = {"High":"🔴 High","Medium":"🟠 Medium","Low":"🟡 Low"}
    df["imp_lbl"] = df["importance"].map(imp_map).fillna(df["importance"])
    return df.sort_values("datetime")

# =========================
#   REFRESH CACHE
# =========================
if refresh:
    st.cache_data.clear()

# =========================
#   LOAD NEWS
# =========================
sources = src_whitelist.strip().replace(" ", "") or None
active_cats = cat_filter
try:
    with st.spinner("Încarc știrile…"):
        news_df = load_news(days, pages, sources, active_cats)
except Exception as e:
    st.error(f"Nu am putut încărca știrile: {e}")
    st.stop()

if news_df is None or news_df.empty:
    st.warning("Nu am găsit știri relevante pe filtrul curent.")
    news_df = pd.DataFrame(columns=["publishedAt","_score","bias","title","source","url","_cats"])

# =========================
#   KPI FROM NEWS
# =========================
mean_score = news_df["_score"].mean() if not news_df.empty else 0.0
bias = "Bullish" if mean_score>0.1 else ("Bearish" if mean_score<-0.1 else "Mixed")
confidence = int(min(100, abs(mean_score)*35 + min(60, (len(news_df)/5) if not news_df.empty else 0)))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Știri relevante", len(news_df))
c2.metric("Bias", bias, delta=f"{mean_score:.2f}")
c3.metric("Confidence", f"{confidence}/100")

today = datetime.now(timezone.utc)
start_day = today.replace(hour=0,minute=0,second=0,microsecond=0)
end_day   = today.replace(hour=23,minute=59,second=59,microsecond=0)

if cal_provider == "TradingEconomics":
    cal_today = load_te_calendar(start_day, end_day, TE_KEY)
else:
    cal_today = load_fmp_calendar(start_day, end_day, FMP_KEY or "demo")

c4.metric("Evenimente Macro azi (US)", len(cal_today) if not cal_today.empty else 0)

# Alerts
if mean_score <= bear_thr:
    st.error(f"⚠️ Alertă Bearish: bias={mean_score:.2f} ≤ {bear_thr:.2f}")
elif mean_score >= bull_thr:
    st.success(f"✅ Alertă Bullish: bias={mean_score:.2f} ≥ {bull_thr:.2f}")

st.caption("Legendă scor știri: + = tentă bullish • − = tentă bearish • |valoare| mare = impact mai puternic.")

# =========================
#   CHARTS
# =========================
if not news_df.empty:
    tmp = news_df.copy()
    tmp["date"] = pd.to_datetime(tmp["publishedAt"], utc=True).dt.tz_convert(None).dt.date
    by_bias = tmp.groupby("bias").size().reset_index(name="count")
    fig1 = px.bar(by_bias, x="bias", y="count", title="Distribuția bias-ului (Bullish / Bearish / Mixed)")
    st.plotly_chart(fig1, use_container_width=True)

    by_day = tmp.groupby("date")["_score"].mean().reset_index()
    fig2 = px.line(by_day, x="date", y="_score", title="Scor mediu zilnic (știri)")
    st.plotly_chart(fig2, use_container_width=True)

    top_src = tmp.groupby("source").size().sort_values(ascending=False).head(12).reset_index(name="count")
    fig3 = px.bar(top_src, x="source", y="count", title="Top surse (număr articole)")
    st.plotly_chart(fig3, use_container_width=True)

# =========================
#   CALENDAR INTELIGENT (US)
# =========================
st.subheader("🗓️ Calendar inteligent (US)")
dleft, dright = st.columns([1,2])
with dleft:
    cal_days = st.selectbox("Interval calendar", ["Azi", "3 zile", "7 zile"], index=1)
    span = {"Azi":1, "3 zile":3, "7 zile":7}[cal_days]
with dright:
    st.caption("Dacă TE Key este setată, folosim TradingEconomics; altfel poți alege FMP în sidebar. Săgeata ⬆️/⬇️/≈ sugerează impact direcțional estimat (euristic).")

start = today
end   = today + timedelta(days=span)
if cal_provider == "TradingEconomics":
    cal_df = load_te_calendar(start, end, TE_KEY)
else:
    cal_df = load_fmp_calendar(start, end, FMP_KEY or "demo")

if cal_df is None or cal_df.empty:
    st.info("Nu am date de calendar disponibile (cheie lipsă sau limitări API).")
else:
    show = cal_df[["datetime","imp_lbl","event","actual","forecast","previous","arrow","category"]].copy()
    show = show.rename(columns={
        "datetime":"Ora (UTC)","imp_lbl":"Impact","event":"Eveniment",
        "actual":"Actual","forecast":"Forecast","previous":"Previous",
        "arrow":"US30 Impact","category":"Categorie"
    })
    st.dataframe(show, use_container_width=True, height=360)

# =========================
#   NEWS TABLE + EXPORT
# =========================
st.subheader("📰 Știri filtrate")
if not news_df.empty:
    show_news = news_df.sort_values("publishedAt", ascending=False)[["publishedAt","source","title","_score","_cats","url","bias"]]
    show_news = show_news.rename(columns={"_score":"score","_cats":"cats"})
    st.dataframe(show_news, use_container_width=True, height=520)
    st.download_button("⬇️ Descarcă CSV filtrat", data=show_news.to_csv(index=False), file_name="us30_news_filtered.csv", mime="text/csv")
else:
    st.write("—")
