import requests, pandas as pd, streamlit as st, plotly.express as px
from datetime import datetime, timedelta, timezone
from math import exp
import xml.etree.ElementTree as ET
from io import BytesIO

# ======================================
#   PAGE CONFIG
# ======================================
st.set_page_config(page_title="US30 / DIA News Radar", layout="wide")
st.title("📈 US30 / DIA News Radar – Dashboard")

# ======================================
#   API KEYS (hardcoded + Secrets)
# ======================================
NEWSAPI_KEY = "a3b28961a34841c98c8f2b95643ee3c1" or st.secrets.get("NEWSAPI_KEY", "")

with st.sidebar:
    st.header("🔐 Chei & Provider Calendar")
    cal_provider = st.radio("Provider calendar preferat", ["TradingEconomics", "FinancialModelingPrep"], index=0)
    TE_KEY = st.text_input("TradingEconomics Key", value=(st.secrets.get("TE_KEY") or "guest:guest"), type="password")
    FMP_KEY = st.text_input("FMP Key (sau 'demo')", value=(st.secrets.get("FMP_KEY") or "demo"), type="password")

# ======================================
#   LEXICON + DOW + SCORING
# ======================================
KEYWORDS = {
    "fed_policy": ["FOMC","rate decision","rate hike","rate cut","policy meeting","press conference","dot plot","SEP","Powell","FOMC minutes","blackout period","balance sheet","quantitative tightening","QT","QE","IOER","reverse repo","RRP","financial conditions","Fed funds futures"],
    "inflation_labor": ["CPI","Core CPI","PCE","Core PCE","supercore","inflation expectations","disinflation","sticky inflation","Nonfarm payrolls","NFP","unemployment rate","average hourly earnings","AHE","jobless claims","JOLTS","ECI"],
    "growth_activity": ["GDP","retail sales","ISM Manufacturing","ISM Services","S&P Global PMI","PMI","durable goods","industrial production","capacity utilization","housing starts","building permits","new home sales","existing home sales","consumer confidence","Michigan sentiment","UMich"],
    "rates_credit_liquidity": ["Treasury yields","2-year","10-year","curve steepening","yield curve inversion","credit spreads","high yield","CDX","repo stress","financial conditions index","auction tails","liquidity"],
    "fx_commodities": ["DXY","US dollar","WTI","Brent","gasoline","EIA inventories","OPEC","gold","copper","natural gas"],
    "dow_micro": ["earnings","guidance","preannouncement","downgrade","upgrade","dividend","buyback","stock split","antitrust","recall","accident","strike","production halt","production restart","investigation","FAA","FDA","NHTSA"],
    "crisis_recession": ["recession","hard landing","soft landing","bank failure","bankruptcy","default","sovereign default","liquidity crisis","credit crunch","sovereign downgrade","IMF bailout","capital controls"],
    "global_geopolitics": ["war","conflict","escalation","ceasefire","missile","strike","troop buildup","border clashes","sanctions","tariffs","trade deal","summit","peace talks","OPEC+","production cut","pipeline attack","shipping disruption","Suez Canal","Red Sea","Strait of Hormuz","ECB decision","BOE decision","BOJ intervention","PBOC stimulus","currency devaluation"],
}
NEGATIVE_FILTERS = ["Fed Cup","animal feed","powell river","fashion summit","trade show","peace concert","movie","series premiere","concert","festival","coach","tournament","game week","celebrity","video game","console war","smartphone leak"]
DOW_COMPONENTS = {"GS":"Goldman Sachs","MSFT":"Microsoft","CAT":"Caterpillar","HD":"Home Depot","SHW":"Sherwin-Williams","V":"Visa","UNH":"UnitedHealth","AXP":"American Express","JPM":"JPMorgan","MCD":"McDonald's","AMGN":"Amgen","TRV":"Travelers","IBM":"IBM","CRM":"Salesforce","AAPL":"Apple","AMZN":"Amazon","BA":"Boeing","HON":"Honeywell","JNJ":"Johnson & Johnson","NVDA":"Nvidia","CVX":"Chevron","PG":"Procter & Gamble","MMM":"3M","DIS":"Disney","WMT":"Walmart","MRK":"Merck","NKE":"Nike","CSCO":"Cisco","KO":"Coca-Cola","VZ":"Verizon"}
DOW_NAMES = set([*DOW_COMPONENTS.values(), *DOW_COMPONENTS.keys()])
CATEGORY_WEIGHTS = {"fed_policy": 2.6, "inflation_labor": 2.2, "growth_activity": 1.6, "rates_credit_liquidity": 1.7, "fx_commodities": 1.5, "dow_micro": 1.5, "crisis_recession": 3.0, "global_geopolitics": 2.3}
SOURCE_WEIGHTS = {"reuters": 1.00, "bloomberg": 1.00, "wsj": 0.95, "financial times": 0.95, "cnbc": 0.90, "ap news": 0.90, "associated press": 0.90, "investing.com": 0.85, "marketwatch": 0.85, "yahoo": 0.80}

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

BULLISH_TERMS = {"rate cut","cooling inflation","disinflation","soft landing","beat expectations","earnings beat","upgrade","buyback","stimulus","ceasefire","peace deal","tariffs lifted","sanctions lifted"}
BEARISH_TERMS = {"rate hike","hot inflation","sticky inflation","recession","hard landing","miss expectations","earnings miss","downgrade","escalation","war","conflict","sanctions","tariffs","default","bank failure","liquidity crisis","credit crunch","shutdown"}

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
    for s in ["surge","plunge","soar","collapse","shock","crash","soaring","spiking"]:
        if s in t: score += 0.3
    for s in ["edges","slight","modest","muted"]:
        if s in t: score -= 0.1
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
    src = article.get("source")
    source = (src.get("name","") if isinstance(src, dict) else (src or ""))
    cats = categorize_strict(title, desc)
    if not cats:
        return None
    text = f"{title} {desc}"
    sign = direction_sign(text)
    inten = intensity(text)
    w_cat = sum(CATEGORY_WEIGHTS.get(c,1.0) for c in cats)/len(cats)
    w_src = normalize_source(source)
    pub = pd.to_datetime(article.get("publishedAt"), errors="coerce", utc=True)
    decay = recency_decay(pub.to_pydatetime() if pub is not None else now_dt, now_dt)
    raw = sign * inten * w_cat * w_src * decay
    if contains_any(text, {"GS","CAT","HD","SHW"}):
        raw *= 1.15
    return {"_score": round(raw, 4), "_cats": cats, "_source_w": w_src, "_source_name": source}

# ======================================
#   SIDEBAR – FILTRE & ALERTS
# ======================================
with st.sidebar:
    st.header("⚙️ Setări & Filtre")
    period = st.selectbox("Perioadă (știri)", ["1 zi","3 zile","7 zile"], index=0)
    days = {"1 zi":1,"3 zile":3,"7 zile":7}[period]
    pages = st.slider("Pagini știri (x100/articole total)", 1, 5, 3)
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

# ======================================
#   NEWS: batching sub 500 chars + 24h pentru „1 zi”
# ======================================
def build_queries(only_dow_flag: bool) -> list[str]:
    base = '("United States" OR US OR USA OR "Federal Reserve" OR "Dow Jones")'
    if not only_dow_flag:
        return [base]
    names = list(DOW_COMPONENTS.values())
    batches, cur = [], []
    def q(lst): return f'({base}) AND (' + " OR ".join(f'"{n}"' for n in lst) + ')'
    for n in names:
        trial = q(cur + [n])
        if len(trial) > 480:
            if cur: batches.append(q(cur)); cur = [n]
            else:   batches.append(q([n]));  cur = []
        else:
            cur.append(n)
    if cur: batches.append(q(cur))
    return batches or [base]

def fetch_news(fr_dt: datetime, to_dt: datetime|None, pages_total: int, sources_csv: str|None, only_dow_flag: bool):
    if not NEWSAPI_KEY: return pd.DataFrame()
    url = "https://newsapi.org/v2/everything"
    headers = {"Authorization": NEWSAPI_KEY}
    queries = build_queries(only_dow_flag)
    per_query = [pages_total // len(queries)] * len(queries)
    for i in range(pages_total % len(queries)): per_query[i] += 1

    all_rows, seen = [], set()
    for qi, q in enumerate(queries):
        for page in range(1, per_query[qi] + 1):
            params = {"q": q, "language": "en", "from": fr_dt.isoformat(), "pageSize": 100, "page": page, "sortBy": "publishedAt"}
            if to_dt is not None: params["to"] = to_dt.isoformat()
            if sources_csv: params["sources"] = sources_csv
            r = requests.get(url, headers=headers, params=params, timeout=20)
            if r.status_code != 200:
                try: msg = r.json()
                except Exception: msg = r.text
                st.warning(f"NewsAPI {r.status_code} batch {qi+1}/{len(queries)}: {str(msg)[:200]}")
                break
            arts = r.json().get("articles", [])
            if not arts: break
            for a in arts:
                key = a.get("url") or a.get("title")
                if key and key in seen: continue
                seen.add(key)
                all_rows.append({"publishedAt": a.get("publishedAt"), "source": (a.get("source") or {}).get("name"), "title": a.get("title"), "description": a.get("description"), "url": a.get("url")})
            if len(arts) < 100: break
    df = pd.DataFrame(all_rows)
    if df.empty: return df
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce")
    return df

@st.cache_data(ttl=600, show_spinner=False)
def load_news(days_back:int, pages_total:int, sources_csv:str|None, active_cats:list, only_dow_flag: bool):
    now = datetime.now(timezone.utc)
    fr_dt, to_dt = ((now - timedelta(hours=24)), None) if days_back == 1 else (now - timedelta(days=days_back), now)
    df = fetch_news(fr_dt, to_dt, pages_total, sources_csv, only_dow_flag)
    if df.empty: return df
    rows = []
    for _, a in df.iterrows():
        art = {"publishedAt": a["publishedAt"], "source": {"name": a["source"]} if a["source"] else a["source"], "title": a["title"], "description": a["description"], "url": a["url"]}
        scored = score_article_v2(art, now)
        if scored is None or not any(c in active_cats for c in scored["_cats"]): continue
        art.update(scored); rows.append(art)
    if not rows: return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["_source_name"] = out["_source_name"].fillna("").astype(str)
    out["bias"] = out["_score"].apply(lambda s: "Bullish" if s>0.1 else ("Bearish" if s<-0.1 else "Mixed"))
    return out

# ======================================
#   CALENDAR – chunking + multi-fallback
# ======================================
def impact_arrow(delta):
    if delta is None: return "≈"
    if delta > 0.1:  return "⬆️"
    if delta < -0.1: return "⬇️"
    return "≈"

def _normalize_calendar_rows(data):
    rows = []
    for it in data:
        rows.append({
            "datetime": it.get("Date") or it.get("date") or it.get("timestamp") or it.get("datetime"),
            "event": it.get("Event") or it.get("event") or it.get("title"),
            "actual": it.get("Actual") or it.get("actual"),
            "forecast": it.get("Forecast") or it.get("forecast") or it.get("previous"),
            "previous": it.get("Previous") or it.get("previous"),
            "importance": it.get("Importance") or it.get("impact") or it.get("impact_text") or "",
            "country": it.get("Country") or it.get("country"),
            "category": it.get("Category") or it.get("category") or "",
        })
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    def to_num(x):
        try:
            if x is None: return None
            return float(str(x).replace("%","").replace(",","").strip())
        except: return None
    df["actual_n"] = df["actual"].apply(to_num)
    df["forecast_n"] = df["forecast"].apply(to_num)
    df["delta"] = df.apply(lambda r: None if r["actual_n"] is None or r["forecast_n"] is None else (r["actual_n"] - r["forecast_n"]), axis=1)
    df["arrow"] = df["delta"].apply(impact_arrow)
    imp_map = {"High":"🔴 High","Medium":"🟠 Medium","Low":"🟡 Low","High Impact":"🔴 High","Medium Impact":"🟠 Medium","Low Impact":"🟡 Low","": ""}
    df["imp_lbl"] = df["importance"].map(imp_map).fillna(df["importance"])
    for col in ["datetime","imp_lbl","event","actual","forecast","previous","arrow","category","country"]:
        if col not in df.columns: df[col] = ""
    return df.sort_values("datetime")

def _te_fetch(d1, d2, key, extra=None):
    base = "https://api.tradingeconomics.com/calendar"
    p = {"d1": d1, "d2": d2, "importance": "1,2,3", "c": key}
    if extra: p.update(extra)
    try:
        r = requests.get(base, params=p, timeout=20)
        if r.status_code != 200:
            return [], f"TE HTTP {r.status_code}: {r.text[:120]}"
        return (r.json() or []), ""
    except Exception as e:
        return [], f"TE exc: {e}"

def _fmp_fetch(d1, d2, key):
    base = "https://financialmodelingprep.com/api/v3/economic_calendar"
    try:
        r = requests.get(base, params={"from": d1, "to": d2, "apikey": (key or "demo")}, timeout=20)
        if r.status_code != 200:
            return [], f"FMP HTTP {r.status_code}: {r.text[:120]}"
        return (r.json() or []), ""
    except Exception as e:
        return [], f"FMP exc: {e}"

def _date_chunks(start_dt: datetime, end_dt: datetime, step_days=3):
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + timedelta(days=step_days), end_dt)
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt

_FF_BASES = [
    "https://nfs.faireconomy.media",
    "https://cdn-nfs.faireconomy.media",
    "http://nfs.faireconomy.media",
    "http://cdn-nfs.faireconomy.media",
]
def _ff_try_fetch(path: str):
    last_err = ""
    for base in _FF_BASES:
        url = f"{base}{path}"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                return r.content, ""
            last_err = f"FF HTTP {r.status_code} @ {base}"
        except Exception as e:
            last_err = f"FF exc @ {base}: {e}"
    return None, last_err

def _ff_parse(content: bytes):
    rows = []
    try:
        root = ET.parse(BytesIO(content)).getroot()
        for ev in root.findall(".//event"):
            def get(tag):
                el = ev.find(tag)
                return el.text if el is not None else None
            rows.append({
                "timestamp": get("timestamp"),
                "datetime": (get("date") + " " + (get("time") or "00:00")) if get("date") else get("timestamp"),
                "event": get("title"),
                "country": get("country"),
                "importance": get("impact"),
                "actual": get("actual"),
                "forecast": get("forecast"),
                "previous": get("previous"),
                "category": get("folder")
            })
    except Exception:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty: return df
    def _to_dt(row):
        if row.get("timestamp"):
            try: return datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc)
            except Exception: pass
        return pd.to_datetime(row.get("datetime"), errors="coerce", utc=True)
    df["datetime"] = df.apply(_to_dt, axis=1)
    impact_map = {"High Impact Expected":"🔴 High","Medium Impact Expected":"🟠 Medium","Low Impact Expected":"🟡 Low","Non-Economic":""}
    df["imp_lbl"] = df["importance"].map(impact_map).fillna(df["importance"].fillna(""))
    def to_num(x):
        try:
            if x is None: return None
            return float(str(x).replace("%","").replace(",","").strip())
        except: return None
    df["actual_n"] = df["actual"].apply(to_num)
    df["forecast_n"] = df["forecast"].apply(to_num)
    df["delta"] = df.apply(lambda r: None if r["actual_n"] is None or r["forecast_n"] is None else (r["actual_n"] - r["forecast_n"]), axis=1)
    df["arrow"] = df["delta"].apply(impact_arrow)
    for col in ["datetime","imp_lbl","event","actual","forecast","previous","arrow","category","country"]:
        if col not in df.columns: df[col] = ""
    return df.sort_values("datetime")

@st.cache_data(ttl=900, show_spinner=False)
def load_calendar_with_diagnostics(span_days:int, provider:str, te_key:str, fmp_key:str):
    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(hours=8)
    end_dt   = now + timedelta(days=span_days)
    diag = {"TE_msg":"", "FMP_msg":"", "FF_msg":""}

    # 1) Provider preferat, CHUNKING 3 zile
    all_rows = []
    if provider == "TradingEconomics":
        te_msgs = []
        for d1, d2 in _date_chunks(start_dt, end_dt, step_days=3):
            data, msg = _te_fetch(d1, d2, te_key, {"country":"united states"})
            if not data:
                data, msg2 = _te_fetch(d1, d2, te_key, None)
                msg = msg or msg2
            te_msgs.append(msg)
            if data: all_rows.extend(data)
        diag["TE_msg"] = "; ".join([m for m in te_msgs if m])[:200]
        df = _normalize_calendar_rows(all_rows)
        if not df.empty:
            df = df[df["country"].astype(str).str.contains("United", case=False, na=False)]
            df = df[(df["datetime"]>=start_dt) & (df["datetime"]<=end_dt)]
            return df, diag

        # FMP fallback (chunking)
        fmp_msgs, all_rows = [], []
        for d1, d2 in _date_chunks(start_dt, end_dt, step_days=3):
            data, msg = _fmp_fetch(d1, d2, fmp_key or "demo")
            fmp_msgs.append(msg)
            if data: all_rows.extend(data)
        diag["FMP_msg"] = "; ".join([m for m in fmp_msgs if m])[:200]
        df = _normalize_calendar_rows(all_rows)
        if not df.empty:
            if "country" in df.columns:
                df = df[df["country"].astype(str).str.contains("United", case=False, na=False)]
            df = df[(df["datetime"]>=start_dt) & (df["datetime"]<=end_dt)]
            return df, diag

    else:
        # FMP preferat
        fmp_msgs, all_rows = [], []
        for d1, d2 in _date_chunks(start_dt, end_dt, step_days=3):
            data, msg = _fmp_fetch(d1, d2, fmp_key or "demo")
            fmp_msgs.append(msg)
            if data: all_rows.extend(data)
        diag["FMP_msg"] = "; ".join([m for m in fmp_msgs if m])[:200]
        df = _normalize_calendar_rows(all_rows)
        if not df.empty:
            if "country" in df.columns:
                df = df[df["country"].astype(str).str.contains("United", case=False, na=False)]
            df = df[(df["datetime"]>=start_dt) & (df["datetime"]<=end_dt)]
            return df, diag

        # TE fallback
        te_msgs, all_rows = [], []
        for d1, d2 in _date_chunks(start_dt, end_dt, step_days=3):
            data, msg = _te_fetch(d1, d2, te_key, {"country":"united states"})
            if not data:
                data, msg2 = _te_fetch(d1, d2, te_key, None)
                msg = msg or msg2
            te_msgs.append(msg)
            if data: all_rows.extend(data)
        diag["TE_msg"] = "; ".join([m for m in te_msgs if m])[:200]
        df = _normalize_calendar_rows(all_rows)
        if not df.empty:
            df = df[df["country"].astype(str).str.contains("United", case=False, na=False)]
            df = df[(df["datetime"]>=start_dt) & (df["datetime"]<=end_dt)]
            return df, diag

    # 2) ForexFactory (două săptămâni), multi-host retry
    c1, m1 = _ff_try_fetch("/ff_calendar_thisweek.xml")
    c2, m2 = _ff_try_fetch("/ff_calendar_nextweek.xml")
    diag["FF_msg"] = (m1 or m2)[:200]
    df1 = _ff_parse(c1) if c1 else pd.DataFrame()
    df2 = _ff_parse(c2) if c2 else pd.DataFrame()
    df = pd.concat([df1, df2], ignore_index=True) if not df1.empty or not df2.empty else pd.DataFrame()
    if not df.empty:
        df = df[(df["country"].astype(str).str.upper().isin(["USD","UNITED STATES"]))]
        df = df[(df["datetime"]>=start_dt) & (df["datetime"]<=end_dt)]
    if df is None or df.empty:
        df = pd.DataFrame(columns=["datetime","imp_lbl","event","actual","forecast","previous","arrow","category","country"])
    for col in ["datetime","imp_lbl","event","actual","forecast","previous","arrow","category","country"]:
        if col not in df.columns: df[col] = ""
    return df.sort_values("datetime"), diag

# ======================================
#   REFRESH CACHE
# ======================================
if refresh:
    st.cache_data.clear()

# ======================================
#   LOAD NEWS
# ======================================
sources_csv = (src_whitelist or "").strip().replace(" ", "") or None
try:
    with st.spinner("Încarc știrile…"):
        news_df = load_news(days, pages, sources_csv, cat_filter, only_dow)
except Exception as e:
    st.error(f"Nu am putut încărca știrile: {e}")
    news_df = pd.DataFrame()

if news_df is None or news_df.empty:
    st.info("Nu am găsit știri relevante (posibil NewsAPI restricționat pe Cloud sau filtre prea stricte).")
    news_df = pd.DataFrame(columns=["publishedAt","_score","bias","title","_source_name","url","_cats"])

# ======================================
#   KPIs
# ======================================
mean_score = news_df["_score"].mean() if not news_df.empty else 0.0
bias = "Bullish" if mean_score>0.1 else ("Bearish" if mean_score<-0.1 else "Mixed")
confidence = int(min(100, abs(mean_score)*35 + min(60, (len(news_df)/5) if not news_df.empty else 0)))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Știri relevante", len(news_df))
c2.metric("Bias", bias, delta=f"{mean_score:.2f}")
c3.metric("Confidence", f"{confidence}/100")

# ======================================
#   CALENDAR INTELIGENT (azi) + DIAGNOSTIC
# ======================================
cal_df_today, diag_today = load_calendar_with_diagnostics(1, cal_provider, TE_KEY, FMP_KEY)
c4.metric("Evenimente Macro azi (US)", len(cal_df_today) if cal_df_today is not None and not cal_df_today.empty else 0)
with st.expander("🧪 Diagnostic calendar (TE/FMP/FF)"):
    st.code(diag_today, language="json")

# ======================================
#   CHARTS (știri)
# ======================================
if not news_df.empty:
    tmp = news_df.copy()
    if "publishedAt" in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp["publishedAt"], utc=True, errors="coerce").dt.tz_convert(None).dt.date
    else:
        tmp["date"] = pd.NaT
    if "bias" in tmp.columns:
        by_bias = tmp.groupby("bias", dropna=False).size().reset_index(name="count")
        if not by_bias.empty:
            st.plotly_chart(px.bar(by_bias, x="bias", y="count", title="Distribuția bias-ului"), use_container_width=True)
    if "_score" in tmp.columns:
        by_day = tmp.groupby("date")["_score"].mean().reset_index()
        if not by_day.empty:
            st.plotly_chart(px.line(by_day, x="date", y="_score", title="Scor mediu zilnic (știri)"), use_container_width=True)
    if "_source_name" not in tmp.columns:
        if "source" in tmp.columns:
            try:
                tmp["_source_name"] = tmp["source"].apply(lambda s: (s.get("name","") if isinstance(s, dict) else (s or ""))).astype(str)
            except Exception:
                tmp["_source_name"] = tmp["source"].astype(str)
        else:
            tmp["_source_name"] = ""
    top_src = tmp.groupby("_source_name", dropna=False).size().sort_values(ascending=False).head(12).reset_index(name="count").rename(columns={"_source_name":"source"})
    if not top_src.empty:
        st.plotly_chart(px.bar(top_src, x="source", y="count", title="Top surse"), use_container_width=True)

# ======================================
#   CALENDAR – UI (Azi / 3 zile / 7 zile)
# ======================================
st.subheader("🗓️ Calendar inteligent (US)")
dleft, dright = st.columns([1,2])
with dleft:
    cal_days_label = st.selectbox("Interval calendar", ["Azi", "3 zile", "7 zile"], index=1)
    span = {"Azi":1, "3 zile":3, "7 zile":7}[cal_days_label]
with dright:
    st.caption("TE/FMP cu chunking pe 3 zile; dacă nu răspund, fallback la ForexFactory (multi-host). Săgeata ⬆️/⬇️/≈ = impact estimat (euristic).")

cal_df2, diag2 = load_calendar_with_diagnostics(span, cal_provider, TE_KEY, FMP_KEY)
if cal_df2 is None or cal_df2.empty:
    st.info("Nu am date de calendar pentru intervalul ales (toți furnizorii goi sau blocați).")
else:
    need = ["datetime","imp_lbl","event","actual","forecast","previous","arrow","category"]
    for c in need:
        if c not in cal_df2.columns: cal_df2[c] = ""
    show = cal_df2[need].rename(columns={"datetime":"Ora (UTC)","imp_lbl":"Impact","event":"Eveniment","actual":"Actual","forecast":"Forecast","previous":"Previous","arrow":"US30 Impact","category":"Categorie"})
    st.dataframe(show, use_container_width=True, height=360)

# ======================================
#   NEWS TABLE + EXPORT
# ======================================
st.subheader("📰 Știri filtrate")
if not news_df.empty:
    show_news = news_df.sort_values("publishedAt", ascending=False)[["publishedAt","_source_name","title","_score","_cats","url","bias"]].rename(columns={"_source_name":"source","_score":"score","_cats":"cats"})
    st.dataframe(show_news, use_container_width=True, height=520)
    st.download_button("⬇️ Descarcă CSV filtrat", data=show_news.to_csv(index=False), file_name="us30_news_filtered.csv", mime="text/csv")
else:
    st.write("—")

# ======================================
#   💵 MONEY IN POLITICS (OpenSecrets + FTM)
# ======================================
st.markdown("## 💵 Money in Politics (OpenSecrets + FollowTheMoney)")

with st.sidebar:
    st.subheader("💾 OpenSecrets CSV")
    st.caption("Încarcă CSV/TSV exportat din OpenSecrets (lobbying/contributions).")
    os_file = st.file_uploader("Fișier OpenSecrets (.csv/.tsv)", type=["csv","tsv"])
    live_ftm = st.checkbox("Activează fallback Live (FollowTheMoney, gratuit)", value=False)

# map nume -> ticker Dow
NAME_TO_TICK = {v:k for k,v in DOW_COMPONENTS.items()}

def _read_os_csv(f):
    import io
    raw = f.read()
    sep = "\t" if f.name.lower().endswith(".tsv") else ","
    df = pd.read_csv(io.BytesIO(raw), sep=sep)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _guess_amount_col(cols):
    keys = ["amount","total","sum","expenditure","expenditures","receipts","contributions","$","spent","spending"]
    for c in cols:
        for k in keys:
            if k in c: return c
    return None

def _guess_name_cols(cols):
    cand = []
    for c in cols:
        if any(k in c for k in ["client","organization","org","recipient","committee","employer","filer","registrant","lobbyist"]):
            cand.append(c)
    return cand or []

def _clean_names(series):
    return series.astype(str).str.replace(r"\b(Inc\.?|Corporation|Corp\.?|LLC|Ltd\.?|Co\.?)\b","", regex=True).str.replace(r"\s+"," ", regex=True).str.strip()

def _attach_dow(df, name_cols):
    ent = None
    for c in name_cols:
        col = _clean_names(df[c].fillna(""))
        ent = col if ent is None else (ent + " " + col)
    df["entity"] = (ent if ent is not None else "").astype(str).str.strip()
    df["dow_match"] = None
    for full_name, tick in NAME_TO_TICK.items():
        mask = df["entity"].str.contains(full_name, case=False, na=False) | df["entity"].str.contains(tick, case=False, na=False)
        df.loc[mask, "dow_match"] = tick
    return df

def _policy_pulse(group_df, amount_col):
    df = group_df.copy()
    # detect time column
    tcol = None
    for c in ["year","yr","fyear","cycle","quarter","period","date"]:
        if c in df.columns: tcol = c; break
    if tcol is None:
        return 0.0
    try:
        df["_t"] = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    except Exception:
        df["_t"] = pd.to_datetime(df[tcol].astype(str) + "-01-01", errors="coerce", utc=True)
    df = df.dropna(subset=["_t"])
    cutoff_1y = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
    last = pd.to_numeric(df.loc[df["_t"] >= cutoff_1y, amount_col], errors="coerce").sum()
    hist = pd.to_numeric(df.loc[(df["_t"] < cutoff_1y) & (df["_t"] >= pd.Timestamp.utcnow() - pd.Timedelta(days=3*365)), amount_col], errors="coerce").sum()
    avg_hist = hist/3.0 if hist>0 else 0
    if last==0 and avg_hist==0: return 0.0
    ratio = (last/(avg_hist+1e-6)) if avg_hist>0 else (1.0 if last>0 else 0.0)
    ratio = min(2.0, max(0.0, ratio))
    return round((ratio-1.0), 2)

if os_file is not None:
    try:
        os_df = _read_os_csv(os_file)
        amt_col = _guess_amount_col(os_df.columns)
        name_cols = _guess_name_cols(os_df.columns)
        if amt_col is None or not name_cols:
            st.warning("Nu găsesc coloanele de sume sau nume entitate în CSV. Încarcă un export cu câmpuri standard (client/organization + amount).")
        else:
            os_df[amt_col] = pd.to_numeric(os_df[amt_col], errors="coerce")
            os_df = _attach_dow(os_df, name_cols)

            dow_rows = os_df.dropna(subset=["dow_match"])
            if dow_rows.empty:
                st.info("Nu am găsit potriviri clare cu companii din Dow în acest fișier.")
            else:
                grp = dow_rows.groupby("dow_match")[amt_col].sum().reset_index().sort_values(amt_col, ascending=False)
                st.plotly_chart(px.bar(grp, x="dow_match", y=amt_col, title="Top cheltuieli (CSV OpenSecrets) pe componente Dow"), use_container_width=True)

                pulses = [{"ticker": t, "policy_pulse": _policy_pulse(g, amt_col)} for t, g in dow_rows.groupby("dow_match")]
                pulse_df = pd.DataFrame(pulses).sort_values("policy_pulse", ascending=False)
                st.plotly_chart(px.bar(pulse_df, x="ticker", y="policy_pulse", title="Policy Pulse (−1..+1) – momentum 12m vs medie 3y"), use_container_width=True)

                cat_col = None
                for c in ["issue","category","sector","industry","lobbying_issues","general_issue"]:
                    if c in os_df.columns: cat_col = c; break
                if cat_col:
                    topcat = dow_rows.groupby(["dow_match", cat_col])[amt_col].sum().reset_index()
                    st.plotly_chart(px.density_heatmap(topcat, x="dow_match", y=cat_col, z=amt_col, title="Heatmap teme/policy vs companii Dow"), use_container_width=True)

                us30_policy = pulse_df["policy_pulse"].mean() if not pulse_df.empty else 0.0
                st.metric("US30 Policy Pulse (CSV)", f"{us30_policy:+.2f}")
    except Exception as e:
        st.error(f"Eroare la citirea CSV: {e}")

# Fallback „live” demo: FollowTheMoney (status simplu; recomand cheie gratuită pentru interogări reale)
def _ftm_quick_demo():
    try:
        r = requests.get("https://www.followthemoney.org/api/1.0/entities", timeout=15)
        return r.status_code
    except Exception as e:
        return str(e)

with st.expander("🔧 FollowTheMoney (demo)"):
    st.caption("Pentru rezultate utile, creează un cont gratuit la FollowTheMoney și adaugă cheia în Secrets. Aici doar verificăm accesul de bază.")
    if st.checkbox("Rulează test acces FTM"):
        st.write("FTM status:", _ftm_quick_demo())
