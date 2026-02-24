"""
소아 허약 검사 - AI analysis v1.0
실행: streamlit run pws_factor_app.py
필요 파일: factor_loadings.csv / corr_matrix.csv / scaler_params.csv / best_svm.pkl
"""

import pickle
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import matplotlib.font_manager as fm

# ── 폰트 설정 (Streamlit Cloud 호환) ──────────────────────────────────────────
font_path = "NanumGothicCoding.ttf" 

fm.fontManager.addfont(font_path)

fontprop = fm.FontProperties(fname=font_path)
KOREAN_FONT = fontprop.get_name()

matplotlib.rcParams['font.family'] = KOREAN_FONT
matplotlib.rcParams['axes.unicode_minus'] = False

ENGLISH_FONT = 'DejaVu Sans'

# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="소아 허약 검사 - AI analysis", layout="wide")

# ── 글로벌 CSS: 폰트 크기 위계 정리 ──────────────────────────────────────────
st.markdown("""
<style>
    /* ── 페이지 제목 (st.title) ─────────────────────────── */
    h1[data-testid="stHeading"] {
        font-size: 2.0rem !important;
    }
    /* ── 섹션 제목 (st.subheader) ──────────────────────── */
    h2[data-testid="stHeading"],
    h3[data-testid="stHeading"] {
        font-size: 1.45rem !important;
    }
    /* ── st.metric value: 섹션 제목보다 약간 작게 ──────── */
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
    }
    [data-testid="stMetric"] label {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }
    /* ── 사이드바 제목 ─────────────────────────────────── */
    section[data-testid="stSidebar"] h1 {
        font-size: 1.3rem !important;
    }
    /* ── 차트 섹션 제목 (커스텀 클래스) ────────────────── */
    .chart-section-title {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        margin-bottom: 8px;
        color: #2D3748;
    }
    /* ── 참고문헌 스타일 ───────────────────────────────── */
    .ref-block {
        font-size: 0.78rem;
        color: #718096;
        line-height: 1.6;
        padding: 8px 0 4px 0;
    }
    .ref-block .ref-note {
        font-style: italic;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_factor_loadings(path="factor_loadings.csv"):
    df = pd.read_csv(path, index_col=0)
    return df.values, list(df.columns)

@st.cache_data
def load_corr_matrix(path="corr_matrix.csv"):
    return pd.read_csv(path, index_col=0).values

@st.cache_data
def load_scaler_params(path="scaler_params.csv"):
    df = pd.read_csv(path, index_col=0)
    return df['mean'].values, df['std'].values

@st.cache_resource
def load_svm_model(path="best_svm.pkl"):
    with open(path, 'rb') as f:
        return pickle.load(f)

def safe_load(fn, label):
    try:
        return fn()
    except FileNotFoundError:
        st.error(f"❌ `{label}` 파일을 찾을 수 없습니다.")
        st.stop()

FACTOR_LOADINGS, factor_col_names = safe_load(load_factor_loadings, "factor_loadings.csv")
CORR_MATRIX                        = safe_load(load_corr_matrix,     "corr_matrix.csv")
_MEAN, _STD                        = safe_load(load_scaler_params,   "scaler_params.csv")
svm_model                          = safe_load(load_svm_model,       "best_svm.pkl")

_weights = np.linalg.solve(CORR_MATRIX, FACTOR_LOADINGS)

# ══════════════════════════════════════════════════════════════════════════════
# 2. 상수
# ══════════════════════════════════════════════════════════════════════════════

QUESTION_LABELS = [
    "Q1. 눈이 자주 충혈된다",
    "Q2. 잠꼬대를 많이 하거나 자다가 자주 놀란다",
    "Q3. 배가 자주 아프다",
    "Q4. 감기에 자주 걸리고, 걸리면 잘 낫지 않는다",
    "Q5. 또래에 비해 체격이 왜소하다",
    "Q6. 눈의 피로를 빨리 느낀다",
    "Q7. 자다가 자주 깨고 울며 보챈다",
    "Q8. 배에 가스가 자주 찬다",
    "Q9. 밤과 새벽에 기침을 자주 한다",
    "Q10. 밥맛이 없고 편식을 하며 먹는 양이 적다",
    "Q11. 눈병이 자주 생긴다",
    "Q12. 화를 잘 낸다",
    "Q13. 구토나 구역질을 자주 한다",
    "Q14. 찬바람을 맞거나 찬 음식만 먹어도 기침을 한다",
    "Q15. 추위에 약하다",
    "Q16. 경련이나 쥐가 자주 난다",
    "Q17. 집중력이 떨어지고 산만하다",
    "Q18. 설사를 자주 한다",
    "Q19. 평소에 가래가 자주 생긴다",
    "Q20. 손발이 찬 편이다",
    "Q21. 자주 넘어지고 팔이나 다리를 자주 뺀다",
    "Q22. 손톱을 입으로 잘 물어 뜯는다",
    "Q23. 자주 어지럽고 머리가 아프다",
    "Q24. 감기 후 축농증, 중이염, 천식 등 합병증이 잘 생긴다",
    "Q25. 치아가 늦게 난다",
    "Q26. 손톱이나 발톱이 약해서 갈라지거나 부러진다",
    "Q27. 예민하고 신경질적이다",
    "Q28. 멀미를 자주 한다",
    "Q29. 편도가 자주 붓는다",
    "Q30. 환경의 변화에 잘 적응하지 못한다",
]

FACTOR_INFO = {
    "Factor1": {"name": "호흡기 건강\n및 면역력",    "full": "호흡기 건강 및 면역력",   "desc": "감기에 자주 걸리거나 오래 앓고, 기침·가래·편도 문제가 반복될 가능성이 있습니다. 찬 바람이나 찬 음식에 민감하게 반응하는 경향도 보일 수 있습니다."},
    "Factor2": {"name": "전반적\n신체 기능 저하",   "full": "전반적 신체 기능 저하",    "desc": "어지럼증이나 두통, 잦은 설사, 수족냉증, 손발톱 약화 등 몸 전반이 다소 허약할 가능성이 있습니다."},
    "Factor3": {"name": "안구 건강",                "full": "안구 건강",               "desc": "눈이 자주 충혈되거나 눈병이 반복되고, 눈의 피로감을 쉽게 느낄 가능성이 있습니다."},
    "Factor4": {"name": "심리·정서적\n취약성",       "full": "심리·정서적 취약성",       "desc": "감정 조절이 다소 어렵거나 예민하고 신경질적인 모습, 집중력 저하나 산만한 경향이 나타날 가능성이 있습니다."},
    "Factor5": {"name": "소화기\n기능 취약성",       "full": "소화기 기능 취약성",       "desc": "배가 자주 아프거나 가스가 차고, 설사가 반복될 가능성이 있습니다."},
    "Factor6": {"name": "성장·영양\n상태",           "full": "성장·영양 상태",           "desc": "또래보다 체격이 작거나 밥을 잘 먹지 않고 편식하는 경향이 있을 가능성이 있습니다."},
    "Factor7": {"name": "수면·신경계\n예민성",       "full": "수면·신경계 예민성",       "desc": "잠을 자다 자주 깨거나 잠꼬대, 수면 중 놀라는 등 예민한 수면 패턴을 보일 가능성이 있습니다."},
}

FACTOR_NAMES  = [FACTOR_INFO.get(c, {}).get("full", c) for c in factor_col_names]
FACTOR_LABELS = [FACTOR_INFO.get(c, {}).get("name", c) for c in factor_col_names]
FACTOR_DESCS  = [FACTOR_INFO.get(c, {}).get("desc", "") for c in factor_col_names]

CLUSTER_INFO = {
    1:  {"name": "건강아",               "emoji": "🟢", "desc": "모든 요인과 PWS 점수가 낮으며, 다차원 건강 지표가 평균 이상으로 양호한 상태일 가능성이 높습니다."},
    2:  {"name": "고도 복합 허약아",     "emoji": "🔴", "desc": "오장(간·심·비·폐·신)의 모든 기능이 허약하고 전반적으로 기능이 크게 떨어져 있을 가능성이 높습니다. 다방면의 건강 관리가 필요할 수 있습니다."},
    3:  {"name": "간폐 허약아",          "emoji": "🟡", "desc": "잦은 호흡기 감염 및 알레르기 성향을 보이며, 결막염이나 안구 충혈 등 안과 질환에 취약할 가능성이 있습니다."},
    4:  {"name": "정서불안형 심계허약아","emoji": "🟡", "desc": "심계 허약 경향이 높으며, 특히 화를 잘 내거나 산만한 모습 등 문제행동이 나타날 가능성이 높으니 정서가 안정될 수 있도록 조기 관리가 필요합니다."},
    5:  {"name": "중등도 복합 허약아",   "emoji": "🟠", "desc": "잦은 호흡기 감염, 높은 피로도, 심리적 취약성 및 낮은 성장 지표가 동시에 나타날 가능성이 있습니다."},
    6:  {"name": "신계허약아",           "emoji": "🟡", "desc": "식욕 부진이나 체구가 왜소한 것이 주된 특징일 가능성이 있습니다. 성장과 영양 관리에 집중이 필요할 수 있습니다."},
    7:  {"name": "폐계허약아",           "emoji": "🟡", "desc": "잦은 호흡기 감염으로 면역력이 약할 가능성이 높아 면역력의 집중 관리가 필요합니다."},
    8:  {"name": "비계허약아",           "emoji": "🟡", "desc": "복통, 설사, 변비 등 소화기 증상이 잦을 가능성이 있으나, 식습관이나 성장에 악영향을 미치지 않도록 조기 관리가 필요합니다."},
    9:  {"name": "신경과민형 심계허약아","emoji": "🟡", "desc": "신경계의 예민성으로 수면 중 자주 깨는 양상을 보일 가능성이 있지만, 신경계가 안정될 수 있도록 조기 관리가 필요합니다."},
    10: {"name": "경도 복합 허약아",     "emoji": "🟠", "desc": "오장 전반에 허약함이 나타날 가능성이 있지만, 그 정도는 심하지 않은 상태입니다. 생활 습관 관리만으로도 개선될 여지가 있습니다."},
}

SCALE_LABELS = {0: "전혀 아니다", 1: "별로 그렇지 않다", 2: "약간 그렇다", 3: "그렇다", 4: "매우 그렇다"}

SUBSCALES = {
    "GN": [0, 5, 10, 15, 20, 25],
    "SM": [1, 6, 11, 16, 21, 26],
    "BE": [2, 7, 12, 17, 22, 27],
    "PH": [3, 8, 13, 18, 23, 28],
    "SN": [4, 9, 14, 19, 24, 29],
}

SUBSCALE_STATS = {
    "GN": {"mean": 3.87, "std": 3.64, "label_kr": "간"},
    "SM": {"mean": 5.25, "std": 4.20, "label_kr": "심"},
    "BE": {"mean": 4.47, "std": 3.86, "label_kr": "비"},
    "PH": {"mean": 5.15, "std": 4.30, "label_kr": "폐"},
    "SN": {"mean": 6.85, "std": 3.84, "label_kr": "신"},
}

TOTAL_MEAN = 25.59
TOTAL_STD  = 16.51

# ══════════════════════════════════════════════════════════════════════════════
# 3. 계산 함수
# ══════════════════════════════════════════════════════════════════════════════

def compute_factor_scores(responses: np.ndarray) -> np.ndarray:
    return (responses - _MEAN) / _STD @ _weights

def compute_subscales(responses: np.ndarray) -> dict:
    return {k: int(responses[idx].sum()) for k, idx in SUBSCALES.items()}

def predict_cluster(factor_scores: np.ndarray) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return int(svm_model.predict(factor_scores.reshape(1, -1))[0])

def compute_percentile(score, mean, std):
    return stats.norm.cdf(score, loc=mean, scale=std) * 100

def get_top_questions_grouped(responses: np.ndarray, labels: list, top_n=3):
    pairs = [(int(responses[i]), labels[i]) for i in range(len(responses))]
    unique_scores = sorted(set(s for s, _ in pairs), reverse=True)
    groups = []
    for rank, sc in enumerate(unique_scores[:top_n], start=1):
        items = [lbl for s, lbl in pairs if s == sc]
        groups.append((rank, sc, items))
    return groups


def build_copy_text(responses_arr, subscale_scores, cluster):
    """EMR 복사용 텍스트 생성 — 5문항씩 슬래시 구분"""
    # Q1-Q5 / Q6-Q10 / Q11-Q15 / Q16-Q20 / Q21-Q25 / Q26-Q30
    groups = []
    for start in range(0, 30, 5):
        chunk = "".join(str(int(v)) for v in responses_arr[start:start + 5])
        groups.append(chunk)
    item_block = "/".join(groups)

    sub_text = "/".join(
        f"{SUBSCALE_STATS[k]['label_kr']}{v}"
        for k, v in subscale_scores.items()
    )
    c_info = CLUSTER_INFO.get(cluster, {})
    c_name = c_info.get("name", f"Cluster {cluster}")
    total = int(responses_arr.sum())
    return f"PWS({item_block}) {sub_text} Total={total} → Cluster {cluster} ({c_name})"


# ══════════════════════════════════════════════════════════════════════════════
# 4. 시각화 함수
# ══════════════════════════════════════════════════════════════════════════════

def plot_radar(factor_scores, factor_labels):
    matplotlib.rcParams['font.family'] = KOREAN_FONT
    N = len(factor_scores)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values = factor_scores.tolist()
    a_closed = angles + angles[:1]
    v_closed = values + values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(a_closed, v_closed, 'o-', linewidth=2, color='#4C72B0')
    ax.fill(a_closed, v_closed, alpha=0.2, color='#4C72B0')
    ax.plot(a_closed, [0] * (N + 1), color='gray', linewidth=0.8, linestyle='--')
    ax.set_ylim(-2, 2)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(["-2", "-1", "0", "+1", "+2"], fontsize=7, color='gray')
    ax.set_rlabel_position(15)
    ax.set_thetagrids(np.degrees(angles), factor_labels, fontsize=12, fontweight='bold')
    ax.set_title("Factor Radar Chart", pad=20, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_subscale_bar(subscale_scores: dict):
    matplotlib.rcParams['font.family'] = ENGLISH_FONT
    names  = list(subscale_scores.keys())
    values = list(subscale_scores.values())

    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(names, values, color='#55A868', width=0.5)
    ax.set_ylim(0, 26)
    ax.set_ylabel("Score")
    ax.set_title("Subscale Scores", fontsize=14, fontweight='bold')
    ax.axhline(5, color='#E8A838', linewidth=1.8, linestyle='--', label='Mild cutoff (5)')
    ax.axhline(8, color='#DD4444', linewidth=1.8, linestyle='--', label='Severe cutoff (8)')
    ax.legend(fontsize=9, loc='upper right')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_normal_dist(total_score):
    matplotlib.rcParams['font.family'] = ENGLISH_FONT
    mean, std = TOTAL_MEAN, TOTAL_STD
    x = np.linspace(mean - 3.5 * std, mean + 3.5 * std, 400)
    y = stats.norm.pdf(x, mean, std)
    pct = compute_percentile(total_score, mean, std)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x, y, color='#4C72B0', linewidth=2)
    x_fill = np.linspace(x[0], min(total_score, x[-1]), 300)
    ax.fill_between(x_fill, stats.norm.pdf(x_fill, mean, std), alpha=0.35, color='#4C72B0')
    ax.axvline(total_score, color='#DD4444', linewidth=2)
    ax.text(total_score + std * 0.1, max(y) * 0.88,
            f"{total_score} pts\n({pct:.1f}th %ile)",
            color='#DD4444', fontsize=10, fontweight='bold')
    ax.set_xlabel("PWS Total Score")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"PWS Total Score Distribution  (Mean={mean}, SD={std})",
                 fontsize=11, fontweight='bold')
    ax.set_xlim(x[0], x[-1])
    fig.tight_layout()
    return fig, pct

# ══════════════════════════════════════════════════════════════════════════════
# 5. 사이드바: 고속 데이터 입력 폼
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📋 PWS Data Input")
    st.caption(
        "각 문항에 0~4 사이 숫자를 입력하세요.  \n"
        "**Tab** 키로 다음 문항으로 빠르게 이동할 수 있습니다.  \n"
        "0 전혀 아니다 · 1 별로 그렇지 않다 · 2 약간 그렇다 · 3 그렇다 · 4 매우 그렇다"
    )

    with st.form("pws_input_form"):
        responses = []
        for i, label in enumerate(QUESTION_LABELS):
            val = st.number_input(
                label, min_value=0, max_value=4, value=0, step=1, key=f"q_{i}",
            )
            responses.append(val)

        submitted = st.form_submit_button("✅ Run Analysis", use_container_width=True)
        if submitted:
            st.session_state.responses = np.array(responses, dtype=float)
            st.session_state.analyzed = True

# ══════════════════════════════════════════════════════════════════════════════
# 6. 메인 영역: 결과 표시
# ══════════════════════════════════════════════════════════════════════════════

st.title("소아 허약 검사 — AI analysis")

# ── 참고문헌 (제목 바로 아래) ─────────────────────────────────────────────────
st.markdown("""
<div class="ref-block">
    <div class="ref-note">※ 본 분석은 아래 연구들을 기반으로 하고 있습니다.</div>
    <div>[1] Chae H, Han SY, Chen JH, Kim KB. Development and Validation of the Pediatric Weakness Scale. J Korean Med Pediatr. 2019;33(3):30-41.</div>
    <div>[2] Kim TH, Yoon SU, Choi SY, Bang MR, Han JH, Chang GT, Lee JY, Lee SH. Pattern Analysis of Pediatric Weakness Using the Pediatric Weakness Scale and Development of a Machine Learning-Based Prediction Model. J Korean Med Pediatr. 2025;39(2):38-53.</div>
    <div>[3] Kim TH, Chae H, Han JH, Bang MR, Chang GT, Lee JY, Lee SH. Cutoff value for Pediatric Weakness Scale for diagnosis of pediatric weakness. J Integr Med. 2026;24:105-114.</div>
</div>
""", unsafe_allow_html=True)

st.caption("한방소아과 연구팀 전용 v1.0")
st.divider()

if not st.session_state.get("analyzed", False):
    st.info("👈 왼쪽 사이드바에서 문항 응답을 입력한 뒤 **Run Analysis** 버튼을 눌러주세요.")
    st.stop()

# ── 분석 실행 ─────────────────────────────────────────────────────────────────
responses_arr   = st.session_state.responses
factor_scores   = compute_factor_scores(responses_arr)
subscale_scores = compute_subscales(responses_arr)
cluster         = predict_cluster(factor_scores)
total_score     = int(responses_arr.sum())

# ── 군집 예측 결과 ────────────────────────────────────────────────────────────
st.subheader("🎯 AI 허약 분석 결과 (SVM)")
c_info  = CLUSTER_INFO.get(cluster, {})
c_name  = c_info.get("name", f"Cluster {cluster}")
c_emoji = c_info.get("emoji", "⚪")
c_desc  = c_info.get("desc", "")

st.metric(label="Predicted Cluster", value=f"{c_emoji}  Cluster {cluster}  |  {c_name}")
st.info(c_desc)
st.warning(
    "⚠️ 이 결과는 설문지 응답을 바탕으로 한 분석 결과입니다. "
    "정확한 진단은 반드시 전문 한의사와의 상담을 통해 이루어져야 합니다."
)

st.divider()

# ── 주요 호소 문항 ────────────────────────────────────────────────────────────
st.subheader("📌 Chief Complaints")

groups = get_top_questions_grouped(responses_arr, QUESTION_LABELS, top_n=3)
SHOW_LIMIT = 5

rank_bg  = {1: "#FFF3F3", 2: "#FFFBF0", 3: "#F5FAF5"}
rank_bdr = {1: "#E57373", 2: "#FFB74D", 3: "#81C784"}

shown_count  = 0
hidden_groups = []

for rank, score, labels in groups:
    if score == 0:
        continue
    if shown_count >= SHOW_LIMIT:
        hidden_groups.append((rank, score, labels))
        continue

    remaining      = SHOW_LIMIT - shown_count
    visible_labels = labels[:remaining]
    hidden_in_rank = labels[remaining:]

    bg  = rank_bg.get(rank, "#FAFAFA")
    bdr = rank_bdr.get(rank, "#BDBDBD")

    items_md = "".join(
        f'<li style="font-size:17px; line-height:2.2;">{lbl}</li>'
        for lbl in visible_labels
    )
    st.markdown(
        f"""
        <div style="
            background:{bg};
            border-left: 5px solid {bdr};
            border-radius: 6px;
            padding: 14px 20px;
            margin-bottom: 10px;
        ">
            <div style="font-weight:700; font-size:17px; margin-bottom:6px; color:#333;">
                {rank}순위 &nbsp;—&nbsp; {score}점
            </div>
            <ul style="margin:0; padding-left:20px;">
                {items_md}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    shown_count += len(visible_labels)
    if hidden_in_rank:
        hidden_groups.append((rank, score, hidden_in_rank))

if hidden_groups:
    total_hidden = sum(len(lbls) for _, _, lbls in hidden_groups)
    with st.expander(f"동점 문항 {total_hidden}개 더 보기"):
        for rank, score, labels in hidden_groups:
            for lbl in labels:
                st.markdown(f"- {rank}순위 ({score}점) — **{lbl}**")

st.divider()

# ── 시각화 ────────────────────────────────────────────────────────────────────
st.subheader("📊 분석 결과")
col_a, col_b = st.columns([1.1, 1])

with col_a:
    st.markdown('<div class="chart-section-title">Factor Radar Chart</div>', unsafe_allow_html=True)
    st.pyplot(plot_radar(factor_scores, FACTOR_LABELS))

with col_b:
    st.markdown('<div class="chart-section-title">Subscale Scores</div>', unsafe_allow_html=True)
    st.pyplot(plot_subscale_bar(subscale_scores))

    # ── Subscale 점수: HTML flex로 차트 바로 밑에 정렬 ─────────────────────
    sub_cells = ""
    for key, sc in subscale_scores.items():
        s   = SUBSCALE_STATS[key]
        pct = compute_percentile(sc, s["mean"], s["std"])
        sub_cells += f"""
        <div style="text-align:center; flex:1; min-width:0;">
            <div style="font-size:12px; color:#666; margin-bottom:2px;">
                {key} ({s['label_kr']})
            </div>
            <div style="font-size:15px; font-weight:700; color:#2D3748;">
                {sc}점
            </div>
            <div style="font-size:13px; color:#38A169;">
                ▲ 상위 {100 - pct:.1f}%
            </div>
        </div>
        """
    st.markdown(
        f"""
        <div style="
            display:flex;
            justify-content:space-around;
            gap:4px;
            padding: 6px 0 0 0;
        ">
            {sub_cells}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── 총점 정규분포 ─────────────────────────────────────────────────────────────
st.subheader("PWS Total Score")
st.caption(
    "⚠️ 본 백분위수는 **6~9세** 아동 데이터를 기반으로 산출되었습니다. "
    "해당 연령 범위 밖의 아동에게는 정확도가 낮을 수 있으니 해석에 유의하시기 바랍니다."
)
fig_nd, total_pct = plot_normal_dist(total_score)
col_nd1, col_nd2 = st.columns([2, 1])
with col_nd1:
    st.pyplot(fig_nd)
with col_nd2:
    st.metric("PWS Total Score", f"{total_score}점")
    st.metric("백분위", f"{total_pct:.1f}%ile")
    st.caption(f"같은 연령대 아동 중 상위 **{100 - total_pct:.1f}%** 수준입니다.")

st.divider()

# ── Factor별 설명 ─────────────────────────────────────────────────────────────
st.subheader("Factor별 설명")
for col_key, score, name, desc in zip(factor_col_names, factor_scores, FACTOR_NAMES, FACTOR_DESCS):
    with st.expander(f"**{col_key} · {name}** — 점수: {score:+.2f}"):
        st.write(desc)

st.divider()

# ── 상세 수치 테이블 ──────────────────────────────────────────────────────────
st.subheader("📋 상세 수치")
col_t1, col_t2 = st.columns(2)

with col_t1:
    st.markdown("**Factor 점수**")
    st.dataframe(
        pd.DataFrame({"요인": FACTOR_NAMES, "점수": factor_scores.round(3)}),
        use_container_width=True, hide_index=True,
    )

with col_t2:
    st.markdown("**하위척도 점수 및 백분위**")
    pct_rows = []
    for key, score in subscale_scores.items():
        s   = SUBSCALE_STATS[key]
        pct = compute_percentile(score, s["mean"], s["std"])
        pct_rows.append({"하위척도": f"{key} ({s['label_kr']})", "합산 점수": score, "백분위": f"{pct:.1f}%"})
    st.dataframe(pd.DataFrame(pct_rows), use_container_width=True, hide_index=True)

st.divider()

# ── 📋 EMR Copy 버튼 (페이지 맨 아래) ────────────────────────────────────────
copy_text = build_copy_text(responses_arr, subscale_scores, cluster)
copy_text_escaped = copy_text.replace("\\", "\\\\").replace("`", "\\`")

st.subheader("📋 EMR Copy")
st.markdown(
    f"""
    <div style="
        background: #F0F4F8;
        border: 1px solid #CBD5E0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
    ">
        <code style="
            flex: 1;
            font-size: 13px;
            word-break: break-all;
            color: #2D3748;
            background: transparent;
        ">{copy_text}</code>
        <button onclick="
            navigator.clipboard.writeText(`{copy_text_escaped}`);
            this.textContent='✅ Copied!';
            setTimeout(()=>this.textContent='📋 Copy', 1500);
        " style="
            background: #4C72B0;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 20px;
            font-size: 14px;
            cursor: pointer;
            white-space: nowrap;
        ">📋 Copy</button>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()
st.caption("한방소아과 연구팀 전용 | 버전 1.0")
