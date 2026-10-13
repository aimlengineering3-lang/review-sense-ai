"""
=========================================================
Review Sense AI
AI-Powered Customer Sentiment Intelligence Dashboard
=========================================================
"""

import time
import traceback
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.inference import predict


# =========================================================
# Page Config
# =========================================================

st.set_page_config(
    page_title="Review Sense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# Cached Model Access
# =========================================================

@st.cache_resource(show_spinner=False)
def load_engine():
    return predict


# =========================================================
# Styling
# =========================================================

st.markdown(
    """
    <style>
    .stApp { background: #0b0e14; color: #e6e8ee; }
    #MainMenu, header, footer {visibility: hidden;}

    .hero {
        padding: 2.2rem 2rem 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #121826 0%, #0d1420 100%);
        border: 1px solid #1f2937;
        margin-bottom: 1.6rem;
    }
    .hero-title {
        font-size: 2.6rem; font-weight: 800;
        background: linear-gradient(90deg, #6ee7ff, #7c7cff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle { font-size: 1.15rem; color: #9aa4b2; font-weight: 500; margin-bottom: 0.6rem; }
    .hero-desc { color: #7c8798; font-size: 0.98rem; max-width: 700px; }
    .badge-row { margin-top: 1rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }
    .badge { background: #131b2a; border: 1px solid #23304a; padding: 0.4rem 0.9rem; border-radius: 999px; font-size: 0.85rem; color: #c7d2e0; }
    .badge-green { border-color: #1f6e46; color: #4ade80; background: #0e1f16; }

    .metric-card { background: #121826; border: 1px solid #1f2937; border-radius: 14px; padding: 1.2rem 1.3rem; text-align: center; }
    .metric-value { font-size: 1.6rem; font-weight: 800; color: #6ee7ff; }
    .metric-label { color: #8b94a7; font-size: 0.85rem; margin-top: 0.2rem; }

    .panel { background: #121826; border: 1px solid #1f2937; border-radius: 16px; padding: 1.6rem; margin-bottom: 1.4rem; }
    .panel-title { font-size: 1.15rem; font-weight: 700; color: #e6e8ee; margin-bottom: 0.8rem; }

    .result-card { border-radius: 16px; padding: 1.8rem; text-align: center; margin-top: 1rem; border: 1px solid; }
    .result-positive { background: linear-gradient(135deg, #0e1f16, #0d1720); border-color: #1f6e46; }
    .result-negative { background: linear-gradient(135deg, #2a1414, #1a0f16); border-color: #7a2d2d; }
    .result-label { font-size: 0.85rem; letter-spacing: 0.12em; color: #8b94a7; text-transform: uppercase; }
    .result-sentiment { font-size: 2.3rem; font-weight: 800; margin: 0.3rem 0 0.9rem 0; letter-spacing: 0.04em; }
    .result-section-label { font-size: 0.78rem; letter-spacing: 0.1em; color: #8b94a7; text-transform: uppercase; margin-top: 1rem; }
    .result-interpretation { color: #c7d2e0; font-size: 0.95rem; max-width: 480px; margin: 0.4rem auto 0 auto; line-height: 1.5; }

    .stat-row { display: flex; gap: 0.8rem; margin-top: 0.6rem; }
    .stat-box { flex: 1; background: #0d1420; border: 1px solid #1f2937; border-radius: 10px; padding: 0.6rem; text-align: center; }
    .stat-value { font-size: 1.1rem; font-weight: 700; color: #e6e8ee; }
    .stat-label { font-size: 0.75rem; color: #7c8798; }

    .confidence-heading { text-align: center; color: #8b94a7; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.8rem; }
    .confidence-number { text-align: center; font-size: 2rem; font-weight: 800; color: #6ee7ff; margin-bottom: 0.4rem; }
    .gauge-caption { text-align: center; color: #7c8798; font-size: 0.85rem; margin-top: -0.6rem; }

    .footer {
        text-align: center; color: #9aa4b2; padding: 2.2rem 1rem 1.4rem 1rem;
        border-top: 1px solid #1f2937; margin-top: 2rem;
    }
    .footer strong { color: #e6e8ee; font-size: 1.15rem; }
    .footer-grid { display: flex; justify-content: center; gap: 3rem; margin: 1.2rem 0; flex-wrap: wrap; }
    .footer-col-label { font-size: 0.75rem; color: #5b6472; text-transform: uppercase; letter-spacing: 0.08em; }
    .footer-col-value { color: #c7d2e0; font-weight: 600; margin-top: 0.2rem; }
    .footer-copyright { color: #5b6472; font-size: 0.8rem; margin-top: 1rem; }

    .stButton>button { background: linear-gradient(90deg, #6ee7ff, #7c7cff); color: #0b0e14; font-weight: 700; border: none; border-radius: 10px; padding: 0.6rem 1.4rem; font-size: 1rem; }
    .stButton>button:hover { opacity: 0.88; color: #0b0e14; }

    /* Text area: default Streamlit styling keeps a white
       input background while .stApp forces light text color
       globally, which makes typed/inserted text (and the
       cursor) nearly invisible. Force dark background, light
       text, and a visible caret explicitly. */
    .stTextArea textarea {
        background-color: #0d1420 !important;
        color: #e6e8ee !important;
        caret-color: #6ee7ff !important;
        border: 1px solid #1f2937 !important;
        border-radius: 10px !important;
    }
    .stTextArea textarea::placeholder {
        color: #5b6472 !important;
        opacity: 1 !important;
    }
    .stTextArea textarea:focus {
        border-color: #6ee7ff !important;
        box-shadow: 0 0 0 1px #6ee7ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Single Source of Truth: Model Stats
#
# Every section (hero badges, dashboard row, performance
# panel, model info panel, footer) reads from this one dict
# instead of hardcoding numbers in multiple places.
# =========================================================

MODEL_STATS = {
    "accuracy": 92.39,
    "precision": 91.92,
    "recall": 93.09,
    "f1": 92.50,
    "architecture": "2-layer LSTM",
    "embedding_dim": 200,
    "hidden_size": 256,
    "vocab_size": 50000,
    "max_seq_len": 200,
    "dataset": "Amazon Reviews (200K)",
    "training_samples": "200,000",
}

CONFUSION_MATRIX = [[18184, 1650], [1393, 18773]]

# label -> full review text inserted on click
# Labels show the actual review (truncated) so it's clear
# what each example contains, rather than an abstract rating.
EXAMPLES = {
    "😊 This product exceeded my expectations. Amazing quality.": "This product exceeded my expectations. Amazing quality.",
    "😊 Fantastic experience, highly recommended.": "Fantastic experience, highly recommended.",
    "😞 Terrible product. Completely disappointed.": "Terrible product. Completely disappointed.",
    "😞 Worst purchase I have made.": "Worst purchase I have made.",
}


# =========================================================
# Session State
#
# st.session_state.review_input is written to in exactly one
# place: set_example(). The text_area widget itself owns and
# updates it on every keystroke via its `key` binding -- no
# other code path assigns to it.
# =========================================================

if "review_input" not in st.session_state:
    st.session_state.review_input = ""

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "history" not in st.session_state:
    st.session_state.history = []


def set_example(text):
    st.session_state.review_input = text
    st.session_state.prediction = None
    st.rerun()


def clear_history():
    st.session_state.history = []


def confidence_label(confidence_pct):
    if confidence_pct >= 95:
        return "Very High Confidence"
    if confidence_pct >= 85:
        return "High Confidence"
    if confidence_pct >= 70:
        return "Moderate Confidence"
    return "Low Confidence"


# =========================================================
# Hero Section
# =========================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">Review Sense AI</div>
        <div class="hero-subtitle">Deep Learning Powered Customer Sentiment Intelligence</div>
        <div class="hero-desc">
            Analyze customer opinions instantly using a neural network powered sentiment engine.
        </div>
        <div class="badge-row">
            <div class="badge badge-green">🟢 AI Engine Online</div>
            <div class="badge">{MODEL_STATS['accuracy']:.2f}% Validation Accuracy</div>
            <div class="badge">Powered by {MODEL_STATS['architecture']} Neural Network</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Dashboard Metrics Row
# =========================================================

dash_cols = st.columns(4)
dash_values = [
    (f"{MODEL_STATS['accuracy']:.2f}%", "Model Accuracy"),
    (MODEL_STATS["architecture"], "Architecture"),
    (MODEL_STATS["training_samples"], "Training Samples"),
    ("Real-Time", "Inference"),
]
for col, (value, label) in zip(dash_cols, dash_values):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")


# =========================================================
# Main Prediction Area
# =========================================================

left, right = st.columns([1.1, 0.9])

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Analyze Customer Feedback</div>', unsafe_allow_html=True)

    # Only `key` -- no `value=`, no `disabled=`. The widget is
    # the single owner of this state; nothing else in the file
    # assigns to st.session_state.review_input except
    # set_example() above.
    st.text_area(
        "Enter customer review",
        key="review_input",
        height=160,
        placeholder="Type or paste a customer review here...",
        label_visibility="collapsed",
    )

    current_text = st.session_state.review_input

    char_count = len(current_text)
    word_count = len(current_text.split())
    reading_seconds = max(int(word_count / 200 * 60), 1) if word_count else 0

    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-box"><div class="stat-value">{char_count}</div><div class="stat-label">Characters</div></div>
            <div class="stat-box"><div class="stat-value">{word_count}</div><div class="stat-label">Words</div></div>
            <div class="stat-box"><div class="stat-value">{reading_seconds}s</div><div class="stat-label">Est. Reading Time</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("**Try an example:**")
    ex_cols = st.columns(2)
    ex_items = list(EXAMPLES.items())
    with ex_cols[0]:
        for label, text in ex_items[:2]:
            st.button(label, key=f"ex_{label}", on_click=set_example, args=(text,), use_container_width=True)
    with ex_cols[1]:
        for label, text in ex_items[2:]:
            st.button(label, key=f"ex_{label}", on_click=set_example, args=(text,), use_container_width=True)

    analyze_clicked = st.button("Analyze Sentiment", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if analyze_clicked:
        text = st.session_state.review_input.strip()

        if not text:
            st.warning("Please enter a review before analyzing.")
        else:
            try:
                engine = load_engine()

                start = time.perf_counter()
                with st.spinner("Running neural network inference..."):
                    sentiment, confidence = engine(text)
                elapsed_ms = (time.perf_counter() - start) * 1000

                confidence_pct = confidence * 100
                is_positive = "Positive" in sentiment

                st.session_state.prediction = {
                    "sentiment": sentiment,
                    "confidence_pct": confidence_pct,
                    "positive_prob": confidence_pct if is_positive else 100 - confidence_pct,
                    "negative_prob": 100 - confidence_pct if is_positive else confidence_pct,
                    "elapsed_ms": elapsed_ms,
                    "words_processed": word_count,
                }

                st.session_state.history.insert(
                    0,
                    {
                        "Review": text[:100] + ("..." if len(text) > 100 else ""),
                        "Prediction": sentiment,
                        "Confidence": f"{confidence_pct:.2f}%",
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )

            except Exception:
                st.session_state.prediction = None
                st.error("Inference failed")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Prediction Result</div>', unsafe_allow_html=True)

    if st.session_state.prediction is None:
        st.markdown(
            '<p style="color:#5b6472;">Submit a review to see the AI sentiment analysis.</p>',
            unsafe_allow_html=True,
        )
    else:
        pred = st.session_state.prediction
        is_positive = "Positive" in pred["sentiment"]
        card_class = "result-positive" if is_positive else "result-negative"
        sentiment_word = "POSITIVE" if is_positive else "NEGATIVE"

        interpretation = (
            "The customer expresses a highly positive opinion. "
            "This review is likely to increase product trust and purchase confidence."
            if is_positive
            else "The customer expresses clear dissatisfaction. "
            "This review may harm product trust and should be flagged for follow-up."
        )

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-label">Customer Sentiment</div>
                <div class="result-sentiment">{sentiment_word}</div>
                <div class="result-section-label">Business Interpretation</div>
                <div class="result-interpretation">{interpretation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="confidence-heading">Prediction Confidence</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-number">{pred["confidence_pct"]:.2f}%</div>', unsafe_allow_html=True)

        fig = go.Figure(
            go.Indicator(
                mode="gauge",
                value=pred["confidence_pct"],
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#5b6472"},
                    "bar": {"color": "#6ee7ff" if is_positive else "#f87171"},
                    "bgcolor": "#0b0e14",
                    "borderwidth": 0,
                },
            )
        )
        fig.update_layout(
            height=170,
            margin=dict(l=20, r=20, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e6e8ee"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="gauge-caption">Model Confidence</div>', unsafe_allow_html=True)

        st.markdown(f'<p class="confidence-heading" style="margin-top:1rem;">{confidence_label(pred["confidence_pct"])}</p>', unsafe_allow_html=True)

        prob_cols = st.columns(2)
        with prob_cols[0]:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value" style="font-size:1.2rem;">{pred["positive_prob"]:.2f}%</div><div class="metric-label">Positive Probability</div></div>',
                unsafe_allow_html=True,
            )
        with prob_cols[1]:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value" style="font-size:1.2rem;">{pred["negative_prob"]:.2f}%</div><div class="metric-label">Negative Probability</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        biz_cols = st.columns(2)
        with biz_cols[0]:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value" style="font-size:1.2rem;">~{pred["elapsed_ms"]:.0f} ms</div><div class="metric-label">Processing Time</div></div>',
                unsafe_allow_html=True,
            )
        with biz_cols[1]:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value" style="font-size:1.2rem;">{pred["words_processed"]}</div><div class="metric-label">Words Processed</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Prediction History
# =========================================================

st.markdown('<div class="panel">', unsafe_allow_html=True)
hist_header, hist_button = st.columns([4, 1])
with hist_header:
    st.markdown('<div class="panel-title">Prediction History</div>', unsafe_allow_html=True)
with hist_button:
    st.button("Clear History", on_click=clear_history, use_container_width=True)

if not st.session_state.history:
    st.markdown('<p style="color:#5b6472;">No predictions yet.</p>', unsafe_allow_html=True)
else:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# How AI Works
# =========================================================

with st.expander("How does Review Sense AI work?"):
    st.markdown(
        """
        1. **Text preprocessing** — raw review text is cleaned and normalized.
        2. **Tokenization** — text is split into individual word tokens.
        3. **Word embeddings** — each token is mapped to a dense vector representation.
        4. **LSTM sequence learning** — a recurrent neural network processes the sequence to capture context and word order.
        5. **Classification** — the final hidden state is passed through a fully connected layer to produce a sentiment prediction.
        """
    )


# =========================================================
# AI Model Performance
# =========================================================

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">AI Model Performance</div>', unsafe_allow_html=True)

perf_cols = st.columns(4)
perf_values = [
    ("Accuracy", MODEL_STATS["accuracy"]),
    ("Precision", MODEL_STATS["precision"]),
    ("Recall", MODEL_STATS["recall"]),
    ("F1 Score", MODEL_STATS["f1"]),
]
for col, (name, value) in zip(perf_cols, perf_values):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{value:.2f}%</div><div class="metric-label">{name}</div></div>',
            unsafe_allow_html=True,
        )

conf_fig = go.Figure(
    data=go.Heatmap(
        z=CONFUSION_MATRIX,
        x=["Predicted Negative", "Predicted Positive"],
        y=["Actual Negative", "Actual Positive"],
        colorscale=[[0, "#121826"], [1, "#6ee7ff"]],
        text=CONFUSION_MATRIX,
        texttemplate="%{text}",
        showscale=False,
    )
)
conf_fig.update_layout(
    height=320,
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color": "#e6e8ee"},
)
st.plotly_chart(conf_fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Model Information Panel
#
# Accuracy / F1 deliberately omitted here -- already shown
# in the "AI Model Performance" panel above. Only structural
# facts live in this panel.
# =========================================================

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Model Information</div>', unsafe_allow_html=True)

info_cols = st.columns(5)
info_values = [
    ("Model", MODEL_STATS["architecture"]),
    ("Embedding", str(MODEL_STATS["embedding_dim"])),
    ("Hidden Size", str(MODEL_STATS["hidden_size"])),
    ("Vocabulary", f"{MODEL_STATS['vocab_size']:,}"),
    ("Dataset", MODEL_STATS["dataset"]),
]
for col, (label, value) in zip(info_cols, info_values):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value" style="font-size:1.0rem;">{value}</div><div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Footer
# =========================================================

st.markdown(
    f"""
    <div class="footer">
        <strong>Review Sense AI</strong><br>
        Deep Learning Customer Sentiment Intelligence Platform
        <div class="footer-grid">
            <div>
                <div class="footer-col-label">Architecture</div>
                <div class="footer-col-value">{MODEL_STATS['architecture']} Neural Network</div>
            </div>
            <div>
                <div class="footer-col-label">Dataset</div>
                <div class="footer-col-value">{MODEL_STATS['dataset']}</div>
            </div>
            <div>
                <div class="footer-col-label">Built using</div>
                <div class="footer-col-value">PyTorch • Streamlit • Plotly</div>
            </div>
        </div>
        <div class="footer-copyright">© 2026 Review Sense AI</div>
    </div>
    """,
    unsafe_allow_html=True,
)