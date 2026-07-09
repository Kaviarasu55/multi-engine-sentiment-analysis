import streamlit as st
import pandas as pd
import numpy as np
import time
from modules.preprocess import load_data
from modules.baseline import train_baseline, predict_baseline
from modules.distilbert_engine import load_distilbert, predict_distilbert
from modules.groq_engine import predict_groq
from modules.evaluator import evaluate_model, plot_confusion_matrix, plot_roc_auc
from modules.logger import log_prediction

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")


@st.cache_resource(show_spinner=False)
def get_baseline():
    X_train, X_test, y_train, y_test, le = load_data()
    model, vectorizer = train_baseline(X_train, y_train)
    return model, vectorizer, X_test, y_test, le


@st.cache_resource(show_spinner=False)
def get_distilbert():
    model, tokenizer = load_distilbert()
    return model, tokenizer


with st.spinner("Loading models, please wait..."):
    baseline_model, vectorizer, X_test, y_test, le = get_baseline()
    distilbert_model, tokenizer = get_distilbert()

class_names = list(le.classes_)

if "results" not in st.session_state:
    st.session_state.results = None

page = st.sidebar.selectbox("Navigate", ["Analyzer", "Evaluation", "History Dashboard"])

if page == "Analyzer":
    st.title("🎯 Sentiment Analyzer")
    st.caption("Compare TF-IDF, DistilBERT, and Groq LLM on the same input.")

    with st.form("analyzer_form"):
        text_input = st.text_area(
            "Enter a review or comment",
            height=120,
            placeholder="e.g. The flight was delayed 3 hours and staff was rude",
        )
        analyze_clicked = st.form_submit_button("Analyze", type="primary")

    if analyze_clicked and text_input.strip():
        start = time.time()
        b_label, b_conf, b_probs = predict_baseline(
            text_input, baseline_model, vectorizer
        )
        b_time = time.time() - start
        b_name = le.inverse_transform([b_label])[0]
        log_prediction(text_input, "tfidf", b_name, b_conf, b_time)

        start = time.time()
        d_label, d_conf, d_probs = predict_distilbert(
            text_input, distilbert_model, tokenizer
        )
        d_time = time.time() - start
        d_name = le.inverse_transform([d_label])[0]
        log_prediction(text_input, "distilbert", d_name, d_conf, d_time)

        start = time.time()
        g_name = predict_groq(text_input)
        g_time = time.time() - start
        log_prediction(text_input, "groq", g_name, None, g_time)

        st.session_state.results = {
            "text": text_input,
            "b_name": b_name,
            "b_conf": b_conf,
            "b_time": b_time,
            "b_label": b_label,
            "d_name": d_name,
            "d_conf": d_conf,
            "d_time": d_time,
            "g_name": g_name,
            "g_time": g_time,
        }

    if st.session_state.results:
        r = st.session_state.results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Engine 1 — TF-IDF")
            st.metric("Sentiment", r["b_name"].upper())
            st.metric("Confidence", f"{r['b_conf'] * 100:.1f}%")
            st.metric("Response Time", f"{r['b_time']:.3f}s")

        with col2:
            st.subheader("Engine 2 — DistilBERT")
            st.metric("Sentiment", r["d_name"].upper())
            st.metric("Confidence", f"{r['d_conf'] * 100:.1f}%")
            st.metric("Response Time", f"{r['d_time']:.3f}s")

        with col3:
            st.subheader("Engine 3 — Groq LLM")
            st.metric("Sentiment", r["g_name"].upper())
            st.metric("Confidence", "N/A")
            st.metric("Response Time", f"{r['g_time']:.3f}s")
            st.caption("⚠️ Groq doesn't expose internal probabilities — label only.")

        st.divider()
        all_labels = [r["b_name"], r["d_name"], r["g_name"]]
        if len(set(all_labels)) == 1:
            st.success(f"✅ All 3 engines agree — **{r['b_name'].upper()}**")
        else:
            st.warning(
                f"⚠️ Engines disagree — TF-IDF: **{r['b_name']}** | DistilBERT: **{r['d_name']}** | Groq: **{r['g_name']}**"
            )

        st.divider()
        st.subheader("🔍 TF-IDF Explainability")
        st.caption("Top words that contributed to TF-IDF's prediction.")
        feature_names = vectorizer.get_feature_names_out()
        coefs = baseline_model.coef_[r["b_label"]]
        text_tfidf = vectorizer.transform([r["text"]]).toarray()[0]
        scores = coefs * text_tfidf
        top_indices = np.argsort(scores)[-5:][::-1]
        top_words = [
            (feature_names[i], round(float(scores[i]), 3))
            for i in top_indices
            if scores[i] > 0
        ]
        if top_words:
            for word, score in top_words:
                st.write(f"`{word}` → score: **{score}**")
        else:
            st.write("No strong contributing words found for this input.")

elif page == "Evaluation":
    st.title("📊 Model Evaluation")
    st.caption("ROC-AUC, F1-score, and Confusion Matrix for TF-IDF vs DistilBERT.")
    st.subheader("Engine 1 — TF-IDF Baseline")

    with st.spinner("Running TF-IDF predictions on test set..."):
        y_pred_b, y_proba_b = [], []
        for text in X_test:
            label, conf, probs = predict_baseline(text, baseline_model, vectorizer)
            y_pred_b.append(label)
            y_proba_b.append(probs)

    report_b, cm_b, roc_b = evaluate_model(
        y_test.tolist(), y_pred_b, np.array(y_proba_b), class_names
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", round(roc_b, 4))
    col2.metric("F1 Weighted", round(report_b["weighted avg"]["f1-score"], 4))
    col3.metric("Accuracy", round(report_b["accuracy"], 4))

    col4, col5 = st.columns(2)
    with col4:
        st.pyplot(plot_confusion_matrix(cm_b, class_names))
    with col5:
        st.pyplot(plot_roc_auc(y_test.tolist(), np.array(y_proba_b), class_names))

    st.divider()
    st.subheader("Engine 2 — DistilBERT Fine-tuned")
    st.caption(
        "Evaluated on full 2,928 test samples. Computed offline to avoid wait time."
    )

    col6, col7, col8 = st.columns(3)
    col6.metric("ROC-AUC", 0.9245)
    col7.metric("F1 Weighted", 0.8199)
    tfidf_f1 = round(report_b["weighted avg"]["f1-score"], 4)
    distilbert_f1 = 0.8199
    diff = round(distilbert_f1 - tfidf_f1, 4)
    col8.metric(
        "F1 vs TF-IDF", distilbert_f1, delta=f"+{diff}" if diff > 0 else str(diff)
    )

    st.info(
        "💡 DistilBERT was fine-tuned incrementally across the full 11,712 samples in 4 rounds — matching production training without a single 6-hour run. Result: F1 0.8199 vs TF-IDF 0.7767 — a clear transformer advantage."
    )

elif page == "History Dashboard":
    st.title("📋 Prediction History")
    st.caption("All predictions logged across all 3 engines.")
    st.info(
        "⚠️ Prediction history resets on every app restart — cloud filesystem is temporary by design."
    )

    try:
        df = pd.read_csv("logs/predictions_log.csv")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", len(df))
        counts = df["label"].value_counts(normalize=True) * 100
        col2.metric("Positive %", f"{counts.get('positive', 0):.1f}%")
        col3.metric("Negative %", f"{counts.get('negative', 0):.1f}%")
        col4.metric("Neutral %", f"{counts.get('neutral', 0):.1f}%")
        st.divider()
        st.subheader("Predictions by Engine")
        st.bar_chart(df["engine"].value_counts())
        st.divider()
        st.subheader("Recent Predictions")
        st.dataframe(
            df.tail(20).iloc[::-1].reset_index(drop=True).fillna("N/A"),
            use_container_width=True,
        )
    except FileNotFoundError:
        st.info(
            "No predictions yet. Go to the Analyzer page and run some predictions first."
        )
