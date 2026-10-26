"""Section: Text Lab - classify text with TF-IDF models and see which words drive each class."""

import plotly.express as px
import streamlit as st

from src.core.code_ds import text_snippet
from src.core.code_snippets import load_kwargs
from src.core.ui_components import (
    render_code_button,
    require_dataset,
)
from src.datascience import text_features as tf

SUBTITLE = (
    "Turn a text column into features, classify it, and read which words push toward each class."
)


def render() -> None:
    """Render this section inside its parent page."""

    dataset = require_dataset("Text Lab")
    df = dataset.working_df
    LOAD = load_kwargs(dataset)
    text_cols = [
        c for c in df.columns if df[c].dtype == object and df[c].astype(str).str.len().median() > 15
    ]
    if not text_cols:
        st.warning(
            "No long text columns were found (looking for text with a median length above 15 characters)."
        )
        return

    c1, c2 = st.columns(2)
    text_col = c1.selectbox("Text column", text_cols, key="tx_col")
    target = c2.selectbox(
        "Label to predict",
        [c for c in df.columns if c != text_col and 2 <= df[c].nunique() <= 30],
        key="tx_target",
    )

    s = tf.text_summary(df, text_col)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents", f"{s['documents']:,}")
    m2.metric("Avg words", f"{s['mean_words']:.1f}")
    m3.metric("Vocabulary", f"{s['vocabulary']:,}")
    m4.metric("Empty", s["empty"])
    with st.expander("Most common words"):
        st.plotly_chart(
            px.bar(
                s["top_words"].iloc[::-1],
                orientation="h",
                template="plotly_dark",
                labels={"value": "count", "index": "word"},
            ),
            use_container_width=True,
        )

    k1, k2, k3, k4 = st.columns(4)
    model = k1.selectbox("Model", list(tf.TEXT_MODELS), key="tx_model")
    ngram = k2.selectbox(
        "Word groups",
        [(1, 1), (1, 2), (1, 3)],
        index=1,
        format_func=lambda g: f"{g[0]}-{g[1]} words",
        key="tx_ngram",
    )
    max_feat = k3.select_slider(
        "Max vocabulary", [2000, 5000, 10000, 20000, 50000], value=20000, key="tx_maxf"
    )
    stop = k4.checkbox("Remove English stop words", value=True, key="tx_stop")

    if st.button("Train and evaluate", type="primary", key="tx_run"):
        with st.spinner("Cross-validating the text model..."):
            try:
                st.session_state["tx_res"] = (
                    text_col,
                    target,
                    tf.text_classification(
                        df,
                        text_col,
                        target,
                        model,
                        ngram,
                        max_feat,
                        stop_words="english" if stop else None,
                    ),
                )
            except tf.TextError as err:
                st.error(str(err))
    stored = st.session_state.get("tx_res")
    if stored and stored[0] == text_col and stored[1] == target:
        res = stored[2]
        a, b, c = st.columns(3)
        a.metric("CV accuracy", f"{res['cv_accuracy']:.3f}", help=f"± {res['cv_accuracy_std']:.3f}")
        b.metric("CV macro F1", f"{res['cv_f1_macro']:.3f}")
        c.metric("Majority-class baseline", f"{res['majority_baseline']:.3f}")
        if res["cv_accuracy"] < res["majority_baseline"] + 0.03:
            st.warning(
                "The model barely beats always guessing the most common label: the text may not carry this signal."
            )
        t_terms, t_conf, t_try = st.tabs(["🔤 Top terms", "🧾 Confusion & report", "✍️ Try it"])
        with t_terms:
            terms = res["top_terms"]
            if len(terms):
                st.plotly_chart(
                    px.bar(
                        terms,
                        x="weight",
                        y="term",
                        color="class",
                        facet_col="class",
                        facet_col_wrap=3,
                        orientation="h",
                        template="plotly_dark",
                        height=max(320, 30 * terms["term"].nunique() // 2),
                    ),
                    use_container_width=True,
                )
            else:
                st.info(
                    "This model type does not expose term weights; pick Logistic Regression or Naive Bayes."
                )
        with t_conf:
            st.plotly_chart(
                px.imshow(
                    res["confusion"],
                    text_auto=True,
                    color_continuous_scale="Blues",
                    template="plotly_dark",
                    title="Held-out confusion matrix",
                ),
                use_container_width=True,
            )
            st.dataframe(res["report"].style.format("{:.3f}"), use_container_width=True)
        with t_try:
            sample = st.text_area("Type or paste text (one document per line)", key="tx_try")
            lines = [ln for ln in sample.splitlines() if ln.strip()]
            if lines:
                st.dataframe(
                    tf.predict_text(res["pipeline"], lines).style.format(
                        {"confidence": "{:.0%}"}, na_rep="-"
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
        render_code_button(
            f"Text classification ({model})",
            lambda o: text_snippet(text_col, target, model, options=o, **LOAD),
            key="code_tx",
        )
    else:
        st.info("Choose the text and the label, then train.")
    return
