"""Credit Risk Scoring: Discriminant Analysis vs Logistic Regression.
Run with:  streamlit run app.py
Both models are trained on the UCI Statlog German Credit data when the app starts."""
import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
COLS = ["checking", "duration", "history", "purpose", "amount", "savings", "employment", "rate",
        "personal", "debtors", "residence", "property", "age", "other_plans", "housing",
        "existing", "job", "dependents", "telephone", "foreign", "target"]
NUM = ["duration", "amount", "rate", "age", "existing", "dependents"]
CAT = ["checking", "history", "savings", "employment", "housing"]
LABELS = {
    "checking": {"A11": "Below 0 DM", "A12": "0 to 200 DM", "A13": "200 DM or more", "A14": "No checking account"},
    "history": {"A30": "No credits / all repaid", "A31": "All repaid at this bank", "A32": "Existing credits repaid on time",
                "A33": "Delay in past payments", "A34": "Critical account"},
    "savings": {"A61": "Below 100 DM", "A62": "100 to 500 DM", "A63": "500 to 1000 DM", "A64": "1000 DM or more", "A65": "Unknown / none"},
    "employment": {"A71": "Unemployed", "A72": "Under 1 year", "A73": "1 to 4 years", "A74": "4 to 7 years", "A75": "7 years or more"},
    "housing": {"A151": "Rent", "A152": "Own", "A153": "Free"},
}
NAMES = {"checking": "Checking account", "history": "Credit history", "savings": "Savings", "employment": "Employment",
         "housing": "Housing", "duration": "Duration (months)", "amount": "Credit amount (DM)", "rate": "Installment rate (% of income)",
         "age": "Age", "existing": "Existing credits", "dependents": "Dependents"}

st.set_page_config(page_title="Credit Risk: LDA vs Logistic", page_icon="📊", layout="wide")


@st.cache_data(show_spinner="Loading dataset…")
def load_data(uploaded=None):
    src = uploaded if uploaded is not None else URL
    df = pd.read_csv(src, sep=r"\s+", header=None, names=COLS)
    df["y"] = (df["target"] == 2).astype(int)  # 1 = bad credit / default
    return df


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


@st.cache_resource(show_spinner="Training both models…")
def train(df):
    X, y = df[NUM + CAT], df["y"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pre = ColumnTransformer([("n", StandardScaler(), NUM), ("c", OneHotEncoder(drop="first"), CAT)], sparse_threshold=0)
    Ztr, Zte = pre.fit_transform(Xtr), pre.transform(Xte)
    models = {"Logistic Regression": LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000).fit(Ztr, ytr),
              "LDA": LinearDiscriminantAnalysis(solver="svd").fit(Ztr, ytr)}
    rows = []
    for name, m in models.items():
        p = m.predict_proba(Zte)[:, 1]
        pred = (p >= 0.5).astype(int)
        tn, fp, _, _ = confusion_matrix(yte, pred).ravel()
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, pred), "AUC-ROC": roc_auc_score(yte, p),
                     "KS": ks_stat(yte, p), "Sensitivity": recall_score(yte, pred), "Specificity": tn / (tn + fp)})
    return pre, models, pd.DataFrame(rows).round(3), len(Xtr), len(Xte)


def contributions(pre, model, row):
    v = pre.transform(row)[0]
    parts = pd.Series(model.coef_[0] * v, index=pre.get_feature_names_out())
    key = [n.split("__")[1].rsplit("_", 1)[0] if n.startswith("c__") else n.split("__")[1] for n in parts.index]
    return parts.groupby(key).sum(), float(model.intercept_[0])


# ---------- data ----------
st.title("Credit Risk Scoring: Discriminant Analysis vs Logistic Regression")
st.caption("Both models are trained live on the UCI Statlog German Credit data. Enter a borrower profile and press Calculate.")
try:
    df = load_data()
except Exception:
    st.warning("Could not download the dataset. Upload `german.data` from the UCI repository.")
    up = st.file_uploader("german.data", type=["data", "txt", "csv"])
    if up is None:
        st.stop()
    df = load_data(up)
pre, models, metrics, ntr, nte = train(df)

# ---------- inputs ----------
left, right = st.columns([1, 2], gap="large")
with left:
    st.subheader("Borrower profile")
    inp = {}
    inp["duration"] = st.slider(NAMES["duration"], 4, 72, 18)
    inp["amount"] = st.slider(NAMES["amount"], 250, 20000, 3000, step=50)
    inp["rate"] = st.slider(NAMES["rate"], 1, 4, 3)
    inp["age"] = st.slider(NAMES["age"], 19, 75, 35)
    inp["existing"] = st.slider(NAMES["existing"], 1, 4, 1)
    inp["dependents"] = st.slider(NAMES["dependents"], 1, 2, 1)
    for c in CAT:
        code = st.selectbox(NAMES[c], list(LABELS[c]), format_func=LABELS[c].get, key=c)
        inp[c] = code
    cutoff = st.slider("Decision cut-off (reject at or above)", 0.10, 0.90, 0.50, 0.01)
    go = st.button("Calculate risk", type="primary", use_container_width=True)

# ---------- processing and results ----------
with right:
    if go:
        steps = ["Validating inputs…", "Standardising and encoding features…", "Logistic Regression: computing logit…",
                 "LDA: computing discriminant score…", "Converting scores to probabilities…", "Comparing with the cut-off…"]
        bar, msg = st.progress(0), st.empty()
        for i in range(50):  # 50 x 0.1 s = 5 seconds
            msg.info(steps[min(len(steps) - 1, i * len(steps) // 50)])
            bar.progress((i + 1) / 50)
            time.sleep(0.1)
        bar.empty(); msg.empty()
        row = pd.DataFrame([inp])
        st.session_state["res"] = (inp.copy(), cutoff, {n: m.predict_proba(pre.transform(row))[0, 1] for n, m in models.items()},
                                   {n: contributions(pre, m, row) for n, m in models.items()})
    res = st.session_state.get("res")
    if res is None:
        st.info("Set the profile on the left and press **Calculate risk**. Processing takes about 5 seconds.")
    else:
        used, cut, prob, contrib = res
        if used != inp or cut != cutoff:
            st.warning("Inputs changed since the last calculation. Press Calculate risk to update.")
        st.subheader("Predicted probability of default")
        c1, c2 = st.columns(2)
        for col, name in zip((c1, c2), prob):
            p = prob[name]
            verdict = "Reject" if p >= cut else ("Review" if p >= cut * 0.7 else "Approve")
            col.metric(name, f"{p * 100:.1f}%", verdict, delta_color="off")
            col.progress(float(p))
        a, b = (prob[n] >= cut for n in prob)
        gap = abs(prob["Logistic Regression"] - prob["LDA"]) * 100
        (st.success if a == b else st.warning)(
            ("Both models agree: " + ("reject or review." if a else "approve.")) if a == b
            else "The models disagree. Send this case to manual review.")
        st.caption(f"Probability gap: {gap:.1f} percentage points")

        st.subheader("What drives the score")
        cdf = pd.DataFrame({n: contrib[n][0] for n in contrib}).rename(index=NAMES).sort_values("Logistic Regression")
        st.bar_chart(cdf)
        st.caption("Positive values raise default risk, negative values lower it.")

        st.subheader("How the scores were calculated")
        st.latex(r"z=\frac{x-\mu}{\sigma}\qquad \text{logit}=\beta_0+\sum_j \beta_j z_j\qquad P(\text{default})=\frac{1}{1+e^{-\text{logit}}}")
        st.latex(r"\delta=c_0+\sum_j w_j z_j\qquad P(\text{default})=\frac{1}{1+e^{-\delta}}")
        for n in contrib:
            s, b0 = contrib[n][0].sum(), contrib[n][1]
            st.write(f"**{n}:** intercept {b0:.3f} + sum of terms {s:.3f} = score **{b0 + s:.3f}**, "
                     f"probability **{prob[n] * 100:.1f}%**")

# ---------- model comparison ----------
st.divider()
st.subheader("Model comparison on the held-out test set")
st.dataframe(metrics.set_index("Model"), use_container_width=True)
st.caption(f"Trained on {ntr} applicants, tested on {nte}. These numbers come from the models trained by this app on startup.")

# ---------- dataset ----------
st.subheader("Dataset and training setup")
st.markdown(f"""
- **Dataset:** Statlog (German Credit Data), UCI Machine Learning Repository, {len(df):,} applicants, label good or bad credit ({int((df.y == 0).sum())} / {int(df.y.sum())}).
- **Source:** archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- **Why this dataset:** public microfinance data is rare, and German Credit is the standard small-loan benchmark for scoring models.
- **Features used:** {", ".join(NAMES[c] for c in NUM + CAT)}.
- **Preprocessing:** 80/20 stratified split (random_state = 42), StandardScaler on numeric columns, one-hot encoding on categorical columns.
- **Logistic Regression:** L2 penalty, C = 1.0, solver lbfgs, max_iter = 1000.
- **LDA:** solver svd, priors estimated from the training data, shared covariance matrix.
""")
