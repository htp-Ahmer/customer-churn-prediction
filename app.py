"""Customer Churn Prediction & Analytics Dashboard."""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    model_input_from_form,
    load_data,
)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
RESULTS_PATH = ROOT / "models" / "model_results.json"

st.set_page_config(
    page_title="ChurnSignal | Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def get_model_bundle() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_results() -> dict:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def style_chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=16, r=16, t=48, b=16),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white"),
    )
    return fig


def format_pct(value: float) -> str:
    return f"{value:.1%}"


def risk_label(probability: float) -> str:
    if probability >= 0.65:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"


def render_overview(df: pd.DataFrame) -> None:
    st.title("Customer churn command center")
    st.caption(
        "A measured view of retention risk across 7,043 historical telecom customers."
    )

    churned = int(df["Churn"].sum())
    churn_rate = df["Churn"].mean()
    metrics = st.columns(4)
    metrics[0].metric("Total customers", f"{len(df):,}")
    metrics[1].metric("Churned customers", f"{churned:,}")
    metrics[2].metric("Churn rate", format_pct(churn_rate))
    metrics[3].metric("Avg. monthly charges", f"${df['MonthlyCharges'].mean():,.2f}")

    st.divider()
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Portfolio at a glance")
        distribution = (
            df.assign(Status=df["Churn"].map({0: "Stayed", 1: "Churned"}))
            .groupby("Status", as_index=False)
            .size()
            .rename(columns={"size": "Customers"})
        )
        fig = px.bar(
            distribution,
            x="Status",
            y="Customers",
            color="Status",
            color_discrete_map={"Stayed": "#1f6f78", "Churned": "#d95d39"},
            text_auto=True,
        )
        st.plotly_chart(style_chart(fig), width="stretch")
    with right:
        st.subheader("Where attention is concentrated")
        contract_summary = (
            df.groupby("Contract", as_index=False)
            .agg(Customers=("Churn", "size"), Churn_rate=("Churn", "mean"))
            .sort_values("Churn_rate", ascending=False)
        )
        contract_summary["Churn rate"] = contract_summary["Churn_rate"].map(format_pct)
        st.dataframe(
            contract_summary[["Contract", "Customers", "Churn rate"]],
            hide_index=True,
            width="stretch",
        )
        st.info(
            "Use Churn Analysis to filter the population and Model Performance to "
            "understand how the risk estimate was measured."
        )


def render_analysis(df: pd.DataFrame) -> None:
    st.title("Churn analysis")
    st.caption("Explore patterns in the historical customer base. Charts update with filters.")

    filters = st.columns(3)
    contract_filter = filters[0].multiselect(
        "Contract",
        sorted(df["Contract"].unique()),
        default=sorted(df["Contract"].unique()),
    )
    internet_filter = filters[1].multiselect(
        "Internet service",
        sorted(df["InternetService"].unique()),
        default=sorted(df["InternetService"].unique()),
    )
    payment_filter = filters[2].multiselect(
        "Payment method",
        sorted(df["PaymentMethod"].unique()),
        default=sorted(df["PaymentMethod"].unique()),
    )
    filtered = df[
        df["Contract"].isin(contract_filter)
        & df["InternetService"].isin(internet_filter)
        & df["PaymentMethod"].isin(payment_filter)
    ].copy()
    if filtered.empty:
        st.warning("No customers match these filters. Select at least one option in each filter.")
        return
    st.caption(f"Showing {len(filtered):,} of {len(df):,} customers.")

    filtered["Churn status"] = filtered["Churn"].map({0: "Stayed", 1: "Churned"})
    filtered["Tenure band"] = pd.cut(
        filtered["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0–12 months", "13–24 months", "25–48 months", "49–72 months"],
    )
    filtered["Monthly charge band"] = pd.cut(
        filtered["MonthlyCharges"],
        bins=[-1, 40, 70, 100, float("inf")],
        labels=["Under $40", "$40–$70", "$70–$100", "Over $100"],
    )

    row_one = st.columns(2)
    with row_one[0]:
        st.subheader("Churn by contract")
        summary = (
            filtered.groupby("Contract", as_index=False)
            .agg(Churn_rate=("Churn", "mean"), Customers=("Churn", "size"))
            .sort_values("Churn_rate", ascending=False)
        )
        fig = px.bar(
            summary,
            x="Contract",
            y="Churn_rate",
            text=summary["Churn_rate"].map(format_pct),
            color="Churn_rate",
            color_continuous_scale=["#b8d8d8", "#d95d39"],
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_chart(fig), width="stretch")
    with row_one[1]:
        st.subheader("Churn by tenure")
        summary = (
            filtered.groupby("Tenure band", observed=False, as_index=False)
            .agg(Churn_rate=("Churn", "mean"), Customers=("Churn", "size"))
        )
        fig = px.bar(
            summary,
            x="Tenure band",
            y="Churn_rate",
            text=summary["Churn_rate"].map(format_pct),
            color="Churn_rate",
            color_continuous_scale=["#b8d8d8", "#d95d39"],
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_chart(fig), width="stretch")

    row_two = st.columns(2)
    with row_two[0]:
        st.subheader("Monthly charges and churn")
        fig = px.histogram(
            filtered,
            x="MonthlyCharges",
            color="Churn status",
            nbins=24,
            barmode="overlay",
            opacity=0.78,
            color_discrete_map={"Stayed": "#1f6f78", "Churned": "#d95d39"},
        )
        st.plotly_chart(style_chart(fig), width="stretch")
    with row_two[1]:
        st.subheader("Churn by payment method")
        summary = (
            filtered.groupby("PaymentMethod", as_index=False)
            .agg(Churn_rate=("Churn", "mean"))
            .sort_values("Churn_rate", ascending=True)
        )
        fig = px.bar(
            summary,
            x="Churn_rate",
            y="PaymentMethod",
            orientation="h",
            text=summary["Churn_rate"].map(format_pct),
            color="Churn_rate",
            color_continuous_scale=["#b8d8d8", "#d95d39"],
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style_chart(fig), width="stretch")

    st.subheader("Churn by internet service")
    summary = (
        filtered.groupby("InternetService", as_index=False)
        .agg(Churn_rate=("Churn", "mean"), Customers=("Churn", "size"))
        .sort_values("Churn_rate", ascending=False)
    )
    fig = px.bar(
        summary,
        x="InternetService",
        y="Churn_rate",
        text=summary["Churn_rate"].map(format_pct),
        color="Churn_rate",
        color_continuous_scale=["#b8d8d8", "#d95d39"],
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style_chart(fig), width="stretch")


def render_model_performance(results: dict) -> None:
    st.title("Model performance")
    st.caption(
        "Both models use the same train/test split and preprocessing pipeline. "
        "The final model is selected by test-set ROC-AUC."
    )
    st.success(
        f"Selected model: {results['selected_model']} · "
        f"ROC-AUC {results['metrics'][results['selected_model']]['roc_auc']:.3f}"
    )

    metric_rows = []
    for name, values in results["metrics"].items():
        metric_rows.append(
            {
                "Model": name,
                "Accuracy": values["accuracy"],
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1-score": values["f1_score"],
                "ROC-AUC": values["roc_auc"],
            }
        )
    table = pd.DataFrame(metric_rows).set_index("Model")
    st.dataframe(
        table.style.format("{:.3f}"),
        width="stretch",
    )

    selected_metrics = results["metrics"][results["selected_model"]]
    chart_col, matrix_col = st.columns(2)
    with chart_col:
        st.subheader("ROC curves")
        fig = go.Figure()
        for name, values in results["metrics"].items():
            fig.add_trace(
                go.Scatter(
                    x=values["roc_curve"]["fpr"],
                    y=values["roc_curve"]["tpr"],
                    mode="lines",
                    name=f"{name} ({values['roc_auc']:.3f})",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
                line=dict(dash="dash", color="#9aa6a8"),
            )
        )
        fig.update_xaxes(title="False positive rate", range=[0, 1])
        fig.update_yaxes(title="True positive rate", range=[0, 1])
        st.plotly_chart(style_chart(fig), width="stretch")
    with matrix_col:
        st.subheader(f"{results['selected_model']} confusion matrix")
        matrix = selected_metrics["confusion_matrix"]
        fig = px.imshow(
            matrix,
            text_auto=True,
            x=["Predicted stay", "Predicted churn"],
            y=["Actual stay", "Actual churn"],
            color_continuous_scale=["#edf4f4", "#1f6f78"],
            labels=dict(x="", y="", color="Customers"),
        )
        st.plotly_chart(style_chart(fig), width="stretch")

    st.info(
        "Recall matters in churn work because missing a customer who is likely to "
        "leave can be costly. ROC-AUC is used for model selection because it evaluates "
        "ranking quality across probability thresholds rather than one arbitrary cutoff."
    )


def render_prediction(bundle: dict) -> None:
    st.title("Churn prediction")
    st.caption(
        "Estimate risk for one customer using the saved "
        f"{bundle['model_name']} pipeline."
    )
    with st.form("prediction_form"):
        st.subheader("Customer profile")
        top = st.columns(4)
        tenure = top[0].number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        monthly = top[1].number_input(
            "Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0
        )
        total = top[2].number_input(
            "Total charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=25.0
        )
        senior = top[3].selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")

        values = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "SeniorCitizen": senior,
        }
        form_cols = st.columns(3)
        option_labels = {
            "gender": ("Gender", ["Female", "Male"]),
            "Partner": ("Partner", ["No", "Yes"]),
            "Dependents": ("Dependents", ["No", "Yes"]),
            "PhoneService": ("Phone service", ["No", "Yes"]),
            "MultipleLines": ("Multiple lines", ["No", "No phone service", "Yes"]),
            "InternetService": ("Internet service", ["DSL", "Fiber optic", "No"]),
            "OnlineSecurity": ("Online security", ["No", "No internet service", "Yes"]),
            "OnlineBackup": ("Online backup", ["No", "No internet service", "Yes"]),
            "DeviceProtection": ("Device protection", ["No", "No internet service", "Yes"]),
            "TechSupport": ("Tech support", ["No", "No internet service", "Yes"]),
            "StreamingTV": ("Streaming TV", ["No", "No internet service", "Yes"]),
            "StreamingMovies": ("Streaming movies", ["No", "No internet service", "Yes"]),
            "Contract": ("Contract", ["Month-to-month", "One year", "Two year"]),
            "PaperlessBilling": ("Paperless billing", ["No", "Yes"]),
            "PaymentMethod": (
                "Payment method",
                [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                ],
            ),
        }
        for index, column in enumerate(CATEGORICAL_FEATURES):
            label, options = option_labels[column]
            values[column] = form_cols[index % 3].selectbox(label, options, key=f"input_{column}")

        submitted = st.form_submit_button("Predict churn", type="primary")

    if submitted:
        model = bundle["model"]
        row = model_input_from_form(values)
        probability = float(model.predict_proba(row)[0, 1])
        prediction = "Likely to churn" if probability >= 0.5 else "Likely to stay"
        risk = risk_label(probability)
        result_cols = st.columns(3)
        result_cols[0].metric("Prediction", prediction)
        result_cols[1].metric("Churn probability", format_pct(probability))
        result_cols[2].metric("Risk category", risk)
        if risk == "High":
            st.error("This profile is in the high-risk band. Consider proactive retention outreach.")
        elif risk == "Medium":
            st.warning("This profile is in the medium-risk band. Monitor engagement and service experience.")
        else:
            st.success("This profile is in the low-risk band based on the model estimate.")
        st.caption(
            "This is a machine-learning estimate from historical patterns, not a certainty "
            "or a decision about an individual customer."
        )


def render_insights(df: pd.DataFrame) -> None:
    st.title("Business insights")
    st.caption("These observations are calculated from the IBM Telco Churn dataset.")
    contract_rates = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    tenure_rates = (
        df.assign(
            TenureBand=pd.cut(
                df["tenure"], [-1, 12, 24, 48, 72],
                labels=["0–12 months", "13–24 months", "25–48 months", "49–72 months"],
            )
        )
        .groupby("TenureBand", observed=False)["Churn"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
    )
    charge_comparison = df.groupby("Churn")["MonthlyCharges"].mean()
    payment_rates = df.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=False)

    insights = [
        (
            "Contract length is a strong separator",
            f"{contract_rates.index[0]} customers have the highest observed churn rate "
            f"({format_pct(contract_rates.iloc[0])}), versus "
            f"{format_pct(contract_rates.iloc[-1])} for {contract_rates.index[-1]} customers.",
        ),
        (
            "Early-tenure customers need attention",
            f"The highest-tenure-risk band is {tenure_rates.index[0]} at "
            f"{format_pct(tenure_rates.iloc[0])}. This supports focusing onboarding and "
            "early-life customer experience work on the first year.",
        ),
        (
            "Churned customers carry higher monthly charges",
            f"Average monthly charges are ${charge_comparison.get(1, 0):.2f} for churned "
            f"customers versus ${charge_comparison.get(0, 0):.2f} for retained customers.",
        ),
        (
            "Payment method patterns are uneven",
            f"{payment_rates.index[0]} shows the highest observed churn rate among payment "
            f"methods ({format_pct(payment_rates.iloc[0])}). This is an association, not proof "
            "that payment method causes churn.",
        ),
    ]
    for title, detail in insights:
        with st.container(border=True):
            st.subheader(title)
            st.write(detail)

    st.warning(
        "The dataset is historical and comes from one telecom context. These insights "
        "should guide questions and experiments, not be treated as causal conclusions."
    )


def main() -> None:
    df = get_data()
    bundle = get_model_bundle()
    results = get_results()
    with st.sidebar:
        st.title("ChurnSignal")
        st.caption("Customer retention analytics")
        st.divider()
        page = st.radio(
            "Navigate",
            ["Overview", "Churn Analysis", "Model Performance", "Churn Prediction", "Business Insights"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("IBM Telco Customer Churn · reproducible seed 42")

    if page == "Overview":
        render_overview(df)
    elif page == "Churn Analysis":
        render_analysis(df)
    elif page == "Model Performance":
        render_model_performance(results)
    elif page == "Churn Prediction":
        render_prediction(bundle)
    else:
        render_insights(df)


if __name__ == "__main__":
    main()