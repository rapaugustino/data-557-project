# promotions_analysis.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Two-proportion z-test
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.api as sm

# Survival analysis
from sksurv.nonparametric import kaplan_meier_estimator

# Sklearn modules
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def prepare_promotion_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares a summary DataFrame for promotion analysis, focusing on faculty who
    reached Associate rank at some point, then checks if they were promoted to Full.

    Steps:
        1. Keep only rows with rank == 'Assoc' or 'Full'.
        2. Find the earliest Associate year (yr_first_assoc) and earliest Full year (yr_first_full).
        3. Merge with sex, field, deg, etc. (assumed constant by id).
        4. Determine if they had admin duties after first becoming Associate (admin_any).
        5. Create a binary 'promoted' indicator if they reached Full by 1995.

    Returns:
        summary (pd.DataFrame) with one row per faculty.
    """
    d = df.copy()

    # Filter to records with rank in ['Assoc', 'Full']
    d = d[d["rank"].isin(["Assoc", "Full"])]

    # Convert columns to numeric where appropriate
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["id"] = pd.to_numeric(d["id"], errors="coerce")

    # Earliest Associate year
    assoc_only = d[d["rank"] == "Assoc"].groupby("id", as_index=False)["year"].min()
    assoc_only.rename(columns={"year": "yr_first_assoc"}, inplace=True)

    # Earliest Full year
    full_only = d[d["rank"] == "Full"].groupby("id", as_index=False)["year"].min()
    full_only.rename(columns={"year": "yr_first_full"}, inplace=True)

    # Merge the two subsets
    summary = pd.merge(assoc_only, full_only, on="id", how="left")

    # Merge with demographic/academic columns (unique at the id level)
    demo_cols = ["id", "sex", "field", "deg", "yrdeg", "startyr", "rank", "salary"]
    demos = df[demo_cols].drop_duplicates("id")
    summary = pd.merge(summary, demos, on="id", how="left")

    # Merge with admin info
    d_admin = df[["id", "year", "admin"]].copy()
    d_admin["year"] = pd.to_numeric(d_admin["year"], errors="coerce")
    summary = pd.merge(summary, d_admin, on="id", how="left")

    # Determine if a faculty member had admin duties after first becoming Associate
    summary["relevant_admin"] = np.where(
        (summary["year"] >= summary["yr_first_assoc"]) & (summary["admin"] == 1), 1, 0
    )
    admin_status = summary.groupby("id")["relevant_admin"].max().reset_index()
    admin_status.rename(columns={"relevant_admin": "admin_any"}, inplace=True)
    summary = pd.merge(
        summary.drop(columns=["year", "admin"]), admin_status, on="id", how="left"
    )

    # Create 'promoted' indicator: 1 if earliest Full <= 95, else 0
    summary["promoted"] = np.where(
        (~summary["yr_first_full"].isna()) & (summary["yr_first_full"] <= 95), 1, 0
    )

    return summary.drop_duplicates("id").reset_index(drop=True)


def create_promotion_bar_chart(summary: pd.DataFrame):
    """
    Creates a Plotly bar chart comparing the count of promoted vs. not promoted by sex.
    """
    if summary.empty:
        return None

    crosstab = summary.groupby(["sex", "promoted"]).size().reset_index(name="count")
    crosstab["Promotion Status"] = crosstab["promoted"].map(
        {0: "Not Promoted", 1: "Promoted"}
    )

    fig = px.bar(
        crosstab,
        x="sex",
        y="count",
        color="Promotion Status",
        barmode="group",
        title="Promotion Counts by Sex",
    )
    fig.update_layout(
        template="plotly_white",
        width=800,
        height=400,
        plot_bgcolor="#e5ecf6",
        paper_bgcolor="#e5ecf6",
        title_font_size=20,
        xaxis_title="Sex",
        yaxis_title="Count of Faculty",
        font=dict(size=14),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_traces(marker_line_width=1.5)

    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    return fig


def two_proportion_z_test(summary: pd.DataFrame):
    """
    Performs a two-proportion z-test to compare promotion rates for men vs. women.

    This test evaluates whether the observed difference in promotion proportions
    (Associate -> Full) between male and female faculty could be due to chance.

    Returns:
        tuple: (test_results, crosstab_table)
               where test_results is a dictionary containing:
                   - z_stat
                   - p_val
                   - p_val_interpretation
                   - ci_lower
                   - ci_upper
                   - diff
                   - ci_interpretation
                   - proportion_men
                   - proportion_women

               and crosstab_table is a DataFrame showing counts & proportions.
    """
    crosstab = pd.crosstab(summary["sex"], summary["promoted"])
    # Need both men (M) and women (F) to proceed
    if not {"M", "F"}.issubset(crosstab.index):
        return None

    men_promoted = crosstab.loc["M", 1]
    men_total = crosstab.loc["M"].sum()
    women_promoted = crosstab.loc["F", 1]
    women_total = crosstab.loc["F"].sum()

    counts = np.array([men_promoted, women_promoted])
    nobs = np.array([men_total, women_total])

    # two-sided two-proportion z-test
    z_stat, p_val = proportions_ztest(counts, nobs)

    # Calculate difference, standard error, and confidence interval
    p_men = men_promoted / men_total
    p_women = women_promoted / women_total
    diff = p_men - p_women

    se_diff = np.sqrt(
        p_men * (1 - p_men) / men_total + p_women * (1 - p_women) / women_total
    )
    z_critical = 1.96  # Approx for 95% CI
    ci_lower = diff - z_critical * se_diff
    ci_upper = diff + z_critical * se_diff

    # Interpretation
    alpha = 0.05
    if p_val < alpha:
        pval_interpretation = (
            f"The two-sided p-value ({p_val:.4f}) is below the significance level of {alpha}. "
            "Thus, the difference in promotion rates between men and women is statistically significant. "
            "This could indicate sex-based disparities in the promotion process."
        )
    else:
        pval_interpretation = (
            f"The two-sided p-value ({p_val:.4f}) is not below {alpha}, so we do not have sufficient "
            "evidence to conclude a difference in promotion rates between men and women. "
            "The observed difference might be due to random chance."
        )

    if ci_lower > 0 or ci_upper < 0:
        ci_interpretation = (
            f"The 95% CI ({ci_lower:.4f}, {ci_upper:.4f}) does not contain zero, reinforcing that the difference "
            "in promotion rates between men and women is statistically significant and suggests a real disparity."
        )
    else:
        ci_interpretation = (
            f"The 95% CI ({ci_lower:.4f}, {ci_upper:.4f}) includes zero, indicating the true difference in promotion rates between men and women could be zero. "
            "We can't rule out that men and women have similar promotion rates."
        )

    test_results = {
        "z_stat": z_stat,
        "p_val": p_val,
        "p_val_interpretation": pval_interpretation,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "diff": diff,
        "ci_interpretation": ci_interpretation,
        "proportion_men": p_men,
        "proportion_women": p_women,
    }

    # Build a table for crosstab results
    crosstab_table = pd.DataFrame(
        {
            "Sex": ["Men (M)", "Women (F)"],
            "Count Promoted": [men_promoted, women_promoted],
            "Total Count": [men_total, women_total],
            "Promotion Proportion": [p_men, p_women],
        }
    )

    return test_results, crosstab_table


def create_survival_analysis(summary: pd.DataFrame):
    """
    Creates Kaplan-Meier survival curves (probability of remaining Associate over time) by sex.
    """
    if summary.empty:
        return None

    km_data = summary.copy()
    km_data["yr_first_assoc"] = pd.to_numeric(km_data["yr_first_assoc"], errors="coerce")
    km_data["yr_first_full"] = pd.to_numeric(km_data["yr_first_full"], errors="coerce")

    km_data["time"] = np.where(
        km_data["promoted"] == 1,
        km_data["yr_first_full"] - km_data["yr_first_assoc"],
        95 - km_data["yr_first_assoc"],
    )
    km_data["event"] = km_data["promoted"]
    km_data["Female"] = np.where(km_data["sex"] == "F", 1, 0)

    # Filter out negative or invalid 'time'
    km_data = km_data[km_data["time"] >= 0]
    if len(km_data) < 2:
        return None

    fig = go.Figure()
    for sex_label, sex_data in km_data.groupby("Female"):
        if len(sex_data) < 2:
            continue

        survival_array = np.array(
            [(bool(e), float(t)) for e, t in zip(sex_data["event"], sex_data["time"])],
            dtype=[("event", "?"), ("time", "<f8")],
        )
        times, survival_prob = kaplan_meier_estimator(
            survival_array["event"], survival_array["time"]
        )
        group_name = "Female" if sex_label == 1 else "Male"
        fig.add_trace(
            go.Scatter(x=times, y=survival_prob, mode="lines", name=group_name)
        )

    fig.update_layout(
        template="plotly_white",
        width=800,
        height=500,
        plot_bgcolor="#e5ecf6",
        paper_bgcolor="#e5ecf6",
        title_font_size=20,
        title="Kaplan-Meier: Probability of Remaining Associate Over Time",
        xaxis_title="Years Since First Associate Rank",
        yaxis_title="Survival Probability (Still Associate)",
        font=dict(size=14),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    fig.update_traces(line=dict(width=3))
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    return fig


def prepare_data_for_modeling(summary: pd.DataFrame):
    """
    Prepares features (X) and target (y) for logistic regression on the 'promoted' outcome.

    We keep:
      - sex (categorical)
      - field (categorical)
      - deg_type (categorical)
      - admin_any (numeric)
      - yrdeg (numeric)
      - startyr (numeric)
      - salary (numeric)

    Then we let a pipeline handle one-hot encoding for categorical columns.
    """
    data = summary.copy()
    data.rename(columns={"deg": "deg_type"}, inplace=True)
    data["admin_any"] = data["admin_any"].fillna(0)

    # Convert numeric fields
    data["yrdeg"] = pd.to_numeric(data["yrdeg"], errors="coerce")
    data["startyr"] = pd.to_numeric(data["startyr"], errors="coerce")
    data["salary"] = pd.to_numeric(data["salary"], errors="coerce")

    feature_cols = [
        "sex",
        "field",
        "deg_type",
        "admin_any",
        "yrdeg",
        "startyr",
        "salary",
    ]
    X = data[feature_cols].copy()
    y = data["promoted"].astype(int)

    return X, y, data


def build_and_run_logreg_model(X: pd.DataFrame, y: pd.Series, selected_features: list):
    """
    Builds and fits a scikit-learn LogisticRegression pipeline with user-selected features.
    Returns:
        pipe (Pipeline)       : The trained pipeline
        preds (np.ndarray)    : Predicted labels (0 or 1)
        probs (np.ndarray)    : Predicted probabilities for the positive class
    """
    cat_cols = [col for col in selected_features if X[col].dtype == object]
    num_cols = [col for col in selected_features if X[col].dtype != object]

    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("logreg", LogisticRegression(solver="lbfgs", max_iter=200)),
        ]
    )

    # Fit the pipeline
    X_model = X[selected_features].copy()
    pipe.fit(X_model, y)

    # Generate predictions
    preds = pipe.predict(X_model)
    probs = pipe.predict_proba(X_model)[:, 1]

    return pipe, preds, probs


def plot_feature_importances_logreg(pipe: Pipeline, X: pd.DataFrame, selected_features: list):
    """
    Creates a bar plot of logistic regression coefficients (log-odds) with user-friendly labels.
    """
    import plotly.express as px

    logreg = pipe.named_steps["logreg"]
    preprocessor = pipe.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out(selected_features)

    coefs = logreg.coef_.flatten()
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs}).sort_values(
        "coef", ascending=False
    )

    def nice_label(col):
        """
        Convert something like "cat__field_Prof" to "Field: Prof".
        """
        parts = col.split("__")
        remainder = parts[1] if len(parts) > 1 else parts[0]
        if "_" in remainder:
            base, cat = remainder.split("_", 1)
            return f"{base.capitalize()}: {cat}"
        else:
            return remainder.capitalize()

    coef_df["feature"] = coef_df["feature"].apply(nice_label)

    fig = px.bar(
        coef_df,
        x="coef",
        y="feature",
        orientation="h",
        title="Logistic Regression Coefficients (Log-Odds)",
    )
    fig.update_layout(
        xaxis_title="Coefficient (Log-Odds)",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        plot_bgcolor="#e5ecf6",
        paper_bgcolor="#e5ecf6",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    return fig