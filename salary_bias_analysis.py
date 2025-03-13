import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import statsmodels.api as sm


def prepare_salary_increase_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares a summary DataFrame for salary increase analysis between 1990-1995.

    Steps:
        1. Filter to include only records from 1990 and 1995.
        2. Pivot the data to get 1990 and 1995 salaries in separate columns.
        3. Calculate salary increase and percentage increase.
        4. Merge with demographic info (sex, field, etc.)

    Returns:
        summary (pd.DataFrame) with one row per faculty.
    """
    # Ensure numeric types
    d = df.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["id"] = pd.to_numeric(d["id"], errors="coerce")
    d["salary"] = pd.to_numeric(d["salary"], errors="coerce")

    # Filter to 1990 and 1995 only
    d = d[d["year"].isin([90, 95])]

    # Pivot to get 1990 and 1995 salaries in separate columns
    salary_pivot = d.pivot_table(
        index="id", columns="year", values="salary", aggfunc="first"
    ).reset_index()

    # Rename columns
    salary_pivot.columns.name = None
    salary_pivot.rename(columns={90: "salary_1990", 95: "salary_1995"}, inplace=True)

    # Keep only faculty present in both years
    salary_pivot = salary_pivot.dropna(subset=["salary_1990", "salary_1995"])

    # Calculate salary increase and percentage
    salary_pivot["salary_increase"] = (
        salary_pivot["salary_1995"] - salary_pivot["salary_1990"]
    )
    salary_pivot["pct_increase"] = (
        salary_pivot["salary_increase"] / salary_pivot["salary_1990"]
    ) * 100

    # Merge with demographic info
    demo_cols = ["id", "sex", "field", "deg", "yrdeg", "startyr", "rank"]

    # For rank, get the 1990 value
    rank_1990 = d[d["year"] == 90][["id", "rank"]].copy()
    rank_1990.rename(columns={"rank": "rank_1990"}, inplace=True)

    # Get the 1995 value
    rank_1995 = d[d["year"] == 95][["id", "rank"]].copy()
    rank_1995.rename(columns={"rank": "rank_1995"}, inplace=True)

    # Get constant demographics (using first occurrence)
    demos = df[demo_cols].groupby("id").first().reset_index()

    # Calculate years experience (years since degree by 1990)
    demos["years_experience"] = 90 - demos["yrdeg"]

    # Merge everything together
    summary = pd.merge(salary_pivot, demos, on="id")
    summary = pd.merge(summary, rank_1990, on="id", how="left")
    summary = pd.merge(summary, rank_1995, on="id", how="left")
    summary.dropna(inplace=True)
    return summary


def create_salary_increase_boxplot(summary: pd.DataFrame):
    """
    Creates a Plotly box plot comparing salary increases by sex.
    """
    if summary.empty:
        return None

    fig = px.box(
        summary,
        x="sex",
        y="salary_increase",
        color="sex",
        title="Salary Increase (1990-1995) by Sex",
        labels={"salary_increase": "Salary Increase ($)", "sex": "Sex"},
    )
    fig.update_layout(xaxis_title="Sex", yaxis_title="Salary Increase ($)",plot_bgcolor="#e5ecf6",
        paper_bgcolor="#e5ecf6")
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    return fig


def create_pct_increase_boxplot(summary: pd.DataFrame):
    """
    Creates a Plotly box plot comparing percentage salary increases by sex.
    """
    if summary.empty:
        return None

    fig = px.box(
        summary,
        x="sex",
        y="pct_increase",
        color="sex",
        title="Percentage Salary Increase (1990-1995) by Sex",
        labels={"pct_increase": "Percentage Increase (%)", "sex": "Sex"},
    )
    fig.update_layout(xaxis_title="Sex", yaxis_title="Percentage Increase (%)", plot_bgcolor="#e5ecf6",
        paper_bgcolor="#e5ecf6")

    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    return fig


def welch_ttest_salary_increase(summary: pd.DataFrame, column="salary_increase"):
    """
    Performs Welch's t-test to compare salary increases for men vs. women.
    Returns a dictionary with test statistics and an interpretation.
    """
    if summary.empty or "sex" not in summary.columns:
        return None

    if not {"M", "F"}.issubset(summary["sex"].unique()):
        return None

    men_data = summary[summary["sex"] == "M"][column].dropna()
    women_data = summary[summary["sex"] == "F"][column].dropna()

    if len(men_data) < 2 or len(women_data) < 2:
        return None

    t_stat, p_val = stats.ttest_ind(men_data, women_data, equal_var=False)

    men_mean = men_data.mean()
    women_mean = women_data.mean()
    diff = men_mean - women_mean

    # Calculate confidence interval
    men_var = men_data.var()
    women_var = women_data.var()
    men_n = len(men_data)
    women_n = len(women_data)

    pooled_se = np.sqrt(men_var / men_n + women_var / women_n)
    df = (men_var / men_n + women_var / women_n) ** 2 / (
        (men_var / men_n) ** 2 / (men_n - 1)
        + (women_var / women_n) ** 2 / (women_n - 1)
    )

    t_critical = stats.t.ppf(0.975, df)
    ci_lower = diff - t_critical * pooled_se
    ci_upper = diff + t_critical * pooled_se

    alpha = 0.05
    if p_val < alpha:
        pval_interpretation = f"p-value = {p_val:.4f} < 0.05. We reject H₀; there's evidence of a difference."
    else:
        pval_interpretation = f"p-value = {p_val:.4f} ≥ 0.05. We fail to reject H₀; no strong evidence of a difference."

    return {
        "t_stat": t_stat,
        "p_val": p_val,
        "p_val_interpretation": pval_interpretation,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "diff": diff,
        "mean_men": men_mean,
        "mean_women": women_mean,
        "n_men": men_n,
        "n_women": women_n,
    }


def create_scatter_experience_increase(summary: pd.DataFrame):
    """
    Creates a scatter plot of salary increase vs. years of experience, colored by sex.
    """
    if summary.empty:
        return None

    fig = px.scatter(
        summary,
        x="years_experience",
        y="salary_increase",
        color="sex",
        title="Salary Increase vs. Years of Experience by Sex",
        labels={
            "years_experience": "Years of Experience (by 1990)",
            "salary_increase": "Salary Increase ($)",
            "sex": "Sex",
        },
        trendline="ols",
    )

    fig.update_layout(
        xaxis_title="Years of Experience", yaxis_title="Salary Increase ($)",
        plot_bgcolor="#e5ecf6",
        paper_bgcolor="#e5ecf6"
    )
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    return fig

import plotly.express as px

def create_salary_increase_by_rank(summary: pd.DataFrame):
    """
    Creates a box plot and a violin plot to visualize salary increase distribution by rank.
    """
    if summary.empty:
        return None

    fig = px.box(
        summary,
        x="rank_1990",
        y="salary_increase",
        color="sex",
        title="Salary Increase Distribution by Rank (1990)",
        labels={"rank_1990": "Rank (1990)", "salary_increase": "Salary Increase ($)"},
        boxmode="group",
    )

    fig.update_layout(xaxis_title="Rank (1990)", yaxis_title="Salary Increase ($)"
        ,plot_bgcolor="#e5ecf6",
        paper_bgcolor="#e5ecf6")
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    return fig


import plotly.express as px

def create_salary_increase_vs_starting_year(summary: pd.DataFrame):
    """
    Creates a line plot of average salary increase by starting year, separated by sex.
    """
    if summary.empty:
        return None

    # Group by starting year and sex, then compute mean salary increase
    salary_by_start = summary.groupby(["startyr", "sex"])["salary_increase"].mean().reset_index()

    fig = px.line(
        salary_by_start,
        x="startyr",
        y="salary_increase",
        color="sex",
        markers=True,
        title="Average Salary Increase vs. Starting Year by Sex",
        labels={"startyr": "Starting Year", "salary_increase": "Average Salary Increase ($)", "sex": "Sex"},
    )

    fig.update_layout(
        xaxis_title="Starting Year",
        yaxis_title="Average Salary Increase ($)",
        legend_title="Sex",
        plot_bgcolor="#e5ecf6",
        paper_bgcolor="#e5ecf6"
    )
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    return fig





def create_pct_salary_increase_vs_initial_salary(summary: pd.DataFrame):
    """
    Creates an overlapping histogram with salary_1990 on the x-axis and 
    percentage salary increase on the y-axis, separated by sex.
    """
    if summary.empty:
        return None

    fig = px.histogram(
        summary,
        x="salary_1990",
        y="pct_increase",
        color="sex",
        histfunc="avg",  # Aggregates percentage increase by salary bins
        barmode="overlay",  # Overlapping bars
        opacity=0.9,  # Adjust opacity for better visualization
        title="Histogram: Initial Salary (1990) vs. Percentage Increase",
        labels={"salary_1990": "Initial Salary (1990) ($)", "pct_increase": "Avg Percentage Increase (%)", "sex": "Sex"},
    )

    fig.update_layout(
        xaxis_title="Initial Salary (1990) ($)", 
        yaxis_title="Average Percentage Increase (%)",
        bargap=0.15  # Adjust bar spacing for better separation
        ,plot_bgcolor="#e5ecf6",
        paper_bgcolor="#e5ecf6"
    )
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    return fig



def create_salary_comparison_plot(summary: pd.DataFrame):
    """
    Creates a scatter plot comparing 1990 salary to 1995 salary, colored by sex.
    """
    if summary.empty:
        return None

    fig = px.scatter(
        summary,
        x="salary_1990",
        y="salary_1995",
        color="sex",
        title="1990 Salary vs. 1995 Salary by Sex",
        labels={
            "salary_1990": "1990 Salary ($)",
            "salary_1995": "1995 Salary ($)",
            "sex": "Sex",
        },
        trendline="ols",
    )

    # Add reference line (y=x)
    max_salary = max(summary["salary_1990"].max(), summary["salary_1995"].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_salary],
            y=[0, max_salary],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Equal Salary Line",
        )
    )

    fig.update_layout(xaxis_title="1990 Salary ($)", yaxis_title="1995 Salary ($)",
        plot_bgcolor="#e5ecf6",
        paper_bgcolor="#e5ecf6")
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    return fig


def prepare_data_for_modeling(summary: pd.DataFrame, target="salary_increase"):
    """
    Prepares X and y for a linear regression model.
    Returns X, y, and feature names.
    """
    data = summary.copy()

    # Create binary indicators
    data["is_female"] = np.where(data["sex"] == "F", 1, 0)

    # Select relevant features
    feature_cols = [
        "is_female",
        "field",
        "deg",
        "rank_1990",
        "rank_1995",
        "years_experience",
        "salary_1990",
        "startyr",
    ]
    y = data[target].astype(float)
    X = data[feature_cols].copy()

    return X, y, feature_cols


def build_and_run_salary_model(X: pd.DataFrame, y: pd.Series, selected_features: list):
    """
    Builds and fits a statsmodels OLS regression with the selected features.
    Returns the fitted model with full statistical results.
    """
    X_model = X[selected_features].copy()

    # Identify categorical and numerical columns
    cat_cols = [col for col in selected_features if X_model[col].dtype == object]
    num_cols = [col for col in selected_features if X_model[col].dtype != object]

    # Convert categorical columns into dummy variables (one-hot encoding, drop first level)
    if cat_cols:
        X_model = pd.get_dummies(X_model, columns=cat_cols, drop_first=True)

    # Add intercept manually (since statsmodels does not add it by default)
    X_model = sm.add_constant(X_model)

    # Fit OLS model
    model = sm.OLS(y, X_model).fit()

    return model


def plot_feature_importances(model):
    """
    Plots the feature importances (coefficients) from a fitted statsmodels OLS model.

    Parameters:
    - model: A fitted statsmodels OLS regression model.

    Returns:
    - Plotly figure showing feature importance.
    """
    # Extract coefficients and feature names directly from the model
    coef_df = pd.DataFrame({"feature": model.params.index, "coef": model.params.values})

    # Sort features by absolute importance
    coef_df = coef_df.sort_values("coef", ascending=False)

    # Create a horizontal bar plot
    fig = px.bar(
        coef_df,
        x="coef",
        y="feature",
        orientation="h",
        title="Feature Importance (Linear Regression Coefficients)",
    )

    # Improve layout
    fig.update_layout(
        xaxis_title="Coefficient",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),  # Highest impact at top
        coloraxis_showscale=False,  # Hide color bar for cleaner look
        template="plotly_white",
        plot_bgcolor="#e5ecf6",
        paper_bgcolor="#e5ecf6"
    )
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    return fig

def build_and_run_salary_model_with_interactions(X: pd.DataFrame, y: pd.Series, selected_features: list, interaction_terms: list):
    """
    Builds and fits a statsmodels OLS regression model with interaction terms.

    Parameters:
    - X: DataFrame of features
    - y: Target variable (salary increase)
    - selected_features: List of features to include in the model
    - interaction_terms: List of tuples specifying interaction terms, e.g., [('sex', 'rank_1995'), ('sex', 'admin')]

    Returns:
    - Fitted statsmodels OLS model
    """
    X_model = X[selected_features].copy()

    # Identify categorical and numerical columns
    cat_cols = [col for col in selected_features if X_model[col].dtype == object]
    num_cols = [col for col in selected_features if X_model[col].dtype != object]

    # Convert categorical columns into dummy variables (one-hot encoding, drop first level)
    if cat_cols:
        X_model = pd.get_dummies(X_model, columns=cat_cols, drop_first=True)

    # Create interaction terms
    for feature1, feature2 in interaction_terms:
        if feature1 in X_model.columns and feature2 in X_model.columns:
            interaction_term = f"{feature1}_x_{feature2}"
            X_model[interaction_term] = X_model[feature1] * X_model[feature2]

    # Convert all columns to numeric, forcing errors to NaN
    X_model = X_model.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    y = pd.to_numeric(y, errors='coerce')

    # Drop rows with missing values in either X_model or y
    mask = X_model.notnull().all(axis=1) & y.notnull()
    X_model = X_model[mask]
    y = y[mask]

    # Force explicit conversion to float
    X_model = X_model.astype(float)
    y = y.astype(float)

    # Add intercept manually (since statsmodels does not add it by default)
    X_model = sm.add_constant(X_model)

    # Fit OLS model
    model = sm.OLS(y, X_model).fit()

    return model

def analyze_salary_increase_bias(summary: pd.DataFrame):
    """
    Comprehensive analysis to determine if sex bias existed in salary increases
    between 1990-1995.

    Returns a dictionary with results and conclusions.
    """
    results = {}

    # 1. Summary statistics
    sex_summary = summary.groupby("sex").agg(
        {
            "salary_1990": ["mean", "median", "count"],
            "salary_1995": ["mean", "median", "count"],
            "salary_increase": ["mean", "median"],
            "pct_increase": ["mean", "median"],
        }
    )
    results["summary"] = sex_summary

    # 2. T-test results for absolute increase
    ttest_abs = welch_ttest_salary_increase(summary, "salary_increase")
    results["ttest_absolute"] = ttest_abs

    # 3. T-test results for percentage increase
    ttest_pct = welch_ttest_salary_increase(summary, "pct_increase")
    results["ttest_percentage"] = ttest_pct

    # 4. Determine if there is evidence of sex bias
    if ttest_abs and ttest_pct:
        bias_absolute = ttest_abs["p_val"] < 0.05 and ttest_abs["diff"] > 0
        bias_percentage = ttest_pct["p_val"] < 0.05 and ttest_pct["diff"] > 0

        results["bias_evident_absolute"] = bias_absolute
        results["bias_evident_percentage"] = bias_percentage

        if bias_absolute and bias_percentage:
            conclusion = "Strong evidence of sex bias in salary increases from 1990-1995, with men receiving significantly higher increases both in absolute dollars and percentage terms."
        elif bias_absolute:
            conclusion = "Evidence of sex bias in absolute salary increases from 1990-1995, with men receiving significantly higher dollar increases."
        elif bias_percentage:
            conclusion = "Evidence of sex bias in percentage salary increases from 1990-1995, with men receiving significantly higher percentage increases."
        else:
            conclusion = "No clear statistical evidence of sex bias in salary increases from 1990-1995 based on direct comparisons."
    else:
        conclusion = (
            "Insufficient data to determine if sex bias existed in salary increases."
        )

    results["conclusion"] = conclusion

    return results
