import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Two-proportion z-test
from statsmodels.stats.proportion import proportions_ztest

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

    # Filter to Associate or Full records
    d = d[d['rank'].isin(['Assoc','Full'])]

    # Ensure numeric types
    d['year'] = pd.to_numeric(d['year'], errors='coerce')
    d['id'] = pd.to_numeric(d['id'], errors='coerce')

    # Earliest Associate year
    assoc_only = d[d['rank'] == 'Assoc'].groupby('id', as_index=False)['year'].min()
    assoc_only.rename(columns={'year': 'yr_first_assoc'}, inplace=True)

    # Earliest Full year
    full_only = d[d['rank'] == 'Full'].groupby('id', as_index=False)['year'].min()
    full_only.rename(columns={'year': 'yr_first_full'}, inplace=True)

    # Merge the two
    summary = pd.merge(assoc_only, full_only, on='id', how='left')

    # Merge with demographics (unique id-level info)
    demo_cols = ['id','sex','field','deg','yrdeg']
    demos = df[demo_cols].drop_duplicates('id')
    summary = pd.merge(summary, demos, on='id', how='left')

    # Merge admin info from the full dataset
    d_admin = df[['id','year','admin']].copy()
    d_admin['year'] = pd.to_numeric(d_admin['year'], errors='coerce')
    summary = pd.merge(summary, d_admin, on='id', how='left')

    # Determine admin status from yr_first_assoc onward
    summary['relevant_admin'] = np.where((summary['year'] >= summary['yr_first_assoc']) &
                                         (summary['admin'] == 1), 1, 0)
    admin_status = summary.groupby('id')['relevant_admin'].max().reset_index()
    admin_status.rename(columns={'relevant_admin': 'admin_any'}, inplace=True)
    summary = pd.merge(summary.drop(columns=['year','admin']), admin_status, on='id', how='left')

    # Create promotion indicator: promoted if first Full <= 95
    summary['promoted'] = np.where((~summary['yr_first_full'].isna()) &
                                   (summary['yr_first_full'] <= 95), 1, 0)

    return summary.drop_duplicates('id').reset_index(drop=True)


def create_promotion_bar_chart(summary: pd.DataFrame):
    """
    Creates a Plotly bar chart comparing the count of promoted vs. not promoted by sex.
    """
    if summary.empty:
        return None

    crosstab = summary.groupby(['sex','promoted']).size().reset_index(name='count')
    crosstab['Promotion Status'] = crosstab['promoted'].map({0: 'Not Promoted', 1: 'Promoted'})

    fig = px.bar(
        crosstab,
        x='sex',
        y='count',
        color='Promotion Status',
        barmode='group',
        title='Promotion Counts by Sex'
    )
    fig.update_layout(xaxis_title="Sex", yaxis_title="Count of Faculty")
    return fig


def welch_two_proportions_test(summary: pd.DataFrame):
    """
    Performs a two-proportion z-test to compare promotion rates for men vs. women.
    Returns a dictionary with test statistics and an interpretation.
    """
    crosstab = pd.crosstab(summary['sex'], summary['promoted'])
    if not {'M', 'F'}.issubset(crosstab.index):
        return None

    men_promoted = crosstab.loc['M', 1]
    men_total = crosstab.loc['M'].sum()
    women_promoted = crosstab.loc['F', 1]
    women_total = crosstab.loc['F'].sum()

    counts = np.array([men_promoted, women_promoted])
    nobs = np.array([men_total, women_total])
    z_stat, p_val = proportions_ztest(counts, nobs)

    p_men = men_promoted / men_total
    p_women = women_promoted / women_total
    diff = p_men - p_women
    se_diff = np.sqrt(p_men*(1-p_men)/men_total + p_women*(1-p_women)/women_total)
    z_critical = 1.96
    ci_lower = diff - z_critical * se_diff
    ci_upper = diff + z_critical * se_diff

    alpha = 0.05
    if p_val < alpha:
        pval_interpretation = f"p-value = {p_val:.4f} < 0.05. We reject H₀; there's evidence of a difference."
    else:
        pval_interpretation = f"p-value = {p_val:.4f} ≥ 0.05. We fail to reject H₀; no strong evidence of a difference."

    return {
        'z_stat': z_stat,
        'p_val': p_val,
        'p_val_interpretation': pval_interpretation,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'diff': diff,
        'proportion_men': p_men,
        'proportion_women': p_women
    }


def create_survival_analysis(summary: pd.DataFrame):
    """
    Creates Kaplan-Meier survival curves (probability of remaining Associate over time)
    by sex.
    """
    if summary.empty:
        return None

    km_data = summary.copy()
    km_data['yr_first_assoc'] = pd.to_numeric(km_data['yr_first_assoc'], errors='coerce')
    km_data['yr_first_full'] = pd.to_numeric(km_data['yr_first_full'], errors='coerce')

    km_data['time'] = np.where(
        km_data['promoted'] == 1,
        km_data['yr_first_full'] - km_data['yr_first_assoc'],
        95 - km_data['yr_first_assoc']
    )
    km_data['event'] = km_data['promoted']
    km_data['Female'] = np.where(km_data['sex'] == 'F', 1, 0)
    km_data = km_data[km_data['time'] >= 0]
    if len(km_data) < 2:
        return None

    fig = go.Figure()
    for sex_label, sex_data in km_data.groupby('Female'):
        if len(sex_data) < 2:
            continue
        survival_array = np.array(
            [(bool(e), float(t)) for e, t in zip(sex_data['event'], sex_data['time'])],
            dtype=[('event', '?'), ('time', '<f8')]
        )
        times, survival_prob = kaplan_meier_estimator(survival_array['event'], survival_array['time'])
        group_name = 'Female' if sex_label == 1 else 'Male'
        fig.add_trace(go.Scatter(
            x=times,
            y=survival_prob,
            mode='lines',
            name=group_name
        ))
    fig.update_layout(
        title="Kaplan-Meier: Probability of Remaining Associate Over Time",
        xaxis_title="Years Since First Associate Rank",
        yaxis_title="Survival Probability (Still Associate)"
    )
    return fig


def prepare_data_for_modeling(summary: pd.DataFrame):
    """
    Prepares X and y for an advanced logistic regression model.
    Returns X, y, and an updated summary DataFrame.
    """
    data = summary.copy()
    data['sex_numeric'] = np.where(data['sex'] == 'F', 1, 0)
    data.rename(columns={'deg': 'deg_type'}, inplace=True)
    data['admin_any'] = data['admin_any'].fillna(0)
    data['yrdeg_min'] = data['yr_first_assoc'] - data['yrdeg']
    feature_cols = ['sex_numeric', 'field', 'deg_type', 'admin_any', 'yrdeg_min']
    y = data['promoted'].astype(int)
    X = data[feature_cols].copy()
    return X, y, data


def build_and_run_sklearn_model(X: pd.DataFrame, y: pd.Series, selected_features: list):
    """
    Builds and fits a scikit-learn logistic regression pipeline with the selected features.
    Returns the pipeline, predictions, and predicted probabilities.
    """
    X_model = X[selected_features].copy()
    cat_cols = [col for col in selected_features if X_model[col].dtype == object]
    num_cols = [col for col in selected_features if X_model[col].dtype != object]
    transformers = []
    if cat_cols:
        transformers.append(('cat', OneHotEncoder(drop='first'), cat_cols))
    if num_cols:
        transformers.append(('num', StandardScaler(), num_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('logreg', LogisticRegression(solver='lbfgs'))
    ])
    pipe.fit(X_model, y)
    preds = pipe.predict(X_model)
    probs = pipe.predict_proba(X_model)[:, 1]
    return pipe, preds, probs


def plot_feature_importances_sklearn(pipe: Pipeline, X: pd.DataFrame, selected_features: list):
    """
    Plots the logistic regression coefficients (log-odds) as a bar chart using Plotly.
    """
    logreg = pipe.named_steps['logreg']
    preprocessor = pipe.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out(selected_features)
    coefs = logreg.coef_.flatten()
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coef': coefs
    }).sort_values('coef', ascending=False)
    fig = px.bar(
        coef_df,
        x='coef',
        y='feature',
        orientation='h',
        title="Feature Coefficients (Log-Odds Scale)"
    )
    fig.update_layout(
        xaxis_title="Coefficient (Log-Odds)",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed")
    )
    return fig