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
    demo_cols = ['id','sex','field','deg','yrdeg', 'startyr', 'rank', 'salary']
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
    fig.update_layout(
    template="plotly_white",       # A minimal, clean template
    width=800,                     # Adjust as desired
    height=400,
    plot_bgcolor="#e5ecf6",        # Light background color
    paper_bgcolor="#e5ecf6",       # Matches the plot background
    title_font_size=20,
    xaxis_title="Sex",
    yaxis_title="Count of Faculty",
    font=dict(size=14),
    margin=dict(l=40, r=40, t=60, b=40)
    )

    fig.update_traces(
        #marker_line_color='white',     # White edge around bars
        marker_line_width=1.5,         # Thicker bar outlines
        #opacity=0.8                    # Slight transparency
    )

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
        pval_interpretation = (
            f"The two-sided p-value ({p_val:.4f}) is less than the significance level of {alpha}. "
            "This suggests that the observed difference in promotion rates between male and female faculty "
            "from Associate to Full Professor is statistically significant. In other words, it is unlikely that "
            "the disparity in promotion outcomes occurred by chance, which may indicate the presence of sex bias "
            "in the promotion process."
        )
    else:
        pval_interpretation = (
            f"The two-sided p-value ({p_val:.4f}) is not below the significance level of {alpha}. "
            "This means that we do not have sufficient statistical evidence to conclude that there is a difference "
            "in promotion rates between male and female faculty. Thus, based on this test alone, we cannot assert "
            "the presence of sex bias in granting promotions from Associate to Full Professor."
        )

    # Check if the confidence interval contains zero
    if ci_lower > 0 or ci_upper < 0:
        ci_interpretation = (
            f"The 95% confidence interval ({ci_lower:.4f}, {ci_upper:.4f}) for the difference in promotion rates does not include zero."
            "This reinforces the conclusion that there is a statistically significant difference between male and female "
            "faculty. This supports the notion that there may be sex bias in the promotion process from Associate to Full Professor."
        )
    else:
        ci_interpretation = (
            f"The 95% confidence interval ({ci_lower:.4f}, {ci_upper:.4f}) for the difference includes zero, suggesting that the true difference in promotion rates "
            "could be zero. This means that the observed disparity in promotion outcomes might not be statistically significant, "
            "and we cannot conclusively claim the presence of sex bias in the promotion process."
        )

    test_results = {
        'z_stat': z_stat,
        'p_val': p_val,
        'p_val_interpretation': pval_interpretation,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'diff': diff,
        'ci_interpretation': ci_interpretation,
        'proportion_men': p_men,
        'proportion_women': p_women
    }
    # Build a table for crosstab results
    crosstab_table = pd.DataFrame({
        "Sex": ["Men (M)", "Women (F)"],
        "Count Promoted": [men_promoted, women_promoted],
        "Total Count": [men_total, women_total],
        "Promotion Proportion": [p_men, p_women]
    })

    return test_results, crosstab_table

    


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
        template="plotly_white",                 # A cleaner, minimal template
        width=800,                               # Figure width in px
        height=500,
        plot_bgcolor="#e5ecf6",        # Light background color
        paper_bgcolor="#e5ecf6",       # Matches the plot background
        title_font_size=20,            
        title="Kaplan-Meier: Probability of Remaining Associate Over Time",
        xaxis_title="Years Since First Associate Rank",
        yaxis_title="Survival Probability (Still Associate)",
        font=dict(size=14),                      # Increase overall font size
        margin=dict(l=40, r=40, t=40, b=40))      # Adjust margins as needed)
        
        fig.update_traces(line=dict(width=3))
    return fig


# --------------------------------------------------------------------
# 5) Prepare Data for Modeling (Logistic Regression)
# --------------------------------------------------------------------
def prepare_data_for_modeling(summary: pd.DataFrame):
    """
    Prepares features (X) and target (y) for logistic regression.
    We'll keep 'sex', 'field', 'deg_type', etc. as is, 
    letting the pipeline handle one-hot encoding for categorical columns.
    """
    data = summary.copy()
    data.rename(columns={'deg':'deg_type'}, inplace=True)
    data['admin_any'] = data['admin_any'].fillna(0)
    # Ensure numeric
    data['yrdeg'] = pd.to_numeric(data['yrdeg'], errors='coerce')
    data['startyr'] = pd.to_numeric(data['startyr'], errors='coerce')
    data['salary'] = pd.to_numeric(data['salary'], errors='coerce')

    feature_cols = [
        'sex',       # Categorical
        'field',     # Categorical
        'deg_type',  # Categorical
        'admin_any', # numeric
        'yrdeg',     # numeric
        'startyr',   # numeric
        'salary'     # numeric
    ]
    X = data[feature_cols].copy()
    y = data['promoted'].astype(int)
    return X, y, data


# --------------------------------------------------------------------
# 6) Build Logistic Regression Pipeline
# --------------------------------------------------------------------

def build_and_run_logreg_model(X: pd.DataFrame, y: pd.Series, selected_features: list):
    """
    Builds and fits a scikit-learn LogisticRegression pipeline with the selected features.
    Returns the pipeline, predictions, and predicted probabilities.
    """
    X_model = X[selected_features].copy()
    cat_cols = [c for c in selected_features if X_model[c].dtype==object]
    num_cols = [c for c in selected_features if X_model[c].dtype!=object]

    transformers = []
    if cat_cols:
        transformers.append(('cat', OneHotEncoder(), cat_cols))
    if num_cols:
        transformers.append(('num', StandardScaler(), num_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('logreg', LogisticRegression(solver='lbfgs', max_iter=200))
    ])
    pipe.fit(X_model, y)
    preds = pipe.predict(X_model)
    probs = pipe.predict_proba(X_model)[:,1]
    return pipe, preds, probs

# --------------------------------------------------------------------
# 7) Plot Logistic Regression Feature Importances
# --------------------------------------------------------------------
def plot_feature_importances_logreg(pipe: Pipeline, X: pd.DataFrame, selected_features: list):
    """
    Plots the logistic regression coefficients (log-odds) with user-friendly labels.
    """
    import plotly.express as px

    logreg = pipe.named_steps['logreg']
    preprocessor = pipe.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out(selected_features)

    coefs = logreg.coef_.flatten()
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coef': coefs
    }).sort_values('coef', ascending=False)

    # Replace underscores or parse for user-friendly text
    # Example: "cat__field_Prof" -> "Field: Prof"
    def nice_label(col):
        # remove prefix like "cat__" or "num__"
        parts = col.split('__')
        if len(parts)>1:
            remainder = parts[1]
        else:
            remainder = parts[0]
        # if remainder looks like "field_Arts" -> "Field: Arts"
        if '_' in remainder:
            base, cat = remainder.split('_', 1)
            return f"{base.capitalize()}: {cat}"
        else:
            return remainder.capitalize()

    coef_df['feature'] = coef_df['feature'].apply(nice_label)

    fig = px.bar(
        coef_df,
        x='coef',
        y='feature',
        orientation='h',
        title="Logistic Regression Coefficients (Log-Odds Scale)"
    )
    fig.update_layout(
        xaxis_title="Coefficient (Log-Odds)",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        plot_bgcolor="#e5ecf6",
        paper_bgcolor="#e5ecf6",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig