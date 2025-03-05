import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf

# scikit-learn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.figure_factory as ff

# scikit-survival
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator

# -----------------------------------------
# 1. Summaries & Basic Tools
# -----------------------------------------
def get_max_rank(group):
    rank_order = {'Assist': 1, 'Assoc': 2, 'Full': 3}
    group = group[group['rank'].isin(rank_order.keys())].copy()
    group['rank_numeric'] = group['rank'].map(rank_order)
    max_rank_numeric = group['rank_numeric'].max()
    for rank_label, numeric_val in rank_order.items():
        if numeric_val == max_rank_numeric:
            return rank_label

def prepare_summary(data, selected_field="All"):
    summary = (
        data.groupby('id')
        .apply(lambda grp: pd.Series({
            'sex': grp['sex'].iloc[0],
            'max_rank': get_max_rank(grp)
        }))
        .reset_index()
    )
    # Keep only those who reached at least Associate rank
    summary = summary[summary['max_rank'].isin(['Assoc', 'Full'])]
    summary['promoted'] = np.where(summary['max_rank'] == 'Full', 1, 0)

    if selected_field != "All":
        valid_ids = data.loc[data['field'] == selected_field, 'id'].unique()
        summary = summary[summary['id'].isin(valid_ids)]

    return summary

def create_promotion_bar_chart(summary_df):
    """
    Enhanced bar chart with a specific color palette and layout adjustments.
    """
    # We define a custom color palette for aesthetic
    color_map = {0: "lightslategray", 1: "#636EFA"}  # e.g., a 2-color scheme
    promo_counts = summary_df.groupby(['sex', 'promoted']).size().reset_index(name='count')
    
    # Convert 'promoted' to string for better labeling
    promo_counts['promoted_str'] = promo_counts['promoted'].replace({0: "Assoc", 1: "Full"})

    fig = px.bar(
        promo_counts,
        x='sex',
        y='count',
        color='promoted_str',
        barmode='group',
        labels={'promoted_str': 'Promotion Status', 'count': 'Count of Faculty'},
        title="Promotion Outcome by Sex",
        color_discrete_map={"Assoc": color_map[0], "Full": color_map[1]}
    )
    fig.update_layout(
        xaxis_title="Sex",
        yaxis_title="Number of Faculty",
        legend_title="Rank",
    )
    return fig


def perform_chi_square(summary_df):
    """
    Return chi2, p, dof, expected, plus a short interpretation
    """
    if summary_df.empty:
        return None, None, None, None, "No data to analyze"
    contingency_table = pd.crosstab(summary_df['sex'], summary_df['promoted'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    # Basic interpretation
    if p < 0.05:
        interpretation = ("A significant chi-square (p<0.05) indicates an association "
                          "between sex and promotion outcome.")
    else:
        interpretation = ("No statistically significant association was found (p>=0.05).")
    return chi2, p, dof, expected, interpretation


# -----------------------------------------
# 2. Logistic Regression (StatsModels)
# -----------------------------------------
def perform_logistic_regression(summary_df):
    """
    Fit a logistic regression model (promoted ~ sex) using statsmodels.
    Return the fitted model, exponentiated odds ratios (OR), and an interpretation text.
    """
    summary_df = summary_df.copy()
    summary_df['sex_cat'] = summary_df['sex'].astype('category')

    if summary_df['sex_cat'].nunique() < 2:
        # If the data is unbalanced or ends up with 1 category, skip
        return None, None, "Insufficient variation in 'sex' for logistic regression."

    logit_model = smf.logit("promoted ~ sex_cat", data=summary_df).fit(disp=False)
    params = logit_model.params
    conf_int = logit_model.conf_int()
    conf_int['OR'] = params
    conf_int.columns = ['2.5%', '97.5%', 'OR']
    odds_ratios = np.exp(conf_int)

    # Basic interpretation logic
    # If sex_cat[T.F] is <1 and p<0.05 => lower odds for female, etc.
    pvals = logit_model.pvalues
    sex_coef = params.get('sex_cat[T.F]', None)
    sex_p = pvals.get('sex_cat[T.F]', None)
    interpretation = []
    if sex_coef is not None and sex_p is not None:
        if sex_p < 0.05:
            if sex_coef < 0:
                interpretation.append("Females have significantly lower odds of promotion (p<0.05).")
            else:
                interpretation.append("Females have significantly higher odds of promotion (p<0.05).")
        else:
            interpretation.append("No statistically significant difference by sex (p>=0.05).")
    else:
        interpretation.append("Sex coefficient not found in the model? Possibly no variation.")
    interpretation_text = " ".join(interpretation)

    return logit_model, odds_ratios, interpretation_text


# -----------------------------------------
# 3. Kaplan-Meier Analysis (scikit-survival)
# -----------------------------------------
def create_survival_analysis(data, summary_df):
    """
    scikit-survival-based KM estimate. 
    If data is unbalanced (e.g. only 1 female in this field), it will still plot, but keep interpretation in mind.
    """
    # Filter to ranks in [Assoc, Full]
    assoc_full = data[data['rank'].isin(['Assoc','Full'])].copy()
    assoc_full['year'] = pd.to_numeric(assoc_full['year'], errors='coerce')

    # earliest Associate
    first_assoc = assoc_full[assoc_full['rank'] == 'Assoc'].groupby('id')['year'].min().rename('year_assoc')
    # earliest Full
    first_full = assoc_full[assoc_full['rank'] == 'Full'].groupby('id')['year'].min().rename('year_full')

    # combine
    summary = pd.DataFrame({'id': assoc_full['id'].unique()}).merge(first_assoc, on='id', how='left')
    summary = summary.merge(first_full, on='id', how='left')

    # exclude those with no Associate data
    summary = summary[~summary['year_assoc'].isna()].copy()
    if summary.empty:
        return go.Figure().add_annotation(text="No data for survival analysis", showarrow=False)

    # Merge 'sex' from summary_df
    # We must ensure the indices match
    summary = summary.merge(
        summary_df[['id','sex','promoted']], on='id', how='inner'
    )

    summary['event'] = np.where(summary['year_full'].notna(), 1, 0)
    summary['time'] = np.where(summary['year_full'].notna(),
                               summary['year_full'] - summary['year_assoc'],
                               1995 - summary['year_assoc'])
    summary = summary[summary['time']>0].copy()
    if summary.empty:
        return go.Figure().add_annotation(text="No valid time intervals after filtering", showarrow=False)

    # Build scikit-survival Surv objects
    times = summary['time'].values
    events = summary['event'].values.astype(bool)

    fig = go.Figure()
    # color palette for sexes
    palette = {"M": "#EF553B", "F": "#00CC96", "Other": "lightslategray"}

    sexes = summary['sex'].unique()
    for s in sexes:
        mask = summary['sex'] == s
        if mask.sum() < 2:
            continue  # skip if only 1 observation
        T, P = kaplan_meier_estimator(events[mask], times[mask])
        color_val = palette.get(s, "lightslategray")
        fig.add_trace(
            go.Scatter(
                x=T, y=P,
                mode='lines',
                name=f"Sex={s}",
                line=dict(color=color_val, width=3)
            )
        )

    fig.update_layout(
        title="Kaplan-Meier: Time to Promotion (Associate -> Full)",
        xaxis_title="Years from First Associate Rank",
        yaxis_title="Probability of Remaining Associate"
    )
    return fig


# -----------------------------------------
# 4. Advanced Sklearn Multi-Feature
# -----------------------------------------
def prepare_data_for_sklearn(data):
    summary = data.groupby('id').apply(
        lambda grp: pd.Series({
            'sex': grp['sex'].iloc[0],
            'field': grp['field'].iloc[0],
            'deg': grp['deg'].iloc[0],
            'admin_any': (grp['admin'] == 1).any(),
            'yrdeg_min': grp['yrdeg'].min(),
            'max_rank': get_max_rank(grp)
        })
    ).reset_index()

    # Keep only Associate or Full
    summary = summary[summary['max_rank'].isin(['Assoc','Full'])]
    summary['promoted'] = np.where(summary['max_rank']=='Full', 1, 0)

    X = summary[['sex','field','deg','admin_any','yrdeg_min']]
    y = summary['promoted'].values
    return X, y, summary

def build_and_run_sklearn_model(X, y, features_to_include):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    X_model = X[features_to_include].copy()
    cat_cols = [c for c in X_model.columns if X_model[c].dtype == 'object' or c in ['sex','field','deg']]
    num_cols = [c for c in X_model.columns if c not in cat_cols]

    transformers = []
    if cat_cols:
        transformers.append(('cat', OneHotEncoder(drop='first'), cat_cols))
    if num_cols:
        transformers.append(('num', StandardScaler(), num_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=1000))
    ])

    pipe.fit(X_model, y)
    preds = pipe.predict(X_model)
    probs = pipe.predict_proba(X_model)[:,1]
    return pipe, preds, probs

def plot_feature_importances_sklearn(pipe, X, features_to_include):
    cat_cols = [c for c in features_to_include if X[c].dtype == 'object' or c in ['sex','field','deg']]
    num_cols = [c for c in features_to_include if c not in cat_cols]
    preprocessor = pipe.named_steps['preprocess']
    clf = pipe.named_steps['clf']

    cat_feature_names = []
    if cat_cols:
        ohe = preprocessor.named_transformers_['cat']
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    numeric_feature_names = num_cols
    all_feature_names = cat_feature_names + numeric_feature_names

    coefs = clf.coef_[0]
    odds_ratios = np.exp(coefs)

    df_imp = pd.DataFrame({
        'Feature': all_feature_names,
        'OddsRatio': odds_ratios
    }).sort_values('OddsRatio', ascending=False)

    # custom color scale
    fig = px.bar(
        df_imp,
        x='OddsRatio',
        y='Feature',
        orientation='h',
        title="Feature Importance (Odds Ratios)",
        color='OddsRatio',
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig

def plot_confusion_matrix_sklearn(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    z_text = [[str(y) for y in x] for x in cm]

    fig = ff.create_annotated_heatmap(
        cm,
        x=["Pred:0", "Pred:1"],
        y=["Actual:0", "Actual:1"],
        annotation_text=z_text,
        colorscale='Blues'
    )
    fig.update_layout(title="Confusion Matrix")
    return fig

def plot_roc_curve_sklearn(y_true, probs):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(
        title=f"ROC Curve (AUC = {roc_auc:.3f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )
    return fig