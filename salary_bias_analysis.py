import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

# Sklearn modules
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

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
    d['year'] = pd.to_numeric(d['year'], errors='coerce')
    d['id'] = pd.to_numeric(d['id'], errors='coerce')
    d['salary'] = pd.to_numeric(d['salary'], errors='coerce')
    
    # Filter to 1990 and 1995 only
    d = d[d['year'].isin([90, 95])]
    
    # Pivot to get 1990 and 1995 salaries in separate columns
    salary_pivot = d.pivot_table(
        index='id',
        columns='year',
        values='salary',
        aggfunc='first'
    ).reset_index()
    
    # Rename columns
    salary_pivot.columns.name = None
    salary_pivot.rename(columns={90: 'salary_1990', 95: 'salary_1995'}, inplace=True)
    
    # Keep only faculty present in both years
    salary_pivot = salary_pivot.dropna(subset=['salary_1990', 'salary_1995'])
    
    # Calculate salary increase and percentage
    salary_pivot['salary_increase'] = salary_pivot['salary_1995'] - salary_pivot['salary_1990']
    salary_pivot['pct_increase'] = (salary_pivot['salary_increase'] / salary_pivot['salary_1990']) * 100
    
    # Merge with demographic info
    demo_cols = ['id', 'sex', 'field', 'deg', 'yrdeg', 'startyr', 'rank']
    
    # For rank, get the 1990 value
    rank_1990 = d[d['year'] == 90][['id', 'rank']].copy()
    rank_1990.rename(columns={'rank': 'rank_1990'}, inplace=True)
    
    # Get the 1995 value
    rank_1995 = d[d['year'] == 95][['id', 'rank']].copy()
    rank_1995.rename(columns={'rank': 'rank_1995'}, inplace=True)
    
    # Get constant demographics (using first occurrence)
    demos = df[demo_cols].groupby('id').first().reset_index()
    
    # Calculate years experience (years since degree by 1990)
    demos['years_experience'] = 90 - demos['yrdeg']
    
    # Merge everything together
    summary = pd.merge(salary_pivot, demos, on='id')
    summary = pd.merge(summary, rank_1990, on='id', how='left')
    summary = pd.merge(summary, rank_1995, on='id', how='left')
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
        x='sex',
        y='salary_increase',
        color='sex',
        title='Salary Increase (1990-1995) by Sex',
        labels={'salary_increase': 'Salary Increase ($)', 'sex': 'Sex'}
    )
    fig.update_layout(xaxis_title="Sex", yaxis_title="Salary Increase ($)")
    return fig


def create_pct_increase_boxplot(summary: pd.DataFrame):
    """
    Creates a Plotly box plot comparing percentage salary increases by sex.
    """
    if summary.empty:
        return None
    
    fig = px.box(
        summary,
        x='sex',
        y='pct_increase',
        color='sex',
        title='Percentage Salary Increase (1990-1995) by Sex',
        labels={'pct_increase': 'Percentage Increase (%)', 'sex': 'Sex'}
    )
    fig.update_layout(xaxis_title="Sex", yaxis_title="Percentage Increase (%)")
    return fig


def welch_ttest_salary_increase(summary: pd.DataFrame, column='salary_increase'):
    """
    Performs Welch's t-test to compare salary increases for men vs. women.
    Returns a dictionary with test statistics and an interpretation.
    """
    if summary.empty or 'sex' not in summary.columns:
        return None
    
    if not {'M', 'F'}.issubset(summary['sex'].unique()):
        return None
    
    men_data = summary[summary['sex'] == 'M'][column].dropna()
    women_data = summary[summary['sex'] == 'F'][column].dropna()
    
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
    
    pooled_se = np.sqrt(men_var/men_n + women_var/women_n)
    df = (men_var/men_n + women_var/women_n)**2 / ((men_var/men_n)**2/(men_n-1) + (women_var/women_n)**2/(women_n-1))
    
    t_critical = stats.t.ppf(0.975, df)
    ci_lower = diff - t_critical * pooled_se
    ci_upper = diff + t_critical * pooled_se
    
    alpha = 0.05
    if p_val < alpha:
        pval_interpretation = f"p-value = {p_val:.4f} < 0.05. We reject H₀; there's evidence of a difference."
    else:
        pval_interpretation = f"p-value = {p_val:.4f} ≥ 0.05. We fail to reject H₀; no strong evidence of a difference."
    
    return {
        't_stat': t_stat,
        'p_val': p_val,
        'p_val_interpretation': pval_interpretation,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'diff': diff,
        'mean_men': men_mean,
        'mean_women': women_mean,
        'n_men': men_n,
        'n_women': women_n
    }


import plotly.express as px
import pandas as pd

def create_scatter_experience_increase(summary: pd.DataFrame):
    """
    Creates a scatter plot of salary increase vs. years of experience, colored by sex.
    """
    if summary.empty:
        return None
    
    fig = px.scatter(
        summary,
        x='years_experience',
        y='salary_increase',
        color='sex',
        title='Salary Increase vs. Years of Experience by Sex',
        labels={
            'years_experience': 'Years of Experience (by 1990)',
            'salary_increase': 'Salary Increase ($)',
            'sex': 'Sex'
        },
        trendline='ols'
    )
    

    
    fig.update_layout(
        xaxis_title="Years of Experience",
        yaxis_title="Salary Increase ($)"
    )
  
    return fig



def create_salary_comparison_plot(summary: pd.DataFrame):
    """
    Creates a scatter plot comparing 1990 salary to 1995 salary, colored by sex.
    """
    if summary.empty:
        return None
    
    fig = px.scatter(
        summary,
        x='salary_1990', 
        y='salary_1995',
        color='sex',
        title='1990 Salary vs. 1995 Salary by Sex',
        labels={
            'salary_1990': '1990 Salary ($)', 
            'salary_1995': '1995 Salary ($)',
            'sex': 'Sex'
        },
        trendline='ols'
    )
    
    # Add reference line (y=x)
    max_salary = max(summary['salary_1990'].max(), summary['salary_1995'].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_salary],
            y=[0, max_salary],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Equal Salary Line'
        )
    )
    
    fig.update_layout(
        xaxis_title="1990 Salary ($)",
        yaxis_title="1995 Salary ($)"
    )
    return fig


def prepare_data_for_modeling(summary: pd.DataFrame, target='salary_increase'):
    """
    Prepares X and y for a linear regression model.
    Returns X, y, and feature names.
    """
    data = summary.copy()
    
    # Create binary indicators
    data['is_female'] = np.where(data['sex'] == 'F', 1, 0)
    
    # Select relevant features
    feature_cols = ['is_female', 'field', 'deg', 'rank_1990','rank_1995', 'years_experience', 'salary_1990','startyr']
    y = data[target].astype(float)
    X = data[feature_cols].copy()
    
    return X, y, feature_cols


def build_and_run_salary_model(X: pd.DataFrame, y: pd.Series, selected_features: list):
    """
    Builds and fits a scikit-learn linear regression pipeline with the selected features.
    Returns the pipeline, predictions, and model statistics.
    """
    X_model = X[selected_features].copy()
    
    # Split features into categorical and numerical
    cat_cols = [col for col in selected_features if X_model[col].dtype == object]
    num_cols = [col for col in selected_features if X_model[col].dtype != object and col != 'is_female']
    
    # Special handling for binary sex variable
    binary_cols = ['is_female'] if 'is_female' in selected_features else []
    
    # Create transformers list
    transformers = []
    if cat_cols:
        transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols))
    if num_cols:
        transformers.append(('num', StandardScaler(), num_cols))
    # Don't transform binary variables
    if binary_cols:
        transformers.append(('bin', 'passthrough', binary_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('linreg', LinearRegression())
    ])
    
    pipe.fit(X_model, y)
    preds = pipe.predict(X_model)
    
    # Calculate R-squared
    r2 = pipe.score(X_model, y)

    # Calculate Adjusted R-squared
    n = X_model.shape[0]
    p = len(pipe.named_steps['linreg'].coef_) + 1  # Add 1 for intercept
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(y - preds))

    # Calculate Residual Standard Error
    residuals = y - preds
    rse = np.sqrt(np.sum(residuals**2) / (n - p))

    return pipe, preds, {
        'r2': r2,
        'adjusted_r2': adjusted_r2,
        'mae': mae,
        'rse': rse,
        'n': n,
        'p': p
    }

def plot_feature_importances(pipe: Pipeline, X: pd.DataFrame, selected_features: list):
    """
    Plots the linear regression coefficients as a bar chart using Plotly.
    """
    linreg = pipe.named_steps['linreg']
    preprocessor = pipe.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out(selected_features)
    
    coefs = linreg.coef_.flatten()
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coef': coefs
    }).sort_values('coef', ascending=False)
    
    fig = px.bar(
        coef_df,
        x='coef',
        y='feature',
        orientation='h',
        title="Feature Coefficients"
    )
    
    fig.update_layout(
        xaxis_title="Coefficient",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def analyze_salary_increase_bias(summary: pd.DataFrame):
    """
    Comprehensive analysis to determine if sex bias existed in salary increases
    between 1990-1995.
    
    Returns a dictionary with results and conclusions.
    """
    results = {}
    
    # 1. Summary statistics
    sex_summary = summary.groupby('sex').agg({
        'salary_1990': ['mean', 'median', 'count'],
        'salary_1995': ['mean', 'median', 'count'],
        'salary_increase': ['mean', 'median'],
        'pct_increase': ['mean', 'median']
    })
    results['summary'] = sex_summary
    
    # 2. T-test results for absolute increase
    ttest_abs = welch_ttest_salary_increase(summary, 'salary_increase')
    results['ttest_absolute'] = ttest_abs
    
    # 3. T-test results for percentage increase
    ttest_pct = welch_ttest_salary_increase(summary, 'pct_increase')
    results['ttest_percentage'] = ttest_pct
    
    # 4. Determine if there is evidence of sex bias
    if ttest_abs and ttest_pct:
        bias_absolute = ttest_abs['p_val'] < 0.05 and ttest_abs['diff'] > 0
        bias_percentage = ttest_pct['p_val'] < 0.05 and ttest_pct['diff'] > 0
        
        results['bias_evident_absolute'] = bias_absolute
        results['bias_evident_percentage'] = bias_percentage
        
        if bias_absolute and bias_percentage:
            conclusion = "Strong evidence of sex bias in salary increases from 1990-1995, with men receiving significantly higher increases both in absolute dollars and percentage terms."
        elif bias_absolute:
            conclusion = "Evidence of sex bias in absolute salary increases from 1990-1995, with men receiving significantly higher dollar increases."
        elif bias_percentage:
            conclusion = "Evidence of sex bias in percentage salary increases from 1990-1995, with men receiving significantly higher percentage increases."
        else:
            conclusion = "No clear statistical evidence of sex bias in salary increases from 1990-1995 based on direct comparisons."
    else:
        conclusion = "Insufficient data to determine if sex bias existed in salary increases."
    
    results['conclusion'] = conclusion
    
    return results