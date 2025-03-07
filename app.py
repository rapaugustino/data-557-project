import streamlit as st
import pandas as pd
import promotions_analysis as pa

# --------------------------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Faculty Salary Analysis Project",
    layout="wide"
)

# --------------------------------------------------------------------
# Persistent Tab Selector in the Sidebar
# --------------------------------------------------------------------
# Define your tab options
tab_options = [
    "Introduction/Background", 
    "Question 1: 1995 Sex Bias", 
    "Question 2: Starting Salaries", 
    "Question 3: Salary Increases (1990-1995)", 
    "Question 4: Promotions", 
    "Summary of Findings"
]

# Initialize the selected tab in session state if it doesn't exist
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = tab_options[0]

# Create a persistent radio button in the sidebar to select a tab
selected_tab = st.sidebar.radio("Select Tab", tab_options, index=tab_options.index(st.session_state.selected_tab))
st.session_state.selected_tab = selected_tab

# --------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------
@st.cache_data
def load_data(file_path="salary.txt"):
    """
    Loads the salary dataset from a text file.
    Expects columns:
    ['case','id','sex','deg','yrdeg','field','startyr','year','rank','admin','salary'].
    """
    df = pd.read_csv(
        file_path,
        sep=r'[\t\s]+',
        header=0,
        engine='python'
    )
    return df

data = load_data()

# --------------------------------------------------------------------
# Render Content Based on the Selected Tab
# --------------------------------------------------------------------
if selected_tab == "Introduction/Background":
    st.header("Introduction and Background")
    st.markdown("""
    **Project Overview:**  
    This project analyzes faculty salary data from a US university in 1995 to investigate 
    potential differences between male and female faculty, focusing on multiple questions.

    **Background & Assumptions:**  
    - Faculty data are assumed independent, though real-world confounders (e.g., teaching evaluations, research productivity) are not fully captured.
    - We focus on group-level differences in promotions.
    - We interpret differences in promotion rates or time-to-promotion as potential sex bias, though causation cannot be established.

    **Dataset Description:**  
    - The dataset includes:  
      `case, id, sex, deg, yrdeg, field, startyr, year, rank, admin, salary`
    - Each row corresponds to a faculty member in a given year (1976-1995).
    """)
    
elif selected_tab == "Question 1: 1995 Sex Bias":
    st.header("Question 1: Does Sex Bias Exist at the University in 1995?")
    st.warning("Placeholder: Implement the 1995 salary analysis here.")

elif selected_tab == "Question 2: Starting Salaries":
    st.header("Question 2: Has Sex Bias Existed in Starting Salaries?")
    st.warning("Placeholder: Implement starting salary analysis here.")

elif selected_tab == "Question 3: Salary Increases (1990-1995)":
    st.header("Question 3: Has Sex Bias Existed in Salary Increases (1990-1995)?")
    st.warning("Placeholder: Salary growth analysis here.")

elif selected_tab == "Question 4: Promotions":
    st.header("Question 4: Has Sex Bias Existed in Granting Promotions from Associate to Full Professor?")
    st.markdown("""
    **Objective:**  
    Determine if there's a disparity in promotion rates (and time-to-promotion) from Associate to Full by sex.
    """)

    # --- Section A: Data Preparation ---
    st.subheader("A) Data Preparation")
    summary = pa.prepare_promotion_data(data)
    if summary.empty:
        st.warning("No eligible records found. Check your dataset.")
        st.stop()

    field_list = ["All"] + sorted(summary['field'].dropna().unique().tolist())
    selected_field = st.selectbox("Field Filter", field_list, index=0)
    if selected_field != "All":
        summary_field = summary[summary['field'] == selected_field].copy()
    else:
        summary_field = summary.copy()
    if summary_field.empty:
        st.warning("No records after applying field filter.")
        st.stop()

    # --- Section B: Descriptive Bar Chart ---
    st.subheader("B) Promotion Counts by Sex")
    bar_fig = pa.create_promotion_bar_chart(summary_field)
    if bar_fig is None:
        st.warning("Not enough data for bar chart.")
    else:
        st.plotly_chart(bar_fig, use_container_width=True)

    # --- Section C: Two-Proportion Z-Test ---
    st.subheader("C) Two-Proportion Z-Test for Promotion Rates")
    test_result = pa.welch_two_proportions_test(summary_field)
    if test_result is None:
        st.info("Not enough data for a two-proportion z-test (need both M and F present).")
    else:
        st.write(f"**Z-Statistic**: {test_result['z_stat']:.4f}")
        st.write(f"**p-value**: {test_result['p_val']:.4f}")
        st.write(f"**Men's Promotion Proportion**: {test_result['proportion_men']:.3f}")
        st.write(f"**Women's Promotion Proportion**: {test_result['proportion_women']:.3f}")
        st.write(f"**Difference (M - F)**: {test_result['diff']:.3f}")
        st.write(f"**95% CI**: ({test_result['ci_lower']:.3f}, {test_result['ci_upper']:.3f})")
        st.info(test_result['p_val_interpretation'])
        st.markdown("""
        - **Null Hypothesis** H₀: Promotion rates for men and women are equal.
        - **Alternative Hypothesis** H₁: Promotion rates differ between men and women.
        """)

    # --- Section D: Survival Analysis ---
    st.subheader("D) Kaplan-Meier Survival Analysis (Time-to-Full)")
    surv_fig = pa.create_survival_analysis(summary_field)
    if surv_fig is None:
        st.warning("Not enough data or only one sex present for survival analysis.")
    else:
        st.plotly_chart(surv_fig, use_container_width=True)
        st.markdown("""
        **Interpretation:**
        - Each curve shows the probability of remaining at Associate rank over time.
        - A faster decline indicates earlier promotion.
        """)

    # --- Section E: Advanced Logistic Regression with Coefficient Feature Importance ---
    st.subheader("E) Advanced Logistic Regression with Coefficient Feature Importance")
    X_all, y_all, summary_all = pa.prepare_data_for_modeling(summary_field)
    if X_all.empty or y_all.nunique() < 2:
        st.warning("Insufficient variation in data for modeling.")
        st.stop()
    st.markdown("**Select Model Features:**")
    possible_features = ["sex_numeric", "field", "deg_type", "admin_any", "yrdeg_min"]
    default_feats = possible_features.copy()
    selected_features = st.multiselect("Features to Include:", options=possible_features, default=default_feats)
    if len(selected_features) == 0:
        st.warning("Please select at least one feature.")
        st.stop()
    pipe, preds, probs = pa.build_and_run_sklearn_model(X_all, y_all, selected_features)
    accuracy = (preds == y_all).mean()
    st.write(f"**Training Accuracy:** {accuracy:.3f}")
    st.markdown("""
    *Note: Accuracy may be misleading if the classes are imbalanced. Consider additional metrics for evaluation.*
    """)
    st.markdown("**Log-Odds Feature Coefficients:**")
    coef_fig = pa.plot_feature_importances_sklearn(pipe, X_all, selected_features)
    st.plotly_chart(coef_fig, use_container_width=True)
    st.info("""
    **Assumptions:**
    - Logistic regression is appropriate for modeling binary promotion outcomes.
    - Faculty records are treated as independent observations.
    - The log-odds scale of coefficients helps interpret how features affect promotion probability.
    """)

elif selected_tab == "Summary of Findings":
    st.header("Summary of Findings")
    st.markdown("""
    **Overall Conclusions:**  
    - Summarize key findings from the analysis (e.g., differences in promotion rates, time-to-promotion, and feature effects).
    
    **Limitations & Future Directions:**  
    - Consider unmeasured confounders (e.g., productivity, teaching quality).
    - Future analyses may incorporate additional data (e.g., departmental context).
    """)