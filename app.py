import streamlit as st
from data_loader import load_data
import promotions_analysis as pa
import pandas as pd

# --------------------------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Faculty Salary Analysis Project",
    layout="wide"
)

# --------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------
data = load_data()

# --------------------------------------------------------------------
# Create Tabs for the App
# --------------------------------------------------------------------
tabs = st.tabs([
    "Introduction/Background", 
    "Question 1: 1995 Sex Bias", 
    "Question 2: Starting Salaries", 
    "Question 3: Salary Increases (1990-1995)", 
    "Question 4: Promotions", 
    "Summary of Findings"
])

# --------------------------------------------------------------------
# Tab 1: Introduction/Background
# --------------------------------------------------------------------
with tabs[0]:
    st.header("Introduction and Background")
    st.markdown("""
    **Project Overview:**  
    This project analyzes faculty salary data from a US university in 1995 to investigate 
    potential differences between male and female faculty.

    **Background:**  
    - Faculty salaries are influenced by multiple factors (degree, experience, field, admin roles, etc.).
    - Historical studies suggest potential sex biases in academia, leading to pay and promotion disparities.

    **Dataset Description:**  
    - The dataset (salary.txt) includes columns:  
      `case, id, sex, deg, yrdeg, field, startyr, year, rank, admin, salary`
    - Each row corresponds to a faculty member in a specific year.
    """)
    st.info("You can add more context, references, or motivations here if desired.")

# --------------------------------------------------------------------
# Tab 2: Question 1
# --------------------------------------------------------------------
with tabs[1]:
    st.header("Question 1: Does Sex Bias Exist at the University in 1995?")
    st.markdown("""
    **Objective:**  
    Determine if there's evidence of salary bias by sex in 1995.

    **Planned Analysis:**  
    - Filter data to 1995
    - Compare male vs. female salary distributions (descriptive stats, boxplots, histograms)
    - T-test, ANOVA, or regression adjusting for confounders
    """)
    st.warning("Placeholder: Implement the 1995 salary analysis here.")

# --------------------------------------------------------------------
# Tab 3: Question 2
# --------------------------------------------------------------------
with tabs[2]:
    st.header("Question 2: Has Sex Bias Existed in Starting Salaries?")
    st.markdown("""
    **Objective:**  
    Investigate whether male and female faculty start at different salary levels.

    **Planned Analysis:**  
    - Identify earliest year for each faculty
    - Compare starting salaries by sex
    - Possibly run t-tests, regression
    """)
    st.warning("Placeholder: Implement starting salary analysis here.")

# --------------------------------------------------------------------
# Tab 4: Question 3
# --------------------------------------------------------------------
with tabs[3]:
    st.header("Question 3: Has Sex Bias Existed in Salary Increases (1990-1995)?")
    st.markdown("""
    **Objective:**  
    Explore if male/female salary growth differs from 1990 to 1995.

    **Planned Analysis:**  
    - Calculate yearly salary changes
    - Visualize trends (line charts)
    - Possibly use repeated-measures or mixed effects models
    """)
    st.warning("Placeholder: Salary growth analysis here.")

# --------------------------------------------------------------------
# Tab 5: Question 4
# --------------------------------------------------------------------
with tabs[4]:
    st.header("Question 4: Has Sex Bias Existed in Granting Promotions from Associate to Full Professor?")
    st.markdown("""
    **Objective:**  
    Determine if there's a disparity in promotion rates from Associate to Full by sex.
    """)

    # Field filter
    st.subheader("Field Filter")
    field_options = sorted(data['field'].dropna().unique().tolist())
    selected_field = st.selectbox("Select Field", ["All"] + field_options)

    # Prepare summary
    summary_filtered = pa.prepare_summary(data, selected_field)

    # Defensive check
    if summary_filtered.empty:
        st.warning("No records found for this field. Please select a different field or 'All'.")
        st.stop()  # End this tab's code early
    else:
        # 1) Promotion Outcome by Sex (Bar Chart)
        st.subheader("Promotion Outcome by Sex")
        bar_fig = pa.create_promotion_bar_chart(summary_filtered)
        st.plotly_chart(bar_fig, use_container_width=True)


        # 2) Chi-Square
        st.subheader("Chi-Square Test (Sex vs. Promotion)")
        chi2, p_val, dof, expected, chi_interp = pa.perform_chi_square(summary_filtered)
        if chi2 is None:
            st.write("No valid data for Chi-square test.")
        else:
            st.write(f"**Chi2 Statistic**: {chi2:.2f}")
            st.write(f"**p-value**: {p_val:.4f}")
            st.write(f"**Degrees of Freedom**: {dof}")
            st.info(chi_interp)

        # 3) Logistic Regression (Statsmodels)
        st.subheader("Logistic Regression (StatsModels) - Sex -> Promotion")
        logit_model, odds_ratios, logit_interpretation = pa.perform_logistic_regression(summary_filtered)
        if logit_model is None:
            st.warning(logit_interpretation)
        else:
            # Display results in a table
            st.markdown("**Model Summary**")
            st.text(logit_model.summary())

            # Convert odds_ratios to a smaller table
            st.markdown("**Odds Ratios (95% CI)**")
            st.dataframe(odds_ratios.style.format("{:.3f}"))

            st.markdown("**Interpretation**")
            st.info(logit_interpretation)

        # 4) Kaplan-Meier Survival with scikit-survival
        st.subheader("Kaplan-Meier Survival: Time to Promotion")
        surv_fig = pa.create_survival_analysis(data, summary_filtered)
        st.plotly_chart(surv_fig, use_container_width=True)
        st.markdown("""
        - Each curve shows the probability of *remaining* at Associate rank over time. 
        - If one sex's curve falls faster, that group generally gets promoted sooner.
        """)

        # 5) Advanced Sklearn Model
        st.markdown("---")
        st.header("Advanced Sklearn Modeling: Multiple Features")
        st.markdown("""
        **Goal**: Include additional variables (e.g., field, degree type, admin duties) 
        to see how they affect promotion likelihood.  
        """)

        X, y, summary_all = pa.prepare_data_for_sklearn(data)
        possible_features = ["sex","field","deg","admin_any","yrdeg_min"]

        selected_features = st.multiselect(
            "Select Features to Include in Model",
            options=possible_features,
            default=["sex"]
        )

        if len(selected_features) == 0:
            st.warning("Please select at least one feature for the sklearn logistic regression.")
        else:
            pipe, preds, probs = pa.build_and_run_sklearn_model(X, y, selected_features)
            accuracy = (preds == y).mean()

            st.write(f"**Model Accuracy (In-sample)**: {accuracy:.3f}")
            st.markdown("""
            *Note*: If data are unbalanced (few 1's or 0's), 
            accuracy might be misleading. Consider other metrics (precision, recall, etc.).
            """)

            st.subheader("Feature Importance (Odds Ratios)")
            imp_fig = pa.plot_feature_importances_sklearn(pipe, X, selected_features)
            st.plotly_chart(imp_fig, use_container_width=True)

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm_fig = pa.plot_confusion_matrix_sklearn(y, preds)
            st.plotly_chart(cm_fig, use_container_width=True)

            # ROC Curve
            st.subheader("ROC Curve")
            roc_fig = pa.plot_roc_curve_sklearn(y, probs)
            st.plotly_chart(roc_fig, use_container_width=True)

            st.markdown("""
            - **Odds Ratios** above 1 indicate a higher likelihood of promotion as that feature increases (or if that category is present).
            - Consider potential interactions not included here.
            """)

# --------------------------------------------------------------------
# Tab 6: Summary of Findings
# --------------------------------------------------------------------
with tabs[5]:
    st.header("Summary of Findings")
    st.markdown("""
    **Overall Conclusions:**  
    - Summarize key takeaways from each question.
    - Discuss statistical significance and effect sizes.

    **Next Steps & Limitations:**  
    - Mention potential confounders not accounted for.
    - Suggest policy implications or further research.
    """)
    st.info("Finalize your project with references, acknowledgments, or deeper analysis as desired.")