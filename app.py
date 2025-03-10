import streamlit as st
import pandas as pd
import promotions_analysis as pa  # We'll assume you have the relevant logistic regression code in promotions_analysis.py

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
    "Introduction",
    "1995 Sex Bias",
    "Starting Salaries",
    "Salary Increases (1990-1995)",
    "Promotions (Associate to Full)",
    "Summary of Findings"
]

# Initialize the selected tab in session state if it doesn't exist
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = tab_options[0]

# Create a persistent radio button in the sidebar to select a tab
selected_tab = st.sidebar.radio(
    "Select Analysis", tab_options, index=tab_options.index(st.session_state.selected_tab))
#st.session_state.selected_tab = selected_tab

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
if selected_tab == "Introduction":
    st.header("Introduction and Background")
    st.markdown("""
        **Project Overview:**  
        In this team project, we analyze faculty salary data from a U.S. university to explore whether there are differences in average salary and career outcomes between men and women. The overarching goal is to determine whether sex bias exists and to describe the magnitude and nature of its effect.

        **Background:**  
        Despite legal protections against discrimination, studies have consistently found differences in salaries and promotion outcomes across genders in academia. These disparities may arise from various factors such as differences in experience, educational attainment, field of study, administrative roles, and productivity. By examining the data at a group level, we aim to uncover potential sex biases that could explain these differences.

        **Questions of Interest:**  
        The project addresses several key questions:
        1. Does sex bias exist in the university in 1995?
        2. Are starting salaries different between male and female faculty?
        3. Have salary increases between 1990 and 1995 been influenced by sex?
        4. Has sex bias existed in granting promotions from Associate to Full Professor?

        **Methodology:**  
        We use a combination of statistical analysis, Kaplan-Meier survival analysis, and logistic regression to address these questions.

        **Dataset Description:**  
        - **Source:** Faculty salary data from a U.S. university in 1995 (excluding medical school faculty).  
        - **Timeframe:** Data spans from 1976 to 1995, capturing monthly salary records and additional demographic and professional information.  
            - **Variables Include:**  
            - **case, id:** Unique identifiers.  
            - **sex:** 'M' (male) or 'F' (female).  
            - **deg:** Highest degree attained (PhD, Prof, or Other).  
            - **yrdeg:** Year of highest degree attainment.  
            - **field:** Faculty field (Arts, Prof, or Other).  
            - **startyr:** Year in which the faculty member was hired.  
            - **year:** Record year.  
            - **rank:** Academic rank (Assistant, Associate, or Full).  
            - **admin:** Indicator of administrative duties (1 = yes, 0 = no).  
            - **salary:** Monthly salary in dollars.

        **Acknowledgements:**  
        - The dataset and research questions for this project were provided by Professor Scott Emerson.
        - We also extend our sincere gratitude to our course instructor, Professor Katie Wilson, for their guidance and support throughout this project. 
        - Team members: Kyle Cullen Bretherton, Aravindh Manavalan, Richard Pallangyo, Akshay Ravi, and Vijay Balaji S
        """)

elif selected_tab == "1995 Sex Bias":
    st.header("Question 1: Does Sex Bias Exist at the University in 1995?")
    st.warning("Placeholder: Implement the 1995 salary analysis here.")

elif selected_tab == "Starting Salaries":
    st.header("Question 2: Has Sex Bias Existed in Starting Salaries?")
    st.warning("Placeholder: Implement starting salary analysis here.")

elif selected_tab == "Salary Increases (1990-1995)":
    st.header("Question 3: Has Sex Bias Existed in Salary Increases (1990-1995)?")
    st.warning("Placeholder: Salary growth analysis here.")

elif selected_tab == "Promotions (Associate to Full)":
    st.header(
        """Question 4: Sex Bias in Promotions from Associate to Full Professor"""
    )
    st.markdown(""" 
   **Objective:** Determine if there is a disparity in promotion rates—and time-to-promotion—by sex.
    """)
    st.markdown("""---""")

    st.header("""
    Comparing Promotion Rates by Sex: Bar Chart & Statistical Test"""
    )
    

    # --- Section A: Data Preparation ---
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

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            - **Null Hypothesis (H₀)**: The promotion rate from Associate to Full Professor is the same for men and women.
            """)
        # --- Section B: Descriptive Bar Chart ---
        st.subheader("Promotion Counts by Sex", divider="violet")
        bar_fig = pa.create_promotion_bar_chart(summary_field)
        if bar_fig is None:
            st.warning("Not enough data for bar chart.")
        else:
            st.plotly_chart(bar_fig, use_container_width=True)

        cross_tab = pa.welch_two_proportions_test(summary_field)[1]
        st.dataframe(cross_tab, hide_index=True, use_container_width=True)

    with col2:
        # --- Section C: Two-Proportion Z-Test ---
        st.markdown("""
            - **Alternative Hypothesis (H₁)**: The promotion rate from Associate to Full Professor differs between men and women.
            """)
        st.subheader("Two-Proportion Z-Test for Promotion Rates",
                     divider="violet")
        test_result = pa.welch_two_proportions_test(summary_field)[0]
        if test_result is None:
            st.info(
                "Not enough data for a two-proportion z-test (need both M and F present).")
        else:
            result = {
                "z_stat": test_result['z_stat'],
                "p_val": test_result['p_val'],
                "proportion_men": test_result['proportion_men'],
                "proportion_women": test_result['proportion_women'],
                "diff": test_result['diff'],
                "ci_lower": test_result['ci_lower'],
                "ci_upper": test_result['ci_upper'],
            }

            results_df = pd.DataFrame({"Statistic": [
                "Z-Statistic",
                "P-value",
                "Men's Promotion Proportion",
                "Women's Promotion Proportion",
                "Difference (M - F)",
                "95% CI"
            ],
                "Value":[
                f"{result['z_stat']:.5f}",
                f"{result['p_val']:.5f}",
                f"{result['proportion_men']:.5f}", 
                f"{result['proportion_women']:.5f}",
                f"{result['diff']:.5f}",
                f"({result['ci_lower']:.5f}, {result['ci_upper']:.5f})"
            ]})

            st.dataframe(results_df, hide_index=True, use_container_width=True)
            st.info(test_result['p_val_interpretation'])
            st.info(test_result['ci_interpretation'])
            

    st.markdown("""---""")
    # --- Section D: Survival Analysis ---
    st.subheader("Kaplan-Meier Analysis of Time-to-Promotion between Men and Women",
                 divider="violet")
    surv_fig = pa.create_survival_analysis(summary_field)
    if surv_fig is None:
        st.warning(
            "Not enough data or only one sex present for survival analysis.")
    else:
        st.plotly_chart(surv_fig, use_container_width=True)
        st.info("""
        **Interpretation:**

        - The Kaplan-Meier curves above illustrate how long faculty members typically remain at the Associate rank before promotion, separated by sex (men vs. women).
        - A curve that declines more rapidly means that faculty from that group are generally promoted sooner.
        - If one sex consistently shows a faster decline, it suggests that faculty of that sex are promoted to Full Professor more quickly than faculty of the other sex.
        - Comparing these curves helps us identify whether there is a potential sex bias in the timing of promotions, such as if men are promoted more rapidly than women.
        """)
    st.markdown("""---""")

    # ----------------------------------------------
    # 3) Logistic Regression
    # ----------------------------------------------
    st.subheader("Logistic Regression with Feature Importance", divider="violet")
    X_all, y_all, summary_all = pa.prepare_data_for_modeling(summary_field)
    if X_all.empty or y_all.nunique() < 2:
        st.warning("Insufficient variation in data for modeling.")
        st.stop()

    st.markdown("**Select Model Features:**")
    possible_features = [
        "sex",         # Categorical
        "field",       # Categorical
        "deg_type",    # Categorical
        "admin_any",   # Numeric (0/1)
        "yrdeg",       # Numeric
        "startyr",     # Numeric
        "salary"       # Numeric
    ]
    selected_features = st.multiselect(
        "Features to Include:", 
        options=possible_features, 
        default=possible_features
    )
    if len(selected_features) == 0:
        st.warning("Please select at least one feature.")
        st.stop()

    # Build and Run Logistic Regression Pipeline
    pipe, preds, probs = pa.build_and_run_logreg_model(X_all, y_all, selected_features)
    accuracy = (preds == y_all).mean()
    st.write(f"**Logistic Regression (Training Accuracy):** {accuracy:.3f}")

    # Plot coefficients
    coef_fig = pa.plot_feature_importances_logreg(pipe, X_all, selected_features)
    st.plotly_chart(coef_fig, use_container_width=True)
    st.info("""
        **Interpretation:**

        - Each **coefficient** in our logistic regression model indicates how strongly a given feature (or category) affects the likelihood of promotion from Associate to Full Professor, *holding other variables constant*.
        - A **positive coefficient** implies that faculty members with that characteristic—or those for whom the variable is higher—have *increased* odds of being promoted. For example, if we see **Sex: M** in the feature list with a **positive** coefficient, it suggests that male faculty are more likely to be promoted compared to the baseline category (often female) under similar conditions.
        - Conversely, a **negative coefficient** means the feature *lowers* the likelihood of promotion. For instance, if **Sex: F** appears with a negative coefficient, it indicates female faculty experience *reduced* odds of being promoted relative to men, assuming the same levels of all other variables (salary, field, degree type, etc.).
        - The **magnitude** of a coefficient reflects how substantial its impact is. Larger absolute values suggest the feature exerts a stronger influence (positive or negative) on promotion chances.
        - Critically, these coefficients help us assess potential **sex bias** in promotions. For instance, being female is associated with a **negative coefficient** in our model, while being male is associated with a **positive coefficient**. This observation points toward a disparity favoring men.
        """)

    st.markdown("""
    ---
    ### Analysis Summary

    **Goal:**  
    Determine whether there is a disparity in promotion outcomes (both promotion rate and time-to-promotion) between male and female faculty, using data from 1976 to 1995.

    ---

    **1. Data Preparation**  
    - We focused on faculty who reached the Associate or Full rank at some point, then marked whether they were promoted to Full by 1995.
    - Extracted columns such as **sex**, **field**, **degree type**, **salary**, etc., relevant to promotion analysis.

    ---

    **2. Bar Chart & Two-Proportion Z-Test**  
    - A bar chart compared *Promoted* vs. *Not Promoted* counts by sex.
    - We subdivided the test by **faculty field** (e.g., Arts, Prof, Other) to see if the difference in promotion rates between men and women held within each field.

    **Hypotheses**  
    - **Null Hypothesis (H₀):** Men and women have the **same** promotion rate from Associate to Full.  
    - **Alternative Hypothesis (H₁):** Men and women have **different** promotion rates from Associate to Full.

    **Key Finding:**  
    - Overall, men were statistically significantly more likely to be promoted than women, except in the **Arts** field, where the data did not show a significant difference.

    ---

    **3. Kaplan-Meier (Time-to-Promotion) Analysis**  
    - Kaplan-Meier survival curves displayed *how quickly* men vs. women become Full Professors over time.
    - A steeper decline means **faster promotions** in that group.

    **Field-Specific Observations**  
    - **Overall:** Men often leave Associate rank faster in the first few years.  
    - **Arts:** Men get promoted sooner initially; after a few years, female Arts faculty catch up or surpass men.  
    - **Other Fields:** Men initially promoted quicker, but women’s promotion rates increase later.  
    - **Professional Fields (Prof):** Women are promoted faster in the earliest years, but men’s promotions accelerate between years 3 and 6.

    Thus, *timing* differences in promotions can vary significantly by field, with men often being promoted earlier in their careers.

    ---

    **4. Logistic Regression: Key Factors Affecting Promotions**  

    A logistic regression model (1976–1995 data) included **sex**, **field**, **degree type**, **salary**, *etc.* The resulting **log-odds coefficients** suggest the following:

    **Factors Promoting Promotion**  
    - **Having a PhD** (`Deg: type PhD`): Strong positive coefficient, indicating higher odds of promotion.  
    - **Professional Field** (`Field: Prof`): Linked to greater promotion likelihood than Arts or Other.  
    - **More Years Since Highest Degree** (`yrdeg`): Increased time since degree attainment correlates with better promotion odds (reflecting accumulated experience).  
    - **Being Male** (`Sex: M`): Men show higher promotion probabilities than women, holding other variables constant.

    **Factors Hindering Promotion**  
    - **Field: Arts**: Shows a negative coefficient, suggesting lower promotion odds compared to the Professional field.  
    - **Being Female** (`Sex: F`): Implies lower promotion likelihood relative to men, which is consistent with our earlier statistical tests and Kaplan-Meier analysis.

    ---

    **Assumptions and Limitations**  
    1. **Independence:**
    Each faculty member’s promotion outcome is treated as an independent observation.

    2. **Data Completeness:**  
    We assume that potential confounders—such as research productivity, teaching evaluations, and departmental affiliation—are either adequately captured or do not heavily bias the results.

    3. **Modeling Approach:**  
    Logistic regression is employed with one-hot encoding for categorical features, assuming an approximate linearity in the log-odds of promotion. This approach allows us to interpret how each factor affects promotion odds, but it may oversimplify complex relationships.

    4. **Time Frame:** 
    Promotions occurring after 1995 are not included, and faculty who remain at the Associate rank by 1995 are censored.

    ---

    **Interpretation & Conclusions**  
    - Across most fields, **men are more likely to be promoted** and often promoted more quickly, consistent with potential sex bias—though significance varies by field (Arts being a notable exception).  
    - Having a PhD, being in a Professional field, and having more years since degree also increase promotion odds, while being female or in the Arts field somehow hinders promotion.  
    - Overall, these findings support the conclusion that sex, field, and highest degree are influential in determining who becomes Full Professor, with **men** benefiting in many disciplines, especially early in the Associate timeline. 
   
   ---
   **Note:**
   This study’s findings are based solely on the 1976–1995 dataset. Unmeasured factors, such as research output or departmental culture, may also influence promotion outcomes. Additionally, our method defines promotion by recording the first year a faculty member reaches Associate rank and then the first year they attain Full Professor status, which does not fully account for the exact duration spent at the Associate rank. This is particularly important for faculty who may have had very short tenures at the Associate level or were recently hired or left the department/institution.
    """)

elif selected_tab == "Summary of Findings":
    st.header("Summary of Findings")
    
    st.subheader("Does sex bias exist in the university in 1995?", divider='violet')
    st.markdown("""
    """)
    
    st.subheader("Are starting salaries different between male and female faculty?", divider='violet')
    st.markdown("""
    """)
    
    st.subheader("Have salary increases between 1990 and 1995 been influenced by sex?", divider='violet')
    st.markdown("""
    """)
    
    st.subheader("Has sex bias existed in granting promotions from Associate to Full Professor?",divider='violet')
    st.markdown("""
    - Across most fields, men are more likely to be promoted and often promoted more quickly, consistent with potential sex bias—though significance varies by field (Arts being a notable exception).
    - Having a PhD, being in a Professional field, and having more years since degree also increase promotion odds, while being female or in the Arts field somehow hinders promotion.
    - Overall, these findings support the conclusion that sex, field, and highest degree are influential in determining who becomes Full Professor, with men benefiting in many disciplines, especially early in the Associate timeline."""
    )


st.markdown("---")
  