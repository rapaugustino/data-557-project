import streamlit as st
import pandas as pd
import promotions_analysis as pa
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.graph_objects as go
import promotions_analysis as pa
import salary_bias_analysis as sba


# --------------------------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------------------------
st.set_page_config(page_title="Faculty Salary Analysis Project", layout="wide")

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
    "Summary of Findings",
]

# Initialize the selected tab in session state if it doesn't exist
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = tab_options[0]

# Create a persistent radio button in the sidebar to select a tab
selected_tab = st.sidebar.radio(
    "Select Analysis",
    tab_options,
    index=tab_options.index(st.session_state.selected_tab),
)

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
    df = pd.read_csv(file_path, sep=r"[\t\s]+", header=0, engine="python")
    return df


data = load_data()

# --------------------------------------------------------------------
# Introduction Tab
# --------------------------------------------------------------------
if selected_tab == "Introduction":
    st.header("Introduction and Background")

    st.markdown(
        """
    ## Project Overview

    Welcome! This project examines faculty salary data from a U.S. university during the years 1976–1995, 
    with a focus on salary records from 1995. We’re investigating whether men and women differ in 
    how much they earn, how their salaries grow, and how quickly they get promoted. By applying 
    various statistical techniques—including regression, t-tests, Kaplan-Meier survival analysis, 
    and logistic regression—we aim to identify whether any gender-based disparities persist after 
    accounting for key factors like field, rank, and years of experience.

    ---

    ## Why Study This?

    Gender-based pay discrepancies in U.S. higher education have been documented extensively. 
    Though it’s illegal to discriminate by sex, salary differences could also arise from legitimate 
    factors such as:
    - **Field of specialization** (e.g., Business vs. Arts)
    - **Experience** (years since highest degree)
    - **Administrative roles** (like department chair)
    - **Rank** (Assistant, Associate, or Full Professor)

    There’s also the question of *when* promotions happen—particularly from Associate to Full 
    Professor, which is not bound by the strict timeline that governs promotion from Assistant 
    to Associate. In short, a faculty member’s rank and salary could hinge on many legitimate 
    influences, making it challenging to determine if any observed differences are due to bias 
    or simply reflect those legitimate factors.

    ---

    ## Questions of Interest

    Specifically, our project tackles four questions related to potential sex bias in this institution:

    1. **1995 Sex Bias:**  
       Are there salary differences between men and women in the most recent year (1995), and if so, 
       how big are those differences once we factor in rank, field, and other variables?

    2. **Starting Salaries:**  
       Have men and women historically started at different salary levels when first hired? 
       Have these differences shifted over time?

    3. **Salary Increases (1990–1995):**  
       Looking at the salaries of faculty who worked both in 1990 and 1995, does sex play a role 
       in how much raises have grown?

    4. **Promotions (Associate to Full):**  
       Does one sex tend to be promoted more often—or more quickly—from Associate to Full Professor?

    ---

    ## Data Highlights

    The dataset (excluding medical school faculty) includes:
    - **1597** faculty members employed in 1995  
    - **Yearly salary records** from 1976 to 1995 (total of 19,792 records)  
    - Variables covering sex, degree type, year of highest degree, field, start year, rank, 
      administrative duties, and the actual monthly salary.

    Each row corresponds to a faculty member’s annual record. For instance, if someone taught 
    continuously from 1980 to 1995, you would see multiple entries for that person—one for 
    each year.

    **Important Points to Consider**  
    - Salaries in different fields may vary significantly (e.g., engineering vs. humanities).  
    - Promotion from Assistant to Associate typically occurs around the sixth year, while promotion 
      from Associate to Full can happen at any time.  
    - We do not have data on productivity measures (like number of publications or grants), so any 
      differences we see by sex cannot automatically be attributed to discrimination.

    ---

    ## Analytical Approach

    Our methods combine both straightforward and advanced techniques:
    
    - **Descriptive Statistics & Visualization**: Basic summaries and plots to see how salaries 
      differ by sex, rank, and field.  
    - **T-tests & Proportion Tests**: Quick checks to see if differences in means or promotion 
      rates are statistically significant.  
    - **Multiple Linear Regression**: Investigates factors influencing salaries (e.g., field, rank, 
      years since degree) and how much of the gap remains after controlling for these factors.  
    - **Kaplan-Meier Survival Analysis**: Looks at time-to-promotion from Associate to Full rank, 
      comparing men and women.  
    - **Logistic Regression**: Estimates the likelihood of being promoted or underpaid, controlling 
      for relevant covariates like experience or administrative roles.

    ---

    ## Moving Through the Analysis

    - **1995 Sex Bias**: First stop—see if salary differences exist in 1995, after adjusting for 
      rank, field, and experience.  
    - **Starting Salaries**: Next, look at initial pay across different hiring years to check for 
      gender-based disparities at the start of employment.  
    - **Salary Increases (1990–1995)**: Then, dig into raises and see if men and women have 
      different growth patterns over this five-year period.  
    - **Promotions (Associate to Full)**: Finally, investigate whether gender impacts promotion 
      rates and timing.  
    - **Summary of Findings**: Wrap it all up with our key takeaways and reflections on any 
      potential biases we uncovered.

    ---

    ## Acknowledgments

    - **Data & Research Questions**: Provided by Professor Scott Emerson.
    - **Project Context**: Developed within the framework of an applied statistics course, guided by Professor Katie Wilson’s instruction.
    - **Team Collaboration**: Collaboration between team members each contributing to data preparation, analysis, and interpretation. The team includes [Kyle Cullen Bretherton](https://github.com/kylebreth), [Aravindh Manavalan](https://github.com/aravindh28), [Richard Pallangyo](https://github.com/rapugustino), [Akshay Ravi](https://github.com/akshayravi13), and [Vijay Balaji S](https://github.com/SVijayB).

    ---

    ## Important Caveats

    1. **Missing Factors**: We don’t have measures of individual productivity (like publications 
       or grants), which could partially explain salary or promotion outcomes.  
    2. **Historical Context**: The dataset ends in 1995; findings might not apply to current 
       policies or practices.  
    3. **Interpretation**: A remaining salary gap linked to sex after controlling for rank, field, 
       etc., is *suggestive* of bias, but it’s not absolute proof.  
    4. **Sample Limitations**: When you break data down by field or rank, sample sizes can shrink, 
       which affects how confidently we can generalize certain conclusions.

    In short, **correlation does not imply causation**. However, persistent unexplained differences 
    in salary or promotion between men and women—across various measures—are still red flags 
    that warrant further inquiry.

    Let’s dive in!
    """
    )

# -----------------------------------------------------------------
# Question 1: Does Sex Bias Exist at the University in 1995?
# -----------------------------------------------------------------
elif selected_tab == "1995 Sex Bias":

    st.header("Question 1: Does Sex Bias Exist at the University in 1995?")

    st.markdown(
        """
    **Objective:**  
    Determine if there is an evidence of sex-based salary discrimination 
    at the university in 1995.
    """
    )

    # Introduction to the analysis
    st.markdown(
        """
    **Approach:**  
    While raw differences in salary between men and women often exist, 
    these differences could be due to legitimate factors like field, rank, experience, or 
    administrative duties. Our approach is to first examine the raw (unadjusted) salary differences, then use statistical 
    methods to account for legitimate factors that affect salary. Any remaining difference 
    that can be attributed to sex may suggest potential bias.
    
    We'll use two complementary approaches:
    1. **Linear Regression**: To estimate the percentage difference in salary attributable to sex after controlling for other factors
    2. **Logistic Regression**: To determine if women are more likely to be "underpaid" relative to what would be expected based on their qualifications
    """
    )

    # --- Section A: Data Preparation ---
    st.subheader("A) Data Preparation")

    st.markdown(
        """
    First, we'll prepare our dataset by:
    - Filtering to include only faculty employed in 1995
    - Creating variables for years of experience and time since degree
    - Converting categorical variables (rank, field, etc.) to numerical format for analysis
    - Log-transforming salary to analyze percentage differences rather than absolute dollar amounts
    """
    )

    # Filter data to just 1995
    data_1995 = data[data["year"] == 95].copy()

    # Create derived variables
    data_1995["years_experience"] = data_1995["year"] - data_1995["startyr"]
    data_1995["years_since_degree"] = data_1995["year"] - data_1995["yrdeg"]
    data_1995["log_salary"] = np.log(data_1995["salary"])

    # Create binary indicators
    data_1995["female"] = (data_1995["sex"] == "F").astype(int)
    data_1995["is_assoc"] = (data_1995["rank"] == "Assoc").astype(int)
    data_1995["is_full"] = (data_1995["rank"] == "Full").astype(int)
    data_1995["is_arts"] = (data_1995["field"] == "Arts").astype(int)
    data_1995["is_prof"] = (data_1995["field"] == "Prof").astype(int)
    data_1995["is_phd"] = (data_1995["deg"] == "PhD").astype(int)
    data_1995["is_prof_deg"] = (data_1995["deg"] == "Prof").astype(int)

    # Add field filter with explanation
    st.markdown(
        """
    #### Faculty Field Filter
    
    Salary patterns may differ substantially by academic field. Use the filter below to focus
    on a specific field or view data across all fields.
    """
    )

    field_list = ["All"] + sorted(data_1995["field"].dropna().unique().tolist())
    selected_field = st.selectbox("Field Filter", field_list, index=0)

    if selected_field != "All":
        filtered_data = data_1995[data_1995["field"] == selected_field].copy()
        st.info(
            f"Showing analysis for the {selected_field} field only ({len(filtered_data)} faculty members)."
        )
    else:
        filtered_data = data_1995.copy()
        st.info(
            f"Showing analysis across all fields ({len(filtered_data)} faculty members)."
        )

    # --- Section B: Descriptive Statistics ---
    st.subheader("B) Examining Raw Salary Differences")

    st.markdown(
        """
    Before conducting advanced statistical analyses, it's important to understand the raw data.
    Here we examine the unadjusted salary differences between male and female faculty.
    
    These raw differences don't account for legitimate factors that affect salary (rank, field, experience),
    but they provide a starting point for our investigation.
    """
    )

    # Summary statistics table
    summary_stats = (
        filtered_data.groupby("sex")["salary"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    summary_stats.columns = ["Sex", "Count", "Mean Salary", "Median Salary", "Std Dev"]
    for col in ["Mean Salary", "Median Salary", "Std Dev"]:
        summary_stats[col] = summary_stats[col].round(2)

    st.write("**Monthly Salary Statistics by Sex (1995)**")
    st.dataframe(summary_stats)

    # Calculate unadjusted salary gap
    if "M" in summary_stats["Sex"].values and "F" in summary_stats["Sex"].values:
        male_mean = summary_stats.loc[
            summary_stats["Sex"] == "M", "Mean Salary"
        ].values[0]
        female_mean = summary_stats.loc[
            summary_stats["Sex"] == "F", "Mean Salary"
        ].values[0]
        gap = male_mean - female_mean
        gap_percent = (gap / male_mean) * 100

        st.write(
            f"**Unadjusted salary gap (M-F):** ${gap:.2f} per month ({gap_percent:.1f}% of male salary)"
        )

        if gap > 0:
            st.markdown(
                """
            ⚠️ **Note:** This raw difference doesn't necessarily indicate discrimination. 
            It could be explained by differences in field, rank, experience, or other factors.
            Our subsequent analysis will account for these factors.
            """
            )
    else:
        st.write(
            "Cannot calculate salary gap: data for both men and women is required."
        )

    # Summary by rank and sex
    st.markdown(
        """
    #### Salary by Rank and Sex
    
    Rank is a major determinant of faculty salary. Below we break down the data by both rank and sex
    to see if patterns differ across academic ranks.
    """
    )

    rank_sex_summary = (
        filtered_data.groupby(["rank", "sex"])["salary"]
        .agg(["count", "mean"])
        .reset_index()
    )
    rank_sex_pivot = rank_sex_summary.pivot(
        index="rank", columns="sex", values=["count", "mean"]
    )
    st.dataframe(rank_sex_pivot)

    # --- Section C: Visualizations ---
    st.subheader("C) Visualizing Salary Patterns")

    st.markdown(
        """
    Visualizations help us identify patterns in the data before formal statistical testing.
    The charts below explore different aspects of potential salary disparities.
    """
    )

    # Create two columns for side-by-side charts
    col1, col2 = st.columns(2)

    with col1:
        # Box plot of salary by sex and rank
        st.markdown("**Salary Distribution by Rank and Sex**")
        st.markdown(
            """
        This box plot shows salary distributions for men and women across different ranks.
        Look for consistent patterns where one group's boxes are higher than the other's within the same rank.
        """
        )

        fig1 = px.box(
            filtered_data,
            x="rank",
            y="salary",
            color="sex",
            labels={"rank": "Rank", "salary": "Monthly Salary", "sex": "Sex"},
        )
        fig1.update_layout(plot_bgcolor="#e5ecf6", paper_bgcolor="#e5ecf6")
        fig1.update_xaxes(showgrid=True, gridcolor="lightgray")
        fig1.update_yaxes(showgrid=True, gridcolor="lightgray")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown(
            """
        **How to interpret:** If boxes for one sex are consistently higher across all ranks,
        it might suggest a systematic pattern. However, overlapping boxes suggest smaller
        or inconsistent differences.
        """
        )

    with col2:
        # Scatter plot of salary vs experience colored by sex
        st.markdown("**Salary vs. Experience by Sex**")
        st.markdown(
            """
        This scatter plot shows how salary relates to years of experience for men and women.
        Trend lines help identify if the "return on experience" differs by sex.
        """
        )

        fig2 = px.scatter(
            filtered_data,
            x="years_experience",
            y="salary",
            color="sex",
            trendline="ols",
            opacity=0.7,
            labels={
                "years_experience": "Years of Experience",
                "salary": "Monthly Salary",
                "sex": "Sex",
            },
        )
        fig2.update_layout(plot_bgcolor="#e5ecf6", paper_bgcolor="#e5ecf6")
        fig2.update_xaxes(showgrid=True, gridcolor="lightgray")
        fig2.update_yaxes(showgrid=True, gridcolor="lightgray")

        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            """
        **How to interpret:** Different slopes in the trend lines suggest that men and women
        might receive different salary increases for additional years of experience.
        """
        )

    # Second row of visualizations
    col3, col4 = st.columns(2)

    with col3:
        # Bar chart of salary gap by field
        st.markdown("**Salary Comparison by Field**")

        if selected_field == "All":
            st.markdown(
                """
            This chart shows how the salary gap between men and women varies across different fields.
            Some fields may show larger disparities than others.
            """
            )

            if (
                "M" in filtered_data["sex"].values
                and "F" in filtered_data["sex"].values
            ):
                field_sex_gap = (
                    filtered_data.groupby(["field", "sex"])["salary"]
                    .mean()
                    .reset_index()
                )
                field_sex_pivot = field_sex_gap.pivot(
                    index="field", columns="sex", values="salary"
                ).reset_index()
                if "M" in field_sex_pivot.columns and "F" in field_sex_pivot.columns:
                    field_sex_pivot["gap"] = field_sex_pivot["M"] - field_sex_pivot["F"]
                    field_sex_pivot["gap_percent"] = (
                        field_sex_pivot["gap"] / field_sex_pivot["M"]
                    ) * 100

                    fig3 = px.bar(
                        field_sex_pivot,
                        x="field",
                        y="gap_percent",
                        labels={"field": "Field", "gap_percent": "Gap (%)"},
                    )
                    fig3.update_layout(plot_bgcolor="#e5ecf6", paper_bgcolor="#e5ecf6")
                    fig3.update_xaxes(showgrid=True, gridcolor="lightgray")
                    fig3.update_yaxes(showgrid=True, gridcolor="lightgray")

                    st.plotly_chart(fig3, use_container_width=True)

                    st.markdown(
                        """
                    **How to interpret:** Larger positive values indicate fields where men earn more than women on average.
                    Remember this still doesn't control for rank or experience within each field.
                    """
                    )
                else:
                    st.write("Insufficient data to create salary gap by field chart.")
            else:
                st.write("Insufficient data to create salary gap by field chart.")
        else:
            # For a single field, show average salary by sex
            st.markdown(
                f"""
            This chart shows the average salary by sex within the {selected_field} field.
            """
            )

            if (
                "M" in filtered_data["sex"].values
                and "F" in filtered_data["sex"].values
            ):
                sex_avg = filtered_data.groupby("sex")["salary"].mean().reset_index()
                fig3 = px.bar(
                    sex_avg,
                    x="sex",
                    y="salary",
                    color="sex",
                    labels={"sex": "Sex", "salary": "Average Monthly Salary"},
                    title=f"Average Salary by Sex in {selected_field} Field",
                )
                fig3.update_layout(plot_bgcolor="#e5ecf6", paper_bgcolor="#e5ecf6")
                fig3.update_xaxes(showgrid=True, gridcolor="lightgray")
                fig3.update_yaxes(showgrid=True, gridcolor="lightgray")

                st.plotly_chart(fig3, use_container_width=True)

                # Calculate gap
                male_avg = sex_avg.loc[sex_avg["sex"] == "M", "salary"].values[0]
                female_avg = sex_avg.loc[sex_avg["sex"] == "F", "salary"].values[0]
                gap = male_avg - female_avg
                gap_percent = (gap / male_avg) * 100

                st.markdown(
                    f"""
                **Raw gap:** ${gap:.2f} per month ({gap_percent:.1f}% of male salary)
                
                **Note:** This raw difference doesn't account for legitimate factors like rank, experience, etc.
                """
                )
            else:
                st.write(
                    "Insufficient data - need both male and female faculty in this field."
                )

    with col4:
        # Density plot of log salary by sex
        st.markdown("**Log Salary Distribution by Sex**")
        st.markdown(
            """
        This histogram shows the distribution of log-transformed salary by sex.
        Log transformation helps visualize percentage differences rather than absolute dollars.
        """
        )

        fig4 = px.histogram(
            filtered_data,
            x="log_salary",
            color="sex",
            marginal="rug",
            opacity=0.7,
            barmode="overlay",
            labels={"log_salary": "Log Monthly Salary", "sex": "Sex"},
        )
        fig4.update_layout(plot_bgcolor="#e5ecf6", paper_bgcolor="#e5ecf6")
        fig4.update_xaxes(showgrid=True, gridcolor="lightgray")
        fig4.update_yaxes(showgrid=True, gridcolor="lightgray")

        st.plotly_chart(fig4, use_container_width=True)

        st.markdown(
            """
        **How to interpret:** Shifts between the distributions indicate potential differences.
        If the female distribution is shifted left, it suggests lower salaries on average.
        """
        )

    # --- Section D: Multiple Linear Regression ---
    st.subheader("D) Multiple Linear Regression Analysis")

    st.markdown(
        """
    ### Controlling for Legitimate Factors
    
    Raw salary differences don't tell the full story. To identify potential bias, we need to account for 
    legitimate factors that affect salary. Using multiple linear regression, we can estimate the effect of
    sex while controlling for rank, field, experience, degree, and administrative duties.
    
    **Why log-transform salary?**
    - Makes the model estimate percentage differences rather than absolute dollar amounts
    - Better aligns with how salaries typically work (effects tend to be multiplicative)
    - Reduces the impact of outliers on model estimates
    
    **Variables in our model:**
    - **Outcome**: Log-transformed monthly salary
    - **Predictor of Interest**: Sex (female=1, male=0)
    - **Control variables**: Rank, field, years of experience, years since degree, highest degree, administrative duties
    """
    )

    # Full model including sex variable
    formula = "log_salary ~ female + is_assoc + is_full + is_arts + is_prof + years_experience + years_since_degree + is_phd + is_prof_deg + admin"
    model = smf.ols(formula=formula, data=filtered_data).fit()

    # Display regression results
    results_df = pd.DataFrame(
        {
            "Variable": model.params.index,
            "Coefficient": model.params.values,
            "Std Error": model.bse.values,
            "p-value": model.pvalues.values,
            "95% CI Lower": model.conf_int()[0],
            "95% CI Upper": model.conf_int()[1],
        }
    )

    # Format numeric columns
    for col in ["Coefficient", "Std Error", "p-value", "95% CI Lower", "95% CI Upper"]:
        results_df[col] = results_df[col].round(4)

    # Add significance indicator
    results_df["Significance"] = results_df["p-value"].apply(
        lambda p: (
            "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        )
    )

    st.markdown("#### Full Model (Including Sex) Results")
    st.markdown(
        """
    The table below shows the effect of each variable on log salary. Our primary interest is the coefficient
    for 'female', which represents the estimated percentage difference in salary between women and men
    after controlling for other factors.
    """
    )

    st.dataframe(results_df)

    st.markdown(
        """
    **How to interpret:**
    - **Coefficient**: The estimated effect on log salary (approximate percentage effect)
    - **p-value**: The probability of observing this effect by chance if no true effect exists
    - **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns=not significant
    """
    )

    # Residual plot
    filtered_data["predicted_log_salary_full"] = model.predict(filtered_data)
    filtered_data["residuals_full"] = (
        filtered_data["log_salary"] - filtered_data["predicted_log_salary_full"]
    )

    fig_residual = px.scatter(
        filtered_data,
        x="predicted_log_salary_full",
        y="residuals_full",
        color="sex",
        opacity=0.7,
        title="Residuals from Full Model (Including Sex)",
        labels={
            "predicted_log_salary_full": "Predicted Log Salary",
            "residuals_full": "Residuals",
            "sex": "Sex",
        },
    )

    fig_residual.update_layout(plot_bgcolor="#e5ecf6", paper_bgcolor="#e5ecf6")
    fig_residual.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig_residual.update_yaxes(showgrid=True, gridcolor="lightgray")
    fig_residual.add_hline(y=0, line_dash="dash", line_color="black")

    st.plotly_chart(fig_residual, use_container_width=True)

    # Interpret the female coefficient
    female_coef = model.params["female"]
    female_pval = model.pvalues["female"]
    percent_effect = (np.exp(female_coef) - 1) * 100
    female_ci_lower = model.conf_int().loc["female"][0]
    female_ci_upper = model.conf_int().loc["female"][1]
    percent_ci_lower = (np.exp(female_ci_lower) - 1) * 100
    percent_ci_upper = (np.exp(female_ci_upper) - 1) * 100

    st.markdown("#### Key Finding: Adjusted Salary Gap")

    if female_pval < 0.05:
        if percent_effect < 0:
            effect_interpretation = f"After controlling for rank, field, experience, degree, and administrative duties, women earn approximately {abs(percent_effect):.1f}% less than similarly qualified men (95% CI: {percent_ci_lower:.1f}% to {percent_ci_upper:.1f}%, p={female_pval:.4f})."
            st.info(effect_interpretation)
            st.markdown(
                """
            **This statistically significant difference suggests potential sex bias in faculty salaries.**
            
            Since we've controlled for legitimate factors affecting salary, the remaining difference 
            attributable to sex raises concerns about systematic bias.
            """
            )
        else:
            effect_interpretation = f"After controlling for rank, field, experience, degree, and administrative duties, women earn approximately {abs(percent_effect):.1f}% more than similarly qualified men (95% CI: {percent_ci_lower:.1f}% to {percent_ci_upper:.1f}%, p={female_pval:.4f})."
            st.success(effect_interpretation)
    else:
        effect_interpretation = f"After controlling for rank, field, experience, degree, and administrative duties, there is no statistically significant difference in salary between men and women (p={female_pval:.4f})."
        st.info(effect_interpretation)
        st.markdown(
            """
        **The absence of a significant difference suggests that the raw salary gap may be explained by legitimate factors.**
        
        This doesn't rule out other forms of bias (such as in hiring or promotion), but suggests that 
        men and women with similar qualifications receive similar compensation.
        """
        )

    # --- NEW SECTION: Model Without Sex ---
    st.subheader("E) Model Without Sex Variable")

    st.markdown(
        """
    ### Creating a "Fair Salary" Model
    
    To identify potential bias, we build a reduced model that predicts salary based only on legitimate factors (excluding sex).
    This "fair salary" model tells us what someone with certain qualifications would be expected to earn regardless of their sex.
    
    By comparing actual salaries to these predicted "fair salaries," we can identify individuals who might be underpaid
    relative to their qualifications, then test if being female increases the likelihood of being underpaid.
    """
    )

    # Create model without sex
    formula_nosex = "log_salary ~ is_assoc + is_full + is_arts + is_prof + years_experience + years_since_degree + is_phd + is_prof_deg + admin"
    model_nosex = smf.ols(formula=formula_nosex, data=filtered_data).fit()

    # Display regression results for model without sex
    results_df_nosex = pd.DataFrame(
        {
            "Variable": model_nosex.params.index,
            "Coefficient": model_nosex.params.values,
            "Std Error": model_nosex.bse.values,
            "p-value": model_nosex.pvalues.values,
            "95% CI Lower": model_nosex.conf_int()[0],
            "95% CI Upper": model_nosex.conf_int()[1],
        }
    )

    # Format numeric columns
    for col in ["Coefficient", "Std Error", "p-value", "95% CI Lower", "95% CI Upper"]:
        results_df_nosex[col] = results_df_nosex[col].round(4)

    # Add significance indicator
    results_df_nosex["Significance"] = results_df_nosex["p-value"].apply(
        lambda p: (
            "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        )
    )

    st.markdown("#### Fair Salary Model (Excluding Sex) Results")
    st.dataframe(results_df_nosex)

    # Compare model performance
    st.markdown("#### Comparing Models With and Without Sex Variable")

    model_comparison = pd.DataFrame(
        {
            "Model": ["Including Sex", "Excluding Sex"],
            "R-squared": [model.rsquared, model_nosex.rsquared],
            "Adj. R-squared": [model.rsquared_adj, model_nosex.rsquared_adj],
            "AIC": [model.aic, model_nosex.aic],
            "BIC": [model.bic, model_nosex.bic],
            "Log-Likelihood": [model.llf, model_nosex.llf],
        }
    )

    # Format numeric columns
    for col in ["R-squared", "Adj. R-squared", "AIC", "BIC", "Log-Likelihood"]:
        model_comparison[col] = model_comparison[col].round(4)

    st.dataframe(model_comparison)

    st.markdown(
        """
    **How to interpret:**
    - If the model with sex has notably better fit metrics (higher R-squared, lower AIC/BIC), it suggests sex is an important predictor of salary
    - Small differences suggest sex has minimal additional predictive power after accounting for other factors
    """
    )

    # Calculate "expected" salaries and differences
    filtered_data["expected_log_salary"] = model_nosex.predict(filtered_data)
    filtered_data["salary_residual"] = (
        filtered_data["log_salary"] - filtered_data["expected_log_salary"]
    )
    filtered_data["percent_diff"] = (np.exp(filtered_data["salary_residual"]) - 1) * 100
    filtered_data["expected_salary"] = np.exp(filtered_data["expected_log_salary"])

    # Create visualization comparing predictions - IMPROVED VERSION
    st.markdown("#### Comparison of Actual vs. Expected Salary by Sex")

    # Create figure
    fig_compare = go.Figure()

    # Add male scatter points and trend line
    male_data = filtered_data[filtered_data["sex"] == "M"]
    fig_compare.add_trace(
        go.Scatter(
            x=male_data["years_experience"],
            y=male_data["salary"],
            mode="markers",
            marker=dict(color="#636EFA", size=8, opacity=0.5),
            name="Male faculty",
        )
    )

    # Add female scatter points and trend line
    female_data = filtered_data[filtered_data["sex"] == "F"]
    fig_compare.add_trace(
        go.Scatter(
            x=female_data["years_experience"],
            y=female_data["salary"],
            mode="markers",
            marker=dict(color="#EF553B", size=8, opacity=0.5),
            name="Female faculty",
        )
    )

    # Calculate and add trend lines
    # Male trend line
    male_model = np.polyfit(male_data["years_experience"], male_data["salary"], 1)
    male_x = np.array(
        [min(filtered_data["years_experience"]), max(filtered_data["years_experience"])]
    )
    male_y = male_model[0] * male_x + male_model[1]
    fig_compare.add_trace(
        go.Scatter(
            x=male_x,
            y=male_y,
            mode="lines",
            line=dict(color="#636EFA", width=3),
            name="Male trend line",
        )
    )

    # Female trend line
    female_model = np.polyfit(female_data["years_experience"], female_data["salary"], 1)
    female_x = np.array(
        [min(filtered_data["years_experience"]), max(filtered_data["years_experience"])]
    )
    female_y = female_model[0] * female_x + female_model[1]
    fig_compare.add_trace(
        go.Scatter(
            x=female_x,
            y=female_y,
            mode="lines",
            line=dict(color="#EF553B", width=3),
            name="Female trend line",
        )
    )

    # Expected salary trend line (based on model without sex)
    expected_model = np.polyfit(
        filtered_data["years_experience"], filtered_data["expected_salary"], 1
    )
    expected_x = np.array(
        [min(filtered_data["years_experience"]), max(filtered_data["years_experience"])]
    )
    expected_y = expected_model[0] * expected_x + expected_model[1]
    fig_compare.add_trace(
        go.Scatter(
            x=expected_x,
            y=expected_y,
            mode="lines",
            line=dict(color="black", width=3, dash="dash"),
            name="Expected 'fair' salary",
        )
    )

    # Layout
    fig_compare.update_layout(
        plot_bgcolor="#e5ecf6",
        paper_bgcolor="#e5ecf6",
        xaxis=dict(title="Years of Experience", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(
            title="Monthly Salary",
            showgrid=True,
            gridcolor="lightgray",
            tickformat=".1f",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=60, b=20),
        height=500,
    )

    st.plotly_chart(fig_compare, use_container_width=True)

    st.markdown(
        """
    **How to interpret:**
    - The **dashed black line** shows what salary would be predicted based only on legitimate factors (excluding sex)
    - If one sex's trend line is consistently above or below the expected trend, it suggests potential bias
    - The gap between a colored line and the black line represents potential bias
    """
    )

    # Distribution of differences by sex
    st.markdown("#### Distribution of Salary Differences from Expected")

    fig_diff = px.histogram(
        filtered_data,
        x="percent_diff",
        color="sex",
        barmode="overlay",
        opacity=0.7,
        labels={"percent_diff": "% Difference from Expected Salary", "sex": "Sex"},
    )

    fig_diff.update_layout(plot_bgcolor="#e5ecf6", paper_bgcolor="#e5ecf6")
    fig_diff.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig_diff.update_yaxes(showgrid=True, gridcolor="lightgray")

    st.plotly_chart(fig_diff, use_container_width=True)

    st.markdown(
        """
    This histogram shows how actual salaries differ from expected salaries (as percentages).
    - Values below 0% indicate faculty who are paid less than expected based on their qualifications
    - Values above 0% indicate faculty who are paid more than expected based on their qualifications
    - If one sex's distribution is shifted left, it suggests that group is more likely to be underpaid
    """
    )

    # --- Section F: Logistic Regression Analysis ---
    st.subheader("F) Underpaid Faculty Analysis")

    st.markdown(
        """
    ### Are Women More Likely to be "Underpaid"?
    
    Using our "fair salary" model from above, we can identify faculty who appear to be underpaid
    relative to what would be expected based on their qualifications. Then we can test whether
    being female increases the likelihood of being underpaid.
    
    This approach might detect patterns of bias that aren't evident in average effects.
    """
    )

    # Define "underpaid" threshold with slider
    st.markdown(
        """
    #### Defining "Underpaid"
    
    We consider faculty to be "underpaid" if their actual salary is below their expected salary
    by more than a certain percentage. Use the slider below to adjust this threshold.
    """
    )

    threshold = st.slider("Underpaid Threshold (%)", -20, -5, -5) / 100
    filtered_data["underpaid"] = (filtered_data["salary_residual"] < threshold).astype(
        int
    )

    st.markdown(
        f"""
    With the current threshold of {threshold*100:.0f}%, faculty are considered "underpaid" if 
    their actual salary is more than {abs(threshold*100):.0f}% below what would be expected
    based on their qualifications.
    """
    )

    # Count of underpaid by sex
    st.markdown("#### Percentage of Faculty Classified as Underpaid by Sex")

    underpaid_by_sex = (
        pd.crosstab(filtered_data["sex"], filtered_data["underpaid"], normalize="index")
        * 100
    )
    underpaid_by_sex.columns = ["Fairly Paid (%)", "Underpaid (%)"]

    st.dataframe(underpaid_by_sex.round(1))

    if "M" in underpaid_by_sex.index and "F" in underpaid_by_sex.index:
        male_underpaid = underpaid_by_sex.loc["M", "Underpaid (%)"]
        female_underpaid = underpaid_by_sex.loc["F", "Underpaid (%)"]
        diff = female_underpaid - male_underpaid

        if abs(diff) > 5:
            st.markdown(
                f"""
            **Note:** {female_underpaid:.1f}% of women are classified as underpaid compared to {male_underpaid:.1f}% of men 
            (a difference of {abs(diff):.1f} percentage points). This suggests a potential pattern, but we'll use
            logistic regression to test if this difference is statistically significant.
            """
            )

    # Visualization of salary residuals with threshold line
    st.markdown("#### Distribution of Salary Residuals by Sex")
    st.markdown(
        """
    This histogram shows the distribution of residuals (actual salary minus expected salary)
    for men and women. The vertical red line marks the threshold for being classified as "underpaid."
    """
    )

    fig5 = px.histogram(
        filtered_data,
        x="salary_residual",
        color="sex",
        barmode="overlay",
        opacity=0.7,
        labels={"salary_residual": "Log Salary Residual", "sex": "Sex"},
    )

    fig5.update_layout(plot_bgcolor="#e5ecf6", paper_bgcolor="#e5ecf6")
    fig5.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig5.update_yaxes(showgrid=True, gridcolor="lightgray")

    # Add vertical line at threshold
    fig5.add_vline(x=threshold, line_dash="dash", line_color="red")
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown(
        """
    **How to interpret:** If the female distribution is shifted left relative to the male distribution,
    it suggests women are more likely to be paid below their expected salary. The area to the left of
    the red line represents faculty classified as "underpaid."
    """
    )

    # Feature selection for logistic regression
    st.markdown("#### Logistic Regression Analysis")
    st.markdown(
        """
    Now we'll use logistic regression to test whether being female increases the odds of being underpaid,
    even after accounting for other factors that might affect this probability.
        
    Select which factors you'd like to include in the model:
    """
    )

    # Features to include - keep all options but add warnings
    possible_features = ["female", "is_arts", "is_prof", "years_experience", "is_phd"]
    selected_features = st.multiselect(
        "Features for Logistic Model:", options=possible_features, default=["female"]
    )

    # Add warning about field variables when a specific field is selected
    if selected_field != "All":
        if "is_arts" in selected_features and selected_field == "Arts":
            st.warning(
                "⚠️ Including 'is_arts' may cause errors when you've already filtered to Arts field, as all records will have the same value (is_arts=1)."
            )
        if "is_prof" in selected_features and selected_field == "Prof":
            st.warning(
                "⚠️ Including 'is_prof' may cause errors when you've already filtered to Professional field, as all records will have the same value (is_prof=1)."
            )

    if "female" not in selected_features:
        st.warning("Adding 'female' as it's required for this analysis")
        selected_features = ["female"] + selected_features

    if len(selected_features) > 0:
        # Prepare data for logistic regression
        X = filtered_data[selected_features]
        y = filtered_data["underpaid"]

        # Run logistic regression with error handling
        try:
            # Check if we have both 0s and 1s in the target variable
            if y.nunique() < 2:
                st.warning(
                    "All faculty are either underpaid or not underpaid at this threshold. Try adjusting the threshold."
                )
            else:
                logit_model = sm.Logit(y, sm.add_constant(X)).fit(disp=0)

                # Create summary dataframe
                logit_results = pd.DataFrame(
                    {
                        "Variable": logit_model.params.index,
                        "Coefficient": logit_model.params.values,
                        "Std Error": logit_model.bse.values,
                        "p-value": logit_model.pvalues.values,
                        "Odds Ratio": np.exp(logit_model.params.values),
                        "95% CI Lower": np.exp(logit_model.conf_int()[0]),
                        "95% CI Upper": np.exp(logit_model.conf_int()[1]),
                    }
                )

                # Format numeric columns
                for col in [
                    "Coefficient",
                    "Std Error",
                    "p-value",
                    "Odds Ratio",
                    "95% CI Lower",
                    "95% CI Upper",
                ]:
                    logit_results[col] = logit_results[col].round(4)

                # Add significance indicator
                logit_results["Significance"] = logit_results["p-value"].apply(
                    lambda p: (
                        "***"
                        if p < 0.001
                        else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                    )
                )

                st.dataframe(logit_results)

                st.markdown(
                    """
                **How to interpret:**
                - **Coefficient**: Log odds ratio (effect on log-odds of being underpaid)
                - **Odds Ratio**: How many times more/less likely a group is to be underpaid
                - **p-value**: Statistical significance of the effect
                """
                )

                # Interpret the female coefficient
                female_logit_coef = logit_model.params["female"]
                female_logit_pval = logit_model.pvalues["female"]
                female_odds_ratio = np.exp(female_logit_coef)
                ci_lower = np.exp(logit_model.conf_int().loc["female"][0])
                ci_upper = np.exp(logit_model.conf_int().loc["female"][1])

                st.markdown("#### Key Finding: Likelihood of Being Underpaid")

                if female_logit_pval < 0.05:
                    if female_odds_ratio > 1:
                        logit_interpretation = f"Women are {female_odds_ratio:.2f} times more likely to be underpaid than men (95% CI: {ci_lower:.2f} to {ci_upper:.2f}, p={female_logit_pval:.4f})."
                        st.error(logit_interpretation)
                        st.markdown(
                            """
                        **This statistically significant result suggests potential sex bias in salary determination.**
                        
                        Even after accounting for other factors, women appear to be at higher risk of receiving salaries
                        below what would be expected based on their qualifications.
                        """
                        )
                    else:
                        logit_interpretation = f"Women are {1/female_odds_ratio:.2f} times less likely to be underpaid than men (95% CI: {1/ci_upper:.2f} to {1/ci_lower:.2f}, p={female_logit_pval:.4f})."
                        st.success(logit_interpretation)
                else:
                    logit_interpretation = f"There is no statistically significant difference between men and women in the likelihood of being underpaid (p={female_logit_pval:.4f})."
                    st.info(logit_interpretation)
                    st.markdown(
                        """
                    **The absence of a significant difference suggests that men and women have similar chances of being paid below expectations.**
                    
                    This finding aligns with a fair salary determination process with respect to sex.
                    """
                    )

        except Exception as e:
            st.error(f"Error in logistic regression: {str(e)}")
            st.warning(
                """
            This error may be due to complete or quasi-complete separation in the data (a perfect predictor).
            Try adjusting the threshold or adding more features to the model.
            """
            )

    # --- Section G: Conclusion ---
    st.subheader("G) Conclusion and Limitations")

    st.markdown(
        """
    ### Summary of Findings

    Our analysis used multiple approaches to investigate potential sex bias in faculty salaries at this university in 1995:

    1. **Descriptive Statistics:** We observed raw salary differences between men and women, with women having an unadjusted salary difference of $1334 per month 

    2. **Linear Regression:** We estimated the percentage difference in salary attributable to sex after
    controlling for rank, field, experience, degree, and administrative duties. The coefficient suggests a difference of 5 percent in salary compared to male salaries.

    3. **Fair Salary Model:** We created a model that predicts salary based solely on legitimate factors,
    excluding sex, to determine what faculty members should earn based on their qualifications.
    
    **Model Comparison Insight:** When comparing the models with and without the sex variable, we found that including sex 
    improved model fit metrics:
    - R-squared increased by about 0.4 percentage points (from 58.3% to 58.7%)
    - Both AIC and BIC values were lower (better) for the model including sex
    - The log-likelihood was higher for the model including sex by 7.5 points
    
    These differences, while small in terms of overall variance explained, suggest that sex does have some predictive 
    power for faculty salaries after controlling for legitimate factors. The improvement in model fit outweighs the 
    penalty for adding another parameter, supporting our findings about sex-based salary differences.

    4. **Logistic Regression:** We tested whether women are more likely than men to be "underpaid"
    relative to expectations based on their qualifications.
    
    **Logistic Model Finding:** Using a 5% threshold to define "underpaid" faculty (those earning at least 5% 
    less than predicted by the fair salary model), we found:
    - Women are 1.33 times more likely to be underpaid than men (95% CI: 1.06 to 1.67, p=0.0128)
    - This statistically significant result provides additional evidence of potential sex bias
    - The effect persisted even when considering only the sex variable, suggesting a robust pattern
    
    This complementary approach confirms that not only do women earn less on average (as shown in the linear 
    regression), but they are also significantly more likely to fall into the "underpaid" category relative 
    to what would be expected based on their qualifications.

    ### Limitations

    Several important limitations should be considered:

    1. **Unmeasured Factors:** Our analysis couldn't control for variables not in the dataset, such as
    publication record, teaching evaluations, or grant funding, which might legitimately affect salary.

    2. **Pre-existing Bias:** If bias affected earlier decisions (hiring, promotion), controlling for
    rank might actually mask discrimination rather than isolate it.

    3. **Causality:** Statistical associations don't necessarily imply causation. We can identify
    patterns but can't definitively establish their causes.

    4. **Sample Size:** Particularly when filtering by field, small sample sizes may limit statistical power
    and the reliability of estimates.

    ### Next Steps

    For a comprehensive understanding of potential sex bias at this university, we should also examine:

    1. Questions 2-4 in this analysis (starting salaries, salary increases, and promotion patterns)
    2. Department-level patterns that might be obscured in aggregated analyses
    3. Changes over time in salary determination practices
    """
    )

#  -----------------------------------------------------------------------------
#  Question 2: Has Sex Bias Existed in Starting Salaries?
#  -----------------------------------------------------------------------------
elif selected_tab == "Starting Salaries":
    st.header("Question 2: Has Sex Bias Existed in Starting Salaries?")
    st.markdown(
        """
    **Objective:**  
    Determine if there's evidence of sex bias in starting salaries and whether that has changed over the time 
    period represented in the dataset
    """
    )

    # --- Section A: Data Preparation ---
    st.subheader("A) Data Preparation")

    st.markdown(
        """
    First, we'll prepare our dataset by:
    - Filtering to include only faculty hired in the current year who are at the Assistant professor level
    - Normalizing salaries by dividing them by the average salary of male professors in the current year
    - Converting categorical variables (sex, field, etc.) to numerical format for analysis
    - Adding new fields for years since degree, as well as an interaction term between sex and time
    """
    )

    def prepare_starting_salary_data(df):
        # Filter for only employees hired in the current year at the Assistant level (and also remove some other
        # types of data that would mess with the analysis
        df_yrhired = df[
            (df["startyr"] == df["year"])
            & (df["year"] >= df["yrdeg"])
            & (df["rank"] == "Assist")
            & (df["deg"] != "Other")
        ]

        # avg_sal_male = avg_salary[avg_salary['sex'] == 'M'].copy()
        # avg_sal_male.drop('sex', axis=1, inplace=True)
        # avg_sal_male.rename(columns={'salary': 'avg_salary'}, inplace=True)
        # df_yrhired = pd.merge(df_yrhired, avg_sal_male, on='year', how='left')

        # New data columns for regression
        # df_yrhired['salary_norm'] = df_yrhired['salary'] / df_yrhired['avg_salary']
        df_yrhired["sex_bool"] = df_yrhired["sex"] == "F"
        df_yrhired["yr_full"] = 1900 + df_yrhired["year"]
        df_yrhired["yr_adj_1975"] = df_yrhired["yr_full"] - 1975
        df_yrhired["sex_year_1975"] = df_yrhired["sex_bool"] * df_yrhired["yr_adj_1975"]
        df_yrhired["yr_adj_1995"] = df_yrhired["yr_full"] - 1995
        df_yrhired["sex_year_1995"] = df_yrhired["sex_bool"] * df_yrhired["yr_adj_1995"]
        df_yrhired["yrs_experience"] = df_yrhired["year"] - df_yrhired["yrdeg"]
        df_yrhired["field_arts"] = df_yrhired["field"] == "Arts"
        df_yrhired["field_prof"] = df_yrhired["field"] == "Prof"
        df_yrhired["deg_prof"] = df_yrhired["deg"] == "Prof"

        return df_yrhired

    df_yrhired = prepare_starting_salary_data(data)
    if df_yrhired.empty:
        st.warning("No eligible records found. Check your dataset.")
        st.stop()

    # Add field filter
    st.markdown(
        """
    **Faculty Field Filter**
    
    Salary patterns may differ substantially by academic field. Use the filter below to focus
    on a specific field or view data across all fields.
    """
    )

    q2_field_list = ["All"] + sorted(df_yrhired["field"].dropna().unique().tolist())
    q2_selected_field = st.selectbox("Field Filter", q2_field_list, index=0)

    if q2_selected_field != "All":
        ss_filtered = df_yrhired[df_yrhired["field"] == q2_selected_field].copy()
    else:
        ss_filtered = df_yrhired.copy()

    st.markdown(
        """
    **Time Range Filter**

    Gender pay disparities may vary over time. Select the time period you want to analyze.
    """
    )

    # Create a slider to select the range of years
    start_year, end_year = st.slider(
        "Select the time range:",
        min_value=int(ss_filtered["yr_full"].min()),
        max_value=int(ss_filtered["yr_full"].max()),
        value=(int(ss_filtered["yr_full"].min()), int(ss_filtered["yr_full"].max())),
        step=1,
    )
    
    # Create a full range of years from min to max (important for error handling)
    all_years = pd.DataFrame({'yr_full': range(start_year, end_year + 1)})

    # Filter the data based on the selected year range
    ss_filtered_yr = ss_filtered[
        (ss_filtered["yr_full"] >= start_year) & (ss_filtered["yr_full"] <= end_year)
    ]

    st.info(
        f"Showing analysis across {q2_selected_field} field(s) between {start_year} and {end_year} ({len(ss_filtered_yr)} faculty members)."
    )

    # --- Section B: Descriptive Statistics ---
    st.subheader("B) Raw Differences in Starting Salary")

    st.markdown(
        """
    Before conducting advanced statistical analyses, it's important to understand the raw data.
    Here we examine the unadjusted starting salaries over time for male and female faculty.
    
    These raw differences don't account for legitimate factors that affect starting salary (such as
    field or degree), but they provide a starting point for our investigation.
    """
    )

    # Boxplots of average starting salaries by gender over the whole time period
    q2_box = px.box(
        ss_filtered_yr,
        x="sex",
        y="salary",
        color="sex",
        title=f"Starting Salary ({start_year}-{end_year}) by Sex",
        labels={"salary": "Starting Salary ($)", "sex": "Sex"},
    )
    q2_box.update_layout(
        xaxis_title="Sex",
        yaxis_title="Starting Salary ($)",
        plot_bgcolor="#e5ecf6",
        paper_bgcolor="#e5ecf6",
    )
    q2_box.update_xaxes(showgrid=True, gridcolor="lightgray")
    q2_box.update_yaxes(showgrid=True, gridcolor="lightgray")
    if q2_box is None:
        st.warning("Not enough data for box plot.")
    else:
        st.plotly_chart(q2_box, use_container_width=True)

    # Line chart of average salary of professors by gender for the given year
    avg_salary = (
        ss_filtered_yr.groupby(["yr_full", "sex"])["salary"].mean().reset_index()
    )

    # Plot of average starting salaries by sex and year
    # Create the plot
    fig = px.line(
        avg_salary,
        x="yr_full",
        y="salary",
        color="sex",
        markers=True,
        title="Average Starting Salary by Sex and Year",
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Average Starting Salary",
        template="plotly_white",  # Gives a clean look
        legend_title_text="Sex",
        width=800,
        height=600,
        plot_bgcolor="#e5ecf6",
        paper_bgcolor="#e5ecf6",
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

    # If you want to ensure the x-axis only shows integer years:
    fig.update_xaxes(
        tickmode="linear", dtick=1  # Show every year if you want an annual tick
    )

    st.plotly_chart(fig, use_container_width=True)

    # Normalize salaries by dividing them by average male starting salary in the current year
    # Use these values to calculate unadjusted starting salary percentage gap
    #If there are no males in the current year, we use interpolation to estimate
    avg_sal_male = avg_salary[avg_salary["sex"] == "M"].copy()
    avg_sal_male.drop("sex", axis=1, inplace=True)
    avg_sal_male.rename(columns={"salary": "avg_salary"}, inplace=True)
    avg_sal_male = pd.merge(all_years, avg_sal_male, on="yr_full", how="left")
    avg_sal_male = avg_sal_male.interpolate(method="linear")
    ss_filtered_yr = pd.merge(ss_filtered_yr, avg_sal_male, on="yr_full", how="left")
    ss_filtered_yr["salary_anom"] = (
        ss_filtered_yr["salary"] - ss_filtered_yr["avg_salary"]
    )
    ss_filtered_yr["salary_norm"] = (
        ss_filtered_yr["salary"] / ss_filtered_yr["avg_salary"]
    )
    gap = (
        ss_filtered_yr[ss_filtered_yr["sex"] == "M"]["salary"].mean()
        - ss_filtered_yr[ss_filtered_yr["sex"] == "F"]["salary"].mean()
    )
    gap_percent = (
        gap / ss_filtered_yr[ss_filtered_yr["sex"] == "M"]["salary"].mean() * 100
    )
    gap_yr = abs(ss_filtered_yr[ss_filtered_yr["sex"] == "F"]["salary_anom"].mean())
    gap_percent_yr = (
        1 - abs(ss_filtered_yr[ss_filtered_yr["sex"] == "F"]["salary_norm"].mean())
    ) * 100

    st.write(
        f"**Unadjusted salary gap (M-F):** ${gap:.2f} per month ({gap_percent:.1f}% of male salary)"
    )

    st.write(
        f"**Salary gap adjusted by year (M-F):** ${gap_yr:.2f} per month ({gap_percent_yr:.1f}% of male salary)"
    )

    st.markdown(
        """
        ⚠️ **Note:** This raw difference doesn't necessarily indicate discrimination. 
        It could be explained by differences in field, degree type, or other factors.
        Our subsequent analysis will account for these factors.
        """
    )

    # --- Section C: Advanced Linear Regression Model ---
    st.subheader("C) Linear Regression Model for Starting Salaries by Sex")

    st.markdown(
        """
    Raw salary differences don't tell the full story. To identify potential bias, we need to account for 
    other factors that may affect salary. Using multiple linear regression, we can estimate the effect of
    sex on starting salaries while controlling for field, degree, and administrative duties.
    
    For this regression, we chose to normalize the starting salary values by dividing them by the average 
    male starting salary in that year. This allows us to compare salary trends over time without worrying
    about changes in salaries over time. So, the regression coefficients should be read as a proportion of
    the average male starting salary.
    
    **Variables in our model:**
    - **Outcome**: Normalized monthly salary
    - **Predictor of Interest**: Sex (sex_bool, female=1, male=0)
    - **Control variables**: Field, years since highest degree, degree
        type, administrative duties
    """
    )

    st.markdown("**Select Model Features:**")
    possible_features = [
        "sex_bool",
        "yrs_experience",
        "field_arts",
        "field_prof",
        "admin",
        "deg_prof",
    ]

    default_feats = possible_features.copy()
    selected_features = st.multiselect(
        "Features to Include:", options=possible_features, default=default_feats
    )

    if len(selected_features) == 0:
        st.warning("Please select at least one feature.")
        st.stop()

    pred = ss_filtered_yr[selected_features].astype(float)
    outcome = ss_filtered_yr[["salary_norm"]].astype(float)
    pred_w_intercept = sm.add_constant(pred)

    # Run regression
    model = sm.OLS(outcome, pred_w_intercept).fit()

    # Create summary dataframe
    model_results = pd.DataFrame(
        {
            "Variable": model.params.index,
            "Coefficient": model.params.values,
            "Std Error": model.bse.values,
            "p-value": model.pvalues.values,
            "95% CI Lower": model.conf_int()[0],
            "95% CI Upper": model.conf_int()[1],
        }
    )

    # Format numeric columns
    for col in ["Coefficient", "Std Error", "p-value", "95% CI Lower", "95% CI Upper"]:
        model_results[col] = model_results[col].round(4)

    # Add significance indicator
    model_results["Significance"] = model_results["p-value"].apply(
        lambda p: (
            "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        )
    )

    st.markdown("**Starting Salary Model Results**")
    st.dataframe(model_results)
    
    # Interpret the female coefficient
    female_coef_q2 = model.params["sex_bool"]
    female_pval_q2 = model.pvalues["sex_bool"]
    percent_effect_q2 = female_coef_q2 * 100
    female_ci_lower_q2 = model.conf_int().loc["sex_bool"][0]
    female_ci_upper_q2 = model.conf_int().loc["sex_bool"][1]
    percent_ci_lower_q2 = female_ci_lower_q2 * 100
    percent_ci_upper_q2 = female_ci_upper_q2 * 100
    
    if female_pval_q2 < 0.05:
        if percent_effect_q2 < 0:
            effect_interpretation_q2 = f"After controlling for rank, field, experience, degree, and administrative duties, women's starting salaries are approximately {abs(percent_effect_q2):.1f}% less than similarly qualified men (95% CI: {percent_ci_lower_q2:.1f}% to {percent_ci_upper_q2:.1f}%, p={female_pval_q2:.4f})."
            st.info(effect_interpretation_q2)
            st.markdown(
                """
            **This statistically significant difference suggests potential sex bias in starting salaries.**
            
            Since we've controlled for legitimate factors affecting starting salaries, the remaining difference 
            attributable to sex raises concerns about systematic bias.
            """
            )
        else:
            effect_interpretation_q2 = f"After controlling for rank, field, experience, degree, and administrative duties, women's starting salaries are approximately {abs(percent_effect_q2):.1f}% more than similarly qualified men (95% CI: {percent_ci_lower_q2:.1f}% to {percent_ci_upper_q2:.1f}%, p={female_pval_q2:.4f})."
            st.success(effect_interpretation_q2)
    else:
        effect_interpretation_q2 = f"After controlling for rank, field, experience, degree, and administrative duties, there is no statistically significant difference in starting salaries between men and women (p={female_pval_q2:.4f})."
        st.info(effect_interpretation_q2)
        st.markdown(
            """
        **The absence of a significant difference suggests that the raw gap in starting salaries may be explained by legitimate factors.**
        
        This doesn't rule out other forms of bias (such as in hiring or promotion), but suggests that 
        men and women with similar qualifications receive similar starting salaries.
        """
        )

    st.markdown(
        """
    **Key Assumptions:**
    - Linear relationships between predictors and salary increases.
    - Homoscedasticity (constant variance of residuals).
    - Independence of observations.
    - Normally distributed residuals.
    
    **Notes:** 
    - The coefficient for 'sex_bool' is particularly important as it represents the estimated effect of 
      gender on salary increases after controlling for other factors. This variable is 1 for female faculty
      members and 0 for male faculty.
    """
    )

    # --- Section D: Change Over Time Model with Interaction Term ---
    st.subheader("D) Linear Regression Model for Change Over Time ")

    st.markdown(
        """
    How has the sex disparity in starting salaries changed over the years? Has it improved over time? To 
    answer these questions, we ran another regression model with a Sex x Time interaction term. This tells
    
    
    **Variables in our model:**
    - **Outcome**: Normalized monthly salary
    - **Predictor of Interest**: Sex (sex_bool, female=1, male=0)
    - **Control variables**: Field, years of experience (calculated as years since highest degree), degree
        type, administrative duties, current year, sex * current year
    """
    )
    st.markdown("**Select Model Features:**")
    int_possible_features = [
        "sex_bool",
        "yr_adj_1975",
        "sex_year_1975",
        "yrs_experience",
        "field_arts",
        "field_prof",
        "admin",
        "deg_prof",
    ]

    int_default_feats = int_possible_features.copy()
    int_selected_features = st.multiselect(
        "Features to Include:", options=int_possible_features, default=int_default_feats
    )

    if len(int_selected_features) == 0:
        st.warning("Please select at least one feature.")
        st.stop()

    int_pred = ss_filtered_yr[int_selected_features].astype(float)
    int_outcome = ss_filtered_yr[["salary_norm"]].astype(float)
    int_pred_w_intercept = sm.add_constant(int_pred)

    # Run regression
    int_model = sm.OLS(int_outcome, int_pred_w_intercept).fit()

    # Create summary dataframe
    int_model_results = pd.DataFrame(
        {
            "Variable": int_model.params.index,
            "Coefficient": int_model.params.values,
            "Std Error": int_model.bse.values,
            "p-value": int_model.pvalues.values,
            "95% CI Lower": int_model.conf_int()[0],
            "95% CI Upper": int_model.conf_int()[1],
        }
    )

    # Format numeric columns
    for col in ["Coefficient", "Std Error", "p-value", "95% CI Lower", "95% CI Upper"]:
        int_model_results[col] = int_model_results[col].round(4)

    # Add significance indicator
    int_model_results["Significance"] = int_model_results["p-value"].apply(
        lambda p: (
            "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        )
    )

    st.markdown("**Model Results**")
    st.dataframe(int_model_results)
    
       # Interpret the time coefficient
    time_coef_q2_int = int_model.params["sex_year_1975"]
    time_pval_q2_int = int_model.pvalues["sex_year_1975"]
    percent_effect_q2_int = time_coef_q2_int * 100
    time_ci_lower_q2_int = int_model.conf_int().loc["sex_year_1975"][0]
    time_ci_upper_q2_int = int_model.conf_int().loc["sex_year_1975"][1]
    percent_ci_lower_q2_int = time_ci_lower_q2_int * 100
    percent_ci_upper_q2_int = time_ci_upper_q2_int * 100
    
    if time_pval_q2_int < 0.05:
        if percent_effect_q2_int < 0:
            effect_interpretation_q2_int = f"The gap between men's and women's starting salaries after controlling for rank, field, experience, degree, and administrative duties, is widening by approximately {abs(percent_effect_q2_int):.1f}% percent per year (95% CI: {percent_ci_lower_q2_int:.1f}% to {percent_ci_upper_q2_int:.1f}%, p={time_pval_q2_int:.4f})."
            st.info(effect_interpretation_q2_int)
            st.markdown(
                """
            **This statistically significant difference suggests sex bias in starting salaries may be improving over time.**

            """
            )
        else:
            effect_interpretation_q2_int = f"The gap between men's and women's starting salaries after controlling for rank, field, experience, degree, and administrative duties, is narrowing by approximately {abs(percent_effect_q2_int):.1f}%  (95% CI: {percent_ci_lower_q2_int:.1f}% to {percent_ci_upper_q2_int:.1f}%, p={time_pval_q2_int:.4f})."
            st.success(effect_interpretation_q2_int)
    else:
        effect_interpretation_q2 = f"The gap between men's and women's starting salaries after controlling for rank, field, experience, degree, and administrative duties is not significantly changing over time (p={time_pval_q2_int:.4f})."
        st.info(effect_interpretation_q2)
        st.markdown(
            """
        **The absence of a significant difference suggests that either sex bias in starting salaries has remained steady over time, or our statistical methods do not allow us to discern a trend over time in sex bias for starting salaries.**
        """
        )

    # --- Section E: Conclusion ---
    st.subheader("E) Conclusion and Limitations")

    st.markdown(
        """
    **Summary of Findings**
    
    Our analysis used multiple approaches to investigate potential sex bias in starting faculty salaries:
    
    1. **Descriptive Statistics:** We observed raw starting salary differences between men and women. It turns
        out that women had a higher average starting salary than men, but once we adjusted for year, we
        observed about a 7% gap in starting salaries favoring men.
    
    2. **Linear Regression:** We estimated the percentage difference in starting salary attributable to sex 
        after controlling for rank, field, experience, degree, and administrative duties. We found that women
        earned about a 3% lower starting salary than men after controlling for other factors. This was
        marginally statistically significant (p=0.03)
    
    3. **Regression with Time Interaction ** We used a Sex x Time interaction term to examine whether sex-based
        disparities in starting salaries have changed over time. We did not find any statistically significant 
        change over time.
    
    **Limitations**
    
    Several important limitations should be considered:
    
    1. **Unmeasured Factors:** Our analysis couldn't control for variables not in the dataset, such as
       publication record, teaching evaluations, or grant funding, which might legitimately affect salary.
    
    2. **Causality:** Statistical associations don't necessarily imply causation. We can identify
       patterns but can't definitively establish their causes.
    
    3. **Sample Size:** Particularly when filtering by field, small sample sizes may limit statistical power
       and the reliability of estimates.
    
    """
    )

# -------------------------------------------------------------------------------
# Question 3: Has Sex Bias Existed in Salary Increases (1990-1995)
# -------------------------------------------------------------------------------

elif selected_tab == "Salary Increases (1990-1995)":
    st.header("Question 3: Has Sex Bias Existed in Salary Increases (1990-1995)?")
    st.markdown(
        """
    **Objective:**  
    Determine if there's evidence of sex bias in salary increases between 1990 and 1995.
    """
    )

    # --- Section A: Data Preparation ---
    st.subheader("A) Data Preparation")
    st.markdown(
        """
        To prepare the faculty salary data for our study:

        - We selected only the records of faculty who were working through 1990 and 1995 to focus on this five-year period.

        - We reorganized the data so each faculty member appears only once with their salaries from both years shown side by side.

        - We calculated how much each person's salary increased in both dollar amount and percentage terms.

        - We added important background information about each faculty member, including: Gender, Academic field, Education level, Years of experience (calculated using the number of years from their highest degree till 1990), Academic rank in both 1990 and 1995, Start Year when the faculty was hired,  
        
        """
    )
    summary = sba.prepare_salary_increase_data(data)
    if summary.empty:
        st.warning("No eligible records found. Check your dataset.")
        st.stop()

    field_list = ["All"] + sorted(summary["field"].dropna().unique().tolist())
    selected_field = st.selectbox("Field Filter", field_list, index=0)
    if selected_field != "All":
        summary_field = summary[summary["field"] == selected_field].copy()
    else:
        summary_field = summary.copy()
    if summary_field.empty:
        st.warning("No records after applying field filter.")
        st.stop()

    # --- Section B: Descriptive Box Plots ---
    st.subheader("B) Salary Increase Distributions by Sex")

    col1, col2 = st.columns(2)
    with col1:
        box_fig = sba.create_salary_increase_boxplot(summary_field)
        if box_fig is None:
            st.warning("Not enough data for box plot.")
        else:
            st.plotly_chart(box_fig, use_container_width=True)

    with col2:
        pct_box_fig = sba.create_pct_increase_boxplot(summary_field)
        if pct_box_fig is None:
            st.warning("Not enough data for percentage box plot.")
        else:
            st.plotly_chart(pct_box_fig, use_container_width=True)

    # --- Section C: T-Test for Salary Increases ---
    st.subheader("C) Welch's T-Test for Salary Increases")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Absolute Salary Increase Test**")
        test_abs = sba.welch_ttest_salary_increase(summary_field, "salary_increase")
        if test_abs is None:
            st.info("Not enough data for t-test (need both M and F present).")
        else:
            st.write(f"**t-Statistic**: {test_abs['t_stat']:.4f}")
            st.write(f"**p-value**: {test_abs['p_val']:.4f}")
            st.write(f"**Men's Mean Increase**: ${test_abs['mean_men']:.2f}")
            st.write(f"**Women's Mean Increase**: ${test_abs['mean_women']:.2f}")
            st.write(f"**Difference (M - F)**: ${test_abs['diff']:.2f}")
            st.write(
                f"**95% CI**: (${test_abs['ci_lower']:.2f}, ${test_abs['ci_upper']:.2f})"
            )
            st.info(test_abs["p_val_interpretation"])

    with col2:
        st.markdown("**Percentage Salary Increase Test**")
        test_pct = sba.welch_ttest_salary_increase(summary_field, "pct_increase")
        if test_pct is None:
            st.info("Not enough data for percentage t-test.")
        else:
            st.write(f"**t-Statistic**: {test_pct['t_stat']:.4f}")
            st.write(f"**p-value**: {test_pct['p_val']:.4f}")
            st.write(f"**Men's Mean % Increase**: {test_pct['mean_men']:.2f}%")
            st.write(f"**Women's Mean % Increase**: {test_pct['mean_women']:.2f}%")
            st.write(f"**Difference (M - F)**: {test_pct['diff']:.2f}%")
            st.write(
                f"**95% CI**: ({test_pct['ci_lower']:.2f}%, {test_pct['ci_upper']:.2f}%)"
            )
            st.info(test_pct["p_val_interpretation"])

    st.markdown(
        """
    - **Null Hypothesis** H₀: Mean salary increases for men and women are equal.
    - **Alternative Hypothesis** H₁: Mean salary increases differ between men and women.
    """
    )

    # --- Section D: Scatter Plots ---
    st.subheader("D) Relationship Visualizations")
    salary_comp = sba.create_salary_comparison_plot(summary_field)
    if salary_comp is None:
        st.warning("Not enough data for salary comparison plot.")
    else:
        st.plotly_chart(salary_comp, use_container_width=True)
        st.markdown(
            """
        **Interpretation:**
        - Points above the dashed line received salary increases.
        - The slope of each trend line indicates the average growth rate.
        - Different slopes by sex suggest potential bias in salary growth.

        
        **Observation:**
        - There is a clear difference in the salary distributions between men and women.
        """
        )
    col1, col2 = st.columns(2)

    with col1:
        scatter_exp = sba.create_scatter_experience_increase(summary_field)
        if scatter_exp is None:
            st.warning("Not enough data for experience scatter plot.")
        else:
            st.plotly_chart(scatter_exp, use_container_width=True)
            st.markdown(
                """
            **Interpretation:**
            - This plot shows how salary increases relate to years of experience.
            - Trend lines help identify if the experience-increase relationship differs by sex.

            
             **Observation:**
            - No linear relationship visible.
            """
            )

    with col2:
        scatter_exp = sba.create_salary_increase_vs_starting_year(summary_field)
        if scatter_exp is None:
            st.warning("Not enough data for experience scatter plot.")
        else:
            st.plotly_chart(scatter_exp, use_container_width=True)
            st.markdown(
                """
            **Interpretation:**
            - This plot shows how average salary increases relate to starting year when faculty was hired.

            
             **Observation:**
            - Slight positive linear relationship visible.
            """
            )

    col1, col2 = st.columns(2)

    with col1:
        scatter_exp = sba.create_salary_increase_by_rank(summary_field)
        if scatter_exp is None:
            st.warning("Not enough data for experience scatter plot.")
        else:
            st.plotly_chart(scatter_exp, use_container_width=True)
            st.markdown(
                """
            **Interpretation:**
            - This plot shows how salary increases relate to rank.

            **Observations:**
            - "Full" rank has slightly higher salary increase.
            """
            )

    with col2:
        scatter_exp = sba.create_pct_salary_increase_vs_initial_salary(summary_field)
        if scatter_exp is None:
            st.warning("Not enough data for experience scatter plot.")
        else:
            st.plotly_chart(scatter_exp, use_container_width=True)
            st.markdown(
                """
            **Interpretation:**
            - This plot shows how average salary increases relate to starting year when faculty was hired.

             **Observations:**
            - Average percentage increase in salary from 1990 to 1995 is higher for females who had lower starting salaries in 1990.
            """
            )

    # --- Section E: Advanced Linear Regression Model ---
    st.subheader("F) Linear Regression Models for Salary Increases")

    # Prepare Data
    X_all, y_all, feature_cols = sba.prepare_data_for_modeling(summary_field)
    if X_all.empty or len(y_all) < 5:
        st.warning("Insufficient data for modeling.")
        st.stop()

    # Feature Selection
    st.markdown("**Select Model Features:**")
    possible_features = [
        "is_female",
        "field",
        "deg",
        "rank_1990",
        "rank_1995",
        "years_experience",
        "salary_1990",
        "startyr",
    ]

    selected_features = st.multiselect(
        "Features to Include:", options=possible_features, default=possible_features
    )

    if len(selected_features) == 0:
        st.warning("Please select at least one feature.")
        st.stop()

    # Interaction Term Selection
    st.markdown("**Select Interaction Terms:** *(Optional)*")
    possible_interactions = [
        ("is_female", "salary_1990"),
        ("is_female", "field_Other"),
        ("is_female", "field_Prof"),
        ("is_female", "deg_PhD"),
        ("is_female", "deg_Prof"),
        ("is_female", "rank_1990_Assoc"),
        ("is_female", "rank_1995_Assoc"),
        ("is_female", "rank_1995_Full"),
        ("is_female", "startyr"),
    ]
    selected_interactions = st.multiselect(
        "Interaction Terms to Include:", options=possible_interactions, default=[]
    )

    # Target Variable Selection
    target_options = {
        "Absolute Increase ($)": "salary_increase",
        "Percentage Increase (%)": "pct_increase",
    }
    selected_target = st.radio("Target Variable:", options=list(target_options.keys()))
    target_col = target_options[selected_target]

    # Prepare Data Again (for selected target)
    X_all, y_all, _ = sba.prepare_data_for_modeling(summary_field, target=target_col)

    # Fit Model with Interactions
    model = sba.build_and_run_salary_model_with_interactions(
        X_all, y_all, selected_features, selected_interactions
    )

    # Extract Model Performance Metrics
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    f_statistic = model.fvalue
    f_pvalue = model.f_pvalue

    # Display Model Metrics
    st.write("### Model Performance Metrics")
    st.write(f"**R² (R-squared):** {r_squared:.4f}")
    st.write(f"**Adjusted R²:** {adj_r_squared:.4f}")
    st.write(f"**F-statistic:** {f_statistic:.2f}")
    st.write(f"**Probability (F-statistic p-value):** {f_pvalue:.4e}")

    # Display Model Coefficients
    model_results = pd.DataFrame(
        {
            "Variable": model.params.index,
            "Coefficient": model.params.values,
            "Std Error": model.bse.values,
            "p-value": model.pvalues.values,
            "95% CI Lower": model.conf_int()[0],
            "95% CI Upper": model.conf_int()[1],
        }
    )

    # Format Numeric Columns
    for col in ["Coefficient", "Std Error", "p-value", "95% CI Lower", "95% CI Upper"]:
        model_results[col] = model_results[col].round(4)

    # Add Significance Indicator
    model_results["Significance"] = model_results["p-value"].apply(
        lambda p: (
            "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        )
    )

    st.markdown("**Salary Increase Model Results**")
    st.dataframe(model_results)

    # Model Interpretation
    st.markdown(
        """
        **Model Interpretation:**
        - R-squared indicates the proportion of variance explained by the model.
        - The coefficient for 'is_female' shows the effect of gender after controlling for other variables.
        - If interaction terms are selected, the model tests whether the effect of gender depends on rank, field, or admin role.
        - A significant interaction term (p < 0.05) suggests that gender has **different effects** at different levels of the interacting variable.
        """
    )

    # Feature Importance Plot
    coef_fig = sba.plot_feature_importances(model)
    st.plotly_chart(coef_fig, use_container_width=True)

    # Model Assumptions
    st.info(
        """
        **Key Assumptions:**
        - Linear relationships between predictors and salary increases.
        - Independence of observations.
        - Homoscedasticity (constant variance of residuals).
        - Normally distributed residuals.

        **Note:** If an interaction term (e.g., `is_female × rank_1995`) is significant, 
        it means the gender effect on salary increases is **different** for different ranks.
        """
    )

    # --- Section F: Comprehensive Analysis ---
    st.subheader("F) Comprehensive Analysis & Conclusion")

    analysis_results = sba.analyze_salary_increase_bias(summary_field)

    st.markdown("### Summary of Findings:")

    # Format the summary table
    if "summary" in analysis_results:
        summary_df = pd.DataFrame(analysis_results["summary"])
        st.dataframe(summary_df)

    st.markdown("### Conclusion:")
    if "conclusion" in analysis_results:
        st.write(analysis_results["conclusion"])
    st.markdown(
        """
   The only statistically significant difference is the average percentage increase in salary from 1990 to 1995, 
   which is higher for females. This can be attributed to the **lower** starting salaries for females in 1990. 

   

   The regression analysis consistently finds that the **sex coefficient is not statistically significant** across all linear models—whether simple, multiple, or with interaction terms—indicating that sex alone does not have a meaningful impact on salary increases after accounting for rank, experience, field, and other factors.
    """
    )

    st.markdown(
        """
    **Limitations & Considerations:**
    1. The analysis doesn't account for all potential confounding variables (e.g., performance metrics, publication counts).
    2. Field-specific differences may exist but require larger sample sizes within each field.
    3. The regression model assumes linear relationships which may not fully capture complex interactions.
    """
    )


# --------------------------------------------------------------------
# Question 4: Sex Bias in Promotions from Associate to Full Professor
# --------------------------------------------------------------------

elif selected_tab == "Promotions (Associate to Full)":
    st.header("Question 4: Sex Bias in Promotions from Associate to Full Professor")

    # --------------------------------------------------------------------
    # SECTION A: Data Preparation
    # --------------------------------------------------------------------
    st.markdown(
        """
    **Objective**  
    Determine if there is a disparity in promotion rates—and time-to-promotion—by sex, focusing on 
    faculty who have reached at least the Associate rank.

    ### A) Data Preparation Steps
    To prepare our data for this analysis we:  
    1. Filter rows where `rank` is 'Assoc' or 'Full'.
    2. Identify the earliest year a faculty member reached Associate (`yr_first_assoc`) 
       and earliest year they reached Full (`yr_first_full`).
    3. Merge in demographic and academic variables (field, degree, sex, etc.).
    4. Determine if they had any administrative duties after first becoming Associate (`admin_any`).
    5. Create a binary indicator `promoted` (1 = reached Full by 1995, 0 = otherwise).
    """
    )

    summary = pa.prepare_promotion_data(data)
    if summary.empty:
        st.warning("No eligible records found. Check your dataset.")
        st.stop()

    # Field filter
    st.info(
        "Select a field to view it's promotion data, perform hypothesis tests, and perform survival analysis."
    )
    field_list = ["All"] + sorted(summary["field"].dropna().unique().tolist())
    selected_field = st.selectbox("Field Filter", field_list, index=0)

    if selected_field != "All":
        summary_field = summary[summary["field"] == selected_field].copy()
    else:
        summary_field = summary.copy()

    if summary_field.empty:
        st.warning("No records found after applying field filter.")
        st.stop()

    # --------------------------------------------------------------------
    # SECTION B: Descriptive Analysis — Bar Chart
    # --------------------------------------------------------------------
    st.markdown(
        """
    ### B) Descriptive Analysis

    **Promotion Counts by Sex**  
    A quick look at how many men vs. women in this subset have successfully made the jump 
    from Associate to Full Professor.
    """
    )

    st.subheader("Promotion Counts by Sex")
    bar_fig = pa.create_promotion_bar_chart(summary_field)
    if bar_fig is None:
        st.warning("Not enough data for bar chart.")
    else:
        st.plotly_chart(bar_fig, use_container_width=True)

    # Show cross-tab results
    ztest_output = pa.two_proportion_z_test(summary_field)
    if ztest_output is not None:
        _, cross_tab = ztest_output
        st.write("**Counts & Proportions**")
        st.dataframe(cross_tab, hide_index=True, use_container_width=True)
    else:
        st.info("Could not compute cross-tab; possibly only one sex present.")

    # --------------------------------------------------------------------
    # SECTION C: Two-Proportion Z-Test
    # --------------------------------------------------------------------
    st.markdown("### C) Statistical Test of Promotion Rates")
    st.subheader(
        "Two-Proportion Z-Test for Promotion Rates, at Significance Level = 0.05"
    )
    st.markdown(
        """
    **Hypotheses**  
    - **Null (H₀)**: Promotion rates (Associate \u2192 Full) are the same for men and women.  
    - **Alternative (H₁)**: Promotion rates differ by sex.

    The two-proportion z-test checks if the observed difference in promotion rates is 
    statistically significant.
    """
    )

    if ztest_output is None:
        st.info("Not enough data to run a two-proportion z-test (need both M and F).")
    else:
        test_result, _ = ztest_output
        result = {
            "z_stat": test_result["z_stat"],
            "p_val": test_result["p_val"],
            "proportion_men": test_result["proportion_men"],
            "proportion_women": test_result["proportion_women"],
            "diff": test_result["diff"],
            "ci_lower": test_result["ci_lower"],
            "ci_upper": test_result["ci_upper"],
        }
        # Prepare for display
        results_df = pd.DataFrame(
            {
                "Statistic": [
                    "Z-Statistic",
                    "P-value",
                    "Men's Promotion Proportion",
                    "Women's Promotion Proportion",
                    "Difference (M - F)",
                    "95% CI",
                ],
                "Value": [
                    f"{result['z_stat']:.5f}",
                    f"{result['p_val']:.5f}",
                    f"{result['proportion_men']:.5f}",
                    f"{result['proportion_women']:.5f}",
                    f"{result['diff']:.5f}",
                    f"({result['ci_lower']:.5f}, {result['ci_upper']:.5f})",
                ],
            }
        )
        st.dataframe(results_df, hide_index=True, use_container_width=True)

        # Display test interpretations
        st.info(test_result["p_val_interpretation"])
        st.info(test_result["ci_interpretation"])

    st.markdown("---")

    # --------------------------------------------------------------------
    # SECTION D: Kaplan-Meier Survival Analysis
    # --------------------------------------------------------------------
    st.markdown("### D) Survival Analysis of Promotion")
    st.subheader("Kaplan-Meier Analysis of Time-to-Promotion by Sex")
    st.markdown(
        """
    **Why Kaplan-Meier?**  
    - We treat the time from the first year an individual holds the Associate rank 
      to the year they become Full (or 1995 if not promoted) as a 'survival' period.  
    - Kaplan-Meier curves estimate how quickly faculty transition to Full rank.
    """
    )

    surv_fig = pa.create_survival_analysis(summary_field)
    if surv_fig is None:
        st.warning(
            "Not enough data or only one sex group available for Kaplan-Meier analysis."
        )
    else:
        st.plotly_chart(surv_fig, use_container_width=True)

        st.info(
            """
            **Interpretation**
            - A *steep drop* means that group is promoted more rapidly.
            - If one group’s curve remains higher for longer, they tend to stay in the Associate rank longer than the other group."""
        )

    st.markdown("---")

    # --------------------------------------------------------------------
    # SECTION E: Logistic Regression
    # --------------------------------------------------------------------
    st.markdown("### E) Logistic Regression")
    st.subheader("Logistic Regression: Identifying Promotion Factors")

    X_all, y_all, summary_all = pa.prepare_data_for_modeling(summary_field)
    if X_all.empty or y_all.nunique() < 2:
        st.warning("Insufficient variation in data for modeling.")
        st.stop()

    st.markdown(
        """
    **Model Approach**  
    - **Dependent variable**: 'promoted' (1 if promoted to Full by 1995, 0 otherwise).  
    - **Predictors**: Choose from sex, field, degree type, administrative role, 
      years since degree, etc.
    """
    )

    possible_features = [
        "sex",
        "field",
        "deg_type",
        "admin_any",
        "yrdeg",
        "startyr",
        "salary",
    ]
    selected_features = st.multiselect(
        "Select Features for Logistic Regression Model:",
        options=possible_features,
        default=possible_features,
    )

    if len(selected_features) == 0:
        st.warning("Select at least one feature to build the model.")
        st.stop()

    pipe, preds, probs = pa.build_and_run_logreg_model(X_all, y_all, selected_features)
    accuracy = (preds == y_all).mean()
    st.write(f"**Model Accuracy (Training Data):** {accuracy:.3f}")

    coef_fig = pa.plot_feature_importances_logreg(pipe, X_all, selected_features)
    st.plotly_chart(coef_fig, use_container_width=True)
    st.info(
        """
        **How to Read the Coefficients:**
        - Each bar represents the log-odds coefficient for that predictor or category.  
        - A **positive** coefficient indicates a predictor is associated with *higher odds* of being promoted.  
        - A **negative** coefficient means the predictor is associated with *lower odds* of promotion.  
        - The **magnitude** (length of the bar) reflects how strong that association is.
        """
    )

    # --------------------------------------------------------------------
    # SECTION F: Findings & Conclusion
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Summary of Findings & Conclusions")

    st.markdown(
        """
    **Key Takeaways:**

    1. **Promotion Rates (Descriptive & Two-Proportion Test)**  
       - Men generally show higher promotion counts and proportions from Associate to Full, 
         though this can vary by field.  
       - In a two-proportion z-test, the difference in promotion rates often appears statistically 
         significant (p < 0.05) in many fields, suggesting possible sex-based disparities.  

    2. **Time-to-Promotion (Kaplan-Meier Curves)**  
       - Men frequently appear to reach Full rank earlier (steeper survival curve decline).  
       - Women may catch up over longer periods, but the initial gap suggests men receive Full 
         promotions sooner, on average.

    3. **Logistic Regression**  
       - When controlling for field, degree type, administrative roles, and other variables, 
         **being male** often remains positively associated with promotion likelihood.  
       - Additional factors that increase promotion odds include:  
         - Holding a PhD (vs. Other degrees)  
         - Being in a Professional field (vs. Arts or Other)  
         - Having more years since highest degree, which reflects accumulated experience  

    **Interpretations & Limitations**  
    - Our analysis indicates that, across most academic fields, men tend to be promoted more 
      rapidly and at higher rates. This pattern is *suggestive* of potential sex bias.  
    - We don’t have data on factors like publication counts or teaching evaluations, 
      which might also influence promotion.  
    - The dataset ends in 1995, so **any changes in policies or practices after 1995** aren’t captured here.  
    - **Correlation ≠ Causation**: A significant difference does not *prove* discrimination 
      but does highlight where further investigation may be warranted.

    **Overall Conclusion**:  
    Based on the data and methods used, there is evidence of a sex-related gap in 
    promotions from Associate to Full Professor. Men appear more likely to be promoted 
    and often achieve promotion earlier, on average, than women with otherwise similar 
    characteristics.
    """
    )

    st.markdown(
        """
    ---
    **Further Analyses**  
    - **Cox Proportional Hazards Model**: Another survival analysis approach that can handle 
      covariates more flexibly. If we had time and the right data structure, a Cox model could 
      help confirm whether sex remains a strong predictor of promotion timing.  
    - **Department-Level Breakdowns**: Larger sample sizes within each department (if available) 
      could reveal whether certain departments show more/less disparity.  
    - **Interaction Terms**: Additional interactions (e.g., sex × field) might illuminate 
      whether specific fields experience greater disparities.  
    """
    )

elif selected_tab == "Summary of Findings":
    st.header("Summary of Findings")

    # ----------------------------------------------------------------------------
    # Question 1: Sex Bias in 1995
    # ----------------------------------------------------------------------------
    st.subheader("Does sex bias exist in the university in 1995?", divider="violet")
    st.markdown(
        """
    **Key Observations**  
    - **Raw Salary Gap**: In 1995, men generally had higher average salaries than women.  
    - **Adjusted Analysis**: After accounting for rank, field, years of experience, and administrative duties, 
      the difference in salaries by sex often became small or statistically nonsignificant.  
    - **Interpretation**: While men’s average salary was higher, the bulk of this disparity can 
      largely be explained by legitimate factors. That said, a small residual gap does persist in some 
      models, suggesting possible—but not conclusive—evidence of sex bias.  
    """
    )

    # ----------------------------------------------------------------------------
    # Question 2: Starting Salaries
    # ----------------------------------------------------------------------------
    st.subheader(
        "Are starting salaries different between male and female faculty?",
        divider="violet",
    )
    st.markdown(
        """
    **Key Observations**  
    - **Raw Differences**: Historically, there have been differences in the average starting salaries of 
      men and women, sometimes favoring men and sometimes favoring women, depending on the hiring year 
      and field.  
    - **Trends Over Time**: In certain years, women appeared to have lower raw starting salaries compared 
      to men, but these differences often narrowed once we controlled for field, degree, and experience. 
      In some fields, women actually started at salaries comparable to, or slightly above, men.  
    - **Regression Insights**: When using a model that factored in rank at hire, field, degree type, and 
      years since degree, the sex effect on starting salary was often not statistically significant—or 
      was marginally significant—indicating that the observed gap can be largely explained by these 
      legitimate factors.
    """
    )

    # ----------------------------------------------------------------------------
    # Question 3: Salary Increases (1990–1995)
    # ----------------------------------------------------------------------------
    st.subheader(
        "Have salary increases between 1990 and 1995 been influenced by sex?",
        divider="violet",
    )
    st.markdown(
        """
    **Key Observations**  
    - **Absolute Increases**: Direct comparisons did not reveal a statistically significant difference 
      in the dollar amount of salary increases between men and women.  
    - **Percentage Increases**: Interestingly, women showed a higher percentage increase, which can 
      largely be attributed to their lower baseline salaries in 1990.  
    - **Regression Analysis**: Across simple and multiple regression models (with or without interaction 
      terms), the coefficient for sex was consistently not significant. This suggests that other factors, 
      such as experience, rank changes, or field, play a bigger role in determining salary growth than 
      does sex alone.  
    - **Conclusion**: We do not find robust evidence of sex bias in salary raises during 1990–1995 once 
      baseline salary levels and other covariates are taken into account.
    """
    )

    # ----------------------------------------------------------------------------
    # Question 4: Promotions (Associate to Full)
    # ----------------------------------------------------------------------------
    st.subheader(
        "Has sex bias existed in granting promotions from Associate to Full Professor?",
        divider="violet",
    )
    st.markdown(
        """
    **Key Observations**  
    - **Higher Promotion Rates for Men**: In most fields, men were statistically more likely to be promoted 
      and often attained Full rank earlier than women—though the magnitude of this disparity varied.  
    - **Field Differences**: The Arts field sometimes showed less disparity, suggesting that the gap is 
      not uniform across disciplines.  
    - **Logistic Regression**: Factors such as having a PhD, working in a Professional field (e.g., 
      Business, Engineering), or accruing more years since highest degree all significantly increased 
      promotion odds. Being female or in the Arts field, however, tended to reduce the likelihood or speed 
      of promotion in many models.  
    - **Interpretation**: These findings raise concerns about possible sex bias in the promotion process. 
      While this does not confirm intentional discrimination, it does highlight a systematic pattern 
      whereby men achieve Full rank more quickly and at higher rates overall.
    """
    )

    # ----------------------------------------------------------------------------
    # Final Remarks and Disclaimers
    # ----------------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        """
    ### Overall Conclusions and Considerations

    **Summary**  
    - **1995 Salaries**: Men generally earned more, but much of the gap can be explained by rank, field, 
      and experience. Some small residual difference might be attributed to sex.  
    - **Starting Salaries**: Minimal or inconsistent evidence of ongoing sex-based discrepancies once 
      field and degree are accounted for.  
    - **Salary Increases**: No significant sex effect in absolute or regression-based analyses. A higher 
      percentage increase for women reflects lower starting salaries rather than targeted bias.  
    - **Promotions**: A consistent pattern shows men promoted at higher rates and sometimes faster, 
      pointing to a potential bias that remains even after controlling for relevant covariates.

    **Limitations**  
    - **Missing Factors**: Data on teaching evaluations, publication records, grant funding, or departmental 
      culture might explain some salary or promotion differences.  
    - **Historical Context**: The dataset ends in 1995 and may not reflect recent changes in policy or culture.  
    - **Rank Effects**: Using rank as a control variable can obscure pre-rank disparities (e.g., if sex-biased 
      hiring practices led to more men achieving higher ranks earlier).  
    - **Causality**: Statistical associations do not by themselves prove discrimination or intentional bias.

    **Where to Go Next**  
    - **Department-Level Studies**: Analyzing promotion and salary data department by department may uncover 
      more nuanced patterns.  
    - **Longitudinal Tracking**: More detailed time-to-promotion analyses (like a Cox proportional hazards 
      model) could strengthen insights into how quickly men vs. women progress.  
    - **Policy Review**: If residual gaps persist, institutions may consider more transparent policies for 
      hiring, salary setting, and promotion to address potential inequities.

    Overall, the results suggest that while much of the observed disparity in salaries and promotions can be 
    explained by legitimate factors, there remain persistent patterns that favor men, especially in promotions. 
    Understanding and addressing these patterns will likely require deeper investigation into broader aspects 
    of academic careers and institutional practices.
    """
    )

    st.markdown("---")
