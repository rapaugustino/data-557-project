# DATA-557-Project

## Overview

This repository contains an applied statistics project that investigates potential sex-based disparities in faculty salaries and promotions at a U.S. university. It includes:

- **app.py** – Main Streamlit application  
- **promotions_analysis.py** – Analysis functions/classes for faculty promotions  
- **salary_bias_analysis.py** – Functions/classes related to salary data and potential bias  
- **salary.txt** – Faculty salary dataset (1976–1995, excluding medical school)  
- **requirements.txt** – Python dependencies  
- **.streamlit/config.toml** – Streamlit configuration (theme, layout)

---

## Key Research Questions

1. **1995 Salaries**  
   Investigate if there is evidence of sex bias in the most recent salary records.  
2. **Starting Salaries**  
   Compare entry-level pay for men and women over time.  
3. **Salary Increases (1990–1995)**  
   Determine whether salary growth rates differ by sex.  
4. **Promotions (Associate to Full)**  
   Examine whether one sex has a higher or quicker promotion rate.

---

## Project Highlights

- **Statistical Methods**: T-tests, two-proportion z-tests, Kaplan-Meier analysis, OLS and logistic regression.  
- **Data Handling**: Pandas for data cleaning and merging.  
- **Visualizations**: Plotly for bar charts, box plots, line plots, and survival curves.  
- **Interactive App**: Streamlit to create a dynamic, filterable dashboard for exploring results.

---

## How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/rapugustino/DATA-557-Project.git
   cd DATA-557-Project
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**
   ```bash
   streamlit run app.py
   ```
   - The app will open in your default web browser at `http://localhost:8501`.

4. **Access the Dashboard**
   Navigate to the dashboard in your web browser and explore the results.   


## Acknowledgments
- **Data & Research Questions**: Provided by Professor Scott Emerson.  
- **Project Context**: Developed within the framework of an applied statistics course, guided by Professor Katie Wilson’s instruction.
- **Team Collaboration**: Collaboration between team members each contributing to data preparation, analysis, and interpretation. The team includes [Kyle Cullen Bretherton](https://github.com/kylebreth), [Aravindh Manavalan](https://github.com/aravindh28), [Richard Pallangyo](https://github.com/rapugustino), [Akshay Ravi](https://github.com/akshayravi13), and [Vijay Balaji S](https://github.com/SVijayB).