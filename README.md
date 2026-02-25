# Data-analyst-job-market-analysis
📊 Data Analyst Job Market Analysis
📌 Project Overview
This project analyzes over 2,000 Data Analyst job postings to understand salary trends, required skills, and overall job market patterns.
The objective is to perform end-to-end data analysis, including data cleaning, exploratory data analysis (EDA), feature engineering, and basic salary prediction modeling.
🎯 Problem Statement
Analyze salary distribution across job roles and locations
Identify key technical skills required by employers
Understand sector and company-based trends
Attempt salary prediction using basic regression models
📂 Dataset Description
The dataset contains job postings with the following features:
Job Title
Salary Estimate
Location
Company Rating
Company Size
Industry & Sector
Job Description
🧹 Data Cleaning & Preprocessing
Removed duplicates and handled missing values
Cleaned salary estimates and created Average Salary feature
Extracted technical skills such as Python and Excel from job descriptions
Converted relevant features into numerical format for modeling
📊 Exploratory Data Analysis (EDA)
Key insights from analysis:
Salaries vary significantly by job title
Certain locations offer higher average salaries
Sector-wise distribution shows concentration in specific industries
Company metadata (rating, size, founded year) has limited impact on salary
Data visualization was performed using Matplotlib and Seaborn.
🤖 Modeling Approach
To predict average salary:
Baseline Model – Dummy Regressor
Linear Regression Model
📈 Model Evaluation Metrics:
Mean Absolute Error (MAE)
R² Score
Both models performed close to the baseline (R² ≈ 0), indicating weak relationships between salary and the selected numerical features.
📌 Key Learning
The modeling results show that salary prediction using basic company metadata is difficult. Salary appears to be influenced more by:
Job role
Location
Market demand
This project highlights the importance of exploratory analysis and realistic model evaluation over forcing predictive performance.
🛠 Tools & Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
📎 Conclusion
This project demonstrates a complete data analysis workflow, from raw data cleaning to insight generation and modeling. While predictive performance was limited, the exploratory insights provide meaningful understanding of the Data Analyst job market.
