#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Load Dataset
data=pd.read_csv(r"C:\Users\rozin\Downloads\DataAnalyst.csv")


# In[3]:


data.head()


# In[4]:


print(data.shape)
print(data.info())


# ### Standardize column Name

# In[5]:


data.columns=data.columns.str.strip().str.replace(' ','_')


# In[6]:


data


# ### Remove Duplicate Rows

# In[7]:


print("Duplicate rows:",data.duplicated().sum())
data=data.drop_duplicates()


# ### Handle Missing Values

# In[8]:


data.isnull().sum()


# In[9]:


data["Company_Name"]=data["Company_Name"].fillna("Unknown")
data.isnull().sum()


# In[10]:


data["Company_Name"].unique()


# In[11]:


data["Sector"].unique()


# In[12]:


data["Type_of_ownership"].unique()


# In[13]:


data["Rating"].unique()


# In[14]:


data["Rating"]=data["Rating"].replace(-1,np.nan)
data["Rating"]=data["Rating"].fillna(data['Rating'].median())


# In[15]:


data["Easy_Apply"].unique()


# In[16]:


categorial_cols=["Company_Name","Industry","Sector","Type_of_ownership"]
data[categorial_cols]=data[categorial_cols].fillna("Unknown")


# In[17]:


object_cols=data.select_dtypes(include="object").columns


# In[18]:


data[object_cols]=data[object_cols].replace([-1,'-1'],np.nan)
data[object_cols]=data[object_cols].fillna("Unknown")


# In[19]:


data["Type_of_ownership"].unique()


# In[20]:


data["Founded"]=data["Founded"].replace(-1,np.nan)
data["Founded"]=data["Founded"].fillna(data['Founded'].median())


# In[21]:


data


# In[22]:


data["Founded"]=data["Founded"].astype(int)


# ### Clean Salary Column

# In[23]:


data["Salary_Estimate"].unique()


# In[24]:


data["Salary_Estimate"]=data["Salary_Estimate"].replace("Unknown",np.nan)
data=data.dropna(subset=['Salary_Estimate'])
data["Salary_Estimate"]=data["Salary_Estimate"].str.replace('[^0-9\-]','',regex=True)
salary_split=data["Salary_Estimate"].str.split('-',expand=True)

data["Min_Salary"]=pd.to_numeric(salary_split[0],errors='coerce')

data["Max_Salary"]=pd.to_numeric(salary_split[1],errors='coerce')

data=data.dropna(subset=['Min_Salary','Max_Salary'])
data["Avg_Salary"]=(data["Min_Salary"]+data["Max_Salary"])/2


# In[25]:


data


# ## Exploratory Data Analysis

# In[26]:


#salary distribution

plt.figure(figsize=(10,6))
sns.histplot(data['Avg_Salary'],bins=20,kde=True)
plt.title("Average Salary Distribution")
plt.show()


# In[27]:


#Company Rating vs Salary
plt.figure(figsize=(8,6))
sns.scatterplot(x='Rating',y='Avg_Salary',data=data)
plt.title("Company Rating vs Salary")
plt.show()


# In[28]:


sns.set(style="whitegrid")


# In[29]:


#Top 10 job Titles
top_jobs=data['Job_Title'].value_counts().head(10)
plt.figure(figsize=(10,5))

sns.barplot(x=top_jobs.values,y=top_jobs.index)
plt.title('Top 10 Job Title')
plt.xlabel('Number of Jobs')
plt.ylabel('Job Title')

plt.show()


# In[30]:


#Avg Salary by job Title



# In[31]:


plt.figure(figsize=(12,6))

sns.barplot(
    data=data,
    x='Avg_Salary',
    y='Job_Title',
    order=data['Job_Title'].value_counts().head(10).index,
    palette='tab10'
)

plt.xlabel('Average Salary ($)')
plt.ylabel('Job Title')
plt.title('Average Salary by Job Title')
plt.show()


# In[32]:


#Salary Trends by Location
location_salary = (
    data.groupby('Location')['Avg_Salary']
    .mean()
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
)

plt.figure(figsize=(10,5))
sns.barplot(data=location_salary, x='Avg_Salary', y='Location')
plt.title('Average Salary by Location')
plt.xlabel('Average Salary')
plt.ylabel('Location')
plt.show()


# In[33]:


#Company_size - Avg salary
company_size_salary = (
    data.groupby('Size')['Avg_Salary']
    .mean()
    .reset_index()
)

plt.figure(figsize=(18,8))
sns.barplot(data=company_size_salary, x='Size', y='Avg_Salary')
plt.title('Average Salary by Company Size')
plt.xlabel('Company Size')
plt.ylabel('Average Salary')
plt.show()


# In[34]:


#top 20 Types of Ownership
ownership = (
    data['Type_of_ownership']
    .value_counts()
    .head(20)
    .reset_index()
)
ownership.columns = ['Type_of_ownership', 'Count']

plt.figure(figsize=(10,6))
sns.barplot(data=ownership, x='Count', y='Type_of_ownership')
plt.title('Top 20 Types of Ownership')
plt.xlabel('Job Count')
plt.ylabel('Ownership Type')
plt.show()


# In[35]:


#top 15 Sector for data analysis
top_sectors = (
    data['Sector']
    .value_counts()
    .head(15)
    .reset_index()
)
top_sectors.columns = ['Sector', 'Count']

plt.figure(figsize=(8,5))
sns.barplot(data=top_sectors, x='Count', y='Sector')
plt.title('Top 10 Sectors with Data Analyst Jobs')
plt.xlabel('Job Count')
plt.ylabel('Sector')
plt.show()


# In[ ]:





# ## Skill Extraction from job Description

# In[36]:


skills=['python','excel','sql','power bi','tableau']

for skill in skills:
    data[skill.upper()]=data['Job_Description'].str.contains(
    skill,case=False,na=False).astype(int)


# In[37]:


#skill score
data['Tech_Skills']=data[['PYTHON','EXCEL','SQL','POWER BI','TABLEAU']].sum(axis=1)


# ## Location Feature Engineering

# In[38]:


data['City']=data['Location'].str.split(',',expand=True)[0]
data['State']=data['Location'].str.split(',',expand=True)[1]


# In[39]:


data.head()


# ## Correlation Analysis

# In[40]:


plt.figure(figsize=(10,8))
sns.heatmap(data[['Rating','Avg_Salary','Tech_Skills','Founded']].corr(),
           annot=True,cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# In[ ]:





# ##  Model Building

# In[41]:


features= ['Rating','Tech_Skills','Founded']
X=data[features]
y = data['Avg_Salary']


# In[42]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# In[43]:


from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score

dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train, y_train)

y_dummy = dummy.predict(X_test)

print("Dummy MAE:", mean_absolute_error(y_test, y_dummy))
print("Dummy R2:", r2_score(y_test, y_dummy))


# In[44]:


y.describe()


# In[45]:


X.nunique()


# In[46]:


data['Avg_Salary'].unique


# In[47]:


from sklearn.linear_model import LinearRegression

X_simple = data[['Founded']]  # only one column
y = data['Avg_Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))


# In[48]:


from sklearn.metrics import mean_absolute_error, r2_score

y_pred_lr = lr.predict(X_test)

print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred_lr))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))


# Conclusion
# This project analyzed data analyst job postings to identify salary patterns,required skills, and market trends.
# Exploratory analysis showed significant variation in salaries across job roles, locations, and sectors,while company-related features had limited impact.
# A baseline and linear regression model were used to predict average salary, both performing close to the baseline with R² values near zero.
# This indicates that salary is influenced more by role and location than by basic company metadata.
# Overall, the project highlights the importance of exploratory data analysis and realistic model evaluation in understanding complex job market data.

# In[ ]:




