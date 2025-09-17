# 📊 Comprehensive Dataset Collection

<div align="center">

![Datasets](https://img.shields.io/badge/Datasets-Collection-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-Open%20Source-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.7+-yellow?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Compatible-red?style=for-the-badge&logo=pandas)

_A curated collection of diverse datasets for data science, machine learning, and analytics projects_

</div>

---

## 🎯 Overview

This repository contains a comprehensive collection of **15+ datasets** spanning various domains including healthcare, entertainment, transportation, demographics, and more. Each dataset is carefully organized and ready for analysis, making it perfect for:

- 🔬 **Data Science Projects**
- 🤖 **Machine Learning Experiments**
- 📈 **Statistical Analysis**
- 🎓 **Educational Purposes**
- 💼 **Business Analytics**

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [📊 Dataset Categories](#-dataset-categories)
- [🔥 Featured Datasets](#-featured-datasets)
- [📁 Dataset Details](#-dataset-details)
- [🚀 Quick Start](#-quick-start)
- [💡 Usage Examples](#-usage-examples)
- [📈 Data Insights](#-data-insights)
- [🛠️ Tools & Libraries](#️-tools--libraries)
- [📝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 📊 Dataset Categories

<div align="center">

| Category              | Count | Description                          |
| --------------------- | ----- | ------------------------------------ |
| 🏥 **Healthcare**     | 4     | Medical data, diabetes, health camps |
| 🚗 **Transportation** | 3     | Cars, traffic, police data           |
| 🏠 **Real Estate**    | 1     | Housing market data                  |
| 🌍 **Demographics**   | 2     | Census, population data              |
| 📚 **Education**      | 3     | Udemy courses, student performance   |
| 🎬 **Entertainment**  | 2     | Netflix content, trending data       |
| 🦠 **Pandemic**       | 1     | COVID-19 statistics                  |
| 🌸 **Science**        | 1     | Iris flower classification           |
| ⚓ **Historical**     | 1     | Titanic passenger data               |
| 💼 **Business**       | 1     | Employee attrition data              |

</div>

---

## 🔥 Featured Datasets

### 🏥 Healthcare Analytics

- **Diabetes Dataset** - Comprehensive health metrics for diabetes prediction
- **Health Camp Data** - Multi-camp attendance and patient profiles

### 🚗 Transportation Intelligence

- **Car Dataset** - Vehicle specifications and market analysis
- **Police Data** - Traffic incidents and law enforcement statistics

### 🎬 Entertainment Insights

- **Netflix Dataset** - Content analysis and viewing patterns
- **Trending Data** - Social media and content trends

---

## 📁 Dataset Details

<details>
<summary><b>🩺 Health & Medical Datasets</b></summary>

### 1. Diabetes Dataset (`diabetes.csv`)

- **Size**: 100,000+ records
- **Features**: Gender, Age, Hypertension, Heart Disease, BMI, HbA1c Level, Blood Glucose
- **Target**: Diabetes prediction (Binary classification)
- **Use Cases**: Predictive modeling, health risk assessment

### 2. Health Camp Dataset (`Health_Care_Dataset/`)

- **Components**: Patient profiles, camp details, attendance records
- **Size**: Multiple files with 10,000+ records
- **Features**: Demographics, health metrics, camp participation
- **Use Cases**: Healthcare analytics, patient behavior analysis

</details>

<details>
<summary><b>🚗 Transportation & Mobility</b></summary>

### 3. Cars Dataset (`Project_2_Cars_Dataset.csv`)

- **Features**: Make, model, year, price, specifications
- **Use Cases**: Price prediction, market analysis, feature comparison

### 4. Police Data (`Project_3_Police Data.csv`)

- **Content**: Incident reports, traffic violations, enforcement data
- **Use Cases**: Crime analysis, traffic pattern studies

</details>

<details>
<summary><b>🏠 Real Estate & Demographics</b></summary>

### 5. Housing Data (`Project_5_Housing_Data.csv`)

- **Features**: Property details, prices, location metrics
- **Use Cases**: Price prediction, market trends, investment analysis

### 6. Census 2011 (`Project_6_Census_2011.csv`)

- **Content**: Demographic statistics, population distribution
- **Use Cases**: Demographic analysis, policy planning

</details>

<details>
<summary><b>🎓 Education & Learning</b></summary>

### 7. Udemy Dataset (`Project_7_Udemy_Dataset.csv`)

- **Features**: Course details, ratings, pricing, enrollment
- **Use Cases**: Course recommendation, pricing strategy

### 8. Student Performance (`student-pass-fail-data.csv`)

- **Content**: Academic performance metrics
- **Use Cases**: Educational analytics, performance prediction

</details>

<details>
<summary><b>🎬 Entertainment & Media</b></summary>

### 9. Netflix Dataset (`Project_8_Netflix_Dataset.csv`)

- **Features**: Content type, ratings, release dates, genres
- **Use Cases**: Content analysis, recommendation systems

### 10. Trending Data (`Trending/trending.csv`)

- **Content**: Social media trends, viral content metrics
- **Use Cases**: Trend analysis, social media insights

</details>

<details>
<summary><b>🔬 Classic ML Datasets</b></summary>

### 11. Iris Dataset (`IRIS.csv`)

- **Size**: 150 records
- **Features**: Sepal/Petal dimensions
- **Target**: Species classification (3 classes)
- **Use Cases**: Classification tutorials, algorithm comparison

### 12. Titanic Dataset (`Titanic_dataset.csv`)

- **Size**: 400+ records
- **Features**: Passenger details, ticket info, survival status
- **Use Cases**: Survival prediction, feature engineering

</details>

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Basic Usage

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load any dataset
df = pd.read_csv('diabetes.csv')

# Quick overview
print(df.info())
print(df.describe())
print(df.head())
```

---

## 💡 Usage Examples

### 🔍 Exploratory Data Analysis

```python
# Diabetes Dataset Analysis
diabetes_df = pd.read_csv('diabetes.csv')

# Distribution of diabetes cases
plt.figure(figsize=(10, 6))
sns.countplot(data=diabetes_df, x='diabetes')
plt.title('Distribution of Diabetes Cases')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(diabetes_df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()
```

### 🤖 Machine Learning Pipeline

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Prepare data
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes']

# Handle categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 📊 Data Visualization

```python
# Netflix content analysis
netflix_df = pd.read_csv('Project_8_Netflix_Dataset.csv')

# Content type distribution
plt.figure(figsize=(10, 6))
netflix_df['type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Netflix Content Distribution')
plt.show()

# Release year trends
plt.figure(figsize=(12, 6))
netflix_df['release_year'].hist(bins=30, edgecolor='black')
plt.title('Netflix Content Release Year Distribution')
plt.xlabel('Release Year')
plt.ylabel('Number of Titles')
plt.show()
```

---

## 📈 Data Insights

### 🎯 Key Statistics

| Dataset  | Records  | Features | Missing Values | Target Variable |
| -------- | -------- | -------- | -------------- | --------------- |
| Diabetes | 100,000+ | 9        | Minimal        | Binary          |
| Iris     | 150      | 5        | None           | Multi-class     |
| Titanic  | 400+     | 12       | Moderate       | Binary          |
| Netflix  | Varies   | 10+      | Low            | None            |

### 📊 Data Quality Overview

```python
# Data quality assessment function
def assess_data_quality(df, dataset_name):
    print(f"\n=== {dataset_name} Quality Assessment ===")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Data types: {df.dtypes.nunique()} unique types")
    return df.info()
```

---

## 🛠️ Tools & Libraries

### Recommended Stack

- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Machine Learning**: `scikit-learn`, `tensorflow`, `pytorch`
- **Statistical Analysis**: `scipy`, `statsmodels`
- **Jupyter Environment**: `jupyter notebook`, `jupyter lab`

### Installation Guide

```bash
# Essential packages
pip install pandas numpy matplotlib seaborn

# Machine Learning
pip install scikit-learn tensorflow

# Advanced visualization
pip install plotly dash

# Statistical analysis
pip install scipy statsmodels

# Jupyter environment
pip install jupyter jupyterlab
```

---

## 📊 Project Structure

```
📁 Datasets/
├── 📄 README.md                    # This comprehensive guide
├── 🩺 diabetes.csv                 # Primary diabetes dataset
├── 🩺 diabetes1.csv                # Secondary diabetes data
├── 🌸 IRIS.csv                     # Classic iris classification
├── ⚓ Titanic_dataset.csv          # Historical passenger data
├── 🧪 testdata.csv                 # Testing dataset
├── 📊 CleaneD_testdata_File.csv    # Cleaned test data
├── 🎓 student-pass-fail-data.csv   # Academic performance
├── 📁 Health_Care_Dataset/         # Comprehensive health data
│   ├── 👥 Patient_Profile.csv
│   ├── 🏥 Health_Camp_Detail.csv
│   ├── 📊 *_Health_Camp_Attended.csv
│   └── 📁 Cleaned_Data/
├── 📁 Trending/                    # Social media trends
├── 📁 Udmey Data/                  # Educational platform data
└── 📊 Project_*_*.csv              # Thematic project datasets
```

---

## 🎯 Use Case Examples

### 🏥 Healthcare Analytics

```python
# Diabetes risk assessment model
def diabetes_risk_model():
    df = pd.read_csv('diabetes.csv')
    # Feature engineering and model training
    return trained_model

# Health camp effectiveness analysis
def analyze_health_camps():
    camp_data = pd.read_csv('Health_Care_Dataset/Health_Camp_Detail.csv')
    attendance = pd.read_csv('Health_Care_Dataset/First_Health_Camp_Attended.csv')
    # Analysis code here
```

### 🚗 Transportation Intelligence

```python
# Car price prediction
def predict_car_price():
    cars_df = pd.read_csv('Project_2_Cars_Dataset.csv')
    # Price prediction model

# Traffic pattern analysis
def analyze_police_data():
    police_df = pd.read_csv('Project_3_Police Data.csv')
    # Traffic and crime pattern analysis
```

### 🎬 Entertainment Insights

```python
# Netflix content recommendation
def netflix_recommender():
    netflix_df = pd.read_csv('Project_8_Netflix_Dataset.csv')
    # Recommendation algorithm

# Trending content predictor
def predict_trending():
    trends_df = pd.read_csv('Trending/trending.csv')
    # Trend prediction model
```

---

## 🔄 Data Processing Workflows

### Standard Pipeline

```python
class DataProcessor:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)

    def clean_data(self):
        # Remove duplicates
        self.df = self.df.drop_duplicates()

        # Handle missing values
        self.df = self.df.fillna(self.df.mean(numeric_only=True))

        return self

    def feature_engineering(self):
        # Create new features
        # Encode categorical variables
        return self

    def split_data(self, target_column):
        # Train-test split logic
        return X_train, X_test, y_train, y_test
```

---

## 📝 Contributing

We welcome contributions! Here's how you can help:

### 🤝 How to Contribute

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/new-dataset`
3. **Add** your dataset with proper documentation
4. **Commit** changes: `git commit -am 'Add new healthcare dataset'`
5. **Push** to branch: `git push origin feature/new-dataset`
6. **Submit** a Pull Request

### 📋 Contribution Guidelines

- Include dataset description and source
- Provide data dictionary/schema
- Add usage examples
- Ensure data quality and cleanliness
- Follow naming conventions

---

## 📜 Dataset Sources & Credits

- **Diabetes Dataset**: Healthcare research compilation
- **Iris Dataset**: R.A. Fisher's classic botanical study
- **Titanic Dataset**: Historical maritime records
- **Netflix Dataset**: Public streaming platform data
- **Health Camp Dataset**: Medical outreach program data

---

## ⚖️ License & Usage

This dataset collection is available under **Open Source License**.

### Usage Terms:

- ✅ **Free** for educational and research purposes
- ✅ **Free** for commercial use with attribution
- ✅ **Modification** and redistribution allowed
- ❌ **No warranty** provided

### Attribution:

When using these datasets, please cite:

```
Dataset Collection by itsluckysharma01
GitHub: https://github.com/itsluckysharma01/Datasets
```

---

## 🎉 Getting Started Today!

### Quick Start Checklist

- [ ] Clone the repository
- [ ] Install required packages
- [ ] Choose a dataset that interests you
- [ ] Load and explore the data
- [ ] Run example analyses
- [ ] Build your own models!

### Need Help?

- 📧 **Email**: [Your Contact]
- 💬 **Issues**: Open a GitHub issue
- 📖 **Wiki**: Check our documentation

---

<div align="center">

### 🌟 Star this repository if you find it useful!

![GitHub stars](https://img.shields.io/github/stars/itsluckysharma01/Datasets?style=social)
![GitHub forks](https://img.shields.io/github/forks/itsluckysharma01/Datasets?style=social)

**Happy Data Science! 🚀📊**

</div>

---

_Last updated: September 2025_
