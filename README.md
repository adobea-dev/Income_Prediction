# Income Prediction
This is a read-me file for Income Prediction project

# 🔍 Dataset Overview
The dataset used for this project is the Adult Census Income Dataset, originally from the U.S. Census Bureau and widely used for classification problems in machine learning. It contains demographic and employment-related information for individuals, with the goal of predicting whether a person earns more than $50,000 per year.

# 📌 Target Variable (Label)

income: A categorical variable with two classes:
">50K" (high income)
"<=50K" (low income)

# 📌 Features (Predictor Variables)
The dataset consists of both numerical and categorical features, requiring careful preprocessing.

# 🔹 Numerical Features
These are continuous variables that provide quantitative insights into an individual's profile:

age: The person's age in years
fnlwgt: Final weight (a population weighting factor)
education-num: Number of years of education completed
capital-gain: Capital gains recorded
capital-loss: Capital losses recorded
hours-per-week: Average working hours per week

# 🔹 Categorical Features
These represent qualitative characteristics and need encoding before feeding them into the model:

workclass: Type of employment (e.g., Private, Government, Self-employed)
education: Level of education (e.g., Bachelors, Masters, PhD)
marital-status: Marital status (e.g., Never-married, Married, Divorced)
occupation: Job type (e.g., Tech-support, Craft-repair, Sales)
relationship: Relationship status (e.g., Husband, Wife, Own-child)
race: Ethnic background (e.g., White, Black, Asian-Pac-Islander)
sex: Gender (Male, Female)
native-country: Country of origin

# 🛠 Tools & Technologies Used
To efficiently handle large-scale data and deploy an ML model, I leveraged the following tools:
🔹 Data Processing & Machine Learning
PySpark – For large-scale data processing and machine learning model training
Pandas & NumPy – For additional data manipulation
Scikit-learn – For feature engineering and model evaluation
Matplotlib & Seaborn – For visualizing data distributions and model insights
🔹 Model Deployment & UI Development
Streamlit – For building an interactive web application
Git & GitHub – For version control and project documentation

# 📊 Data Preprocessing & Feature Engineering
Handling Missing Values
The dataset contained some missing values, particularly in the workclass, occupation, and native-country columns. These were handled by replacing missing values with the mode (most frequent category).

Encoding Categorical Variables
Since machine learning models require numerical inputs, categorical features were converted using:

One-Hot Encoding for nominal variables
Label Encoding for binary variables like sex
Feature Scaling
For numerical features such as age and hours-per-week, I used Min-Max Scaling to normalize the values between 0 and 1, improving model convergence.

# 🤖 Model Development & Training
Machine Learning Algorithm
I trained a Logistic Regression model using PySpark’s MLlib. The steps included:

Splitting the dataset into training (80%) and testing (20%) sets
Training the model on the processed data
Evaluating performance using accuracy, precision, recall, and F1-score
Model Performance Metrics
After training, the model achieved:
✅ Accuracy: 85.2%
✅ Precision: 78.5%
✅ Recall: 72.4%
✅ F1-score: 75.3%

# 🌍 Deploying the Model with Streamlit
To make the income predictor accessible, I built a Streamlit web app, allowing users to input their details and get an instant prediction.

# How the App Works
1️⃣ The user enters key details such as age, education, occupation, hours worked per week, etc.
2️⃣ The trained model processes the input and predicts whether the individual earns above or below $50K per year.
3️⃣ The result is displayed instantly, along with a breakdown of key factors influencing the prediction.
