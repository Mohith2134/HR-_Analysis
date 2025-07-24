import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import time
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df_hr = pd.read_csv("hr_attrition_dataset.csv")
print("loading the dataset...,")
start=time.sleep(5)
print("Shape:", df_hr.shape)
print(df_hr.head())

# 2. Feature Engineering
df_hr['TenureBand'] = pd.cut(df_hr['YearsAtCompany'], bins=[0, 3, 6, 10, 40], labels=['New', 'Junior', 'Mid', 'Senior'])
df_hr['IncomeLevel'] = pd.cut(df_hr['MonthlyIncome'], bins=[0, 5000, 10000, 20000, 100000], labels=['Low', 'Medium', 'High', 'Very High'])

# 3. EDA
sns.countplot(x='Attrition', data=df_hr)
plt.title("Overall Attrition Distribution")
plt.show()

sns.countplot(x='Department', hue='Attrition', data=df_hr)
plt.title("Attrition by Department")
plt.xticks(rotation=45)
plt.show()

sns.boxplot(x='Attrition', y='MonthlyIncome', data=df_hr)
plt.title("Attrition vs Monthly Income")
plt.show()

# 4. Encode categorical variables
df_encoded = pd.get_dummies(df_hr, drop_first=True)
X = df_encoded.drop('Attrition_Yes', axis=1)
y = df_encoded['Attrition_Yes']

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 6. Train multiple models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\n{name} Accuracy: {acc:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Use SHAP with best model (e.g., Random Forest)
best_model = models['Random Forest']
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Global summary plot
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=True)
else:
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

# 8. Individual SHAP explanation (Safe version)
shap.initjs()
instance_index = 5
shap_value_row = shap_values[1][instance_index]
feature_row = X_test.iloc[instance_index:instance_index+1]

print("Length of SHAP values:", len(shap_value_row))
print("Number of features:", feature_row.shape[1])

if len(shap_value_row) == feature_row.shape[1]:
    shap.force_plot(explainer.expected_value[1], shap_value_row, feature_row, matplotlib=True)
else:
    print("❌ SHAP value and feature dimensions do not match. Skipping force plot.")

# 9. Predict probability for new user input
new_input = pd.DataFrame([X_test.iloc[instance_index]])
pred = best_model.predict(new_input)
pred_prob = best_model.predict_proba(new_input)
print("\nPredicted Attrition (0=No, 1=Yes):", pred[0])
print("Attrition Probability (No, Yes):", pred_prob[0])


# Only take numerical features for scaling comparison
numerical_cols = df_hr.select_dtypes(include=[np.number]).columns
df_numerical = df_hr[numerical_cols]

# Apply StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numerical)
scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols)

# Plotting before and after scaling (Boxplots)
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=df_numerical)
plt.title("📊 Before Scaling")
plt.xticks(rotation=90)

plt.subplot(1, 2, 2)
sns.boxplot(data=scaled_df)
plt.title("📊 After Scaling (StandardScaler)")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
