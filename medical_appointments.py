import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\sarat\Downloads\medical_appointments.csv")

# Check first few rows
print(df.head())

# Check dataset info
print(df.info())

# Check number of rows and columns
print("Rows:", df.shape[0], "Columns:", df.shape[1])

# Summary statistics
print(df.describe())

# Check target column distribution
print(df['Showed_up'].value_counts())

# Check for missing values
print(df.isna().sum())
import matplotlib.pyplot as plt
import seaborn as sns

# Show vs No-show
sns.countplot(x='Showed_up', data=df)
plt.title("Show vs No-show Count")
plt.show()

# Age distribution
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train rows:", X_train.shape[0], "Test rows:", X_test.shape[0])

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Train model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature importance:\n", importances)

import joblib
joblib.dump(model, 'no_show_model.pkl')
print("Model saved as 'no_show_model.pkl'")

# SMS impact
if 'SMS_received' in df.columns:
    sns.countplot(x='SMS_received', hue='Showed_up', data=df)
    plt.title("Impact of SMS on Attendance")
    plt.show()

# Weekday impact
sns.countplot(x='Weekday', hue='Showed_up', data=df)
plt.title("No-Show by Weekday")
plt.xticks(rotation=45)
plt.show()
# Export dataset for Power BI
df.to_csv('appointments_cleaned.csv', index=False)
print("Cleaned dataset saved as 'appointments_cleaned.csv'")



