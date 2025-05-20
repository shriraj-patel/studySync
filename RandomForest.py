import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data_path = 'Students_Performance_data_set.csv'
df = pd.read_csv(data_path)

# print("Shape:", df.shape)
# print(df.head(), "\n")

# # Show column names
# print("Columns:")
# print(df.columns.tolist())

# Target variable
target_column = 'What is your current CGPA?'
y = df[target_column]
# print(f"\nTarget variable: {target_column}")
# print(y.describe())

# Select and process relevant features
selected_features = [
    'University Admission year',
    'Do you have meritorious scholarship ?',
    'Do you use University transportation?',
    'What is your preferable learning mode?',
    'Do you have personal Computer?',
    'Do you have any health issues?',
    'Are you engaged with any co-curriculum activities?',
    'How many hour do you study daily?',
    'How many hour do you spent daily in social media?',
    'Average attendance on class',
    'Did you ever fall in probation?',
    'Did you ever got suspension?',
    'Do you attend in teacher consultancy for any kind of academical problems?',
    'What was your previous SGPA?',
    'Do you have any physical disabilities?',
    'How many Credit did you have completed?',
    'What is your monthly family income?'
]

X = df[selected_features].copy()

# Handle categorical values
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Confirm shapes
# print("\nX shape:", X.shape)
# print("y shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# Save cleaned dataset
cleaned_df = X.copy()
cleaned_df[target_column] = y
cleaned_df.to_csv('cleaned_student_data.csv', index=False)

# print("\n✅ Cleaned dataset saved to 'cleaned_student_data.csv'")
