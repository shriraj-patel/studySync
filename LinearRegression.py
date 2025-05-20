import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# : Load the dataset
file_path = 'Students_Performance_data_set.csv'
df = pd.read_csv(file_path)
print("Shape:", df.shape)
print(df.head())

# : Show columns and CGPA
print("\nColumns:")
print(df.columns.tolist())
print("\nTarget variable: What is your current CGPA?")
print(df['What is your current CGPA?'].describe())

# : Select relevant features
selected_features = [
    'Do you have meritorious scholarship ?',
    'Do you use University transportation?',
    'What is your preferable learning mode?',
    'Status of your English language proficiency',
    'Average attendance on class',
    'Do you have personal Computer?',
    'Do you have any health issues?',
    'What was your previous SGPA?'
]

target_variable = 'What is your current CGPA?'
X = df[selected_features].copy()
y = df[target_variable]

# : Clean "Average attendance on class" (e.g., '94-98' → 96.0)
def convert_attendance(value):
    if isinstance(value, str) and '-' in value:
        low, high = value.split('-')
        return (int(low) + int(high)) / 2
    try:
        return float(value)
    except:
        return None

X['Average attendance on class'] = X['Average attendance on class'].apply(convert_attendance)

# Step 5: One-hot encode categorical columns
categorical_cols = [
    'Do you have meritorious scholarship ?',
    'Do you use University transportation?',
    'What is your preferable learning mode?',
    'Status of your English language proficiency',
    'Do you have personal Computer?',
    'Do you have any health issues?'
]

X = pd.get_dummies(X, columns=categorical_cols)

# : Drop rows with missing values
X = X.dropna()
y = y.loc[X.index]

# : Train/Test Split
print("\nX shape:", X.shape)
print("y shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# : Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Predictions and evaluation
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared (R²):", r2_score(y_test, y_pred))
