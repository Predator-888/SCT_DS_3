# 1. Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- DATA LOADING AND PREPARATION ---

# 2. Load the dataset
# Make sure 'bank-full.csv' is in the same folder as your script.
# The data in this file is separated by semicolons, so we use sep=';'
try:
    df = pd.read_csv('bank-full.csv', sep=';')
except FileNotFoundError:
    print("Error: 'bank-full.csv' not found. Please download it from the UCI repository.")
    exit()

# 3. Prepare the data
# The model can't handle text, so we convert categorical columns to numbers.
# For simplicity, we'll use LabelEncoder for this task.
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
le = LabelEncoder()

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# 4. Define Features (X) and Target (y)
# Features are all columns except the one we want to predict ('y')
X = df.drop('y', axis=1) 
# Target is the 'y' column (0 for 'no', 1 for 'yes')
y = df['y'] 


# --- MODEL BUILDING AND TRAINING ---

# 5. Split the data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Initialize and train the Decision Tree model
# random_state ensures we get the same result every time
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print("✅ Model training complete.")


# --- EVALUATION ---

# 7. Make predictions on the test data
y_pred = model.predict(X_test)

# 8. Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Print a detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))