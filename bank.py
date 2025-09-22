# 1. Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- 2. DATA LOADING AND PREPARATION ---

# Load the dataset
try:
    df = pd.read_csv('bank-full.csv', sep=';')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'bank-full.csv' not found. Please ensure it's in the correct directory.")
    exit()

# Convert categorical columns to numbers using LabelEncoder
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
le = LabelEncoder()

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Define Features (X) and Target (y)
X = df.drop('y', axis=1)
y = df['y']

# --- 3. MODEL BUILDING AND TRAINING ---

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# --- 4. EVALUATION ---

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Print a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

# --- 5. VISUALIZATIONS ---

# a) Confusion Matrix
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
class_names = ['No', 'Yes']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix — Customer Purchase Prediction')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# b) Decision Tree Plot
print("Generating Decision Tree visualization...")
feature_names = X.columns
plt.figure(figsize=(25, 12)) # Using a large figure size to make the tree readable
plot_tree(model,
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          fontsize=8,
          max_depth=4) # Limiting depth to 4 for better readability, like in your image
plt.title('Decision Tree for Customer Purchase Prediction')
plt.savefig('decision_tree.png', dpi=300)
plt.show()
print("\nDecision tree visualization saved as decision_tree.png")
