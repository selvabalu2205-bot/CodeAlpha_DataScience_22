import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the Kaggle dataset
df = pd.read_csv("Iris.csv")

print("First five rows of the dataset:")
print(df.head())

# Remove unnecessary ID column if present
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# Separate input and output
X = df.drop(columns=["Species"])
y = df["Species"]

# Convert species names into numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train the model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Predict the test data
y_pred = model.predict(X_test)

# Display accuracy
print("\nIris Flower Classification Results")
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Display classification report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=encoder.classes_
))

# Predict a new flower
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = model.predict(sample)

print("Predicted Flower:", encoder.inverse_transform(prediction)[0])