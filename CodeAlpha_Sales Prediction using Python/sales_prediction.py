import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the Kaggle dataset
df = pd.read_csv("advertising.csv")

# Remove unnecessary index column if present
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Display dataset
print("FIRST FIVE ROWS")
print(df.head())

# Select input features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales
y_pred = model.predict(X_test)

# Evaluate the model
print("\nSALES PREDICTION RESULTS")
print("Mean Absolute Error:",
      round(mean_absolute_error(y_test, y_pred), 3))

print("Mean Squared Error:",
      round(mean_squared_error(y_test, y_pred), 3))

print("R-squared Score:",
      round(r2_score(y_test, y_pred), 3))

# Predict sales for a new advertising campaign
new_campaign = pd.DataFrame({
    "TV": [150],
    "Radio": [25],
    "Newspaper": [20]
})

predicted_sales = model.predict(new_campaign)

print("\nNew Advertising Campaign")
print("TV Advertising: 150")
print("Radio Advertising: 25")
print("Newspaper Advertising: 20")
print("Predicted Sales:", round(predicted_sales[0], 2))

# Plot actual vs predicted sales
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot([minimum, maximum], [minimum, maximum],
         linestyle="--")

plt.title("Actual vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.grid(True)
plt.tight_layout()

plt.savefig("sales_prediction_graph.png")
plt.show()

print("\nSales prediction completed successfully!")