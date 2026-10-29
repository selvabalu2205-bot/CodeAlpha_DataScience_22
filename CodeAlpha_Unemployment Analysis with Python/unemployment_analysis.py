import pandas as pd
import matplotlib.pyplot as plt

# Load the Kaggle dataset
df = pd.read_csv("Unemployment in India.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Remove extra spaces from text values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

print("FIRST FIVE ROWS")
print(df.head())

print("\nDATASET INFORMATION")
print(df.info())

# Clean unemployment rate values
rate_column = "Estimated Unemployment Rate (%)"

df[rate_column] = pd.to_numeric(
    df[rate_column], errors="coerce"
)

df = df.dropna(subset=[rate_column])

# Display statistics
print("\nUNEMPLOYMENT STATISTICS")
print("Average rate:",
      round(df[rate_column].mean(), 2), "%")

print("Highest rate:",
      round(df[rate_column].max(), 2), "%")

print("Lowest rate:",
      round(df[rate_column].min(), 2), "%")

# Graph 1: Unemployment rate by date
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(
        df["Date"], dayfirst=True, errors="coerce"
    )

    date_data = df.dropna(subset=["Date"])

    if not date_data.empty:
        monthly_data = (
            date_data.groupby("Date")[rate_column]
            .mean()
            .sort_index()
        )

        plt.figure(figsize=(10, 5))
        plt.plot(monthly_data.index,
                 monthly_data.values, marker="o")

        plt.title("Unemployment Rate Over Time")
        plt.xlabel("Date")
        plt.ylabel("Unemployment Rate (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("unemployment_trend.png")
        plt.show()

# Graph 2: Average unemployment by region
if "Region" in df.columns:
    region_data = (
        df.groupby("Region")[rate_column]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nAVERAGE UNEMPLOYMENT RATE BY REGION")
    print(region_data.round(2))

    region_data.plot(kind="bar", figsize=(12, 6))

    plt.title("Average Unemployment Rate by Region")
    plt.xlabel("Region")
    plt.ylabel("Unemployment Rate (%)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("unemployment_by_region.png")
    plt.show()

print("\nAnalysis completed successfully!")