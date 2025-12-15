import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

def generate_eda_report(csv_path, output_dir):
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Read dataset
    df = pd.read_csv(csv_path)

    # Basic info
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict()
    }

    # Save summary info
    summary_file = os.path.join(output_dir, "summary.json")
    pd.Series(info).to_json(summary_file, indent=4)

    # Descriptive statistics
    desc = df.describe(include='all')
    desc.to_csv(os.path.join(output_dir, "describe.csv"))

    # Plot numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Histogram of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_hist.png"))
        plt.close()

    # Plot categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        plt.figure(figsize=(6,4))
        df[col].value_counts().plot(kind='bar')
        plt.title(f"Count Plot of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_count.png"))
        plt.close()

    # Correlation heatmap for numeric columns
    if numeric_cols:
        plt.figure(figsize=(8,6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()

    # Generate simple HTML report
    html_path = os.path.join(output_dir, "report.html")
    with open(html_path, "w") as f:
        f.write("<html><head><title>EDA Report</title></head><body>")
        f.write("<h1>Exploratory Data Analysis Report</h1>")
        f.write(f"<h2>Dataset Shape: {df.shape}</h2>")
        f.write("<h3>Missing Values:</h3><pre>")
        f.write(str(df.isnull().sum()))
        f.write("</pre>")
        f.write("<h3>Data Types:</h3><pre>")
        f.write(str(df.dtypes))
        f.write("</pre>")
        f.write("<h3>Summary Statistics:</h3>")
        f.write(desc.to_html())
        f.write("<h3>Plots:</h3>")
        for col in numeric_cols:
            f.write(f"<h4>{col} Histogram</h4><img src='{col}_hist.png' width='600'><br>")
        for col in categorical_cols:
            f.write(f"<h4>{col} Count Plot</h4><img src='{col}_count.png' width='600'><br>")
        if numeric_cols:
            f.write("<h4>Correlation Heatmap</h4><img src='correlation_heatmap.png' width='600'><br>")
        f.write("</body></html>")

    print(f"EDA report generated successfully in '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EDA report from CSV")
    parser.add_argument("csv_path", help="Path to the CSV dataset")
    parser.add_argument("--out", default="eda_report", help="Output directory for report")
    args = parser.parse_args()

    generate_eda_report(args.csv_path, args.out)

