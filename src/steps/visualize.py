# src/steps/visualize.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_missing(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.show()

def show_correlation(df):
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation")
    plt.show()

def feature_impact(df, target="Survived", col_list=None):
    import matplotlib.pyplot as plt
    if col_list is None:
        col_list = df.select_dtypes(include=['int64','float64']).columns.tolist()
        col_list = [c for c in col_list if c != target]
    
    for col in col_list:
        plt.figure()
        df.groupby(col)[target].mean().plot(kind='bar')
        plt.title(f"{col} vs {target}")
        plt.ylabel(f"Average {target}")
        plt.show()
