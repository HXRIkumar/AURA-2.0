# src/steps/missing_values.py
import pandas as pd

def process(df: pd.DataFrame, drop_threshold: float = 0.8) -> pd.DataFrame:
    """
    Automatically handle missing values:
    - Numeric columns → fill with mean
    - Categorical columns → fill with mode
    - Drop columns with missing ratio > drop_threshold
    """

    missing_report = df.isnull().sum()
    total_rows = len(df)
    
    # Drop columns with too many missing values
    for col, count in missing_report.items():
        missing_ratio = count / total_rows
        if missing_ratio > drop_threshold:
            df.drop(columns=[col], inplace=True)
            print(f"🗑️ Dropped column '{col}' ({missing_ratio*100:.2f}% missing)")

    # Fill numeric columns with mean
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"🔢 Filled numeric column '{col}' with mean ({mean_val:.2f})")

    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"🅰️ Filled categorical column '{col}' with mode ('{mode_val}')")

    print("✅ Missing value handling completed automatically.")
    return df
