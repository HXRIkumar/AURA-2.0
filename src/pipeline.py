# src/pipeline.py
import os
import pandas as pd
from src.steps import missing_values
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

class AuraPipeline:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        print(f"✅ Loaded dataset with shape {self.df.shape}")

    def handle_missing_values(self):
        self.df = missing_values.process(self.df)

    def train_model(self):
        df = self.df.copy()

        # Encode categorical columns automatically
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        # Separate features and target
        if 'Survived' not in df.columns:
            raise ValueError("Target column 'Survived' not found in dataset!")
        X = df.drop('Survived', axis=1)
        y = df['Survived']

        # Scale numeric features
        X_scaled = StandardScaler().fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train RandomForest
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"📊 Accuracy after preprocessing + encoding + scaling: {acc:.2f}")

    def save(self, output_path='outputs/cleaned_titanic.csv'):
        # Ensure outputs folder exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"💾 Saved cleaned dataset to {output_path}")
