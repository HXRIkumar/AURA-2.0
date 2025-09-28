# main.py
from src.pipeline import AuraPipeline

if __name__ == "__main__":
    print("=== AURA PREPROCESSOR ===\n")

    # Load Titanic dataset (update path if needed)
    pipeline = AuraPipeline("data/titanic.csv")

    print("\n=== STEP 1: Handle Missing Values ===")
    pipeline.handle_missing_values()

    print("\n=== STEP 2: Train Model ===")
    pipeline.train_model()

    print("\n=== STEP 3: Save Cleaned Dataset ===")
    pipeline.save("outputs/cleaned_titanic.csv")  # Automatically creates outputs folder

    print("\n✅ Pipeline finished successfully!")
