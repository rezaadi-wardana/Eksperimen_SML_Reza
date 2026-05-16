import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import argparse


class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        self.df_scaled = None
        self.scaler = StandardScaler()
        self.numerical_cols = [
            "monthly_income", "monthly_expense", "age",
            "avg_wallet_balance", "on_time_payment_ratio", "num_loans_taken"
        ]

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded. Shape: {self.df.shape}")
        return self

    def handle_missing_values(self):
        missing = self.df.isnull().sum().sum()
        print(f"Missing values: {missing}")
        if missing > 0:
            self.df = self.df.dropna()
            print(f"Missing values handled. Shape: {self.df.shape}")
        return self

    def handle_duplicates(self):
        dup = self.df.duplicated().sum()
        print(f"Duplicates found: {dup}")
        if dup > 0:
            self.df = self.df.drop_duplicates()
            print(f"Duplicates removed. Shape: {self.df.shape}")
        return self

    def handle_outliers(self):
        self.df_clean = self.df.copy()
        for col in self.numerical_cols:
            Q1 = self.df_clean[col].quantile(0.25)
            Q3 = self.df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((self.df_clean[col] < lower) | (self.df_clean[col] > upper)).sum()
            self.df_clean[col] = self.df_clean[col].clip(lower, upper)
            if outliers > 0:
                print(f"Outliers in {col}: {outliers} capped")
        return self

    def scale_features(self):
        self.df_scaled = self.df_clean.copy()
        cols_to_scale = [
            "monthly_income", "monthly_expense",
            "avg_wallet_balance", "on_time_payment_ratio", "num_loans_taken"
        ]
        self.df_scaled[cols_to_scale] = self.scaler.fit_transform(self.df_clean[cols_to_scale])
        print("Feature scaling completed.")
        return self

    def save_data(self, output_path):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        self.df_scaled.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to: {output_path}")
        return self

    def run(self, output_path):
        self.load_data()
        self.handle_missing_values()
        self.handle_duplicates()
        self.handle_outliers()
        self.scale_features()
        self.save_data(output_path)
        return self.df_scaled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate Loan Default Data Preprocessing")
    parser.add_argument("--input", type=str, default="../loan_default_raw/data_for_training.csv",
                        help="Path to raw dataset")
    parser.add_argument("--output", type=str, default="../preprocessing/loan_default_preprocessed.csv",
                        help="Path to save preprocessed dataset")
    args = parser.parse_args()

    preprocessor = DataPreprocessor(args.input)
    df_result = preprocessor.run(args.output)
    print(f"Preprocessing complete. Final shape: {df_result.shape}")
    print(f"Columns: {list(df_result.columns)}")
