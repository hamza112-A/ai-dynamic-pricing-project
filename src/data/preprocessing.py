import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import argparse
import json


class CICIDSPreprocessor:
    
    # Columns to drop (leaky or non-generalizing)
    COLUMNS_TO_DROP = [
        'Flow ID',
        'Source IP',
        'Source Port', 
        'Destination IP',
        'Destination Port',
        'Timestamp',
        'Protocol'  # Can optionally keep this, but often leaky
    ]
    
    def __init__(self, scale_features=True):
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.label_mapping = None
        
    def load_data(self, data_dir):
        data_dir = Path(data_dir)
        csv_files = list(data_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {data_dir}. "
                "Please download CICIDS2017 dataset first."
            )
        
        print(f"Loading {len(csv_files)} CSV files from {data_dir}")
        dfs = []
        
        for csv_file in csv_files:
            print(f"  Loading {csv_file.name}...")
            try:
                df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file, encoding='latin1', low_memory=False)
            dfs.append(df)
        
        # Concatenate all dataframes
        full_df = pd.concat(dfs, ignore_index=True)
        print(f"Total flows loaded: {len(full_df):,}")
        
        return full_df
    
    def clean_data(self, df):
        
        print("\nCleaning data...")
        original_size = len(df)
        
        # Standardize column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Check if 'Label' column exists
        if 'Label' not in df.columns and ' Label' in df.columns:
            df.rename(columns={' Label': 'Label'}, inplace=True)
        
        # Drop leaky columns
        cols_to_drop = [col for col in self.COLUMNS_TO_DROP if col in df.columns]
        print(f"  Dropping {len(cols_to_drop)} leaky columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Separate features and labels
        if 'Label' not in df.columns:
            raise ValueError("Label column not found in dataset!")
        
        y = df['Label'].copy()
        X = df.drop('Label', axis=1)
        
        # Replace infinite values with NaN
        print("  Replacing infinite values with NaN...")
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Convert all feature columns to numeric, coercing errors
        print("  Converting features to numeric...")
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill NaN with median (computed later on training set only)
        print(f"  NaN values before imputation: {X.isna().sum().sum()}")
        
        # Remove duplicate rows
        combined = pd.concat([X, y], axis=1)
        print(f"  Removing duplicates...")
        combined = combined.drop_duplicates()
        
        X = combined.drop('Label', axis=1)
        y = combined['Label']
        
        print(f"  Rows after cleaning: {len(X):,} (removed {original_size - len(X):,})")
        
        return X, y
    
    def fit_transform(self, X, y):
        
        print("\nFitting preprocessor...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Impute NaN with median
        print("  Computing medians for imputation...")
        medians = X.median()
        X = X.fillna(medians)
        self.medians = medians
        
        # Encode labels
        print("  Encoding labels...")
        y_encoded = self.label_encoder.fit_transform(y)
        self.label_mapping = dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
        print(f"  Found {len(self.label_mapping)} classes: {list(self.label_mapping.keys())}")
        
        # Scale features
        if self.scale_features:
            print("  Scaling features...")
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        return X, y_encoded
    
    def transform(self, X, y=None):
        # Impute NaN
        X = X.fillna(self.medians)
        
        # Scale features
        if self.scale_features:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        # Encode labels if provided
        if y is not None:
            y_encoded = self.label_encoder.transform(y)
            return X, y_encoded
        
        return X
    
    def save(self, path):
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, path / 'scaler.pkl')
        joblib.dump(self.label_encoder, path / 'label_encoder.pkl')
        joblib.dump(self.medians, path / 'medians.pkl')
        
        metadata = {
            'feature_names': self.feature_names,
            'label_mapping': {k: int(v) if hasattr(v, 'item') else v for k, v in self.label_mapping.items()},
            'scale_features': self.scale_features
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Preprocessor saved to {path}")
    
    def load(self, path):
        
        path = Path(path)
        
        self.scaler = joblib.load(path / 'scaler.pkl')
        self.label_encoder = joblib.load(path / 'label_encoder.pkl')
        self.medians = joblib.load(path / 'medians.pkl')
        
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.label_mapping = metadata['label_mapping']
        self.scale_features = metadata['scale_features']
        
        print(f"Preprocessor loaded from {path}")


class UNSWPreprocessor:
    
    # Columns to drop (leaky or non-generalizing)
    COLUMNS_TO_DROP = [
        'srcip',
        'sport',
        'dstip',
        'dsport',
        'id',  # Flow ID
    ]
    
    def __init__(self, scale_features=True):
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.label_mapping = None
    
    def load_data(self, data_dir, use_official_split=True):
        data_dir = Path(data_dir)
        
        if use_official_split:
            # Use official train/test split
            train_file = data_dir / "UNSW_NB15_training-set.csv"
            test_file = data_dir / "UNSW_NB15_testing-set.csv"
            
            if not train_file.exists() or not test_file.exists():
                raise FileNotFoundError(
                    f"Official split files not found in {data_dir}. "
                    "Please download UNSW-NB15 dataset first."
                )
            
            print(f"Loading official train/test split from {data_dir}")
            train_df = pd.read_csv(train_file, low_memory=False)
            test_df = pd.read_csv(test_file, low_memory=False)
            
            print(f"Train flows: {len(train_df):,}")
            print(f"Test flows: {len(test_df):,}")
            
            return train_df, test_df
        else:
            # Load all data files
            csv_files = [
                data_dir / f"UNSW-NB15_{i}.csv" for i in range(1, 5)
            ]
            
            print(f"Loading UNSW-NB15 full dataset from {data_dir}")
            dfs = []
            
            for csv_file in csv_files:
                if csv_file.exists():
                    print(f"  Loading {csv_file.name}...")
                    df = pd.read_csv(csv_file, low_memory=False)
                    dfs.append(df)
            
            full_df = pd.concat(dfs, ignore_index=True)
            print(f"Total flows loaded: {len(full_df):,}")
            
            return full_df
    
    def clean_data(self, df):
        
        print("\nCleaning data...")
        original_size = len(df)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Check for label column (might be 'attack_cat' or 'label')
        label_col = 'attack_cat' if 'attack_cat' in df.columns else 'label'
        
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in dataset!")
        
        # Drop leaky columns
        cols_to_drop = [col for col in self.COLUMNS_TO_DROP if col in df.columns]
        print(f"  Dropping {len(cols_to_drop)} leaky columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Separate features and labels
        y = df[label_col].copy()
        X = df.drop([label_col], axis=1, errors='ignore')
        
        # Also drop 'label' if using 'attack_cat' and vice versa
        if 'label' in X.columns:
            X = X.drop('label', axis=1)
        if 'attack_cat' in X.columns and label_col != 'attack_cat':
            X = X.drop('attack_cat', axis=1)
        
        # Replace infinite values with NaN
        print("  Replacing infinite values with NaN...")
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Convert all feature columns to numeric
        print("  Converting features to numeric...")
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        print(f"  NaN values before imputation: {X.isna().sum().sum()}")
        
        # Remove duplicate rows
        combined = pd.concat([X, y], axis=1)
        print(f"  Removing duplicates...")
        combined = combined.drop_duplicates()
        
        X = combined.drop(label_col, axis=1)
        y = combined[label_col]
        
        print(f"  Rows after cleaning: {len(X):,} (removed {original_size - len(X):,})")
        
        return X, y
    
    def fit_transform(self, X, y):
        
        print("\nFitting preprocessor...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Impute NaN with median
        print("  Computing medians for imputation...")
        medians = X.median()
        X = X.fillna(medians)
        self.medians = medians
        
        # Encode labels
        print("  Encoding labels...")
        y_encoded = self.label_encoder.fit_transform(y)
        self.label_mapping = dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
        print(f"  Found {len(self.label_mapping)} classes: {list(self.label_mapping.keys())}")
        
        # Scale features
        if self.scale_features:
            print("  Scaling features...")
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        return X, y_encoded
    
    def transform(self, X, y=None):
        
        X = X.fillna(self.medians)
        
        if self.scale_features:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        if y is not None:
            y_encoded = self.label_encoder.transform(y)
            return X, y_encoded
        
        return X
    
    def save(self, path):
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, path / 'scaler.pkl')
        joblib.dump(self.label_encoder, path / 'label_encoder.pkl')
        joblib.dump(self.medians, path / 'medians.pkl')
        
        metadata = {
            'feature_names': self.feature_names,
            'label_mapping': {k: int(v) if hasattr(v, 'item') else v for k, v in self.label_mapping.items()},
            'scale_features': self.scale_features
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Preprocessor saved to {path}")
    
    def load(self, path):
        
        path = Path(path)
        
        self.scaler = joblib.load(path / 'scaler.pkl')
        self.label_encoder = joblib.load(path / 'label_encoder.pkl')
        self.medians = joblib.load(path / 'medians.pkl')
        
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.label_mapping = metadata['label_mapping']
        self.scale_features = metadata['scale_features']
        
        print(f"Preprocessor loaded from {path}")


def create_stratified_splits(X, y, train_size=0.7, val_size=0.15, random_state=42):
    print("\nCreating stratified splits...")
    
    # First split: train vs (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        train_size=train_size,
        stratify=y,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio = val_size / (1 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_ratio,
        stratify=y_temp,
        random_state=random_state
    )
    
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, output_dir, dataset_name):
    output_dir = Path(output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving splits to {output_dir}...")
    
    # Save as numpy arrays for faster loading
    np.save(output_dir / 'X_train.npy', X_train.values if hasattr(X_train, 'values') else X_train)
    np.save(output_dir / 'X_val.npy', X_val.values if hasattr(X_val, 'values') else X_val)
    np.save(output_dir / 'X_test.npy', X_test.values if hasattr(X_test, 'values') else X_test)
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'y_val.npy', y_val)
    np.save(output_dir / 'y_test.npy', y_test)
    
    print("✓ Splits saved successfully")


def main():
    parser = argparse.ArgumentParser(description="Preprocess NIDS datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cicids2017", "unsw-nb15"],
        help="Which dataset to preprocess"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing raw dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--no-scaling",
        action="store_true",
        help="Disable feature scaling (useful for tree models)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    np.random.seed(args.random_seed)
    scale_features = not args.no_scaling
    
    if args.dataset == "cicids2017":
        print("=" * 70)
        print("Preprocessing CICIDS2017 Dataset")
        print("=" * 70)
        
        preprocessor = CICIDSPreprocessor(scale_features=scale_features)
        
        # Load data
        df = preprocessor.load_data(args.input)
        
        # Clean data
        X, y = preprocessor.clean_data(df)
        
        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = create_stratified_splits(
            X, y, random_state=args.random_seed
        )
        
        # Fit on train, transform all
        X_train, y_train = preprocessor.fit_transform(X_train, y_train)
        X_val, y_val = preprocessor.transform(X_val, y_val)
        X_test, y_test = preprocessor.transform(X_test, y_test)
        
        # Save everything
        preprocessor.save(Path(args.output) / "cicids2017_preprocessor")
        save_splits(X_train, X_val, X_test, y_train, y_val, y_test, 
                   "data/splits", "cicids2017")
    
    elif args.dataset == "unsw-nb15":
        print("=" * 70)
        print("Preprocessing UNSW-NB15 Dataset")
        print("=" * 70)
        
        preprocessor = UNSWPreprocessor(scale_features=scale_features)
        
        # Load official split
        train_df, test_df = preprocessor.load_data(args.input, use_official_split=True)
        
        # Clean data
        X_train_full, y_train_full = preprocessor.clean_data(train_df)
        X_test, y_test = preprocessor.clean_data(test_df)
        
        # Split training into train+val (90/10)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            train_size=0.9,
            stratify=y_train_full,
            random_state=args.random_seed
        )
        
        print(f"\nFinal splits:")
        print(f"  Train: {len(X_train):,} samples")
        print(f"  Val:   {len(X_val):,} samples")
        print(f"  Test:  {len(X_test):,} samples")
        
        # Fit on train, transform all
        X_train, y_train = preprocessor.fit_transform(X_train, y_train)
        X_val, y_val = preprocessor.transform(X_val, y_val)
        X_test, y_test = preprocessor.transform(X_test, y_test)
        
        # Save everything
        preprocessor.save(Path(args.output) / "unsw-nb15_preprocessor")
        save_splits(X_train, X_val, X_test, y_train, y_val, y_test,
                   "data/splits", "unsw-nb15")
    
    print("\n" + "=" * 70)
    print("✓ Preprocessing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
