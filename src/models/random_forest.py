from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import json
from pathlib import Path
import numpy as np


class RandomForestModel:
    
    def __init__(
        self,
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': class_weight,
            'random_state': random_state
        }
    
    def fit(self, X_train, y_train):
        print("Training Random Forest...")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {len(np.unique(y_train))}")
        
        self.model.fit(X_train, y_train)
        
        print("✓ Training complete")
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names=None):
        importance = self.model.feature_importances_
        
        if feature_names is not None:
            importance_dict = dict(zip(feature_names, importance))
            # Sort by importance
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            return importance_dict
        
        return importance
    
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, path / 'model.pkl')
        
        # Save metadata
        metadata = {
            'model_type': 'RandomForest',
            'params': self.params,
            'n_features': self.model.n_features_in_,
            'n_classes': self.model.n_classes_
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {path}")
    
    def load(self, path):
        path = Path(path)
        
        self.model = joblib.load(path / 'model.pkl')
        
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.params = metadata['params']
        
        print(f"Model loaded from {path}")
        return self


def tune_random_forest(X_train, y_train, X_val, y_val, scoring='f1_macro', n_jobs=-1):
    print("=" * 70)
    print("Tuning Random Forest Hyperparameters")
    print("=" * 70)
    
    # Parameter grid based on proposal
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [None, 20, 40],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', None]
    }
    
    base_model = RandomForestClassifier(
        random_state=42,
        n_jobs=n_jobs,
        verbose=0
    )
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=True
    )
    
    print(f"\nSearching over {len(param_grid)} parameters...")
    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    grid_search.fit(X_train, y_train)
    
    print("\n" + "=" * 70)
    print("Best Parameters:")
    print("=" * 70)
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest CV {scoring}: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    best_model = grid_search.best_estimator_
    val_score = best_model.score(X_val, y_val)
    print(f"Validation accuracy: {val_score:.4f}")
    
    return best_model, grid_search.best_params_


if __name__ == "__main__":
    # Quick test
    from sklearn.datasets import make_classification
    
    print("Testing Random Forest model...")
    
    # Generate dummy data
    X_train, y_train = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=30,
        n_classes=5,
        random_state=42
    )
    
    X_test, y_test = make_classification(
        n_samples=200,
        n_features=50,
        n_informative=30,
        n_classes=5,
        random_state=43
    )
    
    # Train model
    rf = RandomForestModel(n_estimators=100, verbose=0)
    rf.fit(X_train, y_train)
    
    # Predict
    y_pred = rf.predict(X_test)
    
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Macro-F1: {f1:.4f}")
    
    # Save and load
    rf.save("test_rf_model")
    
    rf_loaded = RandomForestModel()
    rf_loaded.load("test_rf_model")
    
    print("\n✓ Random Forest model test passed")
