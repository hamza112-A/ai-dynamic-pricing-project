import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import json
from pathlib import Path
import numpy as np


class XGBoostModel:
    
    def __init__(
        self,
        n_estimators=600,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=None,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    ):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=verbosity,
            tree_method='hist',  # Faster training
            eval_metric='mlogloss'
        )
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'scale_pos_weight': scale_pos_weight,
            'random_state': random_state
        }
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=50):
        print("Training XGBoost...")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {len(np.unique(y_train))}")
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=True
            )
            print(f"  Best iteration: {self.model.best_iteration}")
        else:
            self.model.fit(X_train, y_train, verbose=True)
        
        print("✓ Training complete")
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names=None, importance_type='weight'):
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        if feature_names is not None:
            # Map feature indices to names
            importance_dict = {}
            for key, value in importance.items():
                idx = int(key[1:])  # Remove 'f' prefix
                if idx < len(feature_names):
                    importance_dict[feature_names[idx]] = value
            
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
        self.model.save_model(str(path / 'model.json'))
        
        # Also save as pickle for compatibility
        joblib.dump(self.model, path / 'model.pkl')
        
        # Save metadata
        metadata = {
            'model_type': 'XGBoost',
            'params': self.params,
            'n_features': self.model.n_features_in_,
            'n_classes': len(self.model.classes_)
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {path}")
    
    def load(self, path):
        path = Path(path)
        
        # Try loading from pickle first (preserves all attributes)
        try:
            self.model = joblib.load(path / 'model.pkl')
        except:
            # Fallback to JSON format
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(path / 'model.json'))
        
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.params = metadata['params']
        
        print(f"Model loaded from {path}")
        return self


def tune_xgboost(X_train, y_train, X_val, y_val, scoring='f1_macro', n_jobs=-1):
    print("=" * 70)
    print("Tuning XGBoost Hyperparameters")
    print("=" * 70)
    
    # Parameter grid based on proposal
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.3],
        'n_estimators': [200, 400, 600],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    base_model = xgb.XGBClassifier(
        random_state=42,
        n_jobs=n_jobs,
        verbosity=0,
        tree_method='hist',
        eval_metric='mlogloss'
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
    from sklearn.metrics import accuracy_score, f1_score
    
    print("Testing XGBoost model...")
    
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
    xgb_model = XGBoostModel(n_estimators=100, verbosity=0)
    xgb_model.fit(X_train, y_train)
    
    # Predict
    y_pred = xgb_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Macro-F1: {f1:.4f}")
    
    # Save and load
    xgb_model.save("test_xgb_model")
    
    xgb_loaded = XGBoostModel()
    xgb_loaded.load("test_xgb_model")
    
    print("\n✓ XGBoost model test passed")
