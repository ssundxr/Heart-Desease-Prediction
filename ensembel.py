from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import shap
import joblib
import matplotlib.pyplot as plt

class HeartDiseaseModel:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.explainer = None
        
    def create_ensemble(self):
        """Create ensemble of multiple models"""
        # Individual models
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        
        # Voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('xgb', xgb),
                ('gb', gb),
                ('lr', lr)
            ],
            voting='soft'
        )
        
        self.models = {
            'Random Forest': rf,
            'XGBoost': xgb,
            'Gradient Boosting': gb,
            'Logistic Regression': lr,
            'Ensemble': ensemble
        }
        
        return self.models
    
    def train_and_compare(self, X_train, y_train, X_test, y_test):
        """Train all models and compare performance"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            results[name] = {
                'model': model,
                'accuracy': model.score(X_test, y_test),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print(f"{name} - Accuracy: {results[name]['accuracy']:.4f}, ROC-AUC: {results[name]['roc_auc']:.4f}")
        
        # Select best model
        best_name = max(results, key=lambda x: results[x]['roc_auc'])
        self.best_model = results[best_name]['model']
        print(f"\n🏆 Best Model: {best_name}")
        
        return results
    
    def create_explainability(self, X_train):
        """Create SHAP explainer for model interpretability"""
        self.explainer = shap.TreeExplainer(self.best_model)
        shap_values = self.explainer.shap_values(X_train)
        
        return shap_values
    
    def plot_feature_importance(self, X_train, top_n=10):
        """Plot feature importance"""
        shap_values = self.explainer.shap_values(X_train)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[1], X_train, plot_type="bar", max_display=top_n)
        plt.title('Top Feature Importance for Heart Disease Prediction')
        plt.tight_layout()
        return plt.gcf()
    
    def save_model(self, filepath='../models/best_model.pkl'):
        """Save the best model"""
        joblib.dump(self.best_model, filepath)
        print(f"✅ Model saved to {filepath}")