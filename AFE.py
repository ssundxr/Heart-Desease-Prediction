import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class HeartDiseasePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_features(self, df):
        """Create advanced features"""
        df = df.copy()
        
        # BMI-like feature (if thalach represents max heart rate)
        df['heart_rate_age_ratio'] = df['thalach'] / df['age']
        
        # Blood pressure categories
        df['bp_category'] = pd.cut(df['trestbps'], 
                                    bins=[0, 120, 140, 180, 300],
                                    labels=['Normal', 'Elevated', 'High', 'Crisis'])
        
        # Cholesterol risk
        df['chol_risk'] = pd.cut(df['chol'],
                                 bins=[0, 200, 240, 500],
                                 labels=['Desirable', 'Borderline', 'High'])
        
        # Age groups
        df['age_group'] = pd.cut(df['age'],
                                 bins=[0, 40, 55, 70, 100],
                                 labels=['Young', 'Middle', 'Senior', 'Elderly'])
        
        # Interaction features
        df['age_chol_interaction'] = df['age'] * df['chol']
        df['bp_chol_interaction'] = df['trestbps'] * df['chol']
        
        return df
    
    def handle_imbalance(self, X, y):
        """Handle class imbalance with SMOTE"""
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def preprocess(self, df, target_col='target', test_size=0.2):
        """Complete preprocessing pipeline"""
        # Feature engineering
        df = self.create_features(df)
        
        # Encode categorical variables created by pd.cut
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col != target_col:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Split features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Handle imbalance
        X_train, y_train = self.handle_imbalance(X_train, y_train)
        
        # Scale features
        X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns
        )
        
        return X_train, X_test, y_train, y_test