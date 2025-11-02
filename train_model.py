import pandas as pd
from AFE import HeartDiseasePreprocessor
from ensembel import HeartDiseaseModel
import os
import pickle

# Create models directory
os.makedirs('models', exist_ok=True)

# Load data
df = pd.read_csv('heart_disease_uci.csv')

print("Original shape:", df.shape)

# Rename the mismatched column
df = df.rename(columns={'thalch': 'thalach'})

# Drop non-feature columns
df = df.drop(['id', 'dataset'], axis=1)

# Handle target column
df['target'] = (df['num'] > 0).astype(int)
df = df.drop('num', axis=1)

# Handle missing values - drop rows with too many missing values
# Keep only rows where key features are present
key_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'target']
df = df.dropna(subset=key_features)

print("After removing missing values:", df.shape)
print("Remaining columns:", df.columns.tolist())

# Encode categorical variables before preprocessing
from sklearn.preprocessing import LabelEncoder

categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = df[col].fillna('Unknown')  # Fill any remaining NaN
        df[col] = le.fit_transform(df[col].astype(str))

# Fill remaining numeric NaN with median
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    if col != 'target':
        df[col] = df[col].fillna(df[col].median())

print("Final shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())

# Preprocess
preprocessor = HeartDiseasePreprocessor()
X_train, X_test, y_train, y_test = preprocessor.preprocess(df, target_col='target')

# Save the preprocessor
with open('models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Train
model_trainer = HeartDiseaseModel()
model_trainer.create_ensemble()
results = model_trainer.train_and_compare(X_train, y_train, X_test, y_test)

# Save
model_trainer.save_model('models/best_model.pkl')

print("\n✅ Training complete!")
print("Run: streamlit run app.py")