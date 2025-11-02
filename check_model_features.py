import joblib

# Load the model
model = joblib.load('models/best_model.pkl')

# Get feature names from the model
if hasattr(model, 'feature_names_in_'):
    print("Expected feature names and order:")
    for i, feature in enumerate(model.feature_names_in_):
        print(f"{i+1}. {feature}")
else:
    print("Model doesn't have feature_names_in_ attribute")