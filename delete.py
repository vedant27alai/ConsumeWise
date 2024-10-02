import joblib

model = joblib.load('trained_model.pkl')
print(type(model))
