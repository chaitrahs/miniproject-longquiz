import joblib

model = joblib.load("model.joblib")

def predict(data):
    return model.predict([data]).tolist()
