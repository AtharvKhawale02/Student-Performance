import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Example dataset
data = {
    'attendance': [80, 90, 70, 60],
    'test_scores': [85, 95, 75, 65],
    'assignments': [88, 92, 78, 68],
    'performance': [90, 95, 80, 70]
}
df = pd.DataFrame(data)

X = df[['attendance', 'test_scores', 'assignments']]
y = df['performance']

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, 'performance/ml_model.pkl')  # Save the model in the performance folder
