import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("student_performance.csv")

# Features
X = df[[
    "Hours_Studied",
    "Attendance",
    "Previous_Score",
    "Assignments"
]]

# Target
y = df["Performance"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained successfully!")