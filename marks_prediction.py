import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv(r"D:\Porfolio\marks.csv")

# Split into X and Y
X = data[['Hours']]
y = data['Marks']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict test
pred = model.predict(X_test)

print("Actual Marks:")
print(y_test)

print("\nPredicted Marks:")
print(pred)

# Predict your own input
hours = int(input("Enter study hours: "))
result = model.predict(pd.DataFrame([[hours]], columns=['Hours']))
print(f"Predicted Marks for {hours} hours = {result[0]:.2f}")
