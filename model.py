import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset (update path after downloading from Kaggle)
data = pd.read_csv('train.csv')

# Select features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

data = data[features + [target]].dropna()

X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Example prediction
example = [[2000, 3, 2]]
print("Predicted Price:", model.predict(example))
