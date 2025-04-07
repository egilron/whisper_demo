import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate random data
np.random.seed(42)
x = np.random.uniform(0, 10, 50)
noise = np.random.normal(0, 3, 50)
y = 3 * x + 2 + noise
y2 = 3.25673 * x + 1.683736548409 + noise

# Fit regression line
model = LinearRegression()
x_reshaped = x.reshape(-1, 1)
model.fit(x_reshaped, y)
y_pred = model.predict(x_reshaped)

model.fit(x_reshaped, y2)
y_dec = model.predict(x_reshaped)

# Create a DataFrame
# data = pd.DataFrame({'X values': x, 'Y values': y, 'Predicted Y': y_pred})
# data.to_excel('scatter_plot_with_regression.xlsx', index=False)

# Plot scatter plot with regression line
plt.scatter(x, y, label='Data Points')
plt.plot(x, y_pred, color='red', label='Regression Line, int')
plt.plot(x, y_pred, color='yellow', label='Regression Line, dec')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Regresjonslinje, y=3.25673x + 1.683736548409')
plt.show()
