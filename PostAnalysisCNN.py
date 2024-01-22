import os
import glob
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import dask.dataframe as dd
import dask.array as da
import statsmodels.api as sm
from collections import defaultdict
from scipy.sparse import csr_matrix


h5_dir = '.'

# List all H5 files in the directory
h5_files = glob.glob(os.path.join(h5_dir, '*.h5'))

# Check if there are any H5 files in the directory
if not h5_files:
    print("No H5 files found in the directory.")
else:
    # Select the H5 file you want to load
    selected_h5_file = h5_files[0]  # You can change this to select a different file
    # Load the selected model
    model = load_model(selected_h5_file)


# Define the CSV file path for validation_x
csv_file_path = 'alldata_test_x.csv'

# Define the number of rows to load
# rows_to_load = 100000  # Adjust the number of rows based on your testing needs

# Read a portion of the CSV file
X_val_sample = pd.read_csv(csv_file_path)

# Assuming 'X_val' columns are the same as the CSV file
predictions = model.predict(X_val_sample)

# Load the labels directly from the CSV file
validation_y = pd.read_csv('alldata_test_y.csv')['Double_mutant_fitness'].values

# Evaluate the model using regression metrics
mse = mean_squared_error(validation_y, predictions)
mae = mean_absolute_error(validation_y, predictions)
r_squared = r2_score(validation_y, predictions.flatten())

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared: {r_squared}')

# Plot the actual vs. predicted values
plt.scatter(validation_y, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')

# Add annotations for MSE, MAE, and R-squared to the legend
plt.annotate(f'MSE: {mse:.4f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10, color='red')
plt.annotate(f'MAE: {mae:.4f}', xy=(0.05, 0.80), xycoords='axes fraction', fontsize=10, color='red')
plt.annotate(f'R-squared: {r_squared:.4f}', xy=(0.05, 0.75), xycoords='axes fraction', fontsize=10, color='red')

plt.savefig('scatter_plot_with_annotations.png')  # Save the scatter plot with annotations

# Residuals Plot
residuals = validation_y - predictions.flatten()
plt.figure()
plt.scatter(validation_y, residuals)
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.axhline(y=0, color='red', linestyle='--')  # Add a horizontal line at y=0
plt.savefig('residuals_plot.png')

# Histogram of Residuals
plt.figure()
plt.hist(residuals, bins=50, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.savefig('histogram_of_residuals.png')

# Q-Q Plot
plt.figure()
sm.qqplot(residuals, line='45', fit=True)
plt.title('Q-Q Plot of Residuals')
plt.savefig('qq_plot.png')

# Prediction vs. Actual Line Plot
plt.figure()
plt.plot(validation_y, label='Actual Values')
plt.plot(predictions.flatten(), label='Predicted Values', linestyle='--')
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.title('Prediction vs. Actual Line Plot')
plt.legend()
plt.savefig('prediction_vs_actual_line_plot.png')
