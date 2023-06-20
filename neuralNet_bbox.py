import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the data
dtypes = {'Query Strain ID': str, 'Array Strain ID': str, 'Genetic interaction score (ε)': float, 'P-value': float}
chunksize = 100000  # Adjust the chunksize based on available memory and processing capabilities

data_chunks = pd.read_csv("SGA_NxN_trimmed.txt", delimiter=',', dtype=dtypes, chunksize=chunksize)

# Initialize an empty list to store chunks
df_list = []

# Read and append each chunk to the list
for chunk in data_chunks:
    df_list.append(chunk)

# Concatenate all chunks into a single DataFrame
df = pd.concat(df_list, ignore_index=True)

num_rows, num_columns = df.shape
print("Number of rows:", num_rows)
print("Number of columns:", num_columns)

# Sample data
# df = data.sample(n = 50)

inputs = df[['Query Strain ID', 'Array Strain ID']]
output = df['Genetic interaction score (ε)']

# Convert gene pairs to binary vectors
binary_mat = pd.get_dummies(inputs.astype(str), prefix='', prefix_sep='')

# Randomize the rows of the binary_mat DataFrame
binary_inputs = binary_mat.sample(frac=1, random_state=42)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(binary_inputs, output, test_size=0.2)

# Define the model architecture
model = Sequential()

# Add the first layer with input dimension
model.add(Dense(132, activation='relu', input_dim=binary_inputs.shape[1]))

# Add 10 hidden layers
for _ in range(10):
    model.add(Dense(64, activation='relu'))

# Add the final output layer
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Perform cross-validation
num_folds = 10  # Number of cross-validation folds
scores = cross_val_score(model, binary_inputs, output, cv=num_folds, scoring='neg_mean_squared_error')

# Print scores
print("Cross-validation scores:")
print(scores)

# Calculate mean and standard deviation of the scores
mean_score = -np.mean(scores)
std_score = np.std(scores)

print(f"Mean score: {mean_score:.4f}")
print(f"Standard deviation: {std_score:.4f}")

# Train the model
history = model.fit(x_train, y_train, batch_size=100000, epochs=100, validation_data=(x_val, y_val))

# Evaluate the model
score = model.evaluate(x_val, y_val)

# Save model
model.save('rawData_NxN_batch10k_132neurons_12layers.h5')

# Plot the scores
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_folds + 1), -scores, marker='o', linestyle='-', color='blue')
plt.xlabel('Fold')
plt.ylabel('Negative Mean Squared Error')
plt.title('Cross-Validation Scores')
plt.grid(True)

# Save the plot as an image file
plt.savefig('cross_validation_scores.png')

#Print plot of model loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
#Save the plot to a file
plt.savefig('rawData_NxN_100k_100Epoch_132neurons_12layers_plot.png')
