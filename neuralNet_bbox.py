import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
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


# Sample data
# df = data.sample(n = 50)

inputs = df[['Query Strain ID', 'Array Strain ID']]
output = df['Genetic interaction score (ε)']

# Convert gene pairs to binary vectors
binary_inputs = pd.get_dummies(inputs.astype(str), prefix='', prefix_sep='')

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

# Train the model
history = model.fit(x_train, y_train, batch_size=10000, epochs=10, validation_data=(x_val, y_val))

# Evaluate the model
score = model.evaluate(x_val, y_val)

# Save model
model.save('rawData_NxN_batch10k_132neurons_12layers.h5')

#Print plot of model loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
#Save the plot to a file
plt.savefig('rawData_NxN_10k_132neurons_12layers_plot.png')
