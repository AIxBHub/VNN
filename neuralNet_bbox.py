import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the data
df = data = pd.read_csv("GenePair_GrowthRate_Costanzo2016.csv",delimiter=',')

# Sample data
# df = data.sample(n = 50)

# Split the data into inputs (gene pairs) and output (cell growth rate)
inputs = df[['Gene1', 'Gene2']]
output = df['Sga Score']

# Convert gene pairs to binary vectors
binary_inputs = pd.get_dummies(inputs.astype(str), prefix='', prefix_sep='')

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(binary_inputs, output, test_size=0.2)

# Define the model architecture
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=binary_inputs.shape[1]))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))

# Evaluate the model
score = model.evaluate(x_val, y_val)

# Save model
model.save('model.h5')

#Print plot of model loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
#Save the plot to a file
plt.savefig('loss_plot.png')
