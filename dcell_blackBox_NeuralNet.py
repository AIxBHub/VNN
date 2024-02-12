import pandas as pd
from utils import create_binary_matrix, DataGenerator, split_data_generator
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import Sequence
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import glob

# Add arguments for the directory and neuron number
parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, help='Directory name')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (default: 20)')
parser.add_argument('--layers', type=int, default=9, help='Number of layers (default: 9)')
parser.add_argument('--filename', type=str, required=True, help='CSV file to read')
args = parser.parse_args()

# Load and preprocess the data
dtypes = {'Query_allele': str, 'Array_allele': str, 'Double_mutant_fitness': float, 'weighted_eps_pvalue': float, 'Genetic_interaction_score': float}
chunksize = 100000  # Adjust the chunksize based on available memory and processing capabilities

# Construct the full path to the CSV file
file_path = os.path.join('input_files', args.filename)

data_chunks = pd.read_csv(file_path, delimiter=',', dtype=dtypes, chunksize=chunksize)

# Initialize an empty list to store chunks
df_list = []

# Read and append each chunk to the list
for chunk in data_chunks:
    df_list.append(chunk)

# Concatenate all chunks into a single DataFrame
df = pd.concat(df_list, ignore_index=True)

del df_list

num_rows, num_columns = df.shape
print("Number of rows:", num_rows)
print("Number of columns:", num_columns)

# Sample data
# df = data.sample(n = 50)
df.memory_usage(deep=True).sum()

print("randomize sample observations")
df = df.sample(frac=1, random_state=42)

# Split your data into training and validation sets before using the generator
test_size_downstream = 80000
inputs = df[['Query_allele', 'Array_allele']]
output = df['Double_mutant_fitness']
x_model, x_testing, y_model, y_testing = train_test_split(inputs, output, test_size=test_size_downstream, random_state = 42)

# delete dataframe
del df

# Record the indexes of the training set
training_set_indexes = x_testing.index.tolist()

# Save the indexes of the training set to a CSV file
training_set_indexes_df = pd.DataFrame({'Index': training_set_indexes})
training_set_indexes_df.to_csv(f'{args.directory}/training_set_indexes.csv', index=False)

# Save validation data to a CSV file
x_testing.to_csv(f'{args.directory}/{args.layers}layers_{args.epochs}epochs_alldata_test_x.csv', index=False)
y_testing.to_csv(f'{args.directory}/{args.layers}layers_{args.epochs}epochs_alldata_test_y.csv', index=False)

#gets first layer number of neurons based on total number of alleles
neuron_nb = len(set(x_model['Query_allele']).union(set(x_model['Array_allele'])))

print("Call function to create binary matrix for input")
binary_inputs = create_binary_matrix(x_model, 'Query_allele', 'Array_allele')
output = pd.DataFrame(y_model)

print("Split the data into training and validation sets")
# data_split_generator = split_data_generator(binary_inputs, output)
# x_train, y_train, x_val, y_val = next(data_split_generator)

x_train, x_val, y_train, y_val = train_test_split(binary_inputs, output, test_size=0.2)

# Define the model architecture
model = Sequential()

# Add the first layer with input dimension
model.add(Dense(neuron_nb, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(813, activation = 'relu'))
# Add 10 hidden layers
for _ in range(args.layers):
    model.add(Dense(813, activation='relu'))

# Add the final output layer
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
batch_size = 10000
print("Create data generators")
train_data_generator = DataGenerator(x_train, y_train, batch_size=batch_size)
val_data_generator = DataGenerator(x_val, y_val, batch_size=batch_size)
batch_file = batch_size/1000
print("Train the model using data generators")
history = model.fit(train_data_generator, epochs=args.epochs, validation_data=val_data_generator)

# Save the history object to a file
# with open(f'{args.directory}/training_history.pkl', 'wb') as history_file:
#     pickle.dump(history.history, history_file)

# Evaluate the model
score = model.evaluate(val_data_generator)

# Save model
save_model_with_filename(model, neuron_nb, batch=batch_file, directory=args.directory, epochs=args.epochs, layer=args.layers)

# Make predictions using the trained model
predictions = model.predict(x_val)

# Plot the scores
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
plt.savefig(f'{args.directory}/rawData_{args.directory}_{batch_file}kbatch_{args.epochs}Epoch_{neuron_nb}neurons_{args.layers}layers_plot.png')
