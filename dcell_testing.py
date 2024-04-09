import pandas as pd
from utils import *
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import Sequence
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import glob
import scipy
import seaborn as sns

# Add arguments for the directory and neuron number
parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, help='Directory name')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (default: 20)')
parser.add_argument('--layers', type=int, default=9, help='Number of layers (default: 9)')
parser.add_argument('--filename', type=str, required=True, help='CSV file to read')
parser.add_argument('--percent', type=int, required=True, help='Percent of Data')
parser.add_argument('--label', type=str, help='Output label')
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
df = df.sample(frac=1, random_state=42)  # Randomize dataset 

# Split your data into training and validation sets before using the generator
test_size_downstream = len(df.sample(frac = 0.1, random_state = 42))  #Set seed number
inputs = df[['Query_allele', 'Array_allele']]
if args.label == "Double_mutant_fitness":
  output = np.log(df[args.label])
else:
  output = df[args.label]
# output = df[args.label]
x_model, x_testing, y_model, y_testing = train_test_split(inputs, output, test_size=test_size_downstream, random_state = 42)

# delete dataframe
del df

# Record the indexes of the training set
test_set_indexes = x_testing.index.tolist()

# Save the indexes of the training set to a CSV file
directory = f"{args.directory}/{args.percent}_percent/"

# Check if the directory exists, if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)
    
test_set_indexes_df = pd.DataFrame({'Index': test_set_indexes})
test_set_indexes_df.to_csv(f'{args.directory}/{args.percent}_percent/{args.label}_test_set_indexes.csv', index=False)

# Save validation data to a CSV file
x_testing.to_csv(f'{args.directory}/{args.percent}_percent/{args.label}_{args.layers}layers_{args.epochs}epochs_alldata_test_x.csv', index=False)
y_testing.to_csv(f'{args.directory}/{args.percent}_percent/{args.label}_{args.layers}layers_{args.epochs}epochs_alldata_test_y.csv', index=False)

#gets first layer number of neurons based on total number of alleles
neuron_nb = len(set(x_model['Query_allele']).union(set(x_model['Array_allele'])))
neurons = neuron_nb
print("Call function to create binary matrix for input")
binary_inputs = create_binary_matrix(x_model, 'Query_allele', 'Array_allele')
output = pd.DataFrame(y_model)

print("Split the data into training and validation sets")
# data_split_generator = split_data_generator(binary_inputs, output)
# x_train, y_train, x_val, y_val = next(data_split_generator)

x_train, x_val, y_train, y_val = train_test_split(binary_inputs, output, test_size=0.2)

# Define the model architecture
model = Sequential()
activation = 'relu' #tanh
# Add the first layer with input dimension
model.add(Dense(neuron_nb, activation=activation, input_dim=x_train.shape[1]))
#model.add(Dense(813, activation = 'relu'))
# Add 10 hidden layers
for _ in range(args.layers):        #Visualize structure
  neuron_nb //= 2
  neuron_nb = int(neuron_nb)
  model.add(Dense(neuron_nb, activation=activation))
# Add the final output layer
model.add(Dense(1, activation='linear'))

#Rate scheduler 
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0001)  # Adjust parameters as needed

batch_size = 10000
batch_file = batch_size/1000

print("Create data generators")
train_data_generator = DataGenerator(x_train, y_train, batch_size=batch_size)
val_data_generator = DataGenerator(x_val, y_val, batch_size=batch_size)

# Set learning rate
learning_rate = 0.001
optimizer = Adam(lr=learning_rate)

stopEarly = EarlyStopping(monitor='loss', patience=3)

# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[r2score])


# Before the first epoch
predictions_before_training_val = model.predict(x_val)
predictions_before_training_train = model.predict(x_train)

# Train the model for one epoch
history = model.fit(train_data_generator, epochs=1, validation_data=val_data_generator, callbacks=[reduce_lr, stopEarly])

# After the first epoch
predictions_after_training_val = model.predict(x_val)
predictions_after_training_train = model.predict(x_train)

# Evaluate the model
score = model.evaluate(val_data_generator)

# Save model
save_model_with_filename(model, neurons, batch=batch_file, directory=args.directory, label=args.label, percent=args.percent, epochs=args.epochs, layer=args.layers)

plt.figure(figsize=(40, 10))

plt.subplot(1, 3, 1)
sns.kdeplot(y_val.values.flatten(), color='red', alpha = 0.5, label = 'Real Values (Validation)')
sns.kdeplot(y_train.values.flatten(), color='green', alpha = 0.5, label = 'Real Values (Training)')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()

# Plotting before first epoch
plt.subplot(1, 3, 2)
sns.kdeplot(predictions_before_training_val.flatten(), color='red', alpha=0.5, label='Predicted Values (Validation)')
sns.kdeplot(predictions_before_training_train.flatten(), color='blue', alpha=0.5, label='Predicted Values (Training)')
plt.title('Before First Epoch')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()

# Plotting after first epoch
plt.subplot(1, 3, 3)
sns.kdeplot(predictions_after_training_val.flatten(), color='red', alpha=0.5, label='Predicted Values (Validation)')
sns.kdeplot(predictions_after_training_train.flatten(), color='blue', alpha=0.5, label='Predicted Values (Training)')
plt.title('After First Epoch')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()

plt.savefig('overlayed_distributions_realValue.png')
