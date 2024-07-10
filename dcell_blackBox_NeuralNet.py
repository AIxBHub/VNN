import pandas as pd
from utils import *
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import Sequence
from keras.optimizers import Adam, AdamW
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import glob
import scipy


# Add arguments for the directory and neuron number
parser = argparse.ArgumentParser()
parser.add_argument('--inputFiles', type=str, help='directory of input files')
parser.add_argument('--directory', type=str, help='Directory name')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (default: 20)')
parser.add_argument('--layers', type=int, default=9, help='Number of layers (default: 9)')
parser.add_argument('--filename', type=str, required=True, help='CSV file to read')
parser.add_argument('--percent', type=int, required=True, help='Percent of Data')
parser.add_argument('--label', type=str, help='Output label')
parser.add_argument('--batch', type=int, default=800, help='Batch Size')
parser.add_argument('--optimizer', type=str, help='optimizer choice')
args = parser.parse_args()

# Load and preprocess the data
dtypes = {'Query_allele': str, 'Array_allele': str, 'Double_mutant_fitness': float, 'weighted_eps_pvalue': float, 'Genetic_interaction_score': float}
chunksize = 100000  # Adjust the chunksize based on available memory and processing capabilities

# Construct the full path to the CSV file
file_path = os.path.join(args.inputFiles, args.filename)

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
df = df.drop_duplicates(subset=["Query_allele", "Array_allele"], keep="first")

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
x_testing.to_csv(f'{args.directory}/{args.percent}_percent/{args.label}_{args.layers}layers_{args.epochs}epochs_{args.batch}batches_test_x.csv', index=False)
y_testing.to_csv(f'{args.directory}/{args.percent}_percent/{args.label}_{args.layers}layers_{args.epochs}epochs_{args.batch}batches_test_y.csv', index=False)

#gets first layer number of neurons based on total number of alleles
neuron_nb = len(set(x_model['Query_allele']).union(set(x_model['Array_allele'])))
neurons = neuron_nb
print("Call function to create binary matrix for input")
binary_inputs = create_binary_matrix(x_model, 'Query_allele', 'Array_allele')
output = pd.DataFrame(y_model)

print("Split the data into training and validation sets")
x_train, x_val, y_train, y_val = train_test_split(binary_inputs, output, test_size=0.2)

# Define the model architecture
model = Sequential()

# Add the first layer with input dimension
model.add(Dense(neuron_nb, activation='tanh', input_dim=x_train.shape[1]))

# Add 10 hidden layers
for _ in range(args.layers):        #Visualize structure
  neuron_nb //= 2
  neuron_nb = int(neuron_nb)
  model.add(Dense(neuron_nb, activation='tanh'))
# Add the final output layer
model.add(Dense(1, activation='linear'))

#Rate scheduler 
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0001)  # Adjust parameters as needed

batch_size = args.batch
batch_file = batch_size/1000

print("Create data generators")
train_data_generator = DataGenerator(x_train, y_train, batch_size=batch_size)
val_data_generator = DataGenerator(x_val, y_val, batch_size=batch_size)

# Set learning rate
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate, decay=0.004)

stopEarly = EarlyStopping(monitor='val_r2score', mode='max', patience=15)

# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[r2score])

# Sets a checkpoint for model to save best Pearson correlation over epochs
checkpoint_file = f'{directory}/{args.percent}_percent/rawData_{args.directory}_{args.label}_{args.epochs}epochs_{args.batch}k_{neuron_nb}neurons_{args.layers}layers_bestPearsonCorr.keras'
model_checkpoint_callback = ModelCheckpoint(
  filepath=checkpoint_file,
  monitor='val_r2score',
  mode='max',
  verbose=1,
  save_best_only=True)

#Train model
print("Train the model using data generators")
history = model.fit(train_data_generator, epochs=args.epochs, validation_data=val_data_generator, callbacks=[stopEarly, model_checkpoint_callback])

# Save the history object to a file
with open(f'{args.directory}/training_history.pkl', 'wb') as history_file:
  pickle.dump(history.history, history_file)

# Evaluate the model
score = model.evaluate(val_data_generator)

# Save model
save_model_with_filename(model, neurons, batch=batch_file, directory=args.directory, label=args.label, percent=args.percent, epochs=args.epochs, layer=args.layers)

# Make predictions using the trained checkpoint model
model = load_model(checkpoint_file)
predictions = model.predict(x_testing)


# Plot the training and validation loss
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
plt.savefig(f'{args.directory}/{args.percent}_percent/lossVSepochs_{args.directory}_{args.label}_{batch_file}kbatch_{args.epochs}Epoch_{neurons}neurons_{args.layers}layers_plot.png')

# Plot the R2 score
plt.figure()
plt.plot(history.history['r2score'], label='Training Corr')
plt.plot(history.history['val_r2score'], label='Validation Corr')
plt.xlabel('Epochs')
plt.ylabel('Pearson Correlation')
plt.title('Training and Validation Correlation by Epoch')
plt.legend()
plt.show()
plt.savefig(f'{args.directory}/{args.percent}_percent/r2scoreVSepoch_{args.directory}_{args.label}_{batch_file}kbatch_{args.epochs}Epoch_{neurons}neurons_{args.layers}layers_plot.png')

# Plot the predictions vs actual values
plt.figure()
plt.scatter(y_testing, predictions, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predictions vs Actual Values')
plt.plot([min(y_testing), max(y_testing)], [min(y_testing), max(y_testing)], 'r--')  # Diagonal line
plt.show()
plt.savefig(f'{args.directory}/{args.percent}_percent/predictionsVSactual_{args.directory}_{args.label}_{batch_file}kbatch_{args.epochs}Epoch_{neurons}neurons_{args.layers}layers_plot.png')
