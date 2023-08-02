import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import matplotlib.pyplot as plt

def create_binary_matrix(df, gene_col1, gene_col2):
    # Create a set of unique genes from both columns
    unique_genes = set(df[gene_col1]).union(set(df[gene_col2]))
    
    # Create a mapping of unique genes to column indices in the binary matrix
    gene_to_index = {gene: index for index, gene in enumerate(unique_genes)}
    
    # Create an empty binary matrix (all zeros) with rows equal to the number of observations (gene pairs) and columns equal to the number of unique genes
    binary_matrix = np.zeros((len(df), len(unique_genes)), dtype=int)
    
    # Fill the binary matrix based on the presence of genes in each gene pair
    for i, (gene1, gene2) in enumerate(zip(df[gene_col1], df[gene_col2])):
      index1 = gene_to_index[gene1]
      index2 = gene_to_index[gene2]
      binary_matrix[i, index1] = 1
      binary_matrix[i, index2] = 1
      
    # Convert the set 'unique_genes' to a list
    unique_genes_list = list(unique_genes)
    
    # Create a DataFrame from the binary matrix with proper column names
    binary_matrix_df = pd.DataFrame(binary_matrix, columns=unique_genes_list)
    
    return binary_matrix_df


# Load and preprocess the data
dtypes = {'Query_allele': str, 'Array_allele': str, 'aggregated_growth_score': float, 'weighted_average_pvalue': float}
chunksize = 100000  # Adjust the chunksize based on available memory and processing capabilities

data_chunks = pd.read_csv("yeastRawData_StrainID_aggregateGrowth_weightedPval.csv", delimiter=',', dtype=dtypes, chunksize=chunksize)

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

# randomize sample observations
df = df.sample(frac=0.2, random_state=42)

# inputs = df[['Query_allele', 'Array_allele']]

# Call function to create binary matrix for input
binary_inputs = create_binary_matrix(df, 'Query_allele', 'Array_allele')
output = df['aggregated_growth_score']

# delete dataframe
del df

# Convert gene pairs to binary vectors
# binary_mat = pd.get_dummies(inputs.astype(str))

# Randomize the rows of the binary_mat DataFrame
# binary_inputs = binary_matrix_df.sample(frac=1, random_state=42)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(binary_inputs, output, test_size=0.2)

# Define the model architecture
model = Sequential()

# Add the first layer with input dimension
model.add(Dense(4066, activation='relu', input_dim=binary_inputs.shape[1]))

# Add 10 hidden layers
for _ in range(10):
    model.add(Dense(64, activation='relu'))

# Add the final output layer
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Perform cross-validation
# num_folds = 10  # Number of cross-validation folds
# scores = cross_val_score(model, binary_inputs, output, cv=num_folds, scoring='neg_mean_squared_error')
# 
# # Print scores
# print("Cross-validation scores:")
# print(scores)
# 
# # Calculate mean and standard deviation of the scores
# mean_score = -np.mean(scores)
# std_score = np.std(scores)

# print(f"Mean score: {mean_score:.4f}")
# print(f"Standard deviation: {std_score:.4f}")

# Train the model
history = model.fit(x_train, y_train, batch_size=10000, epochs=300, validation_data=(x_val, y_val))

# Evaluate the model
score = model.evaluate(x_val, y_val)

# Save model
model.save('rawData_NxN_300epochs_batch10k_32neurons_12layers.h5')

# Plot the scores
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, num_folds + 1), -scores, marker='o', linestyle='-', color='blue')
# plt.xlabel('Fold')
# plt.ylabel('Negative Mean Squared Error')
# plt.title('Cross-Validation Scores')
# plt.grid(True)

# Save the plot as an image file
# plt.savefig('cross_validation_scores.png')

#Print plot of model loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
#Save the plot to a file
plt.savefig('rawData_NxN_10kbs_300Epoch_32neurons_12layers_plot.png')
