import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict
from keras.utils import Sequence
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


def r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def r2score(y_true, y_pred):
    sum_squares_residuals = K.sum((y_true - y_pred) ** 2)
    sum_squares = K.sum((y_true - K.mean(y_true)) ** 2)
    return(( 1 - sum_squares_residuals ) / sum_squares)

def save_model_with_filename(model, neuron_nb, directory, label, percent, epochs, batch, layer):
    
    # Create the filename using the provided directory argument
    filename = f'{directory}/{percent}_percent/rawData_{directory}_{label}_{epochs}epochs_{batch}k_{neuron_nb}neurons_{layer}layers.h5'
    
    # Save the model with the dynamic filename
    model.save(filename)

def create_binary_matrix(df, gene_col1, gene_col2):
    
    # Create a mapping of unique genes to column indices in the binary matrix
    gene_to_index = defaultdict(lambda: len(gene_to_index))
    
    data = []
    row_indices = []
    col_indices = []
    
    for i, (gene1, gene2) in enumerate(zip(df[gene_col1], df[gene_col2])):
        index1 = gene_to_index[gene1]
        index2 = gene_to_index[gene2]
        
        data.extend([1, 1])  # Add 1 to both gene1 and gene2 positions
        row_indices.extend([i, i])
        col_indices.extend([index1, index2])
        
    # Create a sparse matrix in Compressed Sparse Row (CSR) format
    binary_matrix = csr_matrix((data, (row_indices, col_indices)))
    
    # Create a DataFrame from the sparse matrix
    binary_matrix_df = pd.DataFrame.sparse.from_spmatrix(binary_matrix, columns=list(gene_to_index.keys()))
    
    return binary_matrix_df


class DataGenerator(Sequence):
    def __init__(self, x, y, batch_size, shuffle=True):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.x))
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indexes = self.indexes[start_idx:end_idx]
        batch_x = self.x.iloc[batch_indexes]
        batch_y = self.y.iloc[batch_indexes]

        # Ensure the shape is (None, None) for dynamic batch sizes
        return batch_x.values, batch_y.values
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def split_data_generator(binary_inputs, output, test_size=0.2, batch_size=1000, shuffle=True):
    indexes = np.arange(len(binary_inputs))
    if shuffle:
        np.random.shuffle(indexes)

    start_idx = 0
    while True:
        end_idx = start_idx + batch_size
        if end_idx >= len(binary_inputs):
            break

        batch_indexes = indexes[start_idx:end_idx]
        batch_x = binary_inputs.iloc[batch_indexes]
        batch_y = output.iloc[batch_indexes]

        # Split the batch data
        x_train, x_val, y_train, y_val = train_test_split(batch_x, batch_y, test_size=test_size)

        start_idx = end_idx

        #yield x_train, y_train, x_val, y_val
        yield start_idx, end_idx
