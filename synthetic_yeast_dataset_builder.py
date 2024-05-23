import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os

def extract_genes_for_root(file_path, root):
    genes_set = set()
    with open(file_path, 'r') as file:
        root_found = False
        for line in file:
            line = line.strip()
            if line.startswith("ROOT:") and line.split()[1] == root:
                root_found = True
            elif root_found and line.startswith("GENES:"):
                genes_set.update(line.split()[1:])
            elif root_found and line.startswith("ROOT:"):
                break  # Exit loop if next root is encountered
    return genes_set

parser = argparse.ArgumentParser()
parser.add_argument('--GO_term', type=str, help='GO term for set P')
args = parser.parse_args()
# Example usage:
# GO_0 = "GO:1905037"  # Change this to the desired root
# file_path = "your_file.txt"  # Change this to the actual file path
# genes = extract_genes_for_root(file_path, root)
# print("Genes for root", root, ":", genes)

# Step 1: Select top-level GO term
G_0 = "GO:0044237"

file_path = os.path.join(os.getcwd(),"20240426_0044237_ontology_v1")

P = extract_genes_for_root(file_path, G_0)

# Step 3: Define parameters for the Gaussian distributions
mu_f = 0  # mean for distribution f
sigma = 0.2  # standard deviation for both distributions
delta = 1  # difference in mean between distributions f and g

# Step 4: Read genes from file
with open('../20231219_final_gene_list.txt', 'r') as f:
    all_genes = f.read().splitlines()

# subset_size = 300-len(P)
# all_genes_not_P = set(all_genes) - set(P)
# all_genes_subset = set(np.random.choice(list(all_genes_not_P), subset_size, replace=False))
# all_genes = list(all_genes_subset.union(P))

# Step 5: Generate dataset
N = 3000000  # number of samples
dataset = []
sub_dataset = []

for _ in range(N):
  # Randomly select two genes
  gene1, gene2 = np.random.choice(all_genes, 2, replace=False)
  
  # Sample Y from Gaussian distribution
  if gene1 in P and gene2 in P:
    Y = np.random.normal(mu_f + delta, sigma) # Sample Y from distribution g
    distribution = 'g'
    sub_dataset.append((gene1, gene2, Y)) 
  else:
    Y = np.random.normal(mu_f, sigma)  # Sample Y from distribution f
    distribution = 'f'
  
  dataset.append((gene1, gene2, Y, distribution))
    
    
df = pd.DataFrame(dataset, columns=['Query_allele', 'Array_allele', 'Genetic_interaction_score','Distribution'])
df = df.drop_duplicates(subset=["Query_allele", "Array_allele"], keep="first")
df_sub = pd.DataFrame(sub_dataset, columns=['Query_allele', 'Array_allele', 'Genetic_interaction_score'])
df_sub = df_sub.drop_duplicates(subset=["Query_allele", "Array_allele"], keep="first")

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"synthetic_yeast_data_{G_0}_{current_datetime}_{len(dataset)/1000000}mil.csv"
df.to_csv(filename, index=False)
df_label = f'Full Synthetic F Dist N:{len(dataset)}'
dfsub_label =  f'G Dist N:{len(sub_dataset)}'
fig_file = f"Synthetic_Yeast_data_{G_0}_{len(dataset)/1000000}mil.png"
plt.figure(figsize=(15, 10))
sns.kdeplot(df['Genetic_interaction_score'].values.flatten(), color='red', alpha = 0.5, label = df_label)
sns.kdeplot(df_sub['Genetic_interaction_score'].values.flatten(), color='blue', alpha = 0.5, label = dfsub_label)
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.savefig(fig_file)
