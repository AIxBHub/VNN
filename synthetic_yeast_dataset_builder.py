import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Step 1: Select top-level GO term
G_0 = "GO:0006281"

# Step 2: Find subclasses of G_0 (set P)
# Code to retrieve subclasses of G_0 from Gene Ontology database or any other source

P = [
    "YHR099W", "YDR540C", "YJL081C", "YDR448W", "YMR167W", "YOL133W", "YBR073W", "YMR173W", "YHR079C", "YML095C",
    "YDL154W", "YKL045W", "YOL034W", "YJR043C", "YDR263C", "YPR175W", "YEL018W", "YGL090W", "YDR225W", "YPL164C",
    "YHR154W", "YIL143C", "YER038C", "YDR446W", "YIL128W", "YDR288W", "YOR191W", "YPL235W", "YLR035C", "YOR023C",
    "YDL140C", "YDL200C", "YDR190C", "YML021C", "YPR056W", "YOR386W", "YDR180W", "YER173W", "YLR383W", "YBR223C",
    "YDR311W", "YGL201C", "YML023C", "YIL153W", "YBR010W", "YGR056W", "YJL168C", "YPL262W", "YPL204W", "YPL121C",
    "YML011C", "YEL037C", "YFR038W", "YNL107W", "YCR014C", "YGL127C", "YOL043C", "YPL127C", "YNL215W", "YDR460W",
    "YHR120W", "YOL094C", "YMR076C", "YFR037C", "YGL113W", "YOR368W", "YDL013W", "YFL024C", "YJL142C", "YPL046C",
    "YPL122C", "YDR499W", "YJR068W", "YER147C", "YPL022W", "YLR005W", "YGR002C", "YNL102W", "YOR005C", "YDR079C",
    "YGL163C", "YDR004W", "YGL168W", "YLR265C", "YNL290W", "YBR163W", "YKL067W", "YBR114W", "YDR334W", "YER162C",
    "YDL164C", "YJR046W", "YEL032W", "YLL036C", "YOL095C", "YDR419W", "YPR164W", "YBL023C", "YEL019C", "YDL105W",
    "YNL250W", "YMR114C", "YNL082W", "YOR328W", "YER095W", "YJL047C", "YER171W", "YLR320W", "YKL114C", "YJR006W",
    "YJR082C", "YMR201C", "YJL065C", "YAL019W", "YLR274W", "YDR076W", "YAL015C", "YLR151C", "YBR087W", "YLR394W",
    "YJR052W", "YOL087C", "YDR314C", "YBL003C", "YNL136W", "YDR363W"
]

# Step 3: Define parameters for the Gaussian distributions
mu_f = 0  # mean for distribution f
sigma = 0.2  # standard deviation for both distributions
delta = 1  # difference in mean between distributions f and g

# Step 4: Read genes from file
with open('20231219_final_gene_list.txt', 'r') as f:
    all_genes = f.read().splitlines()

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
df_sub = pd.DataFrame(sub_dataset, columns=['Query_allele', 'Array_allele', 'Genetic_interaction_score','Distribution'])

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"synthetic_yeast_data_{current_datetime}_{len(dataset)/1000000}mil.csv"
df.to_csv(filename, index=False)
df_label = f'Full Synthetic Dist N:{len(dataset)}'
dfsub_label =  f'Full Synthetic Dist N:{len(sub_dataset)}'
fig_file = f"Synthetic_Yeast_data_{len(dataset)/1000000}mil.png"
plt.figure(figsize=(15, 10))
sns.kdeplot(df['Genetic_interaction_score'].values.flatten(), color='red', alpha = 0.5, label = df_label)
sns.kdeplot(df_sub['Genetic_interaction_score'].values.flatten(), color='blue', alpha = 0.5, label = dfsub_label)
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.savefig(fig_file)
