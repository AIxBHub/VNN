#Installation	Install the latest version of this package by entering the following in R:
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
library(tidyr)
library(dplyr)

# BiocManager::install("ctc")
library(data.table)

# Point to data source
setwd("~/RENCI/CreativityHub_NNetworks")
# Home of the data
# https://boonelab.ccbr.utoronto.ca/supplement/costanzo2009/

# Read data in 
s1 <- fread("sgadata_costanzo2009_rawdata_101120.txt")
s2 <- fread("sgadata_costanzo2009_lenientCutoff_101120.txt")

# Format headers to match nomeclature downstream
new_names <- c("Query Strain ID","Query_gene","Array Strain ID","Array_gene","Genetic interaction score (ε)",
               "Standard_deviation","P-value","Query single mutant fitness (SMF)","Query SMF standard deviation",
               "Array SMF","Array SMF standard deviation","Double mutant fitness",
               "Double mutant fitness standard deviation")
names(s1) <- new_names
# Subset data to relevant columns
dat <- s1[,c(1,3,5,7, 12)]

# Separate data into essential (temp-sensitive and DAmP) and non-essential gene data tables
dat_allgenes <- na.omit(dat)
dat_essent <- dat_allgenes[grep("_", dat_allgenes$`Query Strain ID`)] # all genes that are temp sensitive and Damp
dat_nonEssent <- dat_allgenes[!grep("_", dat_allgenes$`Query Strain ID`)] # all other genes (most of which are deletion mutants)
dat_nonEssent2 <- dat_nonEssent[!grep("-", dat_nonEssent$`Array Strain ID`)] # minus alternative alleles
dat_nonEssent2 <- dat_nonEssent2[!grep("-", dat_nonEssent2$`Query Strain ID`)]
dat <- dat_nonEssent2


# dat$`Query Strain ID` <- gsub("_.*","",dat$`Query Strain ID`)
# dat$`Query Strain ID` <- gsub("-.*","",dat$`Query Strain ID`)
# dat$`Array Strain ID` <- gsub("_.*","",dat$`Array Strain ID`)
# dat$`Array Strain ID` <- gsub("-.*","",dat$`Array Strain ID`)

#Check for repeating gene-pairs and how many
dat$allele_pair <- paste(dat$`Query Strain ID`,dat$`Array Strain ID`,sep = "_")
allele_pair_counts <- table(dat$allele_pair)
dim(allele_pair_counts[allele_pair_counts > 1])

# Use arrange within slice to select the top row for each gene pair
dat <- dat %>%
  arrange(allele_pair, `P-value`) %>%
  distinct(allele_pair, .keep_all = TRUE)

# Add a small positive value to all p-values
epsilon <- 1e-10
adjusted_pvalues <- pmax(dat$`P-value`, epsilon)


condensed_data <- data.table(
  Query_allele = dat$`Query Strain ID`,
  Array_allele = dat$`Array Strain ID`,
  Double_mutant_fitness = dat$`Double mutant fitness`,
  weighted_eps_pvalue = adjusted_pvalues,
  Genetic_interaction_score = (dat$`Genetic interaction score (ε)`)
)

# Locate preprocessed gene list and navigate there
setwd("~/PGC_renci/dariusmb/AIxB/input_files_costanzo09/")

 # write.csv(condensed_data, "AllData_lowestPval_weightedPval.csv")

allele_list <- readLines("../20231219_final_gene_list.txt")
allele_list <- as.list(allele_list)

# List of percentage values
percentages <- c(25, 50, 75, 100)

# Iterate over each percentage value
# This creates 4 datasets at varying percentages of the whole for testing purposes. Can just set percent to 100 to avoid extra data tables.
for (percent in percentages) {
  # Set the percentage of genes to sample
  percentage_to_sample <- percent / 100
  
  # Randomly sample genes
  set.seed(123)  
  sampled_genes <- sample(allele_list, size = round(length(allele_list) * percentage_to_sample))
  
  # Filter the dataset based on the sampled genes for both columns
  filtered_data <- condensed_data %>% 
    filter(Query_allele %in% sampled_genes & Array_allele %in% sampled_genes)
  
  # Write the sampled gene list to a file
  write.csv(sampled_genes, paste0("NxN_Sampled_gene_list_", percent, "percent.txt"))
  
  write.csv(filtered_data, paste0("NxN_lowestPval_weightedPval_", percent, "percent.csv"))
}
