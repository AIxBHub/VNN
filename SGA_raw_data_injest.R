#Installation	Install the latest version of this package by entering the following in R:
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
library(tidyr)
library(dplyr)

BiocManager::install("ctc")

library(data.table)
s1 <- fread("Data File S1. Raw genetic interaction datasets: Pair-wise interaction format/SGA_DAmP.txt")
# 
s1_2 <- fread("Data File S1. Raw genetic interaction datasets: Pair-wise interaction format/SGA_ExE.txt")
# 
s1_3 <- fread("Data File S1. Raw genetic interaction datasets: Pair-wise interaction format/SGA_ExN_NxE.txt")

s1_4 <- fread("Data File S1. Raw genetic interaction datasets: Pair-wise interaction format/SGA_NxN.txt")

allData <- rbind(s1,s1_2,s1_3,s1_4)

# s3 <- fread("Data File S3. Genetic interaction profile similarity matrices/cc_ALL.txt")
# s3
# s3_2 <- fread("Data File S3. Genetic interaction profile similarity matrices/cc_EE.txt")
# s3_2
# s3_4 <- fread("Data File S3. Genetic interaction profile similarity matrices/cc_NN.txt")
# s3_4

tempDat <- allData
tempDat <- tempDat[,c(1,3,10,6,7)]
head(tempDat)
dat <- tempDat
dat <- na.omit(dat)

dat$`Query Strain ID` <- gsub("_.*","",dat$`Query Strain ID`)
dat$`Query Strain ID` <- gsub("-.*","",dat$`Query Strain ID`)
dat$`Array Strain ID` <- gsub("_.*","",dat$`Array Strain ID`)
dat$`Array Strain ID` <- gsub("-.*","",dat$`Array Strain ID`)

#Check for repeating gene-pairs and how many
dat$allele_pair <- paste(dat$`Query Strain ID`,dat$`Array Strain ID`,sep = "_")
allele_pair_counts <- table(dat$allele_pair)
dim(allele_pair_counts[allele_pair_counts > 1])

# Use arrange within slice to select the top row for each gene pair
dat <- dat %>%
  arrange(allele_pair, `P-value`) %>%
  distinct(allele_pair, .keep_all = TRUE)


# Aggregate the growth score for each gene pair
# aggregated_growth <- aggregate(dat$`Double mutant fitness`, by = list(dat$allele_pair), FUN = mean)
# 
# # Calculate the weights based on the p-values
# weights <- 1 / dat$`P-value`

# Add a small positive value to all p-values
epsilon <- 1e-10
adjusted_pvalues <- pmax(dat$`P-value`, epsilon)

# Calculate the weighted average Pvalue for each gene pair
# weighted_pvalue <- tapply(dat$`P-value`* adjusted_pvalues,
#                           dat$allele_pair, sum) / tapply(adjusted_pvalues, 
#                                                         dat$allele_pair, sum)

# Create a new condensed data table with allele pairs, aggregated growth scores, and weighted average p-values
# condensed_data <- data.table(
#   allele_pair = unique(dat$allele_pair),
#   aggregated_growth_score = aggregated_growth$x,
#   weighted_eps_pvalue = weighted_pvalue,
#   Genetic_interaction_score = (dat$`Genetic interaction score (ε)`)
# )
condensed_data <- data.table(
  Query_allele = dat$`Query Strain ID`,
  Array_allele = dat$`Array Strain ID`,
  Double_mutant_fitness = dat$`Double mutant fitness`,
  weighted_eps_pvalue = adjusted_pvalues,
  Genetic_interaction_score = (dat$`Genetic interaction score (ε)`)
)

# condensed_data <- separate(condensed_data, allele_pair, 
#                            into = c("Query_allele", "Array_allele"), sep = "_")

write.csv(condensed_data, "AllData_lowestPval_weightedPval.csv")

allele_list <- readLines("../20231219_final_gene_list.txt")
allele_list <- as.list(allele_list)

# Set the percentage of genes to sample
percentage_to_sample <- 0.75 

# Randomly sample genes
set.seed(123)  
sampled_genes <- sample(allele_list, size = round(length(allele_list) * percentage_to_sample))

# Filter the dataset based on the sampled genes for both columns
filtered_data <- condensed_data %>% 
  filter(Query_allele %in% sampled_genes & Array_allele %in% sampled_genes)

# Write the sampled gene list to a file
write.csv(sampled_genes, "Sampled_gene_list_75percent.txt")

write.csv(filtered_data, "AllData_lowestPval_weightedPval_filtered_sampled_75percent.csv")
