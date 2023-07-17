#Installation	Install the latest version of this package by entering the following in R:
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
library(tidyr)
BiocManager::install("ctc")

library(data.table)
s1 <- fread("~/Desktop/RENCI/CreativityHub_NNetworks/Data File S1. Raw genetic interaction datasets: Pair-wise interaction format/SGA_DAmP.txt")

s1_2 <- fread("~/Desktop/RENCI/CreativityHub_NNetworks/Data File S1. Raw genetic interaction datasets: Pair-wise interaction format/SGA_ExE.txt")

s1_3 <- fread("~/Desktop/RENCI/CreativityHub_NNetworks/Data File S1. Raw genetic interaction datasets: Pair-wise interaction format/SGA_ExN_NxE.txt")

s1_4 <- fread("~/Desktop/RENCI/CreativityHub_NNetworks/Data File S1. Raw genetic interaction datasets: Pair-wise interaction format/SGA_NxN.txt")



# s3 <- fread("Data File S3. Genetic interaction profile similarity matrices/cc_ALL.txt")
# s3
# s3_2 <- fread("Data File S3. Genetic interaction profile similarity matrices/cc_EE.txt")
# s3_2
# s3_4 <- fread("Data File S3. Genetic interaction profile similarity matrices/cc_NN.txt")
# s3_4

tempDat <- s1_4
tempDat <- tempDat[,c(1,3,10,7)]
head(tempDat)
dat <- tempDat
dat <- na.omit(dat)

dat$`Query Strain ID` <- gsub("_.*","",dat$`Query Strain ID`)
dat$`Array Strain ID` <- gsub("_.*","",dat$`Array Strain ID`)

#Check for repeating gene-pairs and how many
dat$allele_pair <- paste(dat$`Query Strain ID`,dat$`Array Strain ID`,sep = "_")
allele_pair_counts <- table(dat$allele_pair)
dim(allele_pair_counts[allele_pair_counts > 1])

# Aggregate the growth score for each gene pair
aggregated_growth <- aggregate(dat$`Double mutant fitness`, by = list(dat$allele_pair), FUN = sum)

# Calculate the weights based on the p-values
weights <- 1 / dat$`P-value`

# Add a small positive value to all p-values
epsilon <- 1e-10
adjusted_pvalues <- pmax(dat$`P-value`, epsilon)

# Calculate the weighted average Pvalue for each gene pair
weighted_pvalue <- tapply(dat$`P-value`* adjusted_pvalues,
                          dat$allele_pair, sum) / tapply(adjusted_pvalues, 
                                                        dat$allele_pair, sum)

# Create a new condensed data table with allele pairs, aggregated growth scores, and weighted average p-values
condensed_data <- data.table(
  allele_pair = unique(dat$allele_pair),
  aggregated_growth_score = aggregated_growth$x,
  weighted_average_pvalue = weighted_pvalue
)

condensed_data <- separate(condensed_data, allele_pair, 
                           into = c("Query_allele", "Array_allele"), sep = "_")
