###This is to subset dataset to just gene pairs and growth rate
###

library(data.table)
library(tidyr)
#Read in datafile
dat <- fread("../CreativityHub_NNetworks/Costanzo2016_GeneticInteractions.csv")

#Select Gene columns and growth Column
dat <- dat[,c(1,7,12,13)]

#Rename columns for simpler processing downstream
setnames(dat, old = c('Gene Primary DBID','Gene 2 Primary Identifier') ,
         new = c('Gene1','Gene2'))

#Clean data of missing values
dat <- na.omit(dat)

#Check for repeating gene-pairs and how many
dat$gene_pair <- paste(dat$Gene1, dat$Gene2 ,sep = "_")
gene_pair_counts <- table(dat$gene_pair)
dim(gene_pair_counts[gene_pair_counts > 1])

# Aggregate the growth score for each gene pair
aggregated_growth <- aggregate(dat$`Sga Score`, by = list(dat$gene_pair), FUN = sum)

# Calculate the weights based on the p-values
weights <- 1 / dat$Pvalue

# Add a small positive value to all p-values
epsilon <- 1e-10
adjusted_pvalues <- pmax(dat$Pvalue, epsilon)

# Calculate the weighted average Pvalue for each gene pair
weighted_pvalue <- tapply(dat$Pvalue * adjusted_pvalues,
                          dat$gene_pair, sum) / tapply(adjusted_pvalues, 
                                                       dat$gene_pair, sum)

# Create a new condensed data table with gene pairs, aggregated growth scores, and weighted average p-values
condensed_data <- data.table(
  gene_pair = unique(dat$gene_pair),
  aggregated_growth_score = aggregated_growth$x,
  weighted_average_pvalue = weighted_pvalue
)

condensed_data <- separate(condensed_data, gene_pair, 
                           into = c("Gene1", "Gene2"), sep = "_")

#Write to new table
write.csv(condensed_data, file = "GenePair_GrowthRate_v2_Costanzo2016.csv",row.names = F)
