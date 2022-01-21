library(ggplot2)
library(stringr)
library(tidyverse)
library(reshape2)
library(patchwork)

# ==============================================================================
# ====================== visualize mutational catalog ==========================
# ==============================================================================

Phylogeny <- function(path) {
  # ====== Load Data
  hdf <- read.table(path, sep = ",", header = TRUE, stringsAsFactors = TRUE, check.names=FALSE)
  vdf <- as.data.frame(t(hdf))  # transpose
  
  long_feats <- rownames(vdf)           # save complete mutation features
  vdf$ind <- long_feats                 # add new column (complete mutation features)
  rownames(vdf) <- seq.int(nrow(vdf))   # change index to integer numbers
  
  # ====== add compact mutation features as a new column
  short_feats_list <- c("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
  short_feats <- long_feats
  for (feat in short_feats_list) {
    ind <- str_detect(long_feats, feat)
    short_feats[ind] <- feat
  }
  vdf$indx <- short_feats
  
  # ====== create a new Data.Frame to make all branches in one column
  df <- melt(vdf, id.vars=c("ind", "indx"), variable.name = "branch", value.name = "num_mutations")
  
  # ====== plot
  ggplot(data=df, aes(x=ind, y=num_mutations)) + 
    geom_bar(stat="identity", fill="steelblue", width = 0.5) + 
    facet_wrap(~branch, ncol = 1) + 
    ggtitle("Catalogue Mutations") + 
    xlab("Mutation Features") + 
    ylab("Number of Mutations") + 
    theme_linedraw() + 
    theme(axis.text.x = element_text(angle=90, vjust=.5, hjust=1))
  }

# ==============================================================================
# ====================== visualize signature profile ===========================
# ==============================================================================

Beta <- function(path) {
  # ====== Load Data
  hdf <- read.table(path, sep = ",", header = TRUE, stringsAsFactors = TRUE, check.names=FALSE)
  rownames(hdf) <- hdf[[1]]
  hdf <- select(hdf, -1)
  vdf <- as.data.frame(t(hdf))          # transpose
  
  sig_names <- colnames(vdf)            # save signature names
  long_feats <- rownames(vdf)           # save complete mutation features
  vdf$ind <- long_feats                 # add new column (complete mutation features)
  rownames(vdf) <- seq.int(nrow(vdf))   # change index to integer numbers
  
  # ====== add compact mutation features as a new column
  short_feats_list <- c("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
  short_feats <- long_feats
  for (feat in short_feats_list) {
    ind <- str_detect(long_feats, feat)
    short_feats[ind] <- feat
  }
  vdf$indx <- short_feats
  
  # ====== create a new Data.Frame to make all branches in one column
  df <- melt(vdf, id.vars=c("ind", "indx"), variable.name = "signature_name", value.name = "probability")
  view(df)
  
  # ====== plot
  ggplot(data=df, aes(x=ind, y=probability)) + 
    geom_bar(stat="identity", fill="steelblue") + 
    facet_wrap(~signature_name, ncol = 1) + 
    ggtitle("Signature Profiles") + 
    xlab("Mutation Features") + 
    ylab("Probability") + 
    theme_linedraw() + 
    theme(axis.text.x = element_text(angle=90, vjust=.5, hjust=1))
  }


# ==============================================================================
# ====================== visualize alpha over iterations =======================
# ==============================================================================

alpha_batch <- function(alpha_batch_path, expected_alpha_path) {
  
  alpha_batch <- read.table(alpha_batch_path, sep = ",", header = FALSE)
  expected_alpha <- read.table(expected_alpha_path, sep = ",", header = FALSE)
  
  a <- as.vector(t(data.matrix(expected_alpha)))
  zero <- which(a==0)
  b <- as.data.frame(matrix(a, ncol=length(a), byrow = T))
  w <- do.call("rbind", replicate(nrow(alpha_batch), b, simplify = FALSE))
  ratio <- alpha_batch / w
  
  ratio[zero] <- (1 - alpha_batch[zero])
  ratio$itr <- 0:(nrow(alpha_batch)-1)
  
  df <- melt(ratio, id.vars=c("itr"), variable.name = "alpha", value.name = "value")
  
  # ====== plot
  ggplot(data=df, aes(x=itr, y=value)) + 
    geom_line() + 
    facet_wrap(~alpha, scales = "free_y") + 
    theme_linedraw() + 
    geom_hline(yintercept = 1, linetype="dashed", color="red")
}

# ==============================================================================
# ============================= STORAGE ========================================
# ==============================================================================

