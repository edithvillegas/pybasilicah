library(ggplot2)
library(stringr)
library(tidyverse)
library(reshape2)
library(patchwork)
library(ggpubr)
library(grid)
library(ggthemes)
library(ggjoy)


#-------------------------------------------------------------------------------
# visualize mutational catalog
#-------------------------------------------------------------------------------

Phylogeny <- function(path) {
  #-------------------------------------------------------- Load Data ----------
  hdf <- read.table(path, sep = ",", header = TRUE, stringsAsFactors = TRUE, 
                    check.names=FALSE)
  
  branches <- c()
  for (b in 1:nrow(hdf)) {
    branches[b] <- paste("Branch", b)
  }
  
  vdf <- as.data.frame(t(hdf))  # transpose
  colnames(vdf) <- branches
  
  long_feats <- rownames(vdf)           # save complete mutation features
  rownames(vdf) <- seq.int(nrow(vdf))   # change index to integer numbers
  vdf$ind <- long_feats                 # add new column (complete features)
  
  #-------------------- add compact mutation features as a new column ----------
  short_feats_list <- c("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
  short_feats <- long_feats
  for (feat in short_feats_list) {
    ind <- str_detect(long_feats, feat)
    short_feats[ind] <- feat
  }
  vdf$indx <- short_feats
  
  #---------------------------------- make all branches in one column ----------
  df <- melt(vdf, id.vars=c("ind", "indx"), variable.name = "branch", 
             value.name = "num_mutations")
  
  #------------------------------------------------------------- plot ----------
  plot <- ggplot(data=df, aes(x=ind, y=num_mutations, fill=indx)) + 
    geom_bar(stat="identity", width = 0.5, fill="darkgreen") + 
    facet_wrap(~branch, ncol = 1) + 
    ggtitle("Catalogue Mutations") + 
    xlab("Mutation Features") + 
    ylab("Number of Mutations") + 
    theme_fivethirtyeight() + 
    theme(axis.text.x = element_text(angle=90, vjust=.5, hjust=1))
  
  return(plot)
  }

#-------------------------------------------------------------------------------
# visualize signature profile
#-------------------------------------------------------------------------------

Beta <- function(path) {
  #-------------------------------------------------------- Load Data ----------
  hdf <- read.table(path, sep = ",", header = TRUE, stringsAsFactors = TRUE, 
                    check.names=FALSE)
  rownames(hdf) <- hdf[[1]]
  hdf <- select(hdf, -1)
  vdf <- as.data.frame(t(hdf))          # transpose
  
  sig_names <- colnames(vdf)            # save signature names
  long_feats <- rownames(vdf)           # save complete mutation features
  vdf$ind <- long_feats                 # add new column (complete features)
  rownames(vdf) <- seq.int(nrow(vdf))   # change index to integer numbers
  
  #-------------------- add compact mutation features as a new column ----------
  short_feats_list <- c("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
  short_feats <- long_feats
  for (feat in short_feats_list) {
    ind <- str_detect(long_feats, feat)
    short_feats[ind] <- feat
  }
  vdf$indx <- short_feats
  
  #---------------------------------- make all branches in one column ----------
  df <- melt(vdf, id.vars=c("ind", "indx"), variable.name = "signature_name", 
             value.name = "probability")
  
  #------------------------------------------------------------- plot ----------
  plot <- ggplot(data=df, aes(x=ind, y=probability)) + 
    geom_bar(stat="identity", fill="darkgreen") + 
    facet_wrap(~signature_name, ncol = 1) + 
    ggtitle("Signature Profiles") + 
    xlab("Mutation Features") + 
    ylab("Probability") + 
    theme_fivethirtyeight() + 
    theme(axis.text.x = element_text(angle=90, vjust=.5, hjust=1))
  
  return(plot)
  }

#-------------------------------------------------------------------------------
# visualize alpha over iterations
#-------------------------------------------------------------------------------

alpha_batch <- function(alpha_batch_path, expected_alpha_path, lambda) {
  
  alpha_batch <- read.table(alpha_batch_path, sep = ",", header = FALSE)
  expected_alpha <- read.table(expected_alpha_path, sep = ",", header = FALSE)
  
  a <- as.vector(t(data.matrix(expected_alpha)))
  zero <- which(a==0)
  b <- as.data.frame(matrix(a, ncol=length(a), byrow = T))
  w <- do.call("rbind", replicate(nrow(alpha_batch), b, simplify = FALSE))
  ratio <- alpha_batch / w
  
  ratio[zero] <- (1 - alpha_batch[zero])
  ratio$itr <- 0:(nrow(alpha_batch)-1)
  
  df <- melt(ratio, id.vars=c("itr"), variable.name = "alpha", 
             value.name = "value")
  
  #------------------------------------------------------------- plot ----------
  plot <- ggplot(data=df, aes(x=itr, y=value)) + 
    geom_line(size=0.6) + 
    facet_wrap(~alpha, scales = "free_y", ncol = ncol(expected_alpha)) + 
    ggtitle(paste("Relative Exposure Change, lambda =", lambda)) + 
    xlab("Iterations") + 
    ylab("Ratio (infered / target)") + 
    theme_minimal() + 
    geom_hline(yintercept = 1, color="red")
  
  return(plot)
}

#-------------------------------------------------------------------------------
# visualize likelihood over iterations
#-------------------------------------------------------------------------------

likelihood_iters <- function(path) {
  df <- read.table(path, sep = ",", header = FALSE)
  plot <- ggplot(data=df, aes(x=V1, y=V2)) + 
    geom_line() + 
    #ggtitle("Likelihood changes Over Iterations") + 
    xlab("Iterations") + 
    ylab("Likelihood") + 
    theme_solarized_2()
  
  return(plot)
}

#-------------------------------------------------------------------------------
# visualize likelihood over lambdas
#-------------------------------------------------------------------------------

likelihood_lambdas <- function(path) {
  df <- read.table(path, sep = ",", header = FALSE)
  plot <- ggplot(data=df, aes(x=V1, y=V2)) + 
    ggtitle("Likelihood changes Over Lambdas") + 
    xlab("Lambdas") + 
    ylab("Likelihood") + 
    geom_line() + 
    theme_linedraw()
  
  return(plot)
}

joyplot <- function(path) {
  
  alpha_batch <- read.table(path, sep = ",", header = FALSE)
  alpha_batch$itr <- 0:(nrow(alpha_batch)-1)
  
  df <- melt(alpha_batch, id.vars=c("itr"), variable.name = "alpha", 
             value.name = "value")

  plot <- ggplot(df, aes(x = itr, y = value)) + 
    facet_wrap(~alpha)
    geom_joy() + 
    theme_joy()
  
  return(plot)
}

#-------------------------------------------------------------------------------
# STORAGE
#-------------------------------------------------------------------------------

