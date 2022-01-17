library(ggplot2)
library(stringr)
library(tidyverse)

# ====== visualize mutational catalog ==========================================

Phylogeny <- function(path) {
  df <- read.table(path, sep = "," ,header = TRUE, stringsAsFactors = TRUE, 
                  check.names=FALSE)
  df <- as.data.frame(t(df))
  
  features <- rownames(df)
  compact_features <- c("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
  for (f in compact_features) {
    ind <- str_detect(features, f)
    features[ind] <- f
  }
  df$features <- features
  
  n <- ncol(df)-1
  features <- df[,"features"]
  ind <- rep(c(1),each=96)
  xx <- data.frame(features, V=df[,1], ind)
  
  for (i in 2:n) {
    ind <- rep(c(i),each=96)
    x <- data.frame(features, V=df[,i], ind)
    xx <- rbind(xx, x)
  }
  colnames(xx) <- c("features", "num_mutations", "branch_no")
  
  ggplot(data=xx, aes(x=features, y=num_mutations)) + 
    geom_bar(stat="identity", fill="steelblue", width = 0.5) + 
    facet_wrap(~branch_no, ncol = 3) + 
    ggtitle("Catalogue Mutations") + 
    xlab("Mutation Features") + 
    ylab("Number of Mutations") + 
    theme_linedraw()
}

# ==============================================================================


# ====== visualize signature profile ===========================================

Beta <- function(path) {
  df <- read.table(path, sep = "," 
                  ,header = TRUE, 
                  stringsAsFactors = FALSE, 
                  check.names=FALSE)
  rownames(df) <- df[[1]]
  df <- select(df, -1)
  df <- as.data.frame(t(df))
  sig_names <- colnames(df)
  
  features <- rownames(df)
  compact_features <- c("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
  for (f in compact_features) {
    ind <- str_detect(features, f)
    features[ind] <- f
  }
  df$features <- features
  
  n <- ncol(df)-1
  features <- df[,"features"]
  ind <- rep(c(sig_names[1]),each=96)
  xx <- data.frame(features, V=df[,1], ind)
  for (i in 2:n) {
    ind <- rep(c(sig_names[i]),each=96)
    x <- data.frame(features, V=df[,i], ind)
    xx <- rbind(xx, x)
  }
  colnames(xx) <- c("features", "probability", "sig_names")
  
  ggplot(data=xx, aes(x=features, y=probability)) + 
    geom_bar(stat="identity", fill="steelblue") + 
    facet_wrap(~sig_names) + 
    ggtitle("Signature Profiles") + 
    xlab("Mutation Features") + 
    ylab("Probability") + 
    theme_linedraw()
}

# ==================================================================









