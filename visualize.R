library(ggplot2)
library(stringr)


# visualize mutational catalog
Phylogeny <- function(path) {

  df <- read.csv(path, header = TRUE, stringsAsFactors = TRUE)
  View(df)
  
  features_list <- as.character(df$X)
  target_list <- c("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
  for (f in target_list) {
    ind <- str_detect(features_list, f)
    features_list[ind] <- f
  }
  df$feats <- features_list
  
  ggplot(data=df, aes(x=feats, y=X1)) + 
    geom_bar(stat="identity", fill="steelblue", width = 0.5) + 
    #geom_text(aes(label=X1), vjust=-0.3, size=2.5) + 
    ggtitle("Catalogue Mutations") + 
    xlab("Mutation Features") + 
    ylab("Number of Mutations") +
    theme(axis.text.x=element_text(angle=0,hjust=1,vjust=0.5))
}

# ======================================================================

M_path <- "/home/azad/Documents/thesis/SigPhylo/data/data4R/M4R.csv"
Phylogeny(M_path)

df <- read.csv(M_path, header = TRUE, stringsAsFactors = TRUE)
View(df)

# ======================================================================

# visualize signature profile
beta <- function(path) {

  df <- read.csv(path, header = TRUE)
  
  features_list <- as.character(df$X)
  target_list <- c("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
  for (f in target_list) {
    ind <- str_detect(features_list, f)
    features_list[ind] <- f
  }
  df$feats <- features_list
  
  for (p in colnames(df)[2:(length(colnames(df))-1)]) {
    print(p)
  }
  
  ggplot(data=df, aes(x=feats, y=SBS1, fill=X)) + 
    geom_bar(stat="identity", fill="steelblue") + 
    ggtitle("Signature Profile") + 
    xlab("Mutation Features") + 
    ylab("Probability") + 
    theme(axis.text.x=element_text(angle=0,hjust=1,vjust=0.5))
}

# ==================================================================

beta_path <- "/home/azad/Documents/thesis/SigPhylo/data/data4R/beta4R.csv"
beta(beta_path)

df <- read.csv(beta_path, header = TRUE)
View(df)

for (p in colnames(df)[2:length(colnames(df))]) {
  print(p)
}

# ==================================================================








