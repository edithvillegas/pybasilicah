
M_path <- "/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv"
input_catalog_path <- "/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv"
reference_catalog_path <- "/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv"

M <- read.table(M_path, sep = ",", header = TRUE, check.names = FALSE)
input_catalog <- read.table(input_catalog_path, sep = ",", row.names = 1, header = TRUE, check.names = FALSE)
reference_catalog <- read.table(reference_catalog_path, sep = ",", row.names = 1, header = TRUE, check.names = FALSE)
use_condaenv("pybasilica")
setwd("/home/azad/Documents/thesis/SigPhylo/pybasilica/src")
source_python("basilica.py")

x <- fit(M, input_catalog, k=0:5, reference_catalog, lr=0.05, steps_per_iter=500, fixedLimit=0.05, denovoLimit=0.9)
plot_exposure(x)
plot_signatures(x, useRowNames = TRUE, denovoSignature = TRUE)
plot_signatures(x, useRowNames = TRUE, denovoSignature = FALSE)


#============================ TIPS =============================================
# to install ggpubr package :----> sudo apt install libblas-dev liblapack-dev
# to install Rcurl package :-----> sudo apt install libcurl4-openssl-dev 
#===============================================================================

# installed

library(dplyr)      # add_row()
library(stringr)    # str_detect()
library(reshape2)
library(grid)
library(tidyverse)
library(rjson)      # fromJSON()
library(devtools)
library(ggthemes)   # ggplot themes
library(ggpubr)     # ggarrange()
library(patchwork)
library(plotly)     # ggplotly()
library(ggridges)   # geom_density_ridges()
library(ggtree)


#============================== Reticulate =====================================
use_condaenv("SigPhylo")  # setup conda environment
getwd() # mostly starts at "/home/azad"
setwd("/home/azad/Documents/thesis/SigPhylo") # change the directory
source_python("basilica.py")
py_run_file(main)
py_config()
repl_python()
#===============================================================================


M <- readCatalogue("/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv")
input_catalog <- readBeta("/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv")
reference_catalog <- readBeta("/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv")
beta_expected <- readBeta("/home/azad/Documents/thesis/SigPhylo/data/real/expected_beta.csv")


x <- fit(M, input_catalog, k=0:5, reference_catalog, lr=0.05, steps_per_iter=500, fixedLimit=0.05, denovoLimit=0.9)
x <-fit(M, beta_input, k_list, beta_cosmic, 0.05, 500, 0.05, 0.9)
alpha <- x[["Alpha"]]
beta_fixed <- x[["Beta_Fixed"]]
beta_denovo <- x[["Beta_Denovo"]]

plot_alpha(alpha)
plot_beta(beta_fixed, useRowNames = TRUE)
plot_beta(beta_denovo, useRowNames = TRUE)

a <- plot_beta(beta_denovo, useRowNames = TRUE)
b <- plot_beta(B_exp, useRowNames = TRUE)

grid.arrange(a, b)

beta_expected <- readBeta("/home/azad/Documents/thesis/SigPhylo/data/real/expected_beta.csv")
B_exp <- beta_expected["Signature 3(missing from COSMIC)",]

#-------------------------------------------------------------------------------
# STORAGE
#-------------------------------------------------------------------------------

simRun <- function(Tprofle, Iprofile, cos_path_org, fixedLimit, denovoLimit, startSeed, iterNum) {
  results <- tibble(
    M = list(),
    A_target = list(),
    B_fixed_target = list(),
    B_denovo_target = list(),
    B_input = list(),
    A_inf = list(),
    B_fixed_inf = list(),
    B_denovo_inf = list(),
    GoodnessofFit = numeric(),
    Accuracy = numeric(),
    Quantity = logical(),
    Quality = numeric(),
    TP = character(),
    IP = character(),
    Iter = numeric(),
    Seed = numeric()
  )
  
  counter <- 1
  
  for (i in Tprofle) {
    for (j in Iprofile) {
      for (k in 1:iterNum) {
        #print(paste("Run:", counter, "Started!"))
        seed <- startSeed + counter
        counter <- counter + 1
        
        output <- run_simulated(i,
                                j,
                                cos_path_org,
                                fixedLimit,
                                denovoLimit,
                                seed)
        
        if (class(output)!="list") {
          next
        }
        print(paste("Run:", (counter-1), "is Done!"))
        
        m <- py_to_r(output[["M"]])                              # data.frame
        a_target <- py_to_r(output[["A_target"]])                # data.frame
        b_fixed_target <- py_to_r(output[["B_fixed_target"]])    # data.frame
        b_denovo_target <- py_to_r(output[["B_denovo_target"]])  # data.frame
        b_input <- py_to_r(output[["B_input"]])                  # data.frame
        a_inf <- py_to_r(output[["A_inf"]])                      # data.frame
        b_fixed_inf <- py_to_r(output[["B_fixed_inf"]])          # data.frame
        b_denovo_inf <- py_to_r(output[["B_denovo_inf"]])        # data.frame
        
        goodnessofFit <- output[["GoodnessofFit"]]  # numeric
        accuracy <- output[["Accuracy"]]            # numeric
        quantity <- output[["Quantity"]]            # logical
        quality <- output[["Quality"]]              # numeric
        
        results <- add_row(results,
                           M = list(m),
                           A_target = list(a_target),
                           B_fixed_target = list(b_fixed_target),
                           B_denovo_target = list(b_denovo_target),
                           B_input = list(b_input),
                           A_inf = list(a_inf),
                           B_fixed_inf = list(b_fixed_inf),
                           B_denovo_inf = list(b_denovo_inf),
                           GoodnessofFit = goodnessofFit,
                           Accuracy = accuracy,
                           Quantity = quantity,
                           Quality = quality,
                           TP = i,
                           IP = j,
                           Iter = k,
                           Seed = seed
        )
      }
    }
  }
  return(results)
}




#-------------------------------------------------------------------------------
# Load input Data | JSON ---> tibble (OK)
#-------------------------------------------------------------------------------
load_input <- function(json_path) {
  
  df <- fromJSON(file=json_path)  # list
  x <- df[["input"]]
  
  M <- x[["M"]] # list
  beta_fixed <- x[["beta_fixed"]] # list
  A <- x[["A"]] # list
  beta_fixed_names <- x[["beta_fixed_names"]] # character vector
  alpha_expected <- x[["alpha_expected"]] # list if from simulation
  alpha_expected_names <- x[["alpha_expected_names"]] # character vector if from simulation
  beta_expected <- x[["beta_expected"]] # list
  beta_expected_names <- x[["beta_expected_names"]] # character vector
  mutation_features <- x[["mutation_features"]] # character vector
  
  
  #-------------- phylogeny ------------------------------------------------------
  M_df <- as.data.frame(do.call(rbind, M))
  M_df_n <-nrow(M_df)
  rownames(M_df) <- paste(c("Branch"), 1:M_df_n, sep = "")
  colnames(M_df) <- mutation_features
  M_list <- list(M_df)
  
  #-------------- A --------------------------------------------------------------
  A_df <- as.data.frame(do.call(rbind, A))
  A_df_n <-nrow(A_df)
  rownames(A_df) <- paste(c("Branch"), 1:A_df_n, sep = "")
  colnames(A_df) <- paste(c("Branch"), 1:A_df_n, sep = "")
  A_list <- list(A_df)
  
  #-------------- alpha expected -------------------------------------------------
  
  if (is.character(alpha_expected)) {
    alpha_expected_list <- "NA"
  } else {
    alpha_expected_df <- as.data.frame(do.call(rbind, alpha_expected))
    alpha_expected_n <-nrow(alpha_expected_df)
    alpha_expected_k <- ncol(alpha_expected_df)
    rownames(alpha_expected_df) <- paste(c("Branch"), 1:alpha_expected_n, sep = "")
    colnames(alpha_expected_df) <- alpha_expected_names
    alpha_expected_list <- list(alpha_expected_df)
  }
  
  #-------------- beta fixed -----------------------------------------------------
  beta_fixed_df <- as.data.frame(do.call(rbind, beta_fixed))
  rownames(beta_fixed_df) <- beta_fixed_names
  colnames(beta_fixed_df) <- mutation_features
  beta_fixed_list <- list(beta_fixed_df)
  
  #-------------- beta expected --------------------------------------------------
  beta_expected_df <- as.data.frame(do.call(rbind, beta_expected))
  rownames(beta_expected_df) <- beta_expected_names
  colnames(beta_expected_df) <- mutation_features
  beta_expected_list <- list(beta_expected_df)
  
  
  # create tibble
  data_tibble <- tibble(
    M = M_list,
    A = A_list,
    Alpha_Expected = alpha_expected_list,
    Beta_Fixed = beta_fixed_list,
    Beta_Expected = beta_expected_list
  )
  
  return(data_tibble)
}

#-------------------------------------------------------------------------------
# Load output Data | JSON ---> tibble (OK)
#-------------------------------------------------------------------------------
load_output <- function(json_path) {
  # Load from json file
  data <- fromJSON(file=json_path)  # list
  
  # initialize empty tibble
  data_tibble <- tibble(
    # numeric data ------------
    K_Denovo = numeric(),
    Lambda = numeric(),
    Log_Like = numeric(),
    BIC = numeric(),
    #list data-----------------
    Log_Likes = list(),
    BICs = list(),
    Cosine = list(),
    Alphas = list(),
    Alpha = list(),
    Beta = list(),
    M_R = list()
  )
  
  # mutation feature tags
  features <- data[["input"]][["mutation_features"]] # character
  
  # construct tibble data-set
  # each iteration add a row with specific k and lambda
  for (row in data[["output"]]) {
    # row -> list datatype
    k_denovo <- row[["k_denovo"]]   # numeric
    lambda <- row[["lambda"]]       # numeric
    log_like <- row[["log-like"]]   # numeric
    bic <- row[["BIC"]]             # numeric
    
    log_likes <- list(row[["log-likes"]]) # numeric -> list
    bics <- list(row[["BICs"]])           # numeric -> list
    cosine <- list(row[["cosine"]])       # numeric -> list
    
    alphas <- row[["alphas"]] # list
    alpha <- row[["alpha"]]   # list
    beta <- row[["beta"]]     # list
    m_r <- row[["M_R"]]       # list
    
    #-------------- alphas -----------------------------------------------------
    df <- data.frame()
    iter_num <- 1
    for (alpha in alphas) {
      alphas_df <- as.data.frame(do.call(rbind, alpha))
      alphas_n <-nrow(alphas_df)
      alphas_k <- ncol(alphas_df)
      
      colnames(alphas_df) <- paste(c("Signature"), 1:alphas_k, sep = "")
      alphas_df$Branch <- paste(c("Branch"), 1:alphas_n, sep = "")
      alphas_df$IterNum <- rep(iter_num, alphas_n)
      iter_num <- iter_num + 1
      
      df <- rbind(df, alphas_df)
    }
    alphas_list <- list(df)
    
    #-------------- alpha ------------------------------------------------------
    alpha_df <- as.data.frame(do.call(rbind, alpha))
    alpha_n <-nrow(alpha_df)
    alpha_k <- ncol(alpha_df)
    rownames(alpha_df) <- paste(c("Branch"), 1:alpha_n, sep = "")
    colnames(alpha_df) <- paste(c("Signature"), 1:alpha_k, sep = "")
    alpha_list <- list(alpha_df)
    
    #-------------- beta -------------------------------------------------------
    beta_df <- as.data.frame(do.call(rbind, beta))
    beta_n <-nrow(beta_df)
    beta_k <- ncol(beta_df)
    rownames(beta_df) <- paste(c("Signature"), 1:beta_n, sep = "")
    colnames(beta_df) <- features
    beta_list <- list(beta_df)
    
    #-------------- phylogeny reconstructed ------------------------------------
    m_r_df <- as.data.frame(do.call(rbind, m_r))
    m_r_n <-nrow(m_r_df)
    m_r_k <- ncol(m_r_df)
    rownames(m_r_df) <- paste(c("Branch"), 1:m_r_n, sep = "")
    colnames(m_r_df) <- features
    m_r_list <- list(m_r_df)
    
    # add each run to a row in tibble
    data_tibble <- add_row(data_tibble,
                           K_Denovo = k_denovo,
                           Lambda = lambda,
                           Log_Like = log_like,
                           BIC = bic,
                           Log_Likes = log_likes,
                           BICs = bics,
                           Cosine = cosine,
                           Alphas = alphas_list,
                           Alpha = alpha_list,
                           Beta = beta_list,
                           M_R = m_r_list
    )
  }
  return(data_tibble)
}


#-------------------------------------------------------------------------------
# cosine similarity between two vectors (OK)
#-------------------------------------------------------------------------------
cosine_sim <- function(a, b) {
  numerator <- sum(a * b)
  denominator <- sqrt(sum(a^2)) * sqrt(sum(b^2))
  return(numerator / denominator)
}

#-------------------------------------------------------------------------------
# label inferred beta using expected beta labels (OK)
# input : expected beta, inferred beta ---> inferred beta with correct labels
#-------------------------------------------------------------------------------
beta_labeling <- function(exp, inf) {
  beta_exp <- as.data.frame(t(exp))
  beta_inf <- as.data.frame(t(inf))
  
  # check if they have same dimension
  ncol1 <- ncol(beta_exp)
  ncol2 <- ncol(beta_inf)  # integer vector
  if (ncol1!=ncol2) {
    return("False input!")
  }
  
  m <- matrix(nrow = ncol1, ncol = ncol2)
  for (i in 1:ncol1) {
    col1 <- beta_exp[, i] # numeric vector
    for (j in 1:ncol2) {
      col2 <- beta_inf[, j] # numeric vector
      m[i,j] <- cosine_sim(col1, col2)
    }
  }
  
  new_labels <- rep("NA", ncol2)
  for (i in 1:ncol1) {
    label <- colnames(beta_exp)[i]
    index <- which.max(m[i,])
    new_labels[index] <- label
  }
  rownames(inf) <- new_labels
  return(inf)
}

#-------------------------------------------------------------------------------
# label inferred alpha using expected alpha labels (OK)
# input : expected alpha, inferred alpha ---> inferred alpha with correct labels
#-------------------------------------------------------------------------------
alpha_labeling <- function(exp, inf) {
  # check if they have same dimension
  ncol1 <- as.integer(ncol(exp))
  ncol2 <- as.integer(ncol(inf))  # integer vector
  if (ncol1!=ncol2) {
    return(FALSE)
  }
  
  m <- matrix(nrow = ncol1, ncol = ncol2)
  for (i in 1:ncol1) {
    col1 <- exp[, i] # numeric vector
    for (j in 1:ncol2) {
      col2 <- inf[, j] # numeric vector
      m[i,j] <- cosine_sim(col1, col2)
    }
  }
  
  new_labels <- rep("NA", ncol2)
  for (i in 1:ncol1) {
    label <- colnames(exp)[i]
    index <- which.max(m[i,])
    new_labels[index] <- label
  }
  colnames(inf) <- new_labels
  return(inf)
}


#===============================================================================
# Convert RDs file to mutational catalog
#===============================================================================
#-------------------------------------------------------------------------------
# if get error try these two commands
# sudo apt-get install libbz2-dev
options(timeout = 10000)
BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")
#-------------------------------------------------------------------------------

library("BSgenome.Hsapiens.UCSC.hg19")

counts <- readRDS("/home/azad/Documents/thesis/rds/basilico/basilico_cll_input.rds")
head(counts)

counts  <- counts %>%
  dplyr::mutate(
    CHROMOSOME = chr,
    START = from,
    END = to,
    REFERENCE = ref,
    VARIANT = alt,
    SAMPLE = patient    # sample --> patient
  ) %>% dplyr::filter(!is.na(REFERENCE), !is.na(VARIANT)) %>% dplyr::select(CHROMOSOME, START, END, REFERENCE, VARIANT, SAMPLE)


counts = counts  [c(6, 1, 2, 3, 4, 5)]
counts$CHROMOSOME = gsub(counts$CHROMOSOME,
                         pattern = "chr",
                         replacement = "")
counts$END = counts$START

trinucleotides_counts = get.SBS.counts(data=unique(counts),reference=BSgenome.Hsapiens.UCSC.hg19)

raw_data = rbind(raw_data, trinucleotides_counts)



"get.SBS.counts" <- function( data, reference = NULL ) {
  
  # check that reference is a BSgenome object
  if(is.null(reference)|class(reference)!="BSgenome") {
    stop("The reference genome provided as input needs to be a BSgenome object.")
  }
  
  # preprocessing input data
  data <- as.data.frame(data)
  colnames(data) <- c("sample","chrom","start","end","ref","alt")
  
  # consider only single nucleotide variants involving (A,C,G,T) bases
  data <- data[which(data[,"start"]==data[,"end"]),,drop=FALSE]
  data <- data[which(as.matrix(data[,"ref"])%in%c("A","C","G","T")),,drop=FALSE]
  data <- data[which(as.matrix(data[,"alt"])%in%c("A","C","G","T")),,drop=FALSE]
  data <- data[,c("sample","chrom","start","ref","alt"),drop=FALSE]
  colnames(data) <- c("sample","chrom","pos","ref","alt")
  data <- unique(data)
  data <- data[order(data[,"sample"],data[,"chrom"],data[,"pos"]),,drop=FALSE]
  
  # convert data to GRanges
  data <- GRanges(as.character(data$chrom),IRanges(start=(as.numeric(data$pos)-1),width=3),
                  ref=DNAStringSet(as.character(data$ref)),alt=DNAStringSet(as.character(data$alt)),sample=as.character(data$sample))
  
  # check that all chromosomes match reference
  if(length(setdiff(seqnames(data),GenomeInfoDb::seqnames(reference)))>0) {
    warning("Check chromosome names, not all match reference genome.")
  }
  
  # find context for each mutation
  data$context <- getSeq(reference,data)
  
  # check for any mismatch with BSgenome context
  if(any(subseq(data$context,2,2)!=data$ref)) {
    warning("Check reference bases, not all match context.")
  }
  
  # get complements and reverse complements
  data$cref <- complement(data$ref)
  data$calt <- complement(data$alt)
  data$rccontext <- reverseComplement(data$context)
  
  # identify trinucleotides motif
  data$cat <- ifelse(data$ref%in%c("C","T"),paste0(subseq(data$context,1,1),"[",data$ref,">",data$alt,"]",subseq(data$context,3,3)),
                     paste0(subseq(data$rccontext,1,1),"[",data$cref,">",data$calt,"]",subseq(data$rccontext,3,3)))
  
  # create 96 trinucleotides mutation categories
  categories_context <- NULL
  categories_alt <- rep(c(rep("C>A",4),rep("C>G",4),rep("C>T",4),rep("T>A",4),rep("T>C",4),rep("T>G",4)),4)
  categories_cat <- NULL
  cont <- 0
  for(i in c("A","C","G","T")) {
    for(j in 1:6) {
      for(k in c("A","C","G","T")) {
        cont <- cont + 1
        categories_context <- c(categories_context,paste0(k,":",i))
        categories_cat <- c(categories_cat,paste0(k,"[",categories_alt[cont],"]",i))
      }
    }
  }
  mutation_categories <- data.table(context=categories_context,alt=categories_alt,cat=categories_cat)
  
  # count number of mutations per sample for each category
  data <- merge(mutation_categories[,.(cat)],data.table(sample=data$sample,cat=data$cat)[,.N,by=.(sample,cat)],by="cat",all=TRUE)
  data <- dcast(data,sample~cat,value.var="N")
  data <- data[!is.na(sample),drop=FALSE]
  data[is.na(data)] <- 0
  
  # make trinucleotides counts matrix
  samples_names <- data$sample
  data <- as.matrix(data[,2:ncol(data),drop=FALSE])
  rownames(data) <- samples_names
  data <- data[sort(rownames(data)),,drop=FALSE]
  data <- data[,sort(colnames(data)),drop=FALSE]
  trinucleotides_counts <- array(0,c(nrow(data),96))
  rownames(trinucleotides_counts) <- rownames(data)
  colnames(trinucleotides_counts) <- sort(as.character(mutation_categories$cat))
  rows_contexts <- rownames(data)
  cols_contexts <- colnames(trinucleotides_counts)[which(colnames(trinucleotides_counts)%in%colnames(data))]
  trinucleotides_counts[rows_contexts,cols_contexts] <- data[rows_contexts,cols_contexts]
  
  # return trinucleotides counts matrix
  return(trinucleotides_counts)
  
}





