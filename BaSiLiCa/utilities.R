library(stringr)    # str_detect()
library(ggplot2)    # ggplot()
library(ggthemes)   # ggplot themes
library(dplyr)      # add_row()
library(rjson)      # fromJSON()
library(plotly)     # ggplotly()
library(tidyr)      # tibble() & gather()
library(data.table) # data.table()
library(ggpubr)     # ggarrange()
library(ggridges)   # geom_density_ridges()
library(gridExtra)  # grid.arrange
library(reticulate)



#-------------------------------------------------------------------------------
# plot alpha
#-------------------------------------------------------------------------------
plot_alpha <- function(alpha, title) {
  # alpha: dataframe
  alpha$Branch <- c(1:nrow(alpha))
  alpha_long <- gather(alpha,
                       key="Signature",
                       value="Exposure",
                       c(-Branch)
  )

  ggplot(data = alpha_long, aes(x=Branch, y=Exposure, fill=Signature)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    ggtitle(paste(title)) +
    scale_y_continuous(labels=scales::percent)
}

#-------------------------------------------------------------------------------
# plot alpha error
#-------------------------------------------------------------------------------
plot_alpha_error <- function(alpha_error, title) {
  # alpha: dataframe
  alpha_error$Branch <- c(1:nrow(alpha_error))
  alpha_error_long <- gather(alpha_error,
                       key="Signature",
                       value="Error",
                       c(-Branch)
  )

  ggplot(data = alpha_error_long, aes(x=Branch, y=Error, fill=Signature)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    ggtitle(paste(title))
    #scale_y_continuous(labels=scales::percent)
}


#-------------------------------------------------------------------------------
# plot beta
#-------------------------------------------------------------------------------
plot_beta <- function(beta, title) {

  x <- as_tibble(cbind(cat = colnames(beta), t(beta))) # tibble

  # convert signature columns datatype (chr -> dbl)
  for (i in 2:ncol(x)) {
    x[[i]] <- as.numeric(x[[i]])
  }

  short_feats_list <- c("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
  long_feats <- x$cat
  short_feats <- long_feats
  for (feat in short_feats_list) {
    ind <- str_detect(short_feats, feat)
    short_feats[ind] <- feat
  }

  x$alt <- short_feats

  x <- as.data.table(gather(x, key = "signature", value = "value", c(-cat, -alt)))
  x[, `:=`(Context, paste0(substr(cat, 1, 1), ".", substr(cat, 7, 7)))]
  x[, `:=`(alt, paste0(substr(cat, 3, 3), ">", substr(cat, 5, 5)))]

  glist <- list()
  for (i in 1:nrow(beta)) {
    plt <- ggplot(x[signature == rownames(beta)[i]]) +
      geom_bar(aes(x = Context, y = value, fill = alt), stat = "identity", position = "identity") +
      facet_wrap(~alt, nrow = 1, scales = "free_x") +
      theme(#axis.text.x = element_text(angle = 90, hjust = 1),
        panel.background = element_blank(),
        #axis.line = element_line(colour = "black"),
        axis.line = element_blank(), #added
        axis.ticks.x = element_blank(), # added
        axis.text.x = element_blank()) + # added
      ggtitle(rownames(beta)[i]) +
      theme(legend.position = "none") +
      ylab("Frequency")
    # + CNAqc:::my_ggplot_theme()
    glist[[i]] <- plt
  }

  plot <- ggarrange(plotlist = glist,
                    ncol = 1
                    #nrow = nrow,
                    #common.legend = TRUE,
                    #legend = "bottom"
  )
  annotate_figure(plot,
                  top = text_grob("Visualizing Tooth Growth", color = "red", face = "bold", size = 14)
  )

  # JUST FOR TEST --------------------------------------------------------------
  #plot <- do.call("grid.arrange", c(glist, ncol = 1))
  # JUST FOR TEST --------------------------------------------------------------

  return(plot)
}


#===============================================================================
#===============================================================================
#===============================================================================



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


#=========================== READ ALPHA & BETA =================================
#-------------------------------------------------------------------------------
# read signature profiles from CSV file and return data.frame (OK)
# row-names : signature names
# col-names : mutation features
#-------------------------------------------------------------------------------
beta_read_csv <- function(path) {
  beta <- read.table(path, sep = ",", row.names = 1, header = TRUE, check.names = FALSE)
  return(beta)
}

#-------------------------------------------------------------------------------
# read signature profiles from main file (tibble) and return data.frame (OK)
# row-names : signature names
# col-names : mutation features
#-------------------------------------------------------------------------------
beta_read_tibble <- function(data, k, lambda) {
  df <- filter(data, K_Denovo==k & Lambda==lambda)  # tibble
  beta <- df[["Beta"]][[1]] # data.frame
  return(beta)
}

#============================== VISUALIZATION ==================================

#-------------------------------------------------------------------------------
# heat-map (OK)
# BIC across two axis variables (k_denovo vs lambda)
#-------------------------------------------------------------------------------
heatmap <- function(data) {
  df <- select(data, K_Denovo, BIC, Lambda) # tibble
  df <- df %>%
    mutate(text = paste0("K: ", K_Denovo, "\n",
                         "Lambda: ", Lambda, "\n",
                         "BIC: ", BIC))
  p <- ggplot(data = df, aes(x=K_Denovo, y=Lambda, fill=BIC, text=text)) +
    geom_tile()
  ggplotly(p, tooltip="text")
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


#-------------------------------------------------------------------------------
# plot phylogeny
#-------------------------------------------------------------------------------
library(ggplot2)
library(ggtree)
library(dplyr)

# ggtree BRMA object and plot
brma = ggtree::read.tree("/home/azad/Documents/thesis/simple.nwk")

brma_plot = ggtree(brma, ladderize=F) +
  #geom_tiplab() +
  ggtitle("BRMA (sample tree)") +
  theme_tree() +
  #geom_nodepoint(aes(color=isTip), color = 'purple4', size=3) +
  geom_tippoint()
#geom_cladelabel('B2d', 'jj') +
#geom_tiplab() +
#geom_text(aes(label=label), size=3, color="purple", hjust=-.3) +
#geom_treescale(fontsize=3, linesize=1, offset=.2, x = 0) +
#coord_cartesian(clip = 'off') +
#xlim(0, 9000)

brma_plot + geom_text(aes(label=node))



#===============================================================================
# Convert RDs file to mutational catalogue
#===============================================================================

#-------------------------------------------------------------------------------
# if get error try these two commands
# sudo apt-get install libbz2-dev
# options(timeout = 10000)
BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")
#-------------------------------------------------------------------------------

library(dplyr)
library(data.table)
library("BSgenome.Hsapiens.UCSC.hg19")

#-------------------------------------------------------------------------------

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


#library('BSgenome.Hsapiens.NCBI.GRCh38')

counts = counts  [c(6, 1, 2, 3, 4, 5)]
counts$CHROMOSOME = gsub(counts$CHROMOSOME,
                         pattern = "chr",
                         replacement = "")
counts$END = counts$START

trinucleotides_counts = get.SBS.counts(data=unique(counts),reference=BSgenome.Hsapiens.UCSC.hg19)

raw_data = rbind(raw_data,trinucleotides_counts)


#-------------------------------------------------------------------------------

data=unique(counts)     # added later
reference=BSgenome.Hsapiens.UCSC.hg19  # added later

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



'
#-------------------------------------------------------------------------------
# plot alpha for given k grouped by lambda (OK)
#-------------------------------------------------------------------------------
plot_alpha <- function(data, input, k) {
  sig_share <- data %>% filter(K_Denovo==k) %>% select(Alpha, Lambda) # tibble
  exposure <- tibble()
  for (i in 1:nrow(sig_share)) {
    row <- sig_share[i,]  # tibble
    alpha <- row[["Alpha"]][[1]]  # data.frame

    #df <- df[, order(colnames(df))]  #added
    exp <- input[["Alpha_Expected"]][[1]] #added
    if (exp != "NA") {
      alpha <- alpha_labeling(exp, alpha)    #added
    }

    lambda <- row[["Lambda"]]     # numeric
    lambda_list <- rep(lambda, nrow(alpha)) # numeric (vector)
    df <- tibble(cbind(alpha, Branch=rownames(alpha), Lambda=lambda_list))  # tibble

    exposure <- exposure %>% rbind(df)  # tibble
  }

  # tibble
  exposure_long <- gather(exposure,
                          key="Signature",
                          value="Exposure",
                          c(-Branch, -Lambda))

  ggplot(data = exposure_long, aes(x=Branch, y=Exposure, fill=Signature)) +
    geom_bar(stat = "identity") +
    facet_wrap(~Lambda, labeller = label_both) +
    theme_minimal() +
    ggtitle(paste("No. of Inferred Signature:", k)) +
    scale_y_continuous(labels=scales::percent)
}


#-------------------------------------------------------------------------------
# return best run info (highest log-likelihood) (OK)
#-------------------------------------------------------------------------------
best_loglike <- function(data) {
  index <- which.max(data[["Log_Like"]])
  k <- data[index, ][["K_Denovo"]]
  lambda <- data[index, ][["Lambda"]]
  return(list("k"=k,
              "lambda"=lambda))
}

#-------------------------------------------------------------------------------
# return best run info (lowest BIC) (OK)
#-------------------------------------------------------------------------------
best_bic <- function(data) {
  index <- which.min(data[["BIC"]])
  k <- data[index, ][["K_Denovo"]]
  lambda <- data[index, ][["Lambda"]]
  return(list("k"=k,
              "lambda"=lambda))
}


#-------------------------------------------------------------------------------
# plot no. of reconstructed branches (OK)
# with cosine similarity higher than threshold for all k and lambdas
#-------------------------------------------------------------------------------
plot_cosine <- function(data, threshold) {
  cos <- tibble(K_Denovo=numeric(), Lambda=numeric(), Ratio=numeric())
  for (i in 1:nrow(data)) {
    row <- data[i, ]  # tibble
    k <- row[["K_Denovo"]]  # numeric
    lambda <- row[["Lambda"]] # numeric
    c <- row[["Cosine"]][[1]] # numeric (vector)
    r <- length(c[c > threshold]) / length(c)
    cos <- add_row(cos, K_Denovo=k, Lambda=lambda, Ratio=r)
  }

  ggplot(data = cos, aes(x=K_Denovo, y=Ratio)) +
    geom_bar(stat = "identity") +
    facet_wrap(~Lambda, labeller = label_both)
}


#-------------------------------------------------------------------------------
# plot priors (OK)
#-------------------------------------------------------------------------------
plot_priors <- function(data, k_denovo, lambda) {
  alphas <- filter(data, K_Denovo==k_denovo, Lambda==lambda)[["Alphas"]][[1]]  # data.frame

  long <- gather(alphas, key="Signature", value="Probability", 1:(ncol(alphas) - 2))
  long <- long[c("Branch", "Signature", "IterNum", "Probability")]  # re-order columns

  branches <- unique(long[["Branch"]])
  signatures <- unique(long[["Signature"]])
  iterations <- unique(long[["IterNum"]])

  if (length(iterations) >= 5) {
    iterations <- as.integer(seq(1, length(iterations), length.out = 3))
  }

  c <- data.frame()
  for (branch in branches) {
    for (sig in signatures) {
      for (iter in iterations) {
        a <- filter(long, Branch==branch, Signature==sig, IterNum==iter)  # data.frame
        b <- cbind(a, Samples=rnorm(1000, a[["Probability"]], 1))
        c <- rbind(c, b)
      }
    }
  }

  c$IterNum <- paste(c$IterNum)
  ggplot(c, aes(x = Samples, y = IterNum)) +
    facet_wrap(~ Branch + Signature, ncol = (ncol(alphas) - 2)) +
    geom_density_ridges() +
    theme_ridges()
}

#-------------------------------------------------------------------------------
# plot k_denovo vs. log-likelihood grouped by lambda (OK)
#-------------------------------------------------------------------------------
plot_k_loglike <- function(data) {
  ggplot(data = data, aes(x=K_Denovo, y=Log_Like)) +
    geom_line() +
    facet_wrap(~Lambda, labeller = label_both) +
    #theme_fivethirtyeight() +
    xlab("No. of Inferred Signatures") +
    ylab("Log-Likelihood") +
    ggtitle("Model Performance over Lambdas")
}

#-------------------------------------------------------------------------------
# plot k_denovo vs. BIC grouped by lambda (OK)
#-------------------------------------------------------------------------------
plot_k_bic <- function(data) {
  ggplot(data = data, aes(x=K_Denovo, y=BIC)) +
    geom_line() +
    facet_wrap(~Lambda, labeller = label_both) +
    #theme_fivethirtyeight() +
    xlab("No. of Inferred Signatures") +
    ylab("BIC") +
    ggtitle("Model Performance over Lambdas")
}

#-------------------------------------------------------------------------------
# plot lambda vs. log-likelihood grouped by k (OK)
#-------------------------------------------------------------------------------
plot_lambda_loglike <- function(data) {
  ggplot(data = data, aes(x=Lambda, y=Log_Like)) +
    geom_line() +
    facet_wrap(~K_Denovo, labeller = label_both) +
    #theme_fivethirtyeight() +
    xlab("Lambda") +
    ylab("Log-Likelihood") +
    ggtitle("Model Performance over K")
}

#-------------------------------------------------------------------------------
# plot lambda vs. BIC grouped by k (OK)
#-------------------------------------------------------------------------------
plot_lambda_bic <- function(data) {
  ggplot(data = data, aes(x=Lambda, y=BIC)) +
    geom_line() +
    facet_wrap(~K_Denovo, labeller = label_both, scales = "free_y") +
    #theme_fivethirtyeight() +
    xlab("Lambda") +
    ylab("BIC") +
    ggtitle("Model Performance over K")
}

#-------------------------------------------------------------------------------
# read alpha from main file (tibble) and return data.frame (OK)
# row-names : branches
# col-names : signature names
#-------------------------------------------------------------------------------
alpha_read_tibble <- function(data, k, lambda) {
  df <- filter(data, K_Denovo==k & Lambda==lambda)  # tibble
  alpha <- df[["Alpha"]][[1]] # data.frame
  return(alpha)
}


'





