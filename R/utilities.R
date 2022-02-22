library(stringr)
library(ggplot2)
library(ggthemes)
library(dplyr)
library(rjson)
library(plotly)
library(tidyr)
library(data.table)
library(ggpubr)


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
    
    alpha <- row[["alpha"]]   # list
    beta <- row[["beta"]]     # list
    m_r <- row[["M_R"]]       # list
    
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
                           Alpha = alpha_list, 
                           Beta = beta_list, 
                           M_R = m_r_list
                           )
  }
  return(data_tibble)
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
# read signature profiles from tibble file and return data.frame (OK)
# row-names : signature names
# col-names : mutation features
#-------------------------------------------------------------------------------
beta_read_tibble <- function(data, k, lambda) {
  df <- filter(data, K_Denovo==k & Lambda==lambda)  # tibble
  beta <- df[["Beta"]][[1]] # data.frame
  return(beta)
}

#-------------------------------------------------------------------------------
# read alpha from tibble file and return data.frame (OK)
# row-names : branches
# col-names : signature names
#-------------------------------------------------------------------------------
alpha_read_tibble <- function(data, k, lambda) {
  df <- filter(data, K_Denovo==k & Lambda==lambda)  # tibble
  alpha <- df[["Alpha"]][[1]] # data.frame
  return(alpha)
}


#============================== VISUALIZATION ==================================
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
    facet_wrap(~K_Denovo, labeller = label_both) + 
    #theme_fivethirtyeight() + 
    xlab("Lambda") + 
    ylab("BIC") + 
    ggtitle("Model Performance over K")
}

#-------------------------------------------------------------------------------
# plot alpha for given k grouped by lambda (OK)
#-------------------------------------------------------------------------------
plot_alpha <- function(data, k) {
  sig_share <- data %>% filter(K_Denovo==k) %>% select(Alpha, Lambda) # tibble
  exposure <- tibble()
  for (i in 1:nrow(sig_share)) {
    row <- sig_share[i,]  # tibble
    alpha <- row[["Alpha"]][[1]]  # data.frame
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
# plot beta
#-------------------------------------------------------------------------------
plot_beta <- function(beta) {
  
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
  return(plot)
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
  ncol1 <- ncol(exp)
  ncol2 <- ncol(inf)  # integer vector
  if (ncol1!=ncol2) {
    return("False input!")
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







