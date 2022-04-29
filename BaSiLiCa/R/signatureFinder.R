library(reticulate)
'
#-------------------------------------------------------------------------------
pybasilica <- import("pybasilica")
output <- PyBaSiLiCa$BaSiLiCa(M, B_input, k_list, cosmic_df, fixedLimit, denovoLimit)
#-------------------------------------------------------------------------------
'

#' Title
#'
#' @param catalogue_path
#'
#' @return
#' @export
#'
#' @examples
readCatalogue <- function(catalogue_path) {

  M <- read.table(catalogue_path,
                  sep = ",",
                  header = TRUE,
                  stringsAsFactors = TRUE,
                  check.names=FALSE)
  return(M)
}


#' Title
#'
#' @param beta_path
#'
#' @return
#' @export
#'
#' @examples
readBeta <- function(beta_path) {

  beta <- read.table(beta_path,
                     sep = ",",
                     header = TRUE,
                     stringsAsFactors = TRUE,
                     check.names=FALSE,
                     row.names = 1)
  return(beta)
}


#' Title
#'
#' @param x input mutational counts data (data.frame; rows as samples and columns as 96 mutational categories)
#' @param groups vector of discrete labels with one entry per sample, it defines the groups that will be considered by basilica
#' @param input_catalog input signature profiles, NULL by default
#' @param k vector of possible number of de novo signatures to infer
#' @param reference_catalog a catalog of reference signatures that basilica will use to compare input and de novo signatures
#' @param fixedLimit threshold to discard the signature based on its value in exposure matrix
#' @param denovoLimit threshold to consider inferred signature as COSMIC signature
#'
#' @return inferred exposure matrix, inferred COSMIC signatures and inferred non-COSMIC signatures
#' @export
#'
#' @examples



