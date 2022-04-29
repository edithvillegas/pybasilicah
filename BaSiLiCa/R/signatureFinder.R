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
#' @param catalogue mutational catalog
#' @param beta_input input signature profiles
#' @param k_list list of number of the signatures to infer
#' @param beta_cosmic all signatures from COSMIC
#' @param fixedLimit threshold to discard the signature based on its value in exposure matrix
#' @param denovoLimit threshold to consider inferred signature as COSMIC signature
#'
#' @return inferred exposure matrix, inferred COSMIC signatures and inferred non-COSMIC signatures
#' @export
#'
#' @examples
fitModel <- function(catalog, beta_input, k_list=0:5, beta_cosmic, fixedLimit=0.05, denovoLimit=0.9) {

  setwd("/home/azad/Documents/thesis/SigPhylo/PyBaSiLiCa")
  source_python("basilica.py")

  M <- r_to_py(catalog)
  beta_input <- r_to_py(beta_input)
  #----------------------------- MUST BE CHANGED -------------------------------
  py_run_string("k_list = list(map(int, [0, 1, 2, 3, 4, 5]))")
  k_list <- py$k_list
  k_list <- r_to_py(k_list)
  #-----------------------------------------------------------------------------
  beta_cosmic <- r_to_py(beta_cosmic)
  fixedLimit <- r_to_py(fixedLimit)
  denovoLimit <- r_to_py(denovoLimit)

  output <- BaSiLiCa(M, beta_input, k_list, beta_cosmic, fixedLimit, denovoLimit)

  alpha <- output[[1]]
  beta_fixed <- output[[2]]
  beta_denovo <- output[[3]]

  return(list(Alpha = alpha, Beta_Fixed = beta_fixed, Beta_Denovo = beta_denovo))

}



