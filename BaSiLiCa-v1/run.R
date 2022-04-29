#===============================================================================
#============================== FULL RUN =======================================
#===============================================================================
library(reticulate)
library(tidyr)      # tibble()
library(dplyr)      # add_row()
library(ggplot2)    # ggplot()

getwd() # mostly starts at "/home/azad"
setwd("/home/azad")
use_condaenv("SigPhylo")
setwd("/home/azad/Documents/thesis/SigPhylo/PyBaSiLiCa")
source_python("utilities.py")
source_python("basilica.py")
setwd("/home/azad/Documents/thesis/SigPhylo") # change the directory
import("PyBaSiLiCa")
setwd("/home/azad/Documents/thesis/SigPhylo/PyBaSiLiCa")
source_python("simulation.py")


M <- readCatalogue("/home/azad/Documents/thesis/SigPhylo/data/real/data_sigphylo.csv")
beta_input <- readBeta("/home/azad/Documents/thesis/SigPhylo/data/real/beta_aging.csv")
beta_cosmic <- readBeta("/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv")
beta_expected <- readBeta("/home/azad/Documents/thesis/SigPhylo/data/real/expected_beta.csv")

x <- fitModel(M, beta_input, k_list, beta_cosmic, 0.05, 0.9)
alpha <- x[["Alpha"]]
beta_fixed <- x[["Beta_Fixed"]]
beta_denovo <- x[["Beta_Denovo"]]

plot_alpha(alpha, "alpha")


Tprofle <- c("C")
Iprofile <- c("X", "Y", "Z")
xC <- simRun(Tprofle,
            Iprofile,
            cos_path_org="/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv",
            fixedLimit=0.05,
            denovoLimit=0.9,
            startSeed=100,
            iterNum=20
            )

getwd()
saveRDS(x, file = "fulldata.rds")
x <- rbind(xA, xB, xC)
print(x)


visData <- data.frame(Run=x[["Iter"]],
                  GoodnessofFit = x[["GoodnessofFit"]],
                  Accuracy=x[["Accuracy"]],
                  Quality = x[["Quality"]],
                  Target_Profile = x[["TP"]],
                  Input_Profile = x[["IP"]]
                  )
visData
df <- subset(visData, Quality<0)
df

correct_k_denovo <- sum(x[["Quantity"]], na.rm = TRUE) / length(x[["Quantity"]])
print(paste("ratio of correct K_denovo inference:", correct_k_denovo))

ggplot(data = visData, aes(x=Run, y=GoodnessofFit)) +
  geom_line(stat = "identity") +
  facet_wrap(~Target_Profile + Input_Profile, labeller = label_both) +
  theme_minimal() +
  ggtitle("Goodness of Fit for various types of data") +
  scale_y_continuous(labels=scales::percent)

ggplot(data = visData, aes(x=Accuracy)) +
  geom_histogram() +
  facet_wrap(~Target_Profile + Input_Profile, labeller = label_both) +
  theme_minimal() +
  ggtitle("Accuracy for various types of data")


ggplot(data = visData, aes(x=Run, y=Accuracy)) +
  geom_line(stat = "identity") +
  facet_wrap(~Target_Profile + Input_Profile, labeller = label_both) +
  theme_minimal() +
  ggtitle("Accuracy for various types of data") +
  scale_y_continuous(labels=scales::percent)

ggplot(data = visData, aes(x=Run, y=Quality)) +
  geom_point(stat = "identity") + geom_line() +
  facet_wrap(~Target_Profile + Input_Profile, labeller = label_both) +
  theme_minimal() +
  ggtitle("Beta inferred Quality for various types of data") +
  scale_y_continuous(labels=scales::percent)


#-------------------------------------------------------------------------------

rds_path <- "/home/azad/Documents/thesis/rds/raw_signa.rds"
counts <- readRDS(rds_path)

M <- read.table(M_path, sep = ",", header = TRUE, stringsAsFactors = TRUE, check.names=FALSE)
B_input <- read.table(B_input_path, sep = ",", header = TRUE, stringsAsFactors = TRUE, check.names=FALSE)
k_list <- c(0, 1, 2, 3, 4, 5)
cosmic_df <- read.table(cosmic_path, sep = ",", header = TRUE, stringsAsFactors = TRUE, check.names=FALSE)
fixedLimit <- 0.05
denovoLimit <- 0.9

source_python("basilica.py")
MM <- r_to_py(M)
BB_input <- r_to_py(B_input)
kk_list <- r_to_py(k_list)
ccosmic_df <- r_to_py(cosmic_df)
op <- BaSiLiCa(MM, BB_input, kk_list, ccosmic_df, fixedLimit, denovoLimit)

PyBaSiLiCa$BaSiLiCa
PyBaSiLiCa$basilica$BaSiLiCa




#------------------------ Parallel Run -----------------------------------------
library(parallel)
library(easypar)
numCores <- detectCores()

tprofile <- c("A", "B", "C")
iprofile <- c("X", "Y", "Z")
cos_path <- "/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv"
fixedL <- 0.05
denovoL <- 0.9
seed <- 1:5

my_params <- expand.grid(Tprofile=tprofile,
                         Iprofile=iprofile,
                         cos_path_org=cos_path,
                         fixedLimit=fixedL,
                         denovoLimit=denovoL,
                         seed=1:5)


inputs = lapply(1:nrow(my_params), list)

report <-
  easypar::run(
    FUN = function(x)
    {

      Tprofile <- my_params$Tprofile[x]
      Iprofile <- my_params$Iprofile[x]
      cos_path_org <- my_params$cos_path_org[x]
      fixedLimit <- my_params$fixedLimit[x]
      denovoLimit <- my_params$denovoLimit[x]
      seed <- my_params$seed[x]

      library(reticulate)
      library(tidyr)      # tibble()
      library(dplyr)      # add_row()
      library(ggplot2)    # ggplot()

      #getwd() # mostly starts at "/home/azad"
      setwd("/home/azad")
      use_condaenv("SigPhylo")
      setwd("/home/azad/Documents/thesis/SigPhylo/PyBaSiLiCa")
      source_python("utilities.py")
      setwd("/home/azad/Documents/thesis/SigPhylo") # change the directory
      import("PyBaSiLiCa")
      setwd("/home/azad/Documents/thesis/SigPhylo/PyBaSiLiCa")
      source_python("simulation.py")

      run_simulated(Tprofile, Iprofile, cos_path_org, fixedLimit, denovoLimit, seed)

    },
    PARAMS = inputs,
    parallel = TRUE,
    #cores.ratio = .8,
    filter_errors = FALSE,
    export = ls(globalenv())
  )


output <- run_simulated(Tprofile, Iprofile, cos_path_org, fixedLimit, denovoLimit, seed)


output <- run_simulated("A",
                        "X",
                        "/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv",
                        0.05,
                        0.9,
                        seed)


for (i in Tprofle) {
  for (j in Iprofile) {
    for (k in 1:iterNum) {
      params <- c(i,
                  j,
                  "/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv",
                  0.05,
                  0.9,
                  seed)
      params_list <- list(params_list, params)
      seed <- seed + 1
    }
  }
}

library(xts)
install.packages("xts")

fixedL <- 0.05
denovoL <- 0.9
tprofle <- c("A", "B", "C")
iprofile <- c("X", "Y", "Z")
cos_path <- "/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv"
iterNum <- 10
numRuns <- length(tprofle) * length(iprofile) * iterNum
seed_list <- 1:numRuns

cluster <- makeCluster(detectCores())
clusterEvalQ(cluster, library(xts))
result <- clusterMap(cluster, run_simulated, Tprofile=tprofle, Iprofile=iprofile, seed=1:length(tprofle),
                     MoreArgs=list(cos_path_org=cos_path, fixedLimit=fixedL, denovoLimit=denovoL))



func_parallel <- function(params) {
  Tprofile <- params[1]
  Iprofile <- params[2]
  cos_path <- params[3]
  fixedLimit <- as.numeric(params[4])
  denovoLimit <- as.numeric(params[5])
  seed <- as.numeric(params[6])

  output <- run_simulated(Tprofile,
                          Iprofile,
                          cos_path,
                          fixedLimit,
                          denovoLimit,
                          seed)

  return(output)
}

results <- mclapply(params_list, run_parallel, mc.cores = numCores)


#-------------------------------------------------------------------------------
'






