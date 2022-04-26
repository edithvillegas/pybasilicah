#===============================================================================
#============================== FULL RUN =======================================
#===============================================================================
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

simRun <- function(Tprofle, Iprofile, iterNum, startSeed, cos_path) {
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
        print(paste("Run:", counter, "Started!"))
        seed <- startSeed + counter
        counter <- counter + 1

        output <- run_simulated(i,
                                j,
                                cos_path,
                                0.05,
                                0.9,
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

x <- simRun(c("A", "B", "C"),
            c("X", "Y", "Z"),
            iterNum=2,
            startSeed=123,
            cos_path = "/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv")

print(x)


visData <- data.frame(Run=x[["Iter"]],
                  GoodnessofFit = x[["GoodnessofFit"]],
                  Accuracy=x[["Accuracy"]],
                  Quality = x[["Quality"]],
                  Target_Profile = x[["TP"]],
                  Input_Profile = x[["IP"]]
                  )

correct_k_denovo <- sum(x[["Quantity"]], na.rm = TRUE) / length(x[["Quantity"]])
print(paste("ratio of correct K_denovo inference:", correct_k_denovo))

ggplot(data = visData, aes(x=Run, y=GoodnessofFit)) +
  geom_line(stat = "identity") +
  facet_wrap(~Target_Profile + Input_Profile, labeller = label_both) +
  theme_minimal() +
  ggtitle("Goodness of Fit for various types of data") +
  scale_y_continuous(labels=scales::percent)

ggplot(data = visData, aes(x=Run, y=Accuracy)) +
  geom_line(stat = "identity") +
  facet_wrap(~Target_Profile + Input_Profile, labeller = label_both) +
  theme_minimal() +
  ggtitle("Accuracy for various types of data") +
  scale_y_continuous(labels=scales::percent)

ggplot(data = visData, aes(x=Run, y=Quality)) +
  geom_line(stat = "identity") +
  facet_wrap(~Target_Profile + Input_Profile, labeller = label_both) +
  theme_minimal() +
  ggtitle("Beta inferred Quality for various types of data") +
  scale_y_continuous(labels=scales::percent)

'
#------------------------ Parallel Run -----------------------------------------
library(parallel)
numCores <- detectCores()

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

fixedL <- 0.05
denovoL <- 0.9
tprofle <- c("A", "B", "C")
iprofile <- c("X", "Y", "Z")
cos_path <- "/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv"
iterNum <- 10
numRuns <- length(tprofle) * length(iprofile) * iterNum
seed_list <- 1:numRuns

cluster <- makeCluster(detectCores())
#clusterEvalQ(cluster, library(xts))
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


