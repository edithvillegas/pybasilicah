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

setwd("/home/azad/Documents/thesis/SigPhylo") # change the directory
import("PyBaSiLiCa")
setwd("/home/azad/Documents/thesis/SigPhylo/PyBaSiLiCa/PyBaSiLiCa")
source_python("basilica.py")
setwd("/home/azad/Documents/thesis/SigPhylo/PyBaSiLiCa/test")
source_python("simulation.py")


simRun <- function(Tprofle, Iprofile, iterNum, startSeed) {

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
    Seed = numeric()
  )

  counter <- 1

  for (i in Tprofle) {
    for (j in Iprofile) {
      for (k in 1:iterNum) {
        print(paste("Run:", counter, "Started!"))
        seed <- startSeed + counter
        counter <- counter + 1

        output <- run_simulated("A",
                                "X",
                                "/home/azad/Documents/thesis/SigPhylo/data/cosmic/cosmic_catalogue.csv",
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
                           Seed = seed
        )
      }
    }
  }
  return(results)
}

x <- simRun(c("A", "B", "C"), c("X", "Y", "Z"), iterNum=3, startSeed=123)
print(x)

nrow(x)

visData <- data.frame(Run=1:nrow(x),
                  GoodnessofFit = x[["GoodnessofFit"]],
                  Accuracy=x[["Accuracy"]],
                  Quality = x[["Quality"]]
                  )

correct_k_denovo <- sum(x[["Quantity"]], na.rm = TRUE) / length(x[["Quantity"]])
print(paste("ratio of correct K_denovo inference:", correct_k_denovo))
ggplot(visData, aes(x=Run, y=GoodnessofFit)) + geom_line()
ggplot(visData, aes(x=Run, y=Accuracy)) + geom_line()
ggplot(visData, aes(x=Run, y=Quality)) + geom_line()



#-------------------------------------------------------------------------------

ggplot(data_tibble, aes(x=Sample, y=Accuracy)) +
  geom_line()

ggplot(data_tibble, aes(x=Sample, y=Precision)) +
  geom_line()

ggplot(data_tibble, aes(x=Sample, y=Recall)) +
  geom_line()

ggplot(data_tibble, aes(x=Sample, y=F1)) +
  geom_line()

ggplot(data_tibble, aes(x=Sample, y=alpha_mse)) +
  geom_line()


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




