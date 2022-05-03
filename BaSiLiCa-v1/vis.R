

#-------------------------------------------------------------------------------

#' Plot Exposure Matrix
#'
#' @param alpha exposure matrix data.frame
#'
#' @return plot of the exposure matrix
#' @export
#'
#' @examples


#-------------------------------------------------------------------------------

#' Title
#'
#' @param beta
#' @param useRowNames
#' @param xlabels
#'
#' @return
#' @export
#'
#' @examples

#-------------------------------------------------------------------------------

M_plot <- function( trinucleotides_counts, samples = rownames(trinucleotides_counts), freq = FALSE, xlabels = FALSE ) {
  
  # make samples data
  trinucleotides_counts <- trinucleotides_counts[samples,,drop=FALSE]
  if(freq) {
    trinucleotides_counts <- trinucleotides_counts / rowSums(trinucleotides_counts)
  }
  
  # separate context and alteration
  x <- as.data.table(reshape2::melt(as.matrix(trinucleotides_counts),varnames=c("patient","cat")))
  x[,Context:=paste0(substr(cat,1,1),".",substr(cat,7,7))]
  x[,alt:=paste0(substr(cat,3,3),">",substr(cat,5,5))]
  
  # make the ggplot2 object
  glist <- list()
  for(i in 1:nrow(trinucleotides_counts)) {
    
    plt <- ggplot(x[patient==rownames(trinucleotides_counts)[i]]) +
      geom_bar(aes(x=Context,y=value,fill=alt),stat="identity",position="identity") +
      facet_wrap(~alt,nrow=1,scales="free_x") +
      theme(axis.text.x=element_text(angle=90,hjust=1),panel.background=element_blank(),axis.line=element_line(colour="black")) +
      ggtitle(rownames(trinucleotides_counts)[i]) + theme(legend.position="none") + ylab("Number of mutations")
    
    if(freq) {
      plt <- plt + ylab("Frequency of mutations")
    }
    
    if(!xlabels) {
      plt <- plt + theme(axis.text.x=element_blank(),axis.ticks.x=element_blank())
    }
    
    glist[[i]] <- plt
    
  }
  
  # make the final plot
  grid.arrange(grobs=glist,ncol=ceiling(nrow(trinucleotides_counts)/3))
  
}


#===============================================================================
# plot phylogeny tree
#===============================================================================
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


#-------------------------------------------------------------------------------
# visualize signature profiles shares in branches
#-------------------------------------------------------------------------------
signature_share <- function(path, title) {
  df <- read.table(path, sep = ",")
  df$branch <- seq(1:(nrow(df)))
  ndf <- melt(df, id.vars = "branch")
  
  plot <- ggplot(data=ndf, aes(x=branch, y=value, fill=variable, label = value)) + 
    geom_bar(stat = "identity", show.legend = FALSE) + 
    geom_text(size = 2, position = position_stack(vjust = 0.5)) + 
    ggtitle(title)
  
  return(plot)
  
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


#============================== VISUALIZATION ==================================
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

