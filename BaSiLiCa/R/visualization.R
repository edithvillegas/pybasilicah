library(ggplot2)    # ggplot()
library(tidyr)      # tibble()
library(dplyr)      # add_row()


#' Plot Exposure Matrix
#'
#' @param alpha exposure matrix data.frame
#'
#' @return plot of the exposure matrix
#' @export
#'
#' @examples

plot_exposure <- function(x) {

  alpha$Branch <- c(1:nrow(alpha))
  alpha_long <- gather(alpha,
                       key="Signature",
                       value="Exposure",
                       c(-Branch)
  )

  ggplot(data = alpha_long, aes(x=Branch, y=Exposure, fill=Signature)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    #ggtitle(paste(title)) +
    scale_y_continuous(labels=scales::percent)
}


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
plot_signatures <- function( x, useRowNames = FALSE, xlabels = FALSE ) {

  # set names of the signatures
  if(!useRowNames) {
    rownames(beta) <- paste0("Signature ",1:nrow(beta))
  }

  # separate context and alteration
  x <- as.data.table(reshape2::melt(as.matrix(beta),varnames=c("signature","cat")))
  x[,Context:=paste0(substr(cat,1,1),".",substr(cat,7,7))]
  x[,alt:=paste0(substr(cat,3,3),">",substr(cat,5,5))]

  # make the ggplot2 object
  glist <- list()
  for(i in 1:nrow(beta)) {

    plt <- ggplot(x[signature==rownames(beta)[i]]) +
      geom_bar(aes(x=Context,y=value,fill=alt),stat="identity",position="identity") +
      facet_wrap(~alt,nrow=1,scales="free_x") +
      theme(axis.text.x=element_text(angle=90,hjust=1),panel.background=element_blank(),axis.line=element_line(colour="black")) +
      ggtitle(rownames(beta)[i]) + theme(legend.position="none") + ylab("Frequency of mutations")

    if(!xlabels) {
      plt <- plt + theme(axis.text.x=element_blank(),axis.ticks.x=element_blank())
    }

    glist[[i]] <- plt

  }

  # make the final plot
  grid.arrange(grobs=glist,ncol=ceiling(nrow(beta)/3))

}



