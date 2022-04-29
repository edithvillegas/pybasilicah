library(ggtree)
library(data.table)


M_plot(alpha)


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



