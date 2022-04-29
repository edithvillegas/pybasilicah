library(ggplot2)    # ggplot()
library(tidyr)      # tibble()
library(dplyr)      # add_row()


#' Plot Exposure Matrix
#'
#' @param alpha
#'
#' @return plot of the exposure matrix
#' @export
#'
#' @examples

plot_alpha <- function(alpha) {
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
    #ggtitle(paste(title)) +
    scale_y_continuous(labels=scales::percent)
}





#' Plot Signature Profiles
#'
#' @param beta data.frame of signatures profiles, which rows represents signatures and
#' columns represents 96 mutational categories
#'
#' @return plot of the signature profiles
#' @export
#'
#' @examples
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
  annotate_figure(plot,
                  top = text_grob("Visualizing Tooth Growth", color = "red", face = "bold", size = 14)
  )

  # JUST FOR TEST --------------------------------------------------------------
  #plot <- do.call("grid.arrange", c(glist, ncol = 1))
  # JUST FOR TEST --------------------------------------------------------------

  return(plot)
}




