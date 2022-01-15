library(ggplot2)

path <- "/home/azad/Documents/thesis/SigPhylo/data/data_sigphylo.csv"
#"/home/azad/Documents/thesis/SigPhylo/data/"
df <- read.csv(path, header = TRUE)
print(df)

ggplot(data = df, aes(x = 1:96, y=data[1,])) +
  geom_bar()
