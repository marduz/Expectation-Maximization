library(ggpubr)
# ---------------------------------------------------------------------------------------
for(i in 1:2){
  path = paste("./data/example",i,"_labeled.csv", sep = "")
  data = read_csv(path)
  graph_ = ggplot(data = data, aes(x = X1, y = X2)) +
    geom_point() + theme(legend.position = "none") 
  assign(paste0("g_u", i), graph_)
  graph_ = ggplot(data = data, aes(x = X1, y = X2, color = factor(Cluster))) +
    geom_point() + theme(legend.position = "none") 
  assign(paste0("g_l", i), graph_)
  
}
p = ggarrange(g_u1, g_l1, g_u2, g_l2, ncol=2, nrow=2)
png("./graph/em_clustering.png", width=800, height=800)
p
dev.off()
