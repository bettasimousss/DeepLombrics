full_shap_proj = read.csv('limitingfactors.csv',row.names = 1)

shap_obs = read.csv('shap_observation.csv')

feature_names = read.csv('feature_names.csv')

old2new = feature_names$New_name
names(old2new) = feature_names$Old_name


cols = c("awc_top","bs_top","C","cec_top","clay","clc","CN","crusting","dgh", 
         "dimp","erodi","parmado","PColdQ","pd_top","PDryM","PDryQ","pH",
         "proxi_eau_fast","PSeason","PTotY","PWarmQ","PWetM","PWetQ","silt","TIso",
         "TMaxWarmM","TMeanColdQ","TMeanDryQ","TMeanRngD","TMeanWarmQ","TMeanWetQ",
          "TMeanY","TMinColdM","TRngY","TSeason","wr")

# sel_cols = c("PColdQ","PDryM","PDryQ","PSeason","PTotY","PWarmQ","PWetM","PWetQ",
#              "TIso","TMaxWarmM","TMeanColdQ","TMeanDryQ","TMeanRngD","TMeanWarmQ","TMeanWetQ",
#              "TMeanY","TMinColdM","TRngY","TSeason")

sel_cols = c("TMeanRngD","TMinColdM","PWarmQ","PDryM","PColdQ","TMeanWarmQ","TIso","TSeason")


library(ggplot2)

plot_lim_fact<-function(taxcode){
  shap_proj = subset(full_shap_proj,taxa==taxcode)
  df = shap_proj[,c(c('Longitude','Latitude'),sel_cols)]
  df_melted = reshape2::melt(df, id.vars = c('Longitude','Latitude'), variable.name = "Feature", value.name = "SHAP")
  
  new_names = sapply(df_melted$Feature,function(x) as.character(old2new[as.character(x)]))
  df_melted$EnvFeature = new_names
  
  # High-quality ggplot
  p <- ggplot(df_melted, aes(x = Longitude, y = Latitude, color = SHAP)) +
    geom_point(size = 1, alpha = 0.7) +  # Adjust point size & transparency
    
    # Color scale for SHAP values
    scale_color_gradient2(
      low = "#2166AC", mid = "white", high = "#B2182B", 
      midpoint = 0, space = "Lab", name = "SHAP",
      guide = guide_colorbar(barwidth = 1, barheight = 20)
    ) +
    
    # Facet layout optimized for 36 panels
    facet_wrap(~ EnvFeature, ncol = 4) +  # Adjust ncol for layout (e.g., 6x6 grid)
    
    # Improved theme for publication
    theme_minimal(base_size = 12) +  # Increase base font size for readability
    theme(
      strip.text = element_text(size = 12, face = "bold"),  # Facet labels larger
      axis.text = element_blank(),  # Hide axis text (since it's spatial)
      axis.ticks = element_blank(),
      panel.grid = element_blank(),  # Remove grid for clean look
      legend.position = "right",  # Place legend on right
      legend.text = element_text(size = 12),
      legend.title = element_text(size = 12, face = "bold"),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
    ) +
    
    labs(title = "Geolocalized SHAP Importance of Environmental Factors", 
         color = "SHAP")  # Adjust title & legend label
  
  # Save high-resolution image (Larger for 36 panels)
  ggsave(sprintf("LimFact/%s_limiting_factors.png",taxcode), plot = p, width = 9, height = 5, dpi = 300)
  
}

tax_list = unique(full_shap_proj$taxa)

plot_lim_fact('Apicte')

for (taxcode in tax_list){
  plot_lim_fact(taxcode)
}

#df_sel = subset(df_melted,Feature %in% sel_cols)


# Print the plot
print(p)


library(reshape2)

#library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering visualization
library(dendextend) # for comparing two dendrograms
library(ggdendro)


shaptaxa=reshape2::dcast(shap_global,feature + group + site ~ class,value.var = 'phi')
shapsite=reshape2::dcast(shap_global,feature + group + class ~ site,value.var = 'phi')

codes = unique(shap_global$taxa)

set.seed(123)


fg = 'precipitation'
subshaptaxa=subset(shaptaxa,group %in% c(fg))
df=t(as.matrix(subshaptaxa[,-c(1,2,3)]))
rownames(df)=codes


meth=hcut
optwss=fviz_nbclust(
  x=df,
  FUNcluster = meth,
  method = "wss",
  k.max = 10
)
gap_stat <- cluster::clusGap(df, FUN = meth, K.max = 10, B = 10)
print(gap_stat,method='firstSEmax')
print(gap_stat,method='globalSEmax')
plot(fviz_gap_stat(gap_stat))
plot(optwss)

plot(fviz_cluster(hcut(df,4),data=df))

plot(fviz_cluster(hcut(df,4),df,repel = T,show.clust.cent = T,ellipse.type = 'none',main=paste(fg,'4-hcut')))
plot(fviz_cluster(hcut(df,4),df,repel = T,show.clust.cent = T,ellipse.type = 'norm',main=paste(fg,'4-hcut')))

plot(fviz_cluster(hcut(df,9),df,repel = T,show.clust.cent = T,ellipse.type = 'none',main=paste(fg,'4-hcut')))
plot(fviz_cluster(hcut(df,9),df,repel = T,show.clust.cent = T,ellipse.type = 'norm',main=paste(fg,'4-hcut')))

p=ggdendrogram(hcut(df),rotate = T)+ggtitle(fg)
# ggsave(paste('paper/final_figures/response_groups/',fg,'_dendro.png',sep = ''),p,width = 5 ,height = 10)


######### SAVE
palette = c("#F8766D", "#7CAE00", "#00BFC4", "#C77CFF")
prec_plot = fviz_cluster(hcut(df,4),df,repel = T,show.clust.cent = T,palette=palette,
                         ellipse.type = 'none',main='Precipitation response groups',xlab = 'PC1 (43.1%)',ylab = 'PC2 (13.8%)')
ggsave('prec_clusters.png',prec_plot,width = 10 ,height = 5)

cluster_data = prec_plot$data
# cluster_colors <- c(
#   `1` = "#E76F51",  # Cluster 1 - Red/Pink
#   `2` = "#90BE6D",  # Cluster 2 - Green
#   `3` = "#43AA8B",  # Cluster 3 - Cyan/Teal
#   `4` = "#9D4EDD"   # Cluster 4 - Purple
# )

cluster_colors = palette
names(cluster_colors) = c('1','2','3','4')

taxcolors=cluster_colors[cluster_data$cluster]
names(taxcolors)=cluster_data$name

# 
# taxcolors = rep('black',length(codes))
# names(taxcolors) = codes

lvars = list()
lvars[['bio_15']] = 'PSeason'

get_dendro<-function(hc,titre=''){ 
  ##set order of taxa according to hierarchical clustering 
  txorder=hc$order
  ##Setup colors 
  tcolors=as.character(taxcolors[codes[txorder]])
  ##Prepare dendrogram data 
  dendrf <- dendro_data(hc, type="rectangle") 
  # convert for ggplot 
  clust.dff <- data.frame(label=codes) 
  dendrf[["labels"]] <- merge(dendrf[["labels"]],clust.dff, by="label")
  ##Dendrogram plot 
  g1<- ggplot() + 
    geom_segment(data=segment(dendrf), aes(x=x, y=y, xend=xend, yend=yend)) + 
    geom_text(data=label(dendrf), aes(x, y, label= label, hjust=0.01 ,color=label ),size=6) + 
    coord_flip() + scale_y_reverse(expand=c(1, 1)) + 
    theme_dendro() + 
    scale_color_manual(values=taxcolors)+ 
    theme(legend.position="none", panel.background = element_rect(fill = "transparent"), # bg of the panel 
          plot.background = element_rect(fill = "transparent", color = NA), plot.title = element_text(family = "sans",face='bold', hjust=0.5,vjust=1,size = 20, margin=margin(0,0,5,0))) + ggtitle(titre)
  return(list(tcolors=tcolors,txorder=txorder,g1=g1)) }

## Routine to plot SHAP summary
shap_summary <- function(v, vt, shap_global, txorder = seq(1, 77), 
                         sel_taxa = codes, titre = "", lpos = "right", 
                         tcols = taxcolors, scale = 1, colours = NULL, 
                         shapes = NULL, filt = c("NS")) { 
  
  ## Subset relevant SHAP values
  vn <- lvars[[v]]
  subdata <- subset(shap_global, feature == v & taxa %in% sel_taxa)
  subdata$taxa <- factor(subdata$taxa, levels = as.character(sel_taxa)[txorder])
  
  ## Encode categories
  if (vt == "numerical") {
    subdata$xval <- subdata$xval / scale
    
    g2 <- ggplot(subdata, aes(y = taxa, x = phi, col = xval)) +
      geom_jitter() +
      labs(fill = vn, col = vn) +
      ylab(NULL) + xlab(NULL) +
      scale_y_discrete(position = "left") +
      theme(
        legend.position = lpos,
        axis.text.y = element_text(size = 12, colour = tcols),
        axis.text.x = element_text(size = 10, angle = -90),
        legend.key.size = unit(3, "line"),
        legend.text = element_text(size = 10, angle=-90),
        legend.title = element_text(size = 12,angle = -90),
        plot.title = element_text(family = "sans", face = "bold", 
                                  hjust = 0.5, size = 14, margin = margin(0, 0, 5, 0))
      ) +
      #ggtitle(titre) +
      scale_color_viridis_c(option = "magma")
    
    ggsave('prec_summary.png',g2,width = 10 ,height = 20, dpi = 300)
    
    
  } else {
    subdata$xvalc <- factor(as.character(scale[as.character(subdata$xval)]), 
                            levels = as.character(scale))
    
    g2 <- ggplot(subset(subdata, !(xvalc %in% c(filt, NA))), aes(y = taxa, x = phi, col = xvalc))
    
    if (!is.null(shapes)) {
      g2 <- g2 + geom_jitter(aes(shape = xvalc), size = 5) +
        scale_shape_manual(values = shapes)
    } else {
      g2 <- g2 + geom_jitter()
    }
    
    g2 <- g2 +
      labs(fill = vn, col = vn) +
      ylab(NULL) + xlab(NULL) +
      scale_y_discrete(position = "left") +
      theme(
        legend.position = lpos,
        axis.text.y = element_text(size = 16, colour = tcols),
        axis.text.x = element_text(size = 16),
        legend.key.size = unit(10, "pt"),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 16),
        plot.title = element_text(family = "sans", face = "bold", 
                                  hjust = 0.5, size = 20, margin = margin(0, 0, 5, 0))
      ) +
      ggtitle(titre)
    
    if (is.null(colours)) {
      g2 <- g2 + scale_color_viridis_d(option = "magma")
    } else {
      g2 <- g2 + scale_color_manual(values = colours)
    }
  }
  
  return(g2)
}



###############################################################################
####### PRECIPITATION GROUPS
###############################################################################

hcprec = hcut(df,4)
prep<-get_dendro(hcprec)
g2<-shap_summary(v='bio_15',vt = 'numerical',shap_global = shap_global,txorder = prep$txorder,sel_taxa = codes,
                 titre='Precipitation Seasonality',scale=10,lpos = 'bottom',tcols = prep$tcolors)

pdf('precip_groups_prec.pdf',height=15,width=10)
gridExtra::grid.arrange(prep$g1,g2,ncol=2) 
dev.off()


hcphylo=hclust(as.dist(pdis))
prep<-get_dendro(hcphylo)
g2<-shap_summary(v='bio_15',vt = 'numerical',shap_global = shap_global,txorder = prep$txorder,sel_taxa = codes,
                 titre='Precipitation Seasonality',scale=10,lpos = 'bottom',tcols = prep$tcolors)


pdf('precip_groups_phylo.pdf',height=15,width=10)
gridExtra::grid.arrange(prep$g1,g2,ncol=2) 
dev.off()



###############################################################################
####### LANDCOVER GROUPS
###############################################################################

### Cluster colors
kclc = kmeans(df,6)

cluster_colors <- c("#e066d7", "#ea736c", "#bfa735", "#2cb24a", "#7ab2f2", "#00b9c0")
landcover_plot = fviz_cluster(kclc,df,repel = T,show.clust.cent = T,palette=cluster_colors,
                              ellipse.type = 'none',main='Landcover response groups',xlab = 'PC1 (62%)',ylab = 'PC2 (17.5%)')
ggsave('landcover_clusters.png',landcover_plot,width = 10 ,height = 5)


hclc = hcut(df,6)
prep<-get_dendro(hclc)


species_clusters = kclc$cluster
names(cluster_colors) = c('1','2','3','4','5','6')

taxcolors=cluster_colors[kclc$cluster]
names(taxcolors)=names(kclc$cluster)


### SHAP summary
fg = 'landcover'
subshaptaxa=subset(shaptaxa,group %in% c(fg))
df=t(as.matrix(subshaptaxa[,-c(1,2,3)]))
rownames(df)=codes

sel_taxa = codes
v = 'clc'
scale = clcscales
filt = c()
txorder = prep$txorder#1:length(codes)
tcols = taxcolors[txorder]
lpos = 'right'
titre = 'Landcover SHAP summary'

##subset relevant shapley values
vn=lvars[[v]]
subdata=subset(shap_global,feature==v & taxa %in% sel_taxa)
subdata$taxa=factor(subdata$taxa, levels = as.character(sel_taxa)[txorder])
subdata$xvalc=factor(as.character(scale[as.character(subdata$xval)]),levels = as.character(scale))

g2=ggplot(subset(subdata,!(xvalc %in% c(filt,NA))),aes(y=taxa,x=phi,col=xvalc))
# if(!is.null(shapes)) g2=g2+geom_jitter(aes(shape=xvalc,size=5))+
#   scale_shape_manual(values=shapes) else 

g2=g2+geom_jitter()

g2=g2+labs(fill=vn,col=vn)+
  ylab(NULL)+xlab(NULL)+
  scale_y_discrete(position = "left")+
  theme(
    legend.position = 'bottom',
    axis.text.y = element_text(size = 12, colour = tcols),
    axis.text.x = element_text(size = 10),
    #legend.key.size = unit(8,""),
    legend.text = element_text(size = 10),
    legend.title = element_text(size = 12),
    plot.title = element_text(family = "sans", face = "bold", 
                              hjust = 0.5, size = 14, margin = margin(0, 0, 5, 0)))#+
  #ggtitle(titre)

g2=g2+scale_color_manual('Land cover',values=clccolours) + guides(color = guide_legend(override.aes = list(size=5)))

ggsave('clc_summary.png',g2,width = 10 ,height = 20, dpi = 300)


##### CLUSTERS
nc = 6
kclc = hcut(df,6)
kclc = kmeans(df,6)
fviz_cluster(kclc,df, repel = T,show.clust.cent = F,
             ellipse.type = 'none',
             main=paste('HCLUST',fg,nc))  

landcover_plot = fviz_cluster(kclc,df,repel = T,show.clust.cent = T,palette=cluster_colors,
                         ellipse.type = 'none',main='Landcover response groups',xlab = 'PC1 (62%)',ylab = 'PC2 (17.5%)')
ggsave('landcover_clusters.png',landcover_plot,width = 10 ,height = 5)


meth=kmeans
optwss=fviz_nbclust(
  x=df,
  FUNcluster = meth,
  method = "wss",
  k.max = 10
)
gap_stat <- cluster::clusGap(df, FUN = meth, K.max = 10, B = 10)
print(gap_stat,method='firstSEmax')
print(gap_stat,method='globalSEmax')
plot(fviz_gap_stat(gap_stat))
plot(optwss)

plot(fviz_cluster(hcut(df,4),data=df))

plot(fviz_cluster(hcut(df,4),df,repel = T,show.clust.cent = T,ellipse.type = 'none',main=paste(fg,'4-hcut')))
plot(fviz_cluster(hcut(df,4),df,repel = T,show.clust.cent = T,ellipse.type = 'norm',main=paste(fg,'4-hcut')))

plot(fviz_cluster(hcut(df,9),df,repel = T,show.clust.cent = T,ellipse.type = 'none',main=paste(fg,'4-hcut')))
plot(fviz_cluster(hcut(df,9),df,repel = T,show.clust.cent = T,ellipse.type = 'norm',main=paste(fg,'4-hcut')))

p=ggdendrogram(hcut(df),rotate = T)+ggtitle(fg)
# ggsave(paste('paper/final_figures/response_groups/',fg,'_dendro.png',sep = ''),p,width = 5 ,height = 10)



