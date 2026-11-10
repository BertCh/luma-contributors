# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Direct blockCV references, with explicit projected user blocks and leave-one-point buffer folds.
# PATH=<R-prefix>/bin:$PATH Rscript modules/analysis/test/oracles/statistics-blockcv.R
suppressPackageStartupMessages(library(sf))
suppressPackageStartupMessages(library(blockCV))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(as.character(packageVersion('blockCV')) == '4.0.1')
coordinates <- cbind(rep(0:3, 3), rep(0:2, each=4))
points <- st_as_sf(data.frame(x=coordinates[,1], y=coordinates[,2]), coords=c('x','y'), crs=3857)
polygons <- st_sfc(lapply(0:3,function(column) st_polygon(list(matrix(c(column-.4,-.4,column+.4,-.4,column+.4,2.4,column-.4,2.4,column-.4,-.4),ncol=2,byrow=TRUE)))),crs=3857)
blocks <- st_sf(fold=c(1L,2L,3L,1L),geometry=polygons)
spatial <- cv_spatial(points,k=3,hexagon=FALSE,user_blocks=blocks,folds_column='fold',selection='predefined',seed=17,balance=FALSE,plot=FALSE,report=FALSE,progress=FALSE)
buffer <- cv_buffer(points,size=1.1,report=FALSE,progress=FALSE)
make_reference <- function(model,identifiers,fold_count,distance) {
 list(positions=as.numeric(t(coordinates)),blocks=identifiers,seed=17,foldCount=fold_count,bufferDistance=distance,training=unlist(lapply(model$folds_list,function(fold) as.integer(seq_len(12)%in%fold[[1]]))),testing=unlist(lapply(model$folds_list,function(fold) as.integer(seq_len(12)%in%fold[[2]]))))
}
# IDs select the requested folds under the documented uint32 hash; blockCV independently
# supplies the spatial membership and buffered training masks, not that hash algorithm.
result <- list(versions=list(blockCV=as.character(packageVersion('blockCV')),sf=as.character(packageVersion('sf'))),spatial=make_reference(spatial,rep(c(0,6,1,0),3),3,0),buffer=make_reference(buffer,c(3,8,31,7,16,12,22,6,36,0,10,1),12,1.1))
write_json(result,'modules/analysis/test/oracles/fixtures/statistics-blockcv.json',auto_unbox=TRUE,pretty=TRUE,digits=16)
cat('Executed blockCV 4.0.1 cv_spatial predefined polygons and cv_buffer\n')
