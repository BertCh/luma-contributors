# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
stopifnot(as.character(packageVersion("blockCV")) == "4.0.1")
suppressPackageStartupMessages(library(blockCV));suppressPackageStartupMessages(library(sf));suppressPackageStartupMessages(library(terra));library(jsonlite)
coordinates<-cbind(c(0,1,2,4,8,9,14,17),c(0,0.2,0,0.3,0,0.1,0,0.2))
x<-st_as_sf(data.frame(x=coordinates[,1],y=coordinates[,2],group=c(3,3,8,8,12,12,19,19)),coords=c('x','y'),crs=3857)
prediction<-cbind(c(21,24,28,32,38),c(1,1,2,2,1))
p<-st_as_sf(data.frame(x=prediction[,1],y=prediction[,2]),coords=c('x','y'),crs=3857)
g<-cv_group(x,'group',k=3,report=FALSE)
n<-cv_nndm(x,pred_points=p,size=4,min_train=.2,plot=FALSE,report=FALSE)
k<-cv_knndm(x,pred_points=p,k=2,maxp=.8,clustering='hierarchical',nk_len=5,balance=FALSE,seed=17,plot=FALSE,report=FALSE)
feature<-c(0,.1,.2,.3,9.5,9.7,9.8,10)
r<-rast(nrows=1,ncols=8,xmin=0,xmax=8,ymin=0,ymax=1,crs='EPSG:3857');values(r)<-feature
ex<-st_as_sf(data.frame(x=seq(.5,7.5),y=.5),coords=c('x','y'),crs=3857)
set.seed(17);e<-cv_cluster(ex,r=r,k=2,scale=FALSE,balance=FALSE,report=FALSE)
# Retain actual internal candidate fold partitions for matching selection independently of R stochastic initialization.
d<-as.matrix(st_distance(x));cand<-blockCV:::.knndm_candidates('hierarchical',2,8,5,d,coordinates,'ward.D2',x,TRUE,111325)
cand<-Filter(Negate(is.null),lapply(cand,function(z)blockCV:::.merge_clusters(z,coordinates,2,.8)))
output<-list(version=as.character(packageVersion('blockCV')),positions=as.vector(t(coordinates)),predictionPositions=as.vector(t(prediction)),groups=x$group,groupFolds=g$folds_ids-1,nndm=list(radii=n$exclusion$exclusion_distance,training=unlist(lapply(n$folds_list,function(z)as.integer(seq_len(8)%in%z[[1]])))),knndm=list(folds=k$folds_ids-1,objective=k$W,candidates=as.vector(t(do.call(rbind,cand)))-1,candidateCount=length(cand)),environmental=list(features=feature,folds=e$folds_ids-1))
write_json(output,'modules/graph/test/gpu-spatial-analysis/spatial-regression/spatial-validation-advanced-reference.json',auto_unbox=TRUE,digits=16,pretty=TRUE)
