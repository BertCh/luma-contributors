# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Pinned R4.4.3, spatstat.geom3.8.2, spatstat.explore3.8.2, jsonlite2.0.0.
suppressPackageStartupMessages(library(spatstat.explore))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(as.character(getRversion()) == '4.4.3', as.character(packageVersion('spatstat.explore')) == '3.8.2')
spatstat.options(npixel=1024, maxedgewt=1e30)
ring=function(coordinates)list(x=coordinates[,1],y=coordinates[,2])
outer=matrix(c(0,0,10,0,10,10,0,10),ncol=2,byrow=TRUE)
hole=matrix(c(4,4,4,6,6,6,6,4),ncol=2,byrow=TRUE)
concave=matrix(c(0,0,10,0,10,3,3,3,3,10,0,10),ncol=2,byrow=TRUE)
component=matrix(c(12,0,16,0,16,4,12,4),ncol=2,byrow=TRUE)
fixtures=list()
for (name in c('rectangle','hole','concave','components')) {
 rings=switch(name,rectangle=list(outer),hole=list(outer,hole),concave=list(concave),components=list(outer,component))
 window=owin(poly=lapply(rings,ring))
 positions=matrix(c(.8,.9,1.3,1.4,2.2,.7,2.5,2.1,.9,2.8,1.51,3.31,7.2,1.4,7.5,2.2),ncol=2,byrow=TRUE)
 if(name=='hole') positions=rbind(positions,c(3.8,4.3),c(3.7,5.4),c(6.4,5.2),c(6.6,6.2))
 if(name=='components')positions=rbind(positions,c(12.4,1.2),c(13.2,1.8))
 pattern=ppp(positions[,1],positions[,2],window=window)
 radii=seq(0,3,.25);halfWidth=.7;densityBandwidth=2.5
 modes=list()
 for(mode in c('supplied','include','exclude')) {
  intensity=if(mode=='supplied') seq_len(nrow(positions))/10+.2 else as.numeric(density(pattern,sigma=densityBandwidth/sqrt(6),kernel='epanechnikov',at='points',edge=TRUE,leaveoneout=mode=='exclude',dimyx=1024))
  stopifnot(all(intensity>0))
  distances=pairdist(pattern)
  overlap=area(window)/edge.Trans(pattern,exact=TRUE,trim=Inf)
  weights=1/outer(intensity,intensity)/overlap;diag(weights)=0
  exactK=sapply(radii,function(radius)sum(weights[distances<=radius]))
  directK=Kinhom(pattern,lambda=intensity,r=seq(0,3,.001),correction='translate',renormalise=FALSE)$trans[round(radii/.001)+1]
  correlations=list()
  for(divisor in c('radius','distance'))for(zero in c('none','reflection')) {
   direct=pcfinhom(pattern,lambda=intensity,r=radii,kernel='epanechnikov',h=halfWidth,correction='translate',divisor=if(divisor=='radius')'r' else 'd',zerocor=zero,fast=FALSE,renormalise=FALSE)$trans
   exact=sapply(radii[-1],function(radius){
    kernel=function(delta).75/halfWidth*pmax(0,1-(delta/halfWidth)^2)
    numerator=kernel(radius-distances)
    if(zero=='reflection')numerator=numerator+kernel(radius+distances)
    denominator=if(divisor=='radius')matrix(radius,nrow(positions),nrow(positions))else distances
    diag(denominator)=1
    sum(weights*numerator/denominator)/(2*pi)
   })
   correlations[[length(correlations)+1]]=list(divisor=divisor,zeroCorrection=zero,direct=direct[-1],exact=exact)
  }
  modes[[length(modes)+1]]=list(mode=mode,intensity=intensity,k=exactK[-1],directK=directK[-1],correlations=correlations)
 }
 vertices=do.call(rbind,lapply(rings,function(r)rbind(r,r[1,])))
 offsets=c(0,cumsum(sapply(rings,function(r)nrow(r)+1)))
 fixtures[[length(fixtures)+1]]=list(name=name,overlap=as.vector(t(overlap)),vertices=vertices,ringOffsets=offsets,polygonOffsets=if(name=='components')c(0,1,2)else c(0,length(rings)),positions=positions,radii=radii[-1],bandwidth=densityBandwidth,pairBandwidth=halfWidth,modes=modes)
}
script=commandArgs(FALSE)[grepl('^--file=',commandArgs(FALSE))]
write_json(list(reference=list(R=as.character(getRversion()),explore=as.character(packageVersion('spatstat.explore')),geom=as.character(packageVersion('spatstat.geom')),direct='Kinhom/pcfinhom translation renormaliseFALSE; polygon setcov1024; pcfinhom fastFALSE',exact='edge.Trans exactTRUE polygon clipping plus explicit radial kernel sum; composed reference',halfWidth='Epanechnikov h, standard deviation h/sqrt5'),fixtures=fixtures),file.path(dirname(sub('^--file=','',script)),'translation-pcf-fixtures.json'),auto_unbox=TRUE,pretty=TRUE,digits=16,na='null')
