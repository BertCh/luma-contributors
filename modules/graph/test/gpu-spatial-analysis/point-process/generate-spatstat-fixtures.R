# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run with R4.4, spatstat.explore3.8-2, spatstat.linnet3.5-3, jsonlite2.0.0.
suppressPackageStartupMessages(library(spatstat.explore))
suppressPackageStartupMessages(library(spatstat.linnet))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(as.character(packageVersion('spatstat.explore')) == '3.8.2')
stopifnot(as.character(packageVersion('spatstat.linnet')) == '3.5.3')
planar=list()
for (withHole in c(FALSE,TRUE)) {
 outer=list(x=c(0,10,10,0),y=c(0,0,10,10))
 hole=list(x=c(4,4,6,6),y=c(4,6,6,4))
 window=owin(poly=if(withHole) list(outer,hole) else list(outer))
 coordinates=matrix(c(1,1,2,1,3,5,3,5.5,7,5,8,8),ncol=2,byrow=TRUE)
 pattern=ppp(coordinates[,1],coordinates[,2],window=window)
 intensity=c(.5,1,1.5,1.25,2,3);radii=seq(0,3,.5)
 k=Kinhom(pattern,lambda=intensity,r=radii,correction='border',renormalise=FALSE)$border
 density=as.numeric(density(pattern,sigma=2/sqrt(6),kernel='epanechnikov',at='points',edge=TRUE,leaveoneout=FALSE,dimyx=1024))
 planar[[length(planar)+1]]=list(hole=withHole,positions=coordinates,intensity=intensity,radii=radii[-1],k=k[-1],bandwidth=2,density=density,densityGrid=1024)
}
networks=list()
for (kind in c('branch','cycle','disconnected')) {
 if(kind=='branch') { coordinates=matrix(c(0,0,2,0,0,3,-4,0),ncol=2,byrow=TRUE);edges=matrix(c(1,2,1,3,1,4),ncol=2,byrow=TRUE);events=c(1,2,4) }
 if(kind=='cycle') {coordinates=matrix(c(0,0,2,0,2,2,0,2,4,2),ncol=2,byrow=TRUE);edges=matrix(c(1,2,2,3,3,4,4,1,3,5),ncol=2,byrow=TRUE);events=c(1,2,3,5)}
 if(kind=='disconnected') {coordinates=matrix(c(0,0,2,0,5,0,7,0),ncol=2,byrow=TRUE);edges=matrix(c(1,2,3,4),ncol=2,byrow=TRUE);events=c(1,2,3)}
 window=owin(range(coordinates[,1])+c(-1,1),range(coordinates[,2])+c(-1,1))
 vertices=ppp(coordinates[,1],coordinates[,2],window=window)
 network=linnet(vertices,edges=edges)
 pattern=lpp(data.frame(x=coordinates[events,1],y=coordinates[events,2]),network)
 nodes=lpp(data.frame(x=coordinates[,1],y=coordinates[,2]),network)
 distances=crossdist(pattern,nodes)
 lengths=sqrt(rowSums((coordinates[edges[,1],,drop=FALSE]-coordinates[edges[,2],,drop=FALSE])^2))
 intensity=seq_along(events)/2;radii=seq(0,10.5,.7)
 results=list()
 for(correction in c('none','ang')) for(normalization in c('network-length','inverse-intensity','inverse-intensity-squared')) {
  # spatstat names the uncorrected estimator "none" and geometric correction "Ang".
  value=linearKinhom(pattern,lambda=intensity,r=radii,correction=if(correction=='ang') 'Ang' else correction,normalise=normalization!='network-length',normpower=if(normalization=='inverse-intensity-squared') 2 else 1)$est
  results[[length(results)+1]]=list(correction=correction,normalization=normalization,k=value)
 }
 # Midpoint quadrature converges to an independently integrated continuous radial kernel.
 # This is mass-normalized shortest-path KDE, explicitly distinct from densityEqualSplit.
 queryCoordinates=matrix(numeric(),ncol=2);queryLengths=numeric();bandwidth=4
 for(edge in seq_len(nrow(edges))) {
  fraction=((1:64)-.5)/64
  queryCoordinates=rbind(queryCoordinates,outer(1-fraction,coordinates[edges[edge,1],])+outer(fraction,coordinates[edges[edge,2],]))
  queryLengths=c(queryLengths,rep(lengths[edge]/64,64))
 }
 queries=lpp(data.frame(x=queryCoordinates[,1],y=queryCoordinates[,2]),network)
 queryDistances=crossdist(queries,pattern)
 exactMass=sapply(seq_along(events),function(event)sum(sapply(seq_along(lengths),function(edge) {
  first=distances[event,edges[edge,1]];second=distances[event,edges[edge,2]];length=lengths[edge]
  integrate(function(position)pmax(0,1-pmin(first+position,second+length-position)/bandwidth)/bandwidth,0,length,subdivisions=1000,abs.tol=1e-10)$value
 })))
 kernel=pmax(1-queryDistances/bandwidth,0)/bandwidth
 density=rowSums(sweep(kernel,2,exactMass,'/'))
 kde=list(eventCount=length(events),distances=as.vector(t(queryDistances)),quadratureLengths=queryLengths,bandwidth=bandwidth,continuousMass=exactMass,continuousIntensity=density)
 networks[[length(networks)+1]]=list(name=kind,coordinates=coordinates,nodeCount=nrow(coordinates),edgeNodes=edges-1,edgeLengths=lengths,eventNodeIds=events-1,intensity=intensity,nodeDistances=as.vector(t(distances)),radii=radii,results=results,kde=kde)
}
# Shared pooled reference with leaveout=0 tests DCLF arithmetic; no GPU null calibration is inferred.
set.seed(150)
window=owin(c(0,10),c(0,10)); observed=runifpoint(12,window)
simulations=lapply(1:5,function(index)runifpoint(12,window))
envelopeResult=envelope(observed,Kest,nsim=5,simulate=simulations,r=seq(0,2,.25),correction='translate',savefuns=TRUE,verbose=FALSE)
inference=dclf.test(envelopeResult,leaveout=0)
curves=as.matrix(as.data.frame(attr(envelopeResult,'simfuns'))[,-1])
dclf=list(observed=envelopeResult$obs,simulations=as.vector(curves),reference=rowMeans(cbind(envelopeResult$obs,curves)),integrationWeights=rep(2/nrow(curves),nrow(curves)),simulationCount=ncol(curves),statistic=as.numeric(inference$statistic[1]),pValue=inference$p.value)
script=commandArgs(FALSE)[grepl('^--file=',commandArgs(FALSE))]; path=sub('^--file=','',script)
write_json(list(reference=list(R=as.character(getRversion()),explore=as.character(packageVersion('spatstat.explore')),linnet=as.character(packageVersion('spatstat.linnet')),density='density.ppp Epanechnikov points including self, image edge integral1024',K='Kinhom border renormaliseFALSE',network='linearKinhom all correction/normalise settings'),planar=planar,network=networks,dclf=dclf),file.path(dirname(path),'spatstat-fixtures.json'),pretty=TRUE,auto_unbox=TRUE,digits=16,na='null')
