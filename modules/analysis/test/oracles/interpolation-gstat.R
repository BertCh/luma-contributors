# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Offline independent gstat2.1-6 oracle. Requires interpolation-advanced.py output.
library(gstat)
library(jsonlite)
path <- 'modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/interpolation-advanced-reference.json'
reference <- fromJSON(path, simplifyVector=TRUE)
structureModel <- function(first, second) {
  model <- NULL
  for (index in seq_len(nrow(reference$lmc$structures))) {
    structure <- reference$lmc$structures[index,]
    loadings <- matrix(unlist(structure$loadings), nrow=3, byrow=TRUE)
    coefficients <- loadings %*% t(loadings)
    covariance <- structure$covariance
    name <- if(covariance$model == 'exponential') 'Exp' else 'Gau'
    range <- covariance$range / if(name == 'Exp') 3 else sqrt(3)
    model <- vgm(coefficients[first,second]*covariance$sill, name, range, add.to=model)
  }
  model <- vgm(if(first == second) .05 else 0, 'Err', 0, add.to=model)
  model
}
positions <- reference$lmc$positions
observations <- reference$lmc$observations
multivariate <- NULL
for(variable in 1:3) {
  rows <- observations[,3] == variable-1 & observations[,4]>0
  data <- data.frame(x=positions[rows,1],y=positions[rows,2],value=observations[rows,1])
  multivariate <- gstat(multivariate,id=paste0('variable',variable),formula=value~1,locations=~x+y,data=data,model=structureModel(variable,variable))
}
for(first in 1:2) for(second in (first+1):3) {
 multivariate <- gstat(multivariate,id=c(paste0('variable',first),paste0('variable',second)),model=structureModel(first,second))
}
support <- reference$lmc$supports[rep(1:3,c(2,3,5)),]
block <- data.frame(x=support[,1],y=support[,2])
# The support coordinates are relative to prediction origin(0,0), rational weights are repeated quadrature points.
result <- predict(multivariate, newdata=data.frame(x=0,y=0),block=block,debug.level=0)
# Separate univariate measurement-error block with the same discrete weighted support.
data <- data.frame(x=reference$positions[,1],y=reference$positions[,2],value=reference$values)
model <- vgm(1.7,'Exp',.8,add.to=vgm(.07,'Err',0))
univariate <- krige(value~1,locations=~x+y,data=data,newdata=data.frame(x=0,y=0),model=model,block=block,debug.level=0)
set.seed(19147)
simulation <- krige(value~1,locations=~x+y,data=data,newdata=data.frame(x=reference$queries[1:2,1],y=reference$queries[1:2,2]),model=model,beta=1.2,nsim=4096,debug.level=0)
ensemble <- as.matrix(simulation[,grep('^sim',names(simulation))])
output <- list(ensembleMean=rowMeans(ensemble),ensembleCovariance=cov(t(ensemble)),version=as.character(packageVersion('gstat')),rVersion=as.character(getRversion()),lmc=unname(c(result$variable3.pred,result$variable3.var)),block=unname(c(univariate$var1.pred,univariate$var1.var)))
write_json(output,'modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/interpolation-gstat-reference.json',auto_unbox=TRUE,pretty=TRUE,digits=16)
print(output)
