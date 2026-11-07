# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Direct terra focalPairs/focalReg references. Conda: R4.4, r-terra=1.9_50, r-jsonlite.
suppressPackageStartupMessages(library(terra))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(as.character(packageVersion('terra')) == '1.9.50')
width <- 7L; height <- 5L; radius <- 2L
predictor <- (0:(width*height-1) %% width - 3) + (0:(width*height-1) %/% width)*0.25
response <- 2 + 1.5*predictor + (0:(width*height-1) %% 3)*0.125
predictor[c(3,13)] <- NA_real_; response[18] <- NA_real_
cases <- list()
for (border in c('nodata','clamp','reflect')) {
  for (policy in c('ignore','propagate')) {
    padded_width <- width; padded_height <- height
    first <- predictor; second <- response
    if (border != 'nodata') {
      project <- function(coordinates, size) {
        if (border == 'clamp') return(pmax(0,pmin(size-1,coordinates)))
        folded <- coordinates %% (2*(size-1)); pmin(folded,2*(size-1)-folded)
      }
      columns <- project((-radius):(width+radius-1),width)
      rows <- project((-radius):(height+radius-1),height)
      select <- as.vector(t(outer(rows,columns,function(row,column) row*width+column+1)))
      first <- predictor[select]; second <- response[select]
      padded_width <- length(columns); padded_height <- length(rows)
    }
    raster <- rast(ncols=padded_width,nrows=padded_height,nlyrs=2,xmin=0,xmax=padded_width,ymin=0,ymax=padded_height,crs='')
    values(raster) <- cbind(first,second)
    narm <- policy == 'ignore'
    correlation <- focalPairs(raster,w=2*radius+1,fun='pearson',na.rm=narm)
    covariance <- focalPairs(raster,w=2*radius+1,fun=function(first,second) stats::cov(first,second,use=if(narm)'pairwise.complete.obs' else 'everything'))
    support <- focalPairs(raster,w=2*radius+1,fun=function(first,second) sum(complete.cases(first,second)))
    # focalReg's first band is response; the following band is predictor.
    fit <- focalReg(raster[[c(2,1)]],w=2*radius+1,fun='ols',na.rm=narm,intercept=TRUE)
    select <- seq_len(width*height)
    if(border!='nodata') select <- as.vector(t(outer(0:(height-1),0:(width-1),function(row,column) (row+radius)*padded_width+column+radius+1)))
    cases[[length(cases)+1]] <- list(name=paste(border,policy,sep='-'),width=width,height=height,radius=radius,borderMode=border,noDataPolicy=policy,covarianceNormalization='sample',predictor=list(values=predictor),response=list(values=response),expected=list(covariance=values(covariance)[select,1],correlation=values(correlation)[select,1],pairedCount=values(support)[select,1],intercept=values(fit)[select,1],slope=values(fit)[select,2]))
  }
}
# Temporal composites use the actual terra stack algebra with original QA gaps.
stack_values <- matrix(c(1,3,NA,9,2,7,4,NA,5,0,8,2),nrow=3,byrow=TRUE)
stack <- rast(ncols=3,nrows=1,nlyrs=4,xmin=0,xmax=3,ymin=0,ymax=1,crs='')
values(stack) <- stack_values
composites <- lapply(c('mean','min','max','median'), function(operation) as.vector(values(app(stack,operation,na.rm=TRUE))))
names(composites) <- c('mean','minimum','maximum','median')
result <- list(versions=list(R=as.character(getRversion()),terra=as.character(packageVersion('terra')),jsonlite=as.character(packageVersion('jsonlite'))),cases=cases,temporal=list(cellCount=3,sliceCount=4,values=as.vector(t(stack_values)),expected=composites))
write_json(result,'modules/analysis/test/oracles/fixtures/raster-terra.json',pretty=TRUE,auto_unbox=TRUE,na='null',digits=16)
print(result$versions)
