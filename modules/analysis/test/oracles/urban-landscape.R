# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Direct upstream references. Run from repository root with R 4.4, terra 1.9.50,
# landscapemetrics 2.2.1 and jsonlite; no implementation formulas are duplicated.
library(terra)
library(landscapemetrics)
library(jsonlite)
stopifnot(as.character(packageVersion('landscapemetrics')) == '2.2.1')
stopifnot(as.character(packageVersion('terra')) == '1.9.50')
make_case <- function(name, cells, width, height, spacing) {
  landscape <- rast(nrows=height, ncols=width, xmin=0, xmax=width*spacing[1],
                    ymin=0, ymax=height*spacing[2], crs='EPSG:3857')
  values(landscape) <- cells
  # Both classes form one connected patch in these fixtures; NA is explicit background.
  patches <- ifelse(is.na(cells), 4294967295, cells-1)
  measure <- function(operation, ...) {
    result <- operation(landscape, ...)
    result$value
  }
  list(name=name, width=width, height=height, cellSize=spacing, patches=patches,
       patchClasses=c(0,1),
       area=measure(lsm_p_area, directions=4)*10000,
       perimeter=measure(lsm_p_perim, directions=4),
       internalEdge=measure(lsm_c_te, count_boundary=FALSE, directions=4),
       boundaryEdge=measure(lsm_c_te, count_boundary=TRUE, directions=4),
       mesh=measure(lsm_c_mesh, directions=4)*10000,
       cohesion=measure(lsm_c_cohesion, directions=4),
       aggregation=measure(lsm_c_ai),
       contagion=measure(lsm_l_contag),
       landscapeMesh=measure(lsm_l_mesh, directions=4)*10000,
       landscapeAggregation=measure(lsm_l_ai, directions=4),
       landscapeCohesion=measure(lsm_l_cohesion, directions=4))
}
result <- list(versions=list(R=as.character(getRversion()),terra=as.character(packageVersion('terra')),
                            landscapemetrics=as.character(packageVersion('landscapemetrics'))),
               conventions=list(area='square projected units; upstream hectares multiplied by 10000',
                                connectivity=4,background='NA',edge='count_boundary TRUE versus FALSE'),
               cases=list(make_case('two-square-patches',c(1,1,2,2,1,1,2,2),4,2,c(1,1)),
                          make_case('interior-background',c(1,1,2,2,1,NA,2,2,1,1,2,2),4,3,c(1,1)),
                          make_case('single-cell-class',c(1,2,2,2),4,1,c(1,1))))
write_json(result,'modules/graph/test/oracles/fixtures/urban-landscape.json',auto_unbox=TRUE,pretty=TRUE,digits=16,na='null')
# Raster ENN is edge-cell CENTER distance. landscapemetrics 2.2.1 has no proximity API;
# the last column explicitly composes its patch area and ENN on exactly two patches.
landscape <- rast(nrows=5,ncols=9,xmin=0,xmax=9,ymin=0,ymax=5,crs='EPSG:3857')
values(landscape) <- rep(NA,45)
first <- c(1,2,3,10,11,12,19,20,21)
second <- c(7,8,9,16,17,18,25,26,27)
values(landscape)[c(first,second)] <- 1
patches <- rep(4294967295,45); patches[first] <- 0; patches[second] <- 1
area <- lsm_p_area(landscape,directions=4)$value*10000
nearest <- lsm_p_enn(landscape,directions=4)$value
result$proximity <- list(width=9,height=5,patches=patches,patchClasses=c(0,0),area=area,
 nearest=nearest,composedProximity=rev(area)/(nearest^2),
 coreEdge=lsm_p_core(landscape,directions=4,consider_boundary=FALSE,edge_depth=1)$value*10000,
 coreOpen=lsm_p_core(landscape,directions=4,consider_boundary=TRUE,edge_depth=1)$value*10000,
 convention='Direct ENN/core; proximity is composed, no lsm_p_prox exists in pinned package.')
write_json(result,'modules/graph/test/oracles/fixtures/urban-landscape.json',auto_unbox=TRUE,pretty=TRUE,digits=16,na='null')
