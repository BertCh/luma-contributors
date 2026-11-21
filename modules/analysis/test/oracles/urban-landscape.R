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
for (proximity_values in list(result$proximity$area, result$proximity$nearest,
                             result$proximity$composedProximity, result$proximity$coreEdge,
                             result$proximity$coreOpen)) {
  stopifnot(length(proximity_values) == 2, all(is.finite(proximity_values)))
}

# URB-105: every input cell starts as a separate tile. Establish components by an
# independent flood fill, then map upstream patch IDs by their actual member cells
# (class IDs alone do not identify a component). Keep this separate from cases:
# existing four-direction consumers assume their original two-class schema.
tile_ids <- 0:8
tile_classes <- c(0,1,2,1,0,2,2,2,0)
tile_roots <- rep(NA_integer_, 9)
for (start in seq_along(tile_ids)) {
  if (!is.na(tile_roots[start])) next
  pending <- start
  tile_roots[start] <- tile_ids[start]
  while (length(pending) > 0) {
    current <- pending[1]
    pending <- pending[-1]
    current_row <- (current-1) %/% 3
    current_column <- (current-1) %% 3
    for (other in seq_along(tile_ids)) {
      if (!is.na(tile_roots[other]) || tile_classes[other] != tile_classes[current]) next
      if (max(abs((other-1) %/% 3-current_row), abs((other-1) %% 3-current_column)) != 1) next
      tile_roots[other] <- tile_ids[start]
      pending <- c(pending, other)
    }
  }
}
stopifnot(identical(tile_roots, c(0L,1L,2L,1L,0L,2L,2L,2L,0L)))
tile_landscape <- rast(nrows=3,ncols=3,xmin=0,xmax=6,ymin=0,ymax=9,crs='EPSG:3857')
# Shift class labels so class zero cannot be mistaken for background upstream.
values(tile_landscape) <- tile_classes+1
upstream_patches <- get_patches(tile_landscape, directions=8)[[1]]
patch_mapping <- list()
for (class_patches in upstream_patches) {
  labels <- as.vector(values(class_patches))
  for (patch_id in sort(unique(labels[!is.na(labels)]))) {
    members <- which(!is.na(labels) & labels == patch_id)
    roots <- unique(tile_roots[members])
    classes <- unique(tile_classes[members])
    stopifnot(length(roots) == 1, length(classes) == 1)
    patch_mapping[[length(patch_mapping)+1]] <- data.frame(
      class=classes+1, id=patch_id, root=roots)
  }
}
patch_mapping <- do.call(rbind, patch_mapping)
tile_area <- lsm_p_area(tile_landscape,directions=8)
tile_perimeter <- lsm_p_perim(tile_landscape,directions=8)
tile_metrics <- merge(patch_mapping, tile_area[c('class','id','value')], by=c('class','id'))
names(tile_metrics)[names(tile_metrics) == 'value'] <- 'area'
tile_metrics <- merge(tile_metrics, tile_perimeter[c('class','id','value')], by=c('class','id'))
tile_metrics <- tile_metrics[order(tile_metrics$root),]
stopifnot(identical(as.integer(tile_metrics$root), 0:2))
stopifnot(isTRUE(all.equal(tile_metrics$area*10000, c(18,12,24), tolerance=1e-10)))
result$tileReconciliation <- list(
  width=3,height=3,inputTileIds=tile_ids,classes=tile_classes,cellSize=c(2,3),
  connectivity=8,roots=tile_roots,patchIdToInputRoot=patch_mapping,
  area=tile_metrics$area*10000,perimeter=tile_metrics$value,
  handCounted=list(area=c(18,12,24),perimeter=c(30,20,30)),
  perimeterMatchesPhysicalSides=isTRUE(all.equal(tile_metrics$value,c(30,20,30),tolerance=1e-10)),
  conventions=list(area='square projected units; upstream hectares multiplied by 10000',
    perimeter='projected length; compare upstream with exposed physical sides, not diagonal connectivity edges',
    components='eight-neighbour same-class flood fill; root is minimum input tile ID',
    upstream='get_patches membership maps (class,id) to input roots; lsm_p_area/lsm_p_perim directions=8'))
write_json(result,'modules/analysis/test/oracles/fixtures/urban-landscape.json',auto_unbox=TRUE,pretty=TRUE,digits=16,na='null')
