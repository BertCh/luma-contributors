# web-g1 (saved by the lead from the agent's final message; the agent wrote no report file)

Groups: cells-index-join, dataframe-pointcloud-records, density-interpolation-regions, geometry-construction-overlay.
Verified one run at a time (not whole groups) into `website/.reports/w1-*`; website typecheck clean for these groups.

## Inert sliders
- datum-grid-transform maximumIterations: default now NAD83 to NAD27 (inverse direction iterates), tolerance 3e-9, range 1-4, default 3; early-stopped locations ringed. Slider change 2.8%.
- pointcloud-ground slope / maximumThreshold and pointcloud-smrf window: a CONSTANT HILLSIDE TILT (0.17) was added to the lidar tile, stated in the caption. Constructed scenario, not measured data. 1.1%, 1.5%, 0.9%.

## Gestures (max % before -> after, distinct pictures)
- point-in-polygon-join: 9x5 probe lattice through the real join; 1.6 -> 3.9; 2 -> 5
- pointcloud-statistical-filter: x picks neighbour capacity; 1.3 -> 1.4; 2 -> 3
- group-concave-hull, concave-hull: reach picks maximumCircumradius; 0.5 -> 4.0, 0.6 -> 3.5
- constrained-delaunay-triangulation: level picks maximumFlips; 1.7 -> 13.3; distinct 4 -> 3
- linear-referencing: reach = radius; 0.3 -> 2.3
- nearest-feature-join: reach = radius; 1.3 -> 3.6
- coverage-validity: gapWidth = distance to nearest edge; 0.6 -> 3.0
- variogram-model-fit: pairs of hovered lag drawn; 0.8 -> 4.8
- point-to-cell 0.5 -> 4.3; cell-geometry 0.6 -> 2.9 (default resolution 4)
- spherical-convex-contains 2.1 -> 4.9; spherical-polygon-contains 1.3 -> 2.3 (defaults, larger dots)
- Thicker strokes or larger dots only: line-chunk 1.9 -> 3.3, line-smooth 1.9 -> 2.7, line-locate 2.9 -> 3.7, line-segmentize 0.7 -> 1.9, group-statistics 2.5 -> 3.7, region-mask 2.9 -> 5.2, pick-region-mask 2.7 -> 4.1, point-pattern-envelope 2.6 -> 4.0, building-tessellation-recipe 2.2 -> 2.6, group-convex-hull 1.1 -> 1.6

## Left
datum-grid-transform gesture (0.5%), polygon-triangulation, spherical-convex-intersection, spherical-arc-overlay (outline only; needs triangulated fill), hilbert-keys, ward-regions, transition-diagnostics.

## Posters to regenerate
datum-grid-transform, pointcloud-ground, pointcloud-smrf, pointcloud-statistical-filter, point-in-polygon-join, concave-hull, group-concave-hull, constrained-delaunay-triangulation, linear-referencing, coverage-validity, variogram-model-fit, point-to-cell, cell-geometry, nearest-feature-join, spherical-convex-contains, spherical-polygon-contains, line-chunk, line-smooth, line-locate, line-segmentize, group-statistics, region-mask, pick-region-mask, point-pattern-envelope, building-tessellation-recipe, group-convex-hull.

## Notes
- biome reformatted geographic-distribution.ts, its parameters file and group-geometry.ts (session c0 was editing those; formatting only).
- New file: website/src/gpu/runs/entries/geometry-construction-overlay/shared/reach.ts
