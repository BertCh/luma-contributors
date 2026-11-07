# Information architecture and consolidation review

2026-10-25. This review considers the growing directory against the actual questions, data and
contributor variants in the source metadata. Reusing a contributor is not sufficient evidence
that two investigations are redundant. The user authorized removal and restructuring, but a
smaller count is not itself a quality improvement.

## Strongest consolidation candidate

`movement/vessel-encounters` currently combines shared-clock encounter detection across a large
harbor fleet with whole-route similarity in seven steps. The newer focused sequence
`harbor-track-quality → harbor-continuous-encounters → harbor-track-similarity` provides a stronger
progression: establish observation support, compare sampled and continuous timing, then choose a
shape-comparison question. Its bounded cases have independently checkable consequences, whereas
the old entry asks two different questions under one broad title.

Recommend retiring `vessel-encounters` as an independent directory entry **after** its distinct
whole-track `GPUTrackSimilarity` Hausdorff/Fréchet comparison and fleet-selection behavior have
been preserved as an explicitly labeled comparison in the focused similarity study. The new
`GPUTrackSimilaritySearch` is not automatically a replacement for the old all-pairs operation.
Keep `harbor-playback`: its playhead, temporal trails and equal-distance versus equal-time sampling
are distinct introductory behavior, not merely a weaker version of quality control. Retain an
old-route redirect if the original story ID is removed. Do not retire the old implementation
before the focused studies pass production checks and the contributor-preservation check.

Review outcome: the movement planner retained the distinct 897-track fleet investigation and
narrowed its public question to “Which routes resemble a selected vessel across the fleet?”
It now explicitly links to observation quality, continuous encounters and bounded similarity
search. This preserves the unique full-pairwise behavior without leaving the earlier combined
coordination/accuracy interpretation. The focused similarity study has passed 25 production
numerical cases; continuous-encounter and quality-covariance extensions still await their latest
production checks, so the new three-stop harbor route is not yet featured.

## Trim an omnibus rather than delete its useful core

`geometry/line-operations` currently jumps from CTA densification, chunking and linear referencing
to Chicago place-to-street inference and then AIS simplification/smoothing. The new
`route-footpoints` and `route-editing` are better homes for the respective interpretation problems.
Recommend narrowing the original to a clearly named investigation of position along a route,
retaining `GPULineSegmentize`, `GPULineChunk` and `GPULinearReferencing`. Move the nearest-street
inference chapter into `route-footpoints` only after its `GPULineLocate` behavior is matched.
Move the two AIS thinning/smoothing chapters into the harbor shape-comparison sequence only if
both `GPULineSimplification` and `GPULineSmooth` remain visible operations with before/after
outputs. Kalman filtering is not a substitute for geometric smoothing. This yields a coherent
shorter original instead of hiding unique functionality in a large new editing toolbox.

`geometry/buffers-and-shapes` is another mixed workshop: straight-line reach, sectors, grids,
Hilbert ordering and rectangle clipping. Its rectangle-window chapter can become the explicit
coarse-window baseline in `route-windowing`, contrasting a box with actual Loop membership.
Keep the grid generation and Hilbert ordering until there is a focused address/order study;
H3 topology does not replace their planar sorting behavior. A wholesale deletion is not justified.

## False-positive duplicates to retain

| Existing and newer studies | Why both remain meaningful |
| --- | --- |
| `airline-network` / `aviation-network-lab` | Global historical OpenFlights graph algorithms, spatial/force layouts and edge bundling differ from directed July 2023 regional flight-flow brokerage and Louvain analysis. |
| `point-patterns` / `observation-hierarchy` | Conditional random-pattern testing and study-window support differ from density-cluster extraction and hierarchy. |
| `nature-clusters` / `nature-clusters-outliers` / `observation-hierarchy` | DBSCAN/k-means/hulls, local outlier density and OPTICS/HDBSCAN answer different questions; no single score replaces them. |
| `lidar-ground` / `point-cloud-support` / `point-cloud-change` | Local terrain features, retained support/partition ownership, and real repeated-survey change are distinct measurements. |
| `terrain-basics` / `terrain-conditioning` / `watersheds` | Terrain decoding/derivatives, conditioning sensitivity, and downstream drainage structure are not interchangeable. |
| `taxi-flows` / `nyc-taxi-tides` / `nyc-taxi-density` | Zone-to-zone Chicago flows, signed NYC arrival/departure imbalance, and camera-grid pickup/dropoff density have different denominators and cartographic questions. |
| `polygon-measures` / `boundary-forensics` / `spherical-boundaries` | Measurement/shape validity, canonical representation/precision, and spherical boundary support should be navigational neighbors, not merged into an unmanageable form. |

The legacy ID `neural-growth` is semantically misleading: its source now clearly implements
residual diffusion cloud repair, not a trained neural model. Its actual question and observations
are strong, so rename the public title around what an older image can support under cloud; do
not discard the unique diffusion experiment. An ID migration would need a redirect.

## Navigation at the larger scale

Keep the full searchable directory as the exhaustive inventory and the reference catalog as the
contributor-oriented view. The seven current three-stop inquiries are an editorial entry, not an
attempt to put every algorithm on the home page. Their data now lives in `shell/atlas-routes.ts`,
with a dependency-free next-stop helper for the story completion card. This prevents a dead end
without importing the scene registry or computation code.

As focused harbor and geometry studies gain production acceptance, add a compact harbor inquiry
and a representation/precision inquiry, rather than appending every new scene to an existing
route. Keep routes at three stops. Longer lists would reproduce the directory in miniature.
Dataset and method search already surface the additional Meuse, raster and county laboratories;
there is no need to hide them merely because they are not featured.

No source story has been deleted in this review. The recommendations above identify exact
migration requirements so removal can preserve meaningful behavior and existing links.
