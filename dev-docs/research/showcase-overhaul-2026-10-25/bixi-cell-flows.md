# BIXI cell flows: conservation is not a route

Research and implementation plan, 2026-10-25. Parent assigned this follow-on to the atlas planner
and its existing Luna implementer. New files are isolated to `cells/bixi-cell-flows.*`; existing
station-flow scenes and source data remain read-only.

## Reader question

Do boundary flows account for every trip? Endpoint observations can be grouped into H3 cells,
but only trips whose endpoint cells are immediate neighbors belong to an H3 directed boundary
edge. Same-cell and nonneighbor pairs still contribute to outgoing and incoming cell totals.
No line or boundary in this story reconstructs a bicycle's traveled route.

## Primary sources and decisions

- [BIXI's official open-data page](https://bixi.com/en/open-data/) publishes the 2024 trip-history
  archive used by the existing August 2024 source. The investigation remains historical and
  identifies its retained subset; it does not imply current station availability or service.
- [H3 directed-edge documentation](https://h3geo.org/docs/api/uniedge/) defines a directed edge
  between neighboring cells, with distinct origin and destination identities. Thus a long-distance
  endpoint pair cannot be placed on one arbitrary cell edge. H3's boundary API is the independent
  reference for GPU-produced neighbor keys and coordinates.

## Source audit

The bundled `bixi-flows` monthly OD columns contain 139,295 different-station pairs with at least
three rides during August 2024. Their counts total **1,717,453 rides**. The source's cross-station
marginals total **1,873,846 rides**, leaving **156,393 rare cross-station rides** without an explicit
monthly retained OD pair. A further **54,555 same-station rides** produce the source total of
**1,928,401 rides**. The preprocessing also records 32,216 dropped short rides; they are outside
that retained source total.

Hourly rows preserve only frequent pairs; residual rows store a departure marginal and an arrival
marginal through a sentinel zone for rare pairs. Their sum double-represents those rare rides.
They cannot recover the missing origin–destination linkage. This investigation therefore uses
the known monthly OD subset, not summed hourly/residual rows. Its conservation statement always
names the retained subset. Different stations can lie in one H3 cell, so a same-cell pair is not a
same-station ride.

## Algorithm and visual plan

CPU preprocessing indexes actual station coordinates and sums the retained station-pair counts
by cell-pair key. This explicit aggregation bounds the recipe's direct scan without changing its
mass. `addCellFlowTopologyRecipe` then computes GPU outgoing/incoming totals, six directed neighbor
slots, exact neighbor and edge identities, actual shared-boundary coordinates, edge flows and
per-pair validity/status. The original pair table remains available for endpoint-link inspection.

The Montreal map supports an overview and a selected origin cell. In one view, its endpoint links
include nonneighbor destinations and same-cell mass. In another, only its genuine neighbor-boundary
flows are drawn from GPU output. Selecting one origin avoids ambiguous overlapping opposite-direction
strokes. Charts and a coverage ledger distinguish source exclusion from the recipe's valid-pair
filter. Changing resolution alters the spatial partition while conserving the same retained mass.

| Resolution | Station cells | Aggregated pairs | Same-cell rides | Neighbor rides | Nonneighbor rides |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 23 | 184 | 1,139,301 | 574,379 | 3,773 |
| 7 | 86 | 1,577 | 598,713 | 899,638 | 219,102 |
| 8 | 300 | 12,947 | 133,511 | 605,105 | 978,837 |

These are independent h3-js calculations over the shipped float32 locations and original integer
counts. They are references only, never runtime replacements for GPU output. At each resolution,
the three categories sum to 1,717,453. Default resolution seven shows all three categories clearly.

## Verification plan

Compare every GPU cell marginal, directed edge and neighbor key, boundary coordinate and edge
flow against a separate source-derived CPU reference at all three resolutions. Verify all valid
pair mass, both marginal sums and the same/neighbor/nonneighbor partition; preserve zero-flow
neighbors and distinguish them from invalid keys. Inspect the selected-cell controls and chart,
source coverage disclosure, and both map representations in production. Implementation is in
progress; no GPU or visual acceptance is claimed yet.

## Source review completed

The implemented scene now states the monthly three-ride threshold separately from the hourly
ten-ride threshold, reports omitted cross-station and same-station categories, and checks each
GPU marginal sum against valid input mass. Its first selection uses the highest outgoing GPU
cell; previous/next buttons provide keyboard selection. All genuine neighbor identities remain
visible, including zero-flow neighbors outside the station-cell catalog. Destination charts
name their top-18 truncation and omitted ride mass. Full H3 keys distinguish cells, and borough
labels explicitly describe the most common station borough rather than an H3 administrative
boundary. Independent references are recorded in `bixi-cell-flow-validation.json`.

The durable `check-bixi-cell-flows-reference.mjs` captures actual production readback and compares
all marginals, valid rows, neighbor/edge identities, edge flows and geographic boundary points
at resolutions 6/7/8. Its syntax and CPU references pass; production GPU/visual acceptance remains
pending the coordinated slot. No successful GPU result is inferred from source/type checks.

## Production acceptance

The frozen wave-eleven production build passed all three resolution cases. Every outgoing and
incoming marginal, retained valid row, directed-edge key, neighbor key and edge flow matched the
independent source-derived reference. Both global marginals equal 1,717,453 rides. Boundary
coordinates match h3-js geometry within 1.31 metres across all tested cells/edges, consistent with
the contributor's float32 geographic display precision. No browser or GPU errors occurred.

A separate interaction pass selected the next origin using its button, restored the previous
origin, and confirmed that the selected identity changed and returned correctly. Visual review
covered the neighbor-boundary map, the all-destination endpoint-link map, and the top-18 chart with
its excluded-destination total. The diagram remains explicitly endpoints rather than routes.
Artifacts: `/tmp/bixi-cell-flows-reference-wave11/`. Primary-readout reduction was source-only
after that build; it preserves all computations and readout definitions and awaits parent rebuild.
