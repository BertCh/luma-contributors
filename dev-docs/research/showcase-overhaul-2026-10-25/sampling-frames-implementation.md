# Sampling frames: implementation record

## Investigation

**Question:** What population can this sample represent?

The story in [`sampling-frames.scene.ts`](../../../examples/deck/spatial-analysis-showcase/src/scenes/points/sampling-frames.scene.ts) moves between three eligible frames: a finite Chicago facility inventory, the mapped boundary of one real Northerly Island park polygon, and the polygon's triangulated interior. A fourth mode runs independent uniform draws along that same boundary as a baseline. Seven short chapters show the frame, allocation, support, returned weights, replay key and failure flags.

The symbols use a quiet paper map, botanical green and copper. The finite map leaves the source inventory visible beneath selected facilities. Continuous primary, replacement and zero-weight legacy rows use distinct GPU-derived role colors; a thin line shows the mapped polygon boundary. The boundary is explicitly not an accessible walkway.

## Research basis

The EPA's [technical guide to `spsurvey`](https://usepa.github.io/spsurvey/articles/technical.html) distinguishes the target population from the list or GIS layer actually eligible for selection. It defines first-order inclusion probability `πᵢ`, inverse-probability design weight `1/πᵢ`, and the Horvitz–Thompson total. Its variance discussion depends on joint inclusion probabilities and their design-specific dependence. The [spatial-sampling guide](https://usepa.github.io/spsurvey/articles/sampling.html) describes GRTS for finite and continuous spatial resources, stratified allocation, legacy sites and replacement designs.

These references motivate the question, terms and caveats; this scene does not implement `spsurvey` or GRTS. `GPUSpatialBalancedSample` documents its own hierarchical circular-systematic finite design. `GPUContinuousSpatialSample` documents a random-shift arc-length or square lattice with random sample count. Its replacement waves are separate full designs. None of the output is presented as GRTS, an iid continuous sample, a minimum-distance result or a GRTS replacement list.

## Source frames and reproducibility

### Finite record frame

- Source: bundled `chicago-facilities` manifest and binary `position` / `category` columns. The source has 876 rows; category 1 contains 82 libraries and category 3 contains 92 fire stations (174 eligible rows total). The counts were recomputed from `category.bin` on 2026-10-25.
- Stable frame ID: original source row index + 1, returned by the GPU API and matched in a GPU gather pass. Filtering to the two categories does not renumber the public ID.
- Categories and projection follow [`site-meshes-data.ts`](../../../examples/deck/spatial-analysis-showcase/src/scenes/geometry/site-meshes-data.ts): category 1 / 3 and local projection origin `[-87.7, 41.86]`.
- The map and text describe a record inventory, not facility demand, service quality, access or coverage. It is not a facility census claim.
- Stratum mode supplies library / fire-station strata and equal allocations of 5 / 5 or 10 / 10. Pooled mode omits `strata`, so every eligible record is in API stratum 0. Sample counts are 10 or 20 and stay within the finite frame.
- Optional integer tickets are analyst-chosen. Equal tickets reproduce equal within-stratum probabilities. The unequal control assigns positive integer masses from original source-row parity, not the filtered frame index. The bundled rows contain 41 source rows of each parity among libraries and 46 among fire stations: library ticket totals 820 / 1,640 and fire-station totals 920 / 1,840 divide category allocations 5 / 10, and pooled totals 1,740 / 3,480 divide sample sizes 10 / 20. Ticket values are analyst-chosen, not measured importance.

### Northerly Island continuous frame

- Source: feature index 14 in bundled `chicago-parks/green-space.geojson`, OSM `osmId=2914325`, `name=Northerly Island`, `kind=park`, stored `areaKm2=0.428`.
- Geometry: one Polygon ring with 61 stored coordinates, including its repeated closing coordinate; this particular feature has no interior hole. The source ring topology is passed through `GPUPolygonTriangulation`, so any source holes in a Polygon supplied here remain excluded by the triangle union. The source area attribute is shown as provenance, not recomputed or claimed as the sampler's exact area.
- The same closed exterior is the connected line frame. Arc length is computed from the projected mapped boundary; this is not evidence of a public path or walkway.
- Local metric projection origin is `[-87.6, 41.86]`. The finite inventory and park map use separate Chicago and park cameras; they are not overlaid at a misleading shared scale.
- The default fixed hypothetical legacy site is at arc-length 0 on the line. In area mode it is the centroid of the first valid GPU-produced triangle, labelled synthetic. The `Include fixed legacy site` toggle adds or removes it. Included legacy rows have GPU weight 0 and role 2; their spacing neighborhoods are excluded, which changes the eligible continuous frame.

Source fingerprints (SHA-256, bundled artifacts as read during implementation):

| Artifact | SHA-256 |
| --- | --- |
| `chicago-parks/green-space.geojson` | `208a366c21329c75b41abb44c269033f787a5239478646e4bbc9f8a2cfc2dcc6` |
| `chicago-parks/manifest.json` | `1afec4e935468aa4333c2abef8b5764f0b8f3eb8350f82f495ee3d9137f5cd88` |
| `chicago-facilities/position.bin` | `2bf968e1dc5efe9c37ef1e51c7dc90ed113f1d4209e0237e4a2554f3b47b0e65` |
| `chicago-facilities/category.bin` | `29f634b858e56c6ef39dbd747116069add216c92fdc5e037d6c5079c47f2d173` |
| `chicago-facilities/names.csv` | `a7043c503f3301f61a7c35e80165e5ea5983c2e583c7f0e34119dd298d3656fc` |

## GPU contracts and displayed outputs

### `GPUSpatialBalancedSample`

Inputs are the 174 projected source positions, original stable row IDs, optional strata, positive ticket masses, finite bounds, and a resident `[seed, replicate]` parameter buffer. Allocation/sample-size changes compile the design; changing seed or replicate writes the resident buffer without replacing the compiled graph. The API returns selected IDs, `πᵢ`, `1/πᵢ` and per-stratum status on the GPU. A second GPU pass looks up selected positions and source categories for the map. The small readback exposes actual IDs and paired probabilities/weights. The displayed sum of weights is a realized Horvitz–Thompson estimate of the frame total, not guaranteed to equal `N` under unequal probabilities. No variance estimator is provided.

### `GPUContinuousSpatialSample`

Line mode uses cumulative arc length and area mode uses the valid triangle union from the GPU polygon triangulator. Spacing is 100, 200 or 400 m; returned inverse intensities have units m on the line and m² on area. Each complete wave writes positions, weights, `[source segment/triangle, stream replicate, candidate ordinal, role]` provenance and `[written, required-so-far, random accepted, flags]` status. A GPU role projection colors primary, replacement and legacy rows. The readout separates written rows (including zero-weight legacy rows) from random accepted rows, reports per-wave counts and flags, and shows up to eight provenance/positive-weight examples. It checks role, count and weight consistency before publishing any results.

One, two or three waves use consecutive independent replicate streams. Their weights are not pooled, and the scene makes no claim that selectively retaining a replacement after a failed wave is unbiased. An output capacity of 1 / 256 / 2,048 and candidate budget of 1 / 4,096 / 65,536 expose flags 1 (invalid input), 2 (capacity) and 4 (candidate budget). If any wave is incomplete, its sampled points and inferential weight examples are withheld; the boundary remains visible. After a candidate-budget stop, the readout says **required so far** and that the complete count is unknown.

### `GPURandomPointsOnLine`

The baseline consumes the exact same boundary vertices and one line offset row, declaring the projected park frame's spatial context. It returns a fixed 8 / 16 / 32 IID count, actual positions, line IDs, and normalized arc-length fractions from the GPU API. The readout includes the source line length and fractions. Recompiling with a larger count preserves the seeded prefix, as documented and covered by the contributor test. Independent draws can be close; no minimum Euclidean separation is claimed.

## Verification state

- The finite gather writes each category once, from the `axis == 0` invocation; invalid frame IDs keep their NaN coordinates and invalid category sentinel.
- A compiled graph runs only when newly built or after a resident seed/replicate write. Ordinary frames only flush a pending summary readback. Replay replaces and releases its reader, hides old output and clears selection until the current revision returns.
- Finite results require unique known source IDs, clean per-stratum status, finite positive probabilities and weights, and reciprocal consistency. Any invalid result is withheld from the title, readouts, map and tooltip.
- Continuous results report written rows, random accepted rows, legacy rows and per-wave flags/counts. Legacy rows have zero weight and are not included among random accepted points; weights across waves are never pooled.
- Scoped direct Biome check/format on `sampling-frames.compute.ts` and `sampling-frames.scene.ts`: passed.
- `node scripts/check-cartography.mjs --strict points`: passed. `node scripts/check-story-steps.mjs points`: passed. `node scripts/check-wgsl-identifiers.mjs src/scenes/points`: passed (490 templates scanned).
- No full project build, install, GPU execution or browser render was run in this focused review. The source and metadata checks pass; GPU acceptance and rendered behavior remain unverified.
