# Point-cloud support and measured surface change

Research and implementation record, 2026-10-25. Astra planning/review with the existing Luna
implementer; shared checkout, no separate install or package-wide build. The parent owns final
formatting, build and coordinated GPU validation.

## Audit and question

The existing `terrain/lidar-ground` story already covers neighborhoods, statistical filtering,
local features, ground/height, surface fitting and voxel downsampling on real King County lidar.
The audit found no direct showcase calls to `GPUPointCloudRadiusFilter`, `GPUPointCloudSMRF`,
`GPUPointCloudOctree`, `GPUPointCloudVoxelMerge`, `GPUPointCloudICP` or `GPUPointCloudChange`.
Adding those names to a contributor list would not explain their consequences. Two related
studies instead make support, retained ownership and registration assumptions inspectable.

1. **How much support does a point cloud have?** A bounded 2,000-original-row sample of the
   existing USGS data moves from exact-radius support to morphological ground classification,
   additive octree ownership, then the conservation of voxel sufficient statistics.
2. **When does a moving surface exceed measurement uncertainty?** Real repeated terrestrial
   laser scans of an Austrian rock glacier move from observed epochs to normal-directed
   displacement, support/uncertainty sensitivity and a deliberately questionable local ICP fit.

The first is a scanner/partition laboratory with categorical ownership views. The second is a
survey comparison with a fixed diverging displacement scale and explicit detection intervals.
Neither interprets algorithmic classification as independently verified ground truth.

## Primary sources and design consequences

- [PDAL SMRF documentation](https://pdal.io/en/stable/stages/filters.smrf.html) identifies the
  morphology controls, terrain raster scale and ground-classification context. The showcase
  exposes window and slope, names its fixed 2 m grid, and reports unsupported/inpainted rows.
- [PDAL ICP documentation](https://pdal.io/en/stable/stages/filters.icp.html) documents iterative
  closest-point registration and convergence controls. A lower fitting residual is evidence of
  closer local correspondence, not proof that the surveyed moving surface should be realigned.
- [The authors' py4dgeo M3C2 documentation](https://py4dgeo.readthedocs.io/en/latest/m3c2.html)
  motivates normal-oriented cylindrical comparison at core points. The showcase must report
  support and uncertainty alongside signed distances.
- [The authors' M3C2-EP documentation](https://py4dgeo.readthedocs.io/en/latest/m3c2ep.html)
  gives the affine transform about a reduction point and identifies the richer scanner and
  registration covariance information used by the published method. Preparation applies the
  supplied transform once; the showcase's simpler empirical roughness/registration-error
  calculation is explicitly not a reproduction of the published M3C2-EP result.
- [heiDATA source dataset](https://doi.org/10.11588/data/XHYB10), also inspected through its
  [Dataverse API](https://heidata.uni-heidelberg.de/api/datasets/:persistentId/?persistentId=doi:10.11588/DATA/XHYB10),
  describes genuine 2017/2018 repeated surveys at Äußeres Hochebenkar and stable-area alignment.
  Retain the survey frame by default; fitting only this moving crop is a sensitivity experiment.
- [Author-maintained py4dgeo test data](https://github.com/3dgeo-heidelberg/py4dgeo-test-data)
  provides the bounded original LAZ files, core normals, transform, reduction origin and license.
  Its release archive is small enough to prepare reproducibly without bundling full survey tiles.

The public [USGS repeat bathymetric/topographic comparison dataset](https://www.usgs.gov/data/inputs-and-outputs-3d-point-cloud-comparisons-m3c2-repeat-bathymetric-and-topographic-surveys)
was also considered. The Heidelberg crop was selected because its corresponding epochs,
orientation data and registration parameters are packaged together in a manageable primary source.
No simulated displacement is needed.

## Dataset preparation and provenance

Source attribution: Winiwarter, Anders, Zahs, Hämmerle and Höfle (2021), *M3C2-EP: Pushing the
limits of 3D topographic point cloud change detection by error propagation [Data and Source Code]*,
heiDATA V1, DOI 10.11588/data/XHYB10, CC BY 4.0. Original license text ships alongside the columns.

Pinned archive: release `2026-01-29`, SHA-256
`f4c2edb52a6e2c40206ca07782c4b70e450742fb60aa48478674e776e6a33651`.
Only explicitly named archive members are read; extraction never writes arbitrary archive paths.
The preparation script keeps the first original LAS row in each occupied 0.6 m 3D voxel and
every eighth provided core point. It restores source row ordering and ships source row indices.

| Observations | Full supplied crop | Shipped original rows |
| --- | ---: | ---: |
| 2017 reference | 1,195,012 | 5,192 |
| 2018A comparison | 1,324,416 | 5,407 |
| Core points with provided normals | 2,064 | 258 |

Binary columns total 350,520 bytes. The data builder records full input/output checksums,
native bounds, the transformation and sampling policy. Computational XYZ is in native metres
relative to `[652920, 5189155, 2485]`; this common translation retains normal orientation and
float32 precision. The provider identifies UTM32N but does not fully identify the sample's datum.
EPSG:32632 is assumed only for map placement. The computation never uses projected map XY as
the native normal frame. Exact acquisition-day separation is not established, so report metres,
not annual velocity. Sampling changes roughness and cylinder support.

Three retained source normals are nonfinite (sample indices 117, 118, 184). Preserve them and
count invalid orientations separately from insufficient cylinder support; do not fabricate a
normal. The remaining 255 are unit length to float32 precision.

## Implementation review requirements

Support study: exact all-pairs radius work is bounded to 2,000 rows. The radius mask controls
SMRF selection. Octree node records yield per-level owned counts; overflow invalidates the
hierarchy rather than yielding plausible-looking partial results. Two disjoint row partitions
produce voxel counts, centroids and representatives; a separate full-sample GPU reduction checks
the merged count conservation and per-cell centroid difference. Representative offsets must
refer to the original shared sample. Asynchronous readback must request layers when validity
gates become available. Legends must match all seven source LAS class slots, and amber SMRF
status must include edge/inpainted support rather than describing every flagged row as invalid.

Change study: keep analysis in native metric axes, render separate geographic positions, use
signed comparison-minus-reference distances and fixed display limits. Invalid support is neutral,
not zero movement. Default radius 1.5 m, half-depth 3 m, minimum support 3, registration standard
error 0.03 m and 1.96 multiplier. Linear versus quadrature combination is a visible assumption.
Local ICP must show actual convergence history/status and reject invalid transforms; fitting a
moving patch can remove real motion. The surveyed frame remains the primary interpretation.

## Validation

`scripts/data/ahk-rock-glacier/validate.py` is an independent NumPy float64 reference over the
bundled float32 columns, not a transcription of GPU readback. It normalizes finite supplied
normals, selects each cylindrical neighborhood, computes sample variance and evaluates the
linear detection threshold. Results are saved in `pointcloud-data-validation.json`.
The same script also computes a separate float64 SVD rigid-fit reference on 1,024 systematically
sampled rows per epoch for unweighted, Huber and Tukey ICP. With a 2 m correspondence bound and
12 iterations, all three still make millimetre-to-centimetre updates. An exhausted iteration
budget therefore remains a clearly labeled provisional fit; insufficient matches or a failed
solve invalidate change interpretation. The durable browser checker is
`scripts/check-pointcloud-reference.mjs`, which compares three radii, registration and uncertainty
sensitivities, and all three weighting choices against a stable production preview.

| Radius | Valid cores | Insufficient support | Invalid normal | Detectable difference | Median signed difference | Median detection limit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8 m | 220 | 35 | 3 | 140 | +0.273777 m | 0.192049 m |
| 1.5 m | 251 | 4 | 3 | 181 | +0.281966 m | 0.154629 m |
| 2.0 m | 251 | 4 | 3 | 184 | +0.265066 m | 0.137114 m |

These numbers are reference checks for this bounded sample and chosen assumptions, not the
published study's findings. Runtime UI values must come from GPU outputs. Preparation rerun
and CPU reference completed successfully. GPU comparison, scene controls, overflow/error
handling and final formatted type/build checks remain pending coordinated implementation QA.

The first real WebGPU support review caught a graph ownership error: each partition buffer was
imported a second time under a merge-specific name. The fix reuses each original GraphDataView
across both contributors. The subsequent source-server run produced 2,000 octree-owned rows,
2,000 merged rows matching the direct reduction and zero uncaptured GPU errors. At the original
3 m radius only 49 rows met the support rule; this motivated a 6 m initial support radius and
larger marks on a dark scanner ground. Source HMR briefly restarted one step, so this run is
diagnostic evidence rather than the final production acceptance sweep. Artifacts are in
`/tmp/pointcloud-support-qa/`. The bundled dataset integrity checker reported zero errors after
adding all binary SHA-256 values to `manifest.properties.checksums`.

## Owned files

- `examples/deck/spatial-analysis-showcase/src/scenes/terrain/point-cloud-support.{scene,compute}.ts`
- `examples/deck/spatial-analysis-showcase/src/scenes/terrain/point-cloud-change.{scene,compute}.ts`
- `examples/deck/spatial-analysis-showcase/src/data/datasets/ahk-rock-glacier.dataset.ts`
- `examples/deck/spatial-analysis-showcase/scripts/data/ahk-rock-glacier/{build.py,validate.py,README.md}`
- `examples/deck/spatial-analysis-showcase/public/data/ahk-rock-glacier/`
- This research record and `pointcloud-data-validation.json`.

## Production acceptance of the first two studies

Stable production preview on port 5313 passed all nine story steps with zero browser or GPU
errors. The independent numerical checker passed eight scenarios: three support radii, a larger
registration-error allowance, quadrature combination, and three ICP weighting choices. At the
default 1.5 m cylinder radius, the GPU returned 251 valid cores, four insufficient-support cores,
three invalid normals and 181 intervals excluding zero. The displayed median +0.282 m difference
and 0.155 m limit agree with the independent reference within display rounding. Raising assumed
registration error to 0.2 m reduced detected intervals to 57; quadrature at 0.03 m yielded 197.

On the moving crop, 12-iteration local ICP changed the median from +0.282 m to +0.008 m
(unweighted), +0.021 m (Huber) and +0.152 m (Tukey). The output remains explicitly provisional.
This is a substantive demonstration of fitting away motion, not proof of improved registration.
Final translation steps agree with the independent SVD reference within 0.005 m. Screenshots
of source returns, signed differences, support hierarchy and ICP diagnostics were visually reviewed.
Artifacts: `/tmp/pointcloud-reference/` and `/tmp/pointcloud-production/`. The shared maximum zoom
was raised from 19 to 22 so the actual 37 m survey crop can be framed at zoom 20.3. The parent
approved this shared engine change. Full repository formatting and validation remain parent-owned.

## Authorized extension: vertical-frame sensitivity

After the first two studies passed, the parent authorized `terrain/survey-frame-sensitivity` and
a point-to-plane ICP extension. The new cell-mean comparison reuses the real surveys, not a
perturbed terrain. The builder now adds a north-up 38 × 37 one-metre grid with reference/comparison
arithmetic mean heights, per-epoch return counts and geographic cell centres. At least three
retained returns are required per epoch; 718 cells have paired support. No interpolation fills
the remaining cells. These are sampled mean surfaces, not an independent authoritative DEM.
Total binary payload including these five additional columns is 395,512 bytes.

[xDEM's coregistration guide](https://xdem.readthedocs.io/en/stable/coregistration.html) distinguishes
vertical translation from broader alignment and stresses the use of static control surfaces.
[Its VerticalShift API](https://xdem.readthedocs.io/en/stable/gen_modules/xdem.coreg.VerticalShift.html)
uses a median reduction by default. The local contributor implements this vertical-shift model;
it does not estimate horizontal movement. The chosen patch is moving terrain and has no supplied
stable-control mask. West, east, north or all paired cells are therefore explicitly hypothetical
control selections. A 0.15 m raw-difference standard deviation and exponential correlation length
are adjustable uncertainty assumptions, not calibrated source measurements.

`validate_dem.py` independently computes median bias, NMAD, bias error, corrected volume and
the explicit covariance matrix in float64. Its 12 scenarios are saved in
`pointcloud-dem-validation.json`. With independent 0.15 m difference errors, selecting western
cells as controls yields −33.509 m³ corrected volume; selecting eastern cells yields +91.986 m³.
The raw supported-cell volume is +273.497 m³. This change of sign is the central scientific
question: an assumed reference region can change the inference. Spatial correlation increases
uncertainty without changing the median correction. These are geometric surface-volume scenarios,
not glacier mass balance or observed stable-area registration. Implementation/GPU checks for
this extension are pending.

The point-to-plane extension estimates target normals on the same bounded 1,024-row reference
sample used by local ICP. A 3 m radius and capacity of 64 retain every observed neighborhood in
this sample: independent calculation finds 1,023 usable normals, one unsupported row, no truncated
neighborhoods, and maximum support of 34 other rows. GPU feature flags for degeneracy or a failed
eigensolve are masked to zero normals before ICP. The 258 provided core normals continue to
orient the change cylinders; they are not reused for the different ICP target sample.

An independent NumPy eigenvector and weighted least-squares reference covers all three weighting
choices for the point-to-plane objective. `check-pointcloud-reference.mjs` now checks both metrics,
including normal support and the final translation step. Their residual magnitudes are explicitly
different objectives and cannot be ranked against each other. The DEM checker
`check-survey-frame-reference.mjs` compares all twelve control-mask/correlation cases against the
independent full covariance calculation. Both extensions await the coordinated production GPU slot.


## Production acceptance of the vertical-frame and point-to-plane extension

The stable wave-seven production preview passed 13 independent DEM cases: all four control
masks at correlation lengths 0/2/5 m, plus a doubled assumed raw-difference standard deviation.
It also passed all 11 point-cloud comparison/registration cases, including all three
point-to-plane weighting choices. The GPU returned exactly 1,023 usable target normals, one
weak/unsupported target, zero truncated neighborhoods and maximum support 34. Final plane-fit
translation steps were 0.016 m, 0.001 m and 0.063 m for unweighted/Huber/Tukey, respectively,
within 0.005 m of the independent float64 reference. Corresponding measured median differences
became −0.013 m, −0.011 m and +0.004 m; all fits remain explicitly provisional after 12 iterations.

Both updated stories passed all eight story steps with zero browser or GPU errors. Screenshots
show readable volume/residual charts, a clear signed grid, explicit unknown-support hatching,
and consistent assumed-uncertainty presentation. Artifacts: `/tmp/survey-frame-reference/`,
`/tmp/pointcloud-plane-reference/`, `/tmp/pointcloud-wave7-production/`. A final presentation
adjustment gives the grid a little more camera margin (zoom 20.1) and adds visible source credit,
scale and title furniture; the parent will include it in the next production sweep. No numerical
logic changed after acceptance. The atlas measurement route now connects support, measured
change and hypothetical survey-frame sensitivity in that order.

## Fine-option classification at closure

The contributor audit found two options without explicit showcase controls. Neighborhood
`strategy: serial | cooperative` changes execution decomposition, not the scientific output;
the current default serial path is intentional. SMRF `onlyGround` changes how classification
labels are written: true preserves original non-ground classes, while the default false resets
eligible non-ground rows to `otherClass` (one by default). Neither changes the inferred terrain
or ground mask. The support investigation covers the default classification policy and its
support/mask behavior; preservation of original non-ground labels remains an explicit uncovered
output-policy variant, not an additional ground-detection method. This was reported to the
parent coverage ledger rather than adding another story during closure.
