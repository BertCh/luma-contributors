# Earth systems: evidence, temporal support and physical meaning

Research and implementation ledger, 2026-10-25. Scope: showcase `terrain/`, `hydrology/`,
`raster/` and the new `northeast-pacific-sst` source. Existing scenes and uncommitted work are
preserved. This document separates designed work, written code, static validation and browser
evidence; none of those is a substitute for the others.

## Research findings that changed the design

The existing terrain chapter already runs most of its public contributors. Baseline direct-call
inventory: **38/47 gpu-terrain exports** and **28/95 gpu-raster exports**. These counts include
infrastructure exports and are call-site evidence, not option completeness or execution evidence.
The eight substantive terrain gaps are listed below. Adding a ninth hillshade would have taught
less than improving evidence about raster dates, quality masks and fitted seasonal models.

The existing Dixie NDVI stack contains **16 Sentinel-2 L2A acquisitions, 2017-09-02 through
2024-08-01**, not 16 years or Landsat data. Observations are irregularly spaced. Mean, minimum,
maximum and median are legitimate descriptive acquisition composites; regular-index smoothing
and a seasonal fit would misrepresent elapsed sampling. That constraint motivated a separate,
genuinely monthly scientific dataset instead of pretending the existing stack was regular.

NASA NEO was investigated for a monthly NDVI source. An actual download returned HTML rather than
CSV: NASA says NEO was decommissioned in September 2026. Indexed archive pages were therefore
insufficient acquisition evidence. No NEO-derived numbers or images are bundled.
[NASA retirement notice](https://science.nasa.gov/earth/nasa-earth-observations-neo/).

The replacement is **NOAA ERSSTv5 monthly SST**, accessed successfully from NOAA CoastWatch ERDDAP:
72 months, January 2012–December 2017, a 26 × 21 subset of the original 2° grid in the Northeast
Pacific. The SST and NOAA climatological-anomaly arrays total 314,496 bytes. This is reconstructed
in-situ ocean temperature, not optical imagery or raw observations. The independent NOAA anomaly
uses the 1971–2000 monthly climatology; it must remain distinct from the residual of a fitted
six-year seasonal model. [NOAA archive record](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00927),
[source metadata](https://coastwatch.pfeg.noaa.gov/erddap/info/nceiErsstv5_LonPM180/index.html).

The historical setting is the persistent Northeast Pacific warm anomaly. NOAA PMEL describes the
2014 anomaly and the research into its physical causes. The showcase's bounded one-break model
does **not** reproduce that causal analysis and does not identify daily marine-heatwave events.
[NOAA PMEL](https://www2.pmel.noaa.gov/news-and-media/highlights/causes-and-impacts-2014-warm-anomaly-ne-pacific).

## Data contracts and permissions

| Dataset | Provenance and preprocessing | Rights | Numerical limits |
| --- | --- | --- | --- |
| Existing `ndvi-timeseries` | Earth Search Sentinel-2 L2A scene IDs recorded individually in manifest; B04/B08 averaged to 100 m, SCL nearest sampled; NDVI packed into uint8, code 0 missing, codes 1–255 decode `code/127.5−1`; north-first UTM10N grid | Modified Copernicus Sentinel data, 2017–2024; free/full/open access | Quantization step ≈0.007843 NDVI. Mask excludes SCL 0,1,3,8,9,10,11 but retains cast-shadow/dark-feature class 2 and unclassified 7. Nearest SCL at 100 m does not guarantee all underlying averaged reflectance pixels are clear. |
| New `northeast-pacific-sst` | NOAA ERDDAP exact coordinate/time subset; float32 LE cell-major arrays; 546 cells × 72 months; footprint `[−161,19,−109,61]`; 423 ocean cells and 123 missing land cells in every month | NOAA metadata permits free use/redistribution with no access/use constraints; original disclaimers retained | 2° reconstructed cells do not resolve coastlines or daily extremes. No temporal or spatial interpolation introduced. Calendar slots regular in months, not equal elapsed seconds. |
| Existing terrain DEMs | Terrarium decoded height, spatial metadata and original source used by each scene | Preserve source-specific attribution; do not infer public domain from the encoding | Elevation precision, interpolation and source noise affect derivatives; finite windows clip catchments and shadows. Web Mercator ground-cell correction is separate from projected pixel spacing. |

Sentinel metadata and reflectance offsets require care: processing baseline 04.00 introduced an
additive offset; a multiplicative scale cancels in NDVI, an additive offset does not. The existing
COG builder assumes its source has already had that offset removed. A live per-scene STAC audit is
recorded separately in `ndvi-source-audit.json`; it is evidence about metadata, not a replacement
for an independent reflectance validation. Reapplying a source offset blindly can create a second
correction. [Copernicus processing and SCL specification](https://sentiwiki.copernicus.eu/web/s2-processing),
[Copernicus legal notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice),
[Earth Search source registry](https://registry.opendata.aws/sentinel-2-l2a-cogs/).

All 16 IDs resolved in that audit: ten pre-04.00 products report `boa_offset_applied: false`;
six products from 2022–2024 report `true`. This is consistent with the bundled builder's use of
already harmonized COG digital counts for the newer observations. The raw manifest dates and
existing packed NDVI bytes were not rewritten.

The new NOAA builder is standard-library Python, records exact query URL, source SHA-256, output
SHA-256 hashes, and checks coordinate uniqueness, complete month coverage, valid units and ranges.
It preserves NaNs and row orientation. Its README documents cache/revision behavior, and the
manifest preserves the pinned v5 product identity without claiming it is the latest release.

## Story designs and delivery states

| Story | Analyst question and visible outputs | Planned controls | Delivery state |
| --- | --- | --- | --- |
| `composite-evidence` (new) | Can a green composite hide the fire? Actual GPU composite, valid count, chosen acquisition, selected-pixel observations | Mean/minimum/maximum/median, acquisition window, minimum support, output layer, palette/opacity | Written; reviewer found and requested fixes to masking, stale readback, source sentinel and narrative date counts. Final validation below. |
| `vegetation-trends` (expanded) | What was actually sampled behind a per-pixel trend? Acquisition-date line chart alongside existing statistics/parity | Existing full change-detection controls preserved | Chart written, elapsed-day axis honors irregular dates. Final validation below. |
| `ocean-seasonality` (new) | When does the Pacific remain warm after the seasonal cycle is accounted for? Raw SST, NOAA anomaly, GPU smoothing, fitted trend/seasonality/residual and one selected break | Month, SG window/order/edges, seasonal fit variant, minimum segment, BIC threshold, missing policy, bounded iteration diagnostics as implemented | New real source delivered; implementation in progress. No claim of browser validation yet. |

Suggested inquiry route: **composite-evidence → burn-severity → vegetation-trends**. A second
physical-process route remains **terrain-basics → watersheds → flood-lab**, which moves from
elevation measurements through drainage to an explicitly conditional scenario. The new ocean
story contrasts a recurring cycle, an externally supplied anomaly and an offline fitted shift.

## API and option matrix

“Existing” below denotes inspected existing scene usage, not validation performed in this task.
Each gap is deliberately explicit so new metadata or a code snippet cannot pass for functionality.

| Public family | Existing runtime and meaningful options | Expansion / outstanding gap |
| --- | --- | --- |
| `GPURasterTemporal` | No baseline direct call | Composite reductions added; new monthly dataset supports SG operation, polynomial order, odd window, edge handling, validity support/status. Regular-time temporal capability is a separate source contract. |
| `GPURasterSeasonalBreak` | No baseline direct call | New ocean scene planned: period 12, single harmonic or dummy seasons, minimum segment length, BIC gain, joint/iterative fit, missing policy and diagnostics. Fixed-history monitoring, consecutive-alarm semantics and iteration-limit paths must be inventoried separately; not implied by basic fit. |
| `GPUChangeDetection` | Vegetation trends: difference/log ratio/percent change, Welch t/Mann–Kendall, Sen slope, dates/split/alpha/minimum support/epsilon | Actual-time selected-pixel chart added. Sen slope remains NDVI per acquisition index; no annual-rate label. Uncorrected pixelwise significance is not a familywise inference. |
| Raster local algebra / spectral indices | Burn severity and suitability use algebra, conditional/reclassify/stretch and weighted operations | Direct NDVI/NBR/EVI/NDWI/SAVI wrappers and BandMath remain unproven by the baseline reporter even where equivalent formulas exist. Explicit comparison and QA/scale provenance are needed before claiming full wrapper coverage. |
| Focal/filter/image operators | Existing cellular, terrain and imagery workflows cover some focal effects | Convolution/box/Gaussian blur, gradient/Sobel/Scharr/Laplacian, neighborhood/rank/focal-pairs, opening/closing/dilation/morphology and reconstruction/fill-holes have direct-call gaps. Kernel edge/nodata/halo behavior requires its own inspection. |
| Raster statistics and thresholds | Existing raster cell/zonal/region summaries and stretch workflows | Histogram/contrast/threshold/Otsu, fractional coverage/zonal/categorical/distribution, global histogram/statistics merge and global percentiles remain distinct gaps. No synthetic histogram is presented as GPU output. |
| Raster geometry and transport | Existing contour/isoband/cost-distance/distance-field/flow stories | Registration, conservative remapping, reprojection/tile reprojection, region graph/measurements, skeleton/medial axis and watershed segmentation need substantive output/control stories. |
| Raster remote sensing | Existing change detection and Sentinel source | Spectral unmixing and texture are not covered by temporal additions. Require interpretable endmembers, band units, categorical quantization and sampling assumptions. |
| Raster tile infrastructure | Existing assets load into bounded raster graphs | Tile reader/cache/leases, halo assembly/fill/core extraction, overview/categorical overview, cross-tile components and global reduction initialization/merges remain workflow gaps. A single tile displayed at scale does not demonstrate streaming, seams or global reductions. |
| Terrain decode/repair/derivatives/ruggedness | Terrain basics exposes encoding/repair, slope/aspect/hillshade, ruggedness and ground-cell semantics | Existing functionality; no new evidence of every decoder/repair edge case. Noise repair remains optional and cannot know whether a real cliff is an error. |
| Terrain relief/classification | Relief visualization and landforms: multidirectional/textural relief, horizons/openness/SVF, SLRM/MSRM/local dominance/blends, geomorphons, curvature, TPI/DEV and Weiss classification | Strong existing call coverage; no superficial duplicate scene. Search-radius/window/halo limits and scale dependence remain substantive options to preserve. |
| Terrain illumination/visibility | Sun/shadow and visibility: solar position/irradiance/shadow/cast-shadow, viewsheds, line-of-sight/cumulative viewshed, point horizon/profile | Existing terrain-only models. Clouds/buildings/trees and terrain beyond the tile are not silently modeled. |
| Terrain summit/features | Existing peaks/critical points/summit graph/profile/snap | `GPUTerrainPeakProminence` remains a gap: prominence-tree semantics are not equivalent to nearest local peak height. |
| Terrain flow and drainage | Watersheds: D8/D∞/Freeman/Quinn, fill/flat/accumulation limits, stream threshold, watershed mode/pour points, order, HAND, wetness/power, flow distance and stream graph | Existing flow-distance/network upgrade from previous sprint preserved. HAND remains relative terrain height, not hydraulics. |
| New terrain-processing APIs | No baseline direct calls to the following | `GPUTerrainBranchingFlow`, `GPUTerrainBreach`, `GPUTerrainCatchmentSummary`, `GPUTerrainDEMCoregistration`, `GPUTerrainDepressionHierarchy`, `GPUTerrainFlowFractions`, `GPUTerrainTiledFlow`. These require real branching weights, basin remapping/metric grids, paired DEMs or tile-boundary contracts. Not delivered by the new temporal scenes. |

Catchment-summary integration was inspected: its basin IDs must be **dense, zero-based**, its grid
affine must be planar for metric area/length, and streams must be resolved boolean masks. Existing
automatic watershed labels cannot be passed directly as dense IDs. This is a concrete integration
requirement, not a reason to label the API covered. A later hydrology pass should show basin
hypsometry and drainage density together with clipped-catchment diagnostics.

## Subsequent implementation waves

These are executable plans, not completed additions or a claim that all functionality is covered.

1. **Terrain conditioning decisions.** Block-average the real Grand Canyon DEM to a bounded grid,
   retain an explicit unmodified reference, identify real local-minimum candidates, and compare
   constrained breach cuts with filling. Show candidate/pit statuses, modification depth and
   depression-hierarchy spill/area/volume queries. Follow the conditioned surface with public
   flow fractions and branching transport/expected/min/max distance, then dense-ID catchment
   hypsometry and drainage density. Bound pit count and search length visibly. Any controlled burn
   or synthetic perturbation must be labeled an experiment on sourced terrain, not an observation.
2. **A pixel crosses the boundary.** Show fractional coverage, categorical and continuous zonal
   summaries and distributions, followed by conservative remapping on the real Sentinel projected raster
   (the implementation is Cartesian, so the NOAA degree grid is inappropriate for square-meter claims). Compare valid support, whole-cell estimates and fractional estimates. Use an explicit
   area model; degrees squared cannot be labeled square meters. Remap nodata support separately
   from values so coastlines do not become zero-temperature observations.
3. **Fire-scar topology.** Start with real Sentinel bands and SCL, compute named spectral indices,
   compare explicit and Otsu thresholds, then morphological opening/closing/reconstruction and
   connected-region/region-graph summaries. Skeleton/medial-axis/watershed outputs need distinct
   purpose and legends. Show residual cloud and halo sensitivity; these are mask operators, not
   validated burn-perimeter extraction.
4. **Same landscape, different grid.** Use raster registration/reprojection, local derivatives,
   texture/spectral operations and tile seams to compare representations. A reproducible known
   image transform is acceptable as a labeled registration experiment, but must not be sold as
   independently acquired temporal change. Streaming infrastructure needs actual tiled ownership,
   halo reads and global merges, with bounded status/overflow reporting.

Hydrology method reading: [Tarboton's original D∞ paper](https://hydrology.usu.edu/dtarb/96wr03137.pdf)
motivates proportioned routing; [Barnes, Lehman and Mulla](https://rbarnes.org/sci/2014_depressions.pdf)
explain depression filling; [Lindsay's paper](https://onlinelibrary.wiley.com/doi/10.1002/hyp.10648/full)
compares breaching and filling. These establish scientific context, not an assertion that the
showcase's bounded GPU implementation is identical to the published algorithms. Its actual
documented contract is the authority for runtime claims.

## Verification ledger

- New SST builder executed successfully; 39,312 unique records, 72 months, 546 cells, 314,496 bytes.
- Source checks: valid ocean-cell count is 423 for every month; remaining 123 cells are NaN.
- Independent raw-data inspection at 146°W, 50°N: NOAA annual anomaly means are −0.953°C (2012),
  +0.408 (2013), +1.792 (2014), +1.564 (2015), +1.023 (2016), +0.028 (2017). These are fixture
  checks, not substituted GPU results or causal conclusions.
- Initial composite story-step, WGSL identifier and strict raster cartography checks passed.
  Passing these did not detect the implementation defects found in manual review.
- Root `nvm use`, install, formatting, module build, node/full GPU tests and website checks are
  coordinated by the parent task; this domain agent does not run competing repository-wide edits.
- Final implementation and browser checks remain pending until the delivery update is appended.

### Wave 1 delivery update

The two new scenes and existing trend chart are implemented. Manual review corrected the
minimum-support mask's false value (zero → NaN), unbounded/stale readback, missing source-date
sentinel handling, compile-option labeling, an incorrect acquisition count, numerical select
coercion, initial SST/anomaly selection, graph-rebuild revision acceptance, and signed legends.
SG support/status is exposed. Monthly regional evidence uses cosine-latitude area weights.
The second story step uses reconstructed SST to demonstrate its large seasonal cycle; NOAA's
supplied anomaly and the model residual remain separately labeled.

The implementer's final scoped checks passed: showcase TypeScript, raster story steps, WGSL
identifier scan and strict raster cartography. The new NOAA builder was rerun into a separate
temporary output directory: both binary arrays, manifest and source-metadata snapshot were
byte-for-byte identical. Production build/browser/GPU execution is delegated to the parent
integration sweep and is not yet recorded here as passed.

Independent CPU least-squares spot check at grid cell 137 (146°W, 50°N), harmonic period 12,
joint fit, minimum segment 12, minimum BIC gain 4:

| Input | Selected boundary | BIC improvement | Interpretation |
| --- | --- | --- | --- |
| NOAA supplied anomaly | Index 22, November 2013 | 59.02764 | One-jump fit relative to a seasonal linear baseline |
| Reconstructed SST | Index 18, July 2013 | 30.03132 | Different response and seasonal residual structure produce a different boundary |

The reference was calculated independently from the shipped float32 arrays with a Python
least-squares solver, matching the documented BIC penalty (two additional parameters for jump
and searched date). These are expected browser values with floating-point tolerance, not a
statement that the browser has already matched them. The differing dates make the methodological
point: a selected boundary is not automatically the physical onset of a named ocean event.


### Production review and expanded source delivery

The parent production WebGPU sweep passed all four composite, four ocean and seven vegetation
story steps. The displayed ocean anomaly result matched the independent oracle: November 2013,
BIC improvement 59.03. Visual review found missing geographic context on the ocean map and an
empty initial composite chart. The implementer added Natural Earth land/coast, 10-degree
graticules, accurate regional annotations and a selected-cell marker; composite now initializes
a valid pixel with all 16 observations. These polish changes await the next production sweep.

A second new source is delivered: `dixie-spectral`, six calibrated Sentinel-2 bands on a
250 × 250, 60 m UTM 10N grid for pinned 2021-07-13 and 2021-09-21 scenes. The explicit source
builder, metadata snapshots, calibration assertions and masks are documented in
`scripts/data/dixie-spectral/README.md`. The source supports blue/green/red/NIR/SWIR1/SWIR2
methods that the older three-band fixture could not support. Strict screening requires every
native SCL contributor to be 4/5/6 and every spectral contributor nonzero. Separate land-only
masks require 4/5, excluding any water contribution. This strict aggregation is intentionally
more conservative than nearest-neighbor SCL sampling.

Independent validation: all 16 derived binary hashes matched, every band's finite cells matched
its date mask exactly, and a separate cache rebuild reproduced all 19 shipped files byte for
byte (about 3.31 MB total). July has 60,971 spectrally valid cells / 59,709 land cells;
September has 36,631 / 35,695. Paired valid land is 35,583 cells. Independent float32 dNBR
quartiles over paired land are 0.0794173, 0.4946086, 0.8156452; these are spectral differences,
not validated burn-severity categories. Cell 26,488 has dNBR 0.18530928. The center cell is
invalid in September and must not be used as a default evidence probe.

### Wave 2 review requirements and exact next-wave contracts

Terrain implementation remains **in review**, not delivered. Review caught three scientific
integration errors before publication: hierarchy stages are absolute elevations, not depths
above each sink; hierarchy areas use affine coordinate units, so Web Mercator areas require an
explicit ground approximation; and computed branching products need actual visual outputs,
not just successful-call status. The corrected scene will expose original and filled terrain,
bounded breach, hierarchy sublevel curves, routing fractions, branching products and actual
hypsometry. Geometric sublevel volume is not a physically connected inundation simulation.

The fire topology lab is delegated to the data-coverage track with the new strict Sentinel
land masks; this track will own later fractional/remapping and spectral/grid methods. Their
presence in this plan is **not** completion or coverage credit.

| Planned inquiry | Concrete API/options contract | Evidence and bounds |
| --- | --- | --- |
| A pixel crosses a boundary | `GPURasterFractionalCoverage`, continuous/categorical zonal summaries, `GPURasterFractionalZonalDistribution` exactextract/inverse-CDF, probability controls | Analyst-defined transects explicitly labeled as queries, real spectral input, finite support, tuple overflow and polygon validity visible; max 65,536 distribution tuples |
| What survives a coarser grid? | `GPURasterConservativeRemapping` destination-area/covered-area/extensive, persistent overlap prepare/reuse | Real Sentinel UTM cells; south-first conversion explicit; exact overlap areas, observed-support fraction, source/destination integral and residual, no invented physical mass |
| Does a different grid change the evidence? | `GPURasterReprojection` nearest/bilinear and strict/renormalize, explicit inverse projection; registration periodic/masked-linear | Reprojection source tile includes interpolation support; a known image translation is explicitly a controlled registration experiment, not observed landscape displacement |
| What can six bands distinguish? | Named spectral indices and `GPURasterSpectralUnmixing` unconstrained/nonnegative/simplex with iteration/tolerance/status | Measured source spectra, declared endmember selection, residuals and solver status; abundance weights are a model, not validated land-cover percentages |

Fractional means and quantiles must weight boundary cells by actual coverage, and retain missing
support separately. Method context: [exactextract operations](https://isciences.github.io/exactextract/operations.html).
Conservative remapping's destination-area and covered-area conventions use different denominators;
a covered-area mean must be multiplied by covered area when checking an integral. Method context:
[ESMF field and grid reference](https://earthsystemmodeling.org/docs/release/ESMF_8_4_2/ESMF_refdoc/node5.html).
The local GPU class only handles axis-aligned Cartesian rectangles with at most 256 candidate
source cells per destination. It does not implement ESMF's spherical geometry or higher orders.


#### Spectral/grid implementation handoff notes

The expanded fixture has a completely observed 104 × 104 land square beginning at row 0,
column 32 in the July scene. A 64 × 64 crop within it supports a radix-two, fully finite periodic
registration experiment without filling missing observations. A known wrapped translation must
be labeled as a deliberate image transform. For realistic bounded support, use masked-linear
registration on the same sourced crop with an explicit nonwrapping shift and masks, and expose
maximumShift, minimumOverlapFraction/count, minimumCorrelation, upsampleFactor and
ambiguityTolerance. These are two different boundary models, not two estimates of observed
fire displacement. Status 1/2/3 means unusable/ambiguous/insufficient support respectively.

For a six-band mixture, any endmembers selected from the image should be called measured
spectral exemplars, with locations and six reflectances visible. Vegetation-like, dark low-NBR,
and bright low-NDVI examples are defensible descriptive labels; they are not independently
validated forest, char or mineral classes. Compare unconstrained, nonnegative and simplex
fits, show squared spectral residual and iteration status, and do not silently suppress
iteration-limited fits. The source includes genuine water-valid pixels, but the fire topology
lab deliberately excludes them through the separate land masks.

The independent Grand Canyon block-average check found minimum 759.51904297 m, maximum
2609.56347656 m, and 43 interior eight-neighbor local minima before the candidate cap of 32.
The minimum is source-analysis cell 2064 (row 32, column 16). These are useful checks for the
terrain map range and absolute stage queries. A 0–2200 m legend would clip the North Rim.


### Terrain implementation delivery, awaiting production GPU check

`terrain-conditioning` is implemented and reviewed. It exposes original/fill elevation, bounded
breach cut depth, D∞/Freeman/Quinn outgoing fractions and expected distance; additional D∞
transport, minimum/maximum distance and fractional-catchment layers have explicit source/outlet
markers. The latter products currently use D∞ even when the fraction-method control changes;
this distinction is disclosed in controls and product readouts. The hierarchy uses absolute
source-minimum + 200 m query intervals, nominal owner categories, ground-area affine, a selected
query marker, area/volume curves and persistent invalidation on overflow. The catchment scene
uses one explicitly defined analysis-window label, with actual cumulative hypsometry and profile
status. The area approximation is center-row planar ground spacing, not uncorrected Mercator.

Further review caught and corrected double accumulation of an already cumulative hypsometry
output, incorrect bin centers (the API returns upper edges), stale legend extents, aliased graph
imports under multiple resource names, and overbroad claims about routing-mode changes.

| Terrain API | Runtime delivery | Options deliberately still fixed or absent |
| --- | --- | --- |
| `GPUTerrainBreach` | Real interior pit candidates, cut depth, outcomes, unresolved count | 32-candidate cap, 8 steps, 25 m maximum cut, 0.25 m decrement; optional burnDepth absent |
| `GPUTerrainDepressionHierarchy` | Boundary outlets, owner map, nine absolute stages, root area/volume curves, overflow guard | Capacity 4096, stage selection changes displayed query; outlet model fixed to finite window |
| `GPUTerrainFlowFractions` | D∞, Freeman MFD, Quinn MFD; actual maximum outgoing share | Shared canonical filled surface and D8 fallback; exponent fixed 1.1 |
| `GPUTerrainBranchingFlow` | All five modes produce mapped values and unresolved counts; expected-distance compares three fraction methods | Other four modes use D∞; maximumIterations 128, default walk length; user-selected outlet absent |
| `GPUTerrainCatchmentSummary` | Area/elevation/stream/density, cumulative hypsometry, per-cell profile status | One dense window label, 16 bins; profile numeric channels computed but only status mapped |

The dedicated `scripts/check-terrain-conditioning-browser.mjs` prepares independent numerical
acceptance checks for stage indices 0/3/8 and area/hypsometry closure. Syntax validation passed;
production execution is parent-coordinated and remains pending at this update. The source
oracle uses direct sums of the decoded, block-averaged DEM: at 1359.519043 m, area is
246.315519 km² and geometric volume is 56,224.892601 million m³; at 2359.519043 m, area is
883.464682 km² and volume 600,954.074823 million m³. Total window area is 999.909184 km².

Open terrain classes remain `GPUTerrainDEMCoregistration`, `GPUTerrainPeakProminence`, and
`GPUTerrainTiledFlow`. A direct class call does not establish complete option or workflow
coverage; the fixed/absent options above remain explicit gaps. The next active implementation
is the fractional coverage/distribution/categorical and conservative remapping lab, on a
128 × 128 crop of the new Sentinel projected source.


#### Remaining terrain workflow design

`GPUTerrainPeakProminence` should deepen the existing Matterhorn summit inquiry. Its result is
the upper-excursion-set elder-rule persistence on a closed, finite raster: a key saddle joins a
peak to an elder peak; the surviving component elder has **infinite** prominence. An infinite
result is not the global topographic prominence of the Matterhorn and must receive a separate
boundary category. A bounded downsampled Alps grid, mapped peak/key-saddle/parent relationships,
and a table of named nearby peaks can distinguish this exact topology from the existing local
disc/ring summit drop. Source resolution and finite-window effects remain visible.

`GPUTerrainTiledFlow` requires genuinely disjoint owned rectangles covering the full domain,
with halos excluded from runoff injection. A useful extension of the terrain-conditioning
experiment would compare two resident tiles (one-row halos at their seam) with the untiled
reference, show reach identities crossing the seam, and report maximum accumulation residual
on owned cells. Both D8 and branching modes require globally prepared routing; recomputing
routes independently inside tiles would invalidate this comparison. The implementation is
resident global exchange, not an out-of-core large-basin scalability claim.

`GPUTerrainDEMCoregistration` performs robust **vertical** median-bias correction, not horizontal
image registration. The real repeated AHK point-cloud epochs in the experience track could
supply rasterized DEMs after establishing a stable-surface mask and declared difference
variance. Those variance and spatial-correlation assumptions are model inputs, not uncertainty
estimated from just two rasters. This data synergy has been communicated to that planner;
implementation is not yet assigned or delivered.


### Authorized subsequent inquiry queue (not yet implemented)

After the active boundary/remapping and six-band spectral/grid work, the same implementer will
continue the following coherent investigations. The data-coverage track confirmed that it does
not implement grayscale reconstruction; its fire lab owns binary median/opening/closing,
skeleton, medial axis, watershed segmentation and binary-class region contact graphs.

1. **Which patterns survive a change of scale?** Use real Sentinel NIR/SWIR reflectance and
   source validity. Compare box/Gaussian smoothing, direct rectangular convolution and generic
   weighted neighborhoods; then Sobel/Scharr/gradient magnitude/Laplacian image derivatives.
   These are reflectance gradients, not terrain slopes. GLCM texture maps expose all seven
   metrics (contrast, dissimilarity, homogeneity, ASM, energy, correlation, natural-log entropy),
   pair count, quantization levels/range, radius, direction and symmetry. A clicked local
   histogram and before/after profile make support visible. Paired focal statistics compare
   NIR and SWIR on the same date, with paired count, covariance/correlation/OLS/R²/residual,
   population/sample covariance, minimum support, center validity and explicit border rules.
   Spatially overlapping windows do not justify independent inferential tests.
2. **What did filling invent?** Start from the fully observed July crop and deliberately withhold
   an explicit block or narrow strip of known pixels. Compare nearest, all-donor IDW and GDAL
   four-quadrant fill, bounded search radius, filled provenance and unresolved cells against
   the withheld real observations. Report holdout error by gap width; actual September cloud
   gaps have no withheld truth and are a separate example. All fill methods preserve original
   donors and do not recursively promote filled pixels into donors. Grayscale reconstruction
   uses real NIR as a mask, with explicit marker construction, dilation/erosion, 4/8 connectivity
   and convergence status. Reconstructed features and NIR contour/classifier output remain
   spectral geometry rather than claimed burn perimeters. Gaps are never silently zero-valued.
3. **Can a seam change the answer?** Process the bounded real raster as distinct tiles with
   retained ownership and halo semantics. Compare a local stencil with and without its
   required halo to an untiled reference, then demonstrate the correct pipeline. Tile cache,
   reader/lease, halo fill/assembly/core extraction and tile reprojection are included only
   when their lifecycle is actually executed. Global statistics require an explicit dataset
   initialization, one merge per owned tile, and a second histogram replay after the global
   extent stabilizes. Histogram percentiles are estimates with bin-resolution limits; zero
   and one preserve observed extrema. Overflow bits and valid count remain visible. Categorical
   overviews use categorical aggregation rather than interpolating category IDs. This is a
   bounded correctness exercise, not an out-of-core performance claim.
4. **Which peak belongs to which saddle?** Extend the Alps terrain inquiry with genuine finite
   window prominence and elder relations, followed by the previously described D8/branching
   owned-tile terrain flow comparison. Keep the infinite elder category and domain boundaries
   visible. The point-cloud planner now owns the separate real repeated-survey vertical-bias
   coregistration lab, so it will not be duplicated here.

| Family | Meaningful variants planned | Scientific/runtime constraint |
| --- | --- | --- |
| Neighborhood/convolution/smoothing | clamp, reflect, constant, nodata borders; propagate/ignore-renormalize; square/rectangular and separable kernels; normalized smoothing vs signed derivatives | Missing centers remain invalid; signed kernels cannot renormalize missing weights |
| Focal pairs | nodata/clamp/reflect; ignore/propagate; center propagation; population/sample; all six floating metrics | Same paired-valid samples; constant predictor/response statuses; no inference implied |
| GLCM | seven metrics; direction; symmetric on/off; quantization and radius | Both pair endpoints inside window; empty support NaN; pair count can double under symmetry |
| Nodata fill | nearest, inverse-distance, gdal-four-quadrant; bounded radius | Original donors only; no smoothing pass; held-out truth explicitly artificial omission |
| Reconstruction | dilation/erosion; 4/8 connectivity; iteration budget | Marker inequality must hold; failed convergence is not silently accepted |
| Raster contours | actual contour segments/classification with bounded output capacity | NIR units, source support, capacity/overflow and map placement visible |
| Tile/global processing | actual ownership/halo/cache lifecycle, statistics then histogram replay, percentile endpoints/interior | No double counting halos; no unstable histogram domain; resident and out-of-core behavior distinguished |

Primary implementation references: [GDAL FillNodata](https://gdal.org/en/stable/api/python/utilities.html)
describes its directional interpolation and optional smoothing; our public four-quadrant mode
uses zero smoothing. [scikit-image reconstruction](https://scikit-image.org/docs/stable/api/skimage.morphology)
defines the marker/mask ordering and constrained spread. [scikit-image GLCM documentation](https://scikit-image.org/docs/stable/api/skimage.feature.html)
provides a useful independent definition of co-occurrence normalization and named metrics. These
references guide interpretation and independent checks; the local API contract remains the
source of exact supported bounds and nodata behavior.


### Integration checkpoint

The first terrain production attempt caught a duplicate parameter-buffer graph import before
initialization. It is fixed by importing the flow settings once and reusing that view for
canonical flow and all three fraction producers. Production re-run and numerical assertions
remain pending; static checks alone did not establish runtime correctness. The polished ocean
map passed the parent's production sweep and visual review with geographic context; its
November 2013 / 59.03 anomaly result remains verified.

To parallelize the authorized remaining raster work, the parent approved one additional
`gpt-6-luna` CLI implementer. It owns only new `raster/spectral-texture.*` and a unique
implementation ledger; the existing implementer retains `pixel-boundaries.*` and subsequent
spectral-index/unmixing/grid work. The CLI task has explicit metadata-last, shared-checkout,
no-root-build/lint, no-browser-without-slot and truthful scientific framing constraints. Its
outputs will be reviewed and independently checked before receiving delivery credit.

### Boundary and texture review checkpoint

The independent CPU reference in `earth-source-oracles.json` clips each convex analyst ring
against each 60 m source cell and subtracts the interior hole. It finds 7,228 positive
intersections (per query: 2,254, 429, 2,144, 2,401) and geometric areas of 7.938, 1.3068,
7.418 and 8.410 km². September query 0 has 2.6508 km² of valid NIR support and an
area-weighted mean of 0.085147950235. Full-crop valid-support integrals are
13,132,541.86951518 reflectance·m² for July and 3,626,581.951533258 for September.
These are independent reference numbers, not yet evidence of a production GPU match.

Review of the initial boundary implementation identified and requested corrections to:
raw covered-area displayed as a fraction; NaN remap cells contaminating the total;
unsorted selected-probability quantile curves; an initially cloudy probe; and acquisition
language incorrectly describing a 60 m aggregation as native measurement and July 13
as necessarily pre-fire. Fractional status and quantile validity also need to gate readouts.
The exact transformed crop center is [-121.0005443654, 40.1759228850].

The first CLI texture draft computes several useful kernels, but its initial fixed July
filters, incomplete parameter controls and palette/legend mismatch are not accepted as
complete delivery. The planned method/options matrix above remains the acceptance scope.
Neither a declared contributor nor a computed but unexposed output earns coverage.

Stable composite/ocean readback offset arithmetic was simplified to explicit byte offsets
for lint compliance. Direct scoped Biome checks pass. A Node assertion verifies exact
equality of old/new layouts: ocean offsets are [0,157248,314496,471744,943488,945672,
947856,950040] for 546 cells × 72 months. This edit changes no numerical computation.

The subsequent six-band scene uses calibrated unitless Sentinel reflectances. Primary formula
checks: [USGS EVI](https://www.usgs.gov/landsat-missions/landsat-enhanced-vegetation-index)
uses gain 2.5, red coefficient 6, blue coefficient 7.5 and background term 1;
[USGS SAVI](https://www.usgs.gov/landsat-missions/landsat-soil-adjusted-vegetation-index)
uses the `(1+L)` factor with an explicit soil adjustment (default 0.5). These pages describe
Landsat products; our use transfers the spectral formula to the declared Sentinel bands and
must not claim the fixture is a USGS product. [NASA's vegetation-index explanation](https://science.nasa.gov/earth/earth-observatory/measuring-vegetation-ndvi-evi/)
also makes clear that clouds/aerosols and sampling gaps remain limitations. Independent scalar
index calculations at source cell 26633 are recorded in `earth-source-oracles.json` for both
dates, including both distinctly named NDWI conventions. SWIR1 is a sensor-band adaptation of
Gao's vegetation-water index, not the original 1.24 μm channel.

For unmixing, the [USGS PyHAT user guide](https://pubs.usgs.gov/publication/ofr20251038/full)
distinguishes unconstrained least squares from nonnegative and sum-to-one constraints. Our
three measured exemplars form an explicitly chosen spectral basis; enforcing feasible
coefficients does not validate them as pure ground materials. Displaying fitted spectra and
squared residual alongside solver status is therefore part of the story, not optional debug UI.

Boundary review corrections are now implemented and direct scoped Biome checks pass for both
scene files. `scripts/check-pixel-boundaries-browser.mjs` has passed Node syntax checking and
is ready for the coordinator's exclusive GPU slot: it checks the independent 7,228 tuple
count, September rectangle mean/support, and both dates × both grid phases × all three
normalizations against valid-support source integrals. Production execution is still pending.
The scripts directory is excluded by the current Biome configuration; no formatting check
is claimed for that `.mjs` script. The whole catalog's read-only dataset validator passes
at this checkpoint: 71 catalog datasets, 69 bundled datasets, 714 verified files, zero errors.

### Production numerics and fractional-clipping accuracy repair

Wave 7 production GPU verification passed terrain stages 0/3/8 against independently decoded
and block-averaged source elevations. Reported root areas were 0.244, 246.316 and 883.465 km²;
volumes were 0, 56,224.895 and 600,954.044 million m³, within the declared float32 tolerance.
The final cumulative hypsometry bin and window area both equal 999.909 km². No browser/GPU
errors occurred. Report/screenshot: `/tmp/luma-showcase-overhaul-20261025/terrain-numerics/`.

The same production build passed all twelve conservative-remapping integral cases, plus the
September rectangle's mean, variance and valid area. The raw coverage count was 7,373 versus
7,228 positive intersections from independent float64 clipping. The original exact-count
assertion therefore failed and was retained in the first report. It was not treated as a pass.
Independent float32 simulation then established that fused versus unfused shoelace arithmetic
and clipping interpolation produce different tiny positive slivers; counts alone are not a
stable cross-backend invariant. The four query geometries have a smallest genuine intersection
of 5.0420168 m². The revised diagnostic counts overlaps exceeding 1 m², reports raw tuples
separately, and never removes any weight from the actual algorithms. The revised browser
check also verifies all four query areas, means, variances and effective support.

With parent approval, a minimal core correction now clips triangles in cell-local coordinates
and pins the clipped axis exactly to its boundary, reducing cancellation without an arbitrary
area cutoff. Only that shader logic and a numerical caveat were changed in
`modules/graph/src/gpu-raster/gpu-raster-fractional-coverage.ts`; existing changes were preserved.
The new `fractional-coverage-rounding.spec.ts` compares the same binary-exact fractional rectangle
near the origin and at row/column 460. Independent float32+FMA simulation of that regression
geometry gives old max-cell error 0.0074768 and total 414.951013 against exact 415.125; the new
local calculation gives max error 5.96e-7 and total 415.124998. The regression asserts per-cell
2e-5 and total 0.002 tolerances, and checks meaningful support rather than raw positive counts.
Scoped Biome checks pass. Actual GPU execution of this new test and revised scene is pending
the parent's next build/test slot; CPU reproduction is not presented as GPU validation.

### Current delivery checkpoint (supersedes earlier draft notes)

| Story | Implemented evidence/options | Validation state |
| --- | --- | --- |
| `composite-evidence` | Four temporal composite operations, count/window/source provenance, selected observed series | Production story sweep passed; byte-layout cleanup independently identical and lint clean |
| `ocean-seasonality` | Real 72-month SST/anomaly stack, smooth/seasonal-break alternatives, default-cell chart and Pacific geography | Production story sweep and November 2013 BIC gain 59.03 check passed |
| `vegetation-trends` | Existing methods retained, actual irregular acquisition-date selected series | Production seven-step sweep passed |
| `terrain-conditioning` | Bounded breach/status, stage hierarchy/area/volume, three flow-fraction methods, branching products, window summary/hypsometry | Production stage and cumulative-area numerics passed; subsequent profile-map geographic polish awaits visual review |
| `pixel-boundaries` | Four exact query geometries, two quantile conventions, weighted moments, categorical areas, both dates/phases/three conservative normalizations | Rebuilt wave10 clipping, moments, support diagnostics, all 12 conservation cases and eight paired-support cases passed; wave11 all eight quantity contracts and eight paired-support cases passed individually; full combined harness rerun pending a fixed tab-navigation selector |
| `spectral-evidence` | NDVI/NBR/EVI/SAVI L=0/0.5/1, two distinct NDWI conventions; three mixture constraints, solver budget/tolerance, measured exemplar and fitted spectra, crop solver statuses | Both-date six-index and three-constraint mixture GPU checks passed; wave11 all 18 epsilon/clamp cases passed with baseline fits |
| `spectral-texture` | Planner-reviewed selected-date filtering, derivatives, texture and focal-pair controls; display/transect/transport extension reviewed and source-ready | Baseline independent oracle and browser script ready; no GPU acceptance yet |
| `grid-alignment` | Real CRS resampling and controlled periodic/masked registration; exact analysis/display-frame separation reviewed | Wave11 periodic/masked corrections and aligned counts/residuals passed; CRS selected values/policy checks passed, four support counts differ by one and remain under diagnosis |

Independent spectral-mixture references enumerate every feasible active subset for nonnegative
least squares, and every simplex face with an equality-constrained KKT solve. At source cell
26633, September unconstrained coefficients are [-0.0223718, 1.1931110, 0.2853817], squared
residual 0.0002324358; nonnegative optimum is [0,1.2089660,0.2715585], residual 0.0002350110;
simplex optimum is [0,0.4435755,0.5564245], residual 0.0032001091. These are optimizer
references, not claims that a finite GPU iteration budget reaches the exact optimum.
The three measured September exemplars are source cells 14602, 20119 and 26889; their
six-band matrix has singular values [0.4718073,0.0984417,0.0440666]. Full inputs and both-date
references are in `earth-source-oracles.json`. The new `check-spectral-evidence-browser.mjs`
script passes syntax checking and is ready for a production GPU slot.

### Rebuilt GPU acceptance and remaining waves

The focused `fractional-coverage-rounding.spec.ts` GPU regression passed (1/1, parent run).
A first wave9 browser rerun still failed: inspection of the emitted Vite asset proved it
contained the old global-coordinate shader, although the freshly built package contained
the fix. These failed reports are retained. No area tolerance was increased. After wave10
rebundled the package, all four query means, variances, effective support and areas passed,
as did all twelve date/phase/normalization conservation comparisons. Report:
`/tmp/luma-showcase-overhaul-20261025/pixel-boundary-numerics-wave10/report.json`.
The new raw tuple count is 7,378; exactly 7,228 exceed the independent 1 m² diagnostic
threshold. Reported geometric areas are now within about 1 m² of the ideal polygons.
The raw count remains a diagnostic, never an exact mathematical support invariant.

The paired-observation extension passes all eight date/query comparisons against independent
float64 clipping. `observationWeights` is one only where both real source acquisitions are
valid, otherwise zero; it is not invented confidence. `cellAreas` explicitly supplies the
uniform 3,600 projected m² cell areas; no geodesic-area behavior is claimed. Source-valid area
includes zero-weight observations, while statistical support area is reported separately.
The categorical minority output now reports the least-present numeric NIR interval, excluding
absent intervals and disclosing the contributor's tie rules. Report:
`/tmp/luma-showcase-overhaul-20261025/pixel-boundary-paired-numerics-wave10-rerun/report.json`.
The extension exposed a stale-summary UI race when a support change was immediately followed
by a date change. The source now invalidates the summary on support/probability changes;
that presentation guard awaits the next bundle, while the settled GPU numerical outputs passed.

`spectral-evidence` passed actual wave9 GPU comparisons for both acquisitions, all six indices,
SAVI L=0/1, and all three mixture constraints on both dates. Feasible constrained solutions
were compared with independently enumerated active-set optima, without asserting that finite
iterations recover exact optimal coefficients. No browser/GPU errors occurred. Report:
`/tmp/luma-showcase-overhaul-20261025/spectral-numerics-wave9/report.json`.
New source controls expose denominator epsilon 1e−6/0.01/0.1 and explicit output clamps
none/[−1,1]/[−0.25,0.25]. The narrow cap is labeled an information-loss sensitivity experiment,
not a correction. Rebuilds invalidate old analytical displays. Independent full-crop validity
counts and selected values are recorded in `spectralIndexOptions`; these controls still await
GPU verification. For September, epsilon0.1 reduces the six-index valid total from18,294 to17,489.

`spectral-texture` is planner-reviewed and scoped story/Biome checks pass. It now exposes
selected-date smoothing, named and generic signed derivatives, all seven GLCM metrics and
all eight focal outputs, with meaningful radius/direction/quantization/symmetry/support options.
Rectangular 5×1/1×5 signed convolution and explicit hypothetical constant borders extend the
same inquiry. Detail versus theoretical texture color domains share the map/legend helper;
only display saturation changes. Compile-option guards prevent old acquisition results from
being relabeled while a replacement graph is pending. Production GPU checks remain queued.

`grid-alignment` implements controlled periodic and masked registration, bilinear application
of estimated corrections, valid-support/max/RMS diagnostics, and real CRS-aware resampling.
Planner review corrected the distinction between exact Mercator analysis coordinates
and deck's approximate local display offsets. The destination analysis affine is [60,0,−1920,0,−60,1920],
centered at UTM [666640,4454120]; the independently projected source probe is recorded for all eight
date/interpolation/validity-policy variants. GPU support counts are audited separately from selected
values because float32 projection can move footprints across missing-data boundaries.

`missing-observations` is source-ready after planner review. It withholds 132 isolated July cells
or one 9×9 July block; the separately retained measured values supply truth. September's 1,047
missing cells have no truth score. All three donor rules and radii 2/4/8/12 are implemented;
independent NumPy references are recorded for 36 combinations. The separate native GDAL comparison is retained, with its floating-point tie distinction explained below.
Grayscale square/cross morphology, reconstruction dilation/erosion with convergence, binary
support-hole filling at 4/8 connectivity, and contour cases/geometry are actual GPU paths. Review
fixed a contour allocation that used four rather than sixteen bytes per segment. A browser script
checks support, held-out RMSE/MAE, selected predictions, enclosed holes, reconstruction status and
three contour levels; actual GPU verification remains pending. `GPURasterDilation` and Erosion are
thin wrappers fixing the generic Morphology operation, which this scene executes directly.

The pixel quantity-contract extension executes `GPURasterFractionalZonalStatistics` twice per
acquisition: intensive measured reflectance and extensive per-cell integrals produced on the GPU
by multiplying reflectance by 3,600 m². Both paths should recover the same fractional integral.
Its area-weighted mean remains in reflectance units; the extensive mean is not presented as a
physical density. This is a numerical support contract, not an energy or biomass estimate.
These summaries always use source-valid support, separately from the distribution's paired-date
observation weighting. The existing independent polygon-clipping oracle supplies integral, mean
and observed-area references; `check-pixel-boundaries-browser.mjs --quantity` adds all eight cases.

`tiled-observations` is published source-ready after two CLI passes and planner review. It
executes real reader/cache/halo/core ownership, four Gaussian mosaics, global owned-source
summaries, stable-extent histogram replay, direct/cascaded weighted overviews, categorical
nearest/mode reduction, and half-cell-shift tiled/untiled reprojection. Planner review replaced
256 row-copy kernels with four owned-core kernels, fixed runtime NaN construction, added
explicit support mismatches, and protected empty overview diagnostics. Independent NumPy
references and `check-tiled-observations-browser.mjs` are ready; no GPU acceptance yet.
This decoded in-memory experiment is not a remote streaming or COG benchmark.

The display/transect/transport extension of `spectral-texture` is reviewed and source-ready.
It adds linear/gamma/CDF display transforms, actual GPU transect samples, and buffer↔texture
roundtrips in R/G/B/A channels. Planner review fixed the transect's northing sign and selected
texture-channel readback. The story was consolidated to six steps. Independent references
cover 36 contrast settings, 24 profiles, and both source populations; eight channel comparisons
are prepared. The gamma exponent is 1/gamma. Scientific derivatives and focal statistics still
use calibrated source values. No GPU acceptance is claimed yet.

`terrain-boundaries` is reviewed and source-ready: closed-window upper-excursion peak
prominence with infinite elders; D8 drainage/catchment/distance/geometry status; and four
owned terrain tiles with D8 or D-infinity/Freeman branching exchange. Comparison requires
resolved status in both tiled and reference outputs. Original block-mean elevations supply
branch slopes; filled/resolved D8 directions supply the documented fallback. The exact64²
Float32 DEM CPU union-find reference finds89finite peaks,1elder, and maximum finite prominence
1278.421875m at row64,column58 (one-based). Browser acceptance remains pending.

The finite recipe closure is also reviewed and source-ready. `missing-observations` composes
fill with optional median rank filtering (ignore/propagate, radius1/2), retaining independent
original-valid/filled/filtered provenance and held-out error. `spectral-evidence` adds whole-cell
unweighted zonal distributions, sorted grouped distributions, and paired regional change on
four32²quadrants of the actual64²source. `habitat-connectivity` adds raster circuit maps and
independent dense/sparse16²subset solves with explicit WorldCover resistance scenarios,
4/8neighbors and physical edge lengths. Planner review fixed the circuit readback revision
offset, repeated-rebuild ownership, nonconvergence map gating and map/legend agreement.
Independent NumPy/source oracles exist; all these added recipes await production GPU checks.

### Wave11 numerical findings and remaining acceptance

All runs used the parent-controlled production snapshot on port5313. The spectral extended
suite passed all18epsilon/clampcases and its earlier index/mixture assertions. Grid registration
passed exact known shifts,4096periodic/3782masked aligned counts and zero reported residuals.
CRS probes and policy monotonicity passed, but five option cases (four distinct support decisions) differ by one from
Rasterio; the exact destination-coordinate/support-boundary diagnosis remains open. This is
not recorded as full reprojection acceptance.

The pixel run passed four September distribution zones, eight quantity contracts, eight paired
support cases and raw/meaningful coverage diagnostics; its final remapping repetition stopped
at a hidden Controls tab selector after Story navigation. The selector is corrected for the
next slot. The12remapping combinations already passed wave10. Quantity valid-area tolerance
is distinct from coverage clipping: the contributor sums fractions sequentially in float32.
The July rectangle has2227observed tuples,97partial fractions; independent row-major float32
addition yields7841985.5m² versus exact7842000m², matching the GPU readout. Reversing addition
gives7842014m². An order-independent bound counts partial-addition half-ULPs, exact-one binade
crossings, input fraction representation, final multiplication and displayed rounding. The
result is46m² for this rectangle and3–62m² across cases. The tighter coverage/moment assertions
are unchanged. These are numerical diagnostics, not physical area accuracy claims.

Missing-observation verification passed eight initial fill cases then exposed a donor tie
convention in block/radius4. At zero-based cropcells(33,33)and(34,34), exact integer squared
comparison preserves the first donor while nativeGDAL3.9.3's `QUAD_CHECK` compares a squared
distance to a stored square-root distance squared. Since `sqrt(10)^2>10` and `sqrt(5)^2>5`
in float64, a later equal-distance donor can replace the earlier one. Independent enumeration
of the stated exact-squared rule reproduces GPU MAE0.007641607/RMSE0.009554497; emulating the
native roundtrip reproduces GDAL MAE0.007681987/RMSE0.009608352. The story explicitly calls
this GDAL-style and states the tie rule. Native results remain recorded separately; no kernel
or arbitrary tolerance change was made. The corrected mathematical oracle and remaining
morphology/reconstruction/holes/contours checks await the next numerical slot.

Normal readout rows have no narrative `data-readout` attribute in wave11. The test-only helper
`earth-browser-readouts.mjs` matches exact metadata labels and annotates those DOM values;
this corrected initial selector timeouts without altering app behavior. Parent wave12 adds
stable dedicated readout attributes for future harnesses. All CLI lanes exited successfully;
no worker GPU/browser run occurred outside the parent's reserved slot.

References: [GDAL3.9.3 QUAD_CHECK source](https://raw.githubusercontent.com/OSGeo/gdal/v3.9.3/alg/rasterfill.cpp),
[GDAL overview documentation](https://gdal.org/en/stable/programs/gdaladdo.html),
[OGC COG1.0](https://docs.ogc.org/is/21-026/21-026.html),
[scikit-image exposure](https://scikit-image.org/docs/stable/api/skimage.exposure.html),
[Circuitscape options](https://docs.circuitscape.org/Circuitscape.jl/stable/options/).


### Exact support diagnosis prepared after wave12b

`grid-alignment` now accepts the opt-in query `?gridSupportDiagnostics`. It uses the same
compiled projection and contributor affine arithmetic to read actual GPU source-cell coordinates,
and emits those with the already-read-back eight validity masks. Normal views allocate no
additional diagnostic buffers. `check-grid-alignment-browser.mjs` saves the resulting
`gpu-support.json`; `compare-grid-alignment-support.py` compares every cell with independent
Rasterio coordinates and independently samples the source mask at the actual GPU coordinates.
No projection, count tolerance, or public readout changed. Scoped TypeScript and Biome checks
passed, and the raster WGSL checker passed 697 templates. The comparator's CPU fixture check
passed; real GPU diagnosis remains pending the next reserved slot and production build.

The current fine-option decisions are in `earth-fine-options.md`. Existing curvature controls
cover all three estimators. Bilinear extrema footprints are fixed by ray-sampling support.
Signed zero-sum derivatives retain unnormalized coefficients; positive smoothing is normalized
through the existing weighted-neighborhood path. The generic normalized-convolution variant
and absolute observer input are not claimed as explicitly demonstrated. The existing median
preparation receives a bounded clamp/nodata edge comparison; no additional scene is planned.

Median preparation now exposes clamp versus outside-as-missing borders in the existing fill
step. Ignore omits unavailable neighbors; propagate rejects affected neighborhoods. The
tracked `scripts/surface-preparation-reference.py` independently rebuilds nearest original-donor
fills and upper-median order statistics from the pinned source, reproducing all 48 records
exactly. It updates only its own oracle key. Actual GPU acceptance is still pending.

Every queued earth numerical harness saves served `build-evidence.json` before browser launch
and compares its SHA after browser close, including failing runs. Reports carry the initial and
final bundle hashes and `bundleUnchanged`; a changed build forces failure. A read-only harness
selfcheck confirmed the currently served wave12b hash without starting a browser or GPU task.

### Final source freeze and wave12b review

The finite topology closure is source-ready in `tiled-observation-topology.ts`, called by the
existing tiled ownership step. A 64 × 64 source focus (rows 82–145, columns 92–155) is split into
four 32 × 32 owners. Two calibrated NIR masks at 0.2 feed local components, dense labels and
measurements, then separate low/high cross-tile reconciliation. The resulting global component
IDs feed `addRasterRegionGraphTiles` and the categorical map. Every local/global convergence
and capacity status, plus region-graph status, gates the displayed result. Colors repeat modulo
15 and communicate identity, not rank or magnitude. No ecological class inference is made.

Independent whole-window and owned-core flood calculations agree: July has 4,057 valid cells,
109 local regions and 98 global regions, with 97 low/high adjacency pairs and 763 shared faces
(16 across owner seams). September has 1,935 valid cells, 79 local and 66 global regions, and
only one adjacency face: almost all observed cells lie below the chosen threshold. That date
is a support contrast, not evidence of a richly partitioned field. Production GPU acceptance
of the new topology and fine-option extensions remains pending.

Wave12b completed guided desktop/mobile checks for nine of the eleven earth inquiries:
terrain-conditioning, habitat-connectivity, composite-evidence, ocean-seasonality,
spectral-texture, pixel-boundaries, spectral-evidence, grid-alignment and missing-observations.
`terrain-boundaries` failed because recipe outputs lacked hardware-copy usage; its scene-local
publisher now copies exact stored words through a storage kernel, including float payloads.
`tiled-observations` failed on an undefined WGSL `isFinite` call; its QA-preserving predicate is
now explicit. These two source fixes await the next production run. No core API was changed.

Final scoped checks passed: showcase TypeScript, raster story and strict cartography checks,
Biome, 798 raster shader templates and 261 hydrology shader templates. Runtime source is frozen
for the parent-controlled integration build. The next numerical batch records the served build
hash before and after every suite, beginning with the exact grid-support diagnostic.

### Wave13 grid precision diagnosis

The short GPU diagnostic ran against unchanged bundle
`658033867c43774cdb5ca9b9d9fa4aff28575cb70fc8b6d190d343ac32d0660d` and released the browser before CPU
analysis. Both known-shift registration checks pass. Independent scalar sampling at actual GPU
coordinates reproduces all 32,768 validity decisions exactly. Rasterio coordinate/support
equivalence does not pass: explicitly opted-in Float32 analytic projection stages produce a
maximum observed axis difference of 3.239092 m on 60 m cells. September nearest has 28 spatial
mask differences despite matching aggregate counts. Affine reconstruction contributes less than
0.00052 m, isolating the discrepancy upstream. No tolerance was loosened and no core source was
changed. See `earth-grid-support-diagnostic.md` for exact pixels, masks and acceptance distinction.

The grid default has now been corrected in source: a bounded adaptive plan evaluates a binary64
analytic provider with double-single intermediates, while the raster affine and final output use
local UTM offsets. Independent CPU review of all 4,096 centers gives a maximum simulated final
Float32/affine Euclidean error of 0.000655314 m and exact masks in all eight cases. The one-patch
sampled fit certificate is reported separately. Scoped source checks pass; actual GPU acceptance
awaits wave14. This supersedes treating the earlier metre-scale default as an acceptable limitation.

### Wave14 priority acceptance and remaining tiled reference correction

The corrected grid passes on production wave14: 0.000763260 m maximum Euclidean coordinate error
against independent Rasterio at all 4,096 centers, exact support masks in all eight cases, and
both controlled registration tests. Terrain-boundaries passes the independent prominence oracle
and three tiled/whole-window GPU flow comparisons. Those flow comparisons use shared resolved
support: D-infinity still reports 38 unresolved cells and MFD 319, so no complete resolution or
independent CPU flow validation is claimed. The new tiled component and region-adjacency pipeline
passes both acquisition dates against independent whole-window/core flood-fill and edge counts.

The tiled baseline run passes 11 source population/histogram records, then fails its first blur
support assertion because the NumPy reference incorrectly filled missing center cells. The
public neighborhood contract preserves invalid centers at every separable pass. The corrected
tracked `make-tiled-observations-oracle.py` tests a constant 5 × 5 field with one missing center:
ignore-renormalize retains that hole and 24 finite cells; propagation produces the expected
3 × 3 missing footprint and 16 finite cells. Existing tolerances are unchanged. The failed report
is preserved and the remaining 47 records await rerun. Reader help now states the center rule.

`earth-wave14-priority-acceptance.json` durably records complete reports, test counts, numerical
contracts, limitations and hashes. Every browser in this batch closed before the GPU slot was
released, and every suite verified the same unchanged production bundle.

Wave14 topology screenshots exposed a separate rendering-contract error despite the passing
numeric counts: the category renderer indexes raw uint32 words, while the scene had converted
labels and missing values to float32. Missing cells therefore appeared purple and label colors
were not the declared IDs modulo the palette size. Both categorical overview and topology display
buffers now retain uint32 IDs with `0xffffffff` as missing, and their layers declare uint32 format.
No analysis or numeric readback changed. TypeScript, scoped formatting and WGSL checks pass;
wave15 must confirm correct categorical identity and transparent missing areas visually. The
acceptance packet distinguishes numeric success from this pending visual correction.

The bounded category-format audit found the same display issue in terrain-boundaries’ stream-order
product. Its display kernel and layer now use uint32 orders and the missing sentinel; numeric
routing and prominence are unchanged. Other new earth category products already use uint32.
The emitted wave15 tiled chunk includes its corrected display formats, while the terrain chunk
still contains the older float display. Therefore corrected tiled visuals can be accepted on
wave15, but corrected terrain stream colors require wave16. The durable packet now marks both
numeric passes separately from their pending visual checks.

### Wave16 acceptance, explicit failures and bounded corrections

The frozen wave16 bundle `31e545703f8e9c2990ab47eb3c8ec879c9e3532b9a386c16da552c59782e7eaf`
passes nine complete suites with 266 records: tiled processing 58, tiled topology 2, terrain 4,
fractional boundaries/remapping 33, missing observations 49, texture 46, patch perimeter 6,
surface preparation 48 and regional statistics 20. Full attempt reports, exact commands, hashes,
reference methods and tolerances are retained in `earth-wave16-acceptance.json`. Earlier driver
failures are retained separately from successful retries. Parent and earth planner independently
reviewed the corrected topology palette/transparent outside support and terrain stream-order
colors against their legend. Numeric flow limits from wave14 still apply.

Two numerical suites remain incomplete. Spectral display passes six cases before equalization
uses a hidden gamma setting: the control was disabled for equalization even though the contributor
applies gamma after the CDF. The observed `0.203825` is exactly explained by the retained gamma0.5
applied to the independently expected gamma1 value `0.349707314`. The scene now enables gamma for
both nonlinear modes; the existing oracle and tolerance stay unchanged. Remaining display,
profile and transport cases await the corrected production bundle.

The resistance suite passes four mild scenarios, then correctly withholds the strong four-neighbor
scenario at a current-conservation residual of `2.24e-4`, above the requested `1e-4`. Independent
Float32 CPU review reproduces recursive-residual drift: the solver stops its recurrence while the
recomputed residual still fails. A fixed warm restart of the existing PCG solver reserves 64
iterations inside budgets of at least 512; smaller budgets are unchanged. The strict final
certificate and configured total budget remain unchanged. This is a fixed restart strategy,
not the adaptive algorithm discussed in the cited residual-gap literature. See
`habitat-circuit-cpu-review.md` and its 56-case independent packet. The recipe-only correction and
focused 128/512/1024-budget GPU regression await actual execution.

The original missing-observation harness omitted six grayscale morphology cases. The amended
harness exposes a reference error at cross radius2 closing: the public cross support uses
Manhattan distance, whereas the old oracle included only row/column arms. The new tracked
`make-missing-morphology-oracle.py` verifies a radius2 diagonal fixture and updates only morphology
and reconstruction references, preserving the native GDAL and deterministic-tie evidence. The
reader help now describes the diamond at radii above one. The failed report and corrected
references are retained in `earth-morphology-oracle-review.json`; all six morphology cases still
require a successful rerun at the unchanged chart-precision tolerance.

### Wave17: morphology accepted; remaining numerical corrections

The independent Manhattan-support correction passes all 55 missing-observation records on
wave17 (`56f7715e655d8fe93b59b426b475b87848b7947fb8e4d454faff5353908b4a02`), including the six
morphology configurations and their 24 scalar comparisons. All 36 display transforms now pass,
confirming the gamma-control correction. The remaining profile failure is precisely a support
boundary issue: the authored last vertex at 7290 m is cell-center121, but multiplying by the
Float32 reciprocal of60 produces121.000007629. Under strict bilinear support this introduces a
missing east neighbor;164 samples are emitted but only163 are finite. The bounded caller fix
uses exact half-integer source-cell coordinates and unit cell size for sampling, retaining the
metric map line and converting every output distance/length back by60. No profile core or
reference tolerance changes are involved.

The fixed restart alone did not pass actual GPU validation. Existing advanced/golden44 tests
and the lower-budget128/256 cases pass; the corrected512/1024 cases remain unaccepted. The
independent operator review finds that the default PCG selects subgroup-row accumulation while
the final certificate uses sequential CSR rows. A Float32 subgroup-tree model reproduces the
observed1.14782e-4 certificate and shows a real physical sink residual above1e-4. Both phases of
corrected solves now explicitly select the existing scalar-row strategy; smaller budgets retain
their original adaptive strategy. Strict certificate and0.2 absolute reference bounds are
unchanged. The next focused run will capture final potentials/conductances for an independent
Float64 edge-current audit. All partial reports and failed focused logs are durably preserved in
`earth-wave17-acceptance.json`; CPU agreement has not been promoted to GPU acceptance.

### Wave18: profile and transport accepted; circuit still withheld

All 68 display/profile/transport records pass on wave18
(`942c61078b2db07062a259c0bc88889cf064791c8126a6cbca07d7cca40a8307`). These comprise36 display
transforms,24 listed profile cases (including six explicitly identified nearest-policy duplicates)
and eight exact channel roundtrips. Source-cell sampling restores the original independent
center-based support expectations without epsilon/clamping; reported lengths remain metres.
The final transect map was visually reviewed. `earth-wave18-acceptance.json` retains the report,
reference contracts and test command.

The resistance correction remains unaccepted: the focused set still has46 passes and two
failures, and the browser still passes only four mild cases. Explicit scalar-row produces the
same1024-budget residual as wave17, so the earlier assumption that default subgroup accumulation
caused the observed error is not established for this actual device. Final4096 GPU potentials
and8192 conductances are retained in `habitat-circuit-wave18-gpu-capture.json` for an independent
physical current-balance audit. The original1e-4 certificate and0.2 absolute reference check
remain unchanged; no failed strong scenario is presented as accepted.

### Final circuit scope: restore the original strict publication gate

The wave18 physical audit measured a real Float64 edge-current residual of
2.48359138e-4 at sink2084 and an effective-resistance error of −0.22495486 using the
actual GPU potentials. All8192 conductance slots matched the independent reference.
Neither the fixed warm restart nor explicit scalar-row accumulation established a
robust correction. The earlier subgroup-causality hypothesis is withdrawn; the
actual core test device did not enable optional subgroups. These experimental
results are failures, not evidence that the strong scenario converged.

Only the experimental sparse-recipe restart, explicit strategy selection, and
phase provenance were removed. The recipe is byte-for-byte identical to its
original tracked version: one bounded solve followed by its unchanged strict
current-balance certificate. Generic PCG and SpMV work was untouched. Solver
redesign is outside this showcase integration. The rejected delta is retained in
`habitat-circuit-rejected-restart.patch`; the provisional convergence-demanding
regression and exact source-derived fixture are archived as
`habitat-circuit-experimental-regression.ts.txt` and
`habitat-circuit-residual-fixture.json`. They are historical experimental artifacts,
not executable acceptance tests. Existing advanced/golden44 tests already include
invalid, disconnected, and exhausted sparse-solve withholding, so no redundant
replacement test was added.

The reader now sees explicit conditioning and publication semantics: a failed
certificate withholds both computed maps and accepted resistance/energy; withheld
is not zero connectivity. The main solve spans64×64 cells. Its terminals are
selected within the central16×16 subset, and the separately labeled dense/sparse
comparison solves only that subset. Success of the small comparison cannot certify
the larger solve.

The final browser verifier exercises all eight authored combinations at the
unchanged1024 budget and1e-4 tolerance, requires all four previously accepted mild
cases to remain accepted, and measures strong-case statuses without predeclaring
them. Accepted resistance/energy retain the original independent bounds
`reference × .001 + .003`; publication diagnostics retain their original displayed
certificate bound. Withheld cases must have diagnostic-only residuals and no
accepted resistance/energy or circuit chart. Analysis-canvas screenshots at both
opacity endpoints for potential and current must be identical when withheld;
accepted maps must respond to opacity as a positive control. HTML legends and
MapLibre furniture are hidden only during isolated canvas screenshots. A final
return to mild must restore accepted scalars and visible maps. These checks verify
publication policy and displayed certificates, not physical numerical agreement of
rejected solutions. The final measured eight-case outcome remains pending until
the coordinated next production build and GPU run.

### Wave19 final circuit acceptance and withholding

The final eight-case browser suite and return-to-mild check pass on frozen bundle
`f4742c261276e85b022651df7f3c72f419a7f6d68300c30803839629c67b3675`; the served fingerprint
was unchanged and the browser closed cleanly with no errors. All four mild cases
match independent resistance/energy references. All four strong cases are withheld
under the unchanged1e-4 certificate: reported residuals are2.24e-4/2.20e-4 for
four-neighbor and1.62e-4/1.63e-4 for eight-neighbor (without/with water barrier).
These are four numerical acceptances plus four publication-policy checks, not eight
accepted numerical solutions. Each rejected potential/current product contributes
no rendered pixels across opacity endpoints; only scenario terminals remain. No
accepted full-crop resistance, energy, or circuit chart is published. Returning to
mild restores accepted scalars and maps.

The strong four-neighbor16×16 dense comparison also fails its certificate, so the
small comparison correctly withholds energy; strong eight-neighbor small comparisons
pass independent energy checks. The first verifier attempt assumed that comparison
always converged and is preserved as a failed harness-contract attempt. The corrected
verifier conditions its numerical checks on actual acceptance without changing bounds
or source math. `earth-wave19-acceptance.json` embeds both complete reports and their
hashes, exact command, actual counts, screenshot hashes and remaining limitations.
The experimental solver correction remains withdrawn; the scientific limitation is
explicitly retained in the showcase.

### Wave22 final missing-mask display correction

Final screenshot review found the missing-data orange overlay painted its NaN
background: the mask colormap uses a nonzero test before the continuous-ramp
nonfinite check. The scene now supplies finite0/1 display masks, including enclosed
gap output and its empty fallback. Zero is transparent; numerical rasters and
algorithms are unchanged. Independent source-crop support is81 central block cells,
132 isolated cells, and1047 September gaps.

On frozen bundle`7fc7fb3a993176cdbba18b723eb2fbc4719b53090829c7a1b48b9406db57278e`, all55
original numerical records pass unchanged, as do the six-step/mobile tour and an
isolated-canvas orange-area check against81/4096. Corrected desktop/mobile images
show the central orange block over measured NIR; the September topology image
shows enclosed orange holes and transparent boundary-connected gaps. The earlier
whole-crop orange screenshot remains recorded as a visual failure. Complete reports,
hashes, commands and distinct numerical/display tolerances are embedded in
`earth-wave22-acceptance.json`. All browsers closed before GPU release.
