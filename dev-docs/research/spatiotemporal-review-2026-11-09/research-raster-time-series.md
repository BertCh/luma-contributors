# Spatiotemporal raster, EO time series, gridded climate data and DGGS: directions for a WebGPU analysis library

Research date: 2026-11-09. Scope: 2022-2026 methods and tooling, filtered against the library's constraints
(float32, bounded or compile-time-unrolled loops, no data-dependent iteration without CPU readback,
a few hundred MB resident, no file readers, pinned reference oracles).

Citation rule used below: a URL next to a claim is a page I opened during this research. Claims I could
see only in search-result snippets, or not at all, are marked **UNVERIFIED**.

---

## 0. What the library already has that matters here

Inspected in the working tree (`modules/analysis/src`):

- `raster-analysis/remote-sensing/raster-kernel-utils.ts` exposes `getRasterLeastSquaresWGSL(rows, columns, variableColumns)`,
  a per-pixel Householder-style QR least-squares routine in WGSL with compile-time array sizes. This is the
  key building block for every harmonic / CCDC / BFAST-style method below.
- `gpu-raster-seasonal-break.ts`: harmonic or dummy season + trend + one BIC-selected level shift, fixed-history
  monitoring with `threshold` and `minimumConsecutive`, slices capped at 256 (`validateRasterCount(sliceCount, 256)`),
  regular time slots only.
- `gpu-raster-temporal.ts`: regular-time composites and Savitzky-Golay.
- `change-detection/`, Theil-Sen / Mann-Kendall, Welch.
- `gpu-terrain/terrain-illumination/`: horizon sweep, cast shadow, solar position, and `gpu-solar-irradiance.ts`,
  which already integrates insolation over time and accepts an optional `skyViewFactor` input.
- `raster-pyramid`, `gpu-raster-registration.ts` (phase correlation), `particle-advection`, watershed segmentation.

So the main gaps are: **irregular acquisition dates**, **multi-band recursive monitoring (CCDC/COLD family)**,
**penalised smoothers and phenology metrics**, **multivariate compositing**, **multiple-testing control**,
**field decomposition (EOF)**, **climate indices**, and **compute over embedding rasters**.

---

## 1. Ranked candidates (analyst value over effort)

| Rank | Candidate | Value | Effort | Risk | Oracle (licence) |
|---|---|---|---|---|---|
| 1 | Embedding-raster toolkit (dequantise, cosine similarity search, year-to-year change, k-means, kNN / linear-probe inference, PCA-to-RGB, correct pyramids) | Very high | S-M | Low | NumPy / scikit-learn; AlphaEarth dequantisation formula from Google's readme |
| 2 | Benjamini-Hochberg FDR field significance over any p-value raster | High | S | Low | SciPy `false_discovery_control` / statsmodels (BSD) - UNVERIFIED versions |
| 3 | Irregular-date harmonic regression + CCDC/COLD-style multi-band monitoring | Very high | M-L | Medium | pyxccd (Apache-2.0, C core) |
| 4 | Weighted Whittaker smoother + HANTS, with fixed robust-reweighting passes | High | S | Low | phenofit `smooth_wWHIT`, GRASS `r.hants`; own SciPy sparse solve |
| 5 | Phenology metrics on a smoothed curve (TIMESAT-style SOS/EOS/peak/amplitude/integrals) | High | S-M | Medium (oracle licence) | PhenoloPy (Apache-2.0, UNVERIFIED) or own; TIMESAT is research-only licence |
| 6 | Multivariate compositing: medoid, geometric median (Weiszfeld), best-available-pixel scoring | High | S | Low | hdmedians (Apache-2.0) |
| 7 | Climate indices: SPI/SPEI, consecutive dry/wet days, heat-wave / spell runs, degree days, climatology and anomaly | High | S-M | Medium (stack depth, distribution fitting) | xclim v1.0.0 (Apache-2.0) |
| 8 | EOF/PCA (method of snapshots), varimax rotation, per-pixel correlation and lagged correlation maps | Medium-high | M | Low-medium | xeofs (MIT) |
| 9 | BFAST Monitor (MOSUM on a stable-history harmonic fit), irregular dates | Medium | M | Low | bfast R (GPL, oracle-only) |
| 10 | Precipitation / field nowcasting by optical flow + semi-Lagrangian extrapolation | Medium | M | Medium | pysteps (BSD-3) |
| 11 | DGGS on GPU: HEALPix nested / A5 point-to-cell and cell aggregation; DGGS-JSON zone-data output | Medium | M | Medium | healpix-geo, A5 (Apache-2.0) |
| 12 | Feature (object) tracking through time: threshold + label + frame-to-frame overlap linking | Medium | M | Medium | tobac (BSD-3) |
| 13 | LandTrendr annual segmentation | Medium | M-L | High (oracle) | Only Earth Engine (`ee.Algorithms.TemporalSegmentation.LandTrendr`) |
| 14 | EWMACD control-chart monitoring | Low-medium | S | High (oracle) | Earth Engine only |
| 15 | STARFM-family spatiotemporal fusion | Low-medium | M | Medium | cuSTARFM (GPL-3.0, oracle-only) |

Rationale for the order: rank 1 and 2 are cheap and unlock a lot (embeddings are the fastest-growing EO data
product of 2025-2026; FDR makes the existing Mann-Kendall / Welch / correlation outputs publishable). Rank 3 is
the single most-requested EO time-series capability but needs irregular-date plumbing first, which ranks 4-9
then reuse.

---

## 2. Time-series change detection and monitoring

### 2.1 CCDC -> COLD -> S-CCD (per-pixel recursive harmonic regression)

**Question answered:** when (to the observation) did each pixel's multi-band spectral behaviour depart from its
seasonal-trend model, and what are the per-segment coefficients (usable as classification features)?

**Primary sources**
- CCDC: Zhu & Woodcock 2014, RSE 144:152-171 (citation and parameters from Earth Engine docs:
  https://developers.google.com/earth-engine/apidocs/ee-algorithms-temporalsegmentation-ccdc). Earth Engine defaults:
  `minObservations` 6, `chiSquareProbability` 0.99, `minNumOfYearsScaler` 1.33, `lambda` 20 (LASSO, 0 = OLS),
  `maxIterations` 25000, `dateFormat` 0 = Julian days / 1 = fractional years / 2 = Unix ms (same URL).
- COLD: Zhu et al. 2020, RSE 238:111116. Paper PDF opened at
  https://gerslab.cahnr.uconn.edu/wp-content/uploads/sites/2514/2021/06/cold_2020_zhu.pdf. Key settings from the paper:
  initialisation needs >= 12 clear observations, >= 1 year span, largest gap < 1 year; 4 / 6 / 8 coefficients by clear
  observation count (6 for 18-23, 8 for >= 24, i.e. k = 1/2/3 harmonics); model updated for every observation;
  LASSO lambda 20; change confirmed by 6 consecutive anomalies at change probability 0.99 (chi-squared) normalised by
  temporally adjusted RMSE and a minimum RMSE, plus mean included angle < 45 degrees; reported omission 27 %, commission 28 %.
  Bands Green, Red, NIR, SWIR1, SWIR2 (same PDF).
- S-CCD: Ye, Rogan, Zhu & Eastman 2021, RSE 252:112167 - a state-space (Kalman) variant that avoids refitting on
  new data. Implementation and citation confirmed at https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd.
  Reported accuracy (F1 0.793 vs COLD 0.789, 126 vs 166 day lag) and "4.4x faster" come from search snippets only - **UNVERIFIED**.

**Reference implementation (oracle):** pyxccd, Apache-2.0, "core written in C", functions `cold_detect`,
`cold_detect_flex` (arbitrary bands), `sccd_detect`, `sccd_update` (https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd).
Self-reported: 40-year Landsat tile, ~1 h per-pixel COLD and ~0.5 h S-CCD on 200 EPYC cores (same URL). Docs note S-CCD
does not initialise with <= 18 observations or < 1 year, and that `lam=0` (OLS) can overfit and create false breaks
(https://pyxccd.readthedocs.io/en/latest/1_break_detection_fire_hls.html). Original MATLAB COLD at
https://github.com/GERSL/COLD (licence not shown on the page - **UNVERIFIED**).

**Parallel formulation under the constraints**
- One invocation per pixel; time loop over the pixel's observations is bounded by the slice count (<= 256), which is
  already the library's cap. Number of segments bounded by a compile-time `maximumSegments` (e.g. 8); outputs written
  into fixed-size per-pixel segment records (start, end, break, change probability, coefficients x bands).
- Irregular dates: pass a per-slice `float32` date array (shared across the tile, or per pixel for multi-path
  stacks) plus the existing `validity` mask. The harmonic design row is built on the fly from the date, so the
  QA mask simply removes rows - the gpu BFAST paper shows exactly this "filter missing values per pixel" pattern
  (Section 8).
- Model update "for every observation": either refit with the existing QR routine on the current window (cost
  O(n K^2) per step; with n <= 256 and K = 8 that is ~16k flops per refit, ~4M per pixel worst-case - acceptable),
  or a Givens-rotation QR update (O(K^2) per step). Avoid normal-equation recursive least squares in float32.
- LASSO lambda 20: replace by OLS / ridge with a documented deviation, or run a fixed number (e.g. 20) of
  coordinate-descent sweeps per fit (bounded, unrollable). The pyxccd warning above means an OLS variant needs
  a stronger stability test to stay close to the oracle.
- Change probability: chi-squared CDF of the sum of squared RMSE-normalised residuals over bands; precompute the
  quantile for (bands, probability) on the CPU and pass as a uniform.
- Tmask (robust IRLS on green/SWIR1) can be a fixed 2-3 reweighting passes.

**float32 conditioning:** never use Unix milliseconds (`dateFormat` 2) or raw Julian days in the design matrix; centre
time on the window midpoint and express in fractional years so the trend column is O(1) and harmonics are in [-1,1].
With centred time and QR, K = 8 and n >= 12 is well conditioned; normal equations would square the condition number.

**Reuse:** `getRasterLeastSquaresWGSL`, seasonal-break monitor logic, validity masks, `change-detection` outputs.

**Capacity:** 5 bands x 256 dates x 4 B = 5 KB/pixel input -> ~60k pixels per 300 MB (a 245 x 245 tile) plus outputs;
with 3 bands and 128 dates, ~200k pixels. This is a tiling workload, not whole-scene-resident. Outputs per pixel:
8 segments x (4 + 8 x 5) floats ~ 1.4 KB.

**Effort L** for full COLD parity, **M** for a "COLD-lite" (OLS, fixed thresholds, no Tmask). **Risk** medium: matching
pyxccd break dates exactly requires replicating its stability test and RMSE adjustments.

### 2.2 BFAST, BFAST Monitor, BFAST Lite

- BFAST Lite: `bfastlite` (alias `bfast0n`) in the CRAN `bfast` package: `bfastpp` builds trend + harmonic regressors
  (`order` default 3), `breakpoints` from strucchangeRcpp finds multiple breaks, NAs dropped without interpolation,
  break count by BIC / LWZ (https://search.r-project.org/CRAN/refmans/bfast/html/bfastlite.html). Original paper
  Masiliunas et al. 2021, Remote Sensing 13:3308 (search snippet only - **UNVERIFIED** details).
- GPU BFAST Monitor: Gieseke, Rosca, Henriksen, Verbesselt, Oancea, "Massively-Parallel Change Detection for Satellite
  Time Series Data with Missing Values" (https://www.futhark-lang.org/publications/icde20.pdf). Reports GPU 24-48x faster
  than multi-threaded C (32 threads) and up to three orders of magnitude faster than the R implementation; K = 2 + 2k
  regressors with k typically 2-3; missing values as NaN, per-pixel filtering with padded batched K x K inversions;
  processed continental tropical Africa (38,234 images in ~90 h on one GPU); code at github.com/gieseke/bfast (same PDF).
- **Formulation:** BFAST Monitor = one OLS fit on the stable history + MOSUM of residuals over monitoring period with
  a boundary function - fixed work per pixel, ideal for WebGPU. Multi-break BFAST Lite needs the Bai-Perron dynamic
  program: O(n^2) RSS table per pixel. With n <= 128 and segment minimum h, the triangular RSS table is 8k floats
  = 32 KB per pixel - too big for registers, so store in a storage buffer per workgroup tile, or restrict to n <= 64.
  The library's one-break BIC model already covers the most common case; extending to m <= 3 breaks via DP over a
  precomputed RSS table is **M**.
- **Oracle:** bfast R (GPL - oracle-only; licence of gieseke/bfast stated as GPLv3 in a search snippet, **UNVERIFIED**).

### 2.3 LandTrendr

Annual-composite segmentation (one observation per year), parameters `maxSegments`, `spikeThreshold` 0.9,
`vertexCountOvershoot` 3, `preventOneYearRecovery`, `recoveryThreshold` 0.25, `pvalThreshold` 0.1,
`bestModelProportion` 0.75, `minObservationsNeeded` 6
(https://developers.google.com/earth-engine/apidocs/ee-algorithms-temporalsegmentation-landtrendr). Steps: despike,
overshoot vertices then prune to maxSegments + 1, p-value check with Levenberg-Marquardt refit if it fails, pick model,
apply recovery rules, fit other bands to vertices (https://emapr.github.io/LT-GEE/lt-gee-requirements.html). LT-GEE
code Apache-2.0 but runs only inside Earth Engine (https://github.com/eMapR/LT-GEE); the original IDL code is not a
practical oracle. Kennedy et al. 2018 Remote Sensing 10:691 (citation from the LT-GEE page).

**Formulation:** series length ~40, vertex count <= maxSegments + overshoot - fully bounded; the LM refit can be capped
at a fixed iteration count. **Effort M-L, risk high** because the only bit-exact oracle is the hosted Earth Engine
function. Value is real (forest-disturbance community) but behind COLD.

### 2.4 EWMACD

Harmonic fit on training years, residual Shewhart X-bar screening, EWMA with `lambda` 0.3, `lambdasigs` 3,
`persistence` 3, `harmonicCount` 2; output EWMA per image plus coefficients, RMSE, R^2, residuals
(https://developers.google.com/earth-engine/apidocs/ee-algorithms-temporalsegmentation-ewmacd; Brooks et al. 2014,
IEEE TGRS 52(6)). Trivially per-pixel and **S** effort given the seasonal-break code, but the only oracle is Earth Engine,
so **risk high** for validation. Could be offered as a mode of the existing monitor rather than a new operation.

### 2.5 Near-real-time alerts (RADD, DETER-style)

RADD: Sentinel-1, a single observation raises an alert, later observations iteratively update the disturbance probability;
high-confidence > 97.5 % within at most 90 days, low-confidence > 85 %, 10 m pixels, 0.1 ha MMU; data CC BY 4.0
(https://www.wur.nl/en/research/products-services/radd-forest-disturbance-alert). The page itself does not say
"Bayesian"; Reiche et al. 2021 ERL is cited there. The per-pixel probability update is a fixed-cost Bayes rule given
class-conditional backscatter pdfs - **S** to implement as a generic "sequential probability update" over a time stack,
but the value lies in the calibrated pdfs and SAR preprocessing (out of scope). DETER (INPE): not researched to source
level - **UNVERIFIED**.

### 2.6 BEAST / Rbeast (Bayesian model averaging)

Rbeast 1.0.2, GPL (>= 2), `beast`, `beast.irreg` for irregular series, `beast123`
(https://rdrr.io/cran/Rbeast). Sampling-based (reversible-jump MCMC per the 2019 RSE paper - **UNVERIFIED** from source).
**Not recommended**: thousands of random-length model proposals per pixel, divergent control flow, and a float32 RNG
chain would not match the oracle beyond distributional agreement.

---

## 3. Phenology, smoothing, compositing

### 3.1 Weighted Whittaker smoother (rank 4)

phenofit `smooth_wWHIT`: second-order difference penalty, iterative weights via `wTSM` / `wChen` / `wBisquare`,
`lambda = NULL` selects lambda by V-curve; references Eilers 2003 (Anal. Chem.) and Frasso & Eilers 2015
(https://search.r-project.org/CRAN/refmans/phenofit/html/smooth_wWHIT.html). Python vam.whittaker is a Cython
implementation with V-curve (PyPI page did not render; GitHub URL 404 - **UNVERIFIED** licence).

**Formulation:** (W + lambda D'D) z = W y is a symmetric pentadiagonal system: banded Cholesky in O(T) per pixel,
T <= 256 -> ~2.5k flops, held in a private array of 3 x T floats. Irregular dates: use the divided-difference penalty
matrix built from the date vector (shared across pixels). Robust reweighting: a fixed 3-5 passes (unrolled).
V-curve: evaluate a fixed grid of e.g. 16 log-lambda values per pixel and pick the minimum (bounded, ~40k flops).
float32: pentadiagonal Cholesky with lambda up to ~1e5 loses accuracy; scale the system by 1/lambda when lambda > 1 and
cap lambda; validate against float64 SciPy `solve_banded`.
**Reuse:** temporal SG infrastructure, validity masks. **Capacity:** 1 band x 256 dates = 1 KB/pixel -> ~250k pixels
per 300 MB with scratch. **Effort S. Risk low.** Also gives gap filling (weight 0 where invalid) and interpolation to
a regular daily/5-day grid, which then feeds every regular-time operation already present.

### 3.2 HANTS

GRASS `r.hants` (author Markus Metz) implements Roerink et al. 2000: `nf` frequencies, needs valid observations >= 2nf - 1,
`dod` over-determination, `fet` fit-error tolerance with `-l` / `-h` low/high outlier rejection, non-equidistant
`time_steps` supported (https://grass.osgeo.org/grass85/manuals/addons/r.hants.html). **Formulation:** the existing QR fit
with a fixed number of reject-and-refit passes (each pass drops the worst point beyond `fet`). **S** effort once irregular
dates exist; oracle GRASS (GPL - oracle-only).

### 3.3 Phenology metrics (rank 5)

- TIMESAT site: fitting by Savitzky-Golay, asymmetric Gaussian, double logistic; metrics start/end/length of season,
  base, middle, maximum, amplitude, small and large integrals; free for non-commercial academic research
  (https://web.nateko.lu.se/timesat/timesat.asp). The PyPI page did not render; licence change to a proprietary
  research-only licence from v4.2.1 seen in search snippets only - **UNVERIFIED** but consistent with the site.
  **Treat TIMESAT as unusable even as a CI oracle unless the licence is confirmed to allow it.**
- Copernicus MR-VPP ATBD: TIMESAT 4.2 with cubic smoothing spline (p = 1000) on daily PPI; SOS at 25 % and EOS at 15 %
  of relative amplitude; up to two seasons per year assigned to the peak year; gaps > 91 days filled with a base level;
  QA weights 1 / 0.6 / 0.1 / 0; 13 parameters (SOSD, EOSD, MAXD, SOSV, EOSV, MINV, MAXV, AMPL, LENGTH, LSLOPE, RSLOPE,
  TPROD, SPROD) (https://library.land.copernicus.eu/products/MRVPP_MRVPP_ATBD_v2.html). Notably the operational product
  moved away from double-logistic fitting.
- **Formulation:** Whittaker-smoothed curve on a regular grid (3.1) -> per pixel a bounded scan for up to 2 peaks per
  year, threshold crossings by linear interpolation, trapezoid integrals. All O(T), no nonlinear fitting.
  Double-logistic fitting by Levenberg-Marquardt is possible with a fixed iteration cap but is poorly conditioned in
  float32 (6 parameters, exponentials); defer.
- **Oracle:** phenofit (R; licence not shown on the page - **UNVERIFIED**) or PhenoloPy (Apache-2.0 per a search snippet,
  **UNVERIFIED**), or a NumPy reference written to the MR-VPP threshold definitions. **Effort S-M.**

### 3.4 Compositing: medoid, geometric median, best-available pixel (rank 6)

- Geometric median: Roberts, Mueller & McIntyre 2017, IEEE TGRS 55(11):6254-6264; preserves inter-band relationships;
  equivariant under linear transforms (indices computed after compositing equal those computed before); medoid recommended
  when provenance is needed; output is synthetic (https://knowledge.dea.ga.gov.au/data/version-history/dea-surface-reflectance-geomedian-landsat-2.0.0/).
- hdmedians: Apache-2.0, `geomedian` / `nangeomedian` (Cython, float32 or float64), `medoid` / `nanmedoid`
  (https://github.com/daleroberts/hdmedians). Algorithm not named on the page (Weiszfeld variant - **UNVERIFIED**).
- Best-available-pixel: score = w_doy * Gaussian(DOY) + w_year * linear(year) + w_cloud * linear(cloud distance), example
  weights 0.5 / 0.2 / 0.3, cloud distance 10-100 px (Griffiths et al. 2013 as taught at
  https://eol.pages.cms.hu-berlin.de/gcg_eo/04_baps.html). White et al. 2014 adds sensor and scene-cloud scores (search
  snippet only, **UNVERIFIED**).
- **Formulation:** medoid = O(T^2 B) pairwise distances per pixel (T = 64, B = 6 -> 25k flops); geomedian = Weiszfeld with a
  fixed iteration count (e.g. 32) and epsilon guard - bounded and unrollable; quality-score argmax = O(T). Cloud distance
  comes from the existing distance field run on each slice's mask. **Effort S, risk low**, high everyday value (the most
  common first step in any EO workflow).

---

## 4. Space-time field analysis

### 4.1 FDR field significance (rank 2)

Wilks: per-gridpoint testing without multiplicity adjustment overstates results; control the false discovery rate,
which has "only modest sensitivity to spatial autocorrelation" and needs only sorting of p-values
(https://ams.confex.com/ams/96Annual/webprogram/Paper276937.html; BAMS 97(12) 2016 paper itself not opened -
**UNVERIFIED** beyond the abstract).
**Formulation:** p-value raster from the existing Mann-Kendall / Welch / correlation ops -> a 2-pass histogram on GPU
(e.g. 65,536 log-spaced bins of p) -> CPU finds the largest k with p_(k) <= k q / N inside the bracketing bin -> exact
refinement by a second pass over that bin only -> threshold uniform -> mask. Bounded, no full sort. **Effort S.**
Adds "field significant" outputs to every existing trend op; strongest value-for-effort item in this report.

### 4.2 EOF / PCA, rotated EOF, correlation and teleconnection maps (rank 8)

xeofs: MIT licence; EOF/PCA, varimax-rotated EOF and MCA; randomized SVD; xarray + Dask; Rieger & Levang 2024,
JOSS 9(93):6060 (https://github.com/xarray-contrib/xeofs; https://xeofs.readthedocs.io/en/latest/).
**Formulation (method of snapshots):** with T <= 256 time steps and N pixels, compute the T x T covariance
C = X'X / N on GPU as a reduction over pixels (each workgroup accumulates partial T x T sums; 256^2 = 65k floats,
64 KB - fine), read back C, eigen-decompose on CPU (Jacobi, 256 x 256 is milliseconds), upload the top-k eigenvectors,
and project on GPU to get spatial EOF maps. Area weighting by sqrt(cos lat) applied at load. Varimax on the k x k
loadings is a CPU loop. Per-pixel correlation / lagged correlation against a supplied index (e.g. Nino3.4) is one
dot-product kernel per lag (lags bounded). float32: accumulate the Gram matrix with Kahan or pairwise partial sums,
remove the temporal mean first. **Effort M.** Capacity: 256 dates x 1 M pixels = 1 GB - too much; at 256 dates use
~250k pixels, or 64 dates x 1 M pixels; streaming chunks over pixels works because C is additive.

DMD (PyDMD, MIT, many variants including BOPDMD, HODMD, mrDMD; https://github.com/PyDMD/PyDMD) shares the same
snapshot SVD; the extra step is a small CPU eigenproblem. Offer only as a later extension; analyst demand is low.

### 4.3 Climate indices (rank 7)

xclim latest release v1.0.0 (tag from the GitHub releases API), Apache-2.0, "over 150 climate indicators"
(https://github.com/Ouranosinc/xclim). Source of v1.0.0
(https://raw.githubusercontent.com/Ouranosinc/xclim/v1.0.0/src/xclim/compute/_agro.py):
`standardized_precipitation_index(window=1, dist in {"gamma","fisk","genextreme","lognorm"}, method in {"APP","ML","PWM"}
(default "ML"), cal_start, cal_end, zero-inflation options prob_zero_interpolation / plotting_position_zero)`; output limited
to +/-8.21; degree days via `(tas - thresh).clip(min=0)`.
**Formulation:**
- Degree days, consecutive dry/wet days, spell / heat-wave run lengths, frost days: a single sequential scan per pixel over
  time (bounded by T) - **S**. The 256-slice cap is the issue: daily data for 30 years is ~11k slices. Design these as
  **streaming accumulators** (state per pixel: current run, max run, count, sum), fed one chunk of slices per dispatch,
  so stack depth is unbounded while memory is O(pixels x state).
- Climatology and anomaly: per-pixel per-calendar-slot mean / quantiles, also streaming (sum, sum of squares) for means.
- SPI: per pixel and per calendar month, fit gamma to ~30 calibration values. Use the APP (Thom approximation, closed
  form) or PWM (L-moments, closed form) method so no iterative ML is needed; ML can be a fixed 8-10 Newton steps on the
  shape parameter. Then gamma CDF -> normal quantile (both need float32 series approximations; regularised incomplete gamma
  via series/continued fraction with fixed terms). SPEI with fisk (log-logistic) via PWM is closed form. Requires a PET input
  (out of scope to derive). **Effort M; risk medium** (CDF tail accuracy in float32; validate |dSPI| < 0.01 vs xclim with
  method="APP"/"PWM").

### 4.4 Optical flow nowcasting (rank 10)

pysteps (BSD-3; Pulkkinen et al. 2019 GMD 12:4185) motion methods: constant (max correlation translation), Lucas-Kanade
(OpenCV sparse + interpolation to dense), VET (Laroche & Zawadzki 1995), DARTS, Proesmans anisotropic diffusion; extrapolation
`semilagrangian` (https://pysteps.readthedocs.io/en/stable/pysteps_reference/motion.html; https://github.com/pySTEPS/pysteps).
**Formulation:** pyramidal dense Lucas-Kanade (per-pixel 2 x 2 solve per level, fixed levels and iterations) or Proesmans
(fixed Jacobi iterations) -> semi-Lagrangian backward advection for N lead times (fixed). Reuses `particle-advection`,
`gpu-raster-registration` (phase correlation gives the "constant" method directly), `raster-pyramid`, sampling.
Oracle match: Proesmans and semi-Lagrangian are deterministic in pysteps; LK in pysteps goes through OpenCV feature
selection, so match only the dense-flow statistics. Ensemble STEPS (stochastic cascade) is out of scope. **Effort M.**

### 4.5 Object tracking through time (rank 12)

tobac: BSD-3; features as maxima/minima in a 2-D time-varying field, threshold-based area/volume, linked into trajectories;
v2 plans multiple tracking methods including TINT (https://github.com/tobac-project/tobac).
**Formulation:** per frame: threshold + connected-component labels (existing watershed / patches); GPU builds the
label-overlap histogram between consecutive frames (bounded by label count via hashing into a fixed table); CPU does
assignment/linking (small). Data-dependent linking stays on the CPU by design. **Effort M.**

### 4.6 Spatiotemporal fusion (STARFM family) (rank 15)

cuSTARFM: GPL-3.0, 342x vs sequential STARFM on a Tesla K40, example window 31, 4 classes
(https://www.github.com/HPSCIL/cuSTARFM); cuFSDAF reports 140-182x (search snippet, **UNVERIFIED**). STARFM is a bounded
31 x 31 weighted window per pixel - a natural focal kernel - but needs coarse/fine image pairs that analysts rarely have in
a browser session. **Effort M, value low-medium.**

### 4.7 Wavelet / spectral analysis per pixel

Per-pixel Lomb-Scargle periodogram over a fixed frequency grid is bounded (T x F) and handles irregular dates; a CWT per
pixel is O(T x scales) and also bounded. No primary source fetched - **UNVERIFIED** value; I judge it niche (analysts use
harmonic amplitude/phase maps, which the harmonic fit already gives).

---

## 5. Data cube tooling and how a WebGPU engine receives chunks

- **openEO processes**: latest release 2.0.0-rc.2, stable 1.2.0; stable processes need >= 2 implementations; `proposals/`
  folder is experimental (https://github.com/Open-EO/openeo-processes). Process list from the repository tree
  (https://api.github.com/repos/Open-EO/openeo-processes/git/trees/draft?recursive=1). Families a GPU engine could
  implement (all per-pixel or per-window):
  - Cube reshaping and reduction: `reduce_dimension`, `apply`, `apply_dimension`, `apply_kernel`, `apply_neighborhood`,
    `aggregate_temporal`, `aggregate_temporal_period`, `aggregate_spatial`, `resample_spatial`, `resample_cube_spatial`,
    `resample_cube_temporal`, `merge_cubes`, `mask`, `mask_polygon`, `filter_*`, `trim_cube`, `add_dimension`, `drop_dimension`.
  - Climatology: `climatological_normal`, `anomaly`.
  - Reducers: `mean`, `median`, `min`, `max`, `sd`, `variance`, `quantiles`, `count`, `sum`, `product`, `first`, `last`,
    `extrema`, `order`, `sort`.
  - Arrays and math: `array_*` (incl. `array_interpolate_linear`), arithmetic, trigonometry, comparisons, logic,
    `normalized_difference`, `ndvi`, `linear_scale_range`, `clip`.
  - Proposals: `fit_curve` / `predict_curve` (per-pixel least squares - maps onto the QR routine), `cumsum` / `cummax` /
    `cummin` / `cumproduct`, `date_difference`, `reduce_spatial`, `aggregate_spatial_window`, `cloud_detection`,
    `sar_backscatter`, `ard_*`, `atmospheric_correction` (last four out of scope).
  - No ML processes appear in this repository tree (I did not check where they now live - **UNVERIFIED**).
  An "openEO-shaped" naming of the library's existing ops is a cheap interoperability win.
- **R ecosystem**: sits 1.5.4 (2026-01-14), GPL-2, data cubes from cloud collections, SVM/RF/XGBoost/MLP/TempCNN/attention
  classifiers, smoothing, Bayesian post-classification smoothing (https://cloud.r-project.org/web/packages/sits/index.html);
  gdalcubes 0.7.6 (2026-10-30), MIT, on-demand cubes, reductions, time-window aggregates
  (https://cloud.r-project.org/web/packages/gdalcubes/index.html). Both are usable oracles for cube reductions (gdalcubes is MIT).
- **Zarr v3 / GeoZarr**: three Zarr-v3 conventions - `multiscales`, `geo-proj` (`proj:`), `spatial` (`spatial:`) - declared via
  `zarr_conventions`; all at Proposal maturity targeting Candidate (3+ implementations); OGC GeoZarr SWG on track for
  Architecture Board review summer 2026; multiscales layout lists levels with `derived_from` and relative scale/translation
  plus absolute `spatial:shape` / `spatial:transform`; overviews down to 256 px; shard boundaries aligned to level extent;
  chunks < 100 MB (https://eopf-explorer.github.io/data-model/geozarr-minispec/). The page dates the conventions to the
  "Zarr Summit in Rome (December 2025)"; a search snippet said October 2025 - date **UNVERIFIED**. Whether the OGC review
  happened in summer 2026 - **UNVERIFIED**.
- **Icechunk**: 1.0 on 2025-07-10 (stable on-disk format, transactions, virtual file references), Icechunk 2 announced
  2026-04-09; no JavaScript client mentioned on the blog index (https://icechunk.io/en/stable/blog-posts/).
- **VirtualiZarr**: virtual Zarr stores over netCDF4/HDF5 (also GRIB, TIFF), persisted as Kerchunk references or into
  Icechunk; "almost all of Kerchunk's features" and actively maintained, unlike Kerchunk (https://virtualizarr.readthedocs.io/en/stable/).
- **Browser readers**: zarrita.js - MIT, v2 and v3, ZEP2 sharding, FetchStore / ReferenceStore (Kerchunk refs) / ZipStore,
  returns typed arrays with shape and stride, slicing via `get(arr, [null, zarr.slice(6)])`
  (https://github.com/manzt/zarrita.js). geotiff.js - MIT, `readRasters` with pixel `window` or geographic `bbox`,
  worker-pool decoding, LZW/Deflate/JPEG/LERC/Zstd/WebP, Float16/32/64, automatic overview selection
  (https://github.com/geotiffjs/geotiff.js).
- **How the engine should receive chunks** (design recommendation, not sourced): host adapter decodes each (time, y, x)
  chunk to a `Float32Array` (or `Int8Array` for embeddings) and writes it into a pre-allocated time-major storage buffer
  `[slice][y][x]` via `queue.writeBuffer` at offset `slice * tileArea`; validity as a packed `u32` bitmask per pixel (256 slices
  -> 8 u32 per pixel); dates as a `Float32Array` of centred fractional years. Zarr chunks for time series are usually
  time-contiguous (e.g. `[64, 256, 256]`), which matches pixel-major-in-time access better; offer both slice-major and
  pixel-major layouts and let the kernel generator pick the index expression at compile time.

---

## 6. EO embeddings and foundation models as data products (rank 1)

**Products**
- AlphaEarth Foundations Satellite Embedding V1 (annual): 64 bands A00-A63, 10 m, unit-length vectors (dot product = cosine),
  2017-2025, CC-BY 4.0 with a required attribution line; recommended uses: year-to-year dot-product change, clustering,
  tree classifiers, linear aggregation (https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL).
  Paper: Brown et al., arXiv 2507.22291, July 2025 (https://arxiv.org/abs/2507.22291).
  File format: int8 COGs, 8192 x 8192 x 64 per file, by year and UTM zone; NoData -128; **dequantise as
  sign(q) * (q / 127.5)^2**; overviews built by averaging dequantised vectors and renormalising to unit length -
  `gdaladdo -r average` on raw integers gives wrong results; bucket `gs://alphaearth_foundations`, provider-pays as of
  July 2026 (https://developers.google.com/earth-engine/guides/aef_on_gcs_readme). A Source Cooperative mirror exists
  (search snippet; **UNVERIFIED** beyond the snippet).
- TESSERA (Cambridge): annual 10 m embeddings from a year of Sentinel-1 + Sentinel-2, 2017-2025, v2 has 128-d Matryoshka
  vectors usable at 16/32/64/128 dims; weights CC0-1.0, `geotessera` library MIT, plus an Acceptable Use Policy;
  embedding licence not stated on the page (https://geotessera-readme.static.hf.space/README.md). int8 quantisation of
  released embeddings per arXiv 2506.20380 snippet - **UNVERIFIED** formula.
- TerraMind (IBM/ESA): Apache-2.0, inputs S2L2A/L1C, S1GRD/RTC, DEM, RGB; patch embeddings 196 x 768 for 224 x 224;
  arXiv 2504.11171 (https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base). Patch-level, not per-pixel.
- Clay: Apache-2.0 for code and weights (https://github.com/Clay-foundation/model); embedding size and current release not
  on the page - **UNVERIFIED**.
- Major TOM embeddings (SigLIP, DINOv2, SSL4EO on Core-S2L2A/L1C/S1RTC; arXiv 2412.05600) and Prithvi-EO-2.0, Presto:
  search snippets only - **UNVERIFIED**. These are patch/tile embeddings, i.e. vector tables, not rasters.

**What a client library should compute over embedding rasters** (all bounded, per pixel, D <= 128):
1. Dequantise (format-specific, e.g. AEF signed-square; TESSERA scale) into float32 or keep int8 and dequantise in-kernel
   (4x memory saving: AEF 64 B/pixel).
2. Similarity search: cosine / dot with 1..Q query vectors (Q bounded, e.g. 64) -> similarity rasters, max-over-queries,
   top-k thresholding. Query vectors from user clicks or polygons (mean of pixels, renormalised).
3. Change: per-pixel dot product between years (AEF guidance), angle, and change-vector magnitude; combine with the FDR or a
   percentile threshold.
4. Unsupervised: k-means with fixed K (<= 32) and fixed Lloyd iterations; assignment on GPU, centroid update as a GPU
   reduction - same pattern as the existing zonal statistics. k-means++ seeding on CPU from a sample.
5. Supervised: kNN against a small labelled set (<= few thousand vectors in a uniform/storage buffer, brute force), and
   linear-probe / logistic-regression inference from host-trained weights (D x C matrix). Training stays on CPU or host.
6. Visualisation: PCA to 3 components via the Section 4.2 snapshot method but over D (64 x 64 covariance), then stretch to RGB.
7. Correct pyramids for unit vectors: mean then renormalise (the AEF readme's rule) - the generic `raster-pyramid`
   average is wrong for these data; add a "unit-vector mean" reduction mode.
8. Zonal embedding statistics: mean vector per polygon (renormalised) using the existing exact fractional coverage.

**Capacity:** int8 AEF: 64 B/pixel/year -> 300 MB holds ~4.7 M pixels for one year or two years of ~2.3 M pixels
(~1500 x 1500 px, 15 km square). float32 would be 4x less. **Effort S-M; risk low** (oracle = NumPy / scikit-learn, exact
up to float32 rounding; k-means compare with fixed seeds and iteration count).

---

## 7. Discrete global grid systems

- **H3**: 4.2.0 (2024-12-04) `polygonToCellsExperimental` with containment modes; 4.3.0 (2025-06-17) `gridRing`; 4.4.0
  (2025-11-06) `getIndexDigit`, `isValidIndex`, `constructCell`; 4.5.0 (2026-05-21) `reverseDirectedEdge`, bidirectional
  `gridPathCells`, internal `cellsToMultiPolygon` (https://github.com/uber/h3/blob/master/CHANGELOG.md). `getIndexDigit` /
  `constructCell` make a GPU-side index (de)composition straightforward to validate.
- **A5**: exactly equal-area pentagons, 64-bit IDs on a non-self-crossing space-filling curve, 31 levels, index frozen since
  v1.0, TypeScript/Python/Rust plus a deck.gl integration, Apache-2.0 (https://a5geo.org/; https://github.com/felixpalmer/a5).
  Base solid / projection not stated on those pages (**UNVERIFIED**; the repository contains authalic constants).
- **HEALPix for Earth**: healpix-geo (Rust over cdshealpix, Python, ellipsoid support e.g. WGS84, nested / ring / zuniq
  indexing, range-based cell sets) and xdggs (xarray accessor, Apache-2.0, CF HEALPix and Zarr DGGS conventions)
  (https://healpix-geo.readthedocs.io/en/latest/; https://meetingorganizer.copernicus.org/EGU26/EGU26-18208.html;
  https://github.com/xarray-contrib/xdggs). The 0.9 % sphere-vs-ellipsoid area variation figure is from a search snippet -
  **UNVERIFIED**.
- **DGGRID** 9.2 released 2026-11-05, AGPL-3.0 (https://github.com/sahrk/DGGRID) - oracle-only.
- **OGC API - DGGS Part 1: Core** approved as an OGC Standard, announced 2025-10-24: zone queries (with CQL2), zone data
  retrieval quantised to sub-zones of an ancestral zone, DGGS-JSON and DGGS-JSON-FG encodings, binary zone lists; UBJSON,
  Zarr and netCDF encodings proposed (https://www.ogc.org/announcement/ogc-membership-approves-ogc-api-discrete-global-grid-systems-part-1-core-as-an-official-ogc-standard/).
- **When equal area matters:** any count / density / sum statistic compared across latitudes (H3 cell areas vary with
  position; A5 and HEALPix are equal-area by construction). For means of intensive variables it matters less.
- **GPU-friendly indexing math:** HEALPix nested `ang2pix` is closed-form (z = sin(authalic lat), face from z and longitude,
  then bit interleave) - the cheapest to put in WGSL, and the `Hilbert keys` code already has bit-interleave helpers.
  H3 point-to-cell is gnomonic projection onto 20 icosahedron faces + hex rounding + digit-by-digit aperture-7 rotations:
  bounded (15 resolutions) and fine in f32 for res <= ~10; the 64-bit index needs `vec2<u32>` arithmetic. No GPU H3 port was
  found (search only). **Effort M** for "HEALPix nested + A5 point/pixel-to-cell + cell aggregation of rasters"; value
  medium (DGGS as a cube substitute is still emerging; xdggs is the reference model).

---

## 8. Terrain and environmental dynamics

| Item | Analysis or simulation | Verdict |
|---|---|---|
| Sun-hours / insolation over a day or year with shadows | Analysis | **Already present** (`gpu-solar-irradiance.ts` integrates over time; horizon sweep exists). A small add: annual accumulation by sampling representative days (bounded). |
| Sky-view factor | Analysis | Present as an input to irradiance; ensure it is exposed as its own output from the horizon sweep. SOLWEIG-GPU (PyTorch; SVF, shadows, Tmrt, UTCI; claims 10-100x; Kamath et al. 2026 JOSS 11(118):9535) is a usable comparison (https://solweig-gpu.readthedocs.io/en/latest/). |
| Snow / surface energy balance time integration | Simulation | Out of scope except degree-day melt (a degree-day accumulator from 4.3 times a melt factor) - **S**, borderline. |
| Wildfire spread (Rothermel CA, level set) | Simulation | Out of scope. GPU CA with Rothermel ellipses exists (2012 SciTePress, search snippet only - **UNVERIFIED**). Iteration count is data-dependent (fire front). |
| Urban growth CA (SLEUTH-style) | Simulation | Out of scope (calibration-heavy, stochastic). |
| Pollutant advection-diffusion | Simulation | Out of scope; explicit stencils are bounded but this is a model, not an analysis, and the library rules exclude flood-style simulation. |

---

## 9. Published GPU implementations and speedups

- BFAST (GPU, Futhark -> CUDA/OpenCL): 24-48x vs 32-thread C; up to ~1000x vs R; missing values handled per pixel
  (https://www.futhark-lang.org/publications/icde20.pdf). The 2018 SSDBM predecessor claims up to 10^4x vs public code and 10x vs
  multithreaded CPU (search snippet, arXiv 1807.01751 - **UNVERIFIED**).
- cuSTARFM 342x vs sequential (https://www.github.com/HPSCIL/cuSTARFM).
- SOLWEIG-GPU 10-100x (https://solweig-gpu.readthedocs.io/en/latest/).
- No published CCDC/COLD GPU port was found (search only). No browser/WebGPU EO time-series tool was found in search;
  deck.gl-raster (WebGL band math) is the nearest browser precedent (search snippet). **A WebGPU COLD-lite or BFAST Monitor
  would appear to be the first browser implementation** (absence of evidence, not verified exhaustively).

---

## 10. Not worth doing on this platform

- **BEAST / Rbeast** - per-pixel RJ-MCMC; divergent, data-dependent, stochastic; GPL; cannot match the oracle deterministically.
- **Foundation-model inference (TerraMind, Prithvi, Clay encoders)** inside the library - transformer inference belongs to
  ONNX/WebNN runtimes; the library should consume their output rasters (Section 6), not run encoders.
- **Deep change detection / time-series classifiers training** - host-side.
- **TIMESAT parity** - licence (non-commercial / proprietary from 4.2.1, **UNVERIFIED** for the latter) prevents use as a
  pinned oracle; implement threshold-based metrics on the Whittaker curve instead.
- **Double-logistic / asymmetric-Gaussian nonlinear phenology fits** - LM in float32 with exponentials is fragile; the
  operational Copernicus product itself uses a smoothing spline.
- **Full LASSO-exact COLD reproduction** - possible but expensive; ship OLS/ridge COLD-lite with documented deviation first.
- **ESTARFM / FSDAF** - thin-plate-spline and classification steps; low demand in a browser.
- **Rotated EOF beyond varimax, extended / complex EOF, DMD variants** - small user base; snapshot PCA covers most use.
- **Wildfire, urban CA, pollutant dispersion, snowpack energy balance** - simulation, data-dependent iteration.
- **Ensemble STEPS nowcasting** - stochastic cascades; deterministic extrapolation only.
- **File readers (Zarr, COG, NetCDF, Icechunk)** - stays in host adapters per the library rules; document the chunk upload layout instead.

---

## 11. Cross-cutting prerequisites (do these first)

1. **Irregular-date time axis**: a shared `dates: Float32Array` (centred fractional years) uniform/storage input accepted by
   harmonic fit, Whittaker, HANTS, BFAST Monitor, COLD-lite, Theil-Sen. Unlocks ranks 3, 4, 5, 9.
2. **Streaming temporal accumulators** that consume slice chunks, removing the 256-slice cap for run-length, climatology and
   degree-day style indices (rank 7).
3. **p-value raster -> FDR mask** utility (rank 2) and a common p-value output convention for existing tests.
4. **int8 / unit-vector raster support**: in-kernel dequantisation and a unit-vector-mean pyramid mode (rank 1).

---

## 12. Source list (opened)

- https://developers.google.com/earth-engine/apidocs/ee-algorithms-temporalsegmentation-ccdc
- https://developers.google.com/earth-engine/apidocs/ee-algorithms-temporalsegmentation-landtrendr
- https://developers.google.com/earth-engine/apidocs/ee-algorithms-temporalsegmentation-ewmacd
- https://gerslab.cahnr.uconn.edu/wp-content/uploads/sites/2514/2021/06/cold_2020_zhu.pdf
- https://github.com/GERSL/COLD
- https://github.com/Remote-Sensing-of-Land-Resource-Lab/pyxccd
- https://pyxccd.readthedocs.io/en/latest/1_break_detection_fire_hls.html
- https://search.r-project.org/CRAN/refmans/bfast/html/bfastlite.html
- https://www.futhark-lang.org/publications/icde20.pdf
- https://rdrr.io/cran/Rbeast
- https://www.wur.nl/en/research/products-services/radd-forest-disturbance-alert
- https://github.com/eMapR/LT-GEE ; https://emapr.github.io/LT-GEE/lt-gee-requirements.html
- https://search.r-project.org/CRAN/refmans/phenofit/html/smooth_wWHIT.html
- https://grass.osgeo.org/grass85/manuals/addons/r.hants.html
- https://web.nateko.lu.se/timesat/timesat.asp
- https://library.land.copernicus.eu/products/MRVPP_MRVPP_ATBD_v2.html
- https://knowledge.dea.ga.gov.au/data/version-history/dea-surface-reflectance-geomedian-landsat-2.0.0/
- https://github.com/daleroberts/hdmedians
- https://eol.pages.cms.hu-berlin.de/gcg_eo/04_baps.html
- https://ams.confex.com/ams/96Annual/webprogram/Paper276937.html
- https://github.com/xarray-contrib/xeofs ; https://xeofs.readthedocs.io/en/latest/
- https://github.com/PyDMD/PyDMD
- https://github.com/Ouranosinc/xclim ; https://raw.githubusercontent.com/Ouranosinc/xclim/v1.0.0/src/xclim/compute/_agro.py
- https://pysteps.readthedocs.io/en/stable/pysteps_reference/motion.html ; https://github.com/pySTEPS/pysteps
- https://github.com/tobac-project/tobac
- https://www.github.com/HPSCIL/cuSTARFM
- https://github.com/Open-EO/openeo-processes ; https://api.github.com/repos/Open-EO/openeo-processes/git/trees/draft?recursive=1
- https://cloud.r-project.org/web/packages/sits/index.html ; https://cloud.r-project.org/web/packages/gdalcubes/index.html
- https://eopf-explorer.github.io/data-model/geozarr-minispec/
- https://icechunk.io/en/stable/blog-posts/
- https://virtualizarr.readthedocs.io/en/stable/
- https://github.com/manzt/zarrita.js ; https://github.com/geotiffjs/geotiff.js
- https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- https://developers.google.com/earth-engine/guides/aef_on_gcs_readme
- https://arxiv.org/abs/2507.22291
- https://geotessera-readme.static.hf.space/README.md
- https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base
- https://github.com/Clay-foundation/model
- https://github.com/uber/h3/blob/master/CHANGELOG.md
- https://a5geo.org/ ; https://github.com/felixpalmer/a5
- https://healpix-geo.readthedocs.io/en/latest/ ; https://meetingorganizer.copernicus.org/EGU26/EGU26-18208.html
- https://github.com/xarray-contrib/xdggs ; https://xdggs.readthedocs.io/en/latest/
- https://github.com/sahrk/DGGRID
- https://www.ogc.org/announcement/ogc-membership-approves-ogc-api-discrete-global-grid-systems-part-1-core-as-an-official-ogc-standard/
- https://solweig-gpu.readthedocs.io/en/latest/

Pages attempted but not readable (claims relying on them are marked UNVERIFIED): PyPI pages for timesat and vam.whittaker
(render error), github.com/WFP-VAM/vam.whittaker (404), MDPI LandTrendr GEE paper (403), TU Delft SVF repository (connection
refused), processes.openeo.org (empty render), xclim docs indices pages (404).
