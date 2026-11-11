# Math correctness review: terrain surface (2026-11-10)

Scope id: `terrain-surface`. Scope: `gpu-terrain/terrain-illumination`, `terrain-analysis`,
`topographic-position`, `point-horizon`, `relief-visualization`, `terrain-decode`,
`terrain-curvature` and `gpu-terrain/index.ts`. Paths below are relative to
`modules/analysis/src`.

The math in this scope is in good shape. Most formulas match their reference and most are pinned by
an independent oracle (gdaldem, GRASS, RVT, xDEM, pvlib, SciPy). I found one result that is wrong
on ordinary input (the diffuse term of insolation), two inconsistencies, three boundary cases and
three documentation mismatches. No finding changes slope, aspect, hillshade, curvature, viewshed or
decode values in the interior of a north-up grid.

Scratch script for every CONFIRMED row:
`/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/9972295f-91de-4373-ba95-d2f799611a84/scratchpad/terrain-surface/checks.mjs`
(pure Node, `Math.fround` where float32 matters). I ran no spec files.

Working tree note: `terrain-decode/gpu-terrain-spike-repair.ts` and
`topographic-position/terrain-summed-area-table.ts` are modified by another session. I read the
working-tree versions. Neither looked half-edited.

## 1. Coverage

Read in full (every kernel body and packer; import blocks skipped):

- `gpu-terrain/index.ts`
- `terrain-curvature/gpu-terrain-curvature.ts`, `terrain-curvature/terrain-geomorphometry-utils.ts`
- `terrain-analysis/gpu-terrain-derivatives.ts`, `terrain-analysis/terrain-sight-line.ts`
- `topographic-position/gpu-terrain-ruggedness.ts`, `gpu-terrain-vector-ruggedness.ts`,
  `gpu-terrain-topographic-position.ts`, `gpu-terrain-weiss-landforms.ts`,
  `terrain-summed-area-table.ts`
- `terrain-illumination/solar-position.ts`, `gpu-solar-position.ts`, `gpu-solar-irradiance.ts`,
  `gpu-terrain-cast-shadow.ts`, `gpu-terrain-horizon.ts`, `terrain-horizon-sweep.ts`,
  `gpu-relief-shading.ts`, `gpu-texture-shading.ts`, `terrain-illumination-utils.ts`
- `point-horizon/point-horizon-parameters.ts`, `point-horizon-march.ts`
- `relief-visualization/relief-visualization-utils.ts`, `gpu-multi-scale-relief.ts`
- Supporting, outside the scope: `analysis-core/raster-grid.ts` (full), published
  `@luma.gl/graph` `gpu-raster/gpu-raster-edges.js` (full), `gpu-raster-neighborhood.js` (kernel
  body), `gpu-raster-convolution.js` (Gaussian and box kernel definitions),
  `gpu-core/gpu-finite-difference-2d.js` (stencils).

Read in part (kernel bodies, packers and class documentation read; constructor validation and
binding plumbing skipped):

- `terrain-analysis/gpu-terrain-viewshed.ts` (40-330, 470-655), `gpu-terrain-line-of-sight.ts`
  (40-175, 380-480), `gpu-terrain-cumulative-viewshed.ts` (60-150, 285-480),
  `gpu-terrain-contours.ts` (20-120 and the delegation site only)
- `terrain-illumination/gpu-solar-shadow-mask.ts` (28-177, 270-340)
- `point-horizon/gpu-point-horizon-profile.ts` (30-110, 230-321),
  `gpu-point-horizon-visibility.ts` (36-130, 150-312)
- `relief-visualization/gpu-local-dominance.ts` (30-200, 220-359), `gpu-relief-blend.ts` (28-120,
  290-376), `gpu-simple-local-relief.ts` (30-175)
- `terrain-decode/gpu-terrain-rgb-decode.ts` (25-160, 230-360),
  `gpu-terrain-spike-repair.ts` (class documentation, 396-520, 640-730, 770-900; the size, main
  component and remaining-jump kernels were not read)

Not opened: the seven subdirectory `index.ts` files (re-exports). Nothing else in scope was skipped.

## 2. Findings

| ID | Severity | Status | Where | What the kernel computes | What the reference says | Concrete input and wrong output | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| terrain-surface-01 | wrong-result | CONFIRMED | `terrain-illumination/gpu-solar-irradiance.ts:536`, with `gpu-terrain-horizon.ts:978`; `docs/api/gpu-terrain/operators.md:1113`, `:1184` | Diffuse energy `diffuseIrradiance * skyViewFactor * duration`. The documented pipeline binds the `skyViewFactor` of `GPUTerrainHorizon`, which is Zaksek's `1 - mean(sin h)`. | For an isotropic sky the diffuse irradiance on a horizontal surface is `D * mean(cos^2 h)` (Dozier and Frew 1990; GRASS `r.sun`). `1 - sin h` is the visible solid-angle fraction, not the irradiance fraction. | Uniform horizon 30 degrees, `D = 100 W/m^2`, one hour: kernel 50 Wh/m^2, reference 75 Wh/m^2 (33% low). At 10 degrees: 82.6 against 97.0 (15% low). | Compute the diffuse factor inside `GPUSolarIrradiance` from the horizon map it already binds (`mean(cos^2 max(h, 0))`, or the slope-aware Dozier-Frew form when slope and aspect are bound). Keep the Zaksek output for visualization and say so. |
| terrain-surface-02 | inconsistent | CONFIRMED | `terrain-analysis/gpu-terrain-derivatives.ts:134`, `:275`; `terrain-curvature/gpu-terrain-curvature.ts:227`, `:501`; `topographic-position/gpu-terrain-vector-ruggedness.ts:145`; `terrain-illumination/gpu-relief-shading.ts:316`, `gpu-terrain-horizon.ts:405`, `gpu-terrain-cast-shadow.ts:184`; `raster-analysis/raster-band-utils.ts:61-69` | North is taken from the `rowDirection` prop (default `'south'`). The sign of `grid.affine[4]`, the sign of `affine[0]` and (in `'uniform'` mode) any rotation are never read or checked. | The grid affine already states the row direction. `makeGPURasterMetadata({rowDirection: 'north'})` writes it into the affine. One quantity has two sources. | Grid with `affine[4] = +30` (row 0 in the south), surface rising to the north, contributor props left at default: aspect is 0 (north). The true downslope aspect is 180. Hillshade light, horizon sectors and sun shadows are mirrored the same way. | Derive the row sign from `affine[4]`, or throw when the prop and the affine disagree. Reject `affine[0] < 0` and, for azimuth outputs, rotated grids. `point-horizon-parameters.ts:165` already checks the affine for Web Mercator. |
| terrain-surface-03 | wrong-edge-case | CONFIRMED | `terrain-curvature/gpu-terrain-curvature.ts:228-229`, `:613-616`; `terrain-analysis/gpu-terrain-derivatives.ts:135-136`, `:221`; `topographic-position/gpu-terrain-vector-ruggedness.ts:269-270`; `gpu-terrain-weiss-landforms.ts:329-331`; `terrain-illumination/gpu-relief-shading.ts:438` | Default border rule `'clamp'` repeats the edge sample. On border cells the first derivative across the edge is halved and the second derivative is not zero. | gdaldem marks the outer ring nodata, or with `-compute_edges` extrapolates linearly (exact on a plane). luma.gl's own `GPUFiniteDifference2D` uses second-order one-sided stencils "so physical spacing and accuracy do not silently change at the edge". | Plane with gradient 0.2, cell 10 m. Interior: slope 11.31 degrees, curvature 0. West border column: Horn slope 5.71 degrees; Zevenbergen-Thorne `p = 0.1`, `r = 0.02`, profile curvature -0.0197 1/m, mean curvature -0.0099 1/m. | Default to `'nodata'` for curvature, or replace clamp by the gdaldem `2a - b` extrapolation that `GPUTerrainRuggedness` already has (`gpu-terrain-ruggedness.ts:215-246`). Use one border vocabulary and one default across the five classes. |
| terrain-surface-04 | doc-mismatch | CONFIRMED | `terrain-illumination/terrain-horizon-sweep.ts:597-600`; `gpu-terrain-horizon.ts:78-81`; `docs/api/gpu-terrain/operators.md:1058-1061` | The sweep takes the elevation of a pixel centre on the digital line and the distance `k * hypot(1, slope)` along the ideal line, not the distance to that centre. | Documentation: "exact against brute force over the same samples", "pixel centres within half a pixel of the ray". | Line slope 0.5, 10 m cells, a 10 m step to the next pixel: the sweep reports 41.8 degrees for both neighbours; centre-to-centre geometry gives 45.0 (side) and 35.3 (diagonal). Lateral offset from the ray through the pixel's own centre reaches 0.86 px at slope 0.3. | Say "nearest-pixel sampling of the ray: lateral offset below one pixel, distance measured along the ray". Keep the measured 0.01 sky-view difference. No code change needed. |
| terrain-surface-05 | wrong-edge-case | SUSPECTED | `terrain-illumination/gpu-relief-shading.ts:704`, `:713` | `'imhof-swing'` evaluates `atan2(-0, -0)` on flat cells and multiplies `sin(aspect - azimuth)` by a swing factor of 0. | `atan2(0, 0)` is not defined on every backend. If it returns NaN, `NaN * 0` is NaN and the shade is invalid. | A lake (east and north gradient exactly 0) under `lightWeighting: 'imhof-swing'` would become nodata on such a backend. The fixed and aspect kernel guards this with `isFlat` (`:533`, `:547`). | Guard the swing with the same `isFlat` test: use the base azimuth when both gradients are 0. |
| terrain-surface-06 | wrong-edge-case | CONFIRMED | `point-horizon/gpu-point-horizon-visibility.ts:295-297` | `hidden` if `target < occluder - tol`; `marginal` if `abs(target - occluder) < tol`; else `visible`. | The three classes should be ordered: below the band, inside it, above it. | `target = 10.0`, `occluder = 10.5`, `tol = 0.5`: neither test is true, so the code is `visible`. At 10.01 it is `marginal`, at 9.99 `hidden`. | Make the second test `target <= occluder + tol` after the first, or use `<=` in it. |
| terrain-surface-07 | doc-mismatch | CONFIRMED | `terrain-curvature/gpu-terrain-curvature.ts:52` against `:409-413` | The `'florinsky'` partials are the 5x5 third-order (cubic) polynomial fit: on `z = x^3` the kernel gives `p = 0`. | TSDoc says "5x5 quadratic least squares". A quadratic fit gives `p = 3.4` on the same input. `docs/api` says "5x5 polynomial", which is right. | Documentation only. The kernel matches WhiteboxTools and xDEM. | Change the TSDoc to "5x5 third-order polynomial (Florinsky 2009, 2016)". |
| terrain-surface-08 | doc-mismatch | CONFIRMED | `terrain-illumination/gpu-texture-shading.ts:57-58`; `docs/api/gpu-terrain/operators.md:1320` | Band weight `2^(-k * detail)` approximates the symbol `abs(f)^detail`. | Documentation: "1 approaches a Laplacian". The Laplacian symbol is `abs(f)^2`, so the Laplacian is `detail = 2`; `detail = 1` is the half Laplacian. | Documentation only. | Say "2 approaches a Laplacian; Brown's usual range is 0.5 to 1". |
| terrain-surface-09 | inconsistent | CONFIRMED | `analysis-core/raster-grid.ts:132-142` against `point-horizon/point-horizon-parameters.ts:37`, `point-horizon/gpu-point-horizon-visibility.ts:195`, `:204` | Two ground models for a Web Mercator tile. Viewshed, line of sight, horizon and derivatives scale each row with the WGS84 radii of curvature. Point horizon measures great-circle arcs on a sphere of 6371008.8 m. | One library should measure one tile one way. | Equator: one pixel east-west is `cell` metres in the raster-grid model and `0.99888 * cell` in the point-horizon model; north-south the factors are 0.99331 and 0.99888. A `maximumDistance` cut differs by 0.1 to 0.6% between `GPUTerrainLineOfSight` and `GPUPointHorizonVisibility`. | State the choice in both places, or give point horizon the WGS84 row scale. Low priority. |
| terrain-surface-10 | untested | SUSPECTED | `terrain-illumination/gpu-terrain-horizon.ts:569-585`; `modules/analysis/test/gpu-terrain/terrain-illumination/terrain-horizon-oracle.ts:54-66` | Anisotropic sky-view weights assume RVT direction `i` is `i * 360 / n` counterclockwise from north. | RVT `sky_view_factor_compute`. The only test is a TypeScript copy of the same assumption. No RVT fixture exists for sky-view factor, anisotropic sky-view factor or openness. | I could not check RVT's source here. If RVT's angle is measured differently, the weighted side of the sky is wrong for every pixel. | Add `rvt.vis.sky_view_factor` (svf, asvf, opns) to `parity-wp5-relief.py`, which already calls RVT. |

### terrain-surface-01: diffuse insolation uses the wrong sky factor

Derivation. Isotropic sky radiance `L`. Irradiance on a horizontal surface from the sky above a
horizon at elevation `h` is `integral over azimuth of integral_0^(90-h) L cos(t) sin(t) dt`, which
is `pi L cos^2 h` for a uniform horizon. The unobstructed value is `pi L`. So the factor is
`mean(cos^2 h)`. Zaksek's factor is the same integral without the `cos(t)` weight: the fraction of
the hemisphere's solid angle that is visible. The script integrates numerically and prints both
(0.7500 against 0.5000 at 30 degrees).

The kernel itself only multiplies by whatever raster is bound, and the property is documented as
"scaled by the sky-view factor". The defect is the pairing: the only sky-view factor this library
produces is the Zaksek one, and the documented example feeds it straight in. The pvlib oracle
(`test/oracles/parity-wp5-solar.py`) avoids the question: it uses `(1 + cos(tilt)) / 2` or
arbitrary synthetic factors and states that it makes "not a claim of pvlib terrain-obstruction
parity". So no test pins the physical meaning.

Second point, same line: on a sloped surface the isotropic diffuse term also depends on slope and
aspect (the surface sees less sky, and part of the ground). The kernel applies the incidence angle
to the direct term only. The fix above covers both.

`GPUSolarShadowMask` uses `ambient * svf` too (`gpu-solar-shadow-mask.ts:329`). That output is a
light rig for rendering, not an energy, so I do not count it.

### terrain-surface-02: north comes from a prop, not from the grid

This is a consistency defect, not a formula defect. With a north-up grid and default props every
direction is right. The code is defensible because the prop is documented. The risk is silent: a
user who builds the grid with `rowDirection: 'north'` and forgets to repeat it on the contributor
gets aspect, hillshade, horizon sectors, cast shadows and the anisotropic sky-view direction
mirrored north to south, with no error. In `'uniform'` mode a rotated affine is accepted
(`validateRasterGridAxisAligned` only checks the two curved modes), and "degrees clockwise from
north" then means clockwise from the grid's up axis. The sight-line classes are not affected: they
join pixel positions and use no azimuth.

### terrain-surface-03: clamped borders

The border rule is documented ("repeats edge samples"), so this is not a hidden behaviour. I report
it because the default produces valid-looking wrong numbers, because a sibling class already
implements the gdaldem rule, and because it differs from luma.gl's own finite-difference
convention. The curvature case matters most: 0.02 1/m on a plane is ten times a typical real
curvature. The comment at `gpu-terrain-curvature.ts:237` says the class has "edge extrapolation";
it has edge replication. The gdaldem derivative spec runs with `borderMode: 'nodata'`
(`test/gpu-terrain/terrain-reference-oracle.spec.ts:161`), which equals gdaldem's default outer
ring, so nothing pins the clamped border values against a reference.

## 3. Verified correct

Each line names what I compared the kernel against. "Hand" means I derived the formula from the
reference and checked the WGSL term by term.

- `GPUTerrainDerivatives`: Horn weights and divisor `8 * cell` through published
  `GPURasterGradient` (Sobel responds 8 to a unit ramp); `ROW_NORTH_SIGN`; aspect
  `atan2(-east, -north)` clockwise from north, flat cells -1; slope degrees and percent; ESRI
  hillshade as `max(dot(normal, light), 0)`. Hand, against gdaldem `GDALSlopeHornAlg`,
  `GDALAspectAlg`, `GDALHillshadeAlg`. Strict nodata holds: the two Sobel passes together cover all
  nine cells and the shade kernel tests both raw values.
- `analysis-core/raster-grid.ts` ground scale: geographic `N cos(lat) pi/180` and `M pi/180`; Web
  Mercator `N cos(lat)/a`, `M cos(lat)/a` with `sin(lat) = tanh(y/a)`. Hand, against Snyder.
- `GPUTerrainCurvature`: Evans-Young, Zevenbergen-Thorne and Florinsky 5x5 partials (unit tests on
  `z = x`, `x^2`, `xy`, `x^3` by hand and script); all 15 curvature formulas against Florinsky
  (2016) and Minar et al. (2020); sign convention convex positive (hill `z = -(x^2 + y^2)` gives
  mean curvature +2). Centre-relative differencing is valid because every stencil sums to zero.
- `GPUTerrainRuggedness`: TPI, TRI (Riley and Wilson), roughness and the `-compute_edges` window.
  Hand, against gdaldem `GDALTPIAlg`, `GDALTRIAlg`, `GDALRoughnessAlg`.
- `GPUTerrainVectorRuggedness`: unit normal `normalize(-p, -q, 1)` equals Sappington's
  `(sin s sin a, sin s cos a, cos s)`; `1 - |sum| / count`. Hand.
- `terrain-summed-area-table.ts`: the row-major then column-major double prefix and the
  `readColumnRange` predecessor trick, including `column = -1`, `row = 0` and `width = 1`; 64-bit
  add, subtract, wide multiply. Hand.
- `GPUTerrainTopographicPosition`: TPI over the square annulus; DEV as `(z - mean) / sd`,
  population variance, exact re-centring; DEVmax first-scale tie rule. Hand, against Lindsay et
  al. (2015).
- `GPUTerrainWeissLandforms`: the ten-class table, inclusive thresholds, slope split; global
  z-score with population standard deviation from centred squares. Hand, against Weiss (2001).
- `getSolarPosition`, `GPUSolarPosition`: every NOAA spreadsheet column (mean longitude, anomaly,
  eccentricity, equation of centre, obliquity, declination, equation of time, hour angle, zenith,
  refraction pieces); atan2 azimuth form; the float32 day split with rates `36000.76983 / 36525`
  and `35999.05029 / 36525`. Hand.
- `GPUSolarIrradiance` direct term: Meinel `1353 * 0.7^(AM^0.678)`, Kasten-Young air mass,
  `cos(slope) sin(alt) + sin(slope) cos(alt) cos(azimuth - aspect)`, flat aspect rule, Wh/m^2
  units. Hand.
- `getSolarDiskVisibility` (shadow mask, irradiance) and its inline copy in the cast shadow:
  circular segment area `1 - (acos u - u sqrt(1 - u^2)) / pi`; values 1, 0.5, 0 at `u = 1, 0, -1`.
- `GPUTerrainHorizon` march: pixel direction from ground azimuth, ground distance, tangent
  `(z dh - c d^2) / d`; sky-view `1 - mean(sin(max(h, 0)))` (Zaksek); positive openness
  `mean(90 - h)` and negative openness on the inverted surface (Yokoyama); unorm16 code range
  (script: codes stay in 1..65535 in float32).
- `terrain-horizon-sweep.ts`: digital-line partition and line count; hull pointers as upper hull in
  the transformed height `z h - b a^2` (the curvature term moves out of the argmax); strict
  comparison keeps the nearest on ties; windowed query is exact over in-window samples; int32
  bound `65536 * 32767 + 32768 < 2^31`. Hand.
- `GPUTerrainCastShadow`: sun ground direction `(sin az, -cos az)`, north-row mirroring.
- `GPUReliefShading`: multidirectional weights `sin^2(aspect - azimuth)` normalised by their sum
  (2 for the four USGS lights), per-light clamp at 0, flat cells equal weights. Hand, against Mark
  (1992) and gdaldem `GDALHillshadeMultiDirectionalAlg`.
- `terrain-sight-line.ts`, `GPUTerrainViewshed`, `GPUTerrainLineOfSight`,
  `GPUTerrainCumulativeViewshed`: target slope with drop `c D^2`, sample slope, grazing stays
  visible, tolerance band, batch index arithmetic. `getGPUTerrainCurvatureCoefficient` is
  `(1 - k) / (2R)`.
- `point-horizon-march.ts`: great-circle direct problem in Mercator offsets (`ds`, `den`,
  `atan2`, `atanh(ds / den)`), haversine in Mercator form
  (`sin^2(dphi/2) = cos p1 cos p2 sinh^2(dm/2)`), bearing terms, bilinear form, tangent
  `h / d - c d`. Hand.
- `GPUTerrainRGBDecode`: Terrarium `N / 256 - 32768` and Mapbox `(N - 100000) * 0.1`; texture bytes
  recovered as `round(v * 255)`.
- `GPUTerrainSpikeRepair`: majority-bit candidate equals the mode when agreement exceeds 0.5;
  64-bit percentage tests. Read as a port of a heuristic; there is no reference method.
- `GPUSimpleLocalRelief`, `GPUMultiScaleRelief`, mean filters: edge-replication weights of the
  summed-area filter (strips and corners), separable filter identity, MSRM telescoping
  `(lpf_i - lpf_n) / (n - i)`. Hand, against RVT `slrm`, `msrm`, `mean_filter`.
- `GPULocalDominance`: sum and normaliser; flat surface gives 1. Against RVT `local_dominance`.
- `GPUReliefBlend`: multiply, screen, overlay, soft light. Against RVT `blend_func`.
- `GPUTextureShading`: cascade sigmas `sqrt(s_k^2 - s_(k-1)^2)`, normalised convolution, anchor
  cancels in every band. It is an approximation of Brown's operator, and says so.
- `GPUTerrainContours`: delegates to published `GPURasterContours`. Not re-derived.

## 4. Standards consistency

- North: see terrain-surface-02. All azimuths are compass, clockwise from north, except the RVT
  anisotropy angle, which is documented.
- Border rule: three vocabularies and two defaults. `borderMode` with `'clamp'` default
  (derivatives, relief shading: four published modes; curvature, vector ruggedness: two modes);
  fixed clamp with no option (Weiss slope); `edgeMode` with `'nodata'` default and a gdaldem
  `'extrapolate'` (ruggedness). See terrain-surface-03.
- Horn gradient written three ways: through published `GPURasterGradient` (derivatives, relief
  shading), inline on absolute elevations (`gpu-terrain-vector-ruggedness.ts:280-285`), inline on
  centre-relative differences (`gpu-terrain-weiss-landforms.ts:342-361`). Weights and divisor are
  the same. Only the rounding differs.
- Web Mercator ground model: two (terrain-surface-09).
- Earth radius: `GPU_EARTH_MEAN_RADIUS_METERS` for the curvature drop and point horizon,
  `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` for the projection. Both are named constants. Correct use.
- Tie rule: a grazing sight line is visible (`terrain-sight-line.ts:340-343`,
  `gpu-terrain-viewshed.ts:531`); a grazing sun is shadowed (`terrain-illumination-utils.ts:202`,
  `gpu-terrain-cast-shadow.ts:361`). The user documentation states the second one.
- Solar-disk visibility: the cast shadow repeats the formula inline
  (`gpu-terrain-cast-shadow.ts:353-363`) instead of calling `getSolarDiskVisibility`. Same math.
- Vertical scale: `zFactor` in most classes, `verticalExaggeration` in relief visualization, none
  in the sight-line classes. Reported on 2026-11-09; still present.
- Literals where a shared constant exists: `57.29577951308232` inline at
  `gpu-terrain-weiss-landforms.ts:362`; `PI` and `RADIANS_TO_DEGREES` redeclared in
  `gpu-terrain-derivatives.ts:354-356` and in `point-horizon-march.ts:601-606`. NaN is spelled
  three ways (`bitcast<f32>(0x7fc00000u | (index & 0u))` inline, `getNaN`, `nanValue`). All have
  the same value.
- Finite test: every kernel uses the shared `IS_FINITE_VALUE_WGSL`. Several kernels also rely on a
  NaN failing a comparison (`gpu-local-dominance.ts:346-349`), which is the same assumption.
- Curvature TSDoc (`gpu-terrain-curvature.ts:243-244`) says "spherical" row scale; the shared
  helper is WGS84 ellipsoidal.
- Not math: `docs/api/gpu-terrain/operators.md:1046` says the horizon runs "one node per sector";
  the march is now fused over sectors.

## 5. luma.gl official implementations

Delegates, and the arguments keep their meaning:

- `GPUTerrainDerivatives`, `GPUReliefShading` to `GPURasterGradient` (`operator: 'sobel'`,
  `scale: 1`). The local kernel divides by `8 * groundCell`, which is the Horn divisor. Border and
  nodata are the published ones.
- `GPUTerrainContours` to `GPURasterContours`.
- `terrain-summed-area-table.ts` to `GPUScan`, `GPUScanUint64`, `GPUTranspose` (gpu-core).
- `GPUTerrainWeissLandforms` to `GPUReduction` (gpu-core).
- `GPUTerrainSpikeRepair` to `GPUGraphAlgorithmsConnectedComponents`. The class comment explains
  why `GPURasterConnectedComponents` cannot express its edge rule.
- Texture outputs to `GPURasterBufferToTexture`.

Re-implements something an official entry point publishes:

- `GPUTerrainVectorRuggedness` and `GPUTerrainWeissLandforms` compute Horn inline. Same weights and
  divisor as `GPURasterGradient`. Vector ruggedness has the same clamp and nodata borders; Weiss
  has clamp only. Recommendation: build both on `GPUTerrainDerivatives` (or `GPURasterGradient`),
  as the 2026-11-09 review already suggested.
- `GPUTerrainCurvature`, Zevenbergen-Thorne method: `p`, `q`, `r`, `t` and the `'laplacian'` kind
  are the interior stencils of `GPUFiniteDifference2D` (gpu-core: centred first and second
  differences) and of `GPURasterLaplacian` with connectivity 4. They agree in the interior. At the
  border the official kernel uses second-order one-sided stencils and is exact on a plane; the
  local clamp is not (terrain-surface-03). The official kernel is right. The local class cannot
  delegate today: the official classes have constant spacing, no nodata rule (finite difference)
  and sum absolute float32 values. The local centre-relative differencing is the better numeric
  choice for second derivatives of elevation (at 4000 m with 1 m cells an absolute sum loses about
  0.001 1/m). Recommendation: adopt the one-sided border rule locally; propose upstream a
  centre-relative accumulation option for zero-sum kernels in `GPURasterNeighborhood`.
- `GPUTextureShading` has its own separable Gaussian. `GPURasterGaussianBlur` uses the same sampled
  and sum-normalised weights and the same clamp border, but caps the radius at 8 and has no
  cascade. Local radius is `ceil(3 sigma)`; the official default is `sigma = radius / 2`. No
  conflict. Recommendation: propose upstream a larger radius or a cascade.
- `relief-visualization-utils.ts` has its own box mean. `GPURasterBoxBlur` caps the radius at 8 and
  averages absolute float32 values (the code comment at `:122` says so). The local filter is the
  right choice.
- `GPUTerrainRuggedness` is a fused 3x3 kernel. No official class computes TRI. Roughness could be
  `GPURasterMorphology` dilation minus erosion, at the cost of two extra passes.

No file in scope imports a private luma.gl path.

## 6. Test oracles

I read the generator headers in `modules/analysis/test/oracles` and mapped parity specs to classes.
I did not run them.

Independent oracle (a reference library produced the fixture):

- `GPUTerrainDerivatives`, `GPUTerrainRuggedness`: gdaldem through the libgdal in rasterio
  (`terrain-derivatives.py`). Derivatives are compared with `borderMode: 'nodata'` only; ruggedness
  in both edge modes.
- `GPUTerrainCurvature`: xDEM for Zevenbergen-Thorne and Florinsky; NumPy pseudoinverse fits and a
  SciPy fundamental-form eigenproblem for all 15 kinds (`parity-wp5-terrain-curvature.py`,
  `parity-wp5-round8-curvature.py`).
- `GPUTerrainHorizon` angles, `GPUPointHorizonProfile`: GRASS `r.horizon`, cardinal rays,
  non-negative angles only; SciPy interpolation for point profiles.
- `GPUTerrainCastShadow`: GRASS `r.sunmask`, cardinal azimuths, hard shadow.
- `GPUTerrainViewshed`, `GPUTerrainLineOfSight`, `GPUTerrainCumulativeViewshed`,
  `GPUPointHorizonVisibility`: GDAL viewshed and GRASS `r.viewshed`
  (`parity-wp5-round8-grass.py`), with model differences recorded rather than tolerated.
- `GPUSolarPosition`: NOAA spreadsheet evaluated in NumPy; NREL SPA through pvlib as a separate
  accuracy measure.
- `GPUSolarIrradiance`: pvlib isotropic transposition for the direct term and tilt factor. The
  terrain sky factor is explicitly not covered (terrain-surface-01).
- `GPUSolarShadowMask`: SciPy semicircular distribution for the disk fraction.
- `GPULocalDominance`, `GPUMultiScaleRelief`, `GPUSimpleLocalRelief`, `GPUReliefBlend`: RVT, with
  SciPy composites.
- `GPUTerrainContours`: a parity spec exists; I did not read its generator.
- `GPUTerrainRGBDecode`: exhaustive exact check over all 2^24 codes.

Composite reference written for the test (independent code, not a reference library):

- `GPUTerrainTopographicPosition`, `GPUTerrainWeissLandforms`, `GPUTerrainVectorRuggedness`,
  `GPUTextureShading` (`parity-wp5-round8-composites.py`). The texture case is a component
  comparison, not equivalence to Brown's FFT operator.

Only a CPU copy of the same formula:

- Sky-view factor, positive and negative openness, anisotropic sky-view factor
  (`terrain-horizon-oracle.ts`). The horizon angles under them are independently checked; the
  reductions and the RVT direction convention are not (terrain-surface-10).
- `GPUReliefShading` multidirectional hillshade and the Imhof swing (`relief-shading-oracle.ts`).
  A gdaldem `-multidirectional` fixture would close this.
- `GPUTerrainSpikeRepair` (`terrain-spike-repair-oracle.ts`). There is no outside reference.
- The sweep algorithm against a float64 digital-line oracle of the same sampling model.

Nothing pins: border cells under `'clamp'` against any reference (terrain-surface-03); the
relation between `rowDirection` and the grid affine (terrain-surface-02).

## Resolution (2026-11-10)

Paths are relative to `modules/analysis`. Specs were run on the headless WebGPU device of this
machine (`--project headless`) or in Node (`--project node`).

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| terrain-surface-01 | fixed | `src/gpu-terrain/terrain-illumination/gpu-terrain-horizon.ts:354` new optional `output.irradianceSkyViewFactor`, `mean(cos^2(max(h, 0)))`, reduced from the horizon map by the kernel at `:1060`; `output.skyViewFactor` (Zaksek) is unchanged. `gpu-solar-irradiance.ts:278`, `:301` name which factor to bind and say that the diffuse term is for a horizontal receiver. `docs/api/gpu-terrain/operators.md` (horizon and irradiance bullets, the irradiance example, the `diffuseIrradiance` row). No tilt factor was added, see note 1. | `test/gpu-terrain/terrain-illumination/irradiance-sky-view-factor.spec.ts`: uniform 30 and 10 degree horizons give 0.75 and 0.969846 (march, sweep, float32, unorm16); a plane of gradient 0.5 gives `(1 + cos s) / 2`; the horizon-to-irradiance pipeline gives 75 Wh/m^2 for 100 W/m^2 over one hour. | None for existing callers. A caller who follows the corrected documentation and binds the new output gets the right diffuse energy (75 against 50 Wh/m^2 in the example). Without `output.horizon` the new output keeps a transient float32 horizon map. |
| terrain-surface-02 | fixed | `src/gpu-terrain/terrain-curvature/terrain-geomorphometry-utils.ts:41` `resolveTerrainRowDirection`; used at construction by `gpu-terrain-derivatives.ts:219`, `gpu-terrain-curvature.ts:324`, `gpu-terrain-vector-ruggedness.ts:208`, `gpu-relief-shading.ts:421`, `gpu-terrain-horizon.ts:810`, `gpu-terrain-cast-shadow.ts:284`. Each class has a new read-only `rowDirection` field. `affine[0] < 0` and rotated `'uniform'` grids are not rejected; the class documentation states what they mean. | `test/gpu-terrain/terrain-row-direction.node.spec.ts` (default from the affine, throw on a contradiction, six classes); `test/gpu-terrain/terrain-analysis/terrain-border-row-direction.spec.ts` ("takes north from the grid affine": aspect 180 on the report's input). | `rowDirection` now defaults to the sign of `grid.affine[4]`: a grid built with `rowDirection: 'north'` and contributor props left at default used to be mirrored and is now right. An explicit `rowDirection` that contradicts the affine throws `<id> rowDirection '<value>' must match the grid: ...`. When `affine[4]` is 0 the prop alone decides (`'south'` by default). |
| terrain-surface-03 | fixed in two classes; three unchanged | New `'one-sided'` border: `terrain-geomorphometry-utils.ts:125` (`TERRAIN_ONE_SIDED_BORDER_WGSL`) and `:180` (one-cell padded band for the official `GPURasterGradient`). Default of `GPUTerrainDerivatives` (`gpu-terrain-derivatives.ts:154`, `:251`) and `GPUTerrainCurvature` (`gpu-terrain-curvature.ts:248`, `:659`). `'clamp'` stays available by name. `GPUReliefShading` gets `'one-sided'` as an option, default still `'clamp'` (`gpu-relief-shading.ts:335`, `:458`). `GPUTerrainVectorRuggedness` and the slope of `GPUTerrainWeissLandforms` are unchanged (clamp), see note 2. | `terrain-border-row-direction.spec.ts`: plane slope to the border, quadratic surface, rough terrain against the float64 `GPUFiniteDifference2D` one-sided stencil, validity pattern equal to clamp, zero curvature on a plane for three methods, exact Laplacian on a quadratic, and the report's clamp numbers (5.71 degrees, -0.019704, -0.009852) under `borderMode: 'clamp'`. | Border ring of slope, aspect, hillshade and every curvature changes by default. Plane of gradient 0.2 on 10 m cells: border slope 5.71 to 11.31 degrees; Zevenbergen-Thorne profile curvature -0.0197 to 0. The default derivatives graph has two more nodes (canonicalize, pad). Pass `borderMode: 'clamp'` for the old output. |
| terrain-surface-04 | fixed (documentation) | `terrain-horizon-sweep.ts:380`, `gpu-terrain-horizon.ts:80`, `docs/api/gpu-terrain/operators.md` (sweep option). | none (words) | none |
| terrain-surface-05 | fixed by session `luma-contributors-42` | `gpu-relief-shading.ts:754` now calls `guardedAtan2`. Not my change. | that session's | flat cells under `'imhof-swing'` cannot become nodata |
| terrain-surface-06 | fixed | `src/gpu-terrain/point-horizon/gpu-point-horizon-visibility.ts:305`: the band is `[occluder - tol, occluder + tol)`, as in `GPUTerrainLineOfSight`. | `test/gpu-terrain/point-horizon/gpu-point-horizon-visibility.spec.ts` ("classes are ordered"); it fails on the old rule (`visible` where `marginal` is expected). | A target exactly `tol` below the occluder: `visible` to `marginal`. |
| terrain-surface-07 | fixed (documentation) | `gpu-terrain-curvature.ts:53`; `docs/api/gpu-terrain/operators.md` (curvature paragraph). Also `:264` "spherical" to WGS84. | none (words) | none |
| terrain-surface-08 | fixed (documentation) | `gpu-texture-shading.ts:57-59`; the generated `detail` row in `docs/api/gpu-terrain/operators.md`. The claim about Brown's usual range was not added: I could not check it. | none (words) | none |
| terrain-surface-09 | fixed | `src/gpu-terrain/point-horizon/point-horizon-march.ts:523` (`POINT_HORIZON_GROUND_SCALE_WGSL`, constants from `analysis-core/contracts/earth.ts`), `:1155` (`marchRay` runs in sphere metres with curvature `c s^2` and limits divided by `s`, and returns ground metres); `gpu-point-horizon-visibility.ts:209`; `gpu-point-horizon-profile.ts:289` (azimuths are ground bearings). The great circle is still traced on the mean sphere. See note 3. Float64 oracle updated: `test/gpu-terrain/point-horizon/point-horizon-oracle.ts`. | `gpu-point-horizon-visibility.spec.ts` ("measures a Web Mercator tile in WGS84 ground metres"): meridian arc and equatorial arc closed forms decide a `maximumDistance` cut and the target angle. `gpu-point-horizon-profile.spec.ts` (march equals pyramid bit for bit; 0.002 degrees against the updated oracle). | Web Mercator only. Distances change by -0.56% to +0.45% north-south and +0.11% to +0.45% east-west (equator to pole); tangents, the curvature drop, the per-frame `maximumDistance` cut, `targetIgnoreDistance` and distance outputs follow. Profile azimuths are ground bearings: the traced direction moves by up to 0.19 degrees (45 degree rays at the equator). A ray reaches `min(1, scale)` times the topology `maximumDistance`. Planar is bit-identical. |
| terrain-surface-10 | deferred (untested) | Convention stated in compass terms and marked unverified: `gpu-terrain-horizon.ts:580-600`, class documentation, `docs/api/gpu-terrain/operators.md`. | none | none. Reason: `rvt` is not importable here (`python3 -c "import rvt"` fails), so no fixture could be made. |
| luma-official-03 | fixed | The cast is gone: `createNetworkAnalyticsTopology` and its props type are deleted from `src/analysis-core/network-topology.ts` (it had one caller). `src/gpu-terrain/terrain-decode/gpu-terrain-spike-repair.ts:497` uses `createMaskedWeakComponentNodes`; the unused zero-word buffer and two imports are removed. | `test/gpu-terrain/terrain-decode/spike-repair-component-labels.spec.ts` (labels equal a union-find over the definition; a serpentine component converges within the default 32 iterations); the eight existing spike-repair specs pass unchanged. | Converged output is identical. A run that does not converge within `maximumComponentIterations` may now stop at a different iteration than before, because the labelling algorithm is different (see note 4). |

Extra check asked by the lead (compensated float sums): the one mention in my directories,
`point-horizon-march.ts:65`, is a sentence saying the distance lattice needs no Kahan compensation.
There is no sum: `latDistance` is `LAT_BASE[octave] + f32(j) * LAT_STEP[octave]`, exact by
construction. Nothing to replace.

### Notes

1. Tilt factor (terrain-surface-01). I did not add `(1 + cos(slope)) / 2` to the diffuse term
   although the contributor can have the slope bound. On an infinite plane of slope `s` the plane
   is its own horizon, `tan(h) = tan(s) cos(azimuth - uphill)`, and
   `mean(cos^2(max(h, 0))) = (1 + cos(s)) / 2` exactly. A horizon measured on the same terrain
   already contains the receiver's slope, so multiplying by the tilt factor would count it twice
   (0.870 instead of 0.933 at 30 degrees). The spec pins this identity. The existing pvlib parity
   spec also feeds `(1 + cos(tilt)) / 2` in as the factor, which a built-in tilt factor would have
   doubled. What is still approximate: the slope-azimuth term of Dozier and Frew, and
   terrain-reflected radiation. Both are documented.
2. Border default in three classes (terrain-surface-03). `GPUReliefShading`, vector ruggedness and
   the Weiss slope keep the clamp border by default. So by default the hillshade of
   `GPUReliefShading` now differs from `GPUTerrainDerivatives` in the outer ring of pixels (they
   were equal before). `GPUReliefShading` has `borderMode: 'one-sided'` to match, and the spec that
   compares the two classes now runs that pairing. Making it the default there, and giving the two
   topographic-position classes the same border (the shared WGSL fits both), is a follow-up.
3. Ground model (terrain-surface-09). The scale is the WGS84 metric at the observer latitude,
   constant along a ray, as the raster-grid scale is constant along a row. The bearing conversion
   is pinned only by the float64 oracle, which was updated with the same rule; the distance scale
   has independent closed-form checks on both axes.
4. Existing tests changed because they encoded the old behaviour: `gpu-terrain-derivatives.spec.ts`
   and `gpu-terrain-curvature.spec.ts` now name `borderMode: 'clamp'` (their TypeScript oracles
   model clamp) and build rows-north grids where they pass `rowDirection: 'north'`; the same grid
   change in `gpu-terrain-horizon.spec.ts`, `gpu-terrain-cast-shadow.spec.ts`,
   `parity-wp5-terrain-curvature.parity.spec.ts` and
   `test/analysis-core/kernel-audit/cases-terrain.ts`; `terrain-analysis.node.spec.ts` expects the
   five-node default derivatives graph.

### Upstream gaps

- `GPURasterGradient` (and `GPURasterNeighborhood`) have no one-sided or extrapolating border
  mode. With one, `GPUTerrainDerivatives` and `GPUReliefShading` would not need the padded band.
- `GPUFiniteDifference2D` has constant spacing, no no-data rule and sums absolute float32 values,
  so `GPUTerrainCurvature` cannot delegate. It would need per-row spacing, validity, and a
  centre-relative accumulation for zero-sum stencils.
- `GPUGraphAlgorithmsConnectedComponents` accepts only a `GPUGraphAlgorithmsTopology` rebuilt from
  source edge columns. A constructor (or an algorithm input) that takes a caller-built CSR as
  command-graph views would let `GPUTerrainSpikeRepair` delegate again.

### Review follow-up

From `fix-review-fields.md` and the test typecheck gate.

- **fr-fields-01, fixed.** `GPUPointHorizonProfile` and `GPUPointHorizonVisibility` now resolve
  `rowDirection` with the same helper as the other terrain classes:
  `src/gpu-terrain/point-horizon/point-horizon-march.ts:435`
  (`resolveTerrainRowDirection(id, grid, options.rowDirection)` in `resolvePointHorizonModel`),
  with the prop text at `:312` and the type text in `point-horizon-parameters.ts`. The default is
  the sign of `grid.affine[4]`, an explicit contradiction throws, and the prop alone decides when
  `affine[4]` is 0. The rule is applied for both projections; Web Mercator kernels do not read the
  direction, so the only effect there is that an explicit `'north'` on a rows-south window now
  throws instead of being ignored. No other `rowDirection ??` default is left under
  `src/gpu-terrain`.
  - Pinned by `test/gpu-terrain/terrain-row-direction.node.spec.ts` (now eight classes) and by
    "takes north from the grid affine on a rows-north planar grid" in
    `test/gpu-terrain/point-horizon/gpu-point-horizon-profile.spec.ts`: a plateau 200 m north of
    the observer gives a 13.5 degree skyline at azimuth 0 with the prop omitted, and at azimuth
    180 when the same samples are read as a north-up raster.
  - Behaviour change: a planar point-horizon profile on a rows-north grid with `rowDirection`
    omitted used to be mirrored north to south and is now right.
  - `GPUPointHorizonVisibility` output does not depend on the row direction on a planar grid: the
    target bearing and the march step use the same sign, so the ray joins the same two pixels
    either way. The node spec covers its default and its throw; no headless case can tell the two
    directions apart.
  - `test/analysis-core/kernel-audit/cases-terrain.ts`: the two point-horizon cases that pass
    `rowDirection: 'north'` now build a rows-north grid.
- **Both row signs for horizon sectors and cast shadows.** The existing rows-north cases compare
  against TypeScript models with the prop passed explicitly. Added
  `test/gpu-terrain/terrain-illumination/rows-north-illumination.spec.ts`, closed forms with the
  prop omitted: a plateau 50 m away gives `atan(30 / 50)` in the north sector on a rows-north grid
  and in the south sector on a north-up grid (march and sweep); a 20 degree sun behind the plateau
  is shadowed and the opposite sun is lit, and the two swap with the grid.
- **TS4111 in `terrain-border-row-direction.spec.ts`.** `runCurvature` (`:109`) is now generic in
  the requested curvature kinds and returns `Record<Kind, number[]>`, so `.profile`, `.mean` and
  `.laplacian` are declared properties. No assertion changed.

### Closing pass (2026-11-11)

Sky view stream. Reference: unmodified rvt-py 2.2.1 (`rvt.vis.sky_view_factor`,
`rvt.vis.sky_view_factor_compute`) with NumPy 2.5.4 on CPython 3.14.8, in the pinned environment.
The installed signature is `sky_view_factor(dem, resolution, compute_svf, compute_opns,
compute_asvf, svf_n_dir, svf_r_max, svf_noise, asvf_dir, asvf_level, ve_factor, no_data)`;
`a_main_direction`, `a_poly_level` and `a_min_weight` are arguments of `sky_view_factor_compute`.

| Item | Outcome | Files | Spec that pins it | Evidence |
| --- | --- | --- | --- | --- |
| terrain-surface-10, anisotropic sky-view angle convention | verified, no change to the kernel. The "unverified" wording is removed | `src/gpu-terrain/terrain-illumination/gpu-terrain-horizon.ts` (documentation of `getTerrainHorizonAnisotropicWeightWGSL` and of the class only), `docs/api/gpu-terrain/operators.md` (horizon bullets), new `test/oracles/terrain-sky-view-rvt.py`, `test/oracles/fixtures/terrain-sky-view-rvt.json`, a row in `test/oracles/GENERATORS.md` | `test/gpu-terrain/terrain-illumination/gpu-terrain-horizon-rvt.spec.ts` (headless, 5 tests) | rvt-py 2.2.1. The kernel's reading was right: see below |
| Sky-view factor `1 - mean(sin(max(h, 0)))` (the Zaksek output terrain-surface-01 kept) | verified, no change | none | same spec, "equal rvt where the samples coincide" | rvt `svf`, largest difference 2.2e-7 over 28 case and algorithm pairs (31 with the rows-north runs) |
| Positive and negative openness | verified, no change | none | same | rvt `opns` of the elevation and of its negation, largest difference 2.0e-5 degrees |
| Row direction from the grid affine (terrain-surface-02) under the anisotropic weights | verified, no change | none | same spec, "equals rvt on a rows-north grid with rowDirection left out" | the fixture's terrain stored south row first on a grid with `affine[4] > 0` gives rvt's numbers after flipping the rows back |

What rvt does, from `rvt/vis.py` and from the generator's probe (which asserts it):

- Direction `i` of `n` has angle `i * 360 / n`. `horizon_shift_vector` turns it into the roll
  `(round(cos * r), round(sin * r))` over (rows, columns), and `numpy.roll` brings the neighbour at
  `(row - shift, column - shift)` onto the pixel. So angle 0 looks toward row 0, 90 toward column
  0: on a north-up array 0 is north, 90 west, 180 south, 270 east. The angle runs counterclockwise
  from north.
- The weight is `(1 - a_min_weight) * cos((a_i - a_main_direction) / 2) ** a_poly_level +
  a_min_weight`, with `a_i` in `[0, 360)` and the difference not wrapped (so an odd level is not
  periodic in the main direction). `asvf = sum(w * (1 - sin(max(h, 0)))) / sum(w)`. The presets
  are (4, 0.4) and (8, 0.1); the default `asvf_dir` is 315.
- Probe: a 12 m plateau north, west, south, east and north-east of the centre pixel gives the
  lowest centre `asvf` at main direction 0, 90, 180, 270 and 315, the highest at the opposite
  direction, and equal values at symmetric offsets. With 16 directions a plateau at compass
  azimuth 22.5 gives 337.5. The default 315 therefore weights the north-east sky.
- rvt has no georeference. Row 0 is the first array row; I took it as north, which is what a
  north-up raster is. I did not run `rvt.default` (it needs GDAL, which is not installed).

That is the convention the kernel and its documentation already had (sector `d` is rvt direction
`(n - d) mod n`), so nothing was changed in the kernel. Sensitivity check: with the main direction
mirrored (`360 - A`) on the kernel side, three of the five tests fail (`wall-west-8`: heaviest
direction 270 for an expected 90; anisotropic differences 0.27 and 0.60).

Size of the agreement and of the differences (19 x 19 grids, 2 m cells, radius 6, interior pixels
farther than the radius from every edge):

- Same samples (march with 4 sectors; sweep with 4 or 8; planes with 8 under both): sky-view factor
  within 2.2e-7, anisotropic factor within 2.4e-7 (both presets and level 3 with minimum weight
  0.2, main directions every 45 degrees), openness within 2.0e-5 degrees.
- Different samples. rvt reads whole pixels: 1 to the radius on the axes, `(k, k)` on the
  diagonals, a rounded staircase in between, each at its centre-to-centre distance. The march is
  bilinear along the exact ray; the sweep reads a digital line with distances along the ideal ray.
  - Plane of gradient 0.5 rising north-east, 16 sectors, where the closed form is known
    (`tan(h) = 0.5 cos(t - 45)`, sky-view factor 0.854471, openness 90): march equals the closed
    form (2.8e-7, 8e-5 degrees). rvt reads 0.8328 and 86.68 degrees (0.0217 and 3.32 degrees low).
    The sweep reads 0.835 to 0.842 and 87.4 to 88.3 degrees (up to 0.019 and 2.58 degrees low).
    The spec pins the three statements.
  - Steep test terrain (a 12 m step on 2 m cells; a sum of sines with slopes above 60 degrees),
    march against rvt: 8 sectors up to 0.059 (sky-view factor), 0.20 (anisotropic), 4.1 degrees
    (openness); 16 sectors up to 0.114, 0.20 and 10.9 degrees (sweep: 0.105, 0.20, 14.5 degrees).
    These are upper ends for deliberately rough inputs and are not pinned.
- Edge band. rvt reflect-pads the array by the radius; the kernel stops a ray at the grid edge and
  reports 0 for a ray with no sample. Inside the band the two differ by up to 0.57 (sky-view
  factor) and 52 degrees (openness) on the rough grid. This is the documented halo contract, not a
  defect.

Not verified:

- No-data. From the source, rvt gives an all-NaN ray the slope -1000 (openness term -89.9 degrees);
  the kernel reports 0 for a ray with no valid sample. The sky-view factors agree in that case and
  openness does not. I did not run a no-data case on either side.
- `ve_factor` against `zFactor` (only 1 was run), `svf_noise` above 0 (the kernel has no minimum
  radius), curvature, `maximumDistance`, `stepGrowth` above 1, the `'geographic'` and
  `'web-mercator'` cell modes and the `'unorm16'` horizon format.
- Values for sectors between the axes and the diagonals cannot be compared with rvt number for
  number (the staircase). For 16 sectors the evidence is the ranking test and the plane.

No caller-visible number changed, so `behaviour-changes.md` has no row for this stream.

Reported, not changed: the sweep's bias on smooth slopes (0.019 in sky-view factor and 2.6 degrees
of openness on a plane of gradient 0.5 at 16 sectors, the same size as rvt's). It follows from the
documented nearest-pixel sampling of terrain-surface-04; the horizon bullets of
`docs/api/gpu-terrain/operators.md` now give the number.
