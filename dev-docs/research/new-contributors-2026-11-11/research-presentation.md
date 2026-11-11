# Presentation computations: research for new contributors, 2026-11-11

Scope: GPU computations whose output feeds a map (positions, geometries, rasters, classes): cartograms, labels, terrain presentation, imagery presentation, uncertainty, flow presentation, embeddings and vector generalization. Rendering stays in luma.gl and deck.gl. Each candidate was checked against `census.md` and the source under `modules/analysis/src`.

## 1. Summary

The library already covers more presentation work than the brief assumed. Five items on the brief are already built, so they are not proposed:

- **Sky-view factor, positive and negative openness, anisotropic sky-view factor.** `GPUTerrainHorizon` outputs `skyViewFactor`, `irradianceSkyViewFactor`, positive openness, `negativeOpenness` and `anisotropicSkyViewFactor`, checked against RVT (`docs/api/gpu-terrain/operators.md`, about lines 1083 to 1141). Ambient occlusion for relief is the same quantity.
- **Hypsometric tint.** `GPUReliefShading` has elevation tint stops and an Imhof aspect tint (`gpu-relief-shading.ts`, lines 94 and 163).
- **Evenly spaced streamlines on the GPU.** `GPUStreamlines` is "evenly spaced streamlines (after Jobard and Lefer 1997) resolved by priority instead of processing order" (`gpu-streamlines.ts`, line 129).
- **Overlap removal for circular proportional symbols.** This is `GPUDorlingCartogram`.
- **Colour quantization** (`GPUKMeans`), **OD matrix reordering** (`GPUAdjacencyMatrixOrder`), and **polygon collapse to centre lines** (`GPUVectorStraightSkeleton`, `GPURasterMedialAxis`).

The strongest new candidates are the following. Section 10 ranks them all.

1. A flow-based contiguous cartogram, built on FFT.
2. GPU point-feature label placement, built on summed-area tables.
3. CLAHE together with histogram matching.
4. Displacement interpolation between two density rasters. It reuses the log-domain separable kernel in `GPUSinkhornDivergence`.
5. Hierarchical point clustering for level of detail.
6. A centred travel-time map.
7. Field-based t-SNE.
8. Tanaka contours, value-suppressing uncertainty palettes, pansharpening and colour-vision-deficiency simulation. These four are small.

## 2. Cartograms and deformations

### 2.1 Flow-based density-equalising cartogram (Gastner, Seguy and More 2018)

- **What the reader gains.** A contiguous map in which each region's area is proportional to a value such as population or votes, and neighbours stay neighbours. The examples in the paper are the 2016 US election, Indian and Chinese GDP, and deaths in Kensington and Chelsea ([PNAS full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5877977)).
- **Method** ([PMC5877977](https://pmc.ncbi.nlm.nih.gov/articles/PMC5877977)).
  - Density relaxes linearly, `ρ = (1−t)ρ₀ + tρ̄` for t ≤ 1. Points move with the velocity of the continuity equation.
  - The velocity is a sine/cosine Fourier series. Its coefficients are computed once, by FFT, on an Lx × Ly grid; the examples use 512 × 512.
  - The map sits in a box 1.5 times its larger extent, padded with the mean density, and the density is blurred first.
  - Integration to t = 1 uses a predictor-corrector scheme with an adaptive step, typically about 100 steps.
  - Rounds repeat until the maximum relative area error is below 1%: 5 rounds for the US, 12 for India, 6 for China.
  - Reported times on a 12-core Xeon: 1.5 to 2.7 s, against 59.5 to 178.5 s for the 2004 diffusion method.
- **Diffusion method.** The Gastner-Newman 2004 diffusion method ([PNAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC419634)) is superseded by the flow-based method: same output, slower. Skip it.
- **Follow-up.** A 2025–2026 preprint adds topology-preserving line densification and a quadtree. The authors call it 5FCarto and claim greater accuracy and speed ([arXiv 2511.08121](https://arxiv.org/abs/2511.08121v2)).
- **Oracles.**
  - `go_cart` (C, FFTW): "Adapted from the MIT License", with a clause that images made with it cite the 2018 paper ([LICENSE](https://raw.githubusercontent.com/Flow-Based-Cartograms/go_cart/master/LICENSE)). The repository says it is no longer maintained ([repo](https://github.com/Flow-Based-Cartograms/go_cart)).
  - `cartogram-cpp`: AGPL-3.0, with its output data under MIT ([repo](https://github.com/mgastner/cartogram-cpp)).
  - `go-cart-wasm`: a JS/WASM port, MIT ([npm](https://www.npmjs.com/package/go-cart-wasm)).
  - Use these as offline oracles that produce fixture files. Copy no code; AGPL rules that out in any case.
- **Parallel structure.** Every step is data-parallel:
  - `GPUPolygonRasterization` for the density raster;
  - `GPUConvolution` for the blur;
  - `GPUFFT2D` for the coefficients;
  - a per-grid-point kernel for the velocity;
  - a per-vertex predictor-corrector with bilinear interpolation (as in `GPUParticleAdvection`);
  - `GPUReduction` for the area error and the convergence flag.

  The rounds and steps are the contract's iteration with a budget. One caveat: the published `GPUFFT2D` is a radix-2 complex transform (`gpu-fft2d.js`, "bounded radix-2"). The cosine transform therefore needs an even extension: a 1024² complex FFT for a 512² grid.
- **Neural network.** None. **Effort:** M to L. **Verdict: do.** It is the canonical cartogram, the only known gap next to Dorling, and FFT-native.

### 2.2 Travel-time (time-space) maps

- **What the reader gains.** Distance on the map reads as travel time.
- **All-pairs version.** This uses multidimensional scaling, which minimises a weighted stress function. The geography is then deformed with moving least squares ([ICA proceedings 2019](https://ica-proc.copernicus.org/articles/2/103/2019/ica-proc-2-103-2019.pdf); [Springer 2022](https://link.springer.com/article/10.1007/s42489-022-00124-4)). The sources note that plain MDS can break topology between adjacent nodes ([MDPI 2022](https://mdpi.com/2220-9964/11/8/432/html)).
- **Parallel structure.**
  - A SMACOF update is an O(n²) sum per node, so it is data-parallel per node through `GPUMatMul` for n of a few thousand.
  - Moving least squares is per vertex, with a sum over control points.
  - `GPUNetworkCostMatrix` already supplies the travel-time matrix.
- **Centred variant.** This is my own design, not taken from a paper. It needs one origin and the travel times from `GPUNetworkIsochrones`, then a radial rescale of every vertex by interpolated time over distance. It is per-vertex and per-frame, so the pointer can be the origin (`place`).
- **Oracle.** scikit-learn `MDS`, which uses SMACOF (BSD-3). I did not open its source.
- **Neural network.** None.
- **Effort and verdict.** Centred variant: S, **do**. All-pairs: M, **maybe**, because of the topology faults.

### 2.3 Tile-grid and hexagon cartograms

- **Sources.** McNeill and Hale generate and rank square, hexagon and triangle tile maps ([CGF 2017](https://diglib.eg.org:443/handle/10.1111/cgf13200)); the R package `tilemaps` implements them ([README](https://cran.nics.utk.edu/cran/web/packages/tilemaps/readme/README.html)). Eppstein et al. treat tile assignment as weighted point-set matching ([PDF](https://typeset.io/pdf/improved-grid-map-layout-by-point-set-matching-425z54gks2.pdf)).
- **Parallel structure.** The auction algorithm parallelises, one bidder per thread ([arXiv 1510.02065](https://arxiv.org/pdf/1510.02065)).
- **Verdict: skip.** A few hundred regions do not need a GPU. If wanted, write a CPU twin.

### 2.4 Focus-plus-context lenses and necklace maps

- **Fisheye projection.** A per-vertex formula. It belongs in a deck.gl vertex shader, not in a contributor.
- **Haunert and Sering (2011).** The road-network focus map is a convex quadratic program ([Würzburg](https://informatik.uni-wuerzburg.de/en/algo/research/geographic-information-systems/variable-scale-maps); [code](https://www.github.com/LeonSering/focus_region)). Sources disagree on the author list.
- **Necklace maps** ([TVCG 2010](https://research.tue.nl/en/publications/necklace-maps/)) place symbols on intervals of a curve. This is combinatorial, for small n.
- **Verdict: skip** both.

## 3. Labels and symbols

### 3.1 GPU point-feature label placement (Rapid Labels, Pavlovec and Čmolík 2022)

- **What the reader gains.** Readable names with no overlaps, recomputed at interactive rates.
- **Method** ([TVCG 2022, author PDF](https://dcgi.fel.cvut.cz/wp-content/wpallimport-dist/publications/pdf/publications-2022-pavlovec-tvcg-rapid-paper.pdf)).
  - Eight candidate positions per point.
  - Screen-space buffers: obstacles, conflicts, ambiguity and placed labels. Each candidate is scored in constant time with summed-area tables.
  - The screen is split into blocks twice the largest label's size, each with four quadrants. All points in the same quadrant of different blocks are placed in parallel.
  - An atomic maximum on a priority buffer picks one point per block, and the summed-area tables are updated in place.
  - Priority groups and a zoom-consistent mode are supported.
- **Reported results.**
  - Java and OpenGL on a GTX 1660 Ti.
  - 8,000 points take 79.01 ms, against 2,520 ms for particle-based labelling.
  - It places more labels than the greedy methods compared, but fewer than Zoraster's simulated annealing.
  - On 3,340 US airports it took 11% of particle-based labelling's time.
- **Licence.** No code release or licence was found.
- **Simulated annealing.** Zoraster and Ebner et al. place more labels ([Ebner 2003](https://www.ac.tuwien.ac.at/files/pub/ebner-03.pdf)) but are sequential.
- **Force-based labelling.** It scales to several hundred labels ([Vaaraniemi 2012](https://www.cs.cit.tum.de/fileadmin/w00cfj/cg/Research/Publications/2012/Force_Labeling/2012_Vaaraniemi_Temporally_Coherent_Real-Time_Labeling_of_Dynamic_Scenes.pdf)).
- **Building blocks.** `GPUScan` along rows then columns for the summed-area table, `GPUGridIndex` for blocks, and atomics. The caller supplies label width and height.
- **Oracle.** No public implementation was found. Write a CPU twin of the published algorithm and check the invariants: no overlaps, and a count no greater than the greedy optimum on small cases.
- **Neural network.** None. **Effort:** M. **Verdict: do.** The website now draws text from a signed-distance atlas (roadmap WEB-INT-16), so the labels can appear in the picture.

### 3.2 Hierarchical point clustering for level of detail

- **Supercluster.** It does greedy clustering per zoom, indexed by a KD-tree, with defaults radius 40 px, extent 512, minPoints 2 and maxZoom 16. Licence ISC ([repo](https://github.com/mapbox/supercluster); [Mapbox blog](https://blog.mapbox.com/clustering-millions-of-points-on-a-map-with-supercluster-272046ec5c97)).
- **Why it does not port directly.** Greedy order is sequential.
- **GPU form (my design).** For each zoom level, from fine to coarse:
  - candidates in a `GPUGridIndex` with cell size equal to the radius;
  - a Luby-style maximal independent set, where a point survives if its hash priority beats every neighbour within the radius;
  - every remaining point joins its nearest survivor;
  - `GPUSegmentedReduction` sums counts and weighted centroids.

  Each level takes a few rounds with a budget. The result is not identical to supercluster's, so the oracle checks invariants: counts conserved, no two centres within the radius, and every member within the radius of its parent.
- **Same kernel for sampling.** The same kernel gives blue-noise (Poisson-disk) display sampling. Blue noise preserved scatterplot shape best in a user study ([Yuan et al.](https://deepai.org/publication/evaluation-of-sampling-methods-for-scatterplots)). Multi-class blue noise is from [Chen et al. 2014](https://vis.cs.ucdavis.edu/vis2014papers/TVCG/papers/1683_20tvcg12-chen-2346594.pdf).
- **Overlap.** `GPUSpatialBalancedSample` is a probability design, not a display sampler, so it does not overlap.
- **Effort and verdict.** M. **Do**, with the sampler as an option of the same contributor.

### 3.3 Other symbol layouts

- Overlap removal for squares and rectangles is not covered by Dorling. I found no source to assess it, so it is unassessed.
- Beeswarm layouts are a one-dimensional Dorling. **Skip.**

## 4. Terrain presentation that is missing

| Item | Source | Parallel structure | Effort | Verdict |
| --- | --- | --- | --- | --- |
| Tanaka illuminated contours | Kennelly and Kimerling vary contour width with the cosine of the angle between the surface normal and the light ([CaGIS abstract](https://pure.psu.edu/en/publications/modifications-of-tanakas-illuminated-contour-method/); [ArcUser](https://esri.com/news/arcuser/0701/althillshade.html)) | Per segment of `GPUTerrainContours` output: a tone and width from the segment normal and the azimuth | S | **do**: `aim` gesture, per-frame, striking |
| Terrain generalization (Terrain Sculptor) | Ridges and valleys found by curvature, then raster operations ([Cartographic Perspectives 67](https://cartographicperspectives.org/index.php/journal/article/view/cp67-leonowicz-et-al)) | `GPUTerrainCurvature`, `GPUConvolution`, `GPURasterRankFilter`; per pixel | M | **maybe**: freeware, licence not found, so there is no clean oracle |
| Plan oblique relief | Jenny and Patterson, Cartographic Perspectives 57 (2007) ([article](https://cartographicperspectives.org/index.php/journal/article/view/cp57-jenny-patterson)) | Shift each pixel north by elevation times cot(angle). The visible-surface resolve is a bounded walk per column, like the horizon kernels | S to M | **maybe**: I did not read the full text |
| Hachures and slope lines | Kennelly and Kimerling (2000): one hachure per 3 × 3 cell block ([Cartographic Perspectives 37](https://cartographicperspectives.org/index.php/journal/article/download/cp37-kennelly-kimerling/pdf/3966)); gradient lines from a Gaussian-filtered SRTM DEM ([Magyari-Sáska 2017](https://mail.technicalgeography.org/pdf/1_2017/08_magyari.pdf)) | A recipe: `GPUTerrainDerivatives` gradient into `GPUStreamlines`, with width from slope | S | **maybe**, as a recipe, not a contributor |
| Neural relief shading | U-Net with five down- and up-sampling steps, trained on swisstopo manual shading ([arXiv 2010.01256](https://arxiv.org/pdf/2010.01256)); follow-up at four scales ([IJGI 2024](https://mdpi.com/2220-9964/13/9/326)) | A small U-Net is all convolutions, so `GPUConvolution` could run it | L | **skip**: no public weights, parameter count or licence found. Eduard is a paid macOS app with three models and no stated weight licence ([developer page](https://dilpreet-singh-fi60.squarespace.com/projects/eduard); [Map Room](https://www.maproomblog.com/2022/10/eduard-a-new-mac-only-relief-shading-app/)) |
| Sky-view, openness, ambient occlusion, hypsometric tint | | | | **exists** (see section 1) |

## 5. Raster and imagery presentation

| Item | Source | Oracle (licence) | Parallel structure | Effort | Verdict |
| --- | --- | --- | --- | --- | --- |
| CLAHE | Per-tile histograms, clipped, mapped through a CDF lookup table and interpolated bilinearly; a GPU write-up exists ([Accelerated CLAHE](https://www.github.com/YuAo/Accelerated-CLAHE)) | scikit-image `equalize_adapthist`, default tiles 1/8 of each side ([docs](https://scikit-image.org/docs/stable/api/skimage.exposure)) | `GPUHistogram` or a segmented histogram, clip and redistribute per tile, `GPUSegmentedScan`, a per-pixel bilinear lookup. The clip limit can change per frame | M | **do**: roadmap R7.1 only defers it ("Gate-deferred, no code") |
| Histogram matching | Matches the cumulative histogram of a reference image | scikit-image `match_histograms` (same page) | `GPUHistogram`, `GPUScan`, inverse-CDF lookup | S | **do**, in the same contributor as CLAHE; mosaic colour balancing is a recipe on top |
| Pansharpening | Component substitution (IHS, PCA, Gram-Schmidt) and multiresolution (à trous) families ([Vivone et al. 2015](https://arxiv.org/html/1504.04531)) | GDAL weighted Brovey, the only GDAL method ([header](https://cdn.jsdelivr.net/npm/gdal@0.11.1/deps/libgdal/gdal/alg/gdalpansharpen.cpp); [man page](https://www.mankier.com/1/gdal_pansharpen)). The Vivone toolbox licence was not confirmed ([copyright PDF](https://www.iris.unisa.it/bitstream/11386/4635457/4/Pansharpening_Toolbox_Copyright.pdf)) | Brovey per pixel; Gram-Schmidt needs covariances (`GPUReduction`); à trous uses a separable B3 kernel (`GPUConvolution`) | S | **do**, but the website has no scene with a panchromatic band (roadmap WEB-PIC-09) |
| Haze removal | Dark channel prior ([He, Sun, Tang, CVPR 2009](https://people.csail.mit.edu/kaiming/publications/pami10dehaze.pdf)) | none checked | Minimum filter (`GPURasterRankFilter`), brightest 0.1% (`GPUSort`), transmission refined by a guided filter | M | **maybe**: built for outdoor photographs, untested on satellite scenes |
| Guided filter | Linear time, independent of radius, made of box filters ([He et al. PAMI](https://people.csail.mit.edu/kaiming/publications/pami12guidedfilter.pdf)) | OpenCV contrib (not checked) | Box filters through `GPUScan` summed-area tables or `GPUConvolution` | S | **do**: a shared building block for dehazing and edge-aware smoothing |
| Bilateral filter, bilateral grid | Bilateral grid ([Chen, Paris, Durand 2007](https://groups.csail.mit.edu/graphics/bilagrid/)) | | Direct bilateral filter on a bounded window | S | **maybe**: the grid's code was withdrawn over patents, and US patents 8,731,320 and 9,047,656 cite it ([USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8731320)). Use the direct form or the guided filter |
| Anisotropic diffusion | Perona and Malik 1990, local and parallel ([Caltech](https://authors.library.caltech.edu/6498)) | | Iterated four-neighbour stencil (`GPUFiniteDifference2D`) with a budget | S | **maybe** |
| Colour-vision-deficiency simulation and palette check | Machado, Oliveira and Fernandes 2009 ([TVCG](https://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html)) | `colorspacious`, MIT, implements Machado 2009 ([repo](https://github.com/colour-science/colorspacious)) | A 3 × 3 matrix per pixel in linear RGB; the palette check is pairwise distances over at most a few hundred colours | S | **do**: cheap, useful on every map, `swipe` gesture |
| Colour quantization | | | | | **exists** (`GPUKMeans`) |

**Classified rasters.** Edge-preserving filters do not apply to class labels directly. A workable route, which is my suggestion and not from a source: one-hot encode the classes, smooth each probability band with a guided filter, then take the argmax. This complements `GPURasterSieve`.

## 6. Uncertainty and comparison

| Item | Source | Structure | Oracle | Effort | Verdict |
| --- | --- | --- | --- | --- | --- |
| Value-suppressing uncertainty palette | Value resolution narrows as uncertainty grows. A crowdsourced study found that readers weighted uncertainty more than with bivariate maps ([CHI 2018](https://www.tableau.com/research/publications/value-suppressing-uncertainty-palettes)) | Per-row quantization into the palette's tree of bins; per-frame thresholds | `uwdata/vsup`, MIT ([npm](https://www.npmjs.com/package/vsup)) | S | **do**: a class-assignment sibling of `GPUBivariateClassification` |
| Exceedance probability | Fraction of ensemble members above a threshold, per cell | Per cell over m members, or one `GPUReduction` per cell batch | NumPy | S | **do**: a `level` gesture on the threshold. `GPUEnsembleScores` scores against an observation, which is different |
| Contour boxplots | Band depth over ensemble contours gives a median member and outliers ([Whitaker, Mirzargar, Kirby 2013](https://users.cs.utah.edu/~kirby/Publications/Kirby-82.pdf)) | Band depth over member pairs; data-parallel over (member, pair) | none found | M | **maybe** |
| Hypothetical outcome plots | Animated draws ([PLOS ONE 2015](https://idl.uw.edu/papers/hops)) | The draws already exist (`GPUConditionalGaussianSimulation`); the animation is rendering | | | **skip** as a contributor; make it a website scene |
| Displacement interpolation | Entropic optimal transport with a heat-kernel (Gaussian) kernel makes transport tractable on images ([Solomon et al. 2015](https://graphics.pixar.com/library/OptimalTransportation)); a Lagrangian method decomposes into radial basis functions ([Bonneel et al. 2011](https://www.cs.ubc.ca/labs/imager/tr/2011/DisplacementInterpolation)) | The barycentre of two rasters with weights (1−t, t): iterated Sinkhorn scaling, where each kernel application is the separable row and column log-sum-exp already in `GPUSinkhornDivergence` | POT (MIT, [JMLR](https://www.jmlr.org/beta/papers/v22/20-451.html)). I did not confirm the name of its convolutional barycentre function | M | **do**: `time` gesture morphing one density into another, for example morning into evening trips. Precompute about eight values of t if one solve is too slow for a frame |
| Small multiples of space (OD maps) | Origins laid out as a grid of destination maps ([Wood, Dykes, Slingsby 2010](https://openaccess.city.ac.uk/id/eprint/537/)) | `GPUGridAggregation` with a nested key | | S | **do** (see section 7) |

## 7. Flow and movement presentation

- **Spiral-tree flow maps.** Optimal spiral trees are NP-hard; there is a 2-approximation ([arXiv 1109.3316](https://arxiv.org/pdf/1109.3316); [TVCG 2011](https://research.tue.nl/en/publications/flow-map-layout-via-spiral-trees/)). It is sequential geometric construction. **Skip.**
- **OD maps.** Each OD vector is binned by origin cell, then by destination cell within that origin's sub-grid. The authors queried 10⁵ to 10⁶ vectors interactively on a CPU with a hash grid ([preprint](https://openaccess.city.ac.uk/id/eprint/537/1/wood_visualization_2010.pdf)). On the GPU this is one `GPUGridAggregation` with a composite key, S. **Do**: the `probe` gesture is an origin cell, which lights its destination map.
- **Trajectory density with direction.** Scheepens et al. compose density fields from trajectory attributes ([TVCG 2011](https://typeset.io/papers/composite-density-maps-for-multivariate-trajectories-25f1w4uvme)). A direction-binned kernel density (8 or 16 headings, one band each, via `GPUKernelDensity` or `GPULineDensity`) is S. **Maybe**, because it may already be a recipe.
- **Already covered.** Evenly spaced streamlines (`GPUStreamlines`), OD reordering (`GPUAdjacencyMatrixOrder`), bundling (`GPUEdgeBundling`), animated playback (`GPUTrajectoryPlayhead`).

## 8. High-dimensional views linked to a map

| Item | Source | Structure | Oracle | Effort | Verdict |
| --- | --- | --- | --- | --- | --- |
| t-SNE, field-based | Repulsion approximated by splatting kernel textures; linear complexity; shipped in TensorFlow.js ([Pezzotti et al. TVCG 2020](https://arxiv.org/pdf/1805.10817); [Google blog](https://research.google/blog/realtime-tsne-visualizations-with-tensorflowjs/?m=1)). The CUDA lineage is FIt-SNE ([tsne-cuda](https://arxiv.org/pdf/1807.11824)) | kNN through `GPUSimilaritySearch`; affinities as `GPUGraphCSRMatrix`; attraction by SpMV; repulsion by splatting into a grid and `GPUConvolution` (FFT); iteration with a budget | scikit-learn or openTSNE (BSD-3; not opened) | L | **do**: the picture is points coloured by map region, settling into clusters as the time scrubs; probing a cluster lights its tracts |
| UMAP | GPU UMAP in RAPIDS cuML, up to 100× faster ([Nolet et al. AAAI 2021](https://arxiv.org/abs/2008.00325)) | Edge-wise SGD with negative sampling and atomic updates; results depend on scheduling | umap-learn | L | **maybe**: after t-SNE, sharing its kNN graph |
| Self-organizing map as a legend | SOMs for census trajectories ([Skupin and Hagelman 2005](https://geog.sdsu.edu/People/Pages/skupin/research/pubs/GeoInformatica2005.pdf)) | Batch SOM: best-matching unit by `GPUSimilaritySearch`, then a neighbourhood-weighted mean per node; data-parallel | MiniSom (licence conflicting in sources, so check it) | M | **maybe** |
| Parallel-coordinate and scatterplot-matrix density rasters | Continuous parallel coordinates ([Heinrich and Weiskopf 2009](https://pubmed.ncbi.nlm.nih.gov/19834230)) | Discrete form: each row draws one segment per axis pair into an atomic histogram texture. The segment length is bounded by the raster width, so the walk is bounded | NumPy | S | **do**: `window` brushing on an axis selects tracts on the map |

## 9. Vector generalization

| Item | Source | Parallel? | Effort | Verdict |
| --- | --- | --- | --- | --- |
| Building squaring | Least-squares squaring forces near-right angles to exactly 90° (Lokhat and Touya 2016, in [CartAGen docs](https://ignf.github.io/CartAGen/docs/algorithms/buildings/ls_squaring.html)) | Yes across buildings: one small least-squares solve per building, bounded vertex count | M | **maybe**: buildings per scene are within 100,000 |
| Building simplification | CartAGen lists simplification, rectangle transformation, recursive regression and FER ([manual](https://cartagen.readthedocs.io/en/main/manual.html)) | Yes across buildings, sequential within one | M | **maybe**, together with squaring |
| Aggregation and amalgamation | Morphological amalgamation (same manual) | A recipe of buffer out, dissolve and buffer in from `GPUOffsetExpansion` and `GPUCoverageDissolve` | S | **recipe** |
| Typification | Burghardt-Cecconi, Delaunay-based (CartAGen changelog) | Global and iterative; mostly sequential | L | **skip** |
| Road selection by strokes | Strokes chain edges by good continuation; built by a per-node decision, then assembly ([Thomson, Space Syntax](https://www.spacesyntax.net/symposia-archive/SSS4/fullpapers/50Thomsonfpaper.pdf); [Thomson and Richardson 1999](https://geog.leeds.ac.uk/groups/geocomp/2000/abstracts/e029-thomson.htm)) | Yes. Each node pairs its incident edges by deflection angle in parallel; assembly is connected components (`GPUGraphTraversal`); selection is `GPUSort` by stroke length | M | **do**: the `level` gesture keeps the top k strokes, and the network thins while still reading as a network |
| Polygon collapse | | | | **exists** (`GPUVectorStraightSkeleton`, `GPURasterMedialAxis`) |

CartAGen's documentation is under EUPL 1.2; I did not confirm the software's own licence. Use it only as an offline oracle.

## 10. Ranked candidates

| # | Name | What it computes | Building blocks | Effort | Oracle | Picture on the website | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `GPUFlowCartogram` | Vertex positions of a contiguous density-equalising cartogram | `GPUPolygonRasterization`, `GPUConvolution`, `GPUFFT2D` (even extension), per-vertex predictor-corrector, `GPUReduction` | M–L | go_cart, cartogram-cpp, go-cart-wasm (fixtures only) | Montréal boroughs or US states swell to population; `time` gesture from t = 0 to 1 | **do**: the canonical missing cartogram, FFT-native, with a stated convergence criterion |
| 2 | `GPUPointLabelPlacement` | Chosen candidate (0–7 or none) and zoom of first placement for each labelled point | `GPUScan` (summed-area tables), `GPUGridIndex`, atomics | M | CPU twin of the published algorithm, checked by invariants | BIXI stations or airports labelled live; `reach` gesture on font size, or a dragged obstacle | **do**: measured fast on GPU, useful everywhere, and the site can draw text |
| 3 | `GPURasterContrastEqualization` | CLAHE and histogram matching | `GPUHistogram`, `GPUSegmentedScan`, per-pixel lookup | M | scikit-image (BSD-3) | Swipe between the raw and equalized hillshade or imagery; `reach` on the clip limit | **do**: closes deferred R7.1 |
| 4 | `GPUDisplacementInterpolation` | A raster between two densities at t, by entropic transport | Kernel of `GPUSinkhornDivergence`, Sinkhorn scaling with a budget | M | POT (MIT) | Morning trips flow into evening trips; `time` gesture | **do**: large reuse, a picture no other card makes |
| 5 | `GPUPointClusterHierarchy` | Cluster parents, counts and centroids per zoom, plus a blue-noise display sample | `GPUGridIndex`, independent-set rounds, `GPUSegmentedReduction` | M | supercluster (ISC), checked by invariants | 10 million taxi or trip points merging into bubbles; `level` gesture on zoom | **do**: the level-of-detail need for 10 million points |
| 6 | `GPUTravelTimeMap` (centred) | Map vertices rescaled radially by travel time from an origin | `GPUNetworkIsochrones` output, per-vertex interpolation | S | CPU twin | The street map bends around the pointer; `place` gesture | **do**: cheap, striking, per-frame |
| 7 | `GPUFieldTSNE` | Two-dimensional embedding of tract attributes | `GPUSimilaritySearch`, CSR SpMV, splat plus FFT convolution | L | scikit-learn or openTSNE | Embedding beside the map; `probe` links the two | **do**, after 1–6 |
| 8 | `GPUIlluminatedContours` | Tanaka tone and width per contour segment | `GPUTerrainContours` output, per segment | S | Kennelly-Kimerling formula | Contours light and darken with the sun; `aim` gesture | **do**: S effort, strong picture |
| 9 | `GPUUncertaintyPaletteClasses` | Value-suppressing palette class per row; exceedance probability per cell | per-row and per-cell kernels | S | uwdata/vsup (MIT) | Tracts with uncertain rates fade to grey; `level` gesture | **do** |
| 10 | `GPURoadStrokes` | Stroke identifiers, lengths and a selection rank | per-node pairing, `GPUGraphTraversal`, `GPUSort` | M | CartAGen or a CPU twin | The network thins as `level` drops | **do** |
| 11 | `GPUPansharpen` | Weighted Brovey, Gram-Schmidt and à trous sharpening | per pixel, `GPUReduction`, `GPUConvolution` | S | GDAL | Curtain at 30 m against 15 m | **do** once imagery with a panchromatic band exists |
| 12 | `GPUColorVisionSimulation` | Simulated colours and palette distinctness | 3 × 3 per pixel, pairwise distances | S | colorspacious (MIT) | Swipe between normal and deuteranope views of a choropleth | **do** |
| 13 | `GPUGuidedFilter` | Edge-preserving smoothing; one-hot class generalization | summed-area box filters | S | OpenCV (not checked) | Noisy land cover becomes clean patches | **do**: a building block |
| 14 | `GPUDensityAxes` | Parallel-coordinate and scatterplot-matrix density rasters | atomic histogram texture | S | NumPy | Brushing an axis lights tracts | **do** |
| 15 | OD maps | Nested origin–destination grid | `GPUGridAggregation` | S | CPU twin | Probe an origin cell | **do** (may be a recipe) |
| 16 | Terrain generalization, plan oblique, hachure recipe, building squaring and simplification, self-organizing map, UMAP, contour boxplots, haze removal, anisotropic diffusion, all-pairs travel-time map | | | S–L | | | **maybe** |
| 17 | Neural relief shading, tile-grid assignment, necklace maps, spiral trees, typification, beeswarm, hypothetical outcome plots as a contributor, diffusion cartogram | | | | | | **skip**: no weights or licence, small n, sequential, or superseded |

## 11. What I could not verify

- **No GPU timings were measured.** The one-second targets for the cartogram, t-SNE and displacement interpolation are not established. The only GPU label-placement numbers are from a GTX 1660 Ti under OpenGL, not from an integrated GPU under WebGPU.
- **Rapid Labels.** No code or licence was found.
- **Neural relief shading.** No parameter count, public weights or licence were found, for the paper or for Eduard.
- **Licences not confirmed.** The software licence of CartAGen, MiniSom (sources conflict) and the Vivone pansharpening toolbox. The name of POT's convolutional barycentre function was not confirmed.
- **Papers known only from abstracts or catalogue records.** The full text of Haunert and Sering (sources disagree on the authors), McNeill and Hale's assignment step, Whitaker et al.'s algorithm, and Jenny and Patterson's plan oblique parameters were not read.
- **My own designs.** The GPU independent-set clustering, the centred travel-time warp and the one-hot guided-filter class smoothing are mine, not from a cited source. They need their own correctness review.
- **Untested in this repository.** Whether `GPUFFT2D` at 1024² and the Sinkhorn kernel at barycentre sizes fit the command-node budget.
