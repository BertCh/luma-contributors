| Class | File | Published raster/core classes constructed | Verdict |
| --- | --- | --- | --- |
| `GPUGeomorphons` | `modules/analysis/src/gpu-terrain/geomorphons/gpu-geomorphons.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTerrainFlowDistanceOperation` | `modules/analysis/src/gpu-terrain/hydrology/gpu-terrain-flow-distance-operation.ts` | - | adapter of the local class |
| `GPUTerrainFlowDistance` | `modules/analysis/src/gpu-terrain/hydrology/gpu-terrain-flow-distance.ts` | - | own kernel, no published equivalent |
| `GPUTerrainFlowOperation` | `modules/analysis/src/gpu-terrain/hydrology/gpu-terrain-flow-operation.ts` | - | adapter of the local class |
| `GPUTerrainFlow` | `modules/analysis/src/gpu-terrain/hydrology/gpu-terrain-flow.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTerrainHeightAboveDrainage` | `modules/analysis/src/gpu-terrain/hydrology/gpu-terrain-height-above-drainage.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTerrainHydrologicIndices` | `modules/analysis/src/gpu-terrain/hydrology/gpu-terrain-hydrologic-indices.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTerrainStreamNetwork` | `modules/analysis/src/gpu-terrain/hydrology/gpu-terrain-stream-network.ts` | - | own kernel, no published equivalent |
| `GPUTerrainStreamOrderOperation` | `modules/analysis/src/gpu-terrain/hydrology/gpu-terrain-stream-order-operation.ts` | - | adapter of the local class |
| `GPUTerrainStreamOrder` | `modules/analysis/src/gpu-terrain/hydrology/gpu-terrain-stream-order.ts` | - | own kernel, no published equivalent |
| `GPUTerrainWatershedsOperation` | `modules/analysis/src/gpu-terrain/hydrology/gpu-terrain-watersheds-operation.ts` | - | adapter of the local class |
| `GPUTerrainWatersheds` | `modules/analysis/src/gpu-terrain/hydrology/gpu-terrain-watersheds.ts` | - | own kernel, no published equivalent |
| `GPUPointHorizonProfile` | `modules/analysis/src/gpu-terrain/point-horizon/gpu-point-horizon-profile.ts` | - | own kernel, no published equivalent |
| `GPUPointHorizonVisibility` | `modules/analysis/src/gpu-terrain/point-horizon/gpu-point-horizon-visibility.ts` | - | own kernel, no published equivalent |
| `GPULocalDominance` | `modules/analysis/src/gpu-terrain/relief-visualization/gpu-local-dominance.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUMultiScaleRelief` | `modules/analysis/src/gpu-terrain/relief-visualization/gpu-multi-scale-relief.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside `GPURasterBoxBlur`/`Neighborhood` [NEW row; reason in code holds, RT-07] |
| `GPUReliefBlend` | `modules/analysis/src/gpu-terrain/relief-visualization/gpu-relief-blend.ts` | - | own kernel; percent stretch could use published percentile chain (RT-06) |
| `GPUSimpleLocalRelief` | `modules/analysis/src/gpu-terrain/relief-visualization/gpu-simple-local-relief.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside `GPURasterBoxBlur`/`Neighborhood` [NEW row; reason in code holds, RT-07] |
| `GPUTerrainContours` | `modules/analysis/src/gpu-terrain/terrain-analysis/gpu-terrain-contours.ts` | `GPURasterContours`; band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | delegates (`GPURasterContours`) |
| `GPUTerrainCumulativeViewshed` | `modules/analysis/src/gpu-terrain/terrain-analysis/gpu-terrain-cumulative-viewshed.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTerrainDerivatives` | `modules/analysis/src/gpu-terrain/terrain-analysis/gpu-terrain-derivatives.ts` | `GPURasterBufferToTexture`, `GPURasterGradient`; band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | delegates (`GPURasterGradient`); padded border local [registered] |
| `GPUTerrainLineOfSight` | `modules/analysis/src/gpu-terrain/terrain-analysis/gpu-terrain-line-of-sight.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTerrainViewshed` | `modules/analysis/src/gpu-terrain/terrain-analysis/gpu-terrain-viewshed.ts` | `GPURasterBufferToTexture`; band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTerrainCurvature` | `modules/analysis/src/gpu-terrain/terrain-curvature/gpu-terrain-curvature.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside `GPUFiniteDifference2D`/`GPURasterConvolution` [registered] |
| `GPUTerrainRGBDecode` | `modules/analysis/src/gpu-terrain/terrain-decode/gpu-terrain-rgb-decode.ts` | - | own kernel, no published equivalent |
| `GPUTerrainSpikeRepair` | `modules/analysis/src/gpu-terrain/terrain-decode/gpu-terrain-spike-repair.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside graph CC [registered]; `GPURasterConnectedComponents` correctly not used (edge predicate) |
| `GPUProfilePeaksOperation` | `modules/analysis/src/gpu-terrain/terrain-features/gpu-profile-peaks-operation.ts` | - | adapter of the local class |
| `GPUProfilePeaks` | `modules/analysis/src/gpu-terrain/terrain-features/gpu-profile-peaks.ts` | - | own kernel, no published equivalent |
| `GPUTerrainCriticalPoints` | `modules/analysis/src/gpu-terrain/terrain-features/gpu-terrain-critical-points.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTerrainPeakProminenceOperation` | `modules/analysis/src/gpu-terrain/terrain-features/gpu-terrain-peak-prominence-operation.ts` | - | adapter of the local class |
| `GPUTerrainPeakProminence` | `modules/analysis/src/gpu-terrain/terrain-features/gpu-terrain-peak-prominence.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTerrainPeakSnap` | `modules/analysis/src/gpu-terrain/terrain-features/gpu-terrain-peak-snap.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTerrainSummits` | `modules/analysis/src/gpu-terrain/terrain-features/gpu-terrain-summits.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTerrainFlowField` | `modules/analysis/src/gpu-terrain/terrain-flow-field/gpu-terrain-flow-field.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside `GPUFiniteDifference2D` [registered] |
| `GPUReliefShading` | `modules/analysis/src/gpu-terrain/terrain-illumination/gpu-relief-shading.ts` | `GPURasterBufferToTexture`, `GPURasterGradient`; band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | delegates (`GPURasterGradient`); padded border local [registered] |
| `GPUSolarIrradiance` | `modules/analysis/src/gpu-terrain/terrain-illumination/gpu-solar-irradiance.ts` | `GPURasterBufferToTexture` | own kernel, no published equivalent |
| `GPUSolarPositionOperation` | `modules/analysis/src/gpu-terrain/terrain-illumination/gpu-solar-position-operation.ts` | - | adapter of the local class |
| `GPUSolarPosition` | `modules/analysis/src/gpu-terrain/terrain-illumination/gpu-solar-position.ts` | - | own kernel, no published equivalent |
| `GPUSolarShadowMask` | `modules/analysis/src/gpu-terrain/terrain-illumination/gpu-solar-shadow-mask.ts` | `GPURasterBufferToTexture` | own kernel, no published equivalent |
| `GPUTerrainCastShadow` | `modules/analysis/src/gpu-terrain/terrain-illumination/gpu-terrain-cast-shadow.ts` | `GPURasterBufferToTexture`; band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTerrainHorizon` | `modules/analysis/src/gpu-terrain/terrain-illumination/gpu-terrain-horizon.ts` | `GPURasterBufferToTexture`; band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTextureShading` | `modules/analysis/src/gpu-terrain/terrain-illumination/gpu-texture-shading.ts` | `GPURasterBufferToTexture`; band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside `GPURasterGaussianBlur` (and `GPUFFT2D`) [NEW row, RT-07] |
| `GPUTerrainBranchingFlowOperation` | `modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow-operation.ts` | - | adapter of the local class |
| `GPUTerrainBranchingFlow` | `modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow.ts` | - | own kernel, no published equivalent |
| `GPUTerrainBreachOperation` | `modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-breach-operation.ts` | - | adapter of the local class |
| `GPUTerrainBreach` | `modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-breach.ts` | - | own kernel, no published equivalent |
| `GPUTerrainCatchmentSummaryOperation` | `modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary-operation.ts` | - | adapter of the local class |
| `GPUTerrainCatchmentSummary` | `modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary.ts` | - | beside `GPURasterRegionMeasurements` [registered] |
| `GPUTerrainDEMCoregistration` | `modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration.ts` | - | own kernel, no published equivalent |
| `GPUTerrainDepressionHierarchyOperation` | `modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy-operation.ts` | - | adapter of the local class |
| `GPUTerrainDepressionHierarchy` | `modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy.ts` | - | own kernel, no published equivalent |
| `GPUTerrainFlowFractions` | `modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-flow-fractions.ts` | - | own kernel, no published equivalent |
| `GPUTerrainTiledFlow` | `modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts` | - | own kernel, no published equivalent |
| `GPUTerrainRuggedness` | `modules/analysis/src/gpu-terrain/topographic-position/gpu-terrain-ruggedness.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTerrainTopographicPosition` | `modules/analysis/src/gpu-terrain/topographic-position/gpu-terrain-topographic-position.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTerrainVectorRuggedness` | `modules/analysis/src/gpu-terrain/topographic-position/gpu-terrain-vector-ruggedness.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUTerrainWeissLandforms` | `modules/analysis/src/gpu-terrain/topographic-position/gpu-terrain-weiss-landforms.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUChangeDetection` | `modules/analysis/src/raster-analysis/change-detection/gpu-change-detection.ts` | - | own kernel, no published equivalent |
| `GPUCostDistanceOperation` | `modules/analysis/src/raster-analysis/cost-distance/gpu-cost-distance-operation.ts` | - | adapter of the local class |
| `GPUCostDistancePathOperation` | `modules/analysis/src/raster-analysis/cost-distance/gpu-cost-distance-path-operation.ts` | - | adapter of the local class |
| `GPUCostDistancePath` | `modules/analysis/src/raster-analysis/cost-distance/gpu-cost-distance-path.ts` | - | own kernel, no published equivalent |
| `GPUCostDistance` | `modules/analysis/src/raster-analysis/cost-distance/gpu-cost-distance.ts` | `GPUHistogram`; band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent; band counts delegate (`GPUHistogram`) |
| `GPURasterWalkingCostOperation` | `modules/analysis/src/raster-analysis/cost-distance/gpu-raster-walking-cost-operation.ts` | - | adapter of the local class |
| `GPUDistanceField` | `modules/analysis/src/raster-analysis/distance-field/gpu-distance-field.ts` | `GPURasterBufferToTexture` | own kernel, no published equivalent |
| `GPULineIntegralConvolution` | `modules/analysis/src/raster-analysis/flow-texture/gpu-line-integral-convolution.ts` | `GPURasterBufferToTexture` | own kernel, no published equivalent |
| `GPUStreamlinesOperation` | `modules/analysis/src/raster-analysis/flow-texture/gpu-streamlines-operation.ts` | - | adapter of the local class |
| `GPUStreamlines` | `modules/analysis/src/raster-analysis/flow-texture/gpu-streamlines.ts` | - | own kernel, no published equivalent |
| `GPURasterFocalPairs` | `modules/analysis/src/raster-analysis/focal-pairs/gpu-raster-focal-pairs.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPURasterFillNoData` | `modules/analysis/src/raster-analysis/gpu-raster-fill-nodata.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPURasterFractionalCategoricalStatistics` | `modules/analysis/src/raster-analysis/gpu-raster-fractional-categorical-statistics.ts` | - | own kernel, no published equivalent |
| `GPURasterFractionalCoverage` | `modules/analysis/src/raster-analysis/gpu-raster-fractional-coverage.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPURasterFractionalZonalStatistics` | `modules/analysis/src/raster-analysis/gpu-raster-fractional-coverage.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPURasterFractionalZonalDistribution` | `modules/analysis/src/raster-analysis/gpu-raster-fractional-zonal-distribution.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPURasterRankFilter` | `modules/analysis/src/raster-analysis/gpu-raster-rank-filter.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside `GPURasterErosion`/`Dilation` at q=0/1 [registered] |
| `GPURasterReconstruction` | `modules/analysis/src/raster-analysis/gpu-raster-reconstruction.ts` | - | beside `GPURasterDilation` [registered] |
| `GPURasterFillHoles` | `modules/analysis/src/raster-analysis/gpu-raster-reconstruction.ts` | - | beside `GPURasterConnectedComponents` [NEW, RT-01] |
| `GPURasterReprojection` | `modules/analysis/src/raster-analysis/gpu-raster-reprojection.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPURasterSkeleton` | `modules/analysis/src/raster-analysis/gpu-raster-skeleton.ts` | - | own kernel, no published equivalent |
| `GPURasterMedialAxis` | `modules/analysis/src/raster-analysis/gpu-raster-skeleton.ts` | - | own kernel, no published equivalent |
| `GPURasterNDWI` | `modules/analysis/src/raster-analysis/gpu-raster-spectral-indices.ts` | `GPURasterBandMath` | delegates (`GPURasterBandMath`) |
| `GPURasterNBR` | `modules/analysis/src/raster-analysis/gpu-raster-spectral-indices.ts` | `GPURasterBandMath` | delegates (`GPURasterBandMath`) |
| `GPURasterEVI` | `modules/analysis/src/raster-analysis/gpu-raster-spectral-indices.ts` | `GPURasterBandMath` | delegates (`GPURasterBandMath`) |
| `GPURasterSAVI` | `modules/analysis/src/raster-analysis/gpu-raster-spectral-indices.ts` | `GPURasterBandMath` | delegates (`GPURasterBandMath`) |
| `GPURasterTileReprojection` | `modules/analysis/src/raster-analysis/gpu-raster-tile-reprojection.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | mosaic beside `GPURasterTileHaloFill` [NEW, RT-03]; sampling own |
| `GPURasterWalkingCost` | `modules/analysis/src/raster-analysis/gpu-raster-walking-cost.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPURasterWatershedSegmentation` | `modules/analysis/src/raster-analysis/gpu-raster-watershed-segmentation.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent |
| `GPUIsobandRings` | `modules/analysis/src/raster-analysis/isolines/gpu-isoband-rings.ts` | - | downstream of `GPUIsobands` [registered via isobands] |
| `GPUIsobands` | `modules/analysis/src/raster-analysis/isolines/gpu-isobands.ts` | - | beside `GPURasterContours` [registered] |
| `GPUIsolinesOperation` | `modules/analysis/src/raster-analysis/isolines/gpu-isolines-operation.ts` | - | adapter of the local class |
| `GPUIsolines` | `modules/analysis/src/raster-analysis/isolines/gpu-isolines.ts` | - | beside `GPURasterContours` [registered] |
| `GPUFiniteTimeLyapunovExponent` | `modules/analysis/src/raster-analysis/lagrangian-diagnostics/gpu-finite-time-lyapunov-exponent.ts` | - | beside `GPUFiniteDifference2D` [NEW row, low value] |
| `GPULagrangianFlowMap` | `modules/analysis/src/raster-analysis/lagrangian-diagnostics/gpu-lagrangian-flow-map.ts` | - | own kernel, no published equivalent |
| `GPUParticleAdvection` | `modules/analysis/src/raster-analysis/particle-advection/gpu-particle-advection.ts` | - | own kernel, no published equivalent |
| `GPUPolygonRasterization` | `modules/analysis/src/raster-analysis/polygon-rasterization/gpu-polygon-rasterization.ts` | - | own kernel, no published equivalent |
| `GPURasterJoin` | `modules/analysis/src/raster-analysis/polygon-rasterization/gpu-raster-join.ts` | - | own kernel, no published equivalent |
| `GPURasterArithmetic` | `modules/analysis/src/raster-analysis/raster-algebra/gpu-raster-arithmetic.ts` | - | beside `GPURasterBandMath` [registered] |
| `GPURasterCellStatistics` | `modules/analysis/src/raster-analysis/raster-algebra/gpu-raster-cell-statistics.ts` | - | own kernel, no published equivalent |
| `GPURasterConditional` | `modules/analysis/src/raster-analysis/raster-algebra/gpu-raster-conditional.ts` | - | own kernel, no published equivalent |
| `GPURasterReclassify` | `modules/analysis/src/raster-analysis/raster-algebra/gpu-raster-reclassify.ts` | - | own kernel, no published equivalent |
| `GPUWeightedOverlay` | `modules/analysis/src/raster-analysis/raster-algebra/gpu-weighted-overlay.ts` | - | own kernel, no published equivalent |
| `GPURasterEmbeddingChange` | `modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-change.ts` | - | own kernel, no published equivalent |
| `GPURasterEmbeddingCovariance` | `modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-covariance.ts` | - | own kernel, no published equivalent |
| `GPURasterEmbeddingDequantize` | `modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-dequantize.ts` | - | own kernel, no published equivalent |
| `GPURasterEmbeddingKMeans` | `modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-kmeans.ts` | - | own kernel, no published equivalent |
| `GPURasterEmbeddingKNearestClassifier` | `modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier.ts` | - | own kernel, no published equivalent |
| `GPURasterEmbeddingLinearProbe` | `modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-linear-probe.ts` | - | own kernel, no published equivalent |
| `GPURasterEmbeddingNearestPixels` | `modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-nearest-pixels.ts` | - | own kernel, no published equivalent |
| `GPURasterEmbeddingProjection` | `modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-projection.ts` | - | own kernel, no published equivalent |
| `GPURasterEmbeddingPyramidLevel` | `modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-pyramid-level.ts` | - | own kernel; `GPURasterOverview` wrong for unit vectors (reason in doc holds) |
| `GPURasterEmbeddingSimilarity` | `modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-similarity.ts` | - | own kernel, no published equivalent |
| `GPURasterEmbeddingZoneMean` | `modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-zone-mean.ts` | - | own kernel, no published equivalent |
| `GPURasterFillHolesOperation` | `modules/analysis/src/raster-analysis/raster-morphology/gpu-raster-fill-holes-operation.ts` | - | adapter of the local class |
| `GPURasterReconstructionOperation` | `modules/analysis/src/raster-analysis/raster-morphology/gpu-raster-reconstruction-operation.ts` | - | adapter of the local class |
| `GPURasterPatchMetrics` | `modules/analysis/src/raster-analysis/raster-patches/gpu-raster-patch-metrics.ts` | - | complements `GPURasterRegionMeasurements`; counts/areas overlap [NEW note, RT-04] |
| `GPURasterSieve` | `modules/analysis/src/raster-analysis/raster-patches/gpu-raster-sieve.ts` | - | own kernel on published `GPURasterDenseComponents` labels |
| `GPURasterExtremaPyramid` | `modules/analysis/src/raster-analysis/raster-pyramid/gpu-raster-extrema-pyramid.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel; `GPURasterOverview` has no min/max (reason in code holds) |
| `GPURasterProfile` | `modules/analysis/src/raster-analysis/raster-sampling/gpu-raster-profile.ts` | - | own kernel, no published equivalent |
| `GPURasterSampling` | `modules/analysis/src/raster-analysis/raster-sampling/gpu-raster-sampling.ts` | - | own kernel, no published equivalent |
| `GPURasterStretch` | `modules/analysis/src/raster-analysis/raster-stretch/gpu-raster-stretch.ts` | - | beside `GPURasterContrast`/`Statistics`/`GPUHistogram` [registered; RT-08] |
| `GPURasterBestAvailablePixel` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-best-available-pixel.ts` | - | own kernel, no published equivalent |
| `GPURasterBFASTMonitor` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-bfast-monitor.ts` | - | own kernel, no published equivalent |
| `GPURasterClimateAnomaly` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-climate-anomaly.ts` | - | own kernel, no published equivalent |
| `GPURasterClimateRunAccumulator` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-climate-run-accumulator.ts` | - | own kernel, no published equivalent |
| `GPURasterCOLDMonitor` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-cold-monitor.ts` | - | own kernel, no published equivalent |
| `GPURasterEOFProjection` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-eof-projection.ts` | - | own kernel, no published equivalent |
| `GPURasterGeometricMedianCompositeOperation` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite-operation.ts` | - | adapter of the local class |
| `GPURasterGeometricMedianComposite` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite.ts` | - | own kernel, no published equivalent |
| `GPURasterHANTS` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-hants.ts` | - | own kernel, no published equivalent |
| `GPURasterHarmonicRegression` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-harmonic-regression.ts` | - | own kernel, no published equivalent |
| `GPURasterLaggedCorrelation` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-lagged-correlation.ts` | - | own kernel, no published equivalent |
| `GPURasterMedoidCompositeOperation` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-medoid-composite-operation.ts` | - | adapter of the local class |
| `GPURasterMedoidComposite` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-medoid-composite.ts` | - | own kernel, no published equivalent |
| `GPURasterPhenologyMetrics` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-phenology-metrics.ts` | - | own kernel, no published equivalent |
| `GPURasterRollingMean` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-rolling-mean.ts` | - | own kernel, no published equivalent |
| `GPURasterSnapshotCovariance` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-snapshot-covariance.ts` | - | own kernel, no published equivalent |
| `GPURasterStandardizedIndexCalibration` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-standardized-index-calibration.ts` | - | own kernel, no published equivalent |
| `GPURasterStandardizedIndex` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-standardized-index.ts` | - | own kernel, no published equivalent |
| `GPURasterStreamingAccumulator` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator.ts` | - | own kernel, no published equivalent |
| `GPURasterTemporalResample` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-temporal-resample.ts` | - | own kernel, no published equivalent |
| `GPURasterWhittakerSmoother` | `modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-whittaker-smoother.ts` | - | own kernel, no published equivalent |
| `GPURasterWatershedSegmentationOperation` | `modules/analysis/src/raster-analysis/raster-watershed/gpu-raster-watershed-segmentation-operation.ts` | - | adapter of the local class |
| `GPURasterFractionalZonalDistributionOperation` | `modules/analysis/src/raster-analysis/raster-zonal-statistics/gpu-raster-fractional-zonal-distribution-operation.ts` | - | adapter of the local class |
| `GPURasterFractionalZonalStatisticsOperation` | `modules/analysis/src/raster-analysis/raster-zonal-statistics/gpu-raster-fractional-zonal-statistics-operation.ts` | - | adapter of the local class |
| `GPURasterZonalStatistics` | `modules/analysis/src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside `GPURasterRegionMeasurements` [NEW, RT-04] |
| `GPURasterConservativeRemapping` | `modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-conservative-remapping.ts` | - | own kernel, no published equivalent |
| `GPURasterRegionGraphOperation` | `modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-region-graph-operation.ts` | - | adapter of the local class |
| `GPURasterRegionGraph` | `modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-region-graph.ts` | - | own kernel, no published equivalent |
| `GPURasterRegistration` | `modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-registration.ts` | `GPUFFT2D` | delegates the transforms (`GPUFFT2D`); limits copied by hand (RT-05) |
| `GPURasterSeasonalBreak` | `modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-seasonal-break.ts` | - | own kernel, no published equivalent |
| `GPURasterSpectralUnmixing` | `modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-spectral-unmixing.ts` | - | own kernel, no published equivalent |
| `GPURasterTemporal` | `modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-temporal.ts` | - | own kernel, no published equivalent |
| `GPURasterTexture` | `modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-texture.ts` | - | own kernel, no published equivalent |
