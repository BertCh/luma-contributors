// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CommonJS consumer of the packed declarations, resolved through the `require` condition. Each
// `@ts-expect-error` fails as TS2578 when a declaration silently degrades to `any`.
import spatialAnalysis = require('@luma-contributors/analysis/gpu-spatial-analysis');
import dataframeAnalysis = require('@luma-contributors/analysis/dataframe-analysis');
import analysisCore = require('@luma-contributors/analysis/analysis-core');
import terrainAnalysis = require('@luma-contributors/analysis/gpu-terrain');
import networkAnalysis = require('@luma-contributors/analysis/gpu-network');
import rasterAnalysis = require('@luma-contributors/analysis/raster-analysis');

const segmentizeSchema = spatialAnalysis.GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA;
const segmentizeFormat: 'float32' = segmentizeSchema.format;
void segmentizeFormat;
// @ts-expect-error An invented field must remain a type error in a CommonJS consumer.
segmentizeSchema.notARealField;

const densityWordLength: number = spatialAnalysis.GPU_LINE_DENSITY_PARAMETER_SCHEMA.wordLength;
void densityWordLength;

const fit = dataframeAnalysis.computeVariogramModelFitOnCPU({
  distances: [1],
  semivariances: [1],
  pairCounts: [1]
});
const fitParameters: Float32Array = fit.parameters;
void fitParameters;
// @ts-expect-error An invented result field must remain a type error in a CommonJS consumer.
fit.notARealField;

function requirePublicType<Type>(_value: 0 extends 1 & Type ? never : true): void {}
requirePublicType<terrainAnalysis.CPUTerrainCatchmentSummaryOperationProps>(true);
requirePublicType<networkAnalysis.GPUNetworkIsochronesOutput>(true);
requirePublicType<spatialAnalysis.GPUSpatialScanStatisticOutput>(true);
requirePublicType<rasterAnalysis.GPURasterPatchLabelsStatus>(true);
requirePublicType<spatialAnalysis.CPUPolygonRows>(true);
requirePublicType<spatialAnalysis.GPUMapColoringPriority>(true);
requirePublicType<rasterAnalysis.CPUCompositeDimensions>(true);
requirePublicType<dataframeAnalysis.VariogramModelFitDiagnostics>(true);
requirePublicType<analysisCore.PublishNodeProps>(true);
requirePublicType<analysisCore.WGSLDispatchGate<unknown>>(true);
requirePublicType<rasterAnalysis.CPURasterOperationBand>(true);
requirePublicType<terrainAnalysis.CPURasterOperationBand>(true);
requirePublicType<analysisCore.RasterGridAffine>(true);
requirePublicType<spatialAnalysis.RasterGridAffine>(true);
type CompatibleTypes<Left, Right> = [Left] extends [Right]
  ? [Right] extends [Left] ? true : false
  : false;
const compatibleAliases: [
  CompatibleTypes<rasterAnalysis.CPURasterOperationBand, terrainAnalysis.CPURasterOperationBand>,
  CompatibleTypes<analysisCore.RasterGridAffine, spatialAnalysis.RasterGridAffine>
] = [true, true];
void compatibleAliases;

// @ts-expect-error Deep source folders are not a public package entry.
import privateInterpolation = require('@luma-contributors/analysis/gpu-spatial-analysis/spatial-interpolation');
// @ts-expect-error Analysis testing helpers are intentionally absent from the installed package.
import privateTesting = require('@luma-contributors/analysis/testing');
// @ts-expect-error Workspace specs are not a public package entry.
import privateTestHelper = require('@luma-contributors/analysis/test/utils/gpu-contributor-test-utils');
void (null as unknown as [typeof privateInterpolation, typeof privateTesting, typeof privateTestHelper]);
