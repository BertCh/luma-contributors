// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// ES module consumer of the packed declarations. Each `@ts-expect-error` fails as TS2578 when a
// declaration silently degrades to `any`, for example through an unresolvable relative import type.
import {computeVariogramModelFitOnCPU} from '@luma-contributors/analysis/dataframe-analysis';
import type {VariogramModelFitDiagnostics} from '@luma-contributors/analysis/dataframe-analysis';
import type {
  PublishNodeProps,
  WGSLDispatchGate,
  RasterGridAffine as CoreRasterGridAffine
} from '@luma-contributors/analysis/analysis-core';
import type {
  CPUTerrainCatchmentSummaryOperationProps,
  CPURasterOperationBand as TerrainRasterOperationBand
} from '@luma-contributors/analysis/gpu-terrain';
import type {GPUNetworkIsochronesOutput} from '@luma-contributors/analysis/gpu-network';
import type {
  GPUSpatialScanStatisticOutput,
  CPUPolygonRows,
  GPUMapColoringPriority,
  RasterGridAffine as SpatialRasterGridAffine
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  GPURasterPatchLabelsStatus,
  CPUCompositeDimensions,
  CPURasterOperationBand
} from '@luma-contributors/analysis/raster-analysis';
import {
  GPU_LINE_DENSITY_PARAMETER_SCHEMA,
  GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA
} from '@luma-contributors/analysis/gpu-spatial-analysis';

const schemaWordLength: number = GPU_LINE_DENSITY_PARAMETER_SCHEMA.wordLength;
void schemaWordLength;
// @ts-expect-error An invented field must remain a type error in a native consumer.
GPU_LINE_DENSITY_PARAMETER_SCHEMA.notARealField;

// The emitted type of this schema once named an extensionless path that NodeNext cannot resolve.
const segmentizeFormat: 'float32' = GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA.format;
void segmentizeFormat;
// @ts-expect-error An invented field must remain a type error in a native consumer.
GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA.notARealField;

const fit = computeVariogramModelFitOnCPU({distances: [1], semivariances: [1], pairCounts: [1]});
const fitStatus: Uint32Array = fit.status;
void fitStatus;
// @ts-expect-error An invented result field must remain a type error in a native consumer.
fit.notARealField;

// A missing export or a signature degraded to any must fail in the installed consumer.
function requirePublicType<Type>(_value: 0 extends 1 & Type ? never : true): void {}
requirePublicType<CPUTerrainCatchmentSummaryOperationProps>(true);
requirePublicType<GPUNetworkIsochronesOutput>(true);
requirePublicType<GPUSpatialScanStatisticOutput>(true);
requirePublicType<GPURasterPatchLabelsStatus>(true);
requirePublicType<CPUPolygonRows>(true);
requirePublicType<GPUMapColoringPriority>(true);
requirePublicType<CPUCompositeDimensions>(true);
requirePublicType<VariogramModelFitDiagnostics>(true);
requirePublicType<PublishNodeProps>(true);
requirePublicType<WGSLDispatchGate<unknown>>(true);
requirePublicType<CPURasterOperationBand>(true);
requirePublicType<TerrainRasterOperationBand>(true);
requirePublicType<CoreRasterGridAffine>(true);
requirePublicType<SpatialRasterGridAffine>(true);
type CompatibleTypes<Left, Right> = [Left] extends [Right]
  ? [Right] extends [Left] ? true : false
  : false;
const compatibleAliases: [
  CompatibleTypes<CPURasterOperationBand, TerrainRasterOperationBand>,
  CompatibleTypes<CoreRasterGridAffine, SpatialRasterGridAffine>
] = [true, true];
void compatibleAliases;

// @ts-expect-error Deep source folders are not a public package entry.
import type * as privateInterpolation from '@luma-contributors/analysis/gpu-spatial-analysis/spatial-interpolation';
// @ts-expect-error Analysis testing helpers are intentionally absent from the installed package.
import type * as privateTesting from '@luma-contributors/analysis/testing';
// @ts-expect-error Workspace specs are not a public package entry.
import type * as privateTestHelper from '@luma-contributors/analysis/test/utils/gpu-contributor-test-utils';
void (null as unknown as [typeof privateInterpolation, typeof privateTesting, typeof privateTestHelper]);
