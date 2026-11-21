// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, expectTypeOf, it, vi} from 'vitest';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {runInNewContext} from 'node:vm';
import * as typescript from 'typescript';
import {
  createTransientView,
  GPUCommandGraph,
  GPUGather,
  GPUTranspose,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import * as spatial from '@luma-contributors/analysis/gpu-spatial-analysis';
import * as regression from '../../src/gpu-spatial-analysis/spatial-regression/index';
import * as statusDecoders from '../../src/gpu-spatial-analysis/spatial-regression/regression-status-decoders';
import * as gwrReaders from '../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-model-result';
import * as panelReaders from '../../src/gpu-spatial-analysis/spatial-regression/spatial-panel-error-maximum-likelihood-model-result';
import {validatePermutationInputs} from '../../src/gpu-spatial-analysis/permutation-inference/permutation-inference-kernels';
import * as permutationParameters from '../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';
import {createNullWebGPUDevice} from '../../src/testing/gpu-contributor-test-utils';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import type {
  CPUPolygonRows,
  GPUMapColoringPriority,
  RasterGridAffine,
  GPUSpatialScanStatisticOutput,
  CPUMultipleTestingAdjustmentResult,
  CPUSpatialLagImpactsResult,
  CPUSpatialWeightsValidationResult,
  GPUAnalysisSpacingStatisticsDecodeResult,
  GPUGeographicallyTemporallyWeightedRegressionReadback,
  GPUGeographicallyTemporallyWeightedRegressionModelResult,
  GPUSpatialPanelErrorMaximumLikelihoodModelResult,
  GPUSpatialPanelErrorMaximumLikelihoodModelResultOptions,
  OrdinaryLeastSquaresJoinedTable,
  GPUVecchiaLogLikelihoodReadback,
  GPUVecchiaFisherScoringReadback
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  GPUGeographicallyTemporallyWeightedRegressionReadback as FamilyGTWRReadback,
  GPUGeographicallyTemporallyWeightedRegressionModelResult as FamilyGTWRResult,
  GPUSpatialPanelErrorMaximumLikelihoodModelResult as FamilyPanelResult,
  GPUSpatialPanelErrorMaximumLikelihoodModelResultOptions as FamilyPanelOptions
} from '../../src/gpu-spatial-analysis/spatial-regression/index';
import type {
  CPURasterOperationBand,
  CPUCompositeDimensions
} from '@luma-contributors/analysis/raster-analysis';
import type {CPURasterOperationBand as TerrainBand} from '@luma-contributors/analysis/gpu-terrain';
import type {VariogramModelFitDiagnostics} from '@luma-contributors/analysis/dataframe-analysis';
import type {
  RasterGridAffine as CoreAffine,
  PublishNodeProps,
  WGSLDispatchGate
} from '@luma-contributors/analysis/analysis-core';

it('CMP-01 publishes the joined-table mask with distinct completeness and group-loss views', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const joinedTable: OrdinaryLeastSquaresJoinedTable = {
      matched: createTransientView(graph, 'matched', 'uint32', 5),
      status: {
        complete: createTransientView(graph, 'clean', 'uint32', 1),
        candidateOverflow: createTransientView(graph, 'loss', 'uint32', 1)
      }
    };
    const result = spatial.createOrdinaryLeastSquaresJoinedTableMask(graph, joinedTable);
    expect(result.mask.length).toBe(5);
    expect(result.status.complete).toBe(joinedTable.status.complete);
    expect(result.status.candidateOverflow).toBe(joinedTable.status.candidateOverflow);
    expect(result.status.complete).not.toBe(result.status.candidateOverflow);
    const record = getWGSLKernelNodeRecord(
      graph,
      'ordinary-least-squares-joined-table-training-mask'
    )!;
    expect(record.bindings.find(binding => binding.name === 'complete')?.view).toBe(
      joinedTable.status.complete
    );
    expect(record.bindings.find(binding => binding.name === 'candidateOverflow')?.view).toBe(
      joinedTable.status.candidateOverflow
    );
    expect(record.body).toContain('complete[completeOffset] != 0u');
    expect(record.body).toContain('candidateOverflow[candidateOverflowOffset] == 0u');
  } finally {
    device.destroy();
  }
});

it('W2-03 preserves every named status decoder and reader identity at both barrel levels', () => {
  const requiredStatusDecoders = [
    'decodeGPUGeographicallyWeightedRegressionStatusFields',
    'decodeGPUGeographicallyTemporallyWeightedRegressionStatusFields',
    'decodeGPUMultiscaleGeographicallyTemporallyWeightedRegressionStatusFields',
    'decodeGPUGeneralizedGeographicallyWeightedRegressionStatusFields',
    'decodeGPUMultiscaleGeographicallyWeightedRegressionStatusFields',
    'decodeGPUOrdinaryLeastSquaresStatusFields',
    'decodeGPUSpatialLagMaximumLikelihoodStatusFields',
    'decodeGPUSpatialErrorMaximumLikelihoodStatusFields',
    'decodeGPUSpatialErrorGMStatusFields',
    'decodeGPUSpatialTwoStageLeastSquaresStatusFields',
    'decodeGPUSpatialRegressionDiagnosticsStatusFields',
    'decodeGPUInstrumentalVariablesCovarianceStatusFields',
    'decodeGPUSeeminglyUnrelatedRegressionStatusFields',
    'decodeGPUSpatialSeeminglyUnrelatedRegressionStatusFields',
    'decodeGPUProbitRegressionStatusFields',
    'decodeGPUSpatialPanelRegressionStatusFields',
    'decodeGPUSpatialPanelErrorMaximumLikelihoodStatusFields',
    'decodeGPUSpatialPanelLagMaximumLikelihoodStatusFields',
    'decodeGPUSpatialPanelErrorDiagnosticsStatusFields',
    'decodeGPUSpatialPanelHausmanStatusFields',
    'decodeGPUProbitSpatialDiagnosticsStatusFields',
    'decodeGPURestrictedRegimeRegressionStatusFields',
    'decodeGPUSpatialRegimeRegressionStatusFields',
    'decodeGPUMultiscaleGeographicallyWeightedRegressionInferenceStatusFields',
    'decodeGPUProbitSpatialLagStatusFields'
  ] as const;
  for (const name of requiredStatusDecoders) {
    expect(typeof regression[name], name).toBe('function');
    expect(regression[name], name).toBe(statusDecoders[name]);
    expect(spatial[name], name).toBe(statusDecoders[name]);
  }
  const requiredReaders = {
    getGPUGeographicallyWeightedRegressionModelResult:
      gwrReaders.getGPUGeographicallyWeightedRegressionModelResult,
    getGPUGeographicallyTemporallyWeightedRegressionModelResult:
      gwrReaders.getGPUGeographicallyTemporallyWeightedRegressionModelResult,
    getGPUSpatialPanelErrorMaximumLikelihoodModelResult:
      panelReaders.getGPUSpatialPanelErrorMaximumLikelihoodModelResult
  };
  for (const name of Object.keys(requiredReaders) as (keyof typeof requiredReaders)[]) {
    expect(typeof regression[name], name).toBe('function');
    expect(regression[name], name).toBe(requiredReaders[name]);
    expect(spatial[name], name).toBe(requiredReaders[name]);
  }
  expectTypeOf<spatial.GPUDecodedRegressionStatus>().toEqualTypeOf<regression.GPUDecodedRegressionStatus>();
  expectTypeOf<spatial.GPUDecodedRegressionStatus>().toEqualTypeOf<statusDecoders.GPUDecodedRegressionStatus>();
  expectTypeOf<spatial.GPUSpatialMaximumLikelihoodReadbackInput>().toEqualTypeOf<regression.GPUSpatialMaximumLikelihoodReadbackInput>();
  expectTypeOf<spatial.GPUSpatialMaximumLikelihoodReadbackInput>().toEqualTypeOf<
    import('../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-model-result').GPUSpatialMaximumLikelihoodReadbackInput
  >();
  expectTypeOf<regression.GPUGeographicallyWeightedRegressionReadback>().toEqualTypeOf<gwrReaders.GPUGeographicallyWeightedRegressionReadback>();
  expectTypeOf<regression.GPUGeographicallyWeightedRegressionModelResult>().toEqualTypeOf<gwrReaders.GPUGeographicallyWeightedRegressionModelResult>();
  expectTypeOf<FamilyGTWRReadback>().toEqualTypeOf<GPUGeographicallyTemporallyWeightedRegressionReadback>();
  expectTypeOf<FamilyGTWRResult>().toEqualTypeOf<GPUGeographicallyTemporallyWeightedRegressionModelResult>();
  expectTypeOf<FamilyPanelResult>().toEqualTypeOf<GPUSpatialPanelErrorMaximumLikelihoodModelResult>();
  expectTypeOf<FamilyPanelOptions>().toEqualTypeOf<GPUSpatialPanelErrorMaximumLikelihoodModelResultOptions>();
});

it('STA-06 publishes detailed evidence when the convenience spacing result is unavailable', () => {
  const words = new Uint32Array(spatial.GPU_ANALYSIS_SPACING_SUMMARY_LENGTH);
  words[7] = 1;
  words[1] = 3;
  const result: GPUAnalysisSpacingStatisticsDecodeResult =
    spatial.decodeGPUAnalysisSpacingStatisticsDetailed(words);
  expect(result).toMatchObject({pointCount: 0, invalidCount: 3, unavailableReason: 'all-invalid'});
  expect(spatial.decodeGPUAnalysisSpacingStatistics(words)).toBeUndefined();
  words[1] = 0;
  expect(spatial.decodeGPUAnalysisSpacingStatisticsDetailed(words).unavailableReason).toBe('empty');
});

it('CMP-11 accepts readonly predictor columns and retains their coefficient order', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const first = createTransientView(graph, 'x', 'float32', 5);
    const second = createTransientView(graph, 'z', 'float32', 5);
    expect(spatial.createOrdinaryLeastSquaresDesign(graph, [first])).toBe(first);
    const add = vi.spyOn(graph, 'add');
    const columns: readonly GraphDataView<'float32'>[] = [first, second];
    const predictors = spatial.createOrdinaryLeastSquaresDesign(graph, columns);
    expect(predictors.length).toBe(10);
    // Two length-five column gathers followed by a 2x5 transpose produce row-major [x,z].
    const producers = add.mock.calls.map(([producer]) => producer);
    const gathers = producers.filter(
      (producer): producer is GPUGather<'float32'> => producer instanceof GPUGather
    );
    const transpose = producers.find(
      (producer): producer is GPUTranspose<'float32'> => producer instanceof GPUTranspose
    )!;
    expect(gathers).toHaveLength(2);
    expect(gathers[0].source).toBe(first);
    expect(gathers[1].source).toBe(second);
    expect((gathers[0].output as GraphDataView<'float32'>).byteOffset).toBe(0);
    expect((gathers[1].output as GraphDataView<'float32'>).byteOffset).toBe(20);
    expect(transpose.rows).toBe(2);
    expect(transpose.columns).toBe(5);
    expect(transpose.output).toBe(predictors);
    add.mockRestore();
  } finally {
    device.destroy();
  }
});

it('SCL-12 distinguishes automatic crossovers from forced-route maxima and preserves aliases', () => {
  expect(spatial.GPU_ISLAND_ATTACHMENT_AUTO_SCAN_MAXIMUM_ROWS).toBe(4096);
  expect(spatial.GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROWS).toBe(65535);
  expect(spatial.GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS).toBe(4096);
  expect(spatial.GPU_SPATIAL_WEIGHTS_TRACES_EXACT_MAXIMUM_ROWS).toBe(16384);
});

it('UST-14 exports the actual shared permutation validation ceiling', () => {
  expect(spatial.MAXIMUM_PERMUTATION_COUNT).toBe(1048576);
  expect(spatial.MAXIMUM_PERMUTATION_COUNT).toBe(permutationParameters.MAXIMUM_PERMUTATION_COUNT);
});

it('UST-04 publishes both Philox seed words through the supported spatial entry', () => {
  // 2^32 + 7 has low word 7 and high word 1, rather than two equal words.
  expect(spatial.getPermutationSeedKey(2 ** 32 + 7)).toEqual([7, 1]);
  expect(spatial.getPermutationSeedKey).toBe(permutationParameters.getPermutationSeedKey);
});

it('UCO-16 decodes combined conformal bits from an ordinary package import', () => {
  expect(spatial.decodeConformalStatus(17)).toEqual({
    lowerUnbounded: true,
    upperUnbounded: false,
    invalidQuery: false,
    noCalibration: false,
    neighborsTruncated: true,
    inputIncomplete: false
  });
});

it('ENT-05 exports the documented local-kriging brute-force limits', () => {
  expect(spatial.GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT).toBe(4096);
  expect(spatial.GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK).toBe(4294967296);
});

it('ENT-08 and companion contracts provide consumer-importable public types', () => {
  expectTypeOf<CPUPolygonRows>().toEqualTypeOf<
    Parameters<typeof spatial.computeLabelPointOnCPU>[0]
  >();
  expectTypeOf<GPUMapColoringPriority>().toEqualTypeOf<
    import('../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring').GPUMapColoringPriority
  >();
  expectTypeOf<RasterGridAffine>().toEqualTypeOf<CoreAffine>();
  expectTypeOf<CPURasterOperationBand>().toEqualTypeOf<TerrainBand>();
  expectTypeOf<CPUCompositeDimensions>().toEqualTypeOf<
    import('../../src/raster-analysis/raster-time-series/cpu-composite-observations').CPUCompositeDimensions
  >();
  expectTypeOf<VariogramModelFitDiagnostics>().toEqualTypeOf<
    import('../../src/dataframe-analysis/pair-statistics/variogram-model').VariogramModelFitDiagnostics
  >();
  expectTypeOf<PublishNodeProps>().toEqualTypeOf<
    import('../../src/analysis-core/kernels/wgsl-kernel-nodes').PublishNodeProps
  >();
  expectTypeOf<WGSLDispatchGate<{frameNumber: number}>>().toEqualTypeOf<
    import('../../src/analysis-core/kernels/wgsl-kernel-nodes').WGSLDispatchGate<{
      frameNumber: number;
    }>
  >();
  expectTypeOf<GPUSpatialScanStatisticOutput>().toEqualTypeOf<
    import('../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic').GPUSpatialScanStatisticOutput
  >();
  expectTypeOf<CPUMultipleTestingAdjustmentResult>().toEqualTypeOf<
    ReturnType<typeof spatial.computeMultipleTestingAdjustmentOnCPU>
  >();
  expectTypeOf<CPUSpatialLagImpactsResult>().toEqualTypeOf<
    ReturnType<typeof spatial.computeSpatialLagImpactsOnCPU>
  >();
  expectTypeOf<CPUSpatialWeightsValidationResult>().toEqualTypeOf<
    ReturnType<typeof spatial.computeSpatialWeightsValidationOnCPU>
  >();
});

it('EGR-01 publishes offset-aware Vecchia port readers and the contiguity reader', () => {
  const components = new Float32Array([99, -4, 0.25, 99]);
  const bytes = new Uint8Array(components.buffer, 4, 8);
  const status = new Uint32Array([99, 0, 0, 1, 99]);
  const statusBytes = new Uint8Array(status.buffer, 4, 12);
  const likelihoodPorts: GPUVecchiaLogLikelihoodReadback = {
    logLikelihood: bytes,
    status: statusBytes
  };
  const result = spatial.decodeGPUVecchiaLogLikelihoodResult(likelihoodPorts);
  expect(result.logLikelihood).toBe(-3.75);
  expect(result.usedSiteCount).toBe(1);
  const totals = new Float32Array(20);
  totals.set([-4, 0.25]);
  const fisherPorts: GPUVecchiaFisherScoringReadback = {totals, status: statusBytes};
  expect(spatial.decodeGPUVecchiaFisherScoringResult(fisherPorts).logLikelihood).toBe(-3.75);
  expect(spatial.decodeGPUVecchiaLogLikelihoodStatus(statusBytes).complete).toBe(true);
  expect(spatial.decodeGPUVecchiaFisherScoringStatus(statusBytes).complete).toBe(true);
  expect(typeof spatial.decodeGPUAnalysisContiguityStatistics).toBe('function');
});

it('SCL-12 constructors accept forced scan and exact routes beyond automatic crossovers', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const rowCount of [4096, 4097, 16384, 16385, 65535, 65536]) {
      const graph = new GPUCommandGraph(device);
      const uint32 = (name: string, length: number) =>
        createTransientView(graph, name, 'uint32', length);
      const float32 = (name: string, length: number) =>
        createTransientView(graph, name, 'float32', length);
      const weights = {
        offsets: uint32('offsets', rowCount + 1),
        neighbors: uint32('neighbors', 1),
        weights: float32('weights', 1)
      };
      const islandProps: spatial.GPUIslandAttachmentProps = {
        weights,
        positions: createTransientView(graph, 'positions', 'float32x2', rowCount),
        nearestSearch: 'scan',
        output: {
          offsets: uint32('output-offsets', rowCount + 1),
          neighbors: uint32('output-neighbors', 1),
          weights: float32('output-weights', 1),
          attachment: uint32('attachment', rowCount),
          added: uint32('added', 1),
          status: {overflow: uint32('overflow', 1)}
        }
      };
      if (rowCount === 4096 || rowCount === 4097) {
        expect(
          spatial.getGPUIslandAttachmentExecutionPlan({...islandProps, nearestSearch: undefined})
            .route
        ).toBe(rowCount === 4096 ? 'scan' : 'grid');
      }
      if (rowCount <= spatial.GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROWS) {
        expect(() => new spatial.GPUIslandAttachment(islandProps)).not.toThrow();
      } else {
        expect(() => new spatial.GPUIslandAttachment(islandProps)).toThrow(/65535/);
      }
      if (rowCount <= 16385) {
        const traceProps: spatial.GPUSpatialWeightsTracesProps = {
          weights,
          maximumPower: 1,
          output: {powerTraces: float32('traces', 1), powerSums: float32('sums', 1)}
        };
        expect(new spatial.GPUSpatialWeightsTraces(traceProps).mode).toBe(
          rowCount <= 4096 ? 'exact' : 'stochastic'
        );
        if (rowCount <= spatial.GPU_SPATIAL_WEIGHTS_TRACES_EXACT_MAXIMUM_ROWS) {
          expect(new spatial.GPUSpatialWeightsTraces({...traceProps, mode: 'exact'}).mode).toBe(
            'exact'
          );
        } else {
          expect(() => new spatial.GPUSpatialWeightsTraces({...traceProps, mode: 'exact'})).toThrow(
            /16384/
          );
        }
      }
    }
  } finally {
    device.destroy();
  }
});

it('UST-14 validates permutation counts against the exported declaration', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      id: 'permutation-entry-contract',
      operation: 'PermutationEntryContract',
      weights: {
        offsets: createTransientView(graph, 'offsets', 'uint32', 3),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
        weights: createTransientView(graph, 'weights', 'float32', 2)
      },
      values: createTransientView(graph, 'values', 'float32', 2),
      parameters: createTransientView(
        graph,
        'parameters',
        'uint32',
        spatial.GPU_PERMUTATION_PARAMETER_LENGTH
      ),
      maximumPermutations: spatial.MAXIMUM_PERMUTATION_COUNT
    };
    expect(validatePermutationInputs(props)).toBe(2);
    expect(() =>
      validatePermutationInputs({
        ...props,
        maximumPermutations: spatial.MAXIMUM_PERMUTATION_COUNT + 1
      })
    ).toThrow(/maximumPermutations.*1048576/);
  } finally {
    device.destroy();
  }
});

it('B1 keeps Vecchia readback declaration references resolvable under NodeNext', () => {
  const source = typescript.createSourceFile(
    '/fixture/index.d.mts',
    readFileSync(
      new URL('../../src/gpu-spatial-analysis/spatial-interpolation/index.ts', import.meta.url),
      'utf8'
    ),
    typescript.ScriptTarget.Latest,
    true
  );
  const require = createRequire(import.meta.url);
  const appendExtension = require('@vis.gl/ts-plugins/ts-transform-append-extension').default as (
    program: undefined,
    configuration: {extensions: string[]},
    extras: {ts: typeof typescript}
  ) => typescript.TransformerFactory<typescript.SourceFile>;
  const transformed = typescript.transform(source, [
    appendExtension(
      undefined,
      {extensions: ['.js', '.lens.js', '.example.js', '.run.js', '.scenarios.js']},
      {ts: typescript}
    )
  ]);
  try {
    const references: string[] = [];
    const visit = (node: typescript.Node): void => {
      if (
        (typescript.isImportDeclaration(node) || typescript.isExportDeclaration(node)) &&
        node.moduleSpecifier &&
        typescript.isStringLiteral(node.moduleSpecifier)
      )
        references.push(node.moduleSpecifier.text);
      if (
        typescript.isImportTypeNode(node) &&
        typescript.isLiteralTypeNode(node.argument) &&
        typescript.isStringLiteral(node.argument.literal)
      )
        references.push(node.argument.literal.text);
      typescript.forEachChild(node, visit);
    };
    visit(transformed.transformed[0]);
    const vecchiaReferences = references.filter(
      reference => reference.startsWith('./gpu-vecchia-') && !reference.includes('operation')
    );
    expect(vecchiaReferences.length).toBeGreaterThanOrEqual(3);
    for (const reference of vecchiaReferences) {
      const declaredPath = `/fixture/${reference.slice(2).replace(/\.js$/, '')}.d.ts`;
      const host: typescript.ModuleResolutionHost = {
        fileExists: path => path === declaredPath || path === '/fixture/package.json',
        readFile: path =>
          path === '/fixture/package.json'
            ? '{"type":"module"}'
            : path === declaredPath
              ? 'export {};'
              : undefined
      };
      const result = typescript.resolveModuleName(
        reference,
        '/fixture/index.d.mts',
        {
          module: typescript.ModuleKind.NodeNext,
          moduleResolution: typescript.ModuleResolutionKind.NodeNext
        },
        host,
        undefined,
        undefined,
        typescript.ModuleKind.ESNext
      );
      expect(result.resolvedModule?.resolvedFileName, reference).toBe(declaredPath);
    }
  } finally {
    transformed.dispose();
  }
});

it('B3 publishes the documented space-time Vecchia readers, uploader and limits', () => {
  expect(typeof spatial.decodeGPUVecchiaSpaceTimeLogLikelihoodResult).toBe('function');
  expect(typeof spatial.decodeGPUVecchiaSpaceTimeLogLikelihoodStatus).toBe('function');
  expect(typeof spatial.uploadVecchiaNeighborhoods).toBe('function');
  expect(spatial.MAXIMUM_VECCHIA_NEIGHBOR_COUNT).toBe(32);
  expect(spatial.MAXIMUM_VECCHIA_ENTRY_COUNT).toBe(2147483647);
  const sets: spatial.PreparedVecchiaConditioningSets = {
    offsets: new Uint32Array([0, 0, 1]),
    neighbors: new Uint32Array([0]),
    counts: {rowCount: 2, neighborCount: 7, entryCount: 1},
    formats: {offsets: 'uint32', neighbors: 'uint32'}
  };
  const upload = vi.fn((values: Uint32Array, format: 'uint32') => ({values, format}));
  const uploaded = spatial.uploadVecchiaNeighborhoods(sets, upload);
  expect(upload.mock.calls).toEqual([
    [sets.offsets, 'uint32'],
    [sets.neighbors, 'uint32']
  ]);
  expect(uploaded).toEqual({
    neighborhoodOffsets: {values: sets.offsets, format: 'uint32'},
    neighborhoods: {values: sets.neighbors, format: 'uint32'},
    neighborCount: 7
  });
  const totals = new Float32Array([99, -4, 0.25, 99]);
  const status = new Uint32Array([99, 0, 0, 1, 99]);
  const ports = {
    logLikelihood: new Uint8Array(totals.buffer, 4, 8),
    status: new Uint8Array(status.buffer, 4, 12)
  };
  expect(spatial.decodeGPUVecchiaSpaceTimeLogLikelihoodResult(ports)).toMatchObject({
    logLikelihood: -3.75,
    usedSiteCount: 1
  });
  expect(spatial.decodeGPUVecchiaSpaceTimeLogLikelihoodStatus(ports.status).complete).toBe(true);
  expect(() =>
    spatial.decodeGPUVecchiaSpaceTimeLogLikelihoodResult({
      ...ports,
      logLikelihood: new Uint8Array(4)
    })
  ).toThrow();
  expect(() => spatial.decodeGPUVecchiaSpaceTimeLogLikelihoodStatus(new Uint8Array(8))).toThrow();
  expectTypeOf<spatial.GPUVecchiaSpaceTimeLogLikelihoodOutput>().toEqualTypeOf<
    import('../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood').GPUVecchiaSpaceTimeLogLikelihoodOutput
  >();
  expectTypeOf<spatial.PreparedCPUVecchiaLogLikelihoodResult>().toEqualTypeOf<
    import('../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-log-likelihood-operation').PreparedCPUVecchiaLogLikelihoodResult
  >();
  expectTypeOf<spatial.PreparedCPUVecchiaSpaceTimeLogLikelihoodResult>().toEqualTypeOf<
    import('../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-space-time-log-likelihood-operation').PreparedCPUVecchiaSpaceTimeLogLikelihoodResult
  >();
  expectTypeOf<spatial.PreparedCPUVecchiaFisherScoringResult>().toEqualTypeOf<
    import('../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-fisher-scoring-operation').PreparedCPUVecchiaFisherScoringResult
  >();
});

it('S3 exposes named OLS constructor-input and model-result contracts', () => {
  const source = typescript.createSourceFile(
    'regression-index.ts',
    readFileSync(
      new URL('../../src/gpu-spatial-analysis/spatial-regression/index.ts', import.meta.url),
      'utf8'
    ),
    typescript.ScriptTarget.Latest,
    true
  );
  const exportedTypes = source.statements.flatMap(statement =>
    typescript.isExportDeclaration(statement) &&
    statement.isTypeOnly &&
    statement.exportClause &&
    typescript.isNamedExports(statement.exportClause)
      ? statement.exportClause.elements.map(element => element.name.text)
      : []
  );
  expect(exportedTypes).toContain('GPUOrdinaryLeastSquaresInputProps');
  expect(exportedTypes).toContain('GPUOrdinaryLeastSquaresModelResult');
  expectTypeOf<spatial.GPUOrdinaryLeastSquaresInputProps>().toEqualTypeOf<regression.GPUOrdinaryLeastSquaresInputProps>();
  expectTypeOf<spatial.GPUOrdinaryLeastSquaresInputProps>().toEqualTypeOf<
    ConstructorParameters<typeof spatial.GPUOrdinaryLeastSquares>[0]
  >();
  expectTypeOf<spatial.GPUOrdinaryLeastSquaresModelResult>().toEqualTypeOf<regression.GPUOrdinaryLeastSquaresModelResult>();
  expectTypeOf<spatial.GPUOrdinaryLeastSquaresModelResult>().toEqualTypeOf<
    ReturnType<typeof spatial.getGPUOrdinaryLeastSquaresModelResult>
  >();
});

/** Check erased public names without running a project compiler in the implementer sandbox. */
function expectFamilyTypeExports(family: string, names: readonly string[]): void {
  const source = typescript.createSourceFile(
    `${family}-index.ts`,
    readFileSync(
      new URL(`../../src/gpu-spatial-analysis/${family}/index.ts`, import.meta.url),
      'utf8'
    ),
    typescript.ScriptTarget.Latest,
    true
  );
  const exportedNames = source.statements.flatMap(statement =>
    typescript.isExportDeclaration(statement) &&
    statement.exportClause &&
    typescript.isNamedExports(statement.exportClause)
      ? statement.exportClause.elements
          .filter(element => statement.isTypeOnly || element.isTypeOnly)
          .map(element => element.name.text)
      : []
  );
  for (const name of names) expect(exportedNames, `${family}: ${name}`).toContain(name);
}

it('CPL-08 exposes both prediction decoders and names incomplete neighborhoods', () => {
  const status = new Uint32Array([99, 0, 1, 2, 3, 99]);
  const bytes = new Uint8Array(status.buffer, 4, 16);
  for (const decode of [spatial.decodeGPULocalKrigingStatus, spatial.decodeGPULocalRBFStatus]) {
    expect(typeof decode).toBe('function');
    // Status 3 is candidate loss, distinct from input failure (1) and a singular solve (2).
    expect(decode(bytes, 3)).toEqual({
      code: 3,
      complete: 0,
      invalidInput: 0,
      singular: 0,
      candidateOverflow: 1
    });
    expect(decode(bytes, 0)).toEqual({
      code: 0,
      complete: 1,
      invalidInput: 0,
      singular: 0,
      candidateOverflow: 0
    });
    expect(() => decode(new Uint8Array(3))).toThrow();
    expect(() => decode(bytes, 4)).toThrow();
  }
});

function createPublicPermutationMetadata(
  permutationCount = 9,
  stopAfterExceedances?: number
): spatial.GPUPermutationInferenceMetadata {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new spatial.GPUGlobalPermutationTest({
      weights: {
        offsets: createTransientView(graph, 'metadata-offsets', 'uint32', 5),
        neighbors: createTransientView(graph, 'metadata-neighbors', 'uint32', 4),
        weights: createTransientView(graph, 'metadata-weights', 'float32', 4)
      },
      values: createTransientView(graph, 'metadata-values', 'float32', 4),
      parameters: createTransientView(graph, 'metadata-parameters', 'uint32', 4),
      statistic: 'moran',
      alternative: 'greater',
      maximumPermutations: permutationCount,
      stopAfterExceedances,
      output: {summary: createTransientView(graph, 'metadata-summary', 'float32', 12)}
    });
    return contributor.getInferenceMetadata({seed: 2 ** 32 + 7, permutationCount});
  } finally {
    device.destroy();
  }
}

it('SSR-114 publishes selected permutation rules, actual counts, seed and pair-block limits', () => {
  expectFamilyTypeExports('permutation-inference', [
    'GPUPermutationInferenceMetadata',
    'GPUPermutationCauseStatus'
  ]);
  const metadata = createPublicPermutationMetadata(9, 2);
  expect(metadata.stoppingRule).toEqual({method: 'besag-clifford', stopAfterExceedances: 2});
  const summary = new Float32Array(14);
  summary[0] = summary[13] = 99;
  // Observed and both simulated Moran statistics are 0: two inclusive upper-tail exceedances.
  // The simulated mean/minimum/maximum/deviation are 0; z/p-normal are undefined at zero deviation.
  summary[7] = summary[8] = NaN;
  summary[12] = 4;
  // Published summary positions: p-value at 1, actual draws at 3. Window starts at word 1.
  summary[2] = 1;
  summary[3] = 2;
  summary[4] = 2;
  // A greater-tail stop at its second exceedance on draw two has M/L = 2/2 = 1.
  expect(summary[2]).toBe(1);
  expect(summary[3]).toBe(2);
  const global = spatial.decodeGPUGlobalPermutationTestInferenceMetadata(
    metadata,
    new Uint8Array(summary.buffer, 4, 48)
  );
  expect(global).toMatchObject({
    alternative: 'greater',
    seed: 4294967303,
    requestedPermutationCount: 9,
    actualPermutationCount: 2,
    stoppedEarly: true,
    stoppingRule: {method: 'besag-clifford', stopAfterExceedances: 2}
  });
  const counts = new Uint32Array([99, 9, 2, 0xffffffff, 99]);
  const local = spatial.decodeGPULocalPermutationTestInferenceMetadata(
    metadata,
    new Uint8Array(counts.buffer, 4, 12),
    3
  );
  expect(Array.from(local.actualPermutationCounts)).toEqual([9, 2, 4294967295]);
  expect(local.seed).toBe(4294967303);
  expect(local.alternative).toBe('greater');
  expect(local.stoppingRule).toEqual({method: 'besag-clifford', stopAfterExceedances: 2});
  expect(spatial.GLOBAL_PERMUTATION_PAIR_BLOCK_ROWS).toBe(2048);
  expect(spatial.MAXIMUM_GLOBAL_PERMUTATION_PAIR_BLOCK_COUNT).toBe(16777215);
  expect(() =>
    spatial.decodeGPUGlobalPermutationTestInferenceMetadata(metadata, new Float32Array(3))
  ).toThrow();
  expect(() =>
    spatial.decodeGPULocalPermutationTestInferenceMetadata(metadata, new Uint32Array(2), 3)
  ).toThrow();
  expectTypeOf<spatial.GPUPermutationCauseStatus>().toEqualTypeOf<
    Pick<
      spatial.GPUPermutationCauseStatus,
      'weightsOverflow' | 'candidateOverflow' | 'invalidInput'
    >
  >();
});

it('STA-09 exposes producer loss certificates, unknown requirements and validation options', () => {
  expectFamilyTypeExports('spatial-weights', [
    'GPUSpatialWeightsProducerOutput',
    'GPUSpatialWeightsRequirementObservation',
    'SpatialWeightsValidationOptions'
  ]);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const output: spatial.GPUSpatialWeightsProducerOutput = {
      offsets: createTransientView(graph, 'producer-offsets', 'uint32', 3),
      neighbors: createTransientView(graph, 'producer-neighbors', 'uint32', 4),
      weights: createTransientView(graph, 'producer-weights', 'float32', 4),
      status: {
        overflow: createTransientView(graph, 'producer-overflow', 'uint32', 1),
        candidateOverflow: createTransientView(graph, 'producer-candidate-overflow', 'uint32', 1)
      }
    };
    const described: spatial.GPUSpatialWeights = spatial.describeGPUSpatialWeights(output, {
      information: {rowCount: 2, transform: 'raw', symmetric: 'unknown'}
    });
    expect(described).not.toBe(output);
    expect(described.status?.overflow).toBe(output.status.overflow);
    expect(described.status?.candidateOverflow).toBe(output.status.candidateOverflow);
    const requirement: spatial.GPUSpatialWeightsRequirementObservation = {kind: 'unknown'};
    expect(
      spatial.getGPUSpatialWeightsCapacityPlan({
        rowCount: 2,
        capacity: 4,
        requiredCount: requirement
      })
    ).toMatchObject({capacity: 4, requirement: {kind: 'unknown'}, retryStage: 'candidates'});
    const options: spatial.SpatialWeightsValidationOptions = {columnCount: 3, diagonal: 'forbid'};
    // Target rows 0/1 refer to source columns 2/0 in a 2x3 matrix: both slots are legal.
    const result = spatial.computeSpatialWeightsValidationOnCPU(
      new Uint32Array([0, 1, 2]),
      new Uint32Array([2, 0]),
      new Float32Array([1, 1]),
      options
    );
    expect(Array.from(result.flags)).toEqual([0, 0]);
  } finally {
    device.destroy();
  }
});

it('UST-06 public model consumer reads applied ridge regularization and inference method', () => {
  expectFamilyTypeExports('spatial-regression', [
    'GPUOrdinaryLeastSquaresInputProps',
    'GPUOrdinaryLeastSquaresModelResult'
  ]);
  const summary = new Float32Array(spatial.GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH);
  summary[spatial.GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ROW_COUNT] = 4;
  // A report's applied lambda is 0.25 regardless of what the caller originally requested.
  summary[spatial.GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RIDGE_LAMBDA] = 0.25;
  const result: spatial.GPUOrdinaryLeastSquaresModelResult =
    spatial.getGPUOrdinaryLeastSquaresModelResult({
      coefficients: new Float32Array([1, 2]),
      standardErrors: new Float32Array([1, 1]),
      tStatistics: new Float32Array([1, 2]),
      summary,
      status: new Uint32Array([0])
    });
  expect(result.regularization).toEqual({kind: 'ridge', lambda: 0.25});
  expect(result.inferenceMethod).toBe('ridge-approximate');
  expectTypeOf<spatial.GPUOrdinaryLeastSquaresInputProps>().toEqualTypeOf<
    ConstructorParameters<typeof spatial.GPUOrdinaryLeastSquares>[0]
  >();
});

it.each([
  'exact',
  'tiled'
] as const)('W5-07 %s retains documented completeness and output identities', strategy => {
  expectFamilyTypeExports('spatial-autocorrelation', [
    'GPUMultivariateLocalGearyPermutationTestOutput'
  ]);
  expectTypeOf<spatial.GPUMultivariateLocalGearyPermutationTestOutput>().toEqualTypeOf<
    spatial.GPUMultivariateLocalGearyPermutationTestProps['output']
  >();
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const output: spatial.GPUMultivariateLocalGearyPermutationTestOutput = {
      statistic: createTransientView(graph, 'geary-statistic', 'float32', 3),
      pseudoPValues: createTransientView(graph, 'geary-p-values', 'float32', 3),
      classes: createTransientView(graph, 'geary-classes', 'uint32', 3)
    };
    const status = {
      complete: createTransientView(graph, 'geary-complete', 'uint32', 1)
    };
    const contributor = new spatial.GPUMultivariateLocalGearyPermutationTest({
      values: createTransientView(graph, 'geary-values', 'float32', 3),
      variableCount: 1,
      strategy,
      permutationCount: 9,
      weights: {
        offsets: createTransientView(graph, 'geary-offsets', 'uint32', 4),
        neighbors: createTransientView(graph, 'geary-neighbors', 'uint32', 3),
        weights: createTransientView(graph, 'geary-weights', 'float32', 3),
        status
      },
      output
    });
    for (const name of ['statistic', 'pseudoPValues', 'classes'] as const) {
      expect(output[name]).toBeDefined();
      expect(contributor.props.output[name]).toBe(output[name]);
    }
    expect(contributor.props.weights.status?.complete).toBe(status.complete);
    expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('W5-08 publishes the expectation-scan shape, device limits and actual planner constraints', () => {
  expectFamilyTypeExports('scan-statistics', [
    'GPUExpectationBasedSpaceTimeScanShape',
    'GPUExpectationBasedSpaceTimeScanDeviceLimits'
  ]);
  expect(spatial.GPU_EXPECTATION_BASED_SCAN_LIMITS).toBe(
    spatial.GPUExpectationBasedSpaceTimeScan.limits
  );
  const shape: spatial.GPUExpectationBasedSpaceTimeScanShape = {
    zones: 2,
    timeBuckets: 3,
    maximumNeighbors: 1,
    maximumWindowLength: 2,
    maximumReplicates: 9
  };
  const limits: spatial.GPUExpectationBasedSpaceTimeScanDeviceLimits = {
    maxBufferSize: 134217728,
    maxStorageBufferBindingSize: 134217728
  };
  const plan = spatial.GPUExpectationBasedSpaceTimeScan.getExecutionPlan(shape, limits);
  expect(plan.route).toBe('indexed');
  expect(plan.feasible).toBe(true);
  // (9 replicates + observed) * 2 zones * 2 window cells * 4 bytes = 160 bytes.
  expect(plan.constraints.find(constraint => constraint.name === 'count table')).toMatchObject({
    received: 160,
    satisfied: true
  });
  for (const constraint of plan.constraints) {
    const descriptor = spatial.GPU_EXPECTATION_BASED_SCAN_LIMITS.find(
      entry => entry.name === constraint.name
    )!;
    expect(descriptor).toBeDefined();
    expect(constraint.planner).toBe(descriptor.planner);
    expect(constraint.quantity).toBe(descriptor.quantity);
  }
});

it('W5-09 exposes candidate and harmonic limits separately from row and neighbor bounds', () => {
  expect(spatial.GPU_NESTED_VALIDATION_MAXIMUM_CANDIDATE_COUNT).toBe(16);
  expect(spatial.MAXIMUM_MULTIPLE_TESTING_EXACT_HARMONIC_TERMS).toBe(64);
  expect(spatial.GPU_NESTED_VALIDATION_MAXIMUM_ROW_COUNT).toBe(4096);
  expect(spatial.GPU_NESTED_VALIDATION_REFERENCE_MAXIMUM_ROW_COUNT).toBe(256);
  expect(spatial.GPU_LOCAL_KRIGING_MAXIMUM_NEIGHBOR_COUNT).toBe(32);
});

it('SCL-12 exports the island free planner and route objects used by the public static API', () => {
  expect(typeof spatial.getGPUIslandAttachmentExecutionPlan).toBe('function');
  expect(spatial.GPU_ISLAND_ATTACHMENT_ROUTES).toBe(spatial.GPUIslandAttachment.routes);
  expect(spatial.GPU_ISLAND_ATTACHMENT_PLANNER.getExecutionPlan).toBe(
    spatial.getGPUIslandAttachmentExecutionPlan
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props: spatial.GPUIslandAttachmentProps = {
      weights: {
        offsets: createTransientView(graph, 'island-offsets', 'uint32', 3),
        neighbors: createTransientView(graph, 'island-neighbors', 'uint32', 4),
        weights: createTransientView(graph, 'island-weights', 'float32', 4)
      },
      positions: createTransientView(graph, 'island-positions', 'float32x2', 2),
      output: {
        offsets: createTransientView(graph, 'island-output-offsets', 'uint32', 3),
        neighbors: createTransientView(graph, 'island-output-neighbors', 'uint32', 4),
        weights: createTransientView(graph, 'island-output-weights', 'float32', 4),
        attachment: createTransientView(graph, 'island-attachment', 'uint32', 2),
        added: createTransientView(graph, 'island-added', 'uint32', 4),
        status: {overflow: createTransientView(graph, 'island-overflow', 'uint32', 1)}
      }
    };
    for (const nearestSearch of ['auto', 'scan', 'grid'] as const) {
      const routeProps = {...props, nearestSearch};
      const plan = spatial.getGPUIslandAttachmentExecutionPlan(routeProps, device.limits);
      expect(plan.feasible).toBe(true);
      expect(plan).toEqual(spatial.GPUIslandAttachment.getExecutionPlan(routeProps, device.limits));
      expect(plan).toEqual(
        spatial.GPU_ISLAND_ATTACHMENT_PLANNER.getExecutionPlan(routeProps, device.limits)
      );
      expect(plan.route).toBe(nearestSearch === 'grid' ? 'grid' : 'scan');
      const shortAdded = createTransientView(
        graph,
        `island-short-added-${nearestSearch}`,
        'uint32',
        1
      );
      const invalidProps = {...routeProps, output: {...props.output, added: shortAdded}};
      for (const planner of [
        spatial.getGPUIslandAttachmentExecutionPlan,
        spatial.GPUIslandAttachment.getExecutionPlan,
        spatial.GPU_ISLAND_ATTACHMENT_PLANNER.getExecutionPlan
      ]) {
        const invalidPlan = planner(invalidProps, device.limits);
        expect(invalidPlan.feasible).toBe(false);
      }
    }
    // The public implementation accepts the same optional device argument as the free/static APIs.
    expect(spatial.GPU_ISLAND_ATTACHMENT_PLANNER.getExecutionPlan(props)).toEqual(
      spatial.getGPUIslandAttachmentExecutionPlan(props)
    );
    expectTypeOf<
      Parameters<typeof spatial.GPU_ISLAND_ATTACHMENT_PLANNER.getExecutionPlan>
    >().toEqualTypeOf<Parameters<typeof spatial.getGPUIslandAttachmentExecutionPlan>>();
  } finally {
    device.destroy();
  }
});

const foreignPredictionReadbacks = [
  ['uint32', 'Uint32Array.of(3)'],
  ['bytes', 'new Uint8Array(Uint32Array.of(99, 3, 99).buffer, 4, 4)'],
  ['buffer', 'Uint32Array.of(3).buffer']
] as const;
for (const [name, decode] of [
  ['kriging', spatial.decodeGPULocalKrigingStatus],
  ['rbf', spatial.decodeGPULocalRBFStatus]
] as const) {
  it.each(
    foreignPredictionReadbacks
  )(`B3 ${name} accepts foreign %s status readback`, (_kind, expression) => {
    const input = runInNewContext(expression) as Uint32Array | Uint8Array | ArrayBuffer;
    expect(decode(input)).toEqual({
      code: 3,
      complete: 0,
      invalidInput: 0,
      singular: 0,
      candidateOverflow: 1
    });
  });
}

it.each([
  ['uint32', 'Uint32Array.of(13)'],
  ['bytes', 'new Uint8Array(Uint32Array.of(99, 13, 99).buffer, 4, 4)'],
  ['buffer', 'Uint32Array.of(13).buffer']
] as const)('B3 local metadata accepts foreign %s readback', (_kind, expression) => {
  const input = runInNewContext(expression) as Uint32Array | Uint8Array | ArrayBuffer;
  const metadata = createPublicPermutationMetadata(13);
  const result = spatial.decodeGPULocalPermutationTestInferenceMetadata(metadata, input, 1);
  expect(Array.from(result.actualPermutationCounts)).toEqual([13]);
  expect(result.stoppingRule).toEqual({method: 'fixed-count'});
});

it.each([
  ['float32', 'Float32Array.of(0, 1, 13, 13, 0, 0, NaN, NaN, 0, 0, 0, 4)'],
  [
    'bytes',
    'new Uint8Array(Float32Array.of(99, 0, 1, 13, 13, 0, 0, NaN, NaN, 0, 0, 0, 4, 99).buffer, 4, 48)'
  ],
  ['buffer', 'Float32Array.of(0, 1, 13, 13, 0, 0, NaN, NaN, 0, 0, 0, 4).buffer']
] as const)('B3 global metadata accepts foreign %s readback', (_kind, expression) => {
  const input = runInNewContext(expression) as Float32Array | Uint8Array | ArrayBuffer;
  const result = spatial.decodeGPUGlobalPermutationTestInferenceMetadata(
    createPublicPermutationMetadata(13),
    input
  );
  expect(result.actualPermutationCount).toBe(13);
  expect(result.available).toBe(true);
  expect(result.stoppedEarly).toBe(false);
});

it('NOTE-2 local metadata rejects float32 elements even when their bits encode valid uint32 counts', () => {
  const wrongElements = new Float32Array(Uint32Array.of(13).buffer);
  expect(() =>
    spatial.decodeGPULocalPermutationTestInferenceMetadata(
      createPublicPermutationMetadata(13),
      wrongElements as unknown as Uint32Array,
      1
    )
  ).toThrow();
});

it('NOTE-2 global metadata rejects uint32 elements even when their bits encode valid float32 totals', () => {
  const wrongElements = new Uint32Array(
    Float32Array.of(0, 1, 13, 13, 0, 0, NaN, NaN, 0, 0, 0, 4).buffer
  );
  expect(() =>
    spatial.decodeGPUGlobalPermutationTestInferenceMetadata(
      createPublicPermutationMetadata(13),
      wrongElements as unknown as Float32Array
    )
  ).toThrow();
});

it('NOTE-1 island planning refuses unsupported distances output like the constructor', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props: spatial.GPUIslandAttachmentProps = {
      weights: {
        offsets: createTransientView(graph, 'distance-input-offsets', 'uint32', 4),
        neighbors: createTransientView(graph, 'distance-input-neighbors', 'uint32', 6),
        weights: createTransientView(graph, 'distance-input-weights', 'float32', 6)
      },
      positions: createTransientView(graph, 'distance-positions', 'float32x2', 3),
      output: {
        offsets: createTransientView(graph, 'distance-output-offsets', 'uint32', 4),
        neighbors: createTransientView(graph, 'distance-output-neighbors', 'uint32', 6),
        weights: createTransientView(graph, 'distance-output-weights', 'float32', 6),
        attachment: createTransientView(graph, 'distance-attachment', 'uint32', 3),
        added: createTransientView(graph, 'distance-added', 'uint32', 6),
        distances: createTransientView(graph, 'distance-output-distances', 'float32', 6),
        status: {overflow: createTransientView(graph, 'distance-overflow', 'uint32', 1)}
      }
    };
    expect(() => new spatial.GPUIslandAttachment(props)).toThrow(/distances/);
    expect(spatial.getGPUIslandAttachmentExecutionPlan(props, device.limits).feasible).toBe(false);
    const withoutDistances = {...props, output: {...props.output, distances: undefined}};
    expect(() => new spatial.GPUIslandAttachment(withoutDistances)).not.toThrow();
    expect(
      spatial.getGPUIslandAttachmentExecutionPlan(withoutDistances, device.limits).feasible
    ).toBe(true);
  } finally {
    device.destroy();
  }
});
