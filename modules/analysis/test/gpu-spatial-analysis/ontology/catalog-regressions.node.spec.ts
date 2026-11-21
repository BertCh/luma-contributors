// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {ModuleKind, ScriptTarget, transpileModule} from 'typescript';
import {createTransientView, GPUCommandGraph, GPUGather, GPUMask} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/gpu-key-join';
import {
  computeClassBreaksOnCPU,
  CPUClassBreaksOperation
} from '../../../src/dataframe-analysis/column-classification/cpu-class-breaks-operation';
import {getGPUClassBreaksParameterValues} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {GPUClassBreaks} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import {computeColumnQuantilesOnCPU} from '../../../src/dataframe-analysis/column-classification/cpu-column-quantiles-operation';
import {
  getColumnQuantilesParameterValues,
  getColumnQuantilesRationalProbabilityValues,
  getGPUColumnQuantilesParameterValues,
  getGPUColumnQuantilesRationalProbabilityValues,
  getGPUTimeWindowFilterWordParameterValues,
  joinTimeWords,
  prepareGPUScalarScaleControls,
  GPUCubeLayoutTranspose,
  GPUSpaceTimeVariogram,
  fitVariogramModelWithDiagnostics,
  getGPUSpaceTimeVariogramBinLags,
  fitSpaceTimeVariogramModel,
  evaluateSpaceTimeVariogramModel,
  computeInequalityOnCPU,
  getGPUInequalityParameterValues,
  GPUInequality,
  GPUInequalityOperation,
  CPUInequalityOperation,
  type InequalityOperationProps
} from '../../../src/dataframe-analysis/index';
import {
  GPUOrdinaryLeastSquares,
  type GPUOrdinaryLeastSquaresOutput
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../../src/raster-analysis/raster-grid/index';
import {computeNaturalBreaksMirror} from '../../gpu-dataframe/column-classification/natural-breaks-mirror';
import {
  GPUTerrainTiledFlow,
  type GPUTerrainFlowTile
} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {
  planTerrainD8Tiles,
  validateTerrainD8TileBindings
} from '../../../src/gpu-terrain/terrain-processing/terrain-tiled-plan';
import {resolveClassBreaksMethods} from '../../../src/dataframe-analysis/column-classification/class-breaks-input';
import {GPUIslandAttachment} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {
  GPUClassBreaksOperation,
  type ClassBreaksOperationProps
} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks-operation';
import {GPU_CONTRIBUTOR_OPERATION_LOWERING} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {getOrderedFloat32Key} from '../../../src/dataframe-analysis/column-classification/column-classification-shared';
import {OPERATOR_ANNOTATIONS} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/operator-annotations';
import {
  findGPUAnalysisOperatorPaths,
  getGPUAnalysisNextOperators,
  queryGPUAnalysisOperatorsByKind
} from '../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-operator-kind-supplements';
import {REUSED_ANALYSIS_OPERATORS} from '../../../src/gpu-spatial-analysis/ontology/reused-analysis-catalog';
import {getGPUAnalysisDataKindRecord} from '../../../src/gpu-spatial-analysis/ontology/analysis-experience/data-kind-vocabulary';
import {
  getGPUAnalysisCapabilityCatalog,
  getGPUAnalysisOperator,
  getGPUAnalysisOperatorCatalog,
  queryGPUAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';
import {
  GPU_SPATIAL_ANALYSIS_DATA_KINDS,
  type GPUAnalysisPackage
} from '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-ontology';
import {isGPUSpatialAnalysisDataKind} from '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-operators';
import {
  classifyGPUAnalysisResultValues,
  validateGPUAnalysisResultEncoding
} from '../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-interpretation';

it('counts every nonzero uint32 validity flag as usable and fully encoded', () => {
  const values = Uint32Array.of(0, 1, 2, 0xffffffff);
  const tally = classifyGPUAnalysisResultValues('validity-flag', values);
  expect(tally.counts.get(1)).toBe(3);
  expect(tally.counts.get(0)).toBe(1);
  expect(tally.unrecognized).toBe(0);
  expect(validateGPUAnalysisResultEncoding('validity-flag', 'contributor-integer', values)).toBe(1);
  expect(
    classifyGPUAnalysisResultValues('validity-flag', [-1, 0.5, 2 ** 32, '2']).unrecognized
  ).toBe(4);
  expect(classifyGPUAnalysisResultValues('significance-flag', [2]).unrecognized).toBe(1);
});

it('retains resolvable qualified visibility identities in paths and next operators', () => {
  const records = queryGPUAnalysisOperatorsByKind({input: 'points', output: 'network'});
  const result = findGPUAnalysisOperatorPaths('points', 'network', 1, 500);
  expect(result.totalFound).toBe(records.length);
  expect(result.paths.flat().map(step => getGPUAnalysisOperator(step))).toEqual(
    expect.arrayContaining([...records])
  );
  const next = getGPUAnalysisNextOperators('points');
  const visibility = next.filter(
    step => typeof step !== 'string' && step.exportName === 'addVisibilityGraphRecipe'
  );
  expect(visibility).toHaveLength(2);
  for (const step of visibility) expect(getGPUAnalysisOperator(step)).toBeDefined();
});

it('publishes the actual spatial and network visibility output and status contracts', () => {
  const names = (query: Parameters<typeof queryGPUAnalysisOperators>[0]) =>
    queryGPUAnalysisOperators(query)
      .filter(operator => operator.exportName === 'addVisibilityGraphRecipe')
      .map(operator => operator.package)
      .sort();
  expect(names({status: 'candidateOverflow'})).toEqual([
    '@luma-contributors/analysis/gpu-network',
    '@luma-contributors/analysis/gpu-spatial-analysis'
  ]);
  expect(names({status: 'zeroWeightCycle'})).toEqual(['@luma-contributors/analysis/gpu-network']);
  expect(names({output: 'polygons'})).toEqual([]);
});

it('protects canonical kinds and reused catalog arrays from mutation across callers', () => {
  const index = GPU_SPATIAL_ANALYSIS_DATA_KINDS.indexOf('time-series');
  const changed = Reflect.set(GPU_SPATIAL_ANALYSIS_DATA_KINDS, index, 'replacement');
  if (changed) Reflect.set(GPU_SPATIAL_ANALYSIS_DATA_KINDS, index, 'time-series');
  expect(changed).toBe(false);
  expect(isGPUSpatialAnalysisDataKind('time-series')).toBe(true);
  expect(isGPUSpatialAnalysisDataKind('replacement')).toBe(false);
  expect(getGPUAnalysisDataKindRecord('time-series')).toBeDefined();
  const last = REUSED_ANALYSIS_OPERATORS.at(-1);
  const shortened = Reflect.set(REUSED_ANALYSIS_OPERATORS, 'length', 8);
  if (shortened) Reflect.set(REUSED_ANALYSIS_OPERATORS, 8, last);
  expect(shortened).toBe(false);
  expect(REUSED_ANALYSIS_OPERATORS).toHaveLength(9);
  for (const operator of REUSED_ANALYSIS_OPERATORS)
    expect(
      getGPUAnalysisOperator({package: operator.package, exportName: operator.exportName})
    ).toMatchObject(operator);
});

it('keeps unaudited metadata unknown instead of claiming no status, readback or scale limits', () => {
  const recorder = getGPUAnalysisOperator('GPUComputationRecorder')!;
  expect(recorder.status).toBe('unknown');
  expect(recorder.readback).toBe('unknown');
  expect(recorder.tier).toBe('unknown');
  expect(queryGPUAnalysisOperators({status: 'unknown', includeHelpers: true})).toContain(recorder);
  expect(queryGPUAnalysisOperators({readback: 'unknown', includeHelpers: true})).toContain(
    recorder
  );
});

it('finds audited k-shortest-path statuses and explicit snapshot readback', () => {
  for (const status of ['candidateOverflow', 'converged', 'iterationLimitReached'] as const) {
    expect(queryGPUAnalysisOperators({status}).map(operator => operator.exportName)).toContain(
      'GPUKShortestPaths'
    );
  }
  expect(getGPUAnalysisOperator('GPUSnapshotArena')?.readback).toBe('explicit-adapter');
});

it('discovers Gaussian random fields in both point and grid workflows', () => {
  expect(
    queryGPUAnalysisOperators({input: 'surface', output: 'surface'}).map(
      operator => operator.exportName
    )
  ).toContain('GPUGaussianRandomField');
  expect(
    queryGPUAnalysisOperators({input: 'points', output: 'attributes'}).map(
      operator => operator.exportName
    )
  ).toContain('GPUGaussianRandomField');
});

it('publishes the fitting encoder annotation through the catalog', () => {
  expect(getGPUAnalysisOperator('GPUFittingCoordinateEncoder')?.limitation).toContain(
    'chunked rows'
  );
});

it('applies each public annotation to exactly one final identity', () => {
  for (const [key, annotation] of Object.entries(OPERATOR_ANNOTATIONS)) {
    const records = getGPUAnalysisOperatorCatalog().filter(operator =>
      key.includes(':')
        ? key === `${operator.package}:${operator.exportName}`
        : key === operator.exportName
    );
    expect(records, key).toHaveLength(1);
    expect(records[0], key).toMatchObject(annotation);
  }
});

it('records every enforced network inhomogeneous K size bound', () => {
  const limits = getGPUAnalysisOperator('GPUNetworkInhomogeneousK')!.limits!;
  expect(limits.find(limit => limit.name === 'nodeCount')?.maximum).toBe(65536);
  expect(limits.find(limit => limit.name === 'edgeCount')?.maximum).toBe(65536);
  expect(limits.find(limit => limit.name === 'eventCount * nodeCount')?.maximum).toBe(16777216);
  const source = readFileSync(
    'modules/analysis/src/gpu-network/gpu-network-inhomogeneous-k.ts',
    'utf8'
  );
  for (const limit of limits) {
    const constant = limit.reason.match(/^(MAXIMUM_[A-Z_]+):/)?.[1];
    expect(constant, limit.name).toBeTruthy();
    const value = source.match(new RegExp(`const ${constant} = ([0-9_]+);`))?.[1];
    expect(Number(value?.replaceAll('_', '')), limit.name).toBe(limit.maximum);
  }
});

it('distinguishes the two public visibility recipes by entry point and rejects ambiguous names', () => {
  const exportName = 'addVisibilityGraphRecipe';
  const spatial = getGPUAnalysisOperator({
    package: '@luma-contributors/analysis/gpu-spatial-analysis',
    exportName
  });
  const network = getGPUAnalysisOperator({
    package: '@luma-contributors/analysis/gpu-network',
    exportName
  });
  expect(spatial?.package).toBe('@luma-contributors/analysis/gpu-spatial-analysis');
  expect(network?.package).toBe('@luma-contributors/analysis/gpu-network');
  expect(spatial).not.toBe(network);
  expect(spatial?.limitation).toContain('adjacency');
  expect(network?.limitation).toContain('sampled');
  expect(() => getGPUAnalysisOperator(exportName)).toThrow(/ambiguous.*package/i);
});

it('recognizes every canonical data kind even when historical families omit it', () => {
  for (const kind of GPU_SPATIAL_ANALYSIS_DATA_KINDS) {
    expect(isGPUSpatialAnalysisDataKind(kind), kind).toBe(true);
  }
  expect(isGPUSpatialAnalysisDataKind('not-a-kind')).toBe(false);
});

it('freezes shared catalog records and their nested values', () => {
  const operator = getGPUAnalysisOperator('GPUNetworkFacilityOptimization')!;
  expect(Object.isFrozen(getGPUAnalysisOperatorCatalog())).toBe(true);
  expect(Object.isFrozen(operator)).toBe(true);
  expect(Object.isFrozen(operator.evidence)).toBe(true);
  expect(Object.isFrozen(operator.limits)).toBe(true);
  expect(Object.isFrozen(operator.limits![0])).toBe(true);
  expect(Reflect.set(operator.evidence, 'contract', 'Changed')).toBe(false);
  expect(Reflect.set(operator.limits![0], 'maximum', 1)).toBe(false);
  expect(Reflect.set(operator, 'exportName', 'Changed')).toBe(false);
  expect(getGPUAnalysisOperator('GPUNetworkFacilityOptimization')?.exportName).toBe(
    'GPUNetworkFacilityOptimization'
  );
});

it('derives all-entry family membership from every finalized operator record', () => {
  for (const capability of getGPUAnalysisCapabilityCatalog()) {
    const records = getGPUAnalysisOperatorCatalog().filter(
      operator => operator.primaryCapability === capability.id
    );
    expect(capability.operators, capability.id).toEqual(
      records.map(operator => operator.exportName)
    );
    expect(capability.operatorIdentities, capability.id).toEqual(
      records.map(({package: packageName, exportName}) => ({package: packageName, exportName}))
    );
    expect(Object.isFrozen(capability)).toBe(true);
    expect(Object.isFrozen(capability.operatorIdentities)).toBe(true);
  }
});

// Qualified identities prevent a sibling entry point from satisfying these route contracts.
function getRouteOperator(
  exportName: string,
  family: GPUAnalysisPackage extends `@luma-contributors/analysis/${infer Family}` ? Family : never
) {
  return getGPUAnalysisOperator({package: `@luma-contributors/analysis/${family}`, exportName})!;
}

it('scopes natural-breaks source-row bounds to their enabled packed routes (SCL-05)', () => {
  const operator = getRouteOperator('GPUClassBreaks', 'dataframe-analysis');
  expect(operator.limitation).toContain('when natural-breaks is compiled');
  expect(operator.limits).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        name: 'source rows (natural-breaks exact materialized)',
        maximum: 1024
      }),
      expect.objectContaining({
        name: 'source rows (natural-breaks exact recompute)',
        maximum: 8192
      }),
      expect.objectContaining({name: 'source rows (natural-breaks monotone)', maximum: 262144})
    ])
  );
});

it('separates raster dispatch slices, stream headroom and cell-slot observations (SCL-06)', () => {
  const operator = getRouteOperator('GPURasterStreamingAccumulator', 'raster-analysis');
  expect(operator.limitation).toContain('16,777,216');
  expect(operator.limitation).toContain('per cell and calendar slot');
  expect(operator.limits).toEqual(
    expect.arrayContaining([expect.objectContaining({name: 'chunkSliceCount', maximum: 256})])
  );
  expect(JSON.stringify(operator.limits)).not.toContain('no slice limit');
  expect(
    operator.limits?.some(limit => limit.name === 'valid observations per cell and slot')
  ).toBe(false);
  expect(operator.limitation).toContain('4,294,967,295');
});

it('separates island scan crossover from forced maximum and grid plan feasibility (SCL-12)', () => {
  const operator = getRouteOperator('GPUIslandAttachment', 'gpu-spatial-analysis');
  expect(operator.limitation).toContain('4,096');
  expect(operator.limitation).toContain('268,435,455');
  expect(operator.limitation).toContain('schedule');
  expect(operator.limits).toEqual(
    expect.arrayContaining([
      expect.objectContaining({name: 'candidate slots (grid rows * 16)', maximum: 4294967295}),
      expect.objectContaining({name: 'positions.length (scan)', maximum: 65535})
    ])
  );
  expect(operator.limits?.some(limit => limit.name === 'positions.length (grid)')).toBe(false);
});

it('publishes separate resident and tiled D8 terrain domain bounds (SCL-15)', () => {
  const operator = getRouteOperator('GPUTerrainTiledFlow', 'gpu-terrain');
  expect(operator.limits).toEqual(
    expect.arrayContaining([
      expect.objectContaining({name: 'grid cell count (resident)', maximum: 536870911}),
      expect.objectContaining({name: 'grid cell count (tiled D8)', maximum: 4294967295}),
      expect.objectContaining({
        name: 'owned cells per tile and total coarse boundary cells (tiled D8)',
        maximum: 536870911
      })
    ])
  );
  expect(operator.limitation).toContain('device bindings');
});

it('pins the documented CPU Fisher-Jenks and default GPU-mirror distinction (DEF-06)', () => {
  const values = Float32Array.of(174, 15, 79, 177, 157, 114, 36, 56, 179, 166, 104, 100000);
  const parameters = getGPUClassBreaksParameterValues({method: 'natural-breaks', classCount: 4}, 4);
  const reference = computeClassBreaksOnCPU(values, parameters, {
    maximumClassCount: 4,
    methods: ['natural-breaks']
  });
  // Independent Python: enumerate combinations(range(1,12),3) on sorted(values),
  // minimize sum(sum((x-sum(group)/len(group))**2 for x in group) for group in groups).
  // Optimum inner edges [79,157,100000], total SSD 1819.8666666666666.
  // The binned mirror is evidence about the selected algorithm, not device execution.
  expect(Array.from(reference.breaks)).toEqual([15, 79, 157, 100000, 100000]);
  expect(reference.provenance).toMatchObject({
    naturalBreaksMode: 'approximate',
    naturalBreaksAlgorithm: 'exact-fisher-jenks-float64'
  });
  const mirror = computeNaturalBreaksMirror(values, 15, 100000, 4, 2048);
  expect(mirror.edges).toEqual([104, 157, 100000]);
  expect(reference.breaks.every(Number.isFinite)).toBe(true);
  expect(mirror.edges.every(Number.isFinite)).toBe(true);
  expect(Array.from(reference.breaks.slice(1, 4)).filter(edge => edge <= 79)).toHaveLength(1);
  expect(mirror.edges.filter(edge => edge <= 79)).toHaveLength(0);
});

it('pins the pure two-class empty selection result (FRM-12)', () => {
  const result = computeClassBreaksOnCPU(
    new Float32Array(),
    getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2),
    {maximumClassCount: 2, methods: ['equal-interval']}
  );
  expect(Array.from(result.breaks)).toEqual([NaN, NaN, NaN]);
  expect(Array.from(result.classCount)).toEqual([0]);
  expect(Array.from(result.classCounts)).toEqual([0, 0]);
});

it('packs 65 CPU medians through both neutral helpers (TWN-14)', () => {
  const parameters = getColumnQuantilesParameterValues({quantiles: Array(65).fill(0.5)});
  const rationalProbabilities = getColumnQuantilesRationalProbabilityValues(
    Array.from({length: 65}, () => [1, 2] as const)
  );
  for (const rational of [undefined, rationalProbabilities]) {
    const result = computeColumnQuantilesOnCPU(
      {values: Float32Array.of(0, 2), parameters, rationalProbabilities: rational},
      65
    );
    // Linear median of [0,2] = (0+2)/2 = 1, independently of probability encoding.
    expect(Array.from(result.quantiles)).toEqual(Array(65).fill(1));
  }
});

it('admits a 536870912-cell tiled D8 domain while resident rejects it (SCL-15)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const tiles: GPUTerrainFlowTile[] = [];
    for (let row = 0; row < 16384; row += 1024) {
      for (let column = 0; column < 32768; column += 1024) {
        const prefix = `tile-${row}-${column}`;
        const directions = createTransientView(graph, `${prefix}-directions`, 'uint32', 1048576);
        tiles.push({
          row,
          column,
          width: 1024,
          height: 1024,
          owned: [0, 0, 1024, 1024],
          flowDirections: directions,
          streams: directions,
          runoff: createTransientView(graph, `${prefix}-runoff`, 'float32', 1048576),
          output: {
            accumulation: createTransientView(graph, `${prefix}-accumulation`, 'float32', 1048576),
            status: createTransientView(graph, `${prefix}-status`, 'uint32', 1048576),
            reachIds: createTransientView(graph, `${prefix}-reach`, 'uint32', 1048576)
          }
        });
      }
    }
    const props = {
      grid: {
        width: 32768,
        height: 16384,
        affine: [1, 0, 0, 0, 1, 0] as const,
        pixelInterpretation: 'area' as const
      },
      tiles,
      execution: 'tiled' as const
    };
    const contributor = new GPUTerrainTiledFlow(props);
    const plan = planTerrainD8Tiles(contributor.id, props);
    // 32*16 disjoint tiles, 1024² owned cells and 4*1024-4 boundary cells each.
    expect(tiles).toHaveLength(512);
    expect(plan.maximumCellCount).toBe(1048576);
    expect(plan.boundaryCount).toBe(2095104);
    expect(plan.estimatedCommandNodeCount).toBe(3072);
    expect(() => validateTerrainD8TileBindings(graph, contributor.id, plan)).not.toThrow();
    expect(() => new GPUTerrainTiledFlow({...props, execution: 'resident'})).toThrow(
      /resident grid cell count.*536870911/
    );
  } finally {
    device.destroy();
  }
});

it('admits equal-interval-only above the monotone natural-breaks row bound (SCL-05 control)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUClassBreaks({
      maximumClassCount: 2,
      methods: ['equal-interval'],
      naturalBreaksMode: 'monotone',
      values: createTransientView(graph, 'values', 'float32', 262145),
      parameters: createTransientView(graph, 'parameters', 'float32', 11),
      output: {
        breaks: createTransientView(graph, 'breaks', 'float32', 3),
        classCount: createTransientView(graph, 'count', 'uint32', 1)
      }
    });
    expect(contributor.methods).toEqual(['equal-interval']);
  } finally {
    device.destroy();
  }
});

it('rejects empty resolved class-break graph input so callers can guard it (FRM-12)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    expect(
      () =>
        new GPUClassBreaks({
          maximumClassCount: 2,
          methods: ['equal-interval'],
          values: createTransientView(graph, 'empty-values', 'float32', 0),
          parameters: createTransientView(graph, 'parameters', 'float32', 11),
          output: {
            breaks: createTransientView(graph, 'breaks', 'float32', 3),
            classCount: createTransientView(graph, 'count', 'uint32', 1)
          }
        })
    ).toThrow(/values.*length/);
  } finally {
    device.destroy();
  }
});

it('resolves compatible two-class defaults and rejects explicit incompatible methods (DEF-07)', () => {
  const expected = [
    'equal-interval',
    'quantile',
    'standard-deviation',
    'head-tail',
    'maximum-breaks',
    'natural-breaks',
    'custom'
  ];
  expect(resolveClassBreaksMethods('catalog-defaults', {maximumClassCount: 2})).toEqual(expected);
  expect(resolveClassBreaksMethods('catalog-defaults', {maximumClassCount: 2}, true)).toEqual(
    expected.filter(method => method !== 'maximum-breaks')
  );
  expect(() =>
    resolveClassBreaksMethods('catalog-defaults', {maximumClassCount: 2, methods: ['box-plot']})
  ).toThrow(/box-plot.*6/);
  expect(() =>
    resolveClassBreaksMethods(
      'catalog-defaults',
      {maximumClassCount: 2, methods: ['maximum-breaks']},
      true
    )
  ).toThrow(/packed/);
  const result = computeClassBreaksOnCPU(
    Float32Array.of(0, 2, 4),
    getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2),
    {maximumClassCount: 2}
  );
  expect(Array.from(result.breaks)).toEqual([0, 2, 4]);
});

it('checks island forced-scan admission and rejects the grid arithmetic ceiling schedule (SCL-12)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let sequence = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `island-${sequence++}`, format, length);
    const create = (rows: number, nearestSearch: 'scan' | 'grid') =>
      new GPUIslandAttachment({
        nearestSearch,
        positions: view('float32x2', rows),
        weights: {
          offsets: view('uint32', rows + 1),
          neighbors: view('uint32', rows),
          weights: view('float32', rows)
        },
        output: {
          offsets: view('uint32', rows + 1),
          neighbors: view('uint32', rows * 3),
          weights: view('float32', rows * 3),
          attachment: view('uint32', rows),
          added: view('uint32', rows * 3),
          status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
        }
      });
    expect(() => create(4097, 'scan')).not.toThrow();
    expect(() => create(65535, 'scan')).not.toThrow();
    expect(() => create(65536, 'scan')).toThrow(/65535/);
    // floor((2^32-1)/16) = 268435455 meets candidate addressing but not scheduling.
    expect(() => create(268435455, 'grid')).toThrow(/command node count.*32768/);
    expect(() => create(268435456, 'grid')).toThrow(/candidate slots/);
  } finally {
    device.destroy();
  }
});

function getReviewOperationProps(
  overrides: Partial<ClassBreaksOperationProps> = {}
): ClassBreaksOperationProps {
  return {
    id: 'catalog-review',
    values: 'values',
    parameters: 'parameters',
    maximumClassCount: 2,
    methods: ['equal-interval'],
    output: {breaks: 'breaks', classCount: 'classCount', classCounts: 'classCounts'},
    ...overrides
  };
}

function createReviewCPURows(chunks: Float32Array[]) {
  const values = Float32Array.from(chunks.flatMap(chunk => Array.from(chunk)));
  return {
    format: 'float32' as const,
    length: values.length,
    chunks: chunks.map(chunk => ({values: chunk, length: chunk.length})),
    pack: () => values
  };
}

it('admits fitted and custom empty CPU operations but rejects empty GPU lowering (B1)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    for (const method of ['equal-interval', 'custom'] as const) {
      const props = getReviewOperationProps({rowCount: 0, methods: [method]});
      const parameters = getGPUClassBreaksParameterValues(
        {method, classCount: 2, customEdges: [-5, 0, 7]},
        2
      );
      const result = new CPUClassBreaksOperation(props).execute({
        inputs: {
          values: createReviewCPURows([new Float32Array()]),
          parameters: createReviewCPURows([parameters])
        }
      }) as Record<string, Float32Array | Uint32Array>;
      expect(Array.from(result['breaks'])).toEqual(
        method === 'custom' ? [-5, 0, 7] : [NaN, NaN, NaN]
      );
      expect(Array.from(result['classCount'])).toEqual([method === 'custom' ? 2 : 0]);
      expect(Array.from(result['classCounts'])).toEqual([0, 0]);
      const operation = new GPUClassBreaksOperation(props);
      const empty = createTransientView(graph, `empty-${method}`, 'float32', 0);
      expect(() =>
        GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(operation, {
          resolveVector: () => ({data: [empty]})
        } as unknown as Parameters<typeof GPU_CONTRIBUTOR_OPERATION_LOWERING.lower>[1])
      ).toThrow(/values.length.*at least one row/);
    }
  } finally {
    device.destroy();
  }
});

it('exposes CPU and raw GPU natural-break provenance and preflights both operations (B2)', () => {
  for (const Operation of [CPUClassBreaksOperation, GPUClassBreaksOperation]) {
    expect(
      () =>
        new Operation(
          getReviewOperationProps({
            maximumClassCount: 5,
            methods: ['natural-breaks'],
            naturalBreaksMode: 'approximate',
            naturalBreaksBinCount: 4
          })
        )
    ).toThrow(/maximumClassCount 5.*naturalBreaksBinCount 4/);
  }
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUClassBreaks({
      maximumClassCount: 4,
      methods: ['natural-breaks'],
      values: createTransientView(graph, 'provenance-values', 'float32', 12),
      parameters: createTransientView(graph, 'provenance-parameters', 'float32', 13),
      output: {
        breaks: createTransientView(graph, 'provenance-breaks', 'float32', 5),
        classCount: createTransientView(graph, 'provenance-count', 'uint32', 1)
      }
    });
    expect(contributor.provenance).toMatchObject({
      naturalBreaksMode: 'approximate',
      naturalBreaksAlgorithm: 'histogram-refinement'
    });
  } finally {
    device.destroy();
  }
});

it('checks CPU execution and GPU lowering admission for two input chunks (B3)', () => {
  const chunks = [Float32Array.of(1, 2), Float32Array.of(7, 8)];
  const parameters = getGPUClassBreaksParameterValues({method: 'maximum-breaks', classCount: 2}, 2);
  const inputs = {
    values: createReviewCPURows(chunks),
    parameters: createReviewCPURows([parameters])
  };
  for (const methods of [undefined, ['maximum-breaks']] as const) {
    const operation = new CPUClassBreaksOperation(getReviewOperationProps({methods}));
    const result = operation.execute({inputs}) as Record<string, Float32Array | Uint32Array>;
    // Sorted gaps are 1, 5, 1: the largest gap's midpoint is (2+7)/2=4.5.
    expect(Array.from(result['breaks'])).toEqual([1, 4.5, 8]);
    expect(Array.from(result['classCount'])).toEqual([2]);
    expect(Array.from(result['classCounts'])).toEqual([2, 2]);
  }
  const distinguishingChunks = [Float32Array.of(2, 3), Float32Array.of(4, 19)];
  for (const methods of [undefined, ['maximum-breaks']] as const) {
    const result = new CPUClassBreaksOperation(getReviewOperationProps({methods})).execute({
      inputs: {
        values: createReviewCPURows(distinguishingChunks),
        parameters: createReviewCPURows([parameters])
      }
    }) as Record<string, Float32Array | Uint32Array>;
    // Gaps 1,1,15 select midpoint (4+19)/2=11.5, not the range midpoint 10.5.
    expect(Array.from(result['breaks'])).toEqual([2, 11.5, 19]);
    expect(Array.from(result['classCount'])).toEqual([2]);
    expect(Array.from(result['classCounts'])).toEqual([3, 1]);
  }
  const fallback = computeClassBreaksOnCPU(
    Float32Array.of(2, 3, 4, 19),
    getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2),
    {maximumClassCount: 2, methods: ['equal-interval']}
  );
  expect(Array.from(fallback.breaks)).toEqual([2, 10.5, 19]);
  expect(Array.from(fallback.breaks)).not.toEqual([2, 11.5, 19]);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const views = chunks.map((chunk, index) =>
      createTransientView(graph, `review-chunk-${index}`, 'float32', chunk.length)
    );
    const gpu = new GPUClassBreaksOperation(getReviewOperationProps());
    expect(() =>
      GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(gpu, {
        resolveVector: () => ({data: views})
      } as unknown as Parameters<typeof GPU_CONTRIBUTOR_OPERATION_LOWERING.lower>[1])
    ).toThrow(/one packed chunk, not 2/);
  } finally {
    device.destroy();
  }
});

it('retains inherited-loss-only certification in the documented exact join (CMP-01)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = (length = 1) =>
      createTransientView(graph, `catalog-loss-${serial++}`, 'uint32', length);
    const keys = view(2);
    const overflow = view();
    const candidateOverflow = view();
    const nodes = new GPUKeyLookup({
      leftKeys: view(2),
      rightKeys: keys,
      rightTable: {keys, status: {overflow, candidateOverflow}},
      output: {status: {complete: view()}}
    }).getCommandNodes(graph);
    const certificate = nodes.find(node => node.id.endsWith('-certify'))!;
    for (const loss of [overflow, candidateOverflow]) {
      expect(
        certificate.resources!.some(resource => 'buffer' in resource && resource.buffer === loss)
      ).toBe(true);
    }
    // With final=0 and inherited=1, the OR is true, so the certificate must select 0.
    // Device execution of that literal control is retained in key-join/wave-three.spec.ts.
    const source = getGPUCommandNodeKernelSources(nodes).find(kernel =>
      kernel.id.endsWith('-certify')
    )!.source;
    expect(source).toContain(
      'upstreamOverflow[upstreamOverflowOffset] != 0u || upstreamCandidateOverflow[upstreamCandidateOverflowOffset] != 0u'
    );
    expect(source).toContain('complete[completeOffset] = select(1u, 0u, loss)');
    const page = readFileSync('docs/api/gpu-dataframe-analysis.md', 'utf8');
    expect(page).toContain(
      'Inherited-loss-only control: overflow 0, candidateOverflow 1 also gives complete 0'
    );
    expect(page).toContain("id: 'broadcast-join-complete', source: views.complete");
  } finally {
    device.destroy();
  }
});

it('uses public neutral packers for 65 CPU half-probabilities (TWN-14)', () => {
  const halves = Array.from({length: 65}, () => [1, 2] as const);
  const rationalProbabilities = getColumnQuantilesRationalProbabilityValues(halves);
  for (const ratios of [undefined, rationalProbabilities]) {
    const parameters = getColumnQuantilesParameterValues({
      quantiles: Array(65).fill(ratios ? NaN : 0.5)
    });
    const result = computeColumnQuantilesOnCPU(
      {values: Float32Array.of(0, 2), parameters, rationalProbabilities: ratios},
      65
    );
    // Every R-7 half-quantile on [0,2] is (0+2)/2=1, independent of the packer.
    expect(Array.from(result.quantiles)).toEqual(Array(65).fill(1));
  }
  expect(() => getGPUColumnQuantilesParameterValues({quantiles: Array(65).fill(0.5)})).toThrow(
    /quantileCount/
  );
  expect(() => getGPUColumnQuantilesRationalProbabilityValues(halves)).toThrow(/quantileCount/);
});

it('classifies a raster live prefix while preserving the source byte offset (CMP-05)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const allocation = createTransientView(graph, 'density-capacity', 'float32', 6);
    const source = graph.createDataView(allocation.buffer, {
      format: 'float32',
      byteOffset: 4,
      length: 5
    });
    const band = makeGPURasterBand(source, makeGPURasterMetadata({width: 2, height: 2}));
    const liveValues = band.storage.values;
    expect(liveValues.buffer).toBe(source.buffer);
    expect(liveValues.byteOffset).toBe(4);
    expect(liveValues.length).toBe(4);
    // The leading -999 is outside the source view; 1000 is spare capacity after the grid.
    const backing = Float32Array.of(-999, 2, 4, 6, 8, 1000);
    for (const [length, breaks, counts] of [
      [liveValues.length, [2, 5, 8], [2, 2]],
      [source.length, [2, 501, 1000], [4, 1]]
    ] as const) {
      const result = computeClassBreaksOnCPU(
        new Float32Array(backing.buffer, liveValues.byteOffset, length),
        getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2),
        {maximumClassCount: 2, methods: ['equal-interval']}
      );
      // Equal-interval inner edge is (minimum+maximum)/2; count each side by hand.
      expect(Array.from(result.breaks)).toEqual(breaks);
      expect(Array.from(result.classCounts)).toEqual(counts);
    }
  } finally {
    device.destroy();
  }
});

it('keeps empty and singleton point-domain live counts below allocation capacity (W5-08)', () => {
  for (const domain of [[], [7]]) {
    const controls = prepareGPUScalarScaleControls({scale: 'point', domain});
    expect(controls.minimumDomainCapacity).toBe(2);
    expect(controls.counts).toEqual({domainCount: domain.length, rangeCount: 2, parameterCount: 4});
    expect(controls.formats).toEqual({domain: 'float32', range: 'float32', parameters: 'float32'});
    expect(controls.parameters).toHaveLength(4);
    expect(controls.parameters[1]).toBe(domain.length);
    expect(Array.from(controls.domain)).toEqual(domain);
    expect(Array.from(controls.range)).toEqual([0, 1]);
  }
});

it('checks full numeric endpoint reconstruction before writing word windows (DPC-108)', () => {
  const target = new Uint32Array(8).fill(0xdeadbeef);
  for (const endpoint of [-0.125 + 2 ** -56, 2 ** 53, -(2 ** -26)]) {
    // The first value loses its residual in endpoint-floor(endpoint); the last carries
    // f32(1-2^-26) to 1 and must reject the changed numeric instant, even though 0 is safe.
    expect(() =>
      getGPUTimeWindowFilterWordParameterValues({start: endpoint, end: endpoint}, target)
    ).toThrow();
    expect(Array.from(target)).toEqual(Array(8).fill(0xdeadbeef));
  }
  const wide = getGPUTimeWindowFilterWordParameterValues({
    start: 9007199254740992n,
    end: 9007199254740992n
  });
  expect(wide).toHaveLength(8);
  expect(joinTimeWords(wide[0], wide[1])).toBe(9007199254740992n);
  expect(Array.from(wide.slice(0, 4))).toEqual([0, 2097152, 0, 2097152]);
  for (const [start, end] of [
    [-0.125, -0.125],
    [0.5, -0.125]
  ]) {
    const words = getGPUTimeWindowFilterWordParameterValues({start, end});
    expect(words).toHaveLength(8);
    const fractions = new Float32Array(words.buffer, words.byteOffset + 16, 2);
    expect(Number(joinTimeWords(words[0], words[1])) + fractions[0]).toBe(start);
    expect(Number(joinTimeWords(words[2], words[3])) + fractions[1]).toBe(end);
  }
});

it('canonicalizes signed zero in exact quantiles while raw ordering retains extrema bits (B4)', () => {
  const result = computeColumnQuantilesOnCPU(
    {
      values: Float32Array.of(-0),
      parameters: getColumnQuantilesParameterValues({quantiles: [0], interpolation: 'lower'})
    },
    1
  );
  expect(
    new Uint32Array(
      result.quantiles.buffer,
      result.quantiles.byteOffset,
      result.quantiles.length
    )[0]
  ).toBe(0);
  expect(getOrderedFloat32Key(-0)).toBe(0x7fffffff);
  expect(getOrderedFloat32Key(+0)).toBe(0x80000000);
});

it('prefers rational probabilities over conflicting float probabilities (S2)', () => {
  const input = {
    values: Float32Array.of(0, 8),
    parameters: getColumnQuantilesParameterValues({quantiles: [0.75]})
  };
  // R-7 interpolation on [0,8] is 8*p: float 3/4 -> 6, rational 1/4 -> 2.
  expect(Array.from(computeColumnQuantilesOnCPU(input, 1).quantiles)).toEqual([6]);
  expect(
    Array.from(
      computeColumnQuantilesOnCPU(
        {...input, rationalProbabilities: getColumnQuantilesRationalProbabilityValues([[1, 4]])},
        1
      ).quantiles
    )
  ).toEqual([2]);
});

it('matches all three natural-break limits to planner admission at and above each maximum (S3)', () => {
  const limits = getRouteOperator('GPUClassBreaks', 'dataframe-analysis').limits!;
  const routes = [
    {
      name: 'source rows (natural-breaks exact materialized)',
      naturalBreaksMode: 'exact',
      naturalBreaksExactStorage: 'materialized'
    },
    {
      name: 'source rows (natural-breaks exact recompute)',
      naturalBreaksMode: 'exact',
      naturalBreaksExactStorage: 'recompute'
    },
    {name: 'source rows (natural-breaks monotone)', naturalBreaksMode: 'monotone'}
  ] as const;
  for (const route of routes) {
    const maximum = limits.find(limit => limit.name === route.name)!.maximum;
    for (const rowCount of [maximum, maximum + 1]) {
      const shape = {
        maximumClassCount: 2,
        rowCount,
        naturalBreaksMode: route.naturalBreaksMode,
        naturalBreaksExactStorage:
          'naturalBreaksExactStorage' in route ? route.naturalBreaksExactStorage : undefined
      };
      expect(
        GPUClassBreaks.getExecutionPlan({...shape, methods: ['natural-breaks']}).feasible,
        `${route.name}: ${rowCount}`
      ).toBe(rowCount === maximum);
      expect(
        GPUClassBreaks.getExecutionPlan({...shape, methods: ['equal-interval']}).feasible,
        `${route.name} disabled: ${rowCount}`
      ).toBe(true);
    }
  }
});

function getDataframeGuideFence(marker: string): string {
  const page = readFileSync('docs/api/gpu-dataframe-analysis.md', 'utf8');
  const fence = [...page.matchAll(/```ts\n([\s\S]*?)```/g)].find(match =>
    match[1].includes(marker)
  );
  if (!fence) throw new Error(`Missing dataframe guide fence: ${marker}`);
  return fence[1];
}

// Execute the authored function, retaining the real public constructors and host helpers.
// Transpilation does not perform the strict declaration check reserved to the lead.
function getDataframeGuideFunction(
  name: string,
  overrides: Readonly<Record<string, unknown>> = {}
): (...arguments_: unknown[]) => unknown {
  const fence = getDataframeGuideFence(`function ${name}(`);
  const functionStart = fence.indexOf(
    `export ${name === 'fitEmpiricalSpaceTime' ? 'async ' : ''}function ${name}(`
  );
  const functionEnd =
    name === 'fitCompleteVariogram' ? fence.indexOf('\nconst fit =') : fence.length;
  const imports = fence.slice(
    0,
    fence.indexOf('declare const') >= 0 ? fence.indexOf('declare const') : functionStart
  );
  const code = imports + '\n' + fence.slice(functionStart, functionEnd);
  const exported: Record<string, unknown> = {};
  const dataframe = {
    GPUKeyLookup,
    GPUCubeLayoutTranspose,
    GPUSpaceTimeVariogram,
    fitVariogramModelWithDiagnostics,
    getGPUSpaceTimeVariogramBinLags,
    fitSpaceTimeVariogramModel,
    evaluateSpaceTimeVariogramModel,
    ...overrides
  };
  const dependencies: Readonly<Record<string, Readonly<Record<string, unknown>>>> = {
    '@luma-contributors/analysis/dataframe-analysis': dataframe,
    '@luma-contributors/analysis/gpu-spatial-analysis': {GPUOrdinaryLeastSquares},
    '@luma.gl/graph/gpu-core': {GPUGather, GPUMask},
    '@luma-contributors/analysis/analysis-core': {}
  };
  runInNewContext(
    transpileModule(code, {
      compilerOptions: {
        module: ModuleKind.CommonJS,
        target: ScriptTarget.ES2022
      }
    }).outputText,
    {
      exports: exported,
      require: (entry: string) => {
        if (!dependencies[entry]) throw new Error(`Unexpected guide import ${entry}`);
        return dependencies[entry];
      }
    }
  );
  const callback = exported[name];
  if (typeof callback !== 'function') throw new Error(`Guide must export ${name}`);
  return callback as (...arguments_: unknown[]) => unknown;
}

it('R1 B1 declares the displayed grouped-predictor composition for five observations', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const inheritedLoss of [false, true]) {
      const graph = new GPUCommandGraph(device);
      const declarations = vi.spyOn(graph, 'add');
      let serial = 0;
      const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
        createTransientView(graph, `displayed-${serial++}`, format, length);
      const fitOutput = (predictorCount: number): GPUOrdinaryLeastSquaresOutput => ({
        status: {code: view('uint32', 1)},
        coefficients: view('float32', predictorCount + 1),
        standardErrors: view('float32', predictorCount + 1),
        tStatistics: view('float32', predictorCount + 1),
        summary: view('float32', 22)
      });
      const table = {
        keys: view('uint32', 5),
        meansX: view('float32', 5),
        meansY: view('float32', 5),
        status: {
          overflow: view('uint32', 1),
          ...(inheritedLoss ? {candidateOverflow: view('uint32', 1)} : {})
        }
      };
      const views = {
        onePredictor: view('float32', 5),
        columnPlanes: view('float32', 10),
        predictors: view('float32', 10),
        rightRows: view('uint32', 5),
        matched: view('uint32', 5),
        complete: view('uint32', 1),
        zeroRows: view('uint32', 5),
        broadcastComplete: view('uint32', 5),
        eligible: view('uint32', 5),
        measureValidity: view('uint32', 5),
        fitMask: view('uint32', 5)
      };
      expect(() =>
        getDataframeGuideFunction('addGroupedFits')(
          graph,
          view('uint32', 5),
          table,
          view('float32', 5),
          views,
          {one: fitOutput(1), two: fitOutput(2)}
        )
      ).not.toThrow();
      // graph.add expands each real producer, including its node/resource admission.
      // Re-expanding on the same graph would allocate duplicate transient resource IDs.
      const declaredIds = declarations.mock.calls.map(([producer]) => {
        if (Array.isArray(producer) || !('id' in producer))
          throw new Error('expected a concrete command node from the grouped-fit composition');
        return producer.id;
      });
      expect(declaredIds).toEqual(
        expect.arrayContaining([
          'predictor-lookup',
          'complete-matches',
          'fit-validity',
          'one-predictor-fit',
          'two-predictor-fit'
        ])
      );
      const gathers = declarations.mock.calls
        .map(([producer]) => producer)
        .filter(producer => producer instanceof GPUGather);
      expect(gathers).toHaveLength(4); // three predictor gathers plus completeness broadcast
      declarations.mockRestore();
    }
  } finally {
    device.destroy();
  }
});

it('R1 B2 accepts ordinary producer status without inventing candidate loss ports', () => {
  for (const marker of ["id: 'event-count-lookup'", 'export function addGroupedFits(']) {
    expect(getDataframeGuideFence(marker)).toContain("candidateOverflow?: GraphDataView<'uint32'>");
  }
});

it('R1 B3 B4 documents the landed inequality boundary, gap reductions and admission', () => {
  const page = readFileSync('docs/api/gpu-dataframe-analysis.md', 'utf8');
  // Mean=4.5; e=.5 EDE=(.5*sqrt(9))^2=2.25 -> A=.5. A=1 at e>=1.
  for (const [epsilon, expected] of [
    [0.5, 0.5],
    [1, 1],
    [3, 1]
  ]) {
    const result = computeInequalityOnCPU({
      values: Float32Array.of(0, 9, 9),
      weights: Float32Array.of(2, 1, 1),
      zoneIds: Uint32Array.of(0, 0, 0),
      zoneCount: 1,
      parameters: getGPUInequalityParameterValues({epsilon})
    });
    expect(Array.from(result.atkinson)).toEqual([expected]);
  }
  const props: InequalityOperationProps = {
    values: 'values',
    zoneIds: 'zones',
    parameters: 'parameters',
    zoneCount: 1,
    output: {gini: 'gini'}
  };
  for (const Operation of [CPUInequalityOperation, GPUInequalityOperation]) {
    for (const invalid of [
      {...props, output: {gini: ''}},
      {...props, weights: ''},
      {...props, output: {gini: 'same', counts: 'same'}},
      {...props, output: {gini: 'values'}},
      {...props, output: {gini: 'gini', weightedGini: 'weighted'}}
    ])
      expect(() => new Operation(invalid)).toThrow();
    // Deliberate JavaScript empty selection, built without an invalid TypeScript cast.
    expect(() => Reflect.construct(Operation, [{...props, output: {}}])).toThrow();
  }
  expect
    .soft(
      page.includes('current CPU and both GPU implementations still return NaN'),
      'B3 stale boundary caveat'
    )
    .toBe(false);
  expect
    .soft(
      page.includes('parallel/reference GPU writers still use Lorenz-area complements'),
      'B4 stale Gini caveat'
    )
    .toBe(false);
  expect
    .soft(
      page.includes('but currently needs the same explicit empty-selection'),
      'B4 stale admission caveat'
    )
    .toBe(false);
  expect
    .soft(page.includes('Look at the trapezoid sum 1 - sum'), 'B4 stale kernel caption')
    .toBe(false);
  const device = createNullWebGPUDevice();
  try {
    for (const mode of ['parallel', 'reference'] as const) {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
        createTransientView(graph, `gap-${serial++}`, format, length);
      const nodes = new GPUInequality({
        mode,
        zoneCount: 1,
        values: view('float32', 3),
        zoneIds: view('uint32', 3),
        parameters: view('float32', 4),
        output: {gini: view('float32', 1), globalSummary: view('float32', 8)}
      }).getCommandNodes(graph);
      const ids = nodes.map(node => node.id);
      expect(ids.indexOf('inequality-zone-gini-publish')).toBeGreaterThan(-1);
      expect(ids.indexOf('inequality-zone-gini-publish')).toBeLessThan(
        ids.indexOf('inequality-publish-indices')
      );
      expect(ids.at(-1)).toBe('inequality-global-gini-publish');
    }
  } finally {
    device.destroy();
  }
});

it('R1 B6 names the compiled approximate-natural route for the chunk budget example', () => {
  const shape = {rowCount: 660, chunkRowCounts: Array(660).fill(1), maximumClassCount: 18};
  const equal = GPUClassBreaks.getExecutionPlan({...shape, methods: ['equal-interval']});
  const natural = GPUClassBreaks.getExecutionPlan({
    ...shape,
    methods: ['natural-breaks'],
    naturalBreaksMode: 'approximate'
  });
  expect(equal.estimatedCommandNodeCount).toBe(5384);
  expect(equal.feasible).toBe(true);
  expect(natural.estimatedCommandNodeCount).toBe(32771);
  expect(natural.feasible).toBe(false);
  const page = readFileSync('docs/api/gpu-dataframe-analysis.md', 'utf8');
  expect(page).toContain('with 18 classes and compiled approximate natural breaks');
  expect(page).toContain('Node budget for compiled approximate natural breaks');
});

it('R1 B7 rejects nonfinite positive-count empirical bins before fitting', () => {
  const fit = getDataframeGuideFunction('fitCompleteVariogram');
  const distances = Float32Array.from({length: 96}, (_, index) => index + 1);
  const semivariances = Float32Array.from(distances, distance => Math.min(distance, 10));
  const counts = new Uint32Array(96).fill(1);
  for (const nonfinite of [NaN, Infinity]) {
    const invalidDistances = distances.slice();
    invalidDistances[2] = nonfinite;
    const invalidSemivariances = semivariances.slice();
    invalidSemivariances[2] = nonfinite;
    counts[2] = 100;
    expect(() => fit(distances, invalidSemivariances, counts, Uint32Array.of(0))).toThrow(/bin 2/);
    expect(() => fit(invalidDistances, semivariances, counts, Uint32Array.of(0))).toThrow(/bin 2/);
    counts[2] = 0;
    expect(() =>
      fit(invalidDistances, invalidSemivariances, counts, Uint32Array.of(0))
    ).not.toThrow();
  }
});

it('R1 B8 takes empirical bin counts from props despite extra limit-object fields', async () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `lag-${serial++}`, format, length);
    const binLags = vi.fn(getGPUSpaceTimeVariogramBinLags);
    const fit = getDataframeGuideFunction('fitEmpiricalSpaceTime', {
      getGPUSpaceTimeVariogramBinLags: binLags
    });
    const limits = {maximumDistance: 60, maximumTimeLag: 12, lagCount: 3, temporalLagCount: 1};
    await fit(
      graph,
      {
        positions: view('float32x2', 3),
        values: view('float32', 3),
        times: view('float32', 3),
        parameters: view('float32', 8),
        gridSize: [2, 2],
        lagCount: 1,
        temporalLagCount: 3,
        output: {
          semivariances: view('float32', 8),
          pairCounts: view('uint32', 8),
          status: view('uint32', 8)
        }
      },
      limits,
      async () => ({
        semivariances: Float32Array.of(0, 1, 2, 3, 4, 5, 6, 7),
        pairCounts: new Uint32Array(8).fill(1),
        status: new Uint32Array(8)
      })
    );
    expect(binLags).toHaveBeenCalledWith({
      lagCount: 1,
      temporalLagCount: 3,
      maximumDistance: 60,
      maximumTimeLag: 12
    });
    const lags = binLags.mock.results[0].value;
    // Spatial-major bin 2 is spatial class 0, temporal class 2: (2-.5)*12/3=6.
    expect([lags.distances[2], lags.timeLags[2]]).toEqual([0, 6]);
  } finally {
    device.destroy();
  }
});

it('R1 S1 S3 awaits CPU execution and scopes logarithmic work to sorted range search', () => {
  expect
    .soft(
      getDataframeGuideFence('const floatParameters =').includes(
        'const result = await operation.execute('
      ),
      'S1 awaited public execution'
    )
    .toBe(true);
  const page = readFileSync('docs/api/gpu-dataframe-analysis.md', 'utf8');
  expect
    .soft(page.includes('Finding the range takes O(log n)'), 'S3 range-search complexity')
    .toBe(true);
  expect
    .soft(
      page.includes('one invocation per `output.rows` capacity slot'),
      'S3 capacity dispatch cost'
    )
    .toBe(true);
});
