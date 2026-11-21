// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  getGPUAnalysisOperator,
  queryGPUAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';
import {
  GPU_SPATIAL_ANALYSIS_OPERATOR_DESCRIPTORS,
  getGPUSpatialAnalysisOperator,
  queryGPUSpatialAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-operators';
import {
  projectSpatialOperator,
  resolveOperatorDescriptor
} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/resolve-operator-descriptor';
import {GPUSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {GPUSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transpose';
import {GPUChunkedSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transform';
import {GPUSegmentSnap} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap';
import {AUDITED_EMPTY_OPERATOR_STATUSES} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/operator-status-audits';
import type {GPUSpatialAnalysisStatusKind} from '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-ontology';

const FORWARDED_STATUS_KINDS: GPUSpatialAnalysisStatusKind[] = [
  'count',
  'requiredCount',
  'overflow',
  'candidateOverflow',
  'uncertainCount',
  'invalidCount',
  'converged',
  'iterationLimitReached',
  'iterationCount',
  'approximation'
];

function assertStatusQueries(
  exportName: string,
  expected: readonly GPUSpatialAnalysisStatusKind[]
) {
  expect(getGPUAnalysisOperator(exportName)?.status).toEqual(expected);
  expect(getGPUSpatialAnalysisOperator(exportName)?.status).toEqual(expected);
  for (const query of [queryGPUAnalysisOperators, queryGPUSpatialAnalysisOperators])
    for (const status of FORWARDED_STATUS_KINDS)
      expect(
        query({status}).some(record => record.exportName === exportName),
        `${exportName}: ${status}`
      ).toBe(expected.includes(status));
}

it('persists chunked-transpose destination versus matrix status scopes', () => {
  assertStatusQueries('GPUChunkedSpatialWeightsTranspose', FORWARDED_STATUS_KINDS);
  for (const get of [getGPUAnalysisOperator, getGPUSpatialAnalysisOperator]) {
    const limitation = get('GPUChunkedSpatialWeightsTranspose')?.limitation;
    expect(limitation).toMatch(/outputWeights\.chunks\[i\]\.weights\.status/);
    expect(limitation).toMatch(/count and requiredCount/);
    expect(limitation).toMatch(/outputWeights\.status/);
    expect(limitation).toMatch(/forwards/);
  }
});

it('persists both validators computed invalid count and forwarded evidence', () => {
  for (const name of ['GPUSpatialWeightsValidation', 'GPUChunkedSpatialWeightsValidation']) {
    assertStatusQueries(name, FORWARDED_STATUS_KINDS);
    for (const get of [getGPUAnalysisOperator, getGPUSpatialAnalysisOperator]) {
      expect(get(name)?.limitation).toMatch(/computes invalidCount/);
      expect(get(name)?.limitation).toMatch(/preserves supplied/);
      expect(get(name)?.limitation).toMatch(/outputWeights/);
    }
  }
});

it('persists the space-time hot-spots supplied weights stage status union', () => {
  assertStatusQueries('addSpaceTimeHotSpotsRecipe', FORWARDED_STATUS_KINDS);
  for (const get of [getGPUAnalysisOperator, getGPUSpatialAnalysisOperator]) {
    expect(get('addSpaceTimeHotSpotsRecipe')?.limitation).toMatch(/cells\.weights\.status/);
    expect(get('addSpaceTimeHotSpotsRecipe')?.limitation).toMatch(/weights stage/);
  }
});

it('persists regression scalar solver-code versus GWR row-code guidance', () => {
  for (const get of [getGPUAnalysisOperator, getGPUSpatialAnalysisOperator]) {
    const limitation = get('addSpatialRegressionRecipe')?.limitation;
    expect(limitation).toMatch(
      /ordinary least squares and spatial diagnostics stages publish solver outcome in `stage\.code`/
    );
    expect(limitation).toMatch(
      /optional geographically weighted regression stage publishes per-observation `rowCodes`/
    );
    expect(limitation).not.toMatch(/alias|must not be summed/i);
  }
});

it('B1 restricts the historical projection to complete family evidence', () => {
  const network = getGPUAnalysisOperator({
    package: '@luma-contributors/analysis/gpu-network',
    exportName: 'addVisibilityGraphRecipe'
  })!;
  expect(() => {
    // @ts-expect-error An all-entry descriptor need not have historical family evidence.
    projectSpatialOperator(network);
  }).toThrow(/historical.*evidence/);
  const descriptor = GPU_SPATIAL_ANALYSIS_OPERATOR_DESCRIPTORS[0];
  const projected = projectSpatialOperator(resolveOperatorDescriptor(descriptor));
  expect(projected.evidence).toEqual(descriptor.evidence);
  expect(Number.isFinite(projected.evidence.representativeSourceRows)).toBe(true);
  expect(projected.evidence.transientMemory).toBeDefined();
  expect(projected.evidence.dispatch).toBeDefined();
  expect(projected.evidence.correctnessOracle).toBeTruthy();
});

it('B2 discovers the scalar shape count and nested cell-table status', () => {
  // Publisher contracts: shape-generator offsets writes one vertexCount. CellTable publish
  // writes count/overflow and optional requiredCount; rollup propagates source loss separately.
  const cases: [GPUSpatialAnalysisStatusKind, string[], string[]][] = [
    ['vertexCount', ['GPUShapeGenerator'], ['GPUShapeGenerator']],
    [
      'count',
      ['GPUCellAggregation', 'GPUCellRollup', 'GPUCellPyramid'],
      ['GPUCellAggregation', 'GPUCellRollup', 'GPUCellPyramid']
    ],
    [
      'requiredCount',
      ['GPUCellAggregation', 'GPUCellRollup', 'GPUCellPyramid'],
      ['GPUCellAggregation', 'GPUCellRollup', 'GPUCellPyramid']
    ],
    [
      'overflow',
      ['GPUCellAggregation', 'GPUCellRollup', 'GPUCellPyramid'],
      ['GPUCellAggregation', 'GPUCellRollup', 'GPUCellPyramid']
    ],
    [
      'candidateOverflow',
      ['GPUCellAggregation', 'GPUCellRollup', 'GPUCellPyramid'],
      ['GPUCellRollup', 'GPUCellPyramid']
    ]
  ];
  for (const [status, names, expected] of cases) {
    expect(
      queryGPUAnalysisOperators({status})
        .map(record => record.exportName)
        .filter(name => names.includes(name))
        .sort(),
      status
    ).toEqual([...expected].sort());
    expect(
      queryGPUSpatialAnalysisOperators({status})
        .map(record => record.exportName)
        .filter(name => names.includes(name))
        .sort(),
      status
    ).toEqual([...expected].sort());
  }
});

it('B3 discovers loss views forwarded by whole and chunked weights transforms', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'catalog-forwarded-loss'});
  let sequence = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `forwarded-${sequence++}`, format, length);
  const overflow = view('uint32', 1);
  const count = view('uint32', 1);
  const requiredCount = view('uint32', 1);
  const input = {
    offsets: view('uint32', 3),
    neighbors: view('uint32', 2),
    weights: view('float32', 2),
    status: {count, overflow, requiredCount}
  };
  // Both cardinalities must actually be supplied for their omission checks to discriminate.
  expect(Object.keys(input.status).sort()).toEqual(['count', 'overflow', 'requiredCount']);
  const transformed = new GPUSpatialWeightsTransform({operation: 'binary', weights: input});
  const transposed = new GPUSpatialWeightsTranspose({
    weights: input,
    output: {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 2),
      weights: view('float32', 2)
    }
  });
  const chunked = new GPUChunkedSpatialWeightsTransform({
    operation: 'binary',
    weights: {
      rowCount: 2,
      status: {count, overflow, requiredCount},
      chunks: [{rowStart: 0, rowCount: 2, weights: input}]
    }
  });
  expect(transformed.outputWeights.status?.overflow).toBe(overflow);
  expect(transposed.outputWeights.status?.overflow).toBe(overflow);
  expect(chunked.outputWeights.status?.overflow).toBe(overflow);
  expect(chunked.outputWeights.status?.count).toBe(count);
  expect(chunked.outputWeights.status?.requiredCount).toBe(requiredCount);
  expect(chunked.outputWeights.chunks[0].weights.status?.overflow).toBe(overflow);
  expect(transformed.outputWeights.status?.requiredCount).toBe(requiredCount);
  expect(transformed.outputWeights.status?.count).toBe(count);
  expect(count).not.toBe(requiredCount);
  expect(chunked.outputWeights.chunks[0].weights.status?.count).toBe(count);
  expect(chunked.outputWeights.chunks[0].weights.status?.requiredCount).toBe(requiredCount);
  expect(transposed.outputWeights.status?.count).toBeUndefined();
  expect(transposed.outputWeights.status?.requiredCount).toBeUndefined();
  const names = [
    'GPUSpatialWeightsTransform',
    'GPUSpatialWeightsTranspose',
    'GPUChunkedSpatialWeightsTransform'
  ];
  for (const query of [queryGPUAnalysisOperators, queryGPUSpatialAnalysisOperators]) {
    expect(
      query({status: 'overflow'})
        .map(record => record.exportName)
        .filter(name => names.includes(name))
        .sort()
    ).toEqual([...names].sort());
  }
  const transformStatus: GPUSpatialAnalysisStatusKind[] = [
    'count',
    'requiredCount',
    'overflow',
    'candidateOverflow',
    'uncertainCount',
    'invalidCount',
    'converged',
    'iterationLimitReached',
    'iterationCount',
    'approximation'
  ];
  for (const name of names) {
    const expected =
      name === 'GPUSpatialWeightsTranspose' ? transformStatus.slice(2) : transformStatus;
    expect(getGPUAnalysisOperator(name)?.status).toEqual(expected);
    expect(getGPUSpatialAnalysisOperator(name)?.status).toEqual(expected);
    assertStatusQueries(name, expected);
    expect(getGPUAnalysisOperator(name)?.limitation).toMatch(/forward|preserv/i);
    if (name === 'GPUSpatialWeightsTranspose') {
      for (const get of [getGPUAnalysisOperator, getGPUSpatialAnalysisOperator]) {
        expect(get(name)?.limitation).toMatch(/omits count and requiredCount/);
        expect(get(name)?.limitation).toMatch(/rectangular/);
        expect(get(name)?.limitation).toMatch(/input loss/);
      }
    }
  }
  // Packed transpose can drop out-of-range columns; input cardinalities cannot describe it.
  // Cell-level selection publishes its selected level count and is the positive control.
  for (const query of [queryGPUAnalysisOperators, queryGPUSpatialAnalysisOperators]) {
    const members = query({status: 'requiredCount'}).map(record => record.exportName);
    expect(members.filter(name => names.includes(name)).sort()).toEqual([
      'GPUChunkedSpatialWeightsTransform',
      'GPUSpatialWeightsTransform'
    ]);
    expect(members).toContain('GPUCellLevelSelection');
  }
  assertStatusQueries('GPUCellLevelSelection', ['count', 'requiredCount']);
});

it('B4 excludes per-line segment-snap diagnostics from scalar queries', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'catalog-per-line'});
  const props = {
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    lineOffsets: createTransientView(graph, 'lines', 'uint32', 3),
    referencePositions: createTransientView(graph, 'references', 'float32x2', 2),
    tolerance: 0.5,
    maximumVerticesPerLine: 4,
    output: {
      positions: createTransientView(graph, 'result', 'float32x2', 8),
      sourceEdges: createTransientView(graph, 'edges', 'uint32', 8),
      counts: createTransientView(graph, 'counts', 'uint32x4', 2)
    }
  };
  expect(
    () =>
      new GPUSegmentSnap({
        ...props,
        output: {
          ...props.output,
          status: {
            candidateOverflow: createTransientView(graph, 'scalar-status', 'uint32', 1)
          }
        }
      })
  ).toThrow(/candidateOverflow.length must equal lineCount/);
  expect(
    () =>
      new GPUSegmentSnap({
        ...props,
        output: {
          ...props.output,
          status: {
            candidateOverflow: createTransientView(graph, 'per-line-status', 'uint32', 2)
          }
        }
      })
  ).not.toThrow();

  expect(getGPUAnalysisOperator('GPUSegmentSnap')?.status).toEqual([]);
  expect(getGPUSpatialAnalysisOperator('GPUSegmentSnap')?.status).toEqual([]);
  expect(
    queryGPUAnalysisOperators({status: 'candidateOverflow'}).map(record => record.exportName)
  ).not.toContain('GPUSegmentSnap');
  expect(
    queryGPUSpatialAnalysisOperators({status: 'candidateOverflow'}).map(record => record.exportName)
  ).not.toContain('GPUSegmentSnap');
  expect(
    AUDITED_EMPTY_OPERATOR_STATUSES[
      '@luma-contributors/analysis/gpu-spatial-analysis:GPUSegmentSnap'
    ]
  ).toMatch(/per.line/);
});

it('S1 pins affirmative empty audits against specific publisher contracts', () => {
  // GPUGridGenerator writes fixed-size positions/centers/corners and per-cell masks only.
  // GPULocalFrameProjection writes projected positions and per-position validity only.
  for (const name of ['GPUGridGenerator', 'GPULocalFrameProjection']) {
    expect(getGPUAnalysisOperator(name)?.status, name).toEqual([]);
    expect(
      AUDITED_EMPTY_OPERATOR_STATUSES[`@luma-contributors/analysis/gpu-spatial-analysis:${name}`],
      name
    ).toMatch(/\.ts/);
  }
});

it('earlier-fix note keeps regression excluded-row counts distinct from solver codes', () => {
  const recipe = getGPUAnalysisOperator('addSpatialRegressionRecipe');
  expect(recipe?.limitation).toMatch(
    /solver outcome in `stage\.code` and excluded rows in `stage\.status\.invalidCount`/
  );
  expect(recipe?.limitation).not.toMatch(/alias|must not be summed/i);
  expect(getGPUSpatialAnalysisOperator('addSpatialRegressionRecipe')?.limitation).toBe(
    recipe?.limitation
  );
});

it('B2 publisher re-audit discovers the legacy emerging-hot-spots population count', () => {
  // emerging-hot-spots publish writes output.includedCount or its legacy output.count alias;
  // either spelling exposes one exact uint32 population count, separate from per-cell counts.
  expect(getGPUAnalysisOperator('GPUEmergingHotSpots')?.status).toEqual(['count']);
  expect(getGPUSpatialAnalysisOperator('GPUEmergingHotSpots')?.status).toEqual(['count']);
  expect(queryGPUAnalysisOperators({status: 'count'}).map(record => record.exportName)).toContain(
    'GPUEmergingHotSpots'
  );
});
