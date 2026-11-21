// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUCellCompaction,
  GPUCellSetOutline
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getLensScenario} from '../../src/computation-lenses';
import {declareLensScenario} from './scenario-declaration.test-utils';
import {createNullWebGPUDevice} from '../../../analysis/src/testing/gpu-contributor-test-utils';
import {runExample as runOutline} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/cell-set-outline.example';
import {KEPLER_ANALYSIS_OPERATORS} from '../../src/gpu-spatial-analysis/computation-pack/operators/kepler-analysis';
import {STATISTICS_CLUSTERING_OPERATORS} from '../../src/gpu-spatial-analysis/computation-pack/operators/statistics-clustering';
import {DYNAMICS_OPERATORS} from '../../src/gpu-spatial-analysis/computation-pack/operators/dynamics';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/cell-set-outline.lens';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/voronoi-diagram.lens';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/urban-ecology.lens';

test('W6 outline example accepts a counted source without redundant bare cell bindings', async () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph<undefined>(device);
  const source = {
    cells: createTransientView(graph, 'cells', 'uint32x2', 1),
    status: {
      count: createTransientView(graph, 'count', 'uint32', 1),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1)
    }
  };
  const added = vi.spyOn(graph, 'add');
  try {
    await runOutline(graph, {source, segmentCapacity: 6, ringCapacity: 1});
    const outline = added.mock.calls
      .map(([node]) => node)
      .find(node => node instanceof GPUCellSetOutline);
    if (!(outline instanceof GPUCellSetOutline)) throw new Error('Missing outline');
    expect(outline.props.cells).toBe(source.cells);
    expect(outline.props.count).toBe(source.status.count);
  } finally {
    added.mockRestore();
    device.destroy();
  }
});

test('W6 declaration harness refuses direct submission as well as encoding', async () => {
  const scenario = getLensScenario('gpu-spatial-analysis.voronoi-diagram')!;
  await expect(
    declareLensScenario({
      ...scenario,
      load: async () => ({
        run: async device => {
          device.submit();
          throw new Error('Direct submission was accepted');
        }
      })
    })
  ).rejects.toThrow('Null-device declaration must never submit');
});

test('W6 counted cover reaches the actual outline scenario through loss-aware compaction', async () => {
  const added = vi.spyOn(GPUCommandGraph.prototype, 'add');
  try {
    await declareLensScenario(getLensScenario('gpu-spatial-analysis.cell-set-outline')!);
    const contributors = added.mock.calls.map(([contributor]) => contributor);
    const compaction = contributors.find(contributor => contributor instanceof GPUCellCompaction);
    expect(compaction).toBeInstanceOf(GPUCellCompaction);
    if (!(compaction instanceof GPUCellCompaction)) throw new Error('Missing compaction');
    expect(compaction.props.source?.status.complete).toBeDefined();
    expect(compaction.props.source?.status.candidateOverflow).toBeDefined();
    expect(compaction.props.operation).toEqual({
      type: 'compact',
      resolution: 8,
      minimumResolution: 8
    });
    const outline = contributors.find(contributor => contributor instanceof GPUCellSetOutline);
    if (!(outline instanceof GPUCellSetOutline)) throw new Error('Missing outline');
    expect(outline.props.source?.cells).toBe(compaction.props.output.cells);
    expect(outline.props.source?.status.candidateOverflow).toBe(
      compaction.props.output.status.candidateOverflow
    );
  } finally {
    added.mockRestore();
  }
});

test('W6 Voronoi scenario captures the existing final convergence word', async () => {
  const declaration = await declareLensScenario(
    getLensScenario('gpu-spatial-analysis.voronoi-diagram')!
  );
  expect(declaration.captureNames).toContain('converged');
});

test('W6 synthetic shared wall supplies an aggregate loss certificate without adding a capture', async () => {
  const declaration = await declareLensScenario(
    getLensScenario('gpu-spatial-analysis.building-adjacency-recipe')!
  );
  expect(
    declaration.record.graphs
      .flatMap(graph => graph.nodes)
      .some(node => node.reads.some(resource => resource.includes('urban-ecology-wall-loss')))
  ).toBe(true);
  expect(declaration.captureNames).not.toContain('urban-ecology-wall-loss');
});

test('W6 quantile inference metadata exposes caller-selected alternatives', () => {
  const entry = KEPLER_ANALYSIS_OPERATORS['GPUQuantileLISA'];
  expect(entry.choices.find(choice => choice.name === 'alternative')).toMatchObject({
    alternatives: ['greater', 'two-sided', 'geoda-legacy']
  });
  expect(entry.metadata).toMatchObject({
    permutationTail: 'caller-selected',
    legacyPermutationTail: 'strict-complement',
    legacyPermutationFormula: '(min(upper, P-upper)+1)/(P+1)',
    inclusiveTieRule: 'greater and two-sided return 1 when all draws tie'
  });
});

test('W6 clustering formulas describe the selected cluster origin and link the distance difference', () => {
  const kmeans = STATISTICS_CLUSTERING_OPERATORS['GPUSpatialKMeans'].formula;
  if ('formula' in kmeans) throw new Error('Expected K-means formula');
  expect(kmeans.ascii).toContain("origin of each point's cluster");
  expect(kmeans.ascii).not.toContain('first finite point');
  const similar = STATISTICS_CLUSTERING_OPERATORS['GPUSimilarLocations'].formula;
  if ('formula' in similar) throw new Error('Expected similarity formula');
  // z itself is conceptual on the z-score route; delta is z_ic - r_c, not z_ic.
  expect(similar.symbols.find(symbol => symbol.id === 'standardized')?.binding).toBeUndefined();
});

test('W6 transition formula bindings select the actual Pearson terms and publication', () => {
  const formula = DYNAMICS_OPERATORS['GPUTransitionDiagnostics'].formula;
  if ('formula' in formula) throw new Error('Expected transition formula');
  expect(formula.symbols.find(symbol => symbol.id === 'q')?.binding).toBe('wgsl:statistic');
  expect(formula.symbols.find(symbol => symbol.id === 'observed')?.binding).toBe(
    'wgsl:observedInteger'
  );
  expect(formula.symbols.find(symbol => symbol.id === 'pooled')?.binding).toBe('wgsl:pooledCount');
});
