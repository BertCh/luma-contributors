// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUAZPRegionsParameterValues,
  getGPUMaxPRegionsParameterValues,
  getGPUSkaterRegionsParameterValues,
  getGPUWardRegionsParameterValues,
  GPU_AZP_REGIONS_PARAMETER_LENGTH,
  GPU_MAX_P_REGIONS_PARAMETER_LENGTH,
  GPU_MAX_P_STATUS_LAYOUT,
  GPU_REGION_PARTITION_EVALUATION_LAYOUT,
  GPU_SKATER_REGIONS_PARAMETER_LENGTH,
  GPU_WARD_REGIONS_PARAMETER_LENGTH,
  GPUAZPRegions,
  GPUMaxPRegions,
  GPURegionPartitionEvaluation,
  GPUWardRegions,
  GPUSkaterRegions,
  GPUSpatialWeightsMinimumSpanningTree
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import referenceCases from './skater-reference.json';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  computePseudoF,
  selectMaxPStart,
  standardizeColumns,
  type RegionalizationCase
} from './spatial-regionalization-oracle';

let serial = 0;

function setup(rows: number, capacity: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'regionalization-nodes'});
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const weights = {
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', capacity),
    weights: view('float32', capacity)
  };
  return {graph, view, weights};
}

for (const rows of [2, 3, 4094])
  it(`final3 SKATER includes loss passes in its ${rows}-row admission bound`, () => {
    const {graph, view, weights} = setup(rows, 2);
    const create = () =>
      new GPUSkaterRegions({
        weights: {
          ...weights,
          status: {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)}
        },
        values: view('float32', rows),
        treeEdgeFlags: view('uint32', 2),
        componentLabels: view('uint32', rows),
        parameters: view('uint32', 2),
        maximumRegionCount: rows,
        output: {
          labels: view('uint32', rows),
          regionCount: view('uint32', 2),
          cutEdges: view('uint32', rows),
          cutGains: view('float32', rows)
        }
      });
    try {
      if (rows === 4094) expect(create).toThrow(/estimated command node count.*32768/);
      else {
        const contributor = create();
        const nodes = contributor.getCommandNodes(graph);
        expect(nodes).toHaveLength(rows === 2 ? 26 : 35);
        expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      }
    } finally {
      graph.device.destroy();
    }
  });

it('final3 SKATER suppression dispatches only logical output words', () => {
  const {graph, view, weights} = setup(2, 2);
  recordWGSLKernelNodes(graph);
  try {
    const contributor = new GPUSkaterRegions({
      weights: {...weights, status: {candidateOverflow: view('uint32', 1)}},
      values: view('float32', 2),
      treeEdgeFlags: view('uint32', 2),
      componentLabels: view('uint32', 2),
      parameters: view('uint32', 2),
      maximumRegionCount: 2,
      output: {
        labels: view('uint32', 2),
        regionCount: view('uint32', 2),
        cutEdges: view('uint32', 2),
        cutGains: view('float32', 2)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    for (const name of ['region-count', 'cut-edges', 'cut-gains']) {
      const node = nodes.find(node => node.id.endsWith(`-incomplete-${name}`))!;
      expect(getWGSLKernelNodeRecord(graph, node.id)?.invocationCount, name).toBe(1);
    }
  } finally {
    graph.device.destroy();
  }
});

it('bounds SKATER construction by attainable cuts while keeping the requested log layout', () => {
  const {graph, view, weights} = setup(2, 2);
  const contributor = new GPUSkaterRegions({
    weights,
    values: view('float32', 2),
    treeEdgeFlags: view('uint32', 2),
    componentLabels: view('uint32', 2),
    parameters: view('uint32', 2),
    maximumRegionCount: 100_000,
    output: {
      labels: view('uint32', 2),
      cutEdges: view('uint32', 99_999),
      cutGains: view('float32', 99_999)
    }
  });
  expect(contributor.estimatedCommandNodeCount).toBeLessThan(100);
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.filter(node => node.id.endsWith('-candidates'))).toHaveLength(1);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  const source = getGPUCommandNodeKernelSources(nodes)
    .map(kernel => kernel.source)
    .join('\n');
  expect(source).toContain('index < STEP_COUNT');
  expect(source).toContain('cutEdgesOut[cutEdgesOutOffset + index] = NONE');
});

it('rejects regionalizer graphs above the shared budget at construction', () => {
  const {view, weights} = setup(100_000, 2);
  const values = view('float32', 100_000);
  expect(
    () =>
      new GPUSkaterRegions({
        id: 'skater-budget',
        weights,
        values,
        treeEdgeFlags: view('uint32', 2),
        componentLabels: view('uint32', 100_000),
        parameters: view('uint32', 2),
        maximumRegionCount: 100_000,
        output: {labels: view('uint32', 100_000)}
      })
  ).toThrow(/skater-budget.*estimated command node count/);
  expect(
    () =>
      new GPUWardRegions({
        id: 'ward-budget',
        weights,
        values,
        targetRegionCount: 1,
        maximumIterations: 99_999,
        parameters: view('uint32', 1),
        output: {
          labels: view('uint32', 100_000),
          objective: view('float32', 1),
          status: view('uint32', 4)
        }
      })
  ).toThrow(/ward-budget.*estimated command node count/);
});

it('admits both Ward A/B schedules within the shared command-node budget', () => {
  for (const [rows, mergeSchedule, maximumIterations, admitted] of [
    [1024, 'sequential', 1008, true],
    [1024, 'mutual-nearest', 256, true],
    [1024, 'mutual-nearest', 1008, true],
    [8192, 'mutual-nearest', 1008, false],
    [4096, 'sequential', 4080, false]
  ] as const) {
    const {graph, view, weights} = setup(rows, 4 * rows);
    const createContributor = () =>
      new GPUWardRegions({
        weights,
        values: view('float32', 2 * rows),
        columnCount: 2,
        standardize: false,
        targetRegionCount: 16,
        parameters: view('uint32', 1),
        output: {
          labels: view('uint32', rows),
          objective: view('float32', 1),
          status: view('uint32', 5)
        },
        mergeSchedule,
        maximumIterations
      });
    if (admitted) {
      const contributor = createContributor();
      expect(contributor.estimatedCommandNodeCount).toBeLessThanOrEqual(32768);
      if (mergeSchedule === 'mutual-nearest' && maximumIterations === 1008) {
        expect(contributor.getCommandNodes(graph).length).toBe(18_160);
      }
    } else {
      // Thirteen nodes per sequential round already require 53,040 nodes before the prologue.
      expect(createContributor).toThrow(/estimated command node count.*32768/);
    }
  }
});

it('keeps Ward and AZP lowering within the admitted budget after caller mutation', () => {
  for (const kind of ['ward', 'azp'] as const) {
    const {graph, view, weights} = setup(16, 32);
    const props = {
      weights,
      values: view('float32', 16),
      parameters: view('uint32', 1),
      maximumIterations: 1,
      targetRegionCount: 2,
      mergeSchedule: 'sequential' as 'sequential' | 'mutual-nearest',
      initialLabels: view('uint32', 16),
      regionCapacity: 2,
      output: {labels: view('uint32', 16), objective: view('float32', 1), status: view('uint32', 5)}
    };
    const baseline =
      kind === 'ward'
        ? new GPUWardRegions({...props, id: 'baseline'})
        : new GPUAZPRegions({...props, id: 'baseline'});
    const contributor = kind === 'ward' ? new GPUWardRegions(props) : new GPUAZPRegions(props);
    const nodeCount = baseline.getCommandNodes(graph).length;
    props.maximumIterations = kind === 'ward' ? 14 : 256;
    props.mergeSchedule = 'mutual-nearest';
    expect(contributor.getCommandNodes(graph).length).toBe(nodeCount);
    expect(nodeCount).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  }
});

it('upper-bounds transitive nodes for all four regionalizers and both Ward schedules', () => {
  for (const rows of [6, 512, 5000]) {
    const {graph, view, weights} = setup(rows, rows * 2);
    const common = {
      weights,
      values: view('float32', rows),
      parameters: view('uint32', 1),
      maximumIterations: 2,
      output: {
        labels: view('uint32', rows),
        objective: view('float32', 1),
        status: view('uint32', 6)
      }
    };
    const contributors = [
      new GPUWardRegions({...common, id: 'ward-sequential', targetRegionCount: 2}),
      new GPUWardRegions({
        ...common,
        id: 'ward-mutual',
        targetRegionCount: 2,
        mergeSchedule: 'mutual-nearest'
      }),
      new GPUAZPRegions({...common, initialLabels: view('uint32', rows), regionCapacity: 2}),
      new GPUMaxPRegions({
        ...common,
        extensiveValues: view('float32', rows),
        threshold: view('float32', 1),
        startCount: 1,
        maximumGrowthSteps: 8
      }),
      new GPUMaxPRegions({
        ...common,
        id: 'max-p-no-improvement',
        maximumIterations: 0,
        extensiveValues: view('float32', rows),
        threshold: view('float32', 1),
        startCount: 1,
        maximumGrowthSteps: 8
      })
    ];
    for (const contributor of contributors) {
      expect(contributor.estimatedCommandNodeCount).toBeGreaterThan(0);
      expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
        contributor.estimatedCommandNodeCount
      );
    }
  }
});

it('GPUSpatialWeightsMinimumSpanningTree validates props and declares nodes', () => {
  const {graph, view, weights} = setup(6, 16);
  const base = {
    weights,
    values: view('float32', 12),
    columnCount: 2,
    output: {treeEdgeFlags: view('uint32', 16), componentLabels: view('uint32', 6)}
  };
  expect(() => new GPUSpatialWeightsMinimumSpanningTree({...base, columnCount: 3})).toThrow(
    /values length/
  );
  expect(
    () =>
      new GPUSpatialWeightsMinimumSpanningTree({
        ...base,
        output: {...base.output, treeEdgeFlags: view('uint32', 15)}
      })
  ).toThrow(/treeEdgeFlags length/);
  expect(
    () =>
      new GPUSpatialWeightsMinimumSpanningTree({
        ...base,
        standardize: false,
        output: {...base.output, standardizedValues: view('float32', 12)}
      })
  ).toThrow(/requires standardize/);
  const nodes = new GPUSpatialWeightsMinimumSpanningTree({
    ...base,
    output: {
      ...base.output,
      edges: {
        ids: view('uint32', 5),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      },
      edgeEndpoints: view('uint32', 10),
      edgeDissimilarities: view('float32', 5)
    }
  }).getCommandNodes(graph);
  expect(nodes[0].id).toBe('minimum-spanning-tree-standardize');
  // ceil(log2(6)) = 3 rounds.
  expect(nodes.filter(node => node.id.endsWith('-hook'))).toHaveLength(3);
  expect(nodes.at(-1)?.id).toBe('minimum-spanning-tree-edge-details');
});

it('GPUSkaterRegions validates props and declares one step per allowed cut', () => {
  const {graph, view, weights} = setup(6, 16);
  const base = {
    weights,
    treeEdgeFlags: view('uint32', 16),
    componentLabels: view('uint32', 6),
    values: view('float32', 12),
    columnCount: 2,
    maximumRegionCount: 4,
    parameters: view('uint32', GPU_SKATER_REGIONS_PARAMETER_LENGTH),
    output: {labels: view('uint32', 6)}
  };
  expect(() => new GPUSkaterRegions({...base, maximumRegionCount: 1})).toThrow(
    /maximumRegionCount/
  );
  expect(() => new GPUSkaterRegions({...base, parameters: view('uint32', 1)})).toThrow(
    /parameters/
  );
  expect(
    () => new GPUSkaterRegions({...base, output: {...base.output, cutEdges: view('uint32', 2)}})
  ).toThrow(/cutEdges/);
  // Every kernel stays within the 8 storage bindings the null device enforces.
  const nodes = new GPUSkaterRegions({
    ...base,
    output: {
      ...base.output,
      regionCount: view('uint32', 1),
      cutEdges: view('uint32', 3),
      cutGains: view('float32', 3)
    }
  }).getCommandNodes(graph);
  expect(nodes.filter(node => node.id.endsWith('-candidates'))).toHaveLength(3);
});

it('GPURegionPartitionEvaluation validates props and declares nodes', () => {
  const {graph, view, weights} = setup(6, 16);
  const base = {
    values: view('float32', 12),
    columnCount: 2,
    labels: view('uint32', 6),
    output: {summary: view('float32', GPU_REGION_PARTITION_EVALUATION_LAYOUT.length)}
  };
  expect(
    () => new GPURegionPartitionEvaluation({...base, output: {summary: view('float32', 4)}})
  ).toThrow(/summary/);
  expect(() => new GPURegionPartitionEvaluation({...base, labelCapacity: 0})).toThrow(
    /labelCapacity/
  );
  const nodes = new GPURegionPartitionEvaluation({...base, weights}).getCommandNodes(graph);
  // The stable label sort contributes its own nodes between the key and region kernels.
  const ids = nodes.map(node => node.id);
  expect(ids[0]).toBe('region-partition-evaluation-sort-keys');
  expect(
    ids.filter(
      id =>
        !id.startsWith('region-partition-evaluation-sort-') &&
        !id.startsWith('region-partition-evaluation-summary-')
    )
  ).toEqual([
    'region-partition-evaluation-label-offsets',
    'region-partition-evaluation-region-origins',
    'region-partition-evaluation-origin-signs',
    'region-partition-evaluation-select-origins',
    'region-partition-evaluation-region-sums-prepare',
    'region-partition-evaluation-region-sums-stage-1',
    'region-partition-evaluation-region-sums-stage-2',
    'region-partition-evaluation-region-sums-stage-4',
    'region-partition-evaluation-region-sums-finish',
    'region-partition-evaluation-region-deviations',
    'region-partition-evaluation-region-deviations-fixup',
    'region-partition-evaluation-region-centroids',
    'region-partition-evaluation-region-within',
    'region-partition-evaluation-links',
    'region-partition-evaluation-links-fixup',
    'region-partition-evaluation-row-tiles',
    // The overall means come first: the label tiles read them for the between sum of squares.
    'region-partition-evaluation-means',
    'region-partition-evaluation-label-tiles',
    'region-partition-evaluation-between-tiles',
    'region-partition-evaluation-deviation-tiles',
    'region-partition-evaluation-summary'
  ]);
});

it('standardizeColumns matches sklearn StandardScaler (population standard deviation)', () => {
  // fix/work/C/U6/standard_scaler.py, scikit-learn 1.9.1: StandardScaler().fit_transform(table).
  const table = [1, 10, 2, 20, 4, 60, 7, 0];
  const expected = [
    -1.091089451179962, -0.5488212999484517, -0.6546536707079772, -0.10976425998969035,
    0.2182178902359924, 1.6464638998453551, 1.5275252316519468, -0.9878783399072131
  ];
  standardizeColumns(table, 2).forEach((value, index) => {
    expect(value).toBeCloseTo(expected[index], 12);
  });
  // A constant column is all zero, and a missing entry stays missing without poisoning the rest.
  expect(standardizeColumns([3, 3, 3], 1)).toEqual([0, 0, 0]);
  const withGap = standardizeColumns([1, Number.NaN, 3], 1);
  expect(withGap[0]).toBeCloseTo(-1, 12);
  expect(withGap[2]).toBeCloseTo(1, 12);
  expect(Number.isNaN(withGap[1])).toBe(true);
});

it('computePseudoF matches sklearn calinski_harabasz_score on the SKATER reference runs', () => {
  // pseudoF was written by fix/work/C/U6/pseudo_f.py (scikit-learn 1.9.1).
  let checked = 0;
  for (const data of referenceCases as RegionalizationCase[]) {
    for (const run of data.runs) {
      expect(computePseudoF(data.values, data.columns, run.labels)).toBeCloseTo(run.pseudoF, 9);
      checked++;
    }
  }
  expect(checked).toBeGreaterThan(10);
  // Undefined cases: one region, or as many regions as rows.
  expect(computePseudoF([1, 2, 3], 1, [0, 0, 0])).toBeNaN();
  expect(computePseudoF([1, 2, 3], 1, [0, 1, 2])).toBeNaN();
  // sklearn returns 1.0 when the within-region dispersion is zero.
  expect(computePseudoF([1, 1, 2, 2, 2], 1, [0, 0, 1, 1, 1])).toBe(1);
});

it('selectMaxPStart prefers fewer leftovers, then more regions, then lower SSD, then the lower start', () => {
  const start = (regionCount: number, infeasibleCount: number, withinSquares: number) => ({
    regionCount,
    infeasibleCount,
    withinSquares
  });
  // More regions, but with a leftover below the threshold: the clean construction wins.
  expect(selectMaxPStart([start(5, 1, 1), start(4, 0, 9)])).toBe(1);
  // Among clean constructions, the most regions.
  expect(selectMaxPStart([start(4, 0, 1), start(6, 0, 9), start(5, 0, 0)])).toBe(1);
  // Equal region counts: the smaller within-region sum of squares.
  expect(selectMaxPStart([start(4, 0, 3), start(4, 0, 2), start(4, 0, 2.5)])).toBe(1);
  // Full tie: the lowest start index.
  expect(selectMaxPStart([start(4, 0, 2), start(4, 0, 2)])).toBe(0);
});

function createSolverInputs(rows: number, capacity: number) {
  const {graph, view, weights} = setup(rows, capacity);
  return {graph, view, weights, values: view('float32', rows)};
}

it('GPUMaxPRegions validates props and declares multi-start nodes within the binding limit', () => {
  const {graph, view, weights, values} = createSolverInputs(6, 16);
  const base = {
    weights,
    values,
    extensiveValues: view('float32', 6),
    threshold: view('float32', 1),
    parameters: view('uint32', GPU_MAX_P_REGIONS_PARAMETER_LENGTH),
    output: {
      labels: view('uint32', 6),
      objective: view('float32', 1),
      status: view('uint32', GPU_MAX_P_STATUS_LAYOUT.length)
    }
  };
  expect(() => new GPUMaxPRegions({...base, startCount: 0})).toThrow(/startCount/);
  expect(() => new GPUMaxPRegions({...base, startCount: 65})).toThrow(/startCount/);
  expect(() => new GPUMaxPRegions({...base, maximumIterations: 257})).toThrow(/maximumIterations/);
  expect(() => new GPUMaxPRegions({...base, parameters: view('uint32', 0)})).toThrow(
    /parameters must hold 1 words/
  );
  expect(
    () => new GPUMaxPRegions({...base, output: {...base.output, status: view('uint32', 4)}})
  ).toThrow(/status must hold 6 words/);
  expect(() => new GPUMaxPRegions({...base, maximumGrowthSteps: 0})).toThrow(/maximumGrowthSteps/);
  expect(() => new GPUMaxPRegions({...base, maximumGrowthSteps: 4097})).toThrow(
    /maximumGrowthSteps/
  );
  // Default: gated rounds shared by all starts, one statistics dispatch, then selection and copy;
  // no improvement.
  const nodes = new GPUMaxPRegions(base).getCommandNodes(graph);
  expect(nodes.filter(node => node.id.endsWith('-finalize-starts'))).toHaveLength(1);
  expect(nodes.filter(node => node.id.endsWith('-round-0-control'))).toHaveLength(1);
  expect(nodes.some(node => node.id.includes('-improvement-'))).toBe(false);
  // Directed weights are symmetrized through a transpose before the construction.
  expect(nodes.some(node => node.id.includes('adjacency-transpose'))).toBe(true);
  expect(nodes.some(node => node.id.endsWith('-standardize'))).toBe(true);
  // Symmetric weights skip the transpose; standardize false skips the z-scoring node.
  const symmetric = {
    ...weights,
    information: {rowCount: 6, transform: 'raw' as const, symmetric: true}
  };
  const lean = new GPUMaxPRegions({
    ...base,
    weights: symmetric,
    standardize: false,
    id: 'lean-max-p'
  }).getCommandNodes(graph);
  expect(lean.some(node => node.id.includes('adjacency'))).toBe(false);
  expect(lean.some(node => node.id.endsWith('-standardize'))).toBe(false);
  // The improvement phase appends the AZP search under the extensive constraint.
  const improving = new GPUMaxPRegions({
    ...base,
    id: 'improving-max-p',
    maximumIterations: 3
  }).getCommandNodes(graph);
  expect(improving.filter(node => node.id.includes('-improvement-iteration-'))).not.toHaveLength(0);
});

it('GPUAZPRegions validates the extensive constraint and declares the contiguity check', () => {
  const {graph, view, weights, values} = createSolverInputs(6, 16);
  const base = {
    weights,
    values,
    initialLabels: view('uint32', 6),
    regionCapacity: 2,
    maximumIterations: 2,
    parameters: view('uint32', GPU_AZP_REGIONS_PARAMETER_LENGTH),
    output: {labels: view('uint32', 6), objective: view('float32', 1), status: view('uint32', 4)}
  };
  expect(() => new GPUAZPRegions({...base, extensiveValues: view('float32', 6)})).toThrow(
    /required together/
  );
  expect(
    () =>
      new GPUAZPRegions({
        ...base,
        extensiveValues: view('float32', 5),
        extensiveThreshold: view('float32', 1)
      })
  ).toThrow(/extensiveValues length/);
  const nodes = new GPUAZPRegions({
    ...base,
    extensiveValues: view('float32', 6),
    extensiveThreshold: view('float32', 1)
  }).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids).toContain('azp-regions-initial-contiguity');
  expect(ids).toContain('azp-regions-pack-extensive');
  expect(ids).toContain('azp-regions-standardize');
  expect(ids.indexOf('azp-regions-initial-contiguity')).toBeLessThan(
    ids.indexOf('azp-regions-region-statistics')
  );
});

it('GPUWardRegions declares standardization and undirected region-edge contraction', () => {
  const {graph, view, weights, values} = createSolverInputs(6, 16);
  const base = {
    weights,
    values,
    targetRegionCount: 2,
    maximumIterations: 1,
    parameters: view('uint32', GPU_WARD_REGIONS_PARAMETER_LENGTH),
    output: {labels: view('uint32', 6), objective: view('float32', 1), status: view('uint32', 4)}
  };
  const ids = new GPUWardRegions(base).getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('ward-regions-standardize');
  expect(ids).toContain('ward-regions-initialize-edges');
  expect(ids).toContain('ward-regions-contract-0-unique');
  expect(ids).toContain('ward-regions-contract-0-compact');
  const plain = new GPUWardRegions({...base, id: 'plain-ward', standardize: false})
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(plain).not.toContain('plain-ward-standardize');
});

it('regionalization packers write the documented per-frame words', () => {
  // AZP, Ward and Max-P share the `[seed]` layout, so Max-P hands its view to the AZP phase.
  for (const pack of [
    getGPUAZPRegionsParameterValues,
    getGPUWardRegionsParameterValues,
    getGPUMaxPRegionsParameterValues
  ]) {
    expect(Array.from(pack())).toEqual([0]);
    expect(Array.from(pack({seed: 0xffffffff}))).toEqual([0xffffffff]);
    expect(() => pack({seed: -1})).toThrow(/seed/);
    expect(() => pack({seed: 2 ** 32})).toThrow(/seed/);
    expect(() => pack({seed: 1.5})).toThrow(/seed/);
  }
  expect(GPU_AZP_REGIONS_PARAMETER_LENGTH).toBe(GPU_MAX_P_REGIONS_PARAMETER_LENGTH);
  const skater = getGPUSkaterRegionsParameterValues({regionCount: 5, minimumRegionSize: 3});
  expect(skater).toHaveLength(GPU_SKATER_REGIONS_PARAMETER_LENGTH);
  expect(Array.from(skater)).toEqual([5, 3]);
  expect(Array.from(getGPUSkaterRegionsParameterValues({regionCount: 2}))).toEqual([2, 1]);
  expect(() => getGPUSkaterRegionsParameterValues({regionCount: 0})).toThrow(/regionCount/);
  expect(() => getGPUSkaterRegionsParameterValues({regionCount: 2, minimumRegionSize: 0})).toThrow(
    /minimumRegionSize/
  );
});

it('seeded regionalization kernels read the seed from the parameters binding', () => {
  // A seed is per-frame (A-21): no kernel source may depend on it, and every kernel that hashes
  // with it binds `parameters`.
  const {graph, view, weights, values} = createSolverInputs(6, 16);
  const output = () => ({
    labels: view('uint32', 6),
    objective: view('float32', 1),
    status: view('uint32', GPU_MAX_P_STATUS_LAYOUT.length)
  });
  const producers = [
    new GPUAZPRegions({
      weights,
      values,
      initialLabels: view('uint32', 6),
      regionCapacity: 2,
      maximumIterations: 2,
      parameters: view('uint32', GPU_AZP_REGIONS_PARAMETER_LENGTH),
      output: output()
    }),
    new GPUWardRegions({
      weights,
      values,
      targetRegionCount: 2,
      maximumIterations: 2,
      parameters: view('uint32', GPU_WARD_REGIONS_PARAMETER_LENGTH),
      output: output()
    }),
    new GPUWardRegions({
      id: 'ward-regions-mutual-nearest',
      weights,
      values,
      targetRegionCount: 2,
      maximumIterations: 2,
      mergeSchedule: 'mutual-nearest',
      parameters: view('uint32', GPU_WARD_REGIONS_PARAMETER_LENGTH),
      output: output()
    }),
    new GPUMaxPRegions({
      weights,
      values,
      extensiveValues: view('float32', 6),
      threshold: view('float32', 1),
      maximumIterations: 2,
      parameters: view('uint32', GPU_MAX_P_REGIONS_PARAMETER_LENGTH),
      output: output()
    })
  ];
  for (const producer of producers) {
    const sources = getGPUCommandNodeKernelSources(producer.getCommandNodes(graph));
    const seeded = sources.filter(({source}) => /tieKey|pairKey\(|philox4x32\(/.test(source));
    expect(seeded.length, producer.id).toBeGreaterThan(0);
    for (const {source} of seeded) {
      expect(source, producer.id).not.toMatch(/SEED/);
      expect(source, producer.id).toMatch(/parameters\[parametersOffset\]/);
    }
  }
});
