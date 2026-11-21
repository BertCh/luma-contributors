// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {addNetworkCatchmentAdapter} from '../../../src/gpu-network/network-accessibility/gpu-network-accessibility';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

it('catchment adapter scores demand-major and facility-major costs without reversing directed trips', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const facilityRows of [false, true]) {
    const context = createOracleContext(device);
    const {graph, input, output, read} = context;
    try {
      const scores = output('scores', 'float32', 2);
      const ratios = output('ratios', 'float32', 2);
      const complete = output('complete', 'uint32');
      addNetworkCatchmentAdapter(graph, {
        matrix: {
          values: input('costs', 'float32', facilityRows ? [3, 1, 10, 1] : [3, 10, 1, 1]),
          rowCount: 2,
          columnCount: 2,
          startNodes: input('starts', 'uint32', facilityRows ? [2, 3] : [0, 1]),
          targetNodes: input('targets', 'uint32', facilityRows ? [0, 1] : [2, 3]),
          startsPerRow: 1,
          direction: facilityRows ? 'target-to-start' : 'start-to-target',
          converged: input('converged', 'uint32', [1])
        },
        facilityAxis: facilityRows ? 'rows' : 'columns',
        trips: 'demand-to-facility',
        supply: input('supply', 'float32', [2, 8]),
        demand: input('demand', 'float32', [1, 1]),
        parameters: input('parameters', 'float32', [5, 0, 0, 1]),
        output: scores,
        ratios,
        complete
      });
      context.run();
      // At cutoff 5 facility 0 serves both demands (ratio 2/2=1); facility 1
      // serves only demand 1 (ratio 8/1=8). Thus A=[1,1+8], independently of storage.
      expect(await read(ratios)).toEqual([1, 8]);
      expect(await read(scores)).toEqual([1, 9]);
      expect(await read(complete)).toEqual([1]);
    } finally {
      context.destroy();
    }
  }
});
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUNetworkAccessibility,
  GPUNetworkCostMatrix,
  GPUNetworkSnapping,
  GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH,
  getGPUNetworkAccessibilityParameterValues,
  type GPUNetworkAccessibilitySettings
} from '../../../src/gpu-network/network-accessibility/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  accessibilityOracle,
  buildCSR,
  catchmentOracle,
  costMatrixOracle,
  createGridFixture,
  dijkstra,
  snapOracle,
  ZERO_WEIGHT_EDGES,
  type AccessibilitySeed,
  type OracleParameters
} from './network-accessibility-oracle';
import {createOptimizationHarness} from '../network-optimization/optimization-test-harness';

it('nested catchment retains certified incumbents or suppresses unreported route loss and recovers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const method of ['2sfca', '3sfca'] as const) {
    for (const publicStatus of [false, true]) {
      const harness = createOptimizationHarness(device);
      const costs = harness.input('float32', Float32Array.of(2, 7));
      const costsComplete = harness.input('uint32', Uint32Array.of(1));
      const costLimit = harness.input('float32', Float32Array.of(2));
      const output = harness.output('float32', 2);
      const ratios = harness.output('float32', 1);
      const cumulative = harness.output('float32', 2);
      const gravity = harness.output('float32', 2);
      const complete = publicStatus ? harness.output('uint32', 1) : undefined;
      const read = await harness.run(
        new GPUNetworkAccessibility({
          costs,
          costsComplete,
          costLimit,
          opportunityWeights: harness.input('float32', Float32Array.of(12)),
          parameters: harness.input(
            'float32',
            getGPUNetworkAccessibilityParameterValues({threshold: 5})
          ),
          cumulative,
          gravity,
          catchment: {
            method,
            demand: harness.input('float32', Float32Array.of(3, 4)),
            output,
            ratios
          },
          status: complete ? {complete} : undefined
        }),
        (read, encoding) => {
          // Only population 3 is within cutoff 5: R=12/3=4 and A=[4,0].
          expect(read(output)).toEqual(publicStatus ? [4, 0] : [NaN, NaN]);
          expect(read(ratios)).toEqual(publicStatus ? [4] : [NaN]);
          expect(read(cumulative)).toEqual(publicStatus ? [12, 0] : [NaN, NaN]);
          expect(read(gravity)).toEqual(publicStatus ? [12, 0] : [NaN, NaN]);
          if (complete) expect(read(complete)).toEqual([0]);
          harness.rewrite(costLimit, Float32Array.of(5));
          harness.rewrite(costsComplete, Uint32Array.of(encoding === 0 ? 0 : 1));
        },
        3
      );
      expect(read(output)).toEqual([4, 0]);
      expect(read(ratios)).toEqual([4]);
      expect(read(cumulative)).toEqual([12, 0]);
      expect(read(gravity)).toEqual([12, 0]);
      if (complete) expect(read(complete)).toEqual([1]);
    }
  }
});

for (const evidence of ['costsComplete', 'costLimit'] as const) {
  it(`score-only graph suppresses supplied ${evidence} loss without a public certificate`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createOptimizationHarness(device);
    const costsComplete =
      evidence === 'costsComplete' ? harness.input('uint32', Uint32Array.of(0)) : undefined;
    const costLimit =
      evidence === 'costLimit' ? harness.input('float32', Float32Array.of(2)) : undefined;
    const cumulative = harness.output('float32', 2);
    const gravity = harness.output('float32', 2);
    const read = await harness.run(
      new GPUNetworkAccessibility({
        costs: harness.input('float32', Float32Array.of(2, 7)),
        opportunityWeights: harness.input('float32', Float32Array.of(12)),
        parameters: harness.input(
          'float32',
          getGPUNetworkAccessibilityParameterValues({threshold: 5})
        ),
        costsComplete,
        costLimit,
        cumulative,
        gravity
      }),
      read => {
        expect(read(cumulative)).toEqual([NaN, NaN]);
        expect(read(gravity)).toEqual([NaN, NaN]);
        if (costsComplete) harness.rewrite(costsComplete, Uint32Array.of(1));
        if (costLimit) harness.rewrite(costLimit, Float32Array.of(5));
      }
    );
    expect(read(cumulative)).toEqual([12, 0]);
    expect(read(gravity)).toEqual([12, 0]);
  });
}

function toOracleParameters(parameters: GPUNetworkAccessibilitySettings): OracleParameters {
  return {
    threshold: parameters.threshold,
    decay: parameters.decay ?? 'none',
    beta: parameters.beta ?? 0,
    minimumCost: parameters.minimumCost ?? 1
  };
}

function expectClose(actual: readonly number[], expected: readonly number[]): void {
  expectNumericArrayClose(actual, expected, {
    absoluteTolerance: value => 2e-5 * Math.max(1, Math.abs(value))
  });
}

type OutputName = 'cumulative' | 'gravity' | 'catchment' | 'ratios';
type ScoringResult = Record<OutputName, number[]>;

/** Scoring graph over a caller-owned matrix buffer, compiled once and re-encoded per frame. */
class ScoringFixture {
  readonly graph: GPUCommandGraph;
  readonly parameters: GPUParameterBuffer<'float32'>;
  readonly buffers: Buffer[] = [];
  readonly outputs: Partial<Record<OutputName, {buffer: Buffer; length: number}>> = {};
  compiled?: ReturnType<GPUCommandGraph['compile']>;
  compileCount = 0;

  constructor(
    readonly device: Device,
    matrix: Float32Array,
    props: {
      orientation: 'opportunity-rows' | 'origin-rows';
      weights: number[];
      originCount: number;
      demand?: number[];
    }
  ) {
    this.graph = new GPUCommandGraph(device, {id: 'scoring'});
    this.parameters = new GPUParameterBuffer(device, {
      id: 'accessibility-parameters',
      format: 'float32',
      length: GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH
    });
    const input = (id: string, values: Float32Array) => {
      const buffer = createInputBuffer(device, values);
      this.buffers.push(buffer);
      return importGraphBuffer(this.graph, id, buffer, 'float32', values.length);
    };
    const output = (id: OutputName, length: number): GraphDataView<'float32'> => {
      const buffer = createOutputBuffer(device, length);
      this.buffers.push(buffer);
      this.outputs[id] = {buffer, length};
      return importGraphBuffer(this.graph, id, buffer, 'float32', length);
    };
    const rowCount = props.orientation === 'opportunity-rows' ? props.weights.length : 0;
    this.graph.add(
      new GPUNetworkAccessibility({
        id: 'accessibility',
        costs: input('costs', matrix),
        orientation: props.orientation,
        opportunityWeights: input('weights', Float32Array.from(props.weights)),
        parameters: this.parameters.importToGraph(this.graph),
        cumulative: output('cumulative', props.originCount),
        gravity: output('gravity', props.originCount),
        catchment: props.demand
          ? {
              demand: input('demand', Float32Array.from(props.demand)),
              output: output('catchment', props.originCount),
              ratios: output('ratios', rowCount)
            }
          : undefined
      })
    );
  }

  /** Writes parameters and encodes the same compiled graph. */
  async run(parameters: GPUNetworkAccessibilitySettings): Promise<ScoringResult> {
    if (!this.compiled) {
      this.compiled = this.graph.compile();
      this.compileCount++;
    }
    this.parameters.write(getGPUNetworkAccessibilityParameterValues(parameters));
    submitGraph(this.device, this.compiled, undefined);
    const result: ScoringResult = {
      cumulative: [],
      gravity: [],
      catchment: [],
      ratios: []
    };
    for (const [id, output] of Object.entries(this.outputs)) {
      result[id as OutputName] = await readFloat32(output.buffer, output.length);
    }
    return result;
  }

  destroy(): void {
    this.compiled?.destroy();
    this.parameters.destroy();
    for (const buffer of this.buffers) {
      buffer.destroy();
    }
  }
}

const PARAMETER_SWEEP: GPUNetworkAccessibilitySettings[] = [
  {threshold: 10},
  {threshold: 25, decay: 'exponential', beta: 0.15},
  {threshold: Infinity, decay: 'exponential', beta: 0.05},
  {threshold: 18, decay: 'power', beta: 1.5, minimumCost: 1},
  {threshold: 0},
  {threshold: 40, decay: 'power', beta: 2, minimumCost: 2}
];

it('GPUNetworkAccessibility scores opportunity rows; threshold and decay change without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 16;
  const nodeCount = width * width;
  const {edges} = createGridFixture(11, width, width);
  const csr = buildCSR(nodeCount, edges);
  const opportunityRows: AccessibilitySeed[][] = Array.from({length: 24}, (_, row) => [
    {node: (row * 53 + 7) % nodeCount, cost: 0}
  ]);
  const weights = opportunityRows.map((_, row) => 1 + (row % 5));
  const matrix = costMatrixOracle(csr, nodeCount, opportunityRows, 40);
  const fixture = new ScoringFixture(device, matrix, {
    orientation: 'opportunity-rows',
    weights,
    originCount: nodeCount
  });
  const first: ScoringResult[] = [];
  for (const parameters of PARAMETER_SWEEP) {
    const result = await fixture.run(parameters);
    first.push(result);
    const expected = accessibilityOracle(
      matrix,
      weights.length,
      nodeCount,
      'opportunity-rows',
      weights,
      toOracleParameters(parameters)
    );
    // Integer weights: the fixed-order f32 cumulative sum is exact.
    expect(result.cumulative).toEqual(expected.cumulative);
    expectClose(result.gravity, expected.gravity);
  }
  // Re-encoding with the same parameters is bitwise identical (deterministic gathers).
  for (const [index, parameters] of PARAMETER_SWEEP.entries()) {
    const again = await fixture.run(parameters);
    expect(new Uint32Array(Float32Array.from(again.gravity).buffer)).toEqual(
      new Uint32Array(Float32Array.from(first[index].gravity).buffer)
    );
  }
  expect(fixture.compileCount).toBe(1);
  fixture.destroy();
});

it('GPUNetworkAccessibility scores origin rows with a deterministic per-row tree', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 24;
  const nodeCount = width * width;
  const {edges} = createGridFixture(5, width, width);
  const csr = buildCSR(nodeCount, edges);
  const originRows: AccessibilitySeed[][] = Array.from({length: 9}, (_, row) => [
    {node: (row * 61) % nodeCount, cost: 0}
  ]);
  // Opportunities on every node, zero on most.
  const weights = Array.from({length: nodeCount}, (_, node) => (node % 7 === 0 ? node % 4 : 0));
  const matrix = costMatrixOracle(csr, nodeCount, originRows);
  const fixture = new ScoringFixture(device, matrix, {
    orientation: 'origin-rows',
    weights,
    originCount: originRows.length
  });
  for (const parameters of PARAMETER_SWEEP) {
    const result = await fixture.run(parameters);
    const expected = accessibilityOracle(
      matrix,
      originRows.length,
      nodeCount,
      'origin-rows',
      weights,
      toOracleParameters(parameters)
    );
    expect(result.cumulative).toEqual(expected.cumulative);
    expectClose(result.gravity, expected.gravity);
  }
  expect(fixture.compileCount).toBe(1);
  fixture.destroy();
});

it('GPUNetworkAccessibility computes the two-step floating catchment area', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Facilities on the zero-weight fixture: one in the disconnected component with no demand.
  const nodeCount = 9;
  const csr = buildCSR(nodeCount, ZERO_WEIGHT_EDGES);
  const facilityRows: AccessibilitySeed[][] = [
    [{node: 0, cost: 0}],
    [{node: 4, cost: 0}],
    [{node: 8, cost: 0}],
    [{node: 3, cost: 0}]
  ];
  const supply = [10, 4, 7, 2];
  const demand = [100, 50, 20, 0, 0, 30, 0, 0, 0];
  const matrix = costMatrixOracle(csr, nodeCount, facilityRows);
  const fixture = new ScoringFixture(device, matrix, {
    orientation: 'opportunity-rows',
    weights: supply,
    originCount: nodeCount,
    demand
  });
  for (const parameters of PARAMETER_SWEEP) {
    const result = await fixture.run(parameters);
    const expected = catchmentOracle(
      matrix,
      facilityRows.length,
      nodeCount,
      supply,
      demand,
      toOracleParameters(parameters)
    );
    expectClose(result.ratios, expected.ratios);
    expectClose(result.catchment, expected.accessibility);
  }
  // The facility in the demand-free component has ratio 0, not NaN.
  const result = await fixture.run({threshold: Infinity});
  expect(result.ratios[2]).toBe(0);
  fixture.destroy();
});

it('GPUNetworkAccessibility end to end: snapped opportunities, lane-batched matrix, scores', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 12;
  const nodeCount = width * width;
  const {positions, edges} = createGridFixture(9, width, width);
  const csr = buildCSR(nodeCount, edges);
  // Opportunity points between grid nodes and one exactly on a node.
  const points = Float32Array.from([
    0.25, 0, 3.5, 2.2, 7, 7, 10.75, 4, 5.1, 9.6, 1, 10.5, 8.4, 0.2, 11, 11, 2.5, 6.25, 6.6, 3
  ]);
  const pointCount = points.length / 2;
  const opportunityWeights = Array.from({length: pointCount}, (_, row) => 1 + (row % 3));

  const graph = new GPUCommandGraph(device, {id: 'end-to-end'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    const length = format === 'float32x2' ? values.length / 2 : values.length;
    return importGraphBuffer(graph, id, buffer, format, length);
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const costLimit = 30;
  const matrixBuffer = output(pointCount * nodeCount);
  const gravityBuffer = output(nodeCount);
  const matrixView = importGraphBuffer(
    graph,
    'matrix',
    matrixBuffer,
    'float32',
    pointCount * nodeCount
  );
  const seedNodes = createTransientView(graph, 'seed-nodes', 'uint32', pointCount * 2);
  const seedCosts = createTransientView(graph, 'seed-costs', 'float32', pointCount * 2);
  const offsets = input('offsets', csr.offsets, 'uint32');
  const neighbors = input('neighbors', csr.neighbors, 'uint32');
  const weights = input('weights', csr.weights, 'float32');
  const snappedEdgesBuffer = output(pointCount);
  graph.add(
    new GPUNetworkSnapping({
      id: 'snap',
      positions: input('points', points, 'float32x2'),
      topology: {
        offsets,
        neighbors,
        weights,
        nodePositions: input('positions', positions, 'float32x2')
      },
      output: {
        snappedEdges: importGraphBuffer(
          graph,
          'snapped-edges',
          snappedEdgesBuffer,
          'uint32',
          pointCount
        ),
        startNodes: seedNodes,
        startCosts: seedCosts
      }
    })
  );
  graph.add(
    new GPUNetworkCostMatrix({
      id: 'matrix',
      topology: {offsets, neighbors, weights},
      startNodes: seedNodes,
      startCosts: seedCosts,
      startsPerRow: 2,
      laneCount: 4,
      maximumIterations: 16,
      costLimit: input('cost-limit', Float32Array.from([costLimit]), 'float32'),
      output: {costs: matrixView}
    })
  );
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH,
    values: getGPUNetworkAccessibilityParameterValues({
      threshold: 20,
      decay: 'exponential',
      beta: 0.1
    })
  });
  graph.add(
    new GPUNetworkAccessibility({
      id: 'accessibility',
      costs: matrixView,
      opportunityWeights: input(
        'opportunity-weights',
        Float32Array.from(opportunityWeights),
        'float32'
      ),
      parameters: parameters.importToGraph(graph),
      gravity: importGraphBuffer(graph, 'gravity', gravityBuffer, 'float32', nodeCount)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);

  // Oracle: snap in f64, seed Dijkstra with both endpoint costs (undirected grid).
  const snaps = snapOracle(points, positions, csr.sources, csr.neighbors, {
    edgeCosts: csr.weights
  });
  expect(await readUint32(snappedEdgesBuffer, pointCount)).toEqual(snaps.map(snap => snap.edge));
  const rows = snaps.map(snap => [
    {node: csr.sources[snap.edge], cost: snap.sourceCost},
    {node: csr.neighbors[snap.edge], cost: snap.targetCost}
  ]);
  const expectedMatrix = costMatrixOracle(csr, nodeCount, rows, costLimit);
  const matrix = await readFloat32(matrixBuffer, pointCount * nodeCount);
  expectClose(
    matrix.map(value => (Number.isFinite(value) ? value : -1)),
    Array.from(expectedMatrix, value => (Number.isFinite(value) ? value : -1))
  );
  const expected = accessibilityOracle(
    expectedMatrix,
    pointCount,
    nodeCount,
    'opportunity-rows',
    opportunityWeights,
    {threshold: 20, decay: 'exponential', beta: 0.1, minimumCost: 1}
  );
  expectClose(await readFloat32(gravityBuffer, nodeCount), expected.gravity);
  // The first point sits a quarter along edge 0 -> 1, the first CSR slot.
  expect(snaps[0].edge).toBe(0);
  expect(dijkstra(csr, nodeCount, rows[0])[0]).toBeCloseTo(0.25 * csr.weights[0], 5);

  compiled.destroy();
  parameters.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
});

/** Runs `GPUNetworkAccessibility` once on caller values and reads cumulative and status.complete. */
async function runAccessibilityStatus(
  device: Device,
  props: {
    costs: number[];
    weights: number[];
    costsComplete: number;
    costLimit?: number;
    threshold: number;
  }
): Promise<{cumulative: number[]; complete: number}> {
  const graph = new GPUCommandGraph(device, {id: 'accessibility-status'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'uint32' | 'float32'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, id, buffer, format, values.length);
  };
  const rowCount = props.costs.length / props.weights.length;
  const cumulativeBuffer = createOutputBuffer(device, rowCount);
  const completeBuffer = createOutputBuffer(device, 1);
  buffers.push(cumulativeBuffer, completeBuffer);
  graph.add(
    new GPUNetworkAccessibility({
      id: 'accessibility',
      costs: input('costs', Float32Array.from(props.costs), 'float32'),
      orientation: 'origin-rows',
      opportunityWeights: input('weights', Float32Array.from(props.weights), 'float32'),
      parameters: input(
        'parameters',
        getGPUNetworkAccessibilityParameterValues({threshold: props.threshold}),
        'float32'
      ),
      costsComplete: input('costs-complete', Uint32Array.of(props.costsComplete), 'uint32'),
      costLimit:
        props.costLimit === undefined
          ? undefined
          : input('cost-limit', Float32Array.of(props.costLimit), 'float32'),
      cumulative: importGraphBuffer(graph, 'cumulative', cumulativeBuffer, 'float32', rowCount),
      status: {complete: importGraphBuffer(graph, 'complete', completeBuffer, 'uint32', 1)}
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    cumulative: await readFloat32(cumulativeBuffer, rowCount),
    complete: (await readUint32(completeBuffer, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

it('GPUNetworkAccessibility status.complete follows the matrix word and the cost limit (NET-101, NET-103)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // One origin row: costs 1, 5 and unreached. Expected by hand: complete = costsComplete != 0 and
  // threshold <= costLimit; cumulative counts the finite costs within the threshold.
  const costs = [1, 5, Infinity];
  const weights = [1, 1, 1];
  for (const [costsComplete, costLimit, threshold, expectedComplete, expectedCumulative] of [
    [1, 10, 5, 1, 2],
    [1, 10, 10, 1, 2],
    [0, 10, 10, 0, 2],
    [1, 10, 20, 0, 2],
    [1, 10, Infinity, 0, 2],
    [1, Infinity, Infinity, 1, 2],
    [1, undefined, Infinity, 1, 2],
    [0, undefined, 1, 0, 1]
  ] as const) {
    const result = await runAccessibilityStatus(device, {
      costs,
      weights,
      costsComplete,
      costLimit,
      threshold
    });
    const label = `costsComplete ${costsComplete}, costLimit ${costLimit}, threshold ${threshold}`;
    expect(result.complete, label).toBe(expectedComplete);
    expect(result.cumulative, label).toEqual([expectedCumulative]);
  }
});

it('GPUNetworkAccessibility reports an exhausted cost matrix as incomplete (NET-101)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A 512-node directed path with unit weights, searched from node 0 for one relaxation round of
  // one hop: the search cannot reach node 511 (cost 511), so the matrix is truncated. Before the
  // fix the cumulative count below 512 had no completeness signal.
  const nodeCount = 512;
  const offsets = Uint32Array.from({length: nodeCount + 1}, (_, node) =>
    Math.min(node, nodeCount - 1)
  );
  const neighbors = Uint32Array.from({length: nodeCount - 1}, (_, edge) => edge + 1);
  const edgeWeights = new Float32Array(nodeCount - 1).fill(1);
  const graph = new GPUCommandGraph(device, {id: 'exhausted-matrix'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'uint32' | 'float32'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, id, buffer, format, values.length);
  };
  const converged = createTransientView(graph, 'exhausted-converged', 'uint32', 1);
  const matrix = createTransientView(graph, 'exhausted-costs', 'float32', nodeCount);
  const cumulativeBuffer = createOutputBuffer(device, 1);
  const completeBuffer = createOutputBuffer(device, 1);
  buffers.push(cumulativeBuffer, completeBuffer);
  graph.add(
    new GPUNetworkCostMatrix({
      id: 'matrix',
      topology: {
        offsets: input('offsets', offsets, 'uint32'),
        neighbors: input('neighbors', neighbors, 'uint32'),
        weights: input('edge-weights', edgeWeights, 'float32')
      },
      startNodes: input('start-nodes', Uint32Array.of(0), 'uint32'),
      maximumIterations: 1,
      localIterations: 1,
      output: {costs: matrix, status: {converged}}
    })
  );
  graph.add(
    new GPUNetworkAccessibility({
      id: 'accessibility',
      costs: matrix,
      orientation: 'origin-rows',
      opportunityWeights: input('weights', new Float32Array(nodeCount).fill(1), 'float32'),
      parameters: input(
        'parameters',
        getGPUNetworkAccessibilityParameterValues({threshold: Infinity}),
        'float32'
      ),
      costsComplete: converged,
      cumulative: importGraphBuffer(graph, 'cumulative', cumulativeBuffer, 'float32', 1),
      status: {complete: importGraphBuffer(graph, 'complete', completeBuffer, 'uint32', 1)}
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [cumulative] = await readFloat32(cumulativeBuffer, 1);
  expect(Number.isFinite(cumulative)).toBe(true);
  expect(cumulative).toBeGreaterThanOrEqual(1);
  expect(cumulative).toBeLessThan(nodeCount);
  expect((await readUint32(completeBuffer, 1))[0]).toBe(0);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
});
