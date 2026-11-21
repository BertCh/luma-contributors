// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPUNetworkCostWeights,
  GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH,
  getGPUNetworkCostWeightsParameterValues,
  type GPUNetworkCostWeightsProps
} from '../../../src/gpu-network/network-cost-weights/index';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import type {GPUNetworkTopology} from '../../../src/analysis-core/contracts/network-topology';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount,
  getWorkgroupMemoryBytes,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES
} from '../../utils/gpu-contributor-test-utils';
import {getNetworkCostWeightsReference} from './network-cost-weights-oracle';

let viewCount = 0;

function createView<Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${viewCount++}`, format, length);
}

/** Props for a 3 x 5 matrix with `weightsRows` output rows and `capacity` slots. */
function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkCostWeightsProps> = {},
  weightsRows = 3,
  capacity = 15
): GPUNetworkCostWeightsProps {
  return {
    id: 'cost-weights',
    costs: createView(graph, 'float32', 15),
    rowCount: 3,
    parameters: createView(graph, 'float32', GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH),
    output: {
      offsets: createView(graph, 'uint32', weightsRows + 1),
      neighbors: createView(graph, 'uint32', capacity),
      weights: createView(graph, 'float32', capacity),
      distances: createView(graph, 'float32', capacity),
      status: {
        overflow: createView(graph, 'uint32', 1),
        requiredCount: createView(graph, 'uint32', 1)
      }
    },
    ...overrides
  };
}

it('getGPUNetworkCostWeightsParameterValues packs limit, decay, beta and floor', () => {
  expect(Array.from(getGPUNetworkCostWeightsParameterValues({costLimit: 10}))).toEqual([
    10, 0, 0, 1
  ]);
  expect(
    Array.from(
      getGPUNetworkCostWeightsParameterValues({
        costLimit: Infinity,
        decay: 'power',
        beta: 1.5,
        minimumCost: 0.5
      })
    )
  ).toEqual([Infinity, 2, 1.5, 0.5]);
  expect(
    Array.from(getGPUNetworkCostWeightsParameterValues({costLimit: 4, decay: 'gaussian'}))
  ).toEqual([4, 3, 0, 1]);
  expect(() => getGPUNetworkCostWeightsParameterValues({costLimit: Number.NaN})).toThrow(
    /costLimit/
  );
  expect(() =>
    getGPUNetworkCostWeightsParameterValues({costLimit: Infinity, decay: 'gaussian'})
  ).toThrow(/gaussian/);
  expect(() => getGPUNetworkCostWeightsParameterValues({costLimit: 1, beta: -1})).toThrow(/beta/);
  expect(() =>
    getGPUNetworkCostWeightsParameterValues({costLimit: 1, decay: 'power', beta: 1, minimumCost: 0})
  ).toThrow(/minimumCost/);
  expect(() =>
    getGPUNetworkCostWeightsParameterValues({
      costLimit: 1,
      decay: 'linear' as unknown as 'none'
    })
  ).toThrow(/decay/);
});

it('GPUNetworkCostWeights resolves the matrix shape for both orientations', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const rows = new GPUNetworkCostWeights(createProps(graph));
  expect(rows.rowCount).toBe(3);
  expect(rows.columnCount).toBe(5);
  expect(rows.locationCount).toBe(5);
  expect(rows.weightsRowCount).toBe(3);
  expect(rows.capacity).toBe(15);
  expect(rows.outputWeights.information).toMatchObject({
    rowCount: 3,
    transform: 'kernel',
    provenance: {source: 'GPUNetworkCostWeights', criterion: 'network-cost'}
  });
  expect(rows.outputWeights.status?.overflow).toBe(rows.props.output.status?.overflow);
  expect(rows.outputWeights.status?.requiredCount).toBe(rows.props.output.status?.requiredCount);

  const transposed = new GPUNetworkCostWeights(
    createProps(graph, {orientation: 'columns', columns: createView(graph, 'uint32', 4)}, 4)
  );
  expect(transposed.locationCount).toBe(4);
  expect(transposed.weightsRowCount).toBe(4);
});

it('GPUNetworkCostWeights rejects malformed props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() => new GPUNetworkCostWeights(createProps(graph, {rowCount: 4}))).toThrow(
    /multiple of rowCount/
  );
  expect(() => new GPUNetworkCostWeights(createProps(graph, {rowCount: 0}))).toThrow(/rowCount/);
  expect(
    () =>
      new GPUNetworkCostWeights(createProps(graph, {orientation: 'diagonal' as unknown as 'rows'}))
  ).toThrow(/orientation/);
  expect(
    () =>
      new GPUNetworkCostWeights(createProps(graph, {parameters: createView(graph, 'float32', 3)}))
  ).toThrow(/parameters/);
  // `'columns'` has one weights row per location.
  expect(() => new GPUNetworkCostWeights(createProps(graph, {orientation: 'columns'}))).toThrow(
    /output.offsets/
  );
  const props = createProps(graph);
  expect(
    () =>
      new GPUNetworkCostWeights({
        ...props,
        output: {...props.output, weights: createView(graph, 'float32', 14)}
      })
  ).toThrow(/output.weights length/);
  expect(
    () =>
      new GPUNetworkCostWeights({
        ...props,
        output: {
          ...props.output,
          neighbors: props.costs as unknown as typeof props.output.neighbors
        }
      })
  ).toThrow();
});

it('GPUNetworkCostWeights schedules count, scan, fill and finalize kernels within core limits', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUNetworkCostWeights(
    createProps(graph, {columns: createView(graph, 'uint32', 5)})
  );
  const nodes = contributor.getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids[0]).toBe('cost-weights-count');
  expect(ids.some(id => id.startsWith('cost-weights-offsets'))).toBe(true);
  expect(ids.at(-2)).toBe('cost-weights-fill');
  expect(ids.at(-1)).toBe('cost-weights-finalize');
  const sources = getParsedKernelSources(nodes);
  for (const id of ['cost-weights-count', 'cost-weights-fill', 'cost-weights-finalize']) {
    for (const kernel of sources.getConfigurations(id)) {
      expect(getStorageBindingCount(kernel.source)).toBeLessThanOrEqual(8);
      expect(getWorkgroupMemoryBytes(kernel.source, kernel.constants)).toBeLessThanOrEqual(
        CORE_WORKGROUP_STORAGE_LIMIT_BYTES
      );
    }
  }
  // The fill kernel reads the column map and writes distances; the finalize writes both status words.
  expect(sources.get('cost-weights-fill')).toContain('columns[columnsOffset + neighbor]');
  expect(sources.get('cost-weights-fill')).toContain('distances[distancesOffset + slot]');
  expect(sources.get('cost-weights-finalize')).toContain('requiredCount[requiredCountOffset]');
});

it('GPUNetworkCostWeights transposes through the location index', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUNetworkCostWeights(createProps(graph, {orientation: 'columns'}, 5));
  const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
  // Row = location, neighbor = matrix row.
  expect(sources.get('cost-weights-count')).toContain('let column = row;');
  expect(sources.get('cost-weights-count')).toContain(
    'costs[costsOffset + neighbor * MATRIX_COLUMNS + column]'
  );
});

it('getNetworkCostWeightsReference applies the in-range rule, ordering and capacity', () => {
  const costs = [0, 2, Infinity, -1, 5, Number.NaN, 1, 3];
  const full = getNetworkCostWeightsReference({
    costs,
    rowCount: 2,
    orientation: 'rows',
    settings: {costLimit: 3},
    capacity: 8
  });
  expect(full.offsets).toEqual([0, 2, 4]);
  expect(full.neighbors).toEqual([0, 1, 2, 3]);
  expect(full.distances).toEqual([0, 2, 1, 3]);
  expect(full.weights).toEqual([1, 1, 1, 1]);
  expect(full.requiredCount).toBe(4);
  expect(full.overflow).toBe(0);
  const transposed = getNetworkCostWeightsReference({
    costs,
    rowCount: 2,
    orientation: 'columns',
    settings: {costLimit: 3},
    capacity: 3
  });
  // Location rows 0..3 list the matrix rows that reach them; the fourth slot is dropped.
  expect(transposed.offsets).toEqual([0, 1, 2, 3, 3]);
  expect(transposed.neighbors).toEqual([0, 0, 1]);
  expect(transposed.requiredCount).toBe(4);
  expect(transposed.overflow).toBe(1);
  const mapped = getNetworkCostWeightsReference({
    costs,
    rowCount: 2,
    columns: [3, 9, 1],
    orientation: 'rows',
    settings: {costLimit: 3, decay: 'exponential', beta: 1},
    capacity: 6
  });
  expect(mapped.offsets).toEqual([0, 1, 2]);
  expect(mapped.neighbors).toEqual([2, 0]);
  expect(mapped.weights[0]).toBeCloseTo(Math.exp(-2), 12);
});

it('GPUNetworkTopology is structurally a GPUSpatialWeights CSR', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const topology: GPUNetworkTopology = {
    offsets: createView(graph, 'uint32', 4),
    neighbors: createView(graph, 'uint32', 6),
    weights: createView(graph, 'float32', 6)
  };
  // Compile-time check (the test-tree typecheck): a topology is accepted where weights are read.
  const weights: GPUSpatialWeights = topology;
  expect(weights.neighbors).toBe(topology.neighbors);
});

it('GPUNetworkCostWeights carries the matrix completeness word into status.complete (NET-101)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const base = createProps(graph);
  const costsComplete = createView(graph, 'uint32', 1);
  const complete = createView(graph, 'uint32', 1);
  // Without the matrix word the weights cannot certify completeness.
  expect(
    () =>
      new GPUNetworkCostWeights({
        ...base,
        output: {...base.output, status: {...base.output.status, complete}}
      })
  ).toThrow(/status\.complete requires costsComplete/);
  expect(
    () => new GPUNetworkCostWeights({...base, costsComplete: createView(graph, 'uint32', 2)})
  ).toThrow(/costsComplete must contain exactly one row/);
  expect(
    () =>
      new GPUNetworkCostWeights({
        ...base,
        costsComplete,
        output: {
          ...base.output,
          status: {...base.output.status, complete: createView(graph, 'uint32', 2)}
        }
      })
  ).toThrow(/output\.status\.complete must contain exactly one row/);
  const contributor = new GPUNetworkCostWeights({
    ...base,
    costsComplete,
    output: {...base.output, status: {...base.output.status, complete}}
  });
  const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
  expect(sources.get('cost-weights-finalize')).toContain(
    'complete[completeOffset] = select(0u, 1u, costsComplete[costsCompleteOffset] != 0u && total <= CAPACITY);'
  );
  // Without status.complete the finalize kernel does not bind the word.
  const plainGraph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(contributor.outputWeights.status?.complete).toBe(complete);
  expect(
    () =>
      new GPUNetworkCostWeights({
        ...createProps(plainGraph),
        costsComplete: createView(plainGraph, 'uint32', 1)
      })
  ).toThrow(/output.status.complete/);
});

it('GPUNetworkCostWeights status.complete is 0 when the matrix horizon is below the requested cutoff (NET-124)', () => {
  // Five-node unit chain 0 -> 1 -> 2 -> 3 -> 4 from node 0, one matrix row, weights cutoff 4,
  // capacity 5. A converged matrix built with cutoff 1 holds [0, 1, inf, inf, inf] (2 in-range
  // entries); with cutoff 4 it holds [0, 1, 2, 3, 4] (5). Both searches converge and neither
  // overflows, so only the horizon comparison can tell them apart: expected complete words 0
  // and 1. The finalize kernel is executed with the WGSL interpreter on the recorded source.
  const runFinalize = (matrixCostLimit: number, requestedCostLimit: number, total: number) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    recordWGSLKernelNodes(graph);
    const contributor = new GPUNetworkCostWeights({
      id: 'horizon',
      costs: createView(graph, 'float32', 5),
      rowCount: 1,
      parameters: createView(graph, 'float32', GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH),
      costsComplete: createView(graph, 'uint32', 1),
      costLimit: createView(graph, 'float32', 1),
      output: {
        offsets: createView(graph, 'uint32', 2),
        neighbors: createView(graph, 'uint32', 5),
        weights: createView(graph, 'float32', 5),
        status: {complete: createView(graph, 'uint32', 1)}
      }
    });
    contributor.getCommandNodes(graph);
    const record = getWGSLKernelNodeRecord(graph, 'horizon-finalize')!;
    const storage: Record<string, Uint32Array | Float32Array> = {
      scanTotal: Uint32Array.of(total),
      offsets: Uint32Array.of(0, total),
      costsComplete: Uint32Array.of(1),
      complete: Uint32Array.of(99),
      parameters: getGPUNetworkCostWeightsParameterValues({costLimit: requestedCostLimit}),
      costLimit: Float32Array.of(matrixCostLimit)
    };
    const bindings = Object.fromEntries(
      record.bindings.map((binding, index) => [index, storage[binding.name]])
    );
    new WgslExec(
      new WgslParser().parse(record.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
    ).dispatchWorkgroups('main', [1, 1, 1], {0: bindings}, {constants: record.constants});
    return storage['complete'][0];
  };
  expect(runFinalize(1, 4, 2)).toBe(0);
  expect(runFinalize(4, 4, 5)).toBe(1);
  expect(runFinalize(5, 4, 5)).toBe(1);
  expect(runFinalize(4, Infinity, 5)).toBe(0);
  expect(runFinalize(Infinity, Infinity, 5)).toBe(1);
  expect(runFinalize(Number.NaN, 4, 5)).toBe(0);
  // Without the matrix limit the word keeps the convergence and capacity rule only.
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const base = createProps(graph);
  expect(
    () => new GPUNetworkCostWeights({...base, costLimit: createView(graph, 'float32', 2)})
  ).toThrow(/costLimit must contain exactly one row/);
  expect(
    () =>
      new GPUNetworkCostWeights({
        ...base,
        costLimit: createView(graph, 'uint32', 1) as never
      })
  ).toThrow(/costLimit/);
});
