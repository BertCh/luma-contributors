// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import type {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {getGPUMaxPRegionsParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions';
import {getGPUAZPRegionsParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions';
import type {CPUMaxPRegionsInput} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import type {CPUAZPRegionsInput} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation';
import type {MaxPRegionsOperationProps} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions-operation';
import type {AZPRegionsOperationProps} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import type {OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {getCSRFromAdjacency} from './azp-batched-mirror';

/** Two disconnected pairs force exactly two feasible regions at threshold four. */
export function createMaxPInput(): CPUMaxPRegionsInput {
  return {
    // Directed input: the twin and contributor must form the undirected union.
    offsets: Uint32Array.of(0, 0, 1, 1, 2),
    neighbors: Uint32Array.of(0, 2),
    weights: Float32Array.of(2, 7),
    values: Float32Array.of(0, 1, 2, 3, 10, -1, 14, 5),
    extensiveValues: Float32Array.of(1, 3, 2, 2),
    threshold: Float32Array.of(4),
    parameters: getGPUMaxPRegionsParameterValues({seed: 3})
  };
}

export function createMaxPProps(): MaxPRegionsOperationProps {
  return {
    id: 'max-p-test',
    rowCount: 4,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    values: 'values',
    columnCount: 2,
    standardize: false,
    extensiveValues: 'extensiveValues',
    threshold: 'threshold',
    startCount: 4,
    maximumGrowthSteps: 64,
    parameters: 'parameters',
    output: {labels: 'labels', objective: 'objective', status: 'status'}
  };
}

/** Two cliques: three core rows meet the threshold; the fourth is a zero-extensive enclave. */
export function createMaxPGrowthInput(): CPUMaxPRegionsInput {
  return {
    offsets: Uint32Array.of(0, 0, 1, 3, 6, 6, 7, 9, 12),
    neighbors: Uint32Array.of(0, 0, 1, 0, 1, 2, 4, 4, 5, 4, 5, 6),
    weights: Float32Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    values: Float32Array.of(0, 0, 0, 0, 0, 0, 6, 8, 10, -2, 10, -2, 10, -2, 16, 6),
    extensiveValues: Float32Array.of(1, 2, 3, 0, 1, 2, 3, 0),
    threshold: Float32Array.of(6),
    parameters: getGPUMaxPRegionsParameterValues({seed: 3})
  };
}

/** Two disconnected four-row paths permit two independent improving moves in one round. */
export function createAZPInput(): CPUAZPRegionsInput {
  return {
    offsets: Uint32Array.of(0, 0, 1, 2, 3, 3, 4, 5, 6),
    neighbors: Uint32Array.of(0, 1, 2, 4, 5, 6),
    weights: Float32Array.of(2, 3, 7, 5, 11, 13),
    values: Float32Array.of(0, 0, 10, 20, 11, 22, 20, 40, -10, 4, 0, 24, 1, 26, 10, 44),
    initialLabels: Uint32Array.of(0, 0, 1, 1, 2, 2, 3, 3),
    extensiveValues: Float32Array.of(2, 5, 1, 3, 4, 2, 6, 1),
    extensiveThreshold: Float32Array.of(2),
    parameters: getGPUAZPRegionsParameterValues({seed: 3})
  };
}

export function createAZPProps(): AZPRegionsOperationProps {
  return {
    id: 'azp-test',
    rowCount: 8,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    values: 'values',
    columnCount: 2,
    standardize: false,
    extensiveValues: 'extensiveValues',
    extensiveThreshold: 'extensiveThreshold',
    initialLabels: 'initialLabels',
    regionCapacity: 4,
    maximumIterations: 8,
    parameters: 'parameters',
    output: {labels: 'labels', objective: 'objective', status: 'status'}
  };
}

/** Review regression: source neighbors reconnect only outside the induced two-hop set. */
export function createAZPCycleInput(): CPUAZPRegionsInput {
  const topology = getCSRFromAdjacency([[1, 5, 6], [0, 2], [1, 3], [2, 4], [3, 5], [4, 0], [0]]);
  return {
    ...topology,
    weights: new Float32Array(topology.neighbors.length).fill(1),
    values: Float32Array.of(10, 0, 0, 0, 0, 0, 10),
    initialLabels: Uint32Array.of(0, 0, 0, 0, 0, 0, 1),
    parameters: getGPUAZPRegionsParameterValues({seed: 0})
  };
}

/** Seed 147 constructs the six-row cycle region and the singleton before refinement. */
export function createMaxPCycleInput(): CPUMaxPRegionsInput {
  const input = createAZPCycleInput();
  return {
    offsets: input.offsets,
    neighbors: input.neighbors,
    weights: input.weights,
    values: input.values,
    extensiveValues: Float32Array.of(0, 1, 1, 1, 1, 1, 4),
    threshold: Float32Array.of(4),
    parameters: getGPUMaxPRegionsParameterValues({seed: 147})
  };
}

/** Review regression: a unit extensive remainder survives float64 but is lost in float32. */
export function createAZPExtensiveRoundingInput(): CPUAZPRegionsInput {
  return {
    offsets: Uint32Array.of(0, 1, 3, 4),
    neighbors: Uint32Array.of(1, 0, 2, 1),
    weights: new Float32Array(4).fill(1),
    values: Float32Array.of(0, 10, 10),
    initialLabels: Uint32Array.of(0, 0, 1),
    extensiveValues: Float32Array.of(1, 16777216, 1),
    extensiveThreshold: Float32Array.of(1),
    parameters: getGPUAZPRegionsParameterValues({seed: 0})
  };
}

/** Nine-row lattice used to select and verify an active constrained-refinement comparison. */
export function createMaxPRefinementInput(seed: number): CPUMaxPRegionsInput {
  const adjacency = Array.from({length: 9}, (_, row) =>
    [row - 3, row + 3, ...(row % 3 > 0 ? [row - 1] : []), ...(row % 3 < 2 ? [row + 1] : [])].filter(
      neighbor => neighbor >= 0 && neighbor < 9
    )
  );
  const topology = getCSRFromAdjacency(adjacency);
  return {
    ...topology,
    weights: new Float32Array(topology.neighbors.length).fill(1),
    values: Float32Array.of(0, 9, 2, 12, 5, 15, 1, 10, 4),
    extensiveValues: Float32Array.of(1, 2, 1, 1, 2, 1, 1, 2, 1),
    threshold: Float32Array.of(3),
    parameters: getGPUMaxPRegionsParameterValues({seed})
  };
}

export function getRegionalizationOperationInputs(
  input: CPUMaxPRegionsInput | CPUAZPRegionsInput
): Record<string, OperationGraphInput> {
  return Object.fromEntries(
    Object.entries(input)
      .filter(([, values]) => values !== undefined)
      .map(([name, values]) => [
        name,
        {format: values instanceof Float32Array ? 'float32' : 'uint32', values}
      ])
  );
}

/** GPU props validation happens when the operation constructs its contributor at host lowering. */
export function constructRegionalizationGPU(
  operation: GPUContributorOperation,
  input: CPUMaxPRegionsInput | CPUAZPRegionsInput
): void {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const inputs = getRegionalizationOperationInputs(input);
    const views = new Map<string, GraphDataView>();
    const getView = <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format> => {
      let view = views.get(port);
      if (!view) {
        const descriptor = operation.outputs[port] ?? {
          format: inputs[port].format,
          length: inputs[port].values.length
        };
        view = createTransientView(graph, port, descriptor.format, descriptor.length);
        views.set(port, view);
      }
      return view as GraphDataView<Format>;
    };
    operation.createContributor(getView);
  } finally {
    operation.destroy();
    device.destroy();
  }
}
