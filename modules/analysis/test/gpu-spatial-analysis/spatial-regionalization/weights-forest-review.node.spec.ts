// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {readFileSync} from 'node:fs';
import {expect, it, vi} from 'vitest';
import * as weightsLoss from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-loss';
import {
  GPUSpatialWeightsMinimumSpanningTree,
  GPU_MINIMUM_SPANNING_TREE_STATUS
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree';
import {GPUSkaterRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions';
import {GPURedcapSingleLinkage} from '../../../src/gpu-spatial-analysis/redcap-regions/gpu-redcap-single-linkage';
import {
  GPUSkaterRegionsOperation,
  type SkaterRegionsOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions-operation';
import {
  CPUSkaterRegionsOperation,
  computeSkaterRegionsOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-skater-regions-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

function scene(rows = 3) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `review-${serial++}`, format, length);
  const candidateOverflow = view('uint32', 1);
  const weights = {
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', Math.max(1, 2 * (rows - 1))),
    weights: view('float32', Math.max(1, 2 * (rows - 1))),
    status: {candidateOverflow}
  };
  const output = {
    labels: view('uint32', rows),
    regionCount: view('uint32', 3),
    cutEdges: view('uint32', 4),
    cutGains: view('float32', 4)
  };
  const props = {
    weights,
    treeEdgeFlags: view('uint32', weights.neighbors.length),
    componentLabels: view('uint32', rows),
    values: view('float32', rows),
    maximumRegionCount: 3,
    parameters: view('uint32', 2),
    output
  };
  return {device, graph, view, weights, candidateOverflow, output, props};
}

it('R2 component-only forest publishes adjacency loss independently of invalid attributes', () => {
  const {device, graph, view, weights, candidateOverflow} = scene();
  try {
    const status = view('uint32', 2);
    const labels = view('uint32', 3);
    const values = view('float32', 3);
    const nodes = new GPUSpatialWeightsMinimumSpanningTree({
      weights,
      values,
      standardize: false,
      output: {treeEdgeFlags: view('uint32', 4), componentLabels: labels, status}
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(values, Float32Array.of(1, 2, 4));
    memory.write(candidateOverflow, Uint32Array.of(9));
    memory.write(status, Uint32Array.of(88, 99));
    memory.write(labels, Uint32Array.of(0, 0, 2));
    executeCommandNodesOnCPU(
      nodes.filter(node =>
        /weights-loss|clear-invalid|count-invalid|forest-status|incomplete-component-labels/.test(
          node.id
        )
      ),
      memory
    );
    expect(Array.from(memory.readUint32(labels))).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
    expect(Array.from(memory.readUint32(status))).toEqual([0, 1]);
    expect(GPU_MINIMUM_SPANNING_TREE_STATUS).toMatchObject({
      invalidCount: 0,
      candidateOverflow: 1,
      length: 2
    });
  } finally {
    device.destroy();
  }
});

it('R3 REDCAP retains the validated loss view and both output maps after caller mutation', () => {
  const {device, graph, view, weights, candidateOverflow, output} = scene();
  try {
    const props = {
      weights,
      values: view('float32', 3),
      parameters: view('uint32', 1),
      componentIterations: 3,
      output: {labels: output.labels, regionCount: output.regionCount}
    };
    const contributor = new GPURedcapSingleLinkage(props);
    props.weights.status = {} as typeof weights.status;
    props.output = {labels: view('uint32', 3), regionCount: view('uint32', 1)};
    const nodes = contributor.getCommandNodes(graph);
    for (const prefix of ['redcap-single-linkage-tree-', 'redcap-single-linkage-suppress-'])
      expect(
        nodes.some(
          node =>
            node.id.startsWith(prefix) &&
            getWGSLKernelNodeRecord(graph, node.id)?.bindings.some(
              binding => binding.view === candidateOverflow
            )
        ),
        prefix
      ).toBe(true);
    const memory = createCPUGraphMemory();
    memory.write(candidateOverflow, Uint32Array.of(9));
    memory.write(output.labels, Uint32Array.of(0, 0, 2));
    memory.write(output.regionCount, Uint32Array.of(2, 98, 99));
    executeCommandNodesOnCPU(
      nodes.filter(node => node.id.startsWith('redcap-single-linkage-suppress-')),
      memory
    );
    expect(Array.from(memory.readUint32(output.labels))).toEqual([
      0xffffffff, 0xffffffff, 0xffffffff
    ]);
    expect(Array.from(memory.readUint32(output.regionCount))).toEqual([0, 98, 99]);
  } finally {
    device.destroy();
  }
});

it('X4 REDCAP suppression preserves every scalar tail during loss and recovery', () => {
  const {device, graph, view, weights, candidateOverflow, output} = scene();
  try {
    const converged = view('uint32', 3);
    const iterationLimitReached = view('uint32', 3);
    const nodes = new GPURedcapSingleLinkage({
      weights,
      values: view('float32', 3),
      parameters: view('uint32', 1),
      componentIterations: 3,
      output: {
        labels: output.labels,
        regionCount: output.regionCount,
        status: {converged, iterationLimitReached}
      }
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(output.regionCount, Uint32Array.of(2, 98, 99));
    memory.write(converged, Uint32Array.of(1, 88, 89));
    memory.write(iterationLimitReached, Uint32Array.of(1, 90, 91));
    for (const loss of [0, 9, 0]) {
      memory.write(candidateOverflow, Uint32Array.of(loss));
      memory.write(output.labels, Uint32Array.of(0, 0, 2));
      memory.write(output.regionCount, Uint32Array.of(2));
      memory.write(converged, Uint32Array.of(1));
      memory.write(iterationLimitReached, Uint32Array.of(1));
      executeCommandNodesOnCPU(
        nodes.filter(node => node.id.startsWith('redcap-single-linkage-suppress-')),
        memory
      );
      expect(Array.from(memory.readUint32(output.regionCount))).toEqual([loss ? 0 : 2, 98, 99]);
      expect(Array.from(memory.readUint32(converged))).toEqual([loss ? 0 : 1, 88, 89]);
      expect(Array.from(memory.readUint32(iterationLimitReached))).toEqual([loss ? 0 : 1, 90, 91]);
      expect(Array.from(memory.readUint32(output.labels))).toEqual(
        loss ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 2]
      );
    }
  } finally {
    device.destroy();
  }
});

for (const destination of ['empty-offset', 'float32'] as const)
  it(`X5 REDCAP rejects an invalidCount destination that is ${destination}`, () => {
    const {device, graph, view, weights, output} = scene();
    try {
      const backing = view('uint32', 3);
      const invalidCount =
        destination === 'empty-offset'
          ? graph.createDataView(backing.buffer, {format: 'uint32', length: 0, byteOffset: 4})
          : view('float32', 1);
      expect(
        () =>
          new GPURedcapSingleLinkage({
            weights,
            values: view('float32', 3),
            parameters: view('uint32', 1),
            output: {
              labels: output.labels,
              status: {invalidCount: invalidCount as GraphDataView<'uint32'>}
            }
          })
      ).toThrow(/redcap-single-linkage.*output.status.invalidCount/);
    } finally {
      device.destroy();
    }
  });

it('X5 REDCAP publishes invalidCount into an offset scalar without changing adjacent words', () => {
  const {device, graph, view, weights, output} = scene();
  try {
    const backing = view('uint32', 3);
    const invalidCount = graph.createDataView(backing.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    const values = view('float32', 3);
    const nodes = new GPURedcapSingleLinkage({
      weights,
      values,
      parameters: view('uint32', 1),
      output: {labels: output.labels, status: {invalidCount}}
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(backing, Uint32Array.of(91, 92, 93));
    memory.write(values, Float32Array.of(NaN, 2, NaN));
    executeCommandNodesOnCPU(
      nodes.filter(node =>
        /tree-clear-invalid|tree-count-invalid|^redcap-single-linkage-invalid-count$/.test(node.id)
      ),
      memory
    );
    expect(Array.from(memory.readUint32(backing))).toEqual([91, 2, 93]);
  } finally {
    device.destroy();
  }
});

it('X7 reciprocal zero-weight slots remain forest links', () => {
  const {device, graph, view} = scene(2);
  try {
    const weights = {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 2),
      weights: view('float32', 2)
    };
    const values = view('float32', 2);
    const treeEdgeFlags = view('uint32', 2);
    const componentLabels = view('uint32', 2);
    const nodes = new GPUSpatialWeightsMinimumSpanningTree({
      weights,
      values,
      standardize: false,
      output: {treeEdgeFlags, componentLabels}
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(weights.offsets, Uint32Array.of(0, 1, 2));
    memory.write(weights.neighbors, Uint32Array.of(1, 0));
    memory.write(weights.weights, Float32Array.of(0, 0));
    memory.write(values, Float32Array.of(1, 4));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(treeEdgeFlags))).toEqual([1, 0]);
    expect(Array.from(memory.readUint32(componentLabels))).toEqual([0, 0]);
  } finally {
    device.destroy();
  }
});

it('X7 weights guide documents the topology-only forest exception', () => {
  const guide = readFileSync('docs/api/gpu-spatial-analysis/operators-weights.md', 'utf8');
  expect(guide.includes('One neighbor definition is used everywhere')).toBe(false);
  expect(
    /GPUSpatialWeightsMinimumSpanningTree[\s\S]{0,160}every listed slot is a link/.test(guide)
  ).toBe(true);
});

it('X9 SKATER validates its snapshotted loss sources once at construction', () => {
  const {device, props} = scene();
  const validate = vi.spyOn(weightsLoss, 'validateWeightsLossSources');
  try {
    new GPUSkaterRegions(props);
    expect(validate).toHaveBeenCalledTimes(1);
  } finally {
    validate.mockRestore();
    device.destroy();
  }
});

it('R2 SKATER CPU consumes forest loss without a weights loss port and restores a real cut', () => {
  const backing = Uint32Array.of(77, 0, 0, 98);
  const treeStatus = backing.subarray(1, 3);
  for (const loss of [0, 9, 0]) {
    treeStatus[1] = loss;
    const result = computeSkaterRegionsOnCPU(
      {
        offsets: Uint32Array.from(loss ? [0, 1, 2, 2] : [0, 1, 3, 4]),
        neighbors: Uint32Array.of(1, 0, 2, 1),
        weights: Float32Array.of(1, 1, 1, 1),
        treeEdgeFlags: Uint32Array.from(loss ? [0, 0, 0, 0] : [1, 0, 1, 0]),
        componentLabels: Uint32Array.from(loss ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 0]),
        values: Float32Array.of(1, 2, 4),
        parameters: Uint32Array.of(2, 1),
        treeStatus
      },
      {maximumRegionCount: 3}
    );
    expect(Array.from(result.labels)).toEqual(
      loss ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 2]
    );
    expect(Array.from(result.regionCount)).toEqual([loss ? 0 : 2]);
    expect(Array.from(result.cutEdges)).toEqual([loss ? 0xffffffff : 2, 0xffffffff]);
    if (loss) expect(Array.from(result.cutGains).every(Number.isNaN)).toBe(true);
    else {
      expect(Number.isFinite(result.cutGains[0])).toBe(true);
      expect(result.cutGains[0]).toBeCloseTo(25 / 6, 5);
      expect(result.cutGains[1]).toBe(0);
    }
  }
  expect(backing[0]).toBe(77);
  expect(backing[3]).toBe(98);
});

for (const rows of [1, 3])
  it(`R4 SKATER includes loss nodes in its estimate at ${rows} rows`, () => {
    const {device, graph, props} = scene(rows);
    try {
      const contributor = new GPUSkaterRegions(props);
      expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
        contributor.estimatedCommandNodeCount
      );
    } finally {
      device.destroy();
    }
  });

it('R4 SKATER rejects the loss-bearing command budget boundary at construction', () => {
  const {device, view, props} = scene(4094);
  try {
    const output = {
      ...props.output,
      cutEdges: view('uint32', 4093),
      cutGains: view('float32', 4093)
    };
    expect(
      () =>
        new GPUSkaterRegions({
          ...props,
          weights: {...props.weights, status: undefined},
          maximumRegionCount: 4094,
          output
        })
    ).not.toThrow();
    expect(
      () =>
        new GPUSkaterRegions({
          ...props,
          maximumRegionCount: 4094,
          output
        })
    ).toThrow(/command.*32768|32768.*command/i);
  } finally {
    device.destroy();
  }
});

it('R5 SKATER suppression preserves spare count and cut-log canaries', () => {
  const {device, graph, props, candidateOverflow, output} = scene();
  try {
    const nodes = new GPUSkaterRegions(props).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(candidateOverflow, Uint32Array.of(9));
    memory.write(output.labels, Uint32Array.of(0, 0, 2));
    memory.write(output.regionCount, Uint32Array.of(2, 98, 99));
    memory.write(output.cutEdges, Uint32Array.of(2, 0xffffffff, 98, 99));
    memory.write(output.cutGains, Float32Array.of(25 / 6, 0, 98, 99));
    executeCommandNodesOnCPU(
      nodes.filter(node => /weights-loss|incomplete-/.test(node.id)),
      memory
    );
    expect(Array.from(memory.readUint32(output.regionCount))).toEqual([0, 98, 99]);
    expect(Array.from(memory.readUint32(output.cutEdges))).toEqual([
      0xffffffff, 0xffffffff, 98, 99
    ]);
    expect(Array.from(memory.readFloat32(output.cutGains))).toEqual([NaN, NaN, 98, 99]);
  } finally {
    device.destroy();
  }
});

function operationProps(): SkaterRegionsOperationProps {
  return {
    id: 'review-skater',
    rowCount: 3,
    edgeCount: 4,
    maximumRegionCount: 3,
    weights: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      information: {rowCount: 3, transform: 'raw', symmetric: true}
    },
    treeEdgeFlags: 'treeEdgeFlags',
    componentLabels: 'componentLabels',
    values: 'values',
    parameters: 'parameters',
    output: {labels: 'labels'}
  };
}

it('R6 operation metadata remains at its validated row count after caller mutation', async () => {
  const {device, graph, view} = scene();
  try {
    const props = operationProps();
    const gpu = new GPUSkaterRegionsOperation(props);
    const cpu = new CPUSkaterRegionsOperation(props);
    props.weights.information!.rowCount = 4;
    const getView = <Format extends 'uint32' | 'float32'>(port: string): GraphDataView<Format> =>
      view(
        ['weights', 'values'].includes(port) ? 'float32' : 'uint32',
        ['offsets', 'neighbors', 'weights', 'treeEdgeFlags'].includes(port)
          ? 4
          : port === 'parameters'
            ? 2
            : 3
      ) as GraphDataView<Format>;
    expect(() =>
      gpu
        .createContributor(getView as Parameters<typeof gpu.createContributor>[0])
        .getCommandNodes(graph)
    ).not.toThrow();
    const arrays = {
      offsets: Uint32Array.of(0, 1, 3, 4),
      neighbors: Uint32Array.of(1, 0, 2, 1),
      weights: Float32Array.of(1, 1, 1, 1),
      treeEdgeFlags: Uint32Array.of(1, 0, 1, 0),
      componentLabels: Uint32Array.of(0, 0, 0),
      values: Float32Array.of(1, 2, 4),
      parameters: Uint32Array.of(2, 1)
    };
    const result = await cpu.execute({
      inputs: Object.fromEntries(
        Object.entries(arrays).map(([name, values]) => [
          name,
          {
            format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
            length: values.length,
            chunks: [{values, length: values.length}],
            pack: () => values
          }
        ])
      )
    });
    expect(result['labels']).toEqual(Uint32Array.of(0, 0, 2));
  } finally {
    device.destroy();
  }
});

for (const Backend of [GPUSkaterRegionsOperation, CPUSkaterRegionsOperation])
  for (const name of ['overflow', 'candidateOverflow'] as const)
    it(`R7 ${Backend.name} rejects empty ${name} and accepts undefined`, () => {
      const props = operationProps();
      expect(
        () => new Backend({...props, weights: {...props.weights, status: {[name]: ''}}})
      ).toThrow(new RegExp(`^review-skater.*${name}.*non-empty`));
      expect(
        () => new Backend({...props, weights: {...props.weights, status: {[name]: undefined}}})
      ).not.toThrow();
    });
