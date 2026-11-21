// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUSegmentRingAssembly} from '../../../src/analysis-core/ring-assembly/index';
import {
  GPUCellSetOutline,
  type GPUCellSetOutlineProps
} from '../../../src/gpu-spatial-analysis/cell-set-outline/index';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('nested rings suppress a closed retained prefix when the outline overflows', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `outline-status-${serial++}`, format, length);
  const endpoints = view('float32x4', 4);
  const count = view('uint32', 1),
    overflow = view('uint32', 1);
  const ringCount = view('uint32', 1),
    ringOverflow = view('uint32', 1);
  const ringOffsets = view('uint32', 3);
  const open = view('uint32', 1),
    touching = view('uint32', 1);
  // Capture the actual consumer input. Full ring assembly contains cooperative
  // kernels the CPU interpreter cannot execute; the headless case checks its
  // outputs. This test executes the producer-to-consumer preparation kernel.
  const assemble = vi
    .spyOn(GPUSegmentRingAssembly.prototype, 'getCommandNodes')
    .mockReturnValue([]);
  const nodes = new GPUCellSetOutline({
    family: 'quadbin',
    cells: view('uint32x2', 2),
    output: {
      rows: view('uint32', 4),
      cells: view('uint32x2', 4),
      edgeIndices: view('uint32', 4),
      endpoints,
      status: {count, overflow}
    },
    rings: {
      output: {
        positions: view('float32x2', 10),
        ringOffsets,
        status: {
          count: ringCount,
          overflow: ringOverflow,
          openSegmentCount: open,
          touchingSegmentCount: touching
        }
      }
    }
  }).getCommandNodes(graph);
  const assembly = assemble.mock.contexts[0] as GPUSegmentRingAssembly;
  const memory = createCPUGraphMemory();
  // The deterministic retained prefix is a complete clockwise square. Shapely
  // independently finds one ring from this prefix and two from both squares;
  // see the adjacent .oracle.py. Replay only the production ring-consumer path,
  // injecting the producer's exact capacity-four count and overflow words.
  memory.write(
    endpoints,
    Float32Array.of(-90, 20, 0, 20, 0, 20, 0, 0, 0, 0, -90, 0, -90, 0, -90, 20)
  );
  try {
    for (const loss of [0, 1, 0]) {
      memory.write(count, Uint32Array.of(4));
      memory.write(overflow, Uint32Array.of(loss));
      if (assembly.props.count !== count) memory.write(assembly.props.count!, Uint32Array.of(777));
      executeCommandNodesOnCPU(
        nodes.filter(node => node.id.startsWith('cell-set-outline-rings')),
        memory
      );
      expect(memory.readUint32(assembly.props.count!)[0]).toBe(loss ? 0 : 4);
    }
  } finally {
    assemble.mockRestore();
    device.destroy();
  }
});

it.each([
  'output-map',
  'output-fields',
  'status-fields'
] as const)('outline stages keep the construction-time views after caller mutation (%s)', mutation => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `mutation-${serial++}`, format, length);
  const createOutlineOutput = (): GPUCellSetOutlineProps['output'] => ({
    rows: view('uint32', 4),
    cells: view('uint32x2', 4),
    edgeIndices: view('uint32', 4),
    endpoints: view('float32x4', 4),
    status: {
      count: view('uint32', 1),
      requiredCount: view('uint32', 1),
      overflow: view('uint32', 1)
    }
  });
  const createRingOutput = (): NonNullable<GPUCellSetOutlineProps['rings']>['output'] => ({
    positions: view('float32x2', 5),
    ringOffsets: view('uint32', 2),
    status: {
      count: view('uint32', 1),
      overflow: view('uint32', 1),
      openSegmentCount: view('uint32', 1),
      touchingSegmentCount: view('uint32', 1)
    }
  });
  const output = createOutlineOutput(),
    replacement = createOutlineOutput();
  const rings = createRingOutput(),
    replacementRings = createRingOutput();
  // Independent binding oracle: A is the set supplied at construction; B
  // has the same dimensions but must never appear in this graph. Copy the
  // identity table before mutating any of the caller's objects.
  const expected = {...output, status: {...output.status}};
  const expectedRings = {...rings, status: {...rings.status}};
  const props: GPUCellSetOutlineProps = {
    family: 'quadbin',
    cells: view('uint32x2', 1),
    output,
    rings: {output: rings}
  };
  const outline = new GPUCellSetOutline(props);
  if (mutation === 'output-map') {
    props.output = replacement;
    props.rings = {output: replacementRings};
  } else if (mutation === 'output-fields') {
    Object.assign(output, replacement);
    Object.assign(rings, replacementRings);
  } else {
    Object.assign(output.status, replacement.status);
    Object.assign(rings.status, replacementRings.status);
  }
  try {
    const nodes = outline.getCommandNodes(graph);
    const binding = (nodeId: string, name: string) =>
      getWGSLKernelNodeRecord(graph, nodeId)?.bindings.find(entry => entry.name === name)?.view;
    // Writer, publisher and gate must agree with the assembler about A.
    expect(
      binding('cell-set-outline-write', 'outEndpoints') === expected.endpoints,
      'writer endpoints A'
    ).toBe(true);
    expect(binding('cell-set-outline-write', 'outRows') === expected.rows, 'writer rows A').toBe(
      true
    );
    expect(
      binding('cell-set-outline-publish', 'countOut') === expected.status.count,
      'publisher count A'
    ).toBe(true);
    expect(
      binding('cell-set-outline-publish', 'overflowOut') === expected.status.overflow,
      'publisher overflow A'
    ).toBe(true);
    expect(
      binding('cell-set-outline-rings-prepare-count', 'count') === expected.status.count,
      'gate count A'
    ).toBe(true);
    expect(
      binding('cell-set-outline-rings-capture-loss', 'overflow') === expected.status.overflow,
      'gate overflow A'
    ).toBe(true);
    const records = nodes
      .filter(node => node.id.startsWith('cell-set-outline-rings-'))
      .map(node => getWGSLKernelNodeRecord(graph, node.id))
      .filter(record => record !== undefined);
    const endpoints = records.flatMap(record =>
      record.bindings.filter(entry => entry.name === 'endpoints')
    );
    expect(endpoints.length).toBeGreaterThan(0);
    expect(
      endpoints.every(entry => entry.view === expected.endpoints),
      'assembly endpoints A'
    ).toBe(true);
    const gateCount = binding('cell-set-outline-rings-prepare-count', 'ringInputCount');
    const consumerCounts = records.flatMap(record =>
      record.bindings.filter(entry => entry.name === 'rowCount')
    );
    expect(consumerCounts.length).toBeGreaterThan(0);
    expect(
      consumerCounts.every(entry => entry.view === gateCount),
      'assembly receives gate count'
    ).toBe(true);
    expect(
      binding('cell-set-outline-rings-publish', 'outCount') === expectedRings.status.count,
      'nested count A'
    ).toBe(true);
    expect(
      binding('cell-set-outline-rings-publish', 'outOverflow') === expectedRings.status.overflow,
      'nested overflow A'
    ).toBe(true);
    expect(
      binding('cell-set-outline-rings-publish-counters', 'outOpen') ===
        expectedRings.status.openSegmentCount,
      'nested open count A'
    ).toBe(true);
    expect(
      binding('cell-set-outline-rings-publish-counters', 'outTouching') ===
        expectedRings.status.touchingSegmentCount,
      'nested touching count A'
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it.each([
  false,
  true
])('nested rings suppress either counted source loss even when the four-edge prefix closes (report %s)', reportLoss => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `source-ring-${serial++}`, format, length);
  const source = {
    cells: view('uint32x2', 1),
    status: {
      count: view('uint32', 1),
      overflow: view('uint32', 1),
      candidateOverflow: view('uint32', 1)
    }
  };
  const status = {
    count: view('uint32', 1),
    overflow: view('uint32', 1),
    candidateOverflow: reportLoss ? view('uint32', 1) : undefined
  };
  const assemble = vi
    .spyOn(GPUSegmentRingAssembly.prototype, 'getCommandNodes')
    .mockReturnValue([]);
  try {
    const nodes = new GPUCellSetOutline({
      family: 'quadbin',
      source,
      output: {
        rows: view('uint32', 4),
        cells: view('uint32x2', 4),
        edgeIndices: view('uint32', 4),
        endpoints: view('float32x4', 4),
        status
      },
      rings: {
        output: {
          positions: view('float32x2', 5),
          ringOffsets: view('uint32', 2),
          status: {
            count: view('uint32', 1),
            overflow: view('uint32', 1),
            openSegmentCount: view('uint32', 1),
            touchingSegmentCount: view('uint32', 1)
          }
        }
      }
    }).getCommandNodes(graph);
    const assembly = assemble.mock.contexts[0] as GPUSegmentRingAssembly;
    const memory = createCPUGraphMemory();
    for (const [sourceOverflow, candidate, localOverflow] of [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 1],
      [0, 0, 0]
    ]) {
      memory.write(status.count, Uint32Array.of(4));
      memory.write(status.overflow, Uint32Array.of(localOverflow));
      memory.write(source.status.overflow, Uint32Array.of(sourceOverflow));
      memory.write(source.status.candidateOverflow, Uint32Array.of(candidate));
      executeCommandNodesOnCPU(
        nodes.filter(
          node =>
            node.id.endsWith('-source-loss') || /-rings-(capture-loss|prepare-count)$/.test(node.id)
        ),
        memory
      );
      expect(memory.readUint32(assembly.props.count!)[0]).toBe(
        sourceOverflow || candidate || localOverflow ? 0 : 4
      );
      expect(memory.readUint32(assembly.props.segmentStatus!.overflow)[0]).toBe(
        sourceOverflow || candidate || localOverflow ? 1 : 0
      );
    }
  } finally {
    assemble.mockRestore();
    device.destroy();
  }
});
