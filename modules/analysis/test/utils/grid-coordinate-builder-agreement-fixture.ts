// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUGridAggregation,
  GPUGridBinning,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {createPointDensityGridKeysNode} from '../../src/gpu-spatial-analysis/point-density/point-density-kernels';
import {createFlowZoneNode} from '../../src/gpu-network/flow-aggregation/flow-aggregation-kernels';

/** Same graph used for shader-construction checks and device readback agreement. */
export function createGridBuilderAgreementFixture(
  device: Device,
  testCase: {value: number; minimum: number; maximum: number; size: number}
) {
  const graph = new GPUCommandGraph(device, {id: 'grid-builder-agreement'});
  const buffers: Buffer[] = [];
  function createBuffer(values: Float32Array | Uint32Array): Buffer {
    const buffer = device.createBuffer({
      data: values,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return buffer;
  }
  // Upload once. Two rows make an erroneous capacity-width row stride visible in flow.
  const positionsBuffer = createBuffer(Float32Array.of(testCase.value, 0.25, testCase.value, 0.75));
  const positions = importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 2);
  const weights = importGraphBuffer(
    graph,
    'weights',
    createBuffer(Float32Array.of(7, 13)),
    'float32',
    2
  );
  const gridSize = [testCase.size, 2] as const;
  const cellCount = testCase.size * 2;
  const contributors: (GPUGridBinning | GPUGridAggregation | GPUCommandNode)[] = [];
  // Large literal bounds hit a published-builder WGSL formatter defect. Upload one f32x4 row
  // for all four consumers in the overflow cases to keep the device arithmetic under test.
  const bounds = !Number.isFinite(
    Math.fround(Math.fround(testCase.maximum) - Math.fround(testCase.minimum))
  )
    ? importGraphBuffer(
        graph,
        'shared-bounds',
        createBuffer(Float32Array.of(testCase.minimum, 0, testCase.maximum, 1)),
        'float32x4',
        1
      )
    : ([testCase.minimum, 0, testCase.maximum, 1] as const);
  // Poison outputs so missing dispatch/initialization cannot look like agreement.
  const countsBuffer = createBuffer(new Uint32Array(cellCount).fill(0xffffffff));
  const sumsBuffer = createBuffer(new Float32Array(cellCount).fill(-1));
  const densityBuffer = createBuffer(Uint32Array.of(0xffffffff, 0xffffffff));
  const flowBuffer = createBuffer(Uint32Array.of(0xffffffff, 0xffffffff));
  contributors.push(
    new GPUGridBinning({
      id: 'published-binning',
      positions,
      bounds,
      gridSize,
      output: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', cellCount)
    })
  );
  contributors.push(
    new GPUGridAggregation({
      id: 'published-aggregation',
      positions,
      weights,
      bounds,
      gridSize,
      operation: 'sum',
      output: importGraphBuffer(graph, 'sums', sumsBuffer, 'float32', cellCount)
    })
  );
  contributors.push(
    createPointDensityGridKeysNode(graph, {
      id: 'density-reader',
      positions,
      bounds,
      gridSize,
      keyStart: 0,
      keys: importGraphBuffer(graph, 'density-keys', densityBuffer, 'uint32', 2)
    })
  );
  contributors.push(
    createFlowZoneNode(graph, {
      id: 'flow-reader',
      positions,
      bounds,
      kind: 'grid',
      gridSize: [testCase.size + 1, 3],
      activeGridSize: importGraphBuffer(
        graph,
        'active-grid-size',
        createBuffer(Uint32Array.of(testCase.size, 2)),
        'uint32',
        2
      ),
      rowBase: importGraphBuffer(
        graph,
        'row-base',
        createBuffer(Uint32Array.of(0, 2, 0, 1, 1)),
        'uint32',
        5
      ),
      zones: importGraphBuffer(graph, 'flow-zones', flowBuffer, 'uint32', 2)
    })
  );
  const nodes = contributors.flatMap(contributor =>
    'getCommandNodes' in contributor ? [...contributor.getCommandNodes(graph)] : [contributor]
  );
  graph.add(nodes);
  return {
    nodes,
    graph,
    countsBuffer,
    sumsBuffer,
    densityBuffer,
    flowBuffer,
    cellCount,
    bounds,
    destroy() {
      for (const buffer of buffers) buffer.destroy();
    }
  };
}
