// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {IS_FINITE_FLOAT_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {getCoverageSnapKeyWGSL} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/coverage-topology-kernels';
import {GPUCoverageDissolve} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-dissolve';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

useCPUExecutorTestHooks();

/** Execute the shared WGSL with nonzero binding word offsets as well as position-row padding. */
function readSnapOrigin(positions: Float32Array, firstVertex: number, terminalVertex: number) {
  const output = new Float32Array(2);
  const source = `
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> ringOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const positionsOffset: u32 = 2u;
const ringOffsetsOffset: u32 = 1u;
${IS_FINITE_FLOAT_WGSL}
${getCoverageSnapKeyWGSL(0.25, 2)}
@compute @workgroup_size(1)
fn main() {
  let origin = getSnapOrigin();
  output[0] = origin.x;
  output[1] = origin.y;
}`;
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [1, 1, 1], {
    0: {
      0: positions,
      // Empty first ring: the shared first offset still names the first owned vertex.
      1: Uint32Array.of(0xffffffff, firstVertex, firstVertex, terminalVertex),
      2: output
    }
  });
  return Array.from(output);
}

it('shared snap WGSL reads the first offset rather than leading padding', () => {
  for (const value of [Number.NaN, 3e20, -3e20]) {
    const positions = Float32Array.of(
      99,
      99,
      value,
      value,
      value,
      value,
      10.0625,
      20.0625,
      value,
      value
    );
    expect(readSnapOrigin(positions, 2, 3)).toEqual([10.0625, 20.0625]);
  }
});

it('shared snap WGSL preserves the finite fallback and never uses trailing capacity for empty rings', () => {
  expect(readSnapOrigin(Float32Array.of(99, 99, 3e20, 3e20), 0, 0)).toEqual([0, 0]);
  expect(readSnapOrigin(Float32Array.of(99, 99, 3e20, 3e20), 1, 2)).toEqual([0, 0]);
  expect(readSnapOrigin(Float32Array.of(99, 99, Number.NaN, 20), 0, 1)).toEqual([0, 0]);
  expect(readSnapOrigin(Float32Array.of(99, 99, 10, Number.POSITIVE_INFINITY), 0, 1)).toEqual([
    0, 0
  ]);
});

// Dissolve and ring-emission WGSL, with CPU stand-ins for published scan/sort primitives.
// Ring statistics use barriers unsupported by this executor; they do not affect these outputs.
// This checks the hand-derived fixture and its assembly-tolerance precondition, not GPU execution.
it('dissolve keeps near boundaries separate when assembly tolerance is smaller than their gap', () => {
  const coordinates = Float32Array.of(
    10.0625,
    20.0625,
    11.125,
    20.0625,
    11.125,
    21.0625,
    10.0625,
    21.0625,
    11.1875,
    20.0625,
    12.0625,
    20.0625,
    12.0625,
    21.0625,
    11.1875,
    21.0625
  );
  const runDissolve = (paddingValue?: number) => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const memory = createCPUGraphMemory();
      const leading = paddingValue === undefined ? 0 : 2;
      const positions = new Float32Array(coordinates.length + (leading ? 10 : 0)).fill(
        paddingValue ?? 0
      );
      positions.set(coordinates, 2 * leading);
      const positionsView = createTransientView(
        graph,
        'positions',
        'float32x2',
        positions.length / 2
      );
      const ringOffsets = createTransientView(graph, 'ring-offsets', 'uint32', 3);
      const polygonOffsets = createTransientView(graph, 'polygon-offsets', 'uint32', 3);
      const featureOffsets = createTransientView(graph, 'feature-offsets', 'uint32', 3);
      const labels = createTransientView(graph, 'labels', 'uint32', 2);
      memory.write(positionsView, positions);
      memory.write(ringOffsets, Uint32Array.of(leading, leading + 4, leading + 8));
      memory.write(polygonOffsets, Uint32Array.of(0, 1, 2));
      memory.write(featureOffsets, Uint32Array.of(0, 1, 2));
      memory.write(labels, Uint32Array.of(7, 7));
      const output = {
        positions: createTransientView(graph, 'output-positions', 'float32x2', 16),
        ringOffsets: createTransientView(graph, 'output-ring-offsets', 'uint32', 3),
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1),
          boundarySegmentCount: createTransientView(graph, 'boundary-segment-count', 'uint32', 1),
          uncertainCount: createTransientView(graph, 'uncertain-count', 'uint32', 1)
        }
      };
      const contributor = new GPUCoverageDissolve({
        positions: positionsView,
        ringOffsets,
        polygonOffsets,
        featureOffsets,
        labels,
        snapTolerance: 0.25,
        // The gap is 1/16; the default 0.25 / 4 would match across it.
        vertexTolerance: 1 / 1024,
        output
      });
      const nodes = contributor.getCommandNodes(graph);
      const statisticsIndex = nodes.findIndex(
        node => node.id === 'coverage-dissolve-assembly-ring-stats'
      );
      expect(statisticsIndex).toBeGreaterThan(0);
      executeCommandNodesOnCPU(
        [
          ...nodes.slice(0, statisticsIndex),
          ...nodes.filter(
            node =>
              node.id === 'coverage-dissolve-assembly-publish' ||
              node.id === 'coverage-dissolve-snap-range-clear' ||
              node.id === 'coverage-dissolve-snap-range'
          )
        ],
        memory
      );
      const offsets = Array.from(memory.readUint32(output.ringOffsets));
      return {
        count: memory.readUint32(output.status.count)[0],
        overflow: memory.readUint32(output.status.overflow)[0],
        boundarySegmentCount: memory.readUint32(output.status.boundarySegmentCount)[0],
        uncertainCount: memory.readUint32(output.status.uncertainCount)[0],
        offsets,
        positions: Array.from(memory.readFloat32(output.positions, 2 * offsets[2]))
      };
    } finally {
      device.destroy();
    }
  };
  const reference = runDissolve();
  expect(reference.count).toBe(2);
  expect(reference.boundarySegmentCount).toBe(8);
  expect(reference.uncertainCount).toBe(0);
  expect(reference.overflow).toBe(0);
  for (const value of [Number.NaN, 3e20, -3e20]) {
    expect(runDissolve(value)).toEqual(reference);
  }
});
