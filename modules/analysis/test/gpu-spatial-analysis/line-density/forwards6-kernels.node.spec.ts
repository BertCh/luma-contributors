// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {getInlinedCallSites} from '../segment-intersection/wgsl-call-sites';
import {
  createWalkCountNode,
  createWalkEmitNode
} from '../../../src/gpu-spatial-analysis/line-density/line-density-kernels';
import {GPULineLengthPerPolygon} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-length-per-polygon';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

it('B5 classifies exact stored physical boundaries and their immediate neighbors before permutations', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const axis of ['x', 'y'] as const)
      for (const reverse of [false, true]) {
        const height = Math.fround(0.11);
        const boundary = 2 * height;
        const bits = new Uint32Array(Float32Array.of(boundary).buffer)[0];
        const coordinates = new Float32Array(Uint32Array.of(bits - 1, bits, bits + 1).buffer);
        for (let position = 0; position < 3; position++) {
          const graph = new GPUCommandGraph(device);
          const memory = createCPUGraphMemory();
          const positions = createTransientView(graph, 'positions', 'float32x2', 2);
          const offsets = createTransientView(graph, 'offsets', 'uint32', 2);
          const parameters = createTransientView(graph, 'parameters', 'float32', 4);
          const starts = createTransientView(graph, 'starts', 'uint32', 2);
          const counts = createTransientView(graph, 'counts', 'uint32', 2);
          const keys = createTransientView(graph, 'keys', 'uint32', 4);
          const contributions = createTransientView(graph, 'contributions', 'float32', 4);
          const coordinate = coordinates[position];
          memory.write(
            positions,
            Float32Array.from(
              axis === 'y'
                ? [0.25, coordinate, 0.75, coordinate]
                : [coordinate, 0.25, coordinate, 0.75]
            )
          );
          memory.write(offsets, Uint32Array.of(0, 2));
          memory.write(
            parameters,
            Float32Array.from(axis === 'y' ? [0, -height, 1, height] : [-height, 0, height, 1])
          );
          memory.write(starts, Uint32Array.of(0, 1));
          const constants = {
            positions,
            lineOffsets: offsets,
            columns: axis === 'x' ? 4 : 1,
            rows: axis === 'y' ? 4 : 1,
            coordinateSystem: 'planar' as const,
            radius: 1,
            capacity: 4,
            outputRowOrder: reverse ? ('north-first' as const) : ('south-first' as const),
            outputColumnOrder: reverse ? ('east-first' as const) : ('west-first' as const)
          };
          const nodes = [
            createWalkCountNode(graph, {
              id: 'count',
              operation: 'GPULineDensity',
              constants,
              parameters,
              counts
            }),
            createWalkEmitNode(graph, {
              id: 'emit',
              operation: 'GPULineDensity',
              constants,
              parameters,
              starts,
              keys,
              contributions
            })
          ];
          executeCommandNodesOnCPU(nodes, memory);
          // Exact dyadic arithmetic: (2h - (-h))/h = 3, below => 2, above => 3.
          const physicalCell = position === 0 ? 2 : 3;
          expect(Array.from(memory.readUint32(counts))).toEqual([1, 0]);
          expect(
            memory.readUint32(keys)[0],
            `${axis} reverse=${reverse} neighbor=${position}`
          ).toBe(reverse ? 3 - physicalCell : physicalCell);
          expect(memory.readFloat32(contributions)[0]).toBe(0.5);
        }
      }
  } finally {
    device.destroy();
  }
});

it('T1-F1 line clipping keeps endpoint snaps and membership on one exact call chain', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPULineLengthPerPolygon({
      positions: createTransientView(graph, 'line', 'float32x2', 2),
      lineOffsets: createTransientView(graph, 'line-offsets', 'uint32', 2),
      polygons: {
        kind: 'polygons',
        positions: createTransientView(graph, 'polygon', 'float32x2', 4),
        featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
        polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
        ringOffsets: createTransientView(graph, 'rings', 'uint32', 2)
      },
      output: {
        clippedLengths: createTransientView(graph, 'lengths', 'float32', 1),
        status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
      }
    });
    const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(
      kernel => kernel.id === 'line-length-per-polygon-clip'
    )!.source;
    expect(() => new WgslReflect(source)).not.toThrow();
    for (const [name, count] of getInlinedCallSites(source, 'orientSign'))
      expect(count, `inlined ${name}`).toBeLessThanOrEqual(1);
  } finally {
    device.destroy();
  }
});

it('T1-F1 executed clipping retains tangent snaps, narrow intervals and hole parity', () => {
  const fixtures = [
    // A triangle entirely above y=1.2 meets the line only at its lowest vertex.
    {
      line: [-5, 1.2, 15, 1.2],
      polygon: [0.2, 1.2, -0.8, 2.2, 2.3, 3.3],
      rings: [0, 3],
      expected: 0
    },
    // Positive intersection is exactly 2^-52; no epsilon may erase it.
    {
      line: [-1, 0.5, 1, 0.5],
      polygon: [0, 0, 2 ** -52, 0, 2 ** -52, 1, 0, 1],
      rings: [0, 4],
      expected: 2 ** -52
    },
    // Outer interval [0,4] minus the hole [1,3] has length 2, in either ring orientation.
    {
      line: [-1, 2, 5, 2],
      polygon: [0, 0, 4, 0, 4, 4, 0, 4, 1, 1, 3, 1, 3, 3, 1, 3],
      rings: [0, 4, 8],
      expected: 2
    }
  ];
  const device = createNullWebGPUDevice();
  try {
    for (const fixture of fixtures)
      for (const reverse of [false, true]) {
        const graph = new GPUCommandGraph(device);
        recordWGSLKernelNodes(graph);
        const contributor = new GPULineLengthPerPolygon({
          positions: createTransientView(graph, 'line', 'float32x2', 2),
          lineOffsets: createTransientView(graph, 'line-offsets', 'uint32', 2),
          polygons: {
            kind: 'polygons',
            positions: createTransientView(
              graph,
              'polygon',
              'float32x2',
              fixture.polygon.length / 2
            ),
            featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
            polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
            ringOffsets: createTransientView(graph, 'rings', 'uint32', fixture.rings.length)
          },
          pairCapacity: 1,
          output: {
            clippedLengths: createTransientView(graph, 'lengths', 'float32', 1),
            status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
          }
        });
        const node = contributor
          .getCommandNodes(graph)
          .find(candidate => candidate.id === 'line-length-per-polygon-clip')!;
        const record = getWGSLKernelNodeRecord(graph, node.id)!;
        const memory = createCPUGraphMemory();
        const line = reverse
          ? [...fixture.line.slice(2), ...fixture.line.slice(0, 2)]
          : fixture.line;
        const values: Record<string, number[]> = {
          leftRows: [0],
          rightRows: [0],
          segmentPositions: line,
          featureRings: [0, fixture.rings.length - 1],
          ringOffsets: fixture.rings,
          polygonPositions: fixture.polygon,
          insideLengths: [93],
          crossingOverflow: [0]
        };
        for (const binding of record.bindings)
          memory.write(
            binding.view,
            binding.type === 'f32'
              ? Float32Array.from(values[binding.name])
              : Uint32Array.from(values[binding.name])
          );
        executeCommandNodesOnCPU([node], memory);
        const output = record.bindings.find(binding => binding.name === 'insideLengths')!.view;
        expect(memory.readFloat32(output)[0]).toBe(fixture.expected);
      }
  } finally {
    device.destroy();
  }
});
