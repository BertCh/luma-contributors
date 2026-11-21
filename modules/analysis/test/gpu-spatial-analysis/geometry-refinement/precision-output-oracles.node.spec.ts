// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUPolygonPrecisionReduction} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-precision-reduction';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createTopologyOutput} from './topology-output';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

useCPUExecutorTestHooks();

// These inherited fields have no publisher in polygonization. Refusal is preferable to returning
// caller poison as a diagnostic. The supported uncertainty counter is exercised on GPU separately.
for (const level of ['nodedSegments', 'directedEdges', 'status'] as const)
  for (const field of [
    'invalidCount',
    'iterationCount',
    'converged',
    'iterationLimitReached',
    'approximation'
  ] as const)
    it(`precision refuses unpublished ${level}.${field} at construction`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const output = createTopologyOutput((format, length) =>
        createTransientView(graph, `output-${serial++}`, format, length)
      );
      const status = level === 'status' ? output.status : output[level].status;
      Object.assign(status, {[field]: createTransientView(graph, 'unsupported', 'uint32', 1)});
      try {
        expect(
          () =>
            new GPUPolygonPrecisionReduction({
              id: 'precision-oracle',
              polygons: {
                kind: 'polygons',
                positions: createTransientView(graph, 'positions', 'float32x2', 4),
                ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
                polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
                featureOffsets: createTransientView(graph, 'features', 'uint32', 2)
              },
              intersectionCapacity: 8,
              gridSize: 1,
              lattice: createTransientView(graph, 'lattice', 'float32', 4),
              output
            })
        ).toThrow(
          new RegExp(
            `^precision-oracle output\\.${level === 'status' ? 'status' : `${level}\\.status`}\\.${field} .*unsupported.*uncertainCount`
          )
        );
      } finally {
        device.destroy();
      }
    });

it.each([
  'implicit',
  'explicit',
  'padded'
] as const)('precision %s closures preserve original input edge rows on every ring', mode => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let serial = 0;
  const view = <Format extends GPUGraphVectorFormat>(
    format: Format,
    length: number,
    values?: number[]
  ): GraphDataView<Format> => {
    const result = createTransientView(graph, `precision-${serial++}`, format, length);
    const components = result.rowByteLength / 4;
    const ArrayType = format.startsWith('float') ? Float32Array : Uint32Array;
    memory.write(
      result,
      values ? ArrayType.from(values) : new ArrayType(length * components).fill(913)
    );
    return result;
  };
  const output = createTopologyOutput(view);
  const coordinates =
    mode === 'explicit'
      ? [0, 0, 4, 0, 4, 2, 0, 2, 0, 0, 10, 4, 13, 4, 13, 7, 10, 7, 10, 4]
      : mode === 'padded'
        ? [99, 99, 0, 0, 4, 0, 4, 2, 0, 2, 10, 4, 13, 4, 13, 7, 10, 7, 77, 77]
        : [0, 0, 4, 0, 4, 2, 0, 2, 10, 4, 13, 4, 13, 7, 10, 7];
  const ringOffsets = mode === 'explicit' ? [0, 5, 10] : mode === 'padded' ? [1, 5, 9] : [0, 4, 8];
  const expectedEdges =
    mode === 'explicit'
      ? [0, 1, 2, 3, 5, 6, 7, 8]
      : mode === 'padded'
        ? [1, 2, 3, 4, 5, 6, 7, 8]
        : [0, 1, 2, 3, 4, 5, 6, 7];
  const contributor = new GPUPolygonPrecisionReduction({
    id: 'precision-oracle',
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', coordinates.length / 2, coordinates),
      ringOffsets: view('uint32', 3, ringOffsets),
      polygonOffsets: view('uint32', 3, [0, 1, 2]),
      featureOffsets: view('uint32', 3, [0, 1, 2])
    },
    gridSize: 1,
    intersectionCapacity: 16,
    lattice: view('float32', 4),
    output
  });
  try {
    const nodes = contributor.getCommandNodes(graph);
    const placement = nodes.findIndex(
      node => node.id === 'precision-oracle-polygonize-faces-propagate-duplicate-owners'
    );
    expect(placement).toBeGreaterThan(0);
    // Execute real close/noding/face-placement kernels, with CPU stand-ins only for upstream
    // cooperative scans/sorts/BVH. Skip the later cooperative area reduction and parity filter;
    // provenance remaps depend only on noded source metadata, so execute them after this prefix.
    executeCommandNodesOnCPU(
      [
        ...nodes.slice(0, placement + 1),
        ...nodes.filter(
          node =>
            node.id === 'precision-oracle-source-edge-rows' ||
            node.id === 'precision-oracle-original-edge-rows'
        )
      ],
      memory
    );
    expect(Array.from(memory.readUint32(output.nodedSegments.status.count))).toEqual([8]);
    // No intersections: every outgoing edge is an original position row, including each
    // ring's last distinct vertex. Packing closures and stripping padding must preserve those IDs.
    expect(Array.from(memory.readUint32(output.nodedSegments.sourceEdgeIds)).slice(0, 8)).toEqual(
      expectedEdges
    );
  } finally {
    device.destroy();
  }
});
