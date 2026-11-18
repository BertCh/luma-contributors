// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUVoronoiDiagram} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-voronoi-diagram';
import {GPUConcaveHull} from '../../../src/gpu-spatial-analysis/concave-hull/gpu-concave-hull';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createConcaveHullValidationProps} from '../concave-hull/concave-hull-validation-cases';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

function getResourceView(node: unknown, index: number): GraphDataView {
  return (node as {resources: {buffer: GraphDataView}[]}).resources[index].buffer;
}

// At larger exponent extremes orientSign uses its integer fallback, which this interpreter
// cannot execute reliably (orientExact returns 0 for the 2^100 right triangle). The headless
// sibling also checks 2^100 and 2^-100 on the device, without substituting the predicate.
for (const exponent of [45, -52]) {
  it(`Voronoi circumcenter is analytic at scale 2^${exponent}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const positions = createTransientView(graph, 'positions', 'float32x2', 3);
      const triangles = createTransientView(graph, 'triangles', 'uint32x3', 1);
      const triangleCount = createTransientView(graph, 'triangle-count', 'uint32', 1);
      const contributor = new GPUVoronoiDiagram({
        positions,
        triangles,
        triangleCount,
        clipBounds: createTransientView(graph, 'bounds', 'float32x4', 1),
        output: {
          segments: createTransientView(graph, 'segments', 'float32x4', 3),
          siteIds: createTransientView(graph, 'sites', 'uint32x2', 3),
          status: {
            count: createTransientView(graph, 'count', 'uint32', 1),
            requiredCount: createTransientView(graph, 'required', 'uint32', 1),
            overflow: createTransientView(graph, 'overflow', 'uint32', 1)
          }
        }
      });
      const nodes = contributor.getCommandNodes(graph);
      const centerNode = nodes.find(node => node.id.endsWith('-circumcenters'))!;
      const memory = createCPUGraphMemory();
      const scale = 2 ** exponent;
      memory.write(positions, Float32Array.of(0, 0, scale, 0, 0, scale));
      memory.write(triangles, Uint32Array.of(0, 1, 2));
      memory.write(triangleCount, Uint32Array.of(1));
      executeCommandNodesOnCPU([centerNode], memory);
      expect(Array.from(memory.readFloat32(getResourceView(centerNode, 3)))).toEqual([
        scale / 2,
        scale / 2,
        1
      ]);
    } finally {
      device.destroy();
    }
  });

  it(`alpha hull retention straddles the analytic radius at scale 2^${exponent}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const scale = 2 ** exponent;
      // True radius is scale / sqrt(2): 0.6 excludes it, 0.8 includes it.
      for (const [factor, expected] of [
        [0.6, 0],
        [0.8, 1]
      ]) {
        const graph = new GPUCommandGraph(device);
        const props = {
          ...createConcaveHullValidationProps(graph, 3),
          maximumCircumradius: factor * scale
        };
        const nodes = new GPUConcaveHull(props).getCommandNodes(graph);
        const retain = nodes.find(node => node.id.endsWith('-retain'))!;
        const memory = createCPUGraphMemory();
        memory.write(props.positions, Float32Array.of(0, 0, scale, 0, 0, scale));
        memory.write(props.output.triangles, Uint32Array.of(0, 1, 2));
        memory.write(getResourceView(retain, 2), Uint32Array.of(1));
        memory.write(props.output.status.converged, Uint32Array.of(1));
        executeCommandNodesOnCPU([retain], memory);
        expect(memory.readUint32(props.output.triangleValidity)[0]).toBe(expected);
      }
    } finally {
      device.destroy();
    }
  });
}

it('Voronoi hull rays point away from the opposite site for either triangle winding', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const triangle of [
      [0, 1, 2],
      [0, 2, 1]
    ]) {
      const graph = new GPUCommandGraph(device);
      const props = {
        positions: createTransientView(graph, 'positions', 'float32x2', 3),
        triangles: createTransientView(graph, 'triangles', 'uint32x3', 1),
        triangleCount: createTransientView(graph, 'triangle-count', 'uint32', 1),
        clipBounds: createTransientView(graph, 'bounds', 'float32x4', 1),
        output: {
          segments: createTransientView(graph, 'segments', 'float32x4', 3),
          siteIds: createTransientView(graph, 'sites', 'uint32x2', 3),
          status: {
            count: createTransientView(graph, 'count', 'uint32', 1),
            requiredCount: createTransientView(graph, 'required', 'uint32', 1),
            overflow: createTransientView(graph, 'overflow', 'uint32', 1)
          }
        }
      };
      const nodes = new GPUVoronoiDiagram(props).getCommandNodes(graph);
      const centers = nodes.find(node => node.id.endsWith('-circumcenters'))!;
      const clip = nodes.find(node => node.id.endsWith('-clip-segments'))!;
      const memory = createCPUGraphMemory();
      memory.write(props.positions, Float32Array.of(0, 0, 2, 0, 0, 2));
      memory.write(props.triangles, Uint32Array.from(triangle));
      memory.write(props.triangleCount, Uint32Array.of(1));
      memory.write(props.clipBounds, Float32Array.of(-2, -2, 3, 3));
      memory.write(getResourceView(clip, 4), Uint32Array.of(1, 1, 1));
      memory.write(getResourceView(clip, 5), Uint32Array.of(1, 1, 1));
      executeCommandNodesOnCPU([centers, clip], memory);
      const segments = memory.readFloat32(getResourceView(clip, 6));
      const expectedByPair: Record<string, number[]> = {
        '0,1': [1, 1, 1, -2],
        '1,2': [1, 1, 3, 3],
        '0,2': [1, 1, -2, 1]
      };
      for (let edge = 0; edge < 3; edge++) {
        const pair = [triangle[edge], triangle[(edge + 1) % 3]].sort().join(',');
        Array.from(segments.slice(4 * edge, 4 * edge + 4)).forEach((value, component) => {
          expect(Number.isFinite(value)).toBe(true);
          expect(value).toBeCloseTo(expectedByPair[pair][component], 5);
        });
      }
    }
  } finally {
    device.destroy();
  }
});
