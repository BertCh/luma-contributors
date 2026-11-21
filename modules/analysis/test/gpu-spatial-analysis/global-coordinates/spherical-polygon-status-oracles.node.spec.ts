// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSphericalPolygonContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createSphericalPredicateScenes} from './spherical-predicate-scenes.test-utils';

it('the actual spherical polygon WGSL helper chain separates a short arc from its antipode', () => {
  const source = readFileSync(
    new URL(
      '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains.ts',
      import.meta.url
    ),
    'utf8'
  );
  const helpers = ['polygonEdgeNormal', 'polygonOnArc']
    .map(name => {
      const definition = source.match(new RegExp(`fn ${name}\\([^]*?\\n\\}`))?.[0];
      expect(definition, name).toBeDefined();
      return definition!;
    })
    .join('\n');
  const halfAngle = 0.00075;
  const first = [Math.fround(Math.cos(halfAngle)), Math.fround(-Math.sin(halfAngle)), 0];
  const second = [first[0], -first[1], 0];
  const radialQuery = [1.00002, -3e-7, -4e-7].map(Math.fround);
  const normalizedQuery = radialQuery.map(value => value / Math.hypot(...radialQuery));
  const cases = [
    {first, second, point: [-1, 0, 0], margin: 0.001, expected: 0},
    {first, second, point: [1, 0, 0], margin: 0.001, expected: 1},
    ...[0.0005, 0.002].map(extra => ({
      first,
      second,
      point: [Math.cos(halfAngle + extra), Math.sin(halfAngle + extra), 0],
      margin: 0.001,
      expected: Number(extra === 0.0005)
    })),
    ...[radialQuery, normalizedQuery].flatMap(point => [
      {first: [1, 0, 0], second: [0, 1, 0], point, margin: 1e-6, expected: 1},
      {first: [0, 0, 1], second: [1, 0, 0], point, margin: 1e-6, expected: 1}
    ])
  ];
  // The equatorial interval [-h,h] excludes its antipode. A point .0005 beyond
  // the endpoint is within the .001 cap; .002 is outside. The radial query's
  // direction is atan2(5e-7,1.00002) < 1e-6 from +x, regardless of radial error.
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const inputs = createTransientView(graph, 'predicate-inputs', 'float32', cases.length * 10);
    const output = createTransientView(graph, 'predicate-results', 'uint32', cases.length);
    const memory = createCPUGraphMemory();
    memory.write(
      inputs,
      Float32Array.from(
        cases.flatMap(scene => [...scene.first, ...scene.second, ...scene.point, scene.margin])
      )
    );
    const node = createWGSLKernelNode(graph, {
      id: 'actual-on-arc',
      operation: 'SphericalPredicateOracle',
      invocationCount: cases.length,
      bindings: [
        {name: 'inputs', view: inputs, type: 'f32', access: 'read'},
        {name: 'output', view: output, type: 'u32', access: 'read_write'}
      ],
      declarations: helpers,
      body: `let offset=inputsOffset+index*10u;
let first=vec3f(inputs[offset],inputs[offset+1u],inputs[offset+2u]);
let second=vec3f(inputs[offset+3u],inputs[offset+4u],inputs[offset+5u]);
let point=vec3f(inputs[offset+6u],inputs[offset+7u],inputs[offset+8u]);
output[outputOffset+index]=select(0u,1u,polygonOnArc(first,second,point,inputs[offset+9u]));`
    });
    executeCommandNodesOnCPU([node], memory);
    expect(Array.from(memory.readUint32(output))).toEqual(cases.map(scene => scene.expected));
  } finally {
    device.destroy();
  }
});

it.each(createSphericalPredicateScenes())('interpreted production containment: $name', scene => {
  const device = createNullWebGPUDevice();
  try {
    for (const reversed of [false, true])
      for (const boundary of ['open', 'closed'] as const) {
        const graph = new GPUCommandGraph(device);
        const positions = createTransientView(
          graph,
          'positions',
          'float32x3',
          scene.vertices.length
        );
        const ringOffsets = createTransientView(graph, 'rings', 'uint32', 2);
        const polygonOffsets = createTransientView(graph, 'polygons', 'uint32', 2);
        const queries = createTransientView(graph, 'queries', 'float32x3', 1);
        const polygonIndices = createTransientView(graph, 'indices', 'uint32', 1);
        const contains = createTransientView(graph, 'contains', 'uint32', 1);
        const status = createTransientView(graph, 'status', 'uint32', 1);
        const memory = createCPUGraphMemory();
        memory.write(
          positions,
          Float32Array.from((reversed ? [...scene.vertices].reverse() : scene.vertices).flat())
        );
        memory.write(ringOffsets, Uint32Array.of(0, scene.vertices.length));
        memory.write(polygonOffsets, Uint32Array.of(0, 1));
        memory.write(queries, Float32Array.from(scene.query));
        memory.write(polygonIndices, Uint32Array.of(0));
        const contributor = new GPUSphericalPolygonContains({
          positions,
          ringOffsets,
          polygonOffsets,
          queries,
          polygonIndices,
          tolerance: scene.tolerance,
          boundary,
          output: {contains, status}
        });
        executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
        expect(Array.from(memory.readUint32(status)), `${reversed} ${boundary}`).toEqual([
          scene.status
        ]);
        expect(Array.from(memory.readUint32(contains)), `${reversed} ${boundary}`).toEqual([
          scene.status === 2 ? Number(boundary === 'closed') : scene.contains
        ]);
      }
  } finally {
    device.destroy();
  }
});
