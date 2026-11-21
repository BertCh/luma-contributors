// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it} from 'vitest';
import {describeOutputs} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {GPULineChunk} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-chunk';
import type {GPULineGeometryOutput} from '../../../src/gpu-spatial-analysis/line-segmentize/line-segmentize-types';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());

function expectOutputContract(
  description: ReturnType<typeof describeOutputs>,
  mode: 'chunk' | 'substring'
): void {
  const lineCapacity = mode === 'chunk' ? 5 : 2;
  // Independent full schema: 2 mandatory geometry ports, 3 optional columns, 5 status scalars.
  const actual = description.outputs.map(port => [
    port.path,
    port.format,
    port.capacity,
    port.optional
  ]);
  const expected = [
    ['output.positions', 'float32x2', 12, false],
    ['output.lineOffsets', 'uint32', lineCapacity + 1, false],
    ['output.sourceLineIndices', 'uint32', lineCapacity, true],
    ['output.sourceRows', 'uint32', 12, true],
    ['output.measures', 'float32', 12, true],
    ['output.status.count', 'uint32', 1, false],
    ['output.status.overflow', 'uint32', 1, false],
    ['output.status.requiredCount', 'uint32', 1, true],
    ['output.status.vertexCount', 'uint32', 1, true],
    ['output.status.requiredVertexCount', 'uint32', 1, true]
  ];
  expect(actual).toEqual(expected);
  const layouts = Object.fromEntries(description.outputs.map(port => [port.path, port]));
  expect(layouts['output.positions']).toMatchObject({format: 'float32x2', capacity: 12});
  expect(layouts['output.lineOffsets']).toMatchObject({
    format: 'uint32',
    capacity: lineCapacity + 1,
    activeLengthRule: 'output.status.count + 1'
  });
  expect(layouts['output.sourceLineIndices']).toMatchObject({
    format: 'uint32',
    capacity: lineCapacity
  });
  for (const path of ['output.status.count', 'output.status.overflow']) {
    expect(layouts[path]).toMatchObject({format: 'uint32', capacity: 1, optional: false});
  }
  expect(layouts['output.positions'].status).toEqual([
    'output.status.count',
    'output.status.overflow'
  ]);
}

for (const mode of ['chunk', 'substring'] as const) {
  for (const requiredOnly of [false, true]) {
    it(`describes ${mode} ${requiredOnly ? 'required-only' : 'all-port'} allocations accepted by the actual line contributor`, () => {
      const graph = new GPUCommandGraph(device);
      const description = describeOutputs(
        {package: '@luma-contributors/analysis/gpu-spatial-analysis', exportName: 'GPULineChunk'},
        mode,
        {
          lineCount: 2,
          lineCapacity: 5,
          vertexCapacity: 12,
          positionCount: 6
        }
      );
      expect(description.declaration).toBe('declared');
      expectOutputContract(description, mode);
      const lineCapacity = mode === 'chunk' ? 5 : 2;
      const ports: Record<string, GraphDataView> = {};
      for (const port of description.outputs) {
        if (requiredOnly && port.optional) continue;
        ports[port.path] = createTransientView(graph, port.path, port.format, port.capacity);
      }
      const getPort = <Format extends GraphDataView['format']>(path: string) =>
        ports[path] as GraphDataView<Format>;
      const output: GPULineGeometryOutput = {
        positions: getPort<'float32x2'>('output.positions'),
        lineOffsets: getPort<'uint32'>('output.lineOffsets'),
        sourceLineIndices: getPort<'uint32'>('output.sourceLineIndices'),
        sourceRows: getPort<'uint32'>('output.sourceRows'),
        measures: getPort<'float32'>('output.measures'),
        status: {
          count: getPort<'uint32'>('output.status.count'),
          overflow: getPort<'uint32'>('output.status.overflow'),
          requiredCount: getPort<'uint32'>('output.status.requiredCount'),
          vertexCount: getPort<'uint32'>('output.status.vertexCount'),
          requiredVertexCount: getPort<'uint32'>('output.status.requiredVertexCount')
        }
      };
      const contributor = new GPULineChunk({
        positions: createTransientView(graph, 'positions', 'float32x2', 6),
        lineOffsets: createTransientView(graph, 'lineOffsets', 'uint32', 3),
        parameters: createTransientView(graph, 'parameters', 'float32', 4),
        mode,
        output
      });
      expect(contributor.lineCapacity).toBe(lineCapacity);
      expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
      expect(description.constraints?.every(constraint => constraint.checked)).toBe(true);
    });
  }
}

for (const mode of ['chunk', 'substring'] as const) {
  for (const missing of [
    'output.measures',
    'output.sourceRows',
    'output.status.requiredCount',
    'output.status.vertexCount',
    'output.status.requiredVertexCount'
  ]) {
    it(`rejects incomplete ${mode} declaration missing ${missing}`, () => {
      const description = describeOutputs(
        {package: '@luma-contributors/analysis/gpu-spatial-analysis', exportName: 'GPULineChunk'},
        mode,
        {lineCount: 2, lineCapacity: 5, vertexCapacity: 12, positionCount: 6}
      );
      expect(() =>
        expectOutputContract(
          {...description, outputs: description.outputs.filter(port => port.path !== missing)},
          mode
        )
      ).toThrow();
    });
  }
}
