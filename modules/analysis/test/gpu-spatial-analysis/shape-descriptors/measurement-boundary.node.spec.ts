// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {
  GPUShapeDescriptors,
  type GPUShapeDescriptorsOutput
} from '../../../src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors';
import {
  GPUShapeDescriptorsOperation,
  type ShapeDescriptorsOperationProps
} from '../../../src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors-operation';
import {CPUShapeDescriptorsOperation} from '../../../src/gpu-spatial-analysis/shape-descriptors/cpu-shape-descriptors-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const NAMED: ShapeDescriptorsOperationProps = {
  id: 'measurement-boundary',
  positions: 'positions',
  ringOffsets: 'rings',
  polygonOffsets: 'polygons',
  featureOffsets: 'features',
  parameters: 'parameters',
  featureCount: 1,
  output: {areas: 'areas'}
};

function createRaw(
  spatialContext: GPUSpatialContext,
  column:
    | 'areas'
    | 'perimeters'
    | 'polsbyPopper'
    | 'schwartzberg'
    | 'sliver'
    | 'convexity'
    | 'orientation'
) {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    return new GPUShapeDescriptors({
      id: NAMED.id,
      spatialContext,
      positions: createTransientView(graph, 'positions', 'float32x2', 3),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      parameters: createTransientView(graph, 'parameters', 'float32', 4),
      output: {
        [column]: createTransientView(
          graph,
          'output',
          column === 'sliver' ? 'uint32' : 'float32',
          1
        )
      } as GPUShapeDescriptorsOutput
    });
  } finally {
    device.destroy();
  }
}

for (const frame of [
  {id: 'registry', kind: 'crs' as const, crs: 'EPSG:4326'},
  {id: 'partial', kind: 'crs' as const, crs: 'EPSG:4326', unit: {name: 'degrees'}},
  {id: 'explicit', kind: 'crs' as const, unit: {name: 'degrees', degreesPerUnit: 1}}
]) {
  it(`UNI-02 rejects selected angular measurements at every boundary (${frame.id})`, () => {
    const spatialContext: GPUSpatialContext = {coordinateSpace: 'planar', metric: 'none', frame};
    for (const column of [
      'areas',
      'perimeters',
      'polsbyPopper',
      'schwartzberg',
      'sliver',
      'convexity'
    ] as const) {
      expect(() => createRaw(spatialContext, column)).toThrow(
        /measurement-boundary.*project positions to a planar linear frame/
      );
      for (const Operation of [GPUShapeDescriptorsOperation, CPUShapeDescriptorsOperation]) {
        expect(
          () =>
            new Operation({
              ...NAMED,
              spatialContext,
              output: {[column]: 'result'} as ShapeDescriptorsOperationProps['output']
            })
        ).toThrow(/measurement-boundary.*project positions to a planar linear frame/);
      }
    }
    expect(() => createRaw(spatialContext, 'orientation')).not.toThrow();
    for (const Operation of [GPUShapeDescriptorsOperation, CPUShapeDescriptorsOperation]) {
      expect(
        () => new Operation({...NAMED, spatialContext, output: {orientation: 'orientation'}})
      ).not.toThrow();
    }
  });
}

it('UNI-02 retains projected-meter measurement construction and the closed-form triangle area', async () => {
  const spatialContext: GPUSpatialContext = {
    coordinateSpace: 'planar',
    metric: 'none',
    frame: {id: 'meters', kind: 'crs', crs: 'EPSG:3857'}
  };
  expect(() => createRaw(spatialContext, 'areas')).not.toThrow();
  expect(() => new GPUShapeDescriptorsOperation({...NAMED, spatialContext})).not.toThrow();
  const operation = new CPUShapeDescriptorsOperation({...NAMED, spatialContext});
  const rows = (
    format: 'float32x2' | 'float32' | 'uint32',
    values: Float32Array | Uint32Array,
    length: number
  ) => ({format, length, chunks: [{values, length}], pack: () => values});
  const result = await operation.execute({
    inputs: {
      positions: rows('float32x2', Float32Array.of(0, 0, 2, 0, 0, 2), 3),
      ringOffsets: rows('uint32', Uint32Array.of(0, 3), 2),
      polygonOffsets: rows('uint32', Uint32Array.of(0, 1), 2),
      featureOffsets: rows('uint32', Uint32Array.of(0, 1), 2),
      parameters: rows('float32', Float32Array.of(0.1, 0, 0, 0), 4)
    }
  });
  // A right triangle with legs 2 meters has area 2 * 2 / 2 = 2 square meters.
  expect(Array.from(result['areas'] as Float32Array)).toEqual([2]);
});
