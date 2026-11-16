// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  createCompactnessNode,
  createMomentsNode,
  createOrientationNode
} from '../../../src/gpu-spatial-analysis/shape-descriptors/shape-descriptors-kernels';
import {getShapeMomentTileNodes} from '../../../src/gpu-spatial-analysis/shape-descriptors/shape-moment-tiles';
import {GPUShapeDescriptors} from '../../../src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('keeps closed-form elongation and orientation for huge and tiny finite polygon moments', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const scales = [1, 1e24, 1e-30, 4e37, 1e-37];
  const values = new Float32Array(
    scales.flatMap(scale => [
      // Axis-aligned and 45-degree rotated rectangles, moment eigenvalues 4 and 1.
      4 * scale,
      scale,
      0,
      -scale,
      2.5 * scale,
      2.5 * scale,
      1.5 * scale,
      scale
    ])
  );
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const rowCount = values.length / 4;
  const moments = track(createInputBuffer(device, values));
  const elongation = track(createOutputBuffer(device, rowCount));
  const orientation = track(createOutputBuffer(device, rowCount));
  const clockwise = track(createOutputBuffer(device, rowCount));
  graph.add(
    createOrientationNode(graph, {
      id: 'orientation',
      operation: 'GPUShapeDescriptors',
      featureCount: rowCount,
      moments: importGraphBuffer(graph, 'moments', moments, 'float32x4', rowCount),
      elongation: importGraphBuffer(graph, 'elongation', elongation, 'float32', rowCount),
      orientation: importGraphBuffer(graph, 'angles', orientation, 'float32', rowCount),
      clockwise: importGraphBuffer(graph, 'clockwise', clockwise, 'uint32', rowCount)
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const elongations = await readFloat32(elongation, rowCount);
    const orientations = await readFloat32(orientation, rowCount);
    expect(await readUint32(clockwise, rowCount)).toEqual(
      Array.from({length: rowCount}, (_, row) => (row % 2 === 0 ? 1 : 0))
    );
    for (let row = 0; row < rowCount; row++) {
      expect(elongations[row]).toBeCloseTo(0.5, 5);
      expect(orientations[row]).toBeCloseTo(row % 2 === 0 ? 0 : Math.PI / 4, 5);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('keeps square compactness finite when four pi times area overflows float32', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const areas = track(createInputBuffer(device, new Float32Array([1, 1e38, 1e-30])));
  const perimeters = track(createInputBuffer(device, new Float32Array([4, 4e19, 4e-15])));
  const parameters = track(createInputBuffer(device, new Float32Array([0.05, 0, 0, 0])));
  const polsbyPopper = track(createOutputBuffer(device, 3));
  const schwartzberg = track(createOutputBuffer(device, 3));
  graph.add(
    createCompactnessNode(graph, {
      id: 'compactness',
      operation: 'GPUShapeDescriptors',
      featureCount: 3,
      areas: importGraphBuffer(graph, 'areas', areas, 'float32', 3),
      perimeters: importGraphBuffer(graph, 'perimeters', perimeters, 'float32', 3),
      parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 4),
      polsbyPopper: importGraphBuffer(graph, 'polsby-popper', polsbyPopper, 'float32', 3),
      schwartzberg: importGraphBuffer(graph, 'schwartzberg', schwartzberg, 'float32', 3)
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const value of await readFloat32(polsbyPopper, 3))
      expect(value).toBeCloseTo(Math.PI / 4, 5);
    for (const value of await readFloat32(schwartzberg, 3))
      expect(value).toBeCloseTo(Math.sqrt(Math.PI / 4), 5);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

for (const momentMethod of ['serial', 'tiled'] as const) {
  it(`computes full polygon descriptors at extreme scales through ${momentMethod} moments and packed outputs`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const scales = [1e-7, 1, 4e9, 4.2e9];
    const featureCount = scales.length;
    const coordinates = new Float32Array(
      scales.flatMap(scale => [0, 0, 2 * scale, 0, 2 * scale, scale, 0, scale])
    );
    const buffers: Buffer[] = [];
    const track = (buffer: Buffer) => {
      buffers.push(buffer);
      return buffer;
    };
    const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      values: Uint32Array | Float32Array,
      format: Format,
      length = values.length
    ) => importGraphBuffer(graph, name, track(createInputBuffer(device, values)), format, length);
    // Separate passes write area and orientation to disjoint logical slices inside one alignment window.
    const packed = track(createOutputBuffer(device, 2 * featureCount));
    const backing = importGraphBuffer(graph, 'packed', packed, 'float32', 2 * featureCount);
    const elongations = track(createOutputBuffer(device, featureCount));
    const polygonInputs = {
      positions: input('positions', coordinates, 'float32x2', 4 * featureCount),
      ringOffsets: input(
        'rings',
        Uint32Array.from({length: featureCount + 1}, (_, row) => 4 * row),
        'uint32'
      ),
      polygonOffsets: input(
        'polygons',
        Uint32Array.from({length: featureCount + 1}, (_, row) => row),
        'uint32'
      ),
      featureOffsets: input(
        'features',
        Uint32Array.from({length: featureCount + 1}, (_, row) => row),
        'uint32'
      )
    };
    graph.add(
      new GPUShapeDescriptors({
        momentMethod,
        ...polygonInputs,
        parameters: input('parameters', new Float32Array([0.05, 0, 0, 0]), 'float32'),
        output: {
          areas: graph.createDataView(backing.buffer, {format: 'float32', length: featureCount}),
          orientation: graph.createDataView(backing.buffer, {
            format: 'float32',
            byteOffset: 4 * featureCount,
            length: featureCount
          }),
          elongation: importGraphBuffer(graph, 'elongation', elongations, 'float32', featureCount)
        }
      })
    );
    const rawMoments = track(createOutputBuffer(device, 4 * featureCount));
    graph.add(
      (momentMethod === 'serial' ? createMomentsNode : getShapeMomentTileNodes)(graph, {
        ...polygonInputs,
        id: 'known-centroid-moments',
        operation: 'GPUShapeDescriptors',
        featureCount,
        centroids: input(
          'known-centroids',
          Float32Array.from(scales.flatMap(scale => [scale, scale / 2])),
          'float32x2',
          featureCount
        ),
        moments: importGraphBuffer(graph, 'raw-moments', rawMoments, 'float32x4', featureCount)
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const result = await readFloat32(packed, 2 * featureCount);
      const ratios = await readFloat32(elongations, featureCount);
      const integrals = await readFloat32(rawMoments, 4 * featureCount);
      for (let feature = 0; feature < featureCount; feature++) {
        const fourthPower = scales[feature] ** 4;
        expect(
          integrals[4 * feature] / ((2 / 3) * fourthPower),
          `raw xx at scale ${scales[feature]}`
        ).toBeCloseTo(1, 4);
        expect(
          integrals[4 * feature + 1] / ((1 / 6) * fourthPower),
          `raw yy at scale ${scales[feature]}`
        ).toBeCloseTo(1, 4);
        expect(
          Math.abs(integrals[4 * feature + 2]) / fourthPower,
          `raw xy at scale ${scales[feature]}`
        ).toBeLessThan(2e-6);
        expect(
          result[feature] / (2 * scales[feature] ** 2),
          `area at scale ${scales[feature]}`
        ).toBeCloseTo(1, 4);
        expect(
          result[feature + featureCount],
          `orientation at scale ${scales[feature]}`
        ).toBeCloseTo(0, 4);
        expect(ratios[feature], `elongation at scale ${scales[feature]}`).toBeCloseTo(0.5, 4);
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
