// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPULabelPointOperation} from '../../../src/gpu-spatial-analysis/label-point/gpu-label-point-operation';
import {GPUMinimumBoundsOperation} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds-operation';
import {GPUShapeDescriptorsOperation} from '../../../src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

for (const smallFeatureVertexLimit of [0, 32]) {
  it(`B1 B2 publishes all label columns for a translated diamond (vertex limit ${smallFeatureVertexLimit})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const center = 1048576.125,
      radius = 0.125;
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {
          format: 'float32x2',
          values: new Float32Array([
            center + radius,
            center,
            center,
            center + radius,
            center - radius,
            center,
            center,
            center - radius
          ])
        },
        ringOffsets: {format: 'uint32', values: new Uint32Array([0, 4])},
        polygonOffsets: {format: 'uint32', values: new Uint32Array([0, 1])},
        featureOffsets: {format: 'uint32', values: new Uint32Array([0, 1])}
      },
      operations: [
        new GPULabelPointOperation({
          positions: 'positions',
          ringOffsets: 'ringOffsets',
          polygonOffsets: 'polygonOffsets',
          featureOffsets: 'featureOffsets',
          featureCount: 1,
          smallFeatureVertexLimit,
          output: {
            points: 'points',
            publishedCenterClearance: 'clearance',
            precisionLimited: 'precisionLimited',
            degenerate: 'degenerate'
          }
        })
      ],
      outputs: ['points', 'clearance', 'precisionLimited', 'degenerate']
    });
    expect(Array.from(result['points'])).toEqual([center, center]);
    // Perpendicular distance to either side of the diamond: .125/sqrt(2).
    expect(Array.from(result['clearance'])).toEqual([Math.fround(0.125 / Math.SQRT2)]);
    expect(Array.from(result['precisionLimited'])).toEqual([0]);
    expect(Array.from(result['degenerate'])).toEqual([0]);
  });

  it(`B5 falls back from exterior label publication (vertex limit ${smallFeatureVertexLimit})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const origin = 1048576;
    const positions = new Float32Array(
      [0, 0, 3, 1, 4, 1, 1, 0].map(value => origin + value * 0.125)
    );
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: positions},
        ringOffsets: {format: 'uint32', values: new Uint32Array([0, 4])},
        polygonOffsets: {format: 'uint32', values: new Uint32Array([0, 1])},
        featureOffsets: {format: 'uint32', values: new Uint32Array([0, 1])}
      },
      operations: [
        new GPULabelPointOperation({
          positions: 'positions',
          ringOffsets: 'ringOffsets',
          polygonOffsets: 'polygonOffsets',
          featureOffsets: 'featureOffsets',
          featureCount: 1,
          smallFeatureVertexLimit,
          output: {
            points: 'points',
            publishedCenterClearance: 'clearance',
            precisionLimited: 'precisionLimited',
            degenerate: 'degenerate'
          }
        })
      ],
      outputs: ['points', 'clearance', 'precisionLimited', 'degenerate']
    });
    expect(result['points']).toHaveLength(2);
    expect(
      Array.from({length: 4}, (_, row) => Array.from(positions.slice(row * 2, row * 2 + 2)))
    ).toContainEqual(Array.from(result['points']));
    expect(Array.from(result['clearance'])).toEqual([0]);
    expect(Array.from(result['precisionLimited'])).toEqual([1]);
    expect(Array.from(result['degenerate'])).toEqual([1]);
  });
}

it('B8 executes zero features and one empty feature on owned GPU forms', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const featureCount of [0, 1]) {
    const shared = {
      positions: 'positions',
      ringOffsets: 'rings',
      polygonOffsets: 'polygons',
      featureOffsets: 'features',
      featureCount
    };
    const inputs = {
      positions: {format: 'float32x2' as const, values: new Float32Array()},
      rings: {format: 'uint32' as const, values: new Uint32Array([0])},
      polygons: {format: 'uint32' as const, values: new Uint32Array([0])},
      features: {format: 'uint32' as const, values: new Uint32Array(featureCount + 1)},
      parameters: {format: 'float32' as const, values: new Float32Array([0.05, 0, 0, 0])}
    };
    const label = await runOperationGraph(device, {
      inputs,
      operations: [
        new GPULabelPointOperation({
          ...shared,
          output: {
            points: 'points',
            publishedCenterClearance: 'clearance',
            degenerate: 'degenerate'
          }
        })
      ],
      outputs: ['points', 'clearance', 'degenerate']
    });
    expect(Array.from(label['points'])).toEqual(Array(featureCount * 2).fill(NaN));
    expect(Array.from(label['clearance'])).toEqual(Array(featureCount).fill(NaN));
    expect(Array.from(label['degenerate'])).toEqual(Array(featureCount).fill(1));
    const bounds = await runOperationGraph(device, {
      inputs,
      operations: [
        new GPUMinimumBoundsOperation({
          ...shared,
          rowCount: 0,
          output: {sourceBounds: 'bounds', hullSizes: 'hullSizes', status: {overflow: 'overflow'}}
        })
      ],
      outputs: ['bounds', 'hullSizes', 'overflow']
    });
    expect(Array.from(bounds['bounds'])).toEqual(Array(featureCount * 4).fill(NaN));
    expect(Array.from(bounds['hullSizes'])).toEqual(Array(featureCount).fill(0));
    expect(Array.from(bounds['overflow'])).toEqual([0]);
    const shape = await runOperationGraph(device, {
      inputs,
      operations: [
        new GPUShapeDescriptorsOperation({
          ...shared,
          parameters: 'parameters',
          output: {areas: 'areas', orientation: 'orientation'}
        })
      ],
      outputs: ['areas', 'orientation']
    });
    expect(Array.from(shape['areas'])).toEqual(Array(featureCount).fill(0));
    expect(Array.from(shape['orientation'])).toEqual(Array(featureCount).fill(NaN));
  }
});
