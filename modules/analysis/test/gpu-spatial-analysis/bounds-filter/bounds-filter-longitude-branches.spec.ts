// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/index';
import {expect, it} from 'vitest';
import {
  GPUBoundsFilter,
  getGPUBoundsFilterParameterValues
} from '../../../src/gpu-spatial-analysis/bounds-filter/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';

it('requires equivalent query intervals for measures bounds on different longitude branches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      positions: {
        format: 'float32x2',
        values: Float32Array.of(179, 0, -179, 1, -179, 1, 179, 0, 539, 0, 181, 1)
      },
      lineOffsets: {format: 'uint32', values: Uint32Array.of(0, 2, 4, 6)}
    },
    outputs: {
      ids: {format: 'uint32', length: 3},
      mask: {format: 'uint32', length: 3},
      count: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      bounds: {format: 'float32x4', length: 3}
    },
    create: ({inputs, outputs, parameters}) => {
      const bounds = outputs['bounds'] as GraphDataView<'float32x4'>;
      const measures = new GPUGeometryMeasures({
        geometryType: 'lines',
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          edgeModel: 'linear'
        },
        positions: inputs['positions'] as GraphDataView<'float32x2'>,
        lineOffsets: inputs['lineOffsets'] as GraphDataView<'uint32'>,
        output: {bounds}
      });
      const filter = new GPUBoundsFilter({
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
        bounds,
        parameters,
        output: {
          ids: outputs['ids'] as GraphDataView<'uint32'>,
          outputMask: outputs['mask'] as GraphDataView<'uint32'>,
          status: {
            count: outputs['count'] as GraphDataView<'uint32'>,
            overflow: outputs['overflow'] as GraphDataView<'uint32'>
          }
        }
      });
      return {
        getCommandNodes: graph => [
          ...measures.getCommandNodes(graph),
          ...filter.getCommandNodes(graph)
        ]
      };
    }
  });
  try {
    const masks: number[][] = [];
    for (const turn of [0, 1, 2]) {
      const result = await fixture.run(
        getGPUBoundsFilterParameterValues({
          minX: -179.75 + 360 * turn,
          minY: 0,
          maxX: -179.25 + 360 * turn,
          maxY: 1
        })
      );
      expect(result['overflow']).toEqual([0]);
      expect(result['bounds']).toEqual([179, 0, 181, 1, -181, 0, -179, 1, 539, 0, 541, 1]);
      masks.push(result['mask']);
    }
    expect(masks).toEqual([
      [0, 1, 0],
      [1, 0, 0],
      [0, 0, 1]
    ]);
    expect(masks[0].map((_, row) => Number(masks.some(values => values[row] !== 0)))).toEqual([
      1, 1, 1
    ]);
  } finally {
    fixture.destroy();
  }
});
