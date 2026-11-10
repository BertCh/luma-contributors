// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {
  GPU_RECTANGLE_CLIP_PARAMETER_LENGTH,
  GPURectangleClip
} from '../../../src/gpu-spatial-analysis/rectangle-clip/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;
function view<Format extends 'float32' | 'float32x2' | 'uint32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${serial++}`, format, length);
}

it('GPURectangleClip accepts degrees with straight edges and rejects great-circle edges', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'rectangle-clip'});
  const create = (spatialContext?: GPUSpatialContext) =>
    new GPURectangleClip({
      geometryType: 'lines',
      spatialContext,
      positions: view(graph, 'float32x2', 4),
      lineOffsets: view(graph, 'uint32', 2),
      parameters: view(graph, 'float32', GPU_RECTANGLE_CLIP_PARAMETER_LENGTH),
      output: {
        positions: view(graph, 'float32x2', 8),
        lineOffsets: view(graph, 'uint32', 3),
        status: {count: view(graph, 'uint32', 1), overflow: view(graph, 'uint32', 1)}
      }
    });
  expect(create().spatialContext.coordinateSpace).toBe('planar');
  const degrees: GPUSpatialContext = {coordinateSpace: 'longitude-latitude', metric: 'none'};
  expect(create(degrees).spatialContext).toEqual(degrees);
  expect(() => create({...degrees, metric: 'great-circle', edgeModel: 'great-circle'})).toThrow(
    /edge/
  );
});
