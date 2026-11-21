// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUMinimumBounds} from '../../../src/gpu-spatial-analysis/minimum-bounds/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

const LONGITUDE_LATITUDE: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters'
};
const LOCAL_FRAME: GPUSpatialContext = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'native',
  frame: {id: 'test-frame', kind: 'local-azimuthal-equidistant', origin: [10, 50]}
};
const UNSUPPORTED = /does not support longitude-latitude coordinates/;

let serial = 0;
function view<
  Format extends 'float32' | 'float32x2' | 'float32x3' | 'float32x4' | 'uint32' | 'uint32x3'
>(graph: GPUCommandGraph, format: Format, length: number) {
  return createTransientView(graph, `view-${serial++}`, format, length);
}

it('GPUMinimumBounds is planar only', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'minimum-bounds'});
  const create = (spatialContext?: GPUSpatialContext) =>
    new GPUMinimumBounds({
      spatialContext,
      positions: view(graph, 'float32x2', 8),
      ringOffsets: view(graph, 'uint32', 3),
      polygonOffsets: view(graph, 'uint32', 3),
      featureOffsets: view(graph, 'uint32', 3),
      output: {
        sourceBounds: view(graph, 'float32x4', 2),
        status: {overflow: view(graph, 'uint32', 1)}
      }
    });
  expect(() => create(LONGITUDE_LATITUDE)).toThrow(UNSUPPORTED);
  expect(create().spatialContext.coordinateSpace).toBe('planar');
  expect(create(LOCAL_FRAME).spatialContext.frame?.id).toBe('test-frame');
});

// GEO-106: the circle kernel uses the scaled circumcircle shared with the hull and Voronoi
// kernels, not the unscaled formula; the numeric 2^+-40 check is the headless case "circles are
// the circumcircle at extents 2^40, 2^33, 2^-40 and 2^-45".
it('GPUMinimumBounds circles use the shared scaled circumcircle', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'minimum-bounds-circle'});
  const nodes = new GPUMinimumBounds({
    positions: view(graph, 'float32x2', 8),
    labels: view(graph, 'uint32', 8),
    groupCount: 2,
    output: {circles: view(graph, 'float32x3', 2), status: {overflow: view(graph, 'uint32', 1)}}
  }).getCommandNodes(graph);
  const circleSources = getGPUCommandNodeKernelSources(nodes).filter(({source}) =>
    source.includes('fn getCircumcenter(')
  );
  expect(circleSources).toHaveLength(1);
  const [{source, constants}] = circleSources;
  expect(source).toContain('let circle = getTriangleCircumcircle(a, b, c);');
  expect(source).not.toContain('(ac.y * abSquared - ab.y * acSquared) / determinant');
  expect(auditWGSLModule(source, constants)).toEqual([]);
});
