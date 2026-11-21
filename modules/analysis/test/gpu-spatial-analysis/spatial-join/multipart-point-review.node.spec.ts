// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {GPUBufferSelection} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import {addPointsInPolygonsChoroplethRecipe} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function fixture(queries = 2, polygonCount = 2) {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `review-${serial++}`, format, length);
  const points = {
    kind: 'points' as const,
    positions: view('float32x2', queries),
    featureOffsets: view('uint32', queries + 1)
  };
  const polygons = {
    kind: 'polygons' as const,
    positions: view('float32x2', 4 * polygonCount),
    featureOffsets: view('uint32', polygonCount + 1),
    polygonOffsets: view('uint32', polygonCount + 1),
    ringOffsets: view('uint32', polygonCount + 1)
  };
  return {device, graph, view, points, polygons};
}
it('review R3 refuses strided multipart values before graph mutation', () => {
  const {device, graph, view, points, polygons} = fixture();
  const backing = view('float32', 4),
    values = graph.createDataView(backing.buffer, {format: 'float32', length: 2, byteStride: 8});
  const add = vi.spyOn(graph, 'add');
  try {
    expect(() =>
      addPointsInPolygonsChoroplethRecipe(graph, {
        points,
        polygons: {...polygons, candidateCapacity: 8},
        values,
        statistic: 'sum'
      })
    ).toThrow(/values.*packed|values.*stride/);
    expect(add).not.toHaveBeenCalled();
  } finally {
    add.mockRestore();
    device.destroy();
  }
});
it('review R3 refuses a short multipart featureValues output before graph mutation', () => {
  const {device, graph, view, points, polygons} = fixture();
  const add = vi.spyOn(graph, 'add');
  try {
    expect(() =>
      addPointsInPolygonsChoroplethRecipe(graph, {
        points,
        polygons: {...polygons, candidateCapacity: 8},
        values: view('float32', 2),
        statistic: 'sum',
        output: {featureValues: view('float32', 1)}
      })
    ).toThrow(/featureValues.*2|featureValues.*polygon/);
    expect(add).not.toHaveBeenCalled();
  } finally {
    add.mockRestore();
    device.destroy();
  }
});
it('review R3 refuses strided output and physically shared value/output bindings', () => {
  const {device, graph, view, points, polygons} = fixture();
  const backing = view('float32', 4),
    values = view('float32', 2);
  try {
    const strided = graph.createDataView(backing.buffer, {
      format: 'float32',
      length: 2,
      byteStride: 8
    });
    expect(() =>
      addPointsInPolygonsChoroplethRecipe(graph, {
        points,
        polygons: {...polygons, candidateCapacity: 8},
        values,
        statistic: 'sum',
        output: {featureValues: strided}
      })
    ).toThrow(/featureValues.*packed|featureValues.*stride/);
    expect(() =>
      addPointsInPolygonsChoroplethRecipe(graph, {
        points,
        polygons: {...polygons, candidateCapacity: 8},
        values,
        statistic: 'sum',
        output: {featureValues: values}
      })
    ).toThrow(/share buffers/);
  } finally {
    device.destroy();
  }
});
it('review R8 assignment-only declaration does not allocate a feature-polygon matrix', () => {
  const {device, graph, view, points, polygons} = fixture(8192, 4097);
  try {
    const join = new GPUPointInPolygonJoin({
      points,
      polygons,
      candidateCapacity: 1,
      spatialSort: false,
      querySort: false,
      edgeSlabs: false,
      output: {pointFeatureIds: view('uint32', 8192), status: {overflow: view('uint32', 1)}}
    });
    expect(() => join.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});
it('review R9 rejected leaf capacity permits a same-ID corrected buffer declaration', () => {
  const {device, graph, view, points} = fixture();
  const props = {
    id: 'review-retry',
    points,
    features: {kind: 'points' as const, positions: view('float32x2', 2)},
    radius: view('float32', 1),
    candidateCapacity: 8,
    output: {outputMask: view('uint32', 2)}
  };
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() =>
      new GPUBufferSelection({...props, leafCapacity: 2 ** 24}).getCommandNodes(graph)
    ).toThrow(/leafCapacity|node table|binding/);
    expect(allocate).not.toHaveBeenCalled();
    expect(() =>
      new GPUBufferSelection({...props, leafCapacity: 2}).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    allocate.mockRestore();
    device.destroy();
  }
});
it('review R10 refuses physically shared membership/count buffers and permits separate buffers', () => {
  const {device, graph, view, points, polygons} = fixture();
  const shared = view('uint32', 6);
  const membership = graph.createDataView(shared.buffer, {
      format: 'uint32',
      length: 4,
      byteOffset: 0
    }),
    counts = graph.createDataView(shared.buffer, {format: 'uint32', length: 2, byteOffset: 16});
  const props = {
    points,
    polygons,
    candidateCapacity: 8,
    output: {
      pointFeatureIds: view('uint32', 2),
      featurePolygonMembership: membership,
      polygonMembershipCounts: counts,
      status: {overflow: view('uint32', 1)}
    }
  };
  try {
    expect(() => new GPUPointInPolygonJoin(props)).toThrow(/read|writ/);
    expect(() =>
      new GPUPointInPolygonJoin({
        ...props,
        output: {...props.output, polygonMembershipCounts: view('uint32', 2)}
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});
