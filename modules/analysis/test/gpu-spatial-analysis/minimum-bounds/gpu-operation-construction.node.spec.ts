// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import type {GPUMinimumBounds} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds';
import {
  GPUMinimumBoundsOperation,
  type MinimumBoundsOperationProps
} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds-operation';
import type {GPUConstrainedDelaunayTriangulation} from '../../../src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation';
import {
  GPUConstrainedDelaunayTriangulationOperation,
  type ConstrainedDelaunayTriangulationOperationProps
} from '../../../src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function lowerOperation(
  operation: GPUContributorOperation,
  inputLengths: Readonly<Record<string, number>> = {}
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const views: Record<string, GraphDataView> = {};
  for (const port of Object.keys(operation.inputs)) {
    views[port] = createTransientView(
      graph,
      port,
      port === 'positions' ? 'float32x2' : 'uint32',
      inputLengths[port] ?? (port === 'positions' || port === 'labels' ? 4 : 2)
    );
  }
  for (const [port, output] of Object.entries(operation.outputs)) {
    views[port] = createTransientView(graph, output.value, output.format, output.length);
  }
  try {
    const contributor = operation.createContributor(
      <Format extends GPUGraphVectorFormat>(port: string) => {
        if (!views[port]) throw new Error(`undeclared port ${port}`);
        return views[port] as GraphDataView<Format>;
      }
    );
    return {
      contributor,
      graph,
      views,
      destroy: () => {
        operation.destroy();
        device.destroy();
      }
    };
  } catch (error) {
    operation.destroy();
    device.destroy();
    throw error;
  }
}

function createContext(): GPUSpatialContext {
  return {
    coordinateSpace: 'planar',
    metric: 'native',
    frame: {id: 'original', kind: 'crs', origin: [1, 2]}
  };
}

for (const switchRoute of [false, true]) {
  it(`GPU minimum bounds captures descriptors before lowering (switch route ${switchRoute})`, () => {
    const spatialContext = createContext();
    const props: MinimumBoundsOperationProps = {
      id: 'original',
      spatialContext,
      positions: 'positions',
      labels: 'labels',
      rowCount: 4,
      groupCount: 1,
      maximumHullVertices: 8,
      output: {
        sourceBounds: 'bounds',
        rectangleCorners: 'corners',
        status: {overflow: 'overflow', groupOverflow: 'groupOverflow'}
      }
    };
    const output = props.output;
    const status = output.status;
    const operation = new GPUMinimumBoundsOperation(props);
    Object.assign(props, {
      id: 'changed',
      rowCount: 99,
      groupCount: 2,
      maximumHullVertices: 3,
      positions: 'changed'
    });
    if (switchRoute)
      Object.assign(props, {
        labels: undefined,
        ringOffsets: 'rings',
        polygonOffsets: 'polygons',
        featureOffsets: 'features',
        featureCount: 9
      });
    output.sourceBounds = undefined;
    output.rectangleCorners = 'changed';
    status.groupOverflow = undefined;
    status.overflow = 'changed';
    spatialContext.frame!.id = 'changed';
    spatialContext.frame!.origin![0] = 99;
    const lowered = lowerOperation(operation);
    try {
      const contributor = lowered.contributor as GPUMinimumBounds;
      expect(operation.inputs).toEqual({positions: 'positions', labels: 'labels'});
      expect(operation.outputs['rectangleCorners']).toEqual({
        value: 'corners',
        format: 'float32x2',
        length: 4
      });
      expect(operation.outputs['sourceBounds'].length).toBe(1);
      expect(contributor.props.id).toBe('original');
      expect(contributor.props).toMatchObject({groupCount: 1, maximumHullVertices: 8});
      expect(contributor.props.spatialContext?.frame).toMatchObject({
        id: 'original',
        origin: [1, 2]
      });
      expect(contributor.props.output.sourceBounds).toBe(lowered.views['sourceBounds']);
      expect(contributor.props.output.rectangleCorners).toBe(lowered.views['rectangleCorners']);
      expect(contributor.props.output.status.groupOverflow).toBe(lowered.views['groupOverflow']);
      recordWGSLKernelNodes(lowered.graph);
      contributor.getCommandNodes(lowered.graph);
      const publication = getWGSLKernelNodeRecord(lowered.graph, 'original-hull-publish-status');
      expect(publication?.bindings).toEqual(
        expect.arrayContaining([
          {name: 'overflow', view: lowered.views['overflow'], type: 'u32', access: 'read_write'},
          {
            name: 'groupOverflow',
            view: lowered.views['groupOverflow'],
            type: 'u32',
            access: 'read_write'
          }
        ])
      );
    } finally {
      lowered.destroy();
    }
  });
}

it('GPU constrained triangulation captures options, context and optional outputs before lowering', () => {
  const spatialContext = createContext();
  const props: ConstrainedDelaunayTriangulationOperationProps = {
    id: 'original',
    spatialContext,
    positions: 'positions',
    featureOffsets: 'features',
    polygonOffsets: 'polygons',
    ringOffsets: 'rings',
    polygonCount: 1,
    indexCapacity: 6,
    maximumWork: 64,
    useZOrderHash: true,
    largePolygonMethod: 'none',
    monotoneMinimumVertices: 8,
    monotoneMaximumVertices: 64,
    maximumFlips: 7,
    flipMethod: 'serial',
    maximumVerticesPerPolygon: 8,
    output: {
      indices: 'indices',
      valid: 'valid',
      triangleCount: 'count',
      triangleOffsets: 'offsets',
      status: {converged: 'converged'}
    }
  };
  const output = props.output;
  const operation = new GPUConstrainedDelaunayTriangulationOperation(props);
  Object.assign(props, {
    polygonCount: 99,
    indexCapacity: 3,
    maximumWork: 1,
    useZOrderHash: false,
    largePolygonMethod: 'monotone',
    monotoneMinimumVertices: 9,
    monotoneMaximumVertices: 128,
    maximumFlips: 1,
    flipMethod: 'parallel',
    maximumVerticesPerPolygon: 16,
    positions: 'changed'
  });
  output.triangleCount = undefined;
  output.triangleOffsets = undefined;
  output.status.converged = 'changed';
  spatialContext.frame!.id = 'changed';
  spatialContext.frame!.origin![0] = 99;
  const lowered = lowerOperation(operation);
  try {
    const contributor = lowered.contributor as GPUConstrainedDelaunayTriangulation;
    expect(operation.outputs['indices'].length).toBe(6);
    expect(operation.outputs['converged']).toEqual({
      value: 'converged',
      format: 'uint32',
      length: 1
    });
    expect(contributor.props).toMatchObject({
      maximumWork: 64,
      useZOrderHash: true,
      largePolygonMethod: 'none',
      monotoneMinimumVertices: 8,
      monotoneMaximumVertices: 64,
      maximumFlips: 7,
      flipMethod: 'serial',
      maximumVerticesPerPolygon: 8
    });
    expect(contributor.props.spatialContext?.frame).toMatchObject({id: 'original', origin: [1, 2]});
    expect(contributor.props.output.triangleCount).toBe(lowered.views['triangleCount']);
    expect(contributor.props.output.triangleOffsets).toBe(lowered.views['triangleOffsets']);
    expect(contributor.props.output.status.converged).toBe(lowered.views['converged']);
  } finally {
    lowered.destroy();
  }
});

for (const route of ['labels', 'polygons'] as const) {
  it(`GEO-126 rejects bound positions that disagree with captured rowCount on ${route}`, () => {
    const props: MinimumBoundsOperationProps = {
      id: 'bounds-bind',
      positions: 'positions',
      rowCount: 5,
      ...(route === 'labels'
        ? {labels: 'labels', groupCount: 2}
        : {
            ringOffsets: 'rings',
            polygonOffsets: 'polygons',
            featureOffsets: 'features',
            featureCount: 2
          }),
      output: {sourceBounds: 'bounds', status: {overflow: 'overflow'}}
    };
    const operation = new GPUMinimumBoundsOperation(props);
    props.rowCount = 7; // A later mutation must not legalize an incompatible binding.
    expect(() => lowerOperation(operation, {positions: 7, labels: 7, featureOffsets: 3})).toThrow(
      'bounds-bind positions length must equal rowCount (5)'
    );
  });
}

it('GEO-126 rejects bound feature offsets that disagree with captured featureCount', () => {
  const props: MinimumBoundsOperationProps = {
    id: 'bounds-bind',
    positions: 'positions',
    rowCount: 5,
    ringOffsets: 'rings',
    polygonOffsets: 'polygons',
    featureOffsets: 'features',
    featureCount: 2,
    output: {sourceBounds: 'bounds', status: {overflow: 'overflow'}}
  };
  const operation = new GPUMinimumBoundsOperation(props);
  props.featureCount = 1;
  expect(() => lowerOperation(operation, {positions: 5, featureOffsets: 2})).toThrow(
    'bounds-bind featureOffsets length must equal featureCount + 1 (3)'
  );
});

it('GEO-126 accepts matching packed dimensions for two captured polygon features', () => {
  const props: MinimumBoundsOperationProps = {
    positions: 'positions',
    rowCount: 5,
    ringOffsets: 'rings',
    polygonOffsets: 'polygons',
    featureOffsets: 'features',
    featureCount: 2,
    output: {sourceBounds: 'bounds', status: {overflow: 'overflow'}}
  };
  const operation = new GPUMinimumBoundsOperation(props);
  props.rowCount = 99;
  props.featureCount = 99;
  const lowered = lowerOperation(operation, {positions: 5, featureOffsets: 3});
  try {
    expect(operation.outputs['sourceBounds'].length).toBe(2);
    expect((lowered.contributor as GPUMinimumBounds).groupCount).toBe(2);
    expect((lowered.contributor as GPUMinimumBounds).props.positions.length).toBe(5);
  } finally {
    lowered.destroy();
  }
});
