// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUGraphVectorFormat} from '@luma.gl/graph/gpu-core';
import {AnalysisValidationError} from '../../../src/analysis-core/kernels/analysis-validation-error';
import {validateRefinementNumber} from '../../../src/gpu-spatial-analysis/geometry-refinement/refinement-output-validation';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  GPUCanonicalPolygonEquality,
  GPUConvexStraightSkeleton,
  GPUMaximumInscribedCircle,
  GPUNormalizeGeometry,
  GPUNormalizePolygons,
  GPUPolygonPrecisionReduction,
  GPULinePrecisionReduction,
  GPURatioConcaveHull,
  GPUVectorStraightSkeleton
} from '../../../src/gpu-spatial-analysis/geometry-refinement/index';
import {createTopologyOutput} from './topology-output';

it('all nine refinement producers name the instance, output path, expected and received rows', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends GPUGraphVectorFormat>(format: Format, length: number) =>
    createTransientView(graph, `validation-${serial++}`, format, length);
  const geometry = {
    positions: view('float32x2', 4),
    ringOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    featureOffsets: view('uint32', 2)
  };
  const id = 'my-layer';
  const cases = [
    {
      path: 'output',
      expected: 1,
      received: 0,
      build: () =>
        new GPUCanonicalPolygonEquality({
          id,
          left: geometry,
          right: geometry,
          output: view('uint32', 0)
        })
    },
    ...(['segments', 'sourceEdges', 'offsetPositions', 'status'] as const).map(path => ({
      path: `output.${path}`,
      expected: path === 'status' ? 1 : 8,
      received: path === 'status' ? 0 : 7,
      build: () =>
        new GPUConvexStraightSkeleton({
          id,
          positions: geometry.positions,
          ringOffsets: geometry.ringOffsets,
          maximumVertices: 8,
          segmentCapacity: 8,
          offsetDistance: 1,
          output: {
            segments: view('float32x4', path === 'segments' ? 7 : 8),
            sourceEdges: view('uint32x2', path === 'sourceEdges' ? 7 : 8),
            offsetPositions: view('float32x2', path === 'offsetPositions' ? 7 : 8),
            status: view('uint32x4', path === 'status' ? 0 : 1)
          }
        })
    })),
    ...(['circles', 'witnesses', 'status'] as const).map(path => ({
      path: `output.${path}`,
      expected: 1,
      received: 0,
      build: () =>
        new GPUMaximumInscribedCircle({
          id,
          ...geometry,
          tolerance: 1,
          output: {
            circles: view('float32x4', path === 'circles' ? 0 : 1),
            witnesses: view('float32x2', path === 'witnesses' ? 0 : 1),
            status: view('uint32', path === 'status' ? 0 : 1)
          }
        })
    })),
    ...(['positions', 'children', 'status'] as const).map(path => ({
      path: `output.${path}`,
      expected: path === 'positions' ? 4 : 1,
      received: path === 'positions' ? 3 : 0,
      build: () =>
        new GPUNormalizeGeometry({
          id,
          positions: geometry.positions,
          nodes: view('uint32x4', 1),
          children: view('uint32', 1),
          output: {
            positions: view('float32x2', path === 'positions' ? 3 : 4),
            children: view('uint32', path === 'children' ? 0 : 1),
            status: view('uint32', path === 'status' ? 0 : 1)
          }
        })
    })),
    ...(['positions', 'ringOffsets', 'polygonOffsets'] as const).map(path => ({
      path: `output.${path}`,
      expected: geometry[path].length,
      received: geometry[path].length - 1,
      build: () =>
        new GPUNormalizePolygons({
          id,
          ...geometry,
          output: {
            positions: view('float32x2', path === 'positions' ? 3 : 4),
            ringOffsets: view('uint32', path === 'ringOffsets' ? 1 : 2),
            polygonOffsets: view('uint32', path === 'polygonOffsets' ? 1 : 2)
          }
        })
    })),
    {
      path: 'lattice',
      expected: 4,
      received: 3,
      build: () =>
        new GPUPolygonPrecisionReduction({
          id,
          polygons: {kind: 'polygons', ...geometry},
          gridSize: 1,
          lattice: view('float32', 3),
          intersectionCapacity: 4,
          output: createTopologyOutput(view)
        })
    },
    ...(['positions', 'lineOffsets', 'status'] as const).map(path => ({
      path: `output.${path}`,
      expected: path === 'positions' ? 4 : path === 'lineOffsets' ? 2 : 1,
      received: path === 'positions' ? 3 : path === 'lineOffsets' ? 1 : 0,
      build: () =>
        new GPULinePrecisionReduction({
          id,
          positions: geometry.positions,
          lineOffsets: geometry.ringOffsets,
          gridSize: 1,
          mode: 'valid-output',
          output: {
            positions: view('float32x2', path === 'positions' ? 3 : 4),
            lineOffsets: view('uint32', path === 'lineOffsets' ? 1 : 2),
            status: view('uint32', path === 'status' ? 0 : 1)
          }
        })
    })),
    ...(
      [
        'triangles',
        'triangleValidity',
        'boundaryEdges',
        'edgeValidity',
        'status.converged'
      ] as const
    ).map(path => ({
      path: `output.${path}`,
      expected:
        path === 'status.converged'
          ? 1
          : path === 'triangles' || path === 'triangleValidity'
            ? 3
            : 9,
      received:
        path === 'status.converged'
          ? 0
          : path === 'triangles' || path === 'triangleValidity'
            ? 2
            : 8,
      build: () =>
        new GPURatioConcaveHull({
          id,
          positions: geometry.positions,
          ratio: 1,
          output: {
            triangles: view('uint32x3', path === 'triangles' ? 2 : 3),
            triangleValidity: view('uint32', path === 'triangleValidity' ? 2 : 3),
            boundaryEdges: view('uint32x2', path === 'boundaryEdges' ? 8 : 9),
            edgeValidity: view('uint32', path === 'edgeValidity' ? 8 : 9),
            status: {converged: view('uint32', path === 'status.converged' ? 0 : 1)}
          }
        })
    })),
    ...(['segments', 'sourceEdges', 'offsetPositions', 'offsetLinks', 'status'] as const).map(
      path => ({
        path: `output.${path}`,
        expected: path === 'status' ? 1 : 8,
        received: path === 'status' ? 0 : 7,
        build: () =>
          new GPUVectorStraightSkeleton({
            id,
            ...geometry,
            maximumVertices: 8,
            maximumEvents: 0,
            segmentCapacity: 8,
            offsetDistance: 1,
            output: {
              segments: view('float32x4', path === 'segments' ? 7 : 8),
              sourceEdges: view('uint32x2', path === 'sourceEdges' ? 7 : 8),
              offsetPositions: view('float32x2', path === 'offsetPositions' ? 7 : 8),
              offsetLinks: view('uint32x2', path === 'offsetLinks' ? 7 : 8),
              status: view('uint32x4', path === 'status' ? 0 : 1)
            }
          })
      })
    )
  ];
  try {
    for (const fixture of cases) {
      let message = '';
      let failure: unknown;
      try {
        fixture.build();
      } catch (error) {
        message = (error as Error).message;
        failure = error;
      }
      expect(failure).toBeInstanceOf(AnalysisValidationError);
      const details = failure as AnalysisValidationError;
      expect(details.id).toBe(id);
      expect(details.path).toBe(fixture.path);
      expect(details.contributor).toMatch(/^GPU/);
      expect(details.received).toBe(fixture.received);
      expect(details.expected).toMatchObject(
        fixture.path === 'output.positions' && message.includes('at least')
          ? {minimumRows: fixture.expected}
          : {rows: fixture.expected}
      );
      expect(message, fixture.path).toContain(id);
      expect(message, fixture.path).toContain(fixture.path);
      expect(message, fixture.path).toContain(`${fixture.expected} rows`);
      expect(message, fixture.path).toContain(`received ${fixture.received}`);
    }
  } finally {
    device.destroy();
  }
});

it.each([
  'minimum rows',
  'mode'
] as const)('structured precision validation describes %s', field => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends GPUGraphVectorFormat>(format: Format, length: number) =>
    createTransientView(graph, `details-${serial++}`, format, length);
  try {
    let failure: unknown;
    try {
      new GPULinePrecisionReduction({
        id: 'review',
        positions: view('float32x2', 2),
        lineOffsets: view('uint32', 2),
        gridSize: 1,
        mode: field === 'mode' ? ('unsupported' as never) : 'valid-output',
        output: {
          positions: view('float32x2', field === 'minimum rows' ? 1 : 2),
          lineOffsets: view('uint32', 2),
          status: view('uint32', 1)
        }
      });
    } catch (error) {
      failure = error;
    }
    expect(failure).toBeInstanceOf(AnalysisValidationError);
    const error = failure as AnalysisValidationError;
    expect(error.id).toBe('review');
    expect(error.contributor).toBe('GPULinePrecisionReduction');
    expect(error.path).toBe(field === 'mode' ? 'mode' : 'output.positions');
    expect(error.received).toBe(field === 'mode' ? 'unsupported' : 1);
    expect(error.expected).toEqual(
      field === 'mode'
        ? {values: ['valid-output', 'keep-collapsed']}
        : {minimumRows: 2, source: 'positions.length'}
    );
  } finally {
    device.destroy();
  }
});

it('structured refinement formats keep the caller identity separate from the port path', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends GPUGraphVectorFormat>(format: Format, length: number) =>
    createTransientView(graph, `format-${serial++}`, format, length);
  try {
    let failure: unknown;
    try {
      new GPULinePrecisionReduction({
        id: 'review',
        positions: view('float32x2', 2),
        lineOffsets: view('uint32', 2),
        gridSize: 1,
        mode: 'valid-output',
        output: {
          positions: view('uint32', 2) as never,
          lineOffsets: view('uint32', 2),
          status: view('uint32', 1)
        }
      });
    } catch (error) {
      failure = error;
    }
    expect(failure).toBeInstanceOf(AnalysisValidationError);
    const error = failure as AnalysisValidationError;
    expect(error.id).toBe('review');
    expect(error.path).toBe('output.positions');
    expect(error.received).toBe('uint32');
    expect(error.expected).toMatchObject({formats: ['float32x2']});
  } finally {
    device.destroy();
  }
});
it('refinement scalars preserve the shared float32 upper-edge acceptance', () => {
  expect(() =>
    validateRefinementNumber('review', 'tolerance', 3.4028234663852894e38)
  ).not.toThrow();
});
it('refinement scalar errors carry the actual value and numeric relationship', () => {
  try {
    validateRefinementNumber('review', 'gridSize', -1, true);
    throw new Error('accepted invalid grid');
  } catch (failure) {
    expect(failure).toBeInstanceOf(AnalysisValidationError);
    const error = failure as AnalysisValidationError;
    expect(error.id).toBe('review');
    expect(error.path).toBe('gridSize');
    expect(error.received).toBe(-1);
    expect(error.expected).toMatchObject({minimum: 2 ** -126, precision: 'float32'});
  }
});
