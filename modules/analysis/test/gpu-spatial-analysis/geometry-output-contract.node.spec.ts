// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {GPUConstrainedDelaunayTriangulation} from '../../src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation';
import {GPUGeometryCleanup} from '../../src/gpu-spatial-analysis/geometry-edit/index';
import {GPUGeometryMeasures} from '../../src/gpu-spatial-analysis/geometry-measures/index';
import {
  createLinePrefixNode,
  createSegmentizeCountNode,
  createSegmentizeEmitNode
} from '../../src/gpu-spatial-analysis/line-segmentize/line-segmentize-kernels';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

function createViewFactory() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'geometry-output-contract'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'sint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {graph, view};
}

describe('geometry output named-limit validation (STD-03)', () => {
  it('GPUConstrainedDelaunayTriangulation names maximumFlips and the converged rows', () => {
    const {view} = createViewFactory();
    const props = () => ({
      positions: view('float32x2', 4),
      featureOffsets: view('uint32', 3),
      polygonOffsets: view('uint32', 3),
      ringOffsets: view('uint32', 3),
      output: {
        indices: view('uint32', 12),
        valid: view('uint32', 2),
        status: {converged: view('uint32', 2)}
      }
    });
    expect(() => new GPUConstrainedDelaunayTriangulation({...props(), maximumFlips: 0})).toThrow(
      /^constrained-delaunay maximumFlips must be an integer in \[1, 4294967295\]/
    );
    const undersized = props();
    undersized.output.status.converged = view('uint32', 1);
    expect(() => new GPUConstrainedDelaunayTriangulation(undersized)).toThrow(
      /^constrained-delaunay output.status.converged must have at least 2 rows/
    );
  });

  it('GPUGeometryMeasures names the geodesic area props and their ranges', () => {
    const {view} = createViewFactory();
    const props = {
      spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
      positions: view('float32x2', 4),
      geometryType: 'lines',
      lineOffsets: view('uint32', 2),
      output: {lengths: view('float32', 1)}
    } as const;
    expect(() => new GPUGeometryMeasures({...props, geodesicAreaMaximumSegments: 3})).toThrow(
      /^geometry-measures geodesicAreaMaximumSegments must be a power of two in \[2, 256\]/
    );
    expect(() => new GPUGeometryMeasures({...props, geodesicAreaMaximumSegments: 512})).toThrow(
      /^geometry-measures geodesicAreaMaximumSegments must be a power of two in \[2, 256\]/
    );
    expect(() => new GPUGeometryMeasures({...props, geodesicAreaTolerance: -1})).toThrow(
      /^geometry-measures geodesicAreaTolerance must be a positive finite number/
    );
  });

  it('GPUGeometryCleanup names the missing line offsets', () => {
    const {view} = createViewFactory();
    expect(
      () =>
        new GPUGeometryCleanup({
          geometryType: 'lines',
          positions: view('float32x2', 4),
          parameters: view('float32', 4),
          output: {
            positions: view('float32x2', 4),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)}
          }
        } as never)
    ).toThrow(/^geometry-cleanup lines need input and output line or ring offsets/);
  });

  it('line kernel builders name the conflicting or missing views', () => {
    const {graph, view} = createViewFactory();
    expect(() =>
      createLinePrefixNode(graph, {
        id: 'prefix-check',
        operation: 'GPULineSegmentize',
        rowTurns: view('sint32', 4),
        rowShifts: view('float32', 4)
      } as never)
    ).toThrow(/^prefix-check line prefix takes rowTurns or rowShifts, not both/);
    expect(() =>
      createSegmentizeCountNode(graph, {
        id: 'count-check',
        operation: 'GPULineSegmentize',
        parameters: view('float32', 4),
        parameterIndex: 4
      } as never)
    ).toThrow(/^count-check parameterIndex must be an integer in \[0, 3\]/);
    expect(() =>
      createSegmentizeEmitNode(graph, {
        id: 'emit-check',
        operation: 'GPULineSegmentize',
        closedLines: true,
        outputPositions: view('float32x2', 4)
      } as never)
    ).toThrow(/^emit-check closedLines needs lineOffsets/);
    expect(() =>
      createSegmentizeEmitNode(graph, {
        id: 'emit-check',
        operation: 'GPULineSegmentize',
        outputTurns: view('sint32', 3),
        outputPositions: view('float32x2', 4)
      } as never)
    ).toThrow(/^emit-check outputTurns must have at least 4 rows/);
  });
});
