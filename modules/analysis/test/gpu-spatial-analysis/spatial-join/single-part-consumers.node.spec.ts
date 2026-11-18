// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPULineClipByPolygon} from '../../../src/gpu-spatial-analysis/line-clip/gpu-line-clip-by-polygon';
import {GPUSharedPaths} from '../../../src/gpu-spatial-analysis/line-clip/gpu-shared-paths';
import {GPUSnapRoundedNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import {GPULineNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-line-noding';
import {GPULineSplit} from '../../../src/gpu-spatial-analysis/line-split/gpu-line-split';
import {GPULineLoopRemoval} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-line-loop-removal';
import {GPUTrackSimilarity} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity';
import {GPUTrackSimilaritySearch} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-search';
import {GPUBufferSurface} from '../../../src/gpu-spatial-analysis/polygon-buffer/gpu-buffer-surface';
import {GPUSegmentIntersection} from '../../../src/gpu-spatial-analysis/segment-intersection/gpu-segment-intersection';
import {addClusterAndOutlineRecipe} from '../../../src/gpu-spatial-analysis/recipes/cluster-and-outline-recipe';
import {addStraightLineCatchmentsRecipe} from '../../../src/gpu-spatial-analysis/recipes/straight-line-catchments-recipe';
import {GPUGeometryValidity} from '../../../src/gpu-spatial-analysis/geometry-validity/gpu-geometry-validity';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {GPUBufferSelection} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import type {GPUSpatialJoinGeometry} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-types';

it('point consumers infer a join-geometry union while refusing known multipart inputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'union-positions', 'float32x2', 2);
  const featureOffsets = createTransientView(graph, 'union-features', 'uint32', 2);
  const readGeometry = (grouped: boolean): GPUSpatialJoinGeometry =>
    grouped ? {kind: 'points', positions, featureOffsets} : {kind: 'points', positions};
  const points = readGeometry(false);
  if (points.kind !== 'points') throw new Error('expected point geometry');
  const polygons = {
    kind: 'polygons' as const,
    positions: createTransientView(graph, 'polygon-positions', 'float32x2', 4),
    featureOffsets: createTransientView(graph, 'polygon-features', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2)
  };
  const assignment = createTransientView(graph, 'assignment', 'uint32', 2);
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
  const pipOutput = {pointFeatureIds: assignment, status: {overflow}};
  const radius = createTransientView(graph, 'radius', 'float32', 1);
  const selectionOutput = {outputMask: createTransientView(graph, 'mask', 'uint32', 2)};
  const knownMultipart = {kind: 'points' as const, positions, featureOffsets};
  try {
    expect(
      new GPUPointInPolygonJoin({
        points,
        polygons,
        candidateCapacity: 4,
        output: pipOutput
      }).getCommandNodes(graph).length
    ).toBeGreaterThan(0);
    expect(
      new GPUBufferSelection({
        points,
        features: {kind: 'points', positions},
        candidateCapacity: 4,
        distance: radius,
        output: selectionOutput
      }).getCommandNodes(graph).length
    ).toBeGreaterThan(0);
    const grouped = readGeometry(true);
    if (grouped.kind !== 'points') throw new Error('expected point geometry');
    expect(
      () =>
        new GPUPointInPolygonJoin({
          points: grouped,
          polygons,
          candidateCapacity: 4,
          output: pipOutput
        })
    ).toThrow(/points.featureOffsets multipart points are unsupported/);
    expect(
      () =>
        new GPUBufferSelection({
          points: grouped,
          features: {kind: 'points', positions},
          candidateCapacity: 4,
          distance: radius,
          output: selectionOutput
        })
    ).toThrow(/points.featureOffsets multipart points are unsupported/);
    expect(
      () =>
        new GPUPointInPolygonJoin({
          // @ts-expect-error A known multipart port is not a supported point input.
          points: knownMultipart,
          polygons,
          candidateCapacity: 4,
          output: pipOutput
        })
    ).toThrow(/points.featureOffsets multipart points are unsupported/);
    expect(
      () =>
        new GPUBufferSelection({
          // @ts-expect-error A known multipart port is not a supported point input.
          points: knownMultipart,
          features: {kind: 'points', positions},
          candidateCapacity: 4,
          distance: radius,
          output: selectionOutput
        })
    ).toThrow(/points.featureOffsets multipart points are unsupported/);
  } finally {
    device.destroy();
  }
});

it('X3 grouped ports are rejected before a positions getter is unpacked', () => {
  let positionReads = 0;
  expect(
    () =>
      new GPUGeometryValidity({
        points: {
          kind: 'points',
          featureOffsets: {},
          get positions() {
            positionReads++;
            throw new Error('positions were unpacked');
          }
        }
      } as never)
  ).toThrow(/points.featureOffsets.*multipart/);
  expect(positionReads).toBe(0);
});

it('X3 the single-part snapshot captures inherited position getters once', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let positions = createTransientView(graph, 'positions', 'float32x2', 1);
  const original = positions;
  let positionReads = 0;
  const point = Object.assign(
    Object.create({
      get positions() {
        positionReads++;
        return positions;
      }
    }),
    {kind: 'points'}
  );
  try {
    const contributor = new GPUGeometryValidity({
      points: point,
      output: {issueCode: createTransientView(graph, 'issues', 'uint32', 1)}
    });
    positions = createTransientView(graph, 'replacement', 'float32x2', 2);
    expect(contributor.props.points!.positions).toBe(original);
    expect(positionReads).toBe(1);
  } finally {
    device.destroy();
  }
});

for (const [name, Constructor, fields] of [
  ['line clip', GPULineClipByPolygon, ['lines']],
  ['shared paths', GPUSharedPaths, ['left', 'right']],
  ['snap rounded noding', GPUSnapRoundedNoding, ['lines']],
  ['polygonize', GPUPolygonize, ['lines']],
  ['line noding', GPULineNoding, ['lines']],
  ['line split', GPULineSplit, ['lines']],
  ['line loop removal', GPULineLoopRemoval, ['lines']],
  ['track similarity', GPUTrackSimilarity, ['linesA', 'linesB']],
  ['track similarity search', GPUTrackSimilaritySearch, ['linesA', 'linesB']],
  ['buffer surface', GPUBufferSurface, ['geometry']],
  ['segment intersection', GPUSegmentIntersection, ['left', 'right']]
] as const)
  for (const field of fields)
    it(`X3 ${name} rejects a grouped ${field} before unpacking it`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const line = {
        kind: 'lines' as const,
        positions: createTransientView(graph, 'positions', 'float32x2', 4),
        lineOffsets: createTransientView(graph, 'parts', 'uint32', 3)
      };
      const grouped = {
        ...line,
        featureOffsets: createTransientView(graph, 'features', 'uint32', 2)
      };
      const props = Object.fromEntries(
        fields.map(name => [name, name === field ? {...grouped} : line])
      );
      try {
        expect(() => new Constructor(props as never)).toThrow(
          new RegExp(`${field}\\.featureOffsets.*multipart`)
        );
      } finally {
        device.destroy();
      }
    });

for (const [name, create, field] of [
  ['cluster outline', addClusterAndOutlineRecipe, 'points'],
  ['straight line catchments', addStraightLineCatchmentsRecipe, 'facilities']
] as const)
  it(`X3 ${name} refuses multipart points before graph.add`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const points = {
      kind: 'points' as const,
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2)
    };
    try {
      expect(() => create(graph, {[field]: {...points}} as never)).toThrow(
        /featureOffsets.*multipart/
      );
    } finally {
      device.destroy();
    }
  });
