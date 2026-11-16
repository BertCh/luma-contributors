// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {afterAll, expectTypeOf, it} from 'vitest';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';
import {
  GPULineLocate,
  type GPULineLocateProps
} from '../../../src/gpu-spatial-analysis/linear-referencing/gpu-line-locate';
import {
  GPULinearReferencing,
  type GPULinearReferencingProps
} from '../../../src/gpu-spatial-analysis/linear-referencing/gpu-linear-referencing';
import {
  GPULocalRBF,
  type GPULocalRBFProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {
  GPULocalKriging,
  type GPULocalKrigingProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {
  GPUNaturalNeighbor,
  type GPUNaturalNeighborProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor';
import {
  GPUSphericalConvexContains,
  type GPUSphericalConvexContainsProps
} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-contains';
import {
  GPUSphericalPolygonContains,
  type GPUSphericalPolygonContainsProps
} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());

it('keeps every public props alias and class packed by default', () => {
  expectTypeOf<GPULineLocate['props']>().toEqualTypeOf<GPULineLocateProps>();
  expectTypeOf<GPULinearReferencing['props']>().toEqualTypeOf<GPULinearReferencingProps>();
  expectTypeOf<GPULocalRBF['props']>().toEqualTypeOf<GPULocalRBFProps>();
  expectTypeOf<GPULocalKriging['props']>().toEqualTypeOf<GPULocalKrigingProps>();
  expectTypeOf<GPUNaturalNeighbor['props']>().toEqualTypeOf<GPUNaturalNeighborProps>();
  expectTypeOf<
    GPUSphericalConvexContains['props']
  >().toEqualTypeOf<GPUSphericalConvexContainsProps>();
  expectTypeOf<
    GPUSphericalPolygonContains['props']
  >().toEqualTypeOf<GPUSphericalPolygonContainsProps>();
  expectTypeOf<GPULocalKrigingProps['supports']>().toEqualTypeOf<
    GraphDataView<'float32x2' | 'float32x4'>
  >();
  expectTypeOf<GPUNaturalNeighborProps['output']['weights']>().toEqualTypeOf<
    GraphDataView<'float32' | 'float32x2'> | undefined
  >();
});

it('infers packed port types without caller type arguments or casts', () => {
  const graph = new GPUCommandGraph(device);
  const packed = <
    Format extends 'float32' | 'float32x2' | 'float32x3' | 'float32x4' | 'uint32' | 'sint32'
  >(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const rows = packed;

  const locate = new GPULineLocate({
    positions: packed('locate-positions', 'float32x2', 2),
    lineOffsets: packed('locate-offsets', 'uint32', 2),
    eventLineIndices: rows('events', 'uint32', 6),
    eventMeasures: rows('event-measures', 'float32', 6),
    eventOffsets: rows('event-offsets', 'float32', 6),
    output: {
      positions: rows('located', 'float32x2', 6),
      segmentIndices: rows('located-segments', 'uint32', 6),
      tangents: rows('located-tangents', 'float32x2', 6),
      angles: rows('located-angles', 'float32', 6),
      statuses: rows('located-statuses', 'uint32', 6)
    }
  });
  const referencing = new GPULinearReferencing({
    positions: packed('referencing-positions', 'float32x2', 2),
    lineOffsets: packed('referencing-offsets', 'uint32', 2),
    radius: packed('referencing-radius', 'float32', 1),
    candidateCapacity: 32,
    queryPositions: rows('referencing-queries', 'float32x2', 6),
    output: {
      lineIndices: rows('line-indices', 'uint32', 6),
      segmentIndices: rows('segment-indices', 'uint32', 6),
      fractions: rows('fractions', 'float32', 6),
      footPoints: rows('feet', 'float32x2', 6),
      distances: rows('distances', 'float32', 6),
      measures: rows('measures', 'float32', 6),
      normalizedMeasures: rows('normalized', 'float32', 6),
      sides: rows('sides', 'sint32', 6),
      signedOffsets: rows('signed-offsets', 'float32', 6),
      status: {candidateOverflow: packed('candidate-overflow', 'uint32', 1)}
    }
  });
  const radialBasis = new GPULocalRBF({
    positions: packed('radial-positions', 'float32x2', 3),
    values: packed('radial-samples', 'float32', 3),
    queries: rows('radial-queries', 'float32x2', 6),
    output: {
      values: rows('radial-values', 'float32x4', 6),
      status: rows('radial-status', 'uint32', 6)
    }
  });
  const kriging = new GPULocalKriging({
    positions: packed('kriging-positions', 'float32x2', 3),
    observations: packed('observations', 'float32x4', 3),
    supports: rows('supports', 'float32x2', 6),
    sampleDrift: packed('sample-drift', 'float32', 3),
    queryDrift: rows('query-drift', 'float32', 6),
    driftCount: 1,
    mode: 'external-drift',
    neighborCount: 3,
    covariance: {model: 'exponential', sill: 1, range: 1},
    output: {
      values: rows('kriging-values', 'float32x4', 6),
      status: rows('kriging-status', 'uint32', 6),
      weights: rows('kriging-weights', 'float32', 18)
    }
  });
  const natural = new GPUNaturalNeighbor({
    positions: packed('natural-positions', 'float32x2', 3),
    values: packed('natural-samples', 'float32', 3),
    queries: rows('natural-queries', 'float32x2', 6),
    output: {
      values: rows('natural-values', 'float32x4', 6),
      status: rows('natural-status', 'uint32', 6),
      weights: rows('natural-weights', 'float32', 18)
    }
  });
  const convex = new GPUSphericalConvexContains({
    positions: packed('convex-positions', 'float32x3', 3),
    offsets: packed('convex-offsets', 'uint32', 2),
    queries: rows('convex-queries', 'float32x3', 6),
    polygonIndices: rows('convex-selectors', 'uint32', 6),
    output: {
      contains: rows('convex-contains', 'uint32', 6),
      status: rows('convex-status', 'uint32', 6)
    }
  });
  const polygon = new GPUSphericalPolygonContains({
    positions: packed('polygon-positions', 'float32x3', 3),
    ringOffsets: packed('ring-offsets', 'uint32', 2),
    polygonOffsets: packed('polygon-offsets', 'uint32', 2),
    queries: rows('polygon-queries', 'float32x3', 6),
    polygonIndices: rows('polygon-selectors', 'uint32', 6),
    output: {
      contains: rows('polygon-contains', 'uint32', 6),
      status: rows('polygon-status', 'uint32', 6)
    }
  });
  expectTypeOf(locate.props.output.positions).toEqualTypeOf<GraphDataView<'float32x2'>>();
  expectTypeOf(locate.props.output.segmentIndices).toEqualTypeOf<
    GraphDataView<'uint32'> | undefined
  >();
  expectTypeOf(locate.props.output.tangents).toEqualTypeOf<
    GraphDataView<'float32x2'> | undefined
  >();
  expectTypeOf(locate.props.output.angles).toEqualTypeOf<GraphDataView<'float32'> | undefined>();
  expectTypeOf(locate.props.output.statuses).toEqualTypeOf<GraphDataView<'uint32'> | undefined>();
  expectTypeOf(referencing.props.output.lineIndices).toEqualTypeOf<
    GraphDataView<'uint32'> | undefined
  >();
  expectTypeOf(referencing.props.output.segmentIndices).toEqualTypeOf<
    GraphDataView<'uint32'> | undefined
  >();
  expectTypeOf(referencing.props.output.fractions).toEqualTypeOf<
    GraphDataView<'float32'> | undefined
  >();
  expectTypeOf(referencing.props.output.footPoints).toEqualTypeOf<
    GraphDataView<'float32x2'> | undefined
  >();
  expectTypeOf(referencing.props.output.distances).toEqualTypeOf<
    GraphDataView<'float32'> | undefined
  >();
  expectTypeOf(referencing.props.output.measures).toEqualTypeOf<
    GraphDataView<'float32'> | undefined
  >();
  expectTypeOf(referencing.props.output.normalizedMeasures).toEqualTypeOf<
    GraphDataView<'float32'> | undefined
  >();
  expectTypeOf(referencing.props.output.sides).toEqualTypeOf<GraphDataView<'sint32'> | undefined>();
  expectTypeOf(referencing.props.output.signedOffsets).toEqualTypeOf<
    GraphDataView<'float32'> | undefined
  >();
  expectTypeOf(radialBasis.props.output.values).toEqualTypeOf<GraphDataView<'float32x4'>>();
  expectTypeOf(radialBasis.props.output.status).toEqualTypeOf<GraphDataView<'uint32'>>();
  expectTypeOf(kriging.props.output.values).toEqualTypeOf<GraphDataView<'float32x4'>>();
  expectTypeOf(kriging.props.output.status).toEqualTypeOf<GraphDataView<'uint32'>>();
  expectTypeOf(kriging.props.output.weights).toEqualTypeOf<GraphDataView<'float32'> | undefined>();
  expectTypeOf(kriging.props.queryDrift).toEqualTypeOf<GraphDataView<'float32'> | undefined>();
  expectTypeOf(natural.props.output.values).toEqualTypeOf<GraphDataView<'float32x4'>>();
  expectTypeOf(natural.props.output.status).toEqualTypeOf<GraphDataView<'uint32'>>();
  expectTypeOf(natural.props.output.weights).toEqualTypeOf<GraphDataView<'float32'> | undefined>();
  expectTypeOf(convex.props.output.contains).toEqualTypeOf<GraphDataView<'uint32'>>();
  expectTypeOf(convex.props.output.status).toEqualTypeOf<GraphDataView<'uint32'>>();
  expectTypeOf(polygon.props.output.contains).toEqualTypeOf<GraphDataView<'uint32'>>();
  expectTypeOf(polygon.props.output.status).toEqualTypeOf<GraphDataView<'uint32'>>();
  expectTypeOf(referencing.props.output.status.candidateOverflow).toEqualTypeOf<
    GraphDataView<'uint32'>
  >();
});

it('infers chunked port types without caller type arguments or casts', () => {
  const graph = new GPUCommandGraph(device);
  const packed = <
    Format extends 'float32' | 'float32x2' | 'float32x3' | 'float32x4' | 'uint32' | 'sint32'
  >(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const rows = <
    Format extends 'float32' | 'float32x2' | 'float32x3' | 'float32x4' | 'uint32' | 'sint32'
  >(
    name: string,
    format: Format,
    length: number
  ) => createVectorView(name, format, [packed(name, format, length)]);

  const locate = new GPULineLocate({
    positions: packed('locate-positions', 'float32x2', 2),
    lineOffsets: packed('locate-offsets', 'uint32', 2),
    eventLineIndices: rows('events', 'uint32', 6),
    eventMeasures: rows('event-measures', 'float32', 6),
    eventOffsets: rows('event-offsets', 'float32', 6),
    output: {
      positions: rows('located', 'float32x2', 6),
      segmentIndices: rows('located-segments', 'uint32', 6),
      tangents: rows('located-tangents', 'float32x2', 6),
      angles: rows('located-angles', 'float32', 6),
      statuses: rows('located-statuses', 'uint32', 6)
    }
  });
  const referencing = new GPULinearReferencing({
    positions: packed('referencing-positions', 'float32x2', 2),
    lineOffsets: packed('referencing-offsets', 'uint32', 2),
    radius: packed('referencing-radius', 'float32', 1),
    candidateCapacity: 32,
    queryPositions: rows('referencing-queries', 'float32x2', 6),
    output: {
      lineIndices: rows('line-indices', 'uint32', 6),
      segmentIndices: rows('segment-indices', 'uint32', 6),
      fractions: rows('fractions', 'float32', 6),
      footPoints: rows('feet', 'float32x2', 6),
      distances: rows('distances', 'float32', 6),
      measures: rows('measures', 'float32', 6),
      normalizedMeasures: rows('normalized', 'float32', 6),
      sides: rows('sides', 'sint32', 6),
      signedOffsets: rows('signed-offsets', 'float32', 6),
      status: {candidateOverflow: packed('candidate-overflow', 'uint32', 1)}
    }
  });
  const radialBasis = new GPULocalRBF({
    positions: packed('radial-positions', 'float32x2', 3),
    values: packed('radial-samples', 'float32', 3),
    queries: rows('radial-queries', 'float32x2', 6),
    output: {
      values: rows('radial-values', 'float32x4', 6),
      status: rows('radial-status', 'uint32', 6)
    }
  });
  const kriging = new GPULocalKriging({
    positions: packed('kriging-positions', 'float32x2', 3),
    observations: packed('observations', 'float32x4', 3),
    supports: rows('supports', 'float32x2', 6),
    sampleDrift: packed('sample-drift', 'float32', 3),
    queryDrift: rows('query-drift', 'float32', 6),
    driftCount: 1,
    mode: 'external-drift',
    neighborCount: 3,
    covariance: {model: 'exponential', sill: 1, range: 1},
    output: {
      values: rows('kriging-values', 'float32x4', 6),
      status: rows('kriging-status', 'uint32', 6),
      weights: rows('kriging-weights', 'float32', 18)
    }
  });
  const natural = new GPUNaturalNeighbor({
    positions: packed('natural-positions', 'float32x2', 3),
    values: packed('natural-samples', 'float32', 3),
    queries: rows('natural-queries', 'float32x2', 6),
    output: {
      values: rows('natural-values', 'float32x4', 6),
      status: rows('natural-status', 'uint32', 6),
      weights: rows('natural-weights', 'float32', 18)
    }
  });
  const convex = new GPUSphericalConvexContains({
    positions: packed('convex-positions', 'float32x3', 3),
    offsets: packed('convex-offsets', 'uint32', 2),
    queries: rows('convex-queries', 'float32x3', 6),
    polygonIndices: rows('convex-selectors', 'uint32', 6),
    output: {
      contains: rows('convex-contains', 'uint32', 6),
      status: rows('convex-status', 'uint32', 6)
    }
  });
  const polygon = new GPUSphericalPolygonContains({
    positions: packed('polygon-positions', 'float32x3', 3),
    ringOffsets: packed('ring-offsets', 'uint32', 2),
    polygonOffsets: packed('polygon-offsets', 'uint32', 2),
    queries: rows('polygon-queries', 'float32x3', 6),
    polygonIndices: rows('polygon-selectors', 'uint32', 6),
    output: {
      contains: rows('polygon-contains', 'uint32', 6),
      status: rows('polygon-status', 'uint32', 6)
    }
  });
  expectTypeOf(locate.props.output.positions).toEqualTypeOf<GraphVectorView<'float32x2'>>();
  expectTypeOf(locate.props.output.segmentIndices).toEqualTypeOf<
    GraphVectorView<'uint32'> | undefined
  >();
  expectTypeOf(locate.props.output.tangents).toEqualTypeOf<
    GraphVectorView<'float32x2'> | undefined
  >();
  expectTypeOf(locate.props.output.angles).toEqualTypeOf<GraphVectorView<'float32'> | undefined>();
  expectTypeOf(locate.props.output.statuses).toEqualTypeOf<GraphVectorView<'uint32'> | undefined>();
  expectTypeOf(referencing.props.output.lineIndices).toEqualTypeOf<
    GraphVectorView<'uint32'> | undefined
  >();
  expectTypeOf(referencing.props.output.segmentIndices).toEqualTypeOf<
    GraphVectorView<'uint32'> | undefined
  >();
  expectTypeOf(referencing.props.output.fractions).toEqualTypeOf<
    GraphVectorView<'float32'> | undefined
  >();
  expectTypeOf(referencing.props.output.footPoints).toEqualTypeOf<
    GraphVectorView<'float32x2'> | undefined
  >();
  expectTypeOf(referencing.props.output.distances).toEqualTypeOf<
    GraphVectorView<'float32'> | undefined
  >();
  expectTypeOf(referencing.props.output.measures).toEqualTypeOf<
    GraphVectorView<'float32'> | undefined
  >();
  expectTypeOf(referencing.props.output.normalizedMeasures).toEqualTypeOf<
    GraphVectorView<'float32'> | undefined
  >();
  expectTypeOf(referencing.props.output.sides).toEqualTypeOf<
    GraphVectorView<'sint32'> | undefined
  >();
  expectTypeOf(referencing.props.output.signedOffsets).toEqualTypeOf<
    GraphVectorView<'float32'> | undefined
  >();
  expectTypeOf(radialBasis.props.output.values).toEqualTypeOf<GraphVectorView<'float32x4'>>();
  expectTypeOf(radialBasis.props.output.status).toEqualTypeOf<GraphVectorView<'uint32'>>();
  expectTypeOf(kriging.props.output.values).toEqualTypeOf<GraphVectorView<'float32x4'>>();
  expectTypeOf(kriging.props.output.status).toEqualTypeOf<GraphVectorView<'uint32'>>();
  expectTypeOf(kriging.props.output.weights).toEqualTypeOf<
    GraphVectorView<'float32'> | undefined
  >();
  expectTypeOf(kriging.props.queryDrift).toEqualTypeOf<GraphVectorView<'float32'> | undefined>();
  expectTypeOf(natural.props.output.values).toEqualTypeOf<GraphVectorView<'float32x4'>>();
  expectTypeOf(natural.props.output.status).toEqualTypeOf<GraphVectorView<'uint32'>>();
  expectTypeOf(natural.props.output.weights).toEqualTypeOf<
    GraphVectorView<'float32'> | undefined
  >();
  expectTypeOf(convex.props.output.contains).toEqualTypeOf<GraphVectorView<'uint32'>>();
  expectTypeOf(convex.props.output.status).toEqualTypeOf<GraphVectorView<'uint32'>>();
  expectTypeOf(polygon.props.output.contains).toEqualTypeOf<GraphVectorView<'uint32'>>();
  expectTypeOf(polygon.props.output.status).toEqualTypeOf<GraphVectorView<'uint32'>>();
  expectTypeOf(referencing.props.output.status.candidateOverflow).toEqualTypeOf<
    GraphDataView<'uint32'>
  >();
});
