// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogCoordinateSystem} from './catalog-geojson';
import type {CatalogGpuRunner} from './catalog-gpu-bindings';
import {
  runGpuAffineTransform,
  runGpuGeodesicDestination,
  runGpuGeodesicPairs,
  runGpuGeometryOrientation
} from './catalog-platform4-geometry-edit';
import {
  runGpuGeographicDistribution,
  runGpuLabelPoints,
  runGpuMinimumBounds,
  runGpuMinimumClearance
} from './catalog-platform4-geometry-bounds';
import {
  runGpuDelaunayTriangulation,
  runGpuPolygonTriangulation
} from './catalog-platform4-geometry-tessellation';
import {
  runGpuNetworkCentrality,
  runGpuNetworkStructure
} from './catalog-platform4-geometry-network';
import {getNetworkParameters, MAXIMUM_NETWORK_EDGES} from './catalog-binding-types';
import {
  runGpuGridGenerator,
  runGpuGroupConvexHull,
  runGpuKMeans
} from './catalog-platform4-geometry-groups';
import {
  runGpuLineChunk,
  runGpuLineMerge,
  runGpuLineSimplification,
  runGpuLineSmooth
} from './catalog-platform4-geometry-lines';

const MAXIMUM_LINE_CHUNKS = 50000;
const MAXIMUM_GRID_CELLS = 20000;
const MAXIMUM_CENTRALITY_NODES = 1500;

function getCoordinateSystem(values: Readonly<Record<string, unknown>>): CatalogCoordinateSystem {
  return values['coordinateSystem'] as CatalogCoordinateSystem;
}

/** PLATFORM4 GPU runners for geometry, line and network bindings that were CPU-reference only, keyed by binding id. */
export const PLATFORM4_GEOMETRY_GPU_RUNNERS: Readonly<Record<string, CatalogGpuRunner>> = {
  'line-simplification': {
    contributorNames: ['GPULineSimplification'],
    run: (context, inputs, values) =>
      runGpuLineSimplification(context, inputs['lines'], {
        tolerance: values['tolerance'] as number,
        coordinateSystem: getCoordinateSystem(values)
      })
  },
  'line-smooth': {
    contributorNames: ['GPULineSmooth'],
    run: (context, inputs, values) =>
      runGpuLineSmooth(context, inputs['lines'], {iterations: values['iterations'] as number})
  },
  'line-chunk': {
    contributorNames: ['GPULineChunk'],
    run: (context, inputs, values) =>
      runGpuLineChunk(
        context,
        inputs['lines'],
        {
          maximumChunkLength: values['maximumChunkLength'] as number,
          coordinateSystem: getCoordinateSystem(values)
        },
        MAXIMUM_LINE_CHUNKS
      )
  },
  'line-merge': {
    contributorNames: ['GPULineMerge'],
    run: (context, inputs, values) =>
      runGpuLineMerge(context, inputs['lines'], {coordinateSystem: getCoordinateSystem(values)})
  },
  'geometry-orientation': {
    contributorNames: ['GPUGeometryOrientation'],
    run: (context, inputs, values) =>
      runGpuGeometryOrientation(context, inputs['polygons'], {
        shellOrientation: values['shellOrientation'] as 'counterclockwise' | 'clockwise'
      })
  },
  'affine-transform': {
    contributorNames: ['GPUAffineTransform'],
    run: (context, inputs, values) =>
      runGpuAffineTransform(context, inputs['features'], {
        translateX: values['translateX'] as number,
        translateY: values['translateY'] as number,
        scaleX: values['scaleX'] as number,
        scaleY: values['scaleY'] as number,
        rotationDegrees: values['rotationDegrees'] as number,
        pivot: values['pivot'] as 'origin' | 'bounds-center'
      })
  },
  'geodesic-destination': {
    contributorNames: ['GPUGeodesicDestination'],
    run: (context, inputs, values) =>
      runGpuGeodesicDestination(context, inputs['points'], {
        bearing: values['bearing'] as number,
        distance: values['distance'] as number,
        coordinateSystem: getCoordinateSystem(values)
      })
  },
  'geodesic-pairs': {
    contributorNames: ['GPUGeodesicPairs'],
    run: (context, inputs, values) =>
      runGpuGeodesicPairs(context, inputs['origins'], inputs['destinations'], {
        coordinateSystem: getCoordinateSystem(values)
      })
  },
  'group-convex-hull': {
    contributorNames: ['GPUGroupConvexHull'],
    run: (context, inputs, values) =>
      runGpuGroupConvexHull(context, inputs['points'], {
        groupProperty: values['groupProperty'] as string,
        coordinateSystem: getCoordinateSystem(values)
      })
  },
  'grid-generator': {
    contributorNames: ['GPUGridGenerator'],
    run: (context, inputs, values) =>
      runGpuGridGenerator(
        context,
        inputs['extent'],
        {
          cellWidth: values['cellWidth'] as number,
          cellHeight: values['cellHeight'] as number
        },
        MAXIMUM_GRID_CELLS
      )
  },
  'geometry-minimum-bounds': {
    contributorNames: ['GPUMinimumBounds'],
    run: (context, inputs, values) =>
      runGpuMinimumBounds(
        context,
        inputs['features'],
        values['shape'] as 'rectangle' | 'circle',
        getCoordinateSystem(values)
      )
  },
  'geometry-label-point': {
    contributorNames: ['GPULabelPoint'],
    run: (context, inputs, values) =>
      runGpuLabelPoints(
        context,
        inputs['polygons'],
        false,
        values['precision'] as number,
        getCoordinateSystem(values)
      )
  },
  'geometry-maximum-inscribed-circle': {
    contributorNames: ['GPUMaximumInscribedCircle'],
    run: (context, inputs, values) =>
      runGpuLabelPoints(
        context,
        inputs['polygons'],
        true,
        values['precision'] as number,
        getCoordinateSystem(values)
      )
  },
  'delaunay-triangulation': {
    contributorNames: ['GPUDelaunayTessellation'],
    run: (context, inputs, values) =>
      runGpuDelaunayTriangulation(context, inputs['points'], getCoordinateSystem(values))
  },
  'polygon-triangulation': {
    contributorNames: ['GPUPolygonTriangulation'],
    run: (context, inputs, values) =>
      runGpuPolygonTriangulation(context, inputs['polygons'], getCoordinateSystem(values))
  },
  'network-structure': {
    contributorNames: ['GPUNetworkStructure'],
    run: (context, inputs, values) =>
      runGpuNetworkStructure(
        context,
        inputs['lines'],
        getNetworkParameters(values),
        MAXIMUM_NETWORK_EDGES
      )
  },
  'network-centrality': {
    contributorNames: ['GPUNetworkCentrality'],
    run: (context, inputs, values) =>
      runGpuNetworkCentrality(context, inputs['lines'], getNetworkParameters(values), {
        maximumEdges: MAXIMUM_NETWORK_EDGES,
        maximumNodes: MAXIMUM_CENTRALITY_NODES
      })
  },
  'k-means': {
    contributorNames: ['GPUKMeans'],
    run: (context, inputs, values) =>
      runGpuKMeans(context, inputs['points'], {
        clusterCount: values['clusterCount'] as number,
        maximumIterations: values['maximumIterations'] as number,
        coordinateSystem: getCoordinateSystem(values)
      })
  },
  'geometry-minimum-clearance': {
    contributorNames: ['GPUMinimumClearance'],
    run: (context, inputs, values) =>
      runGpuMinimumClearance(context, inputs['features'], getCoordinateSystem(values))
  },
  'geographic-distribution': {
    contributorNames: ['GPUGeographicDistribution'],
    run: (context, inputs, values) =>
      runGpuGeographicDistribution(context, inputs['features'], {
        groupProperty: values['groupProperty'] as string,
        weightProperty: values['weightProperty'] as string,
        standardDeviations: values['standardDeviations'] as number,
        coordinateSystem: getCoordinateSystem(values)
      })
  }
};
