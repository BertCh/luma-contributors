// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  addSparseResistanceConnectivityRecipe,
  addRasterResistanceConnectivityRecipe,
  addSegregationCounterfactualRecipe,
  addPopulationSupportAlignmentRecipe,
  addStreetProfilesRecipe,
  addBuildingAdjacencyRecipe,
  addIsovistsRecipe,
  addVisibilityGraphRecipe,
  addBuildingTessellationRecipe,
  addLandscapeMetricsRecipe,
  addPatchProximityRecipe,
  addResistanceConnectivityRecipe,
  addSegregationDecompositionRecipe
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {KernelAuditCase} from './kernel-audit-context';

/** Full optional evidence and documented bounds, without allocating input or scratch buffers. */
export const COVERAGE_URBAN_ECOLOGY_CASES: readonly KernelAuditCase[] = [
  {
    exportName: 'addSparseResistanceConnectivityRecipe',
    name: '65538 vertices with sorted CSR and Jacobi PCG',
    build({graph, view}) {
      addSparseResistanceConnectivityRecipe(graph, {
        edges: view('uint32x2', 131072),
        conductances: view('float32', 131072),
        vertexCount: 65538,
        source: 0,
        sink: 65537,
        maximumIterations: 8
      });
    }
  },
  {
    exportName: 'addRasterResistanceConnectivityRecipe',
    name: 'anisotropic raster with eight-neighbor resistance and current density',
    build({graph, view}) {
      addRasterResistanceConnectivityRecipe(graph, {
        resistance: view('float32', 1024),
        width: 32,
        height: 32,
        cellSize: [1, 2],
        connectivity: 8,
        source: 0,
        sink: 1023,
        maximumIterations: 8
      });
    }
  },
  ...(['composition', 'dual-composition', 'share'] as const).map(method => ({
    exportName: 'addSegregationCounterfactualRecipe',
    name: `2048 units with ${method} quantile mapping`,
    build({graph, view}: Parameters<KernelAuditCase['build']>[0]) {
      addSegregationCounterfactualRecipe(graph, {
        baseline: view('float32x2', 2048),
        final: view('float32x2', 2048),
        supportId: 'aligned-zones',
        method
      });
    }
  })),
  {
    exportName: 'addPopulationSupportAlignmentRecipe',
    name: 'explicit stable-ID alignment and missing or duplicate evidence',
    build({graph, view}) {
      addPopulationSupportAlignmentRecipe(graph, {
        sourceIds: view('uint32', 2048),
        targetIds: view('uint32', 2048),
        populations: view('float32x2', 2048)
      });
    }
  },
  {
    exportName: 'addStreetProfilesRecipe',
    name: '1024-segment facade event storage and finite corridor',
    build({graph, view}) {
      addStreetProfilesRecipe(graph, {
        positions: view('float32x2', 16),
        normals: view('float32x2', 16),
        segments: view('float32x4', 1024),
        segmentBuildingIds: view('uint32', 1024),
        buildingHeights: view('float32', 256),
        streetSegment: view('float32x4', 1),
        maximumDistance: 100,
        sampleSpacing: 1
      });
    }
  },
  {
    exportName: 'addIsovistsRecipe',
    name: '4096 sorted vertex-event rays in graph storage',
    build({graph, view}) {
      addIsovistsRecipe(graph, {
        positions: view('float32x2', 8),
        segments: view('float32x4', 512),
        rayCount: 1024,
        sampling: 'vertices',
        region: {fillRule: 'nonzero', freeSpace: 'outside'},
        maximumDistance: 100
      });
    }
  },
  {
    exportName: 'addLandscapeMetricsRecipe',
    name: 'resident tile reconciliation, background policy and joint adjacency contagion',
    build({graph, view}) {
      addLandscapeMetricsRecipe(graph, {
        patches: view('uint32', 1024),
        patchClasses: view('uint32', 16),
        width: 32,
        height: 32,
        classCount: 4,
        cellSize: [1, 2],
        connectivity: 8,
        reconcilePatches: true,
        backgroundClass: 3,
        countBoundary: false,
        countBackground: false,
        contagionWeighting: 'adjacency'
      });
    }
  },
  {
    exportName: 'addStreetProfilesRecipe',
    name: 'binding-limited profiles and obstacles, explicit spacing and tolerance',
    build({graph, view, maximumRows}) {
      const count = maximumRows(16);
      addStreetProfilesRecipe(graph, {
        positions: view('float32x2', count),
        normals: view('float32x2', count),
        segments: view('float32x4', count),
        segmentBuildingIds: view('uint32', count),
        buildingHeights: view('float32', count),
        maximumDistance: 100,
        sampleSpacing: 0.5,
        tolerance: 1e-5
      });
    }
  },
  {
    exportName: 'addBuildingAdjacencyRecipe',
    name: '64 building union structures, exterior perimeter and cardinal alignment',
    build({graph, view}) {
      addBuildingAdjacencyRecipe(graph, {
        orientations: view('float32', 64),
        lineBuildingIds: view('uint32', 64),
        contiguity: view('uint32x2', 128),
        sharedPaths: {
          runs: {
            positions: view('float32x2', 128),
            offsets: view('uint32', 65),
            leftLineIds: view('uint32', 64),
            rightLineIds: view('uint32', 64),
            forward: view('uint32', 64)
          },
          status: {count: view('uint32', 1), overflow: view('uint32', 1)}
        },
        structures: {
          geometry: {
            kind: 'polygons',
            positions: view('float32x2', 256),
            ringOffsets: view('uint32', 65),
            polygonOffsets: view('uint32', 65),
            featureOffsets: view('uint32', 65)
          },
          vertexTolerance: 1 / 1024,
          capacity: {intersections: 1024, nodedSegments: 1024, vertices: 512, rings: 128}
        }
      });
    }
  },
  {
    exportName: 'addBuildingAdjacencyRecipe',
    name: '1024 buildings, all upstream topology status fields, binding-limited runs',
    build({graph, view, maximumRows}) {
      const runs = maximumRows(8) - 1;
      addBuildingAdjacencyRecipe(graph, {
        orientations: view('float32', 1024),
        lineBuildingIds: view('uint32', maximumRows(4)),
        sharedPaths: {
          runs: {
            positions: view('float32x2', maximumRows(8)),
            offsets: view('uint32', runs + 1),
            leftLineIds: view('uint32', runs),
            rightLineIds: view('uint32', runs),
            forward: view('uint32', runs)
          },
          status: {
            count: view('uint32', 1),
            overflow: view('uint32', 1),
            uncertainCount: view('uint32', 1)
          }
        }
      });
    }
  },
  {
    exportName: 'addIsovistsRecipe',
    name: '65536 rays, binding-limited endpoints and obstacles',
    build({graph, view, maximumRows}) {
      addIsovistsRecipe(graph, {
        positions: view('float32x2', maximumRows(8 * 65536)),
        segments: view('float32x4', maximumRows(16)),
        rayCount: 65536,
        maximumDistance: 100,
        tolerance: 1e-5
      });
    }
  },
  {
    exportName: 'addVisibilityGraphRecipe',
    name: '256 viewpoints, binding-limited edge capacity and obstacles',
    build({graph, view, maximumRows}) {
      addVisibilityGraphRecipe(graph, {
        positions: view('float32x2', 256),
        segments: view('float32x4', maximumRows(16)),
        maximumDistance: 100,
        edgeCapacity: maximumRows(8),
        tolerance: 1e-5
      });
    }
  },
  {
    exportName: 'addBuildingTessellationRecipe',
    name: '64 buildings,128 samples,32 enclosure vertices, bounded overlay scratch',
    build({graph, view, maximumRows}) {
      // Overlay scratch is bounded independently by its 144-byte noded segment table.
      const capacity = maximumRows(144);
      addBuildingTessellationRecipe(graph, {
        segments: view('float32x4', maximumRows(16)),
        segmentBuildingIds: view('uint32', maximumRows(16)),
        buildingCount: 64,
        enclosure: view('float32x2', 32),
        sampleSpacing: 1,
        sampleCapacity: 128,
        vertexTolerance: 1 / 1024,
        capacity: {
          intersections: 100000,
          nodedSegments: capacity,
          vertices: capacity,
          rings: Math.floor(capacity / 3)
        }
      });
    }
  },
  ...([4, 8] as const).map(
    (connectivity): KernelAuditCase => ({
      exportName: 'addLandscapeMetricsRecipe',
      name: `256 classes, binding-limited cells and patch metrics, connectivity ${connectivity}`,
      build({graph, view, maximumRows}) {
        addLandscapeMetricsRecipe(graph, {
          patches: view('uint32', maximumRows(4)),
          patchClasses: view('uint32', maximumRows(16)),
          width: maximumRows(4),
          height: 1,
          classCount: 256,
          cellSize: [1, 2],
          connectivity
        });
      }
    })
  ),
  {
    exportName: 'addPatchProximityRecipe',
    name: 'binding-limited cells and pair distances, anisotropic cells',
    build({graph, view, maximumRows}) {
      const patches = Math.floor(Math.sqrt(maximumRows(4)));
      addPatchProximityRecipe(graph, {
        patches: view('uint32', patches),
        patchClasses: view('uint32', patches),
        width: patches,
        height: 1,
        cellSize: [1, 2],
        edgeDepth: 1,
        coreModel: 'raster-erosion',
        coreBoundary: 'open',
        distanceModel: 'cell-center',
        searchRadius: 100
      });
    }
  },
  {
    exportName: 'addResistanceConnectivityRecipe',
    name: '256 vertices,65536 edges,4096 iterations, explicit residual tolerance',
    build({graph, view}) {
      addResistanceConnectivityRecipe(graph, {
        edges: view('uint32x2', 65536),
        conductances: view('float32', 65536),
        vertexCount: 256,
        source: 0,
        sink: 255,
        maximumIterations: 4096,
        tolerance: 1e-5
      });
    }
  },
  {
    exportName: 'addSegregationDecompositionRecipe',
    name: 'four binding-limited aligned counterfactual population columns',
    build({graph, view, maximumRows}) {
      const count = maximumRows(8);
      addSegregationDecompositionRecipe(graph, {
        populations: [
          view('float32x2', count),
          view('float32x2', count),
          view('float32x2', count),
          view('float32x2', count)
        ],
        supportId: 'audit-aligned-zones',
        counterfactualMethod: 'explicit-supplied-counterfactuals'
      });
    }
  }
];
