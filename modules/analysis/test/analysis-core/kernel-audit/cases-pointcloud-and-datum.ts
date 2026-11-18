// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUDatumGridTransform,
  GPUPointBuffer,
  GPU_POINT_BUFFER_PARAMETER_LENGTH
} from '../../../src/geospatial-analysis/index';
import {
  GPUPointCloudChange,
  GPUPointCloudNormalScale,
  GPUPointCloudFeatures,
  GPUPointCloudGround,
  GPUPointCloudHeight,
  GPUPointCloudICP,
  GPUPointCloudNeighborhoods,
  GPUPointCloudSMRF,
  GPUPointCloudStatisticalFilter,
  GPUPointCloudSurfaces,
  GPUPointCloudVoxelMerge,
  GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK
} from '../../../src/pointcloud-analysis/index';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

/** Largest square raster whose float32 or uint32 cells fit one storage binding (5792 x 5792). */
function getRasterGrid(context: KernelAuditContext) {
  const side = Math.floor(Math.sqrt(context.maximumRows(4)));
  return {
    grid: {origin: [0, 0] as const, cellSize: 1, width: side, height: side},
    cells: side * side
  };
}
/** Neighbor, feature and statistical-filter capacity maximum. */
const NEIGHBOR_CAPACITY = 64;
/** Rotation about Z by 30 degrees plus a translation, column-major. */
const ROTATION_TRANSFORM = [0.8660254, 0.5, 0, 0, -0.5, 0.8660254, 0, 0, 0, 0, 1, 0, 10, -20, 3, 1];
const SYMMETRIC_COVARIANCE = [1, 0.1, 0, 1, 0, 1] as const;

/** Builds the point-cloud neighborhoods case with `positions` points and `queries` query rows. */
function buildNeighborhoods(
  context: KernelAuditContext,
  positions: number,
  queries: number,
  extra: {
    capacity?: number;
    grid?: {
      origin: [number, number, number];
      cellSize: number;
      dimensions: [number, number, number];
    };
  }
) {
  const {graph, view} = context;
  const capacity = extra.capacity ?? 4;
  graph.add(
    new GPUPointCloudNeighborhoods({
      positions: view('float32x3', positions),
      queryRows: view('uint32', queries),
      radius: 1,
      capacity,
      ...(extra.grid ? {grid: extra.grid} : {}),
      output: {
        indices: view('uint32', queries * capacity),
        squaredDistances: view('float32', queries * capacity),
        counts: view('uint32', queries)
      }
    })
  );
}

/** Builds a Ground case whose radii are the given list. */
function buildGround(context: KernelAuditContext, radii: readonly number[]) {
  const {graph, view} = context;
  const grid = {origin: [0, 0] as const, cellSize: 1, width: 64, height: 64};
  graph.add(
    new GPUPointCloudGround({
      positions: view('float32x3', 16),
      classifications: view('uint32', 16),
      grid,
      radii,
      initialThreshold: 0.25,
      slope: 0.15,
      maximumThreshold: 3,
      minimumSurface: view('float32', 4096),
      coverage: view('uint32', 4096),
      output: {
        groundMask: view('uint32', 16),
        outputClassifications: view('uint32', 16),
        status: view('uint32', 16)
      }
    })
  );
}

/** Builds an SMRF case with a window and cut of the given physical sizes (cell size one). */
function buildSMRF(context: KernelAuditContext, window: number, cut: number) {
  const {graph, view} = context;
  graph.add(
    new GPUPointCloudSMRF({
      positions: view('float32x3', 16),
      classifications: view('uint32', 16),
      grid: {origin: [0, 0], cellSize: 1, width: 8, height: 8},
      window,
      cut,
      output: {
        groundSurface: view('float32', 64),
        groundMask: view('uint32', 16),
        outputClassifications: view('uint32', 16),
        status: view('uint32', 16)
      }
    })
  );
}

/** Builds an ICP case with the given iteration count and source and target sizes. */
function buildICP(context: KernelAuditContext, iterations: number, source: number, target: number) {
  const {graph, view} = context;
  graph.add(
    new GPUPointCloudICP({
      source: view('float32x3', source),
      target: view('float32x3', target),
      maximumDistance: 1,
      iterations,
      output: {
        transform: view('float32', 16),
        history: view('float32x3', iterations),
        inlierMask: view('uint32', source),
        status: view('uint32', 1)
      }
    })
  );
}

/** Builds a grid-indexed change case with the given cylinder radius and half depth on a 1-unit grid. */
function buildIndexedChange(context: KernelAuditContext, radius: number, halfDepth: number) {
  const {graph, view} = context;
  const cores = 16;
  graph.add(
    new GPUPointCloudChange({
      reference: view('float32x3', 64),
      comparison: view('float32x3', 64),
      cores: view('float32x3', cores),
      normals: view('float32x3', cores),
      alignment: 'already-aligned',
      grid: {origin: [0, 0, 0], cellSize: 1, dimensions: [8, 8, 8]},
      radius,
      halfDepth,
      output: {
        changes: view('float32x3', cores),
        roughness: view('float32x3', cores),
        counts: view('uint32', cores * 2),
        status: view('uint32', cores)
      }
    })
  );
}

/** Cases at and one step above the capacity, iteration, window and unindexed-work limits. */
function getPointCloudLimitCases(): KernelAuditCase[] {
  return [
    {
      exportName: 'GPUPointCloudNormalScale',
      name: '32 scales with indexed nearest-64 support and parameter orientation',
      build: ({graph, view}) =>
        graph.add(
          new GPUPointCloudNormalScale({
            parameters: view('float32', 5),
            positions: view('float32x3', 128),
            queryRows: view('uint32', 16),
            radii: Array.from({length: 32}, (_, index) => 0.1 + index * 0.1),
            capacity: 64,
            grid: {origin: [-4, -4, -4], cellSize: 1, dimensions: [8, 8, 8]},
            output: {
              normals: view('float32x3', 16),
              normalScales: view('float32', 16),
              normalVariances: view('float32', 16),
              status: view('uint32', 16)
            }
          })
        )
    },
    {
      exportName: 'GPUPointCloudNormalScale',
      name: 'scale count one above 32',
      build: ({graph, view}) =>
        graph.add(
          new GPUPointCloudNormalScale({
            positions: view('float32x3', 16),
            parameters: view('float32', 5),
            radii: Array.from({length: 33}, (_, index) => index + 1),
            capacity: 64,
            output: {
              normals: view('float32x3', 16),
              normalScales: view('float32', 16),
              normalVariances: view('float32', 16),
              status: view('uint32', 16)
            }
          })
        )
    },
    {
      exportName: 'GPUPointCloudNeighborhoods',
      name: 'capacity one above 64',
      build: context => buildNeighborhoods(context, 64, 64, {capacity: NEIGHBOR_CAPACITY + 1})
    },
    {
      exportName: 'GPUPointCloudNeighborhoods',
      name: 'unindexed search at the work limit',
      build: context => buildNeighborhoods(context, 65536, 2048, {})
    },
    {
      exportName: 'GPUPointCloudNeighborhoods',
      name: 'unindexed search one query above the work limit',
      build: context => buildNeighborhoods(context, 65536, 2049, {})
    },
    {
      exportName: 'GPUPointCloudNeighborhoods',
      name: 'grid axis of 16777216 cells',
      build: context =>
        buildNeighborhoods(context, 64, 64, {
          grid: {origin: [0, 0, 0], cellSize: 1, dimensions: [16777216, 1, 1]}
        })
    },
    {
      exportName: 'GPUPointCloudNeighborhoods',
      name: 'grid axis one above 16777216 cells',
      build: context =>
        buildNeighborhoods(context, 64, 64, {
          grid: {origin: [0, 0, 0], cellSize: 1, dimensions: [16777217, 1, 1]}
        })
    },
    {
      exportName: 'GPUPointCloudFeatures',
      name: 'capacity one above 64',
      build: context => {
        const {graph, view} = context;
        const rows = 16;
        const capacity = NEIGHBOR_CAPACITY + 1;
        graph.add(
          new GPUPointCloudFeatures({
            positions: view('float32x3', rows),
            indices: view('uint32', rows * capacity),
            counts: view('uint32', rows),
            capacity,
            output: {
              normals: view('float32x3', rows),
              eigenvalues: view('float32x3', rows),
              shapes: view('float32x3', rows),
              status: view('uint32', rows)
            }
          })
        );
      }
    },
    {
      exportName: 'GPUPointCloudStatisticalFilter',
      name: 'capacity one above 64',
      build: context => {
        const {graph, view} = context;
        const rows = 16;
        const capacity = NEIGHBOR_CAPACITY + 1;
        graph.add(
          new GPUPointCloudStatisticalFilter({
            squaredDistances: view('float32', rows * capacity),
            counts: view('uint32', rows),
            capacity,
            output: {meanDistances: view('float32', rows), validity: view('uint32', rows)}
          })
        );
      }
    },
    {
      exportName: 'GPUPointCloudGround',
      name: 'thirty-three radii',
      build: context =>
        buildGround(
          context,
          Array.from({length: 33}, (_, index) => index + 1)
        )
    },
    {
      exportName: 'GPUPointCloudGround',
      name: 'radius one above 32 cells',
      build: context => buildGround(context, [33])
    },
    {
      exportName: 'GPUPointCloudSMRF',
      name: 'window one above 64 cells',
      build: context => buildSMRF(context, 65, 0)
    },
    {
      exportName: 'GPUPointCloudSMRF',
      name: 'cut one above 32 cells',
      build: context => buildSMRF(context, 18, 33)
    },
    {
      exportName: 'GPUPointCloudICP',
      name: 'iterations one above 100',
      build: context => buildICP(context, 101, 16, 16)
    },
    {
      exportName: 'GPUPointCloudICP',
      name: 'unindexed search one source point above the work limit',
      build: context =>
        buildICP(context, 2, Math.floor(GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK / 8192) + 1, 8192)
    },
    {
      exportName: 'GPUPointCloudHeight',
      name: 'inverse-distance reach one above 64 cells',
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUPointCloudHeight({
            positions: view('float32x3', 16),
            grid: {origin: [0, 0], cellSize: 1, width: 8, height: 8},
            terrain: view('float32', 64),
            coverage: view('uint32', 64),
            interpolation: {method: 'idw', radius: 65},
            output: {heights: view('float32', 16), status: view('uint32', 16)}
          })
        );
      }
    },
    {
      exportName: 'GPUPointCloudHeight',
      name: 'inverse-distance power one above 8',
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUPointCloudHeight({
            positions: view('float32x3', 16),
            grid: {origin: [0, 0], cellSize: 1, width: 8, height: 8},
            terrain: view('float32', 64),
            coverage: view('uint32', 64),
            interpolation: {method: 'idw', radius: 4, power: 9},
            output: {heights: view('float32', 16), status: view('uint32', 16)}
          })
        );
      }
    },
    {
      exportName: 'GPUPointCloudChange',
      name: 'indexed cylinder visiting 101 cells per axis',
      // (ceil(2 * hypot(7, 48)) + 3)^3 = 101^3 cells, below the 2^20 limit.
      build: context => buildIndexedChange(context, 48, 7)
    },
    {
      exportName: 'GPUPointCloudChange',
      name: 'indexed cylinder visiting 102 cells per axis',
      // (ceil(2 * hypot(7, 49)) + 3)^3 = 102^3 cells, one axis step above the 2^20 limit.
      build: context => buildIndexedChange(context, 49, 7)
    },
    {
      exportName: 'GPUPointCloudNeighborhoods',
      name: 'grid of more than 4294967294 cells',
      build: context =>
        buildNeighborhoods(context, 64, 64, {
          grid: {origin: [0, 0, 0], cellSize: 1, dimensions: [16777216, 256, 1]}
        })
    }
  ];
}

/** Kernel-audit cases for point clouds and datum grids. */
export const POINTCLOUD_AND_DATUM_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUPointCloudChange',
    name: 'transform alignment, full uncertainty at the unindexed work limit',
    build: context => {
      const {graph, view} = context;
      const cores = 16384;
      const points = Math.floor(GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK / cores / 2);
      graph.add(
        new GPUPointCloudChange({
          reference: view('float32x3', points),
          comparison: view('float32x3', points),
          cores: view('float32x3', cores),
          normals: view('float32x3', cores),
          alignment: {transform: view('float32', 16)},
          radius: 2,
          halfDepth: 4,
          minimumSupport: 5,
          registrationError: 0.01,
          registrationCovariance: SYMMETRIC_COVARIANCE,
          uncertaintyCombination: 'quadrature',
          confidenceMultiplier: 2.58,
          output: {
            changes: view('float32x3', cores),
            roughness: view('float32x3', cores),
            counts: view('uint32', cores * 2),
            status: view('uint32', cores)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPointCloudChange',
    name: 'unindexed scan one core above the work limit',
    build: context => {
      const {graph, view} = context;
      const points = 8192;
      const cores = Math.floor(GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK / (2 * points)) + 1;
      graph.add(
        new GPUPointCloudChange({
          reference: view('float32x3', points),
          comparison: view('float32x3', points),
          cores: view('float32x3', cores),
          normals: view('float32x3', cores),
          alignment: 'already-aligned',
          radius: 2,
          halfDepth: 4,
          output: {
            changes: view('float32x3', cores),
            roughness: view('float32x3', cores),
            counts: view('uint32', cores * 2),
            status: view('uint32', cores)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPointCloudChange',
    name: 'grid, transform alignment and full uncertainty with clouds and cores at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const points = maximumRows(12);
      const cores = maximumRows(12);
      graph.add(
        new GPUPointCloudChange({
          reference: view('float32x3', points),
          comparison: view('float32x3', points),
          cores: view('float32x3', cores),
          normals: view('float32x3', cores),
          alignment: {transform: view('float32', 16)},
          grid: {origin: [-100, -100, -100], dimensions: [1024, 1024, 31], cellSize: 0.5},
          radius: 2,
          halfDepth: 4,
          minimumSupport: 5,
          registrationError: 0.01,
          registrationCovariance: SYMMETRIC_COVARIANCE,
          uncertaintyCombination: 'quadrature',
          confidenceMultiplier: 2.58,
          output: {
            changes: view('float32x3', cores),
            roughness: view('float32x3', cores),
            counts: view('uint32', cores * 2),
            status: view('uint32', cores)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPointCloudChange',
    name: 'sampling covariance at 32768 points per cloud',
    build: context => {
      const {graph, view} = context;
      const cores = 2048;
      graph.add(
        new GPUPointCloudChange({
          reference: view('float32x3', 32768),
          comparison: view('float32x3', 32768),
          cores: view('float32x3', cores),
          normals: view('float32x3', cores),
          alignment: 'already-aligned',
          radius: 2,
          halfDepth: 4,
          samplingCovariance: {
            referenceVariance: 0.01,
            comparisonVariance: 0.02,
            correlationLength: 1
          },
          output: {
            changes: view('float32x3', cores),
            roughness: view('float32x3', cores),
            counts: view('uint32', cores * 2),
            status: view('uint32', cores)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPointCloudChange',
    name: 'sampling covariance one point above 32768 per cloud',
    build: context => {
      const {graph, view} = context;
      const cores = 64;
      graph.add(
        new GPUPointCloudChange({
          reference: view('float32x3', 32769),
          comparison: view('float32x3', 32768),
          cores: view('float32x3', cores),
          normals: view('float32x3', cores),
          alignment: 'already-aligned',
          radius: 2,
          halfDepth: 4,
          samplingCovariance: {
            referenceVariance: 0.01,
            comparisonVariance: 0.02,
            correlationLength: 1
          },
          output: {
            changes: view('float32x3', cores),
            roughness: view('float32x3', cores),
            counts: view('uint32', cores * 2),
            status: view('uint32', cores)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPointCloudFeatures',
    name: 'capacity 64 with covariance and viewpoint at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const rows = maximumRows(NEIGHBOR_CAPACITY * 4);
      graph.add(
        new GPUPointCloudFeatures({
          positions: view('float32x3', maximumRows(12)),
          indices: view('uint32', rows * NEIGHBOR_CAPACITY),
          counts: view('uint32', rows),
          capacity: NEIGHBOR_CAPACITY,
          viewpoint: [0, 0, 100],
          output: {
            normals: view('float32x3', rows),
            eigenvalues: view('float32x3', rows),
            covariance: view('float32', rows * 6),
            shapes: view('float32x3', rows),
            status: view('uint32', rows)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPointCloudGround',
    name: 'thirty-two radii on a binding-limit square grid',
    build: context => {
      const {graph, view, maximumRows} = context;
      const points = maximumRows(12);
      const {grid, cells} = getRasterGrid(context);
      graph.add(
        new GPUPointCloudGround({
          positions: view('float32x3', points),
          classifications: view('uint32', points),
          grid,
          radii: Array.from({length: 32}, (_, index) => index + 1),
          initialThreshold: 0.25,
          slope: 0.15,
          maximumThreshold: 3,
          minimumSurface: view('float32', cells),
          coverage: view('uint32', cells),
          output: {
            groundMask: view('uint32', points),
            outputClassifications: view('uint32', points),
            status: view('uint32', points)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPointCloudICP',
    name: 'unweighted point-to-plane, 100 iterations, origin and target grid',
    build: context => {
      const {graph, view, maximumRows} = context;
      const points = maximumRows(12);
      graph.add(
        new GPUPointCloudICP({
          source: view('float32x3', points),
          target: view('float32x3', points),
          origin: [500000, 4000000, 100],
          initialTransform: ROTATION_TRANSFORM,
          maximumDistance: 5,
          targetGrid: {origin: [-100, -100, -100], dimensions: [1024, 1024, 31], cellSize: 0.5},
          metric: 'point-to-plane',
          targetNormals: view('float32x3', points),
          iterations: 100,
          translationTolerance: 1e-6,
          rotationTolerance: 1e-6,
          output: {
            transform: view('float32', 16),
            history: view('float32x3', 100),
            inlierMask: view('uint32', points),
            status: view('uint32', 1)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPointCloudICP',
    name: 'point-to-point Huber at the unindexed work limit',
    build: context => {
      const {graph, view} = context;
      const source = 16384;
      graph.add(
        new GPUPointCloudICP({
          source: view('float32x3', source),
          target: view('float32x3', GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK / source),
          maximumDistance: 5,
          robust: {method: 'huber', scale: 1},
          iterations: 100,
          output: {
            transform: view('float32', 16),
            history: view('float32x3', 100),
            inlierMask: view('uint32', source),
            status: view('uint32', 1)
          }
        })
      );
    }
  },
  ...(['serial', 'cooperative'] as const).map(
    (strategy): KernelAuditCase => ({
      exportName: 'GPUPointCloudNeighborhoods',
      name: `${strategy} capacity 64 with query rows, support bounds and grid`,
      build: context => {
        const {graph, view, maximumRows} = context;
        const queries = maximumRows(NEIGHBOR_CAPACITY * 4);
        graph.add(
          new GPUPointCloudNeighborhoods({
            positions: view('float32x3', maximumRows(12)),
            queryRows: view('uint32', queries),
            supportBounds: {minimum: [-100, -100, -100], maximum: [100, 100, 100]},
            radius: 2,
            capacity: NEIGHBOR_CAPACITY,
            strategy,
            grid: {origin: [-100, -100, -100], cellSize: 2, dimensions: [1024, 1024, 31]},
            output: {
              indices: view('uint32', queries * NEIGHBOR_CAPACITY),
              squaredDistances: view('float32', queries * NEIGHBOR_CAPACITY),
              counts: view('uint32', queries),
              status: view('uint32', queries)
            }
          })
        );
      }
    })
  ),
  ...(['serial', 'cooperative'] as const).map(
    (strategy): KernelAuditCase => ({
      exportName: 'GPUPointCloudNeighborhoods',
      name: `${strategy} all-pairs capacity 64 without grid`,
      build: context => {
        const {graph, view} = context;
        const queries = 4096;
        graph.add(
          new GPUPointCloudNeighborhoods({
            positions: view('float32x3', queries),
            radius: 1,
            capacity: NEIGHBOR_CAPACITY,
            strategy,
            output: {
              indices: view('uint32', queries * NEIGHBOR_CAPACITY),
              squaredDistances: view('float32', queries * NEIGHBOR_CAPACITY),
              counts: view('uint32', queries)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUPointCloudSMRF',
    name: '64-cell window, 32-cell cut, selection and all outputs on binding-limit square',
    build: context => {
      const {graph, view, maximumRows} = context;
      const points = maximumRows(12);
      const {grid, cells} = getRasterGrid(context);
      graph.add(
        new GPUPointCloudSMRF({
          positions: view('float32x3', points),
          classifications: view('uint32', points),
          selection: view('uint32', points),
          grid,
          slope: 0.2,
          scalar: 1.5,
          threshold: 0.6,
          window: 64,
          cut: 32,
          groundClass: 2,
          otherClass: 1,
          onlyGround: true,
          output: {
            groundSurface: view('float32', cells),
            groundMask: view('uint32', points),
            outputClassifications: view('uint32', points),
            status: view('uint32', points)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPointCloudStatisticalFilter',
    name: 'capacity 64 at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const rows = maximumRows(NEIGHBOR_CAPACITY * 4);
      graph.add(
        new GPUPointCloudStatisticalFilter({
          squaredDistances: view('float32', rows * NEIGHBOR_CAPACITY),
          counts: view('uint32', rows),
          capacity: NEIGHBOR_CAPACITY,
          standardDeviations: 3,
          output: {meanDistances: view('float32', rows), validity: view('uint32', rows)}
        })
      );
    }
  },
  {
    exportName: 'GPUPointCloudSurfaces',
    name: 'ground class on a binding-limit square grid',
    build: context => {
      const {graph, view, maximumRows} = context;
      const points = maximumRows(12);
      const {grid, cells} = getRasterGrid(context);
      graph.add(
        new GPUPointCloudSurfaces({
          positions: view('float32x3', points),
          classifications: view('uint32', points),
          groundClass: 2,
          grid,
          output: {
            terrain: view('float32', cells),
            surface: view('float32', cells),
            canopy: view('float32', cells),
            coverage: view('uint32', cells)
          }
        })
      );
    }
  },
  ...(
    [{method: 'idw', radius: 64, power: 8, minimumSupport: 4}, {method: 'bilinear'}] as const
  ).map(
    (interpolation): KernelAuditCase => ({
      exportName: 'GPUPointCloudHeight',
      name: `${interpolation.method} interpolation with Surfaces coverage bit`,
      build: context => {
        const {graph, view, maximumRows} = context;
        const points = maximumRows(12);
        const {grid, cells} = getRasterGrid(context);
        graph.add(
          new GPUPointCloudHeight({
            positions: view('float32x3', points),
            grid,
            terrain: view('float32', cells),
            coverage: view('uint32', cells),
            coverageBit: 2,
            interpolation,
            output: {heights: view('float32', points), status: view('uint32', points)}
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUPointCloudVoxelMerge',
    name: 'four parts with float32x4 attribute at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const voxels = maximumRows(16);
      const part = (sourceRowOffset: number) => ({
        counts: view('uint32', voxels),
        centroids: view('float32x3', voxels),
        representatives: view('uint32', voxels),
        sourceRowOffset,
        attribute: {means: view('float32x4', voxels), counts: view('uint32', voxels)}
      });
      graph.add(
        new GPUPointCloudVoxelMerge({
          partitionPolicy: 'disjoint-owned-rows',
          parts: [part(0), part(1_000_000), part(2_000_000_000), part(0xfffffffe)],
          output: {
            counts: view('uint32', voxels),
            centroids: view('float32x3', voxels),
            representatives: view('uint32', voxels),
            attribute: {means: view('float32x4', voxels), counts: view('uint32', voxels)},
            status: view('uint32', voxels)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPointCloudVoxelMerge',
    name: 'two parts without attribute',
    build: context => {
      const {graph, view, maximumRows} = context;
      const voxels = maximumRows(12);
      const part = (sourceRowOffset: number) => ({
        counts: view('uint32', voxels),
        centroids: view('float32x3', voxels),
        representatives: view('uint32', voxels),
        sourceRowOffset
      });
      graph.add(
        new GPUPointCloudVoxelMerge({
          partitionPolicy: 'disjoint-owned-rows',
          parts: [part(0), part(1000)],
          output: {
            counts: view('uint32', voxels),
            centroids: view('float32x3', voxels),
            representatives: view('uint32', voxels),
            status: view('uint32', voxels)
          }
        })
      );
    }
  },
  ...getPointCloudLimitCases(),
  ...(['combined', 'horizontal', 'vertical'] as const).map(
    (operation): KernelAuditCase => ({
      exportName: 'GPUDatumGridTransform',
      name: `${operation} inverse with low words, eight grids and 1024 iterations`,
      build: context => {
        const {graph, view, maximumRows} = context;
        const points = maximumRows(12);
        const samples = maximumRows(12);
        // The source documents no maximum grid count; eight is representative.
        const gridCells = Math.floor(samples / 8);
        graph.add(
          new GPUDatumGridTransform({
            positions: view('float32x3', points),
            positionsLow: view('float32x3', points),
            samples: view('float32x3', samples),
            mask: view('uint32', samples),
            grids: Array.from({length: 8}, (_, index) => ({
              width: 2,
              height: Math.floor(gridCells / 2),
              sampleOffset: index * gridCells,
              origin: [index * 10, -45] as const,
              spacing: [0.01, 0.01] as const
            })),
            operation,
            direction: 'inverse',
            sourceVerticalReference: 'source-reference',
            targetVerticalReference: 'target-reference',
            heightUnitMeters: 0.3048,
            maximumIterations: 1024,
            tolerance: 1e-7,
            output: {
              positions: view('float32x3', points),
              positionsLow: view('float32x3', points),
              status: view('uint32', points)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUDatumGridTransform',
    name: 'combined forward without low words',
    build: context => {
      const {graph, view, maximumRows} = context;
      const points = maximumRows(12);
      const samples = maximumRows(12);
      graph.add(
        new GPUDatumGridTransform({
          positions: view('float32x3', points),
          samples: view('float32x3', samples),
          mask: view('uint32', samples),
          grids: [
            {
              width: 2,
              height: Math.floor(samples / 2),
              sampleOffset: 0,
              origin: [0, -45],
              spacing: [0.01, 0.01]
            }
          ],
          operation: 'combined',
          direction: 'forward',
          sourceVerticalReference: 'source-reference',
          targetVerticalReference: 'target-reference',
          output: {positions: view('float32x3', points), status: view('uint32', points)}
        })
      );
    }
  },
  ...(['geographic', 'planar'] as const).map(
    (coordinateSystem): KernelAuditCase => ({
      exportName: 'GPUPointBuffer',
      name: `${coordinateSystem} 256-vertex rings at the binding limit`,
      build: context => {
        const {graph, view, maximumRows} = context;
        // The source documents no maximum ring size; 256 vertices is representative.
        const verticesPerRing = 256;
        const points = Math.floor(maximumRows(8) / verticesPerRing);
        graph.add(
          new GPUPointBuffer({
            points: view('float32x2', points),
            parameters: view('float32', GPU_POINT_BUFFER_PARAMETER_LENGTH),
            offsets: view('float32x2', points * verticesPerRing),
            verticesPerRing,
            coordinateSystem
          })
        );
      }
    })
  )
];
