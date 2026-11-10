// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Named limits shared by the point-cloud contributors. Each constant states why the limit exists,
 * so a refusal at construction can name the limit that failed.
 */

/**
 * Largest retained neighbor count of neighborhoods, features and the statistical filter. Each
 * query lane holds `capacity` ids and distances in private arrays (512 bytes at 64), and the
 * cooperative strategy keeps one such list per lane in 16 KiB of workgroup memory.
 */
export const GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY = 64;

/** Largest ICP iteration count. Iterations are unrolled into graph nodes, a few per iteration. */
export const GPU_POINT_CLOUD_MAXIMUM_ICP_ITERATIONS = 100;

/**
 * Largest cell count along one grid axis. Cell coordinates are computed in float32, which
 * represents every integer up to 2^24 exactly.
 */
export const GPU_POINT_CLOUD_MAXIMUM_GRID_AXIS_CELLS = 0x1000000;

/** Largest total cell count of a grid. Cell keys and offsets are uint32 and 0xffffffff marks invalid points. */
export const GPU_POINT_CLOUD_MAXIMUM_GRID_CELLS = 0xfffffffe;

/** Largest number of square-window radii of the progressive morphological filter; each radius adds three kernels. */
export const GPU_POINT_CLOUD_MAXIMUM_GROUND_RADIUS_COUNT = 32;

/** Largest square-window radius, in cells, of the progressive morphological filter. Each cell scans (2r+1)^2 neighbors. */
export const GPU_POINT_CLOUD_MAXIMUM_GROUND_RADIUS_CELLS = 32;

/** Largest SMRF opening window, in cells. The unrolled openings grow as the square of the radius (about 2,100 nodes at 64). */
export const GPU_POINT_CLOUD_MAXIMUM_SMRF_WINDOW_CELLS = 64;

/** Largest SMRF net-cut spacing, in cells. The cut openings use diamonds of twice this radius. */
export const GPU_POINT_CLOUD_MAXIMUM_SMRF_CUT_CELLS = 32;

/** Largest inverse-distance-weighting reach of the height contributor, in cells. Each point visits (2*reach+3)^2 cells. */
export const GPU_POINT_CLOUD_MAXIMUM_IDW_REACH_CELLS = 64;

/** Largest inverse-distance-weighting power. Higher powers underflow the normalized float32 weights. */
export const GPU_POINT_CLOUD_MAXIMUM_IDW_POWER = 8;

/**
 * Largest cloud size with `samplingCovariance`. One thread visits every pair of in-cylinder
 * points, so the work per core is quadratic in the points the cylinder holds.
 */
export const GPU_POINT_CLOUD_MAXIMUM_SAMPLING_COVARIANCE_POINTS = 32768;

/**
 * Largest number of point-pair tests of a search that has no `grid`, as queries times candidates.
 * Above it the contributor refuses and asks for a `grid`. It is about five times the largest
 * unindexed run in the tests and the website (5,000 by 5,000 points, 25 million pair tests) and
 * about 8 times the largest case of the kernel audit (4,096 squared), so no existing run changes.
 * The ceiling of any exhaustive scan stays `GPU_POINT_CLOUD_MAXIMUM_BRUTE_FORCE_WORK` (2^32).
 */
export const GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK = 2 ** 27;

/**
 * Largest number of grid cells one core of `GPUPointCloudChange` visits, as
 * `(ceil(2 * sqrt(halfDepth^2 + radius^2) / cellSize) + 3)^3`. One invocation walks all of them,
 * so a tiny `cellSize` against a large cylinder could run past the device timeout. The largest
 * use in the tests is 216 cells (radius 1.2, halfDepth 0.6, cellSize 1); the limit is 2^20, about
 * 4,800 times that, which is a cylinder 100 cells across on every axis.
 */
export const GPU_POINT_CLOUD_MAXIMUM_CHANGE_CELLS_PER_CORE = 2 ** 20;
