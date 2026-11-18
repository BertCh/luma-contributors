// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPU_POINT_CLOUD_CHANGE_STATUS_INSUFFICIENT_SUPPORT,
  GPU_POINT_CLOUD_CHANGE_STATUS_INVALID_CORE,
  GPU_POINT_CLOUD_CHANGE_STATUS_NONFINITE_UNCERTAINTY,
  GPU_POINT_CLOUD_CHANGE_STATUS_VALID,
  GPU_POINT_CLOUD_COVERAGE_ANY_POINT,
  GPU_POINT_CLOUD_COVERAGE_GROUND,
  GPU_POINT_CLOUD_FEATURES_STATUS_DEGENERATE,
  GPU_POINT_CLOUD_FEATURES_STATUS_NOT_CONVERGED,
  GPU_POINT_CLOUD_FEATURES_STATUS_TRUNCATED,
  GPU_POINT_CLOUD_GROUND_STATUS_INCOMPLETE_WINDOW,
  GPU_POINT_CLOUD_GROUND_STATUS_UNSUPPORTED,
  GPU_POINT_CLOUD_HEIGHT_STATUS_INVALID_POINT,
  GPU_POINT_CLOUD_HEIGHT_STATUS_NONFINITE,
  GPU_POINT_CLOUD_HEIGHT_STATUS_UNSUPPORTED_TERRAIN,
  GPU_POINT_CLOUD_HEIGHT_STATUS_VALID,
  GPU_POINT_CLOUD_ICP_STATUS_CONVERGED,
  GPU_POINT_CLOUD_ICP_STATUS_DEGENERATE,
  GPU_POINT_CLOUD_ICP_STATUS_EIGENSOLVE_EXHAUSTED,
  GPU_POINT_CLOUD_ICP_STATUS_INSUFFICIENT_MATCHES,
  GPU_POINT_CLOUD_ICP_STATUS_ITERATION_LIMIT,
  GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK,
  GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_INVALID_QUERY,
  GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_TRUNCATED,
  GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_UNSUPPORTED,
  GPU_POINT_CLOUD_SMRF_STATUS_EXCLUDED,
  GPU_POINT_CLOUD_SMRF_STATUS_INPAINTED,
  GPU_POINT_CLOUD_SMRF_STATUS_INVALID,
  GPU_POINT_CLOUD_SMRF_STATUS_NO_TERRAIN,
  GPU_POINT_CLOUD_VOXEL_MERGE_STATUS_COUNT_OVERFLOW,
  GPU_POINT_CLOUD_VOXEL_MERGE_STATUS_REPRESENTATIVE_OVERFLOW,
  GPUPointCloudChange,
  GPUPointCloudICP,
  GPUPointCloudNeighborhoods
} from '@luma-contributors/analysis/pointcloud-analysis';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

const graph = new GPUCommandGraph(createNullWebGPUDevice());
let serial = 0;
const view = <Format extends 'uint32' | 'float32' | 'float32x3'>(format: Format, length: number) =>
  createTransientView(graph, `pointcloud-limits-${serial++}`, format, length);

const GRID = {origin: [0, 0, 0], cellSize: 1, dimensions: [8, 8, 8]} as const;

it('the status constants keep their documented values', () => {
  expect([
    GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_INVALID_QUERY,
    GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_TRUNCATED,
    GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_UNSUPPORTED
  ]).toEqual([1, 2, 4]);
  expect([
    GPU_POINT_CLOUD_FEATURES_STATUS_DEGENERATE,
    GPU_POINT_CLOUD_FEATURES_STATUS_TRUNCATED,
    GPU_POINT_CLOUD_FEATURES_STATUS_NOT_CONVERGED
  ]).toEqual([1, 2, 4]);
  expect([
    GPU_POINT_CLOUD_GROUND_STATUS_UNSUPPORTED,
    GPU_POINT_CLOUD_GROUND_STATUS_INCOMPLETE_WINDOW
  ]).toEqual([1, 2]);
  expect([
    GPU_POINT_CLOUD_SMRF_STATUS_INVALID,
    GPU_POINT_CLOUD_SMRF_STATUS_NO_TERRAIN,
    GPU_POINT_CLOUD_SMRF_STATUS_EXCLUDED,
    GPU_POINT_CLOUD_SMRF_STATUS_INPAINTED
  ]).toEqual([1, 2, 4, 8]);
  expect([
    GPU_POINT_CLOUD_ICP_STATUS_CONVERGED,
    GPU_POINT_CLOUD_ICP_STATUS_ITERATION_LIMIT,
    GPU_POINT_CLOUD_ICP_STATUS_INSUFFICIENT_MATCHES,
    GPU_POINT_CLOUD_ICP_STATUS_DEGENERATE,
    GPU_POINT_CLOUD_ICP_STATUS_EIGENSOLVE_EXHAUSTED
  ]).toEqual([0, 1, 2, 3, 4]);
  expect([
    GPU_POINT_CLOUD_CHANGE_STATUS_VALID,
    GPU_POINT_CLOUD_CHANGE_STATUS_INVALID_CORE,
    GPU_POINT_CLOUD_CHANGE_STATUS_INSUFFICIENT_SUPPORT,
    GPU_POINT_CLOUD_CHANGE_STATUS_NONFINITE_UNCERTAINTY
  ]).toEqual([0, 1, 2, 3]);
  expect([
    GPU_POINT_CLOUD_HEIGHT_STATUS_VALID,
    GPU_POINT_CLOUD_HEIGHT_STATUS_INVALID_POINT,
    GPU_POINT_CLOUD_HEIGHT_STATUS_UNSUPPORTED_TERRAIN,
    GPU_POINT_CLOUD_HEIGHT_STATUS_NONFINITE
  ]).toEqual([0, 1, 2, 3]);
  expect([GPU_POINT_CLOUD_COVERAGE_ANY_POINT, GPU_POINT_CLOUD_COVERAGE_GROUND]).toEqual([1, 2]);
  expect([
    GPU_POINT_CLOUD_VOXEL_MERGE_STATUS_COUNT_OVERFLOW,
    GPU_POINT_CLOUD_VOXEL_MERGE_STATUS_REPRESENTATIVE_OVERFLOW
  ]).toEqual([1, 2]);
});

// 2^27 pairs: 2,048 queries against 65,536 points sit exactly at the limit.
const AT_LIMIT_QUERIES = GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK / 65536;

const makeNeighborhoods = (queries: number, withGrid: boolean) =>
  new GPUPointCloudNeighborhoods({
    positions: view('float32x3', 65536),
    queryRows: view('uint32', queries),
    radius: 1,
    capacity: 4,
    ...(withGrid ? {grid: GRID} : {}),
    output: {
      indices: view('uint32', queries * 4),
      squaredDistances: view('float32', queries * 4),
      counts: view('uint32', queries)
    }
  });

it('neighborhoods without a grid refuse more than the unindexed work limit and ask for a grid', () => {
  expect(() => makeNeighborhoods(AT_LIMIT_QUERIES, false)).not.toThrow();
  expect(() => makeNeighborhoods(AT_LIMIT_QUERIES + 1, false)).toThrow(
    /^gpu-pointcloud-neighborhoods queries x positions is \d+, above the 134217728 limit of a search without an index; pass grid$/
  );
  expect(() => makeNeighborhoods(AT_LIMIT_QUERIES + 1, true)).not.toThrow();
});

const makeICP = (source: number, withGrid: boolean) =>
  new GPUPointCloudICP({
    source: view('float32x3', source),
    target: view('float32x3', 65536),
    maximumDistance: 1,
    iterations: 2,
    ...(withGrid ? {targetGrid: GRID} : {}),
    output: {
      transform: view('float32', 16),
      history: view('float32x3', 2),
      inlierMask: view('uint32', source),
      status: view('uint32', 1)
    }
  });

it('ICP without a target grid refuses more than the unindexed work limit and ask for targetGrid', () => {
  expect(() => makeICP(AT_LIMIT_QUERIES, false)).not.toThrow();
  expect(() => makeICP(AT_LIMIT_QUERIES + 1, false)).toThrow(
    /^gpu-pointcloud-icp source x target is \d+, above the 134217728 limit of a search without an index; pass targetGrid$/
  );
  expect(() => makeICP(AT_LIMIT_QUERIES + 1, true)).not.toThrow();
});

const makeChange = (cores: number, withGrid: boolean) =>
  new GPUPointCloudChange({
    reference: view('float32x3', 32768),
    comparison: view('float32x3', 32768),
    cores: view('float32x3', cores),
    normals: view('float32x3', cores),
    alignment: 'already-aligned',
    radius: 1,
    halfDepth: 1,
    ...(withGrid ? {grid: GRID} : {}),
    output: {
      changes: view('float32x3', cores),
      roughness: view('float32x3', cores),
      counts: view('uint32', cores * 2),
      status: view('uint32', cores)
    }
  });

it('change detection without a grid refuses more than the unindexed work limit and asks for grid', () => {
  expect(() => makeChange(AT_LIMIT_QUERIES, false)).not.toThrow();
  expect(() => makeChange(AT_LIMIT_QUERIES + 1, false)).toThrow(
    /^gpu-pointcloud-change cores x \(reference \+ comparison\) is \d+, above the 134217728 limit of a scan without an index; pass grid$/
  );
  expect(() => makeChange(AT_LIMIT_QUERIES + 1, true)).not.toThrow();
});
