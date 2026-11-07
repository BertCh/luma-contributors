// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  LOAD_BALANCED_WORKGROUP_SIZE,
  LOAD_BALANCED_TILE_SIZE
} from '../utils/load-balanced-for-each';

import {getGPUShaderSubgroupStrategy} from './gpu-subgroup-utils';
import {selectGPUStrategy, type GPUStrategyDecision} from './gpu-strategy';

/** @internal Short rows retain sequential accumulation. */
export const HYBRID_SPMV_MAXIMUM_SCALAR_ROW_LENGTH = 256;

export type GPUSpMVStrategyId =
  | 'scalar-row'
  | 'subgroup-row'
  | 'workgroup-row'
  | 'long-row'
  | 'hybrid-row';

export type GPUSpMVRowStatistics = {
  /** Optional CPU-known or previously profiled maximum nonzeros in a row. */
  maxNonZerosPerRow?: number;
  /** Optional CPU-known or previously profiled fraction of rows with <= 8 nonzeros. */
  shortRowFraction?: number;
};

export type GPUSpMVWorkload = {
  rows: number;
  nonZeros: number;
  statistics?: GPUSpMVRowStatistics;
  /** Physical view layout supplied by the operator, never inferred from logical lengths. */
  storage?: 'flat' | 'chunked';
};

export type GPUSpMVStrategyDetails = {
  /** For hybrid traversal, these row fields describe its scalar stage. */
  rowsPerWorkgroup: number;
  workgroupSize: 64 | 128 | 256;
  workgroupsPerLongRow: number;
  /** Present only for the additional merge and carry stages. */
  merge?: Readonly<{
    workgroupSize: number;
    tileSize: number;
    maximumScalarRowLength: number;
  }>;
};

/** Selects a CSR SpMV execution family from matrix shape and optional row statistics. */
export function selectGPUSpMVStrategy(
  device: Device,
  workload: GPUSpMVWorkload,
  preferredId?: GPUSpMVStrategyId
): GPUStrategyDecision<GPUSpMVStrategyId, GPUSpMVStrategyDetails> {
  if (!Number.isSafeInteger(workload.rows) || workload.rows < 0)
    throw new Error('SpMV rows must be non-negative');
  if (!Number.isSafeInteger(workload.nonZeros) || workload.nonZeros < 0)
    throw new Error('SpMV nonZeros must be non-negative');
  const average = workload.nonZeros / Math.max(1, workload.rows);
  const maxRow = workload.statistics?.maxNonZerosPerRow;
  const shortFraction = workload.statistics?.shortRowFraction;
  const hasSubgroups =
    getGPUShaderSubgroupStrategy(device, {requiresSubgroupId: true}) === 'subgroups';

  return selectGPUStrategy<GPUSpMVStrategyId, GPUSpMVWorkload, GPUSpMVStrategyDetails>({
    device,
    workload,
    preferredId,
    candidates: [
      {
        id: 'hybrid-row',
        isSupported: () =>
          workload.storage === 'flat' && workload.rows + workload.nonZeros <= 0xffffffff,
        score: () => -1,
        reason: () => 'explicit hybrid traversal; sequential short rows and merged long rows',
        createDetails: () => ({
          rowsPerWorkgroup: 64,
          workgroupSize: 64 as const,
          workgroupsPerLongRow: 1,
          merge: Object.freeze({
            workgroupSize: LOAD_BALANCED_WORKGROUP_SIZE,
            tileSize: LOAD_BALANCED_TILE_SIZE,
            maximumScalarRowLength: HYBRID_SPMV_MAXIMUM_SCALAR_ROW_LENGTH
          })
        })
      },
      {
        id: 'scalar-row',
        score: () => (average <= 8 ? 100 : average <= 16 ? 60 : 10),
        reason: () => `average row length ${average.toFixed(1)} favors one invocation per row`,
        createDetails: () => ({
          rowsPerWorkgroup: 64,
          workgroupSize: 64 as const,
          workgroupsPerLongRow: 1
        })
      },
      {
        id: 'subgroup-row',
        isSupported: () => hasSubgroups,
        score: () =>
          average <= 64 && (shortFraction === undefined || shortFraction < 0.9) ? 110 : 35,
        reason: () => `subgroups available; medium CSR rows can reduce within a subgroup`,
        createDetails: () => ({
          rowsPerWorkgroup: 1,
          workgroupSize: 128 as const,
          workgroupsPerLongRow: 1
        })
      },
      {
        id: 'workgroup-row',
        score: () => (average > 32 && (maxRow === undefined || maxRow <= 2048) ? 100 : 50),
        reason: () => `average row length ${average.toFixed(1)} favors cooperative row reduction`,
        createDetails: () => ({
          rowsPerWorkgroup: 1,
          workgroupSize: 256 as const,
          workgroupsPerLongRow: 1
        })
      },
      {
        id: 'long-row',
        score: () => ((maxRow !== undefined && maxRow > 2048) || average > 1024 ? 140 : 0),
        reason: () => `very long rows favor multi-workgroup partial reduction`,
        createDetails: () => ({
          rowsPerWorkgroup: 1,
          workgroupSize: 256 as const,
          workgroupsPerLongRow: 4
        })
      }
    ]
  });
}
