// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_SINKHORN_DIVERGENCE_MAXIMUM_GRID_SIDE,
  GPU_SINKHORN_DIVERGENCE_MAXIMUM_ITERATIONS,
  GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH,
  GPU_SPACE_TIME_HAWKES_MAXIMUM_ITERATIONS,
  GPU_SPACE_TIME_HAWKES_MAXIMUM_PARENTS,
  GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH,
  GPUSinkhornDivergence,
  GPUSpaceTimeHawkes
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

/** Largest event count: counts are exact float32 integers up to 2^24, and the event table is 16 bytes per row. */
const MAXIMUM_HAWKES_ROWS = 8388608;

function addHawkes(
  {graph, view}: KernelAuditContext,
  options: {rows: number; maximumIterations?: number; maximumParents?: number; allOutputs?: boolean}
): void {
  const {rows} = options;
  graph.add(
    new GPUSpaceTimeHawkes({
      positions: view('float32x2', rows),
      times: view('float32', rows),
      parameters: view('float32', GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH),
      maximumIterations: options.maximumIterations,
      maximumParents: options.maximumParents,
      output:
        options.allOutputs === false
          ? {estimates: view('float32', 4)}
          : {
              estimates: view('float32', 4),
              logLikelihood: view('float32', 1),
              backgroundProbability: view('float32', rows),
              status: {
                converged: view('uint32', 1),
                iterationLimitReached: view('uint32', 1),
                iterationCount: view('uint32', 1),
                candidateOverflow: view('uint32', 1),
                invalidCount: view('uint32', 1),
                approximation: view('uint32', 1)
              }
            }
    })
  );
}

function addSinkhorn(
  {graph, view}: KernelAuditContext,
  options: {
    width: number;
    height: number;
    maximumIterations?: number;
    blurRadius?: number;
    allOutputs?: boolean;
  }
): void {
  const cells = options.width * options.height;
  graph.add(
    new GPUSinkhornDivergence({
      grid: makeGPURasterMetadata({width: options.width, height: options.height}),
      source: view('float32', cells),
      target: view('float32', cells),
      parameters: view('float32', GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH),
      maximumIterations: options.maximumIterations,
      blurRadius: options.blurRadius,
      output:
        options.allOutputs === false
          ? {divergence: view('float32', 1)}
          : {
              divergence: view('float32', 1),
              components: view('float32', 3),
              marginalError: view('float32', 1),
              status: {
                converged: view('uint32', 1),
                iterationLimitReached: view('uint32', 1),
                iterationCount: view('uint32', 1),
                invalidCount: view('uint32', 1),
                approximation: view('uint32', 1)
              }
            }
    })
  );
}

/** Audit cases of the space-time Hawkes EM (STM-08) and the Sinkhorn divergence (STM-09). */
export const STM08_STM09_HAWKES_AND_SINKHORN_AUDIT_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUSpaceTimeHawkes',
    name: 'one million events every output default capacities',
    build: context => addHawkes(context, {rows: 1_000_000})
  },
  {
    exportName: 'GPUSpaceTimeHawkes',
    name: 'events, iterations and parents at their maximum with every output',
    build: context =>
      addHawkes(context, {
        rows: MAXIMUM_HAWKES_ROWS,
        maximumIterations: GPU_SPACE_TIME_HAWKES_MAXIMUM_ITERATIONS,
        maximumParents: GPU_SPACE_TIME_HAWKES_MAXIMUM_PARENTS
      })
  },
  {
    exportName: 'GPUSpaceTimeHawkes',
    name: 'estimates only',
    build: context => addHawkes(context, {rows: 1000, maximumIterations: 5, allOutputs: false})
  },
  {
    exportName: 'GPUSpaceTimeHawkes',
    name: 'one event above the row limit',
    build: context => addHawkes(context, {rows: MAXIMUM_HAWKES_ROWS + 1, maximumIterations: 1})
  },
  {
    exportName: 'GPUSinkhornDivergence',
    name: 'largest grid side every output',
    build: context =>
      addSinkhorn(context, {
        width: GPU_SINKHORN_DIVERGENCE_MAXIMUM_GRID_SIDE,
        height: GPU_SINKHORN_DIVERGENCE_MAXIMUM_GRID_SIDE,
        maximumIterations: GPU_SINKHORN_DIVERGENCE_MAXIMUM_ITERATIONS
      })
  },
  {
    exportName: 'GPUSinkhornDivergence',
    name: 'windowed blur on a wide grid divergence only',
    build: context =>
      addSinkhorn(context, {
        width: 2048,
        height: 512,
        blurRadius: 64,
        maximumIterations: 20,
        allOutputs: false
      })
  },
  {
    exportName: 'GPUSinkhornDivergence',
    name: 'one cell above the grid side limit',
    build: context =>
      addSinkhorn(context, {
        width: GPU_SINKHORN_DIVERGENCE_MAXIMUM_GRID_SIDE + 1,
        height: 4,
        maximumIterations: 1
      })
  }
];
