// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_JACQUEZ_MAXIMUM_K,
  GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH,
  GPU_LOCAL_KNOX_SUMMARY_LENGTH,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY_LENGTH,
  GPUJacquezTest,
  GPULocalKnox,
  GPUNearRepeatKnox
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const MAXIMUM_PERMUTATIONS = 65536;
/** Sixteen bands along an axis: the largest table, 16 x 16 = 256 cells. */
const MAXIMUM_EDGES = Array.from({length: 17}, (_, index) => index);

function getPairs({view}: KernelAuditContext, rows: number) {
  const slots = 4;
  return {
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', slots * rows),
    weights: view('float32', slots * rows),
    distances: view('float32', slots * rows)
  };
}

function addNearRepeatKnox(
  context: KernelAuditContext,
  rows: number,
  distanceEdges: number[],
  timeEdges: number[],
  maximumPermutations: number
): void {
  const {graph, view} = context;
  const cells = (distanceEdges.length - 1) * (timeEdges.length - 1);
  graph.add(
    new GPUNearRepeatKnox({
      pairs: getPairs(context, rows),
      times: view('float32', rows),
      temporalContext: {
        unit: 'second',
        epoch: 1_700_000_000n,
        encoding: 'float32-relative',
        axis: {kind: 'irregular'}
      },
      parameters: view('uint32', GPU_SPACE_TIME_PARAMETER_LENGTH),
      distanceEdges,
      timeEdges,
      maximumPermutations,
      output: {
        counts: view('uint32', (maximumPermutations + 1) * cells),
        summary: view('float32', cells * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH)
      }
    })
  );
}

function addLocalKnox(
  context: KernelAuditContext,
  rows: number,
  maximumPermutations: number
): void {
  const {graph, view} = context;
  graph.add(
    new GPULocalKnox({
      pairs: getPairs(context, rows),
      times: view('float32', rows),
      parameters: view('uint32', GPU_SPACE_TIME_PARAMETER_LENGTH),
      maximumPermutations,
      output: {summary: view('float32', rows * GPU_LOCAL_KNOX_SUMMARY_LENGTH)}
    })
  );
}

function addJacquezTest(
  context: KernelAuditContext,
  rows: number,
  k: number,
  maximumPermutations: number,
  stopAfterExceedances?: number
): void {
  const {graph, view} = context;
  graph.add(
    new GPUJacquezTest({
      neighbors: getPairs(context, rows),
      times: view('float32', rows),
      parameters: view('uint32', GPU_SPACE_TIME_PARAMETER_LENGTH),
      k,
      maximumPermutations,
      ...(stopAfterExceedances === undefined ? {} : {stopAfterExceedances}),
      output: {
        statistics: view('uint32', maximumPermutations + 1),
        summary: view('float32', GPU_SPACE_TIME_SUMMARY_LENGTH),
        permutationCount: stopAfterExceedances === undefined ? undefined : view('uint32', 1)
      }
    })
  );
}

/** Audit cases of the near-repeat Knox table, local Knox and Jacquez test. */
export const SPACE_TIME_STATISTICS_AUDIT_CASES: KernelAuditCase[] = [
  ...(
    [
      [300, 99],
      [100000, MAXIMUM_PERMUTATIONS]
    ] as const
  ).map(
    ([rows, permutations]): KernelAuditCase => ({
      exportName: 'GPUNearRepeatKnox',
      name: `${rows} rows ${permutations} permutations 3 by 2 bands`,
      build: context => addNearRepeatKnox(context, rows, [0, 1, 2.5, 4], [0, 1, 6], permutations)
    })
  ),
  {
    exportName: 'GPUNearRepeatKnox',
    name: '100000 rows 65536 permutations 16 by 16 bands',
    build: context =>
      addNearRepeatKnox(context, 100000, MAXIMUM_EDGES, MAXIMUM_EDGES, MAXIMUM_PERMUTATIONS)
  },
  ...(
    [
      [300, 99],
      [100000, MAXIMUM_PERMUTATIONS]
    ] as const
  ).map(
    ([rows, permutations]): KernelAuditCase => ({
      exportName: 'GPULocalKnox',
      name: `${rows} rows ${permutations} permutations`,
      build: context => addLocalKnox(context, rows, permutations)
    })
  ),
  ...(
    [
      [300, 4, 99],
      [300, GPU_JACQUEZ_MAXIMUM_K, MAXIMUM_PERMUTATIONS],
      [100000, 8, 999],
      [100000, GPU_JACQUEZ_MAXIMUM_K, MAXIMUM_PERMUTATIONS]
    ] as const
  ).map(
    ([rows, k, permutations]): KernelAuditCase => ({
      exportName: 'GPUJacquezTest',
      name: `${rows} rows k ${k} ${permutations} permutations`,
      build: context => addJacquezTest(context, rows, k, permutations)
    })
  ),
  // Besag-Clifford stopping at the largest batch count, and with a large h on many rows.
  ...(
    [
      [300, MAXIMUM_PERMUTATIONS, MAXIMUM_PERMUTATIONS],
      [100000, MAXIMUM_PERMUTATIONS, 10]
    ] as const
  ).map(
    ([rows, permutations, stopAfterExceedances]): KernelAuditCase => ({
      exportName: 'GPUJacquezTest',
      name: `${rows} rows k 8 ${permutations} permutations stopAfterExceedances ${stopAfterExceedances}`,
      build: context => addJacquezTest(context, rows, 8, permutations, stopAfterExceedances)
    })
  )
];
