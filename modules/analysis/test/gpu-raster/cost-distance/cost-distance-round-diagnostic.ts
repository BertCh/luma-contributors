// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphBufferUse} from '@luma.gl/graph/gpu-core';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {expect} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  getGPUCostDistanceParameterValues,
  getGPURasterRelaxationContinuationStateLength,
  GPUCostDistance
} from '../../../src/raster-analysis/cost-distance/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

const MAXIMUM_ITERATIONS = 512;
type Visibility = 'parallel' | 'separate-passes' | 'snapshot' | 'forward' | 'reverse';
type RunStatus = {converged: number; iterationCount: number; iterationLimitReached: number};
type CostSummary = {finiteCount: number; barrierCount: number; firstCost: number; lastCost: number};

export function assertCostDistanceSubmissionBudget(
  previousCount: number,
  status: RunStatus,
  budget: number
): void {
  expect(status.iterationCount).toBeGreaterThan(previousCount);
  expect(status.iterationCount).toBeLessThanOrEqual(previousCount + budget);
  if (!status.converged) expect(status.iterationCount).toBe(previousCount + budget);
}

/**
 * A horizontal corridor has the hand-derived exact cost c at column c, Infinity elsewhere.
 * All edge costs and additions are integers below 128: float32 needs no tolerance here.
 * One tile needs one improving round and one unchanged round. With eight tiles a round-start
 * snapshot or a reverse serial visit needs eight improving rounds plus one unchanged round;
 * a forward serial visit needs only one improving round plus one unchanged round.
 */
export function createCostDistanceRoundRun(
  device: Device,
  width: number,
  budget: number,
  visibility: Visibility,
  height = 1,
  options: {sweepAfterIteration?: number; snapshotPlainOnly?: boolean} = {}
) {
  const grid = makeGPURasterMetadata({width, height});
  const cells = width * height;
  // Connected zero-friction serpentine: every even row is open and odd rows have one
  // connector alternating right/left. Every passable cost is exactly zero, barriers Infinity.
  // This oracle follows from connectivity and nonnegative edges; it shares no shortest-path code.
  const friction = Float32Array.from({length: cells}, (_, cell) => {
    if (height === 1) return 1;
    const row = Math.floor(cell / width);
    return row % 2 === 0 || cell % width === (row % 4 === 1 ? width - 1 : 0) ? 0 : NaN;
  });
  const expectedCosts = Array.from(friction, (value, cell) =>
    Number.isNaN(value) ? Infinity : height === 1 ? cell : 0
  );
  const graph = new GPUCommandGraph(device, {
    id: `rounds-${width}x${height}-${budget}-${visibility}`
  });
  recordWGSLKernelNodes(graph);
  const buffers = {
    friction: createInputBuffer(device, friction),
    parameters: createInputBuffer(device, getGPUCostDistanceParameterValues({}, grid)),
    sources: createInputBuffer(device, Uint32Array.of(0)),
    costs: createOutputBuffer(device, cells),
    status: createOutputBuffer(device, 129),
    state: createOutputBuffer(device, getGPURasterRelaxationContinuationStateLength(grid)),
    values: createOutputBuffer(device, cells)
  };
  const continued = budget !== MAXIMUM_ITERATIONS;
  const status = importGraphBuffer(graph, 'status', buffers.status, 'uint32', 129);
  const statusWord = (word: number) =>
    graph.createDataView(status.buffer, {format: 'uint32', byteOffset: word * 256, length: 1});
  const contributor = new GPUCostDistance({
    id: 'diagnostic',
    grid,
    friction: makeGPURasterBand(
      importGraphBuffer(graph, 'friction', buffers.friction, 'float32', cells),
      grid,
      {id: 'friction'}
    ),
    parameters: importGraphBuffer(graph, 'parameters', buffers.parameters, 'float32', 8),
    sources: importGraphBuffer(graph, 'sources', buffers.sources, 'uint32', 1),
    maximumIterations: budget,
    // Corridors isolate halo visibility; the serpentine exercises the production sweep switch.
    sweepAfterIteration: options.sweepAfterIteration ?? (height === 1 ? MAXIMUM_ITERATIONS : 96),
    continuation: continued
      ? {
          state: importGraphBuffer(
            graph,
            'state',
            buffers.state,
            'uint32',
            getGPURasterRelaxationContinuationStateLength(grid)
          ),
          values: importGraphBuffer(graph, 'values', buffers.values, 'float32', cells)
        }
      : undefined,
    output: {
      costs: importGraphBuffer(graph, 'costs', buffers.costs, 'float32', cells),
      status: {
        converged: statusWord(0),
        iterationCount: statusWord(1),
        iterationLimitReached: statusWord(2)
      }
    }
  });
  const snapshot = createTransientView(graph, 'snapshot', 'uint32', cells);
  for (const node of contributor.getCommandNodes(graph)) {
    if (
      visibility === 'parallel' ||
      visibility === 'separate-passes' ||
      !/^diagnostic-relax-\d+$/.test(node.id)
    ) {
      graph.add(node);
      continue;
    }
    // Rebuild the recorded production round with only its global read visibility changed.
    // The tile relaxation, activity stamps, changed bits, gates and continuation are unchanged.
    const kernel = getWGSLKernelNodeRecord(graph, node.id)!;
    const extraResources = node.resources?.filter(
      (resource): resource is GraphBufferUse =>
        'buffer' in resource && resource.usage === 'indirect'
    );
    let declarations = kernel.declarations!;
    let body = kernel.body;
    const bindings = [...kernel.bindings];
    if (visibility === 'snapshot') {
      const working = bindings.find(binding => binding.name === 'values')!.view;
      graph.add(
        createWGSLKernelNode(graph, {
          id: `${node.id}-snapshot`,
          operation: 'CostDistanceRoundDiagnostic',
          bindings: [
            {name: 'working', view: working, type: 'u32', access: 'read'},
            {name: 'snapshot', view: snapshot, type: 'u32', access: 'read_write'}
          ],
          invocationCount: cells,
          workgroupSize: 256,
          condition: node.condition,
          extraResources,
          body: 'snapshot[snapshotOffset + index] = working[workingOffset + index];'
        })
      );
      const originalRead = 'atomicLoad(&values[valuesOffset + cell])';
      expect(declarations).toContain(originalRead);
      declarations = declarations.replace(
        originalRead,
        options.snapshotPlainOnly
          ? `select(snapshot[snapshotOffset + cell], ${originalRead}, roundPhase != 0u)`
          : 'snapshot[snapshotOffset + cell]'
      );
      bindings.push({name: 'snapshot', view: snapshot, type: 'u32', access: 'read'});
    } else {
      // A single workgroup walks the tiles in a prescribed order, publishing each tile before
      // the next loads its halo. This selects two legal visibility extremes without a GPU spinlock.
      body = `if (workgroupIndex != 0u) { return; }
  for (var step = 0u; step < TILE_COUNT; step++) {
    let tile = ${visibility === 'reverse' ? 'TILE_COUNT - 1u - step' : 'step'};
    relaxTile(tile, localInvocationIndex);
    storageBarrier();
    workgroupBarrier();
  }`;
    }
    graph.add(
      createWGSLKernelNode(graph, {
        id: node.id,
        operation: kernel.operation,
        variant: `diagnostic-${visibility}`,
        bindings,
        invocationCount: kernel.invocationCount,
        workgroupSize: 256,
        guardIndex: false,
        declarations,
        body,
        condition: node.condition,
        extraResources
      })
    );
  }
  const compiled = graph.compile();
  const submit = async () => {
    if (visibility === 'separate-passes') {
      const encoder = device.createCommandEncoder();
      compiled.encode(encoder, {parameters: undefined, coalesceComputePasses: false});
      device.submit(encoder.finish());
    } else {
      submitGraph(device, compiled, undefined);
    }
    const [words, costs, state] = await Promise.all([
      readUint32(buffers.status, 129),
      readFloat32(buffers.costs, cells),
      continued ? readUint32(buffers.state, 6) : Promise.resolve(undefined)
    ]);
    const [converged, iterationCount, iterationLimitReached] = [words[0], words[64], words[128]];
    return {status: {converged, iterationCount, iterationLimitReached}, costs, state};
  };
  return {
    submit,
    readWorking: () => readFloat32(buffers.values, cells),
    async execute(): Promise<{status: RunStatus; history: RunStatus[]; costSummary: CostSummary}> {
      const history: RunStatus[] = [];
      for (let execution = 0; execution < 16; execution++) {
        const {status: current, costs, state} = await submit();
        const {converged, iterationCount, iterationLimitReached} = current;
        assertCostDistanceSubmissionBudget(history.at(-1)?.iterationCount ?? 0, current, budget);
        history.push(current);
        expect(iterationLimitReached).toBe(1 - converged);
        if (state) {
          expect(state.slice(0, 4)).toEqual([0, 0, iterationCount, converged]);
        }
        if (converged) {
          // In particular, a continued run may never claim convergence one round early.
          expect(costs, JSON.stringify({width, height, visibility, budget, history})).toEqual(
            expectedCosts
          );
          expect(iterationCount).toBeGreaterThanOrEqual(2);
          expect(iterationCount).toBeLessThanOrEqual(
            height === 1 ? Math.ceil(width / 16) + 1 : MAXIMUM_ITERATIONS
          );
          return {
            status: current,
            history,
            costSummary: {
              finiteCount: costs.filter(Number.isFinite).length,
              barrierCount: costs.filter(cost => cost === Infinity).length,
              firstCost: costs[0],
              lastCost: costs[costs.length - 1]
            }
          };
        }
        expect(
          costs.every((cost, cell) =>
            Number.isNaN(friction[cell]) ? cost === Infinity : Number.isNaN(cost)
          ),
          JSON.stringify({width, height, visibility, budget, history})
        ).toBe(true);
      }
      throw new Error(`No convergence: ${JSON.stringify({visibility, budget, history})}`);
    },
    destroy() {
      compiled.destroy();
      for (const buffer of Object.values(buffers)) buffer.destroy();
    }
  };
}

export async function runCostDistanceRoundMatrix(device: Device, repetitions: number) {
  const started = performance.now();
  const report: Record<string, Record<number, number>> = {};
  const histories: Record<string, RunStatus[][]> = {};
  const finalCosts: Record<string, CostSummary> = {};
  const failures: string[] = [];
  // Budgets 1 and 3 split the default four-round gate group at different boundaries;
  // 64 is the production continuation default. A single tile controls for neighbor visibility.
  for (const width of [16, 128]) {
    for (const visibility of [
      'parallel',
      'separate-passes',
      'snapshot',
      'forward',
      'reverse'
    ] as const) {
      for (const budget of [MAXIMUM_ITERATIONS, 64, 3, 1]) {
        // Isolate alpha.16's compute-pass coalescing on the multi-workgroup production paths.
        if (visibility === 'separate-passes' && (width === 16 || budget < 64)) continue;
        const key = `${width}/${visibility}/${budget}`;
        const frequencies: Record<number, number> = {};
        report[key] = frequencies;
        histories[key] = [];
        const run = createCostDistanceRoundRun(device, width, budget, visibility);
        try {
          for (let repetition = 0; repetition < repetitions; repetition++) {
            const result = await run.execute();
            finalCosts[key] = result.costSummary;
            const count = result.status.iterationCount;
            frequencies[count] = (frequencies[count] ?? 0) + 1;
            // Retain one history per observed count, keeping failure messages bounded.
            if (frequencies[count] === 1) histories[key].push(result.history);
            if ((visibility !== 'parallel' && visibility !== 'separate-passes') || width === 16) {
              const expected = visibility === 'forward' || width === 16 ? 2 : 9;
              if (count !== expected) failures.push(`${key}: expected ${expected}, got ${count}`);
            }
          }
        } finally {
          run.destroy();
        }
      }
    }
  }
  if (failures.length) {
    throw new Error(
      `Cost-distance round diagnostic: ${JSON.stringify({
        failures,
        repetitions,
        elapsedMilliseconds: Math.round(performance.now() - started),
        frequencies: report,
        finalCosts,
        histories
      })}`
    );
  }
  return {
    repetitions,
    elapsedMilliseconds: Math.round(performance.now() - started),
    frequencies: report,
    finalCosts,
    histories
  };
}

export async function runCostDistanceSerpentineDiagnostic(device: Device, repetitions: number) {
  const started = performance.now();
  const frequencies: Record<string, Record<number, number>> = {};
  const histories: Record<string, RunStatus[][]> = {};
  const finalCosts: Record<string, CostSummary> = {};
  for (const budget of [MAXIMUM_ITERATIONS, 64]) {
    const key = String(budget);
    frequencies[key] = {};
    histories[key] = [];
    const run = createCostDistanceRoundRun(device, 64, budget, 'parallel', 257);
    try {
      for (let repetition = 0; repetition < repetitions; repetition++) {
        const result = await run.execute();
        // Literal summary produced by the independent flood in the adjacent .oracle.py.
        // execute() also compares every individual cost exactly: zero or Infinity, with no NaNs.
        expect(result.costSummary).toEqual({
          finiteCount: 8384,
          barrierCount: 8064,
          firstCost: 0,
          lastCost: 0
        });
        finalCosts[key] = result.costSummary;
        const count = result.status.iterationCount;
        frequencies[key][count] = (frequencies[key][count] ?? 0) + 1;
        if (frequencies[key][count] === 1) histories[key].push(result.history);
        if (budget === 64) expect(result.history.length).toBeGreaterThan(1);
      }
    } finally {
      run.destroy();
    }
  }
  return {
    repetitions,
    elapsedMilliseconds: Math.round(performance.now() - started),
    frequencies,
    finalCosts,
    histories
  };
}
