// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {addSparseResistanceSystemNodes} from '../../../src/gpu-spatial-analysis/landscape-ecology/sparse-resistance-system';
import {addSparseResistanceSolveNodes} from '../../../src/gpu-spatial-analysis/landscape-ecology/sparse-resistance-solver';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';

const STATE_ITERATIONS = 1;
const STATE_REACHABLE = 2;
const STATE_INVALID = 3;
const STATE_BREAKDOWN = 8;
const STATE_COMPONENTS = 9;

type Problem = {
  vertexCount: number;
  edges: Uint32Array;
  conductances: Float32Array;
  source: number;
  sink: number;
  maximumIterations: number;
  tolerance: number;
  connectivityIterations?: number;
};

type Result = {solution: number[]; state: number[]; labels: number[]};

function publish(
  fixture: RecipeTestFixture,
  name: string,
  source: GraphDataView<'float32' | 'uint32'>
) {
  const output = fixture.output(name, source.format, source.length);
  const type = source.format === 'float32' ? 'f32' : 'u32';
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: name,
      operation: 'sparse-resistance-readback',
      invocationCount: source.length,
      bindings: [
        {name: 'source', view: source, type, access: 'read'},
        {name: 'output', view: output.view, type, access: 'read_write'}
      ],
      body: 'output[outputOffset+index]=source[sourceOffset+index];'
    })
  );
  return output;
}

async function solveOnGPU(name: string, problem: Problem): Promise<Result> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, name);
  try {
    const edgeCount = problem.conductances.length;
    const edges = fixture.input('edges', problem.edges, 'uint32x2', edgeCount);
    const conductances = fixture.input('conductances', problem.conductances, 'float32', edgeCount);
    const system = addSparseResistanceSystemNodes(fixture.graph, {
      id: name,
      operation: 'sparseResistanceTest',
      edges,
      conductances,
      vertexCount: problem.vertexCount,
      connectivityIterations: problem.connectivityIterations ?? 128
    });
    const solve = addSparseResistanceSolveNodes(fixture.graph, {
      id: name,
      operation: 'sparseResistanceTest',
      system,
      vertexCount: problem.vertexCount,
      source: problem.source,
      sink: problem.sink,
      maximumIterations: problem.maximumIterations,
      tolerance: problem.tolerance
    });
    const solution = publish(fixture, `${name}-solution-copy`, solve.solution);
    const state = publish(fixture, `${name}-state-copy`, system.state);
    const labels = publish(fixture, `${name}-labels-copy`, system.labels);
    fixture.run();
    return {
      solution: await fixture.readFloat32(solution, problem.vertexCount),
      state: await fixture.readUint32(state, 16),
      labels: await fixture.readUint32(labels, problem.vertexCount)
    };
  } finally {
    fixture.destroy();
  }
}

function createLCG(seed: number) {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function createLattice(width: number, height: number, getConductance: () => number) {
  const edgeList: number[] = [];
  const conductanceList: number[] = [];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const vertex = row * width + column;
      if (column + 1 < width) {
        edgeList.push(vertex, vertex + 1);
        conductanceList.push(getConductance());
      }
      if (row + 1 < height) {
        edgeList.push(vertex, vertex + width);
        conductanceList.push(getConductance());
      }
    }
  }
  return {edges: Uint32Array.from(edgeList), conductances: Float32Array.from(conductanceList)};
}

/** Float64 Jacobi-preconditioned CG on the grounded Laplacian with unit injection. */
function solveOnCPU(problem: Problem) {
  const {vertexCount, source, sink} = problem;
  const neighbors: [number, number][][] = Array.from({length: vertexCount}, () => []);
  const degree = new Float64Array(vertexCount);
  for (let edge = 0; edge < problem.conductances.length; edge++) {
    const first = problem.edges[2 * edge]!;
    const second = problem.edges[2 * edge + 1]!;
    const conductance = problem.conductances[edge]!;
    if (first === second || conductance <= 0) continue;
    neighbors[first]!.push([second, conductance]);
    neighbors[second]!.push([first, conductance]);
    degree[first]! += conductance;
    degree[second]! += conductance;
  }
  const multiply = (vector: Float64Array) => {
    const result = new Float64Array(vertexCount);
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      if (vertex === sink) continue;
      let flow = 0;
      for (const [other, conductance] of neighbors[vertex]!) {
        flow += conductance * (vector[vertex]! - vector[other]!);
      }
      result[vertex] = flow;
    }
    return result;
  };
  const inverse = Float64Array.from(degree, value => (value > 0 ? 1 / value : 0));
  const voltage = new Float64Array(vertexCount);
  const residual = new Float64Array(vertexCount);
  residual[source] = 1;
  const direction = Float64Array.from(residual, (value, vertex) => value * inverse[vertex]!);
  let rz = residual[source]! * inverse[source]!;
  for (let iteration = 0; iteration < 20000; iteration++) {
    const norm = Math.sqrt(residual.reduce((total, value) => total + value * value, 0));
    if (norm < 1e-13) break;
    const product = multiply(direction);
    let curvature = 0;
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      curvature += direction[vertex]! * product[vertex]!;
    }
    const alpha = rz / curvature;
    let nextRz = 0;
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      voltage[vertex]! += alpha * direction[vertex]!;
      residual[vertex]! -= alpha * product[vertex]!;
      nextRz += residual[vertex]! * residual[vertex]! * inverse[vertex]!;
    }
    const beta = nextRz / rz;
    rz = nextRz;
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      direction[vertex] = residual[vertex]! * inverse[vertex]! + beta * direction[vertex]!;
    }
  }
  return {voltage: Array.from(voltage), multiply};
}

/** Maximum absolute current-conservation error of `voltage` for unit injection, in float64. */
function getTrueResidual(problem: Problem, voltage: number[]) {
  const flow = new Float64Array(problem.vertexCount);
  for (let edge = 0; edge < problem.conductances.length; edge++) {
    const first = problem.edges[2 * edge]!;
    const second = problem.edges[2 * edge + 1]!;
    const current = problem.conductances[edge]! * (voltage[first]! - voltage[second]!);
    flow[first]! += current;
    flow[second]! -= current;
  }
  flow[problem.source]! -= 1;
  flow[problem.sink]! += 1;
  return Math.max(...Array.from(flow, Math.abs));
}

function expectMatchesReference(
  problem: Problem,
  result: Result,
  voltageTolerance: number,
  residualTolerance: number
) {
  const reference = solveOnCPU(problem);
  let maximumError = 0;
  reference.voltage.forEach((expected, vertex) => {
    const error = Math.abs(result.solution[vertex]! - expected) / (1 + Math.abs(expected));
    maximumError = Math.max(maximumError, error);
  });
  expect(maximumError).toBeLessThanOrEqual(voltageTolerance);
  expect(getTrueResidual(problem, result.solution)).toBeLessThanOrEqual(residualTolerance);
  return maximumError;
}

const WIDTH = 64;
const SOURCE = 32 * WIDTH + 8;
const SINK = 32 * WIDTH + 55;

it('64x64 lattice with random conductances matches a float64 CG reference', async () => {
  const random = createLCG(12345);
  const lattice = createLattice(WIDTH, WIDTH, () => 0.5 * 4 ** random());
  const problem: Problem = {
    vertexCount: WIDTH * WIDTH,
    ...lattice,
    source: SOURCE,
    sink: SINK,
    maximumIterations: 2048,
    tolerance: 1e-4
  };
  const first = await solveOnGPU('lattice', problem);
  expect(first.state[STATE_REACHABLE]).toBe(1);
  expect(first.state[STATE_COMPONENTS]).toBe(1);
  expect(first.state[STATE_INVALID]).toBe(0);
  expect(first.state[STATE_BREAKDOWN]).toBe(0);
  expect(first.state[STATE_ITERATIONS]).toBeGreaterThan(0);
  expect(first.state[STATE_ITERATIONS]).toBeLessThan(problem.maximumIterations);
  expect(first.solution[SINK]).toBe(0);
  const maximumError = expectMatchesReference(problem, first, 2e-3, 2e-4);
  console.log(
    `sparse lattice: iterations ${first.state[STATE_ITERATIONS]}, max relative voltage error ${maximumError}, true residual ${getTrueResidual(problem, first.solution)}`
  );
  // Deterministic trees: a second run is bitwise identical.
  const second = await solveOnGPU('lattice-repeat', problem);
  expect(second.solution).toEqual(first.solution);
});

it('high-contrast lattice with conductances 1/R, R in {1, 6, 15, 120}, meets the probe tolerance', async () => {
  const random = createLCG(777);
  const resistances = [1, 6, 15, 120];
  const lattice = createLattice(
    WIDTH,
    WIDTH,
    () => 1 / resistances[Math.floor(random() * resistances.length)]!
  );
  const problem: Problem = {
    vertexCount: WIDTH * WIDTH,
    ...lattice,
    source: SOURCE,
    sink: SINK,
    maximumIterations: 4096,
    tolerance: 1e-3
  };
  const result = await solveOnGPU('contrast', problem);
  expect(result.state[STATE_REACHABLE]).toBe(1);
  expect(result.state[STATE_ITERATIONS]).toBeLessThan(problem.maximumIterations);
  const maximumError = expectMatchesReference(problem, result, 1e-2, 2e-3);
  console.log(
    `sparse contrast: iterations ${result.state[STATE_ITERATIONS]}, max relative voltage error ${maximumError}, true residual ${getTrueResidual(problem, result.solution)}`
  );
});

it('256x256 unit lattice converges on the signed residual sum and matches the reference', async () => {
  const size = 256;
  const lattice = createLattice(size, size, () => 1);
  const problem: Problem = {
    vertexCount: size * size,
    ...lattice,
    source: 100 * size + 40,
    sink: 150 * size + 210,
    maximumIterations: 4096,
    tolerance: 1e-4
  };
  const start = performance.now();
  const result = await solveOnGPU('lattice-256', problem);
  const elapsed = performance.now() - start;
  expect(result.state[STATE_REACHABLE]).toBe(1);
  expect(result.state[STATE_BREAKDOWN]).toBe(0);
  const iterations = result.state[STATE_ITERATIONS]!;
  expect(iterations).toBeLessThan(problem.maximumIterations);
  const maximumError = expectMatchesReference(problem, result, 2e-3, 1e-4);
  console.log(
    `sparse lattice 256: iterations ${iterations}, GPU solve ${elapsed.toFixed(0)} ms, max relative voltage error ${maximumError}, true residual ${getTrueResidual(problem, result.solution)}`
  );
}, 120000);

it('floating components stay untouched and unreachable or invalid inputs skip the solve', async () => {
  // Vertices 0..5 form a path; 6..8 are a separate path joined to nothing; edge 5-6 is a barrier.
  const connected: Problem = {
    vertexCount: 9,
    edges: Uint32Array.from([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 2, 2]),
    conductances: Float32Array.from([1, 2, 1, 4, 1, 0, 1, 1, 3]),
    source: 0,
    sink: 4,
    maximumIterations: 64,
    tolerance: 1e-5
  };
  const result = await solveOnGPU('floating', connected);
  expect(result.state[STATE_REACHABLE]).toBe(1);
  expect(result.state[STATE_COMPONENTS]).toBe(1);
  // Series path 0-1-2-3-4 with resistances 1, 1/2, 1, 1/4 (the self edge carries no current).
  const expected = [2.75, 1.75, 1.25, 0.25, 0];
  expected.forEach((value, vertex) =>
    expect(Math.abs(result.solution[vertex]! - value)).toBeLessThan(1e-4)
  );
  for (let vertex = 6; vertex < 9; vertex++) {
    expect(result.solution[vertex], `floating vertex ${vertex}`).toBe(0);
    expect(result.labels[vertex]).not.toBe(result.labels[connected.sink]);
  }
  expect(result.labels[connected.source]).toBe(result.labels[connected.sink]);

  const separated = await solveOnGPU('separated', {...connected, source: 7});
  expect(separated.state[STATE_REACHABLE]).toBe(0);
  expect(separated.state[STATE_ITERATIONS]).toBe(0);
  expect(separated.solution.every(value => value === 0)).toBe(true);

  const invalid = await solveOnGPU('invalid', {
    ...connected,
    conductances: Float32Array.from([1, 2, 1, 4, 1, 0, 1, 1, -1])
  });
  expect(invalid.state[STATE_INVALID]).toBe(1);
  expect(invalid.state[STATE_REACHABLE]).toBe(0);
  expect(invalid.state[STATE_ITERATIONS]).toBe(0);

  const undetermined = await solveOnGPU('undetermined', {
    ...connected,
    connectivityIterations: 1
  });
  expect(undetermined.state[STATE_COMPONENTS]).toBe(0);
  expect(undetermined.state[STATE_REACHABLE]).toBe(2);
});
