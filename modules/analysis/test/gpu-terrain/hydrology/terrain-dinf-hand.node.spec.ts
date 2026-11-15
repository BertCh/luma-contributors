// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUTerrainHeightAboveDrainage,
  GPU_TERRAIN_DRAINAGE_UNRESOLVED,
  type GPUTerrainHeightAboveDrainageProps
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-height-above-drainage';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../../src/raster-analysis/raster-grid/gpu-raster-grid';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import reference from '../../oracles/terrain-dinf-hand.json';
import {createTerrainFractionalHANDNodes} from '../../../src/gpu-terrain/hydrology/terrain-fractional-hand';
import {GPUTerrainFlowFractions} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-flow-fractions';
import {
  getGPUTerrainFlowParameterValues,
  GPU_TERRAIN_FLOW_STREAM_UNRESOLVED
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = makeGPURasterMetadata({width: 3, height: 2});
  let identifier = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length = 6) =>
    createTransientView(graph, `view-${identifier++}`, format, length);
  const props: GPUTerrainHeightAboveDrainageProps = {
    id: 'hand',
    grid,
    elevation: makeGPURasterBand(view('float32'), grid),
    flowFractions: view('float32', 48),
    streams: view('uint32'),
    maximumIterations: 3,
    output: {
      heightAboveDrainage: view('float32'),
      drainageCells: view('uint32'),
      status: {
        converged: view('uint32', 1),
        iterationLimitReached: view('uint32', 1),
        iterationCount: view('uint32', 1)
      }
    }
  };
  return {device, graph, props, view};
}

it('accepts fractional routing alone and validates its input shape, routing choice and named limits', () => {
  const {device, graph, props, view} = createFixture();
  const make = (overrides: Partial<GPUTerrainHeightAboveDrainageProps>) =>
    new GPUTerrainHeightAboveDrainage({...props, ...overrides});
  expect(() => make({})).not.toThrow();
  expect(() => make({flowDirections: view('uint32')})).toThrow(/^hand .*exactly one/);
  expect(() => make({flowFractions: undefined})).toThrow(/^hand .*exactly one/);
  expect(() => make({directionEncoding: 'taudem'})).toThrow(/^hand directionEncoding/);
  expect(() => make({flowFractions: view('uint32', 48) as never})).toThrow(
    /^hand flowFractions.*float32/
  );
  expect(() => make({flowFractions: view('float32', 47)})).toThrow(/^hand flowFractions.*48/);
  expect(() => make({flowFractions: view('float32', 49)})).toThrow(/^hand flowFractions.*48/);
  for (const maximumIterations of [0, 1.5, NaN, 1025]) {
    expect(() => make({maximumIterations})).toThrow(/^hand maximumIterations.*1024/);
  }
  expect(() => make({output: {heightAboveDrainage: props.flowFractions}})).toThrow();
  // One input view with two different row layouts would still alias its output buffer.
  const sharesFractions = graph.createDataView(props.flowFractions!.buffer, {
    format: 'float32',
    length: 6
  });
  expect(() => make({output: {heightAboveDrainage: sharesFractions}})).toThrow(/share buffers/);
  device.destroy();
});

it('gates every solve/copy pair and bounds the actual node count for both routing modes', () => {
  const {device, graph, props, view} = createFixture();
  for (const fractional of [false, true]) {
    for (const maximumIterations of [1, 3, 1024]) {
      const contributor = new GPUTerrainHeightAboveDrainage({
        ...props,
        id: `hand-${fractional}-${maximumIterations}`,
        flowFractions: fractional ? props.flowFractions : undefined,
        flowDirections: fractional ? undefined : view('uint32'),
        maximumIterations
      });
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      expect(new Set(nodes.map(node => node.id)).size).toBe(nodes.length);
      if (fractional) {
        const rounds = nodes.filter(node => /average-(round|copy)-/.test(node.id));
        expect(rounds).toHaveLength(2 * maximumIterations);
        expect(rounds.every(node => Boolean(node.condition))).toBe(true);
        expect(nodes.filter(node => /average-gate-/.test(node.id))).toHaveLength(maximumIterations);
      }
    }
  }
  device.destroy();
});

it('builds parseable shared kernels and checks storage binding limits before creating nodes', () => {
  const {device, graph, props} = createFixture();
  const contributor = new GPUTerrainHeightAboveDrainage(props);
  const pipelines = getRasterPipelines(contributor.getCommandNodes(graph));
  for (const pipeline of pipelines)
    expect(() => new WgslReflect(pipeline.source), pipeline.id).not.toThrow();
  const rounds = pipelines.filter(pipeline => /average-round-/.test(pipeline.id));
  expect(new Set(rounds.map(pipeline => pipeline.source)).size).toBe(1);
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 64}
  });
  expect(() => contributor.getCommandNodes(graph)).toThrow(/^hand HAND scratch.*limit 64/);
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 100}
  });
  expect(() => contributor.getCommandNodes(graph)).toThrow(/^hand flowFractions.*limit 100/);
  device.destroy();
});

it('rejects foreign fractions and shares the same caller-owned status ports for drainage-only output', () => {
  const {device, graph, props, view} = createFixture();
  const foreign = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainHeightAboveDrainage({
      ...props,
      flowFractions: createTransientView(foreign, 'foreign', 'float32', 48)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  const labels = new GPUTerrainHeightAboveDrainage({
    ...props,
    output: {drainageCells: view('uint32'), status: props.output.status}
  });
  expect(labels.getCommandNodes(graph).at(-1)?.id).toBe('hand-average-finalize');
  device.destroy();
});

// Executes the actual generated kernels and GPU-written gates in the existing WGSL interpreter.
// This checks the arithmetic here; only the headless spec checks a GPU compiler and scheduler.
function executeHAND(
  input: {
    width: number;
    height: number;
    elevation: number[];
    fractions: number[];
    streams: number[];
    spacing?: number[];
    flowDirections?: number[];
  },
  maximumIterations = 16,
  generateFractions = false
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = makeGPURasterMetadata({
    width: input.width,
    height: input.height,
    cellSize: [input.spacing?.[0] ?? 1, input.spacing?.[1] ?? 1]
  });
  const count = grid.width * grid.height;
  const elevation = createTransientView(graph, 'elevation', 'float32', count);
  const fractions = createTransientView(graph, 'fractions', 'float32', count * 8);
  const streams = createTransientView(graph, 'streams', 'uint32', count);
  const height = createTransientView(graph, 'height', 'float32', count);
  const drainage = createTransientView(graph, 'drainage', 'uint32', count);
  const converged = createTransientView(graph, 'converged', 'uint32', 1);
  const iterationLimitReached = createTransientView(graph, 'limit', 'uint32', 1);
  const iterationCount = createTransientView(graph, 'rounds', 'uint32', 1);
  // Canonical elevations are seeded directly: upstream raster canonicalization has barriers
  // and is covered on the GPU in the companion spec, including validity and calibration.
  const directions = createTransientView(graph, 'directions', 'uint32', count);
  const parameters = createTransientView(graph, 'parameters', 'float32', 8);
  const routingNodes = generateFractions
    ? new GPUTerrainFlowFractions({
        grid,
        elevation,
        flowDirections: directions,
        parameters,
        routing: 'd-infinity',
        output: {
          fractions,
          edgeLengths: createTransientView(graph, 'lengths', 'float32', count * 8)
        }
      }).getCommandNodes(graph)
    : [];
  const nodes = createTerrainFractionalHANDNodes(graph, {
    id: 'terrain-height-above-drainage',
    grid,
    elevation,
    fractions,
    streams,
    maximumIterations,
    output: {
      heightAboveDrainage: height,
      drainageCells: drainage,
      status: {converged, iterationLimitReached, iterationCount}
    }
  });
  const memory = createCPUGraphMemory();
  memory.write(elevation, Float32Array.from(input.elevation));
  memory.write(fractions, Float32Array.from(input.fractions));
  memory.write(streams, Uint32Array.from(input.streams));
  if (generateFractions) {
    memory.write(directions, Uint32Array.from(input.flowDirections!));
    memory.write(parameters, getGPUTerrainFlowParameterValues({}, grid));
  }
  try {
    executeCommandNodesOnCPU([...routingNodes, ...nodes], memory);
    return {
      hand: Array.from(memory.readFloat32(height)),
      drainage: Array.from(memory.readUint32(drainage)),
      status: [converged, iterationLimitReached, iterationCount].map(
        view => memory.readUint32(view)[0]
      ),
      workgroupCounts: memory.workgroupCounts
    };
  } finally {
    device.destroy();
  }
}

it('executes the HAND WGSL and Tarboton composition against the independent NumPy oracle', () => {
  for (const fixture of reference) {
    for (const generateFractions of [false, true]) {
      const result = executeHAND(fixture, 16, generateFractions);
      fixture.hand.forEach((expected, index) =>
        expect(result.hand[index], `${fixture.name}:${index}`).toBeCloseTo(expected, 4)
      );
      expect(result.drainage, fixture.name).toEqual(fixture.drainage);
      expect(result.status.slice(0, 2)).toEqual([1, 0]);
      expect(result.workgroupCounts.get('terrain-height-above-drainage-average-round-15')).toBe(0);
      expect(result.workgroupCounts.get('terrain-height-above-drainage-average-copy-15')).toBe(0);
    }
  }
});

it('executes deterministic short-budget publication and detects unresolved cycles', () => {
  const fractions = new Array(32).fill(0);
  fractions[0] = fractions[8] = fractions[16] = 1;
  const input = {width: 4, height: 1, elevation: [9, 6, 3, 0], fractions, streams: [0, 0, 0, 1]};
  const limited = executeHAND(input, 1);
  expect(limited.hand).toEqual([NaN, NaN, 3, 0]);
  expect(limited.drainage).toEqual([0xfffffffe, 0xfffffffe, 3, 3]);
  expect(limited.status).toEqual([0, 1, 1]);
  const complete = executeHAND(input, 8);
  expect(complete.hand).toEqual([9, 6, 3, 0]);
  expect(complete.status).toEqual([1, 0, 4]);
  fractions.fill(0);
  fractions[0] = fractions[12] = 1;
  const cycle = executeHAND(input, 3);
  expect(cycle.hand).toEqual([NaN, NaN, NaN, 0]);
  expect(cycle.drainage).toEqual([0xfffffffe, 0xfffffffe, 0xffffffff, 3]);
  expect(cycle.status).toEqual([0, 1, 3]);
});

it('distinguishes unknown stream membership from other nonzero stream words', () => {
  const fractions = new Array(32).fill(0);
  fractions[0] = fractions[8] = 1;
  const result = executeHAND({
    width: 4,
    height: 1,
    elevation: [10, 7, 4, 1],
    fractions,
    streams: [0, GPU_TERRAIN_FLOW_STREAM_UNRESOLVED, 1, 1]
  });
  expect(result.hand).toEqual([NaN, NaN, 0, 0]);
  expect(result.drainage).toEqual([
    GPU_TERRAIN_DRAINAGE_UNRESOLVED,
    GPU_TERRAIN_DRAINAGE_UNRESOLVED,
    2,
    3
  ]);
  expect(result.status.slice(0, 2)).toEqual([1, 0]);
  // Unlike the unknown-stream sentinel, this direction sentinel is an ordinary nonzero
  // stream membership word. The original headless fixture confused these two meanings.
  const known = executeHAND({
    width: 4,
    height: 1,
    elevation: [10, 7, 4, 1],
    fractions,
    streams: [0, 0xfffffffe, 1, 1]
  });
  expect(known.hand).toEqual([3, 0, 0, 0]);
  expect(known.drainage).toEqual([1, 1, 2, 3]);
  expect(known.status.slice(0, 2)).toEqual([1, 0]);
});
