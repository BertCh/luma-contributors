// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Rows of a positions view that no line (or no ring) owns are padding: unwritten capacity or a
// transient the graph reuses, holding anything. Results must not depend on them. Before
// 2026-11-11 GPUSnapRoundedNoding measured its lattice frame over every row and flagged every
// non-finite row, and GPUMakeValid's ring closure copied `positions[ringOffsets[last]]` into the
// closing slot of every empty ring, so a NaN there set `overflow` and a far value coarsened the
// lattice. Each case runs once without padding and again with padding rows poisoned by NaN and by
// far finite values, and expects the same result.

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSnapRoundedNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import {GPUMakeValid} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';
import {GPUPolygonPrecisionReduction} from '../../../src/gpu-spatial-analysis/geometry-refinement/index';
import type {GPUPolygonizeOutput} from '../../../src/gpu-spatial-analysis/geometry-topology/topology-types';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createTopologyOutput} from '../geometry-refinement/topology-output';

/** A square whose corners move on a coarser lattice (0.3 and 0.9 are not multiples of 1/2). */
const SQUARE = [0.3, 0.3, 0.9, 0.3, 0.9, 0.9, 0.3, 0.9];
const CLOSED_SQUARE = [...SQUARE, 0.3, 0.3];
/** NaN, and values that would widen the lattice frame to a cell of 1 or more. */
const POISONS = [Number.NaN, 3e6, -3e6];
const PRECISION = {
  vertexTolerance: 1 / 64,
  storage: 'float32',
  predicates: 'exact-or-uncertain'
} as const;

const COMPONENTS = {uint32: 1, float32: 1, float32x2: 2, float32x4: 4} as const;
type Format = keyof typeof COMPONENTS;

function createHarness(device: Device) {
  const graph = new GPUCommandGraph(device, {id: 'noding-padding-rows'});
  const buffers = new Map<GraphDataView, Buffer>();
  let serial = 0;
  const view = <Value extends Format>(format: Value, length: number, values?: number[]) => {
    const buffer = values
      ? createInputBuffer(
          device,
          format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
        )
      : createOutputBuffer(device, length * COMPONENTS[format]);
    const result = importGraphBuffer(graph, `view-${serial++}`, buffer, format, length);
    buffers.set(result, buffer);
    return result as GraphDataView<Value>;
  };
  const read = (value: GraphDataView, rows = value.length) => {
    const components = COMPONENTS[value.format as Format];
    const buffer = buffers.get(value)!;
    return value.format.startsWith('float')
      ? readFloat32(buffer, rows * components)
      : readUint32(buffer, rows * components);
  };
  const run = () => {
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    return () => {
      compiled.destroy();
      buffers.forEach(buffer => buffer.destroy());
    };
  };
  return {graph, view, read, run};
}

async function getDevice() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

/** Positions with `leading` and `trailing` padding rows of `value` around `coordinates`. */
function pad(coordinates: number[], leading: number, trailing: number, value: number) {
  return [
    ...new Array<number>(2 * leading).fill(value),
    ...coordinates,
    ...new Array<number>(2 * trailing).fill(value)
  ];
}

/** Status, lattice and result polygons of a topology output, read back. */
async function readTopology(
  harness: ReturnType<typeof createHarness>,
  output: GPUPolygonizeOutput,
  lattice: GraphDataView
) {
  const {polygons, status} = output;
  const rings = await harness.read(polygons.ringOffsets);
  return {
    count: await harness.read(status.count),
    overflow: await harness.read(status.overflow!),
    nodedOverflow: await harness.read(output.nodedSegments.status.overflow!),
    lattice: await harness.read(lattice),
    rings,
    positions: await harness.read(polygons.positions, rings[rings.length - 1])
  };
}

async function runNoder(leading: number, trailing: number, value: number) {
  const harness = createHarness(await getDevice());
  const {view} = harness;
  const capacity = 32;
  const status = {
    count: view('uint32', 1),
    requiredCount: view('uint32', 1),
    overflow: view('uint32', 1),
    candidateOverflow: view('uint32', 1)
  };
  const lattice = view('float32', 4);
  const latticeEndpoints = view('float32x4', capacity);
  harness.graph.add(
    new GPUSnapRoundedNoding({
      lines: {
        kind: 'lines',
        positions: view(
          'float32x2',
          leading + 5 + trailing,
          pad(CLOSED_SQUARE, leading, trailing, value)
        ),
        lineOffsets: view('uint32', 2, [leading, leading + 5])
      },
      gridSize: 1 / 64,
      intersectionCapacity: 16,
      output: {
        endpoints: view('float32x4', capacity),
        latticeEndpoints,
        lattice,
        sourceFeatureIds: view('uint32', capacity),
        sourceRingIds: view('uint32', capacity),
        sourceEdgeIds: view('uint32', capacity),
        sourceStartParameters: view('float32', capacity),
        sourceEndParameters: view('float32', capacity),
        status
      }
    })
  );
  const destroy = harness.run();
  const count = (await harness.read(status.count))[0];
  const result = {
    count,
    overflow: await harness.read(status.overflow),
    lattice: await harness.read(lattice),
    pieces: await harness.read(latticeEndpoints, count)
  };
  destroy();
  return result;
}

async function runPolygonize(trailing: number, value: number) {
  const harness = createHarness(await getDevice());
  const {view} = harness;
  const output = createTopologyOutput(view);
  const lattice = view('float32', 4);
  harness.graph.add(
    new GPUPolygonize({
      lines: {
        kind: 'lines',
        positions: view('float32x2', 5 + trailing, pad(CLOSED_SQUARE, 0, trailing, value)),
        lineOffsets: view('uint32', 2, [0, 5])
      },
      intersectionCapacity: 16,
      precision: PRECISION,
      snapRounding: {gridSize: 1 / 64, lattice},
      output
    })
  );
  const destroy = harness.run();
  const result = await readTopology(harness, output, lattice);
  destroy();
  return result;
}

/** One square feature; `emptyRings` empty holes follow it and `trailing` rows follow the data. */
async function runMakeValid(emptyRings: number, trailing: number, value: number) {
  const harness = createHarness(await getDevice());
  const {view} = harness;
  const output = createTopologyOutput(view);
  const lattice = view('float32', 4);
  const ringCount = 1 + emptyRings;
  harness.graph.add(
    new GPUMakeValid({
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 4 + trailing, pad(SQUARE, 0, trailing, value)),
        ringOffsets: view('uint32', ringCount + 1, [0, ...new Array<number>(ringCount).fill(4)]),
        polygonOffsets: view('uint32', 2, [0, ringCount]),
        featureOffsets: view('uint32', 2, [0, 1])
      },
      intersectionCapacity: 16,
      precision: PRECISION,
      snapRounding: {gridSize: 1 / 64, lattice},
      output
    })
  );
  const destroy = harness.run();
  const result = await readTopology(harness, output, lattice);
  destroy();
  return result;
}

/** As `runMakeValid`, through GPUPolygonPrecisionReduction (grid 1/64). */
async function runPrecisionReduction(emptyRings: number, trailing: number, value: number) {
  const harness = createHarness(await getDevice());
  const {view} = harness;
  const output = createTopologyOutput(view);
  const lattice = view('float32', 4);
  const ringCount = 1 + emptyRings;
  harness.graph.add(
    new GPUPolygonPrecisionReduction({
      gridSize: 1 / 64,
      lattice,
      intersectionCapacity: 16,
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 4 + trailing, pad(SQUARE, 0, trailing, value)),
        ringOffsets: view('uint32', ringCount + 1, [0, ...new Array<number>(ringCount).fill(4)]),
        polygonOffsets: view('uint32', 2, [0, ringCount]),
        featureOffsets: view('uint32', 2, [0, 1])
      },
      output
    })
  );
  const destroy = harness.run();
  const result = await readTopology(harness, output, lattice);
  destroy();
  return result;
}

it('GPUSnapRoundedNoding ignores rows before and after every line', async () => {
  const reference = await runNoder(0, 0, 0);
  expect(reference.overflow).toEqual([0]);
  expect(reference.count).toBe(4);
  for (const value of POISONS) {
    expect(await runNoder(2, 3, value)).toEqual(reference);
  }
});

it('GPUPolygonize with snap rounding ignores rows after its lines', async () => {
  const reference = await runPolygonize(0, 0);
  expect(reference.overflow).toEqual([0]);
  expect(reference.count).toEqual([1]);
  for (const value of POISONS) {
    expect(await runPolygonize(3, value)).toEqual(reference);
  }
});

it('GPUMakeValid with snap rounding ignores empty rings and rows after its rings', async () => {
  const reference = await runMakeValid(0, 0, 0);
  expect(reference.overflow).toEqual([0]);
  expect(reference.count).toEqual([1]);
  for (const value of POISONS) {
    expect(await runMakeValid(2, 3, value)).toEqual(reference);
  }
});

it('GPUPolygonPrecisionReduction ignores empty rings and rows after its rings', async () => {
  const reference = await runPrecisionReduction(0, 0, 0);
  expect(reference.overflow).toEqual([0]);
  expect(reference.count).toEqual([1]);
  for (const value of POISONS) {
    expect(await runPrecisionReduction(2, 3, value)).toEqual(reference);
  }
});
