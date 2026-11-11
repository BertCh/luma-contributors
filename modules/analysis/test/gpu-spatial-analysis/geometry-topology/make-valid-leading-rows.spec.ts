// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Rows of `polygons.positions` before `ringOffsets[0]` belong to no ring: unwritten capacity or a
// transient the graph reuses, holding anything. Before 2026-11-11 GPUMakeValid's ring closure
// copied each such row into its closed line buffer, where the line split's non-finite check
// (which read every row) set `overflow` for a NaN. Each case runs once with ring offsets from 0
// and again with ring offsets starting at 2, the 2 leading and 3 trailing rows poisoned by NaN and
// by far finite values, with and without snap rounding, and expects the same result.

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUMakeValid} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';
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
/** NaN, and values that would widen the lattice frame to a cell of 1 or more. */
const POISONS = [Number.NaN, 3e6, -3e6];
const LEADING = 2;
const TRAILING = 3;

const COMPONENTS = {uint32: 1, float32: 1, float32x2: 2, float32x4: 4} as const;
type Format = keyof typeof COMPONENTS;

function createHarness(device: Device) {
  const graph = new GPUCommandGraph(device, {id: 'make-valid-leading-rows'});
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

async function runMakeValid(options: {
  leading: number;
  trailing: number;
  value: number;
  snapRounding: boolean;
}) {
  const {leading, trailing, value, snapRounding} = options;
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const {view} = harness;
  const output = createTopologyOutput(view);
  const lattice = view('float32', 4);
  harness.graph.add(
    new GPUMakeValid({
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', leading + 4 + trailing, [
          ...new Array<number>(2 * leading).fill(value),
          ...SQUARE,
          ...new Array<number>(2 * trailing).fill(value)
        ]),
        ringOffsets: view('uint32', 2, [leading, leading + 4]),
        polygonOffsets: view('uint32', 2, [0, 1]),
        featureOffsets: view('uint32', 2, [0, 1])
      },
      intersectionCapacity: 16,
      precision: {vertexTolerance: 1 / 64, storage: 'float32', predicates: 'exact-or-uncertain'},
      ...(snapRounding ? {snapRounding: {gridSize: 1 / 64, lattice}} : {}),
      output
    })
  );
  const destroy = harness.run();
  const {polygons, status} = output;
  const rings = await harness.read(polygons.ringOffsets);
  const result = {
    count: await harness.read(status.count),
    overflow: await harness.read(status.overflow!),
    nodedOverflow: await harness.read(output.nodedSegments.status.overflow!),
    nodedCount: await harness.read(output.nodedSegments.status.count),
    lattice: snapRounding ? await harness.read(lattice) : null,
    rings,
    positions: await harness.read(polygons.positions, rings[rings.length - 1])
  };
  destroy();
  return result;
}

for (const snapRounding of [false, true]) {
  it(`GPUMakeValid ${snapRounding ? 'with' : 'without'} snap rounding ignores rows before ringOffsets[0]`, async () => {
    const reference = await runMakeValid({leading: 0, trailing: 0, value: 0, snapRounding});
    expect(reference.overflow).toEqual([0]);
    expect(reference.count).toEqual([1]);
    for (const value of POISONS) {
      expect(
        await runMakeValid({leading: LEADING, trailing: TRAILING, value, snapRounding}),
        `padding ${value}`
      ).toEqual(reference);
    }
  });
}
