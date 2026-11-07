// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {CompiledGPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  addLensViewSnapshot,
  LensCommandGraph,
  readLensViewSnapshot,
  type LensViewSnapshot
} from '../../../src/computation-lenses/lens-view-snapshot';
import type {
  RasterLensSession,
  RasterLensFormat,
  RasterLensCaptureSpec
} from '../../../src/computation-lenses/scenarios/gpu-raster/raster-lens-harness';

/** Test/benchmark session retaining actual captured buffers; no capture values are fabricated. */
export function createCoverageTrancheOneHarness(device: Device, id: string, cellCount = 9) {
  // Every transient also declares COPY_SRC, so captured buffers can be read back.
  // Transient captures cannot be read from a published compiled graph, so the harness adds one
  // end-of-run snapshot tap for them immediately before compilation.
  let transientSnapshot: LensViewSnapshot | undefined;
  let transientValues: Map<string, number[]> | undefined;
  const addTransientSnapshot = () => {
    const views = Object.fromEntries(
      [...captures]
        .filter(([, {view}]) => !importedBuffers.has(view.buffer.id))
        .map(([name, {view}]) => [name, view])
    );
    if (!transientSnapshot && Object.keys(views).length) {
      transientSnapshot = addLensViewSnapshot(graph, device, {id, views});
    }
  };
  class HarnessCommandGraph extends LensCommandGraph {
    override compile() {
      addTransientSnapshot();
      return super.compile();
    }
    override compileAsync() {
      addTransientSnapshot();
      return super.compileAsync();
    }
  }
  const graph = new HarnessCommandGraph(device, {id});
  const owned: {destroy(): void}[] = [];
  const inputs = new Map<
    string,
    {buffer: Buffer; values: Float32Array | Uint32Array | Int32Array}
  >();
  const captures = new Map<string, {view: GraphDataView; spec: RasterLensCaptureSpec}>();
  // Published compiled graphs do not expose physical buffers, so readback goes through the
  // caller-owned device buffers that back every imported view.
  const importedBuffers = new Map<string, Buffer>();
  const capture = (name: string, view: GraphDataView, spec: RasterLensCaptureSpec = {}) => {
    if (captures.has(name)) throw new Error(`Duplicate capture ${name}`);
    captures.set(name, {view, spec});
  };
  function makeView<Format extends RasterLensFormat>(
    name: string,
    format: Format,
    values: ArrayLike<number>
  ) {
    const data =
      format === 'float32'
        ? Float32Array.from(values)
        : format === 'sint32'
          ? Int32Array.from(values)
          : Uint32Array.from(values);
    const buffer = device.createBuffer({
      id: name,
      byteLength: data.byteLength,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    owned.push(buffer);
    importedBuffers.set(name, buffer);
    const view = graph.createDataView(
      graph.importBuffer({id: name, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
      {format, length: data.length}
    );
    return {view, buffer, data};
  }
  const session: RasterLensSession = {
    graph,
    device,
    input(name, format, values, spec = {role: 'input', geometry: 'raster'}) {
      const {view, buffer, data} = makeView(name, format, values);
      inputs.set(name, {buffer, values: data});
      if (spec !== false) capture(name, view, spec);
      return view;
    },
    output(name, format, spec = {role: 'output', geometry: 'raster'}, length = cellCount) {
      const {view} = makeView(name, format, new Array(length).fill(0));
      if (spec !== false) capture(name, view, spec);
      return view;
    },
    capture,
    track(resource) {
      if (resource) owned.push(resource);
    }
  };
  return {
    device,
    graph,
    session,
    inputs,
    captures,
    upload() {
      for (const {buffer, values} of inputs.values()) buffer.write(values);
    },
    async read(_compiled: CompiledGPUCommandGraph, name: string): Promise<number[]> {
      const capture = captures.get(name);
      if (!capture) throw new Error(`Missing capture ${name}`);
      const {view} = capture;
      const buffer = importedBuffers.get(view.buffer.id);
      if (!buffer) {
        // Transient capture: values were copied by the snapshot tap during the single submission.
        if (!transientSnapshot) throw new Error(`Capture ${name} needs a compiled graph`);
        if (!transientValues) {
          const frames = await readLensViewSnapshot(transientSnapshot, {graphId: id});
          transientValues = new Map(frames.map(frame => [frame.name, Array.from(frame.values)]));
        }
        const values = transientValues.get(name);
        if (!values) throw new Error(`Missing transient capture ${name}`);
        return values;
      }
      const bytes = await buffer.readAsync(view.byteOffset, view.length * 4);
      const typed =
        view.format === 'float32'
          ? new Float32Array(bytes.buffer, bytes.byteOffset, view.length)
          : view.format === 'sint32'
            ? new Int32Array(bytes.buffer, bytes.byteOffset, view.length)
            : new Uint32Array(bytes.buffer, bytes.byteOffset, view.length);
      return Array.from(typed);
    },
    destroy() {
      for (const resource of owned.reverse()) resource.destroy();
    }
  };
}
