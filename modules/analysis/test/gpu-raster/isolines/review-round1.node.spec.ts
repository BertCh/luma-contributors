// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from 'node:buffer';
import {runInNewContext} from 'node:vm';
import {expect, it, vi} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUIsolines,
  decodeGPUIsolinesPolylines
} from '../../../src/raster-analysis/isolines/gpu-isolines';
import {computeIsolinesOnCPU} from '../../../src/raster-analysis/isolines/cpu-isolines-operation';
import {getGPUIsolinesParameterValues} from '../../../src/raster-analysis/isolines/isolines-parameters';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

for (const stitch of [false, true])
  it(`B1 publishes against segment capacity with unequal level width, stitching ${stitch}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const output = {
        segments: createTransientView(graph, 'segments', 'float32x4', 1),
        segmentLevels: createTransientView(graph, 'segmentLevels', 'uint32', 3),
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1),
          polylineOverflow: createTransientView(graph, 'polylineOverflow', 'uint32', 1)
        },
        polylines: stitch
          ? {
              vertices: createTransientView(graph, 'vertices', 'float32x2', 4),
              polylineOffsets: createTransientView(graph, 'polylineOffsets', 'uint32', 2),
              polylineLevels: createTransientView(graph, 'polylineLevels', 'uint32', 1),
              polylineClosed: createTransientView(graph, 'polylineClosed', 'uint32', 1),
              polylineCount: createTransientView(graph, 'polylineCount', 'uint32', 1),
              vertexCount: createTransientView(graph, 'vertexCount', 'uint32', 1)
            }
          : undefined
      };
      const nodes = new GPUIsolines({
        grid: makeGPURasterMetadata({width: 3, height: 2}),
        values: createTransientView(graph, 'values', 'float32', 6),
        levels: createTransientView(graph, 'levels', 'float32', 1),
        parameters: createTransientView(graph, 'parameters', 'float32', 8),
        output
      }).getCommandNodes(graph);
      const publish = getGPUCommandNodeKernelSources(nodes).find(
        kernel => kernel.id === 'isolines-publish'
      )!;
      expect(publish.constants['CAPACITY']).toBe(1);
    } finally {
      device.destroy();
    }
  });

const polylines = {
  vertices: Float32Array.of(0, 0),
  polylineOffsets: Uint32Array.of(0, 1),
  polylineLevels: Uint32Array.of(0),
  polylineClosed: Uint32Array.of(0),
  polylineCount: Uint32Array.of(1),
  vertexCount: Uint32Array.of(1)
};
const status = {overflow: Uint32Array.of(0), polylineOverflow: Uint32Array.of(0)};

it('TSC-04 explicitly reads uint32 CSR words and float32 coordinates from offset bytes', () => {
  // One retained XY vertex at (1.5, -2.25), level 7, offsets [0,1]; trailing words are not live.
  const vertices = new Uint8Array(Float32Array.of(99, 1.5, -2.25, NaN).buffer, 4, 12);
  const offsets = new Uint8Array(Uint32Array.of(99, 0, 1, 99).buffer, 4, 12);
  const input = {polylines: {...polylines, vertices, polylineOffsets: offsets}, status};
  expect(decodeGPUIsolinesPolylines(input, {levels: Float32Array.of(7)}).polylines).toEqual([
    {coordinates: [[1.5, -2.25]], levelIndex: 0, levelValue: 7, closed: false}
  ]);
  const partialOffsets = new Uint8Array(9);
  partialOffsets.set(new Uint8Array(Uint32Array.of(0, 1).buffer));
  expect(() =>
    decodeGPUIsolinesPolylines(
      {polylines: {...input.polylines, polylineOffsets: partialOffsets}, status},
      {levels: Float32Array.of(7)}
    )
  ).toThrow(/byte length must be a multiple of 4; received 9/);
  for (const [polylineOffsets, diagnostic] of [
    [Uint32Array.of(0), /at least 2/],
    [Uint32Array.of(1, 1), /start at 0/]
  ] as const) {
    expect(() =>
      decodeGPUIsolinesPolylines(
        {polylines: {...input.polylines, polylineOffsets}, status},
        {levels: Float32Array.of(7)}
      )
    ).toThrow(diagnostic);
  }
});

it('N1 rejects undocumented typed-array brands before decoding bytes', () => {
  for (const name of Object.keys(polylines)) {
    const input = {polylines: {...polylines}, status};
    Reflect.set(input.polylines, name, Float64Array.of(1, 2));
    expect(() => decodeGPUIsolinesPolylines(input, {levels: Float32Array.of(7)}), name).toThrow(
      /typed-array|Float32Array|Uint32Array/
    );
  }
  for (const invalid of [
    Float64Array.of(1, 2),
    Int32Array.of(1, 2),
    new DataView(new ArrayBuffer(16)),
    Uint32Array.of(1, 2)
  ]) {
    const input = {polylines: {...polylines}, status};
    Reflect.set(input.polylines, 'vertices', invalid);
    expect(() => decodeGPUIsolinesPolylines(input, {levels: Float32Array.of(7)})).toThrow(
      /Float32Array/
    );
  }
  for (const name of ['overflow', 'polylineOverflow']) {
    const invalidStatus = {...status};
    Reflect.set(invalidStatus, name, Float32Array.of(0));
    expect(() =>
      decodeGPUIsolinesPolylines({polylines, status: invalidStatus}, {levels: Float32Array.of(7)})
    ).toThrow(/Uint32Array/);
  }
  const options = {levels: Float32Array.of(7)};
  Reflect.set(options, 'levels', Uint32Array.of(7));
  expect(() => decodeGPUIsolinesPolylines({polylines, status}, options)).toThrow(/Float32Array/);
});

it('N1 retains cross-realm typed words and byte/ArrayBuffer readbacks', () => {
  const vertices = runInNewContext('Float32Array.of(1.5, -2.25)') as Float32Array;
  const polylineOffsets = runInNewContext('Uint32Array.of(0, 1)') as Uint32Array;
  const result = decodeGPUIsolinesPolylines(
    {polylines: {...polylines, vertices, polylineOffsets}, status},
    {levels: Float32Array.of(7).buffer}
  );
  expect(result.polylines[0].coordinates).toEqual([[1.5, -2.25]]);
  expect(result.complete).toBe(true);
});

it('B3 respects Buffer and foreign-realm logical byte ranges for levels and loss words', () => {
  const levels = Buffer.from(Float32Array.of(7, 2).buffer).subarray(4);
  expect(decodeGPUIsolinesPolylines({polylines, status}, {levels}).polylines[0].levelValue).toBe(2);
  const loss = Buffer.from(Uint32Array.of(0, 1).buffer).subarray(4);
  expect(
    decodeGPUIsolinesPolylines({polylines, status: {...status, polylineOverflow: loss}}, {levels})
      .complete
  ).toBe(false);
  const foreignLevels = runInNewContext(
    'new Uint8Array([0,0,224,64,0,0,0,64]).subarray(4)'
  ) as Uint8Array;
  const foreignLoss = runInNewContext(
    'new Uint8Array([0,0,0,0,1,0,0,0]).subarray(4)'
  ) as Uint8Array;
  const decoded = decodeGPUIsolinesPolylines(
    {polylines, status: {...status, polylineOverflow: foreignLoss}},
    {levels: foreignLevels}
  );
  expect(decoded.polylines[0].levelValue).toBe(2);
  expect(decoded.complete).toBe(false);
});

it('S6 refuses to certify a CPU result when stitching was not requested', () => {
  const grid = makeGPURasterMetadata({width: 3, height: 2});
  const input = {
    values: Float32Array.of(0, 0, 0, 4, 4, 4),
    levels: Float32Array.of(2),
    parameters: getGPUIsolinesParameterValues({levelCount: 1, extent: [0, 0, 3, 2]}, grid)
  };
  const segmentsOnly = computeIsolinesOnCPU(input, {grid, segmentCapacity: 2});
  expect(segmentsOnly.count[0]).toBe(2);
  expect(() =>
    decodeGPUIsolinesPolylines(
      {polylines: segmentsOnly, status: segmentsOnly},
      {levels: input.levels}
    )
  ).toThrow(/stitch|vertexCapacity/);
});

it('U1 exposes the decoder through the public raster-analysis entry', async () => {
  const publicAPI = await import('@luma-contributors/analysis/raster-analysis');
  expect(publicAPI).toHaveProperty('decodeGPUIsolinesPolylines', decodeGPUIsolinesPolylines);
});

import {readFileSync} from 'node:fs';
import {ModuleKind, transpileModule} from 'typescript';

it('B6 executes every host-export documentation fence independently with public imports', async () => {
  const publicAPI = await import('@luma-contributors/analysis/raster-analysis');
  const text = readFileSync(
    new URL('../../../../../docs/api/gpu-raster/isolines-tiles.md', import.meta.url),
    'utf8'
  ).split('## Reading stitched geometry on the host')[1];
  const snippets = Array.from(text.matchAll(/```ts\n([\s\S]*?)```/g), match => match[1]);
  const results: unknown[] = [];
  for (const source of snippets) {
    // Transpile and execute only: compiler/typecheck qualification remains the lead's gate.
    const {outputText} = transpileModule(source, {compilerOptions: {module: ModuleKind.CommonJS}});
    runInNewContext(outputText, {
      exports: {},
      require: (name: string) => {
        expect(name).toBe('@luma-contributors/analysis/raster-analysis');
        return publicAPI;
      },
      console: {log: (value: unknown) => results.push(value)}
    });
  }
  expect(snippets).toHaveLength(1);
  expect(results[0]).toMatchObject({
    type: 'FeatureCollection',
    features: [{properties: {level: 0.5}}, {properties: {level: 1.5}}]
  });
});

import * as packedReadback from '../../../src/analysis-core/kernels/packed-readback';
import type {
  GPUIsolinesFloatReadback,
  GPUIsolinesUintReadback,
  GPUIsolinesPolylineReadback,
  GPUIsolinesDecodedPolylines
} from '@luma-contributors/analysis/raster-analysis';

it('B3 copies only live byte prefixes through the shared packed reader', () => {
  const read = vi.spyOn(packedReadback, 'getPackedFloat32Values');
  try {
    const vertices = new Float32Array(10000);
    const levels: GPUIsolinesFloatReadback = new Float32Array(10000).fill(7);
    const overflow: GPUIsolinesUintReadback = Uint32Array.of(0);
    const input: GPUIsolinesPolylineReadback = {
      polylines: {...polylines, vertices},
      status: {...status, overflow}
    };
    const result: GPUIsolinesDecodedPolylines = decodeGPUIsolinesPolylines(input, {levels});
    expect(result.polylines[0].levelValue).toBe(7);
    expect(read.mock.calls.map(([bytes]) => (bytes as Uint8Array).byteLength)).toEqual([8, 4]);
  } finally {
    read.mockRestore();
  }
});
