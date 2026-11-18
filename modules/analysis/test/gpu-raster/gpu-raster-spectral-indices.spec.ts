// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import type {GPURasterBufferBand, GPURasterScalarFormat} from '@luma.gl/graph/gpu-raster';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {RecipeTestFixture} from '../gpu-spatial-analysis/recipes/recipe-harness';
import {GPURasterNBR} from '../../src/raster-analysis/gpu-raster-spectral-indices';
import {
  createSpectralIndex,
  SPECTRAL_CASES,
  SPECTRAL_REFERENCE,
  type SpectralBandFixture,
  type SpectralCase,
  type SpectralExpected
} from './raster-spectral-indices-test-utils';

function makeArray(format: GPURasterScalarFormat, values: number[]) {
  switch (format) {
    case 'uint32':
      return Uint32Array.from([913, ...values]);
    case 'sint32':
      return Int32Array.from([913, ...values]);
    case 'float32':
      return Float32Array.from([913, ...values]);
  }
}

it('G3 NBR retains its validated output and bands after physical-alias mutation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'spectral-snapshot');
  try {
    const nearInfrared = fixture.output('nir', 'float32', 1);
    nearInfrared.buffer.write(Float32Array.of(0.8));
    const output = fixture.output('index', 'float32', 1);
    const validity = fixture.output('validity', 'uint32', 1);
    const band = {
      id: 'nir',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: nearInfrared.view}
    };
    const props = {
      width: 1,
      height: 1,
      nearInfrared: band,
      shortwaveInfrared2: {
        id: 'swir',
        format: 'float32' as const,
        storage: {
          kind: 'buffer' as const,
          values: fixture.input('swir', Float32Array.of(0.2), 'float32', 1)
        }
      },
      output: output.view,
      outputValidity: validity.view
    };
    const contributor = new GPURasterNBR(props);
    props.output = fixture.graph.createDataView(
      fixture.graph.importBuffer(
        {id: 'alias', byteLength: 4, usage: nearInfrared.buffer.usage},
        nearInfrared.buffer
      ),
      {format: 'float32', length: 1}
    );
    band.storage.values = output.view;
    fixture.graph.add(contributor);
    fixture.run();
    expect((await fixture.readFloat32(output, 1))[0]).toBeCloseTo(0.6, 6);
    expect(await fixture.readUint32(validity, 1)).toEqual([1]);
    expect((await fixture.readFloat32(nearInfrared, 1))[0]).toBeCloseTo(0.8, 6);
  } finally {
    fixture.destroy();
  }
});

function importOffsetView<Format extends GPURasterScalarFormat>(
  graph: GPUCommandGraph,
  id: string,
  buffer: Buffer,
  format: Format,
  length: number
): GraphDataView<Format> {
  const handle = graph.importBuffer(
    {id, byteLength: buffer.byteLength, usage: buffer.usage},
    buffer
  );
  return graph.createDataView(handle, {format, length, byteOffset: 4});
}

function makeBand(
  graph: GPUCommandGraph,
  name: string,
  buffer: Buffer,
  fixture: SpectralBandFixture,
  validity?: GraphDataView<'uint32'>
): GPURasterBufferBand {
  const metadata = {
    id: name,
    scale: fixture.scale,
    offset: fixture.offset,
    noDataValue: fixture.noDataValue,
    validity
  };
  const length = fixture.values.length;
  switch (fixture.format) {
    case 'float32':
      return {
        ...metadata,
        format: 'float32',
        storage: {kind: 'buffer', values: importOffsetView(graph, name, buffer, 'float32', length)}
      };
    case 'uint32':
      return {
        ...metadata,
        format: 'uint32',
        storage: {kind: 'buffer', values: importOffsetView(graph, name, buffer, 'uint32', length)}
      };
    case 'sint32':
      return {
        ...metadata,
        format: 'sint32',
        storage: {kind: 'buffer', values: importOffsetView(graph, name, buffer, 'sint32', length)}
      };
  }
}

async function checkOutput(output: Buffer, validity: Buffer, expected: SpectralExpected) {
  const length = expected.values.length;
  const values = await readFloat32(output, length + 1);
  const bits = await readUint32(output, length + 1);
  const flags = await readUint32(validity, length + 1);
  expect(values[0], 'destination prefix outside the view is preserved').toBe(913);
  expect(flags[0]).toBe(913);
  expect(flags.slice(1)).toEqual(expected.validity);
  for (let row = 0; row < length; row++) {
    const reference = expected.values[row];
    if (reference === null) {
      expect(Number.isNaN(values[row + 1]), `invalid row ${row}`).toBe(true);
      expect(bits[row + 1], `canonical NaN row ${row}`).toBe(0x7fc00000);
    } else {
      const {absoluteTolerance, relativeTolerance} = SPECTRAL_REFERENCE.conventions;
      expect(Math.abs(values[row + 1] - reference), `row ${row}`).toBeLessThanOrEqual(
        absoluteTolerance + relativeTolerance * Math.abs(reference)
      );
    }
  }
}

async function runCase(device: Device, scene: SpectralCase) {
  const graph = new GPUCommandGraph(device, {id: scene.name});
  const buffers: Buffer[] = [];
  const sourceBuffers = new Map<string, {values: Buffer; validity?: Buffer}>();
  let compiled: CompiledGPUCommandGraph | undefined;
  try {
    const bands: Record<string, GPURasterBufferBand> = {};
    for (const [name, fixture] of Object.entries(scene.bands)) {
      const values = createInputBuffer(device, makeArray(fixture.format, fixture.values));
      buffers.push(values);
      const validity = fixture.validity
        ? createInputBuffer(device, makeArray('uint32', fixture.validity))
        : undefined;
      if (validity) buffers.push(validity);
      sourceBuffers.set(name, {values, validity});
      bands[name] = makeBand(
        graph,
        name,
        values,
        fixture,
        validity
          ? importOffsetView(graph, `${name}-validity`, validity, 'uint32', fixture.values.length)
          : undefined
      );
    }
    const length = scene.width * scene.height;
    const output = createOutputBuffer(device, length + 1);
    const validity = createOutputBuffer(device, length + 1);
    buffers.push(output, validity);
    output.write(Float32Array.from({length: length + 1}, () => 913));
    validity.write(Uint32Array.from({length: length + 1}, () => 913));
    const contributor = createSpectralIndex(
      scene.index,
      {
        width: scene.width,
        height: scene.height,
        output: importOffsetView(graph, 'result', output, 'float32', length),
        outputValidity: importOffsetView(graph, 'result-validity', validity, 'uint32', length),
        epsilon: scene.epsilon,
        clamp: scene.clamp
      },
      bands,
      scene.soilAdjustment
    );
    const inspected = contributor.getCommandNodes(graph);
    expect(() => Array.prototype.pop.call(inspected)).toThrow(TypeError);
    graph.add(contributor);
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    await checkOutput(output, validity, scene.expected);
    if (scene.updatedBands && scene.updatedExpected) {
      for (const [name, fixture] of Object.entries(scene.updatedBands)) {
        const source = sourceBuffers.get(name)!;
        source.values.write(makeArray(fixture.format, fixture.values));
        if (source.validity && fixture.validity)
          source.validity.write(makeArray('uint32', fixture.validity));
      }
      submitGraph(device, compiled, undefined);
      await checkOutput(output, validity, scene.updatedExpected);
    }
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) {
      expect(buffer.destroyed, 'the recipe borrows imported buffers').toBe(false);
      buffer.destroy();
    }
  }
}

for (const scene of SPECTRAL_CASES) {
  it(`spectral indices match pinned spyndex: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    await runCase(device, scene);
  });
}

it('spectral indices reject nonfinite raw samples and preserve exact integer nodata', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  await runCase(device, {
    name: 'spectral-raw-validity',
    index: 'nbr',
    width: 3,
    height: 2,
    epsilon: 0,
    soilAdjustment: 0.5,
    bands: {
      nearInfrared: {
        format: 'uint32',
        values: [4294967295, 4294967294, 4, 4, 4, 4],
        noDataValue: 4294967295,
        scale: 0.25
      },
      shortwaveInfrared2: {format: 'float32', values: [1, 1, NaN, Infinity, -Infinity, 0.25]}
    },
    expected: {values: [null, 1, null, null, null, 0.6], validity: [0, 1, 0, 0, 0, 1]}
  });
});
