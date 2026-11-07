// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it, vi} from 'vitest';
import {GPUScalarScale} from '../../../src/gpu-dataframe/column-classification/gpu-scalar-scale';
import {prepareGPUScalarScaleControls} from '../../../src/gpu-dataframe/column-classification/scalar-scale-parameters';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {scalarScaleFixtures} from './scalar-scale-fixtures';

function createFixture(device: Device, includeValidity: boolean) {
  const graph = new GPUCommandGraph(device, {id: 'scalar-scale-test'});
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const domain = track(createInputBuffer(device, new Float32Array(8)));
  const range = track(createInputBuffer(device, new Float32Array(2)));
  const parameters = track(createInputBuffer(device, new Float32Array(4)));
  const domainView = importGraphBuffer(graph, 'domain', domain, 'float32');
  const rangeView = importGraphBuffer(graph, 'range', range, 'float32');
  const parametersView = importGraphBuffer(graph, 'parameters', parameters, 'float32');
  const chunks = [3, 0, 4].map((length, chunkIndex) => {
    const values = track(createInputBuffer(device, new Float32Array(length + 2)));
    const validity = track(createInputBuffer(device, new Uint32Array(length + 2)));
    const output = track(createOutputBuffer(device, length + 2, Buffer.VERTEX));
    const outputValidity = track(createOutputBuffer(device, length + 2));
    const valuesView = importGraphBuffer(graph, `values-${chunkIndex}`, values, 'float32');
    const validityView = importGraphBuffer(graph, `validity-${chunkIndex}`, validity, 'uint32');
    const outputView = importGraphBuffer(graph, `output-${chunkIndex}`, output, 'float32');
    const outputValidityView = importGraphBuffer(
      graph,
      `output-validity-${chunkIndex}`,
      outputValidity,
      'uint32'
    );
    graph.add(
      new GPUScalarScale({
        id: `scale-${chunkIndex}`,
        values: graph.createDataView(valuesView.buffer, {format: 'float32', length, byteOffset: 4}),
        validity: includeValidity
          ? graph.createDataView(validityView.buffer, {format: 'uint32', length, byteOffset: 4})
          : undefined,
        domain: domainView,
        range: rangeView,
        parameters: parametersView,
        maximumDomainCount: 8,
        output: {
          values: graph.createDataView(outputView.buffer, {
            format: 'float32',
            length,
            byteOffset: 4
          }),
          validity: includeValidity
            ? graph.createDataView(outputValidityView.buffer, {
                format: 'uint32',
                length,
                byteOffset: 4
              })
            : undefined
        }
      })
    );
    return {length, values, validity, output, outputValidity};
  });
  const compiled = graph.compile();
  return {compiled, chunks, domain, range, parameters, buffers};
}

it.each([
  true,
  false
])('GPUScalarScale maps native oracle fixtures across chunks and replays controls (validity=%s)', async includeValidity => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const pipelineCreation = vi.spyOn(device, 'createComputePipeline');
  const fixture = createFixture(device, includeValidity);
  const pipelineCount = pipelineCreation.mock.calls.length;
  const sourceReads = fixture.chunks.map(chunk => vi.spyOn(chunk.values, 'readAsync'));
  try {
    for (const oracle of scalarScaleFixtures) {
      const controls = prepareGPUScalarScaleControls(oracle.controls);
      const domain = new Float32Array(8);
      domain.set(controls.domain);
      fixture.domain.write(domain);
      fixture.range.write(controls.range);
      fixture.parameters.write(controls.parameters);
      let sourceOffset = 0;
      for (const chunk of fixture.chunks) {
        const values = new Float32Array(chunk.length + 2).fill(NaN);
        const validity = new Uint32Array(chunk.length + 2);
        for (let index = 0; index < chunk.length; index++) {
          const sourceIndex = sourceOffset + index;
          const sourceValue = oracle.values[sourceIndex];
          const valid =
            sourceValue !== null &&
            sourceValue !== undefined &&
            oracle.validity?.[sourceIndex] !== 0;
          values[index + 1] =
            sourceValue !== null && sourceValue !== undefined && (includeValidity || valid)
              ? sourceValue
              : NaN;
          validity[index + 1] = Number(valid);
        }
        chunk.values.write(values);
        chunk.validity.write(validity);
        chunk.output.write(new Float32Array(chunk.length + 2).fill(-777));
        chunk.outputValidity.write(new Uint32Array(chunk.length + 2).fill(777));
        sourceOffset += chunk.length;
      }
      submitGraph(device, fixture.compiled, undefined);
      sourceOffset = 0;
      for (const chunk of fixture.chunks) {
        const output = await readFloat32(chunk.output, chunk.length + 2);
        const outputValidity = await readUint32(chunk.outputValidity, chunk.length + 2);
        expect(output[0]).toBe(-777);
        expect(output[chunk.length + 1]).toBe(-777);
        expect(outputValidity[0]).toBe(777);
        expect(outputValidity[chunk.length + 1]).toBe(777);
        for (let index = 0; index < chunk.length; index++) {
          const expected = oracle.expected[sourceOffset + index] ?? null;
          const expectedValue = expected ?? oracle.controls.nullValue ?? 0;
          expect(
            Math.abs(output[index + 1] - expectedValue),
            `${oracle.name}: row ${sourceOffset + index}`
          ).toBeLessThanOrEqual(1e-5 * Math.max(1, Math.abs(expectedValue)));
          expect(outputValidity[index + 1]).toBe(includeValidity ? Number(expected !== null) : 777);
        }
        sourceOffset += chunk.length;
      }
    }
    expect(pipelineCreation.mock.calls.length).toBe(pipelineCount);
    for (const sourceRead of sourceReads) expect(sourceRead).not.toHaveBeenCalled();
  } finally {
    fixture.compiled.destroy();
    pipelineCreation.mockRestore();
    for (const sourceRead of sourceReads) sourceRead.mockRestore();
    for (const buffer of fixture.buffers) {
      // Imported source/output buffers remain usable after compiled-graph destruction.
      expect(() => buffer.write(new Uint32Array(1))).not.toThrow();
      buffer.destroy();
    }
  }
});

it('GPUScalarScale maps ordered uint32 point codes without aliasing high codes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const source = track(
    createInputBuffer(device, Uint32Array.of(2, 0, 1, 16777216, 16777217, 0xffffffff))
  );
  const domain = track(createInputBuffer(device, Float32Array.of(2, 0, 1, 16777216)));
  const range = track(createInputBuffer(device, Float32Array.of(0, 9)));
  const parameters = track(
    createInputBuffer(
      device,
      prepareGPUScalarScaleControls({
        scale: 'point',
        domain: [2, 0, 1, 16777216],
        range: [0, 9],
        nullValue: -99
      }).parameters
    )
  );
  const output = track(createOutputBuffer(device, 6, Buffer.VERTEX));
  const outputValidity = track(createOutputBuffer(device, 6));
  graph.add(
    new GPUScalarScale({
      values: importGraphBuffer(graph, 'codes', source, 'uint32'),
      domain: importGraphBuffer(graph, 'domain', domain, 'float32'),
      range: importGraphBuffer(graph, 'range', range, 'float32'),
      parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32'),
      maximumDomainCount: 4,
      output: {
        values: importGraphBuffer(graph, 'output', output, 'float32'),
        validity: importGraphBuffer(graph, 'output-validity', outputValidity, 'uint32')
      }
    })
  );
  const compiled = graph.compile();
  const sourceWrite = vi.spyOn(source, 'write');
  try {
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(output, 6)).toEqual([0, 3, 6, 9, -99, -99]);
    expect(await readUint32(outputValidity, 6)).toEqual([1, 1, 1, 1, 0, 0]);
    const singleton = prepareGPUScalarScaleControls({
      scale: 'point',
      domain: [1],
      range: [10, 20],
      nullValue: -99
    });
    domain.write(Float32Array.of(1, 0, 0, 0));
    range.write(singleton.range);
    parameters.write(singleton.parameters);
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(output, 6)).toEqual([-99, -99, 15, -99, -99, -99]);
    parameters.write(
      prepareGPUScalarScaleControls({scale: 'linear', domain: [0, 2], nullValue: -99}).parameters
    );
    submitGraph(device, compiled, undefined);
    expect(await readUint32(outputValidity, 6)).toEqual([0, 0, 0, 0, 0, 0]);
    expect(sourceWrite).not.toHaveBeenCalled();
  } finally {
    sourceWrite.mockRestore();
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
