// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, expect, test, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../analysis/src/testing/gpu-contributor-test-utils';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {CatalogGpuSession} from '../src/analysis/catalog-gpu-session';
import {runKeplerLagrangianDiagnostics} from '../src/analysis/lagrangian-diagnostics';
import {
  GPULagrangianFlowMap,
  GPUFiniteTimeLyapunovExponent
} from '../../analysis/src/raster-analysis';

vi.mock(
  '@luma-contributors/analysis/raster-analysis',
  () => import('../../analysis/src/raster-analysis')
);
afterEach(() => vi.restoreAllMocks());

test('FLT-10 epoch and zero-origin wrapper packets agree and FTLE reads the same position frame', async () => {
  const device = createNullWebGPUDevice();
  function createView<Format extends GPUVectorFormat>(
    graph: GPUCommandGraph,
    name: string,
    format: Format,
    length: number
  ) {
    const buffer = graph.createTransientBuffer({
      id: name,
      byteLength: Math.max(1, length) * getGPUVectorFormatInfo(format).byteLength,
      usage: Buffer.STORAGE
    });
    return graph.createDataView(buffer, {format, length});
  }
  const packets: Float32Array[] = [];
  const producers: GPUCommandNodeProducer[] = [];
  vi.spyOn(CatalogGpuSession.prototype, 'input').mockImplementation(function <
    Format extends GPUVectorFormat
  >(
    this: CatalogGpuSession,
    name: string,
    values: Float32Array | Uint32Array | Int32Array,
    format: Format
  ) {
    if (name === 'flow-parameters') packets.push(values as Float32Array);
    return createView(
      this.graph,
      name,
      format,
      values.byteLength / getGPUVectorFormatInfo(format).byteLength
    );
  });
  vi.spyOn(CatalogGpuSession.prototype, 'output').mockImplementation(function <
    Format extends GPUVectorFormat
  >(this: CatalogGpuSession, name: string, format: Format, length: number) {
    return {
      view: createView(this.graph, name, format, length),
      buffer: {} as never,
      length,
      components: getGPUVectorFormatInfo(format).byteLength / 4
    };
  });
  vi.spyOn(GPUCommandGraph.prototype, 'add').mockImplementation(producer => {
    producers.push(producer as GPUCommandNodeProducer);
  });
  vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {});
  vi.spyOn(CatalogGpuSession.prototype, 'readFloat32').mockImplementation(
    async output => new Float32Array(output.length * output.components)
  );
  vi.spyOn(CatalogGpuSession.prototype, 'readUint32').mockImplementation(
    async output => new Uint32Array(output.length)
  );
  const field = {
    width: 4,
    height: 4,
    extent: [13, -5, 1, 1] as const,
    velocitiesStart: new Float32Array(32)
  };
  for (const origin of [0, 1700000000]) {
    await runKeplerLagrangianDiagnostics(device, field, {
      timeStep: 1,
      stepCount: 2,
      startTime: origin + 3,
      sliceStartTime: origin,
      sliceEndTime: origin + 10
    });
  }
  expect(Array.from(packets[1])).toEqual(Array.from(packets[0]));
  for (let index = 0; index < producers.length; index += 2) {
    const flow = producers[index] as GPULagrangianFlowMap;
    const ftle = producers[index + 1] as GPUFiniteTimeLyapunovExponent;
    expect(ftle.props.positions).toBe(flow.props.output.positions);
    expect(ftle.props.validity).toBe(flow.props.output.validity);
  }
  // B2: preserve the original ABSOLUTE defaults while packing relative to -4.
  await runKeplerLagrangianDiagnostics(device, field, {
    timeStep: 1,
    stepCount: 2,
    sliceStartTime: -4,
    sliceEndTime: 4
  });
  expect(packets[2][4]).toBe(4); // absolute start 0 minus origin -4
  await runKeplerLagrangianDiagnostics(device, field, {
    timeStep: 1,
    stepCount: 2,
    sliceStartTime: -4
  });
  expect(packets[3][7]).toBe(6); // absolute default end |duration|=2 minus origin -4
  device.destroy();
});
