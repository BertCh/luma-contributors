// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {expect} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {
  GPURasterWalkingCost,
  getGPURasterWalkingCostParameterValues
} from '../../src/raster-analysis/gpu-raster-walking-cost';
import {GPURasterWatershedSegmentation} from '../../src/raster-analysis/gpu-raster-watershed-segmentation';
import {createWalkingWatershedTextureScene} from './walking-watershed-texture-scenes';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';

/** Disagreeing RGBA channels, raw nodata, calibration, masks and updates on one compiled graph. */
export async function checkWalkingWatershedTextures(
  route: 'walking' | 'fifo' | 'minimum-spanning-forest'
) {
  const candidateDevice = await getWebGPUTestDevice();
  requireTestDevice(candidateDevice);
  const device = candidateDevice;
  const graph = new GPUCommandGraph(device);
  const scene = createWalkingWatershedTextureScene(route);
  const {grid, model} = scene;
  const pixelCount = grid.width * grid.height;
  const firstFrame = scene.frames[0];
  const buffers: Buffer[] = [];
  const textures: Texture[] = [];
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  try {
    const track = (buffer: Buffer) => {
      buffers.push(buffer);
      return buffer;
    };
    const input = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      values: Uint32Array | Float32Array
    ) => {
      const buffer = track(createInputBuffer(device, values));
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, values.length)};
    };
    const output = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => {
      const buffer = track(createOutputBuffer(device, length));
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
    };
    const pack = (values: readonly number[]) =>
      Float32Array.from(values.flatMap(value => [101, 202, value, 404]));
    function createBand(name: string, raw: number[], scale: number, offset: number) {
      const values = input(`${name}-values`, 'float32', Float32Array.from(raw));
      const mask = input(`${name}-mask`, 'uint32', new Uint32Array(pixelCount).fill(1));
      const texture = device.createTexture({
        id: name,
        format: 'rgba32float',
        width: grid.width,
        height: 1,
        usage: Texture.SAMPLE | Texture.COPY_DST
      });
      textures.push(texture);
      texture.writeData(pack(raw));
      const textureView = graph.createTextureView(
        graph.importTexture(
          {
            id: name,
            format: 'rgba32float',
            width: grid.width,
            height: 1,
            usage: texture.props.usage
          },
          texture
        )
      );
      const metadata = {
        id: name,
        format: 'float32' as const,
        scale,
        offset,
        noDataValue: -9,
        validity: mask.view
      };
      const bands: GPURasterBand[] = [
        {...metadata, storage: {kind: 'buffer', values: values.view}},
        {...metadata, storage: {kind: 'texture', view: textureView, channel: 2}}
      ];
      return {
        bands,
        mask,
        update(rawValues: number[], flags: number[]) {
          values.buffer.write(Float32Array.from(rawValues));
          texture.writeData(pack(rawValues));
          mask.buffer.write(Uint32Array.from(flags));
        }
      };
    }
    const elevation = createBand('elevation', firstFrame.elevation, 2, 10);
    const friction = createBand('friction', firstFrame.friction, 0.5, 10);
    elevation.mask.buffer.write(Uint32Array.from(firstFrame.elevationValidity));
    friction.mask.buffer.write(Uint32Array.from(firstFrame.frictionValidity));
    const seeds = input('seeds', 'uint32', Uint32Array.from(scene.seeds));
    const parameters = input(
      'parameters',
      'float32',
      getGPURasterWalkingCostParameterValues({model, frictionWeight: 1}, grid)
    );
    const results = [0, 1].map(index => {
      const values = output(
        `result-${index}`,
        route === 'walking' ? 'float32' : 'uint32',
        pixelCount
      );
      const validity = output(`validity-${index}`, 'uint32', pixelCount);
      const converged = output(`converged-${index}`, 'uint32', 1);
      const limit = output(`limit-${index}`, 'uint32', 1);
      const status = {converged: converged.view, iterationLimitReached: limit.view};
      graph.add(
        route === 'walking'
          ? new GPURasterWalkingCost({
              id: `walking-${index}`,
              grid,
              elevation: elevation.bands[index],
              friction: friction.bands[index],
              sources: seeds.view,
              parameters: parameters.view,
              model,
              maximumIterations: 8,
              output: {
                costs: values.view as GraphDataView<'float32'>,
                validity: validity.view,
                status
              }
            })
          : new GPURasterWatershedSegmentation({
              id: `watershed-${index}`,
              grid,
              input: elevation.bands[index],
              markers: seeds.view,
              method: route,
              output: {labels: values.view as GraphDataView<'uint32'>, status}
            })
      );
      return {values, validity, converged, limit};
    });
    compiled = graph.compile();
    for (const frame of scene.frames) {
      elevation.update(frame.elevation, frame.elevationValidity);
      friction.update(frame.friction, frame.frictionValidity);
      submitGraph(device, compiled, undefined);
      for (const result of results) {
        const values = await (route === 'walking' ? readFloat32 : readUint32)(
          result.values.buffer,
          pixelCount
        );
        // The shared scene pins a nonzero uphill cost and competing watershed markers.
        // Raw -9 must still be rejected before calibration moves it to a finite usable value.
        const expected = frame.expected;
        expected.forEach((value, cell) => {
          if (Number.isNaN(value)) expect(values[cell]).toBeNaN();
          else {
            expect(Number.isFinite(values[cell])).toBe(true);
            expect(values[cell]).toBeCloseTo(value, 5);
          }
        });
        if (route === 'walking')
          expect(await readUint32(result.validity.buffer, pixelCount)).toEqual(frame.validity);
        expect(await readUint32(result.converged.buffer, 1)).toEqual([1]);
        expect(await readUint32(result.limit.buffer, 1)).toEqual([0]);
      }
    }
  } finally {
    compiled?.destroy();
    textures.forEach(texture => texture.destroy());
    buffers.forEach(buffer => buffer.destroy());
  }
}
