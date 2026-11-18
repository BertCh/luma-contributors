// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {compileProjectionProgram} from '@luma.gl/graph/gpu-project';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {getRasterGroundScale} from '../../src/analysis-core/contracts/raster-grid';
import {
  getGPURasterReprojectionParameterValues,
  GPURasterReprojection
} from '../../src/raster-analysis/gpu-raster-reprojection';
import {
  getGPURasterPatchMetricsParameterValues,
  GPURasterPatchMetrics
} from '../../src/raster-analysis/raster-patches/gpu-raster-patch-metrics';
import {WeightsRig} from '../gpu-spatial-analysis/spatial-weights/spatial-weights-harness';
import {readFloat32, readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

for (const precision of ['same-crs', 'local-f32', 'double-single'] as const) {
  for (const pixelInterpretation of ['area', 'point'] as const) {
    it(`re-encodes affine reprojection with sub-cell origins: ${precision}/${pixelInterpretation}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new WeightsRig(device);
      const inverseProjection =
        precision === 'same-crs'
          ? undefined
          : compileProjectionProgram({
              precision,
              ...(precision === 'local-f32' ? {destinationOrigin: [500000, 4600000] as const} : {}),
              operations: [{type: 'affine', scale: [1, 1], offset: [500000, 4600000]}]
            });
      const makeGrids = (frame: number) => {
        const cellSize = frame === 0 ? 0.25 : 0.5;
        const factor = frame === 0 ? 1 : 0.5;
        const shift = frame === 0 ? [2.125, 1.625] : [5.125, 2.375];
        const sourceGrid: GPURasterMetadata = {
          width: 32,
          height: 8,
          affine: [
            cellSize,
            cellSize / 2,
            500000.3 + frame / 8,
            -cellSize / 2,
            cellSize,
            4600000.1 - frame / 8
          ],
          pixelInterpretation
        };
        const [columnX, rowX, originX, columnY, rowY, originY] = sourceGrid.affine;
        const destinationGrid: GPURasterMetadata = {
          width: 17,
          height: 3,
          pixelInterpretation,
          affine: [
            columnX * factor,
            rowX * factor,
            originX + columnX * shift[0] + rowX * shift[1] - (inverseProjection ? 500000 : 0),
            columnY * factor,
            rowY * factor,
            originY + columnY * shift[0] + rowY * shift[1] - (inverseProjection ? 4600000 : 0)
          ]
        };
        return {sourceGrid, destinationGrid, factor, shift};
      };
      const initial = makeGrids(0);
      // A nonzero view offset also verifies all per-frame reads use the graph view's range.
      const parameters = new GPUParameterBuffer(device, {
        id: 'affines',
        format: 'float32',
        length: 24
      });
      const importedParameters = parameters.importToGraph(rig.graph);
      const parameterView = rig.graph.createDataView(importedParameters.buffer, {
        format: 'float32',
        byteOffset: 16,
        length: 20
      });
      const values = rig.input(
        Float32Array.from({length: 256}, (_, index) => Math.floor(index / 32) * 64 + (index % 32)),
        'float32',
        256
      );
      const outputs = (['nearest', 'bilinear'] as const).map(method => {
        const output = rig.output('float32', 53);
        const validity = rig.output('uint32', 53);
        output.buffer.write(new Float32Array(53).fill(-777));
        validity.buffer.write(new Uint32Array(53).fill(777));
        const contributor = new GPURasterReprojection({
          ...initial,
          id: `reprojection-${method}`,
          inverseProjection,
          method,
          parameters: parameterView,
          input: {id: 'plane', format: 'float32', storage: {kind: 'buffer', values}},
          output: {values: output.view, validity: validity.view}
        });
        rig.graph.add(contributor);
        return {method, output, validity, contributor};
      });
      const compiled = rig.graph.compile();
      try {
        for (let frame = 0; frame < 2; frame++) {
          const grids = makeGrids(frame);
          parameters.write(
            getGPURasterReprojectionParameterValues({...grids, inverseProjection}),
            4
          );
          submitGraph(device, compiled, undefined);
          for (const {method, output, validity} of outputs) {
            const result = await readFloat32(output.buffer, 53);
            expect((await readUint32(validity.buffer, 53)).slice(0, 51)).toEqual(Array(51).fill(1));
            for (let index = 0; index < 51; index++) {
              const pixelShift = pixelInterpretation === 'area' ? 0.5 : 0;
              const column =
                grids.factor * ((index % 17) + pixelShift) + grids.shift[0] - pixelShift;
              const row =
                grids.factor * (Math.floor(index / 17) + pixelShift) + grids.shift[1] - pixelShift;
              const expected =
                method === 'bilinear'
                  ? row * 64 + column
                  : Math.floor(row + 0.5) * 64 + Math.floor(column + 0.5);
              expect(Math.abs(result[index] - expected)).toBeLessThan(0.002);
            }
            // Padding is caller-owned; neither a partial workgroup nor a repeat may overwrite it.
            expect(result.slice(51)).toEqual([-777, -777]);
            expect((await readUint32(validity.buffer, 53)).slice(51)).toEqual([777, 777]);
          }
        }
      } finally {
        compiled.destroy();
        for (const {contributor} of outputs) contributor.destroy();
        parameters.destroy();
        rig.destroy();
      }
    });
  }
}

for (const cellSizeMode of ['uniform', 'geographic', 'web-mercator'] as const) {
  it(`patch area and perimeter follow repacked resolution: ${cellSizeMode}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const grid: GPURasterMetadata = {
      width: 17,
      height: 3,
      affine: [1, 0, 10, 0, -1, 61],
      pixelInterpretation: 'area'
    };
    const parameters = new GPUParameterBuffer(device, {
      id: 'patch-affine',
      format: 'float32',
      length: 8
    });
    const areas = rig.output('float32', 2);
    const perimeters = rig.output('float32', 2);
    const counts = rig.output('uint32', 2);
    rig.graph.add(
      new GPURasterPatchMetrics({
        grid,
        cellSizeMode,
        parameters: parameters.importToGraph(rig.graph),
        labels: rig.input(new Uint32Array(51).fill(1), 'uint32', 51),
        output: {areas: areas.view, perimeters: perimeters.view, pixelCounts: counts.view}
      })
    );
    const compiled = rig.graph.compile();
    try {
      const affines: GPURasterMetadata['affine'][] = [
        [1, 0, 10, 0, -1, 61],
        [2, 0, 20, 0, -0.75, 31]
      ];
      if (cellSizeMode === 'uniform') affines.push([3, 1, 10, 4, 2, 20]);
      for (const affine of affines) {
        parameters.write(getGPURasterPatchMetricsParameterValues({...grid, affine}));
        submitGraph(device, compiled, undefined);
        let expectedArea = 0;
        let expectedPerimeter = 0;
        for (let row = 0; row < 3; row++) {
          const [scaleX, scaleY] = getRasterGroundScale(
            cellSizeMode,
            affine[5] + affine[4] * (row + 0.5)
          );
          expectedArea +=
            17 * Math.abs(affine[0] * affine[4] - affine[1] * affine[3]) * scaleX * scaleY;
          expectedPerimeter += 2 * Math.hypot(affine[1], affine[4]) * scaleY;
          if (row === 0 || row === 2)
            expectedPerimeter += 17 * Math.hypot(affine[0], affine[3]) * scaleX;
        }
        const measuredArea = await readFloat32(areas.buffer, 2);
        const measuredPerimeter = await readFloat32(perimeters.buffer, 2);
        expect(Math.abs(measuredArea[0] / expectedArea - 1)).toBeLessThan(2e-6);
        expect(Math.abs(measuredPerimeter[0] / expectedPerimeter - 1)).toBeLessThan(2e-6);
        expect(measuredArea[1]).toBe(0);
        expect(measuredPerimeter[1]).toBe(0);
        expect(await readUint32(counts.buffer, 2)).toEqual([51, 0]);
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      rig.destroy();
    }
  });
}

for (const cellSizeMode of ['uniform', 'web-mercator'] as const) {
  it(`keeps extreme dynamic patch perimeters finite and unused slots zero: ${cellSizeMode}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const grid: GPURasterMetadata = {
      width: 1,
      height: 1,
      affine: [1e20, 0, 0, 0, 1e-20, 0],
      pixelInterpretation: 'area'
    };
    const parameters = new GPUParameterBuffer(device, {
      id: 'extreme-patch-affine',
      format: 'float32',
      length: 8
    });
    const perimeters = rig.output('float32', 2);
    rig.graph.add(
      new GPURasterPatchMetrics({
        grid,
        cellSizeMode,
        parameters: parameters.importToGraph(rig.graph),
        labels: rig.input(Uint32Array.of(1), 'uint32', 1),
        output: {perimeters: perimeters.view}
      })
    );
    const compiled = rig.graph.compile();
    try {
      const frames: GPURasterMetadata[] = [
        grid,
        // Keep the enormous row's center at world y=0, where the ground scale is nonzero.
        {...grid, affine: [1e-20, 0, 0, 0, 1e20, -5e19]},
        {...grid, affine: [1e-20, 0, 0, 0, 1e-20, 0]}
      ];
      for (const frame of frames) {
        parameters.write(getGPURasterPatchMetricsParameterValues(frame));
        submitGraph(device, compiled, undefined);
        const result = await readFloat32(perimeters.buffer, 2);
        const rowCenter = frame.affine[5] + frame.affine[4] * 0.5;
        const [scaleX, scaleY] = getRasterGroundScale(cellSizeMode, rowCenter);
        // One rectangle: two column edges plus two row edges, evaluated in float64.
        const expected =
          2 * (Math.abs(frame.affine[0]) * scaleX + Math.abs(frame.affine[4]) * scaleY);
        expect(Number.isFinite(result[0])).toBe(true);
        expect(Math.abs(result[0] / expected - 1)).toBeLessThan(2e-6);
        expect(result[1]).toBe(0);
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      rig.destroy();
    }
  });
}
