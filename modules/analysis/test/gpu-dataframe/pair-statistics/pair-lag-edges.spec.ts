// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUCrossVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-cross-variogram';
import {
  GPUSpaceTimeVariogram,
  getGPUSpaceTimeVariogramParameterValues
} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-variogram';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

for (const legacy of [true, false])
  for (const gridSize of [undefined, [1, 1] as const, [3, 2] as const])
    for (const [lagCount, temporalLagCount] of [
      [29, 1],
      [1, 29]
    ]) {
      it(`B1 exact lag edges cross cells: ${legacy ? 'legacy' : 'positive'} ${JSON.stringify(gridSize)} (${lagCount},${temporalLagCount})`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const graph = new GPUCommandGraph(device);
        const buffers: Buffer[] = [];
        const input = <Format extends 'float32' | 'float32x2'>(
          name: string,
          format: Format,
          values: Float32Array
        ) => {
          const buffer = createInputBuffer(device, values);
          buffers.push(buffer);
          return importGraphBuffer(
            graph,
            name,
            buffer,
            format,
            values.length / (format === 'float32x2' ? 2 : 1)
          );
        };
        const bins = legacy ? lagCount * temporalLagCount : (lagCount + 1) * (temporalLagCount + 1);
        const output = <Format extends 'float32' | 'uint32'>(name: string, format: Format) => {
          const buffer = createOutputBuffer(device, bins);
          buffers.push(buffer);
          return {buffer, view: importGraphBuffer(graph, name, buffer, format, bins)};
        };
        const semivariances = output('semivariances', 'float32');
        const pairCounts = output('counts', 'uint32');
        const status = output('status', 'uint32');
        const props = {
          positions: input('positions', 'float32x2', Float32Array.of(0, 0, 30, 0, 60, 0)),
          values: input('values', 'float32', Float32Array.of(0, 2, 4)),
          secondValues: input('second', 'float32', Float32Array.of(0, 1, 2)),
          times: input('times', 'float32', Float32Array.of(0, 30, 60)),
          parameters: input(
            'parameters',
            'float32',
            legacy
              ? Float32Array.of(58, 58)
              : getGPUSpaceTimeVariogramParameterValues({
                  bounds: [0, -1, 60, 1],
                  maximumDistance: 58,
                  maximumTimeLag: 58
                })
          ),
          lagCount,
          temporalLagCount,
          output: {
            semivariances: semivariances.view,
            pairCounts: pairCounts.view,
            status: status.view
          }
        };
        graph.add(
          legacy
            ? new GPUCrossVariogram({...props, gridSize})
            : new GPUSpaceTimeVariogram({...props, gridSize: gridSize ?? [256, 256]})
        );
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          // 01 and 12 each have distance/time lag 30 and cross term (2*1)/2=1.
          // 02 has lag 60>58 and is excluded. Width=58/29=2; both edges start bin 15.
          // With two effective columns, row 2 at x=60 lies beyond the cell boundary x=58.
          const spatialClass = (lagCount === 29 ? 15 : 0) + Number(!legacy);
          const temporalClass = (temporalLagCount === 29 ? 15 : 0) + Number(!legacy);
          const selected = spatialClass * (temporalLagCount + Number(!legacy)) + temporalClass;
          const counts = new Array<number>(bins).fill(0);
          counts[selected] = 2;
          expect(await readUint32(pairCounts.buffer, bins)).toEqual(counts);
          expect(await readUint32(status.buffer, bins)).toEqual(
            counts.map(count => (count ? 0 : 1))
          );
          const estimates = await readFloat32(semivariances.buffer, bins);
          expect(estimates).toHaveLength(bins);
          for (let bin = 0; bin < bins; bin++) {
            if (bin === selected) {
              expect(Number.isFinite(estimates[bin])).toBe(true);
              expect(estimates[bin]).toBeCloseTo(1, 5);
            } else expect(Number.isNaN(estimates[bin])).toBe(true);
          }
        } finally {
          compiled.destroy();
          buffers.forEach(buffer => buffer.destroy());
        }
      });
    }
