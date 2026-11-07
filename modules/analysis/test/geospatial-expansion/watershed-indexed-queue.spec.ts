// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterWatershedSegmentation} from '../../src/raster-analysis/gpu-raster-watershed-segmentation';
import {makeGPURasterMetadata} from '../../src/raster-analysis/raster-grid/index';
import {InterpolationBenchmarkScope} from '../gpu-spatial-analysis/spatial-interpolation/interpolation-performance-harness';
import {requireTestDevice} from '../utils/require-test-device';
import {submitGraph} from '../utils/gpu-contributor-test-utils';
import {calculateWatershedScanReference} from './watershed-scan-reference';
import {calculateWatershedReference, createWatershedWorkload} from './backlog-performance-fixtures';

for (const featureLevel of ['core', 'max'] as const)
  for (const connectivity of [4, 8] as const)
    for (const maximumSteps of [1, 9, 35]) {
      it(`watershed ${featureLevel} parity for ${connectivity} neighbors, ${maximumSteps} steps and changed inputs`, async () => {
        const device = await getWebGPUTestDevice(featureLevel);
        requireTestDevice(device);
        const scope = new InterpolationBenchmarkScope(
          device,
          `watershed-${connectivity}-${maximumSteps}`
        );
        const count = 35;
        const values = new Float32Array(count);
        const validity = new Uint32Array(count).fill(1);
        const markers = new Uint32Array(count);
        const elevation = scope.input('elevation', values, 'float32');
        const valid = scope.input('validity', validity, 'uint32');
        const seeds = scope.input('markers', markers, 'uint32');
        scope.graph.add(
          new GPURasterWatershedSegmentation({
            grid: makeGPURasterMetadata({width: 7, height: 5}),
            input: {
              id: 'surface',
              format: 'float32',
              validity: valid,
              scale: 2,
              offset: -3,
              noDataValue: -9999,
              storage: {kind: 'buffer', values: elevation}
            },
            markers: seeds,
            connectivity,
            maximumSteps,
            output: {
              labels: scope.output('labels', count, 'uint32'),
              status: {converged: scope.output('converged', 1, 'uint32')}
            }
          })
        );
        const compiled = scope.graph.compile();
        try {
          for (const scenario of [
            'rough',
            'plateau',
            'none',
            'all',
            'invalid',
            'barriers',
            'rough'
          ]) {
            values.set(
              Float32Array.from({length: count}, (_, cell) =>
                scenario === 'plateau' ? -2 : ((cell * 17 + 13) % 11) - 5
              )
            );
            validity.fill(1);
            markers.fill(0);
            if (scenario === 'all')
              markers.set(Uint32Array.from({length: count}, (_, cell) => cell + 1));
            else if (scenario !== 'none') {
              markers[0] = 1;
              markers[count - 1] = 2;
            }
            if (scenario === 'invalid') validity.fill(0);
            if (scenario === 'barriers') {
              markers[2] = 0xfffffffe;
              markers[3] = 0xffffffff;
              markers[7] = 0xfffffffd;
              markers[8] = 3;
              validity[8] = 0;
              values[9] = NaN;
              values[10] = -9999;
              for (let row = 0; row < 5; row++) validity[row * 7 + 4] = 0;
            }
            scope.initialWrites[0].buffer.write(values);
            scope.initialWrites[1].buffer.write(validity);
            scope.initialWrites[2].buffer.write(markers);
            submitGraph(device, compiled, undefined);
            const expected = calculateWatershedScanReference({
              width: 7,
              height: 5,
              values,
              validity,
              markers,
              connectivity,
              maximumSteps,
              scale: 2,
              offset: -3,
              noDataValue: -9999
            });
            expect(await scope.read(), scenario).toEqual(expected);
          }
        } finally {
          compiled.destroy();
          scope.destroy();
        }
      });
    }

for (const maximumSteps of [6, 7, 8]) {
  it(`watershed parity at the exact ${maximumSteps}-step completion boundary`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scope = new InterpolationBenchmarkScope(device, `watershed-boundary-${maximumSteps}`);
    const values = new Float32Array(7),
      markers = new Uint32Array([1, 0, 0, 0, 0, 0, 0]);
    scope.graph.add(
      new GPURasterWatershedSegmentation({
        grid: makeGPURasterMetadata({width: 7, height: 1}),
        input: {
          id: 'chain',
          format: 'float32',
          storage: {kind: 'buffer', values: scope.input('elevation', values, 'float32')}
        },
        markers: scope.input('markers', markers, 'uint32'),
        maximumSteps,
        output: {
          labels: scope.output('labels', 7, 'uint32'),
          status: {converged: scope.output('converged', 1, 'uint32')}
        }
      })
    );
    const compiled = scope.graph.compile();
    try {
      for (const write of scope.initialWrites) write.buffer.write(write.values);
      submitGraph(device, compiled, undefined);
      expect(await scope.read()).toEqual({
        labels: maximumSteps < 7 ? [1, ...Array(6).fill(0xfffffffe)] : Array(7).fill(1),
        converged: [Number(maximumSteps >= 7)]
      });
    } finally {
      compiled.destroy();
      scope.destroy();
    }
  });
}

it('watershed maximum supported region agrees with the independent lazy-heap oracle', {
  timeout: 120000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scope = new InterpolationBenchmarkScope(device, 'watershed-maximum');
  const workload = createWatershedWorkload(128);
  scope.graph.add(
    new GPURasterWatershedSegmentation({
      grid: makeGPURasterMetadata({width: 128, height: 128}),
      input: {
        id: 'maximum',
        format: 'float32',
        storage: {kind: 'buffer', values: scope.input('elevation', workload.values, 'float32')}
      },
      markers: scope.input('markers', workload.markers, 'uint32'),
      output: {
        labels: scope.output('labels', 16384, 'uint32'),
        status: {converged: scope.output('converged', 1, 'uint32')}
      }
    })
  );
  const compiled = scope.graph.compile();
  try {
    for (const write of scope.initialWrites) write.buffer.write(write.values);
    for (const values of [workload.values, workload.updatedValues]) {
      scope.initialWrites[0].buffer.write(values);
      submitGraph(device, compiled, undefined);
      expect(await scope.read()).toEqual(calculateWatershedReference(workload, values));
    }
  } finally {
    compiled.destroy();
    scope.destroy();
  }
});

for (const featureLevel of ['core', 'max'] as const)
  for (const boundaryOffset of [-1, 0, 1]) {
    it(`watershed ${featureLevel} queue capacity boundary ${boundaryOffset}`, async () => {
      const device = await getWebGPUTestDevice(featureLevel);
      requireTestDevice(device);
      const count = Math.floor(device.limits.maxComputeWorkgroupStorageSize / 16) + boundaryOffset;
      const scope = new InterpolationBenchmarkScope(device, 'watershed-local-boundary');
      const values = new Float32Array(count);
      const markers = new Uint32Array(count);
      const elevation = scope.input('elevation', values, 'float32');
      const seeds = scope.input('markers', markers, 'uint32');
      scope.graph.add(
        new GPURasterWatershedSegmentation({
          grid: makeGPURasterMetadata({width: count, height: 1}),
          input: {id: 'surface', format: 'float32', storage: {kind: 'buffer', values: elevation}},
          markers: seeds,
          output: {
            labels: scope.output('labels', count, 'uint32'),
            status: {converged: scope.output('converged', 1, 'uint32')}
          }
        })
      );
      const compiled = scope.graph.compile();
      try {
        for (const scenario of ['alternating', 'all', 'none', 'plateau', 'alternating']) {
          for (let cell = 0; cell < count; cell++) {
            values[cell] = scenario === 'plateau' ? -2 : ((cell * 17 + 13) % 101) - 50;
            markers[cell] =
              scenario === 'none' ? 0 : scenario === 'all' || cell % 2 === 0 ? cell + 1 : 0;
          }
          scope.initialWrites[0].buffer.write(values);
          scope.initialWrites[1].buffer.write(markers);
          submitGraph(device, compiled, undefined);
          expect(await scope.read(), scenario).toEqual(
            calculateWatershedScanReference({
              width: count,
              height: 1,
              values,
              markers,
              connectivity: 4,
              maximumSteps: count
            })
          );
        }
      } finally {
        compiled.destroy();
        scope.destroy();
      }
    });
  }
