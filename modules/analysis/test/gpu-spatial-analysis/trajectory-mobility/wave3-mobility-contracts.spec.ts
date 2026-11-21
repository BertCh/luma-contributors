// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUMobilityIndicatorsOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators-operation';
import {CPUMobilityIndicatorsOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-mobility-indicators-operation';
import {GPUModalSplitOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-modal-split-operation';
import {CPUModalSplitOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-modal-split-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
it('suppresses unusable start rows even without an output loss view', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUMobilityIndicatorsOperation, CPUMobilityIndicatorsOperation])
    for (const start of [0, 0xffffffff])
      for (const withStatus of [false, true]) {
        const result = await runOperationGraph(device, {
          inputs: {
            timestamps: {format: 'float32', values: Float32Array.of(0)},
            tracks: {format: 'uint32', values: Uint32Array.of(0)},
            centroids: {format: 'float32x2', values: Float32Array.of(0, 0)},
            starts: {format: 'uint32', values: Uint32Array.of(start)},
            durations: {format: 'float32', values: Float32Array.of(1)},
            locations: {format: 'uint32', values: Uint32Array.of(0)},
            active: {format: 'uint32', values: Uint32Array.of(1)},
            loss: {format: 'uint32', values: Uint32Array.of(1)}
          },
          operations: [
            new Operation({
              trackCount: 1,
              rowCount: 1,
              staypointCapacity: 1,
              timestamps: 'timestamps',
              staypoints: {
                tracks: 'tracks',
                centroids: 'centroids',
                startRows: 'starts',
                durations: 'durations',
                locationIds: 'locations',
                status: {count: 'active', overflow: 'loss'}
              },
              output: {
                radiusOfGyration: 'radius',
                homeLocations: 'home',
                ...(withStatus ? {status: {candidateOverflow: 'lossOut'}} : {})
              }
            })
          ],
          outputs: ['radius', 'home', ...(withStatus ? ['lossOut'] : [])]
        });
        expect(Number.isNaN(result['radius'][0])).toBe(true);
        expect(result['home'][0]).toBe(0xffffffff);
        if (withStatus) expect(result['lossOut'][0]).toBe(1);
      }
});
it('uses float32 speed boundaries and missing IDs for supplied modes in both named forms', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUModalSplitOperation, CPUModalSplitOperation])
    for (const prediction of [false, true]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: Float32Array.of(0, 0, 15 / 3.6, 0)},
          timestamps: {format: 'float32', values: Float32Array.of(0, 1)},
          offsets: {format: 'uint32', values: Uint32Array.of(0, 2)},
          tracks: {format: 'uint32', values: Uint32Array.of(0)},
          starts: {format: 'uint32', values: Uint32Array.of(0)},
          ends: {format: 'uint32', values: Uint32Array.of(2)},
          active: {format: 'uint32', values: Uint32Array.of(1)},
          modes: {format: 'uint32', values: Uint32Array.of(7)}
        },
        operations: [
          new Operation({
            positions: 'positions',
            timestamps: 'timestamps',
            trackOffsets: 'offsets',
            rowCount: 2,
            trackCount: 1,
            triplegCapacity: 1,
            triplegs: {
              tracks: 'tracks',
              startRows: 'starts',
              endRows: 'ends',
              status: {count: 'active'}
            },
            modeCount: 3,
            metric: 'count',
            ...(prediction
              ? {modeSpeedUpperBoundsFloat32: [15 / 3.6, 100 / 3.6, Infinity]}
              : {modes: 'modes'}),
            output: {modalSplit: 'split', predictedModes: 'predictions'}
          })
        ],
        outputs: ['split', 'predictions']
      });
      expect(Array.from(result['split'])).toEqual(prediction ? [0, 1, 0] : [0, 0, 1]);
      expect(Array.from(result['predictions'])).toEqual([prediction ? 1 : 0xffffffff]);
    }
});
it('offers spherical gyration with an undefined cancelling center in both named forms', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUMobilityIndicatorsOperation, CPUMobilityIndicatorsOperation])
    for (const longitudes of [
      [-1, 1],
      [179, -179],
      [0, 180]
    ])
      for (const gyrationCenter of [undefined, 'coordinate-mean', 'spherical-mean'] as const) {
        const antipodes = longitudes[0] === 0;
        const result = await runOperationGraph(device, {
          inputs: {
            timestamps: {format: 'float32', values: Float32Array.of(0, 1)},
            tracks: {format: 'uint32', values: Uint32Array.of(0, 0)},
            centroids: {
              format: 'float32x2',
              values: Float32Array.of(longitudes[0], 0, longitudes[1], 0)
            },
            starts: {format: 'uint32', values: Uint32Array.of(0, 1)},
            durations: {format: 'float32', values: Float32Array.of(1, 1)},
            locations: {format: 'uint32', values: Uint32Array.of(0, 1)},
            active: {format: 'uint32', values: Uint32Array.of(2)}
          },
          operations: [
            new Operation({
              trackCount: 1,
              rowCount: 2,
              staypointCapacity: 2,
              timestamps: 'timestamps',
              spatialContext: {
                coordinateSpace: 'longitude-latitude',
                metric: 'great-circle',
                units: 'meters',
                sphereRadius: 6371000
              },
              gyrationCenter,
              staypoints: {
                tracks: 'tracks',
                centroids: 'centroids',
                startRows: 'starts',
                durations: 'durations',
                locationIds: 'locations',
                status: {count: 'active'}
              },
              output: {radiusOfGyration: 'radius', jumpLengths: 'jumps'}
            })
          ],
          outputs: ['radius', 'jumps']
        });
        expect(result['radius'].length).toBe(1);
        expect(result['jumps'].length).toBe(2);
        expect(result['jumps'][0]).toBe(0);
        const radius = result['radius'][0];
        if (antipodes && gyrationCenter !== 'coordinate-mean')
          expect(Number.isNaN(radius)).toBe(true);
        else {
          expect(Number.isFinite(radius)).toBe(true);
          expect(radius).toBeCloseTo(
            antipodes
              ? 10007543
              : gyrationCenter === 'coordinate-mean' && longitudes[0] === 179
                ? 19903892
                : 111194.9296875,
            -2
          );
          if (!antipodes && gyrationCenter !== 'coordinate-mean') {
            // Two equal masses: radius R*pi/180 = half the two-degree jump.
            expect(Number.isFinite(result['jumps'][1])).toBe(true);
            expect(radius).toBeCloseTo(result['jumps'][1] / 2, -1);
          }
        }
      }
});
it('review classifies non-unit-duration and multistep speed boundaries consistently', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUModalSplitOperation, CPUModalSplitOperation])
    for (const multistep of [false, true]) {
      const rowCount = multistep ? 16 : 2;
      const positions = multistep
        ? Float32Array.from({length: 32}, (_, index) =>
            index % 2 === 0 && Math.floor(index / 2) % 2 === 1 ? 0.277777761220932 : 0
          )
        : Float32Array.of(0, 0, 5, 0);
      const timestamps = multistep
        ? Float32Array.from({length: 16}, (_, index) => index / 15)
        : Float32Array.of(0, 25);
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: positions},
          timestamps: {format: 'float32', values: timestamps},
          offsets: {format: 'uint32', values: Uint32Array.of(0, rowCount)},
          tracks: {format: 'uint32', values: Uint32Array.of(0)},
          starts: {format: 'uint32', values: Uint32Array.of(0)},
          ends: {format: 'uint32', values: Uint32Array.of(rowCount)},
          active: {format: 'uint32', values: Uint32Array.of(1)}
        },
        operations: [
          new Operation({
            positions: 'positions',
            timestamps: 'timestamps',
            trackOffsets: 'offsets',
            rowCount,
            trackCount: 1,
            triplegCapacity: 1,
            triplegs: {
              tracks: 'tracks',
              startRows: 'starts',
              endRows: 'ends',
              status: {count: 'active'}
            },
            metric: 'count',
            modeCount: 2,
            modeSpeedUpperBoundsFloat32: [multistep ? 15 / 3.6 : Math.fround(0.2), Infinity],
            output: {modalSplit: 'split', predictedModes: 'predicted'}
          })
        ],
        outputs: ['split', 'predicted']
      });
      expect(Array.from(result['split'])).toEqual([1, 0]);
      expect(Array.from(result['predicted'])).toEqual([0]);
    }
});
