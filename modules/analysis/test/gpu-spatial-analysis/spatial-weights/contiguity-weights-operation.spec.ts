// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUContiguityWeightsOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation';
import {GPUContiguityWeightsOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {flattenPolygons} from './spatial-weights-oracle';

it('compares adjacency-changing snaps and signed half-cells at a nonzero origin', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scenes: {
    name: string;
    polygons: Parameters<typeof flattenPolygons>[0];
    joinsAfterSnap: boolean;
  }[] = [
    {
      name: 'off-grid gap closes',
      polygons: [
        [
          [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
          ]
        ],
        [
          [
            [1.0625, 0],
            [2, 0],
            [2, 1],
            [1.0625, 1]
          ]
        ]
      ],
      joinsAfterSnap: true
    },
    {
      name: 'positive half-cell stays separate at nonzero origin',
      polygons: [
        [
          [
            [10.0625, 20.0625],
            [11.125, 20.0625],
            [11.125, 21.0625],
            [10.0625, 21.0625]
          ]
        ],
        [
          [
            [11.1875, 20.0625],
            [12.0625, 20.0625],
            [12.0625, 21.0625],
            [11.1875, 21.0625]
          ]
        ]
      ],
      joinsAfterSnap: false
    },
    {
      name: 'negative half-cell joins at nonzero origin',
      polygons: [
        [
          [
            [10.1875, 20.1875],
            [11.1875, 20.1875],
            [11.1875, 21.1875],
            [10.1875, 21.1875]
          ]
        ],
        [
          [
            [9.1875, 20.1875],
            [10.0625, 20.1875],
            [10.0625, 21.1875],
            [9.1875, 21.1875]
          ]
        ]
      ],
      joinsAfterSnap: true
    }
  ];
  // These dyadic coordinates and tolerance are exactly representable. Local snap cells are
  // 4 versus 5 at the positive half-cell, and 0 versus 0 at the negative half-cell.
  for (const scene of scenes) {
    // Prefix offsets encode contiguous ring ranges; position gaps between them are not representable.
    for (const paddingValue of [undefined, Number.NaN, 3e20, -3e20]) {
      const original = flattenPolygons(scene.polygons);
      const leading = paddingValue === undefined ? 0 : 2;
      const positions = new Float32Array(original.positions.length + (leading ? 10 : 0)).fill(
        paddingValue ?? 0
      );
      positions.set(original.positions, 2 * leading);
      const padded = {
        ...original,
        positions,
        ringOffsets: Uint32Array.from(original.ringOffsets, offset => offset + leading)
      };
      for (const criterion of ['queen', 'rook'] as const) {
        for (const snapTolerance of [0, 0.25]) {
          const props = {
            criterion,
            featureCount: 2,
            capacity: 2,
            pairCapacity: 32,
            snapTolerance,
            polygons: {
              positions: 'positions',
              ringOffsets: 'ringOffsets',
              polygonOffsets: 'polygonOffsets',
              featureOffsets: 'featureOffsets'
            },
            output: {
              offsets: 'offsets',
              neighbors: 'neighbors',
              weights: 'weights',
              status: {
                overflow: 'overflow',
                requiredCount: 'required',
                candidateOverflow: 'candidate'
              }
            }
          };
          const results: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
          for (const Operation of [GPUContiguityWeightsOperation, CPUContiguityWeightsOperation]) {
            results.push(
              await runOperationGraph(device, {
                inputs: {
                  positions: {format: 'float32x2', values: padded.positions},
                  ringOffsets: {format: 'uint32', values: padded.ringOffsets},
                  polygonOffsets: {format: 'uint32', values: padded.polygonOffsets},
                  featureOffsets: {format: 'uint32', values: padded.featureOffsets}
                },
                operations: [new Operation(props)],
                outputs: ['offsets', 'neighbors', 'weights', 'overflow', 'required', 'candidate']
              })
            );
          }
          const joins = snapTolerance > 0 && scene.joinsAfterSnap;
          for (const result of results) {
            expect(Array.from(result['offsets']), scene.name).toEqual(
              joins ? [0, 1, 2] : [0, 0, 0]
            );
            expect(Array.from(result['required'])).toEqual([joins ? 2 : 0]);
            expect(Array.from(result['overflow'])).toEqual([0]);
            expect(Array.from(result['candidate'])).toEqual([0]);
          }
          for (const name of [
            'offsets',
            'neighbors',
            'weights',
            'overflow',
            'required',
            'candidate'
          ]) {
            // Compare every defined output row exactly. Unused neighbor/weight slots are unspecified.
            const length =
              name === 'neighbors' || name === 'weights'
                ? results[1]['offsets'][2]
                : results[1][name].length;
            expect(Array.from(results[0][name].slice(0, length)), scene.name).toEqual(
              Array.from(results[1][name].slice(0, length))
            );
            if (name === 'neighbors')
              expect(Array.from(results[0][name].slice(0, length))).toEqual(joins ? [1, 0] : []);
            if (name === 'weights')
              expect(Array.from(results[0][name].slice(0, length))).toEqual(joins ? [1, 1] : []);
          }
        }
      }
    }
  }
});

it('both operations reject an unrounded negative snap tolerance before execution', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = flattenPolygons([
    [
      [
        [0, 0],
        [1, 0],
        [0, 1]
      ]
    ]
  ]);
  for (const Operation of [GPUContiguityWeightsOperation, CPUContiguityWeightsOperation]) {
    await expect(
      (async () =>
        runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: polygons.positions},
            ringOffsets: {format: 'uint32', values: polygons.ringOffsets},
            polygonOffsets: {format: 'uint32', values: polygons.polygonOffsets},
            featureOffsets: {format: 'uint32', values: polygons.featureOffsets}
          },
          operations: [
            new Operation({
              id: 'negative-tolerance',
              criterion: 'queen',
              featureCount: 1,
              capacity: 1,
              snapTolerance: -Number.MIN_VALUE,
              polygons: {
                positions: 'positions',
                ringOffsets: 'ringOffsets',
                polygonOffsets: 'polygonOffsets',
                featureOffsets: 'featureOffsets'
              },
              output: {
                offsets: 'offsets',
                neighbors: 'neighbors',
                weights: 'weights',
                status: {overflow: 'overflow'}
              }
            })
          ],
          outputs: ['offsets', 'neighbors', 'weights', 'overflow']
        }))()
    ).rejects.toThrow(/^negative-tolerance snapTolerance/);
  }
});

for (const criterion of ['queen', 'rook'] as const) {
  it(`${criterion}: compares all CSR and status rows on ordinary input and final-capacity overflow`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const polygons = flattenPolygons([
      [
        [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 1]
        ]
      ],
      [
        [
          [1, 0],
          [2, 0],
          [2, 1],
          [1, 1]
        ]
      ],
      [
        [
          [0, 1],
          [1, 1],
          [1, 2],
          [0, 2]
        ]
      ],
      [
        [
          [1, 1],
          [2, 1],
          [2, 2],
          [1, 2]
        ]
      ]
    ]);
    for (const capacity of [criterion === 'queen' ? 12 : 8, 3]) {
      for (const snapTolerance of [0, 0.25]) {
        const props = {
          criterion,
          featureCount: 4,
          capacity,
          snapTolerance,
          pairCapacity: 64,
          polygons: {
            positions: 'positions',
            ringOffsets: 'ringOffsets',
            polygonOffsets: 'polygonOffsets',
            featureOffsets: 'featureOffsets'
          },
          output: {
            offsets: 'offsets',
            neighbors: 'neighbors',
            weights: 'weights',
            status: {
              overflow: 'overflow',
              requiredCount: 'required',
              candidateOverflow: 'candidate'
            }
          }
        };
        const results = [];
        for (const Operation of [GPUContiguityWeightsOperation, CPUContiguityWeightsOperation]) {
          results.push(
            await runOperationGraph(device, {
              inputs: {
                positions: {format: 'float32x2', values: polygons.positions},
                ringOffsets: {format: 'uint32', values: polygons.ringOffsets},
                polygonOffsets: {format: 'uint32', values: polygons.polygonOffsets},
                featureOffsets: {format: 'uint32', values: polygons.featureOffsets}
              },
              operations: [new Operation(props)],
              outputs: ['offsets', 'neighbors', 'weights', 'overflow', 'required', 'candidate']
            })
          );
        }
        // Zero tolerance: all outputs are integers or binary weights and must match exactly.
        for (const name of ['offsets', 'neighbors', 'weights', 'overflow', 'required', 'candidate'])
          expect(Array.from(results[0][name])).toEqual(Array.from(results[1][name]));
      }
    }
  });
}
