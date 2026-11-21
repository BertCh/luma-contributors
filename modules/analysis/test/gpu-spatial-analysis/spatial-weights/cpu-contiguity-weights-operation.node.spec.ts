// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  CPUContiguityWeightsOperation,
  computeContiguityWeightsOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation';
import {GPUContiguityWeightsOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights-operation';
import {flattenFeatures, flattenPolygons} from './spatial-weights-oracle';
import {createPairStarvedContiguityFixture} from './contiguity-loss.test-utils';

for (const Operation of [GPUContiguityWeightsOperation, CPUContiguityWeightsOperation])
  it(`${Operation.name} validates scalar props before a missing candidate destination`, () => {
    const props = {
      id: 'negative-tolerance',
      criterion: 'queen' as const,
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
    };
    expect(() => new Operation(props as never)).toThrow(
      /^negative-tolerance snapTolerance must be >= 0$/
    );
    expect(() => new Operation({...props, snapTolerance: 0} as never)).toThrow(
      /^negative-tolerance output.status.candidateOverflow/
    );
  });

it('final1 explicit pairCapacity binds the CPU twin while final slots remain sufficient', () => {
  const fixture = createPairStarvedContiguityFixture();
  const partial = computeContiguityWeightsOnCPU(fixture.input, fixture.options);
  expect(Array.from(partial.status.candidateOverflow)).toEqual([1]);
  expect(Array.from(partial.status.overflow)).toEqual([0]);
  expect(Array.from(partial.status.requiredCount)).toEqual([1]);
  expect(Array.from(partial.offsets)).toEqual([0, 1, 1]);
  expect(partial.neighbors[0]).toBe(1);
  const complete = computeContiguityWeightsOnCPU(fixture.input, {
    ...fixture.options,
    pairCapacity: 4
  });
  expect(Array.from(complete.status.candidateOverflow)).toEqual([0]);
  expect(Array.from(complete.status.requiredCount)).toEqual([2]);
});

it('checks adjacency-changing snaps and signed half-cells at a nonzero origin', () => {
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
  // With tolerance 1/4: the first gap is 1/16 and closes. Relative to (10.0625,20.0625),
  // x=11.125 is cell 4 and x=11.1875 is the positive half-cell 4.5, rounded up to 5.
  // Relative to (10.1875,20.1875), x=10.0625 is cell -0.5, rounded up to 0.
  // A grid anchored at zero incorrectly joins the positive case and separates the negative case.
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
          const joins = snapTolerance > 0 && scene.joinsAfterSnap;
          const result = computeContiguityWeightsOnCPU(padded, {
            criterion,
            capacity: 2,
            snapTolerance
          });
          expect(Array.from(result.offsets), scene.name).toEqual(joins ? [0, 1, 2] : [0, 0, 0]);
          const usedCount = result.offsets[2];
          expect(Array.from(result.neighbors.slice(0, usedCount)), scene.name).toEqual(
            joins ? [1, 0] : []
          );
          expect(Array.from(result.weights.slice(0, usedCount)), scene.name).toEqual(
            joins ? [1, 1] : []
          );
          expect(result.status.requiredCount[0]).toBe(joins ? 2 : 0);
          expect(result.status.overflow[0]).toBe(0);
          expect(result.status.candidateOverflow[0]).toBe(0);
        }
      }
    }
  }
});

it('rejects an invalid GPU criterion before lowering', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format> =>
      createTransientView(
        graph,
        port,
        port === 'positions' ? 'float32x2' : port === 'weights' ? 'float32' : 'uint32',
        2
      ) as GraphDataView<Format>;
    const construct = () =>
      new GPUContiguityWeightsOperation({
        id: 'gpu-contiguity',
        criterion: 'invalid' as 'queen',
        featureCount: 1,
        capacity: 2,
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
          status: {overflow: 'overflow', candidateOverflow: 'candidate'}
        }
      });
    expect(() => construct().createContributor(getView)).toThrow(/gpu-contiguity criterion/);
  } finally {
    device.destroy();
  }
});

it('the GPU operation rejects a negative tolerance even when float32 would round it to zero', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format> =>
      createTransientView(
        graph,
        port,
        port === 'positions' ? 'float32x2' : port === 'weights' ? 'float32' : 'uint32',
        2
      ) as GraphDataView<Format>;
    const construct = () =>
      new GPUContiguityWeightsOperation({
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
          status: {overflow: 'overflow', candidateOverflow: 'candidate'}
        }
      });
    expect(() => construct().createContributor(getView)).toThrow(
      /^negative-tolerance snapTolerance must be >= 0$/
    );
  } finally {
    device.destroy();
  }
});

it('matches hand-drawn 2x2 square adjacency for queen and rook, including truncation', () => {
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
  for (const [criterion, expectedOffsets, expectedNeighbors] of [
    ['queen', [0, 3, 6, 9, 12], [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]],
    ['rook', [0, 2, 4, 6, 8], [1, 2, 0, 3, 0, 3, 1, 2]]
  ] as const) {
    for (const capacity of [0, 3, expectedNeighbors.length]) {
      const result = computeContiguityWeightsOnCPU(polygons, {criterion, capacity});
      expect(Array.from(result.offsets)).toEqual(
        expectedOffsets.map(offset => Math.min(offset, capacity))
      );
      expect(Array.from(result.neighbors)).toEqual(expectedNeighbors.slice(0, capacity));
      expect(Array.from(result.weights)).toEqual(new Array(capacity).fill(1));
      expect(result.status.requiredCount[0]).toBe(expectedNeighbors.length);
      expect(result.status.overflow[0]).toBe(Number(capacity < expectedNeighbors.length));
      expect(result.status.candidateOverflow[0]).toBe(0);
    }
  }
});

it('includes holes and multipart features, ignores bad vertices and deduplicates repeated edges', () => {
  const polygons = flattenFeatures([
    [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4]
        ],
        [
          [1, 1],
          [2, 1],
          [2, 2],
          [1, 2]
        ]
      ],
      [
        [
          [8, 0],
          [9, 0],
          [9, 1],
          [8, 1]
        ]
      ]
    ],
    [
      [
        [
          [1, 1],
          [1, 2],
          [2, 2],
          [2, 1],
          [1, 1]
        ]
      ]
    ],
    [
      [
        [
          [9, 0],
          [10, 0],
          [10, 1],
          [9, 1]
        ]
      ]
    ],
    [
      [
        [
          [NaN, 0],
          [Infinity, 1],
          [20, 20]
        ]
      ]
    ]
  ]);
  for (const criterion of ['queen', 'rook'] as const) {
    const result = computeContiguityWeightsOnCPU(polygons, {criterion, capacity: 4});
    expect(Array.from(result.offsets)).toEqual([0, 2, 3, 4, 4]);
    expect(Array.from(result.neighbors)).toEqual([1, 2, 0, 0]);
    expect(result.status.requiredCount[0]).toBe(4);
  }
});

it('uses the first vertex as snap origin, half-up for negative coordinates and implicit ring closure', () => {
  const polygons = flattenPolygons([
    [
      [
        [0, 0],
        [1, 0],
        [1, 1]
      ]
    ],
    [
      [
        [1.125, 0],
        [2, 0],
        [2, 1]
      ]
    ],
    [
      [
        [-0.125, 0],
        [-1, 0],
        [-1, 1]
      ]
    ]
  ]);
  expect(
    computeContiguityWeightsOnCPU(polygons, {criterion: 'queen', capacity: 4, snapTolerance: 0.25})
      .status.requiredCount[0]
  ).toBe(2);
  expect(
    computeContiguityWeightsOnCPU(polygons, {criterion: 'queen', capacity: 4}).status
      .requiredCount[0]
  ).toBe(0);
  const broken = {...polygons, ringOffsets: Uint32Array.of(0, 20)};
  expect(() => computeContiguityWeightsOnCPU(broken, {criterion: 'rook', capacity: 1})).toThrow(
    'ringOffsets'
  );
  expect(() =>
    computeContiguityWeightsOnCPU(
      {...polygons, positions: Float32Array.of(1)},
      {criterion: 'rook', capacity: 1}
    )
  ).toThrow('positions');
  expect(() =>
    computeContiguityWeightsOnCPU(
      {...polygons, polygonOffsets: Uint32Array.of(1, 0)},
      {criterion: 'rook', capacity: 1}
    )
  ).toThrow('polygonOffsets');
  expect(() =>
    computeContiguityWeightsOnCPU(
      {...polygons, featureOffsets: Uint32Array.of(0, 20)},
      {criterion: 'rook', capacity: 1}
    )
  ).toThrow('featureOffsets');
});

it('validates construction and keeps both graph output contracts identical', () => {
  const props = {
    id: 'contiguity-test',
    criterion: 'queen' as const,
    polygons: {
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonOffsets: 'polygonOffsets',
      featureOffsets: 'featureOffsets'
    },
    featureCount: 3,
    capacity: 4,
    output: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      status: {overflow: 'overflow', requiredCount: 'required', candidateOverflow: 'candidate'}
    }
  };
  expect(() => new CPUContiguityWeightsOperation({...props, featureCount: 0})).toThrow(
    /contiguity-test featureCount/
  );
  expect(() => new CPUContiguityWeightsOperation({...props, snapTolerance: -1})).toThrow(
    /contiguity-test snapTolerance/
  );
  expect(
    () => new CPUContiguityWeightsOperation({...props, snapTolerance: -Number.MIN_VALUE})
  ).toThrow(/^contiguity-test snapTolerance must be >= 0$/);
  expect(
    () => new CPUContiguityWeightsOperation({...props, criterion: 'other' as 'queen'})
  ).toThrow(/contiguity-test criterion/);
  expect(() => new GPUContiguityWeightsOperation({...props, capacity: 0})).toThrow(
    /^contiguity-test capacity/
  );
  expect(new CPUContiguityWeightsOperation(props).contract.outputs).toEqual(
    new GPUContiguityWeightsOperation(props).contract.outputs
  );
  expect(
    () =>
      new CPUContiguityWeightsOperation({
        ...props,
        pairCapacity: 1,
        snapTolerance: 1,
        coordinateExtent: 2 ** 25
      })
  ).not.toThrow();
});

it('CPU contiguity uses the zero fallback for a non-finite owned origin and supports empty rings', () => {
  // Anchored at zero, the vertical sides x=11.125 and x=11.1875 both snap to cell 45.
  // The bad first vertex is ignored; the two vertical sides still establish rook adjacency.
  const input = {
    positions: Float32Array.of(
      3e20,
      3e20,
      Number.NaN,
      20,
      11.125,
      20,
      11.125,
      21,
      10,
      21,
      11.1875,
      20,
      12,
      20,
      12,
      21,
      11.1875,
      21,
      3e20,
      3e20
    ),
    ringOffsets: Uint32Array.of(1, 1, 5, 9),
    polygonOffsets: Uint32Array.of(0, 0, 2, 3),
    featureOffsets: Uint32Array.of(0, 1, 2, 3)
  };
  for (const criterion of ['queen', 'rook'] as const) {
    const result = computeContiguityWeightsOnCPU(input, {
      criterion,
      capacity: 2,
      snapTolerance: 0.25
    });
    expect(Array.from(result.offsets)).toEqual([0, 0, 1, 2]);
    expect(Array.from(result.neighbors)).toEqual([2, 1]);
    expect(result.status.requiredCount[0]).toBe(2);
    const empty = computeContiguityWeightsOnCPU(
      {...input, ringOffsets: Uint32Array.of(1, 1, 1, 1)},
      {criterion, capacity: 2, snapTolerance: 0.25}
    );
    expect(Array.from(empty.offsets)).toEqual([0, 0, 0, 0]);
    expect(empty.status.requiredCount[0]).toBe(0);
  }
});

it('DGN-09 accepts terminal-only empty hierarchy levels and preserves one empty parent', () => {
  for (const [ringOffsets, polygonOffsets, featureOffsets, expected] of [
    [[0], [0], [0], [0]],
    [[0], [0], [0, 0], [0, 0]],
    [[0], [0, 0], [0, 1], [0, 0]],
    [
      [0, 0],
      [0, 1],
      [0, 1],
      [0, 0]
    ]
  ]) {
    const result = computeContiguityWeightsOnCPU(
      {
        positions: new Float32Array(),
        ringOffsets: Uint32Array.from(ringOffsets),
        polygonOffsets: Uint32Array.from(polygonOffsets),
        featureOffsets: Uint32Array.from(featureOffsets)
      },
      {criterion: 'rook', capacity: 0}
    );
    expect([...result.offsets]).toEqual(expected);
    expect([...result.status.requiredCount]).toEqual([0]);
    expect([...result.status.overflow]).toEqual([0]);
  }
  expect(() =>
    computeContiguityWeightsOnCPU(
      {
        positions: new Float32Array(),
        ringOffsets: new Uint32Array(),
        polygonOffsets: Uint32Array.of(0),
        featureOffsets: Uint32Array.of(0)
      },
      {criterion: 'rook', capacity: 0}
    )
  ).toThrow(/terminal offset/);
});
