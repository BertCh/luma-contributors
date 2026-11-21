// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {GPUCommandGraph, GraphVectorView, createTransientView} from '@luma.gl/graph/gpu-core';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {expect, it, vi} from 'vitest';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  GPULocalRBF,
  decodeGPULocalRBFStatus
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {
  GPULocalKriging,
  decodeGPULocalKrigingStatus
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {snapshotInterpolationConstructionProps} from '../../../src/gpu-spatial-analysis/spatial-interpolation/interpolation-construction-snapshot';
import {getGPUInverseDistanceWeightingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/spatial-interpolation-parameters';
import {getGPUKrigingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/kriging-parameters';
import {
  getGPULocalKrigingParameterValues,
  getGPUSpaceTimeKrigingParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {getGPUGaussianRandomFieldParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gaussian-random-field-parameters';

import {GPUInverseDistanceWeighting} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting';
import {
  GPUFocalStatistics,
  type GPUFocalStatisticsOutput
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics';
import type {GPUGaussianRandomFieldProps} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-gaussian-random-field';

import {GPUNaturalNeighbor} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor';
import {expectPredictionReadback} from './prediction-readback-expectations';

it('S3 rejects extra or missing scalar and packed prediction readback elements', () => {
  expectPredictionReadback([7], 7);
  expectPredictionReadback([7, 0, 0, 0], 7, 4);
  expectPredictionReadback([NaN, 0, 0, 0], NaN, 4);
  for (const values of [[], [7, 99]]) expect(() => expectPredictionReadback(values, 7)).toThrow();
  for (const values of [[7], [7, 0, 0], [7, 0, 0, 0, 99]])
    expect(() => expectPredictionReadback(values, 7, 4)).toThrow();
  expect(() => expectPredictionReadback([NaN], NaN, 4)).toThrow();
});

const covariance = {model: 'spherical', range: 1, sill: 1} as const;
const grid = (origin = 0, step = 1, width = 2) =>
  makeGPURasterMetadata({width, height: 2, cellSize: [step, 1], origin: [origin, 2]});
const rasterSettings = {searchRadius: 10, variogram: {...covariance, nugget: 0}};

for (const placement of ['own', 'inherited'] as const) {
  it(`W5-09 ignores ${placement} unrelated getters and captures predictor context metadata once`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const reads = {coordinateSpace: 0, metric: 0, frame: 0, unrelated: 0};
    const frame = {
      id: 'prediction-frame',
      kind: 'crs' as const,
      origin: [10, 20] as [number, number],
      unit: {name: 'meters', metersPerUnit: 1},
      areaOfUse: [-10, -20, 10, 20] as [number, number, number, number]
    };
    const spatialContext = {
      get coordinateSpace() {
        reads.coordinateSpace++;
        return 'planar' as const;
      },
      get metric() {
        reads.metric++;
        return 'native' as const;
      },
      get frame() {
        reads.frame++;
        return frame;
      }
    };
    const unrelated = {
      get unrelated() {
        reads.unrelated++;
        throw new Error('unrelated getter');
      }
    };
    if (placement === 'inherited') Object.setPrototypeOf(spatialContext, unrelated);
    else
      Object.defineProperty(
        spatialContext,
        'unrelated',
        Object.getOwnPropertyDescriptor(unrelated, 'unrelated')!
      );
    const props = {
      spatialContext,
      degree: 0 as const,
      kernel: 'gaussian' as const,
      neighborCount: 1,
      positions: createTransientView(graph, 'context-positions', 'float32x2', 1),
      values: createTransientView(graph, 'context-values', 'float32', 1),
      queries: createTransientView(graph, 'context-queries', 'float32x2', 1),
      output: {
        values: createTransientView(graph, 'context-output', 'float32x4', 1),
        status: createTransientView(graph, 'context-status', 'uint32', 1)
      }
    };
    try {
      const predictor = new GPULocalRBF(props);
      expect(reads).toEqual({coordinateSpace: 1, metric: 1, frame: 1, unrelated: 0});
      const captured = predictor.props.spatialContext!;
      // Chai may inspect own getters while formatting object identity assertions.
      expect(Object.is(captured, spatialContext)).toBe(false);
      expect(captured.frame).not.toBe(frame);
      expect(captured.frame?.origin).not.toBe(frame.origin);
      expect(captured.frame?.unit).not.toBe(frame.unit);
      expect(captured.frame?.areaOfUse).not.toBe(frame.areaOfUse);
      frame.origin[0] = 99;
      frame.unit.metersPerUnit = 99;
      frame.areaOfUse[0] = 99;
      expect(captured.frame?.origin).toEqual([10, 20]);
      expect(captured.frame?.unit?.metersPerUnit).toBe(1);
      expect(captured.frame?.areaOfUse).toEqual([-10, -20, 10, 20]);
      expect(predictor.props.positions).toBe(props.positions);
      expect(predictor.props.values).toBe(props.values);
      expect(predictor.props.queries).toBe(props.queries);
      expect(predictor.props.output.values).toBe(props.output.values);
      expect(predictor.props.output.status).toBe(props.output.status);
      expect(Object.isFrozen(props.positions.buffer)).toBe(false);
      expect(predictor.getCommandNodes(graph).length).toBeGreaterThan(0);
    } finally {
      device.destroy();
    }
  });
}

it('CPL-08 preserves producer loss on a shared CSR in both predictors', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const neighborhoodWeights = {
    offsets: createTransientView(graph, 'offsets', 'uint32', 2),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
    status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
  };
  const shared = {
    id: 'loss-test',
    positions: createTransientView(graph, 'positions', 'float32x2', 2),
    neighborhoodWeights,
    output: {
      values: createTransientView(graph, 'output', 'float32x4', 1),
      status: createTransientView(graph, 'status', 'uint32', 1)
    }
  };
  for (const contributor of [
    new GPULocalRBF({
      ...shared,
      queries: createTransientView(graph, 'queries', 'float32x2', 1),
      values: createTransientView(graph, 'data', 'float32', 2),
      neighborCount: 2,
      degree: 0,
      kernel: 'gaussian'
    }),
    new GPULocalKriging({
      ...shared,
      supports: createTransientView(graph, 'supports', 'float32x2', 1),
      observations: createTransientView(graph, 'observations', 'float32x4', 2),
      mode: 'ordinary',
      covariance,
      neighborCount: 2
    })
  ]) {
    const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    expect(
      sources.some(({source}) => source.includes('loss[lossOffset]') && source.includes('= 3u'))
    ).toBe(true);
  }
  device.destroy();
});

for (const [name, length, pack] of [
  [
    'raster',
    12,
    (target: Float32Array) => getGPUKrigingParameterValues(rasterSettings, grid(), target)
  ],
  ['local', 8, (target: Float32Array) => getGPULocalKrigingParameterValues({covariance}, target)],
  [
    'local-co',
    12,
    (target: Float32Array) =>
      getGPULocalKrigingParameterValues({covariance, coregionalization: [1, 0, 0, 1]}, target)
  ],
  [
    'space-time',
    28,
    (target: Float32Array) =>
      getGPUSpaceTimeKrigingParameterValues(
        {covariance: {type: 'metric', joint: {...covariance, nugget: 0}, anisotropy: 1}},
        target
      )
  ],
  [
    'field',
    16,
    (target: Float32Array) =>
      getGPUGaussianRandomFieldParameterValues({variogram: covariance, seed: 0xffffffff}, target)
  ]
] as const) {
  it(`UPT-02 ${name} publishes only its exact prefix in an interior arena view`, () => {
    const arena = new Float32Array(length + 6).fill(77);
    arena[0] = 11;
    arena[arena.length - 1] = 99;
    const target = arena.subarray(2, 2 + length + 2);
    expect(pack(target)).toBe(target);
    // Literal record words independently pin publication as well as arena boundaries.
    const expectedPrefix =
      name === 'raster'
        ? [0, 2, 2, 0, 10, 16, 0, 0, 1, 1, 3, 0]
        : name === 'space-time'
          ? [1, 2, 1, 0]
          : [1, 0, 0, 0, 0, 0, 1, 1];
    expect(Array.from(target.subarray(0, expectedPrefix.length))).toEqual(expectedPrefix);
    if (name === 'local-co') expect(Array.from(target.subarray(8, 12))).toEqual([1, 0, 0, 1]);
    if (name === 'field')
      expect(new Uint32Array(target.buffer, target.byteOffset, 16)[9]).toBe(0xffffffff);
    expect(Array.from(arena.subarray(0, 2))).toEqual([11, 77]);
    expect(Array.from(arena.subarray(2 + length))).toEqual([77, 77, 77, 99]);
  });
}

for (const [name, pack] of [
  [
    'shear',
    (target: Float32Array) =>
      getGPUKrigingParameterValues(rasterSettings, {...grid(), affine: [1, 1, 0, 0, -1, 2]}, target)
  ],
  [
    'range',
    (target: Float32Array) =>
      getGPULocalKrigingParameterValues({covariance: {...covariance, range: -1}}, target)
  ],
  [
    'anisotropy',
    (target: Float32Array) =>
      getGPUSpaceTimeKrigingParameterValues(
        {covariance: {type: 'metric', joint: {...covariance, nugget: 0}, anisotropy: -1}},
        target
      )
  ],
  [
    'origin',
    (target: Float32Array) =>
      getGPUGaussianRandomFieldParameterValues(
        {variogram: covariance, seed: 1, origin: [1e100, 0]},
        target
      )
  ],
  [
    'coefficient',
    (target: Float32Array) =>
      getGPULocalKrigingParameterValues({covariance, coregionalization: [1e20, 0, 0, 1]}, target)
  ],
  [
    'scale',
    (target: Float32Array) =>
      getGPULocalKrigingParameterValues(
        {covariance: {...covariance, sill: 1e30}, coregionalization: [1e10, 0, 0, 1]},
        target
      )
  ]
] as const) {
  it(`UPT-02/W3-01 rejected ${name} leaves every destination byte intact`, () => {
    const arena = new Float32Array(36).fill(77);
    arena[0] = 11;
    arena[35] = 99;
    const before = new Uint8Array(arena.buffer).slice();
    expect(() => pack(arena.subarray(2, 34))).toThrow(
      name === 'coefficient'
        ? /coefficient.*\[0\]/
        : name === 'scale'
          ? /normalizationScale/
          : undefined
    );
    expect(new Uint8Array(arena.buffer)).toEqual(before);
  });
}

it('UPT-03 rejects collapsed encoded axes and preserves translated local grids', () => {
  for (const projected of [grid(5000000, 0.125, 1), grid(10000000, 0.1)]) {
    expect(() =>
      getGPUInverseDistanceWeightingParameterValues({searchRadius: 1}, projected)
    ).toThrow(/x.*original.*encoded.*common.*origin/);
    expect(() => getGPUKrigingParameterValues(rasterSettings, projected)).toThrow(
      /x.*original.*encoded/
    );
  }
  expect(
    Array.from(
      getGPUInverseDistanceWeightingParameterValues({searchRadius: 1}, grid(0, 0.1)).subarray(0, 4)
    )
  ).toEqual([0, 2, Math.fround(0.2), 0]);
  expect(() =>
    getGPUInverseDistanceWeightingParameterValues(
      {searchRadius: 1e50, taper: 'modified-shepard'},
      grid()
    )
  ).toThrow(/searchRadius/);
  expect(() =>
    getGPUKrigingParameterValues(
      {...rasterSettings, variogram: {...rasterSettings.variogram, range: 1e-50}},
      grid()
    )
  ).toThrow(/range/);
});

it('VAL-01 names the wrong values format and custom instance before interpreting integer bytes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    id: 'custom-rbf',
    positions: createTransientView(graph, 'positions', 'float32x2', 3),
    queries: createTransientView(graph, 'queries', 'float32x2', 1),
    values: createTransientView(graph, 'data', 'sint32', 3),
    output: {
      values: createTransientView(graph, 'output', 'float32x4', 1),
      status: createTransientView(graph, 'status', 'uint32', 1)
    }
  };
  // @ts-expect-error Deliberately substitute integer storage for the float values port.
  expect(() => new GPULocalRBF(props)).toThrow(/custom-rbf.*values.*float32.*sint32/);
  device.destroy();
});

it('ENT-02 captures cross-format vector status identity without freezing GPU resources', () => {
  const require = createRequire(import.meta.url);
  const common = require('@luma.gl/graph/gpu-core') as typeof import('@luma.gl/graph/gpu-core');
  const device = createNullWebGPUDevice();
  const status = new common.GraphVectorView({
    id: 'foreign',
    name: 'foreign',
    format: 'uint32',
    data: [],
    length: 0,
    valueLength: 0,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4
  });
  const props = snapshotInterpolationConstructionProps(
    {output: {status}},
    ['output'],
    ['status'],
    ['complete']
  );
  expect(props.output.status).toBe(status);
  expect(Object.isFrozen(status)).toBe(false);
  device.destroy();
});

it('LIF-11 releases partial retained allocation and permits retry on a fresh graph', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    positions: createTransientView(graph, 'positions', 'float32x2', 1),
    observations: createTransientView(graph, 'observations', 'float32x4', 1),
    queries: createTransientView(graph, 'queries', 'float32x2', 1),
    covariance,
    seed: 1,
    replicateCount: 1,
    reusePreparation: true,
    output: {
      values: createTransientView(graph, 'values', 'float32', 1),
      moments: createTransientView(graph, 'moments', 'float32x2', 1),
      status: createTransientView(graph, 'status', 'uint32', 1)
    }
  };
  const contributor = new GPUConditionalGaussianSimulation(props);
  const original = device.createBuffer.bind(device);
  const acquired: ReturnType<typeof original>[] = [];
  const allocate = vi.spyOn(device, 'createBuffer').mockImplementation(options => {
    if (acquired.length === 1) throw new Error('allocation two');
    const buffer = original(options);
    acquired.push(buffer);
    return buffer;
  });
  expect(() => graph.add(contributor)).toThrow('allocation two');
  allocate.mockRestore();
  expect(acquired.every(buffer => buffer.destroyed)).toBe(true);
  expect(contributor.preparationStatistics).toBeUndefined();
  // Views are bound to their owner; rebuilding imports is required for a fresh graph.
  expect(() => contributor.getCommandNodes(graph)).toThrow(/already in use/);
  const rebuiltGraph = new GPUCommandGraph(device);
  const rebuilt = new GPUConditionalGaussianSimulation({
    ...props,
    positions: createTransientView(rebuiltGraph, 'positions', 'float32x2', 1),
    observations: createTransientView(rebuiltGraph, 'observations', 'float32x4', 1),
    queries: createTransientView(rebuiltGraph, 'queries', 'float32x2', 1),
    output: {
      values: createTransientView(rebuiltGraph, 'values', 'float32', 1),
      moments: createTransientView(rebuiltGraph, 'moments', 'float32x2', 1),
      status: createTransientView(rebuiltGraph, 'status', 'uint32', 1)
    }
  });
  expect(() => rebuiltGraph.add(rebuilt)).not.toThrow();
  rebuilt.destroy();
  contributor.destroy();
  device.destroy();
});

it('UPT-06 packs against the actual IDW nearest-neighbor topology and capacity', () => {
  const device = createNullWebGPUDevice();
  for (const capacity of [0, 2]) {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUInverseDistanceWeighting({
      id: 'custom-idw',
      grid: grid(),
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      values: createTransientView(graph, 'data', 'float32', 2),
      parameters: createTransientView(graph, 'params', 'float32', 9),
      indexGridSize: [1, 1],
      indexBounds: [0, 0, 4, 4],
      maximumNeighborCount: capacity,
      output: {values: createTransientView(graph, 'values', 'float32', 4)}
    });
    expect(contributor.getParameterValues({searchRadius: 3, neighborCount: capacity})[6]).toBe(
      capacity
    );
    expect(() =>
      contributor.getParameterValues({searchRadius: 3, neighborCount: capacity + 1})
    ).toThrow(/custom-idw.*neighborCount/);
    expect(() => contributor.getParameterValues({searchRadius: 3}, grid(0, 1, 1))).toThrow(
      /custom-idw.*grid/
    );
  }
  device.destroy();
});

it('DEF-09 preserves every minimal focal output and rejects an empty JavaScript selection', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const float = createTransientView(graph, 'float', 'float32', 4);
  const counts = createTransientView(graph, 'counts', 'uint32', 4);
  const selections: GPUFocalStatisticsOutput[] = [
    {mean: float},
    {sum: float},
    {min: float},
    {max: float},
    {range: float},
    {standardDeviation: float},
    {counts}
  ];

  const props = {
    grid: grid(),
    values: createTransientView(graph, 'data', 'float32', 4),
    parameters: createTransientView(graph, 'params', 'float32', 4),
    maximumRadius: 1
  };
  for (const output of selections)
    expect(() => new GPUFocalStatistics({...props, output})).not.toThrow();
  // @ts-expect-error Empty output deliberately exercises the JavaScript runtime guard.
  expect(() => new GPUFocalStatistics({...props, output: {}})).toThrow(/at least one output/);
  device.destroy();
});

// These functions are never called: the lead's typecheck verifies required input alternatives.
const invalidDomainFixtures = (
  parameters: GPUGaussianRandomFieldProps['parameters'],
  output: GPUGaussianRandomFieldProps['output'],
  positions: NonNullable<GPUGaussianRandomFieldProps['positions']>
) => {
  // @ts-expect-error A domain is required.
  const missing: GPUGaussianRandomFieldProps = {parameters, output};
  // @ts-expect-error A grid and positions are mutually exclusive.
  const both: GPUGaussianRandomFieldProps = {parameters, output, grid: grid(), positions};
  const point: GPUGaussianRandomFieldProps = {parameters, output, positions};
  const raster: GPUGaussianRandomFieldProps = {parameters, output, grid: grid()};
  return [missing, both, point, raster];
};
void invalidDomainFixtures;

it('ENT-02 selects ordered chunks from an actual CommonJS vector in an ESM predictor', () => {
  const common = createRequire(import.meta.url)(
    '@luma.gl/graph/gpu-core'
  ) as typeof import('@luma.gl/graph/gpu-core');
  expect(common.GraphVectorView).not.toBe(GraphVectorView);
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const queries = new common.GraphVectorView({
    id: 'queries',
    name: 'queries',
    format: 'float32x2',
    data: [
      createTransientView(graph, 'query-0', 'float32x2', 1),
      createTransientView(graph, 'query-1', 'float32x2', 1)
    ],
    length: 2,
    valueLength: 4,
    stride: 2,
    byteStride: 8,
    rowByteLength: 8
  });
  const contributor = new GPULocalRBF({
    queries,
    positions: createTransientView(graph, 'positions', 'float32x2', 3),
    values: createTransientView(graph, 'data', 'float32', 3),
    output: {
      values: createVectorView('output', 'float32x4', [
        createTransientView(graph, 'output-0', 'float32x4', 1),
        createTransientView(graph, 'output-1', 'float32x4', 1)
      ]),
      status: createVectorView('status', 'uint32', [
        createTransientView(graph, 'status-0', 'uint32', 1),
        createTransientView(graph, 'status-1', 'uint32', 1)
      ])
    }
  });
  expect(contributor.props.queries).toBe(queries);
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.length).toBe(2);
  expect(queries.data.map(view => view.buffer.id)).toEqual(['query-0', 'query-1']);
  expect(Object.isFrozen(queries.data[0].buffer)).toBe(false);
  device.destroy();
});

it('CPL-08 status decoders accept offset readbacks and reject missing or partial words', () => {
  for (const decode of [decodeGPULocalKrigingStatus, decodeGPULocalRBFStatus]) {
    const arena = Uint32Array.of(77, 3, 0, 77);
    expect(decode(new Uint8Array(arena.buffer, 4, 4))).toEqual({
      code: 3,
      complete: 0,
      invalidInput: 0,
      singular: 0,
      candidateOverflow: 1
    });
    expect(decode(arena.subarray(1, 3), 1).complete).toBe(1);
    expect(decode(Uint32Array.of(2).buffer).singular).toBe(1);
    expect(() => decode(new Uint8Array(3))).toThrow(/whole uint32/);
    expect(() => decode(new Uint32Array())).toThrow(/contain row/);
    expect(() => decode(Uint32Array.of(9))).toThrow(/status code/);
  }
});

it('VAL-01 validates named fixed ports in simulation, local kriging and natural neighbors', () => {
  const device = createNullWebGPUDevice();
  for (const kind of ['simulation', 'kriging', 'natural'] as const) {
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 3);
    const queries = createTransientView(graph, 'queries', 'float32x2', 1);
    const observations = createTransientView(graph, 'data', 'float32x4', 3);
    const bad = createTransientView(graph, 'bad', 'sint32', 3);
    if (kind === 'simulation') {
      const props = {
        id: 'custom-simulation',
        positions,
        queries,
        observations,
        covariance,
        seed: 1,
        replicateCount: 1,
        output: {
          values: createTransientView(graph, 'values', 'float32', 1),
          moments: createTransientView(graph, 'moments', 'float32x2', 1),
          status: createTransientView(graph, 'status', 'uint32', 1)
        }
      };
      expect(() => new GPUConditionalGaussianSimulation(props)).not.toThrow();
      // @ts-expect-error Deliberately substitute a wrong fixed-format observation port.
      expect(() => new GPUConditionalGaussianSimulation({...props, observations: bad})).toThrow(
        /custom-simulation.*observations.*float32x4.*sint32/
      );
    } else if (kind === 'kriging') {
      const props = {
        id: 'custom-kriging',
        positions,
        supports: queries,
        observations,
        covariance,
        mode: 'ordinary' as const,
        output: {
          values: createTransientView(graph, 'values', 'float32x4', 1),
          status: createTransientView(graph, 'status', 'uint32', 1)
        }
      };
      expect(() => new GPULocalKriging(props)).not.toThrow();
      // @ts-expect-error Deliberately substitute a wrong fixed-format observation port.
      expect(() => new GPULocalKriging({...props, observations: bad})).toThrow(
        /custom-kriging.*observations.*float32x4.*sint32/
      );
      const weightedSupports = createTransientView(graph, 'weighted-supports', 'float32x4', 1);
      expect(() => new GPULocalKriging({...props, supports: weightedSupports})).not.toThrow();
    } else {
      const props = {
        id: 'custom-natural',
        positions,
        queries,
        values: createTransientView(graph, 'data-values', 'float32', 3),
        output: {
          values: createTransientView(graph, 'values', 'float32x4', 1),
          status: createTransientView(graph, 'status', 'uint32', 1)
        }
      };
      expect(() => new GPUNaturalNeighbor(props)).not.toThrow();
      // @ts-expect-error Deliberately substitute integer storage for the float values port.
      expect(() => new GPUNaturalNeighbor({...props, values: bad})).toThrow(
        /custom-natural.*values.*float32.*sint32/
      );
    }
  }
  device.destroy();
});

it('UPT-03 preserves Gaussian-field origin/step layouts at projected origins', () => {
  const values = getGPUGaussianRandomFieldParameterValues({
    variogram: covariance,
    seed: 1,
    grid: grid(5000000, 0.125, 1)
  });
  // This route stores the step independently; it never subtracts rounded outer x corners.
  expect(values[13]).toBe(0.125);
  expect(values[14]).toBe(-1);
});
