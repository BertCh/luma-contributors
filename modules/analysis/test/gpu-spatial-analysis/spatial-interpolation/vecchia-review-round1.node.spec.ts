// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {runInNewContext} from 'node:vm';
import {readFileSync} from 'node:fs';
import {
  expectVecchiaMarginalTotals,
  expectVecchiaSuppressedTotals,
  VECCHIA_MARGINAL_EXPECTED
} from './vecchia-marginal-expectations';
import {expect, expectTypeOf, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  buildVecchiaConditioningSets,
  buildVecchiaPredictionNeighborhoods,
  type VecchiaConditioningSets
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-conditioning-sets';
import {
  GPUVecchiaLogLikelihood,
  getVecchiaLogLikelihoodResult,
  decodeGPUVecchiaLogLikelihoodResult
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood';
import {
  GPUVecchiaFisherScoring,
  decodeGPUVecchiaFisherScoringResult
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring';
import {
  GPUVecchiaSpaceTimeLogLikelihood,
  getGPUVecchiaSpaceTimeLogLikelihoodParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood';
import {GPUVecchiaLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood-operation';
import {GPUVecchiaSpaceTimeLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood-operation';
import {GPUVecchiaFisherScoringOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring-operation';
import {
  CPUVecchiaLogLikelihoodOperation,
  computeVecchiaLogLikelihoodOnCPU,
  type CPUVecchiaLogLikelihoodResult
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-log-likelihood-operation';
import {
  CPUVecchiaSpaceTimeLogLikelihoodOperation,
  type CPUVecchiaSpaceTimeLogLikelihoodResult
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-space-time-log-likelihood-operation';
import {
  CPUVecchiaFisherScoringOperation,
  computeVecchiaFisherScoringOnCPU,
  type CPUVecchiaFisherScoringResult
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-fisher-scoring-operation';
import type {SpaceTimeVariogramModel} from '../../../src/dataframe-analysis/pair-statistics/space-time-variogram-model';
import {getGPUVecchiaLogLikelihoodParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-parameters';

it('B1 keeps prediction widths used by conformal calibration without dropping allocation checks', () => {
  for (const neighborCount of [33, 64, 256]) {
    const result = buildVecchiaPredictionNeighborhoods({
      positions: [2, 3],
      queries: [2, 3],
      neighborCount
    });
    expect(Array.from(result.offsets)).toEqual([0, 1]);
    expect(Array.from(result.neighbors)).toEqual([0]);
    expect(result.counts?.neighborCount).toBe(neighborCount);
  }
  expect(() =>
    buildVecchiaPredictionNeighborhoods({positions: [], queries: [0, 0], neighborCount: 2 ** 31})
  ).toThrow(/neighborCount/);
});

it('B2 keeps nonzero setup masks distinct from positive observation validity', () => {
  const props = {positions: [0, 0, 1, 0, 2, 0], validity: Int8Array.of(-1, 1, 1), neighborCount: 1};
  const conditioning = buildVecchiaConditioningSets(props);
  expect(Array.from(conditioning.offsets)).toEqual([0, 0, 1, 2]);
  expect(Array.from(conditioning.neighbors)).toEqual([0, 1]);
  expect(
    Array.from(buildVecchiaPredictionNeighborhoods({...props, queries: [0, 0]}).neighbors)
  ).toEqual([0]);
  const evaluated = computeVecchiaLogLikelihoodOnCPU({
    positions: Float32Array.of(0, 0),
    observations: Float32Array.of(1, 0, 0, -1),
    neighborhoodOffsets: Uint32Array.of(0, 0),
    neighborhoods: new Uint32Array(),
    parameters: getGPUVecchiaLogLikelihoodParameterValues({model: 'exponential', sill: 1, range: 3})
  });
  expect(Array.from(evaluated.status)).toEqual([0, 1, 0]);
});

it('B3 decodes foreign-realm byte views and rejects foreign partial elements', () => {
  const totals = runInNewContext(
    'new Uint8Array(new Float32Array([77, -8, .125, 77]).buffer, 4, 8)'
  ) as Uint8Array;
  const status = runInNewContext(
    'new Uint8Array(new Uint32Array([77, 0, 0, 1, 77]).buffer, 4, 12)'
  ) as Uint8Array;
  expect(getVecchiaLogLikelihoodResult(totals, status)).toEqual({
    logLikelihood: -7.875,
    failedSiteCount: 0,
    excludedSiteCount: 0,
    usedSiteCount: 1
  });
  expect(decodeGPUVecchiaLogLikelihoodResult({logLikelihood: totals, status}).logLikelihood).toBe(
    -7.875
  );
  const foreignBuffer = runInNewContext('new Float32Array([-8, .125]).buffer') as ArrayBuffer;
  expect(getVecchiaLogLikelihoodResult(foreignBuffer, status).logLikelihood).toBe(-7.875);
  const partial = runInNewContext('new Uint8Array(9)') as Uint8Array;
  expect(() => getVecchiaLogLikelihoodResult(partial, status)).toThrow(/logLikelihood/);
});

it('B3 rejects wrong typed elements only at the named decoder boundary', () => {
  const floats = Float32Array.of(-8, 0.125);
  const words = new Uint32Array(floats.buffer);
  const status = Uint32Array.of(0, 0, 1);
  expect(() =>
    decodeGPUVecchiaLogLikelihoodResult({logLikelihood: words as unknown as Float32Array, status})
  ).toThrow(/logLikelihood.*Float32Array/);
  expect(() =>
    decodeGPUVecchiaLogLikelihoodResult({
      logLikelihood: floats,
      status: Float32Array.of(0, 0, 1) as unknown as Uint32Array
    })
  ).toThrow(/status.*Uint32Array/);
  expect(() =>
    decodeGPUVecchiaFisherScoringResult({
      totals: new Uint32Array(20) as unknown as Float32Array,
      status
    })
  ).toThrow(/totals.*Float32Array/);
  expect(() =>
    decodeGPUVecchiaLogLikelihoodResult({
      logLikelihood: floats,
      status,
      neighborhoodStatus: Float32Array.of(1, 0, 0) as unknown as Uint32Array
    })
  ).toThrow(/neighborhoodStatus.*Uint32Array/);
  expect(getVecchiaLogLikelihoodResult([-8, 0.125], [0, 0, 1]).logLikelihood).toBe(-7.875);
  expect(getVecchiaLogLikelihoodResult(Float64Array.of(-8, 0.125), [0, 0, 1]).logLikelihood).toBe(
    -7.875
  );
  expect(getVecchiaLogLikelihoodResult(words, status).logLikelihood).toBe(words[0] + words[1]);
});

const operationProps = {
  id: 'round-one-fit',
  rowCount: 1,
  neighborCount: 1,
  positions: 'positions',
  observations: 'observations',
  neighborhoodOffsets: 'offsets',
  neighborhoods: 'neighbors',
  parameters: 'parameters',
  times: 'times',
  temporalContext: {
    unit: 'second' as const,
    epoch: 0n,
    encoding: 'float32-relative' as const,
    axis: {kind: 'irregular' as const}
  },
  output: {
    conditionals: 'conditionals',
    conditionalStatus: 'conditional-status',
    siteTerms: 'terms',
    siteStatus: 'sites',
    logLikelihood: 'likelihood',
    totals: 'totals',
    status: 'status'
  }
};

for (const [name, Operation] of [
  ['gpu spatial', GPUVecchiaLogLikelihoodOperation],
  ['cpu spatial', CPUVecchiaLogLikelihoodOperation],
  ['gpu temporal', GPUVecchiaSpaceTimeLogLikelihoodOperation],
  ['cpu temporal', CPUVecchiaSpaceTimeLogLikelihoodOperation],
  ['gpu fisher', GPUVecchiaFisherScoringOperation],
  ['cpu fisher', CPUVecchiaFisherScoringOperation]
] as const) {
  it(`B4 rejects empty graph names in ${name}`, () => {
    expect(
      () => new Operation({...operationProps, neighborhoodStatus: {candidateOverflow: ''}})
    ).toThrow(/neighborhoodCandidateOverflow/);
    expect(() => new Operation({...operationProps, neighborhoodStatus: {overflow: ''}})).toThrow(
      /neighborhoodOverflow/
    );
    expect(
      () =>
        new Operation({
          ...operationProps,
          output: {...operationProps.output, neighborhoodStatus: ''}
        })
    ).toThrow(/neighborhoodStatus/);
  });
}

it('B5 retains assignability of legacy CSR and CPU result shapes', () => {
  // Static compatibility fixture for the lead's typecheck (runtime Vitest does not typecheck).
  const sets: VecchiaConditioningSets = {
    offsets: Uint32Array.of(0, 0),
    neighbors: new Uint32Array()
  };
  const legacy = {
    conditionals: new Float32Array(4),
    conditionalStatus: new Uint32Array(1),
    siteTerms: new Float32Array(1),
    siteStatus: new Uint32Array(1),
    logLikelihood: new Float32Array(2),
    status: new Uint32Array(3)
  };
  const spatial: CPUVecchiaLogLikelihoodResult = legacy;
  const temporal: CPUVecchiaSpaceTimeLogLikelihoodResult = legacy;
  const fisher: CPUVecchiaFisherScoringResult = {
    siteTerms: new Float32Array(10),
    siteStatus: new Uint32Array(1),
    totals: new Float32Array(20),
    status: new Uint32Array(3)
  };
  expect(spatial).toBe(legacy);
  expect(temporal).toBe(legacy);
  expect(fisher.totals).toHaveLength(20);
  expect(sets.offsets).toHaveLength(2);
  const input = {
    positions: Float32Array.of(0, 0),
    observations: Float32Array.of(1, 0, 0, 1),
    neighborhoodOffsets: sets.offsets,
    neighborhoods: sets.neighbors,
    parameters: getGPUVecchiaLogLikelihoodParameterValues({model: 'exponential', sill: 1, range: 3})
  };
  expectTypeOf(
    computeVecchiaLogLikelihoodOnCPU(input).neighborhoodStatus
  ).toEqualTypeOf<Uint32Array>();
  expectTypeOf(
    computeVecchiaFisherScoringOnCPU(input).neighborhoodStatus
  ).toEqualTypeOf<Uint32Array>();
  expectTypeOf(
    buildVecchiaConditioningSets({positions: [], neighborCount: 1}).counts
  ).toEqualTypeOf<{rowCount: number; neighborCount: number; entryCount: number}>();
});

it('B6 accepts full mask/time columns covering an explicitly selected prefix', () => {
  const props = {
    positions: [0, 0, 1, 0, 2, 0],
    rowCount: 2,
    neighborCount: 1,
    validity: [1, 1, 0],
    times: [0, 0, NaN],
    timeScale: 1,
    order: [0, 1]
  };
  const result = buildVecchiaConditioningSets(props);
  expect(Array.from(result.offsets)).toEqual([0, 0, 1]);
  expect(Array.from(result.neighbors)).toEqual([0]);
  expect(() => buildVecchiaConditioningSets({...props, validity: [1]})).toThrow(/validity/);
  expect(() => buildVecchiaConditioningSets({...props, times: [0]})).toThrow(/times/);
  expect(() => buildVecchiaConditioningSets({...props, order: [0, 1, 2]})).toThrow(/order/);
});

it('S1 treats either loss word as incomplete even if the complete word disagrees', () => {
  for (const neighborhoodStatus of [Uint32Array.of(1, 1, 0), Uint32Array.of(1, 0, 1)]) {
    const result = decodeGPUVecchiaLogLikelihoodResult({
      logLikelihood: Float32Array.of(-8, 0.125),
      status: Uint32Array.of(0, 0, 1),
      neighborhoodStatus
    });
    expect(result.status.complete).toBe(false);
    expect(Number.isNaN(result.logLikelihood)).toBe(true);
    const fisher = decodeGPUVecchiaFisherScoringResult({
      totals: new Float32Array(20),
      status: Uint32Array.of(0, 0, 1),
      neighborhoodStatus
    });
    expect(fisher.status.complete).toBe(false);
    expect(fisher.score).toHaveLength(3);
    expect(fisher.information).toHaveLength(6);
    expect([...fisher.score, ...fisher.information].every(Number.isNaN)).toBe(true);
  }
});

for (const kind of ['spatial', 'temporal', 'fisher'] as const) {
  it(`S2 rejects overlapping aligned publication bindings for ${kind}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `round-one-${serial++}`, format, length);
    const backing = view('uint32', 32);
    const totals = graph.createDataView(backing.buffer, {
      format: 'float32',
      length: kind === 'fisher' ? 20 : 2,
      byteOffset: 0
    });
    const evidence = graph.createDataView(backing.buffer, {
      format: 'uint32',
      length: 3,
      byteOffset: kind === 'fisher' ? 80 : 8
    });
    const props = {
      positions: view('float32x2', 1),
      observations: view('float32x4', 1),
      neighborhoodOffsets: view('uint32', 2),
      neighborhoods: view('uint32', 1),
      neighborCount: 1,
      parameters: view('float32', kind === 'temporal' ? 28 : 8)
    };
    const output = {
      conditionals: view('float32x4', 1),
      conditionalStatus: view('uint32', 1),
      siteTerms: view('float32', kind === 'fisher' ? 10 : 1),
      siteStatus: view('uint32', 1),
      logLikelihood: totals,
      totals,
      status: view('uint32', 3),
      neighborhoodStatus: evidence
    };
    try {
      expect(() =>
        kind === 'fisher'
          ? new GPUVecchiaFisherScoring({...props, output})
          : kind === 'temporal'
            ? new GPUVecchiaSpaceTimeLogLikelihood({
                ...props,
                times: view('float32', 1),
                temporalContext: operationProps.temporalContext,
                output
              })
            : new GPUVecchiaLogLikelihood({...props, output})
      ).toThrow(/overlap/);
    } finally {
      device.destroy();
    }
  });
}

it('S4 publishes independently empty storage when no data row is eligible', () => {
  for (const props of [{positions: []}, {positions: [NaN, NaN], validity: [0]}]) {
    const prediction = buildVecchiaPredictionNeighborhoods({
      ...props,
      queries: new Float32Array(4096 * 2),
      neighborCount: 24
    });
    expect(prediction.offsets).toHaveLength(4097);
    expect(Array.from(prediction.offsets).every(value => value === 0)).toBe(true);
    expect(prediction.neighbors).toHaveLength(0);
    expect(prediction.neighbors.buffer.byteLength).toBe(0);
    expect(prediction.counts?.neighborCount).toBe(24);
    const conditioning = buildVecchiaConditioningSets({...props, neighborCount: 24});
    expect(conditioning.neighbors.buffer.byteLength).toBe(0);
  }
});

it('S3 rejects vacuous loss checks and incomplete Fisher recovery', () => {
  expect(() => expectVecchiaSuppressedTotals([], false)).toThrow();
  expect(() => expectVecchiaSuppressedTotals(Float32Array.of(NaN, NaN), true)).toThrow();
  const recovered = Float32Array.from(VECCHIA_MARGINAL_EXPECTED.flatMap(value => [value, 0]));
  expectVecchiaMarginalTotals(recovered, true);
  recovered.fill(NaN, 2);
  expect(() => expectVecchiaMarginalTotals(recovered, true)).toThrow();
});

it('S5 shows the likelihood reader object result in the guide', () => {
  const guide = readFileSync('docs/api/gpu-spatial-analysis/vecchia-likelihood.md', 'utf8');
  expect(guide).toContain('const {logLikelihood} = getVecchiaLogLikelihoodResult(total, words)');
});

it('N1 rejects zero smooth nugget at a tiny range without changing target bytes', () => {
  const target = new Float32Array(30).fill(77);
  const before = new Uint8Array(target.buffer).slice();
  expect(() =>
    getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(
      {
        type: 'metric',
        joint: {model: 'matern-3/2', sill: 2, range: 1e-10, nugget: 0},
        anisotropy: 1
      },
      target
    )
  ).toThrow(/nugget/);
  expect(new Uint8Array(target.buffer)).toEqual(before);
});

it('N1 uses the exact coincident discontinuity in every space-time family', () => {
  const structure = {model: 'matern-3/2' as const, sill: 1, range: 1e-10, nugget: 0};
  const models: SpaceTimeVariogramModel[] = [
    {type: 'separable', sill: 2, spatial: structure, temporal: structure},
    {type: 'product-sum', k: 1, spatial: structure, temporal: structure},
    {type: 'sum-metric', spatial: structure, temporal: structure, joint: structure, anisotropy: 1},
    {
      type: 'gneiting',
      sill: 2,
      nugget: 0,
      spatialRange: 1e-10,
      temporalRange: 1e-10,
      alpha: 0.75,
      gamma: 0.75,
      beta: 0.5,
      tau: 1
    }
  ];
  for (const model of models)
    expect(() => getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model)).toThrow(/nugget/);
  // n >= .001 * (2+n), so n=2/999 lies exactly on the accepted metric floor.
  expect(() =>
    getGPUVecchiaSpaceTimeLogLikelihoodParameterValues({
      type: 'metric',
      joint: {...structure, sill: 2, nugget: 2 / 999},
      anisotropy: 1
    })
  ).not.toThrow();
});

it('N1 pins exact accepted and rejected floor boundaries for all space-time families', () => {
  const structure = {model: 'matern-3/2' as const, sill: 1, range: 1e-10, nugget: 0};
  const metric = (nugget: number): SpaceTimeVariogramModel => ({
    type: 'metric',
    joint: {...structure, sill: 2, nugget},
    anisotropy: 1
  });
  const separable = (nugget: number): SpaceTimeVariogramModel => ({
    type: 'separable',
    sill: 2,
    spatial: {...structure, sill: 1 - nugget, nugget},
    temporal: structure
  });
  const product = (nugget: number): SpaceTimeVariogramModel => ({
    type: 'product-sum',
    k: 1,
    spatial: {...structure, sill: 1 - nugget, nugget},
    temporal: structure
  });
  const sum = (nugget: number): SpaceTimeVariogramModel => ({
    type: 'sum-metric',
    spatial: {...structure, sill: 1 - nugget, nugget},
    temporal: structure,
    joint: structure,
    anisotropy: 1
  });
  const gneiting = (nugget: number): SpaceTimeVariogramModel => ({
    type: 'gneiting',
    sill: 2,
    nugget,
    spatialRange: 1e-10,
    temporalRange: 1e-10,
    alpha: 0.75,
    gamma: 0.75,
    beta: 0.5,
    tau: 1
  });
  // Discontinuities: metric/gneiting=n with total 2+n; separable=2n with total 2;
  // product-sum=2n with total 3; sum-metric=n with total 3.
  for (const [create, floor] of [
    [metric, 2 / 999],
    [separable, 0.001],
    [product, 0.0015],
    [sum, 0.003],
    [gneiting, 2 / 999]
  ] as const) {
    expect(() => getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(create(floor))).not.toThrow();
    expect(() => getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(create(floor * 0.99))).toThrow(
      /nugget/
    );
  }
});

it('U1 requires every documented Vecchia public-entry addition (EGR-01)', async () => {
  const entry = await import('@luma-contributors/analysis/gpu-spatial-analysis');
  const exports: Record<string, unknown> = entry;
  const functions = [
    'decodeGPUVecchiaLogLikelihoodResult',
    'decodeGPUVecchiaLogLikelihoodStatus',
    'decodeGPUVecchiaSpaceTimeLogLikelihoodResult',
    'decodeGPUVecchiaSpaceTimeLogLikelihoodStatus',
    'decodeGPUVecchiaFisherScoringResult',
    'decodeGPUVecchiaFisherScoringStatus',
    'uploadVecchiaNeighborhoods'
  ];
  const names = [...functions, 'MAXIMUM_VECCHIA_NEIGHBOR_COUNT', 'MAXIMUM_VECCHIA_ENTRY_COUNT'];
  const missing = names.filter(name => exports[name] === undefined);
  expect(missing, `Missing documented Vecchia exports: ${missing.join(', ')}`).toEqual([]);
  for (const name of functions) expect(typeof exports[name]).toBe('function');
  expect(exports['MAXIMUM_VECCHIA_NEIGHBOR_COUNT']).toBe(32);
  expect(exports['MAXIMUM_VECCHIA_ENTRY_COUNT']).toBe(0x7fffffff);
});
