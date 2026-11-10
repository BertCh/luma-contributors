// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUEnsembleScores,
  GPUGaussianPredictiveScores
} from '../../../src/gpu-spatial-analysis/scoring-rules/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import fixtureJson from '../../oracles/fixtures/scoring-rules.json';

/** One ensemble case of `test/oracles/scoring-rules.py`. */
export type EnsembleCase = {
  name: string;
  memberCount: number;
  dimension: number;
  ensemble: number[];
  observations: number[];
  crps: number[];
  crpsFair: number[];
  energyScore: number;
  energyScoreFair: number;
  variogramOrder: number;
  variogramPairsAll: [number, number][];
  variogramScoreAll: number;
  variogramPairsNeighbours: [number, number][];
  variogramWeights: number[];
  variogramScoreNeighbours: number;
  variogramScoreNeighboursOrderOne: number;
};

/** The decoded oracle fixture. */
export const SCORING_FIXTURE = decodeOracleFixture<{
  tolerances: {
    crpsAbsolute: number;
    crpsRelative: number;
    energyRelative: number;
    variogramRelative: number;
    gaussianAbsolute: number;
    gaussianRelative: number;
  };
  hand: {members: number[]; observation: number; crps: number; crpsFair: number};
  ensembles: EnsembleCase[];
  gaussian: {mean: number; variance: number; observation: number; crps: number}[];
}>(fixtureJson);

/** Scores of one ensemble; `null` for a score that was not requested. */
export type EnsembleScoreResult = {
  crps: Float32Array | null;
  energyScore: number | null;
  variogramScore: number | null;
};

/** Runs `GPUEnsembleScores` once. */
export async function runEnsembleScores(
  device: Device,
  entry: Pick<EnsembleCase, 'memberCount' | 'dimension' | 'ensemble' | 'observations'>,
  options: {
    fair?: boolean;
    crps?: boolean;
    energyScore?: boolean;
    variogramPairs?: readonly (readonly number[])[];
    variogramWeights?: readonly number[];
    variogramOrder?: number;
  }
): Promise<EnsembleScoreResult> {
  const graph = new GPUCommandGraph(device, {id: 'ensemble-scores'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const crpsBuffer = options.crps ? output(entry.dimension) : undefined;
  const energyBuffer = options.energyScore ? output(1) : undefined;
  const variogramBuffer = options.variogramPairs ? output(1) : undefined;
  const pairs = options.variogramPairs
    ? Uint32Array.from(options.variogramPairs.flat())
    : undefined;
  graph.add(
    new GPUEnsembleScores({
      ensemble: importGraphBuffer(
        graph,
        'ensemble',
        input(Float32Array.from(entry.ensemble)),
        'float32'
      ),
      observations: importGraphBuffer(
        graph,
        'observations',
        input(Float32Array.from(entry.observations)),
        'float32'
      ),
      memberCount: entry.memberCount,
      dimension: entry.dimension,
      fair: options.fair,
      variogramOrder: options.variogramOrder,
      variogramPairs: pairs
        ? importGraphBuffer(graph, 'pairs', input(pairs), 'uint32x2')
        : undefined,
      variogramWeights: options.variogramWeights
        ? importGraphBuffer(
            graph,
            'weights',
            input(Float32Array.from(options.variogramWeights)),
            'float32'
          )
        : undefined,
      output: {
        crps: crpsBuffer
          ? importGraphBuffer(graph, 'o-crps', crpsBuffer, 'float32', entry.dimension)
          : undefined,
        energyScore: energyBuffer
          ? importGraphBuffer(graph, 'o-energy', energyBuffer, 'float32', 1)
          : undefined,
        variogramScore: variogramBuffer
          ? importGraphBuffer(graph, 'o-variogram', variogramBuffer, 'float32', 1)
          : undefined
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      crps: crpsBuffer ? Float32Array.from(await readFloat32(crpsBuffer, entry.dimension)) : null,
      energyScore: energyBuffer ? (await readFloat32(energyBuffer, 1))[0] : null,
      variogramScore: variogramBuffer ? (await readFloat32(variogramBuffer, 1))[0] : null
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

/** Runs `GPUGaussianPredictiveScores` once. */
export async function runGaussianScores(
  device: Device,
  means: readonly number[],
  variances: readonly number[],
  observations: readonly number[]
): Promise<Float32Array> {
  const graph = new GPUCommandGraph(device, {id: 'gaussian-scores'});
  const buffers: Buffer[] = [];
  const input = (values: readonly number[]) => {
    const buffer = createInputBuffer(device, Float32Array.from(values));
    buffers.push(buffer);
    return buffer;
  };
  const crps = createOutputBuffer(device, means.length);
  buffers.push(crps);
  graph.add(
    new GPUGaussianPredictiveScores({
      means: importGraphBuffer(graph, 'means', input(means), 'float32'),
      variances: importGraphBuffer(graph, 'variances', input(variances), 'float32'),
      observations: importGraphBuffer(graph, 'observations', input(observations), 'float32'),
      output: {crps: importGraphBuffer(graph, 'o-crps', crps, 'float32', means.length)}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return Float32Array.from(await readFloat32(crps, means.length));
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}
