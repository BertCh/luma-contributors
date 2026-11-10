// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPU repeat of dev-docs/research/spatiotemporal-review-2026-11-09/experiments/vecchia-float32.mjs.
// Off by default: VITE_VECCHIA_FLOAT32_STUDY=1 npx vitest run --project headless <this file>
// Optional VITE_VECCHIA_FLOAT32_STUDY_SIZES is a comma-separated row-count list.

import {describe, expect, it} from 'vitest';
import {getGPULocalKrigingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {
  buildVecchiaConditioningSets,
  getVecchiaMaxMinOrdering
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-conditioning-sets';
import {fitVecchiaCovarianceFisherScoring} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-fisher-fit';
import {getGPUVecchiaLogLikelihoodParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-parameters';
import {
  createExperimentGaussian,
  createExperimentRandom,
  evaluateReferenceVecchiaFisher,
  evaluateReferenceVecchiaLogLikelihood,
  getEffectiveRangeForExperiment,
  simulateReferenceVecchiaField,
  type ReferenceVecchiaKernel
} from './vecchia-likelihood-reference';
import {
  createVecchiaFisherHarness,
  createVecchiaLikelihoodHarness
} from './vecchia-likelihood-harness';

const SIDE = 100000;
const NEIGHBOR_COUNT = 30;
const TRUE_RANGE = 10000;
const RANGE_GRID_MULTIPLIERS = [0.5, 0.7, 0.85, 0.95, 1, 1.05, 1.15, 1.3, 1.5, 2];
const KERNELS: ReferenceVecchiaKernel[] = ['exponential', 'matern-3/2', 'matern-5/2', 'gaussian'];
const NUGGETS = [0, 1e-4, 1e-3, 1e-2, 1e-1];

const ENVIRONMENT = (import.meta as unknown as {env: Record<string, string | undefined>}).env;
const isEnabled = Boolean(ENVIRONMENT['VITE_VECCHIA_FLOAT32_STUDY']);
const TIMING_SIZE = Number(ENVIRONMENT['VITE_VECCHIA_FLOAT32_STUDY_TIMING_SIZE'] ?? 0);
const SIZES = (ENVIRONMENT['VITE_VECCHIA_FLOAT32_STUDY_SIZES'] ?? '10000').split(',').map(Number);

/** Experiment scene: points then fields from one mulberry32 stream, in the experiment's order. */
function createBaseScene(
  rowCount: number,
  ordering: 'random' | 'max-min'
): {
  positions: Float64Array;
  offsets: Uint32Array;
  neighbors: Uint32Array;
  gaussian: () => number;
} {
  const random = createExperimentRandom(20261109);
  let positions = new Float64Array(rowCount * 2);
  for (let index = 0; index < rowCount * 2; index++) {
    positions[index] = (random() - 0.5) * SIDE;
  }
  if (ordering === 'max-min') {
    const order = getVecchiaMaxMinOrdering(positions);
    const reordered = new Float64Array(rowCount * 2);
    for (let row = 0; row < rowCount; row++) {
      reordered[row * 2] = positions[order[row] * 2];
      reordered[row * 2 + 1] = positions[order[row] * 2 + 1];
    }
    positions = reordered;
  }
  const {offsets, neighbors} = buildVecchiaConditioningSets({
    positions,
    neighborCount: NEIGHBOR_COUNT
  });
  return {positions, offsets, neighbors, gaussian: createExperimentGaussian(random)};
}

function formatScientific(value: number): string {
  if (!Number.isFinite(value)) {
    return String(value);
  }
  return Number(value.toPrecision(3)).toString();
}

type StudyRow = {
  kernel: ReferenceVecchiaKernel;
  nugget: number;
  failures: number;
  gridMultiplierBestF64: number;
  gridMultiplierBestF32: number | 'n/a';
  maximumError: number;
  differenceError: number;
  gap: number;
  bestF64: number;
  milliseconds: number;
};

async function runGridRow(
  harness: Awaited<ReturnType<typeof createVecchiaLikelihoodHarness>>,
  scene: ReturnType<typeof createBaseScene>,
  values: Float64Array,
  kernel: ReferenceVecchiaKernel,
  nugget: number
): Promise<StudyRow> {
  const reference: number[] = [];
  const gpu: number[] = [];
  let failures = 0;
  let elapsed = 0;
  for (const multiplier of RANGE_GRID_MULTIPLIERS) {
    const range = getEffectiveRangeForExperiment(kernel, multiplier * TRUE_RANGE);
    const covariance = {model: kernel, range, sill: 1, nugget};
    const exact = evaluateReferenceVecchiaLogLikelihood(
      values,
      scene.positions,
      scene.offsets,
      scene.neighbors,
      covariance
    );
    reference.push(exact.logLikelihood);
    const result = await harness.evaluate(getGPULocalKrigingParameterValues({covariance}));
    gpu.push(result.logLikelihood);
    failures = Math.max(failures, result.failedSiteCount);
    elapsed += result.elapsedMilliseconds;
  }
  const argumentBest = (curve: number[]): number =>
    curve.indexOf(Math.max(...curve.filter(Number.isFinite)));
  // Row 'gaussian, nugget 0' fails in binary64 too at n = 10,000: no finite reference curve.
  const best = argumentBest(reference);
  if (best < 0) {
    return {
      kernel,
      nugget,
      failures,
      gridMultiplierBestF64: Number.NaN,
      gridMultiplierBestF32: 'n/a',
      maximumError: Number.NaN,
      differenceError: Number.NaN,
      gap: Number.NaN,
      bestF64: Number.NaN,
      milliseconds: elapsed / RANGE_GRID_MULTIPLIERS.length
    };
  }
  const hasGPU = failures === 0;
  const gpuBest = hasGPU ? argumentBest(gpu) : -1;
  return {
    kernel,
    nugget,
    failures,
    gridMultiplierBestF64: RANGE_GRID_MULTIPLIERS[best],
    gridMultiplierBestF32: hasGPU ? RANGE_GRID_MULTIPLIERS[gpuBest] : 'n/a',
    maximumError: hasGPU
      ? Math.max(...reference.map((value, index) => Math.abs(value - gpu[index])))
      : Number.NaN,
    differenceError: hasGPU
      ? Math.max(
          ...reference.map((value, index) =>
            Math.abs(value - reference[best] - (gpu[index] - gpu[best]))
          )
        )
      : Number.NaN,
    gap: Math.min(
      ...reference.map((value, index) => (index === best ? Infinity : reference[best] - value))
    ),
    bestF64: reference[best],
    milliseconds: elapsed / RANGE_GRID_MULTIPLIERS.length
  };
}

function printStudyTable(title: string, rows: readonly StudyRow[]): void {
  const lines = [
    title,
    '| Kernel | Nugget / sill | Failed sites | GPU error | Difference error | Gap | Same best range | GPU best / truth | Best f64 LL | ms per evaluation |',
    '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |'
  ];
  for (const row of rows) {
    lines.push(
      `| ${row.kernel} | ${row.nugget} | ${row.failures} | ${formatScientific(row.maximumError)} | ${formatScientific(row.differenceError)} | ${formatScientific(row.gap)} | ${
        row.gridMultiplierBestF32 === 'n/a'
          ? 'n/a'
          : row.gridMultiplierBestF32 === row.gridMultiplierBestF64
            ? 'yes'
            : 'no'
      } | ${row.gridMultiplierBestF32} | ${row.bestF64.toFixed(3)} | ${row.milliseconds.toFixed(1)} |`
    );
  }
  console.log(`VECCHIA_STUDY_TABLE\n${lines.join('\n')}\nVECCHIA_STUDY_TABLE_END`);
}

describe.skipIf(!isEnabled)('Vecchia float32 GPU study', () => {
  for (const rowCount of SIZES) {
    it(`repeats the float32 experiment on the GPU at n=${rowCount}`, {
      timeout: 7200_000
    }, async () => {
      const scene = createBaseScene(rowCount, 'random');
      const harness = await createVecchiaLikelihoodHarness({
        positions: scene.positions,
        values: new Float64Array(rowCount),
        offsets: scene.offsets,
        neighbors: scene.neighbors,
        neighborCount: NEIGHBOR_COUNT
      });
      // Each row needs its own field in the observations buffer, so rebuild the harness per row.
      harness.destroy();
      const isFull = rowCount <= 10000;
      const kernels = isFull ? KERNELS : KERNELS.slice(0, 2);
      const nuggets = isFull ? NUGGETS : [0, 1e-3, 1e-2];
      const rows: StudyRow[] = [];
      for (const kernel of KERNELS) {
        for (const nugget of NUGGETS) {
          // The experiment simulates every kernel and nugget in this order from one stream;
          // keep the stream aligned even for rows that are not evaluated.
          const values = simulateReferenceVecchiaField(
            scene.gaussian,
            scene.positions,
            scene.offsets,
            scene.neighbors,
            {
              model: kernel,
              range: getEffectiveRangeForExperiment(kernel, TRUE_RANGE),
              sill: 1,
              nugget
            }
          );
          if (!kernels.includes(kernel) || !nuggets.includes(nugget)) {
            continue;
          }
          const rowHarness = await createVecchiaLikelihoodHarness({
            positions: scene.positions,
            values,
            offsets: scene.offsets,
            neighbors: scene.neighbors,
            neighborCount: NEIGHBOR_COUNT
          });
          try {
            rows.push(await runGridRow(rowHarness, scene, values, kernel, nugget));
            expect(rowHarness.buildCount).toBe(1);
          } finally {
            rowHarness.destroy();
          }
        }
      }
      printStudyTable(
        `n=${rowCount} m=${NEIGHBOR_COUNT} ordering=random device=${JSON.stringify(harness.device.info)}`,
        rows
      );
    });
  }

  it.runIf(TIMING_SIZE > 0)(
    `measures one evaluation at n=${TIMING_SIZE} (no reference)`,
    {timeout: 3600_000},
    async () => {
      const setupStart = performance.now();
      const scene = createBaseScene(TIMING_SIZE, 'random');
      const setupMilliseconds = performance.now() - setupStart;
      const values = simulateReferenceVecchiaField(
        scene.gaussian,
        scene.positions,
        scene.offsets,
        scene.neighbors,
        {model: 'exponential', range: 3 * TRUE_RANGE, sill: 1, nugget: 1e-3}
      );
      const harness = await createVecchiaLikelihoodHarness({
        positions: scene.positions,
        values,
        offsets: scene.offsets,
        neighbors: scene.neighbors,
        neighborCount: NEIGHBOR_COUNT
      });
      const times: number[] = [];
      let lastLogLikelihood = 0;
      for (let attempt = 0; attempt < 5; attempt++) {
        const result = await harness.evaluate(
          getGPULocalKrigingParameterValues({
            covariance: {model: 'exponential', range: 3 * TRUE_RANGE, sill: 1, nugget: 1e-3}
          })
        );
        times.push(result.elapsedMilliseconds);
        lastLogLikelihood = result.logLikelihood;
      }
      harness.destroy();
      console.log(
        `VECCHIA_TIMING n=${TIMING_SIZE} setup(random order + conditioning sets) ms=${setupMilliseconds.toFixed(0)} evaluation ms=${times.map(time => time.toFixed(1)).join(', ')} ll=${lastLogLikelihood}`
      );
    }
  );

  it.runIf(Boolean(ENVIRONMENT['VITE_VECCHIA_FLOAT32_STUDY_FIT']))(
    'fits by Fisher scoring in float32 at the experiment size and compares with binary64',
    {timeout: 3600_000},
    async () => {
      const lines: string[] = [];
      for (const [kernel, trueRange, nugget] of [
        ['exponential', 3 * TRUE_RANGE, 0.05],
        ['matern-3/2', getEffectiveRangeForExperiment('matern-3/2', TRUE_RANGE), 0.02]
      ] as const) {
        const scene = createBaseScene(SIZES[0], 'random');
        const covariance = {model: kernel, range: trueRange, sill: 1, nugget};
        const values = simulateReferenceVecchiaField(
          scene.gaussian,
          scene.positions,
          scene.offsets,
          scene.neighbors,
          covariance
        );
        const harness = await createVecchiaFisherHarness({
          positions: scene.positions,
          values,
          offsets: scene.offsets,
          neighbors: scene.neighbors,
          neighborCount: NEIGHBOR_COUNT
        });
        const initial = {model: kernel, range: trueRange * 1.8, sill: 0.6, nugget: nugget * 3};
        let elapsed = 0;
        let evaluations = 0;
        const gpuFit = await fitVecchiaCovarianceFisherScoring({
          initial,
          evaluate: async next => {
            const result = await harness.evaluate(getGPUVecchiaLogLikelihoodParameterValues(next));
            elapsed += result.elapsedMilliseconds;
            evaluations++;
            return result;
          }
        });
        const referenceFit = await fitVecchiaCovarianceFisherScoring({
          initial,
          logLikelihoodNoise: 1e-9,
          evaluate: next => ({
            ...evaluateReferenceVecchiaFisher(
              values,
              scene.positions,
              scene.offsets,
              scene.neighbors,
              next
            ),
            failedSiteCount: 0
          })
        });
        lines.push(
          `${kernel} n=${SIZES[0]} m=${NEIGHBOR_COUNT} gpuIterations=${gpuFit.iterationCount} referenceIterations=${referenceFit.iterationCount} converged=${gpuFit.converged}/${referenceFit.converged} gpu=[sill ${gpuFit.covariance.sill.toPrecision(6)}, range ${gpuFit.covariance.range.toFixed(1)}, nugget ${gpuFit.covariance.nugget!.toPrecision(5)}] ref=[sill ${referenceFit.covariance.sill.toPrecision(6)}, range ${referenceFit.covariance.range.toFixed(1)}, nugget ${referenceFit.covariance.nugget!.toPrecision(5)}] deltaLL=${(gpuFit.logLikelihood - referenceFit.logLikelihood).toFixed(4)} msPerEvaluation=${(elapsed / evaluations).toFixed(1)} evaluations=${evaluations}`
        );
        harness.destroy();
      }
      console.log(`VECCHIA_FISHER_STUDY\n${lines.join('\n')}\nVECCHIA_FISHER_STUDY_END`);
    }
  );

  it('repeats key rows under max-min ordering', {timeout: 7200_000}, async () => {
    const rowCount = SIZES[0];
    const scene = createBaseScene(rowCount, 'max-min');
    const rows: StudyRow[] = [];
    const cases: [ReferenceVecchiaKernel, number][] = [
      ['exponential', 0],
      ['exponential', 1e-3],
      ['matern-3/2', 1e-4],
      ['matern-3/2', 1e-3],
      ['matern-5/2', 1e-3],
      ['gaussian', 1e-3]
    ];
    for (const [kernel, nugget] of cases) {
      const values = simulateReferenceVecchiaField(
        scene.gaussian,
        scene.positions,
        scene.offsets,
        scene.neighbors,
        {
          model: kernel,
          range: getEffectiveRangeForExperiment(kernel, TRUE_RANGE),
          sill: 1,
          nugget
        }
      );
      const harness = await createVecchiaLikelihoodHarness({
        positions: scene.positions,
        values,
        offsets: scene.offsets,
        neighbors: scene.neighbors,
        neighborCount: NEIGHBOR_COUNT
      });
      try {
        rows.push(await runGridRow(harness, scene, values, kernel, nugget));
      } finally {
        harness.destroy();
      }
    }
    printStudyTable(`n=${rowCount} m=${NEIGHBOR_COUNT} ordering=max-min`, rows);
  });
});
