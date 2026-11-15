// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkLouvain} from '../../../src/gpu-network/network-optimization/gpu-network-louvain';
import {GPUNetworkRationalAccess} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access';
import {ADAPTIVE_INCIRCLE_WGSL} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {
  GPUTransitionDiagnostics,
  GPU_TRANSITION_HOMOGENEITY
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics';
import {GPULabelPoint} from '../../../src/gpu-spatial-analysis/label-point/index';
import {
  GPUSpatialScanStatistic,
  GPU_SPATIAL_SCAN_STATISTIC_CLUSTER,
  GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX,
  GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS,
  GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS,
  GPU_SPATIAL_SCAN_STATISTIC_SUMMARY,
  GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH,
  getGPUSpatialScanStatisticParameterValues
} from '../../../src/gpu-spatial-analysis/scan-statistics/index';
import {GPUSpatialCrossValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import {createOptimizationHarness} from '../../gpu-network/network-optimization/optimization-test-harness';
import {flattenPolygonFeatures} from '../../gpu-spatial-analysis/geometry-measures/polygon-layout';
import {requireTestDevice} from '../../utils/require-test-device';
import {createSeededRandom, runWGSLModuleKernel} from '../../utils/wgsl-module-harness';

// Library audit of the "lost table fill" fault of the macOS test GPU (contributor contract
// section 11; dev-docs/research/math-correctness-review-2026-11-10/shader-compiler-findings.md,
// section "Library audit (2026-11-11)"): a local `array<u32, N>` or `array<i32, N>` whose stores
// are all at the index of a loop with a constant trip count, read at a run-time index, can read
// back as zeros: at 33 to about 250 entries in a shader that does nothing else, and from 12
// entries inside the old Louvain kernel.
//
// This file runs the contributors whose integer table has only such stores, at every size class
// their limits allow from 4 entries up, on inputs whose answer is known without the kernel: a
// closed form, a property of the construction, or exact integer arithmetic. Every case is built
// so that a zeroed table gives a different answer. `table-audit-network.spec.ts` and
// `table-audit-raster.spec.ts` run the contributors whose table also receives stores at
// data-dependent indices.

/** Table sizes of the audit; each test keeps the ones its contributor accepts. */
const AUDIT_SIZES = [4, 8, 12, 16, 20, 24, 28, 32, 33, 36, 40, 48, 64, 96, 128, 200, 256];

it('GPUNetworkRationalAccess keeps its minima and maxima tables from 4 to 64 origins', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const wrongCases: string[] = [];
  for (const originCount of AUDIT_SIZES.filter(size => size <= 64)) {
    // Two facilities of capacity `originCount`, every origin with a demand of 2. All demand
    // starts at the nearer facility (load 2 per unit of capacity against 0), so the equalizing
    // shift is the whole demand, the early-cycle cap halves it, and one unit per origin moves:
    // both facilities then carry `originCount`, the costs are equal and the run has converged.
    // `nearer` selects which facility is nearer, so the origin's cheapest facility (`minima`) is
    // facility 1 in one case and its dearest loaded facility (`maxima`) is facility 1 in the
    // other. With either table read as zeros the shift goes from facility 0 to facility 0 and
    // the allocation never changes.
    for (const nearer of [0, 1]) {
      const harness = createOptimizationHarness(device);
      const output = {
        assignments: harness.output('float32', 2 * originCount),
        scores: harness.output('float32', originCount),
        unservedDemand: harness.output('float32', originCount),
        summary: harness.output('uint32', 4)
      };
      const costs = new Float32Array(2 * originCount);
      for (let origin = 0; origin < originCount; origin++) {
        costs[2 * origin + (1 - nearer)] = nearer === 0 ? 0 : 0.001;
      }
      const result = await harness.run(
        new GPUNetworkRationalAccess({
          costs: harness.input('float32', costs),
          demand: harness.input('float32', new Float32Array(originCount).fill(2)),
          supply: harness.input('float32', Float32Array.of(originCount, originCount)),
          parameters: harness.input('float32', Float32Array.of(1, 1, 1, 1, 50, 0.01)),
          maximumCycles: 8,
          output
        })
      );
      const assignments = result(output.assignments);
      const isRight =
        assignments.every(value => value === 1) &&
        result(output.unservedDemand).every(value => value === 0) &&
        result(output.scores).every(value => Math.abs(value - 1) < 2e-3) &&
        result(output.summary)[0] === 1 &&
        result(output.summary)[1] === 1;
      if (!isRight) {
        wrongCases.push(
          `${originCount} origins, nearer ${nearer}: assignments ${assignments.slice(0, 4)}, status ${result(output.summary)}`
        );
      }
    }
  }
  expect(wrongCases).toEqual([]);
});

it('GPUTransitionDiagnostics keeps its pooled row from 4 to 64 classes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const wrongCases: string[] = [];
  for (const classCount of AUDIT_SIZES.filter(size => size <= 64)) {
    // Two conditions. From state `origin`, condition 0 only ever goes to `origin` (`stay` times)
    // and condition 1 only to `origin + 7` (`move` times). Each from-state is a 2 x 2 table with
    // an empty off diagonal, for which Pearson's chi-square is the table total, `stay + move`,
    // and the likelihood ratio is 2 (stay ln((stay + move) / stay) + move ln((stay + move) /
    // move)), with one degree of freedom. A pooled row of zeros gives Q = 0.
    const counts = new Uint32Array(2 * classCount * classCount);
    let expectedQ = 0;
    let expectedLikelihoodRatio = 0;
    for (let origin = 0; origin < classCount; origin++) {
      const stay = 3 + (origin % 5);
      const move = 2 + (origin % 3);
      counts[origin * classCount + origin] = stay;
      counts[(classCount + origin) * classCount + ((origin + 7) % classCount)] = move;
      expectedQ += stay + move;
      expectedLikelihoodRatio +=
        2 * (stay * Math.log((stay + move) / stay) + move * Math.log((stay + move) / move));
    }
    const harness = createOptimizationHarness(device);
    const homogeneity = harness.output('float32', 5);
    const result = await harness.run(
      new GPUTransitionDiagnostics({
        counts: harness.input('uint32', counts),
        classCount,
        conditionCount: 2,
        output: {homogeneity}
      })
    );
    const actual = result(homogeneity);
    const isRight =
      Math.abs(actual[GPU_TRANSITION_HOMOGENEITY.Q] / expectedQ - 1) < 1e-4 &&
      Math.abs(actual[GPU_TRANSITION_HOMOGENEITY.LR] / expectedLikelihoodRatio - 1) < 1e-4 &&
      actual[GPU_TRANSITION_HOMOGENEITY.DEGREES_OF_FREEDOM] === classCount;
    if (!isRight) {
      wrongCases.push(
        `${classCount} classes: Q ${actual[0]} (${expectedQ}), LR ${actual[2]} (${expectedLikelihoodRatio}), dof ${actual[4]}`
      );
    }
  }
  expect(wrongCases).toEqual([]);
});

it('the environmental fold design keeps its assignment table from 4 to 256 rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const wrongCases: string[] = [];
  for (const rowCount of AUDIT_SIZES) {
    // One feature with two well separated groups: every third row near 0, the others near 100.
    // Two clusters of any k-means are those groups. A zeroed table puts every row in fold 0.
    const isFirstGroup = (row: number): boolean => row % 3 === 0;
    const features = Float32Array.from(
      {length: rowCount},
      (_, row) => (isFirstGroup(row) ? 0 : 100) + 0.001 * row
    );
    const positions = Float32Array.from({length: 2 * rowCount}, (_, word) =>
      word % 2 === 0 ? word / 2 : 0
    );
    const harness = createOptimizationHarness(device);
    const diagnostics = harness.output('float32', 4);
    const output = {
      folds: harness.output('uint32', rowCount),
      training: harness.output('uint32', 2 * rowCount),
      testing: harness.output('uint32', 2 * rowCount),
      counts: harness.output('uint32', 6),
      status: harness.output('uint32', 2)
    };
    const result = await harness.run(
      new GPUSpatialCrossValidation({
        positions: harness.input('float32x2', positions),
        blocks: harness.input('uint32', new Uint32Array(rowCount)),
        foldCount: 2,
        seed: 17,
        design: {
          mode: 'environmental',
          features: harness.input('float32', features),
          featureCount: 1,
          standardize: false,
          diagnostics
        },
        output
      })
    );
    const folds = result(output.folds);
    const firstGroupFold = folds[0];
    const isRight =
      result(diagnostics)[0] === 0 &&
      firstGroupFold < 2 &&
      folds.every(
        (fold, row) => fold === (isFirstGroup(row) ? firstGroupFold : 1 - firstGroupFold)
      );
    if (!isRight) {
      wrongCases.push(
        `${rowCount} rows: folds ${folds.slice(0, 8)}, state ${result(diagnostics)[0]}`
      );
    }
  }
  expect(wrongCases).toEqual([]);
});

/** Modularity of a partition of a symmetric weight matrix (Newman 2006), in float64. */
function getModularity(weights: Float32Array, nodeCount: number, communities: number[]): number {
  let total = 0;
  const strengths = new Map<number, number>();
  let internal = 0;
  for (let source = 0; source < nodeCount; source++) {
    for (let destination = 0; destination < nodeCount; destination++) {
      const weight = weights[source * nodeCount + destination];
      total += weight;
      strengths.set(communities[source], (strengths.get(communities[source]) ?? 0) + weight);
      if (communities[source] === communities[destination]) {
        internal += weight;
      }
    }
  }
  let penalty = 0;
  for (const strength of strengths.values()) {
    penalty += (strength / total) ** 2;
  }
  return internal / total - penalty;
}

it('GPUNetworkLouvain keeps its group table at every size from 2 to 32 vertices', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const wrongCases: string[] = [];
  const runLouvain = async (nodeCount: number, weights: Float32Array, seed?: number) => {
    const harness = createOptimizationHarness(device);
    const output = {
      communities: harness.output('uint32', nodeCount),
      objectiveTrace: harness.output('float32', 129),
      summary: harness.output('uint32', 4)
    };
    const result = await harness.run(
      new GPUNetworkLouvain({
        nodeCount,
        weights: harness.input('float32', weights),
        parameters: harness.input('float32', Float32Array.of(1, 0)),
        ...(seed === undefined ? {} : {seed}),
        output
      })
    );
    return {
      communities: result(output.communities),
      objective: result(output.objectiveTrace).filter(Number.isFinite).at(-1) ?? Number.NaN,
      summary: result(output.summary)
    };
  };
  for (let nodeCount = 2; nodeCount <= 32; nodeCount++) {
    for (const seed of [undefined, 0, 7]) {
      // Disjoint cliques of four vertices (the last one smaller). No move between two cliques
      // is ever a candidate, and inside a clique every merge raises modularity, so the answer
      // is the cliques, for any visit order.
      const cliques = new Float32Array(nodeCount * nodeCount);
      for (let source = 0; source < nodeCount; source++) {
        for (let destination = 0; destination < nodeCount; destination++) {
          if (source !== destination && Math.floor(source / 4) === Math.floor(destination / 4)) {
            cliques[source * nodeCount + destination] = 1;
          }
        }
      }
      const expectedCliques = Array.from({length: nodeCount}, (_, vertex) =>
        Math.floor(vertex / 4)
      );
      const cliqueRun = await runLouvain(nodeCount, cliques, seed);
      const isSamePartition = cliqueRun.communities.every((community, vertex) =>
        cliqueRun.communities.every(
          (other, otherVertex) =>
            (community === other) === (expectedCliques[vertex] === expectedCliques[otherVertex])
        )
      );
      const expectedObjective = getModularity(cliques, nodeCount, expectedCliques);
      if (
        !isSamePartition ||
        Math.abs(cliqueRun.objective - expectedObjective) > 1e-4 ||
        cliqueRun.summary[0] !== 1 ||
        cliqueRun.summary[1] !== 1
      ) {
        wrongCases.push(
          `cliques, ${nodeCount} vertices, seed ${seed}: ${cliqueRun.communities}, objective ${cliqueRun.objective} (${expectedObjective})`
        );
      }
      if (nodeCount < 12 || nodeCount % 2 === 1) {
        continue;
      }
      // A ring of pairs: the two vertices of a pair are joined with weight 1.5 and consecutive
      // pairs with weight 1. The first level makes the pairs. On the graph of pairs, joining two
      // neighbouring pairs raises modularity by (2 / T) (1 - 2.5 / (pairs / 2)) > 0 for six
      // pairs or more (T the total weight), so a run that stops at the pairs has not carried
      // its groups to the second level. That is what a zeroed group table gives.
      const pairCount = nodeCount / 2;
      const ring = new Float32Array(nodeCount * nodeCount);
      for (let pair = 0; pair < pairCount; pair++) {
        const first = 2 * pair;
        const second = 2 * pair + 1;
        const next = (2 * pair + 2) % nodeCount;
        ring[first * nodeCount + second] = 1.5;
        ring[second * nodeCount + first] = 1.5;
        ring[second * nodeCount + next] = 1;
        ring[next * nodeCount + second] = 1;
      }
      const ringRun = await runLouvain(nodeCount, ring, seed);
      const pairs = Array.from({length: nodeCount}, (_, vertex) => Math.floor(vertex / 2));
      const keepsPairs = pairs.every(
        (pair, vertex) => ringRun.communities[vertex] === ringRun.communities[2 * pair]
      );
      const communityCount = new Set(ringRun.communities).size;
      const partitionObjective = getModularity(ring, nodeCount, ringRun.communities);
      if (
        !keepsPairs ||
        communityCount >= pairCount ||
        Math.abs(ringRun.objective - partitionObjective) > 1e-4 ||
        !(partitionObjective > getModularity(ring, nodeCount, pairs) + 1e-6) ||
        ringRun.summary[0] !== 1 ||
        ringRun.summary[1] !== 1
      ) {
        wrongCases.push(
          `ring, ${nodeCount} vertices, seed ${seed}: ${ringRun.communities}, objective ${ringRun.objective} (${partitionObjective}), summary ${ringRun.summary}`
        );
      }
    }
  }
  expect(wrongCases).toEqual([]);
}, 600_000);

/** Exact sign of the incircle determinant of float32 coordinates, in integer arithmetic. */
function getExactIncircleSign(coordinates: Float32Array): number {
  // Every float32 is an integer multiple of 2^-149; scaling by 2^149 keeps the sign.
  const toInteger = (value: number): bigint => {
    const view = new DataView(new ArrayBuffer(4));
    view.setFloat32(0, value);
    const bits = view.getUint32(0);
    const exponent = (bits >>> 23) & 0xff;
    const fraction = BigInt(bits & 0x7fffff);
    const magnitude = exponent === 0 ? fraction : (fraction | 0x800000n) << BigInt(exponent - 1);
    return bits >>> 31 === 1 ? -magnitude : magnitude;
  };
  const [ax, ay, bx, by, cx, cy, px, py] = Array.from(coordinates, toInteger);
  const rows = [
    [ax - px, ay - py],
    [bx - px, by - py],
    [cx - px, cy - py]
  ].map(([x, y]) => [x, y, x * x + y * y]);
  const determinant =
    rows[0][2] * (rows[1][0] * rows[2][1] - rows[2][0] * rows[1][1]) -
    rows[1][2] * (rows[0][0] * rows[2][1] - rows[2][0] * rows[0][1]) +
    rows[2][2] * (rows[0][0] * rows[1][1] - rows[1][0] * rows[0][1]);
  return determinant > 0n ? 1 : determinant < 0n ? -1 : 0;
}

it('the exact incircle workspace (140 limbs, filled by constant loops) gives the exact sign', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Quadruples on or within one float32 step of a common circle, at four scales, so the float
  // filter cannot decide and every limb of the workspace matters. The oracle is the determinant
  // in BigInt.
  const random = createSeededRandom(20261111);
  const lattice = [
    [5, 0],
    [4, 3],
    [3, 4],
    [0, 5],
    [-3, 4],
    [-4, 3],
    [-5, 0],
    [-4, -3],
    [-3, -4],
    [0, -5],
    [3, -4],
    [4, -3]
  ];
  const quadrupleCount = 512;
  const coordinates = new Float32Array(8 * quadrupleCount);
  for (let quadruple = 0; quadruple < quadrupleCount; quadruple++) {
    const scale = 2 ** (Math.floor(random() * 4) * 5 - 10);
    const offsetX = Math.floor(random() * 64) - 32;
    const offsetY = Math.floor(random() * 64) - 32;
    const chosen = [...lattice].sort(() => random() - 0.5).slice(0, 4);
    chosen.forEach(([x, y], point) => {
      const base = 8 * quadruple + 2 * point;
      coordinates[base] = (x + offsetX) * scale;
      coordinates[base + 1] = (y + offsetY) * scale;
    });
    // Every other quadruple gets its last point moved by a few float32 steps (a zero stays
    // zero: a subnormal beside it would put the exponents further apart than the exact path
    // accepts).
    const word = new Uint32Array(coordinates.buffer, 4 * (8 * quadruple + 6), 1);
    if (quadruple % 2 === 1 && word[0] << 1 !== 0) {
      word[0] += Math.floor(random() * 7) - 3;
    }
  }
  const words = await runWGSLModuleKernel(device, {
    id: 'table-audit-incircle-exact',
    inputs: {coordinates},
    outputLength: quadrupleCount,
    invocationCount: quadrupleCount,
    declarations: ADAPTIVE_INCIRCLE_WGSL,
    body: `let base = coordinatesOffset + 8u * index;
  let sign = incircleExact(
    vec2<f32>(coordinates[base], coordinates[base + 1u]),
    vec2<f32>(coordinates[base + 2u], coordinates[base + 3u]),
    vec2<f32>(coordinates[base + 4u], coordinates[base + 5u]),
    vec2<f32>(coordinates[base + 6u], coordinates[base + 7u]));
  output[outputOffset + index] = u32(sign + 1);`
  });
  const wrongCases: string[] = [];
  const signCounts = [0, 0, 0];
  for (let quadruple = 0; quadruple < quadrupleCount; quadruple++) {
    const expected = getExactIncircleSign(coordinates.subarray(8 * quadruple, 8 * quadruple + 8));
    signCounts[expected + 1]++;
    if (words[quadruple] !== expected + 1) {
      wrongCases.push(`quadruple ${quadruple}: ${words[quadruple] - 1} (${expected})`);
    }
  }
  // The cases cover all three signs.
  expect(Math.min(...signCounts)).toBeGreaterThan(20);
  expect(wrongCases).toEqual([]);
});

it('GPUNetworkLouvain keeps its group table for other level, sweep, direction and seed settings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Every combination below is a different compiled kernel. The ring of pairs of the test above:
  // a correct run joins neighbouring pairs at the second level.
  const wrongCases: string[] = [];
  for (const nodeCount of [12, 16, 24, 32]) {
    const pairCount = nodeCount / 2;
    const ring = new Float32Array(nodeCount * nodeCount);
    for (let pair = 0; pair < pairCount; pair++) {
      const first = 2 * pair;
      const second = 2 * pair + 1;
      const next = (2 * pair + 2) % nodeCount;
      ring[first * nodeCount + second] = 1.5;
      ring[second * nodeCount + first] = 1.5;
      ring[second * nodeCount + next] = 1;
      ring[next * nodeCount + second] = 1;
    }
    const pairs = Array.from({length: nodeCount}, (_, vertex) => Math.floor(vertex / 2));
    const pairObjective = getModularity(ring, nodeCount, pairs);
    for (const maximumLevels of [2, 8, 32]) {
      for (const maximumIterations of [4, 16, 64]) {
        for (const directed of [false, true]) {
          for (const seed of [undefined, 7]) {
            const harness = createOptimizationHarness(device);
            const output = {
              communities: harness.output('uint32', nodeCount),
              objectiveTrace: harness.output('float32', 1 + maximumLevels * maximumIterations),
              summary: harness.output('uint32', 4)
            };
            const result = await harness.run(
              new GPUNetworkLouvain({
                nodeCount,
                weights: harness.input('float32', ring),
                parameters: harness.input('float32', Float32Array.of(1, 0)),
                maximumLevels,
                maximumIterations,
                directed,
                ...(seed === undefined ? {} : {seed}),
                output
              })
            );
            const communities = result(output.communities);
            const objective = getModularity(ring, nodeCount, communities);
            const reported =
              result(output.objectiveTrace).filter(Number.isFinite).at(-1) ?? Number.NaN;
            if (
              !pairs.every((pair, vertex) => communities[vertex] === communities[2 * pair]) ||
              new Set(communities).size >= pairCount ||
              Math.abs(reported - objective) > 1e-4 ||
              !(objective > pairObjective + 1e-6) ||
              result(output.summary)[0] !== 1
            ) {
              wrongCases.push(
                `${nodeCount} vertices, ${maximumLevels} levels, ${maximumIterations} sweeps, directed ${directed}, seed ${seed}: ${communities}`
              );
            }
          }
        }
      }
    }
  }
  expect(wrongCases).toEqual([]);
}, 600_000);

it('GPUSpatialScanStatistic keeps its per-bucket case sums from 8 to 32 time buckets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const wrongCases: string[] = [];
  for (const timeBuckets of [8, 12, 16, 20, 24, 28, 32]) {
    // A 5 x 5 lattice, a population of 100 and 5 cases in every (zone, bucket) cell, except 60
    // cases in the centre zone in three consecutive buckets. The most likely cluster is that
    // block: one zone, those three buckets. Its statistic is Kulldorff's Poisson likelihood
    // ratio c ln(c / e) + (C - c) ln((C - c) / (C - e)) with e = C * (block population / total).
    const side = 5;
    const zones = side * side;
    const positions = new Float32Array(2 * zones);
    const cases = new Uint32Array(zones * timeBuckets).fill(5);
    const baseline = new Float32Array(zones * timeBuckets).fill(100);
    for (let zone = 0; zone < zones; zone++) {
      positions[2 * zone] = 10 * (zone % side);
      positions[2 * zone + 1] = 10 * Math.floor(zone / side);
    }
    const centre = 12;
    const firstBucket = timeBuckets - 5;
    const lastBucket = timeBuckets - 3;
    for (let bucket = firstBucket; bucket <= lastBucket; bucket++) {
      cases[centre * timeBuckets + bucket] = 60;
    }
    const totalCases = 5 * zones * timeBuckets + 3 * 55;
    const observedCases = 180;
    const expectedCases = (totalCases * 3) / (zones * timeBuckets);
    const expectedRatio =
      observedCases * Math.log(observedCases / expectedCases) +
      (totalCases - observedCases) *
        Math.log((totalCases - observedCases) / (totalCases - expectedCases));
    const harness = createOptimizationHarness(device);
    const output = {
      clusterIndices: harness.output('uint32', 2 * GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS),
      clusterStatistics: harness.output('float32', 2 * GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS),
      statistics: harness.output('float32', 33),
      summary: harness.output('uint32', GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH)
    };
    const result = await harness.run(
      new GPUSpatialScanStatistic({
        positions: harness.input('float32x2', positions),
        cases: harness.input('uint32', cases),
        baseline: harness.input('float32', baseline),
        timeBuckets,
        maximumWindowZones: 8,
        maximumPermutations: 32,
        maximumClusters: 2,
        parameters: harness.input(
          'uint32',
          getGPUSpatialScanStatisticParameterValues({
            seed: 1,
            permutationCount: 19,
            maximumPopulationFraction: 0.25,
            maximumTimeBuckets: 4,
            windowShape: 'circle'
          })
        ),
        output
      })
    );
    const indices = result(output.clusterIndices);
    const statistics = result(output.clusterStatistics);
    const isRight =
      result(output.summary)[GPU_SPATIAL_SCAN_STATISTIC_SUMMARY.clusterCount] >= 1 &&
      indices[GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX.center] === centre &&
      indices[GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX.zoneCount] === 1 &&
      indices[GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX.firstBucket] === firstBucket &&
      indices[GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX.lastBucket] === lastBucket &&
      statistics[GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.observedCases] === observedCases &&
      Math.abs(statistics[GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.expectedCases] / expectedCases - 1) <
        1e-4 &&
      Math.abs(
        statistics[GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.logLikelihoodRatio] / expectedRatio - 1
      ) < 2e-3;
    if (!isRight) {
      wrongCases.push(
        `${timeBuckets} buckets: indices ${indices.slice(0, GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS)}, statistics ${statistics.slice(0, 3)} (${expectedRatio}, ${observedCases}, ${expectedCases})`
      );
    }
  }
  expect(wrongCases).toEqual([]);
}, 600_000);

it('GPULabelPoint keeps its candidate cell table from 1 to 16 refinement candidates', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A comb: a strip of width 1 on the left, a bar of height 1 along the bottom and a 20 x 20
  // square on the right. The three scanline seeds fall in the strip (distance 0.5); the pole of
  // inaccessibility is the centre of the square, (13, 10), at distance 10 from four sides. It is
  // found only through the table of best first-grid cells. The second feature is the same ring
  // with 48 vertices, which takes the cooperative path.
  const comb = [
    [0, 0],
    [23, 0],
    [23, 20],
    [3, 20],
    [3, 1],
    [1, 1],
    [1, 20],
    [0, 20]
  ];
  const denseComb = comb.flatMap(([x, y], vertex) => {
    const [nextX, nextY] = comb[(vertex + 1) % comb.length];
    return Array.from({length: 6}, (_, step) => [
      x + ((nextX - x) * step) / 6,
      y + ((nextY - y) * step) / 6
    ]);
  });
  const layout = flattenPolygonFeatures([[comb], [denseComb]]);
  const wrongCases: string[] = [];
  for (const refinementCandidates of [1, 2, 4, 8, 10, 12, 13, 14, 15, 16]) {
    const harness = createOptimizationHarness(device);
    const output = {
      points: harness.output('float32x2', 2),
      distances: harness.output('float32', 2),
      degenerate: harness.output('uint32', 2)
    };
    const result = await harness.run(
      new GPULabelPoint({
        positions: harness.input('float32x2', Float32Array.from(layout.positions)),
        ringOffsets: harness.input('uint32', Uint32Array.from(layout.ringOffsets)),
        polygonOffsets: harness.input('uint32', Uint32Array.from(layout.polygonOffsets)),
        featureOffsets: harness.input('uint32', Uint32Array.from(layout.featureOffsets)),
        refinementCandidates,
        output
      })
    );
    const points = result(output.points);
    const distances = result(output.distances);
    for (let feature = 0; feature < 2; feature++) {
      const isRight =
        Math.hypot(points[2 * feature] - 13, points[2 * feature + 1] - 10) < 0.02 &&
        distances[feature] > 9.98 &&
        distances[feature] < 10.001 &&
        result(output.degenerate)[feature] === 0;
      if (!isRight) {
        wrongCases.push(
          `${refinementCandidates} candidates, feature ${feature}: (${points[2 * feature]}, ${points[2 * feature + 1]}) at ${distances[feature]}`
        );
      }
    }
  }
  expect(wrongCases).toEqual([]);
});
