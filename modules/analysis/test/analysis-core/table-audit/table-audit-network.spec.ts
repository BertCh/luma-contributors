// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUNetworkFlow} from '../../../src/gpu-network/network-optimization/gpu-network-flow';
import {addResistanceConnectivityRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/resistance-connectivity';
import {addVisibilityGraphRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/visibility-graph';
import {
  getGPUExpectationBasedScanParameterValues,
  GPUExpectationBasedSpaceTimeScan,
  GPU_EXPECTATION_BASED_SCAN_NO_ZONE
} from '../../../src/gpu-spatial-analysis/scan-statistics/index';
import {GPUSpatialCrossValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createSeededRandom} from '../../gpu-spatial-analysis/catchment-accessibility/rig';

// Audit of local WGSL tables of 33 to about 250 entries (shader-compiler-findings.md section 3):
// each real contributor runs at table sizes inside and outside the window on an input whose
// answer is known without the kernel. A wrong size is recorded, not thrown, so one run lists all.

const COMPONENTS: Record<string, number> = {
  float32: 1,
  uint32: 1,
  float32x2: 2,
  uint32x2: 2,
  float32x4: 4,
  uint32x4: 4
};

/** Test-owned graph, buffers and readback. */
class TableAuditRig {
  readonly graph: GPUCommandGraph;
  private readonly buffers: Buffer[] = [];
  private readonly outputBuffers = new Map<GraphDataView, Buffer>();
  private serial = 0;
  private compiled?: {destroy(): void};

  constructor(readonly device: Device) {
    this.graph = new GPUCommandGraph(device, {id: 'table-audit'});
  }

  input<Format extends GPUVectorFormat>(
    values: Float32Array | Uint32Array,
    format: Format
  ): GraphDataView<Format> {
    const buffer = createInputBuffer(this.device, values);
    this.buffers.push(buffer);
    return importGraphBuffer(
      this.graph,
      `input-${this.serial++}`,
      buffer,
      format,
      values.length / COMPONENTS[format]
    );
  }

  output<Format extends GPUVectorFormat>(format: Format, rows: number): GraphDataView<Format> {
    const buffer = createOutputBuffer(this.device, rows * COMPONENTS[format]);
    buffer.write(new Uint32Array(buffer.byteLength / 4).fill(0x7f7f7f7f));
    this.buffers.push(buffer);
    const view = importGraphBuffer(this.graph, `output-${this.serial++}`, buffer, format, rows);
    this.outputBuffers.set(view, buffer);
    return view;
  }

  run(...producers: Parameters<GPUCommandGraph['add']>[0][]): void {
    for (const producer of producers) this.graph.add(producer);
    const compiled = this.graph.compile();
    this.compiled = compiled;
    submitGraph(this.device, compiled, undefined);
  }

  readFloat(view: GraphDataView): Promise<number[]> {
    return readFloat32(this.outputBuffers.get(view)!, view.length * COMPONENTS[view.format]);
  }

  readUint(view: GraphDataView): Promise<number[]> {
    return readUint32(this.outputBuffers.get(view)!, view.length * COMPONENTS[view.format]);
  }

  destroy(): void {
    this.compiled?.destroy();
    for (const buffer of this.buffers) buffer.destroy();
  }
}

async function createRig(): Promise<TableAuditRig> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return new TableAuditRig(device);
}

function isClose(actual: number, expected: number, relative = 1e-4, absolute = 1e-4): boolean {
  return Math.abs(actual - expected) <= absolute + relative * Math.abs(expected);
}

/** First index at which `actual` differs from `expected`, as text, or undefined. */
function describeFirstMismatch(
  actual: readonly number[],
  expected: readonly number[],
  relative = 1e-4
): string | undefined {
  for (let index = 0; index < expected.length; index++) {
    if (!isClose(actual[index], expected[index], relative)) {
      return `[${index}] got ${actual[index]} expected ${expected[index]}`;
    }
  }
  return undefined;
}

const CONTROL_AND_WINDOW_SIZES = [16, 32, 33, 36, 40, 48, 64];

it('GPUNetworkFlow: Bellman-Ford distance and predecessor tables of 16 to 128 nodes', async () => {
  const wrongSizes: string[] = [];
  for (const nodeCount of [16, 32, 33, 36, 40, 48, 64, 96, 128]) {
    const rig = await createRig();
    try {
      // A chain 0 -> 1 -> ... -> n-1 listed in REVERSE order (capacity 5, cost 1), so each
      // Bellman-Ford round relaxes one vertex, plus a direct edge 0 -> n-1 (capacity 2, cost 100).
      // Both routes saturate: flow 7, cost 5 (n - 1) + 200, whichever the solver takes first.
      const chainLength = nodeCount - 1;
      const edgePairs: number[] = [];
      const capacities: number[] = [];
      const costs: number[] = [];
      for (let edge = chainLength - 1; edge >= 0; edge--) {
        edgePairs.push(edge, edge + 1);
        capacities.push(5);
        costs.push(1);
      }
      edgePairs.push(0, nodeCount - 1);
      capacities.push(2);
      costs.push(100);
      const edgeCount = capacities.length;
      const flow = rig.output('uint32', edgeCount);
      const cut = rig.output('uint32', nodeCount);
      const summary = rig.output('uint32', 8);
      const exactCost = rig.output('uint32x2', 1);
      rig.run(
        new GPUNetworkFlow({
          nodeCount,
          edges: rig.input(Uint32Array.from(edgePairs), 'uint32x2'),
          capacities: rig.input(Uint32Array.from(capacities), 'uint32'),
          costs: rig.input(Uint32Array.from(costs), 'uint32'),
          parameters: rig.input(Uint32Array.of(0, nodeCount - 1, 0xffffffff), 'uint32'),
          output: {flow, cut, summary, exactCost}
        })
      );
      const actualFlow = await rig.readUint(flow);
      const actualSummary = await rig.readUint(summary);
      const actualCut = await rig.readUint(cut);
      const actualCost = await rig.readUint(exactCost);
      const expectedFlow = capacities.map(capacity => capacity);
      const expectedCost = 5 * chainLength + 200;
      const problems: string[] = [];
      if (actualFlow.join() !== expectedFlow.join()) {
        problems.push(`flow ${actualFlow.slice(0, 6).join(',')}... expected all saturated`);
      }
      // [valid, complete, value, remaining, augmentations, cost bits, cut capacity, certified]
      const summaryNumbers = [
        actualSummary[0],
        actualSummary[1],
        actualSummary[2],
        actualSummary[3],
        actualSummary[4],
        actualSummary[6],
        actualSummary[7]
      ];
      if (summaryNumbers.join() !== [1, 1, 7, 0, 2, 7, 1].join()) {
        problems.push(`summary ${summaryNumbers.join(',')} expected 1,1,7,0,2,7,1`);
      }
      if (actualCost[0] !== expectedCost || actualCost[1] !== 0) {
        problems.push(`exact cost ${actualCost.join(',')} expected ${expectedCost},0`);
      }
      // The residual graph reaches only the source: both routes are saturated.
      if (actualCut.join() !== [1, ...new Array(nodeCount - 1).fill(0)].join()) {
        problems.push(`cut ${actualCut.slice(0, 4).join(',')}...`);
      }
      if (problems.length > 0) wrongSizes.push(`nodes ${nodeCount}: ${problems.join('; ')}`);
    } finally {
      rig.destroy();
    }
  }
  expect(wrongSizes).toEqual([]);
});

it('addResistanceConnectivityRecipe: connected table of 16 to 256 vertices on a resistor chain', async () => {
  const wrongSizes: string[] = [];
  for (const vertexCount of [...CONTROL_AND_WINDOW_SIZES, 96, 128, 200, 256]) {
    const rig = await createRig();
    try {
      // Unit resistors in series, source at vertex 0, sink at the far end: R = n - 1, and the
      // potential falls linearly, V(i) = n - 1 - i, with unit current in every edge.
      const edgeCount = vertexCount - 1;
      const edgePairs: number[] = [];
      for (let edge = 0; edge < edgeCount; edge++) edgePairs.push(edge, edge + 1);
      const potentials = rig.output('float32', vertexCount);
      const currents = rig.output('float32', edgeCount);
      const diagnostics = rig.output('float32x4', 1);
      const status = rig.output('uint32x4', 1);
      const result = addResistanceConnectivityRecipe(rig.graph, {
        edges: rig.input(Uint32Array.from(edgePairs), 'uint32x2'),
        conductances: rig.input(new Float32Array(edgeCount).fill(1), 'float32'),
        vertexCount,
        source: 0,
        sink: vertexCount - 1,
        maximumIterations: 4096,
        output: {potentials, currents, diagnostics, status}
      });
      expect(result.status).toBe(status);
      rig.run();
      const actualPotentials = await rig.readFloat(potentials);
      const actualCurrents = await rig.readFloat(currents);
      const actualDiagnostics = await rig.readFloat(diagnostics);
      const actualStatus = await rig.readUint(status);
      const problems: string[] = [];
      const potentialMismatch = describeFirstMismatch(
        actualPotentials,
        Array.from({length: vertexCount}, (_, vertex) => vertexCount - 1 - vertex),
        1e-3
      );
      if (potentialMismatch) problems.push(`potentials ${potentialMismatch}`);
      const currentMismatch = describeFirstMismatch(
        actualCurrents,
        new Array(edgeCount).fill(1),
        1e-3
      );
      if (currentMismatch) problems.push(`currents ${currentMismatch}`);
      if (!isClose(actualDiagnostics[0], edgeCount, 1e-3)) {
        problems.push(`resistance ${actualDiagnostics[0]} expected ${edgeCount}`);
      }
      if (actualStatus[2] !== 1 || actualStatus[3] !== 0) {
        problems.push(
          `status ${actualStatus.join(',')} (reachable and invalid words expected 1,0)`
        );
      }
      if (problems.length > 0) {
        wrongSizes.push(
          `vertices ${vertexCount} (converged ${actualStatus[0]}, iterations ${actualStatus[1]}): ${problems.join('; ')}`
        );
      }
    } finally {
      rig.destroy();
    }
  }
  expect(wrongSizes).toEqual([]);
});

it('addVisibilityGraphRecipe: breadth-first distance table of 16 to 256 viewpoints on a visibility chain', async () => {
  const wrongSizes: string[] = [];
  for (const count of [...CONTROL_AND_WINDOW_SIZES, 96, 128, 200, 256]) {
    const rig = await createRig();
    try {
      // Viewpoints on a line at x = 0 .. n-1 with maximumDistance 1.1: only neighbours see each
      // other, so the visibility graph is the path on n vertices, far from the one obstacle.
      // Path graph facts: depth sum from i is the sum of |i - j|, maximum depth max(i, n-1-i),
      // betweenness i (n - 1 - i), and n - 1 visible pairs.
      const positions = new Float32Array(count * 2);
      for (let point = 0; point < count; point++) positions[point * 2] = point;
      const metrics = rig.output('float32x4', count);
      const choice = rig.output('float32', count);
      const extendedIntegration = rig.output('float32x4', count);
      const status = rig.output('uint32x4', 1);
      addVisibilityGraphRecipe(rig.graph, {
        positions: rig.input(positions, 'float32x2'),
        segments: rig.input(Float32Array.of(0, 1000, 1, 1000), 'float32x4'),
        maximumDistance: 1.1,
        edgeCapacity: count,
        output: {metrics, choice, extendedIntegration, status}
      });
      rig.run();
      const actualMetrics = await rig.readFloat(metrics);
      const actualChoice = await rig.readFloat(choice);
      const actualExtended = await rig.readFloat(extendedIntegration);
      const actualStatus = await rig.readUint(status);
      const expectedMetrics: number[] = [];
      const expectedChoice: number[] = [];
      const expectedMaximumDepth: number[] = [];
      for (let point = 0; point < count; point++) {
        let depthSum = 0;
        for (let other = 0; other < count; other++) depthSum += Math.abs(point - other);
        expectedMetrics.push(
          point > 0 && point < count - 1 ? 2 : 1,
          count - 1,
          depthSum,
          (count - 1) / depthSum
        );
        expectedChoice.push(point * (count - 1 - point));
        expectedMaximumDepth.push(Math.max(point, count - 1 - point));
      }
      const problems: string[] = [];
      const metricsMismatch = describeFirstMismatch(actualMetrics, expectedMetrics);
      if (metricsMismatch) problems.push(`metrics ${metricsMismatch}`);
      const choiceMismatch = describeFirstMismatch(actualChoice, expectedChoice);
      if (choiceMismatch) problems.push(`choice ${choiceMismatch}`);
      const depthMismatch = describeFirstMismatch(
        expectedMaximumDepth.map((_, point) => actualExtended[point * 4 + 3]),
        expectedMaximumDepth
      );
      if (depthMismatch) problems.push(`maximum depth ${depthMismatch}`);
      if (actualStatus.join() !== [count - 1, count - 1, 0, 0].join()) {
        problems.push(`status ${actualStatus.join(',')} expected ${count - 1},${count - 1},0,0`);
      }
      if (problems.length > 0) wrongSizes.push(`points ${count}: ${problems.join('; ')}`);
    } finally {
      rig.destroy();
    }
  }
  expect(wrongSizes).toEqual([]);
});

it('GPUExpectationBasedSpaceTimeScan: sorting-network order table of 16 to 64 neighbours', async () => {
  const wrongSizes: string[] = [];
  for (const neighborCount of CONTROL_AND_WINDOW_SIZES) {
    const rig = await createRig();
    try {
      // neighborCount zones on a line, one time bucket, every baseline 1, every neighbourhood
      // is the whole set. Five zones (13 j + 5 mod n) carry 10, 9, 8, 7, 6 cases, the rest 0.
      // The best subset is the elevated zones in ratio order; adding a zero zone only lowers
      // the score (F = C ln(C/B) + B - C falls as B grows beyond C / e... checked below).
      const elevatedCounts = [10, 9, 8, 7, 6];
      const elevatedZones = elevatedCounts.map((_, rank) => (13 * rank + 5) % neighborCount);
      expect(new Set(elevatedZones).size).toBe(elevatedCounts.length);
      const cases = new Uint32Array(neighborCount);
      elevatedZones.forEach((zone, rank) => (cases[zone] = elevatedCounts[rank]));
      // Independent best prefix of the ratio-sorted list, computed in doubles.
      const sortedCounts = [...Array.from(cases)].sort((first, second) => second - first);
      let bestScore = 0;
      let bestMemberCount = 0;
      let caseTotal = 0;
      sortedCounts.forEach((count, index) => {
        caseTotal += count;
        const baselineTotal = index + 1;
        const score =
          caseTotal > baselineTotal
            ? caseTotal * Math.log(caseTotal / baselineTotal) + baselineTotal - caseTotal
            : 0;
        if (score > bestScore) {
          bestScore = score;
          bestMemberCount = index + 1;
        }
      });
      expect(bestMemberCount).toBe(elevatedCounts.length);
      const positions = new Float32Array(neighborCount * 2);
      for (let zone = 0; zone < neighborCount; zone++) positions[zone * 2] = zone * 10;
      const clusterZones = rig.output('uint32', neighborCount);
      const clusterStatistics = rig.output('float32', 4);
      const statistics = rig.output('float32', 2);
      const summary = rig.output('uint32', 5);
      rig.run(
        new GPUExpectationBasedSpaceTimeScan({
          positions: rig.input(positions, 'float32x2'),
          cases: rig.input(cases, 'uint32'),
          baseline: rig.input(new Float32Array(neighborCount).fill(1), 'float32'),
          timeBuckets: 1,
          maximumNeighbors: neighborCount,
          maximumWindowLength: 1,
          maximumReplicates: 1,
          parameters: rig.input(
            getGPUExpectationBasedScanParameterValues({seed: 7, replicateCount: 1}),
            'uint32'
          ),
          output: {clusterZones, clusterStatistics, statistics, summary}
        })
      );
      const actualZones = await rig.readUint(clusterZones);
      const actualStatistics = await rig.readFloat(clusterStatistics);
      const actualSummary = await rig.readUint(summary);
      const expectedZones = [
        ...elevatedZones,
        ...new Array(neighborCount - elevatedZones.length).fill(GPU_EXPECTATION_BASED_SCAN_NO_ZONE)
      ];
      const problems: string[] = [];
      if (actualZones.join() !== expectedZones.join()) {
        problems.push(
          `clusterZones ${actualZones.slice(0, 8).join(',')}... expected ${expectedZones.slice(0, 6).join(',')}...`
        );
      }
      if (actualSummary[0] !== 1 || actualSummary[1] !== 5 || actualSummary[2] !== 1) {
        problems.push(`summary ${actualSummary.join(',')} expected found 1, members 5, window 1`);
      }
      if (
        !isClose(actualStatistics[0], bestScore, 1e-4) ||
        actualStatistics[1] !== 40 ||
        actualStatistics[2] !== 5
      ) {
        problems.push(`statistics ${actualStatistics.join(',')} expected ${bestScore},40,5`);
      }
      if (problems.length > 0)
        wrongSizes.push(`neighbours ${neighborCount}: ${problems.join('; ')}`);
    } finally {
      rig.destroy();
    }
  }
  expect(wrongSizes).toEqual([]);
});

it('GPUSpatialCrossValidation NNDM: neighbours, remaining and nearest tables of 16 to 64 rows', async () => {
  const wrongSizes: string[] = [];
  for (const rowCount of CONTROL_AND_WINDOW_SIZES) {
    // Rows at pseudo-random integer positions plus one isolated row; one prediction location
    // very far away, so the prediction distance function stays at 0 over every radius reached.
    // The algorithm then removes a row's nearest-neighbour link whenever its nearest distance is
    // at most maximumDistance, as often as minimumTrainingFraction allows. With no removals
    // allowed the radius is the untouched nearest-neighbour distance. With rowCount - 2 allowed
    // (never reached, because the isolated row is excluded) every row sheds all links within
    // maximumDistance, so its radius is its distance to the nearest other row farther than
    // maximumDistance. (A limit of a few removals per row is not used: a mutual nearest pair at
    // equal distance interacts with the "step to a strictly larger radius" rule of the
    // algorithm, which this expectation does not model; a CPU transcription of the kernel agrees
    // with the GPU there, so it is not a table effect.)
    const maximumDistance = 150.5;
    for (const unbounded of [false, true]) {
      const removals = unbounded ? rowCount - 2 : 0;
      const rig = await createRig();
      try {
        const random = createSeededRandom(1000 + rowCount);
        const positions = new Float32Array(rowCount * 2);
        for (let index = 0; index < positions.length; index++) {
          positions[index] = Math.floor(random() * 1000);
        }
        positions[(rowCount - 1) * 2] = 50000;
        positions[(rowCount - 1) * 2 + 1] = 50000;
        const expectedRadii: number[] = [];
        const sortedDistances: number[][] = [];
        for (let row = 0; row < rowCount; row++) {
          const distances: number[] = [];
          for (let other = 0; other < rowCount; other++) {
            if (other !== row) {
              distances.push(
                Math.hypot(
                  positions[row * 2] - positions[other * 2],
                  positions[row * 2 + 1] - positions[other * 2 + 1]
                )
              );
            }
          }
          distances.sort((first, second) => first - second);
          sortedDistances.push(distances);
          const withinReach = distances.filter(distance => distance <= maximumDistance).length;
          expectedRadii.push(distances[Math.min(removals, withinReach)]);
        }
        const folds = rig.output('uint32', rowCount);
        const training = rig.output('uint32', rowCount * rowCount);
        const testing = rig.output('uint32', rowCount * rowCount);
        const counts = rig.output('uint32', 3 * rowCount);
        const status = rig.output('uint32', rowCount);
        const diagnostics = rig.output('float32', 4 + rowCount);
        rig.run(
          new GPUSpatialCrossValidation({
            positions: rig.input(positions, 'float32x2'),
            blocks: rig.input(new Uint32Array(rowCount), 'uint32'),
            foldCount: rowCount,
            seed: 17,
            design: {
              mode: 'nndm',
              predictionPositions: rig.input(Float32Array.of(1e6, 1e6), 'float32x2'),
              maximumDistance,
              minimumTrainingFraction: (rowCount - removals - 0.5) / rowCount,
              diagnostics
            },
            output: {folds, training, testing, counts, status}
          })
        );
        const actualDiagnostics = await rig.readFloat(diagnostics);
        const actualFolds = await rig.readUint(folds);
        const problems: string[] = [];
        if (actualDiagnostics[0] !== 0) problems.push(`design status ${actualDiagnostics[0]}`);
        const radiusMismatch = describeFirstMismatch(
          actualDiagnostics.slice(4),
          expectedRadii,
          1e-4
        );
        if (radiusMismatch) {
          // Which sorted neighbour distance each wrong radius equals: -1 when none does.
          const wrongRows = expectedRadii
            .map((expected, row) => ({
              row,
              rank: sortedDistances[row].findIndex(distance =>
                isClose(actualDiagnostics[4 + row], distance, 1e-4)
              ),
              wanted: sortedDistances[row].findIndex(distance => isClose(expected, distance, 1e-4))
            }))
            .filter(entry => !isClose(actualDiagnostics[4 + entry.row], expectedRadii[entry.row]));
          problems.push(
            `radii ${radiusMismatch}; ${wrongRows.length} of ${rowCount} rows wrong (row:got rank/wanted rank) ${wrongRows
              .map(entry => `${entry.row}:${entry.rank}/${entry.wanted}`)
              .join(' ')}`
          );
        }
        if (actualFolds.join() !== Array.from({length: rowCount}, (_, row) => row).join()) {
          problems.push(`folds ${actualFolds.slice(0, 6).join(',')}...`);
        }
        if (problems.length > 0) {
          wrongSizes.push(`rows ${rowCount}, removals ${removals}: ${problems.join('; ')}`);
        }
      } finally {
        rig.destroy();
      }
    }
  }
  expect(wrongSizes).toEqual([]);
});
