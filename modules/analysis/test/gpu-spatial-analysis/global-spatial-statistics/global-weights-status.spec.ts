// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUGlobalSpatialStatistics,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY,
  GPU_GLOBAL_SPATIAL_STATISTIC_FIELD
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {
  CLASSIFY_BODY,
  getGlobalQuadrantCounts,
  getGlobalQuadrantLegend,
  getGlobalJoinTooltip,
  getGlobalMoranQuadrant,
  getGlobalObservedJoinCounts
} from '../../../../../examples/deck/spatial-analysis-showcase/src/scenes/weights/global-autocorrelation-display';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-lag';
import {GPUChunkedSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose';
import {GPUContiguityWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights';
import {
  GPUNeighborSearch,
  getGPUNeighborSearchParameterValues
} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {importWeightsCoreWeights} from '../../../../../examples/deck/spatial-analysis-showcase/src/scenes/weights/b4-weights-imports';
import {importWeightsKitLoss} from '../../../../../examples/deck/spatial-analysis-explorer/modes/spatial-weights-imports';

const LAYOUT = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT;
const FIELD = GPU_GLOBAL_SPATIAL_STATISTIC_FIELD;

it('final2 suppressed inference displays no quadrant or observed join counts and recovers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const candidate = rig.input(Uint32Array.of(0), 'uint32', 1);
    const weights = {
      ...rig.uploadWeights({offsets: [0, 1, 2], neighbors: [1, 0], weights: [1, 1], distances: []}),
      status: {candidateOverflow: candidate}
    };
    const values = rig.input(Float32Array.of(1, 4), 'float32', 2);
    const mask = rig.input(Uint32Array.of(1, 1), 'uint32', 2);
    const summary = rig.output('float32', LAYOUT.length);
    const lag = rig.output('float32', 2);
    const quadrant = rig.output('uint32', 2);
    const joinClass = rig.output('uint32', 2);
    const binary = rig.output('float32', 2);
    const joinCounts = rig.output('uint32', 3);
    const joinSummary = rig.output('float32', LAYOUT.length);
    rig.graph.add(
      new GPUGlobalSpatialStatistics({
        weights,
        values,
        statistics: ['moran'],
        output: {summary: summary.view}
      })
    );
    rig.graph.add(new GPUSpatialLag({weights, values, normalize: true, output: lag.view}));
    rig.graph.add(
      createWGSLKernelNode(rig.graph, {
        id: 'showcase-classification',
        operation: 'DisplayTest',
        invocationCount: 2,
        bindings: [
          {name: 'x', view: values, type: 'f32', access: 'read'},
          {name: 'lag', view: lag.view, type: 'f32', access: 'read'},
          {name: 'mask', view: mask, type: 'u32', access: 'read'},
          {name: 'summary', view: summary.view, type: 'f32', access: 'read'},
          {
            name: 'threshold',
            view: rig.input(Float32Array.of(2.5), 'float32', 1),
            type: 'f32',
            access: 'read'
          },
          {name: 'quadrant', view: quadrant.view, type: 'u32', access: 'read_write'},
          {name: 'joinClass', view: joinClass.view, type: 'u32', access: 'read_write'},
          {name: 'binary', view: binary.view, type: 'f32', access: 'read_write'}
        ],
        body: CLASSIFY_BODY
      })
    );
    rig.graph.add(
      new GPUGlobalSpatialStatistics({
        id: 'display-joins',
        weights,
        values: binary.view,
        statistics: ['joinCount'],
        output: {summary: joinSummary.view, joinCounts: joinCounts.view}
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const flag of [0, 1, 0]) {
        rig.bufferOf(candidate).write(Uint32Array.of(flag));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(quadrant.buffer, 2)).toEqual(flag ? [0, 0] : [2, 4]);
        const colors = {
          highHigh: [1, 0, 0, 255] as const,
          lowHigh: [2, 0, 0, 255] as const,
          lowLow: [3, 0, 0, 255] as const,
          highLow: [4, 0, 0, 255] as const
        };
        const globals = await readFloat32(summary.buffer, LAYOUT.length);
        const islands = globals[LAYOUT.summary + GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY.islandCount];
        expect(islands).toBe(0);
        const legend = getGlobalQuadrantLegend(
          getGlobalQuadrantCounts(await readUint32(quadrant.buffer, 2), islands),
          colors,
          [0, 0, 0, 0]
        );
        expect(legend.entries.find(entry => entry.label === 'Unavailable quadrant')?.count).toBe(
          flag ? 2 : 0
        );
        expect(legend.entries.find(entry => entry.label === 'Low-High')?.count).toBe(flag ? 0 : 1);
        expect(legend.entries.find(entry => entry.label === 'High-Low')?.count).toBe(flag ? 0 : 1);
        expect(legend.note).toContain('Islands: 0');
        const joinClasses = await readUint32(joinClass.buffer, 2);
        expect(
          getGlobalJoinTooltip(joinClasses[0], !flag, [0, 0, 0, 255], [255, 255, 255, 255]).value
        ).toBe(flag ? 'Unavailable (incomplete weights)' : 'White');
        expect(
          getGlobalJoinTooltip(joinClasses[1], !flag, [0, 0, 0, 255], [255, 255, 255, 255]).value
        ).toBe(flag ? 'Unavailable (incomplete weights)' : 'Black');
        expect(await readUint32(joinClass.buffer, 2)).toEqual(
          flag ? [0xffffffff, 0xffffffff] : [0, 1]
        );
        const observed = getGlobalObservedJoinCounts(
          flag === 0,
          await readUint32(joinCounts.buffer, 3)
        );
        expect(observed).toEqual(flag ? undefined : [0, 1, 0]);
        expect(
          getGlobalMoranQuadrant(1, flag ? Number.NaN : 4, flag ? Number.NaN : 2.5, !flag)
        ).toBe(flag ? 0 : 2);
        expect(
          getGlobalMoranQuadrant(4, flag ? Number.NaN : 1, flag ? Number.NaN : 2.5, !flag)
        ).toBe(flag ? 0 : 4);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('final1 transpose cannot erase a chunk-local candidate signal before global inference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const candidate = rig.input(new Uint32Array([0]), 'uint32', 1);
    const input = {
      rowCount: 2,
      chunks: [
        {
          rowStart: 0,
          rowCount: 2,
          weights: {
            ...rig.uploadWeights({
              offsets: [0, 1, 2],
              neighbors: [1, 0],
              weights: [1, 1],
              distances: []
            }),
            status: {candidateOverflow: candidate}
          }
        }
      ]
    };
    const output = rig.weightsOutput(2, 2).spatialWeights;
    for (const flag of [0, 1, 0]) {
      rig.bufferOf(candidate).write(new Uint32Array([flag]));
      expect(() => new GPUChunkedSpatialWeightsTranspose({weights: input, output})).toThrow(
        /chunk.*status.*outer/
      );
    }
    // No packed result is published for a port whose chunk completeness has not been aggregated.
  } finally {
    rig.destroy();
  }
});

it('final1 showcase importer suppresses contiguity candidate loss and recovers after switching to search', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const resources = new WeightsRig(device);
  const contiguityRig = new WeightsRig(device);
  const searchRig = new WeightsRig(device);
  const statisticRig = new WeightsRig(device);
  try {
    const offsets = resources.output('uint32', 5);
    const neighbors = resources.output('uint32', 12);
    const weights = resources.output('float32', 12);
    const overflow = resources.output('uint32', 1);
    const candidate = resources.output('uint32', 1);
    const core = {
      csr: {
        offsets: offsets.buffer,
        neighbors: neighbors.buffer,
        weights: weights.buffer,
        distances: weights.buffer
      },
      slots: 12,
      overflow: overflow.buffer,
      candidateOverflow: candidate.buffer
    };
    const vertices = new Float32Array(32);
    for (let row = 0; row < 4; row++)
      vertices.set([row, 0, row + 1, 0, row + 1, 1, row, 1], row * 8);
    contiguityRig.graph.add(
      new GPUContiguityWeights({
        criterion: 'queen',
        pairCapacity: 1,
        polygons: contiguityRig.polygons({
          positions: vertices,
          ringOffsets: new Uint32Array([0, 4, 8, 12, 16]),
          polygonOffsets: new Uint32Array([0, 1, 2, 3, 4]),
          featureOffsets: new Uint32Array([0, 1, 2, 3, 4])
        }),
        output: importWeightsCoreWeights(contiguityRig.graph, core, 4)
      })
    );
    const settings = getGPUNeighborSearchParameterValues({bounds: [-1, -1, 4, 1], radius: 1.1});
    searchRig.graph.add(
      new GPUNeighborSearch({
        mode: 'radius',
        gridSize: [4, 2],
        positions: searchRig.input(new Float32Array([0, 0, 1, 0, 2, 0, 3, 0]), 'float32x2', 4),
        parameters: searchRig.input(settings, 'float32', settings.length),
        output: importWeightsCoreWeights(searchRig.graph, core, 4)
      })
    );
    const summary = statisticRig.output('float32', LAYOUT.length);
    const kitSummary = statisticRig.output('float32', LAYOUT.length);
    const statisticWeights = importWeightsCoreWeights(statisticRig.graph, core, 4);
    statisticRig.graph.add(
      new GPUGlobalSpatialStatistics({
        weights: statisticWeights,
        values: statisticRig.input(new Float32Array([1, 2, 4, 8]), 'float32', 4),
        statistics: ['moran'],
        output: {summary: summary.view}
      })
    );
    statisticRig.graph.add(
      new GPUGlobalSpatialStatistics({
        id: 'explorer-kit-statistic',
        weights: {
          ...statisticWeights,
          status: importWeightsKitLoss(statisticRig.graph, 'kit', {
            overflow: overflow.buffer,
            candidateOverflow: candidate.buffer
          })
        },
        values: statisticRig.input(new Float32Array([1, 2, 4, 8]), 'float32', 4),
        statistics: ['moran'],
        output: {summary: kitSummary.view}
      })
    );
    const contiguity = contiguityRig.graph.compile();
    const search = searchRig.graph.compile();
    const statistic = statisticRig.graph.compile();
    try {
      for (const incomplete of [true, false, true, false]) {
        submitGraph(device, incomplete ? contiguity : search, undefined);
        submitGraph(device, statistic, undefined);
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(candidate.buffer, 1)).toEqual([Number(incomplete)]);
        const result = await readFloat32(summary.buffer, LAYOUT.length);
        const kitResult = await readFloat32(kitSummary.buffer, LAYOUT.length);
        if (incomplete) {
          expect(result.slice(LAYOUT.moran).every(Number.isNaN)).toBe(true);
          expect(kitResult.slice(LAYOUT.moran).every(Number.isNaN)).toBe(true);
        } else {
          expect(Number.isFinite(result[LAYOUT.moran])).toBe(true);
          expect(result[LAYOUT.moran]).toBeCloseTo(29 / 115, 5);
          expect(Number.isFinite(kitResult[LAYOUT.moran])).toBe(true);
          expect(kitResult[LAYOUT.moran]).toBeCloseTo(29 / 115, 5);
        }
      }
    } finally {
      contiguity.destroy();
      search.destroy();
      statistic.destroy();
    }
  } finally {
    statisticRig.destroy();
    searchRig.destroy();
    contiguityRig.destroy();
    resources.destroy();
  }
});

for (const mode of ['tiled', 'reference'] as const) {
  for (const chunked of [false, true]) {
    for (const publishStatus of [false, true]) {
      it(`global ${mode} ${chunked ? 'chunked' : 'packed'} inference suppresses weights loss and recovers, status ${publishStatus}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const rig = new WeightsRig(device);
        try {
          const weights = chunked
            ? {
                rowCount: 4,
                chunks: [
                  {
                    rowStart: 0,
                    rowCount: 2,
                    weights: rig.uploadWeights({
                      offsets: [0, 1, 3],
                      neighbors: [1, 0, 2],
                      weights: [1, 1, 1],
                      distances: []
                    })
                  },
                  {
                    rowStart: 2,
                    rowCount: 2,
                    weights: rig.uploadWeights({
                      offsets: [0, 2, 3],
                      neighbors: [1, 3, 2],
                      weights: [1, 1, 1],
                      distances: []
                    })
                  }
                ]
              }
            : rig.uploadWeights({
                offsets: [0, 1, 3, 5, 6],
                neighbors: [1, 0, 2, 1, 3, 2],
                weights: [1, 1, 1, 1, 1, 1],
                distances: []
              });
          const overflow = rig.input(new Uint32Array([0]), 'uint32', 1);
          const candidateOverflow = rig.input(new Uint32Array([0]), 'uint32', 1);
          const outputOverflow = rig.output('uint32', 1);
          const outputCandidateOverflow = rig.output('uint32', 1);
          const summary = rig.output('float32', LAYOUT.length);
          const counts = rig.output('uint32', 3);
          const joinCounts = rig.output('uint32', 3);
          rig.graph.add(
            new GPUGlobalSpatialStatistics({
              mode,
              weights: {...weights, status: {overflow, candidateOverflow}},
              values: rig.input(new Float32Array([1, 2, 4, 8]), 'float32', 4),
              secondValues: rig.input(new Float32Array([2, 0, 5, 9]), 'float32', 4),
              statistics: ['moran', 'geary', 'getisOrdG', 'bivariateMoran', 'joinCount'],
              output: {
                summary: summary.view,
                counts: counts.view,
                joinCounts: joinCounts.view,
                status: publishStatus
                  ? {overflow: outputOverflow.view, candidateOverflow: outputCandidateOverflow.view}
                  : undefined
              }
            })
          );
          const compiled = rig.graph.compile();
          try {
            let completeBits: number[] | undefined;
            for (const [finalLoss, candidateLoss] of [
              [0, 0],
              [0, 7],
              [0, 0],
              [3, 0],
              [0, 0],
              [1, 1],
              [0, 0]
            ]) {
              rig.bufferOf(overflow).write(new Uint32Array([finalLoss]));
              rig.bufferOf(candidateOverflow).write(new Uint32Array([candidateLoss]));
              // Poison outputs before every submission to catch missing writes and sticky status.
              summary.buffer.write(new Uint32Array(LAYOUT.length).fill(0x7f7f7f7f));
              counts.buffer.write(new Uint32Array(3).fill(0xffffffff));
              joinCounts.buffer.write(new Uint32Array(3).fill(0xffffffff));
              outputOverflow.buffer.write(new Uint32Array([0xffffffff]));
              outputCandidateOverflow.buffer.write(new Uint32Array([0xffffffff]));
              submitGraph(device, compiled, undefined);
              const result = await readFloat32(summary.buffer, LAYOUT.length);
              expect(await readUint32(counts.buffer, 3)).toEqual([4, 0, 4]);
              if (publishStatus) {
                expect(await readUint32(outputOverflow.buffer, 1)).toEqual([
                  Number(finalLoss !== 0)
                ]);
                expect(await readUint32(outputCandidateOverflow.buffer, 1)).toEqual([
                  Number(candidateLoss !== 0)
                ]);
              }
              if (finalLoss || candidateLoss) {
                expect(result.slice(LAYOUT.moran).every(Number.isNaN)).toBe(true);
                expect(result.slice(1, 6).every(Number.isNaN)).toBe(true);
                expect(await readUint32(joinCounts.buffer, 3)).toEqual([0, 0, 0]);
              } else {
                // Four-point line: n=4, S0=6, centered sum of squares=115/4;
                // directed cross product=87/8, directed squared differences=42.
                expect(Number.isFinite(result[LAYOUT.moran + FIELD.statistic])).toBe(true);
                expect(result[LAYOUT.moran + FIELD.statistic]).toBeCloseTo(29 / 115, 5);
                expect(Number.isFinite(result[LAYOUT.geary + FIELD.statistic])).toBe(true);
                expect(result[LAYOUT.geary + FIELD.statistic]).toBeCloseTo(42 / 115, 5);
                expect(Number.isFinite(result[LAYOUT.moran + FIELD.pNormality])).toBe(true);
                expect(await readUint32(joinCounts.buffer, 3)).toEqual([6, 0, 0]);
                const bits = await readUint32(summary.buffer, LAYOUT.length);
                if (completeBits) expect(bits).toEqual(completeBits);
                completeBits = bits;
              }
              expect(result[0]).toBe(4);
              expect(result[6]).toBe(4);
              expect(result[7]).toBe(0);
            }
          } finally {
            compiled.destroy();
          }
        } finally {
          rig.destroy();
        }
      });
    }
  }
}
