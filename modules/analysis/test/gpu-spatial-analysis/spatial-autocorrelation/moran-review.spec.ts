// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPULocalMoran,
  GPUHotSpotAnalysis,
  getGPUSpatialAutocorrelationParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  GPUGlobalSpatialStatistics,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {
  GPUNeighborSearch,
  getGPUNeighborSearchParameterValues
} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {GPUContiguityWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {importHotSpotsWeights} from '../../../../../examples/deck/spatial-analysis-shared/hot-spots-analysis';

const VALUES = new Float32Array([1, 2, 4, 8]);
const POSITIONS = new Float32Array([0, 0, 1, 0, 2, 0, 3, 0]);
const CSR = {
  offsets: [0, 1, 3, 5, 6],
  neighbors: [1, 0, 2, 1, 3, 2],
  weights: [1, 1, 1, 1, 1, 1],
  distances: []
};

it('R1 refuses a loss flag on the second chunk instead of silently analyzing the retained line', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const candidateOverflow = rig.input(new Uint32Array([0]), 'uint32', 1);
    const weights = {
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
          weights: {
            ...rig.uploadWeights({
              offsets: [0, 2, 3],
              neighbors: [1, 3, 2],
              weights: [1, 1, 1],
              distances: []
            }),
            status: {candidateOverflow}
          }
        }
      ]
    };
    const values = rig.input(VALUES, 'float32', 4);
    const parameters = rig.input(getGPUSpatialAutocorrelationParameterValues(), 'float32', 8);
    const zScores = rig.output('float32', 4).view;
    const summary = rig.output('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length).view;
    for (const flag of [0, 1, 0]) {
      rig.bufferOf(candidateOverflow).write(new Uint32Array([flag]));
      expect(() => new GPULocalMoran({weights, values, parameters, output: {zScores}})).toThrow(
        /chunks\[1\].*status.*outer/
      );
      for (const mode of ['tiled', 'reference'] as const)
        expect(
          () =>
            new GPUGlobalSpatialStatistics({
              mode,
              weights,
              values,
              statistics: ['moran'],
              output: {summary}
            })
        ).toThrow(/chunks\[1\].*status.*outer/);
    }
  } finally {
    rig.destroy();
  }
});

it('R2 a complete radius search clears a stale candidate flag before Moran on every submission', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const weights = rig.weightsOutput(4, 6).spatialWeights;
    const overflow = rig.output('uint32', 1);
    const candidateOverflow = rig.output('uint32', 1);
    const searchParameters = getGPUNeighborSearchParameterValues({
      bounds: [-1, -1, 4, 1],
      radius: 1.1
    });
    const search = new GPUNeighborSearch({
      mode: 'radius',
      gridSize: [4, 2],
      positions: rig.input(POSITIONS, 'float32x2', 4),
      parameters: rig.input(searchParameters, 'float32', searchParameters.length),
      output: {
        ...weights,
        status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
      }
    });
    rig.graph.add(search);
    const zScores = rig.output('float32', 4);
    const pValues = rig.output('float32', 4);
    rig.graph.add(
      new GPULocalMoran({
        weights: search.outputWeights,
        values: rig.input(VALUES, 'float32', 4),
        parameters: rig.input(getGPUSpatialAutocorrelationParameterValues(), 'float32', 8),
        output: {zScores: zScores.view, pValues: pValues.view}
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (let submission = 0; submission < 2; submission++) {
        candidateOverflow.buffer.write(new Uint32Array([1]));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([0]);
        expect((await readFloat32(pValues.buffer, 4)).every(Number.isFinite)).toBe(true);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('R3 a candidate-bounded self-join supplies the candidate signal even when the caller omitted it', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const points = {
      kind: 'points' as const,
      positions: rig.input(new Float32Array(8), 'float32x2', 4)
    };
    const overall = rig.output('uint32', 1);
    const weightsOverflow = rig.output('uint32', 1);
    const join = new GPUSpatialPredicateJoin({
      left: points,
      right: points,
      predicate: 'dwithin',
      distance: 1,
      candidateCapacity: 14,
      output: {
        weights: {
          ...rig.weightsOutput(4, 16).spatialWeights,
          status: {overflow: weightsOverflow.view}
        },
        status: {overflow: overall.view}
      }
    });
    rig.graph.add(join);
    const candidateStatus = rig.output('uint32', 1);
    const statisticStatus = rig.output('uint32', 1);
    const zScores = rig.output('float32', 4);
    const pValues = rig.output('float32', 4);
    rig.run(
      new GPULocalMoran({
        weights: join.outputWeights!,
        values: rig.input(VALUES, 'float32', 4),
        parameters: rig.input(getGPUSpatialAutocorrelationParameterValues(), 'float32', 8),
        output: {
          zScores: zScores.view,
          pValues: pValues.view,
          status: {overflow: statisticStatus.view, candidateOverflow: candidateStatus.view}
        }
      })
    );
    expect(await readUint32(overall.buffer, 1)).toEqual([1]);
    expect(await readUint32(weightsOverflow.buffer, 1)).toEqual([0]);
    expect(await readUint32(statisticStatus.buffer, 1)).toEqual([0]);
    expect(await readUint32(candidateStatus.buffer, 1)).toEqual([1]);
    expect((await readFloat32(pValues.buffer, 4)).every(Number.isNaN)).toBe(true);
  } finally {
    rig.destroy();
  }
});

for (const mode of ['tiled', 'reference'] as const) {
  it(`R5 global ${mode} reads only the first word of a spare-capacity overflow view`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    try {
      const overflow = rig.input(new Uint32Array([0, 1]), 'uint32', 2);
      const summary = rig.output('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length);
      rig.graph.add(
        new GPUGlobalSpatialStatistics({
          mode,
          weights: {...rig.uploadWeights(CSR), status: {overflow}},
          values: rig.input(VALUES, 'float32', 4),
          statistics: ['moran'],
          output: {summary: summary.view}
        })
      );
      const compiled = rig.graph.compile();
      try {
        for (const words of [
          [0, 1],
          [1, 0],
          [0, 1]
        ]) {
          rig.bufferOf(overflow).write(new Uint32Array(words));
          submitGraph(device, compiled, undefined);
          const result = await readFloat32(
            summary.buffer,
            GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length
          );
          const statistic = result[GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran];
          if (words[0]) expect(Number.isNaN(statistic)).toBe(true);
          else {
            expect(Number.isFinite(statistic)).toBe(true);
            expect(statistic).toBeCloseTo(29 / 115, 5);
            expect(Number.isFinite(result[12])).toBe(true);
          }
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  });
}

for (const Contributor of [GPULocalMoran, GPUHotSpotAnalysis]) {
  it(`R4 explorer imports incomplete search weights across graphs for ${Contributor.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const capacity of [1, 6]) {
      const rig = new WeightsRig(device);
      try {
        const offsets = rig.output('uint32', 5);
        const neighbors = rig.output('uint32', capacity);
        const weights = rig.output('float32', capacity);
        const overflow = rig.output('uint32', 1);
        const searchParameters = getGPUNeighborSearchParameterValues({
          bounds: [-1, -1, 4, 1],
          radius: 1.1
        });
        rig.graph.add(
          new GPUNeighborSearch({
            mode: 'radius',
            gridSize: [4, 2],
            positions: rig.input(POSITIONS, 'float32x2', 4),
            parameters: rig.input(searchParameters, 'float32', searchParameters.length),
            output: {
              offsets: offsets.view,
              neighbors: neighbors.view,
              weights: weights.view,
              status: {overflow: overflow.view}
            }
          })
        );
        const searchCompiled = rig.graph.compile();
        const statisticRig = new WeightsRig(device);
        try {
          const imported = importHotSpotsWeights(
            statisticRig.graph,
            {
              offsets: offsets.buffer,
              neighbors: neighbors.buffer,
              weights: weights.buffer,
              overflow: overflow.buffer
            },
            4,
            capacity
          );
          const zScores = statisticRig.output('float32', 4);
          const classes = statisticRig.output('uint32', 4);
          const bins = statisticRig.graph.createDataView(classes.view.buffer, {
            format: 'sint32',
            length: 4
          });
          statisticRig.graph.add(
            new Contributor({
              weights: imported,
              values: statisticRig.input(VALUES, 'float32', 4),
              parameters: statisticRig.input(
                getGPUSpatialAutocorrelationParameterValues(),
                'float32',
                8
              ),
              output: {zScores: zScores.view, quadrants: classes.view, bins}
            })
          );
          const compiled = statisticRig.graph.compile();
          try {
            submitGraph(device, searchCompiled, undefined);
            submitGraph(device, compiled, undefined);
            const scores = await readFloat32(zScores.buffer, 4);
            if (capacity < 6) {
              expect(scores.every(Number.isNaN)).toBe(true);
              expect(await readUint32(classes.buffer, 4)).toEqual([0, 0, 0, 0]);
            } else expect(scores.every(Number.isFinite)).toBe(true);
          } finally {
            compiled.destroy();
          }
        } finally {
          statisticRig.destroy();
          searchCompiled.destroy();
        }
      } finally {
        rig.destroy();
      }
    }
  });
}

it('R6 contiguity pair loss remains visible as candidate loss in Moran without final overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const polygons = {
      kind: 'polygons' as const,
      positions: rig.input(
        new Float32Array([0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1]),
        'float32x2',
        8
      ),
      ringOffsets: rig.input(new Uint32Array([0, 4, 8]), 'uint32', 3),
      polygonOffsets: rig.input(new Uint32Array([0, 1, 2]), 'uint32', 3),
      featureOffsets: rig.input(new Uint32Array([0, 1, 2]), 'uint32', 3)
    };
    const overflow = rig.output('uint32', 1);
    const candidateOverflow = rig.output('uint32', 1);
    const producer = new GPUContiguityWeights({
      criterion: 'queen',
      polygons,
      pairCapacity: 1,
      output: {
        ...rig.weightsOutput(2, 8).spatialWeights,
        status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
      }
    });
    rig.graph.add(producer);
    const statisticOverflow = rig.output('uint32', 1);
    const statisticCandidateOverflow = rig.output('uint32', 1);
    const zScores = rig.output('float32', 2);
    const pValues = rig.output('float32', 2);
    rig.run(
      new GPULocalMoran({
        weights: producer.outputWeights,
        values: rig.input(new Float32Array([1, 2]), 'float32', 2),
        parameters: rig.input(getGPUSpatialAutocorrelationParameterValues(), 'float32', 8),
        output: {
          zScores: zScores.view,
          pValues: pValues.view,
          status: {
            overflow: statisticOverflow.view,
            candidateOverflow: statisticCandidateOverflow.view
          }
        }
      })
    );
    expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
    expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([1]);
    expect(await readUint32(statisticOverflow.buffer, 1)).toEqual([0]);
    expect(await readUint32(statisticCandidateOverflow.buffer, 1)).toEqual([1]);
    expect((await readFloat32(pValues.buffer, 2)).every(Number.isNaN)).toBe(true);
  } finally {
    rig.destroy();
  }
});
