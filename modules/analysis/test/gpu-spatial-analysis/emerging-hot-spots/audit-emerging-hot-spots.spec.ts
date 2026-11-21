// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {it} from 'vitest';
import {GPUEmergingHotSpots} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/gpu-emerging-hot-spots';
import {
  getGPUEmergingHotSpotsParameterValues,
  GPU_EMERGING_HOT_SPOTS_CATEGORIES
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/emerging-hot-spot-parameters';
import type {
  GPUSpatialWeights,
  GPUChunkedSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {createInputBuffer, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  AUDIT_EMERGING,
  type AuditEmergingCube,
  type AuditEmergingTrendOutputs
} from './audit-emerging-hot-spots-fixture';
import {
  ACCUMULATION_FIXTURE_CASES,
  createAccumulationFixture,
  createDenseCoefficientFixture,
  createMaskedCoefficientFixture,
  createTemporalScaleFixture,
  expectFocalCoefficientReadbacks,
  TEMPORAL_SCALE_FIXTURE_SCORE
} from './emerging-coefficient-review-fixture';

function createOutputs(rig: AnalysisRig, cells = 8, slices = 4) {
  return {
    giZScores: rig.output('float32', cells * slices),
    pValues: rig.output('float32', cells * slices),
    adjustedPValues: rig.output('float32', cells * slices),
    trendZ: rig.output('float32', cells),
    trendP: rig.output('float32', cells),
    trendS: rig.output('sint32', cells),
    category: rig.output('uint32', cells),
    hotSliceCount: rig.output('uint32', cells),
    coldSliceCount: rig.output('uint32', cells),
    lastValidSlice: rig.output('uint32', cells),
    globalStatistics: rig.output('float32', 4),
    includedCount: rig.output('uint32', 1)
  };
}

it('retains near-uniform weighted Gi* scores and suppresses the exactly uniform control', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const coefficientScale of [1, 2 ** -66])
    for (const selfCoefficient of [1 + 2 ** -12, 1]) {
      const selfWeight = selfCoefficient * coefficientScale;
      const rig = new AnalysisRig(device);
      try {
        const output = createOutputs(rig, 8, 1);
        const neighbors = Array.from({length: 8}, (_, row) =>
          Array.from({length: 8}, (_, neighbor) => neighbor).filter(neighbor => neighbor !== row)
        ).flat();
        rig.run(
          new GPUEmergingHotSpots({
            values: rig.input(Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7]), 'float32'),
            weights: rig.weights(
              {
                offsets: [0, 7, 14, 21, 28, 35, 42, 49, 56],
                neighbors,
                weights: new Array<number>(56).fill(coefficientScale)
              },
              0
            ),
            selfWeight,
            sliceCount: 1,
            multipleTesting: 'none',
            parameters: rig.input(Float32Array.of(0, 0, 0.75, 0.05, 0.9, 1, 0.05, 0), 'float32'),
            output
          })
        );
        const uniform = selfCoefficient === 1;
        const scores = [
          -1.5275252316519468, -1.091089451179962, -0.6546536707079772, -0.2182178902359924,
          0.2182178902359924, 0.6546536707079772, 1.091089451179962, 1.5275252316519468
        ];
        const probabilities = [
          0.12663045794761713, 0.27523352407483426, 0.5126907602619233, 0.8272593465627113,
          0.8272593465627113, 0.5126907602619233, 0.27523352407483426, 0.12663045794761713
        ];
        const absent = new Array<number>(8).fill(NaN);
        expectNumericArrayClose(await rig.readFloat(output.giZScores), uniform ? absent : scores, {
          absoluteTolerance: 2e-6,
          relativeTolerance: 2e-5
        });
        for (const port of ['pValues', 'adjustedPValues'] as const)
          expectNumericArrayClose(
            await rig.readFloat(output[port]),
            uniform ? absent : probabilities,
            {absoluteTolerance: 2e-6}
          );
        compareTrend(
          await readTrend(rig, output),
          {
            trendS: new Array<number>(8).fill(0),
            trendZ: uniform ? absent : new Array<number>(8).fill(0),
            trendP: uniform ? absent : new Array<number>(8).fill(1),
            category: uniform ? new Array<number>(8).fill(0) : [9, 9, 0, 0, 0, 0, 1, 1],
            hotSliceCount: uniform ? new Array<number>(8).fill(0) : [0, 0, 0, 0, 0, 0, 1, 1],
            coldSliceCount: uniform ? new Array<number>(8).fill(0) : [1, 1, 0, 0, 0, 0, 0, 0],
            lastValidSlice: new Array<number>(8).fill(uniform ? 0xffffffff : 0)
          },
          'near-uniform coefficients'
        );
        expectNumericArrayClose(
          await rig.readFloat(output.globalStatistics),
          [8, 3.5, 5.25, 2.29128784747792],
          {absoluteTolerance: 1e-6}
        );
        expectNumericArrayClose(await rig.readUint(output.includedCount), [8]);
      } finally {
        rig.destroy();
      }
    }
});

it('keeps a significant cold bin in a dense near-uniform eight-thousand-cell row', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const excessAt of ['self', 'first', 'relabeled'] as const) {
    const rig = new AnalysisRig(device);
    try {
      const fixture = createDenseCoefficientFixture(excessAt);
      const output = createOutputs(rig, 8000, 1);
      rig.run(
        new GPUEmergingHotSpots({
          values: rig.input(fixture.values, 'float32'),
          weights: rig.weights(
            {
              offsets: [0, ...new Array<number>(8000).fill(7999)],
              neighbors: fixture.neighbors,
              weights: fixture.weights
            },
            0
          ),
          selfWeight: fixture.selfWeight,
          sliceCount: 1,
          multipleTesting: 'none',
          parameters: rig.input(
            Float32Array.of(0, 0, 1.959963984540054, 0.05, 0.9, 1, 0.05, 0),
            'float32'
          ),
          output
        })
      );
      // Unit coefficients cancel over the complete population: focal standardized value is -2.
      expectFocalCoefficientReadbacks(
        {
          scores: await rig.readFloat(output.giZScores),
          probabilities: await rig.readFloat(output.pValues),
          coldCounts: await rig.readUint(output.coldSliceCount)
        },
        8000
      );
    } finally {
      rig.destroy();
    }
  }
});

for (const fixture of ACCUMULATION_FIXTURE_CASES) {
  it(fixture.name, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const {cellCount, sliceCount, confidenceLevel, hotSliceCount} = fixture;
      const {values, neighbors, weights, selfWeight} = createAccumulationFixture(fixture);
      const output = createOutputs(rig, cellCount, sliceCount);
      const offsets = [0, ...new Array<number>(cellCount).fill(neighbors.length)];
      rig.run(
        new GPUEmergingHotSpots({
          values: rig.input(values, 'float32'),
          weights: rig.weights({offsets, neighbors, weights}, 0),
          selfWeight,
          sliceCount,
          multipleTesting: 'none',
          parameters: rig.input(
            getGPUEmergingHotSpotsParameterValues({
              temporalWindow: sliceCount - 1,
              confidenceLevel,
              multipleTesting: 'none',
              minimumTrendSliceCount: 1
            }),
            'float32'
          ),
          output
        })
      );
      const scores = await rig.readFloat(output.giZScores);
      const probabilities = await rig.readFloat(output.pValues);
      expectNumericArrayClose(
        [scores.length, probabilities.length],
        [cellCount * sliceCount, cellCount * sliceCount]
      );
      expectNumericArrayClose(scores.slice(0, sliceCount), fixture.scores, {
        absoluteTolerance: 5e-5
      });
      // Two-sided normal tails computed independently with Python math.erfc.
      expectNumericArrayClose(probabilities.slice(0, sliceCount), fixture.probabilities, {
        absoluteTolerance: 1e-5
      });
      const hotCounts = await rig.readUint(output.hotSliceCount);
      const coldCounts = await rig.readUint(output.coldSliceCount);
      const categories = await rig.readUint(output.category);
      expectNumericArrayClose(
        [hotCounts[0], coldCounts[0], categories[0]],
        [
          hotSliceCount,
          0,
          hotSliceCount === 1
            ? GPU_EMERGING_HOT_SPOTS_CATEGORIES.NEW_HOT
            : GPU_EMERGING_HOT_SPOTS_CATEGORIES.NO_PATTERN
        ]
      );
    } finally {
      rig.destroy();
    }
  });
}

it('B3 masked coefficients preserve a cold bin under ascending-CSR relabeling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const relabeled of [false, true]) {
    const rig = new AnalysisRig(device);
    try {
      const fixture = createMaskedCoefficientFixture(relabeled);
      const output = createOutputs(rig, 257, 1);
      rig.run(
        new GPUEmergingHotSpots({
          values: rig.input(fixture.values, 'float32'),
          mask: rig.input(fixture.mask, 'uint32'),
          weights: rig.weights(
            {
              offsets: [0, ...new Array<number>(257).fill(256)],
              neighbors: fixture.neighbors,
              weights: fixture.weights
            },
            0
          ),
          selfWeight: 1,
          sliceCount: 1,
          multipleTesting: 'none',
          parameters: rig.input(
            Float32Array.of(0, 0, 1.959963984540054, 0.05, 0.9, 1, 0.05, 0),
            'float32'
          ),
          output
        })
      );
      // mean=0, variance=7; the excess coefficient at -7 gives z=-sqrt(7).
      // Python math.erfc(sqrt(7/2)) = .0081509715935027.
      expectFocalCoefficientReadbacks(
        {
          scores: await rig.readFloat(output.giZScores),
          probabilities: await rig.readFloat(output.pValues),
          coldCounts: await rig.readUint(output.coldSliceCount)
        },
        257,
        -2.6457513110645907,
        0.0081509715935027
      );
    } finally {
      rig.destroy();
    }
  }
});

it('B3 temporal Gi* merges different usable coefficient origins', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const output = createOutputs(rig, 5, 2);
    rig.run(
      new GPUEmergingHotSpots({
        values: rig.input(Float32Array.of(1, 2, 3, NaN, 4, NaN, NaN, 5, NaN, 6), 'float32'),
        weights: rig.weights(
          {offsets: [0, 4, 4, 4, 4, 4], neighbors: [1, 2, 3, 4], weights: [0.5, 0.5, 1, 1]},
          0
        ),
        sliceCount: 2,
        selfWeight: 1,
        multipleTesting: 'none',
        parameters: rig.input(
          Float32Array.of(0, 1, 1.959963984540054, 0.05, 0.9, 1, 0.05, 0),
          'float32'
        ),
        output
      })
    );
    const scores = await rig.readFloat(output.giZScores);
    const probabilities = await rig.readFloat(output.pValues);
    expectNumericArrayClose([scores.length, probabilities.length], [10, 10]);
    // Valid observations are 1..6. The two half-weight values, 3 and 4, cancel about 3.5.
    expectNumericArrayClose([scores[1]], [0], {absoluteTolerance: 1e-6});
    expectNumericArrayClose([probabilities[1]], [1], {absoluteTolerance: 1e-6});
  } finally {
    rig.destroy();
  }
});

it('merges temporal Gi* in one coefficient scale across a missing large-weight bin', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scale of [1, 2 ** -66, 2 ** 126]) {
    const rig = new AnalysisRig(device);
    try {
      const fixture = createTemporalScaleFixture(scale);
      const output = createOutputs(rig, 3, 2);
      rig.run(
        new GPUEmergingHotSpots({
          values: rig.input(fixture.values, 'float32'),
          weights: rig.weights(
            {offsets: [0, 2, 2, 2], neighbors: fixture.neighbors, weights: fixture.weights},
            0
          ),
          selfWeight: fixture.selfWeight,
          sliceCount: 2,
          multipleTesting: 'none',
          parameters: rig.input(
            Float32Array.of(0, 1, 1.959963984540054, 0.05, 0.9, 1, 0.05, 0),
            'float32'
          ),
          output
        })
      );
      const scores = await rig.readFloat(output.giZScores);
      const probabilities = await rig.readFloat(output.pValues);
      // Closed-form Gi* over the five observed bins, with the missing sixth bin explicitly NaN.
      expectNumericArrayClose(
        scores,
        [
          0.5669467095138409,
          TEMPORAL_SCALE_FIXTURE_SCORE,
          -0.7071067811865475,
          0,
          1.414213562373095,
          NaN
        ],
        {absoluteTolerance: 2e-6}
      );
      // Python math.erfc(abs(z) / sqrt(2)) for each of the closed-form scores.
      expectNumericArrayClose(
        probabilities,
        [0.570750388058174, 0.30169958247834805, 0.47950012218695354, 1, 0.15729920705028516, NaN],
        {absoluteTolerance: 2e-6}
      );
    } finally {
      rig.destroy();
    }
  }
});

function createWeights(
  rig: AnalysisRig,
  chunked: boolean
): GPUSpatialWeights | GPUChunkedSpatialWeights {
  const source = AUDIT_EMERGING.weights;
  if (!chunked) return rig.weights(source, 0);
  // Uneven independent row ranges retain global neighbor row identities.
  return {
    rowCount: 8,
    chunks: [
      [0, 3],
      [3, 5]
    ].map(([rowStart, rowCount]) => {
      const first = source.offsets[rowStart];
      const last = source.offsets[rowStart + rowCount];
      return {
        rowStart,
        rowCount,
        weights: rig.weights(
          {
            offsets: source.offsets
              .slice(rowStart, rowStart + rowCount + 1)
              .map(offset => offset - first),
            neighbors: source.neighbors.slice(first, last),
            weights: source.weights.slice(first, last)
          },
          0
        )
      };
    })
  };
}

async function readTrend(
  rig: AnalysisRig,
  output: ReturnType<typeof createOutputs>
): Promise<AuditEmergingTrendOutputs> {
  // Read the signed statistic's bits through the rig, then interpret them as int32 rather than float32.
  const statistic = await rig.readUint(output.trendS as unknown as GraphDataView<'uint32'>);
  return {
    trendS: statistic.map(value => value | 0),
    trendZ: await rig.readFloat(output.trendZ),
    trendP: await rig.readFloat(output.trendP),
    category: await rig.readUint(output.category),
    hotSliceCount: await rig.readUint(output.hotSliceCount),
    coldSliceCount: await rig.readUint(output.coldSliceCount),
    lastValidSlice: await rig.readUint(output.lastValidSlice)
  };
}

function compareTrend(
  actual: AuditEmergingTrendOutputs,
  expected: AuditEmergingTrendOutputs,
  label: string
) {
  for (const port of [
    'trendS',
    'trendZ',
    'trendP',
    'category',
    'hotSliceCount',
    'coldSliceCount',
    'lastValidSlice'
  ] as const) {
    // Fixed small samples: f32 normal tails and corrected variance allow 0.1% relative p error.
    // Pure relative tail checks reject a zero substituted for a representable small probability.
    expectNumericArrayClose(actual[port], expected[port], {
      absoluteTolerance: port === 'trendZ' ? 3e-5 : 0,
      relativeTolerance: port === 'trendP' ? 1e-3 : port === 'trendZ' ? 5e-5 : 0,
      label: `${label}, ${port}`
    });
  }
}

async function compareCube(
  rig: AnalysisRig,
  output: ReturnType<typeof createOutputs>,
  fixture: AuditEmergingCube,
  suppressed = false
) {
  const cells = 8;
  const expected = fixture.expected;
  const absentBins = new Array<number>(32).fill(NaN);
  expectNumericArrayClose(
    await rig.readFloat(output.giZScores),
    suppressed ? absentBins : expected.giZScores,
    {absoluteTolerance: 2e-5, relativeTolerance: 5e-5, label: `${fixture.name}, giZScores`}
  );
  for (const port of ['pValues', 'adjustedPValues'] as const)
    expectNumericArrayClose(
      await rig.readFloat(output[port]),
      suppressed ? absentBins : expected[port],
      {absoluteTolerance: 1e-7, relativeTolerance: 5e-4, label: `${fixture.name}, ${port}`}
    );
  compareTrend(
    await readTrend(rig, output),
    suppressed
      ? {
          trendS: new Array<number>(cells).fill(0),
          trendZ: new Array<number>(cells).fill(NaN),
          trendP: new Array<number>(cells).fill(NaN),
          category: new Array<number>(cells).fill(17),
          hotSliceCount: new Array<number>(cells).fill(0),
          coldSliceCount: new Array<number>(cells).fill(0),
          lastValidSlice: new Array<number>(cells).fill(0xffffffff)
        }
      : expected,
    fixture.name
  );
  // Input moments/count do not depend on whether neighborhood edges were retained.
  expectNumericArrayClose(await rig.readUint(output.includedCount), expected.includedCount);
  expectNumericArrayClose(await rig.readFloat(output.globalStatistics), expected.globalStatistics, {
    absoluteTolerance: (value, index) =>
      index === 0 || value === 0
        ? 0
        : index === 1
          ? Math.max(1e-6, Math.abs(value) * 2 ** -23)
          : Math.max(1e-6, Math.abs(value) * 2e-5),
    label: `${fixture.name}, globalStatistics`
  });
}

for (const fixture of AUDIT_EMERGING.cubes) {
  it(`audits every emerging-hot-spot output against independent SciPy literals: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const chunked of fixture.mode === 'weights' ? [false, true] : [false]) {
      const rig = new AnalysisRig(device);
      try {
        const output = createOutputs(rig);
        const contributor = new GPUEmergingHotSpots({
          values: fixture.unsigned
            ? rig.input(Uint32Array.from(fixture.values), 'uint32')
            : rig.input(Float32Array.from(fixture.values), 'float32'),
          ...(fixture.mode === 'weights'
            ? {weights: createWeights(rig, chunked), selfWeight: fixture.selfWeight}
            : {gridWidth: 4, gridHeight: 2}),
          sliceCount: 4,
          maximumRadius: 1,
          multipleTesting: fixture.method,
          finalSlice: fixture.finalSlice,
          trendAutocorrelationCorrection: fixture.correction,
          trendCorrectionMaximumLag: fixture.maximumLag,
          trendCorrectionCriticalZ: fixture.correctionCriticalZ,
          mask: rig.input(Uint32Array.from(fixture.mask), 'uint32'),
          parameters: rig.input(Float32Array.from(fixture.parameters), 'float32'),
          output
        });
        rig.run(contributor);
        await compareCube(rig, output, fixture);
      } finally {
        rig.destroy();
      }
    }
  });
}

// Corrected trend fixtures include [9,8,...,-2] at row 5: Sen slope -1 leaves a constant signal.
// Real Metal mean-division rounding previously invented autocorrelation in all four methods/lag
// cases. The independent expected z remains -4.45721562860432; no tolerance was widened.
for (const fixture of AUDIT_EMERGING.stages) {
  it(`audits direct Gi* trend/category inputs on the real shader path: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const cells = fixture.scores.length / fixture.sliceCount;
    try {
      const output = createOutputs(rig, cells, fixture.sliceCount);
      const producer = new GPUEmergingHotSpots({
        values: rig.input(new Float32Array(fixture.scores.length), 'float32'),
        gridWidth: cells,
        gridHeight: 1,
        sliceCount: fixture.sliceCount,
        multipleTesting: 'none',
        finalSlice: fixture.finalSlice,
        trendAutocorrelationCorrection: fixture.correction,
        trendCorrectionMaximumLag: fixture.maximumLag || undefined,
        trendCorrectionCriticalZ: fixture.correctionCriticalZ,
        parameters: rig.input(Float32Array.from(fixture.parameters), 'float32'),
        output: {...output, giZScores: rig.input(Float32Array.from(fixture.scores), 'float32')}
      });
      const nodes = producer
        .getCommandNodes(rig.graph)
        .filter(node => node.id.endsWith('-slice-counts') || node.id.endsWith('-mann-kendall'));
      for (const node of nodes) rig.graph.add(node);
      const compiled = rig.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        compareTrend(await readTrend(rig, output), fixture.expected, fixture.name);
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  });
}

for (const minimumTrendSliceCount of [0, 2])
  for (const chunked of [false, true])
    for (const flag of ['overflow', 'candidateOverflow'] as const)
      for (const chunkFlag of chunked ? [false, true] : [false]) {
        it(`suppresses and restores all dependent outputs per frame: chunks=${chunked}, ${flag}, nested=${chunkFlag}, minimum trend=${minimumTrendSliceCount}`, async () => {
          const device = await getWebGPUTestDevice();
          requireTestDevice(device);
          const rig = new AnalysisRig(device);
          const fixture = AUDIT_EMERGING.cubes.find(cube => cube.name === 'weighted')!;
          const lossBuffer = createInputBuffer(device, Uint32Array.of(0));
          try {
            const output = createOutputs(rig);
            const weights = createWeights(rig, chunked);
            const loss = importGraphBuffer(rig.graph, 'loss-flag', lossBuffer, 'uint32', 1);
            if (chunkFlag && 'chunks' in weights) weights.chunks[1].weights.status = {[flag]: loss};
            else weights.status = {[flag]: loss};
            rig.graph.add(
              new GPUEmergingHotSpots({
                values: rig.input(Float32Array.from(fixture.values), 'float32'),
                weights,
                selfWeight: 1,
                sliceCount: 4,
                multipleTesting: 'none',
                mask: rig.input(Uint32Array.from(fixture.mask), 'uint32'),
                parameters: rig.input(
                  Float32Array.from(fixture.parameters, (value, index) =>
                    index === 5 ? minimumTrendSliceCount : value
                  ),
                  'float32'
                ),
                output
              })
            );
            const compiled = rig.graph.compile();
            try {
              for (const lossValue of [0, 1, 0]) {
                lossBuffer.write(Uint32Array.of(lossValue));
                submitGraph(device, compiled, undefined);
                await compareCube(
                  rig,
                  output,
                  minimumTrendSliceCount === 0
                    ? {
                        ...fixture,
                        expected: {
                          ...fixture.expected,
                          trendZ: fixture.expected.trendZ.map((value, index) =>
                            fixture.expected.lastValidSlice[index] === 0xffffffff ? 0 : value
                          ),
                          trendP: fixture.expected.trendP.map((value, index) =>
                            fixture.expected.lastValidSlice[index] === 0xffffffff ? 1 : value
                          )
                        }
                      }
                    : fixture,
                  lossValue !== 0
                );
              }
            } finally {
              compiled.destroy();
            }
          } finally {
            lossBuffer.destroy();
            rig.destroy();
          }
        });
      }

it('publishes the same independent population count through the legacy count alias', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const fixture = AUDIT_EMERGING.cubes[0];
  try {
    const output = createOutputs(rig);
    rig.run(
      new GPUEmergingHotSpots({
        values: rig.input(Float32Array.from(fixture.values), 'float32'),
        gridWidth: 4,
        gridHeight: 2,
        sliceCount: 4,
        maximumRadius: 1,
        multipleTesting: 'none',
        mask: rig.input(Uint32Array.from(fixture.mask), 'uint32'),
        parameters: rig.input(Float32Array.from(fixture.parameters), 'float32'),
        output: {...output, includedCount: undefined, count: output.includedCount}
      })
    );
    await compareCube(rig, output, fixture);
  } finally {
    rig.destroy();
  }
});

it('retains the tiny focal Gi* coefficient beside a missing unit-weight neighbor', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const coefficientScale of [1, 2 ** -66]) {
    const rig = new AnalysisRig(device);
    try {
      const output = createOutputs(rig, 3, 1);
      rig.run(
        new GPUEmergingHotSpots({
          values: rig.input(Float32Array.of(1, NaN, 5), 'float32'),
          weights: rig.weights({
            offsets: [0, 1, 1, 1],
            neighbors: [1],
            weights: [coefficientScale]
          }),
          selfWeight: 1e-8 * coefficientScale,
          sliceCount: 1,
          multipleTesting: 'none',
          parameters: rig.input(Float32Array.of(0, 0, 1.96, 0.05, 0.9, 0, 0.05, 0), 'float32'),
          output
        })
      );
      // Usable population [1,5]: mean=3, deviation=2. Each usable row has only self weight w,
      // hence Gi* numerator=+/-2w and denominator=2w independently of w and the missing edge.
      expectNumericArrayClose(await rig.readFloat(output.giZScores), [-1, NaN, 1], {
        absoluteTolerance: 1e-6
      });
      for (const port of ['pValues', 'adjustedPValues'] as const)
        expectNumericArrayClose(
          await rig.readFloat(output[port]),
          [0.31731050786291415, NaN, 0.31731050786291415],
          {absoluteTolerance: 1e-7}
        );
      expectNumericArrayClose(await rig.readFloat(output.globalStatistics), [2, 3, 4, 2]);
      expectNumericArrayClose(await rig.readUint(output.includedCount), [2]);
    } finally {
      rig.destroy();
    }
  }
});

for (const flag of ['overflow', 'candidateOverflow'] as const)
  it(`S4 cooperative 64-slice loss and recovery with minimum trend zero: ${flag}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const lossBuffer = createInputBuffer(device, Uint32Array.of(0));
    try {
      const output = createOutputs(rig, 3, 64);
      const loss = importGraphBuffer(rig.graph, 'cooperative-loss', lossBuffer, 'uint32', 1);
      const weights = rig.weights({offsets: [0, 0, 0, 0], neighbors: [], weights: []});
      weights.status = {[flag]: loss};
      rig.graph.add(
        new GPUEmergingHotSpots({
          values: rig.input(
            Float32Array.from([
              ...new Array<number>(64).fill(1),
              ...new Array<number>(64).fill(5),
              ...new Array<number>(64).fill(NaN)
            ]),
            'float32'
          ),
          weights,
          sliceCount: 64,
          multipleTesting: 'none',
          parameters: rig.input(Float32Array.of(0, 0, 1.96, 0.05, 0.9, 0, 0.05, 0), 'float32'),
          output
        })
      );
      const compiled = rig.graph.compile();
      try {
        for (const lost of [false, true, false]) {
          lossBuffer.write(Uint32Array.of(lost ? 1 : 0));
          submitGraph(device, compiled, undefined);
          // Population has 64 ones and 64 fives: n=128, mean=3, variance=4. Self-only
          // windows give 64 scores -1 and 64 scores +1, each constant series having S=0.
          const scores = lost
            ? new Array<number>(192).fill(NaN)
            : [
                ...new Array<number>(64).fill(-1),
                ...new Array<number>(64).fill(1),
                ...new Array<number>(64).fill(NaN)
              ];
          const probabilities = lost
            ? new Array<number>(192).fill(NaN)
            : [
                ...new Array<number>(128).fill(0.31731050786291415),
                ...new Array<number>(64).fill(NaN)
              ];
          expectNumericArrayClose(await rig.readFloat(output.giZScores), scores, {
            absoluteTolerance: 1e-6
          });
          for (const port of ['pValues', 'adjustedPValues'] as const)
            expectNumericArrayClose(await rig.readFloat(output[port]), probabilities, {
              absoluteTolerance: 1e-7
            });
          compareTrend(
            await readTrend(rig, output),
            {
              trendS: [0, 0, 0],
              trendZ: lost ? [NaN, NaN, NaN] : [0, 0, 0],
              trendP: lost ? [NaN, NaN, NaN] : [1, 1, 1],
              category: lost ? [17, 17, 17] : [0, 0, 0],
              hotSliceCount: [0, 0, 0],
              coldSliceCount: [0, 0, 0],
              lastValidSlice: lost ? [0xffffffff, 0xffffffff, 0xffffffff] : [63, 63, 0xffffffff]
            },
            `cooperative ${flag}, lost=${lost}`
          );
          expectNumericArrayClose(await rig.readFloat(output.globalStatistics), [128, 3, 4, 2]);
          expectNumericArrayClose(await rig.readUint(output.includedCount), [128]);
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      lossBuffer.destroy();
      rig.destroy();
    }
  });
