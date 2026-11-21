// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {AnalysisValidationError} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {addSpaceTimeHotSpotsRecipe} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {
  GPUEmergingHotSpots,
  type GPUEmergingHotSpotsProps
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/gpu-emerging-hot-spots';
import {
  getGPUEmergingHotSpotsCriticalZ,
  getGPUEmergingHotSpotsParameterValues,
  GPU_EMERGING_HOT_SPOTS_CRITICAL_Z_SCORES,
  type GPUEmergingHotSpotsSettings
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {AUDIT_EMERGING} from './audit-emerging-hot-spots-fixture';
import {
  executeAuditEmergingStage,
  getInterpretedEmergingCompensationWGSL
} from './audit-emerging-hot-spots-wgsl';
import {
  computeEmergingHotSpotMoments,
  computeSpaceTimeGiStar,
  computeSpaceTimeGiStarWeighted,
  computeBinPValues,
  computeAdjustedBinPValues,
  computeEmergingHotSpotCells
} from './emerging-hot-spots-oracle';

let serial = 0;

it('has independent numerical fixtures at the maximum accepted corrected trend length', () => {
  for (const correction of ['hamed-rao', 'yue-wang'])
    expect(
      AUDIT_EMERGING.stages.some(
        stage => stage.sliceCount === 63 && stage.correction === correction
      )
    ).toBe(true);
});

it('reports the encoded custom critical value normal tail and preserves the separate trend level', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const contributor = new GPUEmergingHotSpots(createAuditEmergingProps(graph));
    const settings = {temporalWindow: 0, criticalZ: 1, trendSignificanceLevel: 0.025};
    expectNumericArrayClose(
      [contributor.getInferenceMetadata(settings).significanceLevel!],
      [0.31731050786291415],
      {absoluteTolerance: 1e-7}
    );
    expect(contributor.getParameterValues(settings)[3]).toBe(Math.fround(0.025));
    expectNumericArrayClose(
      [contributor.getInferenceMetadata({...settings, criticalZ: 0.75000001}).significanceLevel!],
      [0.4532547047537364],
      {absoluteTolerance: 1e-7}
    );
  } finally {
    device.destroy();
  }
});

it('updates the space-time recipe legend from the custom encoded critical threshold', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters = new GPUParameterBuffer(device, {id: 'legend', format: 'float32', length: 8});
  const writes = vi.spyOn(parameters, 'write');
  try {
    const recipe = addSpaceTimeHotSpotsRecipe(graph, {
      timestamps: createTransientView(graph, 'timestamps', 'uint32x2', 1),
      calendarParameters: createTransientView(graph, 'calendar', 'sint32', 2),
      slices: {step: 'month', origin: {year: 2023}, count: 1},
      cells: {
        kind: 'lattice',
        positions: createTransientView(graph, 'positions', 'float32x2', 1),
        width: 2,
        height: 4,
        bounds: [0, 0, 2, 4]
      },
      multipleTesting: 'none',
      settings: {temporalWindow: 0, criticalZ: 1, trendSignificanceLevel: 0.025},
      parameters
    });
    expectNumericArrayClose([recipe.inference.significanceLevel!], [0.31731050786291415], {
      absoluteTolerance: 1e-7
    });
    let thresholdReads = 0;
    recipe.setSettings({
      temporalWindow: 0,
      get criticalZ() {
        thresholdReads++;
        return thresholdReads === 1 ? 1 : 0.75;
      },
      trendSignificanceLevel: 0.025
    });
    expect(thresholdReads).toBe(1);
    expect(writes.mock.calls.at(-1)![0][2]).toBe(1);
    expectNumericArrayClose([recipe.inference.significanceLevel!], [0.31731050786291415], {
      absoluteTolerance: 1e-7
    });
    recipe.setSettings({temporalWindow: 0, criticalZ: 0.75000001, trendSignificanceLevel: 0.025});
    expectNumericArrayClose([recipe.inference.significanceLevel!], [0.4532547047537364], {
      absoluteTolerance: 1e-7
    });
    const packed = Array.from(writes.mock.calls.at(-1)![0]);
    expect(packed[2]).toBe(0.75);
    expect(packed[3]).toBe(Math.fround(0.025));
  } finally {
    writes.mockRestore();
    parameters.destroy();
    device.destroy();
  }
});

it('executes centered coefficient moments for nearly uniform, uniform and zero-weight populations', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createAuditEmergingProps(graph, 1);
    props.gridWidth = undefined;
    props.gridHeight = undefined;
    props.weights = {
      offsets: createTransientView(graph, 'offsets', 'uint32', 9),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 56),
      weights: createTransientView(graph, 'weights', 'float32', 56)
    };
    const source = getGPUCommandNodeKernelSources(
      new GPUEmergingHotSpots(props).getCommandNodes(graph)
    ).find(kernel => kernel.id.includes('gi-star'))!.source;
    const moment = source.match(/fn addEmergingWeightMoment[\s\S]*?\n}/)![0];
    const scale = source.match(/fn getEmergingCoefficientScale[\s\S]*?\n}/)![0];
    const program = `${getInterpretedEmergingCompensationWGSL(source)}\n${moment}\n${scale}
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(1,1,1) fn main() {
 var state = vec3<f32>(0.0);
 var maximum = 0.0;
 for (var row=0u;row<8u;row++) {maximum=max(maximum,abs(weights[row]));}
 let scale=getEmergingCoefficientScale(maximum);
 for (var row=0u;row<8u;row++) {state=addEmergingWeightMoment(state,row+1u,weights[row]/scale-weights[1]/scale);}
 output[0]=getCompensated(state.yz)*8.0/7.0;
}`;
    for (const [weights, expected] of [
      [Float32Array.of(1 + 2 ** -12, 1, 1, 1, 1, 1, 1, 1), 2 ** -24],
      [Float32Array.from([1 + 2 ** -12, 1, 1, 1, 1, 1, 1, 1], value => value * 2 ** -66), 2 ** -24],
      [new Float32Array(8).fill(1), 0],
      // Moving the excess from self to a neighboring coefficient leaves the same variance:
      // delta²(1 - 1/8) * 8/7 = delta² = 2^-24.
      [Float32Array.of(1, 1 + 2 ** -12, 1, 1, 1, 1, 1, 1), 2 ** -24],
      [Float32Array.of(0, 1, 1, 1, 1, 1, 1, 1), 1]
    ] as const) {
      const output = new Float32Array(1);
      new WgslExec(new WgslParser().parse(program)).dispatchWorkgroups('main', 1, {
        0: {0: weights, 1: output}
      });
      expectNumericArrayClose(output, [expected], {relativeTolerance: 2e-6});
    }
  } finally {
    device.destroy();
  }
});
export function createAuditEmergingProps(
  graph: GPUCommandGraph,
  sliceCount = 4
): GPUEmergingHotSpotsProps {
  const view = <Format extends 'float32' | 'uint32' | 'sint32'>(format: Format, length: number) =>
    createTransientView(graph, `emerging-audit-${serial++}`, format, length);
  return {
    values: view('float32', 8 * sliceCount),
    gridWidth: 4,
    gridHeight: 2,
    sliceCount,
    multipleTesting: 'none',
    parameters: view('float32', 8),
    output: {
      giZScores: view('float32', 8 * sliceCount),
      trendZ: view('float32', 8),
      trendP: view('float32', 8),
      trendS: view('sint32', 8),
      category: view('uint32', 8),
      hotSliceCount: view('uint32', 8),
      coldSliceCount: view('uint32', 8)
    }
  };
}

it('packs the independent normal critical values and all eight parameter positions', () => {
  // scipy.stats.norm.ppf(.975) and norm.ppf(.995); remaining positions follow the declared layout.
  expectNumericArrayClose(
    getGPUEmergingHotSpotsParameterValues({temporalWindow: 0}),
    [0, 0, 1.959963984540054, 0.05, 0.9, 10, 0.05, 0],
    {absoluteTolerance: 1e-7}
  );
  expectNumericArrayClose(
    getGPUEmergingHotSpotsParameterValues({
      radius: 2.5,
      temporalWindow: 3,
      confidenceLevel: 0.99,
      trendSignificanceLevel: 0.025,
      persistentFraction: 0.75,
      minimumTrendSliceCount: 6,
      sporadicRequiresFinalSlice: false
    }),
    [2.5, 3, 2.5758293035489004, 0.025, 0.75, 6, 0.01, 1],
    {absoluteTolerance: 1e-7}
  );
});

for (const [field, value, extra] of [
  ['radius', 1e100, {}],
  ['temporalWindow', 1e100, {}],
  ['criticalZ', 1e-60, {multipleTesting: 'none'}],
  ['trendSignificanceLevel', 1e-60, {}],
  ['persistentFraction', 0.50000000001, {}],
  ['minimumTrendSliceCount', 1e100, {}],
  ['confidenceLevel', 1e-60, {trendSignificanceLevel: 0.05}]
] as const) {
  it(`rejects ${field} that leaves its domain when encoded as float32`, () => {
    const settings = {temporalWindow: 0, ...extra, [field]: value} as GPUEmergingHotSpotsSettings;
    expect(() => getGPUEmergingHotSpotsParameterValues(settings)).toThrow(
      new RegExp(`GPUEmergingHotSpots ${field}`)
    );
  });
}

it('rejects unsupported correction names at the exported packer as well as the contributor', () => {
  expect(() =>
    getGPUEmergingHotSpotsParameterValues({temporalWindow: 0, multipleTesting: 'invalid' as 'none'})
  ).toThrow(/GPUEmergingHotSpots multipleTesting/);
});

it('snapshots emerging-hot-spot topology and output handles', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createAuditEmergingProps(graph);
    const contributor = new GPUEmergingHotSpots(props);
    props.sliceCount = 7;
    props.output.category = createTransientView(graph, 'late-category', 'uint32', 8);
    expect(contributor.props.sliceCount).toBe(4);
    expect(contributor.props.output.category).not.toBe(props.output.category);
  } finally {
    device.destroy();
  }
});

for (const chunked of [false, true]) {
  it(`snapshots nested neighborhood descriptors before lowering: chunked=${chunked}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      const props = createAuditEmergingProps(graph);
      props.gridWidth = undefined;
      props.gridHeight = undefined;
      const overflow = createTransientView(graph, 'original-overflow', 'uint32', 1);
      const original = {
        offsets: createTransientView(graph, 'original-offsets', 'uint32', 9),
        neighbors: createTransientView(graph, 'original-neighbors', 'uint32', 8),
        weights: createTransientView(graph, 'original-weights', 'float32', 8),
        status: {overflow}
      };
      const chunks = [{rowStart: 0, rowCount: 8, weights: original}];
      props.weights = chunked ? {rowCount: 8, chunks} : original;
      const contributor = new GPUEmergingHotSpots(props);
      original.status.overflow = createTransientView(graph, 'replacement-overflow', 'uint32', 1);
      original.offsets = createTransientView(graph, 'replacement-offsets', 'uint32', 1);
      chunks[0].rowCount = 1;
      chunks.length = 0;
      const nodes = contributor.getCommandNodes(graph);
      expect(
        nodes.filter(node =>
          node.resources?.some(resource => 'buffer' in resource && resource.buffer === overflow)
        )
      ).toHaveLength(1);
      expect(
        nodes.some(node =>
          node.resources?.some(
            resource => 'buffer' in resource && resource.buffer === original.status.overflow
          )
        )
      ).toBe(false);
    } finally {
      device.destroy();
    }
  });
}

it('reads both neighborhood loss flags before deriving hot spots', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createAuditEmergingProps(graph);
    props.gridWidth = undefined;
    props.gridHeight = undefined;
    const status = {
      overflow: createTransientView(graph, 'overflow', 'uint32', 1),
      candidateOverflow: createTransientView(graph, 'candidate-overflow', 'uint32', 1)
    };
    props.weights = {
      offsets: createTransientView(graph, 'offsets', 'uint32', 9),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 8),
      weights: createTransientView(graph, 'weights', 'float32', 8),
      status
    };
    const nodes = new GPUEmergingHotSpots(props).getCommandNodes(graph);
    for (const flag of Object.values(status))
      expect(
        nodes.filter(node =>
          node.resources?.some(resource => 'buffer' in resource && resource.buffer === flag)
        )
      ).toHaveLength(1);
  } finally {
    device.destroy();
  }
});

it('bounds correction lags to uint32 and ignores valid lags beyond the series', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    expect(
      () =>
        new GPUEmergingHotSpots({
          ...createAuditEmergingProps(graph),
          trendAutocorrelationCorrection: 'yue-wang',
          trendCorrectionMaximumLag: Number.MAX_SAFE_INTEGER
        })
    ).toThrow(/trendCorrectionMaximumLag/);
    const contributor = new GPUEmergingHotSpots({
      ...createAuditEmergingProps(graph),
      trendAutocorrelationCorrection: 'yue-wang',
      trendCorrectionMaximumLag: 0xffffffff
    });
    const kernel = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(source =>
      source.id.endsWith('-mann-kendall')
    )!;
    expect(kernel.source).toContain('min(4294967295u, valid - 1u)');
  } finally {
    device.destroy();
  }
});

// The serial interpreter cannot synchronize cooperative barriers or reliably run the Sen heap's
// function-pointer arrays on tied/nonmonotone input. The headless spec executes every fixture,
// including both corrected methods and the 64/65-slice cooperative boundaries, without shims.
for (const fixture of AUDIT_EMERGING.stages.filter(
  stage => stage.sliceCount < 64 && stage.correction === 'none'
)) {
  it(`executes actual trend/classification WGSL against the independent oracle: ${fixture.name}`, () => {
    const actual = executeAuditEmergingStage(fixture);
    for (const port of Object.keys(fixture.expected) as (keyof typeof fixture.expected)[]) {
      expectNumericArrayClose(actual[port], fixture.expected[port], {
        absoluteTolerance: port === 'trendZ' ? 3e-5 : 0,
        relativeTolerance: port === 'trendP' ? 1e-3 : port === 'trendZ' ? 5e-5 : 0,
        label: `${fixture.name}, ${port}`
      });
    }
  });
}

for (const fixture of AUDIT_EMERGING.stages.filter(
  stage => stage.correction !== 'none' && stage.sliceCount === 12
)) {
  it(`keeps a constant detrended signal uncorrected despite rounded mean division: ${fixture.name}`, () => {
    // Row 5 is [9,8,...,-2]. Every Sen slope is -1, so the detrended series is exactly 10,
    // or midrank 6.5 for Hamed-Rao. The independent SciPy oracle therefore uses factor 1.
    // Repeat eight cells to isolate this arithmetic from unrelated interpreter heap limits.
    const scores = fixture.scores.slice(5 * fixture.sliceCount, 6 * fixture.sliceCount);
    const expected = Object.fromEntries(
      Object.entries(fixture.expected).map(([port, values]) => [
        port,
        new Array<number>(8).fill(values[5])
      ])
    ) as typeof fixture.expected;
    const actual = executeAuditEmergingStage(
      {...fixture, scores: Array.from({length: 8}, () => scores).flat(), expected},
      {roundCorrectionMeanDown: true}
    );
    for (const port of Object.keys(expected) as (keyof typeof expected)[]) {
      expectNumericArrayClose(actual[port], expected[port], {
        absoluteTolerance: port === 'trendZ' ? 3e-5 : 0,
        relativeTolerance: port === 'trendP' ? 1e-3 : 0,
        label: `${fixture.name}, ${port}`
      });
    }
  });
}

for (const fixture of AUDIT_EMERGING.cubes.filter(cube => cube.correction === 'none')) {
  it(`checks the host reference against independent SciPy literals on every cube output: ${fixture.name}`, () => {
    const cube = {
      gridWidth: 4,
      gridHeight: 2,
      sliceCount: 4,
      values: fixture.unsigned
        ? Uint32Array.from(fixture.values)
        : Float32Array.from(fixture.values),
      mask: Uint32Array.from(fixture.mask)
    };
    const packed = Float32Array.from(fixture.parameters);
    const moments = computeEmergingHotSpotMoments(cube);
    expectNumericArrayClose(
      [moments.count, moments.mean, moments.variance, moments.standardDeviation],
      fixture.expected.globalStatistics,
      {absoluteTolerance: 1e-10, relativeTolerance: 1e-12}
    );
    expectNumericArrayClose([moments.count], fixture.expected.includedCount);
    const scores =
      fixture.mode === 'lattice'
        ? computeSpaceTimeGiStar(cube, packed, 1)
        : computeSpaceTimeGiStarWeighted(cube, AUDIT_EMERGING.weights, fixture.selfWeight, packed);
    expectNumericArrayClose(scores, fixture.expected.giZScores, {
      absoluteTolerance: 1e-10,
      relativeTolerance: 1e-12
    });
    const published = Float32Array.from(scores);
    expectNumericArrayClose(computeBinPValues(published), fixture.expected.pValues, {
      absoluteTolerance: 1e-10,
      relativeTolerance: 2e-7
    });
    expectNumericArrayClose(
      computeAdjustedBinPValues(published, fixture.method),
      fixture.expected.adjustedPValues,
      {absoluteTolerance: 1e-10, relativeTolerance: 2e-7}
    );
    const cells = computeEmergingHotSpotCells(published, cube, packed, {
      multipleTesting: fixture.method,
      finalSlice: fixture.finalSlice
    });
    for (const port of Object.keys(cells) as (keyof typeof cells)[])
      expectNumericArrayClose(cells[port], fixture.expected[port], {
        absoluteTolerance: 1e-10,
        relativeTolerance: port === 'trendP' ? 2e-7 : 1e-12,
        label: port
      });
  });
}

it('keeps contributor packing and inference metadata consistent with all independent cube settings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    for (const fixture of AUDIT_EMERGING.cubes) {
      const contributor = new GPUEmergingHotSpots({
        ...createAuditEmergingProps(graph),
        multipleTesting: fixture.method
      });
      const target = new Float32Array(11).fill(123);
      expect(contributor.getParameterValues(fixture.settings, target)).toBe(target);
      expectNumericArrayClose(target.slice(0, 8), fixture.parameters);
      expect(Array.from(target.slice(8))).toEqual([123, 123, 123]);
      const metadata = contributor.getInferenceMetadata(fixture.settings);
      expect({...metadata, significanceLevel: 0}).toEqual({
        alternative: 'two-sided',
        permutationCount: 0,
        multipleTesting: fixture.method,
        significanceLevel: 0,
        testedRowsDefinition: 'valid space-time bins'
      });
      expectNumericArrayClose(
        [metadata.significanceLevel!],
        [
          fixture.method === 'none' && fixture.settings.criticalZ !== undefined
            ? 0.4532547047537364
            : 1 - fixture.settings.confidenceLevel!
        ],
        {absoluteTolerance: 1e-7}
      );
    }
    for (const [index, confidenceLevel] of [0.9, 0.95, 0.99].entries()) {
      expectNumericArrayClose(
        [
          getGPUEmergingHotSpotsParameterValues({
            temporalWindow: 0,
            confidenceLevel,
            multipleTesting: 'none'
          })[2]
        ],
        [AUDIT_EMERGING.parameterCriticalValues[index]],
        {absoluteTolerance: 1e-7}
      );
    }
  } finally {
    device.destroy();
  }
});

it('TYP-12 exposes definite tabulated critical values and checks dynamic confidence levels', () => {
  const known: readonly number[] = [
    GPU_EMERGING_HOT_SPOTS_CRITICAL_Z_SCORES[0.9],
    GPU_EMERGING_HOT_SPOTS_CRITICAL_Z_SCORES['0.95'],
    GPU_EMERGING_HOT_SPOTS_CRITICAL_Z_SCORES[0.99]
  ];
  expectNumericArrayClose(known, AUDIT_EMERGING.parameterCriticalValues, {
    absoluteTolerance: 1e-14
  });
  expect([0.9, 0.95, 0.99].map(level => getGPUEmergingHotSpotsCriticalZ(level))).toEqual(known);
  // @ts-expect-error The critical-value table has no key for this confidence level.
  expect(GPU_EMERGING_HOT_SPOTS_CRITICAL_Z_SCORES['0.975']).toBeUndefined();
  expect(() => getGPUEmergingHotSpotsCriticalZ(0.975)).toThrow(
    /^emerging-hot-spots confidenceLevel must be 0.9, 0.95 or 0.99; received 0.975/
  );
  expect(() =>
    getGPUEmergingHotSpotsParameterValues(
      {temporalWindow: 0, confidenceLevel: 0.975, multipleTesting: 'none'},
      undefined,
      'critical-z'
    )
  ).toThrow(/^critical-z confidenceLevel must be 0.9, 0.95 or 0.99; received 0.975/);
  expect(
    getGPUEmergingHotSpotsParameterValues({temporalWindow: 0, confidenceLevel: 0.975})[2]
  ).toBe(0);
});

it('W2-15 captures nested weights provenance and reports named construction requirements', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createAuditEmergingProps(graph);
    props.gridWidth = undefined;
    props.gridHeight = undefined;
    const provenance = {source: 'upload' as const, radius: 2};
    const information = {rowCount: 8, transform: 'raw' as const, symmetric: true, provenance};
    props.weights = {
      offsets: createTransientView(graph, 'capture-offsets', 'uint32', 9),
      neighbors: createTransientView(graph, 'capture-neighbors', 'uint32', 8),
      weights: createTransientView(graph, 'capture-weights', 'float32', 8),
      information
    };
    const contributor = new GPUEmergingHotSpots(props);
    provenance.radius = 99;
    const captured = contributor.props.weights!;
    expect(captured.information?.provenance?.radius).toBe(2);
    expect('chunks' in captured ? captured.chunks[0].weights.offsets : captured.offsets).toBe(
      props.weights.offsets
    );
    expect(
      () =>
        new GPUEmergingHotSpots({...props, id: 'custom-hot-spots', finalSlice: 'typo' as 'require'})
    ).toThrow(AnalysisValidationError);
    expect(
      () =>
        new GPUEmergingHotSpots({...props, id: 'custom-hot-spots', finalSlice: 'typo' as 'require'})
    ).toThrow(/^custom-hot-spots.*finalSlice.*require.*last-valid.*typo/);
  } finally {
    device.destroy();
  }
});

it('W2-15 parameter packing retains the instance ID in named validation failures', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const contributor = new GPUEmergingHotSpots({
      ...createAuditEmergingProps(graph),
      id: 'custom-packer'
    });
    expect(() => contributor.getParameterValues({temporalWindow: -1})).toThrow(
      AnalysisValidationError
    );
    expect(() => contributor.getParameterValues({temporalWindow: -1})).toThrow(
      /^custom-packer.*temporalWindow.*-1/
    );
    expect(() => contributor.getParameterValues({temporalWindow: 0}, new Float32Array(1))).toThrow(
      /^custom-packer.*target.*8.*1/
    );
  } finally {
    device.destroy();
  }
});

for (const inherited of [{criticalZ: 0.75}, {criticalZ: 0.75, temporalWindow: 2}])
  it(`B2 shares named settings capture between standalone and instance packers: ${JSON.stringify(inherited)}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      const contributor = new GPUEmergingHotSpots(createAuditEmergingProps(graph));
      const settings = Object.assign(Object.create(inherited), {
        multipleTesting: 'none',
        ...('temporalWindow' in inherited ? {} : {temporalWindow: 1})
      }) as GPUEmergingHotSpotsSettings;
      const expected = [
        0,
        'temporalWindow' in inherited ? 2 : 1,
        0.75,
        Math.fround(0.05),
        Math.fround(0.9),
        10,
        Math.fround(0.05),
        0
      ];
      expect(Array.from(getGPUEmergingHotSpotsParameterValues(settings))).toEqual(expected);
      expect(Array.from(contributor.getParameterValues(settings))).toEqual(expected);
      expectNumericArrayClose(
        [contributor.getInferenceMetadata(settings).significanceLevel!],
        [0.4532547047537364],
        {absoluteTolerance: 1e-7}
      );
    } finally {
      device.destroy();
    }
  });

it('S5 unsupported multiple testing reports structured errors and the caller identity at both entries', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const attempts = [
      () =>
        new GPUEmergingHotSpots({
          ...createAuditEmergingProps(graph),
          id: 'invalid-method',
          multipleTesting: 'invalid' as 'none'
        }),
      () =>
        getGPUEmergingHotSpotsParameterValues(
          {temporalWindow: 0, multipleTesting: 'invalid' as 'none'},
          undefined,
          'invalid-method'
        )
    ];
    for (const attempt of attempts) {
      expect(attempt).toThrow(AnalysisValidationError);
      expect(attempt).toThrow(
        /^invalid-method.*multipleTesting.*none.*bonferroni.*holm.*benjamini-hochberg.*benjamini-yekutieli.*invalid/
      );
    }
  } finally {
    device.destroy();
  }
});

it('earlier weighted Gi* control retains a tiny focal coefficient beside a missing unit-weight neighbor', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createAuditEmergingProps(graph, 1);
    props.gridWidth = undefined;
    props.gridHeight = undefined;
    props.selfWeight = 1e-8;
    props.weights = {
      offsets: createTransientView(graph, 'missing-offsets', 'uint32', 9),
      neighbors: createTransientView(graph, 'missing-neighbors', 'uint32', 1),
      weights: createTransientView(graph, 'missing-weights', 'float32', 1)
    };
    const source = getGPUCommandNodeKernelSources(
      new GPUEmergingHotSpots(props).getCommandNodes(graph)
    ).find(kernel => kernel.id.includes('gi-star'))!.source;
    // Execute the actual instant-sum body without its cooperative wrapper: for one slice the
    // focal invocation reads only its own instant slot. This is execution, never the oracle.
    const start = source.indexOf('var weighted = ');
    const end = source.indexOf('\n  }\n  // Every invocation', start);
    expect(start).toBeGreaterThan(0);
    expect(end).toBeGreaterThan(start);
    const instant = source.slice(start, end);
    const moment = source.match(/fn addEmergingWeightMoment[\s\S]*?\n}/)![0];
    const scale = source.match(/fn getEmergingCoefficientScale[\s\S]*?\n}/)![0];
    const spreadStart = source.indexOf('let outsideCount = count - min(count, supportCount);');
    const spreadEnd = source.indexOf('if (count >=', spreadStart);
    const spread = source.slice(spreadStart, spreadEnd);
    const sumStart = source.indexOf('let residualTotal = residualMean[', spreadStart);
    const sumEnd = source.indexOf('zScore = ', sumStart);
    const centeredSum = source.slice(sumStart, sumEnd);
    const program = `${getInterpretedEmergingCompensationWGSL(source)}\n${moment}\n${scale}
const SELF_WEIGHT: f32 = 1e-8;
const CELL_COUNT: u32 = 3u;
const SLICE_COUNT: u32 = 1u;
@group(0) @binding(0) var<storage, read> centeredValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
const offsets = array<u32, 4>(0u, 1u, 1u, 1u);
const neighbors = array<u32, 1>(1u);
const weights = array<f32, 1>(1.0);
const offsetsOffset = 0u;
const neighborsOffset = 0u;
const weightsOffset = 0u;
var<private> instantWeighted: array<vec2<f32>, 1>;
var<private> instantWeight: array<f32, 1>;
var<private> instantSquare: array<f32, 1>;
var<private> instantCount: array<u32, 1>;
var<private> instantUnweighted: array<vec2<f32>, 1>;
var<private> instantOrigin: array<f32, 1>;
var<private> instantScale: array<f32, 1>;
const residualMeanOffset = 0u;
const residualMean = array<f32, 1>(0.0);
fn readDeviation(bin: u32) -> f32 {return centeredValues[bin];}
fn isBinValid(bin: u32, cell: u32) -> bool {return (bitcast<u32>(readDeviation(bin)) & 0x7f800000u) != 0x7f800000u;}
fn multiplyBySpatialCount(value: f32, count: u32) -> f32 {return value * f32(count);}
fn divideBySpatialCount(value: f32, count: u32) -> f32 {return value / f32(count);}
@compute @workgroup_size(1,1,1) fn main() {
  let cell = 0u; let slice = 0u; let local = 0u; let chunkRowStart = 0u;
  var coefficientOrigin = 0.0;
  ${instant}
  let count = 2u;
  let supportCount = instantCount[0];
  let weightMean = instantWeight[0];
  let weightSquares = instantSquare[0];
  let weightedSum = instantWeighted[0];
  let unweightedSum = instantUnweighted[0];
  let windowOrigin = coefficientOrigin;
  ${spread}
  ${centeredSum}
  output[0] = getCompensated(centeredSum) / (2.0 * sqrt(spread));
}`;
    const result = new Float32Array(1);
    // Values [1,NaN,5] have mean 3, population deviation 2. Only self has usable weight w:
    // numerator=-2w, denominator=2sqrt((2w²-w²)/(2-1))=2w, hence z=-1.
    new WgslExec(new WgslParser().parse(program)).dispatchWorkgroups('main', 1, {
      0: {0: Float32Array.of(-2, NaN, 2), 1: result}
    });
    expectNumericArrayClose(result, [-1], {absoluteTolerance: 1e-6});
  } finally {
    device.destroy();
  }
});
