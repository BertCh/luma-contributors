// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH,
  type GPUClassBreaksMethod,
  type GPUClassBreaksSettings
} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {GPUClassBreaks} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../../src/raster-analysis/raster-grid/gpu-raster-grid';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  computeClassBreaksOracle,
  computeExactNaturalBreaks,
  countClassesOracle,
  getWithinClassCost,
  type ClassBreaksOracleResult
} from './class-breaks-oracle';
import {
  CLASS_BREAKS_REFERENCE,
  getGoodnessOfVarianceFit,
  getReferenceEdgeMismatches,
  getReferenceParameters,
  NATURAL_BREAKS_LOCAL_OPTIMA,
  NATURAL_BREAKS_LOCAL_OPTIMUM_FIT_TOLERANCE
} from './class-breaks-reference';
import {LegacyNaturalBreaks} from './legacy-natural-breaks';
import {CLASS_BREAKS_EDGE_POLICY} from '../dataframe-edge-policy';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createClassificationKernelNode} from '../../../src/dataframe-analysis/column-classification/classification-kernel-node';

const MAXIMUM_CLASS_COUNT = 10;

/** Deterministic xorshift in [0, 1). */
function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

/** Heavy-tailed (Pareto-like) values with a few clusters, NaNs, infinities, and ties. */
function createColumn(
  seed: number,
  rows: number,
  options: {specials?: boolean} = {}
): Float32Array {
  const random = createRandom(seed);
  const values = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    const choice = random();
    if (options.specials && choice < 0.01) {
      values[row] = [NaN, Infinity, -Infinity, -0, 0][Math.floor(random() * 5)];
    } else if (choice < 0.3) {
      values[row] = 10 + Math.round(random() * 4);
    } else if (choice < 0.6) {
      values[row] = 100 / (1 - random() * 0.999) ** 0.8;
    } else {
      values[row] = (random() - 0.5) * 50;
    }
  }
  return values;
}

/** Three tight, well separated clusters, so natural breaks have one clear optimum. */
function createClusteredColumn(seed: number, rows: number): Float32Array {
  const random = createRandom(seed);
  const centers = [5, 40, 90];
  const values = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    values[row] = centers[row % 3] + (random() - 0.5) * 4;
  }
  values[0] = 0;
  values[1] = 100;
  return values;
}

type Result = {
  breaks: number[];
  breakBits: number[];
  classCount: number;
  classCounts: number[];
  gapKeys?: number[];
};

type Harness = {
  /** `parameters` may be a raw packed view, to write values the CPU packer refuses. */
  run(
    values: Float32Array,
    mask: Uint32Array | undefined,
    parameters: GPUClassBreaksSettings | Float32Array
  ): Promise<Result>;
  /** Submits the compiled graph `frameCount` times and waits for one small readback. */
  submit(frameCount: number): Promise<void>;
  buildCount: number;
  destroy(): void;
};

function createHarness(
  device: Device,
  props: {
    rows: number;
    /** Allocation may contain a spare tail; the declared values view still has rows rows. */
    allocationRows?: number;
    captureGapKeys?: boolean;
    logicalGridSize?: readonly [number, number];
    withMask: boolean;
    naturalBreaksBinCount: number;
    naturalBreaksMode?: 'approximate' | 'exact' | 'monotone';
    naturalBreaksExactStorage?: 'auto' | 'materialized' | 'recompute';
    methods?: GPUClassBreaksMethod[];
    valueFormat?: 'float32' | 'uint32' | 'sint32';
    /** Defaults to MAXIMUM_CLASS_COUNT. */
    maximumClassCount?: number;
  }
): Harness {
  const {rows, withMask, naturalBreaksBinCount} = props;
  const maximumClassCount = props.maximumClassCount ?? MAXIMUM_CLASS_COUNT;
  const buffers: Buffer[] = [];
  const keep = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const valuesBuffer = keep(
    createInputBuffer(device, new Float32Array(props.allocationRows ?? rows))
  );
  const maskBuffer = withMask ? keep(createInputBuffer(device, new Uint32Array(rows))) : undefined;
  const breaksBuffer = keep(createOutputBuffer(device, maximumClassCount + 1));
  const classCountBuffer = keep(createOutputBuffer(device, 1));
  const classCountsBuffer = keep(createOutputBuffer(device, maximumClassCount));
  const gapKeysBuffer = props.captureGapKeys ? keep(createOutputBuffer(device, rows)) : undefined;
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'class-breaks-parameters',
    format: 'float32',
    length: getGPUClassBreaksParameterLength(maximumClassCount)
  });
  const graph = new GPUCommandGraph(device, {id: 'class-breaks-graph'});
  if (props.captureGapKeys) recordWGSLKernelNodes(graph);
  const allocationView = importGraphBuffer(
    graph,
    'values',
    valuesBuffer,
    props.valueFormat ?? 'float32',
    props.allocationRows ?? rows
  );
  const values = props.logicalGridSize
    ? makeGPURasterBand(
        allocationView,
        makeGPURasterMetadata({width: props.logicalGridSize[0], height: props.logicalGridSize[1]})
      ).storage.values
    : allocationView;
  const contributor = new GPUClassBreaks({
    id: 'breaks',
    values,
    mask: maskBuffer ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows) : undefined,
    parameters: parameterBuffer.importToGraph(graph),
    maximumClassCount,
    methods: props.methods,
    naturalBreaksBinCount,
    naturalBreaksMode: props.naturalBreaksMode,
    naturalBreaksExactStorage: props.naturalBreaksExactStorage,
    output: {
      breaks: importGraphBuffer(graph, 'o-breaks', breaksBuffer, 'float32', maximumClassCount + 1),
      classCount: importGraphBuffer(graph, 'o-class-count', classCountBuffer, 'uint32', 1),
      classCounts: importGraphBuffer(
        graph,
        'o-class-counts',
        classCountsBuffer,
        'uint32',
        maximumClassCount
      )
    }
  });
  const harness: Harness = {
    buildCount: 0,
    async run(values, mask, parameters) {
      valuesBuffer.write(
        props.valueFormat === 'uint32'
          ? Uint32Array.from(values)
          : props.valueFormat === 'sint32'
            ? Int32Array.from(values)
            : values
      );
      if (maskBuffer) {
        maskBuffer.write(mask ?? new Uint32Array(rows).fill(1));
      }
      parameterBuffer.write(
        parameters instanceof Float32Array
          ? parameters
          : getGPUClassBreaksParameterValues(parameters, maximumClassCount)
      );
      submitGraph(device, compiled, undefined);
      const breaks = await readFloat32(breaksBuffer, maximumClassCount + 1);
      const breakBits = await readUint32(breaksBuffer, maximumClassCount + 1);
      const [classCount] = await readUint32(classCountBuffer, 1);
      const classCounts = await readUint32(classCountsBuffer, maximumClassCount);
      return {
        breaks,
        breakBits,
        classCount,
        classCounts,
        ...(gapKeysBuffer ? {gapKeys: await readUint32(gapKeysBuffer, rows)} : {})
      };
    },
    async submit(frameCount) {
      for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        submitGraph(device, compiled, undefined);
      }
      await readUint32(classCountBuffer, 1);
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
  graph.add(contributor);
  if (gapKeysBuffer) {
    const gapKeys = getWGSLKernelNodeRecord(graph, 'breaks-maximum-breaks-gaps')?.bindings.find(
      binding => binding.name === 'gapKeys'
    );
    expect(gapKeys).toBeDefined();
    const destination = importGraphBuffer(
      graph,
      'captured-gap-keys',
      gapKeysBuffer,
      'uint32',
      rows
    );
    // Copy the actual production gap-kernel output; a second arithmetic implementation would
    // fail to detect a saturated key. Extending its read lifetime also prevents transient aliasing.
    graph.add(
      createClassificationKernelNode(graph, {
        id: 'capture-production-gap-keys',
        operation: 'GPUClassBreaksGapKeyTest',
        invocationCount: rows,
        bindings: [
          {name: 'source', view: gapKeys!.view, type: 'u32', access: 'read'},
          {name: 'destination', view: destination, type: 'u32', access: 'read_write'}
        ],
        body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
      })
    );
  }
  harness.buildCount++;
  const compiled = graph.compile();
  return harness;
}

/** Relative closeness for edges that come out of f32 arithmetic. */
function expectEdgesClose(
  actual: number[],
  expected: ClassBreaksOracleResult,
  tolerance: number,
  label: string
): void {
  const scale = Math.max(
    Math.abs(expected.maximum - expected.minimum),
    Math.abs(expected.maximum),
    1e-30
  );
  for (let edge = 0; edge <= MAXIMUM_CLASS_COUNT; edge++) {
    const want = expected.breaks[edge];
    const got = actual[edge];
    if (Number.isNaN(want)) {
      expect(Number.isNaN(got), `${label} edge ${edge} should be NaN, got ${got}`).toBe(true);
    } else {
      expect(Math.abs(got - want), `${label} edge ${edge}: ${got} vs ${want}`).toBeLessThanOrEqual(
        tolerance * scale
      );
    }
  }
}

function expectEdgesExact(
  actual: number[],
  expected: ClassBreaksOracleResult,
  label: string
): void {
  for (let edge = 0; edge <= MAXIMUM_CLASS_COUNT; edge++) {
    const want = expected.breaks[edge];
    if (Number.isNaN(want)) {
      expect(Number.isNaN(actual[edge]), `${label} edge ${edge}`).toBe(true);
    } else {
      expect(
        Object.is(actual[edge], Math.fround(want)) || actual[edge] === Math.fround(want),
        `${label} edge ${edge}: ${actual[edge]} vs ${want}`
      ).toBe(true);
    }
  }
}

function checkFrame(
  result: Result,
  values: Float32Array,
  mask: Uint32Array | undefined,
  parameters: GPUClassBreaksSettings,
  naturalBreaksBinCount: number
): void {
  const label = `${parameters.method} k=${parameters.classCount ?? ''}`;
  const oracle = computeClassBreaksOracle({
    values,
    mask,
    parameters,
    maximumClassCount: MAXIMUM_CLASS_COUNT,
    naturalBreaksBinCount
  });
  if (parameters.method === 'natural-breaks' && oracle.naturalHistogram && result.classCount > 1) {
    // The CPU mirror runs the same algorithm; f32 rounding on the GPU may resolve a near tie
    // differently, in which case the partition must cost as little (relative 1e-6: the f32 class
    // cost has a relative error below 1e-6, see natural-breaks.node.spec.ts).
    expect(result.classCount, label).toBe(oracle.classCount);
    const gpuEdges = result.breaks.slice(1, result.classCount);
    const mirrorEdges = Array.from(oracle.breaks.slice(1, oracle.classCount));
    if (gpuEdges.some((edge, index) => edge !== mirrorEdges[index])) {
      const finite = Array.from(values).filter((value, row) => !mask || mask[row] !== 0);
      expect(getWithinClassCost(finite, gpuEdges), label).toBeLessThanOrEqual(
        getWithinClassCost(finite, mirrorEdges) * (1 + 1e-6)
      );
    }
  } else {
    expect(result.classCount, label).toBe(oracle.classCount);
    switch (parameters.method) {
      case 'quantile':
      case 'maximum-breaks':
      case 'custom':
        if (parameters.method === 'quantile') {
          // Linear interpolation may differ by one ulp from a fused multiply-add.
          expectEdgesClose(result.breaks, oracle, 1e-6, label);
        } else {
          expectEdgesExact(result.breaks, oracle, label);
        }
        break;
      case 'head-tail':
        // f32 fixed-order means versus f64 means.
        expectEdgesClose(result.breaks, oracle, 1e-5, label);
        break;
      default:
        expectEdgesClose(result.breaks, oracle, 1e-5, label);
    }
  }
  // Class counts are exact for the edges the GPU produced.
  expect(result.classCounts, `${label} class counts`).toEqual(
    Array.from(
      countClassesOracle(values, mask, result.breaks, result.classCount, MAXIMUM_CLASS_COUNT)
    )
  );
}

const FRAMES: GPUClassBreaksSettings[] = [
  {method: 'equal-interval', classCount: 5},
  {method: 'quantile', classCount: 7},
  {method: 'standard-deviation', classCount: 6, standardDeviationInterval: 0.5},
  {method: 'head-tail', classCount: 10},
  {method: 'head-tail', classCount: 10, headTailRatio: 1},
  {method: 'box-plot'},
  {method: 'maximum-breaks', classCount: 6},
  {method: 'natural-breaks', classCount: 5},
  {method: 'custom', customEdges: [-100, 0, 10, 50, 1000]},
  {method: 'quantile', classCount: 3},
  {method: 'equal-interval', classCount: 1},
  {method: 'natural-breaks', classCount: 10}
];

it('GPUClassBreaks matches the oracle for every method across per-frame changes without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 6000;
  const binCount = 256;
  const harness = createHarness(device, {rows, withMask: true, naturalBreaksBinCount: binCount});
  try {
    const random = createRandom(99);
    for (const [frameIndex, frame] of FRAMES.entries()) {
      const values = createColumn(frameIndex + 1, rows, {specials: frameIndex % 2 === 0});
      const mask = new Uint32Array(rows).map(() => (random() < 0.8 ? 1 : 0));
      const result = await harness.run(values, mask, frame);
      checkFrame(result, values, mask, frame, binCount);
      if (frame.method === 'custom')
        expect(result.classCounts.slice(0, 4)).toEqual([996, 396, 1900, 1463]);
      const again = await harness.run(values, mask, frame);
      expect(again.breakBits, `${frame.method} run-to-run`).toEqual(result.breakBits);
      expect(again.classCounts).toEqual(result.classCounts);
    }
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }
}, 120000);

it('GPUClassBreaks natural breaks find the clear optimum on clustered data', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 3000;
  const binCount = 1024;
  const harness = createHarness(device, {
    rows,
    withMask: false,
    naturalBreaksBinCount: binCount,
    methods: ['natural-breaks', 'head-tail']
  });
  try {
    const values = createClusteredColumn(5, rows);
    const parameters: GPUClassBreaksSettings = {method: 'natural-breaks', classCount: 3};
    const result = await harness.run(values, undefined, parameters);
    const oracle = computeClassBreaksOracle({
      values,
      parameters,
      maximumClassCount: MAXIMUM_CLASS_COUNT,
      naturalBreaksBinCount: binCount
    });
    expect(result.classCount).toBe(3);
    expectEdgesClose(result.breaks, oracle, 1e-6, 'clustered natural');
    // The breaks are the smallest values of the second and third cluster (centres 40 and 90, half
    // width 2): data values, not bin edges.
    expect(result.breaks[1]).toBeGreaterThanOrEqual(38);
    expect(result.breaks[1]).toBeLessThan(38.1);
    expect(result.breaks[2]).toBeGreaterThanOrEqual(88);
    expect(result.breaks[2]).toBeLessThan(88.1);
    expect(Array.from(values)).toContain(result.breaks[1]);
    expect(Array.from(values)).toContain(result.breaks[2]);
    expect(result.classCounts.slice(0, 3)).toEqual([1000, 999, 1001]);
    // A method that is not compiled in yields no classes.
    const missing = await harness.run(values, undefined, {method: 'quantile', classCount: 4});
    expect(missing.classCount).toBe(0);
    expect(missing.breaks.every(Number.isNaN)).toBe(true);
  } finally {
    harness.destroy();
  }
}, 60000);

it('GPUClassBreaks handles empty, constant, and tiny columns', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 64;
  const harness = createHarness(device, {rows, withMask: true, naturalBreaksBinCount: 64});
  try {
    const methods: GPUClassBreaksSettings[] = FRAMES.filter(frame => frame.method !== 'custom');
    const constant = new Float32Array(rows).fill(3.5);
    const allNaN = new Float32Array(rows).fill(NaN);
    const twoValues = new Float32Array(rows).map((_, row) => (row % 2 ? 1 : 2));
    const oneRowMask = new Uint32Array(rows);
    oneRowMask[7] = 1;
    for (const frame of methods) {
      for (const [label, values, mask] of [
        ['constant', constant, undefined],
        ['all NaN', allNaN, undefined],
        ['fully masked', twoValues, new Uint32Array(rows)],
        ['one row', twoValues, oneRowMask],
        ['two values', twoValues, undefined]
      ] as const) {
        const result = await harness.run(values, mask, frame);
        const oracle = computeClassBreaksOracle({
          values,
          mask,
          parameters: frame,
          maximumClassCount: MAXIMUM_CLASS_COUNT,
          naturalBreaksBinCount: 64
        });
        expect(result.classCount, `${frame.method} ${label}`).toBe(oracle.classCount);
        if (oracle.classCount <= 1) {
          expectEdgesExact(result.breaks, oracle, `${frame.method} ${label}`);
        } else {
          checkFrame(result, values, mask, frame, 64);
        }
      }
    }
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }
}, 120000);

it('GPUClassBreaks scales to 1M rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 1 << 20;
  const harness = createHarness(device, {
    rows,
    withMask: false,
    naturalBreaksBinCount: 1024,
    methods: [
      'quantile',
      'natural-breaks',
      'standard-deviation',
      'head-tail',
      'equal-interval',
      'box-plot'
    ]
  });
  try {
    const values = createColumn(77, rows);
    for (const frame of [
      {method: 'quantile', classCount: 9},
      {method: 'natural-breaks', classCount: 7},
      {method: 'head-tail', classCount: 10},
      {method: 'standard-deviation', classCount: 6}
    ] as GPUClassBreaksSettings[]) {
      const result = await harness.run(values, undefined, frame);
      checkFrame(result, values, undefined, frame, 1024);
      await harness.submit(2);
      const frameCount = 20;
      const start = performance.now();
      await harness.submit(frameCount);
      // Graph encode, submit, and GPU time per frame; the column is already resident.
      // biome-ignore lint/suspicious/noConsole: benchmark output for the handoff
      console.log(
        `GPUClassBreaks ${frame.method} 1M rows: ${((performance.now() - start) / frameCount).toFixed(2)} ms/frame`
      );
    }
  } finally {
    harness.destroy();
  }
}, 180000);

it('GPUClassBreaks classifies uint32 and sint32 columns like the converted float column', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 4000;
  const binCount = 256;
  for (const valueFormat of ['uint32', 'sint32'] as const) {
    const harness = createHarness(device, {
      rows,
      withMask: true,
      naturalBreaksBinCount: binCount,
      valueFormat
    });
    try {
      const random = createRandom(valueFormat === 'uint32' ? 5 : 6);
      const mask = new Uint32Array(rows).map(() => (random() < 0.85 ? 1 : 0));
      for (const [frameIndex, frame] of FRAMES.entries()) {
        // Count-like integers: many ties, a heavy tail, and negatives for sint32.
        const values = new Float32Array(rows).map(() => {
          const count = Math.floor(3 / (1 - random() * 0.999) ** 1.5);
          return valueFormat === 'sint32' ? count - 4 : count;
        });
        const result = await harness.run(values, mask, frame);
        checkFrame(result, values, mask, frame, binCount);
        expect(result.classCount, `${valueFormat} frame ${frameIndex}`).toBeGreaterThan(0);
      }
      expect(harness.buildCount).toBe(1);
    } finally {
      harness.destroy();
    }
  }
}, 120000);

it('GPUClassBreaks matches the mapclassify and jenkspy references', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Fixture: test/oracles/dataframe-class-breaks.py (mapclassify 2.11, jenkspy 0.4.1); each case
  // carries its derived tolerance. The CPU mirror passes the same cases in
  // class-breaks-reference.node.spec.ts.
  const mismatches: string[] = [];
  const localOptima: string[] = [];
  const unexpectedLocalOptima: string[] = [];
  for (const [name, scene] of Object.entries(CLASS_BREAKS_REFERENCE.scenes)) {
    const values = Float32Array.from(scene.values);
    const harness = createHarness(device, {
      rows: values.length,
      withMask: false,
      naturalBreaksBinCount: 2048
    });
    try {
      for (const referenceCase of scene.cases) {
        const result = await harness.run(values, undefined, getReferenceParameters(referenceCase));
        mismatches.push(
          ...getReferenceEdgeMismatches(
            `${name} ${referenceCase.method} k=${referenceCase.classCount}`,
            result.breaks,
            result.classCount,
            referenceCase
          )
        );
      }
      for (const reference of scene.naturalBreaks) {
        const result = await harness.run(values, undefined, {
          method: 'natural-breaks',
          classCount: reference.classCount
        });
        const label = `${name} ${reference.classCount}`;
        expect(result.classCount, label).toBe(reference.classCount);
        const inner = result.breaks.slice(1, reference.classCount);
        // Exact lower edges of the exact Fisher-Jenks partition: equal data values. The JSON of the
        // fixture is f32 data, so compare after Math.fround.
        const expected = reference.exactLowerEdges.map(Math.fround);
        if (inner.every((edge, index) => edge === expected[index])) {
          expect(result.classCounts.slice(0, reference.classCount), label).toEqual(
            reference.exactClassSizes
          );
        } else {
          // A neighbouring local optimum (see NATURAL_BREAKS_LOCAL_OPTIMA), also where f32 rounding
          // on the GPU resolved a near tie differently from the CPU mirror: allowed only within the
          // documented goodness of variance fit.
          const shortfall =
            reference.exactGoodnessOfVarianceFit - getGoodnessOfVarianceFit(values, inner);
          localOptima.push(`${label} (${shortfall.toExponential(1)})`);
          expect(shortfall, label).toBeLessThanOrEqual(NATURAL_BREAKS_LOCAL_OPTIMUM_FIT_TOLERANCE);
          if (!NATURAL_BREAKS_LOCAL_OPTIMA[label]) {
            unexpectedLocalOptima.push(label);
          }
        }
      }
    } finally {
      harness.destroy();
    }
  }
  expect(mismatches).toEqual([]);
  // Beyond the documented local optima the GPU must find the exact partition; log any other.
  console.log(`natural breaks local optima: ${localOptima.join(', ') || 'none'}`);
  expect(unexpectedLocalOptima).toEqual([]);
}, 120000);

it('GPUClassBreaks natural breaks handle ties, outliers, constant columns and missing rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const run = async (
    values: Float32Array,
    classCount: number,
    mask?: Uint32Array,
    maximumClassCount = MAXIMUM_CLASS_COUNT
  ) => {
    const harness = createHarness(device, {
      rows: values.length,
      withMask: Boolean(mask),
      naturalBreaksBinCount: 2048,
      methods: ['natural-breaks'],
      maximumClassCount
    });
    try {
      return await harness.run(values, mask, {method: 'natural-breaks', classCount});
    } finally {
      harness.destroy();
    }
  };
  // A constant column is one class, whatever was requested.
  const constant = await run(new Float32Array(200).fill(4.5), 5);
  expect(constant.classCount).toBe(1);
  expect(constant.breaks.slice(0, 2)).toEqual([4.5, 4.5]);
  // A class count above the distinct values gives one class per distinct value.
  for (const name of ['plateaus', 'two-values']) {
    const scene = CLASS_BREAKS_REFERENCE.scenes[name];
    const result = await run(Float32Array.from(scene.values), MAXIMUM_CLASS_COUNT);
    const distinct = scene.distinctValueEdges!.map(Math.fround);
    expect(result.classCount, name).toBe(distinct.length + 1);
    expect(result.breaks.slice(1, result.classCount), name).toEqual(distinct);
  }
  // Heavy ties, k = 2 and a single extreme outlier against the exact dynamic program.
  const random = createRandom(31);
  const ties = new Float32Array(900).map(() => Math.floor(random() * 11));
  const outlier = new Float32Array(500).map(() => 10 + (random() + random() + random() - 1.5) * 2);
  outlier[123] = 1e7;
  for (const [label, values, classCounts] of [
    ['ties', ties, [2, 3, 5, 7]],
    ['outlier', outlier, [2, 3, 4, 5, 6, 7]]
  ] as const) {
    for (const classCount of classCounts) {
      const exact = computeExactNaturalBreaks(values, classCount);
      const result = await run(values, classCount);
      expect(result.classCount, `${label} ${classCount}`).toBe(classCount);
      const inner = Array.from(result.breaks.slice(1, classCount));
      if (exact.unique) {
        expect(inner, `${label} ${classCount}`).toEqual(exact.edges.map(Math.fround));
      }
    }
  }
  // NaN, infinities and masked rows are missing: the result is that of the remaining column.
  const clean = Float32Array.from(CLASS_BREAKS_REFERENCE.scenes['clusters'].values);
  const noisy = new Float32Array(clean.length * 2);
  const mask = new Uint32Array(noisy.length);
  clean.forEach((value, row) => {
    noisy[2 * row] = value;
    mask[2 * row] = 1;
    // Between every two rows: a decoy (masked), and every few rows NaN or an infinity.
    noisy[2 * row + 1] = [1e6, NaN, Infinity, -Infinity][row % 4];
    mask[2 * row + 1] = row % 4 === 0 ? 0 : 1;
  });
  const cleanResult = await run(clean, 4);
  const noisyResult = await run(noisy, 4, mask);
  expect(noisyResult.breakBits).toEqual(cleanResult.breakBits);
  expect(noisyResult.classCount).toBe(4);
}, 120000);

it('GPUClassBreaks natural breaks A/B: O(k B^2) binned path against divide and conquer with refinement', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Same device, same column, same run: the previous pipeline (test-only legacy-natural-breaks.ts)
  // against the current one, at 1024 and 2048 bins, k = 8, 2^20 rows. Logs the time per frame and
  // the within-class cost of both partitions; the new cost must not be above the old one, whose
  // breaks snap to bin edges.
  const rows = 1 << 20;
  const classCount = 8;
  const values = createColumn(77, rows);
  const frameCount = 20;
  for (const binCount of [1024, 2048]) {
    const harness = createHarness(device, {
      rows,
      withMask: false,
      naturalBreaksBinCount: binCount,
      methods: ['natural-breaks'],
      maximumClassCount: classCount
    });
    const valuesBuffer = createInputBuffer(device, values);
    const breaksBuffer = createOutputBuffer(device, classCount + 1);
    const legacyGraph = new GPUCommandGraph(device, {id: `legacy-${binCount}`});
    legacyGraph.add(
      new LegacyNaturalBreaks({
        id: 'legacy',
        values: importGraphBuffer(legacyGraph, 'legacy-values', valuesBuffer, 'float32', rows),
        classCount,
        binCount,
        breaks: importGraphBuffer(
          legacyGraph,
          'legacy-breaks',
          breaksBuffer,
          'float32',
          classCount + 1
        )
      })
    );
    const legacy = legacyGraph.compile();
    try {
      const frame: GPUClassBreaksSettings = {method: 'natural-breaks', classCount};
      const current = await harness.run(values, undefined, frame);
      submitGraph(device, legacy, undefined);
      const legacyBreaks = Array.from(await readFloat32(breaksBuffer, classCount + 1));
      // Warm up, then time whole frames (encode, submit, GPU) with one small readback each.
      await harness.submit(2);
      const start = performance.now();
      await harness.submit(frameCount);
      const currentTime = (performance.now() - start) / frameCount;
      for (let index = 0; index < 2; index++) {
        submitGraph(device, legacy, undefined);
      }
      await readFloat32(breaksBuffer, 1);
      const legacyStart = performance.now();
      for (let index = 0; index < frameCount; index++) {
        submitGraph(device, legacy, undefined);
      }
      await readFloat32(breaksBuffer, 1);
      const legacyTime = (performance.now() - legacyStart) / frameCount;
      const currentCost = getWithinClassCost(values, current.breaks.slice(1, classCount));
      const legacyCost = getWithinClassCost(values, legacyBreaks.slice(1, classCount));
      console.log(
        `natural breaks A/B ${binCount} bins, k=${classCount}, ${rows} rows: ` +
          `old ${legacyTime.toFixed(2)} ms/frame, new ${currentTime.toFixed(2)} ms/frame, ` +
          `within-class cost old ${legacyCost.toExponential(6)} new ${currentCost.toExponential(6)}`
      );
      expect(current.classCount).toBe(classCount);
      expect(currentCost).toBeLessThanOrEqual(legacyCost * (1 + 1e-9));
    } finally {
      legacy.destroy();
      valuesBuffer.destroy();
      breaksBuffer.destroy();
      harness.destroy();
    }
  }
}, 240000);

it('GPUClassBreaks follows the class breaks edge policy', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const maximumClassCount = 4;
  for (const row of CLASS_BREAKS_EDGE_POLICY) {
    const values = Float32Array.from(row.values);
    const harness = createHarness(device, {
      rows: values.length,
      withMask: false,
      naturalBreaksBinCount: 16,
      maximumClassCount,
      // Explicitly compile only the methods the policy rows use. Omitted methods would also
      // select the other compatible routes, excluding box-plot at this four-class capacity.
      methods: ['equal-interval', 'quantile', 'standard-deviation', 'custom']
    });
    try {
      // Pack a valid frame (in-range counts as given, so quantile probabilities are packed), then
      // write the raw class count: out-of-range rows test the GPU clamp.
      const isInRange = row.classCount >= 1 && row.classCount <= maximumClassCount;
      const parameters = getGPUClassBreaksParameterValues(
        {
          method: row.method,
          classCount: isInRange ? row.classCount : 1,
          // NaN edges are written raw below: the packer refuses them.
          customEdges: row.customEdges?.map(edge => (Number.isNaN(edge) ? 0 : edge))
        },
        maximumClassCount
      );
      parameters[1] = row.classCount;
      row.customEdges?.forEach((edge, index) => {
        parameters[GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH + index] = edge;
      });
      if (row.standardDeviationInterval !== undefined) {
        parameters[2] = row.standardDeviationInterval;
      }
      const result = await harness.run(values, undefined, parameters);
      expect(result.classCount, row.name).toBe(row.expectedClassCount);
      for (const [index, expected] of row.expectedBreaks.entries()) {
        expect(
          Object.is(result.breaks[index], expected),
          `${row.name} e[${index}]: ${result.breaks[index]}`
        ).toBe(true);
      }
    } finally {
      harness.destroy();
    }
  }
});

it('GPUClassBreaks exact mode exhaustively optimizes distinct partitions and replays changed inputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, {
    rows: 8,
    withMask: true,
    naturalBreaksBinCount: 2,
    naturalBreaksMode: 'exact',
    methods: ['natural-breaks'],
    maximumClassCount: 4
  });
  // Independent exhaustive partition enumeration, using direct centered double sums.
  const oracle = (values: number[], classes: number): number[] => {
    const sorted = values.filter(Number.isFinite).sort((left, right) => left - right);
    let best = Infinity;
    let result: number[] = [];
    const visit = (start: number, cuts: number[]) => {
      if (cuts.length === classes - 1) {
        let cost = 0;
        let begin = 0;
        for (const end of [...cuts, sorted.length]) {
          const group = sorted.slice(begin, end);
          const mean = group.reduce((sum, value) => sum + value, 0) / group.length;
          cost += group.reduce((sum, value) => sum + (value - mean) ** 2, 0);
          begin = end;
        }
        if (cost < best) {
          best = cost;
          result = cuts.map(cut => sorted[cut]);
        }
        return;
      }
      for (let split = start; split < sorted.length; split++) {
        if (sorted[split] !== sorted[split - 1]) {
          visit(split + 1, [...cuts, split]);
        }
      }
    };
    visit(1, []);
    return result;
  };
  try {
    for (const source of [
      [0, 1, 2, 9, 10, 12, 30, 31],
      [0, 0, 1, 1, 10, 10, 11, 11],
      [-20, -19, -18, 0, 4, 30, 33, 35]
    ]) {
      const values = new Float32Array(source);
      for (const classCount of [2, 3, 4]) {
        const result = await harness.run(values, undefined, {method: 'natural-breaks', classCount});
        expect(result.breaks.slice(1, classCount)).toEqual(oracle(source, classCount));
        expect(result.classCounts.reduce((sum, count) => sum + count, 0)).toBe(8);
      }
    }
    // Scaling must not turn ordinary tiny columns into zero-cost ties or overflow huge columns.
    for (const scale of [1e-37, 1e-20, 1e20, 1e37]) {
      const scaled = Float32Array.from([0, 1, 2, 9, 10, 12, 30, 31], value => value * scale);
      const result = await harness.run(scaled, undefined, {
        method: 'natural-breaks',
        classCount: 3
      });
      expect(result.breaks.slice(1, 3)).toEqual([scaled[3], scaled[6]]);
    }
    const offset = Float32Array.from([0, 1, 2, 9, 10, 12, 30, 31], value => 1e7 + value);
    const offsetResult = await harness.run(offset, undefined, {
      method: 'natural-breaks',
      classCount: 3
    });
    expect(offsetResult.breaks.slice(1, 3)).toEqual([offset[3], offset[6]]);
    const extreme = new Float32Array([
      -3.4028234663852886e38, -3.4028234663852886e38, 0, 0, 0, 0, 3.4028234663852886e38,
      3.4028234663852886e38
    ]);
    const extremeResult = await harness.run(extreme, undefined, {
      method: 'natural-breaks',
      classCount: 3
    });
    expect(extremeResult.breaks.slice(1, 3)).toEqual([0, extreme[7]]);
    const masked = await harness.run(new Float32Array(8).fill(1), new Uint32Array(8), {
      method: 'natural-breaks',
      classCount: 3
    });
    expect(masked.classCount).toBe(0);
    const tiedValues = new Float32Array([0, 1, 2, NaN, NaN, NaN, NaN, NaN]);
    const tied = await harness.run(tiedValues, undefined, {
      method: 'natural-breaks',
      classCount: 2
    });
    expect(tied.breaks.slice(0, 3)).toEqual([0, 1, 2]);
    expect(
      (await harness.run(tiedValues, undefined, {method: 'natural-breaks', classCount: 2}))
        .breakBits
    ).toEqual(tied.breakBits);
    const special = await harness.run(
      new Float32Array([NaN, Infinity, -Infinity, 1, 1, 2, 2, 100]),
      new Uint32Array([1, 1, 1, 1, 1, 1, 1, 0]),
      {method: 'natural-breaks', classCount: 4}
    );
    expect(special.classCount).toBe(2);
    expect(special.breaks.slice(0, 3)).toEqual([1, 2, 2]);
    expect(special.classCounts).toEqual([3, 3, 0, 0]);
    const empty = await harness.run(new Float32Array(8).fill(NaN), undefined, {
      method: 'natural-breaks',
      classCount: 4
    });
    expect(empty.classCount).toBe(0);
    const constant = await harness.run(new Float32Array(8).fill(7), undefined, {
      method: 'natural-breaks',
      classCount: 4
    });
    expect(constant.classCount).toBe(1);
    expect(constant.classCounts).toEqual([8, 0, 0, 0]);
  } finally {
    harness.destroy();
  }
});

it('GPUClassBreaks exact mode escapes known approximate local minima', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [name, classes] of [
    ['lognormal', 7],
    ['outlier-pair', 5]
  ] as const) {
    const scene = CLASS_BREAKS_REFERENCE.scenes[name];
    const values = Float32Array.from(scene.values);
    const harness = createHarness(device, {
      rows: values.length,
      withMask: false,
      naturalBreaksBinCount: 2,
      naturalBreaksMode: 'exact',
      methods: ['natural-breaks'],
      maximumClassCount: classes
    });
    try {
      const result = await harness.run(values, undefined, {
        method: 'natural-breaks',
        classCount: classes
      });
      const reference = scene.naturalBreaks.find(item => item.classCount === classes)!;
      expect(result.breaks.slice(1, classes)).toEqual(reference.exactLowerEdges);
      expect(result.classCounts).toEqual(reference.exactClassSizes);
    } finally {
      harness.destroy();
    }
  }
});

it('GPUClassBreaks exact mode runs the maximum row and class capacities', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = Float32Array.from(
    {length: 1024},
    (_, row) => Math.floor(row / 16) * 1000 + (row % 16)
  );
  const harness = createHarness(device, {
    rows: 1024,
    withMask: false,
    naturalBreaksBinCount: 2,
    naturalBreaksMode: 'exact',
    methods: ['natural-breaks'],
    maximumClassCount: 64
  });
  try {
    const result = await harness.run(values, undefined, {method: 'natural-breaks', classCount: 64});
    expect(result.classCount).toBe(64);
    expect(result.breaks.slice(1, 64)).toEqual(
      Array.from({length: 63}, (_, index) => (index + 1) * 1000)
    );
    expect(result.classCounts).toEqual(new Array(64).fill(16));
  } finally {
    harness.destroy();
  }
});

it('GPUClassBreaks recomputed interval costs preserve exact partitions, ties, and changed inputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const options = {
    rows: 16,
    withMask: true,
    naturalBreaksBinCount: 2,
    naturalBreaksMode: 'exact' as const,
    methods: ['natural-breaks'] as GPUClassBreaksMethod[],
    maximumClassCount: 4
  };
  const materialized = createHarness(device, {
    ...options,
    naturalBreaksExactStorage: 'materialized'
  });
  const recomputed = createHarness(device, {...options, naturalBreaksExactStorage: 'recompute'});
  try {
    const sources = [
      Array.from({length: 16}, (_, row) => (row < 8 ? row : 100 + row)),
      [0, 1, 2, 0, 1, 2, -0, 0, 1, 2, NaN, Infinity, -Infinity, 1, 2, 3],
      Array.from({length: 16}, (_, row) => Math.fround(1e7 + row)),
      Array.from({length: 16}, (_, row) => Math.fround((row - 8) * 1e-37)),
      Array.from({length: 16}, (_, row) => Math.fround((row - 8) * 1e37)),
      new Array(16).fill(NaN),
      new Array(16).fill(7)
    ];
    for (const source of sources) {
      const values = Float32Array.from(source);
      for (const classCount of [2, 4, 3]) {
        const settings = {method: 'natural-breaks' as const, classCount};
        const expected = await materialized.run(values, undefined, settings);
        const actual = await recomputed.run(values, undefined, settings);
        expect(actual.breakBits).toEqual(expected.breakBits);
        expect(actual.classCount).toBe(expected.classCount);
        expect(actual.classCounts).toEqual(expected.classCounts);
      }
    }
  } finally {
    materialized.destroy();
    recomputed.destroy();
  }
});

it('GPUClassBreaks exact auto storage handles more than 1024 rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, {
    rows: 1025,
    withMask: false,
    naturalBreaksBinCount: 2,
    naturalBreaksMode: 'exact',
    methods: ['natural-breaks'],
    maximumClassCount: 2
  });
  try {
    const values = Float32Array.from({length: 1025}, (_, row) =>
      row < 512 ? row / 1024 : 100 + row / 1024
    );
    const actual = await harness.run(values, undefined, {method: 'natural-breaks', classCount: 2});
    expect(actual.breaks[1]).toBe(values[512]);
    expect(actual.classCounts).toEqual([512, 513]);
  } finally {
    harness.destroy();
  }
});

it('GPUClassBreaks monotone mode preserves data boundaries across large and degenerate frames', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The final tree level crosses the default 65535-workgroup dispatch dimension.
  const rows = 65537;
  const harness = createHarness(device, {
    rows,
    withMask: true,
    naturalBreaksBinCount: 2,
    naturalBreaksMode: 'monotone',
    methods: ['natural-breaks'],
    maximumClassCount: 4
  });
  try {
    const values = Float32Array.from({length: rows}, (_, row) =>
      row < 32768 ? row / rows : 100 + row / rows
    );
    const actual = await harness.run(values, undefined, {method: 'natural-breaks', classCount: 2});
    expect(actual.breaks[1]).toBe(values[32768]);
    expect(actual.classCounts).toEqual([32768, 32769, 0, 0]);
    const tied = new Float32Array(rows).fill(7);
    const constant = await harness.run(tied, undefined, {method: 'natural-breaks', classCount: 4});
    expect(constant.classCount).toBe(1);
    expect(constant.classCounts).toEqual([rows, 0, 0, 0]);
    const mask = new Uint32Array(rows);
    const empty = await harness.run(tied, mask, {method: 'natural-breaks', classCount: 4});
    expect(empty.classCount).toBe(0);
    // Reuse a large source allocation with only a handful of finite, selected rows.
    values.fill(NaN);
    values.set([-0, 0, 1, 1, 10, 10, 11, 11]);
    const distinct = await harness.run(values, undefined, {
      method: 'natural-breaks',
      classCount: 4
    });
    // Numeric zeros form one class; the shared integer-key minimum preserves the input -0 bits.
    expect(distinct.breaks.slice(0, 5)).toEqual([-0, 1, 10, 11, 11]);
    expect(distinct.classCounts).toEqual([2, 2, 2, 2]);
  } finally {
    harness.destroy();
  }
});

// DPC-111. For float32 [-3e38, 0, 3e38] the span 6e38 is past the float32 maximum. A float32
// replay of `min + ((max - min) / k) * i` gives an infinite width, every inner edge clamped to the
// maximum and counts [2, 1] for k = 2. Expected (float64, then fround; the CPU twin agrees):
// k = 2 has the middle edge 0 and counts [1, 2] under the lower-edge rule; k = 4 has the edges
// -1.5e38, 0, 1.5e38 and counts [1, 0, 1, 1].
it('GPUClassBreaks equal interval keeps its cuts when finite extrema span past float32', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, {
    rows: 3,
    withMask: false,
    naturalBreaksBinCount: 64,
    methods: ['equal-interval']
  });
  try {
    const values = Float32Array.from([-3e38, 0, 3e38]);
    const minimum = Math.fround(-3e38);
    const maximum = Math.fround(3e38);
    const two = await harness.run(values, undefined, {method: 'equal-interval', classCount: 2});
    expect(two.classCount).toBe(2);
    expect(two.breaks.slice(0, 3)).toEqual([minimum, 0, maximum]);
    expect(two.classCounts.slice(0, 2)).toEqual([1, 2]);
    const four = await harness.run(values, undefined, {method: 'equal-interval', classCount: 4});
    expect(four.classCount).toBe(4);
    const expectedEdges = [minimum, Math.fround(-1.5e38), 0, Math.fround(1.5e38), maximum];
    for (const [edge, expected] of expectedEdges.entries()) {
      const actual = four.breaks[edge];
      expect(Number.isFinite(actual), `edge ${edge} ${actual}`).toBe(true);
      expect(Math.abs(actual - expected), `edge ${edge} ${actual}`).toBeLessThanOrEqual(
        1e-6 * Math.abs(expected)
      );
    }
    expect(four.classCounts.slice(0, 4)).toEqual([1, 0, 1, 1]);
  } finally {
    harness.destroy();
  }
});

it('DPC-110: class breaks merges signed zeros in membership and maximum gaps', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const custom = createHarness(device, {
    rows: 2,
    withMask: false,
    naturalBreaksBinCount: 64,
    methods: ['custom'],
    maximumClassCount: 2
  });
  const maximum = createHarness(device, {
    rows: 4,
    withMask: false,
    naturalBreaksBinCount: 64,
    methods: ['maximum-breaks'],
    maximumClassCount: 4
  });
  try {
    const counted = await custom.run(Float32Array.of(-0, +0), undefined, {
      method: 'custom',
      customEdges: [-1, +0, 1]
    });
    expect(counted.classCounts).toEqual([0, 2]);
    const distinct = await maximum.run(Float32Array.of(-1, -0, +0, 1), undefined, {
      method: 'maximum-breaks',
      classCount: 4
    });
    expect(distinct.classCount).toBe(3);
    expect(distinct.breaks).toEqual([-1, -0.5, 0.5, 1, NaN]);
    expect(distinct.classCounts).toEqual([1, 2, 1, 0]);
  } finally {
    custom.destroy();
    maximum.destroy();
  }
});

it('DPC-111: approximate natural breaks keeps finite cuts across a float32-overflowing span', async () => {
  // Two tight symmetric clusters, separated by a gap 56 times their width: the unique two-class
  // least-squares partition separates the signs (within-cluster SSD = 4 * (1e37)^2).
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, {
    rows: 6,
    withMask: false,
    naturalBreaksBinCount: 64,
    methods: ['natural-breaks'],
    maximumClassCount: 2
  });
  try {
    const values = Float32Array.of(-3e38, -2.9e38, -2.8e38, 2.8e38, 2.9e38, 3e38);
    const result = await harness.run(values, undefined, {method: 'natural-breaks', classCount: 2});
    expect(result.classCount).toBe(2);
    expect(result.breaks.every(Number.isFinite)).toBe(true);
    expect(result.breaks).toEqual([values[0], values[3], values[5]]);
    expect(result.classCounts).toEqual([3, 3]);
    // A stale 2^-64 scale erases this entire normal-float column. Both clusters must still
    // classify, and refinement must publish the actual smallest value of the upper cluster.
    const tinyValues = Float32Array.from([-3, -2.9, -2.8, 2.8, 2.9, 3], value => value * 2 ** -90);
    const ordinary = await harness.run(tinyValues, undefined, {
      method: 'natural-breaks',
      classCount: 2
    });
    expect(ordinary.classCounts).toEqual([3, 3]);
    expect(ordinary.breaks.every(Number.isFinite)).toBe(true);
    expect(ordinary.breaks).toEqual([tinyValues[0], tinyValues[3], tinyValues[5]]);
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }
});

it('DPC-110 ordering: retains adjacent nonzero floats and excludes NaN and masked rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const adjacent = 1 + 2 ** -23;
  const harness = createHarness(device, {
    rows: 8,
    withMask: true,
    naturalBreaksBinCount: 64,
    methods: ['custom'],
    maximumClassCount: 3
  });
  try {
    const result = await harness.run(
      Float32Array.of(-adjacent, -1, -0, +0, 1, adjacent, NaN, adjacent),
      Uint32Array.of(1, 1, 1, 1, 1, 1, 1, 0),
      {method: 'custom', customEdges: [-2, -1, adjacent, 2]}
    );
    // Hand membership: -adjacent below -1; -1, both zeros and 1 in the middle; adjacent upper.
    expect(result.classCounts).toEqual([1, 4, 1]);
  } finally {
    harness.destroy();
  }
});

it('B1: approximate natural breaks merges signed-zero classes before publishing', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, {
    rows: 4,
    withMask: false,
    naturalBreaksBinCount: 4,
    methods: ['natural-breaks'],
    maximumClassCount: 4
  });
  try {
    const result = await harness.run(Float32Array.of(-2, -0, +0, 2), undefined, {
      method: 'natural-breaks',
      classCount: 4
    });
    // Three numeric distinct values have a zero-cost partition; both zeros share one class.
    expect(result.classCount).toBe(3);
    expect(result.classCounts).toEqual([1, 2, 1, 0]);
    expect(result.breaks.slice(0, 4).map(value => (value === 0 ? 0 : value))).toEqual([
      -2, 0, 2, 2
    ]);
    expect(result.breaks[4]).toBeNaN();
  } finally {
    harness.destroy();
  }
});

it('B2: approximate natural breaks preserves the tiny bulk beside a huge normal outlier', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const tiny = 2 ** -90;
  const large = 2 ** 70;
  const harness = createHarness(device, {
    rows: 4,
    withMask: false,
    naturalBreaksBinCount: 16,
    methods: ['natural-breaks'],
    maximumClassCount: 3
  });
  try {
    const result = await harness.run(Float32Array.of(tiny, tiny, 2 * tiny, large), undefined, {
      method: 'natural-breaks',
      classCount: 3
    });
    // Each distinct value has its own class: unique zero-SSD partition, counts [2,1,1].
    expect(result.classCount).toBe(3);
    expect(result.classCounts).toEqual([2, 1, 1]);
    expect(result.breaks.every(Number.isFinite)).toBe(true);
    expect(result.breaks).toEqual([tiny, 2 * tiny, large, large]);
  } finally {
    harness.destroy();
  }
});

it('B2: equal intervals retain the zero cut with three classes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, {
    rows: 4,
    withMask: false,
    naturalBreaksBinCount: 64,
    methods: ['equal-interval'],
    maximumClassCount: 3
  });
  try {
    const result = await harness.run(Float32Array.of(-2, -(2 ** -58), 0, 1), undefined, {
      method: 'equal-interval',
      classCount: 3
    });
    // Exact unit width: the tiny negative value belongs below the zero cut.
    expect(result.classCount).toBe(3);
    expect(result.breaks).toEqual([-2, -1, 0, 1]);
    expect(result.classCounts).toEqual([1, 1, 2]);
  } finally {
    harness.destroy();
  }
});

it.each([
  {
    values: [1, 1.0000001192092896],
    edges: [1, 1.0000001192092896, 1.0000001192092896],
    bits: [0x3f800000, 0x3f800001, 0x3f800001]
  },
  // Python struct.unpack('<f', struct.pack('<I', bits)) for 0x08800000 and 0x08800001.
  {
    values: [7.703719777548943e-34, 7.703720695903905e-34],
    edges: [7.703719777548943e-34, 7.703720695903905e-34, 7.703720695903905e-34],
    bits: [0x08800000, 0x08800001, 0x08800001]
  }
])('W5-01: maximum breaks separates adjacent float32 observations: %j', async fixture => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, {
    rows: 2,
    withMask: false,
    naturalBreaksBinCount: 64,
    methods: ['maximum-breaks'],
    maximumClassCount: 2
  });
  try {
    const result = await harness.run(Float32Array.from(fixture.values), undefined, {
      method: 'maximum-breaks',
      classCount: 2
    });
    // Exact midpoint 1 + 2^-24 rounds to the lower row; lower-edge membership needs the upper.
    expect(result.classCount).toBe(2);
    expect(result.breaks).toEqual(fixture.edges);
    expect(result.breakBits).toEqual(fixture.bits);
    expect(result.classCounts).toEqual([1, 1]);
  } finally {
    harness.destroy();
  }
});

it('WGS-05: maximum breaks keeps extreme midpoints finite and unequal large gaps ordered', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const values of [
    Float32Array.of(-(2 ** 127), 2 ** 127),
    Float32Array.of(-3 * 2 ** 126, 3 * 2 ** 126),
    Float32Array.of(-3 * 2 ** 126, -(2 ** 127), 2 ** 127, 3 * 2 ** 126),
    // The largest gap is 2^128; the next is 2^127. Literal midpoint is -2^126.
    Float32Array.of(-3 * 2 ** 126, 2 ** 126, 3 * 2 ** 126)
  ]) {
    const harness = createHarness(device, {
      rows: values.length,
      withMask: false,
      naturalBreaksBinCount: 2048,
      methods: ['maximum-breaks'],
      maximumClassCount: 2
    });
    try {
      const result = await harness.run(values, undefined, {
        method: 'maximum-breaks',
        classCount: 2
      });
      expect(result.breaks.every(Number.isFinite)).toBe(true);
      expect(result.breaks[1]).toBe(values.length === 3 ? -(2 ** 126) : 0);
      expect(result.classCounts).toEqual(
        values.length === 4 ? [2, 2] : values.length === 3 ? [1, 2] : [1, 1]
      );
    } finally {
      harness.destroy();
    }
  }
});

it('CMP-05: a four-row density-grid view ignores its spare allocation tail', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A 2x2 raster's logical prefix over the five-float producer allocation.
  for (const rows of [4, 5]) {
    const harness = createHarness(device, {
      rows,
      allocationRows: 5,
      logicalGridSize: rows === 4 ? [2, 2] : undefined,
      withMask: false,
      naturalBreaksBinCount: 2048,
      methods: ['equal-interval'],
      maximumClassCount: 2
    });
    try {
      const result = await harness.run(Float32Array.of(2, 4, 6, 8, 1000), undefined, {
        method: 'equal-interval',
        classCount: 2
      });
      expect(result.breaks.every(Number.isFinite)).toBe(true);
      expect(result.breaks).toEqual(rows === 4 ? [2, 5, 8] : [2, 501, 1000]);
      expect(result.classCounts).toEqual(rows === 4 ? [2, 2] : [4, 1]);
    } finally {
      harness.destroy();
    }
  }
});

it('DEF-06: reads default and exact natural-breaks edges for the twelve-value provenance fixture', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = Float32Array.of(174, 15, 79, 177, 157, 114, 36, 56, 179, 166, 104, 100000);
  for (const naturalBreaksMode of [undefined, 'exact'] as const) {
    const harness = createHarness(device, {
      rows: 12,
      withMask: false,
      maximumClassCount: 4,
      naturalBreaksBinCount: 2048,
      methods: ['natural-breaks'],
      naturalBreaksMode
    });
    try {
      const result = await harness.run(values, undefined, {
        method: 'natural-breaks',
        classCount: 4
      });
      expect(result.breaks.every(Number.isFinite)).toBe(true);
      // Default [104,157,100000] is the approximate host-mirror observation until this GPU case runs.
      // Exact [79,157,100000] comes from independently enumerating Fisher-Jenks partitions.
      expect(result.breaks.slice(1, 4)).toEqual(
        naturalBreaksMode === 'exact' ? [79, 157, 100000] : [104, 157, 100000]
      );
    } finally {
      harness.destroy();
    }
  }
});

it('R1 B6: reads distinct production keys for unequal overflowing gaps across two runs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, {
    rows: 2,
    maximumClassCount: 2,
    withMask: false,
    methods: ['maximum-breaks'],
    naturalBreaksBinCount: 2048,
    captureGapKeys: true
  });
  try {
    const observed: number[] = [];
    // Widths 2.25*2^127 = 1.125*2^128 and 3*2^127 = 1.5*2^128.
    // Extended positive exponent encodings 0x7f900000/0x7fc00000 invert to the literals below.
    for (const [lower, expectedKey, midpoint] of [
      [-(2 ** 127), 0x806fffff, 2 ** 124],
      [-1.75 * 2 ** 127, 0x803fffff, -(2 ** 125)]
    ]) {
      const result = await harness.run(Float32Array.of(lower, 1.25 * 2 ** 127), undefined, {
        method: 'maximum-breaks',
        classCount: 2
      });
      expect(result.gapKeys).toEqual([0xffffffff, expectedKey]);
      expect(Number.isFinite(result.breaks[1])).toBe(true);
      expect(result.breaks[1]).toBe(midpoint);
      expect(result.classCounts).toEqual([1, 1]);
      observed.push(result.gapKeys![1]);
    }
    expect(observed[1]).toBeLessThan(observed[0]);
  } finally {
    harness.destroy();
  }
});

it('R1 B3: one-class defaults use exactly-sized public parameters', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, {
    rows: 3,
    maximumClassCount: 1,
    withMask: false,
    naturalBreaksBinCount: 2048
  });
  try {
    const result = await harness.run(Float32Array.of(0, 2, 4), undefined, {
      method: 'equal-interval',
      classCount: 1
    });
    expect(result.breaks.every(Number.isFinite)).toBe(true);
    expect(result.breaks).toEqual([0, 4]);
    expect(result.classCounts).toEqual([3]);
  } finally {
    harness.destroy();
  }
});
