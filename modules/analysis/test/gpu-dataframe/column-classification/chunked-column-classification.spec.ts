// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUScalarScale} from '../../../src/dataframe-analysis/column-classification/gpu-scalar-scale';
import {GPUColorScale} from '../../../src/dataframe-analysis/column-classification/gpu-color-scale';
import {GPUBivariateClassification} from '../../../src/dataframe-analysis/column-classification/gpu-bivariate-classification';
import {GPUColumnQuantiles} from '../../../src/dataframe-analysis/column-classification/gpu-column-quantiles';
import {GPUClassBreaks} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import {getGPUScalarScaleParameterValues} from '../../../src/dataframe-analysis/column-classification/scalar-scale-parameters';
import {getGPUColorScaleParameterValues} from '../../../src/dataframe-analysis/column-classification/color-scale-parameters';
import {getGPUBivariateClassificationParameterValues} from '../../../src/dataframe-analysis/column-classification/bivariate-classification-parameters';
import {
  getGPUColumnQuantilesParameterValues,
  getGPUColumnQuantilesRationalProbabilityValues,
  type GPUColumnQuantileInterpolation
} from '../../../src/dataframe-analysis/column-classification/column-quantiles-parameters';
import {
  getGPUClassBreaksParameterValues,
  type GPUClassBreaksMethod
} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {computeColumnQuantilesOracle} from './column-quantiles-oracle';
import {countClassesOracle} from './class-breaks-oracle';
import {computeColorScaleOnCPU} from './color-scale-oracle';
import {computeBivariateClassificationOnCPU} from './bivariate-classification-oracle';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

const GUARD_WORD = 0xdeadbeef;
/** Forty million rows need a longer GPU execution allowance than the small parity fixtures. */
const LARGE_QUANTILE_TEST_TIMEOUT_MS = 60_000;

/** Uneven chunks, including empties and boundaries inside the fixed-order moment tiles. */
function createHarness(
  device: Device,
  length: number,
  chunkLengths = [1, 0, 3, 2047, 0, length - 2051]
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const inputs = new Map<string, Buffer>();
  const rowInputs = new Map<string, {buffer: Buffer; length: number}[]>();
  const outputs = new Map<string, {buffer: Buffer; length: number}[]>();
  const input = <Format extends 'float32' | 'uint32' | 'sint32'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array | Int32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    inputs.set(name, buffer);
    return importGraphBuffer(graph, name, buffer, format);
  };
  const inputRows = <Format extends 'float32' | 'uint32' | 'sint32'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array | Int32Array,
    chunked: boolean
  ) => {
    let rowOffset = 0;
    const records: {buffer: Buffer; length: number}[] = [];
    const chunks = (chunked ? chunkLengths : [length]).map((chunkLength, chunkIndex) => {
      const padded =
        format === 'float32'
          ? new Float32Array(chunkLength + 2)
          : format === 'uint32'
            ? new Uint32Array(chunkLength + 2)
            : new Int32Array(chunkLength + 2);
      padded.set(values.subarray(rowOffset, rowOffset + chunkLength), 1);
      rowOffset += chunkLength;
      const view = input(`${name}-${chunkIndex}`, format, padded);
      records.push({buffer: inputs.get(`${name}-${chunkIndex}`)!, length: chunkLength});
      return graph.createDataView(view.buffer, {format, length: chunkLength, byteOffset: 4});
    });
    rowInputs.set(name, records);
    return chunked ? createVectorView(name, format, chunks) : chunks[0];
  };
  const output = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    lengths: number[],
    chunked: boolean
  ) => {
    const records = lengths.map((chunkLength, chunkIndex) => {
      const buffer = createOutputBuffer(device, chunkLength + 2);
      buffer.write(new Uint32Array(chunkLength + 2).fill(GUARD_WORD));
      buffers.push(buffer);
      const view = importGraphBuffer(graph, `${name}-${chunkIndex}`, buffer, format);
      return {
        buffer,
        length: chunkLength,
        view: graph.createDataView(view.buffer, {format, length: chunkLength, byteOffset: 4})
      };
    });
    outputs.set(name, records);
    return chunked
      ? createVectorView(
          name,
          format,
          records.map(record => record.view)
        )
      : records[0].view;
  };
  const packedOutput = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    count: number
  ) => output(name, format, [count], false) as ReturnType<typeof input<Format>>;
  const outputRows = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    chunked: boolean
  ) => output(name, format, chunked ? chunkLengths : [length], chunked);
  const readWords = async (name: string) => {
    const records = outputs.get(name)!;
    const words = new Uint32Array(records.reduce((sum, record) => sum + record.length, 0));
    let rowOffset = 0;
    for (const record of records) {
      const bytes = await record.buffer.readAsync();
      const padded = new Uint32Array(bytes.buffer, bytes.byteOffset, record.length + 2);
      expect(padded[0]).toBe(GUARD_WORD);
      expect(padded[record.length + 1]).toBe(GUARD_WORD);
      words.set(padded.subarray(1, record.length + 1), rowOffset);
      rowOffset += record.length;
    }
    return words;
  };
  const writeRows = (name: string, values: Float32Array | Uint32Array | Int32Array) => {
    let rowOffset = 0;
    for (const record of rowInputs.get(name)!) {
      const padded =
        values instanceof Float32Array
          ? new Float32Array(record.length + 2)
          : values instanceof Uint32Array
            ? new Uint32Array(record.length + 2)
            : new Int32Array(record.length + 2);
      padded.set(values.subarray(rowOffset, rowOffset + record.length), 1);
      record.buffer.write(padded);
      rowOffset += record.length;
    }
  };
  const writeInput = (name: string, values: Float32Array | Uint32Array) =>
    inputs.get(name)!.write(values);
  const checkParity = async (names: string[]) => {
    for (const name of names)
      expect(await readWords(`chunked-${name}`), name).toEqual(await readWords(`packed-${name}`));
  };
  return {
    graph,
    buffers,
    input,
    inputRows,
    packedOutput,
    outputRows,
    readWords,
    checkParity,
    writeRows,
    writeInput
  };
}

function createValues(length: number) {
  const values = Float32Array.from({length}, (_, row) => ((row * 47) % 997) / 8 - 30);
  values.set([NaN, Infinity, -Infinity, -0, 0, 3, 3], 0);
  return values;
}

it('GPUScalarScale produces bitwise packed results from chunked values, masks and outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const length = 4103;
  const harness = createHarness(device, length);
  const {graph, input, inputRows, outputRows} = harness;
  const values = createValues(length);
  const mask = Uint32Array.from({length}, (_, row) => Number(row % 11 !== 0));
  const domain = input('domain', 'float32', Float32Array.of(-30, 100));
  const range = input('range', 'float32', Float32Array.of(1, 10));
  const parameters = input(
    'parameters',
    'float32',
    getGPUScalarScaleParameterValues({scale: 'sqrt', domainCount: 2, nullValue: -99})
  );
  for (const chunked of [false, true]) {
    const route = chunked ? 'chunked' : 'packed';
    graph.add(
      new GPUScalarScale({
        id: route,
        values: inputRows(`${route}-source-values`, 'float32', values, chunked),
        mask: inputRows(`${route}-source-mask`, 'uint32', mask, chunked),
        domain,
        range,
        parameters,
        maximumDomainCount: 2,
        output: {
          values: outputRows(`${route}-values`, 'float32', chunked),
          validity: outputRows(`${route}-validity`, 'uint32', chunked)
        }
      })
    );
  }
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    await harness.checkParity(['values', 'validity']);
  } finally {
    compiled.destroy();
    harness.buffers.forEach(buffer => buffer.destroy());
  }
});

it.each([
  'float32',
  'uint32',
  'sint32'
] as const)('GPUColorScale accumulates chunked counts once and preserves %s packed results on replay', async format => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const length = 4103;
  const harness = createHarness(device, length);
  const {graph, input, inputRows, outputRows, packedOutput} = harness;
  const values =
    format === 'float32'
      ? createValues(length)
      : format === 'uint32'
        ? Uint32Array.from({length}, (_, row) => row % 101)
        : Int32Array.from({length}, (_, row) => (row % 101) - 50);
  const mask = Uint32Array.from({length}, (_, row) => Number(row % 13 !== 0));
  const domain = input('domain', 'float32', Float32Array.of(-50, 0, 100));
  const palette = input('palette', 'uint32', Uint32Array.of(0xff0000ff, 0xff00ff00, 0xffff0000));
  const parameters = input(
    'parameters',
    'float32',
    getGPUColorScaleParameterValues({
      scale: 'linear',
      domainCount: 3,
      paletteCount: 3,
      clamp: true,
      interpolation: 'linear'
    })
  );
  for (const chunked of [false, true]) {
    const route = chunked ? 'chunked' : 'packed';
    graph.add(
      new GPUColorScale({
        id: route,
        values: inputRows(`${route}-source-values`, format, values, chunked),
        integerValues: 'numeric',
        mask: inputRows(`${route}-source-mask`, 'uint32', mask, chunked),
        domain,
        palette,
        parameters,
        maximumDomainCount: 3,
        maximumPaletteCount: 3,
        output: {
          colors: outputRows(`${route}-colors`, 'uint32', chunked),
          classIndices: outputRows(`${route}-classes`, 'uint32', chunked),
          classCounts: packedOutput(`${route}-counts`, 'uint32', 3)
        }
      })
    );
  }
  const compiled = graph.compile();
  try {
    for (let frame = 0; frame < 4; frame++) {
      const frameMask =
        frame === 2
          ? new Uint32Array(length)
          : frame === 1
            ? Uint32Array.from({length}, (_, row) => Number(row % 3 === 0))
            : mask;
      for (const route of ['packed', 'chunked'])
        harness.writeRows(`${route}-source-mask`, frameMask);
      submitGraph(device, compiled);
      await harness.checkParity(['colors', 'classes', 'counts']);
      const expected = computeColorScaleOnCPU({
        values,
        mask: frameMask,
        domain: [-50, 0, 100],
        activeDomainCount: 3,
        palette: [0xff0000ff, 0xff00ff00, 0xffff0000],
        scale: 'linear',
        paletteCount: 3,
        interpolation: 'linear',
        clamp: true,
        maximumDomainCount: 3,
        maximumPaletteCount: 3
      });
      for (const route of ['packed', 'chunked'])
        expect(await harness.readWords(`${route}-counts`)).toEqual(expected.classCounts);
    }
  } finally {
    compiled.destroy();
    harness.buffers.forEach(buffer => buffer.destroy());
  }
});

it('GPUBivariateClassification preserves packed classes, alpha colours and counts from aligned chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const length = 4103;
  const harness = createHarness(device, length);
  const {graph, input, inputRows, outputRows, packedOutput} = harness;
  const values = createValues(length);
  const valuesY = Float32Array.from(values, value => -value);
  const mask = Uint32Array.from({length}, (_, row) => Number(row % 7 !== 0));
  const edges = input('edges', 'float32', Float32Array.of(-100, 0, 100));
  const palette = input(
    'palette',
    'uint32',
    Uint32Array.of(0xff0000ff, 0xff00ff00, 0xffff0000, 0xffffffff)
  );
  const parameters = input(
    'parameters',
    'float32',
    getGPUBivariateClassificationParameterValues({
      classCountX: 2,
      classCountY: 2,
      valueByAlpha: {domain: [-30, 100], minimumAlpha: 0.1}
    })
  );
  for (const chunked of [false, true]) {
    const route = chunked ? 'chunked' : 'packed';
    graph.add(
      new GPUBivariateClassification({
        id: route,
        valuesX: inputRows(`${route}-source-x`, 'float32', values, chunked),
        valuesY: inputRows(`${route}-source-y`, 'float32', valuesY, chunked),
        alphaValues: inputRows(`${route}-source-alpha`, 'float32', valuesY, chunked),
        mask: inputRows(`${route}-source-mask`, 'uint32', mask, chunked),
        breaksX: edges,
        breaksY: edges,
        palette,
        parameters,
        maximumClassCount: 2,
        output: {
          colors: outputRows(`${route}-colors`, 'uint32', chunked),
          classIds: outputRows(`${route}-classes`, 'uint32', chunked),
          classCounts: packedOutput(`${route}-counts`, 'uint32', 4)
        }
      })
    );
  }
  const compiled = graph.compile();
  try {
    for (let frame = 0; frame < 4; frame++) {
      const frameMask =
        frame === 2
          ? new Uint32Array(length)
          : frame === 1
            ? Uint32Array.from({length}, (_, row) => Number(row % 5 === 0))
            : mask;
      const classCountX = frame === 1 ? 1 : 2;
      for (const route of ['packed', 'chunked'])
        harness.writeRows(`${route}-source-mask`, frameMask);
      harness.writeInput(
        'parameters',
        getGPUBivariateClassificationParameterValues({
          classCountX,
          classCountY: 2,
          valueByAlpha: {domain: [-30, 100], minimumAlpha: 0.1}
        })
      );
      submitGraph(device, compiled);
      await harness.checkParity(['colors', 'classes', 'counts']);
      const expected = computeBivariateClassificationOnCPU({
        valuesX: values,
        valuesY,
        mask: frameMask,
        breaksX: [-100, 0, 100],
        breaksY: [-100, 0, 100],
        classCountX,
        classCountY: 2,
        maximumClassCount: 2,
        palette: [0xff0000ff, 0xff00ff00, 0xffff0000, 0xffffffff],
        alphaValues: valuesY,
        valueByAlpha: {domain: [-30, 100], minimumAlpha: 0.1}
      });
      for (const route of ['packed', 'chunked'])
        expect(await harness.readWords(`${route}-counts`)).toEqual(expected.classCounts);
    }
  } finally {
    compiled.destroy();
    harness.buffers.forEach(buffer => buffer.destroy());
  }
});

it.each<GPUColumnQuantileInterpolation>([
  'lower',
  'higher',
  'nearest',
  'linear',
  'midpoint'
])('GPUColumnQuantiles preserves packed %s quantiles and percentile masks across chunks', async interpolation => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const length = 4103;
  const harness = createHarness(device, length);
  const {graph, input, inputRows, outputRows, packedOutput} = harness;
  const values = createValues(length);
  const mask = Uint32Array.from({length}, (_, row) => Number(row % 9 !== 0));
  const parameters = input(
    'parameters',
    'float32',
    getGPUColumnQuantilesParameterValues({
      quantiles: [0, 0.25, 0.5, 0.95, 1],
      interpolation,
      filterRange: [0.15, 0.9]
    })
  );
  for (const chunked of [false, true]) {
    const route = chunked ? 'chunked' : 'packed';
    graph.add(
      new GPUColumnQuantiles({
        id: route,
        values: inputRows(`${route}-source-values`, 'float32', values, chunked),
        mask: inputRows(`${route}-source-mask`, 'uint32', mask, chunked),
        parameters,
        quantileCount: 5,
        output: {
          quantiles: packedOutput(`${route}-quantiles`, 'float32', 5),
          validCount: packedOutput(`${route}-count`, 'uint32', 1),
          outputMask: outputRows(`${route}-mask`, 'uint32', chunked),
          filterBounds: packedOutput(`${route}-bounds`, 'float32', 2)
        }
      })
    );
  }
  const compiled = graph.compile();
  try {
    for (let frame = 0; frame < 2; frame++) {
      submitGraph(device, compiled);
      await harness.checkParity(['quantiles', 'count', 'mask', 'bounds']);
      const oracle = computeColumnQuantilesOracle({
        values,
        mask,
        quantiles: [0, 0.25, 0.5, 0.95, 1],
        interpolation,
        filterRange: [0.15, 0.9]
      });
      expect(await harness.readWords('chunked-quantiles')).toEqual(
        new Uint32Array(oracle.quantiles.buffer)
      );
      expect(await harness.readWords('chunked-count')).toEqual(Uint32Array.of(oracle.validCount));
      expect(await harness.readWords('chunked-mask')).toEqual(oracle.outputMask);
      expect(await harness.readWords('chunked-bounds')).toEqual(
        new Uint32Array(oracle.filterBounds.buffer)
      );
    }
  } finally {
    compiled.destroy();
    harness.buffers.forEach(buffer => buffer.destroy());
  }
});

it.each<GPUClassBreaksMethod>([
  'equal-interval',
  'quantile',
  'box-plot',
  'custom',
  'standard-deviation',
  'head-tail',
  'natural-breaks'
])('GPUClassBreaks preserves packed %s edges and counts with tile-splitting chunks', async method => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const length = 4103;
  const harness = createHarness(device, length);
  const {graph, input, inputRows, packedOutput} = harness;
  const values = createValues(length);
  const mask = Uint32Array.from({length}, (_, row) => Number(row % 17 !== 0));
  const parameters = input(
    'parameters',
    'float32',
    getGPUClassBreaksParameterValues(
      {method, classCount: 6, customEdges: [-30, 0, 20, 40, 60, 80, 100]},
      6
    )
  );
  for (const chunked of [false, true]) {
    const route = chunked ? 'chunked' : 'packed';
    graph.add(
      new GPUClassBreaks({
        id: route,
        values: inputRows(`${route}-source-values`, 'float32', values, chunked),
        mask: inputRows(`${route}-source-mask`, 'uint32', mask, chunked),
        parameters,
        maximumClassCount: 6,
        methods: [method],
        naturalBreaksBinCount: 16,
        output: {
          breaks: packedOutput(`${route}-breaks`, 'float32', 7),
          classCount: packedOutput(`${route}-class-count`, 'uint32', 1),
          classCounts: packedOutput(`${route}-counts`, 'uint32', 6)
        }
      })
    );
  }
  const compiled = graph.compile();
  try {
    for (let frame = 0; frame < 2; frame++) {
      submitGraph(device, compiled);
      await harness.checkParity(['breaks', 'class-count', 'counts']);
      for (const route of ['packed', 'chunked']) {
        const classCount = (await harness.readWords(`${route}-class-count`))[0];
        expect(classCount).toBeGreaterThan(0);
        expect(classCount).toBeLessThanOrEqual(6);
        if (method === 'custom')
          expect(Array.from(await harness.readWords(`${route}-counts`))).toEqual([
            927, 626, 614, 617, 620, 457
          ]);
        const breakWords = await harness.readWords(`${route}-breaks`);
        expect(await harness.readWords(`${route}-counts`)).toEqual(
          countClassesOracle(values, mask, new Float32Array(breakWords.buffer), classCount, 6)
        );
      }
    }
  } finally {
    compiled.destroy();
    harness.buffers.forEach(buffer => buffer.destroy());
  }
});

it('scales and radix-selects two million chunked rows without a packed source binding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const length = 2_000_003;
  const harness = createHarness(device, length, [1_000_001, 0, 1_000_002]);
  const {graph, input, inputRows, outputRows, packedOutput} = harness;
  const values = Float32Array.from({length}, (_, row) => row % 1001);
  const source = inputRows('values', 'float32', values, true);
  graph.add(
    new GPUScalarScale({
      values: source,
      domain: input('domain', 'float32', Float32Array.of(0, 1000)),
      range: input('range', 'float32', Float32Array.of(0, 1000)),
      parameters: input(
        'scale-parameters',
        'float32',
        getGPUScalarScaleParameterValues({scale: 'identity', domainCount: 2})
      ),
      maximumDomainCount: 2,
      output: {values: outputRows('scaled', 'float32', true)}
    })
  );
  graph.add(
    new GPUColumnQuantiles({
      values: source,
      parameters: input(
        'quantile-parameters',
        'float32',
        getGPUColumnQuantilesParameterValues({quantiles: [0.5], interpolation: 'lower'})
      ),
      rationalProbabilities: input(
        'ratios',
        'uint32',
        getGPUColumnQuantilesRationalProbabilityValues([[1, 2]])
      ),
      quantileCount: 1,
      output: {
        quantiles: packedOutput('quantile', 'float32', 1),
        validCount: packedOutput('count', 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    const scaled = await harness.readWords('scaled');
    expect(scaled).toEqual(new Uint32Array(values.buffer));
    expect(await harness.readWords('count')).toEqual(Uint32Array.of(length));
    const quantile = await harness.readWords('quantile');
    expect(new Float32Array(quantile.buffer)[0]).toBe(500);
  } finally {
    compiled.destroy();
    harness.buffers.forEach(buffer => buffer.destroy());
  }
});

it('rejects a filter-mask chunk that would overwrite the median reduction', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, 4, [2, 2]);
  const {graph, input, inputRows, packedOutput} = harness;
  try {
    const shared = packedOutput('median-storage', 'float32', 2);
    const maskFirst = graph.createDataView(shared.buffer, {format: 'uint32', length: 2});
    expect(
      () =>
        new GPUColumnQuantiles({
          values: inputRows('values', 'float32', Float32Array.of(1, 2, 3, 4), true),
          parameters: input(
            'parameters',
            'float32',
            getGPUColumnQuantilesParameterValues({quantiles: [0.5]})
          ),
          quantileCount: 1,
          output: {
            quantiles: shared,
            validCount: packedOutput('count', 'uint32', 1),
            outputMask: createVectorView('filter-output', 'uint32', [
              maskFirst,
              packedOutput('mask-tail', 'uint32', 2)
            ])
          }
        })
    ).toThrow(/byte ranges must not overlap/);
  } finally {
    harness.buffers.forEach(buffer => buffer.destroy());
  }
});

it(
  'selects rational quantiles from 20M/empty/20M chunks against exact integer ranks',
  async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const lengths = [20_000_000, 0, 20_000_000];
    const harness = createHarness(device, 40_000_000, lengths);
    const {graph, input, packedOutput} = harness;
    // Generate the source on the GPU, avoiding a row-sized host fixture or a packed GPU binding.
    let startingRow = 0;
    const chunks = lengths.map((length, chunkIndex) => {
      const globalStartingRow = startingRow;
      startingRow += length;
      const view = createTransientView(graph, `values-${chunkIndex}`, 'float32', length);
      if (length > 0)
        graph.add(
          createWGSLKernelNode(graph, {
            id: `fill-${chunkIndex}`,
            operation: 'QuantileTestFixture',
            invocationCount: length,
            bindings: [{name: 'values', view, type: 'f32', access: 'read_write'}],
            body: `let globalRow = ${globalStartingRow}u + index;
  values[valuesOffset + index] = select(1.0, 0.0, globalRow <= 13333333u);`
          })
        );
      return view;
    });
    const ratios: [number, number][] = [
      [0, 1],
      [1, 3],
      [1, 2],
      [19, 20],
      [1, 1]
    ];
    graph.add(
      new GPUColumnQuantiles({
        values: createVectorView('values', 'float32', chunks),
        parameters: input(
          'parameters',
          'float32',
          getGPUColumnQuantilesParameterValues({
            quantiles: ratios.map(([numerator, denominator]) => numerator / denominator),
            interpolation: 'lower'
          })
        ),
        rationalProbabilities: input(
          'ratios',
          'uint32',
          getGPUColumnQuantilesRationalProbabilityValues(ratios)
        ),
        quantileCount: ratios.length,
        output: {
          quantiles: packedOutput('quantiles', 'float32', ratios.length),
          validCount: packedOutput('count', 'uint32', 1)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      // floor((40,000,000 - 1) / 3) = 13,333,333; float32 ranking gives 13,333,334.
      // The transition after the exact rank makes the two routes observably different.
      const expected = Float32Array.of(0, 0, 1, 1, 1);
      expect(await harness.readWords('quantiles')).toEqual(new Uint32Array(expected.buffer));
      expect(await harness.readWords('count')).toEqual(Uint32Array.of(40_000_000));
    } finally {
      compiled.destroy();
      harness.buffers.forEach(buffer => buffer.destroy());
    }
  },
  LARGE_QUANTILE_TEST_TIMEOUT_MS
);

it('preserves natural-break outlier refinement with empty first/middle chunks across method replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const length = 6144;
  const harness = createHarness(device, length, [0, 2047, 0, 2048, 2049]);
  const {graph, input, inputRows, packedOutput} = harness;
  const values = Float32Array.from({length}, (_, row) => row % 17);
  values[length - 1] = 1e20;
  const parameters = input(
    'parameters',
    'float32',
    getGPUClassBreaksParameterValues({method: 'natural-breaks', classCount: 3}, 3)
  );
  for (const chunked of [false, true]) {
    const route = chunked ? 'chunked' : 'packed';
    graph.add(
      new GPUClassBreaks({
        id: route,
        values: inputRows(`${route}-source`, 'float32', values, chunked),
        parameters,
        maximumClassCount: 3,
        methods: ['natural-breaks', 'custom'],
        naturalBreaksBinCount: 16,
        output: {
          breaks: packedOutput(`${route}-breaks`, 'float32', 4),
          classCount: packedOutput(`${route}-class-count`, 'uint32', 1),
          classCounts: packedOutput(`${route}-counts`, 'uint32', 3)
        }
      })
    );
  }
  const compiled = graph.compile();
  try {
    let initialBreaks: Uint32Array | undefined;
    let initialCounts: Uint32Array | undefined;
    for (const method of ['natural-breaks', 'custom', 'natural-breaks'] as const) {
      harness.writeInput(
        'parameters',
        getGPUClassBreaksParameterValues({method, classCount: 3, customEdges: [-1, 4, 10, 1e20]}, 3)
      );
      submitGraph(device, compiled);
      await harness.checkParity(['breaks', 'class-count', 'counts']);
      const frameBreakWords = await harness.readWords('chunked-breaks');
      const frameBreaks = new Float32Array(frameBreakWords.buffer);
      expect(await harness.readWords('chunked-class-count')).toEqual(Uint32Array.of(3));
      expect(await harness.readWords('chunked-counts')).toEqual(
        countClassesOracle(values, undefined, frameBreaks, 3, 3)
      );
      if (method === 'custom') {
        expect(frameBreaks).toEqual(Float32Array.of(-1, 4, 10, 1e20));
      } else {
        expect(frameBreaks[0]).toBe(0);
        expect(frameBreaks[3]).toBe(Math.fround(1e20));
      }
      if (method === 'natural-breaks') {
        const breaks = await harness.readWords('chunked-breaks');
        const counts = await harness.readWords('chunked-counts');
        if (initialBreaks) {
          expect(breaks).toEqual(initialBreaks);
          expect(counts).toEqual(initialCounts);
        } else {
          initialBreaks = breaks;
          initialCounts = counts;
        }
      }
    }
  } finally {
    compiled.destroy();
    harness.buffers.forEach(buffer => buffer.destroy());
  }
});

it.each([
  'float32',
  'uint32',
  'sint32'
] as const)('preserves moments and %s conversion at 2047/2048/2049 chunk boundaries near 2^24', async format => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const length = 6144;
  const harness = createHarness(device, length, [2047, 0, 2048, 0, 2049]);
  const {graph, input, inputRows, packedOutput} = harness;
  const integers = Int32Array.from(
    {length},
    (_, row) => (row % 2 === 0 || format === 'uint32' ? 1 : -1) * (2 ** 24 + (row % 19) - 9)
  );
  const values =
    format === 'float32'
      ? Float32Array.from(integers)
      : format === 'uint32'
        ? Uint32Array.from(integers)
        : integers;
  const mask = Uint32Array.from({length}, (_, row) => Number(row % 11 !== 0));
  const parameters = input(
    'parameters',
    'float32',
    getGPUClassBreaksParameterValues({method: 'standard-deviation', classCount: 3}, 3)
  );
  for (const chunked of [false, true]) {
    const route = chunked ? 'chunked' : 'packed';
    const source = chunked
      ? inputRows(`${route}-source`, format, values, true)
      : inputRows(`${route}-source`, 'float32', Float32Array.from(values), false);
    graph.add(
      new GPUClassBreaks({
        id: route,
        values: source,
        mask: inputRows(`${route}-mask`, 'uint32', mask, chunked),
        parameters,
        maximumClassCount: 3,
        methods: ['standard-deviation', 'head-tail', 'quantile'],
        output: {
          breaks: packedOutput(`${route}-breaks`, 'float32', 4),
          classCount: packedOutput(`${route}-class-count`, 'uint32', 1),
          classCounts: packedOutput(`${route}-counts`, 'uint32', 3)
        }
      })
    );
  }
  const compiled = graph.compile();
  try {
    for (const method of ['standard-deviation', 'head-tail', 'quantile'] as const) {
      harness.writeInput(
        'parameters',
        getGPUClassBreaksParameterValues({method, classCount: 3}, 3)
      );
      submitGraph(device, compiled);
      await harness.checkParity(['breaks', 'class-count', 'counts']);
      const frameBreaks = await harness.readWords('chunked-breaks');
      const frameClassCount = (await harness.readWords('chunked-class-count'))[0];
      expect(frameClassCount).toBeGreaterThan(0);
      expect(await harness.readWords('chunked-counts')).toEqual(
        countClassesOracle(
          Float32Array.from(values),
          mask,
          new Float32Array(frameBreaks.buffer),
          frameClassCount,
          3
        )
      );
    }
  } finally {
    compiled.destroy();
    harness.buffers.forEach(buffer => buffer.destroy());
  }
});

it('maps chunked scalar identity against independent values and validity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, 6, [0, 2, 1, 3]);
  const {graph, input, inputRows, outputRows} = harness;
  graph.add(
    new GPUScalarScale({
      values: inputRows(
        'values',
        'float32',
        Float32Array.of(NaN, -Infinity, 0, 1, 3, Infinity),
        true
      ),
      mask: inputRows('mask', 'uint32', Uint32Array.of(1, 1, 1, 1, 0, 1), true),
      domain: input('domain', 'float32', Float32Array.of(0, 10)),
      range: input('range', 'float32', Float32Array.of(0, 10)),
      parameters: input(
        'parameters',
        'float32',
        getGPUScalarScaleParameterValues({scale: 'identity', domainCount: 2, nullValue: -99})
      ),
      maximumDomainCount: 2,
      output: {
        values: outputRows('scaled', 'float32', true),
        validity: outputRows('validity', 'uint32', true)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await harness.readWords('scaled')).toEqual(
      new Uint32Array(Float32Array.of(-99, -99, 0, 1, -99, -99).buffer)
    );
    expect(await harness.readWords('validity')).toEqual(Uint32Array.of(0, 0, 1, 1, 0, 0));
  } finally {
    compiled.destroy();
    harness.buffers.forEach(buffer => buffer.destroy());
  }
});

it('keeps a median beside consecutive mask chunks in one output buffer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, 4, [2, 2]);
  const {graph, input, inputRows, packedOutput} = harness;
  const shared = packedOutput('shared', 'float32', 5);
  const quantiles = graph.createDataView(shared.buffer, {
    format: 'float32',
    length: 1,
    byteOffset: 4
  });
  const masks = [8, 16].map(byteOffset =>
    graph.createDataView(shared.buffer, {format: 'uint32', length: 2, byteOffset})
  );
  graph.add(
    new GPUColumnQuantiles({
      values: inputRows('values', 'float32', Float32Array.of(1, 2, 3, 4), true),
      parameters: input(
        'parameters',
        'float32',
        getGPUColumnQuantilesParameterValues({quantiles: [0.5]})
      ),
      quantileCount: 1,
      output: {
        quantiles,
        validCount: packedOutput('count', 'uint32', 1),
        outputMask: createVectorView('mask-output', 'uint32', masks)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await harness.readWords('shared')).toEqual(
      Uint32Array.of(new Uint32Array(Float32Array.of(2.5).buffer)[0], 1, 1, 1, 1)
    );
    expect(await harness.readWords('count')).toEqual(Uint32Array.of(4));
  } finally {
    compiled.destroy();
    harness.buffers.forEach(buffer => buffer.destroy());
  }
});

it('replays head-tail-only packed and chunked graphs against independent edges and counts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, 6, [0, 2, 0, 1, 3]);
  const {graph, input, inputRows, packedOutput} = harness;
  const values = Float32Array.of(0, 1, 2, 9, 10, 11);
  const allRows = new Uint32Array(6).fill(1);
  const parameters = input(
    'parameters',
    'float32',
    getGPUClassBreaksParameterValues({method: 'head-tail', classCount: 2}, 2)
  );
  for (const chunked of [false, true]) {
    const route = chunked ? 'chunked' : 'packed';
    graph.add(
      new GPUClassBreaks({
        id: route,
        values: inputRows(`${route}-values`, 'float32', values, chunked),
        mask: inputRows(`${route}-mask`, 'uint32', allRows, chunked),
        parameters,
        maximumClassCount: 2,
        methods: ['head-tail'],
        output: {
          breaks: packedOutput(`${route}-breaks`, 'float32', 3),
          classCount: packedOutput(`${route}-class-count`, 'uint32', 1),
          classCounts: packedOutput(`${route}-counts`, 'uint32', 2)
        }
      })
    );
  }
  const frames = [
    {mask: allRows, edges: Float32Array.of(0, 5.5, 11), counts: Uint32Array.of(3, 3)},
    {
      mask: Uint32Array.of(1, 1, 1, 1, 0, 0),
      edges: Float32Array.of(0, 3, 9),
      counts: Uint32Array.of(3, 1)
    },
    {mask: new Uint32Array(6), edges: undefined, counts: Uint32Array.of(0, 0)},
    {mask: allRows, edges: Float32Array.of(0, 5.5, 11), counts: Uint32Array.of(3, 3)}
  ];
  const compiled = graph.compile();
  try {
    for (const frame of frames) {
      for (const route of ['packed', 'chunked']) harness.writeRows(`${route}-mask`, frame.mask);
      submitGraph(device, compiled);
      for (const route of ['packed', 'chunked']) {
        const breakWords = await harness.readWords(`${route}-breaks`);
        if (frame.edges) {
          expect(breakWords).toEqual(new Uint32Array(frame.edges.buffer));
        } else {
          expect([...new Float32Array(breakWords.buffer)].every(Number.isNaN)).toBe(true);
        }
        expect(await harness.readWords(`${route}-class-count`)).toEqual(
          Uint32Array.of(frame.edges ? 2 : 0)
        );
        expect(await harness.readWords(`${route}-counts`)).toEqual(frame.counts);
      }
    }
  } finally {
    compiled.destroy();
    harness.buffers.forEach(buffer => buffer.destroy());
  }
});

it.each<GPUClassBreaksMethod>([
  'equal-interval',
  'quantile',
  'standard-deviation',
  'head-tail',
  'natural-breaks',
  'box-plot',
  'custom'
])('checks chunked %s against independent textbook edges and histograms', async method => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device, 6, [0, 2, 1, 3]);
  const {graph, input, inputRows, packedOutput} = harness;
  const maximumClassCount = method === 'box-plot' ? 6 : 2;
  graph.add(
    new GPUClassBreaks({
      values: inputRows('values', 'float32', Float32Array.of(0, 1, 2, 9, 10, 11), true),
      parameters: input(
        'parameters',
        'float32',
        getGPUClassBreaksParameterValues(
          {method, classCount: maximumClassCount, customEdges: [0, 3, 11]},
          maximumClassCount
        )
      ),
      maximumClassCount,
      methods: [method],
      naturalBreaksBinCount: 16,
      output: {
        breaks: packedOutput('breaks', 'float32', maximumClassCount + 1),
        classCount: packedOutput('class-count', 'uint32', 1),
        classCounts: packedOutput('counts', 'uint32', maximumClassCount)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    const expected =
      method === 'box-plot'
        ? Float32Array.of(0, 0, 1.25, 5.5, 9.75, 11, 11)
        : Float32Array.of(0, method === 'natural-breaks' ? 9 : method === 'custom' ? 3 : 5.5, 11);
    expect(await harness.readWords('breaks')).toEqual(new Uint32Array(expected.buffer));
    expect(await harness.readWords('class-count')).toEqual(Uint32Array.of(maximumClassCount));
    expect(await harness.readWords('counts')).toEqual(
      method === 'box-plot' ? Uint32Array.of(0, 2, 1, 1, 1, 1) : Uint32Array.of(3, 3)
    );
  } finally {
    compiled.destroy();
    harness.buffers.forEach(buffer => buffer.destroy());
  }
});
