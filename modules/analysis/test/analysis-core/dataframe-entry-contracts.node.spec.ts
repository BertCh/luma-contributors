// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createSourceFile, isExportDeclaration, isNamedExports, ScriptTarget} from 'typescript';
import * as arrow from 'apache-arrow';
import type {GPUAnalyticsDictionary} from '@luma.gl/graph/arrow';
import {expect, expectTypeOf, it} from 'vitest';
import {
  MAXIMUM_ARROW_ANALYTICS_CATEGORY_COUNT,
  MAXIMUM_ARROW_ANALYTICS_DICTIONARY_BYTE_LENGTH,
  MAXIMUM_ARROW_ANALYTICS_SCALAR_ROW_COUNT,
  prepareArrowBatchesForGPUAnalytics,
  prepareArrowKeyPairForGPUAnalytics,
  prepareArrowMeasureColumnsForGPUAnalytics,
  type PreparedArrowAnalyticsKeyColumn,
  type PreparedArrowAnalyticsKeyPair,
  type PreparedArrowAnalyticsMeasureColumn
} from '@luma-contributors/analysis/analysis-core';
import {
  computeColumnQuantilesOnCPU,
  decodeGPUGroupSum,
  evaluateSpaceTimeVariogramModel,
  fitSpaceTimeVariogramModel,
  fitVariogramModel,
  fitVariogramModelWithDiagnostics,
  getColumnQuantilesParameterValues,
  getColumnQuantilesRationalProbabilityValues,
  GPUClassBreaks,
  GPUClassBreaksOperation,
  CPUClassBreaksOperation,
  getGPUPointPatternIndicesParameterValues,
  type GPUClassBreaksShape,
  type GPUClassBreaksDeviceLimits,
  type GPUClassBreaksOperationProps,
  type CPUClassBreaksOperationProps,
  type GPUPointPatternIndicesHintSettings,
  type GPUPointPatternIndicesSettings,
  getGPUCellKeyFreeBits,
  getGPUCellPeriodKeyPeriodBits,
  getGPUSpaceTimeVariogramBinLags,
  packGPUCellPeriodKey,
  unpackGPUCellPeriodKeyChecked,
  type DecodedGPUGroupSum,
  type GPUCellPeriodKeyLayout,
  type SpaceTimeVariogramFit,
  type SpaceTimeVariogramFitOptions,
  type SpaceTimeVariogramInput,
  type SpaceTimeVariogramModel,
  type SpaceTimeVariogramModelType,
  type SpaceTimeVariogramStructure,
  type VariogramModel,
  type VariogramModelFitDiagnostics
} from '@luma-contributors/analysis/dataframe-analysis';
import * as spatial from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  deserializeGPUComputationCaptures,
  GPU_COMPUTATION_CAPTURES_SCHEMA,
  serializeGPUComputationCaptures,
  type GPUComputationCaptures,
  type GPUComputationDeclaredBinding
} from '@luma-contributors/analysis/computation-record';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

function getNamedEntryExports(relativePath: string): string[] {
  const source = createSourceFile(
    relativePath,
    readFileSync(new URL(relativePath, import.meta.url), 'utf8'),
    ScriptTarget.Latest
  );
  return source.statements.flatMap(statement =>
    isExportDeclaration(statement) &&
    statement.exportClause &&
    isNamedExports(statement.exportClause)
      ? statement.exportClause.elements.map(element => element.name.text)
      : []
  );
}

it('SCL-05 exposes named planner and operation props at dataframe', () => {
  expect(
    getNamedEntryExports('../../src/dataframe-analysis/column-classification/index.ts')
  ).toEqual(
    expect.arrayContaining([
      'GPUClassBreaksShape',
      'GPUClassBreaksDeviceLimits',
      'GPUClassBreaksOperationProps',
      'CPUClassBreaksOperationProps'
    ])
  );
  const shape: GPUClassBreaksShape = {
    rowCount: 3,
    maximumClassCount: 2,
    methods: ['equal-interval']
  };
  const device = createNullWebGPUDevice();
  try {
    const limits: GPUClassBreaksDeviceLimits = device.limits;
    expect(GPUClassBreaks.getExecutionPlan(shape, limits).feasible).toBe(true);
  } finally {
    device.destroy();
  }
  const gpuProps: GPUClassBreaksOperationProps = {
    values: 'values',
    parameters: 'parameters',
    maximumClassCount: 2,
    methods: ['equal-interval'],
    output: {breaks: 'breaks', classCount: 'class-count'}
  };
  const cpuProps: CPUClassBreaksOperationProps = gpuProps;
  expect(new GPUClassBreaksOperation(gpuProps).methods).toEqual(['equal-interval']);
  expect(new CPUClassBreaksOperation(cpuProps)).toBeInstanceOf(CPUClassBreaksOperation);
});

it('NAM-11 exposes hint settings with a required index cell size', () => {
  expect(getNamedEntryExports('../../src/dataframe-analysis/pair-statistics/index.ts')).toContain(
    'GPUPointPatternIndicesHintSettings'
  );
  interface ExtendedSettings extends GPUPointPatternIndicesSettings {
    label: string;
  }
  const legacy: ExtendedSettings = {bounds: [0, 0, 4, 4], indexCellSizeHint: 2, label: 'canonical'};
  const hint: GPUPointPatternIndicesHintSettings = {bounds: [0, 0, 4, 4], indexCellSizeHint: 2};
  expectTypeOf<ExtendedSettings['indexCellSizeHint']>().toEqualTypeOf<number>();
  expect(Array.from(getGPUPointPatternIndicesParameterValues(hint))).toEqual([
    0, 0, 4, 4, 2, 1, 0, 0
  ]);
  expect(getGPUPointPatternIndicesParameterValues(legacy)).toEqual(
    getGPUPointPatternIndicesParameterValues(hint)
  );
});

function getUsableArrowKeyCodes(
  column: PreparedArrowAnalyticsKeyColumn,
  expectedChunkLengths: readonly number[]
): number[][] {
  expect(column.chunks).toHaveLength(expectedChunkLengths.length);
  expect(column.validity).toHaveLength(expectedChunkLengths.length);
  return column.chunks.map((chunk, chunkIndex) => {
    expect(chunk).toHaveLength(expectedChunkLengths[chunkIndex]);
    expect(column.validity[chunkIndex]).toHaveLength(expectedChunkLengths[chunkIndex]);
    return Array.from(chunk).filter((_, row) => column.validity[chunkIndex][row]);
  });
}

it('EGR-04 exposes explicit identity layout without reinterpreting persisted one-bit keys', () => {
  const identity: GPUCellPeriodKeyLayout = 'zero-period-bit-identity';
  const legacy: GPUCellPeriodKeyLayout = 'legacy-one-bit';
  expect(getGPUCellPeriodKeyPeriodBits(1, identity)).toBe(0);
  expect(getGPUCellKeyFreeBits('s2', 30)).toBe(0);
  // A level-30 S2 leaf has no free low bits: its one-period key must stay 1.
  expect(packGPUCellPeriodKey(1n, 0, 1, 'zeros', identity)).toBe(1n);
  expect(unpackGPUCellPeriodKeyChecked(1n, 1, 'zeros', identity)).toEqual({
    cellKey: 1n,
    period: 0
  });
  expect(packGPUCellPeriodKey(3n, 0, 1, 'ones', identity)).toBe(3n);
  expect(unpackGPUCellPeriodKeyChecked(3n, 1, 'ones', identity)).toEqual({
    cellKey: 3n,
    period: 0
  });
  // Persisted pre-identity fixture: replace the low one bit of cell 3 with period 0 => 2.
  expect(getGPUCellPeriodKeyPeriodBits(1, legacy)).toBe(1);
  expect(packGPUCellPeriodKey(3n, 0, 1, 'ones', legacy)).toBe(2n);
  expect(unpackGPUCellPeriodKeyChecked(2n, 1, 'ones', legacy)).toEqual({
    cellKey: 3n,
    period: 0
  });
});

it('EGR-05 exposes checked unpacking with missing and malformed keys distinguished', () => {
  const legacy = 'legacy-one-bit';
  expectTypeOf<ReturnType<typeof unpackGPUCellPeriodKeyChecked>>().toEqualTypeOf<
    {cellKey: bigint; period: number} | undefined
  >();
  expect(unpackGPUCellPeriodKeyChecked(0xffffffffffffffffn, 3, 'zeros', legacy)).toBeUndefined();
  // Two period bits admit payload 3, but periodCount 3 promises only 0, 1, 2.
  expect(() => unpackGPUCellPeriodKeyChecked(3n, 3, 'zeros', legacy)).toThrow(
    /period 3.*periodCount 3/
  );
  expect(unpackGPUCellPeriodKeyChecked(6n, 3, 'zeros', legacy)).toEqual({cellKey: 4n, period: 2});
});

it('UDF-01 exposes paired Arrow key chunks, validity and the shared decoding dictionary at core', () => {
  expectTypeOf<PreparedArrowAnalyticsKeyColumn['chunks']>().toEqualTypeOf<readonly Uint32Array[]>();
  expectTypeOf<PreparedArrowAnalyticsKeyColumn['validity']>().toEqualTypeOf<
    readonly Uint32Array[]
  >();
  expectTypeOf<PreparedArrowAnalyticsKeyPair['dictionary']>().toEqualTypeOf<
    GPUAnalyticsDictionary | undefined
  >();
  const left = new arrow.Table({label: arrow.vectorFromArray(['A', null, 'B'], new arrow.Utf8())});
  const right = new arrow.Table({name: arrow.vectorFromArray(['B', 'A'], new arrow.Utf8())});
  const pair: PreparedArrowAnalyticsKeyPair = prepareArrowKeyPairForGPUAnalytics(
    left,
    'label',
    right,
    'name'
  );
  const column: PreparedArrowAnalyticsKeyColumn = pair.left;
  expect(column.validity.map(chunk => Array.from(chunk))).toEqual([[1, 0, 1]]);
  expect(getUsableArrowKeyCodes(column, [3])).toEqual([[0, 1]]);
  expect(getUsableArrowKeyCodes(pair.right, [2])).toEqual([[1, 0]]);
  expect(pair.right.chunks.map(chunk => Array.from(chunk))).toEqual([[1, 0]]);
  expect(pair.dictionary?.values).toEqual(['A', 'B']);
  // Hand join: left A -> right row 1 -> 10; left B -> right row 0 -> 20.
  const rightValues = [20, 10];
  expect(
    [0, 2].map(row => rightValues[pair.right.chunks[0].indexOf(column.chunks[0][row])])
  ).toEqual([10, 20]);
  const integers = new arrow.Table({
    id: arrow.vectorFromArray([9007199254740992n, 9007199254740993n], new arrow.Uint64())
  });
  const integerPair = prepareArrowKeyPairForGPUAnalytics(integers, 'id', integers, 'id');
  // 2^53 and 2^53 + 1 are [low, high] = [0, 2^21] and [1, 2^21].
  expect(integerPair.keyBits).toBe(64);
  expect(integerPair.left.chunks.map(chunk => Array.from(chunk))).toEqual([
    [0, 2097152, 1, 2097152]
  ]);
});

it('review S1 rejects extra code lanes or mismatched Arrow chunk membership before filtering nulls', () => {
  const column: PreparedArrowAnalyticsKeyColumn = {
    name: 'label',
    sourceType: 'Utf8',
    chunks: [Uint32Array.of(0, 42, 1)],
    validity: [Uint32Array.of(1, 0, 1)]
  };
  expect(getUsableArrowKeyCodes(column, [3])).toEqual([[0, 1]]);
  expect(() =>
    getUsableArrowKeyCodes({...column, chunks: [Uint32Array.of(0, 42, 1, 99)]}, [3])
  ).toThrow();
  expect(() => getUsableArrowKeyCodes({...column, chunks: []}, [3])).toThrow();
  expect(() =>
    getUsableArrowKeyCodes({...column, validity: [Uint32Array.of(1, 0, 1, 1)]}, [3])
  ).toThrow();
});

it('review B1 exposes nullable Arrow measure preparation and its public column type', () => {
  const source = new arrow.Table({
    x: arrow.vectorFromArray([null, 0, 2], new arrow.Float32()),
    y: arrow.vectorFromArray([12, null, 4], new arrow.Float32())
  });
  const measures = prepareArrowMeasureColumnsForGPUAnalytics(source, ['x', 'y']);
  const first: PreparedArrowAnalyticsMeasureColumn = measures['x'];
  expect(first.chunks).toHaveLength(1);
  expect(Array.from(first.chunks[0])).toEqual([NaN, 0, 2]);
  expect(first.validity.map(chunk => Array.from(chunk))).toEqual([[0, 1, 1]]);
  expect(measures['y'].chunks.map(chunk => Array.from(chunk))).toEqual([[12, NaN, 4]]);
  expect(measures['y'].validity.map(chunk => Array.from(chunk))).toEqual([[1, 0, 1]]);
});

it('review B1 exposes batch preparation and the Arrow route limit constants at core', () => {
  expect(typeof prepareArrowBatchesForGPUAnalytics).toBe('function');
  expect(MAXIMUM_ARROW_ANALYTICS_SCALAR_ROW_COUNT).toBe(4294967295);
  expect(MAXIMUM_ARROW_ANALYTICS_CATEGORY_COUNT).toBe(4294967295);
  expect(MAXIMUM_ARROW_ANALYTICS_DICTIONARY_BYTE_LENGTH).toBe(2147483647);
  const first = new arrow.Table({label: arrow.vectorFromArray(['A', 'B'], new arrow.Utf8())});
  const second = new arrow.Table({label: arrow.vectorFromArray(['B', 'C'], new arrow.Utf8())});
  const batches = Array.from(prepareArrowBatchesForGPUAnalytics(first.concat(second)));
  expect(batches).toHaveLength(2);
  expect(batches.map(batch => batch.sourceBatches)).toEqual([
    [{sourceBatchIndex: 0, sourceRowIndexOffset: 0, sourceRowCount: 2}],
    [{sourceBatchIndex: 1, sourceRowIndexOffset: 2, sourceRowCount: 2}]
  ]);
  expect(batches.map(batch => Array.from(batch.table.getChild('label')!.data[0].values))).toEqual([
    [0, 1],
    [1, 2]
  ]);
  expect(batches.map(batch => Array.from(batch.table.getChild('label')!))).toEqual([
    ['A', 'B'],
    ['B', 'C']
  ]);
  expect(batches[1].dictionaries['label'].values).toEqual(['A', 'B', 'C']);
});

it('UDF-04 publicly round trips finite, missing and signed-zero capture bits', () => {
  // IEEE-754 literals: 1.5, quiet NaN with payload 0x12345, negative zero; guards excluded.
  const words = Uint32Array.of(99, 0x3fc00000, 0x7fc12345, 0x80000000, 99);
  const captures: GPUComputationCaptures = {
    schema: GPU_COMPUTATION_CAPTURES_SCHEMA,
    runId: 'public-codec',
    snapshots: [
      {
        id: 'result',
        name: 'result',
        elementType: 'f32',
        values: new Float32Array(words.buffer, 4, 3),
        provenance: 'gpu-readback'
      }
    ],
    stamps: [],
    replays: []
  };
  const restored = deserializeGPUComputationCaptures(serializeGPUComputationCaptures(captures));
  const values = restored.snapshots[0].values;
  expect(values).toBeInstanceOf(Float32Array);
  const floats = values as Float32Array;
  expect(Array.from(new Uint32Array(floats.buffer, floats.byteOffset, floats.length))).toEqual([
    0x3fc00000, 0x7fc12345, 0x80000000
  ]);
});

it('EGR-12 exposes signed scaled group sums at a nonzero row through dataframe', () => {
  // Row 1 is signed int64 -65536 (two's complement), divided by 65536 => -1.
  const words = Uint32Array.of(123, 0, 0xffff0000, 0xffffffff);
  const decoded: DecodedGPUGroupSum = decodeGPUGroupSum(words, 1, 65536);
  expect(Number.isFinite(decoded.value)).toBe(true);
  expect(decoded).toEqual({scaledInteger: -65536n, value: -1});
  expect(decodeGPUGroupSum(new Uint8Array(words.buffer), 1, 65536)).toEqual(decoded);
  expect(decodeGPUGroupSum(words.buffer, 1, 65536)).toEqual(decoded);
});

it('TYP-02 and ENT-08 expose the separately named variogram diagnostics and stable result type', () => {
  // Independent spherical fixture: gamma(h) = 2 + 4 * (1.5t - 0.5t^3), t = h/2 < 1;
  // gamma(h) = 6 for h >= 2. Thus h=.5 => 3.46875, h=1 => 4.75, h>=2 => 6.
  const input = {
    distances: [0.5, 1, 2, 3, 4],
    semivariances: [3.46875, 4.75, 6, 6, 6],
    pairCounts: [1, 1, 1, 1, 1]
  };
  expectTypeOf<
    ReturnType<typeof fitVariogramModelWithDiagnostics>
  >().toEqualTypeOf<VariogramModelFitDiagnostics>();
  const diagnostics: VariogramModelFitDiagnostics = fitVariogramModelWithDiagnostics(input, {
    model: 'spherical',
    weighting: 'none'
  });
  const model: VariogramModel = fitVariogramModel(input, {model: 'spherical', weighting: 'none'});
  expect(Number.isFinite(diagnostics.model.nugget)).toBe(true);
  expect(Number.isFinite(diagnostics.model.sill)).toBe(true);
  expect(Number.isFinite(diagnostics.model.range)).toBe(true);
  expect(diagnostics.model.nugget).toBeCloseTo(2, 8);
  expect(diagnostics.model.sill).toBeCloseTo(4, 8);
  expect(diagnostics.model.range).toBeCloseTo(2, 8);
  expect(Number.isFinite(diagnostics.model.residual)).toBe(true);
  expect(diagnostics.model.residual).toBeGreaterThanOrEqual(0);
  expect(diagnostics.model.residual).toBeLessThan(1e-16);
  expect(Number.isFinite(diagnostics.logRangeBracket)).toBe(true);
  expect(typeof diagnostics.atBoundary).toBe('boolean');
  expect(typeof diagnostics.converged).toBe('boolean');
  expect(diagnostics.model).toEqual(model);
});

it('ENT-06 exposes the same space-time helpers and types at dataframe and spatial entries', () => {
  expect(getGPUSpaceTimeVariogramBinLags).toBe(spatial.getGPUSpaceTimeVariogramBinLags);
  expect(fitSpaceTimeVariogramModel).toBe(spatial.fitSpaceTimeVariogramModel);
  expect(evaluateSpaceTimeVariogramModel).toBe(spatial.evaluateSpaceTimeVariogramModel);
  expectTypeOf<SpaceTimeVariogramFit>().toEqualTypeOf<spatial.SpaceTimeVariogramFit>();
  expectTypeOf<SpaceTimeVariogramFitOptions>().toEqualTypeOf<spatial.SpaceTimeVariogramFitOptions>();
  expectTypeOf<SpaceTimeVariogramInput>().toEqualTypeOf<spatial.SpaceTimeVariogramInput>();
  expectTypeOf<SpaceTimeVariogramModel>().toEqualTypeOf<spatial.SpaceTimeVariogramModel>();
  expectTypeOf<SpaceTimeVariogramModelType>().toEqualTypeOf<spatial.SpaceTimeVariogramModelType>();
  expectTypeOf<SpaceTimeVariogramStructure>().toEqualTypeOf<spatial.SpaceTimeVariogramStructure>();
  const lags = getGPUSpaceTimeVariogramBinLags({
    lagCount: 1,
    temporalLagCount: 1,
    maximumDistance: 4,
    maximumTimeLag: 8
  });
  // Spatial-major bin order: (0,0), (0,4), (2,0), (2,4).
  expect(Array.from(lags.distances)).toEqual([0, 0, 2, 2]);
  expect(Array.from(lags.timeLags)).toEqual([0, 4, 0, 4]);
});

it('TWN-14 exposes neutral quantile packers for more than 64 CPU quantiles', () => {
  const quantileCount = 65;
  const parameters = getColumnQuantilesParameterValues({
    quantiles: Array(quantileCount).fill(0.5),
    interpolation: 'linear'
  });
  const rationalProbabilities = getColumnQuantilesRationalProbabilityValues(
    Array.from({length: quantileCount}, () => [1, 2] as const)
  );
  expect(Array.from(parameters.slice(0, 4))).toEqual([3, 0, 1, 0]);
  expect(Array.from(rationalProbabilities)).toEqual(
    Array.from({length: 130}, (_, index) => (index % 2 === 0 ? 1 : 2))
  );
  // Every requested median of [0, 2] is exactly 1, with or without rational ranks.
  for (const probabilities of [undefined, rationalProbabilities]) {
    const result = computeColumnQuantilesOnCPU(
      {values: Float32Array.of(0, 2), parameters, rationalProbabilities: probabilities},
      quantileCount
    );
    expect(Array.from(result.quantiles)).toEqual(Array(quantileCount).fill(1));
  }
});

it('W3-12 publicly names a declared slot without claiming a bound graph resource', () => {
  const declared: GPUComputationDeclaredBinding = {
    name: 'values',
    type: 'array<f32>',
    group: 0,
    location: 1
  };
  expectTypeOf<GPUComputationDeclaredBinding>().not.toHaveProperty('resource');
  expect(declared).toEqual({name: 'values', type: 'array<f32>', group: 0, location: 1});
});
