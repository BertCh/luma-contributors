// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expectTypeOf, it} from 'vitest';
import {expect} from 'vitest';
import {readFileSync} from 'node:fs';
import typeScript from 'typescript';
import {resolveInventoryParameterType} from '../../../../website/scripts/extract-inventory';

it('catalog extraction resolves generic constructor constraints before collecting union props', () => {
  const source = typeScript.createSourceFile(
    '/inventory-fixture.ts',
    `
    type ForestProps = {model: string; rowCount: number} & (
      {features: number[]; rasterStack?: never} | {features?: never; rasterStack: {bands: number[]}}
    );
    class Forest<Props extends ForestProps> {constructor(props: Props) {}}
  `,
    typeScript.ScriptTarget.Latest,
    true
  );
  const options = {noLib: true, noEmit: true};
  const host = typeScript.createCompilerHost(options);
  host.getSourceFile = file => (file === source.fileName ? source : undefined);
  host.fileExists = file => file === source.fileName;
  host.readFile = file => (file === source.fileName ? source.text : undefined);
  const program = typeScript.createProgram([source.fileName], options, host);
  const checker = program.getTypeChecker();
  const declaration = source.statements.find(typeScript.isClassDeclaration)!;
  const parameter = declaration.members.find(typeScript.isConstructorDeclaration)!.parameters[0];
  const resolved = resolveInventoryParameterType(checker, checker.getTypeAtLocation(parameter));
  expect(resolved.isUnion()).toBe(true);
  const members = resolved.isUnion() ? resolved.types : [resolved];
  expect([
    ...new Set(
      members.flatMap(member => checker.getPropertiesOfType(member).map(property => property.name))
    )
  ]).toEqual(expect.arrayContaining(['features', 'rasterStack', 'model', 'rowCount']));
});
import type {GPUForestInferenceProps} from '../../src/dataframe-analysis/forest-inference/gpu-forest-inference';
import type {
  GPUCellAggregation,
  GPUCellAggregationProps
} from '../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation';
import type {GPUCellRollupProps} from '../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-rollup';
import type {GPUGeometryCleanupProps} from '../../src/gpu-spatial-analysis/geometry-edit/gpu-geometry-cleanup';
import type {GPUExpectationBasedSpaceTimeScanProps} from '../../src/gpu-spatial-analysis/scan-statistics/gpu-expectation-based-space-time-scan';
import type {GPUSpatialScanStatisticProps} from '../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic';
import type {GPULocalKrigingProps} from '../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import type {GPUGeneralizedGeographicallyWeightedRegressionProps} from '../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import type {GPUSpatialLagProps} from '../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-lag';
import type {
  GPUNetworkCentrality,
  GPUNetworkCentralityProps
} from '../../src/gpu-network/network-centrality/gpu-network-centrality';
import type {GPUGeneralizedGeographicallyWeightedRegression} from '../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';

it('website inventory excludes private composition switches including inherited props', () => {
  const inventory = JSON.parse(
    readFileSync('website/src/catalog/generated/inventory.json', 'utf8')
  ) as {
    symbols: {symbol: string; props: {name: string}[]}[];
  };
  for (const [symbol, internalProp] of [
    ['GPUCellAggregation', 'extraCounts'],
    ['GPUCellRollup', 'extraCounts'],
    ['GPUCellPyramid', 'extraCounts'],
    ['GPUGeometryCleanup', 'forceSequentialRepeatRemoval'],
    ['GPUSpatialScanStatistic', 'batchReplicates'],
    ['GPUExpectationBasedSpaceTimeScan', 'batchReplicates'],
    ['GPULocalKriging', 'sampleIndexChunkQueryCount'],
    ['GPUGeneralizedGeographicallyWeightedRegression', 'fittedScale'],
    ['GPUSpatialLag', 'rowStart']
  ]) {
    const entries = inventory.symbols.filter(entry => entry.symbol === symbol);
    expect(entries.length, symbol).toBeGreaterThan(0);
    for (const entry of entries)
      expect(
        entry.props.map(prop => prop.name),
        symbol
      ).not.toContain(internalProp);
  }
});

it('forest catalog includes both input arms and shared required fields', () => {
  const inventory = JSON.parse(
    readFileSync('website/src/catalog/generated/inventory.json', 'utf8')
  ) as {
    symbols: {symbol: string; propsType: string; props: {name: string; optional: boolean}[]}[];
  };
  const forest = inventory.symbols.find(entry => entry.symbol === 'GPUForestInference');
  expect(forest).toBeDefined();
  expect(forest!.props.map(prop => prop.name)).toEqual(
    expect.arrayContaining(['features', 'rasterStack', 'model', 'rowCount'])
  );
  expect(
    forest!.props
      .filter(prop => ['model', 'rowCount'].includes(prop.name))
      .every(prop => !prop.optional)
  ).toBe(true);
  expect(forest!.propsType).not.toBe('Props');
  expect(forest!.props.find(prop => prop.name === 'features')).toMatchObject({
    type: expect.stringContaining('GraphDataView'),
    optional: true
  });
  expect(forest!.props.find(prop => prop.name === 'rasterStack')).toMatchObject({
    type: expect.stringContaining('bands'),
    optional: true
  });
});

it('public props exclude internal composition and benchmark switches', () => {
  expectTypeOf<GPUCellAggregation>().not.toHaveProperty('validateBindings');
  expectTypeOf<GPUNetworkCentrality>().not.toHaveProperty('validateGraphBindings');
  expectTypeOf<GPUCellAggregationProps>().not.toHaveProperty('extraCounts');
  expectTypeOf<GPUCellRollupProps>().not.toHaveProperty('extraCounts');
  expectTypeOf<GPUGeometryCleanupProps>().not.toHaveProperty('forceSequentialRepeatRemoval');
  expectTypeOf<GPUExpectationBasedSpaceTimeScanProps>().not.toHaveProperty('batchReplicates');
  expectTypeOf<GPUSpatialScanStatisticProps>().not.toHaveProperty('batchReplicates');
  expectTypeOf<GPULocalKrigingProps>().not.toHaveProperty('sampleIndexChunkQueryCount');
  expectTypeOf<GPUGeneralizedGeographicallyWeightedRegressionProps>().not.toHaveProperty(
    'fittedScale'
  );
  expectTypeOf<GPUSpatialLagProps>().not.toHaveProperty('rowStart');
  expectTypeOf<ConstructorParameters<typeof GPUNetworkCentrality>>().toEqualTypeOf<
    [props: GPUNetworkCentralityProps]
  >();
  expectTypeOf<
    Parameters<GPUGeneralizedGeographicallyWeightedRegression['getCommandNodes']>['length']
  >().toEqualTypeOf<1>();
});

it('forest inputs require exactly one source at the type boundary', () => {
  type Shared = Omit<GPUForestInferenceProps, 'features' | 'rasterStack' | 'featureLayout'>;
  type Features = NonNullable<GPUForestInferenceProps['features']>;
  type RasterStack = NonNullable<GPUForestInferenceProps['rasterStack']>;
  expectTypeOf<Shared>().not.toExtend<GPUForestInferenceProps>();
  expectTypeOf<
    Shared & {features: Features; rasterStack: RasterStack}
  >().not.toExtend<GPUForestInferenceProps>();
  expectTypeOf<Shared & {features: Features}>().toExtend<GPUForestInferenceProps>();
  expectTypeOf<Shared & {rasterStack: RasterStack}>().toExtend<GPUForestInferenceProps>();
});
