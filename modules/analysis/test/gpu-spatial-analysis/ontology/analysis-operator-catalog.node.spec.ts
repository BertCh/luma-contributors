// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import * as analysisCore from '../../../src/analysis-core/index';
import * as computationRecord from '../../../src/computation-record/index';
import * as dataframeAnalysis from '../../../src/dataframe-analysis/index';
import * as geospatialAnalysis from '../../../src/geospatial-analysis/index';
import * as gpuNetwork from '../../../src/gpu-network/index';
import * as gpuSpatialAnalysis from '../../../src/gpu-spatial-analysis/index';
import * as gpuTerrain from '../../../src/gpu-terrain/index';
import * as pointcloudAnalysis from '../../../src/pointcloud-analysis/index';
import * as rasterAnalysis from '../../../src/raster-analysis/index';
import {
  REUSED_ANALYSIS_OPERATORS,
  GPU_SPATIAL_ANALYSIS_AUDIT,
  GPU_SPATIAL_ANALYSIS_CAPABILITIES,
  GPU_SPATIAL_ANALYSIS_OPERATORS,
  getGPUAnalysisCapabilityCatalog,
  getGPUAnalysisOperator,
  getGPUAnalysisOperatorCatalog,
  queryGPUAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/index';
import {FITTING_KERNEL_OPERATORS} from '../../../src/gpu-spatial-analysis/ontology/fitting-kernel-catalog';
import {SPATIOTEMPORAL_REVIEW_OPERATORS} from '../../../src/gpu-spatial-analysis/ontology/spatiotemporal-review-catalog';

const ENTRY_POINTS: Record<string, Record<string, unknown>> = {
  '@luma-contributors/analysis/analysis-core': analysisCore,
  '@luma-contributors/analysis/computation-record': computationRecord,
  '@luma-contributors/analysis/gpu-spatial-analysis': gpuSpatialAnalysis,
  '@luma-contributors/analysis/gpu-network': gpuNetwork,
  '@luma-contributors/analysis/gpu-terrain': gpuTerrain,
  '@luma-contributors/analysis/raster-analysis': rasterAnalysis,
  '@luma-contributors/analysis/dataframe-analysis': dataframeAnalysis,
  '@luma-contributors/analysis/pointcloud-analysis': pointcloudAnalysis,
  '@luma-contributors/analysis/geospatial-analysis': geospatialAnalysis
};

/** Classes that declare getCommandNodes, and add*Recipe functions, exported by an entry point. */
function listExportedContributors(entry: Record<string, unknown>): string[] {
  return Object.entries(entry)
    .filter(([name, value]) => {
      if (typeof value !== 'function') return false;
      if (/^add\w+Recipe$/.test(name)) return true;
      const prototype = (value as {prototype?: Record<string, unknown>}).prototype;
      return typeof prototype?.['getCommandNodes'] === 'function';
    })
    .map(([name]) => name)
    .sort();
}

it('has a catalog entry for every exported contributor and recipe of every entry point', () => {
  const missing: Record<string, string[]> = {};
  for (const [packageName, entry] of Object.entries(ENTRY_POINTS)) {
    const names = listExportedContributors(entry).filter(name => !getGPUAnalysisOperator(name));
    if (names.length > 0) missing[packageName] = names;
  }
  // A new contributor or add*Recipe export needs an entry in the operator catalog.
  expect(missing).toEqual({});
});

it('resolves every catalog operator from the entry point it declares', () => {
  const operators = getGPUAnalysisOperatorCatalog();
  for (const operator of operators) {
    expect(
      ENTRY_POINTS[operator.package]?.[operator.exportName],
      `${operator.package}: ${operator.exportName}`
    ).toBeTypeOf('function');
  }
  expect(new Set(operators.map(({exportName}) => exportName)).size).toBe(operators.length);
});

it('keeps the original exports unchanged and adds the other entry points beside them', () => {
  // Consumers (the website inventory generator, the kepler catalog bindings) enumerate these.
  expect(GPU_SPATIAL_ANALYSIS_OPERATORS).toHaveLength(261);
  expect(GPU_SPATIAL_ANALYSIS_CAPABILITIES).toHaveLength(49);
  expect(GPU_SPATIAL_ANALYSIS_AUDIT.operatorCount).toBe(261);
  const packages = new Set(GPU_SPATIAL_ANALYSIS_OPERATORS.map(({package: name}) => name));
  // GPUParameterBuffer is catalogued here but exported only from analysis-core (STD-09).
  expect([...packages].sort()).toEqual([
    '@luma-contributors/analysis/analysis-core',
    '@luma-contributors/analysis/gpu-spatial-analysis',
    '@luma-contributors/analysis/gpu-terrain'
  ]);

  const all = getGPUAnalysisOperatorCatalog();
  // 226 operators of the other entry points, then the operators of `gpu-spatial-analysis` added
  // after the original 261 (spatiotemporal review, fitting kernels, reused analysis operators).
  expect(all.length).toBe(
    261 +
      226 +
      SPATIOTEMPORAL_REVIEW_OPERATORS.length +
      FITTING_KERNEL_OPERATORS.length +
      REUSED_ANALYSIS_OPERATORS.length
  );
  expect(all.slice(0, 261)).toEqual(GPU_SPATIAL_ANALYSIS_OPERATORS);
  const capabilities = getGPUAnalysisCapabilityCatalog();
  expect(capabilities.slice(0, 49)).toEqual(GPU_SPATIAL_ANALYSIS_CAPABILITIES);
  expect(new Set(capabilities.map(({id}) => id)).size).toBe(capabilities.length);
  for (const operator of all) {
    expect(capabilities.some(({id}) => id === operator.primaryCapability)).toBe(true);
  }
});

it('names the new capability families for what they do, never for a development wave', () => {
  const original = new Set(GPU_SPATIAL_ANALYSIS_CAPABILITIES.map(({id}) => id));
  for (const capability of getGPUAnalysisCapabilityCatalog()) {
    if (original.has(capability.id)) continue;
    expect(capability.id).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
    expect(capability.id).not.toMatch(/^(expansion|coverage|wave|tranche|platform)-/);
    expect(capability.operators.length).toBeGreaterThan(0);
  }
});

it('records a tier for every operator and the limits of every reference operator', () => {
  for (const operator of getGPUAnalysisOperatorCatalog()) {
    expect(['scale', 'reference'], operator.exportName).toContain(operator.tier);
    if (operator.tier === 'reference') {
      expect(
        (operator.limits?.length ?? 0) > 0 || Boolean(operator.limitation),
        `${operator.exportName} is a reference operator and must record its limits`
      ).toBe(true);
    }
    for (const limit of operator.limits ?? []) {
      expect(limit.maximum, `${operator.exportName} ${limit.name}`).toBeGreaterThan(0);
      expect(limit.reason, `${operator.exportName} ${limit.name}`).toBeTruthy();
      if (limit.source) {
        expect(existsSync(resolve(limit.source)), `${operator.exportName}: ${limit.source}`).toBe(
          true
        );
      }
    }
  }
  const reference = queryGPUAnalysisOperators({tier: 'reference'}).map(
    ({exportName}) => exportName
  );
  expect(reference).toContain('GPUNetworkFacilityOptimization');
  expect(reference).not.toContain('GPUNetworkFlow');
  expect(reference).not.toContain('GPUNetworkReachability');
});

it('records the facility search limit as candidates, not demand rows', () => {
  const facility = getGPUAnalysisOperator('GPUNetworkFacilityOptimization');
  expect(facility?.limits?.find(({name}) => name === 'candidateCount')?.maximum).toBe(2 ** 20);
});

it('describes added operators with a summary, existing documentation and a spec', () => {
  const added = getGPUAnalysisOperatorCatalog().slice(261);
  for (const operator of added) {
    expect(operator.summary, operator.exportName).toBeTruthy();
    expect(existsSync(resolve(operator.evidence.tests)), operator.evidence.tests).toBe(true);
    expect(
      existsSync(resolve(operator.evidence.documentation)),
      operator.evidence.documentation
    ).toBe(true);
  }
});

it('marks temporal operators as 2d+time', () => {
  const temporal = queryGPUAnalysisOperators({dimensionality: '2d+time'}).map(
    ({exportName}) => exportName
  );
  for (const name of [
    'GPUNetworkTimeRouting',
    'GPUMapMatching',
    'GPUCalendarBuckets',
    'GPUTemporalReduction',
    'GPUTimeWindowFilter',
    'GPUSpaceTimeVariogram',
    'GPUChangeDetection',
    'GPURasterTemporal',
    'GPURasterSeasonalBreak',
    'GPUSolarIrradiance',
    // TC-08: operators of gpu-spatial-analysis that read times or tracks.
    'GPUKnoxTest',
    'GPUMantelTest',
    'GPUEmergingHotSpots',
    'GPUSpatialMarkov',
    'GPUSpatialPanelRegression',
    'GPUTrajectoryMetrics',
    'GPUTrackSimilarity',
    'addSpaceTimeHotSpotsRecipe',
    'addPeriodComparisonRecipe',
    ...SPATIOTEMPORAL_REVIEW_OPERATORS.filter(
      ({dimensionality}) => dimensionality === '2d+time'
    ).map(({exportName}) => exportName)
  ]) {
    expect(temporal, name).toContain(name);
  }
  // Every trajectory contributor is time-aware.
  const untagged = getGPUAnalysisOperatorCatalog()
    .filter(({exportName}) => /^GPUTraj|^GPUTrack|^GPUMobility|^GPUModalSplit/.test(exportName))
    .filter(({dimensionality}) => dimensionality !== '2d+time')
    .map(({exportName}) => exportName);
  expect(untagged).toEqual([]);
});

it('finds time-series consumers of the dataframe and raster entry points by data kind', () => {
  const consumers = queryGPUAnalysisOperators({input: 'time-series'}).map(
    ({exportName}) => exportName
  );
  for (const name of [
    'GPUCalendarBuckets',
    'GPUCubeLayoutTranspose',
    'GPUPeriodIndex',
    'GPUTimeWordsRelative',
    'GPUValueCube',
    'GPUCellPeriodKeys',
    'GPURasterTemporal',
    'GPUChangeDetection',
    'GPUSpaceTimeCrossCorrelation',
    'GPUSpaceTimeKriging'
  ]) {
    expect(consumers, name).toContain(name);
  }
});

it('lists the parameter buffer as a helper and leaves it out of operator queries', () => {
  expect(getGPUAnalysisOperator('GPUParameterBuffer')?.role).toBe('helper');
  expect(queryGPUAnalysisOperators().map(({exportName}) => exportName)).not.toContain(
    'GPUParameterBuffer'
  );
  expect(
    queryGPUAnalysisOperators({includeHelpers: true}).map(({exportName}) => exportName)
  ).toContain('GPUParameterBuffer');
  // Counts of the original exports are unchanged: the helper is still listed.
  expect(GPU_SPATIAL_ANALYSIS_AUDIT.helperOperatorCount).toBe(1);
});

it('queries operators by package, tier and capability', () => {
  const network = queryGPUAnalysisOperators({
    package: '@luma-contributors/analysis/gpu-network'
  });
  expect(network.length).toBeGreaterThanOrEqual(37);
  expect(network.map(({exportName}) => exportName)).toContain('GPUNetworkReachability');
  expect(
    queryGPUAnalysisOperators({primaryCapability: 'pointcloud-ground'}).map(
      ({exportName}) => exportName
    )
  ).toEqual([
    'GPUPointCloudGround',
    'GPUPointCloudSMRF',
    'GPUPointCloudSurfaces',
    'GPUPointCloudHeight'
  ]);
});

it('classifies extracted preparation and exact summaries as operations, and their arithmetic/upload building blocks as helpers', () => {
  for (const operator of REUSED_ANALYSIS_OPERATORS) {
    expect(getGPUAnalysisOperator(operator.exportName)).toBe(operator);
    expect(operator.evidence.contract).not.toBe('unaudited');
  }
  for (const helper of [
    'makeDecimal128WordVectors',
    'makeDecimal128SelectionExpression',
    'makeInteger64RangeSelectionExpression',
    'mergeNumericNarrowingReports',
    'runWideNumericKernel',
    'splitBigIntToWords'
  ] as const) {
    expect(dataframeAnalysis[helper]).toBeTypeOf('function');
    expect(getGPUAnalysisOperator(helper)).toBeUndefined();
  }
});
