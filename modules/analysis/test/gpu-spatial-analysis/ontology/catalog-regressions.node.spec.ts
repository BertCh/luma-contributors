// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {readFileSync} from 'node:fs';
import {OPERATOR_ANNOTATIONS} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/operator-annotations';
import {
  findGPUAnalysisOperatorPaths,
  getGPUAnalysisNextOperators,
  queryGPUAnalysisOperatorsByKind
} from '../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-operator-kind-supplements';
import {REUSED_ANALYSIS_OPERATORS} from '../../../src/gpu-spatial-analysis/ontology/reused-analysis-catalog';
import {getGPUAnalysisDataKindRecord} from '../../../src/gpu-spatial-analysis/ontology/analysis-experience/data-kind-vocabulary';
import {
  getGPUAnalysisCapabilityCatalog,
  getGPUAnalysisOperator,
  getGPUAnalysisOperatorCatalog,
  queryGPUAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';
import {GPU_SPATIAL_ANALYSIS_DATA_KINDS} from '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-ontology';
import {isGPUSpatialAnalysisDataKind} from '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-operators';
import {
  classifyGPUAnalysisResultValues,
  validateGPUAnalysisResultEncoding
} from '../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-interpretation';

it('counts every nonzero uint32 validity flag as usable and fully encoded', () => {
  const values = Uint32Array.of(0, 1, 2, 0xffffffff);
  const tally = classifyGPUAnalysisResultValues('validity-flag', values);
  expect(tally.counts.get(1)).toBe(3);
  expect(tally.counts.get(0)).toBe(1);
  expect(tally.unrecognized).toBe(0);
  expect(validateGPUAnalysisResultEncoding('validity-flag', 'contributor-integer', values)).toBe(1);
  expect(
    classifyGPUAnalysisResultValues('validity-flag', [-1, 0.5, 2 ** 32, '2']).unrecognized
  ).toBe(4);
  expect(classifyGPUAnalysisResultValues('significance-flag', [2]).unrecognized).toBe(1);
});

it('retains resolvable qualified visibility identities in paths and next operators', () => {
  const records = queryGPUAnalysisOperatorsByKind({input: 'points', output: 'network'});
  const result = findGPUAnalysisOperatorPaths('points', 'network', 1, 500);
  expect(result.totalFound).toBe(records.length);
  expect(result.paths.flat().map(step => getGPUAnalysisOperator(step))).toEqual(
    expect.arrayContaining([...records])
  );
  const next = getGPUAnalysisNextOperators('points');
  const visibility = next.filter(
    step => typeof step !== 'string' && step.exportName === 'addVisibilityGraphRecipe'
  );
  expect(visibility).toHaveLength(2);
  for (const step of visibility) expect(getGPUAnalysisOperator(step)).toBeDefined();
});

it('publishes the actual spatial and network visibility output and status contracts', () => {
  const names = (query: Parameters<typeof queryGPUAnalysisOperators>[0]) =>
    queryGPUAnalysisOperators(query)
      .filter(operator => operator.exportName === 'addVisibilityGraphRecipe')
      .map(operator => operator.package)
      .sort();
  expect(names({status: 'candidateOverflow'})).toEqual([
    '@luma-contributors/analysis/gpu-network',
    '@luma-contributors/analysis/gpu-spatial-analysis'
  ]);
  expect(names({status: 'zeroWeightCycle'})).toEqual(['@luma-contributors/analysis/gpu-network']);
  expect(names({output: 'polygons'})).toEqual([]);
});

it('protects canonical kinds and reused catalog arrays from mutation across callers', () => {
  const index = GPU_SPATIAL_ANALYSIS_DATA_KINDS.indexOf('time-series');
  const changed = Reflect.set(GPU_SPATIAL_ANALYSIS_DATA_KINDS, index, 'replacement');
  if (changed) Reflect.set(GPU_SPATIAL_ANALYSIS_DATA_KINDS, index, 'time-series');
  expect(changed).toBe(false);
  expect(isGPUSpatialAnalysisDataKind('time-series')).toBe(true);
  expect(isGPUSpatialAnalysisDataKind('replacement')).toBe(false);
  expect(getGPUAnalysisDataKindRecord('time-series')).toBeDefined();
  const last = REUSED_ANALYSIS_OPERATORS.at(-1);
  const shortened = Reflect.set(REUSED_ANALYSIS_OPERATORS, 'length', 8);
  if (shortened) Reflect.set(REUSED_ANALYSIS_OPERATORS, 8, last);
  expect(shortened).toBe(false);
  expect(REUSED_ANALYSIS_OPERATORS).toHaveLength(9);
  for (const operator of REUSED_ANALYSIS_OPERATORS)
    expect(
      getGPUAnalysisOperator({package: operator.package, exportName: operator.exportName})
    ).toMatchObject(operator);
});

it('keeps unaudited metadata unknown instead of claiming no status, readback or scale limits', () => {
  const recorder = getGPUAnalysisOperator('GPUComputationRecorder')!;
  expect(recorder.status).toBe('unknown');
  expect(recorder.readback).toBe('unknown');
  expect(recorder.tier).toBe('unknown');
  expect(queryGPUAnalysisOperators({status: 'unknown', includeHelpers: true})).toContain(recorder);
  expect(queryGPUAnalysisOperators({readback: 'unknown', includeHelpers: true})).toContain(
    recorder
  );
});

it('finds audited k-shortest-path statuses and explicit snapshot readback', () => {
  for (const status of ['candidateOverflow', 'converged', 'iterationLimitReached'] as const) {
    expect(queryGPUAnalysisOperators({status}).map(operator => operator.exportName)).toContain(
      'GPUKShortestPaths'
    );
  }
  expect(getGPUAnalysisOperator('GPUSnapshotArena')?.readback).toBe('explicit-adapter');
});

it('discovers Gaussian random fields in both point and grid workflows', () => {
  expect(
    queryGPUAnalysisOperators({input: 'surface', output: 'surface'}).map(
      operator => operator.exportName
    )
  ).toContain('GPUGaussianRandomField');
  expect(
    queryGPUAnalysisOperators({input: 'points', output: 'attributes'}).map(
      operator => operator.exportName
    )
  ).toContain('GPUGaussianRandomField');
});

it('publishes the fitting encoder annotation through the catalog', () => {
  expect(getGPUAnalysisOperator('GPUFittingCoordinateEncoder')?.limitation).toContain(
    'chunked rows'
  );
});

it('applies each public annotation to exactly one final identity', () => {
  for (const [key, annotation] of Object.entries(OPERATOR_ANNOTATIONS)) {
    const records = getGPUAnalysisOperatorCatalog().filter(operator =>
      key.includes(':')
        ? key === `${operator.package}:${operator.exportName}`
        : key === operator.exportName
    );
    expect(records, key).toHaveLength(1);
    expect(records[0], key).toMatchObject(annotation);
  }
});

it('records every enforced network inhomogeneous K size bound', () => {
  const limits = getGPUAnalysisOperator('GPUNetworkInhomogeneousK')!.limits!;
  expect(limits.find(limit => limit.name === 'nodeCount')?.maximum).toBe(65536);
  expect(limits.find(limit => limit.name === 'edgeCount')?.maximum).toBe(65536);
  expect(limits.find(limit => limit.name === 'eventCount * nodeCount')?.maximum).toBe(16777216);
  const source = readFileSync(
    'modules/analysis/src/gpu-network/gpu-network-inhomogeneous-k.ts',
    'utf8'
  );
  for (const limit of limits) {
    const constant = limit.reason.match(/^(MAXIMUM_[A-Z_]+):/)?.[1];
    expect(constant, limit.name).toBeTruthy();
    const value = source.match(new RegExp(`const ${constant} = ([0-9_]+);`))?.[1];
    expect(Number(value?.replaceAll('_', '')), limit.name).toBe(limit.maximum);
  }
});

it('distinguishes the two public visibility recipes by entry point and rejects ambiguous names', () => {
  const exportName = 'addVisibilityGraphRecipe';
  const spatial = getGPUAnalysisOperator({
    package: '@luma-contributors/analysis/gpu-spatial-analysis',
    exportName
  });
  const network = getGPUAnalysisOperator({
    package: '@luma-contributors/analysis/gpu-network',
    exportName
  });
  expect(spatial?.package).toBe('@luma-contributors/analysis/gpu-spatial-analysis');
  expect(network?.package).toBe('@luma-contributors/analysis/gpu-network');
  expect(spatial).not.toBe(network);
  expect(spatial?.limitation).toContain('adjacency');
  expect(network?.limitation).toContain('sampled');
  expect(() => getGPUAnalysisOperator(exportName)).toThrow(/ambiguous.*package/i);
});

it('recognizes every canonical data kind even when historical families omit it', () => {
  for (const kind of GPU_SPATIAL_ANALYSIS_DATA_KINDS) {
    expect(isGPUSpatialAnalysisDataKind(kind), kind).toBe(true);
  }
  expect(isGPUSpatialAnalysisDataKind('not-a-kind')).toBe(false);
});

it('freezes shared catalog records and their nested values', () => {
  const operator = getGPUAnalysisOperator('GPUNetworkFacilityOptimization')!;
  expect(Object.isFrozen(getGPUAnalysisOperatorCatalog())).toBe(true);
  expect(Object.isFrozen(operator)).toBe(true);
  expect(Object.isFrozen(operator.evidence)).toBe(true);
  expect(Object.isFrozen(operator.limits)).toBe(true);
  expect(Object.isFrozen(operator.limits![0])).toBe(true);
  expect(Reflect.set(operator.evidence, 'contract', 'Changed')).toBe(false);
  expect(Reflect.set(operator.limits![0], 'maximum', 1)).toBe(false);
  expect(Reflect.set(operator, 'exportName', 'Changed')).toBe(false);
  expect(getGPUAnalysisOperator('GPUNetworkFacilityOptimization')?.exportName).toBe(
    'GPUNetworkFacilityOptimization'
  );
});

it('derives all-entry family membership from every finalized operator record', () => {
  for (const capability of getGPUAnalysisCapabilityCatalog()) {
    const records = getGPUAnalysisOperatorCatalog().filter(
      operator => operator.primaryCapability === capability.id
    );
    expect(capability.operators, capability.id).toEqual(
      records.map(operator => operator.exportName)
    );
    expect(capability.operatorIdentities, capability.id).toEqual(
      records.map(({package: packageName, exportName}) => ({package: packageName, exportName}))
    );
    expect(Object.isFrozen(capability)).toBe(true);
    expect(Object.isFrozen(capability.operatorIdentities)).toBe(true);
  }
});
