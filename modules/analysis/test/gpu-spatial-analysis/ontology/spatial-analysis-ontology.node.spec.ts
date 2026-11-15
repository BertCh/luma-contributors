// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {existsSync} from 'node:fs';
import {resolve} from 'node:path';
import * as spatialAnalysis from '../../../src/gpu-spatial-analysis/index';
import {GPUTerrainPeakProminence} from '../../../src/gpu-terrain/index';
import {GPU_SPATIAL_ANALYSIS_CONTRACT_NAMES} from '../../../src/analysis-core/contracts/index';
import {GPUContributorOperation, GPUParameterBuffer} from '../../../src/analysis-core/index';
import {
  REUSED_ANALYSIS_OPERATORS,
  getGPUAnalysisOperator,
  GPU_SPATIAL_ANALYSIS_CAPABILITIES,
  GPU_SPATIAL_ANALYSIS_AUDIT,
  GPU_SPATIAL_ANALYSIS_DATA_ONTOLOGY,
  GPU_SPATIAL_ANALYSIS_OPERATORS,
  formatGPUSpatialAnalysisAuditMarkdown,
  getGPUSpatialAnalysisCapability,
  getGPUSpatialAnalysisCapabilityConnections,
  getGPUSpatialAnalysisCapabilityForOperator,
  getGPUSpatialAnalysisOperator,
  queryGPUSpatialAnalysisCapabilities,
  queryGPUSpatialAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/index';
import {SPATIOTEMPORAL_REVIEW_OPERATORS} from '../../../src/gpu-spatial-analysis/ontology/spatiotemporal-review-catalog';

/**
 * Operators of `gpu-spatial-analysis` added after the original 261. They are catalogued in
 * `SPATIOTEMPORAL_REVIEW_OPERATORS` so that `GPU_SPATIAL_ANALYSIS_OPERATORS` and
 * `GPU_SPATIAL_ANALYSIS_CAPABILITIES` keep returning what the website generator and the kepler
 * bindings enumerate.
 */
const ADDED_SPATIAL_OPERATOR_NAMES = SPATIOTEMPORAL_REVIEW_OPERATORS.map(
  ({exportName}) => exportName
);

/**
 * A `GPU<Name>Operation` is the luma.gl GPUGraph form of a contributor that has its own catalogue
 * entry, not a further contributor (contributor contract, section 14), so it is not catalogued.
 */
function isOperationForm(value: unknown): boolean {
  return (
    typeof value === 'function' &&
    (value as {prototype?: unknown}).prototype instanceof GPUContributorOperation
  );
}

it('catalogues every public contributor and recipe exactly once', () => {
  const publicOperators = Object.entries({
    ...spatialAnalysis,
    GPUTerrainPeakProminence,
    GPUParameterBuffer
  })
    .filter(
      ([name, value]) =>
        typeof value === 'function' &&
        // The catalog also exposes this CPU display-boundary operation.
        (/^GPU[A-Z]/.test(name) || /^add[A-Z]/.test(name) || name === 'getHealpixCellBoundary') &&
        !isOperationForm(value)
    )
    .map(([name]) => name)
    .sort();
  const cataloguedOperators = [
    ...GPU_SPATIAL_ANALYSIS_CAPABILITIES.flatMap(capability => capability.operators),
    ...ADDED_SPATIAL_OPERATOR_NAMES
  ].sort();

  expect(new Set(cataloguedOperators).size).toBe(cataloguedOperators.length);
  expect(cataloguedOperators).toEqual(publicOperators);
});

it('describes every public contributor and recipe at operator level', () => {
  const publicOperators = Object.entries({
    ...spatialAnalysis,
    GPUTerrainPeakProminence,
    GPUParameterBuffer
  })
    .filter(
      ([name, value]) =>
        typeof value === 'function' &&
        (/^GPU[A-Z]/.test(name) || /^add[A-Z]/.test(name) || name === 'getHealpixCellBoundary') &&
        !isOperationForm(value)
    )
    .map(([name]) => name)
    .sort();

  expect(
    [
      ...GPU_SPATIAL_ANALYSIS_OPERATORS.map(({exportName}) => exportName),
      ...ADDED_SPATIAL_OPERATOR_NAMES
    ].sort()
  ).toEqual(publicOperators);
  expect(new Set(GPU_SPATIAL_ANALYSIS_OPERATORS.map(({exportName}) => exportName)).size).toBe(
    GPU_SPATIAL_ANALYSIS_OPERATORS.length
  );
  for (const operator of GPU_SPATIAL_ANALYSIS_OPERATORS) {
    expect(getGPUSpatialAnalysisOperator(operator.exportName)).toBe(operator);
    expect(getGPUSpatialAnalysisCapability(operator.primaryCapability)).toBeDefined();
    if (operator.evidence.status === 'proven') {
      expect(operator.inputs.length + operator.outputs.length).toBeGreaterThan(0);
    }
    expect(operator.evidence.tests).toBeTruthy();
    expect(operator.evidence.documentation).toBeTruthy();
    expect(operator.evidence.contract).toBeTruthy();
    expect(operator.evidence.representativeSourceRows).toBeGreaterThan(0);
    expect(operator.evidence.transientMemory).toBeTruthy();
    expect(operator.evidence.dispatch).toBeTruthy();
    expect(
      existsSync(resolve(operator.evidence.correctnessOracle)),
      operator.evidence.correctnessOracle
    ).toBe(true);
    expect(existsSync(resolve(operator.evidence.tests)), operator.evidence.tests).toBe(true);
    expect(
      existsSync(resolve(operator.evidence.documentation)),
      operator.evidence.documentation
    ).toBe(true);
    if (operator.evidence.status === 'proven') {
      for (const contract of operator.evidence.contract.split(' & ')) {
        expect(
          GPU_SPATIAL_ANALYSIS_CONTRACT_NAMES.includes(contract as never) ||
            contract in spatialAnalysis,
          `${operator.exportName} contract ${contract}`
        ).toBe(true);
      }
    }
  }
  expect(GPU_SPATIAL_ANALYSIS_AUDIT.operatorCount).toBe(
    publicOperators.length - ADDED_SPATIAL_OPERATOR_NAMES.length
  );
});

it('requires scale, memory, dispatch and correctness evidence for every available family', () => {
  const available = GPU_SPATIAL_ANALYSIS_CAPABILITIES.filter(
    capability => capability.status === 'available'
  );
  expect(GPU_SPATIAL_ANALYSIS_AUDIT.fullySpecifiedCapabilityFamilyCount).toBe(available.length);
  for (const capability of available) {
    const evidence = capability.evidence;
    expect(evidence, capability.id).toBeDefined();
    expect(evidence?.representativeSourceRows, capability.id).toBeGreaterThan(0);
    if (evidence?.representativeOutputRows !== undefined) {
      expect(evidence.representativeOutputRows, capability.id).toBeGreaterThan(0);
    }
    expect(evidence?.transientMemory, capability.id).toBeTruthy();
    expect(evidence?.dispatch, capability.id).toBeTruthy();
    expect(
      existsSync(resolve(evidence?.correctnessOracle ?? '')),
      `${capability.id}: ${evidence?.correctnessOracle}`
    ).toBe(true);
    for (const operator of queryGPUSpatialAnalysisOperators({
      primaryCapability: capability.id
    })) {
      expect(operator.evidence.representativeSourceRows).toBe(evidence?.representativeSourceRows);
      expect(operator.evidence.transientMemory).toBe(evidence?.transientMemory);
      expect(operator.evidence.dispatch).toBe(evidence?.dispatch);
      expect(operator.evidence.correctnessOracle).toBe(evidence?.correctnessOracle);
    }
  }
});

it('discovers operators by task, contract and evidence', () => {
  expect(
    queryGPUSpatialAnalysisOperators({output: 'pairs'}).map(({exportName}) => exportName)
  ).toContain('GPUSpatialPredicateJoin');
  expect(
    queryGPUSpatialAnalysisOperators({status: 'uncertainCount'}).map(({exportName}) => exportName)
  ).toContain('GPUSpatialPredicateJoin');
  expect(
    queryGPUSpatialAnalysisOperators({search: 'GPUCompactPairPort'}).map(
      ({exportName}) => exportName
    )
  ).toEqual(['GPUSpatialPredicateJoin', 'GPUPairGather']);
  expect(formatGPUSpatialAnalysisAuditMarkdown()).toContain('| spatial-joins | available |');
});

it('uses unique concept and capability identifiers with resolvable relationships', () => {
  const dataKindIds = GPU_SPATIAL_ANALYSIS_DATA_ONTOLOGY.map(concept => concept.id);
  const capabilityIds = GPU_SPATIAL_ANALYSIS_CAPABILITIES.map(capability => capability.id);
  expect(new Set(dataKindIds).size).toBe(dataKindIds.length);
  expect(new Set(capabilityIds).size).toBe(capabilityIds.length);

  for (const capability of GPU_SPATIAL_ANALYSIS_CAPABILITIES) {
    expect(getGPUSpatialAnalysisCapability(capability.id)).toBe(capability);
    for (const relatedId of capability.relatedCapabilities ?? []) {
      expect(getGPUSpatialAnalysisCapability(relatedId), `${capability.id} -> ${relatedId}`).toBe(
        definedCapability(relatedId)
      );
    }
  }
});

it('queries capabilities by analytical role, data flow and public operator', () => {
  expect(
    queryGPUSpatialAnalysisCapabilities({
      status: 'available',
      stage: 'relate',
      output: 'spatial-weights'
    }).map(capability => capability.id)
  ).toEqual(['spatial-joins', 'spatial-neighborhoods']);
  expect(queryGPUSpatialAnalysisCapabilities({search: 'voronoi'}).map(({id}) => id)).toEqual([
    'tessellation'
  ]);
  expect(getGPUSpatialAnalysisCapabilityForOperator('GPULocalMoran')?.id).toBe(
    'spatial-autocorrelation-and-inference'
  );
  expect(
    getGPUSpatialAnalysisCapabilityConnections('spatial-regression')?.related.map(({id}) => id)
  ).toEqual(['spatial-neighborhoods', 'advanced-spatial-models']);
});

function definedCapability(id: string) {
  const capability = getGPUSpatialAnalysisCapability(id);
  expect(capability).toBeDefined();
  return capability;
}

it('registers terrain prominence in its actual package with the independent reference', () => {
  const operator = getGPUSpatialAnalysisOperator('GPUTerrainPeakProminence');
  expect(operator).toMatchObject({
    package: '@luma-contributors/analysis/gpu-terrain',
    primaryCapability: 'terrain-peak-prominence',
    inputs: ['surface'],
    outputs: ['surface', 'attributes'],
    coordinateModels: ['grid'],
    cardinality: 'source-sized',
    deterministic: true,
    readback: 'none',
    evidence: {
      status: 'partial',
      contract: 'GPUTerrainPeakProminenceOutput',
      documentation: 'docs/api/gpu-terrain/operators.md',
      correctnessOracle: 'modules/analysis/test/oracles/terrain-peak-prominence.json'
    }
  });
  expect(getGPUSpatialAnalysisCapabilityForOperator('GPUHierarchicalDensityClustering')?.id).toBe(
    'group-and-regional-analysis'
  );
});

it('discovers exact-source topology preparation beside contiguity without treating tokens as metric coordinates', () => {
  const prepared = REUSED_ANALYSIS_OPERATORS.filter(
    operator => operator.package === '@luma-contributors/analysis/gpu-spatial-analysis'
  );
  expect(prepared.map(operator => operator.exportName)).toEqual([
    'preparePolygonContiguityTopology',
    'computePolygonContiguityOnCPU'
  ]);
  for (const operator of prepared) {
    expect(spatialAnalysis[operator.exportName as keyof typeof spatialAnalysis]).toBeTypeOf(
      'function'
    );
    expect(getGPUAnalysisOperator(operator.exportName)).toBe(operator);
    expect(operator.primaryCapability).toBe('spatial-neighborhoods');
    expect(operator.coordinateModels).toEqual(['topological']);
    expect(operator.limitation).toContain('zero snap tolerance');
  }
});
