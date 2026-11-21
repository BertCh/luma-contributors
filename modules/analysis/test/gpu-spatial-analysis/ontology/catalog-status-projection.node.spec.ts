// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';

const ONTOLOGY = '../../../src/gpu-spatial-analysis/ontology';

it('preserves omitted, audited-empty and named status through both public projections', async () => {
  // Hand-worked fixture: omission has no audit evidence; Empty was affirmatively audited;
  // Named declares overflow. Unqualified evidence must never audit a sibling package.
  const packageName = '@luma-contributors/analysis/gpu-spatial-analysis';
  const names = [
    'FixtureOmitted',
    'FixtureEmpty',
    'FixtureNamed',
    'FixtureUnqualified',
    'FixtureComplete'
  ];
  const evidence = {
    representativeSourceRows: 1,
    transientMemory: 'constant',
    dispatch: 'single-pass',
    correctnessOracle: 'hand-authored fixture'
  };
  let qualifiedAnnotationReads = 0;
  vi.resetModules();
  vi.doMock(`${ONTOLOGY}/spatial-analysis-catalog`, () => ({
    GPU_SPATIAL_ANALYSIS_CAPABILITIES: [
      {id: 'fixture', operators: names, evidence},
      {id: 'visibility-fixture', operators: ['addVisibilityGraphRecipe'], evidence}
    ]
  }));
  vi.doMock(`${ONTOLOGY}/coverage-expansion-catalog`, () => ({
    COVERAGE_EXPANSION_OPERATOR_OVERRIDES: {
      FixtureEmpty: {status: []},
      FixtureNamed: {status: ['overflow']},
      FixtureComplete: {status: ['complete']},
      FixtureUnqualified: {status: []}
    }
  }));
  vi.doMock(`${ONTOLOGY}/all-entry-points/operator-status-audits`, () => ({
    AUDITED_EMPTY_OPERATOR_STATUSES: {
      [`${packageName}:FixtureEmpty`]: 'Hand-audited fixture output has no scalar status',
      FixtureUnqualified: 'Bare names do not constitute qualified audit evidence',
      '@luma-contributors/analysis/gpu-network:FixtureOmitted': 'Sibling output only'
    }
  }));
  vi.doMock(`${ONTOLOGY}/all-entry-points/operator-annotations`, () => ({
    OPERATOR_ANNOTATIONS: {
      FixtureNamed: {tier: 'reference', limitation: 'bare fallback'},
      [`${packageName}:addVisibilityGraphRecipe`]: {
        limits: [{name: 'fixtureBound', maximum: 1, reason: 'spatial identity only'}]
      },
      '@luma-contributors/analysis/gpu-network:addVisibilityGraphRecipe': {
        limitation: 'network identity only'
      },
      get [`${packageName}:FixtureNamed`]() {
        qualifiedAnnotationReads++;
        return {tier: 'scale', limitation: 'qualified winner'};
      }
    }
  }));
  vi.doMock(`${ONTOLOGY}/all-entry-points/entry-point-capabilities`, () => ({
    ENTRY_POINT_CAPABILITIES: []
  }));
  try {
    const catalog = await import(
      '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog'
    );
    const historical = await import(
      '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-operators'
    );
    const fixtureRecords = catalog.queryGPUAnalysisOperators({primaryCapability: 'fixture'});
    const [omitted, empty, named, unqualified, complete] = names.map(exportName =>
      catalog.getGPUAnalysisOperator({package: packageName, exportName})
    );
    expect(fixtureRecords).toEqual([omitted, empty, named, unqualified, complete]);
    expect(omitted?.status).toBe('unknown');
    expect(empty?.status).toEqual([]);
    expect(named?.status).toEqual(['overflow']);
    expect(unqualified?.status).toBe('unknown');
    expect(complete?.status).toEqual(['complete']);
    expect(
      catalog.queryGPUAnalysisOperators({primaryCapability: 'fixture', status: 'complete'})
    ).toEqual([complete]);
    expect(historical.queryGPUSpatialAnalysisOperators({status: 'complete'})).toEqual([
      historical.getGPUSpatialAnalysisOperator('FixtureComplete')
    ]);
    expect(
      catalog.queryGPUAnalysisOperators({primaryCapability: 'fixture', status: 'unknown'})
    ).toEqual([omitted, unqualified]);
    expect(
      catalog.queryGPUAnalysisOperators({primaryCapability: 'fixture', status: 'overflow'})
    ).toEqual([named]);
    expect(names.map(name => historical.getGPUSpatialAnalysisOperator(name)?.status)).toEqual([
      [],
      [],
      ['overflow'],
      [],
      ['complete']
    ]);
    expect(historical.queryGPUSpatialAnalysisOperators({status: 'overflow'})).toEqual([
      historical.getGPUSpatialAnalysisOperator('FixtureNamed')
    ]);
    expect(named).toMatchObject({tier: 'scale', limitation: 'qualified winner'});
    expect(qualifiedAnnotationReads).toBe(1);
    const networkVisibility = catalog.getGPUAnalysisOperator({
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'addVisibilityGraphRecipe'
    });
    expect(networkVisibility?.limitation).toBe('network identity only');
    // A spatial-only annotation must never become a network default.
    expect(networkVisibility?.limits?.map(limit => [limit.name, limit.maximum])).toEqual([
      ['positions.length (csr)', 1000000],
      ['edgeCapacity (csr)', 16777216],
      ['positions.length (reference)', 256]
    ]);
    expect(networkVisibility?.limits?.some(limit => limit.name === 'fixtureBound')).toBe(false);
    expect(historical.getGPUSpatialAnalysisOperator('addVisibilityGraphRecipe')?.limits).toEqual([
      {name: 'fixtureBound', maximum: 1, reason: 'spatial identity only'}
    ]);
    expect(Object.isFrozen(empty?.status)).toBe(true);
  } finally {
    for (const path of [
      'spatial-analysis-catalog',
      'coverage-expansion-catalog',
      'all-entry-points/operator-status-audits',
      'all-entry-points/operator-annotations',
      'all-entry-points/entry-point-capabilities'
    ])
      vi.doUnmock(`${ONTOLOGY}/${path}`);
    vi.resetModules();
  }
});

it('S3 resolves an unaudited reused declaration through both public access paths', async () => {
  vi.resetModules();
  // Simulate the evidence state of a newly added reused descriptor: its explicit [] has no
  // affirmative audit. Removing only this audit leaves every unrelated production audit intact.
  vi.doMock(`${ONTOLOGY}/all-entry-points/operator-status-audits`, async () => {
    const actual = await vi.importActual<{
      AUDITED_EMPTY_OPERATOR_STATUSES: Readonly<Record<string, string>>;
    }>(`${ONTOLOGY}/all-entry-points/operator-status-audits`);
    const audits = {...actual.AUDITED_EMPTY_OPERATOR_STATUSES};
    delete audits['@luma-contributors/analysis/dataframe-analysis:GPUWideNumericScan'];
    return {AUDITED_EMPTY_OPERATOR_STATUSES: audits};
  });
  try {
    const {REUSED_ANALYSIS_OPERATORS} = await import(
      '../../../src/gpu-spatial-analysis/ontology/reused-analysis-catalog'
    );
    const {getGPUAnalysisOperator} = await import(
      '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog'
    );
    const direct = REUSED_ANALYSIS_OPERATORS.find(
      record => record.exportName === 'GPUWideNumericScan'
    );
    expect(direct?.status).toBe('unknown');
    expect(getGPUAnalysisOperator('GPUWideNumericScan')).toBe(direct);
    expect(Object.isFrozen(direct)).toBe(true);
  } finally {
    vi.doUnmock(`${ONTOLOGY}/all-entry-points/operator-status-audits`);
    vi.resetModules();
  }
});
