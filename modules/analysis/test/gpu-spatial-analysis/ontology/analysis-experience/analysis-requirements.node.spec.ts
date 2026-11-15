// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  checkGPUAnalysisCoordinateDeclaration,
  evaluateGPUAnalysisRequirement,
  evaluateGPUAnalysisRequirements,
  summarizeGPUAnalysisRequirementResults
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-requirements';
import {
  GPU_ANALYSIS_REQUIREMENT_IDS,
  type GPUAnalysisColumnProfile,
  type GPUAnalysisDataProfile,
  type GPUAnalysisRequirementId,
  type GPUAnalysisRequirementResult,
  type GPUAnalysisRequirementSpec,
  type GPUAnalysisSpacingStatistics
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-experience-types';

function makeColumn(
  name: string,
  overrides: Partial<GPUAnalysisColumnProfile> = {}
): GPUAnalysisColumnProfile {
  return {name, type: 'number', rowCount: 1000, nullCount: 0, ...overrides};
}

function makeSpacing(
  overrides: Partial<GPUAnalysisSpacingStatistics> = {}
): GPUAnalysisSpacingStatistics {
  return {
    pointCount: 1000,
    sampleSize: 1000,
    sampled: false,
    coordinateModel: 'geographic',
    unit: 'metre',
    extent: [0, 0, 1, 1],
    extentWidth: 100,
    extentHeight: 100,
    extentDiagonal: 141,
    center: [0.5, 0.5],
    standardDistance: 40,
    medianDistanceFromCenter: 35,
    nearestNeighbor: {
      minimum: 1,
      lowerQuartile: 3,
      median: 5,
      upperQuartile: 8,
      mean: 6,
      maximum: 30,
      upperFence: 15,
      aboveUpperFenceCount: 2
    },
    duplicateLocationCount: 0,
    ...overrides
  };
}

function makeProfile(overrides: Partial<GPUAnalysisDataProfile> = {}): GPUAnalysisDataProfile {
  return {
    kind: 'points',
    rowCount: 1000,
    columns: [makeColumn('value', {minimum: 1, maximum: 9, distinctCount: 8})],
    ...overrides
  };
}

function spec(
  id: GPUAnalysisRequirementId,
  overrides: Partial<GPUAnalysisRequirementSpec> = {}
): GPUAnalysisRequirementSpec {
  return {id, severity: 'block', ...overrides};
}

const bound = {value: 'value'};

describe('evaluateGPUAnalysisRequirement: counts', () => {
  it('minimum-rows passes at the limit and fails one below it', () => {
    const limit = spec('minimum-rows', {limit: 30, remedy: 'Load more rows'});
    const atLimit = evaluateGPUAnalysisRequirement(limit, makeProfile({rowCount: 30}), {});
    expect(atLimit).toMatchObject({
      state: 'passed',
      evidence: 'profile',
      message: '30 rows; at least 30 are needed'
    });
    expect(atLimit.remedy).toBeUndefined();
    const below = evaluateGPUAnalysisRequirement(limit, makeProfile({rowCount: 29}), {});
    expect(below).toMatchObject({state: 'failed', remedy: 'Load more rows'});
  });

  it('minimum-rows formats numbers for reading and uses the singular', () => {
    const result = evaluateGPUAnalysisRequirement(
      spec('minimum-rows', {limit: 1_000_000}),
      makeProfile({rowCount: 12400}),
      {}
    );
    expect(result.message).toBe('12,400 rows; at least 1,000,000 are needed');
    expect(
      evaluateGPUAnalysisRequirement(
        spec('minimum-rows', {limit: 2}),
        makeProfile({rowCount: 1}),
        {}
      ).message
    ).toBe('1 row; at least 2 are needed');
    expect(
      evaluateGPUAnalysisRequirement(
        spec('minimum-rows', {limit: 1}),
        makeProfile({rowCount: 0}),
        {}
      )
    ).toMatchObject({
      state: 'failed',
      message: '0 rows; at least 1 is needed'
    });
  });

  it('maximum-rows passes at the limit and fails one above it', () => {
    const limit = spec('maximum-rows', {limit: 5000});
    expect(evaluateGPUAnalysisRequirement(limit, makeProfile({rowCount: 5000}), {}).state).toBe(
      'passed'
    );
    const above = evaluateGPUAnalysisRequirement(limit, makeProfile({rowCount: 5001}), {});
    expect(above).toMatchObject({
      state: 'failed',
      message: '5,001 rows; at most 5,000 are supported'
    });
  });

  it('maximum-vertices uses the vertex count, the row count for points, and is unknown for polygons without one', () => {
    const limit = spec('maximum-vertices', {limit: 100000});
    expect(
      evaluateGPUAnalysisRequirement(
        limit,
        makeProfile({kind: 'polygons', rowCount: 10, vertexCount: 100000}),
        {}
      ).state
    ).toBe('passed');
    expect(
      evaluateGPUAnalysisRequirement(
        limit,
        makeProfile({kind: 'polygons', rowCount: 10, vertexCount: 100001}),
        {}
      )
    ).toMatchObject({
      state: 'failed',
      message: '100,001 vertices; at most 100,000 are supported'
    });
    expect(
      evaluateGPUAnalysisRequirement(limit, makeProfile({kind: 'points', rowCount: 100001}), {})
        .state
    ).toBe('failed');
    expect(
      evaluateGPUAnalysisRequirement(limit, makeProfile({kind: 'polygons', rowCount: 10}), {})
    ).toMatchObject({
      state: 'unknown',
      evidence: 'not-checkable'
    });
  });

  it('throws for a missing, negative or non-finite limit', () => {
    for (const id of ['minimum-rows', 'maximum-rows', 'maximum-vertices'] as const) {
      expect(() => evaluateGPUAnalysisRequirement(spec(id), makeProfile(), {})).toThrow(/limit/);
      expect(() =>
        evaluateGPUAnalysisRequirement(spec(id, {limit: -1}), makeProfile(), {})
      ).toThrow(/at least 0/);
      expect(() =>
        evaluateGPUAnalysisRequirement(spec(id, {limit: Number.NaN}), makeProfile(), {})
      ).toThrow(/evaluateGPUAnalysisRequirement/);
    }
  });
});

describe('evaluateGPUAnalysisRequirement: coordinates', () => {
  it('planar-coordinates and geographic-coordinates are unknown without coordinates', () => {
    for (const id of ['planar-coordinates', 'geographic-coordinates'] as const) {
      expect(evaluateGPUAnalysisRequirement(spec(id), makeProfile(), {})).toMatchObject({
        state: 'unknown',
        evidence: 'not-checkable'
      });
    }
  });

  it('passes and fails on a declared model with evidence declaration', () => {
    const planar = makeProfile({coordinates: {model: 'planar', source: 'declared'}});
    expect(evaluateGPUAnalysisRequirement(spec('planar-coordinates'), planar, {})).toMatchObject({
      state: 'passed',
      evidence: 'declaration'
    });
    expect(
      evaluateGPUAnalysisRequirement(spec('geographic-coordinates'), planar, {})
    ).toMatchObject({state: 'failed', evidence: 'declaration'});
  });

  it('gives evidence guess for a guessed model', () => {
    const geographic = makeProfile({
      coordinates: {model: 'geographic', source: 'guess', reason: 'extent'}
    });
    const passed = evaluateGPUAnalysisRequirement(spec('geographic-coordinates'), geographic, {});
    expect(passed).toMatchObject({state: 'passed', evidence: 'guess'});
    expect(passed.message).toContain('guessed from the extent');
    const contradicted = evaluateGPUAnalysisRequirement(spec('planar-coordinates'), geographic, {});
    expect(contradicted).toMatchObject({
      state: 'unknown',
      evidence: 'guess',
      remedy: 'Confirm the coordinate system'
    });
    expect(contradicted.message).toBe(
      'The coordinates look geographic (guessed from the extent); planar coordinates are needed'
    );
  });

  it('never fails on a guess in either direction and keeps a remedy the spec supplies', () => {
    const planarGuess = makeProfile({coordinates: {model: 'planar', source: 'guess'}});
    const result = evaluateGPUAnalysisRequirement(
      spec('geographic-coordinates', {remedy: 'Reproject to degrees'}),
      planarGuess,
      {}
    );
    expect(result).toMatchObject({state: 'unknown', evidence: 'guess'});
    expect(result.remedy).toBe('Confirm the coordinate system; Reproject to degrees');
    expect(evaluateGPUAnalysisRequirement(spec('planar-coordinates'), planarGuess, {}).state).toBe(
      'passed'
    );
  });

  it('keeps a declared model failing outright, with no confirm remedy', () => {
    const declared = makeProfile({coordinates: {model: 'geographic', source: 'declared'}});
    const result = evaluateGPUAnalysisRequirement(spec('planar-coordinates'), declared, {});
    expect(result).toMatchObject({state: 'failed', evidence: 'declaration'});
    expect(result.remedy).toBeUndefined();
  });

  it('fails a declared geographic model whose extent cannot be degrees', () => {
    const declared = makeProfile({
      coordinates: {model: 'geographic', source: 'declared'},
      extent: [500000, 4100000, 600000, 4200000]
    });
    for (const id of ['geographic-coordinates', 'planar-coordinates'] as const) {
      const result = evaluateGPUAnalysisRequirement(spec(id), declared, {});
      expect(result, id).toMatchObject({state: 'failed', evidence: 'declaration'});
      expect(result.message, id).toContain('cannot be degrees');
    }
    const fromSpacing = makeProfile({
      coordinates: {model: 'geographic', source: 'declared'},
      spacing: makeSpacing({extent: [0, 0, 2000, 1000]})
    });
    expect(
      evaluateGPUAnalysisRequirement(spec('geographic-coordinates'), fromSpacing, {}).state
    ).toBe('failed');
  });

  it('accepts a declared geographic extent, including longitudes 0 to 360, and never contradicts planar', () => {
    for (const extent of [
      [-122, 37, -121, 38],
      [140, -40, 290, 30]
    ] as const) {
      expect(
        evaluateGPUAnalysisRequirement(
          spec('geographic-coordinates'),
          makeProfile({coordinates: {model: 'geographic', source: 'declared'}, extent}),
          {}
        ).state
      ).toBe('passed');
    }
    expect(
      evaluateGPUAnalysisRequirement(
        spec('planar-coordinates'),
        makeProfile({coordinates: {model: 'planar', source: 'declared'}, extent: [-5, -5, 9, 9]}),
        {}
      ).state
    ).toBe('passed');
  });

  it('checkGPUAnalysisCoordinateDeclaration judges the extent at its limits', () => {
    expect(
      checkGPUAnalysisCoordinateDeclaration([-180, -90, 360, 90], 'geographic').plausible
    ).toBe(true);
    expect(checkGPUAnalysisCoordinateDeclaration([-180.001, 0, 0, 0], 'geographic').plausible).toBe(
      false
    );
    expect(checkGPUAnalysisCoordinateDeclaration([0, 0, 360.001, 0], 'geographic').plausible).toBe(
      false
    );
    expect(checkGPUAnalysisCoordinateDeclaration([0, -90.001, 0, 0], 'geographic').plausible).toBe(
      false
    );
    expect(checkGPUAnalysisCoordinateDeclaration([0, 0, 0, 90.001], 'geographic').plausible).toBe(
      false
    );
    expect(checkGPUAnalysisCoordinateDeclaration([-9e9, -9e9, 9e9, 9e9], 'planar').plausible).toBe(
      true
    );
    expect(
      checkGPUAnalysisCoordinateDeclaration([Number.NaN, 0, 1, 1], 'geographic').plausible
    ).toBe(true);
  });

  it('declared-coordinates passes only for a declaration', () => {
    expect(
      evaluateGPUAnalysisRequirement(spec('declared-coordinates'), makeProfile(), {})
    ).toMatchObject({state: 'unknown', evidence: 'not-checkable'});
    expect(
      evaluateGPUAnalysisRequirement(
        spec('declared-coordinates'),
        makeProfile({coordinates: {model: 'planar', source: 'guess'}}),
        {}
      )
    ).toMatchObject({state: 'unknown', evidence: 'guess'});
    expect(
      evaluateGPUAnalysisRequirement(
        spec('declared-coordinates'),
        makeProfile({coordinates: {model: 'planar', source: 'declared'}}),
        {}
      )
    ).toMatchObject({state: 'passed', evidence: 'declaration'});
  });
});

describe('evaluateGPUAnalysisRequirement: row semantics', () => {
  const requirement = spec('row-semantics', {expects: ['events', 'objects']});

  it('is unknown until semantics are present', () => {
    expect(evaluateGPUAnalysisRequirement(requirement, makeProfile(), {})).toMatchObject({
      state: 'unknown',
      evidence: 'not-checkable'
    });
  });

  it('passes and fails with evidence from the source', () => {
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        makeProfile({rowSemantics: {value: 'events', source: 'guess'}}),
        {}
      )
    ).toMatchObject({state: 'passed', evidence: 'guess'});
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        makeProfile({rowSemantics: {value: 'objects', source: 'declared'}}),
        {}
      )
    ).toMatchObject({state: 'passed', evidence: 'declaration'});
    const failed = evaluateGPUAnalysisRequirement(
      requirement,
      makeProfile({rowSemantics: {value: 'surface-samples', source: 'declared'}}),
      {}
    );
    expect(failed).toMatchObject({state: 'failed', evidence: 'declaration'});
    expect(failed.message).toContain('events or objects');
  });

  it('throws when the requirement expects nothing', () => {
    expect(() => evaluateGPUAnalysisRequirement(spec('row-semantics'), makeProfile(), {})).toThrow(
      /expects/
    );
    expect(() =>
      evaluateGPUAnalysisRequirement(spec('row-semantics', {expects: []}), makeProfile(), {})
    ).toThrow(/expects/);
  });
});

describe('evaluateGPUAnalysisRequirement: value requirements', () => {
  const profileWith = (column: GPUAnalysisColumnProfile) => makeProfile({columns: [column]});

  it('positive-values compares the minimum with 0', () => {
    const requirement = spec('positive-values', {role: 'value'});
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        profileWith(makeColumn('value', {minimum: 0.0001})),
        bound
      )
    ).toMatchObject({state: 'passed', evidence: 'profile', role: 'value'});
    const zero = evaluateGPUAnalysisRequirement(
      requirement,
      profileWith(makeColumn('value', {minimum: 0})),
      bound
    );
    expect(zero).toMatchObject({
      state: 'failed',
      message: 'value has a minimum of 0; every value must be above 0'
    });
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        profileWith(makeColumn('value', {minimum: -5})),
        bound
      ).state
    ).toBe('failed');
  });

  it('non-negative-values accepts exactly 0', () => {
    const requirement = spec('non-negative-values', {role: 'value'});
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        profileWith(makeColumn('value', {minimum: 0})),
        bound
      ).state
    ).toBe('passed');
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        profileWith(makeColumn('value', {minimum: -0.001})),
        bound
      ).state
    ).toBe('failed');
  });

  it('no-missing-values compares the null count with 0', () => {
    const requirement = spec('no-missing-values', {role: 'value'});
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        profileWith(makeColumn('value', {nullCount: 0})),
        bound
      ).state
    ).toBe('passed');
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        profileWith(makeColumn('value', {nullCount: 1})),
        bound
      )
    ).toMatchObject({
      state: 'failed',
      message: 'value has 1 missing value; none are allowed'
    });
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        profileWith(makeColumn('value', {nullCount: 1200})),
        bound
      ).message
    ).toContain('1,200 missing values');
  });

  it('varying-values uses minimum below maximum, then the distinct count', () => {
    const requirement = spec('varying-values', {role: 'value'});
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        profileWith(makeColumn('value', {minimum: 3, maximum: 3})),
        bound
      ).state
    ).toBe('failed');
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        profileWith(makeColumn('value', {minimum: 3, maximum: 3.0001})),
        bound
      ).state
    ).toBe('passed');
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        profileWith(makeColumn('value', {type: 'category', distinctCount: 1})),
        bound
      ).state
    ).toBe('failed');
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        profileWith(makeColumn('value', {type: 'category', distinctCount: 2})),
        bound
      ).state
    ).toBe('passed');
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        profileWith(makeColumn('value', {type: 'category'})),
        bound
      )
    ).toMatchObject({state: 'unknown', evidence: 'not-checkable'});
  });

  it('minimum-distinct-values compares at the limit and says about for an estimate', () => {
    const requirement = spec('minimum-distinct-values', {role: 'value', limit: 5});
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        profileWith(makeColumn('value', {distinctCount: 5})),
        bound
      ).state
    ).toBe('passed');
    const below = evaluateGPUAnalysisRequirement(
      requirement,
      profileWith(makeColumn('value', {distinctCount: 4})),
      bound
    );
    expect(below).toMatchObject({
      state: 'failed',
      message: 'value has 4 distinct values; at least 5 are needed'
    });
    const estimate = evaluateGPUAnalysisRequirement(
      requirement,
      profileWith(makeColumn('value', {distinctCount: 4000, distinctCountIsEstimate: true})),
      bound
    );
    expect(estimate).toMatchObject({state: 'unknown'});
    expect(estimate.message).toContain(
      'value has about 4,000 distinct values; at least 5 are needed'
    );
    expect(estimate.message).toContain('no guaranteed error bound');
    expect(
      evaluateGPUAnalysisRequirement(requirement, profileWith(makeColumn('value')), bound)
    ).toMatchObject({state: 'unknown', evidence: 'not-checkable'});
    expect(() =>
      evaluateGPUAnalysisRequirement(
        spec('minimum-distinct-values', {role: 'value'}),
        profileWith(makeColumn('value', {distinctCount: 5})),
        bound
      )
    ).toThrow(/limit/);
  });

  it('is unknown when no column is bound, the column is missing, or the statistic is absent', () => {
    const requirement = spec('positive-values', {role: 'value'});
    expect(evaluateGPUAnalysisRequirement(requirement, makeProfile(), {})).toMatchObject({
      state: 'unknown',
      evidence: 'not-checkable'
    });
    expect(evaluateGPUAnalysisRequirement(requirement, makeProfile(), {value: []})).toMatchObject({
      state: 'unknown'
    });
    expect(
      evaluateGPUAnalysisRequirement(requirement, makeProfile(), {value: 'absent'}).message
    ).toContain('no column named absent');
    expect(
      evaluateGPUAnalysisRequirement(requirement, profileWith(makeColumn('value')), bound).message
    ).toContain('No minimum was computed');
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        profileWith(makeColumn('value', {minimum: Number.NaN})),
        bound
      ).state
    ).toBe('unknown');
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        profileWith(makeColumn('value', {rowCount: 5, nullCount: 5, type: 'number'})),
        bound
      ).state
    ).toBe('unknown');
  });

  it('throws when a value requirement names no role', () => {
    expect(() =>
      evaluateGPUAnalysisRequirement(spec('positive-values'), makeProfile(), bound)
    ).toThrow(/role/);
  });

  describe('sampled statistics', () => {
    const sampled = (overrides: Partial<GPUAnalysisColumnProfile>) =>
      makeColumn('value', {sampleSize: 500, rowCount: 1900, ...overrides});
    const evaluate = (
      id: GPUAnalysisRequirementId,
      column: GPUAnalysisColumnProfile,
      extra: Partial<GPUAnalysisRequirementSpec> = {}
    ) =>
      evaluateGPUAnalysisRequirement(
        spec(id, {role: 'value', ...extra}),
        profileWith(column),
        bound
      );

    it('universal properties: a sample can refute them but never establish them', () => {
      for (const [id, passing, failing] of [
        ['positive-values', {minimum: 2}, {minimum: 0}],
        ['non-negative-values', {minimum: 0}, {minimum: -1}],
        ['no-missing-values', {nullCount: 0}, {nullCount: 2}]
      ] as const) {
        const passes = evaluate(id, sampled(passing));
        expect(passes, id).toMatchObject({state: 'unknown', evidence: 'sampled-profile'});
        expect(passes.message).toContain('in a sample of 500 of 1,900 rows');
        expect(passes.message).toContain('cannot show it holds for every row');
        expect(evaluate(id, sampled(failing)), id).toMatchObject({
          state: 'failed',
          evidence: 'sampled-profile'
        });
      }
    });

    it('existential properties: a sample can establish them but never refute them', () => {
      for (const [id, passing, failing, extra] of [
        ['varying-values', {minimum: 5, maximum: 9}, {minimum: 5, maximum: 5}, {}],
        ['minimum-distinct-values', {distinctCount: 6}, {distinctCount: 1}, {limit: 5}]
      ] as const) {
        expect(evaluate(id, sampled(passing), extra), id).toMatchObject({
          state: 'passed',
          evidence: 'sampled-profile'
        });
        const refuted = evaluate(id, sampled(failing), extra);
        expect(refuted, id).toMatchObject({state: 'unknown', evidence: 'sampled-profile'});
        expect(refuted.message).toContain('rule it out for the whole column');
      }
    });

    it('does not treat a full-size sampleSize as sampled', () => {
      const exact = makeColumn('value', {sampleSize: 1900, rowCount: 1900, minimum: 2});
      expect(evaluate('positive-values', exact)).toMatchObject({
        state: 'passed',
        evidence: 'profile'
      });
    });

    it('blocks nothing on a sample that happens to see a constant column', () => {
      const results = evaluateGPUAnalysisRequirements(
        [spec('varying-values', {role: 'value'})],
        profileWith(sampled({minimum: 5, maximum: 5})),
        bound
      );
      expect(summarizeGPUAnalysisRequirementResults(results).status).toBe('needs-confirmation');
    });

    it('lets a refutation win over an unknown across bound columns', () => {
      const profile = makeProfile({
        columns: [
          makeColumn('a', {sampleSize: 500, rowCount: 1900, minimum: 3}),
          makeColumn('b', {sampleSize: 500, rowCount: 1900, minimum: -1})
        ]
      });
      const result = evaluateGPUAnalysisRequirement(
        spec('positive-values', {role: 'predictors'}),
        profile,
        {predictors: ['a', 'b']}
      );
      expect(result).toMatchObject({state: 'failed', evidence: 'sampled-profile'});
    });

    it('no-duplicate-locations: sampled zero duplicates is unknown, sampled duplicates fail', () => {
      const zero = evaluateGPUAnalysisRequirement(
        spec('no-duplicate-locations'),
        makeProfile({spacing: makeSpacing({sampled: true, sampleSize: 500, pointCount: 50000})}),
        {}
      );
      expect(zero).toMatchObject({state: 'unknown', evidence: 'sampled-profile'});
      expect(zero.message).toContain('in a sample of 500 of 50,000 points');
    });
  });

  it('evaluates every column of a multiple role and summarises', () => {
    const requirement = spec('positive-values', {role: 'predictors'});
    const profile = makeProfile({
      columns: [
        makeColumn('a', {minimum: 1}),
        makeColumn('b', {minimum: -1}),
        makeColumn('c', {minimum: 0}),
        makeColumn('d')
      ]
    });
    const failed = evaluateGPUAnalysisRequirement(requirement, profile, {
      predictors: ['a', 'b', 'c']
    });
    expect(failed).toMatchObject({state: 'failed'});
    expect(failed.message).toBe(
      'b has a minimum of -1; every value must be above 0; 1 other bound column also fails'
    );
    const passed = evaluateGPUAnalysisRequirement(requirement, profile, {predictors: ['a']});
    expect(passed.state).toBe('passed');
    const allPass = evaluateGPUAnalysisRequirement(
      requirement,
      makeProfile({columns: [makeColumn('a', {minimum: 1}), makeColumn('b', {minimum: 2})]}),
      {predictors: ['a', 'b']}
    );
    expect(allPass.message).toContain('1 other bound column passes too');
    const unknown = evaluateGPUAnalysisRequirement(requirement, profile, {predictors: ['a', 'd']});
    expect(unknown).toMatchObject({state: 'unknown'});
  });
});

describe('evaluateGPUAnalysisRequirement: neighbors and duplicates', () => {
  const requirement = spec('every-row-has-a-neighbor');

  it('points always have a nearest neighbor from two rows on', () => {
    const two = evaluateGPUAnalysisRequirement(
      requirement,
      makeProfile({kind: 'points', rowCount: 2}),
      {}
    );
    expect(two).toMatchObject({state: 'passed', evidence: 'profile'});
    expect(two.message).toContain('nearest neighbor');
    expect(
      evaluateGPUAnalysisRequirement(requirement, makeProfile({kind: 'points', rowCount: 1}), {})
        .state
    ).toBe('failed');
    expect(
      evaluateGPUAnalysisRequirement(requirement, makeProfile({kind: 'points', rowCount: 0}), {})
        .state
    ).toBe('failed');
  });

  it('areas use the island count and are unknown without contiguity', () => {
    for (const kind of ['polygons', 'cells', 'regions'] as const) {
      expect(evaluateGPUAnalysisRequirement(requirement, makeProfile({kind}), {})).toMatchObject({
        state: 'unknown',
        evidence: 'not-checkable'
      });
    }
    const contiguity = {minimum: 0, median: 5, maximum: 9};
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        makeProfile({kind: 'polygons', contiguity: {...contiguity, islandCount: 0}}),
        {}
      ).state
    ).toBe('passed');
    const failed = evaluateGPUAnalysisRequirement(
      requirement,
      makeProfile({kind: 'polygons', contiguity: {...contiguity, islandCount: 1}}),
      {}
    );
    expect(failed).toMatchObject({
      state: 'failed',
      message: '1 area has no neighbor under contiguity; none may be without one'
    });
    expect(
      evaluateGPUAnalysisRequirement(
        requirement,
        makeProfile({kind: 'cells', contiguity: {...contiguity, islandCount: 12}}),
        {}
      ).message
    ).toContain('12 areas have no neighbor');
  });

  it('other kinds are unknown', () => {
    expect(
      evaluateGPUAnalysisRequirement(requirement, makeProfile({kind: 'lines'}), {})
    ).toMatchObject({state: 'unknown', evidence: 'not-checkable'});
  });

  it('no-duplicate-locations uses the duplicate count and notes sampling', () => {
    const duplicates = spec('no-duplicate-locations');
    expect(evaluateGPUAnalysisRequirement(duplicates, makeProfile(), {})).toMatchObject({
      state: 'unknown',
      evidence: 'not-checkable'
    });
    expect(
      evaluateGPUAnalysisRequirement(duplicates, makeProfile({spacing: makeSpacing()}), {})
    ).toMatchObject({state: 'passed', evidence: 'profile'});
    const failed = evaluateGPUAnalysisRequirement(
      duplicates,
      makeProfile({spacing: makeSpacing({duplicateLocationCount: 1})}),
      {}
    );
    expect(failed).toMatchObject({
      state: 'failed',
      message: '1 point shares a location with an earlier point; none may'
    });
    const sampled = evaluateGPUAnalysisRequirement(
      duplicates,
      makeProfile({
        spacing: makeSpacing({
          sampled: true,
          sampleSize: 500,
          pointCount: 50000,
          duplicateLocationCount: 2
        })
      }),
      {}
    );
    expect(sampled).toMatchObject({state: 'failed', evidence: 'sampled-profile'});
    expect(sampled.message).toContain('in a sample of 500 of 50,000 points');
  });
});

describe('evaluateGPUAnalysisRequirement: not checkable', () => {
  it('connected-weights and disjoint-coverage are always unknown', () => {
    const richProfile = makeProfile({
      kind: 'polygons',
      coordinates: {model: 'planar', source: 'declared'},
      contiguity: {minimum: 1, median: 5, maximum: 9, islandCount: 0},
      spacing: makeSpacing()
    });
    for (const id of ['connected-weights', 'disjoint-coverage'] as const) {
      for (const profile of [makeProfile(), richProfile]) {
        expect(evaluateGPUAnalysisRequirement(spec(id), profile, {})).toMatchObject({
          id,
          state: 'unknown',
          evidence: 'not-checkable'
        });
      }
    }
  });
});

describe('every requirement id', () => {
  const profile = makeProfile({
    coordinates: {model: 'planar', source: 'declared'},
    rowSemantics: {value: 'events', source: 'declared'},
    spacing: makeSpacing(),
    columns: [makeColumn('value', {minimum: 1, maximum: 9, distinctCount: 8})]
  });
  const specs: Readonly<Record<GPUAnalysisRequirementId, GPUAnalysisRequirementSpec>> = {
    'minimum-rows': spec('minimum-rows', {limit: 10}),
    'maximum-rows': spec('maximum-rows', {limit: 10000}),
    'maximum-vertices': spec('maximum-vertices', {limit: 10000}),
    'planar-coordinates': spec('planar-coordinates'),
    'geographic-coordinates': spec('geographic-coordinates'),
    'declared-coordinates': spec('declared-coordinates'),
    'row-semantics': spec('row-semantics', {expects: ['events']}),
    'positive-values': spec('positive-values', {role: 'value'}),
    'non-negative-values': spec('non-negative-values', {role: 'value'}),
    'no-missing-values': spec('no-missing-values', {role: 'value'}),
    'varying-values': spec('varying-values', {role: 'value'}),
    'minimum-distinct-values': spec('minimum-distinct-values', {role: 'value', limit: 3}),
    'every-row-has-a-neighbor': spec('every-row-has-a-neighbor'),
    'no-duplicate-locations': spec('no-duplicate-locations'),
    'connected-weights': spec('connected-weights'),
    'disjoint-coverage': spec('disjoint-coverage')
  };

  it('has a spec for every id in the contract', () => {
    expect(Object.keys(specs).sort()).toEqual([...GPU_ANALYSIS_REQUIREMENT_IDS].sort());
  });

  it('returns a one-sentence message, the id and the severity for every id', () => {
    for (const id of GPU_ANALYSIS_REQUIREMENT_IDS) {
      const result = evaluateGPUAnalysisRequirement(
        {...specs[id], severity: 'warn'},
        profile,
        bound
      );
      expect(result.id).toBe(id);
      expect(result.severity).toBe('warn');
      expect(result.message.length, id).toBeGreaterThan(10);
      expect(result.message.trim().endsWith('.'), id).toBe(false);
    }
  });

  it('reaches the passed state for every id that a profile can pass', () => {
    const passable = GPU_ANALYSIS_REQUIREMENT_IDS.filter(
      id =>
        id !== 'connected-weights' && id !== 'disjoint-coverage' && id !== 'geographic-coordinates'
    );
    for (const id of passable) {
      expect(evaluateGPUAnalysisRequirement(specs[id], profile, bound).state, id).toBe('passed');
    }
    expect(
      evaluateGPUAnalysisRequirement(
        specs['geographic-coordinates'],
        makeProfile({coordinates: {model: 'geographic', source: 'declared'}}),
        bound
      ).state
    ).toBe('passed');
  });

  it('reaches unknown for every id on an empty profile', () => {
    const empty: GPUAnalysisDataProfile = {kind: 'polygons', rowCount: 0, columns: []};
    for (const id of GPU_ANALYSIS_REQUIREMENT_IDS) {
      if (id === 'minimum-rows' || id === 'maximum-rows') {
        continue;
      }
      expect(evaluateGPUAnalysisRequirement(specs[id], empty, {}).state, id).toBe('unknown');
    }
  });

  it('passes the remedy only with a failure', () => {
    const failed = evaluateGPUAnalysisRequirement(
      {...specs['minimum-rows'], limit: 5000, remedy: 'Add rows'},
      profile,
      {}
    );
    expect(failed.remedy).toBe('Add rows');
    const unknown = evaluateGPUAnalysisRequirement(
      {...specs['connected-weights'], remedy: 'Check the weights'},
      profile,
      {}
    );
    expect(unknown.remedy).toBe('Check the weights');
    const passed = evaluateGPUAnalysisRequirement(
      {...specs['minimum-rows'], remedy: 'Add rows'},
      profile,
      {}
    );
    expect(passed.remedy).toBeUndefined();
  });
});

describe('evaluateGPUAnalysisRequirements and summarizeGPUAnalysisRequirementResults', () => {
  const profile = makeProfile({rowCount: 10});
  const blockFailed = spec('minimum-rows', {limit: 30});
  const blockUnknown = spec('connected-weights');
  const blockPassed = spec('maximum-rows', {limit: 100});
  const warnFailed = spec('minimum-rows', {limit: 50, severity: 'warn'});
  const warnUnknown = spec('disjoint-coverage', {severity: 'warn'});
  const warnPassed = spec('maximum-rows', {limit: 100, severity: 'warn'});

  it('evaluates in order', () => {
    const results = evaluateGPUAnalysisRequirements([blockPassed, blockFailed], profile, {});
    expect(results.map(result => result.state)).toEqual(['passed', 'failed']);
    expect(evaluateGPUAnalysisRequirements([], profile, {})).toEqual([]);
  });

  const summarize = (...requirements: GPUAnalysisRequirementSpec[]) =>
    summarizeGPUAnalysisRequirementResults(
      evaluateGPUAnalysisRequirements(requirements, profile, {})
    );

  it('is ready with no results and with only passes', () => {
    expect(summarizeGPUAnalysisRequirementResults([])).toEqual({
      status: 'ready',
      blocking: [],
      unknown: [],
      warnings: []
    });
    expect(summarize(blockPassed, warnPassed).status).toBe('ready');
  });

  it('is blocked when a blocking requirement failed, even with unknowns', () => {
    const summary = summarize(blockUnknown, blockFailed, blockPassed);
    expect(summary.status).toBe('blocked');
    expect(summary.blocking.map(result => result.id)).toEqual(['minimum-rows']);
    expect(summary.unknown.map(result => result.id)).toEqual(['connected-weights']);
  });

  it('needs confirmation when a blocking requirement is unknown and none failed', () => {
    const summary = summarize(blockPassed, blockUnknown);
    expect(summary.status).toBe('needs-confirmation');
    expect(summary.blocking).toEqual([]);
  });

  it('never lets warning-level results change the status', () => {
    const summary = summarize(blockPassed, warnFailed, warnUnknown, warnPassed);
    expect(summary.status).toBe('ready');
    expect(summary.warnings.map(result => result.state)).toEqual(['failed', 'unknown']);
    const results: readonly GPUAnalysisRequirementResult[] = [];
    expect(summarizeGPUAnalysisRequirementResults(results).warnings).toEqual([]);
  });

  it('needs confirmation, not blocked, when a block coordinate requirement meets a contradicting guess', () => {
    const guessed = makeProfile({coordinates: {model: 'geographic', source: 'guess'}});
    const summary = summarizeGPUAnalysisRequirementResults(
      evaluateGPUAnalysisRequirements([spec('planar-coordinates')], guessed, {})
    );
    expect(summary.status).toBe('needs-confirmation');
    expect(summary.blocking).toEqual([]);
    expect(summary.unknown.map(result => result.id)).toEqual(['planar-coordinates']);
    const declared = makeProfile({coordinates: {model: 'geographic', source: 'declared'}});
    expect(
      summarizeGPUAnalysisRequirementResults(
        evaluateGPUAnalysisRequirements([spec('planar-coordinates')], declared, {})
      ).status
    ).toBe('blocked');
  });
});

describe('reviewed requirement uncertainty', () => {
  it('keeps low-confidence guesses unknown for agreement and disagreement, but declarations decisive', () => {
    for (const source of ['guess', 'declared'] as const) {
      for (const matches of [true, false]) {
        const rowRequirement = spec('row-semantics', {expects: ['events']});
        const coordinateRequirement = spec('geographic-coordinates');
        const profile = makeProfile({
          coordinates: {model: matches ? 'geographic' : 'planar', source, confidence: 'low'},
          rowSemantics: {value: matches ? 'events' : 'surface-samples', source, confidence: 'low'}
        });
        for (const requirement of [rowRequirement, coordinateRequirement]) {
          const result = evaluateGPUAnalysisRequirement(requirement, profile, {});
          expect(result).toMatchObject({
            state: source === 'guess' ? 'unknown' : matches ? 'passed' : 'failed',
            evidence: source === 'guess' ? 'guess' : 'declaration'
          });
          expect(summarizeGPUAnalysisRequirementResults([result]).status).toBe(
            source === 'guess' ? 'needs-confirmation' : matches ? 'ready' : 'blocked'
          );
        }
      }
    }
  });

  it('retains high-confidence guess decisions', () => {
    for (const matches of [true, false]) {
      const profile = makeProfile({
        rowSemantics: {
          value: matches ? 'events' : 'surface-samples',
          source: 'guess',
          confidence: 'high'
        },
        coordinates: {model: matches ? 'geographic' : 'planar', source: 'guess', confidence: 'high'}
      });
      expect(
        evaluateGPUAnalysisRequirement(spec('row-semantics', {expects: ['events']}), profile, {})
          .state
      ).toBe(matches ? 'passed' : 'failed');
      expect(
        evaluateGPUAnalysisRequirement(spec('geographic-coordinates'), profile, {}).state
      ).toBe(matches ? 'passed' : 'unknown');
    }
  });

  it('keeps the actual-row review estimates unknown near their exact Set thresholds', () => {
    // Values recorded by the review probe: Set sizes and real profiler estimates.
    for (const [exactCount, estimate] of [
      [10001, 10007],
      [15000, 15007],
      [20001, 20018]
    ]) {
      for (const limit of [exactCount, exactCount + 1]) {
        const result = evaluateGPUAnalysisRequirement(
          spec('minimum-distinct-values', {role: 'value', limit}),
          makeProfile({
            columns: [makeColumn('value', {distinctCount: estimate, distinctCountIsEstimate: true})]
          }),
          bound
        );
        expect(result.state).toBe('unknown');
        expect(result.message).toContain('no guaranteed error bound');
      }
    }
  });

  it('keeps estimates unknown without a contractual bound, even far from a threshold', () => {
    for (const count of [0, 50, 99, 100, 101, 102, 10000]) {
      const result = evaluateGPUAnalysisRequirement(
        spec('minimum-distinct-values', {role: 'value', limit: 100}),
        makeProfile({
          columns: [makeColumn('value', {distinctCount: count, distinctCountIsEstimate: true})]
        }),
        bound
      );
      expect(result.state, String(count)).toBe('unknown');
    }
    // Estimated variety must not bypass the same rule through varying-values.
    expect(
      evaluateGPUAnalysisRequirement(
        spec('varying-values', {role: 'value'}),
        makeProfile({
          columns: [makeColumn('value', {distinctCount: 2, distinctCountIsEstimate: true})]
        }),
        bound
      ).state
    ).toBe('unknown');
    const sampled = makeProfile({
      columns: [
        makeColumn('value', {
          rowCount: 1000,
          sampleSize: 100,
          distinctCount: 50,
          distinctCountIsEstimate: true
        })
      ]
    });
    expect(
      evaluateGPUAnalysisRequirement(
        spec('minimum-distinct-values', {role: 'value', limit: 100}),
        sampled,
        bound
      )
    ).toMatchObject({state: 'unknown', evidence: 'sampled-profile'});
  });
});
