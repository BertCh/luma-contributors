// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  createWorkflowOperationRegistry,
  getWorkflowOperationRegistry,
  getWorkflowPortCompatibility
} from '../../src/workflows/workflow-operations';
import {planWorkflow} from '../../src/workflows/workflow-plan';
import {
  applyWorkflowRecommendation,
  getWorkflowRecommendationSubject,
  recommendWorkflowOperations,
  WORKFLOW_RECOMMENDATION_CANDIDATES,
  type WorkflowRecommendationSubject
} from '../../src/workflows/workflow-recommendations';
import type {WorkflowDatasetProfile, WorkflowPortKind} from '../../src/workflows/workflow-types';
import {makeFixtureDocument, makeFixtureRegistry} from './workflow-fixtures';

const registry = getWorkflowOperationRegistry();
const columns: WorkflowRecommendationSubject['columns'] = [
  {name: 'feature_id', type: 'int32'},
  {name: 'grid', type: 'float64'},
  {name: 'rowIndex', type: 'int32'},
  {name: 'category', type: 'string'},
  {name: 'value', type: 'float64'},
  {name: 'population', type: 'int32'}
];

function makeSubject(kind: WorkflowPortKind): WorkflowRecommendationSubject {
  return {kind, rowCount: 100, columns};
}

function recommendAll(subject: WorkflowRecommendationSubject) {
  return recommendWorkflowOperations(registry, subject, {limit: 100});
}

function makeProfile(overrides: Partial<WorkflowDatasetProfile>): WorkflowDatasetProfile {
  return {
    id: 'observations',
    label: 'Observations',
    rowCount: 42,
    resident: false,
    fields: [],
    geometryFields: [],
    ...overrides
  };
}

describe('workflow recommendation subjects', () => {
  test('geometry takes precedence and coordinate fields are excluded', () => {
    const subject = getWorkflowRecommendationSubject(
      makeProfile({
        geometryFields: ['shape', 'boundary'],
        pointColumns: {longitude: 'lon', latitude: 'lat'},
        fields: [
          {name: 'shape', type: 'geojson'},
          {name: 'boundary', type: 'geojson'},
          {name: 'lon', type: 'real'},
          {name: 'lat', type: 'real'},
          {name: 'value', type: 'real'},
          {name: 'count', type: 'integer'},
          {name: 'enabled', type: 'boolean'},
          {name: 'time', type: 'timestamp'}
        ]
      })
    );
    expect(subject).toEqual({
      kind: 'features',
      rowCount: 42,
      columns: [
        {name: 'value', type: 'float64'},
        {name: 'count', type: 'int32'},
        {name: 'enabled', type: 'boolean'},
        {name: 'time', type: 'string'}
      ]
    });
  });

  test('point columns identify points without retaining coordinate attributes', () => {
    expect(
      getWorkflowRecommendationSubject(
        makeProfile({
          pointColumns: {longitude: 'longitude', latitude: 'latitude'},
          fields: [
            {name: 'longitude', type: 'real'},
            {name: 'latitude', type: 'real'},
            {name: 'name', type: 'string'}
          ]
        })
      )
    ).toEqual({kind: 'point', rowCount: 42, columns: [{name: 'name', type: 'string'}]});
  });

  test('a profile without geometry stays a table and keeps potential coordinates', () => {
    expect(
      getWorkflowRecommendationSubject(
        makeProfile({
          fields: [
            {name: 'longitude', type: 'real'},
            {name: 'latitude', type: 'real'},
            {name: 'unknown', type: 'object'}
          ]
        })
      )
    ).toEqual({
      kind: 'table',
      rowCount: 42,
      columns: [
        {name: 'longitude', type: 'float64'},
        {name: 'latitude', type: 'float64'},
        {name: 'unknown', type: 'string'}
      ]
    });
  });
});

describe('dataset-aware workflow recommendations', () => {
  test('every candidate uses real operation, port and preset parameter names', () => {
    expect(WORKFLOW_RECOMMENDATION_CANDIDATES.length).toBeGreaterThanOrEqual(25);
    for (const candidate of WORKFLOW_RECOMMENDATION_CANDIDATES) {
      const operation = registry.get(candidate.operationId);
      expect(operation, candidate.operationId).toBeDefined();
      expect(operation!.inputs.map(input => input.name)).toContain(candidate.inputName);
      for (const parameter of Object.keys({
        ...candidate.parameters,
        ...candidate.columnParameters
      })) {
        expect(
          operation!.parameters.map(specification => specification.name),
          candidate.operationId
        ).toContain(parameter);
      }
      expect(candidate.label.length).toBeLessThanOrEqual(40);
    }
  });

  test('point defaults are ordered, short and deterministic', () => {
    const subject = makeSubject('point');
    const recommendations = recommendWorkflowOperations(registry, subject);
    expect(recommendations.map(recommendation => recommendation.operationId)).toEqual([
      'density-clustering',
      'kernel-density-grid',
      'recipe-hot-spot-lattice',
      'hotspot-gi-star',
      'voronoi-diagram',
      'group-convex-hull'
    ]);
    expect(recommendWorkflowOperations(registry, subject)).toEqual(recommendations);
    expect(recommendations.every(recommendation => recommendation.label.length <= 40)).toBe(true);
  });

  test('numeric presets skip identifier suffixes and use a different OLS predictor', () => {
    const recommendations = recommendAll(makeSubject('point'));
    for (const operationId of [
      'hotspot-gi-star',
      'idw-grid',
      'kriging-grid',
      'local-moran-statistics'
    ]) {
      expect(
        recommendations.find(recommendation => recommendation.operationId === operationId)
          ?.parameters
      ).toEqual({valueProperty: 'value'});
    }
    expect(
      recommendations.find(recommendation => recommendation.operationId === 'regression-ols')
        ?.parameters
    ).toEqual({responseProperty: 'value', predictorProperties: 'population'});
    expect(
      recommendations.find(recommendation => recommendation.operationId === 'group-convex-hull')
        ?.parameters
    ).toEqual({groupProperty: 'category'});
  });

  test('no numeric column excludes value-based operations but keeps geometry analysis', () => {
    const recommendations = recommendAll({
      kind: 'point',
      columns: [{name: 'name', type: 'string'}]
    });
    expect(recommendations.map(recommendation => recommendation.operationId)).toEqual([
      'density-clustering',
      'kernel-density-grid',
      'recipe-hot-spot-lattice',
      'voronoi-diagram',
      'group-convex-hull',
      'local-outlier-factor',
      'k-means',
      'dggs-cell-aggregation',
      'geographic-distribution',
      'geometry-measures',
      'spatial-weights'
    ]);
    expect(
      recommendAll({kind: 'point', columns: [{name: 'value', type: 'float64'}]}).some(
        recommendation => recommendation.operationId === 'regression-ols'
      )
    ).toBe(false);
    expect(
      recommendAll({kind: 'point', columns: []}).some(
        recommendation => recommendation.operationId === 'group-convex-hull'
      )
    ).toBe(false);
  });

  test('lines offer simplification, polygon length, networks and density with the correct port', () => {
    const recommendations = recommendWorkflowOperations(registry, makeSubject('line'));
    expect(
      recommendations.map(recommendation => [recommendation.operationId, recommendation.inputName])
    ).toEqual([
      ['line-simplification', 'lines'],
      ['line-length-per-polygon', 'lines'],
      ['network-service-area', 'lines'],
      ['network-centrality', 'lines'],
      ['line-density-grid', 'lines'],
      ['geometry-measures', 'features']
    ]);
    expect(recommendations[1].reason).toContain('also needs a connected polygon');
  });

  test('polygons start with measures, validity, repair and interior labels', () => {
    const recommendations = recommendAll(makeSubject('polygon'));
    expect(recommendations.slice(0, 6).map(recommendation => recommendation.operationId)).toEqual([
      'geometry-measures',
      'geometry-validity',
      'geometry-topology-repair',
      'geometry-label-point',
      'geometry-overlay-intersection',
      'spatial-weights'
    ]);
    expect(
      recommendations.find(
        recommendation => recommendation.operationId === 'local-moran-statistics'
      )?.parameters
    ).toEqual({valueProperty: 'value'});
  });

  test('table coordinate names unlock points and are preset exactly', () => {
    expect(
      recommendAll({
        kind: 'table',
        columns: [
          {name: 'pickupLongitude', type: 'float64'},
          {name: 'pickup_latitude', type: 'int32'}
        ]
      })
    ).toEqual([
      {
        id: 'table-points-from-columns',
        operationId: 'table-points-from-columns',
        label: 'Make points from coordinates',
        reason: 'Create point geometry from numeric longitude and latitude columns.',
        inputName: 'rows',
        parameters: {longitudeColumn: 'pickupLongitude', latitudeColumn: 'pickup_latitude'}
      },
      {
        id: 'table-summarize',
        operationId: 'table-summarize',
        label: 'Summarise rows',
        reason: 'Count the rows in the table.',
        inputName: 'rows',
        parameters: {}
      },
      {
        id: 'table-filter-rows',
        operationId: 'table-filter-rows',
        label: 'Filter rows',
        reason:
          'Keep rows matching an expression; edit the initial expression to choose your subset.',
        inputName: 'rows',
        parameters: {expression: 'true'}
      }
    ]);
    for (const coordinateColumns of [
      [
        {name: 'longitude', type: 'string'},
        {name: 'latitude', type: 'float64'}
      ],
      [{name: 'longitude', type: 'float64'}],
      [
        {name: 'weight', type: 'float64'},
        {name: 'height', type: 'float64'}
      ]
    ] as const) {
      expect(
        recommendAll({kind: 'table', columns: coordinateColumns}).map(
          recommendation => recommendation.operationId
        )
      ).toEqual(['table-summarize', 'table-filter-rows']);
    }
  });

  test('unknown geometry ranks general operations first and labels specific requirements', () => {
    const recommendations = recommendAll(makeSubject('features'));
    expect(recommendations.slice(0, 2).map(recommendation => recommendation.operationId)).toEqual([
      'geometry-measures',
      'geographic-distribution'
    ]);
    expect(
      recommendations.find(recommendation => recommendation.operationId === 'density-clustering')
        ?.label
    ).toBe('Find clusters (points)');
    expect(
      recommendations.find(recommendation => recommendation.operationId === 'line-simplification')
        ?.label
    ).toBe('Simplify lines (lines)');
    expect(recommendations.every(recommendation => recommendation.label.length <= 40)).toBe(true);
    expect(new Set(recommendations.map(recommendation => recommendation.id)).size).toBe(
      recommendations.length
    );
  });

  test('missing operations and incompatible ports are refused', () => {
    expect(recommendWorkflowOperations(makeFixtureRegistry(), makeSubject('point'))).toEqual([]);
    const operation = registry.get('density-clustering')!;
    const incompatibleRegistry = createWorkflowOperationRegistry([
      {
        ...operation,
        inputs: [{...operation.inputs[0], accepts: ['line']}]
      }
    ]);
    expect(recommendWorkflowOperations(incompatibleRegistry, makeSubject('point'))).toEqual([]);
  });

  test('row limits include the boundary and exclude larger inputs', () => {
    const operation = registry.get('density-clustering')!;
    const maximumRows = operation.inputs[0].maximumRows;
    const limitedRegistry = createWorkflowOperationRegistry([operation]);
    expect(
      recommendWorkflowOperations(limitedRegistry, {
        kind: 'point',
        columns: [],
        rowCount: maximumRows
      }).map(recommendation => recommendation.operationId)
    ).toEqual(['density-clustering']);
    expect(
      recommendWorkflowOperations(limitedRegistry, {
        kind: 'point',
        columns: [],
        rowCount: maximumRows + 1
      })
    ).toEqual([]);
    for (const recommendation of recommendAll({...makeSubject('point'), rowCount: 20001})) {
      const input = registry
        .get(recommendation.operationId)!
        .inputs.find(port => port.name === recommendation.inputName)!;
      expect(input.maximumRows).toBeGreaterThanOrEqual(20001);
      expect(getWorkflowPortCompatibility('point', input)).not.toBe('no');
    }
  });

  test('an oversized dataset gets one bounded sampling suggestion', () => {
    expect(recommendAll({...makeSubject('point'), rowCount: 100001})).toEqual([
      {
        id: 'table-sort-rows',
        operationId: 'table-sort-rows',
        label: 'Take a sample first',
        reason:
          'Keep a bounded subset sorted by a column before applying smaller analysis operations.',
        inputName: 'rows',
        parameters: {limit: 1000, column: 'feature_id'}
      }
    ]);
  });

  test('limit truncates deterministically and never turns a zero limit into a sample', () => {
    expect(recommendWorkflowOperations(registry, makeSubject('point'), {limit: 1.9})).toEqual(
      recommendAll(makeSubject('point')).slice(0, 1)
    );
    for (const limit of [0, -1, Number.NaN]) {
      expect(
        recommendWorkflowOperations(registry, {...makeSubject('point'), rowCount: 100001}, {limit})
      ).toEqual([]);
    }
  });

  test('applying every suggestion connects the advertised port and plans valid parameters', () => {
    const fixtureSource = makeFixtureRegistry().get('fixture-source')!;
    const combinedRegistry = createWorkflowOperationRegistry([...registry.list(), fixtureSource]);
    const original = makeFixtureDocument();
    const document = {...original, steps: original.steps.slice(0, 1)};
    const subjects = ['point', 'line', 'polygon', 'features', 'table'].map(kind =>
      makeSubject(kind as WorkflowPortKind)
    );
    subjects.push({
      kind: 'table',
      columns: [
        {name: 'longitude', type: 'float64'},
        {name: 'latitude', type: 'float64'}
      ]
    });
    subjects.push({...makeSubject('point'), rowCount: 100001});
    for (const subject of subjects) {
      for (const recommendation of recommendAll(subject)) {
        const result = applyWorkflowRecommendation(document, combinedRegistry, recommendation, {
          stepId: 'source',
          output: 'data'
        });
        const step = result.document.steps.find(candidate => candidate.id === result.stepId)!;
        expect(step.inputs).toEqual({
          [recommendation.inputName]: {stepId: 'source', output: 'data'}
        });
        expect(step.parameters).toEqual(
          Object.fromEntries(
            Object.entries(recommendation.parameters).map(([name, value]) => [
              name,
              {from: 'static', value}
            ])
          )
        );
        const plan = planWorkflow(result.document, {registry: combinedRegistry});
        expect(plan.steps[result.stepId].parameters, recommendation.operationId).toBeDefined();
        expect(
          plan.steps[result.stepId].issues
            .filter(issue => issue.severity === 'error')
            .map(issue => issue.code),
          recommendation.operationId
        ).toEqual(
          registry.get(recommendation.operationId)!.inputs.length > 1 ? ['input-required'] : []
        );
      }
    }
    expect(document.steps).toHaveLength(1);
  });

  test('applying refuses an unknown upstream output with its exact issue code', () => {
    const fixtureSource = makeFixtureRegistry().get('fixture-source')!;
    const combinedRegistry = createWorkflowOperationRegistry([...registry.list(), fixtureSource]);
    const original = makeFixtureDocument();
    const document = {...original, steps: original.steps.slice(0, 1)};
    const recommendation = recommendAll(makeSubject('point'))[0];
    expect(() =>
      applyWorkflowRecommendation(document, combinedRegistry, recommendation, {
        stepId: 'source',
        output: 'missing'
      })
    ).toThrowError(expect.objectContaining({name: 'WorkflowError', code: 'unknown-output-port'}));
  });
});
