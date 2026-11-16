// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  buildKeplerCuratedAnalysisRequest,
  createKeplerCuratedAnalysisMapping,
  getKeplerCuratedAnalysis,
  getWorkflowOperationRegistry,
  profileKeplerFeatureCollection,
  type CatalogFeatureCollection
} from '@luma-contributors/kepler';
import {
  createWorkflowAnalysisSelection,
  searchWorkflowAnalyses
} from '../../src/workflows/workflow-analysis-search';

import {createAnalysisDatasetContext} from '../../../../examples/deck/kepler-compute/src/analysis-dataset-context';
import {getOriginalFieldType} from '../../../../examples/deck/kepler-compute/src/original-column-values';
import {createSandboxSample} from '../../../../examples/deck/kepler-compute/src/sandbox-samples';

function makeProfile(datasetId: string, geometry: 'Point' | 'Polygon' = 'Point') {
  const collection: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: Array.from({length: 12}, (_, index) => ({
      type: 'Feature',
      geometry:
        geometry === 'Point'
          ? {type: 'Point', coordinates: [100 + (index % 4) * 2, 200 + Math.floor(index / 4) * 3]}
          : {
              type: 'Polygon',
              coordinates: [
                [
                  [index, 0],
                  [index + 1, 0],
                  [index + 1, 1],
                  [index, 1],
                  [index, 0]
                ]
              ]
            },
      properties: {renamed_amount: index + 1, population: index + 10}
    }))
  };
  return {
    ...profileKeplerFeatureCollection(collection, {
      datasetId,
      label: datasetId,
      coordinateSystem: 'planar',
      declarations: {rowSemantics: 'events'}
    }),
    geometryColumns: ['geometry']
  };
}

const registry = getWorkflowOperationRegistry();

describe('dataset-scoped AX command search', () => {
  test('hot ranks the template on the published observations-through-time sample', () => {
    const sample = createSandboxSample('points-time');
    const context = createAnalysisDatasetContext();
    context.synchronize([
      {
        id: 'published-observations',
        label: 'Observations through time',
        rowCount: sample.table.numRows,
        coordinates: sample.coordinates,
        timeColumns: ['timestamp'],
        fieldTypes: Object.fromEntries(
          sample.table.schema.fields.map(field => [field.name, getOriginalFieldType(field.type)])
        ),
        bounds: [-74.04, 40.7, -73.95, 40.79],
        provider: {
          fields: sample.table.schema.fields.map(field => ({name: field.name, sortable: true})),
          maximumPageSize: 512,
          getRevision: () => '1',
          readPage: async () => {
            throw new Error('Unexpected read');
          }
        },
        isCurrent: () => true
      }
    ]);
    const profile = context.get('published-observations')!.profile;
    const matches = searchWorkflowAnalyses(registry, 'hot', {
      profiles: [profile],
      activeDatasetId: profile.datasetId
    });
    expect(
      matches[0],
      JSON.stringify(matches.map(({id, status, reason}) => ({id, status, reason})))
    ).toMatchObject({kind: 'template', id: 'template:hot-spot-map'});
  });

  test('hot ranks the point hot spot template first and opens mapped slots without running', () => {
    const profile = makeProfile('uploaded-events');
    const matches = searchWorkflowAnalyses(registry, 'hot', {
      profiles: [profile],
      activeDatasetId: profile.datasetId
    });
    expect(matches[0]).toMatchObject({kind: 'template', id: 'template:hot-spot-map'});
    const first = matches[0];
    if (first.kind !== 'template') throw new Error('Expected a template');
    expect(first.available).toBe(true);
    expect(first.selection.mapping.slots[0]).toMatchObject({
      slot: 'points',
      datasetId: 'uploaded-events'
    });
    expect(first.selection.mapping.progress).toEqual({mapped: 1, required: 1});
    expect(first.selection.template.curatedAnalysisId).toBe('hot-spot-map');
    expect(first.status).not.toBe('blocked');
    expect(
      buildKeplerCuratedAnalysisRequest(first.selection.analysis, first.selection.mapping)
        .inputDatasetIds
    ).toBeDefined();
    expect(
      matches.some(
        match =>
          match.kind === 'operation' && match.match.operation.id === 'recipe-hot-spot-lattice'
      )
    ).toBe(true);
  });

  test('indexes shared question labels and examples as well as operation names', () => {
    const profiles = [makeProfile('events')];
    const questions = searchWorkflowAnalyses(registry, 'Where do values cluster', {profiles});
    expect(questions.some(match => match.id === 'template:hot-spot-map')).toBe(true);
    expect(questions.some(match => match.kind === 'operation')).toBe(true);
    const examples = searchWorkflowAnalyses(registry, 'disease rate', {profiles});
    expect(examples.some(match => match.id === 'template:hot-spot-map')).toBe(true);
    expect(
      searchWorkflowAnalyses(registry, 'kernel-density-grid', {profiles}).some(
        match => match.id === 'operation:kernel-density-grid'
      )
    ).toBe(true);
    expect(searchWorkflowAnalyses(registry, 'unfindable-question-xyz', {profiles})).toEqual([]);
  });

  test('anchors the active dataset despite a competing point dataset and preserves paired slots', () => {
    const points = makeProfile('a-other-points');
    const active = makeProfile('z-active-points');
    const areas = makeProfile('areas', 'Polygon');
    const profiles = [points, active, areas];
    const selection = createWorkflowAnalysisSelection(
      getKeplerCuratedAnalysis('hot-spot-map')!,
      profiles,
      active.datasetId
    );
    expect(selection.mapping.slots[0].datasetId).toBe(active.datasetId);
    const paired = createWorkflowAnalysisSelection(
      getKeplerCuratedAnalysis('count-by-area')!,
      profiles,
      active.datasetId
    );
    expect(paired.mapping.slots.find(slot => slot.kind === 'points')?.datasetId).toBe(
      active.datasetId
    );
    expect(paired.mapping.slots.find(slot => slot.kind === 'polygons')?.datasetId).toBe(
      areas.datasetId
    );
  });

  test('preserves before/after direction when the active dataset already fills a slot', () => {
    const profiles = [makeProfile('before'), makeProfile('after')];
    const analysis = getKeplerCuratedAnalysis('period-comparison')!;
    const original = createKeplerCuratedAnalysisMapping(analysis, profiles);
    expect(Object.fromEntries(original.slots.map(slot => [slot.slot, slot.datasetId]))).toEqual({
      before: 'before',
      after: 'after'
    });
    for (const activeDatasetId of ['before', 'after']) {
      const selection = createWorkflowAnalysisSelection(analysis, profiles, activeDatasetId);
      expect(selection.mapping).toEqual(original);
      const searched = searchWorkflowAnalyses(registry, 'Change between two point sets', {
        profiles,
        activeDatasetId
      }).find(match => match.kind === 'template' && match.selection.analysis.id === analysis.id);
      expect(searched?.kind === 'template' && searched.selection.mapping).toEqual(original);
    }
  });

  test('keeps wrong-kind, excessive capacity and unavailable templates visible with reasons', () => {
    const polygon = makeProfile('areas', 'Polygon');
    const wrongKind = searchWorkflowAnalyses(registry, 'hot', {
      profiles: [polygon],
      activeDatasetId: polygon.datasetId
    });
    expect(wrongKind.find(match => match.id === 'template:hot-spot-map')).toMatchObject({
      status: 'blocked',
      reason: expect.stringContaining('active dataset')
    });
    const tooLarge = {...makeProfile('large'), rowCount: 100_000_000};
    const excessive = searchWorkflowAnalyses(registry, 'hot', {
      profiles: [tooLarge],
      activeDatasetId: tooLarge.datasetId
    });
    expect(excessive.find(match => match.id === 'template:hot-spot-map')).toMatchObject({
      status: 'blocked',
      reason: expect.stringMatching(/limit|maximum|at most|rows|features/i)
    });
    const unsupported = {
      ...getKeplerCuratedAnalysis('hot-spot-map')!,
      execution: {
        status: 'needs-binding' as const,
        implementation: 'fixture',
        reason: 'This fixture has no execution binding.'
      },
      exposedParameters: []
    };
    const unavailable = searchWorkflowAnalyses(registry, 'hot', {
      profiles: [makeProfile('events')],
      analyses: [unsupported]
    }).find(match => match.kind === 'template');
    expect(unavailable).toMatchObject({
      available: false,
      status: 'blocked',
      reason: unsupported.execution.reason
    });
  });

  test('is deterministic, handles empty profiles and limits results', () => {
    const profile = makeProfile('events');
    const options = {profiles: [profile], activeDatasetId: profile.datasetId, limit: 5};
    expect(searchWorkflowAnalyses(registry, '', options).map(match => match.id)).toEqual(
      searchWorkflowAnalyses(registry, '', options).map(match => match.id)
    );
    expect(searchWorkflowAnalyses(registry, '', options)).toHaveLength(5);
    expect(
      searchWorkflowAnalyses(registry, 'hot', {
        profiles: [],
        analyses: [getKeplerCuratedAnalysis('hot-spot-map')!]
      }).find(match => match.kind === 'template')
    ).toMatchObject({status: 'blocked'});
    expect(searchWorkflowAnalyses(registry, 'hot', {...options, limit: 0})).toEqual([]);
  });
});
