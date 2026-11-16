// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {evaluateGPUAnalysisParameterCondition} from '@luma-contributors/analysis/gpu-spatial-analysis';

import {describe, expect, it} from 'vitest';
import type {
  GPUAnalysisColumnProfile,
  GPUAnalysisColumnType,
  GPUAnalysisMeasurementLevel,
  GPUAnalysisRowSemantics,
  GPUSpatialAnalysisDataKind
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  getCatalogBinding,
  normalizeCatalogParameters,
  runCatalogOperationAsync
} from '../../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {getKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {profileKeplerNativeDataset} from '../../src/profile/native-dataset-profile';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {
  buildKeplerCuratedAnalysisRequest,
  createKeplerCuratedAnalysisMapping,
  listKeplerCuratedAnalysesForDatasets,
  parseKeplerCuratedAnalysisMapping,
  replayKeplerCuratedAnalysisMapping,
  serializeKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  validateKeplerCuratedAnalysis,
  type KeplerCuratedAnalysisMapping
} from '../../src/curated/curated-analysis-mapping';
import {
  getKeplerCuratedAnalysis,
  KEPLER_CURATED_ANALYSES
} from '../../src/curated/curated-analyses';
import {
  createKeplerCuratedAnalysisSample,
  KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS
} from '../../src/curated/curated-analysis-samples';
import type {KeplerCuratedAnalysis} from '../../src/curated/curated-analysis-types';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';

// ---------------------------------------------------------------------------------------------
// Hand-built profiles
// ---------------------------------------------------------------------------------------------

function makeColumn(
  name: string,
  type: GPUAnalysisColumnType,
  measurement?: GPUAnalysisMeasurementLevel
): GPUAnalysisColumnProfile {
  return {
    name,
    type,
    rowCount: 100,
    nullCount: 0,
    ...(measurement === undefined ? {} : {measurement, measurementSource: 'declared' as const})
  };
}

function makeProfile(
  datasetId: string,
  kind: GPUSpatialAnalysisDataKind,
  columns: readonly GPUAnalysisColumnProfile[],
  rowSemantics?: GPUAnalysisRowSemantics,
  label: string = datasetId
): KeplerDatasetProfile {
  return {
    datasetId,
    label,
    kind,
    rowCount: 100,
    columns,
    ...(rowSemantics === undefined
      ? {}
      : {rowSemantics: {value: rowSemantics, source: 'declared' as const}}),
    geometryColumns: [],
    ...(kind === 'points' || kind === 'trajectories' ? {catalogGeometry: 'point' as const} : {}),
    ...(kind === 'polygons' ? {catalogGeometry: 'polygon' as const} : {}),
    ...(kind === 'lines' || kind === 'network' ? {catalogGeometry: 'line' as const} : {})
  };
}

function getAnalysis(id: string): KeplerCuratedAnalysis {
  const analysis = getKeplerCuratedAnalysis(id);
  if (analysis === undefined) {
    throw new Error(`test setup: no curated analysis "${id}"`);
  }
  return analysis;
}

it('rejects wrong-type field choices and invalidates a saved field when its type changes', () => {
  const analysis = getAnalysis('idw-surface');
  const profile = makeProfile(
    'gauges',
    'points',
    [
      makeColumn('rainfall', 'number', 'amount'),
      makeColumn('station_label', 'category', 'nominal')
    ],
    'surface-samples'
  );
  const mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
  expect(mapping.slots[0].roles[0].candidates.map(candidate => candidate.name)).not.toContain(
    'station_label'
  );
  expect(() =>
    updateKeplerCuratedAnalysisMapping(
      mapping,
      {
        slot: 'points',
        role: 'value',
        columns: ['station_label']
      },
      analysis,
      [profile]
    )
  ).toThrow();
  const chosen = updateKeplerCuratedAnalysisMapping(
    mapping,
    {
      slot: 'points',
      role: 'value',
      columns: ['rainfall']
    },
    analysis,
    [profile]
  );
  const changed = {...profile, columns: [makeColumn('rainfall', 'category', 'nominal')]};
  const replayed = replayKeplerCuratedAnalysisMapping(analysis, chosen.choices, [changed]);
  expect(replayed.ready).toBe(false);
  expect(replayed.problems.some(problem => problem.kind === 'saved-choice-lost')).toBe(true);
  expect(() => buildKeplerCuratedAnalysisRequest(analysis, replayed)).toThrow();
});

/** One dataset per slot whose columns carry the sample's column names, as the sample's own data would. */
function makeSampleShapedProfiles(analysis: KeplerCuratedAnalysis): KeplerDatasetProfile[] {
  const sampleColumns =
    analysis.sampleId === undefined
      ? {}
      : (KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS as Record<string, Record<string, string>>)[
          analysis.sampleId
        ];
  return analysis.slots.map(slot =>
    makeProfile(
      `${slot.name}-dataset`,
      slot.kind === 'network' ? 'lines' : slot.kind,
      [
        makeColumn('name', 'text', 'nominal'),
        ...slot.fields.map(role =>
          makeColumn(
            sampleColumns[`${slot.name}.${role.role}`] ?? role.role,
            role.accepts[0],
            role.measurement?.[0]
          )
        )
      ],
      slot.rowSemantics?.[0],
      slot.name
    )
  );
}

const AVAILABLE_ANALYSES = KEPLER_CURATED_ANALYSES.filter(
  analysis => analysis.execution.status === 'available'
);

// Ids the task named as available when this was written; the list may only grow.
const KNOWN_AVAILABLE_IDS = [
  'density-surface',
  'hot-spot-map',
  'idw-surface',
  'count-by-area',
  'distribution-by-zone',
  'straight-line-catchments',
  'cluster-and-outlier-map',
  'change-of-support',
  'period-comparison',
  'drive-time-catchments'
];

/** A synthetic analysis with every feature the real ones lack: two roles from one pool, a multiple role, a conditional role. */
const SYNTHETIC_ANALYSIS: KeplerCuratedAnalysis = {
  id: 'synthetic-mapping-probe',
  question: 'how-related',
  title: 'Probe?',
  name: 'Probe',
  method: 'Test only.',
  wave: 1,
  slots: [
    {
      name: 'areas',
      label: 'Areas',
      kind: 'polygons',
      fields: [
        {
          role: 'cases',
          label: 'Cases',
          accepts: ['number', 'integer'],
          measurement: ['count'],
          required: true
        },
        {
          role: 'population',
          label: 'Population',
          accepts: ['number', 'integer'],
          measurement: ['count'],
          required: true
        },
        {
          role: 'predictors',
          label: 'Predictors',
          accepts: ['number', 'integer'],
          measurement: ['rate'],
          required: false,
          multiple: {minimum: 1, maximum: 2}
        },
        {
          role: 'weight',
          label: 'Weight',
          accepts: ['number'],
          required: {whenParameter: 'mode', equals: ['weighted']}
        }
      ]
    }
  ],
  execution: {
    status: 'available',
    bindingId: 'synthetic-no-descriptor',
    inputs: {areas: 'features'},
    fieldParameters: {'areas.cases': 'valueProperty'},
    parameters: {mode: 'plain'}
  },
  exposedParameters: []
};

const PROBE_AREAS = makeProfile('areas', 'polygons', [
  makeColumn('n_cases', 'integer', 'count'),
  makeColumn('pop_total', 'integer', 'count'),
  makeColumn('other_count', 'integer', 'count'),
  makeColumn('share_a', 'number', 'rate'),
  makeColumn('share_b', 'number', 'rate'),
  makeColumn('label', 'text', 'nominal')
]);

// ---------------------------------------------------------------------------------------------
// Every available analysis
// ---------------------------------------------------------------------------------------------

describe('curated analyses that are available', () => {
  it('lists at least the analyses the stream was told about', () => {
    const ids = AVAILABLE_ANALYSES.map(analysis => analysis.id);
    expect(ids).toEqual(expect.arrayContaining(KNOWN_AVAILABLE_IDS));
  });

  for (const analysis of AVAILABLE_ANALYSES) {
    it(`${analysis.id}: sample-shaped datasets map with no action and build a catalog request`, () => {
      const datasets = makeSampleShapedProfiles(analysis);
      const mapping = createKeplerCuratedAnalysisMapping(analysis, datasets);

      expect(mapping.problems).toEqual([]);
      expect(mapping.complete).toBe(true);
      expect(mapping.ready).toBe(true);
      expect(mapping.progress.mapped).toBe(mapping.progress.required);
      expect(mapping.progress.required).toBeGreaterThanOrEqual(analysis.slots.length);
      for (const slot of mapping.slots) {
        expect(slot.datasetId).toBe(`${slot.slot}-dataset`);
        expect(slot.state).toBe('settled');
        expect(slot.candidates).toHaveLength(datasets.length);
      }

      const request = buildKeplerCuratedAnalysisRequest(analysis, mapping);
      if (analysis.execution.status !== 'available') throw new Error('unreachable');
      const binding = getCatalogBinding(request.bindingId);
      expect(binding).toBeDefined();
      const normalized = normalizeCatalogParameters(binding!, request.parameters);
      expect(Object.keys(normalized).length).toBeGreaterThan(0);
      expect(Object.keys(request.inputDatasetIds).sort()).toEqual(
        binding!.inputs.map(input => input.name).sort()
      );
      // The column each field parameter received is the column of that role.
      for (const [roleKey, parameterName] of Object.entries(analysis.execution.fieldParameters)) {
        const [slotName, roleName] = roleKey.split('.');
        const role = mapping.slots
          .find(slot => slot.slot === slotName)
          ?.roles.find(entry => entry.role === roleName);
        // Optional roles are never filled automatically, so their parameter stays absent.
        if (role?.columns.length) {
          expect(request.parameters[parameterName]).toBe(role.columns.join(','));
        } else {
          expect(role?.required).toBe(false);
          expect(request.parameters).not.toHaveProperty(parameterName);
        }
      }
      // Presets survive and nothing else is invented.
      for (const [name, value] of Object.entries(analysis.execution.parameters)) {
        expect(request.parameters[name]).toEqual(value);
      }
      const allowed = new Set([
        ...Object.keys(analysis.execution.parameters),
        ...Object.values(analysis.execution.fieldParameters)
      ]);
      expect(Object.keys(request.parameters).every(name => allowed.has(name))).toBe(true);
    });

    it(`${analysis.id}: no datasets gives needs-data naming the first slot's kind`, () => {
      const mapping = createKeplerCuratedAnalysisMapping(analysis, []);
      expect(mapping.complete).toBe(false);
      expect(mapping.progress.mapped).toBe(0);
      const [availability] = listKeplerCuratedAnalysesForDatasets([analysis], []);
      expect(availability.status).toBe('needs-data');
      expect(availability.reason).toContain(analysis.slots[0].label);
      expect(() => buildKeplerCuratedAnalysisRequest(analysis, mapping)).toThrow(
        /buildKeplerCuratedAnalysisRequest.*not complete/
      );
    });
  }
});

// ---------------------------------------------------------------------------------------------
// Automatic choice
// ---------------------------------------------------------------------------------------------

describe('automatic column choice', () => {
  const idw = getAnalysis('idw-surface');

  it('maps differently named columns by type and measurement level', () => {
    const areas = getAnalysis('cluster-and-outlier-map');
    const datasets = [
      makeProfile('tracts', 'polygons', [
        makeColumn('n_cases', 'integer', 'count'),
        makeColumn('pop_total', 'integer', 'count'),
        makeColumn('VALUE', 'number', 'rate'),
        makeColumn('name', 'text', 'nominal')
      ])
    ];
    const mapping = createKeplerCuratedAnalysisMapping(areas, datasets);
    const role = mapping.slots[0].roles[0];
    expect(role.columns).toEqual(['VALUE']);
    expect(role.state).toBe('settled');
    expect(role.source).toBe('automatic');
    // Text is not offered at all; counts are offered with a warning after the rate.
    expect(role.candidates.map(candidate => candidate.name)).toEqual([
      'VALUE',
      'n_cases',
      'pop_total'
    ]);
    expect(role.candidates[0].warning).toBeUndefined();
    expect(mapping.ready).toBe(true);
  });

  it('carries over a column named like the sample column when several fit', () => {
    const datasets = [
      makeProfile(
        'gauges',
        'points',
        [
          makeColumn('depth', 'number', 'amount'),
          makeColumn('rainfall', 'number', 'amount'),
          makeColumn('height', 'number', 'amount')
        ],
        'surface-samples'
      )
    ];
    const role = createKeplerCuratedAnalysisMapping(idw, datasets).slots[0].roles[0];
    expect(role.columns).toEqual(['rainfall']);
    expect(role.state).toBe('settled');
  });

  it('never matches by substring: "accident" is not "id", "rainfalls" is not "rainfall"', () => {
    const datasets = [
      makeProfile(
        'gauges',
        'points',
        [makeColumn('rainfalls_total', 'number', 'amount'), makeColumn('rain', 'number', 'amount')],
        'surface-samples'
      )
    ];
    const role = createKeplerCuratedAnalysisMapping(idw, datasets).slots[0].roles[0];
    // Neither name equals "rainfall" as a word list, so nothing distinguishes them.
    expect(role.state).toBe('choose');
  });

  it('flags two plausible value columns as a choice and needs-mapping, and a user choice settles it', () => {
    const areas = getAnalysis('cluster-and-outlier-map');
    const datasets = [
      makeProfile('tracts', 'polygons', [
        makeColumn('rate_a', 'number', 'rate'),
        makeColumn('rate_b', 'number', 'rate')
      ])
    ];
    const mapping = createKeplerCuratedAnalysisMapping(areas, datasets);
    expect(mapping.slots[0].roles[0].columns).toEqual(['rate_a']);
    expect(mapping.slots[0].roles[0].state).toBe('choose');
    expect(mapping.slots[0].roles[0].reason).toContain('rate_b');
    expect(mapping.complete).toBe(true);
    expect(mapping.ready).toBe(false);
    expect(mapping.problems.map(problem => problem.kind)).toEqual(['needs-choice']);
    expect(listKeplerCuratedAnalysesForDatasets([areas], datasets)[0].status).toBe('needs-mapping');

    const chosen = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'areas', role: 'value', columns: ['rate_b']},
      areas,
      datasets
    );
    expect(chosen.slots[0].roles[0].columns).toEqual(['rate_b']);
    expect(chosen.slots[0].roles[0].source).toBe('user');
    expect(chosen.ready).toBe(true);
    expect(chosen.problems).toEqual([]);
  });

  it('carries the analysis warning when only a count is offered to a role that prefers rates', () => {
    const areas = getAnalysis('cluster-and-outlier-map');
    const datasets = [
      makeProfile('tracts', 'polygons', [makeColumn('n_cases', 'integer', 'count')])
    ];
    const mapping = createKeplerCuratedAnalysisMapping(areas, datasets);
    const role = mapping.slots[0].roles[0];
    expect(role.columns).toEqual(['n_cases']);
    expect(role.candidates[0].fit).toBe('accepted');
    expect(role.candidates[0].warning).toContain('count');
    expect(role.warnings[0]).toContain('is a count');
    expect(role.state).toBe('choose');
    expect(mapping.complete).toBe(true);
    expect(mapping.ready).toBe(false);
    expect(listKeplerCuratedAnalysesForDatasets([areas], datasets)[0].status).toBe('needs-mapping');
  });

  it('never fills an optional role, offers its candidates and says what filling would change', () => {
    const density = getAnalysis('density-surface');
    const one = makeProfile(
      'incidents',
      'points',
      [makeColumn('weight', 'number', 'count')],
      'events'
    );
    const two = makeProfile(
      'incidents',
      'points',
      [makeColumn('count_a', 'number', 'count'), makeColumn('count_b', 'number', 'count')],
      'events'
    );
    for (const dataset of [one, two]) {
      const mapping = createKeplerCuratedAnalysisMapping(density, [dataset]);
      const role = mapping.slots[0].roles[0];
      // Even a single column named exactly like the role is left for the user.
      expect(role.columns).toEqual([]);
      expect(role.state).toBe('empty');
      expect(role.source).toBe('none');
      expect(role.candidates.length).toBe(dataset.columns.length);
      expect(role.reason).toContain('never filled automatically');
      expect(role.reason).toContain('changes what');
      expect(mapping.ready).toBe(true);
      expect(buildKeplerCuratedAnalysisRequest(density, mapping).parameters).not.toHaveProperty(
        'weightProperty'
      );
    }
    const chosen = updateKeplerCuratedAnalysisMapping(
      createKeplerCuratedAnalysisMapping(density, [one]),
      {slot: 'points', role: 'weight', columns: ['weight']},
      density,
      [one]
    );
    expect(buildKeplerCuratedAnalysisRequest(density, chosen).parameters).toMatchObject({
      weightProperty: 'weight'
    });
  });

  it('keeps a column taken by one role out of the automatic choice of another, but selectable with a warning', () => {
    const mapping = createKeplerCuratedAnalysisMapping(SYNTHETIC_ANALYSIS, [PROBE_AREAS]);
    const [cases, population] = mapping.slots[0].roles;
    expect(cases.columns).toEqual(['n_cases']);
    expect(cases.state).toBe('settled');
    expect(population.columns).not.toContain('n_cases');
    expect(population.columns).toHaveLength(1);
    // Two count columns remain and neither is named for population.
    expect(population.state).toBe('choose');
    const reused = population.candidates.find(candidate => candidate.name === 'n_cases');
    expect(reused?.warning).toContain('Also used for Cases');

    const insisted = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'areas', role: 'population', columns: ['n_cases']},
      SYNTHETIC_ANALYSIS,
      [PROBE_AREAS]
    );
    // The user's column is reserved, so Cases moves to a remaining column.
    expect(insisted.slots[0].roles[1].columns).toEqual(['n_cases']);
    expect(insisted.slots[0].roles[0].columns).not.toEqual(['n_cases']);
    const both = updateKeplerCuratedAnalysisMapping(
      insisted,
      {slot: 'areas', role: 'cases', columns: ['n_cases']},
      SYNTHETIC_ANALYSIS,
      [PROBE_AREAS]
    );
    expect(both.slots[0].roles[0].columns).toEqual(['n_cases']);
    expect(both.slots[0].roles[1].columns).toEqual(['n_cases']);
    expect(both.problems.filter(problem => problem.kind === 'column-warning')).toHaveLength(2);
    expect(both.complete).toBe(true);
  });

  it('fills a multiple role with its minimum and lets the user add up to the maximum', () => {
    const mapping = createKeplerCuratedAnalysisMapping(SYNTHETIC_ANALYSIS, [PROBE_AREAS]);
    const predictors = mapping.slots[0].roles[2];
    // Two rates fit and none is named "predictors": optional and unclear, so empty.
    expect(predictors.columns).toEqual([]);
    const two = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'areas', role: 'predictors', columns: ['share_a', 'share_b']},
      SYNTHETIC_ANALYSIS,
      [PROBE_AREAS]
    );
    expect(two.slots[0].roles[2].columns).toEqual(['share_a', 'share_b']);
    expect(() =>
      updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'areas', role: 'predictors', columns: ['share_a', 'share_b', 'other_count']},
        SYNTHETIC_ANALYSIS,
        [PROBE_AREAS]
      )
    ).toThrow(/updateKeplerCuratedAnalysisMapping.*areas\.predictors.*1 to 2/);
  });
});

// ---------------------------------------------------------------------------------------------
// Datasets as candidates
// ---------------------------------------------------------------------------------------------

describe('dataset candidates', () => {
  const idw = getAnalysis('idw-surface');
  const gauges = makeProfile(
    'gauges',
    'points',
    [makeColumn('rainfall', 'number', 'amount')],
    'surface-samples'
  );
  const districts = makeProfile('districts', 'polygons', [makeColumn('rate', 'number', 'rate')]);
  const bare = makeProfile(
    'bare-points',
    'points',
    [makeColumn('name', 'text', 'nominal')],
    'events'
  );

  it('lists every loaded dataset with its fit and a reason, never dropping the wrong ones', () => {
    const mapping = createKeplerCuratedAnalysisMapping(idw, [districts, bare, gauges]);
    const candidates = mapping.slots[0].candidates;
    expect(candidates.map(candidate => [candidate.datasetId, candidate.fit])).toEqual([
      ['gauges', 'fits'],
      ['bare-points', 'missing-columns'],
      ['districts', 'wrong-kind']
    ]);
    expect(candidates[2].reason).toContain('needs points');
    expect(candidates[2].reason).toContain('polygons');
    expect(candidates[1].reason).toContain('Measured value');
    expect(mapping.slots[0].datasetId).toBe('gauges');
  });

  it('warns, without blocking, when row semantics are not the ones the analysis expects', () => {
    const events = makeProfile(
      'events',
      'points',
      [makeColumn('rainfall', 'number', 'amount')],
      'events'
    );
    const mapping = createKeplerCuratedAnalysisMapping(idw, [events]);
    expect(mapping.slots[0].candidates[0].fit).toBe('fits');
    expect(mapping.slots[0].candidates[0].warning).toContain('surface-samples');
    expect(mapping.complete).toBe(true);
  });

  it('prefers the dataset whose row semantics match when two fit', () => {
    const events = makeProfile(
      'a-events',
      'points',
      [makeColumn('rainfall', 'number', 'amount')],
      'events'
    );
    const mapping = createKeplerCuratedAnalysisMapping(idw, [events, gauges]);
    expect(mapping.slots[0].datasetId).toBe('gauges');
    expect(mapping.slots[0].state).toBe('settled');
  });

  it('with only wrong-kind data the slot is empty, the analysis needs-data and the problem names the kind', () => {
    const mapping = createKeplerCuratedAnalysisMapping(idw, [districts]);
    expect(mapping.slots[0].datasetId).toBeUndefined();
    expect(mapping.problems[0].kind).toBe('slot-empty');
    expect(mapping.problems[0].message).toContain('needs points');
    const [availability] = listKeplerCuratedAnalysesForDatasets([idw], [districts]);
    expect(availability.status).toBe('needs-data');
    expect(availability.reason).toContain('points');
  });

  it('treats a lines dataset as a network', () => {
    const analysis = getAnalysis('drive-time-catchments');
    const streets = makeProfile('streets', 'lines', []);
    const slot = createKeplerCuratedAnalysisMapping(analysis, [streets]).slots.find(
      entry => entry.slot === 'network'
    );
    expect(slot?.candidates[0].fit).toBe('fits');
    expect(slot?.datasetId).toBe('streets');
  });

  it('ignores a repeated dataset id after the first and reports it', () => {
    const other = makeProfile(
      'gauges',
      'points',
      [makeColumn('depth', 'number', 'amount')],
      'surface-samples'
    );
    const mapping = createKeplerCuratedAnalysisMapping(idw, [gauges, other]);
    expect(mapping.slots[0].candidates).toHaveLength(1);
    expect(mapping.slots[0].roles[0].columns).toEqual(['rainfall']);
    expect(mapping.problems.map(problem => [problem.kind, problem.severity])).toEqual([
      ['dataset-duplicate', 'check']
    ]);
    expect(mapping.problems[0].message).toContain('"gauges"');
    expect(mapping.ready).toBe(true);
    const list = listKeplerCuratedAnalysesForDatasets(KEPLER_CURATED_ANALYSES, [gauges, gauges]);
    expect(list).toHaveLength(KEPLER_CURATED_ANALYSES.length);
  });

  it('uses the one dataset of a kind for the facilities slot and another for observations by row semantics', () => {
    // The real analysis may leave row semantics open; this copy states them, as the check is about the scoring.
    const open = getAnalysis('straight-line-catchments');
    const analysis: KeplerCuratedAnalysis = {
      ...open,
      slots: open.slots.map(slot => ({
        ...slot,
        rowSemantics: slot.name === 'facilities' ? ['objects'] : ['events', 'objects']
      }))
    };
    const facilities = makeProfile('depots', 'points', [], 'objects');
    const observations = makeProfile(
      'waits',
      'points',
      [makeColumn('minutes', 'number', 'amount')],
      'events'
    );
    const mapping = createKeplerCuratedAnalysisMapping(analysis, [observations, facilities]);
    expect(mapping.slots.map(slot => [slot.slot, slot.datasetId, slot.state])).toEqual([
      ['facilities', 'depots', 'settled'],
      ['observations', 'waits', 'settled']
    ]);
    expect(mapping.ready).toBe(true);
  });
});

describe('two slots of one kind', () => {
  const period = getAnalysis('period-comparison');
  const incidents = (id: string, label: string = id) =>
    makeProfile(id, 'points', [makeColumn('severity', 'number', 'count')], 'events', label);

  it('reports a conflict, not a duplicate, when one dataset is the only candidate for both', () => {
    const mapping = createKeplerCuratedAnalysisMapping(period, [incidents('incidents')]);
    expect(mapping.slots.map(slot => [slot.datasetId, slot.conflict])).toEqual([
      ['incidents', false],
      ['incidents', true]
    ]);
    expect(mapping.complete).toBe(false);
    expect(mapping.progress).toEqual({mapped: 1, required: 2});
    expect(mapping.problems.map(problem => problem.kind)).toEqual(['slot-conflict']);
    expect(() => buildKeplerCuratedAnalysisRequest(period, mapping)).toThrow(/not complete/);
    const [availability] = listKeplerCuratedAnalysesForDatasets([period], [incidents('incidents')]);
    expect(availability.status).toBe('needs-data');
    expect(availability.reason).toContain('Earlier points and Later points');
  });

  it('uses two datasets once, in slot order by name affinity, and settles', () => {
    const mapping = createKeplerCuratedAnalysisMapping(period, [
      incidents('z-2024', 'after'),
      incidents('a-2023', 'before')
    ]);
    expect(mapping.slots.map(slot => [slot.slot, slot.datasetId, slot.state])).toEqual([
      ['before', 'a-2023', 'settled'],
      ['after', 'z-2024', 'settled']
    ]);
    expect(mapping.ready).toBe(true);
  });

  it('flags equal datasets as a choice, orders by id, and a user choice swaps them', () => {
    const datasets = [incidents('b-set'), incidents('a-set')];
    const mapping = createKeplerCuratedAnalysisMapping(period, datasets);
    expect(mapping.slots.map(slot => [slot.datasetId, slot.state])).toEqual([
      ['a-set', 'choose'],
      ['b-set', 'choose']
    ]);
    expect(mapping.ready).toBe(false);
    expect(mapping.complete).toBe(true);

    const swapped = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'before', datasetId: 'b-set'},
      period,
      datasets
    );
    // The other slot re-derives to the remaining dataset; the user's slot is settled.
    expect(
      swapped.slots.map(slot => [slot.datasetId, slot.source, slot.state, slot.conflict])
    ).toEqual([
      ['b-set', 'user', 'settled', false],
      ['a-set', 'automatic', 'settled', false]
    ]);
    expect(swapped.ready).toBe(true);
  });

  it('flags a conflict when the user puts one dataset in both slots', () => {
    const datasets = [incidents('b-set'), incidents('a-set')];
    let mapping = createKeplerCuratedAnalysisMapping(period, datasets);
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'before', datasetId: 'a-set'},
      period,
      datasets
    );
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'after', datasetId: 'a-set'},
      period,
      datasets
    );
    expect(mapping.slots.map(slot => [slot.datasetId, slot.conflict])).toEqual([
      ['a-set', false],
      ['a-set', true]
    ]);
    expect(mapping.complete).toBe(false);
  });
});

// ---------------------------------------------------------------------------------------------
// Conditional roles
// ---------------------------------------------------------------------------------------------

describe('conditional roles', () => {
  const noWeight = makeProfile('areas', 'polygons', [
    makeColumn('n_cases', 'integer', 'count'),
    makeColumn('pop_total', 'integer', 'count')
  ]);
  const withWeight = makeProfile('areas', 'polygons', [
    makeColumn('n_cases', 'integer', 'count'),
    makeColumn('pop_total', 'integer', 'count'),
    makeColumn('weight', 'number', 'amount')
  ]);

  it('is not required while the preset parameter says so, and appears with another value', () => {
    const plain = createKeplerCuratedAnalysisMapping(SYNTHETIC_ANALYSIS, [noWeight]);
    expect(plain.slots[0].roles[3].required).toBe(false);
    expect(plain.progress.required).toBe(3);
    expect(plain.slots[0].candidates[0].fit).toBe('fits');

    const weighted = createKeplerCuratedAnalysisMapping(SYNTHETIC_ANALYSIS, [noWeight], {
      parameters: {mode: 'weighted'}
    });
    expect(weighted.slots[0].roles[3].required).toBe(true);
    expect(weighted.progress.required).toBe(4);
    expect(weighted.slots[0].candidates[0].fit).toBe('missing-columns');
    expect(weighted.slots[0].candidates[0].reason).toContain('Weight');
    expect(weighted.complete).toBe(false);

    const fits = createKeplerCuratedAnalysisMapping(SYNTHETIC_ANALYSIS, [withWeight], {
      parameters: {mode: 'weighted'}
    });
    expect(fits.slots[0].roles[3].columns).toEqual(['weight']);
    expect(fits.complete).toBe(true);
  });

  it('keeps its parameter values through update', () => {
    const weighted = createKeplerCuratedAnalysisMapping(SYNTHETIC_ANALYSIS, [withWeight], {
      parameters: {mode: 'weighted'}
    });
    const updated = updateKeplerCuratedAnalysisMapping(
      weighted,
      {slot: 'areas', role: 'cases', columns: ['n_cases']},
      SYNTHETIC_ANALYSIS,
      [withWeight]
    );
    expect(updated.slots[0].roles[3].required).toBe(true);
    expect(() =>
      updateKeplerCuratedAnalysisMapping(
        updated,
        {slot: 'areas', role: 'weight', columns: []},
        SYNTHETIC_ANALYSIS,
        [withWeight]
      )
    ).toThrow(/areas\.weight.*required/);
  });

  it('requires, in the request, a role that an override makes required', () => {
    const plain = createKeplerCuratedAnalysisMapping(SYNTHETIC_ANALYSIS, [noWeight]);
    expect(plain.slots[0].roles[3].columns).toEqual([]);
    expect(() =>
      buildKeplerCuratedAnalysisRequest(SYNTHETIC_ANALYSIS, plain, {
        parameterOverrides: {mode: 'weighted'}
      })
    ).toThrow(/Weight has no column/);
    expect(buildKeplerCuratedAnalysisRequest(SYNTHETIC_ANALYSIS, plain).parameters).toEqual({
      mode: 'plain',
      valueProperty: 'n_cases'
    });
    // A mapping derived with the same parameter is complete and passes the same override.
    const weighted = createKeplerCuratedAnalysisMapping(SYNTHETIC_ANALYSIS, [withWeight], {
      parameters: {mode: 'weighted'}
    });
    expect(weighted.slots[0].roles[3].columns).toEqual(['weight']);
    expect(
      buildKeplerCuratedAnalysisRequest(SYNTHETIC_ANALYSIS, weighted, {
        parameterOverrides: {mode: 'weighted'}
      }).parameters
    ).toMatchObject({mode: 'weighted'});
  });
});

// ---------------------------------------------------------------------------------------------
// Update
// ---------------------------------------------------------------------------------------------

describe('updateKeplerCuratedAnalysisMapping', () => {
  const areasAnalysis = getAnalysis('cluster-and-outlier-map');
  const tractsA = makeProfile('tracts-a', 'polygons', [
    makeColumn('rate', 'number', 'rate'),
    makeColumn('label', 'text', 'nominal')
  ]);
  const tractsB = makeProfile('tracts-b', 'polygons', [
    makeColumn('index_value', 'number', 'rate')
  ]);
  const noValue = makeProfile('no-value', 'polygons', [makeColumn('label', 'text', 'nominal')]);
  const points = makeProfile('points', 'points', [makeColumn('v', 'number', 'amount')]);
  const datasets = [tractsA, tractsB, noValue, points];

  it('re-derives the roles when the dataset changes, and is pure', () => {
    const mapping = createKeplerCuratedAnalysisMapping(areasAnalysis, datasets);
    expect(mapping.slots[0].datasetId).toBe('tracts-a');
    const before = JSON.stringify(mapping);
    const changed = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'areas', datasetId: 'tracts-b'},
      areasAnalysis,
      datasets
    );
    expect(JSON.stringify(mapping)).toBe(before);
    expect(changed).not.toBe(mapping);
    expect(changed.slots[0].datasetId).toBe('tracts-b');
    expect(changed.slots[0].source).toBe('user');
    expect(changed.slots[0].roles[0].columns).toEqual(['index_value']);
  });

  it('drops the column choices of a slot when its dataset changes, but keeps them for the same dataset', () => {
    const wide = makeProfile('wide', 'polygons', [
      makeColumn('rate_a', 'number', 'rate'),
      makeColumn('rate_b', 'number', 'rate')
    ]);
    const wide2 = makeProfile('wide-2', 'polygons', [
      makeColumn('rate_a', 'number', 'rate'),
      makeColumn('rate_b', 'number', 'rate')
    ]);
    const all = [wide, wide2];
    let mapping = createKeplerCuratedAnalysisMapping(areasAnalysis, all);
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'areas', role: 'value', columns: ['rate_b']},
      areasAnalysis,
      all
    );
    const same = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'areas', datasetId: 'wide'},
      areasAnalysis,
      all
    );
    expect(same.slots[0].roles[0].columns).toEqual(['rate_b']);
    const other = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'areas', datasetId: 'wide-2'},
      areasAnalysis,
      all
    );
    expect(other.slots[0].roles[0].columns).toEqual(['rate_a']);
    expect(other.slots[0].roles[0].state).toBe('choose');
    expect(other.choices.columns).toEqual({});
  });

  it('rejects every invalid change with an error naming the function, the slot or role and the reason', () => {
    const mapping = createKeplerCuratedAnalysisMapping(areasAnalysis, datasets);
    const update = (change: Parameters<typeof updateKeplerCuratedAnalysisMapping>[1]) => () =>
      updateKeplerCuratedAnalysisMapping(mapping, change, areasAnalysis, datasets);
    const prefix = 'updateKeplerCuratedAnalysisMapping';
    expect(update({slot: 'nowhere', datasetId: 'tracts-a'})).toThrow(
      new RegExp(
        `${prefix}: slot "nowhere" is not a slot of "cluster-and-outlier-map"; accepted: areas`
      )
    );
    expect(update({slot: 'areas', datasetId: 'ghost'})).toThrow(
      /slot "areas": dataset "ghost" is not loaded/
    );
    expect(update({slot: 'areas', datasetId: 'points'})).toThrow(
      /dataset "points" cannot be used \(wrong-kind\).*needs polygons/
    );
    expect(update({slot: 'areas', datasetId: 'no-value'})).toThrow(
      /dataset "no-value" cannot be used \(missing-columns\).*Value/
    );
    expect(update({slot: 'areas', role: 'nope', columns: ['rate']})).toThrow(
      /role "nope" is not a role of slot "areas"; accepted: value/
    );
    expect(update({slot: 'areas', role: 'value', columns: ['ghost']})).toThrow(
      /areas\.value.*"ghost" is not a column of tracts-a/
    );
    expect(update({slot: 'areas', role: 'value', columns: ['label']})).toThrow(
      /areas\.value.*"label" is text; Value takes number or integer/
    );
    expect(update({slot: 'areas', role: 'value', columns: []})).toThrow(/areas\.value.*required/);
    expect(update({slot: 'areas', role: 'value', columns: ['rate', 'rate']})).toThrow(
      /areas\.value.*takes 1 column\(s\); got 2/
    );
    expect(() =>
      updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'areas', datasetId: 'tracts-a'},
        getAnalysis('idw-surface'),
        datasets
      )
    ).toThrow(/mapping is for analysis "cluster-and-outlier-map", not "idw-surface"/);
    const empty = createKeplerCuratedAnalysisMapping(areasAnalysis, [points]);
    expect(() =>
      updateKeplerCuratedAnalysisMapping(
        empty,
        {slot: 'areas', role: 'value', columns: ['v']},
        areasAnalysis,
        [points]
      )
    ).toThrow(/slot "areas" has no dataset/);
  });

  it('lets the user choose an optional role and clearing it removes the choice', () => {
    const density = getAnalysis('density-surface');
    const incidents = makeProfile(
      'incidents',
      'points',
      [makeColumn('weight', 'number', 'count')],
      'events'
    );
    const mapping = createKeplerCuratedAnalysisMapping(density, [incidents]);
    const chosen = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'points', role: 'weight', columns: ['weight']},
      density,
      [incidents]
    );
    expect(chosen.slots[0].roles[0].columns).toEqual(['weight']);
    expect(chosen.slots[0].roles[0].source).toBe('user');
    const cleared = updateKeplerCuratedAnalysisMapping(
      chosen,
      {slot: 'points', role: 'weight', columns: []},
      density,
      [incidents]
    );
    expect(cleared.slots[0].roles[0].columns).toEqual([]);
    expect(cleared.slots[0].roles[0].source).toBe('none');
    expect(cleared.choices.columns).toEqual({});
    expect(buildKeplerCuratedAnalysisRequest(density, cleared).parameters).not.toHaveProperty(
      'weightProperty'
    );
  });

  it('applies an update to datasets that changed since the mapping was made', () => {
    const mapping = createKeplerCuratedAnalysisMapping(areasAnalysis, [tractsA]);
    const next = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'areas', datasetId: 'tracts-b'},
      areasAnalysis,
      [tractsA, tractsB]
    );
    expect(next.slots[0].datasetId).toBe('tracts-b');
  });
});

// ---------------------------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------------------------

describe('buildKeplerCuratedAnalysisRequest', () => {
  const idw = getAnalysis('idw-surface');
  const gauges = makeProfile(
    'gauges',
    'points',
    [makeColumn('rainfall', 'number', 'amount')],
    'surface-samples'
  );

  it('applies presets, then field parameters, then overrides', () => {
    const mapping = createKeplerCuratedAnalysisMapping(idw, [gauges]);
    const request = buildKeplerCuratedAnalysisRequest(idw, mapping, {
      parameterOverrides: {power: 3, cellSize: 250}
    });
    expect(request.bindingId).toBe('idw-grid');
    expect(request.inputDatasetIds).toEqual({points: 'gauges'});
    expect(request.parameters).toMatchObject({
      power: 3,
      neighborCount: 12,
      valueProperty: 'rainfall',
      cellSize: 250
    });
    // Parameters nobody set are absent so the data-derived default can fill them.
    expect(request.parameters).not.toHaveProperty('searchRadius');
    expect(
      normalizeCatalogParameters(getCatalogBinding('idw-grid')!, request.parameters)
    ).toMatchObject({power: 3});
  });

  it('retains the mapping settings before applying mapped fields and explicit overrides', () => {
    const mapping = createKeplerCuratedAnalysisMapping(idw, [gauges], {
      parameters: {cellSize: 17, power: 4, valueProperty: 'stale-column'}
    });
    expect(buildKeplerCuratedAnalysisRequest(idw, mapping).parameters).toMatchObject({
      cellSize: 17,
      power: 4,
      valueProperty: 'rainfall'
    });
    expect(
      buildKeplerCuratedAnalysisRequest(idw, mapping, {
        parameterOverrides: {power: 3}
      }).parameters
    ).toMatchObject({cellSize: 17, power: 3, valueProperty: 'rainfall'});
  });

  it('rejects mapping settings inconsistent with the conditional roles used to derive it', () => {
    const analysis = getAnalysis('count-by-area');
    const mapping = createKeplerCuratedAnalysisMapping(analysis, [
      makeProfile('events', 'points', [makeColumn('amount', 'number', 'amount')], 'events'),
      makeProfile('areas', 'polygons', [])
    ]);
    expect(() =>
      buildKeplerCuratedAnalysisRequest(analysis, {
        ...mapping,
        parameterValues: {...mapping.parameterValues, statistic: 'mean'}
      })
    ).toThrow(
      /buildKeplerCuratedAnalysisRequest.*mapping.parameterValues.*inconsistent.*points.value/
    );
  });

  it('refuses an override of a field parameter, a needs-binding analysis and an incomplete mapping', () => {
    const mapping = createKeplerCuratedAnalysisMapping(idw, [gauges]);
    expect(() =>
      buildKeplerCuratedAnalysisRequest(idw, mapping, {parameterOverrides: {valueProperty: 'x'}})
    ).toThrow(/valueProperty.*change the mapping/);
    const pending: KeplerCuratedAnalysis = {
      ...idw,
      exposedParameters: [],
      execution: {
        status: 'needs-binding',
        implementation: 'addFutureRecipe',
        reason: 'No binding has been registered for this future recipe.'
      }
    };
    expect(() =>
      buildKeplerCuratedAnalysisRequest(
        pending,
        createKeplerCuratedAnalysisMapping(pending, [gauges])
      )
    ).toThrow(/needs-binding/);
    expect(() =>
      buildKeplerCuratedAnalysisRequest(idw, createKeplerCuratedAnalysisMapping(idw, []))
    ).toThrow(/buildKeplerCuratedAnalysisRequest.*Measurements: no loaded dataset fits/);
    expect(() => buildKeplerCuratedAnalysisRequest(getAnalysis('hot-spot-map'), mapping)).toThrow(
      /mapping is for analysis "idw-surface"/
    );
  });

  it('builds from an ambiguous but complete mapping, using the first pick', () => {
    const areas = getAnalysis('cluster-and-outlier-map');
    const datasets = [
      makeProfile('tracts', 'polygons', [
        makeColumn('rate_a', 'number', 'rate'),
        makeColumn('rate_b', 'number', 'rate')
      ])
    ];
    const mapping = createKeplerCuratedAnalysisMapping(areas, datasets);
    expect(mapping.ready).toBe(false);
    expect(buildKeplerCuratedAnalysisRequest(areas, mapping).parameters['valueProperty']).toBe(
      'rate_a'
    );
  });
});

// ---------------------------------------------------------------------------------------------
// Listing
// ---------------------------------------------------------------------------------------------

describe('listKeplerCuratedAnalysesForDatasets', () => {
  const gauges = makeProfile(
    'gauges',
    'points',
    [makeColumn('rainfall', 'number', 'amount')],
    'surface-samples'
  );
  const districts = makeProfile('districts', 'polygons', [makeColumn('rate', 'number', 'rate')]);

  it('covers every analysis once and orders ready, needs-mapping, needs-data, needs-binding, then wave and table order', () => {
    const list = listKeplerCuratedAnalysesForDatasets(KEPLER_CURATED_ANALYSES, [gauges, districts]);
    expect(list.map(entry => entry.analysisId).sort()).toEqual(
      KEPLER_CURATED_ANALYSES.map(analysis => analysis.id).sort()
    );
    const rank = {ready: 0, 'needs-mapping': 1, 'needs-data': 2, 'needs-binding': 3};
    const ranks = list.map(entry => rank[entry.status]);
    expect(ranks).toEqual([...ranks].sort((left, right) => left - right));
    for (const status of Object.keys(rank)) {
      const group = list.filter(entry => entry.status === status);
      const keys = group.map(entry => {
        const index = KEPLER_CURATED_ANALYSES.findIndex(
          analysis => analysis.id === entry.analysisId
        );
        return KEPLER_CURATED_ANALYSES[index].wave * 1000 + index;
      });
      expect(keys).toEqual([...keys].sort((left, right) => left - right));
    }
    const byId = new Map(list.map(entry => [entry.analysisId, entry]));
    expect(byId.get('idw-surface')?.status).toBe('ready');
    // Rain gauges declared as surface samples contradict the density surface's events.
    expect(byId.get('density-surface')?.status).toBe('needs-mapping');
    expect(byId.get('cluster-and-outlier-map')?.status).toBe('ready');
    expect(byId.get('hot-spot-map')?.status).toBe('needs-mapping');
    expect(byId.get('drive-time-catchments')?.status).toBe('needs-data');
    expect(byId.get('drive-time-catchments')?.reason).toContain('lines');
    for (const entry of list) {
      const analysis = getAnalysis(entry.analysisId);
      expect(entry.mapping === undefined).toBe(analysis.execution.status === 'needs-binding');
      expect(entry.reason.length).toBeGreaterThan(0);
    }
  });

  it('puts a ready wave 2 analysis before a needs-mapping wave 1 analysis', () => {
    const ambiguous = makeProfile('tracts', 'polygons', [
      makeColumn('rate_a', 'number', 'rate'),
      makeColumn('rate_b', 'number', 'rate')
    ]);
    const list = listKeplerCuratedAnalysesForDatasets(
      [getAnalysis('cluster-and-outlier-map'), getAnalysis('idw-surface')],
      [ambiguous, gauges]
    );
    expect(list.map(entry => [entry.analysisId, entry.status])).toEqual([
      ['idw-surface', 'ready'],
      ['cluster-and-outlier-map', 'needs-mapping']
    ]);
  });

  it('is empty for no analyses', () => {
    expect(listKeplerCuratedAnalysesForDatasets([], [gauges])).toEqual([]);
  });
});

// ---------------------------------------------------------------------------------------------
// Save and replay
// ---------------------------------------------------------------------------------------------

describe('serialise, parse and replay', () => {
  const areas = getAnalysis('cluster-and-outlier-map');
  const wide = makeProfile('wide', 'polygons', [
    makeColumn('rate_a', 'number', 'rate'),
    makeColumn('rate_b', 'number', 'rate')
  ]);

  it("saves only the user's choices", () => {
    const automatic = createKeplerCuratedAnalysisMapping(areas, [wide]);
    expect(JSON.parse(serializeKeplerCuratedAnalysisMapping(automatic))).toEqual({
      version: 1,
      analysisId: 'cluster-and-outlier-map',
      datasets: {},
      columns: {}
    });
    const chosen = updateKeplerCuratedAnalysisMapping(
      automatic,
      {slot: 'areas', role: 'value', columns: ['rate_b']},
      areas,
      [wide]
    );
    expect(JSON.parse(serializeKeplerCuratedAnalysisMapping(chosen)).columns).toEqual({
      'areas.value': ['rate_b']
    });
  });

  it('round-trips and replays to an equal mapping on the same data', () => {
    const datasets = [
      wide,
      makeProfile(
        'wide-2',
        'polygons',
        wide.columns.map(column => makeColumn(column.name, 'number', 'rate'))
      )
    ];
    let mapping = createKeplerCuratedAnalysisMapping(areas, datasets);
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'areas', datasetId: 'wide-2'},
      areas,
      datasets
    );
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'areas', role: 'value', columns: ['rate_b']},
      areas,
      datasets
    );
    const text = serializeKeplerCuratedAnalysisMapping(mapping);
    expect(text).toBe(serializeKeplerCuratedAnalysisMapping(mapping));
    const choices = parseKeplerCuratedAnalysisMapping(text);
    expect(parseKeplerCuratedAnalysisMapping(JSON.parse(text))).toEqual(choices);
    const replayed = replayKeplerCuratedAnalysisMapping(areas, choices, datasets);
    expect(replayed).toEqual(mapping);
    expect(replayed.problems).toEqual([]);
  });

  it('reports a column that no longer exists as a problem and chooses again, without throwing', () => {
    const mapping = updateKeplerCuratedAnalysisMapping(
      createKeplerCuratedAnalysisMapping(areas, [wide]),
      {slot: 'areas', role: 'value', columns: ['rate_b']},
      areas,
      [wide]
    );
    const choices = parseKeplerCuratedAnalysisMapping(
      serializeKeplerCuratedAnalysisMapping(mapping)
    );
    const dropped = makeProfile('wide', 'polygons', [
      makeColumn('rate_a', 'number', 'rate'),
      makeColumn('label', 'text', 'nominal')
    ]);
    const replayed = replayKeplerCuratedAnalysisMapping(areas, choices, [dropped]);
    expect(
      replayed.problems.map(problem => [problem.kind, problem.severity, problem.slot, problem.role])
    ).toEqual([['saved-choice-lost', 'check', 'areas', 'value']]);
    expect(replayed.problems[0].message).toContain('"rate_b"');
    expect(replayed.slots[0].roles[0].columns).toEqual(['rate_a']);
    expect(replayed.slots[0].roles[0].state).toBe('choose');
    expect(replayed.ready).toBe(false);
    expect(replayed.complete).toBe(true);
  });

  it('reports a column that is still there but of a type the role no longer accepts', () => {
    const choices = {analysisId: areas.id, datasets: {}, columns: {'areas.value': ['label']}};
    const text = makeProfile('wide', 'polygons', [
      makeColumn('label', 'text', 'nominal'),
      makeColumn('rate_a', 'number', 'rate')
    ]);
    const replayed = replayKeplerCuratedAnalysisMapping(areas, choices, [text]);
    expect(replayed.problems[0].kind).toBe('saved-choice-lost');
    expect(replayed.slots[0].roles[0].columns).toEqual(['rate_a']);
  });

  it('reports a saved dataset that is gone or no longer fits, and unknown slots', () => {
    const choices = {
      analysisId: areas.id,
      datasets: {areas: 'ghost', elsewhere: 'wide'},
      columns: {'elsewhere.value': ['x']}
    };
    const replayed = replayKeplerCuratedAnalysisMapping(areas, choices, [wide]);
    expect(replayed.problems.map(problem => problem.kind)).toEqual([
      'saved-choice-lost',
      'saved-choice-lost',
      'saved-choice-lost',
      'needs-choice'
    ]);
    expect(replayed.problems[2].message).toContain('Saved dataset "ghost"');
    expect(replayed.slots[0].datasetId).toBe('wide');
    expect(replayed.slots[0].state).toBe('choose');
    expect(replayed.ready).toBe(false);
  });

  it('throws only for choices that belong to another analysis', () => {
    expect(() =>
      replayKeplerCuratedAnalysisMapping(areas, {analysisId: 'other', datasets: {}, columns: {}}, [
        wide
      ])
    ).toThrow(/replayKeplerCuratedAnalysisMapping.*"other"/);
  });

  it('parses strictly', () => {
    const good = {version: 1, analysisId: 'x', datasets: {a: 'd'}, columns: {'a.v': ['c']}};
    expect(parseKeplerCuratedAnalysisMapping(good)).toEqual({
      analysisId: 'x',
      datasets: {a: 'd'},
      columns: {'a.v': ['c']}
    });
    const bad: [string, unknown][] = [
      ['not valid JSON', '{nope'],
      ['must be an object', 42],
      ['must be an object', [1]],
      ['must be an object', null],
      ['version must be 1', {...good, version: 2}],
      ['version must be 1', {analysisId: 'x', datasets: {}, columns: {}}],
      ['unknown key "extra"', {...good, extra: 1}],
      ['analysisId', {...good, analysisId: ''}],
      ['analysisId', {...good, analysisId: 3}],
      ['datasets and columns must be objects', {...good, datasets: []}],
      ['datasets and columns must be objects', {...good, columns: 'x'}],
      ['datasets["a"]', {...good, datasets: {a: 3}}],
      ['datasets["a"]', {...good, datasets: {a: ''}}],
      ['columns["a.v"]', {...good, columns: {'a.v': 'c'}}],
      ['columns["a.v"]', {...good, columns: {'a.v': [1]}}],
      ['columns["nodot"]', {...good, columns: {nodot: ['c']}}]
    ];
    for (const [message, input] of bad) {
      expect(() => parseKeplerCuratedAnalysisMapping(input)).toThrow(
        new RegExp(`parseKeplerCuratedAnalysisMapping.*${message.replace(/[[\]"().]/g, '\\$&')}`)
      );
    }
  });
});

// ---------------------------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------------------------

describe('determinism', () => {
  it('gives equal mappings for equal input and for datasets in any order', () => {
    for (const analysis of AVAILABLE_ANALYSES) {
      const datasets = [
        ...makeSampleShapedProfiles(analysis),
        makeProfile(
          'zz-extra-points',
          'points',
          [makeColumn('count_a', 'number', 'count')],
          'events'
        ),
        makeProfile('aa-extra-areas', 'polygons', [makeColumn('rate_a', 'number', 'rate')])
      ];
      const first = createKeplerCuratedAnalysisMapping(analysis, datasets);
      expect(createKeplerCuratedAnalysisMapping(analysis, datasets)).toEqual(first);
      expect(createKeplerCuratedAnalysisMapping(analysis, [...datasets].reverse())).toEqual(first);
      expect(listKeplerCuratedAnalysesForDatasets(KEPLER_CURATED_ANALYSES, datasets)).toEqual(
        listKeplerCuratedAnalysesForDatasets(KEPLER_CURATED_ANALYSES, [...datasets].reverse())
      );
    }
  });

  it('does not change the datasets or the analysis it is given', () => {
    const analysis = getAnalysis('idw-surface');
    const datasets = makeSampleShapedProfiles(analysis);
    const before = JSON.stringify([analysis, datasets]);
    updateKeplerCuratedAnalysisMapping(
      createKeplerCuratedAnalysisMapping(analysis, datasets),
      {slot: 'points', role: 'value', columns: ['rainfall']},
      analysis,
      datasets
    );
    expect(JSON.stringify([analysis, datasets])).toBe(before);
  });
});

// ---------------------------------------------------------------------------------------------
// Fix round: lists of columns and their limits
// ---------------------------------------------------------------------------------------------

function makePredictorProfile(names: readonly string[]): KeplerDatasetProfile {
  return makeProfile(
    'districts',
    'polygons',
    [
      makeColumn('outcome', 'number', 'amount'),
      ...names.map(name => makeColumn(name, 'number', 'amount'))
    ],
    'objects'
  );
}

describe('a role that holds several columns', () => {
  const regression = getAnalysis('regression-with-diagnostics');

  it('writes the predictors as a comma-separated list the catalog accepts', () => {
    const profile = makePredictorProfile(['income', 'density', 'age', 'rent', 'jobs']);
    for (const count of [1, 2, 5]) {
      const names = ['income', 'density', 'age', 'rent', 'jobs'].slice(0, count);
      let mapping = createKeplerCuratedAnalysisMapping(regression, [profile]);
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'areas', role: 'response', columns: ['outcome']},
        regression,
        [profile]
      );
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'areas', role: 'predictors', columns: names},
        regression,
        [profile]
      );
      expect(mapping.complete).toBe(true);
      const request = buildKeplerCuratedAnalysisRequest(regression, mapping);
      expect(request.parameters['predictorProperties']).toBe(names.join(','));
      expect(request.parameters['responseProperty']).toBe('outcome');
      expect(() =>
        normalizeCatalogParameters(getCatalogBinding(request.bindingId)!, request.parameters)
      ).not.toThrow();
    }
  });

  it('lists a name with a comma as unavailable, never chooses it and rejects it with a clear message', () => {
    const profile = makePredictorProfile(['a,b', 'income']);
    const mapping = createKeplerCuratedAnalysisMapping(regression, [profile]);
    const predictors = mapping.slots[0].roles.find(role => role.role === 'predictors')!;
    const comma = predictors.candidates.find(candidate => candidate.name === 'a,b');
    expect(comma?.unavailable).toContain('comma');
    expect(predictors.columns).not.toContain('a,b');
    expect(() =>
      updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'areas', role: 'predictors', columns: ['a,b']},
        regression,
        [profile]
      )
    ).toThrow(/updateKeplerCuratedAnalysisMapping.*areas\.predictors.*"a,b" contains a comma/);
    // A single-column role has no list, so a comma there is harmless.
    const response = mapping.slots[0].roles.find(role => role.role === 'response')!;
    expect(
      response.candidates.find(candidate => candidate.name === 'a,b')?.unavailable
    ).toBeUndefined();
  });

  it('checks the written length against the parameter limit before the mapping is complete', () => {
    const longNames = Array.from(
      {length: 15},
      (_, index) => `${'p'.repeat(38)}${String(index).padStart(2, '0')}`
    );
    const profile = makePredictorProfile(longNames);
    const mapping = createKeplerCuratedAnalysisMapping(regression, [profile]);
    // The limit is the binding parameter's maximumLength, read from the catalog.
    const limit = getCatalogBinding('curated-regression-with-diagnostics')!.parameters.find(
      parameter => parameter.name === 'predictorProperties'
    )!.maximumLength!;
    expect(longNames.join(',').length).toBeGreaterThan(limit);
    expect(() =>
      updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'areas', role: 'predictors', columns: longNames},
        regression,
        [profile]
      )
    ).toThrow(/areas\.predictors.*characters written out.*predictorProperties holds at most/);
    // Fewer of them fit and build.
    const fits = longNames.slice(0, Math.floor(limit / 41));
    const updated = updateKeplerCuratedAnalysisMapping(
      updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'areas', role: 'response', columns: ['outcome']},
        regression,
        [profile]
      ),
      {slot: 'areas', role: 'predictors', columns: fits},
      regression,
      [profile]
    );
    const request = buildKeplerCuratedAnalysisRequest(regression, updated);
    expect(() =>
      normalizeCatalogParameters(getCatalogBinding(request.bindingId)!, request.parameters)
    ).not.toThrow();
  });

  it('lists a name longer than a single-column parameter holds as unavailable, so the mapping is not complete', () => {
    const idw = getAnalysis('idw-surface');
    const longName = 'v'.repeat(97);
    const profile = makeProfile(
      'gauges',
      'points',
      [makeColumn(longName, 'number', 'amount')],
      'surface-samples'
    );
    const mapping = createKeplerCuratedAnalysisMapping(idw, [profile]);
    const role = mapping.slots[0].roles[0];
    expect(role.candidates[0].unavailable).toContain('97 characters');
    expect(role.columns).toEqual([]);
    expect(mapping.complete).toBe(false);
    expect(mapping.problems[0].kind).toBe('role-empty');
    expect(() => buildKeplerCuratedAnalysisRequest(idw, mapping)).toThrow(/not complete/);
    expect(() =>
      updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'points', role: 'value', columns: [longName]},
        idw,
        [profile]
      )
    ).toThrow(/97 characters/);
  });
});

// ---------------------------------------------------------------------------------------------
// Fix round: when `ready` is honest
// ---------------------------------------------------------------------------------------------

describe('ready and what the user declared or chose', () => {
  const gauges = (source: 'declared' | 'guess' | undefined): KeplerDatasetProfile => ({
    ...makeProfile('gauges', 'points', [makeColumn('rainfall', 'number', 'amount')]),
    ...(source === undefined ? {} : {rowSemantics: {value: 'surface-samples' as const, source}})
  });

  it('is not ready when declared row semantics contradict the slot, and says so in the problems', () => {
    for (const id of ['hot-spot-map', 'density-surface']) {
      const analysis = getAnalysis(id);
      const mapping = createKeplerCuratedAnalysisMapping(analysis, [gauges('declared')]);
      expect(mapping.complete).toBe(true);
      expect(mapping.ready).toBe(false);
      expect(mapping.slots[0].state).toBe('choose');
      expect(mapping.problems).toHaveLength(1);
      expect(mapping.problems[0]).toMatchObject({
        kind: 'needs-choice',
        severity: 'check',
        acknowledged: false
      });
      expect(mapping.problems[0].message).toContain('surface-samples (declared)');
      const [availability] = listKeplerCuratedAnalysesForDatasets([analysis], [gauges('declared')]);
      expect(availability.status).toBe('needs-mapping');
      expect(availability.reason).toContain('surface-samples');
    }
  });

  it('stays a warning for guessed or absent semantics, and the user choosing the dataset acknowledges it', () => {
    const hotSpot = getAnalysis('hot-spot-map');
    const guessed = createKeplerCuratedAnalysisMapping(hotSpot, [gauges('guess')]);
    expect(guessed.ready).toBe(true);
    expect(guessed.problems).toMatchObject([
      {kind: 'dataset-warning', severity: 'check', acknowledged: false}
    ]);
    expect(createKeplerCuratedAnalysisMapping(hotSpot, [gauges(undefined)]).problems).toEqual([]);

    const declared = createKeplerCuratedAnalysisMapping(hotSpot, [gauges('declared')]);
    const chosen = updateKeplerCuratedAnalysisMapping(
      declared,
      {slot: 'points', datasetId: 'gauges'},
      hotSpot,
      [gauges('declared')]
    );
    expect(chosen.ready).toBe(true);
    expect(chosen.problems).toMatchObject([{kind: 'dataset-warning', acknowledged: true}]);
  });

  it('keeps an explicitly chosen count for a rate role ready, with an acknowledged column warning', () => {
    const areas = getAnalysis('cluster-and-outlier-map');
    const profile = makeProfile('tracts', 'polygons', [
      makeColumn('rate_a', 'number', 'rate'),
      makeColumn('n_cases', 'integer', 'count')
    ]);
    const mapping = createKeplerCuratedAnalysisMapping(areas, [profile]);
    expect(mapping.ready).toBe(true);
    const chosen = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'areas', role: 'value', columns: ['n_cases']},
      areas,
      [profile]
    );
    expect(chosen.ready).toBe(true);
    expect(chosen.problems).toHaveLength(1);
    expect(chosen.problems[0]).toMatchObject({
      kind: 'column-warning',
      severity: 'check',
      acknowledged: true
    });
    expect(chosen.problems[0].message).toContain('count');
    // Automatic acceptance of the same column is not acknowledged: it stays unready.
    const only = createKeplerCuratedAnalysisMapping(areas, [
      makeProfile('tracts', 'polygons', [makeColumn('n_cases', 'integer', 'count')])
    ]);
    expect(only.ready).toBe(false);
    expect(only.problems[0].acknowledged).toBe(false);
  });

  it('does not turn the incident hot spot map into a map of loss amounts by itself', () => {
    const hotSpot = getAnalysis('hot-spot-map');
    const incidents = makeProfile(
      'incidents',
      'points',
      [
        makeColumn('incident_id', 'text', 'identifier'),
        makeColumn('loss_amount', 'number', 'amount')
      ],
      'events'
    );
    const mapping = createKeplerCuratedAnalysisMapping(hotSpot, [incidents]);
    const role = mapping.slots[0].roles[0];
    expect(role.columns).toEqual([]);
    expect(role.candidates.map(candidate => candidate.name)).toEqual(['loss_amount']);
    expect(role.reason).toContain('changes what');
    expect(mapping.ready).toBe(true);
    expect(buildKeplerCuratedAnalysisRequest(hotSpot, mapping).parameters).not.toHaveProperty(
      'valueProperty'
    );
  });
});

// ---------------------------------------------------------------------------------------------
// Fix round: roles required by a parameter, with the real analyses
// ---------------------------------------------------------------------------------------------

/**
 * The real analysis with its value role required while `statistic` is not `count`, exactly as the
 * binding requires it. When the analyses already declare that, the copy is the analysis itself.
 */
function withStatisticRequiredValue(
  analysis: KeplerCuratedAnalysis,
  slotName: string
): KeplerCuratedAnalysis {
  return {
    ...analysis,
    slots: analysis.slots.map(slot =>
      slot.name !== slotName
        ? slot
        : {
            ...slot,
            fields: slot.fields.map(role =>
              role.role === 'value' && typeof role.required === 'boolean'
                ? {
                    ...role,
                    required: {
                      whenParameter: 'statistic',
                      equals: ['sum', 'mean', 'minimum', 'maximum']
                    }
                  }
                : role
            )
          }
    )
  };
}

describe('count-by-area and period-comparison with a statistic', () => {
  const textOnly = makeProfile(
    'incidents',
    'points',
    [makeColumn('label', 'text', 'nominal')],
    'events'
  );
  const withNumber = makeProfile(
    'incidents',
    'points',
    [makeColumn('label', 'text', 'nominal'), makeColumn('severity', 'number', 'amount')],
    'events'
  );
  const districts = makeProfile('districts', 'polygons', []);

  it('count-by-area: the value is required once the statistic is not count, and the user can change it', () => {
    const analysis = withStatisticRequiredValue(getAnalysis('count-by-area'), 'points');
    const counting = createKeplerCuratedAnalysisMapping(analysis, [textOnly, districts]);
    expect(counting.ready).toBe(true);
    expect(counting.slots[0].roles[0].required).toBe(false);
    expect(buildKeplerCuratedAnalysisRequest(analysis, counting).parameters).not.toHaveProperty(
      'valueProperty'
    );

    const meaning = createKeplerCuratedAnalysisMapping(analysis, [textOnly, districts], {
      parameters: {statistic: 'mean'}
    });
    expect(
      meaning.slots[0].candidates.find(candidate => candidate.datasetId === 'incidents')?.fit
    ).toBe('missing-columns');
    expect(meaning.complete).toBe(false);
    expect(() => buildKeplerCuratedAnalysisRequest(analysis, meaning)).toThrow(/not complete/);

    const mean = createKeplerCuratedAnalysisMapping(analysis, [withNumber, districts], {
      parameters: {statistic: 'mean'}
    });
    expect(mean.slots[0].roles[0].required).toBe(true);
    expect(mean.slots[0].roles[0].columns).toEqual(['severity']);
    expect(mean.progress.required).toBe(analysis.slots.length + 1);
    const request = buildKeplerCuratedAnalysisRequest(analysis, mean);
    expect(request.parameters).toMatchObject({statistic: 'mean', valueProperty: 'severity'});
    expect(() =>
      normalizeCatalogParameters(getCatalogBinding(request.bindingId)!, request.parameters)
    ).not.toThrow();

    // The user switches the statistic on an existing mapping through the options.
    const switched = updateKeplerCuratedAnalysisMapping(
      createKeplerCuratedAnalysisMapping(analysis, [withNumber, districts]),
      {slot: 'areas', datasetId: 'districts'},
      analysis,
      [withNumber, districts],
      {parameters: {statistic: 'sum'}}
    );
    expect(switched.slots[0].roles[0].required).toBe(true);
    expect(switched.slots[0].roles[0].columns).toEqual(['severity']);
    // Without options, the parameter values of the mapping carry over.
    const kept = updateKeplerCuratedAnalysisMapping(
      switched,
      {slot: 'points', role: 'value', columns: ['severity']},
      analysis,
      [withNumber, districts]
    );
    expect(kept.parameterValues).toMatchObject({statistic: 'sum'});
    // A request that sets the statistic without a value column is refused even for a count mapping.
    expect(() =>
      buildKeplerCuratedAnalysisRequest(analysis, counting, {
        parameterOverrides: {statistic: 'sum'}
      })
    ).toThrow(/Value.*has no column/);
  });

  it('period-comparison: the earlier value is required once the statistic is not count', () => {
    const analysis = withStatisticRequiredValue(getAnalysis('period-comparison'), 'before');
    const before = {...withNumber, datasetId: 'before', label: 'before'};
    const after = {...withNumber, datasetId: 'after', label: 'after'};
    const counting = createKeplerCuratedAnalysisMapping(analysis, [before, after]);
    expect(counting.slots[0].roles[0].required).toBe(false);
    const mean = createKeplerCuratedAnalysisMapping(analysis, [before, after], {
      parameters: {statistic: 'mean'}
    });
    expect(mean.slots[0].roles[0].required).toBe(true);
    expect(mean.slots[0].roles[0].columns).toEqual(['severity']);
    const bare = {...textOnly, datasetId: 'before', label: 'before'};
    const bareAfter = {...textOnly, datasetId: 'after', label: 'after'};
    const missing = createKeplerCuratedAnalysisMapping(analysis, [bare, bareAfter], {
      parameters: {statistic: 'mean'}
    });
    expect(missing.complete).toBe(false);
  });
});

// ---------------------------------------------------------------------------------------------
// Fix round: one column name read from two datasets
// ---------------------------------------------------------------------------------------------

/**
 * Period comparison as the binding reads it: one `valueProperty` name for both datasets. The real
 * analysis declares only the earlier role until its owner adds the later one, so the copy adds it
 * when it is absent.
 */
function withSharedValue(analysis: KeplerCuratedAnalysis): KeplerCuratedAnalysis {
  if (analysis.execution.status !== 'available') {
    throw new Error('test setup: needs an available analysis');
  }
  if (analysis.execution.fieldParameters['after.value'] !== undefined) {
    return analysis;
  }
  return {
    ...analysis,
    slots: analysis.slots.map(slot =>
      slot.name === 'after'
        ? {
            ...slot,
            fields: [
              {
                role: 'value',
                label: 'Value (optional)',
                accepts: ['number', 'integer'],
                measurement: ['count', 'amount'],
                required: {whenParameter: 'statistic', equals: ['sum', 'mean']}
              }
            ]
          }
        : slot
    ),
    execution: {
      ...analysis.execution,
      fieldParameters: {...analysis.execution.fieldParameters, 'after.value': 'valueProperty'}
    }
  } as KeplerCuratedAnalysis;
}

describe('two slots that read one column name', () => {
  const base = withStatisticRequiredValue(
    withSharedValue(getAnalysis('period-comparison')),
    'before'
  );
  const analysis = withStatisticRequiredValue(base, 'after');
  const incidents = (id: string, columns: readonly GPUAnalysisColumnProfile[]) =>
    makeProfile(id, 'points', columns, 'events', id);
  const options = {parameters: {statistic: 'sum'}};

  it('reports a problem naming both datasets when no column is shared', () => {
    const before = incidents('before', [makeColumn('amount', 'number', 'amount')]);
    const after = incidents('after', [makeColumn('total', 'number', 'amount')]);
    const mapping = createKeplerCuratedAnalysisMapping(analysis, [before, after], options);
    expect(mapping.complete).toBe(false);
    const problem = mapping.problems.find(entry => entry.kind === 'shared-column-missing');
    expect(problem?.severity).toBe('blocks');
    expect(problem?.message).toContain('before');
    expect(problem?.message).toContain('after');
    expect(problem?.message).toContain('valueProperty');
    expect(mapping.slots.flatMap(slot => slot.roles.map(role => role.columns))).toEqual([[], []]);
    // The mapping reports it once, not once more as an empty role.
    expect(mapping.problems.filter(entry => entry.kind === 'role-empty')).toEqual([]);
    expect(listKeplerCuratedAnalysesForDatasets([analysis], [before, after])[0].status).toBe(
      'ready'
    );
    expect(() => buildKeplerCuratedAnalysisRequest(analysis, mapping)).toThrow(/not complete/);
  });

  it('offers only columns both datasets have, with a compatible type, and fills both roles alike', () => {
    const before = incidents('before', [
      makeColumn('amount', 'number', 'amount'),
      makeColumn('total', 'number', 'amount'),
      makeColumn('note', 'number', 'amount'),
      makeColumn('only_before', 'number', 'amount')
    ]);
    const after = incidents('after', [
      makeColumn('total', 'integer', 'amount'),
      makeColumn('amount', 'number', 'amount'),
      makeColumn('note', 'text', 'nominal'),
      makeColumn('only_after', 'number', 'amount')
    ]);
    const mapping = createKeplerCuratedAnalysisMapping(analysis, [before, after], options);
    const [earlier, later] = mapping.slots.map(slot => slot.roles[0]);
    for (const role of [earlier, later]) {
      expect(role.candidates.map(candidate => candidate.name).sort()).toEqual(['amount', 'total']);
    }
    expect(earlier.sharedWith).toEqual(['after.value']);
    expect(later.sharedWith).toEqual(['before.value']);
    expect(later.columns).toEqual(earlier.columns);
    expect(earlier.columns).toHaveLength(1);
    // Two plausible columns: still a choice, for the pair.
    expect(earlier.state).toBe('choose');
    expect(later.state).toBe('choose');

    const chosen = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'after', role: 'value', columns: ['total']},
      analysis,
      [before, after],
      options
    );
    expect(chosen.slots.map(slot => slot.roles[0].columns)).toEqual([['total'], ['total']]);
    expect(chosen.slots.map(slot => slot.roles[0].source)).toEqual(['user', 'user']);
    expect(chosen.ready).toBe(true);
    expect(chosen.choices.columns).toEqual({'before.value': ['total'], 'after.value': ['total']});
    const request = buildKeplerCuratedAnalysisRequest(analysis, chosen, {
      parameterOverrides: {statistic: 'sum'}
    });
    expect(request.parameters['valueProperty']).toBe('total');
    expect(() =>
      updateKeplerCuratedAnalysisMapping(
        chosen,
        {slot: 'before', role: 'value', columns: ['only_before']},
        analysis,
        [before, after],
        options
      )
    ).toThrow(/not shared with after\.value/);
    expect(() =>
      updateKeplerCuratedAnalysisMapping(
        chosen,
        {slot: 'before', role: 'value', columns: ['note']},
        analysis,
        [before, after],
        options
      )
    ).toThrow(/not shared with after\.value/);

    // Changing a dataset drops the shared choice on both sides.
    const moved = updateKeplerCuratedAnalysisMapping(
      chosen,
      {slot: 'after', datasetId: 'before'},
      analysis,
      [before, after],
      options
    );
    expect(moved.choices.columns).toEqual({});
  });

  it('is not required while the statistic is count, so nothing is reported', () => {
    const before = incidents('before', [makeColumn('amount', 'number', 'amount')]);
    const after = incidents('after', [makeColumn('total', 'number', 'amount')]);
    const mapping = createKeplerCuratedAnalysisMapping(analysis, [before, after]);
    expect(mapping.complete).toBe(true);
    expect(mapping.problems.filter(entry => entry.kind === 'shared-column-missing')).toEqual([]);
  });

  it('refuses a request whose shared roles hold different columns', () => {
    const before = incidents('before', [makeColumn('total', 'number', 'amount')]);
    const after = incidents('after', [makeColumn('total', 'number', 'amount')]);
    const mapping = createKeplerCuratedAnalysisMapping(analysis, [before, after], options);
    const forged = {
      ...mapping,
      slots: mapping.slots.map(slot =>
        slot.slot === 'after'
          ? {...slot, roles: slot.roles.map(role => ({...role, columns: ['other']}))}
          : slot
      )
    };
    expect(() => buildKeplerCuratedAnalysisRequest(analysis, forged)).toThrow(
      /both feed parameter "valueProperty" but hold different columns/
    );
  });
});

// ---------------------------------------------------------------------------------------------
// Fix round: robustness
// ---------------------------------------------------------------------------------------------

describe('input that is not what the types promise', () => {
  const idw = getAnalysis('idw-surface');
  const good = makeProfile(
    'gauges',
    'points',
    [makeColumn('rainfall', 'number', 'amount')],
    'surface-samples'
  );

  it('names the function and the member of a malformed profile', () => {
    const noLabel = {...good} as Partial<KeplerDatasetProfile>;
    delete noLabel.label;
    const noRows = {...good} as Partial<KeplerDatasetProfile>;
    delete noRows.rowCount;
    const cases: [unknown, RegExp][] = [
      [noLabel, /createKeplerCuratedAnalysisMapping: datasets\[1\]\.label must be a string/],
      [noRows, /datasets\[1\]\.rowCount must be a finite number/],
      [{...good, rowCount: Number.NaN}, /datasets\[1\]\.rowCount/],
      [{...good, datasetId: ''}, /datasets\[1\]\.datasetId must be a non-empty string/],
      [{...good, columns: undefined}, /datasets\[1\]\.columns must be an array/],
      [
        {...good, columns: [{type: 'number'}]},
        /datasets\[1\]\.columns\[0\] must have a string name/
      ],
      [null, /datasets\[1\] must be a dataset profile object; got null/]
    ];
    for (const [bad, message] of cases) {
      expect(() =>
        createKeplerCuratedAnalysisMapping(idw, [good, bad as KeplerDatasetProfile])
      ).toThrow(message);
    }
    expect(() =>
      listKeplerCuratedAnalysesForDatasets(KEPLER_CURATED_ANALYSES, [
        {...good, label: undefined as never}
      ])
    ).toThrow(/datasets\[0\]\.label/);
  });

  it('rejects columns given as a string, not an array of characters', () => {
    const mapping = createKeplerCuratedAnalysisMapping(idw, [good]);
    expect(() =>
      updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'points', role: 'value', columns: 'rainfall' as never},
        idw,
        [good]
      )
    ).toThrow(
      /updateKeplerCuratedAnalysisMapping.*points\.value.*columns must be an array of column names; got string/
    );
    expect(() =>
      updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'points', role: 'value', columns: [3] as never},
        idw,
        [good]
      )
    ).toThrow(/array of column names/);
  });

  it('drops keys of slots and roles that no longer exist and repeated columns when it replays, then saves only what holds', () => {
    const choices = {
      analysisId: idw.id,
      datasets: {points: 'gauges', gone: 'x'},
      columns: {
        'points.value': ['rainfall', 'rainfall'],
        'points.removed': ['rainfall'],
        'gone.value': ['x']
      }
    };
    const replayed = replayKeplerCuratedAnalysisMapping(idw, choices, [good]);
    const kinds = replayed.problems.map(problem => problem.kind);
    expect(kinds.filter(kind => kind === 'saved-choice-lost')).toHaveLength(4);
    expect(replayed.slots[0].roles[0].columns).toEqual(['rainfall']);
    expect(replayed.slots[0].roles[0].state).toBe('choose');
    expect(replayed.choices).toEqual({
      analysisId: idw.id,
      datasets: {points: 'gauges'},
      columns: {}
    });
    expect(JSON.parse(serializeKeplerCuratedAnalysisMapping(replayed))).toEqual({
      version: 1,
      analysisId: idw.id,
      datasets: {points: 'gauges'},
      columns: {}
    });
  });

  it('parses without silently dropping a __proto__ key and rejects an empty column list', () => {
    const base = '{"version":1,"analysisId":"x","datasets":{},"columns":{}}';
    expect(() =>
      parseKeplerCuratedAnalysisMapping(
        '{"version":1,"analysisId":"x","datasets":{"__proto__":"d"},"columns":{}}'
      )
    ).toThrow(/parseKeplerCuratedAnalysisMapping: datasets key "__proto__" is not allowed/);
    expect(() =>
      parseKeplerCuratedAnalysisMapping(
        '{"version":1,"analysisId":"x","datasets":{},"columns":{"__proto__.v":["c"]}}'
      )
    ).toThrow(/columns key "__proto__.v" is not allowed/);
    expect(() =>
      parseKeplerCuratedAnalysisMapping(
        '{"version":1,"analysisId":"x","datasets":{},"columns":{"a.v":[]}}'
      )
    ).toThrow(/columns\["a\.v"\] is empty/);
    expect(() =>
      parseKeplerCuratedAnalysisMapping(
        '{"version":1,"analysisId":"x","datasets":{},"columns":{"a.v":[""]}}'
      )
    ).toThrow(/non-empty column names/);
    expect(parseKeplerCuratedAnalysisMapping(base)).toEqual({
      analysisId: 'x',
      datasets: {},
      columns: {}
    });
    expect(Object.getPrototypeOf(parseKeplerCuratedAnalysisMapping(base).datasets)).toBe(
      Object.prototype
    );
  });
});

// ---------------------------------------------------------------------------------------------
// Fix round: the data's coordinate model
// ---------------------------------------------------------------------------------------------

describe('coordinate model of the data', () => {
  const idw = getAnalysis('idw-surface');
  // The preset the analyses carry today; the copy makes these tests independent of its removal.
  const presetCopy: KeplerCuratedAnalysis = {
    ...idw,
    execution: {
      ...(idw.execution as Extract<KeplerCuratedAnalysis['execution'], {status: 'available'}>),
      parameters: {power: 2, coordinateSystem: 'EPSG:4326'}
    }
  };
  const gauges = (
    model: 'geographic' | 'planar',
    source: 'declared' | 'guess'
  ): KeplerDatasetProfile => ({
    ...makeProfile(
      'gauges',
      'points',
      [makeColumn('rainfall', 'number', 'amount')],
      'surface-samples'
    ),
    coordinates: {
      model,
      source,
      reason: source === 'guess' ? 'extent looks like metres' : undefined
    }
  });

  it('shows the model of each slot and whether it is a guess', () => {
    const planar = createKeplerCuratedAnalysisMapping(presetCopy, [gauges('planar', 'declared')]);
    expect(planar.slots[0].coordinates).toMatchObject({model: 'planar', source: 'declared'});
    expect(planar.problems).toEqual([]);
    const guessed = createKeplerCuratedAnalysisMapping(presetCopy, [gauges('planar', 'guess')]);
    expect(guessed.slots[0].coordinates).toMatchObject({model: 'planar', source: 'guess'});
    expect(guessed.problems).toMatchObject([
      {kind: 'coordinate-guess', severity: 'check', slot: 'points'}
    ]);
    expect(guessed.problems[0].message).toContain('extent looks like metres');
    expect(guessed.ready).toBe(true);
    const none = createKeplerCuratedAnalysisMapping(presetCopy, [
      makeProfile(
        'gauges',
        'points',
        [makeColumn('rainfall', 'number', 'amount')],
        'surface-samples'
      )
    ]);
    expect(none.slots[0].coordinates).toBeUndefined();
  });

  it('retains explicit mapping coordinate settings even when equal to an inherited preset', () => {
    const datasets = [gauges('planar', 'declared')];
    for (const coordinateSystem of ['EPSG:4326', 'planar']) {
      const mapping = createKeplerCuratedAnalysisMapping(presetCopy, datasets, {
        parameters: {coordinateSystem}
      });
      expect(mapping.ready).toBe(true);
      expect(
        buildKeplerCuratedAnalysisRequest(presetCopy, mapping).parameters['coordinateSystem']
      ).toBe(coordinateSystem);
      const updated = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'points', datasetId: 'gauges'},
        presetCopy,
        datasets
      );
      expect(
        buildKeplerCuratedAnalysisRequest(presetCopy, updated).parameters['coordinateSystem']
      ).toBe(coordinateSystem);
      const replayed = replayKeplerCuratedAnalysisMapping(presetCopy, mapping.choices, datasets, {
        parameters: {coordinateSystem}
      });
      expect(
        buildKeplerCuratedAnalysisRequest(presetCopy, replayed).parameters['coordinateSystem']
      ).toBe(coordinateSystem);
      const serialized: KeplerCuratedAnalysisMapping = JSON.parse(JSON.stringify(mapping));
      expect(
        buildKeplerCuratedAnalysisRequest(presetCopy, serialized).parameters['coordinateSystem']
      ).toBe(coordinateSystem);
      expect(
        buildKeplerCuratedAnalysisRequest(presetCopy, mapping, {
          parameterOverrides: {coordinateSystem: 'planar'}
        }).parameters['coordinateSystem']
      ).toBe('planar');
    }
  });

  it('leaves coordinateSystem to the defaults module for planar data, and keeps an explicit override', () => {
    const planar = createKeplerCuratedAnalysisMapping(presetCopy, [gauges('planar', 'declared')]);
    expect(buildKeplerCuratedAnalysisRequest(presetCopy, planar).parameters).not.toHaveProperty(
      'coordinateSystem'
    );
    expect(
      buildKeplerCuratedAnalysisRequest(presetCopy, planar, {
        parameterOverrides: {coordinateSystem: 'planar'}
      }).parameters['coordinateSystem']
    ).toBe('planar');
    const geographic = createKeplerCuratedAnalysisMapping(presetCopy, [
      gauges('geographic', 'declared')
    ]);
    expect(
      buildKeplerCuratedAnalysisRequest(presetCopy, geographic).parameters['coordinateSystem']
    ).toBe('EPSG:4326');
  });
});

// ---------------------------------------------------------------------------------------------
// Fix round: agreement with the descriptor
// ---------------------------------------------------------------------------------------------

describe('validateKeplerCuratedAnalysis', () => {
  type Available = Extract<KeplerCuratedAnalysis['execution'], {status: 'available'}>;

  /** An analysis that says exactly what the descriptor says. */
  function mirrorDescriptor(analysis: KeplerCuratedAnalysis): KeplerCuratedAnalysis {
    const execution = analysis.execution as Available;
    const descriptor = getKeplerOperationDescriptor(execution.bindingId)!;
    return {
      ...analysis,
      question: descriptor.questions[0],
      exposedParameters: descriptor.parameters
        .filter(
          parameter =>
            parameter.level === 'basic' &&
            parameter.name !== 'coordinateSystem' &&
            evaluateGPUAnalysisParameterCondition(parameter.activeWhen, {
              ...Object.fromEntries(
                getCatalogBinding(execution.bindingId)!.parameters.map(entry => [
                  entry.name,
                  entry.defaultValue
                ])
              ),
              ...execution.parameters
            })
        )
        .map(parameter => parameter.name),
      slots: analysis.slots.map(slot => {
        const input = descriptor.inputs.find(entry => entry.name === execution.inputs[slot.name])!;
        const geometry =
          slot.kind === 'polygons' ? 'polygon' : slot.kind === 'points' ? 'point' : 'line';
        return {
          ...slot,
          kind: input.kind,
          rowSemantics: input.rowSemantics,
          fields: input.fields
            .filter(field =>
              slot.fields.some(
                role => execution.fieldParameters[`${slot.name}.${role.role}`] === field.parameter
              )
            )
            .map(({parameter: _parameter, measurementByGeometry, ...field}) => ({
              ...field,
              ...(measurementByGeometry?.[geometry] === undefined
                ? {}
                : {measurement: measurementByGeometry[geometry]})
            }))
        };
      })
    };
  }

  it('finds no disagreement between any available analysis and its descriptor', () => {
    for (const analysis of AVAILABLE_ANALYSES) {
      expect(validateKeplerCuratedAnalysis(analysis), analysis.id).toEqual([]);
    }
    const pending: KeplerCuratedAnalysis = {
      ...getAnalysis('density-surface'),
      exposedParameters: [],
      execution: {
        status: 'needs-binding',
        implementation: 'addFutureRecipe',
        reason: 'No binding has been registered for this future recipe.'
      }
    };
    expect(validateKeplerCuratedAnalysis(pending)).toEqual([]);
  });

  it('ignores coordinateSystem and parameters that are inactive under the presets, and reports exposing one', () => {
    const mirror = mirrorDescriptor(getAnalysis('cluster-and-outlier-map'));
    const execution = mirror.execution as Available;
    const descriptor = getKeplerOperationDescriptor(execution.bindingId)!;
    const inactive = descriptor.parameters.find(
      parameter =>
        parameter.activeWhen !== undefined &&
        !evaluateGPUAnalysisParameterCondition(parameter.activeWhen, {
          ...Object.fromEntries(
            getCatalogBinding(execution.bindingId)!.parameters.map(entry => [
              entry.name,
              entry.defaultValue
            ])
          ),
          ...execution.parameters
        })
    );
    expect(inactive).toBeDefined();
    // The mirror lists the basic ones; it must not list the inactive one or coordinateSystem.
    expect(mirror.exposedParameters).not.toContain('coordinateSystem');
    expect(validateKeplerCuratedAnalysis(mirror)).toEqual([]);
    expect(
      validateKeplerCuratedAnalysis({
        ...mirror,
        exposedParameters: [...mirror.exposedParameters, inactive!.name]
      })
    ).toEqual([
      expect.stringMatching(
        inactive!.level === 'basic' ? /inactive under the presets/ : /is advanced in the descriptor/
      )
    ]);
  });

  it('finds no disagreement in an analysis written from its descriptor, and one for every change', () => {
    const source = getAnalysis('cluster-and-outlier-map');
    const mirror = mirrorDescriptor(source);
    expect(validateKeplerCuratedAnalysis(mirror)).toEqual([]);

    const slot = mirror.slots[0];
    const role = slot.fields[0];
    const change = (replace: Partial<typeof role>): KeplerCuratedAnalysis => ({
      ...mirror,
      slots: [{...slot, fields: [{...role, ...replace}]}]
    });
    expect(validateKeplerCuratedAnalysis(change({accepts: ['category']}))).toEqual([
      expect.stringContaining('role "areas.value" accepts category but the descriptor accepts')
    ]);
    expect(validateKeplerCuratedAnalysis(change({required: false}))).toEqual([
      expect.stringContaining('role "areas.value" is required false but the descriptor says true')
    ]);
    // `measurementByGeometry` makes polygons rates only, so amounts are not allowed.
    expect(validateKeplerCuratedAnalysis(change({measurement: ['rate', 'amount']}))).toEqual([
      expect.stringContaining('not allowed: amount')
    ]);
    expect(validateKeplerCuratedAnalysis(change({measurement: undefined}))).toEqual([
      expect.stringContaining('not allowed: (any level)')
    ]);
    expect(validateKeplerCuratedAnalysis(change({multiple: {minimum: 1, maximum: 2}}))).toEqual([
      expect.stringContaining('takes {"minimum":1,"maximum":2} but the descriptor says one column')
    ]);
    expect(validateKeplerCuratedAnalysis({...mirror, exposedParameters: ['nope']})).toEqual(
      expect.arrayContaining([
        expect.stringContaining('exposed parameter "nope" has no entry in the descriptor'),
        expect.stringContaining('descriptor basic parameters not exposed')
      ])
    );
    expect(
      validateKeplerCuratedAnalysis({...mirror, slots: [{...slot, rowSemantics: ['events']}]})
    ).toEqual([expect.stringContaining('slot "areas" expects events rows')]);
    expect(validateKeplerCuratedAnalysis({...mirror, slots: [{...slot, fields: []}]})).toEqual([
      expect.stringContaining(
        'descriptor role "value" (parameter "valueProperty") of input "features" is not offered'
      )
    ]);
    const execution = mirror.execution as Available;
    expect(
      validateKeplerCuratedAnalysis({
        ...mirror,
        execution: {...execution, fieldParameters: {'areas.value': 'otherProperty'}}
      })
    ).toEqual([
      expect.stringContaining('feeds "otherProperty"'),
      expect.stringContaining('is not offered by slot')
    ]);
  });

  it('matches columns with the descriptor measurement for the slot geometry', () => {
    const areas = getAnalysis('cluster-and-outlier-map');
    const profile = makeProfile('tracts', 'polygons', [
      makeColumn('total_amount', 'number', 'amount'),
      makeColumn('share', 'number', 'rate')
    ]);
    const mapping = createKeplerCuratedAnalysisMapping(areas, [profile]);
    const role = mapping.slots[0].roles[0];
    // The descriptor lists rates only for polygons, whatever the analysis's own list says.
    expect(role.columns).toEqual(['share']);
    expect(role.candidates.map(candidate => [candidate.name, candidate.fit])).toEqual([
      ['share', 'preferred'],
      ['total_amount', 'accepted']
    ]);
    expect(role.candidates[1].warning).toContain('meant for rates');
  });
});

// ---------------------------------------------------------------------------------------------
// Fix round: the whole chain on real profiles
// ---------------------------------------------------------------------------------------------

type SampleCollections = Record<string, CatalogFeatureCollection>;

function renameProperties(collection: CatalogFeatureCollection): CatalogFeatureCollection {
  return {
    ...collection,
    features: collection.features.map(feature => ({
      ...feature,
      properties: Object.fromEntries(
        Object.entries(feature.properties ?? {}).map(([name, value]) => [`renamed_${name}`, value])
      )
    }))
  } as CatalogFeatureCollection;
}

/** Profiles of every slot's sample (geographic coordinates declared), with the ids the slots use. */
function profileSample(
  analysis: KeplerCuratedAnalysis,
  collections: SampleCollections,
  labelByName: boolean
): {profiles: KeplerDatasetProfile[]; collectionsById: Record<string, CatalogFeatureCollection>} {
  const profiles: KeplerDatasetProfile[] = [];
  const collectionsById: Record<string, CatalogFeatureCollection> = {};
  analysis.slots.forEach((slot, index) => {
    const id = labelByName ? slot.name : `dataset-${index + 1}`;
    profiles.push(
      profileKeplerFeatureCollection(collections[slot.name], {
        datasetId: id,
        label: id,
        coordinateSystem: 'EPSG:4326'
      })
    );
    collectionsById[id] = collections[slot.name];
  });
  return {profiles, collectionsById};
}

async function runRequest(
  request: ReturnType<typeof buildKeplerCuratedAnalysisRequest>,
  profiles: readonly KeplerDatasetProfile[],
  collectionsById: Record<string, CatalogFeatureCollection>
) {
  const binding = getCatalogBinding(request.bindingId)!;
  const descriptor = getKeplerOperationDescriptor(request.bindingId)!;
  const profilesByInput = Object.fromEntries(
    Object.entries(request.inputDatasetIds).map(([input, datasetId]) => [
      input,
      profiles.find(profile => profile.datasetId === datasetId)
    ])
  );
  const resolved = resolveKeplerOperationParameters(binding, descriptor, profilesByInput, {
    overrides: request.parameters
  });
  const inputs = Object.fromEntries(
    Object.entries(request.inputDatasetIds).map(([input, datasetId]) => [
      input,
      collectionsById[datasetId]
    ])
  );
  return runCatalogOperationAsync(binding, resolved.parameters, inputs);
}

describe('from a real profile to a finished run', () => {
  // Typed GPU recipes have mapped execution checks in their dedicated headless specs.
  for (const analysis of AVAILABLE_ANALYSES.filter(
    analysis =>
      analysis.execution.status === 'available' &&
      getCatalogBinding(analysis.execution.bindingId)?.engine !== 'gpu'
  )) {
    for (const variant of ['sample columns', 'renamed columns'] as const) {
      it(`${analysis.id} (${variant}) builds, resolves its parameters and runs`, async () => {
        const sample = createKeplerCuratedAnalysisSample(analysis.sampleId!) as SampleCollections;
        const collections: SampleCollections =
          variant === 'sample columns'
            ? sample
            : Object.fromEntries(
                Object.entries(sample).map(([name, collection]) => [
                  name,
                  renameProperties(collection)
                ])
              );
        const {profiles, collectionsById} = profileSample(analysis, collections, true);
        const mapping = createKeplerCuratedAnalysisMapping(analysis, profiles);
        expect(mapping.complete).toBe(true);
        expect(mapping.slots.every(slot => slot.datasetId === slot.slot)).toBe(true);
        const request = buildKeplerCuratedAnalysisRequest(analysis, mapping);
        if (getCatalogBinding(request.bindingId)!.engine === 'gpu') {
          // Dedicated headless specs verify both mappings against the real recipe fixture.
          await expect(runRequest(request, profiles, collectionsById)).rejects.toThrow(
            'GPU device required'
          );
          return;
        }
        const result = await runRequest(request, profiles, collectionsById);
        expect(result.diagnostics.filter(diagnostic => diagnostic.severity === 'error')).toEqual(
          []
        );
        expect(result.tables.length).toBeGreaterThan(0);
        expect(result.tables.some(table => table.rowCount > 0)).toBe(true);
      }, 120000);
    }
  }

  it('regression runs with two and with five predictors chosen through the mapping', async () => {
    const regression = getAnalysis('regression-with-diagnostics');
    const sample = createKeplerCuratedAnalysisSample(regression.sampleId!) as SampleCollections;
    // Three more numeric columns, independent of the planted ones, so five predictors are not collinear.
    const widened: CatalogFeatureCollection = {
      ...sample['areas'],
      features: sample['areas'].features.map((feature, index) => ({
        ...feature,
        properties: {
          ...feature.properties,
          age: 30 + 10 * Math.sin(index * 0.7) + (index % 7),
          rent: 900 + 200 * Math.cos(index * 1.3) + (index % 5) * 15,
          jobs: 50 + 20 * Math.sin(index * 2.1 + 1) + (index % 3) * 4
        }
      }))
    } as CatalogFeatureCollection;
    const {profiles, collectionsById} = profileSample(regression, {areas: widened}, true);
    for (const predictors of [
      ['income', 'density'],
      ['income', 'density', 'age', 'rent', 'jobs']
    ]) {
      let mapping = createKeplerCuratedAnalysisMapping(regression, profiles);
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'areas', role: 'response', columns: ['outcome']},
        regression,
        profiles
      );
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'areas', role: 'predictors', columns: predictors},
        regression,
        profiles
      );
      const request = buildKeplerCuratedAnalysisRequest(regression, mapping);
      expect(request.parameters['predictorProperties']).toBe(predictors.join(','));
      const result = await runRequest(request, profiles, collectionsById);
      expect(result.diagnostics.filter(diagnostic => diagnostic.severity === 'error')).toEqual([]);
      const coefficients = result.tables.find(table => table.id === 'coefficients')!;
      // One row per predictor plus the intercept.
      expect(coefficients.rowCount).toBe(predictors.length + 1);
    }
  }, 120000);
});

describe('curated network sample slots', () => {
  it('maps facilities to origins and demand to opportunities regardless of dataset order', () => {
    const analysis = getAnalysis('drive-time-catchments');
    const sample = createKeplerCuratedAnalysisSample('street-grid-facilities');
    const profiles = Object.entries(sample).map(([slot, collection]) =>
      profileKeplerFeatureCollection(collection, {
        datasetId: slot,
        label: slot,
        coordinateSystem: 'EPSG:4326'
      })
    );
    for (const datasets of [profiles, [...profiles].reverse()]) {
      let mapping = createKeplerCuratedAnalysisMapping(analysis, datasets);
      // Curated slot names carry the sample's meaning before any user reassignment.
      expect(Object.fromEntries(mapping.slots.map(slot => [slot.slot, slot.datasetId]))).toEqual({
        network: 'network',
        facilities: 'facilities',
        demand: 'demand'
      });
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'demand', role: 'weight', columns: ['residents']},
        analysis,
        datasets
      );
      const request = buildKeplerCuratedAnalysisRequest(analysis, mapping);
      expect(request.inputDatasetIds).toEqual({
        lines: 'network',
        origins: 'facilities',
        opportunities: 'demand'
      });
      expect(request.parameters['weightProperty']).toBe('residents');
    }
  });
});

it('requires a mapped role when either of two different parameter conditions holds', () => {
  const analysis: KeplerCuratedAnalysis = {
    ...SYNTHETIC_ANALYSIS,
    slots: SYNTHETIC_ANALYSIS.slots.map(slot => ({
      ...slot,
      fields: slot.fields.map(role =>
        role.role === 'weight'
          ? {
              ...role,
              required: {
                anyOf: [
                  {whenParameter: 'mode', equals: ['weighted']},
                  {parameter: 'useWeight', equals: [true]}
                ]
              }
            }
          : role
      )
    }))
  };
  const dataset = makeProfile('areas', 'polygons', [
    makeColumn('n_cases', 'integer', 'count'),
    makeColumn('pop_total', 'integer', 'count')
  ]);
  for (const [mode, useWeight, required] of [
    ['plain', false, false],
    ['weighted', false, true],
    ['plain', true, true]
  ] as const) {
    const mapping = createKeplerCuratedAnalysisMapping(analysis, [dataset], {
      parameters: {mode, useWeight}
    });
    expect(mapping.slots[0].roles.find(role => role.role === 'weight')?.required).toBe(required);
    expect(mapping.complete).toBe(!required);
  }
});

it('maps native polygon-backed cells through shared compatibility while retaining polygon slot restrictions', () => {
  const profile = profileKeplerNativeDataset(
    {
      rowCount: 4,
      geometryColumn: 'geometry',
      columns: [
        {name: 'rate', type: 'float64', values: [1, 2, 3, 4]},
        {
          name: 'geometry',
          type: 'string',
          values: [0, 1, 2, 3].map(column =>
            JSON.stringify({
              type: 'Polygon',
              coordinates: [
                [
                  [column, 0],
                  [column + 1, 0],
                  [column + 1, 1],
                  [column, 1],
                  [column, 0]
                ]
              ]
            })
          )
        }
      ]
    },
    {
      datasetId: 'native-cells',
      label: 'Native cells',
      declarations: {
        coordinateModel: 'planar',
        rowSemantics: 'objects',
        measurement: {rate: 'rate'}
      }
    },
    {kind: 'cells'}
  );
  expect(profile).toMatchObject({kind: 'cells', catalogGeometry: 'polygon'});
  const analysis = getAnalysis('cluster-and-outlier-map');
  const binding = getCatalogBinding('local-moran-statistics')!;
  const applicability = evaluateKeplerOperationApplicability(
    binding,
    getKeplerOperationDescriptor(binding.id)!,
    [profile],
    {executionEngine: 'cpu'}
  );
  expect(applicability.status).toBe('ready');
  const mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
  expect(mapping.slots[0].candidates[0].fit).toBe('fits');
  expect(mapping.slots[0].datasetId).toBe(profile.datasetId);
  expect(mapping.ready).toBe(true);
  for (const catalogGeometry of ['point', 'line'] as const) {
    const incompatible = createKeplerCuratedAnalysisMapping(analysis, [
      {...profile, catalogGeometry}
    ]);
    expect(incompatible.slots[0].candidates[0].fit).toBe('wrong-kind');
  }
});
