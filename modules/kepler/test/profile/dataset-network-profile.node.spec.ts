// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {evaluateGPUAnalysisRequirement} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {describe, expect, it, vi} from 'vitest';
import {
  getNetworkParameters,
  type CatalogParameterValue
} from '../../src/analysis/catalog-binding-types';
import {getCatalogBinding, getCatalogDefaultParameters} from '../../src/analysis/catalog-bindings';
import type {CatalogFeature, CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {prepareCuratedDriveTimeBands} from '../../src/curated/bindings/curated-drive-time-bands-binding';
import {
  createDriveTimeBandsSecondDataset,
  DRIVE_TIME_BANDS_SECOND_SAMPLE
} from '../../src/curated/samples/drive-time-bands-data';
import {buildNetworkGraph} from '../../src/analysis/catalog-network';
import {runSnapPoints} from '../../src/analysis/catalog-snap-binding';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import {NETWORK_OPERATION_DESCRIPTORS} from '../../src/descriptors/descriptors-network';
import {
  getKeplerDatasetNetworkProfile,
  getKeplerDatasetNetworkTopologyKey,
  profileKeplerDatasetNetwork
} from '../../src/profile/dataset-network-profile';
import {
  applyKeplerDatasetProfileDeclarations,
  getKeplerDatasetProfileUndeclared,
  profileKeplerFeatureCollection
} from '../../src/profile/dataset-profile';
import {createKeplerDatasetProfileCache} from '../../src/profile/dataset-profile-cache';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';

type Parameters = Record<string, CatalogParameterValue>;
const PARAMETERS: Parameters = {
  coordinateSystem: 'planar',
  snapTolerance: 0,
  splitAtSharedVertices: true,
  costMode: 'length',
  forwardCostProperty: '',
  reverseCostProperty: '',
  onewayProperty: '',
  crossings: 'not-noded'
};
function line(coordinates: number[][], properties: Record<string, unknown> = {}): CatalogFeature {
  return {geometry: {type: 'LineString', coordinates}, properties};
}
function collection(features: CatalogFeature[]): CatalogFeatureCollection {
  return {type: 'FeatureCollection', features};
}
function profile(revision = 1): KeplerDatasetProfile {
  return {
    datasetId: 'streets',
    label: 'Streets',
    revision,
    kind: 'lines',
    catalogGeometry: 'line',
    geometryColumns: [],
    rowCount: 1,
    columns: []
  };
}
const SHARED = collection([
  line([
    [0, 0],
    [1, 0],
    [1, 0],
    [2, 0]
  ]),
  line([
    [1, 0],
    [1, 1]
  ])
]);
function assess(bindingId: string, source: KeplerDatasetProfile, overrides: Parameters = {}) {
  const binding = getCatalogBinding(bindingId)!;
  const descriptor = NETWORK_OPERATION_DESCRIPTORS.find(
    candidate => candidate.bindingId === bindingId
  )!;
  return evaluateKeplerOperationApplicability(binding, descriptor, [source], {
    executionEngine: 'cpu',
    overrides: {...PARAMETERS, ...overrides}
  });
}

describe('topology-specific dataset network facts', () => {
  it('matches drive-time bend segments with its fixed vertex preparation', () => {
    const inputs = createDriveTimeBandsSecondDataset();
    const parameters = {
      ...DRIVE_TIME_BANDS_SECOND_SAMPLE.parameters,
      costBreaks: [...DRIVE_TIME_BANDS_SECOND_SAMPLE.parameters.costBreaks]
    };
    const prepared = prepareCuratedDriveTimeBands(inputs, parameters);
    const facts = profileKeplerDatasetNetwork(
      profile(),
      inputs['network'],
      parameters,
      'gpu',
      'curated-drive-time-bands'
    )!;
    expect(facts.nodeCount).toBe(prepared.graph.nodes.length);
    expect(facts.arcCount).toBe(prepared.topology.weights.length);
    expect(facts.edgeCount).toBe(3);
    expect(facts.topologyKey).not.toBe(
      getKeplerDatasetNetworkTopologyKey(
        {...parameters, snapTolerance: 0, splitAtSharedVertices: true, costMode: 'property'},
        'gpu'
      )
    );
  });

  it.each([true, false])('matches shared and duplicate vertices with splitting=%s', split => {
    const parameters = {...PARAMETERS, splitAtSharedVertices: split};
    const facts = profileKeplerDatasetNetwork(profile(), SHARED, parameters)!;
    expect(facts).toMatchObject({nodeCount: 4, edgeCount: split ? 3 : 2, arcCount: split ? 6 : 4});
    const graph = buildNetworkGraph(SHARED, getNetworkParameters(parameters), 100);
    expect([facts.nodeCount, facts.edgeCount, facts.arcCount]).toEqual([
      graph.nodes.length,
      graph.edges.length,
      graph.outgoing.flat().length
    ]);
  });
  it('counts multipart and duplicate physical edges separately', () => {
    const lines = collection([
      {
        geometry: {
          type: 'MultiLineString',
          coordinates: [
            [
              [0, 0],
              [1, 0]
            ],
            [
              [1, 0],
              [2, 0]
            ]
          ]
        },
        properties: {}
      },
      line([
        [0, 0],
        [1, 0]
      ])
    ]);
    expect(profileKeplerDatasetNetwork(profile(), lines, PARAMETERS)).toMatchObject({
      nodeCount: 3,
      edgeCount: 3,
      arcCount: 6
    });
  });
  it('counts empty and degenerate networks exactly', () => {
    for (const lines of [
      collection([]),
      collection([
        line([
          [0, 0],
          [0, 0]
        ])
      ])
    ]) {
      expect(profileKeplerDatasetNetwork(profile(), lines, PARAMETERS)).toMatchObject({
        nodeCount: 0,
        edgeCount: 0,
        arcCount: 0
      });
    }
  });
  it('uses CRS-specific snapping units', () => {
    const lines = collection([
      line([
        [0, 0],
        [1, 0]
      ]),
      line([
        [1.00001, 0],
        [2, 0]
      ])
    ]);
    expect(profileKeplerDatasetNetwork(profile(), lines, PARAMETERS)?.nodeCount).toBe(4);
    expect(
      profileKeplerDatasetNetwork(profile(), lines, {...PARAMETERS, snapTolerance: 0.001})
        ?.nodeCount
    ).toBe(3);
    expect(
      profileKeplerDatasetNetwork(profile(), lines, {
        ...PARAMETERS,
        coordinateSystem: 'EPSG:4326',
        snapTolerance: 0.001
      })?.nodeCount
    ).toBe(4);
    expect(
      profileKeplerDatasetNetwork(profile(), lines, {
        ...PARAMETERS,
        coordinateSystem: 'EPSG:4326',
        snapTolerance: 2
      })?.nodeCount
    ).toBe(3);
  });
  it('distinguishes physical edges from one-way and missing-cost arcs', () => {
    const lines = collection([
      line(
        [
          [0, 0],
          [1, 0]
        ],
        {direction: 'forward', cost: 2}
      ),
      line(
        [
          [1, 0],
          [2, 0]
        ],
        {direction: 'reverse', cost: 4}
      ),
      line(
        [
          [2, 0],
          [3, 0]
        ],
        {cost: 8}
      ),
      line(
        [
          [3, 0],
          [4, 0]
        ],
        {cost: null}
      )
    ]);
    const parameters = {
      ...PARAMETERS,
      costMode: 'property',
      forwardCostProperty: 'cost',
      onewayProperty: 'direction'
    };
    expect(profileKeplerDatasetNetwork(profile(), lines, parameters)).toMatchObject({
      nodeCount: 5,
      edgeCount: 4,
      arcCount: 4
    });
    expect(
      profileKeplerDatasetNetwork(profile(), lines, {...parameters, reverseCostProperty: 'missing'})
    ).toMatchObject({edgeCount: 4, arcCount: 2});
    expect(() =>
      profileKeplerDatasetNetwork(
        profile(),
        collection([
          line(
            [
              [0, 0],
              [1, 0]
            ],
            {cost: -1}
          )
        ]),
        parameters
      )
    ).toThrow(/profileKeplerDatasetNetwork.*forwardCostProperty.*at least 0/);
  });
  it('keeps GPU crossing-noded facts unknown', () => {
    const parameters = {...PARAMETERS, crossings: 'noded'};
    const source = {
      ...profile(),
      network: profileKeplerDatasetNetwork(profile(), SHARED, parameters)
    };
    expect(getKeplerDatasetNetworkProfile(source, parameters, 'cpu')).toBeDefined();
    expect(getKeplerDatasetNetworkProfile(source, parameters, 'gpu')).toBeUndefined();
    expect(profileKeplerDatasetNetwork(profile(), SHARED, parameters, 'gpu')).toBeUndefined();
  });
  it('rejects facts from another dataset, revision or graph setting', () => {
    const source = {
      ...profile(),
      network: profileKeplerDatasetNetwork(profile(), SHARED, PARAMETERS)
    };
    expect(getKeplerDatasetNetworkProfile(source, PARAMETERS, 'gpu')).toBe(source.network);
    for (const changed of [{datasetId: 'other'}, {revision: 2}])
      expect(
        getKeplerDatasetNetworkProfile({...source, ...changed}, PARAMETERS, 'cpu')
      ).toBeUndefined();
    for (const changed of [
      {coordinateSystem: 'EPSG:4326'},
      {snapTolerance: 1},
      {splitAtSharedVertices: false},
      {costMode: 'property'},
      {onewayProperty: 'direction'}
    ] as Parameters[])
      expect(
        getKeplerDatasetNetworkProfile(source, {...PARAMETERS, ...changed}, 'cpu')
      ).toBeUndefined();
    const property = {...PARAMETERS, costMode: 'property', forwardCostProperty: 'cost'};
    expect(getKeplerDatasetNetworkTopologyKey(property, 'cpu')).not.toBe(
      getKeplerDatasetNetworkTopologyKey({...property, reverseCostProperty: 'returnCost'}, 'cpu')
    );
  });
  it('validates graph settings with named errors', () => {
    for (const changed of [
      {snapTolerance: -1},
      {snapTolerance: NaN},
      {coordinateSystem: 'unknown'},
      {splitAtSharedVertices: 'yes'},
      {costMode: 'unknown'}
    ] as Parameters[])
      expect(() =>
        profileKeplerDatasetNetwork(profile(), SHARED, {...PARAMETERS, ...changed})
      ).toThrow(/profileKeplerDatasetNetwork/);
  });
  it.each([
    'cpu',
    'gpu'
  ] as const)('keys snap facts by effective splitting and ignored crossings on %s', executionEngine => {
    const bindingId = 'network-snap-points';
    const parameters = {...PARAMETERS, splitAtSharedVertices: false, crossings: 'noded'};
    const cache = createKeplerDatasetProfileCache({maximumEntries: 1});
    cache.set(profile());
    const loadLines = vi.fn(() => SHARED);
    const refined = cache.profileNetwork(
      'streets',
      1,
      parameters,
      loadLines,
      executionEngine,
      bindingId
    )!;
    expect(refined.network).toMatchObject({nodeCount: 4, edgeCount: 3, arcCount: 6});
    expect(getKeplerDatasetNetworkProfile(refined, parameters, executionEngine, bindingId)).toBe(
      refined.network
    );
    expect(
      profileKeplerDatasetNetwork(profile(), SHARED, parameters, executionEngine, bindingId)
    ).toEqual(refined.network);
    expect(
      cache.profileNetwork('streets', 1, PARAMETERS, loadLines, executionEngine, bindingId)?.network
    ).toBe(refined.network);
    expect(loadLines).toHaveBeenCalledTimes(1);
  });
});

describe('lazy revision and topology cache', () => {
  it('loads only on request and reuses previously selected settings', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 2});
    const loadLines = vi.fn(() => SHARED);
    cache.set(profile());
    expect(cache.get('streets', 1)?.network).toBeUndefined();
    expect(loadLines).not.toHaveBeenCalled();
    const first = cache.profileNetwork('streets', 1, PARAMETERS, loadLines)?.network;
    expect(first?.edgeCount).toBe(3);
    expect(
      cache.profileNetwork('streets', 1, {...PARAMETERS, splitAtSharedVertices: false}, loadLines)
        ?.network?.edgeCount
    ).toBe(2);
    expect(cache.profileNetwork('streets', 1, PARAMETERS, loadLines)?.network).toBe(first);
    expect(loadLines).toHaveBeenCalledTimes(2);
    cache.set(profile(2));
    expect(cache.profileNetwork('streets', 1, PARAMETERS, loadLines)).toBeUndefined();
    expect(loadLines).toHaveBeenCalledTimes(2);
    expect(cache.profileNetwork('streets', 2, PARAMETERS, loadLines)?.network?.revision).toBe(2);
    cache.invalidate('streets');
    expect(cache.profileNetwork('streets', 2, PARAMETERS, loadLines)).toBeUndefined();
    cache.set(profile(2));
    cache.profileNetwork('streets', 2, PARAMETERS, loadLines);
    expect(loadLines).toHaveBeenCalledTimes(4);
  });
  it('does not load unsupported topology or publish sources replaced during loading', () => {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 1});
    cache.set(profile());
    const loadLines = vi.fn(() => SHARED);
    expect(
      cache.profileNetwork('streets', 1, {...PARAMETERS, crossings: 'noded'}, loadLines, 'gpu')
        ?.network
    ).toBeUndefined();
    expect(loadLines).not.toHaveBeenCalled();
    expect(
      cache.profileNetwork('streets', 1, PARAMETERS, () => {
        cache.set(profile(2));
        return SHARED;
      })
    ).toBeUndefined();
    expect(cache.get('streets', 2)?.network).toBeUndefined();
  });
  it('retains initial declarations and evicts facts with the dataset', () => {
    const original = profileKeplerFeatureCollection(SHARED, {
      datasetId: 'streets',
      label: 'Streets',
      coordinateSystem: 'planar',
      declarations: {coordinateModel: 'geographic', rowSemantics: 'objects'}
    });
    const cache = createKeplerDatasetProfileCache({maximumEntries: 1});
    const loadLines = vi.fn(() => SHARED);
    cache.set(original);
    const refined = cache.profileNetwork('streets', undefined, PARAMETERS, loadLines);
    expect(refined?.coordinates).toMatchObject({model: 'geographic', source: 'declared'});
    expect(refined?.rowSemantics?.value).toBe('objects');
    cache.set({...profile(), datasetId: 'other'});
    expect(cache.profileNetwork('streets', undefined, PARAMETERS, loadLines)).toBeUndefined();
    expect(loadLines).toHaveBeenCalledTimes(1);
    cache.set(original);
    cache.profileNetwork('streets', undefined, PARAMETERS, loadLines);
    expect(loadLines).toHaveBeenCalledTimes(2);
  });
  it('can change coordinate declarations after refinement using retained summaries', () => {
    const original = profileKeplerFeatureCollection(SHARED, {
      datasetId: 'streets',
      label: 'Streets',
      revision: 1,
      coordinateSystem: 'planar'
    });
    const cache = createKeplerDatasetProfileCache({maximumEntries: 1});
    cache.set(original);
    cache.profileNetwork('streets', 1, PARAMETERS, () => SHARED);
    const declared = cache.setDeclarations('streets', {coordinateModel: 'geographic'});
    expect(declared?.network?.edgeCount).toBe(3);
    expect(declared?.spacing?.coordinateModel).toBe('geographic');
    expect(cache.setDeclarations('streets', undefined)?.coordinates?.model).toBe('planar');
  });
  it.each([
    'refinement',
    'declarations'
  ] as const)('preserves private summaries and baseline on returned %s profiles and reinsertion', operation => {
    const original = profileKeplerFeatureCollection(SHARED, {
      datasetId: 'streets',
      label: 'Streets',
      revision: 1,
      coordinateSystem: 'planar',
      declarations: {coordinateModel: 'geographic', rowSemantics: 'objects'}
    });
    const baseline = getKeplerDatasetProfileUndeclared(original);
    const cache = createKeplerDatasetProfileCache({maximumEntries: 1});
    cache.set(original);
    let refined = cache.profileNetwork('streets', 1, PARAMETERS, () => SHARED)!;
    if (operation === 'declarations')
      refined = cache.setDeclarations('streets', {
        coordinateModel: 'geographic',
        rowSemantics: 'objects'
      })!;
    const planar = applyKeplerDatasetProfileDeclarations(refined, {coordinateModel: 'planar'});
    expect(planar.spacing).toEqual(baseline.spacing);
    expect(planar.spacing?.coordinateModel).toBe('planar');
    expect(planar.extent).toEqual(baseline.extent);
    expect(planar.network).toBe(refined.network);
    expect(
      applyKeplerDatasetProfileDeclarations(planar, {coordinateModel: 'geographic'}).spacing
    ).toEqual(original.spacing);
    expect(getKeplerDatasetProfileUndeclared(refined)).toMatchObject({
      coordinates: baseline.coordinates,
      spacing: baseline.spacing,
      network: refined.network
    });
    cache.set(refined);
    cache.setDeclarations('streets', {coordinateModel: 'geographic', rowSemantics: 'objects'});
    const reverted = cache.setDeclarations('streets', undefined)!;
    expect(reverted.coordinates).toEqual(baseline.coordinates);
    expect(reverted.spacing).toEqual(baseline.spacing);
    expect(reverted.rowSemantics).toEqual(baseline.rowSemantics);
    expect(reverted.network).toBe(refined.network);
    expect(original.network).toBeUndefined();
    expect(original.coordinates?.model).toBe('geographic');
  });
});

describe('network capacity requirements', () => {
  it.each([
    'cpu',
    'gpu'
  ] as const)('checks the forced snap splitting on %s before authorizing capacity', executionEngine => {
    const lines = collection([line(Array.from({length: 50002}, (_, index) => [index % 2, 0]))]);
    const parameters = {...PARAMETERS, splitAtSharedVertices: false, maximumSnapDistance: 1};
    const source = {
      ...profile(),
      vertexCount: 50002,
      network: profileKeplerDatasetNetwork(profile(), lines, parameters)
    };
    expect(source.network?.edgeCount).toBe(1);
    const points = collection([{geometry: {type: 'Point', coordinates: [0, 0]}, properties: {}}]);
    const pointProfile: KeplerDatasetProfile = {
      datasetId: 'points',
      label: 'Points',
      kind: 'points',
      rowCount: 1,
      columns: [],
      geometryColumns: [],
      catalogGeometry: 'point'
    };
    const binding = getCatalogBinding('network-snap-points')!;
    const descriptor = NETWORK_OPERATION_DESCRIPTORS.find(
      candidate => candidate.bindingId === binding.id
    )!;
    const evaluate = (streets: KeplerDatasetProfile) =>
      evaluateKeplerOperationApplicability(binding, descriptor, [streets, pointProfile], {
        executionEngine,
        overrides: parameters
      });
    const initial = evaluate(source);
    expect(
      initial.requirements.find(requirement => requirement.id === 'maximum-network-edges')?.state
    ).toBe('unknown');
    expect(initial.status).not.toBe('ready');
    const cache = createKeplerDatasetProfileCache({maximumEntries: 1});
    cache.set(source);
    const refined = cache.profileNetwork(
      'streets',
      1,
      parameters,
      () => lines,
      executionEngine,
      binding.id
    )!;
    expect(refined.network?.edgeCount).toBe(50001);
    expect(refined.network?.topologyKey).toBe(
      getKeplerDatasetNetworkTopologyKey(
        {...parameters, splitAtSharedVertices: true},
        executionEngine
      )
    );
    const assessment = evaluate(refined);
    expect(
      assessment.requirements.find(requirement => requirement.id === 'maximum-network-edges')?.state
    ).toBe('failed');
    expect(assessment.status).toBe('blocked');
    expect(
      assessment.requirements.find(requirement => requirement.id === 'maximum-network-edges')
        ?.remedy
    ).not.toContain('disable splitAtSharedVertices');
    expect(() => runSnapPoints({lines, points}, parameters)).toThrow(
      /network-snap-points.*edgeCount/
    );
  });
  it('keeps graph capacities unknown without matching topology facts', () => {
    const result = assess('network-centrality', {...profile(), vertexCount: 49000});
    for (const id of ['maximum-network-edges', 'maximum-network-nodes'])
      expect(result.requirements.find(requirement => requirement.id === id)).toMatchObject({
        state: 'unknown',
        severity: 'block',
        evidence: 'not-checkable'
      });
    expect(result.status).toBe('needs-confirmation');
  });
  it('uses all builder nodes for centrality, including unsplit interior nodes', () => {
    for (const count of [1500, 1501]) {
      const lines = collection([line(Array.from({length: count}, (_, index) => [index, 0]))]);
      const parameters = {...PARAMETERS, splitAtSharedVertices: false};
      const source = {
        ...profile(),
        vertexCount: count,
        network: profileKeplerDatasetNetwork(profile(), lines, parameters)
      };
      const result = assess('network-centrality', source, parameters);
      expect(
        result.requirements.find(requirement => requirement.id === 'maximum-network-nodes')
      ).toMatchObject({state: count === 1500 ? 'passed' : 'failed', evidence: 'profile'});
      expect(source.network?.edgeCount).toBe(1);
      expect(result.status).toBe(count === 1500 ? 'ready' : 'blocked');
      expect(
        assess('network-centrality', {...source, revision: 2}, parameters).requirements.find(
          requirement => requirement.id === 'maximum-network-nodes'
        )?.state
      ).toBe('unknown');
    }
  });
  it('checks the inclusive physical edge cap without substituting arcs', () => {
    for (const count of [50000, 50001]) {
      const lines = collection([
        {
          geometry: {
            type: 'MultiLineString',
            coordinates: Array.from({length: count}, () => [
              [0, 0],
              [1, 0]
            ])
          },
          properties: {}
        }
      ]);
      const source = {
        ...profile(),
        vertexCount: count * 2,
        network: profileKeplerDatasetNetwork(profile(), lines, PARAMETERS)
      };
      const result = assess('network-build-graph', source);
      expect(source.network?.arcCount).toBe(count * 2);
      expect(
        result.requirements.find(requirement => requirement.id === 'maximum-network-edges')?.state
      ).toBe(count === 50000 ? 'passed' : 'failed');
      expect(result.status).toBe(count === 50000 ? 'ready' : 'blocked');
    }
  });
  it('matches resolved catalog defaults and rejects changed settings', () => {
    const binding = getCatalogBinding('network-build-graph')!;
    const parameters = {...getCatalogDefaultParameters(binding), coordinateSystem: 'planar'};
    const source = {
      ...profile(),
      network: profileKeplerDatasetNetwork(profile(), SHARED, parameters)
    };
    expect(
      assess(binding.id, source).requirements.find(
        requirement => requirement.id === 'maximum-network-edges'
      )?.state
    ).toBe('passed');
    expect(
      assess(binding.id, source, {snapTolerance: 1}).requirements.find(
        requirement => requirement.id === 'maximum-network-edges'
      )?.state
    ).toBe('unknown');
  });
  it('supports independent traversable arc capacity', () => {
    const source = {
      ...profile(),
      network: profileKeplerDatasetNetwork(profile(), SHARED, PARAMETERS)
    };
    expect(
      evaluateGPUAnalysisRequirement(
        {id: 'maximum-network-arcs', severity: 'block', limit: 5},
        source,
        {}
      )
    ).toMatchObject({state: 'failed', evidence: 'profile'});
    expect(
      evaluateGPUAnalysisRequirement(
        {id: 'maximum-network-arcs', severity: 'block', limit: 6},
        source,
        {}
      ).state
    ).toBe('passed');
    expect(
      evaluateGPUAnalysisRequirement(
        {id: 'maximum-network-arcs', severity: 'block', limit: 6},
        profile(),
        {}
      ).state
    ).toBe('unknown');
  });
});
