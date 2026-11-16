// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it, vi} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../../analysis/test/utils/gpu-contributor-test-utils';
import {CatalogGpuSession} from '../../src/analysis/catalog-gpu-session';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import type {
  CatalogCoordinateSystem,
  CatalogFeatureCollection
} from '../../src/analysis/catalog-geojson';
import {buildNetworkGraph, searchNetwork} from '../../src/analysis/catalog-network';
import {getNetworkParameters} from '../../src/analysis/catalog-binding-types';
import {
  CURATED_NETWORK_DISRUPTION_BINDING,
  MAXIMUM_NETWORK_DISRUPTION_ARC_COST,
  NETWORK_DISRUPTION_LIMITS,
  prepareCuratedNetworkDisruption,
  publishCuratedNetworkDisruption,
  type CuratedNetworkDisruptionReadback
} from '../../src/curated/bindings/curated-network-disruption-binding';
import {
  createNetworkDisruptionSample,
  createNetworkDisruptionSecondDataset,
  CURATED_NETWORK_DISRUPTION_ANALYSIS,
  NETWORK_DISRUPTION_SAMPLE,
  NETWORK_DISRUPTION_EXPECTATIONS,
  NETWORK_DISRUPTION_SECOND_SAMPLE,
  NETWORK_DISRUPTION_SECOND_EXPECTATIONS
} from '../../src/curated/samples/network-disruption';
import {CURATED_NETWORK_DISRUPTION_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-network-disruption';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import * as operationDescriptorRegistry from '../../src/descriptors/operation-descriptors';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest,
  validateKeplerCuratedAnalysis
} from '../../src/curated/curated-analysis-mapping';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';
import {
  createWorkflowOperationRegistry,
  getWorkflowOperationRegistry,
  makeWorkflowOperationFromCatalogBinding
} from '../../src/workflows/workflow-operations';
import {createWorkflowDocument} from '../../src/workflows/workflow-document';
import {
  parseWorkflowAnalysisTemplate,
  serializeWorkflowAnalysisTemplate,
  validateWorkflowAnalysisTemplate
} from '../../src/workflows/workflow-analysis-template-parser';
import {readFileSync} from 'node:fs';

const binding = CURATED_NETWORK_DISRUPTION_BINDING;
const sampleParameters = NETWORK_DISRUPTION_SAMPLE.parameters;

/** Closed cost-1 shortcut; the only scenario route is the two-arc detour. */
function makeOverflowDetour(arcCost: number): Record<string, CatalogFeatureCollection> {
  const nodes = [
    [0, 0],
    [1, 0],
    [2, 0]
  ];
  return {
    network: {
      type: 'FeatureCollection',
      features: [
        [0, 1, arcCost, false],
        [1, 2, arcCost, false],
        [0, 2, 1, true]
      ].map(([from, to, minutes, closed], index) => ({
        id: `road-${index}`,
        geometry: {type: 'LineString', coordinates: [nodes[Number(from)], nodes[Number(to)]]},
        properties: {minutes, closed, direction: 'forward'}
      }))
    },
    demand: {
      type: 'FeatureCollection',
      features: [{id: 'origin', geometry: {type: 'Point', coordinates: nodes[0]}}]
    },
    facilities: {
      type: 'FeatureCollection',
      features: [
        {
          id: 'destination',
          geometry: {type: 'Point', coordinates: nodes[2]},
          properties: {supply: 1}
        }
      ]
    }
  };
}

/** Oracle reuses the existing independent Float64 Dijkstra implementation, with no new solver. */
function computeReference(
  prepared: ReturnType<typeof prepareCuratedNetworkDisruption>
): CuratedNetworkDisruptionReadback {
  const scenario = {
    ...prepared.graph,
    outgoing: prepared.graph.outgoing.map(row =>
      row.filter(
        arc =>
          prepared.edgeMask[
            prepared.arcIdentities.findIndex(
              identity => identity.edge === arc.edge && identity.forward === arc.forward
            )
          ] !== 0
      )
    )
  };
  const baselineCosts: number[] = [];
  const scenarioCosts: number[] = [];
  const baselineAccessibility: number[] = [];
  const scenarioAccessibility: number[] = [];
  for (const node of prepared.demandNodes) {
    const baseline = searchNetwork(prepared.graph, [{node, cost: 0}]);
    const changed = searchNetwork(scenario, [{node, cost: 0}]);
    let before = 0;
    let after = 0;
    prepared.facilityNodes.forEach((target, column) => {
      const baselineCost = baseline.distance[target];
      const scenarioCost = changed.distance[target];
      baselineCosts.push(baselineCost);
      scenarioCosts.push(scenarioCost);
      if (baselineCost <= prepared.maximumCost) before += prepared.opportunityWeights[column];
      if (scenarioCost <= prepared.maximumCost) after += prepared.opportunityWeights[column];
    });
    baselineAccessibility.push(before);
    scenarioAccessibility.push(after);
  }
  return {
    baselineCosts,
    scenarioCosts,
    costDifferences: baselineCosts.map((cost, index) =>
      Number.isFinite(cost) && Number.isFinite(scenarioCosts[index])
        ? scenarioCosts[index] - cost
        : NaN
    ),
    newlyUnreachable: baselineCosts.map((cost, index) =>
      Number(Number.isFinite(cost) && scenarioCosts[index] === Infinity)
    ),
    baselineAccessibility,
    scenarioAccessibility,
    accessibilityDifferences: baselineAccessibility.map(
      (value, index) => scenarioAccessibility[index] - value
    ),
    stages: [
      {stage: 'baseline-routing', converged: 1},
      {stage: 'scenario-routing', converged: 1}
    ]
  };
}

function mapInputs(
  inputs: Record<string, CatalogFeatureCollection>,
  columns: Readonly<Record<string, string>>,
  identifier: string,
  maximumCost: number,
  coordinateSystem: CatalogCoordinateSystem
) {
  const analysis = CURATED_NETWORK_DISRUPTION_ANALYSIS;
  const profiles = Object.entries(inputs).map(([slot, collection]) =>
    profileKeplerFeatureCollection(collection, {
      datasetId: `${identifier}-${slot}`,
      label: `${identifier} ${slot}`,
      coordinateSystem,
      declarations: {
        coordinateModel: coordinateSystem === 'planar' ? 'planar' : 'geographic',
        rowSemantics: 'objects',
        measurement: Object.fromEntries(
          Object.entries(columns)
            .filter(
              ([key]) =>
                key.startsWith(`${slot}.`) && (key.endsWith('cost') || key.endsWith('weight'))
            )
            .map(([, column]) => [column, 'amount'])
        )
      }
    })
  );
  let mapping = createKeplerCuratedAnalysisMapping(analysis, profiles, {
    parameters: {maximumCost, maximumSnapDistance: 0}
  });
  for (const slot of analysis.slots)
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: slot.name, datasetId: `${identifier}-${slot.name}`},
      analysis,
      profiles
    );
  for (const [key, column] of Object.entries(columns)) {
    const [slot, role] = key.split('.');
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot, role, columns: [column]},
      analysis,
      profiles
    );
  }
  expect(mapping.ready, JSON.stringify(mapping.problems)).toBe(true);
  const request = buildKeplerCuratedAnalysisRequest(analysis, mapping);
  // Mapping requests leave coordinate defaults unresolved. Resolve from the declared profiles
  // before catalog normalization can fall back to its geographic constant.
  const profilesByInput = Object.fromEntries(
    Object.entries(request.inputDatasetIds).map(([input, datasetId]) => [
      input,
      profiles.find(profile => profile.datasetId === datasetId)!
    ])
  );
  const resolved = resolveKeplerOperationParameters(
    CURATED_NETWORK_DISRUPTION_BINDING,
    CURATED_NETWORK_DISRUPTION_DESCRIPTOR,
    profilesByInput,
    {overrides: request.parameters, executionEngine: 'gpu'}
  );

  return {mapping, profiles, request: {...request, parameters: resolved.parameters}, resolved};
}

describe('curated network disruption preparation and mappings', () => {
  it('rejects finite arcs whose detour sum can overflow float32 before GPU allocation', async () => {
    const inputs = makeOverflowDetour(2e38);
    const graph = buildNetworkGraph(
      inputs['network'],
      getNetworkParameters({...sampleParameters, snapTolerance: 0, splitAtSharedVertices: true}),
      NETWORK_DISRUPTION_LIMITS.edges
    );
    const scenario = {
      ...graph,
      outgoing: graph.outgoing.map(row => row.filter(arc => arc.edge !== 2))
    };
    expect(searchNetwork(scenario, [{node: 0, cost: 0}]).distance[2]).toBe(4e38);
    expect(Math.fround(2e38)).toBeLessThan(Infinity);
    expect(Math.fround(Math.fround(2e38) + Math.fround(2e38))).toBe(Infinity);
    expect(() => prepareCuratedNetworkDisruption(inputs, sampleParameters)).toThrow(
      /curated-network-disruption maximumArcCost.*rescale/
    );
    const device = createNullWebGPUDevice();
    const uploads = vi.spyOn(CatalogGpuSession.prototype, 'input');
    const submitted = vi.spyOn(CatalogGpuSession.prototype, 'run');
    try {
      await expect(binding.runGpu!({device}, inputs, sampleParameters)).rejects.toThrow(
        /maximumArcCost.*rescale/
      );
      expect(uploads).not.toHaveBeenCalled();
      expect(submitted).not.toHaveBeenCalled();
    } finally {
      uploads.mockRestore();
      submitted.mockRestore();
      device.destroy();
    }
  });

  it('accepts the arc-cost ceiling and keeps worst supported float32 sums finite', () => {
    let pathCost = 0;
    for (let step = 0; step < NETWORK_DISRUPTION_LIMITS.nodes; step++)
      pathCost = Math.fround(pathCost + MAXIMUM_NETWORK_DISRUPTION_ARC_COST);
    expect(pathCost).toBe(2 ** 126);
    const prepared = prepareCuratedNetworkDisruption(
      makeOverflowDetour(MAXIMUM_NETWORK_DISRUPTION_ARC_COST),
      sampleParameters
    );
    const output = publishCuratedNetworkDisruption(prepared, computeReference(prepared));
    expect(output.tables[1].columns['scenario_cost']).toEqual([
      2 * MAXIMUM_NETWORK_DISRUPTION_ARC_COST
    ]);
    expect(output.tables[0].columns['cut_off']).toEqual([false]);
    const nextFloat32 = MAXIMUM_NETWORK_DISRUPTION_ARC_COST * (1 + 2 ** -23);
    expect(() =>
      prepareCuratedNetworkDisruption(makeOverflowDetour(nextFloat32), sampleParameters)
    ).toThrow(/maximumArcCost/);
  });

  it.each([
    'planar',
    'EPSG:4326'
  ] as const)('resolves the declared %s model without fixing a coordinate system in the reusable analysis', coordinateSystem => {
    const {request, mapping} = mapInputs(
      createNetworkDisruptionSample(),
      NETWORK_DISRUPTION_SAMPLE.columns,
      'declared-coordinate-model',
      NETWORK_DISRUPTION_SAMPLE.parameters.maximumCost,
      coordinateSystem
    );
    expect(mapping.explicitParameterNames).not.toContain('coordinateSystem');
    expect(request.parameters['coordinateSystem']).toBe(coordinateSystem);
    expect(CURATED_NETWORK_DISRUPTION_ANALYSIS.execution).toMatchObject({
      parameters: {costMode: 'property', costUnit: 'minutes'}
    });
  });

  it('matches both hand-computed recipe fixtures with the existing Float64 routing oracle', () => {
    for (const [inputs, specification, expected] of [
      [createNetworkDisruptionSample(), NETWORK_DISRUPTION_SAMPLE, NETWORK_DISRUPTION_EXPECTATIONS],
      [
        createNetworkDisruptionSecondDataset(),
        NETWORK_DISRUPTION_SECOND_SAMPLE,
        NETWORK_DISRUPTION_SECOND_EXPECTATIONS
      ]
    ] as const) {
      const prepared = prepareCuratedNetworkDisruption(inputs, specification.parameters);
      expect(computeReference(prepared)).toEqual(expected);
    }
  });

  it('maps sample and independently identified/renamed columns through the existing mapper', () => {
    for (const [inputs, specification] of [
      [createNetworkDisruptionSample(), NETWORK_DISRUPTION_SAMPLE],
      [createNetworkDisruptionSecondDataset(), NETWORK_DISRUPTION_SECOND_SAMPLE]
    ] as const) {
      const {request, mapping, resolved} = mapInputs(
        inputs,
        specification.columns,
        specification.id,
        specification.parameters.maximumCost,
        specification.parameters.coordinateSystem
      );
      expect(mapping.explicitParameterNames).not.toContain('coordinateSystem');
      expect(request.parameters['coordinateSystem'], specification.id).toBe(
        specification.parameters.coordinateSystem
      );
      expect(
        resolved.resolutions.find(resolution => resolution.parameter === 'coordinateSystem')
      ).toMatchObject({source: 'data'});
      expect(request.bindingId).toBe(binding.id);
      expect(request.inputDatasetIds).toEqual({
        network: `${specification.id}-network`,
        facilities: `${specification.id}-facilities`,
        demand: `${specification.id}-demand`
      });
      expect(request.parameters).toMatchObject(specification.parameters);
      const mappedInputs = Object.fromEntries(
        Object.entries(request.inputDatasetIds).map(([slot, datasetId]) => [
          slot,
          Object.fromEntries(
            Object.entries(inputs).map(([name, collection]) => [
              `${specification.id}-${name}`,
              collection
            ])
          )[datasetId]
        ])
      );
      expect(
        prepareCuratedNetworkDisruption(mappedInputs, request.parameters).facilityNodes.length
      ).toBe(inputs['facilities'].features.length);
    }
  });

  it('greys incompatible datasets and rejects wrong-type closure columns through mapping', () => {
    const inputs = createNetworkDisruptionSample();
    inputs['network'].features.forEach(feature => {
      feature.properties!['closed'] = 'closed';
    });
    const profiles = Object.entries(inputs).map(([slot, collection]) =>
      profileKeplerFeatureCollection(collection, {
        datasetId: slot,
        label: slot,
        coordinateSystem: 'planar'
      })
    );
    const mapping = createKeplerCuratedAnalysisMapping(
      CURATED_NETWORK_DISRUPTION_ANALYSIS,
      profiles
    );
    expect(mapping.ready).toBe(false);
    expect(
      mapping.slots[0].candidates.find(candidate => candidate.datasetId === 'network')?.fit
    ).toBe('missing-columns');
    expect(
      mapping.slots[0].candidates.find(candidate => candidate.datasetId === 'demand')?.fit
    ).toBe('wrong-kind');
  });

  it('keeps original node, source, parallel-edge and CSR identities while changing only the mask', () => {
    const inputs = createNetworkDisruptionSecondDataset();
    const prepared = prepareCuratedNetworkDisruption(
      inputs,
      NETWORK_DISRUPTION_SECOND_SAMPLE.parameters
    );
    expect(Array.from(prepared.topology.offsets)).toEqual([0, 2, 3, 3]);
    expect(Array.from(prepared.topology.neighbors)).toEqual([1, 1, 2]);
    expect(Array.from(prepared.topology.weights)).toEqual([1, 4, 2]);
    expect(Array.from(prepared.edgeMask)).toEqual([0, 1, 1]);
    expect(Array.from(prepared.facilityNodes)).toEqual([2, 0, 1]);
    const original = JSON.stringify(inputs);
    prepareCuratedNetworkDisruption(inputs, NETWORK_DISRUPTION_SECOND_SAMPLE.parameters);
    expect(JSON.stringify(inputs)).toBe(original);
    inputs['network'].features[0].properties!['blocked'] = 0;
    const reopened = prepareCuratedNetworkDisruption(
      inputs,
      NETWORK_DISRUPTION_SECOND_SAMPLE.parameters
    );
    expect(reopened.arcIdentities).toEqual(prepared.arcIdentities);
    expect(reopened.graph.nodes).toEqual(prepared.graph.nodes);
    expect(reopened.topology).toEqual(prepared.topology);
    expect(Array.from(reopened.edgeMask)).toEqual([1, 1, 1]);
  });

  it('masks both directions and every split/multipart piece of a source street', () => {
    const inputs = createNetworkDisruptionSample();
    inputs['network'].features[0] = {
      id: 'multipart',
      geometry: {
        type: 'MultiLineString',
        coordinates: [
          [
            [0, 0],
            [1, 0],
            [2, 0]
          ],
          [
            [2, 0],
            [3, 0]
          ]
        ]
      },
      properties: {closed: true, minutes: 8, direction: 'both'}
    };
    const prepared = prepareCuratedNetworkDisruption(inputs, sampleParameters);
    const slots = prepared.arcIdentities
      .map((arc, slot) => ({arc, slot}))
      .filter(({arc}) => prepared.graph.edges[arc.edge].sourceIndex === 0);
    expect(slots).toHaveLength(6);
    expect(slots.every(({slot}) => prepared.edgeMask[slot] === 0)).toBe(true);
    expect(slots.map(({slot}) => prepared.topology.weights[slot]).sort()).toEqual([
      4, 4, 4, 4, 8, 8
    ]);
    expect(
      prepared.graph.edges.filter(edge => edge.sourceIndex === 0).map(edge => edge.edgeId)
    ).toEqual(['0:0:0', '0:0:1', '0:1:0']);
  });

  it('preserves reverse-only directions, cost units and explicit zero facility weights', () => {
    const inputs = createNetworkDisruptionSecondDataset();
    inputs['network'].features[0].properties!['restriction'] = 'reverse';
    inputs['facilities'].features[0].properties!['capacity'] = 0;
    const prepared = prepareCuratedNetworkDisruption(inputs, {
      ...NETWORK_DISRUPTION_SECOND_SAMPLE.parameters,
      costUnit: 'seconds'
    });
    expect(
      prepared.arcIdentities.some(
        arc => arc.edge === 0 && arc.from === 1 && arc.to === 0 && !arc.forward
      )
    ).toBe(true);
    expect(prepared.opportunityWeights[0]).toBe(0);
    expect(prepared.costUnit).toBe('seconds');
    expect(
      prepareCuratedNetworkDisruption(inputs, {
        ...NETWORK_DISRUPTION_SECOND_SAMPLE.parameters,
        costMode: 'length'
      }).costUnit
    ).toBe('coordinate-unit');
    expect(
      prepareCuratedNetworkDisruption(createNetworkDisruptionSample(), {
        ...sampleParameters,
        coordinateSystem: 'EPSG:4326',
        costMode: 'length'
      }).costUnit
    ).toBe('meters');
  });

  it('snaps only to routing endpoints, never unused interior CSR rows', () => {
    const inputs = createNetworkDisruptionSecondDataset();
    inputs['network'].features = [
      {
        id: 'bent-road',
        geometry: {
          type: 'LineString',
          coordinates: [
            [100, 200],
            [110, 200],
            [120, 200]
          ]
        },
        properties: {travelTime: 4, blocked: 0, restriction: 'forward'}
      }
    ];
    // The middle coordinate exists as graph node 1, but is not an edge endpoint.
    expect(() =>
      prepareCuratedNetworkDisruption(inputs, NETWORK_DISRUPTION_SECOND_SAMPLE.parameters)
    ).toThrow(/facilities row 2.*maximumSnapDistance 0/);
    const prepared = prepareCuratedNetworkDisruption(inputs, {
      ...NETWORK_DISRUPTION_SECOND_SAMPLE.parameters,
      maximumSnapDistance: 10
    });
    expect(Array.from(prepared.facilityNodes)).toEqual([2, 0, 0]);
    expect(Array.from(prepared.facilitySnapDistances)).toEqual([0, 0, 10]);
    expect(prepared.graph.nodes).toHaveLength(3);
  });

  it('respects asymmetric costs and reports missing cost arcs while rejecting negatives', () => {
    const inputs = createNetworkDisruptionSecondDataset();
    inputs['network'].features[0].properties = {
      travelTime: 1,
      inbound: 7,
      blocked: 0,
      restriction: 'both'
    };
    const parameters = {
      ...NETWORK_DISRUPTION_SECOND_SAMPLE.parameters,
      reverseCostProperty: 'inbound'
    };
    const prepared = prepareCuratedNetworkDisruption(inputs, parameters);
    const reverseSlot = prepared.arcIdentities.findIndex(arc => arc.edge === 0 && !arc.forward);
    expect(prepared.topology.weights[reverseSlot]).toBe(7);
    expect(
      prepared.graph.diagnostics.some(diagnostic => diagnostic.code === 'invalid-cost-arc-removed')
    ).toBe(false);
    inputs['network'].features[1].properties!['travelTime'] = null;
    const missing = prepareCuratedNetworkDisruption(inputs, parameters);
    expect(
      missing.graph.diagnostics.find(diagnostic => diagnostic.code === 'invalid-cost-arc-removed')
        ?.count
    ).toBe(1);
    inputs['network'].features[1].properties!['travelTime'] = -1;
    expect(() => prepareCuratedNetworkDisruption(inputs, parameters)).toThrow(
      /curated-network-disruption.*forwardCostProperty/
    );
  });

  it('rejects missing inputs, ambiguous closure values, bad weights, parameters and distant points before GPU allocation', () => {
    const inputs = createNetworkDisruptionSample();
    for (const name of ['network', 'demand', 'facilities']) {
      const missing = {...inputs};
      delete missing[name];
      expect(() => prepareCuratedNetworkDisruption(missing, sampleParameters)).toThrow(
        new RegExp(`curated-network-disruption.*${name}`)
      );
    }
    for (const value of [null, undefined, 'true', '1', 2, -1, NaN]) {
      const invalid = createNetworkDisruptionSample();
      invalid['network'].features[0].properties!['closed'] = value;
      expect(() => prepareCuratedNetworkDisruption(invalid, sampleParameters)).toThrow(
        /curated-network-disruption.*closureProperty/
      );
    }
    for (const value of [-1, NaN, Infinity, null, '4']) {
      const invalid = createNetworkDisruptionSample();
      invalid['facilities'].features[0].properties!['supply'] = value;
      expect(() => prepareCuratedNetworkDisruption(invalid, sampleParameters)).toThrow(
        /curated-network-disruption.*weightProperty/
      );
    }
    const invalidParameters: Record<string, CatalogParameterValue>[] = [
      {maximumCost: NaN},
      {maximumCost: -1},
      {snapTolerance: -1},
      {maximumSnapDistance: Infinity},
      {costMode: 'speed'},
      {splitAtSharedVertices: 'yes'},
      {coordinateSystem: 'guessed'},
      {closureProperty: ''}
    ];
    for (const overrides of invalidParameters)
      expect(() =>
        prepareCuratedNetworkDisruption(inputs, {...sampleParameters, ...overrides})
      ).toThrow(/curated-network-disruption/);
    inputs['demand'].features[0].geometry = {type: 'Point', coordinates: [100, 100]};
    expect(() => prepareCuratedNetworkDisruption(inputs, sampleParameters)).toThrow(
      /maximumSnapDistance 0/
    );
    inputs['demand'].features[0].geometry = {type: 'Point', coordinates: [NaN, 0]};
    expect(() => prepareCuratedNetworkDisruption(inputs, sampleParameters)).toThrow(/finite Point/);
  });

  it('checks feature, vertex, derived physical-edge and node capacities independently', () => {
    const inputs = createNetworkDisruptionSample();
    expect(() =>
      prepareCuratedNetworkDisruption(
        {
          ...inputs,
          demand: {
            type: 'FeatureCollection',
            features: Array(NETWORK_DISRUPTION_LIMITS.points + 1).fill(inputs['demand'].features[0])
          }
        },
        sampleParameters
      )
    ).toThrow(/demand feature count/);
    const tooManyVertices = {
      type: 'FeatureCollection' as const,
      features: [
        {
          ...inputs['network'].features[0],
          geometry: {
            type: 'LineString',
            coordinates: Array(NETWORK_DISRUPTION_LIMITS.vertices + 1).fill([0, 0])
          }
        }
      ]
    };
    expect(() =>
      prepareCuratedNetworkDisruption({...inputs, network: tooManyVertices}, sampleParameters)
    ).toThrow(/network vertex count/);
    const sharedVertices = {
      type: 'FeatureCollection' as const,
      features: Array.from({length: 3}, (_, index) => ({
        id: index,
        geometry: {
          type: 'LineString',
          coordinates: Array.from({length: 1501}, (_, node) => [node, 0])
        },
        properties: {closed: false, minutes: 1, direction: 'forward'}
      }))
    };
    expect(() =>
      prepareCuratedNetworkDisruption({...inputs, network: sharedVertices}, sampleParameters)
    ).toThrow(/edgeCount.*4096/);
    const manyNodes = {
      type: 'FeatureCollection' as const,
      features: [
        {
          geometry: {
            type: 'LineString',
            coordinates: Array.from({length: 4097}, (_, node) => [node, 0])
          },
          properties: {closed: false, minutes: 1, direction: 'forward'}
        }
      ]
    };
    expect(() =>
      prepareCuratedNetworkDisruption(
        {...inputs, network: manyNodes},
        {...sampleParameters, splitAtSharedVertices: false}
      )
    ).toThrow(/nodeCount.*4096/);
  });
});

describe('curated network disruption execution and publication', () => {
  it('keeps every active basic descriptor parameter exposed by the supplied curated record', () => {
    // Simulate only registration, without editing stream X's shared descriptor registry.
    const originalLookup = operationDescriptorRegistry.getKeplerOperationDescriptor;
    const lookup = vi
      .spyOn(operationDescriptorRegistry, 'getKeplerOperationDescriptor')
      .mockImplementation(identifier =>
        identifier === binding.id
          ? CURATED_NETWORK_DISRUPTION_DESCRIPTOR
          : originalLookup(identifier)
      );
    try {
      expect(validateKeplerCuratedAnalysis(CURATED_NETWORK_DISRUPTION_ANALYSIS)).toEqual([]);
      expect(
        validateKeplerCuratedAnalysis({
          ...CURATED_NETWORK_DISRUPTION_ANALYSIS,
          exposedParameters: CURATED_NETWORK_DISRUPTION_ANALYSIS.exposedParameters.filter(
            name => name !== 'costMode'
          )
        })
      ).toEqual(['network-disruption: descriptor basic parameters not exposed: costMode.']);
    } finally {
      lookup.mockRestore();
    }
  });

  it('constructs actual recipe routing, closure and difference kernels and releases its session on submission failure', async () => {
    const device = createNullWebGPUDevice();
    const destroyed = vi.spyOn(CatalogGpuSession.prototype, 'dispose');
    const submitted = vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(function (
      this: CatalogGpuSession
    ) {
      const kernels = getGPUCommandNodeKernelSources(
        (this.graph as unknown as {nodes: unknown[]}).nodes
      );
      for (const id of [
        'network-disruption-recipe-closures',
        'network-disruption-recipe-differences',
        'network-disruption-recipe-accessibility-delta',
        'publish-baseline-routing',
        'publish-scenario-routing'
      ])
        expect(
          kernels.some(kernel => kernel.id === id),
          id
        ).toBe(true);
      expect(
        kernels.some(kernel => kernel.id.startsWith('network-disruption-recipe-baseline'))
      ).toBe(true);
      expect(
        kernels.some(kernel => kernel.id.startsWith('network-disruption-recipe-scenario'))
      ).toBe(true);
      throw new Error('construction probe reached submission');
    });
    try {
      await expect(
        binding.runGpu!({device}, createNetworkDisruptionSample(), sampleParameters)
      ).rejects.toThrow('construction probe reached submission');
      expect(submitted).toHaveBeenCalledOnce();
      expect(destroyed).toHaveBeenCalledOnce();
    } finally {
      submitted.mockRestore();
      destroyed.mockRestore();
      device.destroy();
    }
  });

  it('requires a GPU without silently using a CPU fallback', async () => {
    expect(binding.run).toBeUndefined();
    await expect(
      runCatalogOperationAsync(binding, sampleParameters, createNetworkDisruptionSample())
    ).rejects.toThrow(/GPU device required/);
  });

  it('publishes original identities, explicit null unreachable costs, exact difference columns and geometric presentation plans', () => {
    const prepared = prepareCuratedNetworkDisruption(
      createNetworkDisruptionSample(),
      sampleParameters
    );
    const output = publishCuratedNetworkDisruption(prepared, NETWORK_DISRUPTION_EXPECTATIONS);
    expect(output.summary).toMatchObject({
      newlyUnreachableCount: 2,
      cutOffCount: 0,
      closedArcCount: 1,
      costUnit: 'minutes',
      converged: 1
    });
    const [demand, pairs, arcs] = output.tables;
    expect(demand.columns['demand_source_id']).toEqual(['demand-0', 'demand-1']);
    expect(demand.columns['accessibility_difference']).toEqual([-7, 0]);
    expect(pairs.columns['facility_source_id'].slice(0, 5)).toEqual([
      'facility-0',
      'facility-1',
      'facility-2',
      'facility-3',
      'facility-4'
    ]);
    expect(pairs.columns['baseline_cost']).toEqual([0, 1, 3, 4, null, null, null, 0, 1, null]);
    expect(pairs.columns['scenario_cost']).toEqual([
      0,
      1,
      null,
      null,
      null,
      null,
      null,
      0,
      1,
      null
    ]);
    expect(pairs.columns['cost_difference']).toEqual([
      0,
      0,
      null,
      null,
      null,
      null,
      null,
      0,
      0,
      null
    ]);
    expect(pairs.columns['status'][2]).toBe('newly-unreachable');
    expect(pairs.columns['status'][4]).toBe('baseline-unreachable');
    expect(pairs.columns['demand_snap_distance']).toEqual(Array(10).fill(0));
    expect(pairs.columns['facility_snap_distance']).toEqual(Array(10).fill(0));
    expect(arcs.columns['csr_slot']).toEqual([0, 1, 2, 3]);
    expect(arcs.columns['source_id']).toEqual(['street-0', 'street-1', 'street-2', 'street-3']);
    expect(arcs.columns['scenario_cost']).toEqual([1, null, 1, 1]);
    for (const table of output.tables) {
      const specification = binding.outputs.find(candidate => candidate.tableId === table.id)!;
      expect(specification.identityColumns.every(column => column in table.columns)).toBe(true);
      expect(Object.keys(table.columns)).toEqual(Object.keys(table.columnTypes));
      expect(Object.values(table.columns).every(column => column.length === table.rowCount)).toBe(
        true
      );
      if (table.geometryColumn)
        expect(
          planKeplerResultPresentation(
            table,
            CURATED_NETWORK_DISRUPTION_DESCRIPTOR.outputs.find(
              candidate => candidate.tableId === table.id
            )!
          )
        ).toBeDefined();
    }
  });

  it('distinguishes threshold access loss from complete cut-off and baseline disconnection', () => {
    const inputs = createNetworkDisruptionSecondDataset();
    let prepared = prepareCuratedNetworkDisruption(
      inputs,
      NETWORK_DISRUPTION_SECOND_SAMPLE.parameters
    );
    let output = publishCuratedNetworkDisruption(prepared, NETWORK_DISRUPTION_SECOND_EXPECTATIONS);
    expect(output.tables[0].columns['cut_off']).toEqual([false]);
    expect(output.tables[0].columns['accessibility_difference']).toEqual([-7]);
    inputs['network'].features.forEach(feature => {
      feature.properties!['blocked'] = 1;
    });
    inputs['facilities'].features = [inputs['facilities'].features[0]];
    prepared = prepareCuratedNetworkDisruption(inputs, NETWORK_DISRUPTION_SECOND_SAMPLE.parameters);
    output = publishCuratedNetworkDisruption(prepared, computeReference(prepared));
    expect(output.tables[0].columns['cut_off']).toEqual([true]);
    expect(output.tables[0].columns['status']).toEqual(['cut-off']);
    expect(output.summary['newlyUnreachableCount']).toBe(1);
  });

  it('rejects nonconverged, malformed and nonfinite readbacks with readable errors', () => {
    const prepared = prepareCuratedNetworkDisruption(
      createNetworkDisruptionSample(),
      sampleParameters
    );
    for (const stage of ['baseline-routing', 'scenario-routing'])
      expect(() =>
        publishCuratedNetworkDisruption(prepared, {
          ...NETWORK_DISRUPTION_EXPECTATIONS,
          stages: NETWORK_DISRUPTION_EXPECTATIONS.stages.map(status => ({
            ...status,
            converged: status.stage === stage ? 0 : 1
          }))
        })
      ).toThrow(new RegExp(stage));
    expect(() =>
      publishCuratedNetworkDisruption(prepared, {...NETWORK_DISRUPTION_EXPECTATIONS, stages: []})
    ).toThrow(/baseline-routing/);
    expect(() =>
      publishCuratedNetworkDisruption(prepared, {
        ...NETWORK_DISRUPTION_EXPECTATIONS,
        baselineCosts: []
      })
    ).toThrow(/baselineCosts.*length/);
    expect(() =>
      publishCuratedNetworkDisruption(prepared, {
        ...NETWORK_DISRUPTION_EXPECTATIONS,
        baselineAccessibility: [NaN, 7]
      })
    ).toThrow(/baselineAccessibility.*finite/);
    expect(() =>
      publishCuratedNetworkDisruption(prepared, {
        ...NETWORK_DISRUPTION_EXPECTATIONS,
        newlyUnreachable: Array(10).fill(0)
      })
    ).toThrow(/inconsistent/);
  });

  it('has complete policies, typed workflow ports, and a recorded synthetic licence for both fixtures', () => {
    expect(
      validateKeplerOperationDescriptor(CURATED_NETWORK_DISRUPTION_DESCRIPTOR, binding)
    ).toEqual([]);
    const operation = makeWorkflowOperationFromCatalogBinding(binding);
    expect(operation.inputs.map(input => input.accepts)).toEqual([['line'], ['point'], ['point']]);
    expect(operation.outputs.map(output => output.kind)).toEqual(['point', 'table', 'line']);
    expect(NETWORK_DISRUPTION_SAMPLE.license).toBe('MIT');
    const licenses = readFileSync(
      new URL('../../../../THIRD-PARTY-DATA.md', import.meta.url),
      'utf8'
    );
    expect(licenses).toContain('## Curated network disruption synthetic samples');
    expect(licenses).toContain('modules/kepler/src/curated/samples/network-disruption.ts');
    expect(licenses).toContain('independently identified parallel-street network');
  });

  it('round trips the template and identifies the shared descriptor-registration dependency for executable validation', () => {
    const analysis = CURATED_NETWORK_DISRUPTION_ANALYSIS;
    const document = createWorkflowDocument({id: analysis.id, title: analysis.title});
    document.steps = [
      ...analysis.slots.map(slot => ({
        id: `source-${slot.name}`,
        operationId: 'source-dataset',
        inputs: {},
        parameters: {
          dataset: {from: 'static' as const, value: slot.name},
          geometryField: {from: 'static' as const, value: 'geometry'}
        },
        position: {x: 0, y: 0}
      })),
      {
        id: 'analysis',
        operationId: binding.id,
        inputs: Object.fromEntries(
          analysis.slots.map(slot => [slot.name, {stepId: `source-${slot.name}`, output: 'data'}])
        ),
        parameters: Object.fromEntries(
          Object.entries(sampleParameters).map(([name, value]) => [
            name,
            {from: 'static' as const, value}
          ])
        ),
        position: {x: 1, y: 0}
      }
    ];
    const template = parseWorkflowAnalysisTemplate(
      JSON.stringify({
        schemaVersion: 1,
        id: analysis.id,
        title: analysis.title,
        question: analysis.question,
        method: analysis.method,
        curatedAnalysisId: analysis.id,
        document,
        slots: analysis.slots.map(slot => ({
          ...slot,
          source: {stepId: `source-${slot.name}`, output: 'data'},
          fields: slot.fields.map(field => ({
            ...field,
            sources: [
              {
                stepId: 'analysis',
                parameter:
                  analysis.execution.status === 'available'
                    ? analysis.execution.fieldParameters[`${slot.name}.${field.role}`]
                    : field.role
              }
            ]
          }))
        })),
        exposed: [],
        sample: {
          id: NETWORK_DISRUPTION_SAMPLE.id,
          license: NETWORK_DISRUPTION_SAMPLE.license,
          attribution: NETWORK_DISRUPTION_SAMPLE.attribution,
          reference: NETWORK_DISRUPTION_SAMPLE.reference
        },
        reads: {summary: analysis.method, cautions: CURATED_NETWORK_DISRUPTION_DESCRIPTOR.notes}
      })
    );
    const registry = createWorkflowOperationRegistry([
      ...getWorkflowOperationRegistry()
        .list()
        .filter(operation => operation.id !== binding.id),
      makeWorkflowOperationFromCatalogBinding(binding)
    ]);
    const validation = validateWorkflowAnalysisTemplate(template, {registry});
    // Stream X owns the descriptor registry. This local binding registry establishes port/schema
    // validity, but cannot substitute for registering the field roles in the shared descriptor map.
    // Once X registers this descriptor, this same envelope must be executable with no issues.
    const descriptorRegistered = registry
      .get(binding.id)!
      .analysisDescriptor?.inputs.some(input => input.fields.length > 0);
    expect(validation.issues.map(issue => issue.code)).toEqual(
      descriptorRegistered ? [] : Array(5).fill('invalid-role-parameter')
    );
    expect(validation.executable).toBe(Boolean(descriptorRegistered));
    expect(
      parseWorkflowAnalysisTemplate(JSON.stringify(serializeWorkflowAnalysisTemplate(template)))
    ).toEqual(template);
    expect(
      validateWorkflowAnalysisTemplate({...template, slots: template.slots.slice(1)}, {registry})
        .executable
    ).toBe(false);
  });
});
