// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readdirSync, readFileSync} from 'node:fs';
import {expect, test, vi} from 'vitest';
import * as records from '@luma-contributors/analysis/computation-record';
import {fromRecordedLensFile, listLensScenarios} from '../../src/computation-lenses/index';
import {
  declareLensScenario,
  getMissingScenarioChoices,
  getRecordedCaptureNames,
  getRecordedStorageBufferLimit,
  getStructureSignature
} from './scenario-declaration.test-utils';
import '../../src/computation-lenses/scenarios/index';

vi.mock(import('@luma-contributors/analysis/testing'), async importOriginal => {
  const testing = await importOriginal();
  return {
    ...testing,
    createNullWebGPUDevice() {
      const device = testing.createNullWebGPUDevice();
      // The null device reports no alignment. Use the WebGPU core alignment used by the captures
      // so declaration allocates the same aligned scratch views as the captured GPU graph.
      Object.defineProperty(device.limits, 'minStorageBufferOffsetAlignment', {value: 256});
      return device;
    }
  };
});

const scenarios = listLensScenarios();

test('PTS-138: the assembled Ripley receipt records its capped Hanisch search', async () => {
  const scenario = scenarios.find(
    candidate => candidate.id === 'gpu-dataframe.ripley-distance-functions'
  )!;
  const declared = await declareLensScenario(scenario);
  expect(declared.missingChoices).toEqual([]);
  expect(declared.record.choices.find(choice => choice.name === 'hanischSearch')).toMatchObject({
    stage: 'aggregate',
    value: 'capped',
    alternatives: ['capped', 'exact'],
    sweepable: true
  });
  expect(
    getMissingScenarioChoices(scenario, {
      ...declared.record,
      choices: declared.record.choices.filter(choice => choice.name !== 'hanischSearch')
    })
  ).toEqual(['hanischSearch']);
});

test.each([
  'elementOffset',
  'bindingByteOffset'
] as const)('LENS-106: live-only %s drift fails before archive comparison', async field => {
  const scenario = scenarios.find(
    candidate => candidate.id === 'gpu-spatial-analysis.pair-gather'
  )!;
  const originalDescribe = records.describeGPUCommandGraph;
  const describe = vi
    .spyOn(records, 'describeGPUCommandGraph')
    .mockImplementationOnce((...parameters) => {
      const described = originalDescribe(...parameters);
      const graph = structuredClone(described.graph);
      const node = graph.nodes.find(candidate => candidate.bindings?.length)!;
      node.bindings![0][field] =
        (node.bindings![0][field] ?? 0) + (field === 'bindingByteOffset' ? 256 : 1);
      return {...described, graph};
    });
  try {
    // Only the live compile description changes; record assembly and archives are untouched.
    await expect(declareLensScenario(scenario)).rejects.toThrow(
      /compiled graph .*structurally different/
    );
    expect(describe).toHaveBeenCalled();
  } finally {
    describe.mockRestore();
  }
  // The next declaration must use the real description after the fixture's spy is gone.
  await expect(declareLensScenario(scenario)).resolves.toHaveProperty('record');
});

/**
 * Network scenarios whose recorded capture names are assembled by scenario code after readback
 * (`NetworkElementScenario.capture`, `captureInputGeometry`: `road-context`, gathered start points,
 * derived node points), so the null-device declaration sees the graph's snapshot slots instead.
 * Their graphs, nodes, resources and binding layouts take the full comparison; only the capture
 * names are exempt, and only by exactly this difference (LENS-105). A changed difference, or an
 * entry whose names now agree, fails the comparison.
 */
const CAPTURE_NAMES_DERIVED_AFTER_READBACK: Readonly<
  Record<string, {liveOnly: readonly string[]; recordedOnly: readonly string[]}>
> = {
  'gpu-network.adjacency-matrix': {
    liveOnly: ['neighbors', 'nodePositions', 'offsets'],
    recordedOnly: []
  },
  'gpu-network.adjacency-matrix-order': {
    liveOnly: ['neighbors', 'nodePositions', 'offsets'],
    recordedOnly: ['groupsSample', 'orderSample', 'sampleNodes']
  },
  'gpu-network.drive-time-catchment-recipe': {
    liveOnly: ['converged'],
    recordedOnly: ['catchment-demand', 'road-context']
  },
  'gpu-network.edge-bundling': {
    liveOnly: ['zonePositions'],
    recordedOnly: ['bundleStart', 'straightFlowOffsets', 'straightFlows']
  },
  'gpu-network.flow-aggregation': {
    liveOnly: ['destinationZoneIds', 'originZoneIds'],
    recordedOnly: ['rankedFlowLines', 'rankedFlowOffsets', 'recordLines', 'recordOffsets']
  },
  'gpu-network.map-matching': {
    liveOnly: ['matchCandidateOverflow', 'snappedPositions'],
    recordedOnly: ['matchedRunOffsets', 'matchedRuns', 'road-context']
  },
  'gpu-network.network-accessibility': {
    liveOnly: ['accessibilityComplete', 'converged'],
    recordedOnly: ['facility-positions', 'gravity-nodes', 'road-context']
  },
  'gpu-network.network-analytics-columns': {
    liveOnly: [],
    recordedOnly: ['degree-nodes', 'road-context']
  },
  'gpu-network.network-centrality': {
    liveOnly: ['converged'],
    recordedOnly: ['closeness-pivots', 'pivot-nodes', 'road-context']
  },
  'gpu-network.network-centrality-schedule-step': {
    liveOnly: ['converged'],
    recordedOnly: ['closeness-pivots', 'pivot-nodes', 'road-context']
  },
  'gpu-network.network-coarsening': {
    liveOnly: ['candidateOverflow', 'overflow', 'superedgeCount'],
    recordedOnly: ['road-context', 'superedgeOffsets', 'superedges', 'supernodes', 'zone-nodes']
  },
  'gpu-network.network-cost-matrix': {
    liveOnly: ['converged', 'startCosts', 'startNodes'],
    recordedOnly: ['matrix-row-nodes', 'road-context']
  },
  'gpu-network.network-cost-table': {
    liveOnly: ['tableCosts', 'tableOffsets', 'tableTargets'],
    recordedOnly: [
      'road-context',
      'rowCosts',
      'rowTargets',
      'start-intersection',
      'table-row-targets',
      'tableRowEnd',
      'tableRowStart'
    ]
  },
  'gpu-network.network-cost-weights': {
    liveOnly: [
      'converged',
      'matrixCosts',
      'network-cost-weights-row-counts',
      'weightComplete',
      'weightDistances',
      'weightNeighbors',
      'weightOffsets',
      'weightValues'
    ],
    recordedOnly: [
      'retained-nodes',
      'road-context',
      'rowNeighbors',
      'rowSeconds',
      'rowWeights',
      'weightRowCounts',
      'weightRowStart'
    ]
  },
  'gpu-network.network-isochrones': {
    liveOnly: ['converged', 'count', 'overflow', 'triangles'],
    recordedOnly: ['isochrone-triangles', 'road-context', 'start-intersection', 'triangle-offsets']
  },
  'gpu-network.network-k-function': {
    liveOnly: ['converged', 'envelope', 'neighbors', 'nodePositions', 'offsets'],
    recordedOnly: ['bandThreshold', 'envelope-maximum', 'envelope-mean', 'envelope-minimum']
  },
  'gpu-network.network-line-graph': {
    liveOnly: ['arcCount', 'overflow'],
    recordedOnly: ['edgeMidpoints', 'legalTurns', 'road-context']
  },
  'gpu-network.network-neighborhood': {
    liveOnly: [],
    recordedOnly: ['ego-network-nodes', 'road-context', 'start-intersection']
  },
  'gpu-network.network-noding': {liveOnly: ['nodingOverflow', 'pieceOverflow'], recordedOnly: []},
  'gpu-network.network-path-extraction': {
    liveOnly: ['candidateOverflow', 'converged', 'count', 'nodeRows', 'overflow'],
    recordedOnly: ['extracted-path', 'road-context', 'start-and-target']
  },
  'gpu-network.network-reachability': {
    liveOnly: [],
    recordedOnly: ['lastImprovedRound', 'reached-nodes', 'road-context', 'start-intersections']
  },
  'gpu-network.network-reachability.chained': {
    liveOnly: [],
    recordedOnly: ['lastQueuedRound', 'reached-nodes', 'road-context', 'start-intersections']
  },
  'gpu-network.network-service-areas': {
    liveOnly: ['converged', 'facilityNodes'],
    recordedOnly: ['facility-nodes', 'road-context', 'served-nodes']
  },
  'gpu-network.network-snapping': {liveOnly: [], recordedOnly: ['road-context']},
  'gpu-network.network-statistics': {
    liveOnly: ['neighbors', 'nodePositions', 'offsets', 'summary'],
    recordedOnly: ['networkStatistics', 'outDegreeHistogram']
  },
  'gpu-network.network-subgraph-filter': {
    liveOnly: [],
    recordedOnly: ['kept-edges', 'keptEdgeOffsets', 'road-context']
  },
  'gpu-network.visibility-graph-recipe': {
    liveOnly: ['extendedIntegration', 'integration', 'metrics', 'normalizedChoice'],
    recordedOnly: ['visibility-centrality']
  }
};

test.each([
  ['gpu-network.network-accessibility', 'accessibilityComplete'],
  ['gpu-network.network-cost-weights', 'weightComplete']
])('LENS-105: %s pins its exact capture difference and rejects certificate drift', async (scenarioId, certificate) => {
  const scenario = scenarios.find(candidate => candidate.id === scenarioId)!;
  const declared = await declareLensScenario(scenario);
  const recorded = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(`../../../../docs/lens-records/${scenarioId}.json`, import.meta.url),
        'utf8'
      )
    )
  ).record;
  const actual = getStructureSignature(declared.record, declared.captureNames);
  // Isolate names here; the separate archive test retains the complete graph/layout comparison.
  const expected = getStructureSignature(declared.record, getRecordedCaptureNames(recorded));
  const exemption = CAPTURE_NAMES_DERIVED_AFTER_READBACK[scenarioId];
  expect(declared.captureNames).toContain(certificate);
  expect(getStructureMismatches(scenarioId, actual, expected, exemption)).toEqual([]);
  for (const names of [
    [...declared.captureNames, 'unexpected-capture'],
    declared.captureNames.filter(name => name !== certificate)
  ]) {
    expect(
      getStructureMismatches(
        scenarioId,
        getStructureSignature(declared.record, names),
        expected,
        exemption
      )
    ).toHaveLength(1);
  }
});

for (const scenario of scenarios) {
  let declaration: ReturnType<typeof declareLensScenario> | undefined;
  const declare = () => (declaration ??= declareLensScenario(scenario));
  test(`${scenario.id}: declares its graph and every choice without a GPU`, async () => {
    const declared = await declare();
    expect(declared.record.graphs.length).toBeGreaterThan(0);
    expect(declared.missingChoices).toEqual([]);
  });

  test(`${scenario.id}: committed record matches the declared observation graph`, async () => {
    // Scenarios with a hand-authored describeStructure (the gpu-network family) take the same
    // archive comparison as every other scenario (LENS-105); their structure checks remain too.
    const recorded = fromRecordedLensFile(
      JSON.parse(
        readFileSync(
          new URL(`../../../../docs/lens-records/${scenario.id}.json`, import.meta.url),
          'utf8'
        )
      )
    ).record;
    const storageBufferLimit = getRecordedStorageBufferLimit(recorded);
    const declared =
      storageBufferLimit === 8
        ? await declare()
        : await declareLensScenario(scenario, storageBufferLimit);
    const actual = getStructureSignature(declared.record, declared.captureNames);
    const expected = getStructureSignature(recorded, getRecordedCaptureNames(recorded));
    expect(
      getStructureMismatches(
        scenario.id,
        actual,
        expected,
        CAPTURE_NAMES_DERIVED_AFTER_READBACK[scenario.id]
      )
    ).toEqual([]);
  });
}

test('capture-name exemptions name only registered network scenarios with hand-authored structure', async () => {
  for (const scenarioId of Object.keys(CAPTURE_NAMES_DERIVED_AFTER_READBACK)) {
    const scenario = scenarios.find(candidate => candidate.id === scenarioId);
    expect(scenario, scenarioId).toBeDefined();
    expect(scenario!.entry).toBe('gpu-network');
    expect(typeof (await scenario!.load()).describeStructure).toBe('function');
  }
});

test('registry-wide declaration gate includes every scenario without exceptions', () => {
  const archived = enumerateArchiveScenarioIds(
    readdirSync(new URL('../../../../docs/lens-records/', import.meta.url))
  );
  expect(
    getScenarioMembershipDifferences(
      scenarios.map(scenario => scenario.id),
      archived
    )
  ).toEqual({missing: [], extra: []});
});

function enumerateArchiveScenarioIds(filenames: readonly string[]) {
  return filenames.filter(name => name.endsWith('.json')).map(name => name.slice(0, -5));
}

test('registry archive enumeration reports an orphan regardless of its namespace', () => {
  // A closed fixture isolates orphan detection from the separate live missing-archive gate.
  const registered = ['gpu-spatial-analysis.fixture', 'gpu-dataframe.fixture'];
  const filenames = registered.map(id => `${id}.json`);
  const inventory = enumerateArchiveScenarioIds([...filenames, 'gpu-core.new-orphan.json']);
  expect(getScenarioMembershipDifferences(registered, inventory)).toEqual({
    missing: ['gpu-core.new-orphan'],
    extra: []
  });
});

function getScenarioMembershipDifferences(actual: readonly string[], expected: readonly string[]) {
  return {
    missing: expected.filter(id => !actual.includes(id)).sort(),
    extra: actual.filter(id => !expected.includes(id)).sort()
  };
}

test('registry membership reports a missing and extra ID even when the count is unchanged', () => {
  expect(getScenarioMembershipDifferences(['first', 'replacement'], ['first', 'second'])).toEqual({
    missing: ['second'],
    extra: ['replacement']
  });
});

test('signature detects kind, resource, capture and order changes with unchanged node IDs', () => {
  const record = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(
          '../../../../docs/lens-records/gpu-spatial-analysis.pair-gather.json',
          import.meta.url
        ),
        'utf8'
      )
    )
  ).record;
  const captureNames = getRecordedCaptureNames(record);
  const signature = getStructureSignature(record, captureNames);
  const graph = record.graphs[0];
  const firstNode = graph.nodes[0];
  for (const changedNode of [
    {...firstNode, kind: firstNode.kind === 'copy' ? ('compute' as const) : ('copy' as const)},
    {...firstNode, reads: ['different-input-resource']},
    {...firstNode, writes: ['different-output-resource']}
  ]) {
    const changed = {
      ...record,
      graphs: [{...graph, nodes: [changedNode, ...graph.nodes.slice(1)]}, ...record.graphs.slice(1)]
    };
    expect(changed.graphs[0].nodes.map(node => node.id)).toEqual(graph.nodes.map(node => node.id));
    expect(getStructureSignature(changed, captureNames)).not.toEqual(signature);
  }
  // LENS-106: shifting one binding by one element, or changing its length, changes the signature.
  const boundNode = graph.nodes.find(node =>
    node.bindings?.some(b => b.elementOffset !== undefined)
  )!;
  const boundIndex = graph.nodes.indexOf(boundNode);
  const bindingIndex = boundNode.bindings!.findIndex(
    binding => binding.elementOffset !== undefined
  );
  for (const change of [
    {elementOffset: boundNode.bindings![bindingIndex].elementOffset! + 1},
    {length: (boundNode.bindings![bindingIndex].length ?? 0) + 1}
  ]) {
    const bindings = boundNode.bindings!.map((binding, index) =>
      index === bindingIndex ? {...binding, ...change} : binding
    );
    const nodes = graph.nodes.map((node, index) =>
      index === boundIndex ? {...node, bindings} : node
    );
    const changed = {...record, graphs: [{...graph, nodes}, ...record.graphs.slice(1)]};
    expect(getStructureSignature(changed, captureNames)).not.toEqual(signature);
    const mismatches = getStructureMismatches(
      record.id,
      getStructureSignature(changed, captureNames),
      signature
    );
    expect(mismatches).toHaveLength(1);
    expect(mismatches[0]).toContain(`binding layout changed: ${boundNode.id}`);
  }
  expect(getStructureSignature(record, [...captureNames, 'new-capture'])).not.toEqual(signature);
  expect(
    getStructureSignature(
      {
        ...record,
        graphs: [{...graph, nodes: [...graph.nodes].reverse()}, ...record.graphs.slice(1)]
      },
      captureNames
    )
  ).not.toEqual(signature);
});

type StructureSignature = ReturnType<typeof getStructureSignature>;

function getStructureMismatches(
  scenarioId: string,
  actual: StructureSignature,
  expected: StructureSignature,
  captureNameExemption?: {liveOnly: readonly string[]; recordedOnly: readonly string[]}
): string[] {
  const mismatches: string[] = [];
  const captureNameDifference = {
    liveOnly: actual.captureNames.filter(name => !expected.captureNames.includes(name)),
    recordedOnly: expected.captureNames.filter(name => !actual.captureNames.includes(name))
  };
  if (captureNameExemption) {
    if (JSON.stringify(captureNameDifference) !== JSON.stringify(captureNameExemption))
      mismatches.push(
        `${scenarioId}: capture names changed from the exempt difference: live-only=${JSON.stringify(captureNameDifference.liveOnly)} recorded-only=${JSON.stringify(captureNameDifference.recordedOnly)}`
      );
  } else if (JSON.stringify(actual.captureNames) !== JSON.stringify(expected.captureNames))
    mismatches.push(
      `${scenarioId}: capture names changed: live=${JSON.stringify(actual.captureNames)} recorded=${JSON.stringify(expected.captureNames)}`
    );
  if (actual.graphs.length !== expected.graphs.length)
    mismatches.push(`${scenarioId}: graph count differs`);
  for (const [index, graph] of actual.graphs.entries()) {
    const recorded = expected.graphs[index];
    const prefix = `${scenarioId}: graph ${graph.id}`;
    if (!recorded || graph.id !== recorded.id) {
      mismatches.push(`${prefix}: graph changed: recorded=${recorded?.id}`);
      continue;
    }
    const liveNodes = new Map(graph.nodes.map(node => [node.id, node]));
    const recordedNodes = new Map(recorded.nodes.map(node => [node.id, node]));
    for (const node of graph.nodes)
      if (!recordedNodes.has(node.id))
        mismatches.push(`${prefix}: recorded node removed: ${node.id}`);
    for (const node of recorded.nodes)
      if (!liveNodes.has(node.id)) mismatches.push(`${prefix}: recorded node added: ${node.id}`);
    const liveOrder = graph.nodes.filter(node => recordedNodes.has(node.id)).map(node => node.id);
    const recordedOrder = recorded.nodes
      .filter(node => liveNodes.has(node.id))
      .map(node => node.id);
    if (JSON.stringify(liveOrder) !== JSON.stringify(recordedOrder))
      mismatches.push(
        `${prefix}: node order changed: live=${JSON.stringify(liveOrder)} recorded=${JSON.stringify(recordedOrder)}`
      );
    for (const node of graph.nodes) {
      const previous = recordedNodes.get(node.id);
      if (!previous) continue;
      if (node.kind !== previous.kind)
        mismatches.push(
          `${prefix}: node kind changed: ${node.id}, live=${node.kind} recorded=${previous.kind}`
        );
      for (const access of ['reads', 'writes'] as const) {
        if (JSON.stringify(node[access]) !== JSON.stringify(previous[access]))
          mismatches.push(
            `${prefix}: ${access === 'reads' ? 'read' : 'write'} resources changed: ${node.id}, live=${JSON.stringify(node[access])} recorded=${JSON.stringify(previous[access])}`
          );
      }
      const comparableBindings = node.bindings.map((binding, index) => {
        const archived = previous.bindings[index];
        const {bindingByteOffset, bindingByteLength, ...logicalBinding} = binding;
        return {
          ...logicalBinding,
          ...(archived?.bindingByteOffset !== undefined ? {bindingByteOffset} : {}),
          ...(archived?.bindingByteLength !== undefined ? {bindingByteLength} : {})
        };
      });
      if (JSON.stringify(comparableBindings) !== JSON.stringify(previous.bindings))
        mismatches.push(
          `${prefix}: binding layout changed: ${node.id}, live=${JSON.stringify(node.bindings)} recorded=${JSON.stringify(previous.bindings)}`
        );
    }
  }
  return mismatches;
}

const structuralDrifts: {name: string; mutate(signature: StructureSignature): void}[] = [
  {
    name: 'node removed',
    mutate: signature => {
      signature.graphs[0].nodes.splice(1, 1);
    }
  },
  {
    name: 'node added',
    mutate: signature => {
      const graph = signature.graphs[0];
      graph.nodes.splice(1, 0, {...structuredClone(graph.nodes[0]), id: 'stale-extra-node'});
    }
  },
  {
    name: 'node order changed',
    mutate: signature => {
      const nodes = signature.graphs[0].nodes;
      [nodes[0], nodes[1]] = [nodes[1], nodes[0]];
    }
  },
  {
    name: 'node kind changed',
    mutate: signature => {
      const node = signature.graphs[0].nodes[0];
      node.kind = node.kind === 'copy' ? 'compute' : 'copy';
    }
  },
  {
    name: 'read resources changed',
    mutate: signature => {
      const node = signature.graphs[0].nodes.find(node => node.reads.length)!;
      node.reads[0] += '-stale';
      node.reads.sort();
    }
  },
  {
    name: 'write resources changed',
    mutate: signature => {
      const node = signature.graphs[0].nodes.find(node => node.writes.length)!;
      node.writes[0] += '-stale';
      node.writes.sort();
    }
  },
  {
    name: 'binding layout changed',
    mutate: signature => {
      const node = signature.graphs[0].nodes.find(node =>
        node.bindings.some(binding => binding.elementOffset !== undefined)
      )!;
      const binding = node.bindings.find(binding => binding.elementOffset !== undefined)!;
      binding.elementOffset = (binding.elementOffset ?? 0) + 1;
    }
  },
  {
    name: 'capture names changed',
    mutate: signature => {
      signature.captureNames[0] += '-stale';
      signature.captureNames.sort();
    }
  }
];

for (const scenarioId of [
  'gpu-spatial-analysis.pair-gather',
  'gpu-spatial-analysis.nearest-feature-join'
]) {
  const record = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(`../../../../docs/lens-records/${scenarioId}.json`, import.meta.url),
        'utf8'
      )
    )
  ).record;
  const signature = getStructureSignature(record, getRecordedCaptureNames(record));
  test(`${scenarioId}: unmodified committed signature passes`, () => {
    expect(getStructureMismatches(scenarioId, signature, structuredClone(signature))).toEqual([]);
  });
  test.each(structuralDrifts)(`${scenarioId}: stale $name is reported without Git`, drift => {
    const stale = structuredClone(signature);
    drift.mutate(stale);
    const mismatches = getStructureMismatches(scenarioId, signature, stale);
    expect(mismatches).toHaveLength(1);
    expect(mismatches[0]).toContain(scenarioId);
    expect(mismatches[0]).toContain(drift.name);
  });
}

test.each([
  ['local-moran', 9],
  ['spatial-regression-recipe', 10]
] as const)('%s replays the recorded fusion limit instead of the core limit', async (name, expectedLimit) => {
  const scenarioId = `gpu-spatial-analysis.${name}`;
  const scenario = scenarios.find(candidate => candidate.id === scenarioId)!;
  const core = await declareLensScenario(scenario);
  // An in-memory higher-limit recording keeps this regression independent of the adapter used
  // by a future recapture of the committed file.
  const recorded = await declareLensScenario(scenario, expectedLimit);
  expect(getRecordedStorageBufferLimit(recorded.record)).toBe(expectedLimit);
  const replay = await declareLensScenario(
    scenario,
    getRecordedStorageBufferLimit(recorded.record)
  );
  const expected = getStructureSignature(recorded.record, recorded.captureNames);
  expect(getStructureSignature(core.record, core.captureNames)).not.toEqual(expected);
  expect(getStructureSignature(replay.record, replay.captureNames)).toEqual(expected);
  const graph = recorded.record.graphs[0];
  const node = graph.nodes.find(candidate => candidate.bindings?.length === expectedLimit)!;
  const resource = node.bindings![0].resource;
  const withOtherBindings = {
    ...recorded.record,
    graphs: [
      {
        ...graph,
        nodes: [
          {
            ...node,
            bindings: [
              ...node.bindings!,
              {name: 'uniformFixture', resource, access: 'uniform' as const},
              {name: 'sampledFixture', resource, access: 'sampled' as const}
            ]
          }
        ]
      }
    ]
  };
  expect(getRecordedStorageBufferLimit(withOtherBindings)).toBe(expectedLimit);
});

test('LENS-106 absolute binding windows remain distinct and legacy fields stay omitted', () => {
  const record = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(
          '../../../../docs/lens-records/gpu-spatial-analysis.pair-gather.json',
          import.meta.url
        ),
        'utf8'
      )
    )
  ).record;
  const graph = record.graphs.find(candidate =>
    candidate.nodes.some(node => node.bindings?.length)
  )!;
  const node = graph.nodes.find(candidate => candidate.bindings?.length)!;
  const binding = node.bindings![0];
  const resource = graph.resources!.find(candidate => candidate.id === binding.resource)!;
  resource.bytes = Math.max(resource.bytes, 512);
  delete binding.bindingByteOffset;
  delete binding.bindingByteLength;
  const legacy = getStructureSignature(record, []);
  const graphIndex = record.graphs.indexOf(graph);
  const nodeIndex = graph.nodes.indexOf(node);
  const layout = () =>
    getStructureSignature(record, []).graphs[graphIndex].nodes[nodeIndex].bindings;
  expect(layout()[0]).not.toHaveProperty('bindingByteOffset');
  expect(layout()[0]).not.toHaveProperty('bindingByteLength');
  binding.elementOffset = 0;
  binding.length = 4;
  binding.bindingByteOffset = 0;
  binding.bindingByteLength = 16;
  expect(records.validateGPUComputationRecord(record).valid).toBe(true);
  const firstWindow = layout();
  const firstDescription = structuredClone(record);
  binding.bindingByteOffset = 256;
  expect(records.validateGPUComputationRecord(record).valid).toBe(true);
  expect(layout()).not.toEqual(firstWindow);
  expect(layout()[0].bindingByteOffset).toBe(256);
  expect(layout()[0].elementOffset).toBe(0);
  // Both valid descriptions differ only in the aligned window offset of this one binding.
  const secondDescription = structuredClone(record);
  secondDescription.graphs[graphIndex].nodes[nodeIndex].bindings![0].bindingByteOffset = 0;
  expect(secondDescription).toEqual(firstDescription);
  const secondWindow = layout();
  binding.bindingByteLength = 32;
  expect(records.validateGPUComputationRecord(record).valid).toBe(true);
  expect(layout()).not.toEqual(secondWindow);
  expect(getStructureSignature(record, [])).not.toEqual(legacy);
});

test('LENS-106 archive comparison checks each physical field only when archived', () => {
  const record = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(
          '../../../../docs/lens-records/gpu-spatial-analysis.pair-gather.json',
          import.meta.url
        ),
        'utf8'
      )
    )
  ).record;
  const binding = record.graphs.flatMap(graph => graph.nodes).find(node => node.bindings?.length)!
    .bindings![0];
  delete binding.bindingByteOffset;
  delete binding.bindingByteLength;
  const legacy = getStructureSignature(record, []);
  binding.bindingByteOffset = 0;
  binding.bindingByteLength = 16;
  const withWindow = getStructureSignature(record, []);
  expect(getStructureMismatches(record.id, withWindow, legacy)).toEqual([]);
  binding.bindingByteOffset = 256;
  const shifted = getStructureSignature(record, []);
  expect(getStructureMismatches(record.id, shifted, legacy)).toEqual([]);
  expect(getStructureMismatches(record.id, shifted, withWindow).join('\n')).toContain(
    'binding layout changed'
  );
  // Offset-only archives must still check their offset, while ignoring new length evidence.
  delete binding.bindingByteLength;
  const offsetOnly = getStructureSignature(record, []);
  expect(getStructureMismatches(record.id, shifted, offsetOnly)).toEqual([]);
  expect(getStructureMismatches(record.id, withWindow, offsetOnly).join('\n')).toContain(
    'binding layout changed'
  );
  binding.bindingByteOffset = 0;
  binding.bindingByteLength = 32;
  const longer = getStructureSignature(record, []);
  expect(getStructureMismatches(record.id, longer, withWindow).join('\n')).toContain(
    'binding layout changed'
  );
});
