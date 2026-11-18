// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {fromRecordedLensFile, listLensScenarios} from '../../src/computation-lenses/index';
import {
  declareLensScenario,
  getRecordedCaptureNames,
  getRecordedStorageBufferLimit,
  getStructureSignature
} from './scenario-declaration.test-utils';
import '../../src/computation-lenses/scenarios/index';

const scenarios = listLensScenarios();

for (const scenario of scenarios) {
  let declaration: ReturnType<typeof declareLensScenario> | undefined;
  const declare = () => (declaration ??= declareLensScenario(scenario));
  test(`${scenario.id}: declares its graph and every choice without a GPU`, async () => {
    const declared = await declare();
    expect(declared.record.graphs.length).toBeGreaterThan(0);
    expect(declared.missingChoices).toEqual([]);
  });

  test(`${scenario.id}: committed record matches the declared observation graph`, async () => {
    const loaded = await scenario.load();
    if (loaded.describeStructure) {
      // Preserve the existing hand-authored Node/headless structure checks unchanged.
      return;
    }
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
    expect(getStructureMismatches(scenario.id, actual, expected)).toEqual([]);
  });
}

test('registry-wide declaration gate includes every scenario without exceptions', () => {
  expect(scenarios.length).toBeGreaterThan(400);
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
  expected: StructureSignature
): string[] {
  const mismatches: string[] = [];
  if (JSON.stringify(actual.captureNames) !== JSON.stringify(expected.captureNames))
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
