// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync, statSync} from 'node:fs';
import {afterEach, describe, expect, it, vi} from 'vitest';
import {ComputePipeline, type ComputePipelineProps} from '@luma.gl/core';
import {CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {NullDevice} from '@luma.gl/core/test-utils';
import * as graphExports from '@luma.gl/graph/gpu-graph-algorithms';
import {
  describeGPUCommandGraph,
  validateGPUComputationRecord,
  validateGPUComputationSidecars
} from '@luma-contributors/analysis/computation-record';
import * as snapshotModule from '../../../src/computation-lenses/lens-view-snapshot';
import {
  extractLensExample,
  findNonPublicImports,
  getLensExamplePath,
  listLensScenarios
} from '../../../src/computation-lenses';
import {
  GPU_GRAPH_COMPUTATION_PACK,
  GPU_GRAPH_FORMULAS
} from '../../../src/computation-lenses/scenarios/gpu-graph/computation-pack';
import {createRailGraphFixture} from '../../../src/computation-lenses/scenarios/gpu-graph/rail-graph-fixture';
import {
  buildGPUGraphScenario,
  compileGPUGraphScenario,
  getGPUGraphCaptureView,
  recordGPUGraphScenario
} from '../../../src/computation-lenses/scenarios/gpu-graph/gpu-graph-scenario';
import {runExample} from '../../../src/computation-lenses/scenarios/gpu-graph/graph-page-rank.example';
import railData from '../../../src/computation-lenses/data/g-nl-rail-graph.json';
import '../../../src/computation-lenses/scenarios/gpu-graph/index';

afterEach(() => vi.restoreAllMocks());

const scenarios = listLensScenarios().filter(scenario => scenario.id.startsWith('gpu-graph.'));
const sourceRoot = new URL('../../../src/computation-lenses/', import.meta.url);

// Match the gpu-graph recording specs: the null backend supplies allocation and shader objects,
// while WebGPU capability metadata and a no-op pipeline permit construction without a GPU.
class StructurePipeline extends ComputePipeline {
  readonly handle = null;
  constructor(
    readonly device: NullDevice,
    props: ComputePipelineProps
  ) {
    super(device, props);
  }
  setBindings(): void {}
}

function createStructureDevice(id: string): NullDevice {
  const device = new NullDevice({id});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxStorageBuffersPerShaderStage: 8
    }
  });
  vi.spyOn(device, 'createComputePipeline').mockImplementation(
    props => new StructurePipeline(device, props!)
  );
  return device;
}

describe('gpu-graph lens substrate', () => {
  it('preserves the real, compact CC0 station and edge dataset', () => {
    expect(statSync(new URL('data/g-nl-rail-graph.json', sourceRoot)).size).toBeLessThanOrEqual(
      150_000
    );
    expect(railData.stationNames).toHaveLength(485);
    expect(railData.lonLat).toHaveLength(970);
    expect(railData.edges.source).toHaveLength(1411);
    expect(railData.edges.target).toHaveLength(1411);
    expect(railData.edges.travelTimeSeconds).toHaveLength(1411);
    expect(railData.stationNames[409]).toBe('Utrecht Centraal');
    expect(railData.licence).toBe('CC0');
    expect(railData.serviceDate).toBe('2026-10-09');
    expect(railData.edges.travelTimeSeconds.every(Number.isInteger)).toBe(true);
    expect(railData.lonLat.every(value => value === Number(value.toFixed(5)))).toBe(true);
  });

  it('deduplicates undirected source pairs and reserves complete symmetric weighted CSR', () => {
    const device = createStructureDevice('rail-undirected');
    const submit = vi.spyOn(device, 'submit');
    const fixture = createRailGraphFixture(device, {
      id: 'rail-undirected',
      directed: false,
      weights: true
    });
    try {
      const pairs = new Set(
        railData.edges.source.flatMap((source, index) => {
          const target = railData.edges.target[index];
          return source === target
            ? []
            : [`${Math.min(source, target)}:${Math.max(source, target)}`];
        })
      );
      expect(fixture.edgeCount).toBe(pairs.size);
      expect(fixture.graph.directed).toBe(false);
      expect(fixture.railTopology.reverse).toBeUndefined();
      expect(fixture.railTopology.forward.neighbors.length).toBe(pairs.size * 2);
      expect(fixture.railTopology.forward.edgeWeights?.length).toBe(pairs.size * 2);
      expect(fixture.graph.edgeWeights?.length).toBe(pairs.size);
      expect(fixture.railTopology.forward.offsets.length).toBe(486);
      expect(fixture.railTopology.invalidEdgeCount.length).toBe(1);
      expect(submit).not.toHaveBeenCalled();
    } finally {
      fixture.destroy();
      device.destroy();
    }
  });

  it('reuses the actual imported output handle and destroys owned columns exactly once', async () => {
    const device = createStructureDevice('rail-output-handle');
    const submit = vi.spyOn(device, 'submit');
    const {fixture, outputs} = await buildGPUGraphScenario(device, {
      id: 'rail-output-handle',
      directed: true,
      reverse: true,
      example: runExample
    });
    const destroyRanks = vi.spyOn(outputs.ranks, 'destroy');
    const destroyBuffer = vi.spyOn(outputs.ranks.data[0].buffer, 'destroy');
    const first = getGPUGraphCaptureView(fixture, 'observation-a', outputs.ranks);
    const second = getGPUGraphCaptureView(fixture, 'observation-b', outputs.ranks);
    expect(first.buffer).toBe(second.buffer);
    const compiled = fixture.commandGraph.compile();
    try {
      const description = describeGPUCommandGraph(compiled, {domain: 'gpu-graph'});
      expect(first.buffer.id).toBe('gpu-graph-page-rank-output');
      // Published nodes carry no `op` for this algorithm; the node id names it instead.
      expect(
        description.graph.nodes.some(
          node => node.id.includes('page-rank') && node.writes.includes(first.buffer.id)
        )
      ).toBe(true);
      expect(fixture.railTopology.reverse?.neighbors.length).toBe(1411);
      expect(submit).not.toHaveBeenCalled();
    } finally {
      compiled.destroy();
      fixture.destroy();
      fixture.destroy();
      device.destroy();
    }
    expect(destroyRanks).toHaveBeenCalledTimes(1);
    expect(destroyBuffer).toHaveBeenCalledTimes(1);
  });
});

describe('gpu-graph scenario contracts', () => {
  it('registers at least the PageRank worked example', () => {
    expect(scenarios.some(scenario => scenario.element === 'GPUGraphAlgorithmsPageRank')).toBe(
      true
    );
  });
  for (const scenario of scenarios) {
    it(`${scenario.id} has a public example, pack entry and coherent descriptor`, () => {
      expect(scenario.entry).toBe('gpu-graph');
      expect(scenario.element in graphExports).toBe(true);
      expect(scenario.audience).toBe('engineer');
      expect(scenario.defaultRung).toBe('result');
      expect(scenario.rungs).toContain('result');
      expect(scenario.dataset).toMatchObject({id: 'g-nl-rail-graph', licence: 'CC0'});
      expect(GPU_GRAPH_COMPUTATION_PACK.operators?.[scenario.element]).toBeDefined();
      const formula = GPU_GRAPH_FORMULAS[scenario.element];
      expect(formula).toBeDefined();
      expect(scenario.rungs.includes('formula')).toBe(
        !('formula' in formula) && formula.derivation === 'curated'
      );
      const source = readFileSync(
        new URL(scenario.example?.path ?? getLensExamplePath(scenario.id), sourceRoot),
        'utf8'
      );
      expect(extractLensExample(source).lineCount).toBeLessThanOrEqual(25);
      expect(findNonPublicImports(source)).toEqual([]);
      expect(source).toMatch(/export async function runExample\b/);
    });

    it(`${scenario.id} describes its compiled graph and binds curated terms to resources`, async () => {
      const device = createStructureDevice(`${scenario.id}-structure`);
      const submit = vi.spyOn(device, 'submit');
      try {
        const loaded = await scenario.load();
        expect(loaded.describeStructure).toBeDefined();
        const record = await loaded.describeStructure!(device);
        expect(submit).not.toHaveBeenCalled();
        expect(validateGPUComputationRecord(record).issues).toEqual([]);
        const graph = record.graphs.find(candidate => candidate.form === 'compiled')!;
        // Published contributors set no `op` on their nodes, so the scenario's own nodes are found
        // by the element name in the node id (the topology scenario names them `topology-*`).
        const elementName = scenario.id.replace('gpu-graph.graph-', '');
        expect(graph.nodes.filter(node => node.id.includes(elementName)).length).toBeGreaterThan(0);
        // The end-of-run snapshot tap is the last node of the observation graph.
        expect(graph.nodes.at(-1)?.id).toMatch(/-snapshot-tap$/);
        const formula = GPU_GRAPH_FORMULAS[scenario.element];
        if ('formula' in formula) return;
        for (const symbol of formula.symbols) {
          // Dropped coverage: `wgsl:` bindings are no longer verified against recorded shader text,
          // because published luma.gl contributors expose no source. Resource bindings are checked.
          expect(symbol.binding, symbol.id).toMatch(/^(wgsl|resource):.+/);
          const [kind, ...parts] = symbol.binding!.split(':');
          if (kind === 'resource') {
            const identifier = parts.join(':');
            expect(
              graph.resources?.some(resource => resource.id === identifier),
              symbol.binding
            ).toBe(true);
          }
        }
        const termIds = [...formula.tex.matchAll(/\\term\{([^}]+)\}/g)].map(match => match[1]);
        expect(termIds.length).toBeGreaterThan(0);
        for (const identifier of termIds)
          expect(formula.symbols.some(symbol => symbol.id === identifier)).toBe(true);
      } finally {
        device.destroy();
      }
    });
  }
});

describe('gpu-graph recorder plumbing (mock readback)', () => {
  it('records one end-of-run snapshot with real capture identities, geometry, highlight and residual', async () => {
    const device = createStructureDevice('record-plumbing');
    // Only the submission and readback transport is mocked; these synthetic values never leave tests.
    const submit = vi.spyOn(snapshotModule, 'submitLensGraph').mockReturnValue(undefined as never);
    const destroyCompiled = vi.spyOn(CompiledGPUCommandGraph.prototype, 'destroy');
    vi.spyOn(snapshotModule, 'readLensViewSnapshot').mockImplementationOnce(
      async (snapshot, props) =>
        Object.entries(snapshot.resources).map(([name, resource]) => {
          const components = name === 'station-positions' ? 2 : 1;
          const length = name === 'csr-offsets' ? 486 : name === 'csr-neighbors' ? 1411 : 485;
          const values = new Float32Array(length * components);
          if (name === 'page-rank') values[409] = 100;
          if (name === 'page-rank-residual') values[0] = 0.125;
          return {
            id: name,
            name,
            graph: props.graphId,
            resource,
            afterNode: snapshot.tapNodeId,
            elementType: 'f32' as const,
            values,
            components,
            provenance: 'gpu-readback' as const
          };
        })
    );
    try {
      const {run} = await import(
        '../../../src/computation-lenses/scenarios/gpu-graph/graph-page-rank.run'
      );
      const record = await run(device);
      expect(submit).toHaveBeenCalledOnce();
      expect(destroyCompiled).toHaveBeenCalledOnce();
      expect(validateGPUComputationSidecars(record, record).issues).toEqual([]);
      const snapshots = record.captures!.snapshots;
      expect(snapshots.filter(snapshot => snapshot.name === 'page-rank')).toHaveLength(1);
      expect(snapshots.find(snapshot => snapshot.id === 'csr-offsets')).toMatchObject({
        role: 'input',
        geometry: {
          kind: 'graph',
          offsetsCapture: 'csr-offsets',
          targetsCapture: 'csr-neighbors',
          layoutCapture: 'station-positions',
          coordinateModel: 'geographic'
        }
      });
      expect(snapshots.find(snapshot => snapshot.id === 'page-rank-stations')).toMatchObject({
        role: 'output',
        components: 2,
        keySpace: 'vertex',
        geometry: {kind: 'points', coordinateModel: 'geographic', valuesCapture: 'page-rank'},
        highlight: {key: 409, label: 'Utrecht Centraal: rank 1'}
      });
      expect(snapshots.every(snapshot => snapshot.provenance === 'gpu-readback')).toBe(true);
      expect(record.metrics![0].encoding).toEqual({
        coalesced: false,
        timestamped: false,
        taps: 1,
        lensTwin: true,
        retained: []
      });
      expect(record.metrics![0].nodes).toEqual({});
      expect(record.metrics![0].series).toEqual([
        {name: 'final L1 residual', axis: 'publication', values: [0.125]}
      ]);
    } finally {
      device.destroy();
    }
  });

  it('always releases the snapshot arena, compiled graph and fixture when submission fails', async () => {
    const device = createStructureDevice('failed-record');
    const fixture = createRailGraphFixture(device, {id: 'failed-record', directed: true});
    // A real snapshot arena and tap, so the cleanup path frees the same objects as production.
    const {compiled, snapshot} = compileGPUGraphScenario(fixture, {});
    const destroyArena = vi.spyOn(snapshot.arena, 'destroy');
    const destroyFixture = vi.spyOn(fixture, 'destroy');
    const destroyCompiled = vi.spyOn(compiled, 'destroy');
    vi.spyOn(snapshotModule, 'submitLensGraph').mockImplementationOnce(() => {
      throw new Error('readback failed');
    });
    try {
      await expect(
        recordGPUGraphScenario({
          scenarioId: 'gpu-graph.failed-record',
          element: 'GPUGraphAlgorithmsTopology',
          fixture,
          compiled,
          snapshot,
          outputCaptures: {},
          choices: []
        })
      ).rejects.toThrow('readback failed');
      expect(destroyArena).toHaveBeenCalled();
      expect(destroyCompiled).toHaveBeenCalledOnce();
      expect(destroyFixture).toHaveBeenCalledOnce();
    } finally {
      device.destroy();
    }
  });
});
