// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {readFileSync} from 'node:fs';
import {GPUMapMatching} from '../../../src/gpu-network/map-matching/gpu-map-matching';
import {
  computeMapMatchingOnCPU,
  CPUMapMatchingOperation
} from '../../../src/gpu-network/map-matching/cpu-map-matching-operation';
import {
  GPUMapMatchingOperation,
  type MapMatchingOperationProps
} from '../../../src/gpu-network/map-matching/gpu-map-matching-operation';
import {addTrackQualityRecipe} from '../../../src/gpu-network/recipes/track-quality-recipe';
import {GPUMapMatchedEdgeVolumes} from '../../../src/gpu-network/flow-aggregation/gpu-map-matched-edge-volumes';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {getResourceBuffer} from '../network-resource-buffers';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {
  createFinalMatchingInput,
  createParallelRoadMatchingInput
} from './map-matching-final-review-fixture';
import {createMatchingFixture, getMatchingGraphInputs} from './map-matching-operation-fixture';

it('network:G1 refuses traversal input without a completeness declaration', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = (name: string, length: number) => createTransientView(graph, name, 'uint32', length);
  expect(
    () =>
      new GPUMapMatchedEdgeVolumes({
        // @ts-expect-error missing completeness is rejected for JavaScript callers too
        traversedEdges: {edges: view('edges', 16), counts: view('counts', 1)},
        trackCount: 1,
        traversedEdgeCapacity: 16,
        edgeCount: 11,
        output: {
          edgeVolumes: view('volumes', 11),
          status: {candidateOverflow: view('incomplete', 1)}
        }
      })
  ).toThrow(/complet/);
});

it('network:G1 fixed external sequences explicitly assert known completeness', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = (name: string, length: number) => createTransientView(graph, name, 'uint32', length);
  const edges = view('edges', 2),
    counts = view('counts', 1),
    loss = view('loss', 1);
  const nodes = new GPUMapMatchedEdgeVolumes({
    traversedEdges: {edges, counts, completeness: 'known-complete'},
    trackCount: 1,
    traversedEdgeCapacity: 2,
    edgeCount: 2,
    output: {edgeVolumes: view('volumes', 2), status: {candidateOverflow: loss}}
  }).getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  memory.write(edges, Uint32Array.of(0, 1));
  for (const count of [3, 2]) {
    memory.write(counts, Uint32Array.of(count));
    executeCommandNodesOnCPU(nodes, memory);
    expect(memory.readUint32(loss)[0]).toBe(Number(count > 2));
  }
});

it('network:G1 transition status inherits traversal loss and clears it on reuse', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = (name: string, length: number) => createTransientView(graph, name, 'uint32', length);
  const edges = view('edges', 16),
    counts = view('counts', 1),
    loss = view('loss', 1);
  const upstreamOverflow = view('upstream-overflow', 1),
    volumeLoss = view('volume-loss', 1),
    transitionLoss = view('transition-loss', 1);
  const nodes = new GPUMapMatchedEdgeVolumes({
    traversedEdges: {edges, counts, status: {overflow: upstreamOverflow, candidateOverflow: loss}},
    trackCount: 1,
    traversedEdgeCapacity: 16,
    edgeCount: 11,
    transitionPairCapacity: 32,
    output: {
      edgeVolumes: view('volumes', 11),
      status: {candidateOverflow: volumeLoss},
      transitions: {
        ids: view('ids', 16),
        status: {
          count: view('transition-count', 1),
          overflow: view('transition-overflow', 1),
          candidateOverflow: transitionLoss
        }
      }
    }
  }).getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  memory.write(edges, Uint32Array.of(0, 0xffffffff, 10));
  memory.write(counts, Uint32Array.of(3));
  for (const flag of [1, 0]) {
    memory.write(loss, Uint32Array.of(flag));
    // The nested compact publisher's own pair-loss flag starts clean. Execute it and the
    // volume status kernels without the unrelated transition hash/scan implementation.
    executeCommandNodesOnCPU(
      nodes.filter(
        node =>
          !node.id.includes('-transitions-') ||
          node.resources?.some(binding => getResourceBuffer(binding) === transitionLoss)
      ),
      memory
    );
    expect(memory.readUint32(volumeLoss)[0]).toBe(flag);
    expect(memory.readUint32(transitionLoss)[0]).toBe(flag);
  }
});

it.each([
  false,
  true
])('network:G2 rejects aliased input/output loss for class status=%s', classStatus => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = (name: string, length: number) => createTransientView(graph, name, 'uint32', length);
  const loss = view('loss', 1);
  class OutputStatus {
    get candidateOverflow() {
      return loss;
    }
  }
  expect(
    () =>
      new GPUMapMatchedEdgeVolumes({
        traversedEdges: {
          edges: view('edges', 16),
          counts: view('counts', 1),
          status: {overflow: view('upstream-overflow', 1), candidateOverflow: loss}
        },
        trackCount: 1,
        traversedEdgeCapacity: 16,
        edgeCount: 11,
        output: {
          edgeVolumes: view('volumes', 11),
          status: classStatus ? new OutputStatus() : {candidateOverflow: loss}
        }
      })
  ).toThrow(/map-matched-edge-volumes outputs must not share buffers with inputs/);
});

it('network:G2 lowering uses the one captured class status view after its getter changes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = (name: string, length: number) => createTransientView(graph, name, 'uint32', length);
  const edges = view('edges', 2),
    counts = view('counts', 1),
    inputLoss = view('input-loss', 1),
    outputLoss = view('output-loss', 1);
  let selected = outputLoss,
    reads = 0;
  class OutputStatus {
    get candidateOverflow() {
      reads++;
      return selected;
    }
  }
  const producer = new GPUMapMatchedEdgeVolumes({
    traversedEdges: {edges, counts, status: {candidateOverflow: inputLoss}},
    trackCount: 1,
    traversedEdgeCapacity: 2,
    edgeCount: 2,
    output: {edgeVolumes: view('volumes', 2), status: new OutputStatus()}
  });
  selected = inputLoss;
  const memory = createCPUGraphMemory();
  memory.write(inputLoss, Uint32Array.of(1));
  memory.write(edges, Uint32Array.of(0, 1));
  memory.write(counts, Uint32Array.of(2));
  executeCommandNodesOnCPU(producer.getCommandNodes(graph), memory);
  expect(reads).toBe(1);
  expect(memory.readUint32(outputLoss)[0]).toBe(1);
  expect(memory.readUint32(inputLoss)[0]).toBe(1);
});

it.each([
  false,
  true
])('network:G7 CPU matching retains own/inherited route budget (inherited=%s)', async inherited => {
  const input = {...createFinalMatchingInput(), routeTable: undefined};
  let budget = 8;
  const props: MapMatchingOperationProps = Object.assign(
    inherited
      ? Object.create({
          get routeNodeBudget() {
            return budget;
          }
        })
      : {},
    {
      pointCount: 2,
      trackCount: 1,
      nodeCount: 12,
      edgeCount: 11,
      positions: 'positions',
      trackOffsets: 'trackOffsets',
      parameters: 'parameters',
      topology: {offsets: 'offsets', neighbors: 'neighbors', nodePositions: 'nodePositions'},
      cellSize: 1,
      candidateCount: 1,
      bounds: {minimum: [-1, -1], maximum: [12, 1]},
      output: {matchedEdges: 'edges', breaks: 'breaks', status: {candidateOverflow: 'loss'}}
    }
  );
  if (!inherited) props.routeNodeBudget = budget;
  const operation = new CPUMapMatchingOperation(props);
  budget = 16;
  if (!inherited) props.routeNodeBudget = budget;
  const graphInputs = getMatchingGraphInputs(input);
  const inputs = Object.fromEntries(
    Object.entries(operation.inputs).map(([port, value]) => {
      const row = graphInputs[value];
      const length = row.values.length / (row.format === 'float32x2' ? 2 : 1);
      return [
        port,
        {format: row.format, length, chunks: [{values: row.values, length}], pack: () => row.values}
      ];
    })
  );
  const result = await operation.execute({inputs});
  expect(Array.from(result['breaks'] as Uint32Array)).toEqual([0, 2]);
  expect(Array.from(result['loss'] as Uint32Array)).toEqual([1]);
});

it('network:G7 contributor lowering retains the validated budget after mutation', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `budget-${serial++}`, format, length);
  const props = {
    positions: view('float32x2', 2),
    trackOffsets: view('uint32', 2),
    parameters: view('float32', 8),
    topology: {
      offsets: view('uint32', 13),
      neighbors: view('uint32', 11),
      nodePositions: view('float32x2', 12)
    },
    cellSize: 1,
    bounds: {minimum: [-1, -1] as const, maximum: [12, 1] as const},
    candidateCount: 1,
    routeNodeBudget: 8,
    output: {matchedEdges: view('uint32', 2)}
  };
  const operation = new GPUMapMatching(props);
  props.routeNodeBudget = 0;
  expect(
    getParsedKernelSources(operation.getCommandNodes(graph)).get('map-matching-routes')
  ).toContain('const S: u32 = 8u;');
});

it('network:G7 GPU operation retains named budgets and ports for deferred lowering', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const {props, input} = createMatchingFixture();
  props.routeNodeBudget = 8;
  const operation = new GPUMapMatchingOperation(props);
  props.routeNodeBudget = 0;
  props.output.matchedEdges = 'mutatedEdges';
  const rows = getMatchingGraphInputs(input);
  const views: Record<string, GraphDataView> = {};
  for (const [port, name] of Object.entries(operation.inputs)) {
    const row = rows[name];
    views[port] = createTransientView(
      graph,
      port,
      row.format,
      row.values.length / (row.format === 'float32x2' ? 2 : 1)
    );
  }
  for (const [port, output] of Object.entries(operation.outputs))
    views[port] = createTransientView(graph, port, output.format, output.length);
  const matching = operation.createContributor(port => views[port] as never) as GPUMapMatching;
  expect(matching.props.routeNodeBudget).toBe(8);
  expect(matching.props.output.matchedEdges).toBe(views['matchedEdges']);
});

it('network:G7 track quality recipe retains an inherited matching budget exactly once', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0,
    reads = 0,
    budget = 8;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `recipe-${serial++}`, format, length);
  const model = Object.assign(
    Object.create({
      get routeNodeBudget() {
        reads++;
        return budget;
      }
    }),
    {
      topology: {
        offsets: view('uint32', 13),
        neighbors: view('uint32', 11),
        nodePositions: view('float32x2', 12)
      },
      parameters: view('float32', 8),
      cellSize: 1,
      candidateCount: 1,
      bounds: {minimum: [-1, -1], maximum: [12, 1]}
    }
  );
  const result = addTrackQualityRecipe(graph, {
    trajectory: {
      positions: view('float32x2', 2),
      trackOffsets: view('uint32', 2),
      timestamps: view('float32', 2)
    },
    thresholds: view('float32', 2),
    mapMatching: model
  });
  budget = 0;
  const matching = result.contributors.find(
    contributor => contributor instanceof GPUMapMatching
  ) as GPUMapMatching;
  expect(reads).toBe(1);
  expect(matching.props.routeNodeBudget).toBe(8);
});

it('network:G8 counts are conditional on the retained match after scoring loss', () => {
  const input = createParallelRoadMatchingInput();
  const complete = computeMapMatchingOnCPU(input, {
    cellSize: 1,
    candidateCount: 2,
    routeNodeBudget: 16,
    traversedEdgeCapacity: 16
  });
  const limited = computeMapMatchingOnCPU(input, {
    cellSize: 1,
    candidateCount: 2,
    routeNodeBudget: 8,
    traversedEdgeCapacity: 16
  });
  expect(Array.from(complete.traversedEdges.slice(0, complete.traversedCounts[0]))).toEqual(
    Array.from({length: 11}, (_, row) => row)
  );
  expect(Array.from(limited.traversedEdges.slice(0, limited.traversedCounts[0]))).toEqual([
    11, 12, 13
  ]);
  expect(limited.candidateOverflow[0]).toBe(1);
  const alternativeRoadCounts: number[] = [];
  for (const match of [complete, limited]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const view = (name: string, length: number) =>
      createTransientView(graph, name, 'uint32', length);
    const edges = view('edges', 16),
      counts = view('counts', 1),
      loss = view('loss', 1),
      volumes = view('volumes', 14);
    const nodes = new GPUMapMatchedEdgeVolumes({
      traversedEdges: {edges, counts, status: {candidateOverflow: loss}},
      trackCount: 1,
      traversedEdgeCapacity: 16,
      edgeCount: 14,
      output: {edgeVolumes: volumes, status: {candidateOverflow: view('volume-loss', 1)}}
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(edges, match.traversedEdges);
    memory.write(counts, match.traversedCounts);
    memory.write(loss, match.traversedCandidateOverflow);
    executeCommandNodesOnCPU(nodes, memory);
    alternativeRoadCounts.push(memory.readUint32(volumes)[11]);
  }
  expect(alternativeRoadCounts).toEqual([0, 1]);
  expect(readFileSync('docs/api/gpu-network/map-matched-edge-volumes-mob05.md', 'utf8')).toContain(
    'conditional on the retained match'
  );
});

it('network:G11 certified misses search only the CSR row regardless of spare capacity', () => {
  const searchSpans: number[] = [];
  for (const capacity of [16, 4_000_000]) {
    let searched = 0;
    const storage = new Uint32Array(capacity);
    storage.set([0, 1, 1, 2, 3, 3]);
    const costs = new Float32Array(capacity);
    costs.set([0, 1, 0, 0, 1, 0]);
    const targets = new Proxy(storage, {
      get(target, key) {
        if (key === 'indexOf')
          return (value: number, from = 0) => {
            searched += target.length - from;
            return target.indexOf(value, from);
          };
        if (typeof key === 'string' && /^\d+$/.test(key)) searched++;
        const value = Reflect.get(target, key, target);
        return typeof value === 'function' ? value.bind(target) : value;
      }
    });
    const result = computeMapMatchingOnCPU(
      {
        positions: Float32Array.from({length: 128}, (_, component) =>
          component % 2 === 0 ? 0.5 : (((component - 1) / 2) % 2) * 2
        ),
        trackOffsets: Uint32Array.of(0, 64),
        parameters: createFinalMatchingInput().parameters,
        topology: {
          offsets: Uint32Array.of(0, 1, 1, 2, 2),
          neighbors: Uint32Array.of(1, 3),
          nodePositions: Float32Array.of(0, 0, 1, 0, 0, 2, 1, 2)
        },
        routeTable: {
          offsets: Uint32Array.of(0, 2, 3, 5, 6),
          targets,
          costs,
          costLimit: Float32Array.of(40),
          complete: Uint32Array.of(1)
        }
      },
      {cellSize: 1, candidateCount: 1}
    );
    expect(Array.from(result.breaks)).toEqual([0, ...new Array(63).fill(1)]);
    expect(searched, `capacity ${capacity}`).toBeLessThanOrEqual(256);
    searchSpans.push(searched);
  }
  expect(searchSpans[1]).toBe(searchSpans[0]);
});

it('network:F6 binds an explicit CPU route budget when the table certificate is cleared', () => {
  const input = createFinalMatchingInput();
  const options = {cellSize: 1, candidateCount: 1, routeNodeBudget: 8, traversedEdgeCapacity: 16};
  const certified = computeMapMatchingOnCPU(input, options);
  expect(certified.breaks).toEqual(Uint32Array.of(0, 0));
  expect(certified.traversedEdges.slice(0, 3)).toEqual(Uint32Array.of(0, 0xffffffff, 10));
  expect(certified.traversedOverflow[0]).toBe(0);
  expect(certified.traversedCandidateOverflow[0]).toBe(1);
  input.routeTable.complete[0] = 0;
  const fallback = computeMapMatchingOnCPU(input, options);
  expect(fallback.breaks).toEqual(Uint32Array.of(0, 2));
  expect(fallback.routeOverflow[0]).toBeGreaterThan(0);
  expect(fallback.overflow[0]).toBe(0);
  expect(fallback.candidateOverflow[0]).toBe(1);
  expect(fallback.traversedCandidateOverflow[0]).toBe(1);
});

it('network:F1 propagates traversal loss to volume status and clears it on reuse', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = (name: string, length: number) => createTransientView(graph, name, 'uint32', length);
  const edges = view('edges', 16),
    counts = view('counts', 1),
    candidateOverflow = view('lost', 1);
  const edgeVolumes = view('volumes', 11),
    incomplete = view('incomplete', 1);
  const nodes = new GPUMapMatchedEdgeVolumes({
    traversedEdges: {edges, counts, status: {candidateOverflow}},
    trackCount: 1,
    traversedEdgeCapacity: 16,
    edgeCount: 11,
    output: {edgeVolumes, status: {candidateOverflow: incomplete}}
  }).getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  memory.write(edges, Uint32Array.from([0, 0xffffffff, 10, ...new Array(13).fill(0)]));
  memory.write(counts, Uint32Array.of(3));
  memory.write(candidateOverflow, Uint32Array.of(1));
  executeCommandNodesOnCPU(nodes, memory);
  expect(memory.readUint32(incomplete)).toEqual(Uint32Array.of(1));
  memory.write(edges, Uint32Array.from([0, 1, ...new Array(14).fill(0)]));
  memory.write(counts, Uint32Array.of(2));
  memory.write(candidateOverflow, Uint32Array.of(0));
  executeCommandNodesOnCPU(nodes, memory);
  expect(memory.readUint32(incomplete)).toEqual(Uint32Array.of(0));
});

it('network:F7 route drops write candidateOverflow and unrecovered hops emit a separator', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `final-${serial++}`, format, length);
  const overflow = view('uint32', 1),
    candidateOverflow = view('uint32', 1);
  const nodes = new GPUMapMatching({
    positions: view('float32x2', 2),
    trackOffsets: view('uint32', 2),
    parameters: view('float32', 8),
    topology: {
      offsets: view('uint32', 13),
      neighbors: view('uint32', 11),
      nodePositions: view('float32x2', 12)
    },
    cellSize: 1,
    bounds: {minimum: [-1, -1], maximum: [12, 1]},
    candidateCount: 1,
    routeNodeBudget: 8,
    traversedEdgeCapacity: 16,
    output: {
      matchedEdges: view('uint32', 2),
      status: {overflow, candidateOverflow},
      traversedEdges: {edges: view('uint32', 16), counts: view('uint32', 1)}
    }
  }).getCommandNodes(graph);
  const bindings =
    nodes
      .find(node => node.id === 'map-matching-route-status')
      ?.resources?.map(getResourceBuffer) ?? [];
  expect(bindings).toContain(candidateOverflow);
  expect(bindings).not.toContain(overflow);
  const source = getParsedKernelSources(nodes).get('map-matching-route-edges') ?? '';
  expect(source.slice(source.indexOf('if (!isRecovered)'))).toContain('emitEdge(index, NONE);');
});
