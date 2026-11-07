// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  cellToVertexes,
  originToDirectedEdges,
  getRes0Cells,
  cellToChildren,
  latLngToCell,
  getPentagons,
  getDirectedEdgeDestination,
  directedEdgeToBoundary
} from 'h3-js';
import {GPUNetworkStructure} from '../../src/gpu-network/network-structure';
import {GPUAlternativeRoutes} from '../../src/gpu-network/alternative-routes';
import {GPUNetworkCentrality} from '../../src/gpu-network/network-centrality';
import {GPUH3CellIdentities} from '../../src/gpu-spatial-analysis/cell-topology/gpu-h3-cell-identities';
import {GPUBoundedFacilityAllocation} from '../../src/gpu-spatial-analysis/location-allocation/gpu-bounded-facility-allocation';
import {addCellFlowTopologyRecipe} from '../../src/gpu-spatial-analysis/recipes/cell-flow-topology-recipe';
import {GPUTrackSimilarity} from '../../src/gpu-spatial-analysis/track-similarity';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import reference from '../oracles/fixtures/network-expansion.json';
import {requireTestDevice} from '../utils/require-test-device';

async function harness() {
  const candidateDevice = await getWebGPUTestDevice();
  requireTestDevice(candidateDevice);
  const device = candidateDevice;
  const graph = new GPUCommandGraph(device);
  let next = 0;
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  function input(
    values: number[],
    format: 'uint32'
  ): ReturnType<typeof graph.createDataView<'uint32'>>;
  function input(
    values: number[],
    format: 'float32'
  ): ReturnType<typeof graph.createDataView<'float32'>>;
  function input(values: number[], format: 'uint32' | 'float32') {
    const buffer = createInputBuffer(
      device,
      format === 'float32' ? Float32Array.from(values) : Uint32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(graph, `input-${next++}`, buffer, format, values.length);
  }
  function output<Format extends 'uint32' | 'float32' | 'uint32x2'>(
    length: number,
    format: Format
  ) {
    const buffer = createOutputBuffer(device, length * (format === 'uint32x2' ? 2 : 1));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${next++}`, buffer, format, length)};
  }
  return {
    device,
    graph,
    input,
    output,
    run() {
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      return () => {
        compiled.destroy();
        buffers.forEach(buffer => buffer.destroy());
      };
    }
  };
}

it('network expansion finds SCCs, bridges, articulation points and multigraph blocks', async () => {
  const setup = await harness();
  const {graph, input, output} = setup;
  // Triangle 0-1-2 with a bridge to 3 and two parallel edges 3-4; isolated 5.
  const offsets = input([0, 2, 4, 7, 10, 12, 12], 'uint32');
  const neighbors = input([1, 2, 0, 2, 0, 1, 3, 2, 4, 4, 3, 3], 'uint32');
  const reverseEdgeSlots = input([2, 4, 0, 5, 1, 3, 7, 6, 10, 11, 8, 9], 'uint32');
  const strong = output(6, 'uint32');
  const bridges = output(12, 'uint32');
  const articulation = output(6, 'uint32');
  const blocks = output(12, 'uint32');
  graph.add(
    new GPUNetworkStructure({
      topology: {offsets, neighbors},
      reverseEdgeSlots,
      output: {
        strongComponents: strong.view,
        bridges: bridges.view,
        articulationPoints: articulation.view,
        biconnectedComponents: blocks.view
      }
    })
  );
  const directed = output(6, 'uint32');
  graph.add(
    new GPUNetworkStructure({
      id: 'directed',
      topology: {
        offsets: input([0, 1, 2, 4, 5, 6, 6], 'uint32'),
        neighbors: input([1, 0, 1, 3, 4, 3], 'uint32')
      },
      output: {strongComponents: directed.view}
    })
  );
  const destroy = setup.run();
  expect(Array.from(await readUint32(strong.buffer, 6))).toEqual([0, 0, 0, 0, 0, 5]);
  expect(Array.from(await readUint32(directed.buffer, 6))).toEqual(reference.strongComponents);
  expect(Array.from(await readUint32(bridges.buffer, 12))).toEqual([
    0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0
  ]);
  expect(Array.from(await readUint32(articulation.buffer, 6))).toEqual([0, 0, 1, 1, 0, 0]);
  expect(Array.from(await readUint32(blocks.buffer, 12))).toEqual([
    0, 0, 0, 0, 0, 0, 6, 6, 8, 8, 8, 8
  ]);
  destroy();
});

it('alternative routes orders simple paths and flags bounded search', async () => {
  for (const maximumExpansions of [1000, 1]) {
    const setup = await harness();
    const {graph, input, output} = setup;
    const vertices = output(12, 'uint32');
    const edges = output(12, 'uint32');
    const lengths = output(3, 'uint32');
    const costs = output(3, 'float32');
    const counts = output(1, 'uint32');
    const complete = output(1, 'uint32');
    graph.add(
      new GPUAlternativeRoutes({
        topology: {
          offsets: input([0, 3, 5, 6, 6], 'uint32'),
          neighbors: input([1, 2, 3, 2, 3, 3], 'uint32'),
          weights: input([1, 2, 5, 0.5, 2, 1], 'float32')
        },
        sources: input([0], 'uint32'),
        targets: input([3], 'uint32'),
        routeCount: 3,
        maximumExpansions,
        output: {
          vertices: vertices.view,
          edges: edges.view,
          lengths: lengths.view,
          costs: costs.view,
          counts: counts.view,
          complete: complete.view
        }
      })
    );
    const destroy = setup.run();
    expect(Array.from(await readUint32(complete.buffer, 1))).toEqual([
      maximumExpansions === 1 ? 0 : 1
    ]);
    if (maximumExpansions > 1) {
      expect(Array.from(await readFloat32(costs.buffer, 3))).toEqual(reference.routeCosts);
      expect(Array.from(await readUint32(lengths.buffer, 3))).toEqual([4, 3, 3]);
      expect(Array.from(await readUint32(vertices.buffer, 12))).toEqual([
        0, 1, 2, 3, 0, 1, 3, 0xffffffff, 0, 2, 3, 0xffffffff
      ]);
    }
    destroy();
  }
});

it('edge betweenness works without requesting node betweenness', async () => {
  const setup = await harness();
  const {graph, input, output} = setup;
  const result = output(2, 'float32');
  graph.add(
    new GPUNetworkCentrality({
      topology: {
        offsets: input([0, 1, 2, 2], 'uint32'),
        neighbors: input([1, 2], 'uint32'),
        weights: input([1, 1], 'float32')
      },
      reverseTopology: {
        offsets: input([0, 0, 1, 2], 'uint32'),
        neighbors: input([0, 1], 'uint32'),
        weights: input([1, 1], 'float32')
      },
      pivotNodes: input([0, 1, 2], 'uint32'),
      maximumIterations: 8,
      laneCount: 2,
      output: {edgeBetweenness: result.view}
    })
  );
  const destroy = setup.run();
  expect(Array.from(await readFloat32(result.buffer, 2))).toEqual(reference.edgeBetweenness);
  destroy();
});

it('H3 identities match h3-js across base cells, face crossings and resolutions', async () => {
  const cells = [
    ...getRes0Cells().flatMap(cell => [cell, ...cellToChildren(cell, 1)]),
    ...Array.from({length: 16}, (_, resolution) => latLngToCell(37.7, -122.4, resolution))
  ];
  for (const operation of ['vertices', 'directed-edges'] as const) {
    const setup = await harness();
    const {graph, output} = setup;
    const words = Uint32Array.from(
      cells.flatMap(cell => {
        const key = BigInt(`0x${cell}`);
        return [Number(key & 0xffffffffn), Number(key >> 32n)];
      })
    );
    const buffer = createInputBuffer(setup.device, words);
    const result = output(cells.length * 6, 'uint32x2');
    graph.add(
      new GPUH3CellIdentities({
        cells: importGraphBuffer(graph, 'cells', buffer, 'uint32x2', cells.length),
        operation,
        output: result.view
      })
    );
    const destroy = setup.run();
    const actual = await readUint32(result.buffer, cells.length * 12);
    for (let row = 0; row < cells.length; row++) {
      const expected =
        operation === 'vertices' ? cellToVertexes(cells[row]) : originToDirectedEdges(cells[row]);
      const actualKeys = Array.from({length: 6}, (_, slot) =>
        (
          (BigInt(actual[row * 12 + slot * 2 + 1]) << 32n) |
          BigInt(actual[row * 12 + slot * 2])
        ).toString(16)
      ).filter(key => key !== '0');
      expect(actualKeys, `${operation} ${cells[row]}`).toEqual(expected);
    }
    destroy();
    buffer.destroy();
  }
});

it('facility allocation certifies p-center and respects indivisible capacity', async () => {
  for (const operation of ['p-center', 'capacitated-p-median'] as const) {
    const setup = await harness();
    const {graph, input, output} = setup;
    const selected = output(3, 'uint32'),
      assignments = output(3, 'uint32'),
      objective = output(1, 'float32'),
      status = output(3, 'uint32');
    graph.add(
      new GPUBoundedFacilityAllocation({
        operation,
        facilityCount: 3,
        selectCount: 2,
        costs: input([1, 4, 5, 2, 1, 4, 5, 3, 1], 'float32'),
        demand: input([1, 1, 1], 'float32'),
        capacities: input([1, 1, 1], 'float32'),
        output: {
          selectedFacilities: selected.view,
          assignments: assignments.view,
          objective: objective.view,
          status: status.view
        }
      })
    );
    const destroy = setup.run();
    if (operation === 'p-center') {
      expect(Array.from(await readUint32(selected.buffer, 3))).toEqual([1, 0, 1]);
      expect(Array.from(await readFloat32(objective.buffer, 1))).toEqual([2]);
      expect(Array.from(await readUint32(status.buffer, 3))).toEqual([0, 1, 1]);
    } else {
      expect(Array.from(await readUint32(status.buffer, 3))).toEqual([1, 1, 0]);
      const assigned = Array.from(await readUint32(assignments.buffer, 3)).filter(
        value => value !== 0xffffffff
      );
      expect(new Set(assigned).size).toBe(2);
      expect(Array.from(await readFloat32(objective.buffer, 1))).toEqual([Infinity]);
    }
    destroy();
  }
});

it('LCSS matches partial tracks and carries dynamic-program rows across strips', async () => {
  const setup = await harness();
  const {graph, input, output} = setup;
  const long = Array.from({length: 300}, (_, index) => [index, 0]).flat();
  const short = Array.from({length: 10}, (_, index) => [270 + index * 2, 0]).flat();
  const positions = input([...long, ...short], 'float32');
  const result = output(1, 'float32');
  graph.add(
    new GPUTrackSimilarity({
      linesA: {
        kind: 'lines',
        positions: graph.createDataView(positions.buffer, {format: 'float32x2', length: 310}),
        lineOffsets: input([0, 300, 310], 'uint32')
      },
      pairA: input([0], 'uint32'),
      pairB: input([1], 'uint32'),
      lcssTolerance: 0,
      output: {longestCommonSubsequence: result.view}
    })
  );
  const destroy = setup.run();
  expect(Array.from(await readFloat32(result.buffer, 1))).toEqual([1]);
  destroy();
});

it('cell-flow recipe conserves arbitrary OD totals and matches H3 shared boundaries', async () => {
  for (const origin of [getPentagons(1)[0], latLngToCell(0, 179.99, 2)]) {
    const neighbor = getDirectedEdgeDestination(originToDirectedEdges(origin)[0]);
    const cells = [origin, neighbor, latLngToCell(-55, 25, origin === getPentagons(1)[0] ? 1 : 2)];
    const setup = await harness();
    const {graph, input, output} = setup;
    const words = Uint32Array.from(
      cells.flatMap(cell => {
        const key = BigInt(`0x${cell}`);
        return [Number(key & 0xffffffffn), Number(key >> 32n)];
      })
    );
    const cellsBuffer = createInputBuffer(setup.device, words);
    const cellView = importGraphBuffer(graph, 'flow-cells', cellsBuffer, 'uint32x2', 3);
    const outgoing = output(3, 'float32'),
      incoming = output(3, 'float32'),
      edgeFlows = output(18, 'float32'),
      edgeKeys = output(18, 'uint32x2'),
      counts = output(18, 'uint32'),
      invalid = output(1, 'uint32');
    const boundaryBuffer = createOutputBuffer(setup.device, 108);
    const boundaries = importGraphBuffer(graph, 'flow-boundaries', boundaryBuffer, 'float32x2', 54);
    const result = addCellFlowTopologyRecipe(graph, {
      cells: cellView,
      origins: input([0, 0, 1, 0, 0, 5], 'uint32'),
      destinations: input([1, 1, 0, 2, 0, 1], 'uint32'),
      values: input([2, 3, 4, 7, 11, 17], 'float32'),
      outputs: {
        outgoingTotals: outgoing.view,
        incomingTotals: incoming.view,
        edgeFlows: edgeFlows.view,
        directedEdges: edgeKeys.view,
        boundaries,
        boundaryCounts: counts.view,
        status: {invalidCount: invalid.view}
      }
    });
    expect(result.pairTable.values.length).toBe(6);
    const destroy = setup.run();
    expect(await readFloat32(outgoing.buffer, 3)).toEqual([23, 4, 0]);
    expect(await readFloat32(incoming.buffer, 3)).toEqual([15, 5, 7]);
    expect((await readFloat32(edgeFlows.buffer, 18)).reduce((sum, value) => sum + value, 0)).toBe(
      9
    );
    expect(await readUint32(invalid.buffer, 1)).toEqual([1]);
    const keys = await readUint32(edgeKeys.buffer, 36),
      actualCounts = await readUint32(counts.buffer, 18),
      positions = await readFloat32(boundaryBuffer, 108);
    for (let edge = 0; edge < 18; edge++) {
      const key = (BigInt(keys[edge * 2 + 1]) << 32n) | BigInt(keys[edge * 2]);
      if (key === 0n) {
        expect(actualCounts[edge]).toBe(0);
        continue;
      }
      const expected = directedEdgeToBoundary(key.toString(16));
      expect(actualCounts[edge]).toBe(expected.length);
      for (let vertex = 0; vertex < expected.length; vertex++) {
        const longitude = positions[edge * 6 + vertex * 2];
        const latitude = positions[edge * 6 + vertex * 2 + 1];
        expect(Math.abs(((longitude - expected[vertex][1] + 540) % 360) - 180)).toBeLessThan(
          0.0003
        );
        expect(Math.abs(latitude - expected[vertex][0])).toBeLessThan(0.0003);
      }
    }
    destroy();
    cellsBuffer.destroy();
    boundaryBuffer.destroy();
  }
});

it('GPU undirected structure matches NetworkX on seeded random disconnected graphs', async () => {
  const setup = await harness();
  const {graph, input, output} = setup;
  const cases = reference.randomUndirected.map((fixture, index) => {
    const bridges = output(fixture.neighbors.length, 'uint32'),
      blocks = output(fixture.neighbors.length, 'uint32'),
      articulation = output(8, 'uint32');
    graph.add(
      new GPUNetworkStructure({
        id: `random-${index}`,
        topology: {
          offsets: input(fixture.offsets, 'uint32'),
          neighbors: input(fixture.neighbors, 'uint32')
        },
        reverseEdgeSlots: input(fixture.reverse, 'uint32'),
        output: {
          bridges: bridges.view,
          biconnectedComponents: blocks.view,
          articulationPoints: articulation.view
        }
      })
    );
    return {fixture, bridges, blocks, articulation};
  });
  const destroy = setup.run();
  for (const item of cases) {
    expect(await readUint32(item.bridges.buffer, item.fixture.neighbors.length)).toEqual(
      item.fixture.bridges
    );
    expect(await readUint32(item.blocks.buffer, item.fixture.neighbors.length)).toEqual(
      item.fixture.blocks
    );
    expect(await readUint32(item.articulation.buffer, 8)).toEqual(item.fixture.articulation);
  }
  destroy();
});

it('capacitated facility allocation finds feasible indivisible assignments', async () => {
  const setup = await harness();
  const {graph, input, output} = setup;
  const selected = output(3, 'uint32'),
    assignments = output(3, 'uint32'),
    objective = output(1, 'float32'),
    status = output(3, 'uint32');
  graph.add(
    new GPUBoundedFacilityAllocation({
      operation: 'capacitated-p-median',
      facilityCount: 3,
      selectCount: 2,
      costs: input([1, 4, 5, 2, 1, 4, 5, 3, 1], 'float32'),
      demand: input([1, 1, 1], 'float32'),
      capacities: input([2, 1, 1], 'float32'),
      output: {
        selectedFacilities: selected.view,
        assignments: assignments.view,
        objective: objective.view,
        status: status.view
      }
    })
  );
  const destroy = setup.run();
  expect(await readUint32(selected.buffer, 3)).toEqual([1, 0, 1]);
  expect(await readUint32(assignments.buffer, 3)).toEqual([0, 0, 2]);
  expect(await readFloat32(objective.buffer, 1)).toEqual([4]);
  expect(await readUint32(status.buffer, 3)).toEqual([0, 1, 0]);
  destroy();
});

it('LCSS matches an independent NetworkX longest-chain oracle with skipped and repeated observations', async () => {
  const setup = await harness();
  const {graph, input, output} = setup;
  const cases = reference.lcss.map((fixture, index) => {
    const result = output(1, 'float32');
    const positions = input(
      [...fixture.first, ...fixture.second].flatMap(value => [value, 0]),
      'float32'
    );
    graph.add(
      new GPUTrackSimilarity({
        id: `lcss-${index}`,
        linesA: {
          kind: 'lines',
          positions: graph.createDataView(positions.buffer, {
            format: 'float32x2',
            length: fixture.first.length + fixture.second.length
          }),
          lineOffsets: input(
            [0, fixture.first.length, fixture.first.length + fixture.second.length],
            'uint32'
          )
        },
        pairA: input([0], 'uint32'),
        pairB: input([1], 'uint32'),
        lcssTolerance: fixture.tolerance,
        output: {longestCommonSubsequence: result.view}
      })
    );
    return {fixture, result};
  });
  const destroy = setup.run();
  for (const item of cases)
    expect((await readFloat32(item.result.buffer, 1))[0]).toBeCloseTo(item.fixture.similarity, 6);
  destroy();
});

it('capacitated allocation exposes a feasible heuristic without claiming the exact assignment optimum', async () => {
  const setup = await harness();
  const {graph, input, output} = setup;
  const fixture = reference.capacitated;
  const selected = output(2, 'uint32'),
    assignments = output(2, 'uint32'),
    objective = output(1, 'float32'),
    status = output(3, 'uint32');
  graph.add(
    new GPUBoundedFacilityAllocation({
      operation: 'capacitated-p-median',
      facilityCount: 2,
      selectCount: 2,
      costs: input(fixture.costs, 'float32'),
      demand: input(fixture.demand, 'float32'),
      capacities: input(fixture.capacities, 'float32'),
      output: {
        selectedFacilities: selected.view,
        assignments: assignments.view,
        objective: objective.view,
        status: status.view
      }
    })
  );
  const destroy = setup.run();
  expect(await readUint32(assignments.buffer, 2)).toEqual([0, 1]);
  const actual = (await readFloat32(objective.buffer, 1))[0];
  expect(actual).toBe(101);
  expect(fixture.exactObjective).toBe(3);
  expect(actual).toBeGreaterThan(fixture.exactObjective);
  expect(await readUint32(status.buffer, 3)).toEqual([0, 1, 0]);
  destroy();
});
