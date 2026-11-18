// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUNetworkInhomogeneousK} from '../../src/gpu-network/gpu-network-inhomogeneous-k';
import {GPUAdjacencyMatrix} from '../../src/gpu-network/adjacency-matrix/gpu-adjacency-matrix';
import {GPUNetworkCostWeights} from '../../src/gpu-network/network-cost-weights/gpu-network-cost-weights';
import {GPUNetworkSubgraphFilter} from '../../src/gpu-network/network-subgraph-filter/gpu-network-subgraph-filter';
import {GPUNetworkNoding} from '../../src/gpu-network/network-noding/gpu-network-noding';
import {GPUNetworkAccessibility} from '../../src/gpu-network/network-accessibility/gpu-network-accessibility';
import {GPUNetworkCostMatrix} from '../../src/gpu-network/network-accessibility/gpu-network-cost-matrix';
import {GPUNetworkCostTable} from '../../src/gpu-network/network-cost-table/gpu-network-cost-table';
import {createKernelAuditContext} from '../analysis-core/kernel-audit/kernel-audit-context';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';
import {getParsedKernelSources} from '../utils/gpu-contributor-test-utils';

it('accessibility accepts scores packed beside catchment output but separates ratio reads from writes', () => {
  const {graph, device, view} = createKernelAuditContext('packed-accessibility');
  const storage = view('uint32', 130).buffer;
  const props = {
    costs: view('float32', 4),
    opportunityWeights: view('float32', 2),
    parameters: view('float32', 4),
    cumulative: graph.createDataView(storage, {format: 'float32', length: 2}),
    gravity: graph.createDataView(storage, {format: 'float32', length: 2, byteOffset: 256}),
    catchment: {
      demand: view('float32', 2),
      output: graph.createDataView(storage, {format: 'float32', length: 2, byteOffset: 8}),
      ratios: view('float32', 2)
    }
  };
  try {
    expect(() => graph.add(new GPUNetworkAccessibility(props))).not.toThrow();
    expect(
      () =>
        new GPUNetworkAccessibility({
          ...props,
          gravity: graph.createDataView(storage, {format: 'float32', length: 2, byteOffset: 16})
        })
    ).toThrow(/aligned binding byte ranges must not overlap/);
    expect(
      () =>
        new GPUNetworkAccessibility({
          ...props,
          catchment: {
            ...props.catchment,
            ratios: graph.createDataView(storage, {format: 'float32', length: 2, byteOffset: 512})
          }
        })
    ).toThrow(/must not share buffers with inputs/);
  } finally {
    device.destroy();
  }
});

it('cost matrix accepts costs adjacent to separately published status and aligns co-bound flags', () => {
  const {graph, device, view} = createKernelAuditContext('packed-cost-matrix');
  const storage = view('uint32', 65).buffer;
  const props = {
    topology: {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 2),
      weights: view('float32', 2)
    },
    startNodes: view('uint32', 1),
    laneCount: 1,
    maximumIterations: 2,
    output: {
      costs: graph.createDataView(storage, {format: 'float32', length: 2}),
      status: {
        converged: graph.createDataView(storage, {format: 'uint32', length: 1, byteOffset: 8}),
        iterationLimitReached: graph.createDataView(storage, {
          format: 'uint32',
          length: 1,
          byteOffset: 256
        })
      }
    }
  };
  try {
    expect(() => graph.add(new GPUNetworkCostMatrix(props))).not.toThrow();
    for (const [byteOffset, message] of [
      [4, /logical byte ranges must not overlap/],
      [12, /aligned binding byte ranges must not overlap/]
    ] as const) {
      expect(
        () =>
          new GPUNetworkCostMatrix({
            ...props,
            output: {
              ...props.output,
              status: {
                ...props.output.status,
                iterationLimitReached: graph.createDataView(storage, {
                  format: 'uint32',
                  length: 1,
                  byteOffset
                })
              }
            }
          })
      ).toThrow(message);
    }
  } finally {
    device.destroy();
  }
});

it('cost table accepts packed offsets, entries and separately published status', () => {
  const {graph, device, view} = createKernelAuditContext('packed-cost-table');
  const storage = view('uint32', 130).buffer;
  const props = {
    topology: {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 2),
      weights: view('float32', 2)
    },
    costLimit: view('float32', 1),
    laneCount: 1,
    maximumIterations: 2,
    output: {
      offsets: graph.createDataView(storage, {format: 'uint32', length: 3}),
      targets: graph.createDataView(storage, {format: 'uint32', length: 2, byteOffset: 256}),
      costs: graph.createDataView(storage, {format: 'float32', length: 2, byteOffset: 512}),
      status: {
        overflow: graph.createDataView(storage, {format: 'uint32', length: 1, byteOffset: 12})
      }
    }
  };
  try {
    expect(() => graph.add(new GPUNetworkCostTable(props))).not.toThrow();
    expect(
      () =>
        new GPUNetworkCostTable({
          ...props,
          output: {
            ...props.output,
            costs: graph.createDataView(storage, {format: 'float32', length: 2, byteOffset: 264})
          }
        })
    ).toThrow(/aligned binding byte ranges must not overlap/);
    expect(
      () =>
        new GPUNetworkCostTable({
          ...props,
          output: {
            ...props.output,
            status: {
              overflow: graph.createDataView(storage, {format: 'uint32', length: 1, byteOffset: 8})
            }
          }
        })
    ).toThrow(/logical byte ranges must not overlap/);
  } finally {
    device.destroy();
  }
});

it('cost weights accepts adjacent output ranges in separate dispatches and rejects mixed storage usage', () => {
  const {graph, device, view} = createKernelAuditContext('packed-cost-weights');
  const storage = view('uint32', 130).buffer;
  const props = {
    costs: view('float32', 2),
    parameters: view('float32', 4),
    output: {
      offsets: view('uint32', 2),
      neighbors: graph.createDataView(storage, {format: 'uint32', length: 2}),
      weights: view('float32', 2),
      status: {
        overflow: graph.createDataView(storage, {format: 'uint32', length: 1, byteOffset: 8})
      }
    }
  };
  try {
    const nodes = new GPUNetworkCostWeights(props).getCommandNodes(graph);
    for (const {id: name, source, constants} of getParsedKernelSources(nodes).configurations)
      expect(auditWGSLModule(source, constants), name).toEqual([]);
    expect(getParsedKernelSources(nodes).get('network-cost-weights-fill')).toContain(
      'getScaledQuotient(cost, costLimit, 1.0, 0)'
    );
    expect(() => nodes.forEach(node => graph.add(node))).not.toThrow();
    expect(
      () =>
        new GPUNetworkCostWeights({
          ...props,
          output: {
            ...props.output,
            offsets: graph.createDataView(storage, {format: 'uint32', length: 2, byteOffset: 256})
          }
        })
    ).toThrow(/must not share buffers with inputs/);
  } finally {
    device.destroy();
  }
});

it('subgraph filtering accepts counts packed beside induced neighbors from a separate dispatch', () => {
  const {graph, device, view} = createKernelAuditContext('packed-subgraph');
  const storage = view('uint32', 8).buffer;
  try {
    expect(() =>
      graph.add(
        new GPUNetworkSubgraphFilter({
          topology: {offsets: view('uint32', 4), neighbors: view('uint32', 4)},
          output: {
            nodeMask: view('uint32', 3),
            edgeMask: view('uint32', 4),
            counts: graph.createDataView(storage, {format: 'uint32', length: 4}),
            inducedTopology: {
              offsets: view('uint32', 4),
              neighbors: graph.createDataView(storage, {
                format: 'uint32',
                length: 4,
                byteOffset: 16
              }),
              status: {overflow: view('uint32', 1)}
            }
          }
        })
      )
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('noding accepts adjacent node-position and CSR-offset outputs and atomically reports snap overflow', () => {
  const {graph, device, view} = createKernelAuditContext('packed-noding');
  const storage = view('uint32', 25).buffer;
  try {
    const contributor = new GPUNetworkNoding({
      lines: {kind: 'lines', positions: view('float32x2', 4), lineOffsets: view('uint32', 3)},
      tolerance: view('float32', 1),
      intersectionCapacity: 4,
      output: {
        pieces: {
          geometry: {
            kind: 'lines',
            positions: view('float32x2', 8),
            lineOffsets: view('uint32', 5)
          },
          sourceIds: view('uint32', 4),
          status: {
            count: view('uint32', 1),
            overflow: view('uint32', 1),
            vertexCount: view('uint32', 1)
          }
        },
        nodes: {
          positions: graph.createDataView(storage, {format: 'float32x2', length: 8}),
          status: {count: view('uint32', 1)}
        },
        edges: {
          fromNodes: view('uint32', 4),
          toNodes: view('uint32', 4),
          lengths: view('float32', 4)
        },
        topology: {
          offsets: graph.createDataView(storage, {format: 'uint32', length: 9, byteOffset: 64}),
          neighbors: view('uint32', 8),
          weights: view('float32', 8)
        },
        status: {overflow: view('uint32', 1)}
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    const sources = getParsedKernelSources(nodes);
    expect(
      [...sources.values()].some(source =>
        source.includes('atomicStore(&snapOverflow[snapOverflowOffset], 1u)')
      )
    ).toBe(true);
    for (const {id: name, source, constants} of sources.configurations)
      expect(auditWGSLModule(source, constants), name).toEqual([]);
    expect(() => nodes.forEach(node => graph.add(node))).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('inhomogeneous K audits scaled arithmetic and rejects normalization as writable storage', () => {
  const {graph, device, view} = createKernelAuditContext('inhomogeneous-scale');
  const props = {
    eventNodeIds: view('uint32', 2),
    intensity: view('float32', 2),
    nodeDistances: view('float32', 4),
    nodeCount: 2,
    edgeNodes: view('uint32x2', 1),
    edgeLengths: view('float32', 1),
    radii: view('float32', 1),
    output: {
      k: view('float32', 1),
      numerator: view('float32', 1),
      normalization: view('float32', 3),
      status: view('uint32', 1)
    }
  };
  try {
    const nodes = new GPUNetworkInhomogeneousK(props).getCommandNodes(graph);
    for (const {id: name, source, constants} of getParsedKernelSources(nodes).configurations)
      expect(auditWGSLModule(source, constants), name).toEqual([]);
    expect(getParsedKernelSources(nodes).get('network-inhomogeneous-k-pairs')).toContain(
      'getScaledQuotient(1.0, sourceIntensity, destinationIntensity, 0)'
    );
    const storage = view('uint32', 65).buffer;
    expect(
      () =>
        new GPUNetworkInhomogeneousK({
          ...props,
          output: {
            ...props.output,
            normalization: graph.createDataView(storage, {format: 'float32', length: 3}),
            k: graph.createDataView(storage, {format: 'float32', length: 1, byteOffset: 256})
          }
        })
    ).toThrow(/must not share buffers with inputs/);
  } finally {
    device.destroy();
  }
});

it('adjacency matrix rejects scales that would create infinite or zero WGSL constants', () => {
  const {device, view} = createKernelAuditContext('adjacency-scale');
  const props = {
    topology: {offsets: view('uint32', 3), neighbors: view('uint32', 1)},
    resolution: 1,
    output: {counts: view('uint32', 1)}
  };
  try {
    for (const weightScale of [1e100, 1e-100])
      expect(() => new GPUAdjacencyMatrix({...props, weightScale})).toThrow(
        /adjacency-matrix.*weightScale.*finite/
      );
    const nodes = new GPUAdjacencyMatrix({...props, weightScale: 1e20}).getCommandNodes(
      props.topology.offsets.buffer.graph as GPUCommandGraph
    );
    for (const {id: name, source, constants} of getParsedKernelSources(nodes).configurations) {
      expect(auditWGSLModule(source, constants), name).toEqual([]);
    }
    expect(getParsedKernelSources(nodes).get('adjacency-matrix-bin-counts')).toMatch(
      /const WEIGHT_SCALE: f32 = [0-9]+\.[0-9]+e[+-][0-9]+f;/
    );
  } finally {
    device.destroy();
  }
});

it('adjacency bin dispatches allow packed counts and weights when the binding budget splits them', () => {
  const {graph, device, view} = createKernelAuditContext('split-adjacency');
  const storage = view('uint32', 2).buffer;
  try {
    const nodes = new GPUAdjacencyMatrix({
      topology: {
        offsets: view('uint32', 4),
        neighbors: view('uint32', 2),
        weights: view('float32', 2)
      },
      resolution: 1,
      parameters: view('uint32', 4),
      order: view('uint32', 3),
      nodeMask: view('uint32', 3),
      edgeMask: view('uint32', 2),
      output: {
        counts: graph.createDataView(storage, {format: 'uint32', length: 1}),
        weightSums: graph.createDataView(storage, {format: 'uint32', length: 1, byteOffset: 4})
      }
    }).getCommandNodes(graph);
    expect(nodes.map(node => node.id)).toContain('adjacency-matrix-bin-counts');
    expect(nodes.map(node => node.id)).toContain('adjacency-matrix-bin-weights');
    expect(() => nodes.forEach(node => graph.add(node))).not.toThrow();
  } finally {
    device.destroy();
  }
});
