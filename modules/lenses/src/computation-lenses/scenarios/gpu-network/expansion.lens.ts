// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {runExample as runStructure} from './network-structure.example';
import {runExample as runStrongComponents} from './network-strongly-connected-components.example';
import {runExample as runRoutes} from './alternative-routes.example';
import {runExample as runKShortestPaths} from './k-shortest-paths.example';
import {runExample as runDisruption} from './network-disruption-recipe.example';
import {runExample as runFacility} from './facility-scenario-recipe.example';
import {
  recordNetworkElement,
  describeNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';

export type ExpansionKind =
  | 'network-structure'
  | 'network-strongly-connected-components'
  | 'alternative-routes'
  | 'k-shortest-paths'
  | 'network-disruption-recipe'
  | 'facility-scenario-recipe';

/** Small explicitly synthetic CSR keeps bounded detours and component inspection bounded. @internal */
export function createNetworkExpansionScenario(kind: ExpansionKind): NetworkElementScenario {
  return {
    id: `gpu-network.${kind}`,
    weights: 'lengthMeters',
    withoutRoads: true,
    roadContext: false,
    datasetVersion:
      kind === 'network-strongly-connected-components'
        ? 'synthetic-four-node-directed-components-v1'
        : 'synthetic-four-node-directed-diamond-v1',
    seed: 0,
    retain:
      kind === 'k-shortest-paths'
        ? [
            'vertices',
            'edges',
            'lengths',
            'costs',
            'counts',
            'converged',
            'iterationCount',
            'iterationLimitReached',
            'candidateOverflow'
          ]
        : kind === 'alternative-routes'
          ? ['vertices', 'edges', 'lengths', 'costs', 'counts', 'complete']
          : kind === 'network-disruption-recipe'
            ? [
                'network-disruption-recipe-baseline-costs',
                'network-disruption-recipe-scenario-costs',
                'network-disruption-recipe-cost-differences',
                'network-disruption-recipe-newly-unreachable',
                'network-disruption-recipe-baseline-accessibility',
                'network-disruption-recipe-scenario-accessibility',
                'network-disruption-recipe-accessibility-differences'
              ]
            : kind === 'facility-scenario-recipe'
              ? [
                  'facility-scenario-recipe-accessibility',
                  'facility-scenario-recipe-nearest-costs',
                  'facility-scenario-recipe-nearest-facilities',
                  'facility-scenario-recipe-covered',
                  'facility-scenario-recipe-uncovered',
                  'facility-scenario-recipe-invalid'
                ]
              : [],
    formulas: [],
    keySpaces: [
      {id: 'node', label: 'Synthetic node', size: 4, indexSpace: '1d'},
      // The gpu-graph strong-component kernels used by GPUNetworkStructure address graph vertices.
      {id: 'vertex', label: 'Graph vertex', size: 4, indexSpace: '1d'},
      {id: 'node-offset', label: 'CSR node boundary', size: 5, indexSpace: '1d'},
      {id: 'edge', label: 'Directed edge slot', size: 5, indexSpace: '1d'},
      {id: 'route', label: 'Ranked route', size: 2, indexSpace: '1d'},
      {id: 'frontier-slot', label: 'Alternating frontier queue slot', size: 8, indexSpace: '1d'},
      {id: 'query', label: 'Query', size: 1, indexSpace: '1d'},
      {id: 'route-vertex', label: 'Padded route vertex', size: 8, indexSpace: '1d'}
    ],
    entityKeySpace: 'node',
    ...(kind === 'network-disruption-recipe' && {
      frontierRelationAliases: {
        'network-disruption-recipe-baseline-': 'baseline-frontier-queue',
        'network-disruption-recipe-scenario-': 'scenario-frontier-queue'
      },
      relations: ['baseline', 'scenario'].map(stage => ({
        id: `${stage}-frontier-queue`,
        from: 'frontier-slot',
        to: 'node',
        cardinality: 'N:1' as const,
        kind: 'buffer' as const,
        timeDependent: true,
        ref: {
          layout: 'index' as const,
          resource: `network-disruption-recipe-${stage}-batch-0-queues`
        }
      }))
    }),
    choices: [
      {
        stage: 'representation',
        name: 'topology',
        value:
          kind === 'network-strongly-connected-components'
            ? 'two-node cycle, outgoing singleton with self-loop, isolated node and unused CSR slot'
            : 'four-node directed diamond with return edge',
        sweepable: false
      }
    ],
    async build(context) {
      const {graph} = context;
      const upload = context.upload;
      const view = <Format extends 'uint32' | 'float32'>(
        name: string,
        format: Format,
        length: number
      ) => createTransientView(graph, name, format, length, Buffer.STORAGE | Buffer.COPY_SRC);
      if (kind === 'network-strongly-connected-components') {
        const componentTopology = {
          offsets: upload('componentOffsets', new Uint32Array([0, 1, 3, 4, 4]), 'uint32'),
          neighbors: upload(
            'componentNeighbors',
            new Uint32Array([1, 0, 2, 2, 0xffffffff]),
            'uint32'
          )
        };
        const strongComponents = view('strongComponents', 'uint32', 4);
        const status = {
          converged: view('converged', 'uint32', 1),
          iterationCount: view('iterationCount', 'uint32', 1),
          iterationLimitReached: view('iterationLimitReached', 'uint32', 1)
        };
        await runStrongComponents(context.device, graph, {
          topology: componentTopology,
          maximumIterations: 32,
          output: {strongComponents, status}
        });
        return {views: {...componentTopology, strongComponents, ...status}};
      }
      const topology = {
        offsets: upload('offsets', new Uint32Array([0, 2, 3, 4, 5]), 'uint32'),
        neighbors: upload('neighbors', new Uint32Array([1, 2, 3, 3, 0]), 'uint32'),
        weights: upload('weights', new Float32Array([1, 2, 1, 1, 4]), 'float32')
      };
      if (kind === 'network-structure') {
        const strongComponents = view('strongComponents', 'uint32', 4);
        await runStructure(context.device, graph, {
          topology,
          output: {strongComponents}
        });
        return {views: {...topology, strongComponents}};
      }
      if (kind === 'k-shortest-paths') {
        const output = {
          vertices: view('vertices', 'uint32', 8),
          edges: view('edges', 'uint32', 8),
          lengths: view('lengths', 'uint32', 2),
          costs: view('costs', 'float32', 2),
          counts: view('counts', 'uint32', 1),
          status: {
            converged: view('converged', 'uint32', 1),
            iterationCount: view('iterationCount', 'uint32', 1),
            iterationLimitReached: view('iterationLimitReached', 'uint32', 1),
            candidateOverflow: view('candidateOverflow', 'uint32', 1)
          }
        };
        await runKShortestPaths(context.device, graph, {
          topology,
          sources: upload('sources', new Uint32Array([0]), 'uint32'),
          targets: upload('targets', new Uint32Array([3]), 'uint32'),
          routeCount: 2,
          maximumPathLength: 4,
          candidateCapacity: 8,
          maximumIterations: 4,
          output
        });
        const {status, ...routes} = output;
        return {views: {...topology, ...routes, ...status}};
      }
      if (kind === 'alternative-routes') {
        const output = {
          vertices: view('vertices', 'uint32', 8),
          edges: view('edges', 'uint32', 8),
          lengths: view('lengths', 'uint32', 2),
          costs: view('costs', 'float32', 2),
          counts: view('counts', 'uint32', 1),
          complete: view('complete', 'uint32', 1)
        };
        await runRoutes(context.device, graph, {
          topology,
          sources: upload('sources', new Uint32Array([0]), 'uint32'),
          targets: upload('targets', new Uint32Array([3]), 'uint32'),
          routeCount: 2,
          maximumExpansions: 1000,
          maximumFrontierSize: 32,
          output
        });
        return {views: {...topology, ...output}};
      }
      const parameters = upload('parameters', new Float32Array([10, 0, 0, 1]), 'float32');
      if (kind === 'network-disruption-recipe') {
        const outputs = await runDisruption(context.device, graph, {
          search: {
            topology,
            startNodes: upload('starts', new Uint32Array([0]), 'uint32'),
            maximumIterations: 16
          },
          rowCount: 1,
          parameters,
          opportunityWeights: upload('opportunities', new Float32Array([1, 1, 1, 1]), 'float32'),
          edgeMask: upload('edgeMask', new Uint32Array([0, 1, 1, 1, 1]), 'uint32')
        });
        return {views: {...topology, ...outputs}};
      }
      const outputs = await runFacility(context.device, graph, {
        costs: upload('facilityCosts', new Float32Array([0, 1, 2, 2, 4, 5, 0, 1]), 'float32'),
        demand: upload('demand', new Float32Array([1, 2, 3, 4]), 'float32'),
        supply: upload('supply', new Float32Array([5, 10]), 'float32'),
        facilityMask: upload('facilityMask', new Uint32Array([1, 0]), 'uint32'),
        parameters
      });
      return {views: {...topology, ...outputs}};
    },
    capture(final, _context, captured) {
      if (kind === 'alternative-routes' && final('complete').values[0] !== 1)
        throw new Error('Synthetic route search was incomplete');
      if (kind === 'network-strongly-connected-components' && final('converged').values[0] !== 1)
        throw new Error('Synthetic strong-component search did not converge');
      if (
        kind === 'k-shortest-paths' &&
        (final('converged').values[0] !== 1 ||
          final('candidateOverflow').values[0] !== 0 ||
          final('iterationLimitReached').values[0] !== 0)
      )
        throw new Error('Synthetic k-shortest path search did not complete');
      return captured.map(capture => ({
        ...capture,
        ...(kind === 'network-strongly-connected-components' && {
          role:
            capture.name === 'offsets' || capture.name === 'neighbors'
              ? ('input' as const)
              : ('output' as const)
        }),
        keySpace:
          capture.name === 'offsets' || capture.name === 'componentOffsets'
            ? 'node-offset'
            : capture.values.length === 5
              ? 'edge'
              : capture.values.length === 8
                ? 'route-vertex'
                : capture.values.length === 4
                  ? 'node'
                  : capture.values.length === 2
                    ? 'route'
                    : 'query'
      }));
    }
  };
}
export function runExpansion(device: Device, kind: ExpansionKind) {
  return recordNetworkElement(device, createNetworkExpansionScenario(kind));
}
export function describeExpansion(device: Device, kind: ExpansionKind) {
  return describeNetworkElement(device, createNetworkExpansionScenario(kind));
}
