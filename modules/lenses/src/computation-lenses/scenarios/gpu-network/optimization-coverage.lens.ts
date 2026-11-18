// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUNetworkLouvainParameterValues} from '@luma-contributors/analysis/gpu-network';
import {runExample as runNetworkPointRouting} from './network-point-routing.example';
import {runExample as runNetworkFlow} from './network-flow.example';
import {runExample as runNetworkLouvain} from './network-louvain.example';
import {runExample as runNetworkFacilityOptimization} from './network-facility-optimization.example';
import {runExample as runNetworkTimeRouting} from './network-time-routing.example';
import {runExample as runNetworkRationalAccess} from './network-rational-access.example';
import {Buffer, type Device, type VertexFormat} from '@luma.gl/core';
import {createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  recordNetworkElement,
  describeNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';

export type NetworkOptimizationKind =
  | 'point-routing'
  | 'flow'
  | 'louvain'
  | 'louvain-sparse'
  | 'facility-optimization'
  | 'time-routing'
  | 'rational-access';

/** Builds the network scenario, including its retained outputs and capture checks. @internal */
export function createNetworkOptimizationScenario(
  kind: NetworkOptimizationKind
): NetworkElementScenario {
  return {
    id:
      kind === 'louvain-sparse'
        ? 'gpu-network.network-louvain.sparse'
        : `gpu-network.network-${kind}`,
    weights: 'lengthMeters',
    withoutRoads: true,
    roadContext: false,
    datasetVersion: `synthetic-network-${kind}-v1`,
    seed: 0,
    retain:
      kind === 'flow'
        ? ['flow', 'cut', 'summary']
        : kind === 'point-routing'
          ? ['costs', 'status', 'predecessorEdges']
          : kind === 'louvain' || kind === 'louvain-sparse'
            ? ['communities', 'objectiveTrace', 'summary']
            : kind === 'time-routing'
              ? ['arrivals', 'predecessorTransitions', 'summary']
              : kind === 'facility-optimization'
                ? ['selectedFacilities', 'assignments', 'objective', 'summary']
                : ['assignments', 'scores', 'unservedDemand', 'summary'],
    formulas: [],
    keySpaces: [{id: 'entry', label: 'Scenario input/output row', indexSpace: '1d'}],
    entityKeySpace: 'entry',
    choices: [
      {
        stage: 'representation',
        name: 'dataset',
        value: `Explicit small ${kind} scenario`,
        sweepable: false
      }
    ],
    async build(context) {
      const {graph, upload} = context;
      const views: Record<string, GraphDataView> = {};
      const floats = (name: string, values: number[]) => {
        const view = upload(name, Float32Array.from(values), 'float32');
        views[name] = view;
        return view;
      };
      const integers = (name: string, values: number[]) => {
        const view = upload(name, Uint32Array.from(values), 'uint32');
        views[name] = view;
        return view;
      };
      const integerPairs = (name: string, values: number[]) => {
        const scalar = integers(`${name}-words`, values);
        const view = graph.createDataView(scalar.buffer, {
          format: 'uint32x2',
          length: values.length / 2
        });
        views[name] = view;
        delete views[`${name}-words`];
        return view;
      };
      const floatPairs = (name: string, values: number[]) => {
        const view = upload(name, Float32Array.from(values), 'float32x2');
        views[name] = view;
        return view;
      };
      const output = <Format extends VertexFormat>(
        name: string,
        format: Format,
        length: number
      ) => {
        const view = createTransientView(
          graph,
          name,
          format,
          length,
          Buffer.STORAGE | Buffer.COPY_SRC
        );
        views[name] = view;
        return view;
      };
      if (kind === 'flow') {
        await runNetworkFlow(graph, {
          nodeCount: 4,
          edges: integerPairs('edges', [0, 1, 0, 2, 1, 2, 2, 1, 1, 3, 2, 3]),
          capacities: integers('capacities', [3, 2, 1, 1, 2, 3]),
          parameters: integers('parameters', [0, 3, 0xffffffff]),
          output: {
            flow: output('flow', 'uint32', 6),
            cut: output('cut', 'uint32', 4),
            summary: output('summary', 'uint32', 8)
          }
        });
      } else if (kind === 'point-routing') {
        await runNetworkPointRouting(graph, {
          costs: floats('edgeCosts', [10, 20, 30]),
          turns: integerPairs('turns', [0, 1, 1, 2]),
          queries: integerPairs('queries', [0, 0, 0, 2, 2, 0]),
          fractions: floatPairs('fractions', [0.2, 0.7, 0.5, 0.5, 0.2, 0.7]),
          output: {
            costs: output('costs', 'float32', 3),
            status: output('status', 'uint32', 3),
            predecessorEdges: output('predecessorEdges', 'uint32', 9)
          }
        });
      } else if (kind === 'louvain' || kind === 'louvain-sparse') {
        // Two disconnected pairs need one contraction and one convergence level, not four.
        const maximumLevels = kind === 'louvain-sparse' ? 2 : 4;
        await runNetworkLouvain(graph, {
          nodeCount: 4,
          maximumLevels,
          maximumIterations: 4,
          ...(kind === 'louvain-sparse'
            ? {
                topology: {
                  offsets: integers('offsets', [0, 1, 2, 3, 4]),
                  neighbors: integers('neighbors', [1, 0, 3, 2]),
                  weights: floats('weights', [1, 1, 1, 1])
                }
              }
            : {
                weights: floats('weights', [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
              }),
          parameters: floats('parameters', Array.from(getGPUNetworkLouvainParameterValues())),
          output: {
            communities: output('communities', 'uint32', 4),
            objectiveTrace: output('objectiveTrace', 'float32', 1 + maximumLevels * 4),
            summary: output('summary', 'uint32', 4)
          }
        });
      } else if (kind === 'time-routing') {
        await runNetworkTimeRouting(graph, {
          stateVertices: integers('stateVertices', [0, 1, 2]),
          transitions: integerPairs('transitions', [0, 1, 1, 2, 0, 2]),
          costs: floatPairs('edgeCosts', [1, 0, 1, 1, 10, 0]),
          endpoints: integers('endpoints', [0, 2]),
          departure: floats('departure', [3]),
          output: {
            arrivals: output('arrivals', 'float32', 3),
            predecessorTransitions: output('predecessorTransitions', 'uint32', 3),
            summary: output('summary', 'uint32', 4)
          }
        });
      } else if (kind === 'facility-optimization') {
        await runNetworkFacilityOptimization(graph, {
          operation: 'capacitated-p-median',
          facilityCount: 2,
          selectCount: 2,
          costs: floats('facilityCosts', [1, 2, 1, Infinity]),
          demand: floats('demand', [1, 1]),
          capacities: floats('capacities', [1, 1]),
          threshold: floats('threshold', [0]),
          output: {
            selectedFacilities: output('selectedFacilities', 'uint32', 2),
            assignments: output('assignments', 'uint32', 2),
            objective: output('objective', 'float32', 1),
            summary: output('summary', 'uint32', 4)
          }
        });
      } else {
        await runNetworkRationalAccess(graph, {
          costs: floats('travelCosts', [0, 0, Infinity, Infinity]),
          demand: floats('demand', [10, 3]),
          supply: floats('supply', [5, 5]),
          parameters: floats('parameters', [1, 1, 1, 1, 50, 0.0001]),
          maximumCycles: 4,
          initialLimitCycles: 0,
          output: {
            assignments: output('assignments', 'float32', 4),
            scores: output('scores', 'float32', 2),
            unservedDemand: output('unservedDemand', 'float32', 2),
            summary: output('summary', 'uint32', 4)
          }
        });
      }
      return {views};
    },
    capture(final, _context, captures) {
      if (kind === 'point-routing') {
        if (Array.from(final('status').values).some(value => value !== 0))
          throw new Error('Point-routing example did not complete');
      } else if (
        kind === 'flow' ||
        kind === 'louvain' ||
        kind === 'louvain-sparse' ||
        kind === 'time-routing'
      ) {
        if (final('summary').values[0] !== 1 || final('summary').values[1] !== 1)
          throw new Error('Network example did not converge');
      } else if (final('summary').values[0] !== 1 || final('summary').values[1] !== 1) {
        throw new Error('Network allocation example did not produce a valid result');
      }
      return captures.map(capture => ({...capture, keySpace: 'entry'}));
    }
  };
}

export function runNetworkOptimizationCoverage(device: Device, kind: NetworkOptimizationKind) {
  return recordNetworkElement(device, createNetworkOptimizationScenario(kind));
}
export function describeNetworkOptimizationCoverage(device: Device, kind: NetworkOptimizationKind) {
  return describeNetworkElement(device, createNetworkOptimizationScenario(kind));
}
