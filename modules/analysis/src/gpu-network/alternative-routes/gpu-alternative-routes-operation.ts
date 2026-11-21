// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUAlternativeRoutes, getGPUAlternativeRoutesOutputLayout} from './gpu-alternative-routes';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** Graph-value properties. Route and path padding dimensions are explicit. */
export type AlternativeRoutesOperationProps = {
  id?: string;
  nodeCount: number;
  queryCount: number;
  topology: {offsets: string; neighbors: string; weights: string; nodePositions?: string};
  sources: string;
  targets: string;
  routeCount: number;
  maximumExpansions?: number;
  maximumFrontierSize?: number;
  output: {
    vertices: string;
    edges: string;
    lengths: string;
    costs: string;
    counts: string;
    complete: string;
  };
};
/** Captures the same values for CPU execution and GPU descriptor construction. @internal */
export function snapshotAlternativeRoutesOperationProps(
  props: AlternativeRoutesOperationProps
): AlternativeRoutesOperationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'nodeCount',
    'queryCount',
    'topology',
    'sources',
    'targets',
    'routeCount',
    'maximumExpansions',
    'maximumFrontierSize',
    'output'
  ]);
  return Object.freeze({
    ...snapshot,
    topology: snapshotConstructionFields(snapshot.topology, [
      'offsets',
      'neighbors',
      'weights',
      'nodePositions'
    ]),
    output: snapshotConstructionFields(snapshot.output, [
      'vertices',
      'edges',
      'lengths',
      'costs',
      'counts',
      'complete'
    ])
  });
}

/** Vector operation form, retaining the contributor's per-query counts and complete flag. */
export class GPUAlternativeRoutesOperation extends GPUContributorOperation {
  constructor(props: AlternativeRoutesOperationProps) {
    props = snapshotAlternativeRoutesOperationProps(props);
    const id = props.id ?? 'alternative-routes';
    getGPUAlternativeRoutesOutputLayout({
      id,
      nodeCount: props.nodeCount,
      queryCount: props.queryCount,
      routeCount: props.routeCount
    });
    const inputs: Record<string, string> = {
      offsets: props.topology.offsets,
      neighbors: props.topology.neighbors,
      weights: props.topology.weights,
      sources: props.sources,
      targets: props.targets
    };
    if (props.topology.nodePositions !== undefined)
      inputs['nodePositions'] = props.topology.nodePositions;
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    for (const [port, value] of Object.entries(props.output))
      outputs[port] = {
        value,
        format: port === 'costs' ? 'float32' : 'uint32',
        length:
          props.queryCount *
          (['counts', 'complete'].includes(port)
            ? 1
            : props.routeCount * (['vertices', 'edges'].includes(port) ? props.nodeCount : 1))
      };
    super({
      emptyOutputPorts: ['vertices', 'edges', 'lengths', 'costs', 'counts', 'complete'],
      id,
      inputs,
      outputs,
      createContributor: getView =>
        new GPUAlternativeRoutes({
          id: props.id,
          topology: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            nodePositions:
              props.topology.nodePositions === undefined
                ? undefined
                : getView<'float32x2'>('nodePositions')
          },
          sources: getView<'uint32'>('sources'),
          targets: getView<'uint32'>('targets'),
          routeCount: props.routeCount,
          maximumExpansions: props.maximumExpansions,
          maximumFrontierSize: props.maximumFrontierSize,
          output: {
            vertices: getView<'uint32'>('vertices'),
            edges: getView<'uint32'>('edges'),
            lengths: getView<'uint32'>('lengths'),
            costs: getView<'float32'>('costs'),
            counts: getView<'uint32'>('counts'),
            complete: getView<'uint32'>('complete')
          }
        })
    });
  }
}
