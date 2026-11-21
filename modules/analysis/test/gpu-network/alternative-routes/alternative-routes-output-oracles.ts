// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {AlternativeRoutesOperationProps} from '../../../src/gpu-network/alternative-routes/gpu-alternative-routes-operation';
import type {OperationGraphInput} from '../../utils/mixed-execution-test-utils';

export const alternativeRoutesInput = {
  topology: {
    offsets: Uint32Array.of(0, 2, 3, 4, 4),
    neighbors: Uint32Array.of(1, 2, 3, 3),
    weights: Float32Array.of(1, 1, 1, 2)
  },
  sources: Uint32Array.of(0, 3),
  targets: Uint32Array.of(3, 0)
};

// By hand: the only 0->3 paths are (0,1,3), edges (0,2), cost 2 and
// (0,2,3), edges (1,3), cost 3. There is no 3->0 path. The third route is padding.
export const alternativeRoutesExpected = {
  vertices: [0, 1, 3, 4294967295, 0, 2, 3, 4294967295, ...new Array<number>(16).fill(4294967295)],
  edges: [0, 2, 4294967295, 4294967295, 1, 3, ...new Array<number>(18).fill(4294967295)],
  lengths: [3, 3, 0, 0, 0, 0],
  costs: [2, 3, Infinity, Infinity, Infinity, Infinity],
  counts: [2, 0],
  complete: [1, 1]
};

export function createAlternativeBudgetOracle(scene: 'expansion' | 'frontier' | 'retained') {
  // Chain 0->1->2 needs two examinations. A one-examination budget stops before edge 1.
  // The branching scene 0->{1,2} overflows a one-prefix heap while scanning the source.
  // In the retained scene, 0->1 is published first; expanding 2 then overflows on {3,4}.
  const offsets =
    scene === 'expansion'
      ? [0, 1, 2, 2]
      : scene === 'frontier'
        ? [0, 2, 2, 2]
        : [0, 2, 2, 5, 5, 5, 5];
  const neighbors = scene === 'retained' ? [1, 2, 3, 4, 5] : [1, 2];
  const props: AlternativeRoutesOperationProps = {
    nodeCount: offsets.length - 1,
    queryCount: 1,
    routeCount: scene === 'retained' ? 2 : 1,
    maximumExpansions: scene === 'expansion' ? 1 : 16,
    maximumFrontierSize: scene === 'expansion' ? 4 : scene === 'frontier' ? 1 : 2,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    sources: 'sources',
    targets: 'targets',
    output: {
      vertices: 'vertices',
      edges: 'edges',
      lengths: 'lengths',
      costs: 'costs',
      counts: 'counts',
      complete: 'complete'
    }
  };
  const inputs: Record<string, OperationGraphInput> = {
    offsets: {format: 'uint32', values: Uint32Array.from(offsets)},
    neighbors: {format: 'uint32', values: Uint32Array.from(neighbors)},
    weights: {
      format: 'float32',
      values: Float32Array.from(neighbors, (_, row) => (scene === 'retained' && row === 1 ? 2 : 1))
    },
    sources: {format: 'uint32', values: Uint32Array.of(0)},
    targets: {format: 'uint32', values: Uint32Array.of(scene === 'expansion' ? 2 : 1)}
  };
  const pathRows = props.nodeCount * props.routeCount;
  const vertices = new Array<number>(pathRows).fill(4294967295);
  const edges = new Array<number>(pathRows).fill(4294967295);
  if (scene === 'retained') {
    vertices[0] = 0;
    vertices[1] = 1;
    edges[0] = 0;
  }
  const expected = {
    vertices,
    edges,
    lengths: scene === 'retained' ? [2, 0] : [0],
    costs: scene === 'retained' ? [1, Infinity] : [Infinity],
    counts: [scene === 'retained' ? 1 : 0],
    complete: [0]
  };
  return {props, inputs, expected};
}
