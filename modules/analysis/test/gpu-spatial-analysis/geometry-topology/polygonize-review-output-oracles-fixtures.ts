// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUPolygonize,
  type GPUPolygonizeProps
} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import {createTopologyOutput} from '../geometry-refinement/topology-output';
import type {OracleAllocate} from './noding-polygonize-output-oracles-fixtures';

export type ReviewPolygonizeSubject =
  | 'spur'
  | 'chain'
  | 'short-spur'
  | 'overlap-square-first'
  | 'overlap-exterior-first';

export function createReviewPolygonizeProps(
  allocate: OracleAllocate,
  subject: ReviewPolygonizeSubject = 'spur'
): GPUPolygonizeProps {
  let serial = 0;
  const output = createTopologyOutput((format, length) =>
    allocate(`review-polygon-${serial++}`, format, length)
  );
  const overlap = subject.startsWith('overlap-');
  const exteriorFirst = subject === 'overlap-exterior-first';
  const square = [0, 0, 4, 0, 4, 4, 0, 4, 0, 3 / 32];
  const exterior = [-1, -1, 0, -3 / 32];
  const coordinates =
    subject === 'chain'
      ? [0, 0, 1, 0, 2, 0, 3, 0]
      : subject === 'short-spur'
        ? [0, 0, 4, 0, 4, 4, 0, 4, 0, 0, 0, 0, 1, 1, 1 + 1 / 128, 1 + 1 / 128]
        : overlap
          ? exteriorFirst
            ? [...exterior, ...square]
            : [...square, ...exterior]
          : [0, 0, 4, 0, 4, 4, 0, 4, 0, 0, 0, 0, 1, 1];
  const offsets =
    subject === 'chain'
      ? [0, 4]
      : subject === 'short-spur'
        ? [0, 5, 8]
        : exteriorFirst
          ? [0, 2, 7]
          : [0, 5, 7];
  return {
    lines: {
      kind: 'lines',
      positions: allocate('review-polygon-input', 'float32x2', coordinates.length / 2, coordinates),
      lineOffsets: allocate('review-polygon-offsets', 'uint32', offsets.length, offsets)
    },
    intersectionCapacity: 16,
    precision: {
      vertexTolerance: overlap ? 1 / 8 : 1 / 64,
      predicates: 'exact-or-uncertain',
      storage: 'float32'
    },
    output
  };
}

export function createReviewPolygonizeCase(
  graph: GPUCommandGraph,
  allocate: OracleAllocate,
  subject: ReviewPolygonizeSubject
) {
  const props = createReviewPolygonizeProps(allocate, subject);
  const hasSquare = subject !== 'chain';
  const dangles = subject === 'chain' ? 3 : subject === 'short-spur' ? 2 : 1;
  const exteriorFirst = subject === 'overlap-exterior-first';
  // Shapely polygonize_full: inward spur -> one area-16 polygon, one dangle;
  // open chain -> no polygons, three atomic dangles. No cut edges in either.
  const expected: Record<string, number[]> = {
    'status.count': [hasSquare ? 1 : 0],
    'status.overflow': [0],
    'status.candidateOverflow': [0],
    'diagnostics.edgeClasses': hasSquare
      ? exteriorFirst
        ? [26, 1, 1, 1, 1]
        : [1, 1, 1, 1, ...new Array(dangles).fill(26)]
      : [26, 26, 26],
    'diagnostics.closedRingCount': [hasSquare ? 4 : 0],
    'diagnostics.openChainCount': [dangles],
    'diagnostics.cutEdgeCount': [0],
    'diagnostics.dangleCount': [dangles],
    'diagnostics.unusedEdgeCount': [dangles],
    ...(hasSquare
      ? {
          'polygons.ringOffsets': [0, 4],
          'polygons.positions': [0, 0, 4, 0, 4, 4, 0, 4],
          'directedEdges.ringIds': exteriorFirst
            ? [0xffffffff, 0xffffffff, 0, 1, 0, 1, 0, 1, 0, 1]
            : [0, 1, 0, 1, 0, 1, 0, 1, ...new Array(2 * dangles).fill(0xffffffff)],
          'directedEdges.flags': exteriorFirst
            ? [2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
            : [0, 0, 0, 0, 0, 0, 0, 0, ...new Array(2 * dangles).fill(2)]
        }
      : {
          'directedEdges.ringIds': new Array(6).fill(0xffffffff),
          'directedEdges.flags': [2, 2, 2, 2, 2, 2]
        })
  };
  return {output: props.output, expected, nodes: new GPUPolygonize(props).getCommandNodes(graph)};
}
