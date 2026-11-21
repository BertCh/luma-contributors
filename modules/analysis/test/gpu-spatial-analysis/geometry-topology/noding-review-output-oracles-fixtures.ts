// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPULineNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-line-noding';
import {GPUSnapRoundedNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding';
import type {OracleAllocate} from './noding-polygonize-output-oracles-fixtures';

export const REVIEW_NODING_CASES = [
  'translated',
  'translated-snap',
  'retrace',
  'retrace-snap',
  'large',
  'small',
  'ordinary',
  'ordinary-snap',
  'uncertain',
  'snapped'
] as const;
export type ReviewNodingCase = (typeof REVIEW_NODING_CASES)[number];

/**
 * The moved crossing projects to 52/160 = 13/40 on the lattice. Rounded division
 * gives 0.32499998807907104; multiplying by a float32 reciprocal gives the next
 * float, 0.32500001788139343. Allow one float32 ULP (2^-25 at this magnitude)
 * for that non-dyadic parameter only. Zero, one and the other edge's 1/4 remain
 * exact, and non-finite values must fail.
 */
export function matchesSnappedReviewParameters(
  actual: readonly number[],
  expected: readonly number[]
): boolean {
  return (
    actual.length === expected.length &&
    actual.every(
      (value, index) =>
        Number.isFinite(value) &&
        Number.isFinite(expected[index]) &&
        (expected[index] === 0.32499998807907104
          ? Math.abs(value - expected[index]) <= 2 ** -25
          : Object.is(value, expected[index]))
    )
  );
}

export function createReviewNodingCase(
  graph: GPUCommandGraph,
  allocate: OracleAllocate,
  subject: ReviewNodingCase,
  uncertainty = true
) {
  const geometryCase =
    subject === 'translated-snap'
      ? 'translated'
      : subject === 'retrace-snap'
        ? 'retrace'
        : subject === 'ordinary-snap'
          ? 'ordinary'
          : subject;
  const coordinates = {
    translated: [1000000.0625, 0, 1000000.125, 0, 1000000.1875, 0],
    retrace: [0, 0, 2, 0, 0, 0],
    large: [0, 0, 1e20, 0],
    small: [0, 0, 1e-25, 0],
    ordinary: [0, 0, 12, 4, 9, 3, 9, 5],
    uncertain: [1e-30, 1e-30, 2e-30, 2e-30, 1.5e-30, 1.5e-30, 1e38, 1e38],
    snapped: [0, 0, 6, 2, 2, 0, 2, 2]
  }[geometryCase];
  const multiple = subject === 'uncertain' || subject === 'snapped' || geometryCase === 'ordinary';
  const scalar = (name: string) => allocate(name, 'uint32', 1);
  const output = {
    endpoints: allocate('review-endpoints', 'float32x4', 4),
    sourceFeatureIds: allocate('review-features', 'uint32', 4),
    sourceRingIds: allocate('review-rings', 'uint32', 4),
    sourceEdgeIds: allocate('review-edges', 'uint32', 4),
    sourceStartParameters: allocate('review-starts', 'float32', 4),
    sourceEndParameters: allocate('review-ends', 'float32', 4),
    status: {
      count: scalar('review-count'),
      requiredCount: scalar('review-required'),
      overflow: scalar('review-overflow'),
      candidateOverflow: scalar('review-candidate'),
      ...(uncertainty ? {uncertainCount: scalar('review-uncertain')} : {})
    }
  };
  const lines = {
    kind: 'lines' as const,
    positions: allocate('review-positions', 'float32x2', coordinates.length / 2, coordinates),
    lineOffsets: allocate(
      'review-offsets',
      'uint32',
      multiple ? 3 : 2,
      multiple ? [0, 2, 4] : [0, coordinates.length / 2]
    )
  };
  const expected: Record<string, number[]> = {
    'status.overflow': [0],
    'status.candidateOverflow': [subject === 'uncertain' ? 1 : 0],
    ...(uncertainty ? {'status.uncertainCount': [subject === 'uncertain' ? 1 : 0]} : {})
  };
  if (subject !== 'uncertain') {
    const count =
      geometryCase === 'ordinary'
        ? 3
        : subject === 'snapped'
          ? 4
          : subject === 'large' || subject === 'small'
            ? 1
            : 2;
    Object.assign(expected, {
      'status.count': [count],
      'status.requiredCount': [count],
      sourceEdgeIds: subject === 'snapped' ? [0, 0, 2, 2] : [0, 1].slice(0, count),
      sourceStartParameters:
        subject === 'snapped' ? [0, 0.32499998807907104, 0, 0.25] : new Array(count).fill(0),
      sourceEndParameters:
        subject === 'snapped' ? [0.32499998807907104, 1, 0.25, 1] : new Array(count).fill(1)
    });
    if (subject === 'snapped')
      expected['endpoints'] = [0, 0, 2, 0.5, 2, 0.5, 6, 2, 2, 0, 2, 0.5, 2, 0.5, 2, 2];
    if (geometryCase === 'ordinary')
      Object.assign(expected, {
        endpoints: [0, 0, 9, 3, 9, 3, 12, 4, 9, 3, 9, 5],
        sourceEdgeIds: [0, 0, 2],
        sourceStartParameters: [0, 0.75, 0],
        sourceEndParameters: [0.75, 1, 1]
      });
  }
  // R2: original traversal identifies the edge, including a retraced edge; a rounded
  // midpoint cannot identify it. R3: nonzero finite horizontal directions have [0,1]
  // parameters at either scale, irrespective of overflow/underflow in their square.
  // R5: one candidate's float32 exponent span exceeds the exact predicate's documented
  // certifiable range (200 binary orders). It must be counted, not silently certified.
  // R6: rounded crossing (2,.5) projects onto (6,2) at (12+1)/40=13/40;
  // the vertical source projection is .5/2=1/4. Literals are IEEE float32 values.
  const contributor =
    subject === 'snapped' || subject.endsWith('-snap')
      ? new GPUSnapRoundedNoding({
          lines,
          intersectionCapacity: 8,
          gridSize: subject === 'translated-snap' ? 1 / 16 : 0.3,
          output: {
            ...output,
            lattice: allocate('review-lattice', 'float32', 4),
            latticeEndpoints: allocate('review-lattice-endpoints', 'float32x4', 4)
          }
        })
      : new GPULineNoding({lines, intersectionCapacity: 8, output});
  return {output, expected, nodes: contributor.getCommandNodes(graph)};
}
