// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPULineSplit} from '../../../src/gpu-spatial-analysis/line-split/gpu-line-split';
import {GPULineClipByPolygon} from '../../../src/gpu-spatial-analysis/line-clip/gpu-line-clip-by-polygon';
import {GPUSharedPaths} from '../../../src/gpu-spatial-analysis/line-clip/gpu-shared-paths';
import {GPUGeometryPredicates} from '../../../src/gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates';

export type WrapperView = <Format extends 'uint32' | 'float32x2'>(
  format: Format,
  length: number,
  values?: number[]
) => GraphDataView<Format>;

export const LOSS_WRAPPERS = ['split', 'clip', 'shared', 'predicates'] as const;

export function createLateCrossingFixture(leafCapacity: number, view: WrapperView) {
  const lines = {
    kind: 'lines' as const,
    positions: view('float32x2', 9, [-5, 0, -4, 0, -3, 0, -2, 0, -1, 0, 0, 0, 2, 2, 0, 2, 2, 0]),
    lineOffsets: view('uint32', 2, [0, 9])
  };
  const intersectionCount = view('uint32', 1);
  const predicateOverflow = view('uint32', 1);
  const simple = view('uint32', 1);
  const pieceCount = view('uint32', 1);
  const pieceOverflow = view('uint32', 1);
  const candidateOverflow = view('uint32', 1);
  const predicates = new GPUGeometryPredicates({
    geometry: lines,
    leafCapacity,
    intersectionCapacity: 32,
    output: {isSimple: simple, status: {intersectionCount, overflow: predicateOverflow}}
  });
  const split = new GPULineSplit({
    lines,
    leafCapacity,
    intersectionCapacity: 32,
    output: {
      pieces: {
        geometry: {
          kind: 'lines',
          positions: view('float32x2', 64),
          lineOffsets: view('uint32', 33)
        },
        sourceIds: view('uint32', 32),
        status: {count: pieceCount, overflow: pieceOverflow, candidateOverflow}
      }
    }
  });
  return {
    predicates,
    split,
    intersectionCount,
    predicateOverflow,
    simple,
    pieceCount,
    pieceOverflow,
    candidateOverflow
  };
}

export function createLateSharedFixture(leafCapacity: number, view: WrapperView) {
  const count = view('uint32', 1);
  const overflow = view('uint32', 1);
  const candidateOverflow = view('uint32', 1);
  const contributor = new GPUSharedPaths({
    leafCapacity,
    intersectionCapacity: 32,
    left: {
      kind: 'lines',
      positions: view('float32x2', 2, [0, 2, 1, 2]),
      lineOffsets: view('uint32', 2, [0, 2])
    },
    right: {
      kind: 'lines',
      positions: view('float32x2', 7, [10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 0, 2, 1, 2]),
      lineOffsets: view('uint32', 2, [0, 7])
    },
    output: {
      runs: {
        positions: view('float32x2', 16),
        offsets: view('uint32', 5),
        leftSourceIds: view('uint32', 4),
        rightSourceIds: view('uint32', 4),
        forward: view('uint32', 4)
      },
      status: {count, overflow, candidateOverflow}
    }
  });
  return {contributor, count, overflow, candidateOverflow};
}

export function createCapacityBoundaryFixture(
  kind: 'split' | 'clip' | 'shared',
  view: WrapperView
) {
  // Two disjoint two-vertex lines, and only two output vertex slots: retain exactly line 0.
  const lines = {
    kind: 'lines' as const,
    positions: view('float32x2', 4, [0, 0, 1, 0, 0, 2, 1, 2]),
    lineOffsets: view('uint32', 3, [0, 2, 4])
  };
  const positions = view('float32x2', 2);
  const offsets = view('uint32', 3);
  const count = view('uint32', 1);
  const overflow = view('uint32', 1);
  const candidateOverflow = view('uint32', 1);
  const status = {count, overflow, candidateOverflow};
  const pieces = {
    geometry: {kind: 'lines' as const, positions, lineOffsets: offsets},
    sourceIds: view('uint32', 2),
    status
  };
  const contributor =
    kind === 'split'
      ? new GPULineSplit({lines, intersectionCapacity: 16, output: {pieces}})
      : kind === 'clip'
        ? new GPULineClipByPolygon({
            lines,
            intersectionCapacity: 16,
            candidateCapacity: 64,
            polygons: {
              kind: 'polygons',
              positions: view('float32x2', 4, [-1, -1, 2, -1, 2, 3, -1, 3]),
              ringOffsets: view('uint32', 2, [0, 4]),
              polygonOffsets: view('uint32', 2, [0, 1]),
              featureOffsets: view('uint32', 2, [0, 1])
            },
            output: {pieces}
          })
        : new GPUSharedPaths({
            left: lines,
            right: lines,
            intersectionCapacity: 16,
            output: {
              runs: {
                positions,
                offsets,
                leftSourceIds: view('uint32', 2),
                rightSourceIds: view('uint32', 2),
                forward: view('uint32', 2)
              },
              status
            }
          });
  return {contributor, positions, offsets, count, overflow, candidateOverflow};
}

export function createWrapperLossFixture(
  kind: (typeof LOSS_WRAPPERS)[number],
  view: WrapperView,
  leafCapacity = 4
) {
  // Six live edges exceed four leaves; 64 pairs/pieces and 256 vertices leave final capacity spare.
  const lines = {
    kind: 'lines' as const,
    positions: view('float32x2', 7, [0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]),
    lineOffsets: view('uint32', 2, [0, 7])
  };
  const isSimple = view('uint32', 1);
  const overflow = view('uint32', 1);
  const candidateOverflow = view('uint32', 1);
  const requiredCount = view('uint32', 1);
  const count = view('uint32', 1);
  const status = {count, overflow, candidateOverflow, requiredCount};
  const pieces = {
    geometry: {
      kind: 'lines' as const,
      positions: view('float32x2', 256),
      lineOffsets: view('uint32', 65)
    },
    sourceIds: view('uint32', 64),
    status
  };
  const common = {leafCapacity, intersectionCapacity: 64};
  const contributor =
    kind === 'split'
      ? new GPULineSplit({...common, lines, output: {pieces}})
      : kind === 'clip'
        ? new GPULineClipByPolygon({
            ...common,
            lines,
            polygons: {
              kind: 'polygons',
              positions: view('float32x2', 6, [0, -1, 2, -1, 4, -1, 6, 1, 4, 1, 0, 1]),
              ringOffsets: view('uint32', 2, [0, 6]),
              polygonOffsets: view('uint32', 2, [0, 1]),
              featureOffsets: view('uint32', 2, [0, 1])
            },
            candidateCapacity: 512,
            output: {pieces}
          })
        : kind === 'shared'
          ? new GPUSharedPaths({
              ...common,
              left: lines,
              right: lines,
              output: {
                runs: {
                  positions: pieces.geometry.positions,
                  offsets: pieces.geometry.lineOffsets,
                  leftSourceIds: pieces.sourceIds,
                  rightSourceIds: view('uint32', 64),
                  forward: view('uint32', 64)
                },
                status
              }
            })
          : new GPUGeometryPredicates({
              ...common,
              geometry: lines,
              output: {isSimple, status: {overflow}}
            });
  return {contributor, overflow, candidateOverflow, requiredCount, count, isSimple};
}

export function createStatusOnlyFixture(
  request: 'count' | 'overflow' | 'both',
  view: WrapperView,
  intersectionCapacity = 1
) {
  const values =
    request === 'overflow' ? [0, 0, 4, 4, 0, 4, 4, 0, 0, 2, 4, 2] : [0, 0, 1, 1, 0, 1, 1, 0];
  const positions = view('float32x2', values.length / 2, values);
  const cleanPositions = Float32Array.from(
    Array.from({length: values.length / 2}, (_, index) => [index, 0]).flat()
  );
  const intersectionCount = view('uint32', 1, [99]);
  const overflow = view('uint32', 1, [99]);
  const contributor = new GPUGeometryPredicates({
    id: 'status-only',
    geometry: {kind: 'lines', positions, lineOffsets: view('uint32', 2, [0, values.length / 2])},
    intersectionCapacity,
    output: {
      status: {
        ...(request !== 'overflow' ? {intersectionCount} : {}),
        ...(request !== 'count' ? {overflow} : {})
      }
    }
  });
  return {contributor, positions, intersectionCount, overflow, cleanPositions};
}
