// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CPUTrajectoryRestrictionInput,
  CPUTrajectoryRestrictionOptions
} from '../../../src/gpu-spatial-analysis/trajectory-restriction/cpu-trajectory-restriction-operation';
import type {TrajectoryRestrictionOperationProps} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction-operation';
import {getInt64TimeWords} from '../../../src/dataframe-analysis/time-window-filter/time-words';
import type {OperationGraphInput} from '../../utils/mixed-execution-test-utils';

/** Shared input scenes only; the node spec supplies independent hand-computed expectations. */
export type RestrictionScene = {
  input: CPUTrajectoryRestrictionInput;
  options: CPUTrajectoryRestrictionOptions;
};

export function createRectangleScene(): RestrictionScene {
  return {
    input: {
      positions: Float32Array.of(-2, 0, 2, 0, 2, 2, 0, 0, 0, 0),
      times: Float32Array.of(0, 4, 6, 0, 3),
      offsets: Uint32Array.of(0, 3, 5),
      parameters: Float32Array.of(0, 6, -1, -1, 1, 1, 0)
    },
    options: {}
  };
}

export function createPolygonScene(): RestrictionScene {
  return {
    input: {
      positions: Float32Array.of(-2, 0, 10, 0, 1, -1, 7, -1),
      times: Float32Array.of(0, 12, 0, 6),
      offsets: Uint32Array.of(0, 2, 4),
      parameters: Float32Array.of(
        0,
        12,
        0,
        0,
        0,
        0,
        0,
        0,
        -2,
        8,
        -2,
        8,
        2,
        0,
        2,
        2,
        -1,
        6,
        -1,
        6,
        1,
        2,
        1
      )
    },
    options: {
      polygonRingVertexCounts: [4, 4],
      intervalsPerSegment: 4,
      compactCapacity: 6,
      mergeIntervals: true
    }
  };
}

export function createPairScene(): RestrictionScene {
  return {
    input: {
      positions: Float32Array.of(-2, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 2),
      times: Float32Array.of(0, 4, 0, 4, 0, 4),
      offsets: Uint32Array.of(0, 2, 4, 6),
      parameters: Float32Array.of(0, 4, 1, 0, 0, 0, 0),
      pairs: {
        first: Uint32Array.of(0, 1, 0, 1, 0),
        second: Uint32Array.of(1, 1, 2, 99, 1),
        count: Uint32Array.of(4)
      }
    },
    options: {intervalsPerPair: 8}
  };
}

export function createStaggeredPairScene(): RestrictionScene {
  return {
    input: {
      positions: Float32Array.of(-2, 0, -1, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0),
      times: Float32Array.of(0, 1, 3, 4, 0, 2, 4),
      offsets: Uint32Array.of(0, 4, 7),
      parameters: Float32Array.of(0, 4, 1, 0, 0, 0, 0)
    },
    options: {
      pairwiseTracks: [
        [0, 1],
        [1, 0]
      ],
      intervalsPerPair: 12
    }
  };
}

export function createWordScene(): RestrictionScene {
  const origin = (1n << 60n) - 1n;
  return {
    input: {
      positions: Float32Array.of(0, 0, 2, 0, 2, 0),
      times: getInt64TimeWords(BigInt64Array.of(origin, origin + 2n, origin + 7n)),
      timeOrigin: getInt64TimeWords(BigInt64Array.of(origin)),
      offsets: Uint32Array.of(0, 3),
      parameters: Float32Array.of(0, 7, 0, -1, 2, 1, 2)
    },
    options: {intervalsPerSegment: 1, compactCapacity: 4, mergeIntervals: true}
  };
}

export function createDecreasingOffsetPairScene(): RestrictionScene {
  return {
    input: {
      positions: Float32Array.of(0, 0),
      times: Float32Array.of(0),
      offsets: Uint32Array.of(1, 0, 1),
      parameters: Float32Array.of(0, 1, 1, 0, 0, 0, 0)
    },
    options: {pairwiseTracks: [[0, 1]], intervalsPerPair: 1}
  };
}

export function createZeroRowScene(trackCount: 0 | 1 = 1): RestrictionScene {
  return {
    input: {
      positions: new Float32Array(0),
      times: new Float32Array(0),
      offsets: new Uint32Array(trackCount + 1),
      parameters: Float32Array.of(0, 1, -1, -1, 1, 1, 0)
    },
    options: trackCount === 0 ? {compactCapacity: 0} : {}
  };
}

export function createDecreasingOffsetGeometryScene(): RestrictionScene {
  const scene = createDecreasingOffsetPairScene();
  return {
    input: {...scene.input, parameters: Float32Array.of(0, 1, -1, -1, 1, 1, 0)},
    options: {compactCapacity: 1}
  };
}

export function createReversedRangeOrderGeometryScene(mergeTouching = false): RestrictionScene {
  return {
    input: {
      positions: Float32Array.of(0, 0, 0, 0),
      times: Float32Array.of(0, mergeTouching ? 0 : 1),
      offsets: Uint32Array.of(1, 2, 0, 1),
      parameters: Float32Array.of(0, 1, -1, -1, 1, 1, 0)
    },
    options: {compactCapacity: 2, mergeIntervals: mergeTouching}
  };
}

export function getSceneProps(scene: RestrictionScene): TrajectoryRestrictionOperationProps {
  const {input, options} = scene;
  return {
    ...options,
    positions: 'sourcePositions',
    times: 'times',
    offsets: 'offsets',
    parameters: 'parameters',
    rowCount: input.positions.length / 2,
    trackCount: input.offsets.length - 1,
    timestampFormat: input.times instanceof Uint32Array ? 'uint32x2' : 'float32',
    timeOrigin: input.timeOrigin ? 'timeOrigin' : undefined,
    pairs: input.pairs
      ? {first: 'first', second: 'second', count: input.pairs.count ? 'pairCount' : undefined}
      : undefined,
    pairCapacity: input.pairs?.first.length,
    output: {
      intervals: 'intervals',
      positions: 'positions',
      status: 'status',
      relationships: 'relationships',
      ...(options.compactCapacity !== undefined
        ? {
            compactIntervals: 'compactIntervals',
            compactPositions: 'compactPositions',
            compactSources: 'compactSources',
            compactStatus: 'compactStatus',
            compactSummary: 'compactSummary'
          }
        : {})
    }
  };
}

export function getSceneInputs(scene: RestrictionScene): Record<string, OperationGraphInput> {
  const {input} = scene;
  return {
    sourcePositions: {format: 'float32x2', values: input.positions},
    times: {
      format: input.times instanceof Uint32Array ? 'uint32x2' : 'float32',
      values: input.times
    },
    offsets: {format: 'uint32', values: input.offsets},
    parameters: {format: 'float32', values: input.parameters},
    ...(input.timeOrigin
      ? {timeOrigin: {format: 'uint32' as const, values: input.timeOrigin}}
      : {}),
    ...(input.pairs
      ? {
          first: {format: 'uint32' as const, values: input.pairs.first},
          second: {format: 'uint32' as const, values: input.pairs.second},
          ...(input.pairs.count
            ? {pairCount: {format: 'uint32' as const, values: input.pairs.count}}
            : {})
        }
      : {})
  };
}
