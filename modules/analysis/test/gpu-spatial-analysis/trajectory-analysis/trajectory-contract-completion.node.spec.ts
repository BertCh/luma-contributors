// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTrackSimilaritySearch,
  type GPUTrackSimilaritySearchProps
} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-search';
import {GPUTrajectoryThinning} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/index';
import {
  GPUTrajectoryClosestApproach,
  GPUTrajectoryEncounters
} from '../../../src/gpu-spatial-analysis/trajectory-encounters/index';
import {GPUTrajectoryKalman} from '../../../src/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman';
import {prepareGPUTrajectoryObservations} from '../../../src/gpu-spatial-analysis/trajectory-estimation/prepare-trajectory-observations';
import {GPUTrajectoryRestriction} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;
type Format = 'float32' | 'uint32' | 'float32x2' | 'float32x4' | 'uint32x4';

function createFixture() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <F extends Format>(format: F, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {graph, view};
}

const LONGITUDE_LATITUDE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
const LAMBERT_FRAME = {
  coordinateSpace: 'planar',
  metric: 'native',
  frame: {id: 'lambert', kind: 'local-lambert-equal-area'}
} as const;

function createSearch(
  overrides: Partial<Omit<GPUTrackSimilaritySearchProps, 'output'>> = {},
  outputOverrides: Partial<GPUTrackSimilaritySearchProps['output']> = {}
) {
  const {view} = createFixture();
  const metric = overrides.metric ?? 'dynamic-time-warping';
  return new GPUTrackSimilaritySearch({
    metric,
    linesA: {kind: 'lines', positions: view('float32x2', 8), lineOffsets: view('uint32', 3)},
    pairA: view('uint32', 4),
    pairB: view('uint32', 4),
    threshold: view('float32', 1),
    output: {
      ...(metric === 'longest-common-subsequence'
        ? {similarity: view('float32', 4), upperBound: view('float32', 4)}
        : {distance: view('float32', 4), lowerBound: view('float32', 4)}),
      matched: view('uint32', 4),
      status: view('uint32', 4),
      ...outputOverrides
    },
    ...overrides
  });
}

// Compile-only controls for the lead's test typecheck. Never invoke invalid constructors here.
function checkSearchFixtureTypes() {
  // @ts-expect-error Unknown metrics must be rejected by the fixture type.
  createSearch({metric: 'misspelled-metric'});
  // @ts-expect-error Misspelled output ports must be rejected by the fixture type.
  createSearch({}, {similarty: undefined});
}
void checkSearchFixtureTypes;

it('GPUTrackSimilaritySearch refuses invalid props with its ID, the prop and the limit', () => {
  expect(() => createSearch()).not.toThrow();
  expect(() => createSearch({}, {lowerBound: undefined})).toThrow(
    /^track-similarity-search output\.distance and output\.lowerBound are required/
  );
  expect(() => createSearch({metric: 'longest-common-subsequence'})).not.toThrow();
  for (const name of ['similarity', 'upperBound'] as const) {
    expect(() => createSearch({metric: 'longest-common-subsequence'}, {[name]: undefined})).toThrow(
      /^track-similarity-search output\.similarity and output\.upperBound are required/
    );
  }
  const {view} = createFixture();
  for (const name of ['distance', 'lowerBound'] as const) {
    expect(() =>
      createSearch({metric: 'longest-common-subsequence'}, {[name]: view('float32', 4)})
    ).toThrow(
      new RegExp(
        `^track-similarity-search output\\.${name} is inactive for metric longest-common-subsequence; omit output\\.${name}`
      )
    );
  }
  expect(() => createSearch({lcssTolerance: -1})).toThrow(
    /^track-similarity-search lcssTolerance must be a finite number in \[0, Infinity\]/
  );
  expect(() => createSearch({survivorCapacity: 5})).toThrow(
    /^track-similarity-search survivorCapacity must be an integer in \[0, 4\]/
  );
  expect(() => createSearch({topK: 1025})).toThrow(
    /^track-similarity-search topK must be an integer in \[1, 1024\]/
  );
  expect(() => createSearch({topK: 2})).toThrow(
    /^track-similarity-search topK needs output\.topKIndices and output\.topKSummary/
  );
  expect(() => createSearch({threshold: view('float32', 2)})).toThrow(
    /^track-similarity-search threshold must have 1 rows/
  );
  expect(() => createSearch({}, {topKIndices: view('uint32', 2)})).toThrow(
    /^track-similarity-search output\.topKIndices and output\.topKSummary need topK/
  );
  expect(() => createSearch({}, {survivorSummary: view('uint32', 2)})).toThrow(
    /^track-similarity-search output\.survivorSummary must have 3 rows/
  );
  const shared = view('uint32', 4);
  expect(() => createSearch({}, {matched: shared, status: shared})).toThrow(
    /^track-similarity-search outputs must not overlap each other/
  );
});

it('prepareGPUTrajectoryObservations refuses invalid input with its name and the limit', () => {
  const props = {
    longitudeLatitude: [10, 50, 10.1, 50.1],
    timestamps: [0n, 1000n],
    epoch: 0n,
    unitsPerSecond: 1000n,
    origin: [10, 50] as const
  };
  expect(prepareGPUTrajectoryObservations(props).positions).toHaveLength(4);
  expect(() => prepareGPUTrajectoryObservations({...props, timestamps: [0n]})).toThrow(
    /^prepareGPUTrajectoryObservations longitudeLatitude must hold two values per timestamp/
  );
  expect(() => prepareGPUTrajectoryObservations({...props, unitsPerSecond: 0n})).toThrow(
    /^prepareGPUTrajectoryObservations unitsPerSecond must be a positive finite integer/
  );
  expect(() => prepareGPUTrajectoryObservations({...props, origin: [200, 0]})).toThrow(
    /^prepareGPUTrajectoryObservations origin longitude must be a finite number in \[-180, 180\]/
  );
  expect(() => prepareGPUTrajectoryObservations({...props, origin: [0, 91]})).toThrow(
    /^prepareGPUTrajectoryObservations origin latitude must be a finite number in \[-90, 90\]/
  );
  expect(() =>
    prepareGPUTrajectoryObservations({...props, longitudeLatitude: [10, 50, 181, 50]})
  ).toThrow(/^prepareGPUTrajectoryObservations longitudeLatitude longitude at row 1 must be/);
  // Nonfinite observations stay missing measurements.
  expect(
    prepareGPUTrajectoryObservations({...props, longitudeLatitude: [10, 50, NaN, NaN]}).positions[2]
  ).toBeNaN();
});

it('prepareGPUTrajectoryObservations compares the origin in double precision', () => {
  const props = {
    longitudeLatitude: [10, 50],
    timestamps: [0n],
    epoch: 0n,
    unitsPerSecond: 1000n,
    origin: [10, 50] as const
  };
  // Both values round to the bound in float32, but lie outside it as doubles.
  expect(Math.fround(180.000001)).toBe(180);
  expect(Math.fround(90.000003)).toBe(90);
  expect(() => prepareGPUTrajectoryObservations({...props, origin: [180.000001, 0]})).toThrow(
    /^prepareGPUTrajectoryObservations origin longitude must be a finite number in \[-180, 180\]/
  );
  expect(() => prepareGPUTrajectoryObservations({...props, origin: [0, 90.000003]})).toThrow(
    /^prepareGPUTrajectoryObservations origin latitude must be a finite number in \[-90, 90\]/
  );
  expect(() => prepareGPUTrajectoryObservations({...props, origin: [180, 90]})).not.toThrow();
});

function createKalman(spatialContext?: unknown) {
  const {view} = createFixture();
  return new GPUTrajectoryKalman({
    positions: view('float32x2', 8),
    times: view('float32', 8),
    offsets: view('uint32', 2),
    parameters: view('float32', 4),
    spatialContext,
    output: {
      filteredState: view('float32x4', 8),
      filteredCovariance: view('float32x4', 8),
      status: view('uint32', 8)
    }
  } as never);
}

it('GPUTrajectoryKalman takes an optional planar spatialContext and refuses the others', () => {
  expect(createKalman().spatialContext).toMatchObject({coordinateSpace: 'planar'});
  expect(
    createKalman({coordinateSpace: 'planar', metric: 'native', units: 'meters'})
  ).toBeDefined();
  expect(() => createKalman(LONGITUDE_LATITUDE)).toThrow(
    /^trajectory-kalman does not support longitude-latitude coordinates/
  );
  expect(() => createKalman(LAMBERT_FRAME)).toThrow(/^trajectory-kalman measures distance/);
});

function createRestriction(overrides: Record<string, unknown> = {}) {
  const {view} = createFixture();
  return new GPUTrajectoryRestriction({
    positions: view('float32x2', 8),
    times: view('float32', 8),
    offsets: view('uint32', 2),
    parameters: view('float32', 7),
    output: {
      intervals: view('float32x2', 8),
      positions: view('float32x4', 8),
      status: view('uint32', 8),
      relationships: view('uint32x4', 1)
    },
    ...overrides
  } as never);
}

it('GPUTrajectoryRestriction takes an optional planar spatialContext and refuses the others', () => {
  expect(createRestriction().spatialContext).toMatchObject({coordinateSpace: 'planar'});
  expect(() => createRestriction({spatialContext: LONGITUDE_LATITUDE})).toThrow(
    /^trajectory-restriction does not support longitude-latitude coordinates/
  );
  // Without a distance predicate no distance is measured, so any planar frame is fine.
  expect(() => createRestriction({spatialContext: LAMBERT_FRAME})).not.toThrow();
  expect(() =>
    createRestriction({spatialContext: LAMBERT_FRAME, pairwiseTracks: [[0, 1]]})
  ).toThrow(/^trajectory-restriction measures distance/);
});

it('trajectory encounter classes take `mask` and no longer accept `trackValid`', () => {
  const {graph, view} = createFixture();
  const encounterProps = (overrides: Record<string, unknown>) =>
    ({
      samples: view('float32x2', 8),
      trackCount: 2,
      bucketCount: 4,
      distance: view('float32', 1),
      cellSize: 10,
      bounds: [0, 0, 100, 100],
      hitCapacity: 16,
      output: {
        pairs: {
          ids: view('uint32', 4),
          status: {count: view('uint32', 1), overflow: view('uint32', 1)}
        },
        status: {candidateOverflow: view('uint32', 1)}
      },
      ...overrides
    }) as never;
  const maskView = view('uint32', 2);
  expect(new GPUTrajectoryEncounters(encounterProps({mask: maskView})).props.mask).toBe(maskView);
  expect('trackValid' in new GPUTrajectoryEncounters(encounterProps({mask: maskView})).props).toBe(
    false
  );
  expect(() => new GPUTrajectoryEncounters(encounterProps({mask: view('uint32', 3)}))).toThrow(
    /^trajectory-encounters mask length must equal trackCount/
  );

  const closestProps = (overrides: Record<string, unknown>, status = {}) =>
    ({
      positions: view('float32x2', 16),
      timestamps: view('float32', 16),
      trackOffsets: view('uint32', 3),
      distance: view('float32', 1),
      cellSize: 10,
      timeCellSize: 60,
      entryCapacity: 64,
      hitCapacity: 64,
      output: {
        pairs: {
          ids: view('uint32', 4),
          status: {count: view('uint32', 1), overflow: view('uint32', 1)}
        },
        status: {entryOverflow: view('uint32', 1), hitOverflow: view('uint32', 1), ...status}
      },
      ...overrides
    }) as never;
  expect(new GPUTrajectoryClosestApproach(closestProps({mask: maskView})).props.mask).toBe(
    maskView
  );
  expect(() => new GPUTrajectoryClosestApproach(closestProps({mask: view('uint32', 3)}))).toThrow(
    /^trajectory-closest-approach mask length must equal the track count/
  );
  // The optional `candidateOverflow` joins the two overflow words as one more kernel binding.
  const withStatus = new GPUTrajectoryClosestApproach(
    closestProps({}, {candidateOverflow: view('uint32', 1)})
  );
  const statusNode = withStatus.getCommandNodes(graph).find(node => node.id.endsWith('-status'));
  expect(statusNode?.id).toBe('trajectory-closest-approach-status');
});

it('GPUTrajectoryThinning takes `output.sourceSelectionMask` and names it in validation messages', () => {
  const {view} = createFixture();
  const props = (output: Record<string, unknown>) =>
    ({
      positions: view('float32x2', 8),
      timestamps: view('float32', 8),
      trackOffsets: view('uint32', 2),
      parameters: view('float32', 4),
      mode: 'minimum-distance',
      output: {
        sourceRows: view('uint32', 8),
        trackOffsets: view('uint32', 2),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1),
          candidateOverflow: view('uint32', 1)
        },
        ...output
      }
    }) as never;
  const sourceSelectionMask = view('uint32', 8);
  expect(
    new GPUTrajectoryThinning(props({sourceSelectionMask})).props.output.sourceSelectionMask
  ).toBe(sourceSelectionMask);
  expect(() => new GPUTrajectoryThinning(props({sourceSelectionMask: view('uint32', 7)}))).toThrow(
    /^trajectory-thinning output\.sourceSelectionMask must have 8 rows/
  );
});
