// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUHistogram,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPUPairGather,
  type GPUSpatialPredicateJoinProps,
  type GPUPointInPolygonJoinProps,
  GPUSpatialPredicateJoin
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  createWGSLKernelNode,
  validateRequirement,
  type GPUSpatialContext
} from '@luma-contributors/analysis/analysis-core';

/** Uploaded inputs and caller-owned outputs for this recorded example. */
export type PairGatherExampleInput = {
  parks: GPUPointInPolygonJoinProps['polygons'];
  tracts: GPUPointInPolygonJoinProps['polygons'];
  pairs: NonNullable<GPUSpatialPredicateJoinProps['output']['pairs']>;
  spatialContext: GPUSpatialContext;
  parkAreas: GraphDataView<'float32'>;
  /** Optional small candidate budget for completeness demonstrations. */
  candidateCapacity?: number;
};

/** Captures polygon view identities once; the contributor validates the geometry. */
function capturePolygonPort(polygon: PairGatherExampleInput['parks']) {
  return Object.freeze({
    kind: polygon.kind,
    positions: polygon.positions,
    featureOffsets: polygon.featureOffsets,
    polygonOffsets: polygon.polygonOffsets,
    ringOffsets: polygon.ringOffsets,
    sourceIds: polygon.sourceIds
  });
}

/** Adds the public contributor and returns the views captured by the figure. */
export async function runExample(graph: GPUCommandGraph<undefined>, input: PairGatherExampleInput) {
  const {parks, tracts, pairs: suppliedPairs, spatialContext, parkAreas, candidateCapacity} = input;
  validateRequirement(
    'pair-gather-example',
    !('leftIds' in suppliedPairs) && !('rightIds' in suppliedPairs),
    'pairs.leftRows and pairs.rightRows are required; leftIds and rightIds are unsupported'
  );
  const {leftRows, rightRows, status: suppliedStatus} = suppliedPairs;
  const {
    count,
    overflow: pairOverflow,
    requiredCount,
    candidateOverflow: pairCandidateOverflow
  } = suppliedStatus;
  const capturedParks = capturePolygonPort(parks);
  const capturedTracts = capturePolygonPort(tracts);
  const pairs = Object.freeze({
    leftRows,
    rightRows,
    status: Object.freeze({
      count,
      requiredCount,
      overflow: pairOverflow,
      candidateOverflow:
        pairCandidateOverflow ??
        createTransientView(
          graph,
          'pairCandidateOverflow',
          'uint32',
          1,
          Buffer.STORAGE | Buffer.COPY_SRC
        )
    })
  });
  input = Object.freeze({
    parks: capturedParks,
    tracts: capturedTracts,
    pairs,
    spatialContext,
    parkAreas,
    candidateCapacity
  });
  const tractMatches = createTransientView(
    graph,
    'tractMatches',
    'uint32',
    input.tracts.featureOffsets.length - 1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const parkRows = createTransientView(
    graph,
    'parkRows',
    'uint32',
    pairs.leftRows.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const tractRows = createTransientView(
    graph,
    'tractRows',
    'uint32',
    pairs.leftRows.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const joinedAreas = createTransientView(
    graph,
    'joinedAreas',
    'float32',
    pairs.leftRows.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const rowCount = createTransientView(
    graph,
    'rowCount',
    'uint32',
    1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const overflow = createTransientView(
    graph,
    'overflow',
    'uint32',
    1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const candidateOverflow = createTransientView(
    graph,
    'candidateOverflow',
    'uint32',
    1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const addTractHistogram = () => {
    // Keep setup local so this example needs only published package imports.
    const pairMask = createTransientView(
      graph,
      'pair-gather-active-pairs',
      'uint32',
      pairs.rightRows.length,
      Buffer.STORAGE
    );
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'pair-gather-active-pairs',
        operation: 'lens-pair-histogram-membership',
        invocationCount: pairMask.length,
        bindings: [
          {name: 'pairCount', view: pairs.status.count, type: 'u32', access: 'read'},
          {name: 'mask', view: pairMask, type: 'u32', access: 'read_write'}
        ],
        body: 'mask[maskOffset + index] = select(0u, 1u, index < pairCount[pairCountOffset]);'
      })
    );
    graph.add(
      new GPUHistogram({
        input: pairs.rightRows,
        mask: pairMask,
        output: tractMatches,
        domain: [0, tractMatches.length - 1]
      })
    );
  };
  // #region example
  graph.add(
    new GPUSpatialPredicateJoin({
      left: input.parks,
      right: input.tracts,
      spatialContext: input.spatialContext,
      predicate: 'intersects',
      engine: 'relate',
      candidateCapacity: input.candidateCapacity ?? pairs.leftRows.length,
      output: {pairs}
    })
  );
  addTractHistogram();
  graph.add(
    new GPUPairGather({
      pairs,
      leftColumns: [{source: input.parkAreas, output: joinedAreas}],
      output: {
        leftRows: parkRows,
        rightRows: tractRows,
        rowCount,
        status: {overflow, candidateOverflow}
      }
    })
  );
  // tractMatches colours tracts; pairs identify parks and tracts; joinedAreas carries source park area per pair.
  // #endregion example
  return {tractMatches, parkRows, tractRows, joinedAreas, rowCount, overflow, candidateOverflow};
}
