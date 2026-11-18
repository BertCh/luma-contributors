// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUHistogram, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUSpatialJoinPrepared,
  type GPUSpatialPredicateJoinProps,
  type GPUPointInPolygonJoinProps,
  GPUSpatialPredicateJoin,
  GPUSpatialJoinCandidates
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  createWGSLKernelNode,
  type GPUSpatialContext
} from '@luma-contributors/analysis/analysis-core';

/** Uploaded inputs and caller-owned outputs for this recorded example. */
export type SpatialJoinPreparedExampleInput = {
  parks: GPUPointInPolygonJoinProps['polygons'];
  tracts: GPUPointInPolygonJoinProps['polygons'];
  pairs: NonNullable<GPUSpatialPredicateJoinProps['output']['pairs']>;
  spatialContext: GPUSpatialContext;
  candidatePairs: NonNullable<GPUSpatialPredicateJoinProps['output']['pairs']>;
};

/** Adds the public contributor and returns the views captured by the figure. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: SpatialJoinPreparedExampleInput
) {
  const tractMatches = createTransientView(
    graph,
    'tractMatches',
    'uint32',
    input.tracts.featureOffsets.length - 1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const addTractHistogram = () => {
    // Keep setup local so this example needs only published package imports.
    const pairMask = createTransientView(
      graph,
      'prepared-join-active-pairs',
      'uint32',
      input.pairs.rightRows.length,
      Buffer.STORAGE
    );
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'prepared-join-active-pairs',
        operation: 'lens-pair-histogram-membership',
        invocationCount: pairMask.length,
        bindings: [
          {name: 'pairCount', view: input.pairs.status.count, type: 'u32', access: 'read'},
          {name: 'mask', view: pairMask, type: 'u32', access: 'read_write'}
        ],
        body: 'mask[maskOffset + index] = select(0u, 1u, index < pairCount[pairCountOffset]);'
      })
    );
    graph.add(
      new GPUHistogram({
        input: input.pairs.rightRows,
        mask: pairMask,
        output: tractMatches,
        domain: [0, tractMatches.length - 1]
      })
    );
  };
  // #region example
  const prepared = new GPUSpatialJoinPrepared({geometry: input.tracts, spatialSort: false});
  graph.add(prepared);
  graph.add(
    new GPUSpatialJoinCandidates({
      left: input.parks,
      right: input.tracts,
      prepared,
      spatialContext: input.spatialContext,
      output: input.candidatePairs
    })
  );
  graph.add(
    new GPUSpatialPredicateJoin({
      left: input.parks,
      right: input.tracts,
      spatialContext: input.spatialContext,
      predicate: 'intersects',
      engine: 'relate',
      candidateCapacity: input.pairs.leftRows.length,
      prepared,
      output: {pairs: input.pairs}
    })
  );
  addTractHistogram();
  // tractMatches colours tracts; pairs identify parks and tracts.
  // #endregion example
  return {tractMatches, prepared};
}
