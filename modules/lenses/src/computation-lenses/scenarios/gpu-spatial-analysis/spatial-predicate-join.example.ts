// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUHistogram, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUSpatialPredicateJoin,
  type GPUSpatialPredicateJoinProps,
  type GPUPointInPolygonJoinProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  createWGSLKernelNode,
  type GPUSpatialContext
} from '@luma-contributors/analysis/analysis-core';

/** Uploaded inputs and caller-owned outputs for this recorded example. */
export type SpatialPredicateJoinExampleInput = {
  parks: GPUPointInPolygonJoinProps['polygons'];
  tracts: GPUPointInPolygonJoinProps['polygons'];
  pairs: NonNullable<GPUSpatialPredicateJoinProps['output']['pairs']>;
  spatialContext: GPUSpatialContext;
};

/** Adds the public contributor and returns the views captured by the figure. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: SpatialPredicateJoinExampleInput
) {
  const tractMatches = createTransientView(
    graph,
    'tractMatches',
    'uint32',
    input.tracts.featureOffsets.length - 1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const addPairHistogramMask = (
    graph: GPUCommandGraph<undefined>,
    pairs: SpatialPredicateJoinExampleInput['pairs'],
    id: string
  ) => {
    // Keep setup local so this example needs only published package imports.
    const pairMask = createTransientView(
      graph,
      id,
      'uint32',
      pairs.rightRows.length,
      Buffer.STORAGE
    );
    graph.add(
      createWGSLKernelNode(graph, {
        id,
        operation: 'lens-pair-histogram-membership',
        invocationCount: pairMask.length,
        bindings: [
          {name: 'pairCount', view: pairs.status.count, type: 'u32', access: 'read'},
          {name: 'mask', view: pairMask, type: 'u32', access: 'read_write'}
        ],
        body: 'mask[maskOffset + index] = select(0u, 1u, index < pairCount[pairCountOffset]);'
      })
    );
    return pairMask;
  };
  // #region example
  graph.add(
    new GPUSpatialPredicateJoin({
      left: input.parks,
      right: input.tracts,
      spatialContext: input.spatialContext,
      predicate: 'intersects',
      engine: 'relate',
      candidateCapacity: input.pairs.leftRows.length,
      output: {pairs: input.pairs}
    })
  );
  graph.add(
    new GPUHistogram({
      input: input.pairs.rightRows,
      mask: addPairHistogramMask(graph, input.pairs, 'predicate-join-active-pairs'),
      output: tractMatches,
      domain: [0, tractMatches.length - 1]
    })
  );
  // tractMatches colours tracts; pairs identify parks and tracts.
  // #endregion example
  return {tractMatches};
}
