// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUGeographicallyWeightedRegressionNonstationarityTest} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression-nonstationarity-test';
import {getGPUGeographicallyWeightedRegressionParameterLength} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';

/** Isolates actual GWR batch reset/group/merge/advance nodes with hand-derived partial moments.
 * The unchanged local refit mathematics has separate integration/oracle tests. This fixture
 * exercises the changed batch cursor and both destination rows without replaying a permutation.
 */
export function createGWRBatchMergeFixture<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  parameters: GraphDataView<'uint32'>,
  standardDeviations: GraphDataView<'float32'>
) {
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2'>(format: Format, length: number) =>
    createTransientView(graph, `batch-fixture-${serial++}`, format, length);
  const contributor = new GPUGeographicallyWeightedRegressionNonstationarityTest({
    id: 'batch-review',
    positions: view('float32x2', 4097),
    predictors: view('float32', 4097),
    response: view('float32', 4097),
    predictorCount: 1,
    coefficients: view('float32', 8194),
    bandwidthParameters: view('float32', getGPUGeographicallyWeightedRegressionParameterLength()),
    selectedBandwidth: view('float32', 2),
    parameters,
    tileRowCount: 1,
    maximumPermutations: 33,
    output: {table: view('float32', 12), summary: view('float32', 4), standardDeviations}
  });
  const allNodes = contributor.getCommandNodes(graph);
  const select = (suffix: string) => {
    const node = allNodes.find(candidate => candidate.id === `batch-review-${suffix}`);
    if (!node) throw new Error(`missing batch fixture node ${suffix}`);
    return node;
  };
  const firstGroup = select('batch-0-merge-partial-groups');
  const partial = firstGroup.resources!.filter(resource => 'buffer' in resource)[0]
    .buffer as GraphDataView;
  const cursorSlot = contributor.batchRunCount * 4097 * 5;
  const nodes = [select('batch-reset')];
  for (let batch = 0; batch < 2; batch++) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `batch-review-known-moments-${batch}`,
        operation: 'GWRBatchMergeFixture',
        invocationCount: contributor.batchRunCount * 4097,
        bindings: [{name: 'partial', view: partial, type: 'f32', access: 'read_write'}],
        body: `let run = u32(partial[partialOffset + ${cursorSlot}u]) + index / 4097u;
  let tile = index % 4097u;
  var amplitude = 1.0;
  if (run == 32u) { amplitude = 3.0; }
  if (run == 33u) { amplitude = 5.0; }
  let count = select(1.0, 0.0, tile == 4096u);
  let deviation = select(-amplitude, amplitude, (tile & 1u) == 1u) * count;
  let base = partialOffset + index * 5u;
  partial[base] = count;
  partial[base + 1u] = deviation; partial[base + 2u] = 2.0 * deviation;
  partial[base + 3u] = amplitude * amplitude * count;
  partial[base + 4u] = 4.0 * amplitude * amplitude * count;`
      }),
      select(`batch-${batch}-merge-partial-groups`),
      select(`batch-${batch}-merge`),
      select(`batch-${batch}-advance`)
    );
  }
  const counts = select('batch-0-merge')
    .resources!.filter(resource => 'buffer' in resource)
    .filter(resource => resource.usage === 'storage-read-write')[0].buffer as GraphDataView;
  return {nodes, counts, partial, cursorSlot};
}
