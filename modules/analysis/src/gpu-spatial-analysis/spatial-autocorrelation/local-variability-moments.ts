// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {SPATIAL_STATISTICS_COUNTS_WGSL} from '../spatial-weights/spatial-statistics-counts-wgsl';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';

/** Bounded Welford blocks have exactly representable local counts, followed by uint32 Chan merges. */
export const LOCAL_VARIABILITY_MOMENT_BLOCK_ROWS = 4096;
/** Padding a row dispatch and addressing a row-major value must both fit uint32. */
export const LOCAL_VARIABILITY_MAXIMUM_ROWS = 0xffffffff - LOCAL_VARIABILITY_MOMENT_BLOCK_ROWS;
/** Original serial Welford reference counts are float32 and must stay strictly below 2^24. */
export const LOCAL_VARIABILITY_REFERENCE_MAXIMUM_ROWS = 2 ** 24 - 1;
/** A bounded variable count keeps the joint complete-case predicate and row work finite. */
export const LOCAL_VARIABILITY_MAXIMUM_VARIABLE_COUNT = 32;

/** Parallel bounded moments, fixed-order compensated Chan merge, exact population port. @internal */
export function getLocalVariabilityMomentNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rows: number;
    columns: number;
    bindings: WGSLKernelBinding[];
    declarations: string;
    variance: GraphDataView<'float32'>;
    count?: GraphDataView<'uint32'>;
  }
): readonly GPUCommandNode<Parameters>[] {
  const blocks = Math.ceil(props.rows / LOCAL_VARIABILITY_MOMENT_BLOCK_ROWS);
  // Counts within a 4096-row block are exactly represented by the first float word.
  const partials = createTransientView(
    graph,
    `${props.id}-partials`,
    'float32',
    blocks * props.columns * 3
  );
  return [
    createWGSLKernelNode(graph, {
      id: `${props.id}-blocks`,
      operation: 'GPUMultivariateLocalGeary',
      variant: 'moment-blocks',
      invocationCount: blocks * props.columns,
      bindings: [
        ...props.bindings,
        {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
      ],
      declarations: props.declarations,
      body: `let column = index / ${blocks}u; let blockIndex = index % ${blocks}u;
var count = 0u; var mean = 0.0; var squares = 0.0;
let begin = blockIndex * ${LOCAL_VARIABILITY_MOMENT_BLOCK_ROWS}u;
for (var row = begin; row < min(begin + ${LOCAL_VARIABILITY_MOMENT_BLOCK_ROWS}u, ${props.rows}u); row++) {
 if (!included(row)) { continue; }
 let value = values[valuesOffset + row * ${props.columns}u + column]; count++;
 let delta = value - mean; mean += delta / f32(count); squares += delta * (value - mean);
}
let destination = partialsOffset + index * 3u; partials[destination] = f32(count); partials[destination + 1u] = mean; partials[destination + 2u] = squares;`
    }),
    createWGSLKernelNode(graph, {
      id: `${props.id}-merge`,
      operation: 'GPUMultivariateLocalGeary',
      variant: 'moment-merge',
      invocationCount: props.columns,
      bindings: [
        {name: 'partials', view: partials, type: 'f32', access: 'read'},
        {name: 'variance', view: props.variance, type: 'f32', access: 'read_write'},
        ...(props.count
          ? [
              {
                name: 'populationCount',
                view: props.count,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      declarations: `${props.declarations.split('fn included')[0]}\n${SPATIAL_STATISTICS_COUNTS_WGSL}\n${ADDITION_ROUNDING_ERROR_WGSL}`,
      body: `var count = 0u; var mean = 0.0; var squares = 0.0; var correction = 0.0;
for (var blockIndex = 0u; blockIndex < ${blocks}u; blockIndex++) {
 let source = partialsOffset + (index * ${blocks}u + blockIndex) * 3u;
 let blockCount = u32(partials[source]); if (blockCount == 0u) { continue; }
 // The empty aggregate has no mean: avoid squaring a large first block mean.
 if (count == 0u) { count = blockCount; mean = partials[source + 1u]; squares = partials[source + 2u]; continue; }
 let combined = count + blockCount; let delta = partials[source + 1u] - mean;
 let ratio = divideSpatialCounts(blockCount, combined);
 let term = partials[source + 2u] + multiplyBySpatialCount(delta * delta * ratio, count);
 let next = squares + term; correction += getAdditionRoundingError(squares, term, next); squares = next;
 mean += delta * ratio; count = combined;
}
variance[varianceOffset + index] = select(getNaN(), divideBySpatialCount(squares + correction, count), count > 0u);
${props.count ? 'if (index == 0u) { populationCount[populationCountOffset] = count; }' : ''}`
    })
  ];
}
