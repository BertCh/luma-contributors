// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_BITS_WGSL, GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {GPUTransitionMatrix, type GPUTransitionMatrixOutput} from './gpu-transition-matrix';

export type GPURankMarkovProps = {
  id?: string;
  /** Period-major balanced panel. Any nonfinite observation excludes its row from ALL periods. */
  values: GraphDataView<'float32'>;
  periodCount: number;
  mask?: GraphDataView<'uint32'>;
  mode?: 'full-rank' | 'geographic-rank';
  /** giddy reverses ascending ordinal ranks: later row IDs win descending ties. */
  tieOrder?: 'giddy' | 'row-id';
  output: GPUTransitionMatrixOutput & {
    /** Zero-based descending ranks; ties follow tieOrder (giddy by default). Excluded rows: 0xffffffff. */
    ranks: GraphDataView<'uint32'>;
    /** Mean absolute rank change per adjacent period, NaN when fewer than two included rows. */
    mobility?: GraphDataView<'float32'>;
  };
};

/**
 * giddy 2.3.9 full-rank and geographic-rank transition conventions. Full rank follows each
 * location through rank states. Geographic rank follows the occupant of each fixed rank through
 * location states. No discretization, rank averaging or synthetic self-loops. Common complete
 * cases keep the population comparable over time. The n-by-n matrix retains original capacity
 * even when rows are excluded; zero transition rows remain zero. Existing transition diagnostics
 * explicitly report nonergodicity instead of inventing a unique stationary distribution.
 */
export class GPURankMarkov implements GPUCommandNodeProducer {
  readonly id: string;
  readonly rowCount: number;
  constructor(readonly props: GPURankMarkovProps) {
    this.id = props.id ?? 'rank-markov';
    validateCapacity(this.id, 'periodCount', props.periodCount, 2, 65536);
    this.rowCount = props.values.length / props.periodCount;
    validateCapacity(this.id, 'rowCount', this.rowCount, 1, 64);
    if (props.tieOrder && !['giddy', 'row-id'].includes(props.tieOrder))
      throw new Error(`${this.id}: invalid tie order`);
    if (props.mode && props.mode !== 'full-rank' && props.mode !== 'geographic-rank')
      throw new Error(`${this.id}: invalid rank mode`);
    for (const [view, format, length] of [
      [props.values, 'float32', props.values.length],
      [props.mask, 'uint32', this.rowCount],
      [props.output.ranks, 'uint32', props.values.length],
      [props.output.mobility, 'float32', props.periodCount - 1]
    ] as const)
      if (view) {
        validatePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
      }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [
        ...(Object.values(props.output).filter(
          value => value && 'buffer' in value
        ) as GraphDataView[])
      ],
      [props.values, props.mask]
    );
    // Validate optional diagnostics at declaration, before constructing transient rank occupants.
    new GPUTransitionMatrix({
      classes: props.output.ranks,
      periodCount: props.periodCount,
      classCount: this.rowCount,
      output: props.output
    });
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount: rows} = this;
    const ranks = props.output.ranks;
    const valid = createTransientView(graph, `${id}-complete-cases`, 'uint32', rows);
    const classes =
      props.mode === 'geographic-rank'
        ? createTransientView(graph, `${id}-occupants`, 'uint32', ranks.length)
        : ranks;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.mask,
      ranks,
      props.output.mobility
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-complete`,
        operation: 'GPURankMarkov',
        variant: 'complete-cases',
        invocationCount: rows,
        bindings: [
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          ...(props.mask
            ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'valid', view: valid, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_BITS_WGSL,
        body: `var included = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
for (var period = 0u; period < ${props.periodCount}u; period++) { included = included && isFiniteBits(values[valuesOffset + period * ${rows}u + index]); }
valid[validOffset + index] = select(0u, 1u, included);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-rank`,
        operation: 'GPURankMarkov',
        variant: 'rank',
        invocationCount: ranks.length,
        bindings: [
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'valid', view: valid, type: 'u32', access: 'read'},
          {name: 'ranks', view: ranks, type: 'u32', access: 'read_write'}
        ],
        body: `let row = index % ${rows}u; let base = index - row; var rank = 0xffffffffu;
if (valid[validOffset + row] != 0u) { rank = 0u; let value = values[valuesOffset + index];
  for (var other = 0u; other < ${rows}u; other++) { let candidate = values[valuesOffset + base + other];
    if (valid[validOffset + other] != 0u && (candidate > value || (candidate == value && other ${props.tieOrder === 'row-id' ? '<' : '>'} row))) { rank++; }
  }
}
ranks[ranksOffset + index] = rank;`
      })
    ];
    if (props.mode === 'geographic-rank')
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-occupants`,
          operation: 'GPURankMarkov',
          variant: 'occupants',
          invocationCount: ranks.length,
          bindings: [
            {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
            {name: 'classes', view: classes, type: 'u32', access: 'read_write'}
          ],
          body: `let rank = index % ${rows}u; let base = index - rank; var occupant = 0xffffffffu;
for (var row = 0u; row < ${rows}u; row++) { if (ranks[ranksOffset + base + row] == rank) { occupant = row; break; } }
classes[classesOffset + index] = occupant;`
        })
      );
    nodes.push(
      ...new GPUTransitionMatrix({
        id: `${id}-transitions`,
        classes,
        periodCount: props.periodCount,
        classCount: rows,
        output: props.output
      }).getCommandNodes(graph)
    );
    if (props.output.mobility)
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-mobility`,
          operation: 'GPURankMarkov',
          variant: 'mobility',
          invocationCount: props.periodCount - 1,
          bindings: [
            {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
            {name: 'mobility', view: props.output.mobility, type: 'f32', access: 'read_write'}
          ],
          declarations: GET_NAN_WGSL,
          body: `var count = 0u; var displacement = 0.0;
for (var row = 0u; row < ${rows}u; row++) { let first = ranks[ranksOffset + index * ${rows}u + row]; let second = ranks[ranksOffset + (index + 1u) * ${rows}u + row];
if (first < ${rows}u && second < ${rows}u) { count++; displacement += abs(f32(first) - f32(second)); } }
mobility[mobilityOffset + index] = select(getNaN(), displacement / f32(count), count > 1u);`
        })
      );
    return nodes;
  }
}
