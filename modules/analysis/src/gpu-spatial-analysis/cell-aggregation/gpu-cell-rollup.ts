// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {createCellLevelKernelNode} from './cell-level-kernel';
import {
  validateCapacity,
  validateRequirement,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  CELL_KEY_WGSL,
  getCellKeyLayout,
  resolveCellSpatialContext,
  type CellKeyLayout,
  type GPUCellAggregationFamily
} from './cell-keys';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  getCellTableNodes,
  getCellTableViews,
  hasCellKeyHighWord,
  snapshotCellTable,
  validateCellTable,
  type GPUCellTable
} from './cell-table';
import {
  validateCellPublishBindingCount,
  validateCellViewBindingRanges,
  validateCellTableScratchBindingSizes,
  validateCellTableWritableViews
} from './cell-validation';
import {
  GPU_CELL_DEFAULT_SUM_SCALE,
  validateSumScale
} from '../../analysis-core/kernels/cell-key-wgsl';

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';

const OPERATION = 'GPUCellRollup';

/**
 * Properties for {@link GPUCellRollup}.
 *
 * Per-frame (no recompile): the contents of the source table. Topology: family, resolutions,
 * `sumScale`, which columns are present, and capacities.
 */
export type GPUCellRollupProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'cell-rollup'`. */
  id?: string;
  /** Grid family of both tables. */
  family: GPUCellAggregationFamily;
  /**
   * Optional `discrete-grid` context of the keys. Defaults to
   * `{coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: family}`; a given
   * `gridFamily` must equal `family` and a given `gridResolution` must be at least `resolution`.
   */
  spatialContext?: GPUSpatialContext;
  /** Resolution of every key in `source`. */
  sourceResolution: number;
  /** Coarser output resolution, at most `sourceResolution`. */
  resolution: number;
  /**
   * A table written by {@link GPUCellAggregation} or another {@link GPUCellRollup} at
   * `sourceResolution`: ascending keys in rows `[0, count)`. Output sums need `source.sums`, and
   * output extremes need the matching source extremes.
   */
  source: GPUCellTable;
  /** Fixed-point scale used by the source sums. Defaults to 65536. */
  sumScale?: number;
  /** Caller-owned table at `resolution`; `output.status.candidateOverflow` is required. */
  output: GPUCellTable;
};

/**
 * Rolls a sorted cell table up to a coarser resolution without re-reading the source rows.
 *
 * The parent key truncates the cell path (`cellToParent` for Quadbin and H3), which is monotone
 * in the child key, so the source order is already the parent order and no sort is needed: one
 * key pass, a head scan, and per-parent reductions of the child cells. Counts and fixed-point sums
 * are integer sums, so the result equals a direct aggregation at `resolution` bit for bit. The
 * output `candidateOverflow` includes source final and inherited loss; `overflow` describes only
 * this destination. Its requirement is a lower bound on the full result after source loss.
 */
export class GPUCellRollup implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellRollupProps;
  /** Key layout of the output resolution. */
  readonly layout: CellKeyLayout;
  /** Fixed-point scale of the sums. */
  readonly sumScale: number;
  /** Resolved `discrete-grid` spatial context. */
  readonly spatialContext: GPUSpatialContext;

  private readonly extraCounts: readonly GraphDataView<'uint32'>[];

  constructor(props: GPUCellRollupProps);
  constructor(props: GPUCellRollupProps, extraCounts: readonly GraphDataView<'uint32'>[] = []) {
    const fields = snapshotConstructionFields(props, [
      'id',
      'family',
      'spatialContext',
      'sourceResolution',
      'resolution',
      'source',
      'sumScale',
      'output'
    ]);
    props = Object.freeze({
      ...fields,
      source: snapshotCellTable(fields.source),
      output: snapshotCellTable(fields.output),
      spatialContext: fields.spatialContext
        ? snapshotConstructionSpatialContext(fields.spatialContext)
        : undefined
    });
    this.extraCounts = [...extraCounts];
    this.id = props.id ?? 'cell-rollup';
    this.props = props;
    const id = this.id;
    this.layout = getCellKeyLayout(props.family, props.resolution);
    getCellKeyLayout(props.family, props.sourceResolution);
    this.spatialContext = resolveCellSpatialContext(
      id,
      props.family,
      props.spatialContext,
      'cells',
      props.resolution
    );
    if (props.resolution > props.sourceResolution) {
      throw new Error(`${id} resolution must not exceed sourceResolution`);
    }
    this.sumScale = props.sumScale ?? GPU_CELL_DEFAULT_SUM_SCALE;
    validateSumScale(id, this.sumScale);
    validateCellTable(id, 'source', props.source);
    validateCellTable(id, 'output', props.output);
    validateRequirement(
      id,
      Boolean(props.output.status.candidateOverflow),
      'output.status.candidateOverflow is required to report source loss'
    );
    const {source, output} = props;
    if ((output.sums || output.sumValues) && !source.sums) {
      throw new Error(`${id} source.sums is required to roll up sums`);
    }
    if ((output.minimums && !source.minimums) || (output.maximums && !source.maximums)) {
      throw new Error(`${id} source extremes are required to roll up extremes`);
    }
    for (const [index, view] of this.extraCounts.entries()) {
      validateSinglePackedView(view, ['uint32'], `${id} extraCounts[${index}]`);
      validateCapacity(id, `extraCounts[${index}] rows`, view.length);
    }
    validateCellTableWritableViews(id, output, this.extraCounts, {rollup: true});
    validateGraphOutputsDisjointFromInputs(
      id,
      [...getCellTableViews(output), ...this.extraCounts],
      getCellTableViews(source)
    );
  }

  /** Returns parent-key, boundary, table, reduction, and publish nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, layout} = this;
    validateCellPublishBindingCount(
      graph.device,
      id,
      Boolean(props.output.status.requiredCount),
      0,
      this.extraCounts.length
    );
    validateCellViewBindingRanges(graph, id, [
      ...getCellTableViews(props.source),
      ...getCellTableViews(props.output),
      ...this.extraCounts
    ]);
    const rows = props.source.cells.length;
    validateCellTableScratchBindingSizes(graph.device, id, rows, props.output.cells.length);
    const twoWords = hasCellKeyHighWord(layout);
    const keyLow = createTransientView(graph, `${id}-key-low`, 'uint32', rows);
    const keyHigh = twoWords
      ? createTransientView(graph, `${id}-key-high`, 'uint32', rows)
      : undefined;
    const keyNode = createCellLevelKernelNode<Parameters>(
      graph,
      {
        id: `${id}-keys`,
        operation: OPERATION,
        variant: 'parent-keys',
        bindings: [
          {
            name: 'childCells',
            view: props.source.cells,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'childCount',
            view: props.source.status.count,
            type: 'u32',
            access: 'read'
          },
          {name: 'keyLow', view: keyLow, type: 'u32', access: 'read_write'},
          ...(keyHigh
            ? [
                {
                  name: 'keyHigh',
                  view: keyHigh,
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        invocationCount: rows,
        declarations: `const RESOLUTION_MASK: u32 = ${props.family === 'quadbin' ? '0x1fu' : '0xfu'};
${CELL_KEY_WGSL}`,
        body: `var compact = cellShiftLeft(vec2u(0u, 1u), CELL_WIDTH);
  let key = vec2u(childCells[childCellsOffset + 2u * index + 1u], childCells[childCellsOffset + 2u * index]);
  if (index < childCount[childCountOffset] && ((key.x >> 20u) & RESOLUTION_MASK) == SOURCE_RESOLUTION) {
    compact = cellGetCompactKey(key, CELL_LOW_BIT, CELL_WIDTH);
  }
  keyLow[keyLowOffset + index] = compact.y;
  ${keyHigh ? 'keyHigh[keyHighOffset + index] = compact.x;' : ''}`
      },
      {
        SOURCE_RESOLUTION: props.sourceResolution,
        CELL_WIDTH: layout.width,
        CELL_LOW_BIT: layout.lowBit
      }
    );
    return [
      ...keyNode.uploads,
      keyNode.node,
      ...getCellTableNodes<Parameters>(graph, {
        id,
        operation: OPERATION,
        layout,
        keyLow,
        keyHigh,
        sorted: true,
        source: {kind: 'cells', table: props.source},
        sumScale: this.sumScale,
        output: props.output,
        candidateSources: [
          props.source.status.overflow,
          ...(props.source.status.candidateOverflow ? [props.source.status.candidateOverflow] : [])
        ],
        extraCounts: this.extraCounts
      })
    ];
  }
}

/** Adds composition-only destinations for a pyramid's level counts. @internal */
export function createCellRollup(
  props: GPUCellRollupProps & {extraCounts?: readonly GraphDataView<'uint32'>[]}
): GPUCellRollup {
  return Reflect.construct(GPUCellRollup, [props, props.extraCounts]) as GPUCellRollup;
}
