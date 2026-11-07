// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Binding, type CommandEncoder, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import type {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUReduction,
  GPUSort,
  alignGraphVectorViews,
  createGPUComputeCommandNode,
  createTransientView,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getViewBinding,
  getViewElementOffset,
  type CompiledGPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPUColumnQuantiles,
  GPUGroupStatistics,
  getGPUColumnQuantilesParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {
  GPU_POINT_DENSITY_D3_HEXAGON_WGSL,
  GPU_POINT_DENSITY_HEXAGON_WGSL
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPUKeplerAggregationLayout, GPUResidentScalarFormat} from '@luma-contributors/kepler';
import {
  GROUP_STATISTIC_BY_AGGREGATION,
  type AggregationChannelPlan,
  type AggregationStylePlan,
  type GroupStatisticName
} from './aggregation-statistics-plan';
import {
  SCALE_LOOKUP_WGSL,
  getScaleLookupLength,
  packScaleLookup,
  type OrdinalLookup
} from './aggregation-scale-lookup';

const WORKGROUP_SIZE = 256;
/** Stand-in for infinity in WGSL, which has no infinity literal. */
const UNBOUNDED = 3.0e38;
const IS_FINITE_WGSL = /* wgsl */ `fn isFiniteBits(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}`;
const SHADER_TYPES = {float32: 'f32', sint32: 'i32', uint32: 'u32'} as const;
/** Heterogeneous alignment: operands are validated by alignGraphVectorViews at runtime. */
const alignViews = alignGraphVectorViews as unknown as (
  graph: GPUCommandGraph,
  vectors: readonly unknown[]
) => readonly (readonly GraphDataView[])[];

type KernelBinding = {
  name: string;
  view: GraphDataView;
  type: 'f32' | 'i32' | 'u32' | 'atomic<u32>';
  access: 'read' | 'read_write';
};
/**
 * Largest dictionary whose first-appearance indices are computed on the GPU. The rank pass sorts
 * the representative cells (at most one per label and never more than the cell budget), so cost
 * no longer grows with the square of the dictionary; only two dictionary-sized scratch arrays remain.
 */
export const MAXIMUM_CATEGORY_REMAP_COUNT = 4194304;

function getLiteral(value: number): string {
  return `${Math.fround(value).toExponential(9)}`;
}

/** Elementwise kernel over one aligned span; every binding is addressed with `index`. */
function createKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    invocationCount: number;
    bindings: readonly KernelBinding[];
    declarations?: string;
    body: string;
  }
): GPUCommandNode<Parameters> {
  const layout = getBoundedDispatchLayout(
    'AggregationStatistics',
    Math.max(props.invocationCount, 1),
    WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const declarations = props.bindings
    .map(
      (
        binding,
        location
      ) => `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n');
  const references = props.bindings.map(binding => `_ = &${binding.name};`).join('\n  ');
  const source = /* wgsl */ `const LENGTH: u32 = ${props.invocationCount}u;
${declarations}
${IS_FINITE_WGSL}
${props.declarations ?? ''}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${references}
  ${getBoundedInvocationIndexSource(layout, WORKGROUP_SIZE)}
  if (index >= LENGTH) { return; }
  ${props.body}
}`;
  let readByteLength = 0;
  let writeByteLength = 0;
  for (const binding of props.bindings) {
    const byteLength = binding.view.length * binding.view.rowByteLength;
    if (binding.access === 'read') readByteLength += byteLength;
    else writeByteLength += byteLength;
  }
  return createGPUComputeCommandNode<Parameters>({
    id: props.id,
    workload: {
      operation: 'AggregationStatistics',
      commandCount: 1,
      maximumWorkgroupCount: layout.x * layout.y * layout.z,
      maximumInvocationCount: layout.x * layout.y * layout.z * WORKGROUP_SIZE,
      readByteLength,
      writeByteLength
    },
    resources: props.bindings.map(binding => ({
      buffer: binding.view,
      usage: binding.access === 'read' ? ('storage-read' as const) : ('storage-read-write' as const)
    })),
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id: props.id,
        source,
        shaderLayout: {
          bindings: props.bindings.map((binding, location) => ({
            name: binding.name,
            type: binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const),
            group: 0,
            location
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          const bindings: Record<string, Binding> = {};
          for (const binding of props.bindings)
            bindings[binding.name] = getViewBinding(binding.view, getBuffer);
          kernel.dispatch(computePass, {bindings, x: layout.x, y: layout.y, z: layout.z});
        },
        destroy: () => kernel.destroy()
      };
    }
  });
}

/** WGSL that writes the cell key of one position, or 0xffffffff for rejected rows. Exported for tests. */
export function getCellKeySource(layout: GPUKeplerAggregationLayout): {
  declarations: string;
  body: string;
} {
  const [minimumX, minimumY, maximumX, maximumY] = layout.bounds.map(getLiteral);
  const [columns, rows] = layout.gridSize;
  const common = `const COLUMNS: u32 = ${columns}u;
const ROWS: u32 = ${rows}u;
const MAXIMUM_FLOAT: f32 = 3.402823466e+38;`;
  const prelude = `let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var key = 0xffffffffu;
  let included = selection[selectionOffset + index] != 0u && isFiniteBits(x) && isFiniteBits(y);
  let minimumX = ${minimumX}; let minimumY = ${minimumY};
  let maximumX = ${maximumX}; let maximumY = ${maximumY};`;
  if (layout.kind === 'grid')
    return {
      declarations: `${common}
fn getCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
  if (maximum == minimum || value == minimum) { return 0u; }
  if (value == maximum) { return size - 1u; }
  if (minimum < 0.0 && maximum > 0.0) {
    let scale = max(abs(minimum), abs(maximum));
    let scaledValue = value / scale;
    let scaledMinimum = minimum / scale;
    let scaledMaximum = maximum / scale;
    return min(
      u32((scaledValue - scaledMinimum) / (scaledMaximum - scaledMinimum) * f32(size)),
      size - 1u
    );
  }
  return min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u);
}`,
      body: `${prelude}
  let inX = x >= minimumX && x <= maximumX && (maximumX != minimumX || x == minimumX);
  let inY = y >= minimumY && y <= maximumY && (maximumY != minimumY || y == minimumY);
  if (included && inX && inY) {
    key = getCoordinate(y, minimumY, maximumY, ROWS) * COLUMNS + getCoordinate(x, minimumX, maximumX, COLUMNS);
  }
  keys[keysOffset + index] = key;`
    };
  const d3 = layout.assignment === 'd3';
  return {
    declarations: `${common}
${d3 ? GPU_POINT_DENSITY_D3_HEXAGON_WGSL : GPU_POINT_DENSITY_HEXAGON_WGSL}`,
    body: `${prelude}
  let radius = ${getLiteral(layout.radius!)};
  let inside = x >= minimumX && x <= maximumX && y >= minimumY && y <= maximumY;
  if (included && inside) {
    let cell = ${d3 ? 'getPointDensityD3HexagonCell' : 'getPointDensityHexagonCell'}(x, y, minimumX, minimumY, radius);
    if (cell.x >= 0 && cell.y >= 0 && u32(cell.x) < COLUMNS && u32(cell.y) < ROWS) {
      key = u32(cell.y) * COLUMNS + u32(cell.x);
    }
  }
  keys[keysOffset + index] = key;`
  };
}

const OUTPUT_FIELDS: Record<GroupStatisticName, string> = {
  mean: 'means',
  sum: 'sumValues',
  minimum: 'minimums',
  maximum: 'maximums',
  median: 'medians',
  standardDeviation: 'standardDeviations',
  variance: 'variances',
  mode: 'modes',
  uniqueCount: 'uniqueCounts'
};

export type AggregationStatisticsProps = {
  device: Device;
  id: string;
  positions: GPUVector<'float32x2'>;
  selectionMask: GPUVector<'uint32'>;
  /**
   * Rows that form bins in native Kepler (CPU-filtered topology, before GPU range filters hide rows
   * inside a bin). Bin order, and therefore categorical first-appearance indices, follow the first
   * such row of every cell. Defaults to `selectionMask`.
   */
  binOrderMask?: GPUVector<'uint32'>;
  layout: GPUKeplerAggregationLayout;
  /** Selected finite in-bounds points per cell; the count measure and emptiness mask. */
  counts: GPUData<'uint32'>;
  columns: Readonly<
    Record<string, {values: GPUVector<GPUResidentScalarFormat>; validity?: GPUVector<'uint32'>}>
  >;
  plan: AggregationStylePlan;
  /** Packed RGBA palette words, already reversed when the native range is reversed. */
  palette: readonly number[];
  /**
   * Dictionary sizes of categorical (dictionary-coded) mode channels. Native wraps these
   * accessors so every distinct label becomes its index of first appearance in bin order (bins
   * ordered by their first source row); the scales then see those indices, not dictionary codes.
   */
  categoryCounts?: {color?: number; elevation?: number};
};

type ChannelStage = {
  /** Four floats: shader domain start, end, hide-below, hide-above. */
  parameters: GraphDataView<'float32'>;
  scaled: GraphDataView<'float32'>;
};

/**
 * GPU channel pipeline for native Grid/Hexagon color and elevation. It owns per-cell colors,
 * visibility (the percentile/cutoff filter) and extrusion heights, reproducing the deck.gl 9
 * aggregation scale/cutoff/domain rules, and computes every native aggregation with
 * GPUGroupStatistics (sample variance, d3 median, smallest-value mode) keyed by the same cell
 * assignment as GPUPointDensity. No source row data is read back.
 */
export class AggregationStatistics {
  /** Packed RGBA per cell. */
  readonly colors: Buffer;
  /** Nonzero when the cell survives the percentile and domain tests. */
  readonly visible: Buffer;
  /** Interpolated elevation per cell before native elevationScale; only with an elevation plan. */
  readonly elevations?: Buffer;
  /** Raw color channel value per cell (NaN when undefined), the hover measure. */
  readonly colorValues: Buffer;
  /**
   * The value the color scale classifies: `colorValues`, or for a categorical mode channel the
   * native first-appearance index of the label (computed on the GPU, ordered by the minimum
   * source row of each label's cells, which equals native bin order).
   */
  readonly scaleInput: Buffer;
  /**
   * Jenks or ordinal color lookup (see aggregation-scale-lookup.ts). Not ready until the owner
   * reads `colorValues` back, derives the lookup and calls `setScaleLookup`; cells stay hidden
   * (NaN) meanwhile.
   */
  readonly scaleLookup?: Buffer;
  readonly scaleLookupKind?: 'jenks' | 'ordinal';
  private readonly cellCount: number;
  private readonly buffers: Buffer[] = [];
  private readonly compiled: CompiledGPUCommandGraph;
  private destroyed = false;

  constructor(props: AggregationStatisticsProps) {
    const {device, id, layout, plan} = props;
    const cellCount = layout.cellCount;
    this.cellCount = cellCount;
    const rowCount = props.positions.length;
    const graph = new GPUCommandGraph(device, {id: `${id}-statistics`});
    const own = (name: string, length: number, values?: Float32Array | Uint32Array): Buffer => {
      const buffer = device.createBuffer({
        id: `${id}-${name}`,
        ...(values ? {data: values} : {byteLength: Math.max(4, length * 4)}),
        usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      this.buffers.push(buffer);
      return buffer;
    };
    const viewOf = <Format extends 'uint32' | 'float32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ): GraphDataView<Format> =>
      graph.createDataView(
        graph.importBuffer(
          {id: `${id}-${name}`, byteLength: buffer.byteLength, usage: buffer.usage},
          buffer
        ),
        {format, length}
      );
    const transient = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const lookupKind =
      plan.color.scale === 'jenks' || plan.color.scale === 'ordinal' ? plan.color.scale : undefined;
    const scaleLookupLength = getScaleLookupLength(cellCount);
    const scaleLookupBuffer = lookupKind ? own('scale-lookup', scaleLookupLength) : undefined;
    this.scaleLookup = scaleLookupBuffer;
    this.scaleLookupKind = lookupKind;
    const scaleLookupView = scaleLookupBuffer
      ? viewOf('scale-lookup', scaleLookupBuffer, 'float32', scaleLookupLength)
      : undefined;
    try {
      const channels = [plan.color, ...(plan.elevation ? [plan.elevation] : [])];
      const counts = graph.importGPUData(`${id}-counts`, props.counts);
      // Weighted channels: one packed key/value stream, reduced by GPUGroupStatistics.
      const weighted = channels.filter(channel => channel.aggregation !== 'count');
      const statisticOutputs = new Map<
        string,
        GraphDataView<'float32'> | GraphDataView<'uint32'>
      >();
      let groupCounts: GraphDataView<'uint32'> | undefined;
      const categoryCounts = props.categoryCounts ?? {};
      const isRemapped = (count: number | undefined): count is number =>
        count !== undefined && count > 0 && count <= MAXIMUM_CATEGORY_REMAP_COUNT;
      // First source row (selected, finite, in-bounds) of every cell: the native bin order key.
      let firstRows: GraphDataView<'uint32'> | undefined;
      if (weighted.length) {
        const keys = transient('keys', 'uint32', rowCount);
        const key = getCellKeySource(layout);
        if (isRemapped(categoryCounts.color) || isRemapped(categoryCounts.elevation)) {
          firstRows = transient('first-rows', 'uint32', cellCount);
          graph.add(
            createKernelNode(graph, {
              id: `${id}-first-rows-clear`,
              invocationCount: cellCount,
              bindings: [
                {name: 'firstRows', view: firstRows, type: 'atomic<u32>', access: 'read_write'}
              ],
              body: 'atomicStore(&firstRows[firstRowsOffset + index], 0xffffffffu);'
            })
          );
        }
        let spanRowBase = 0;
        const positions = graph.importGPUVector(`${id}-positions`, props.positions);
        const selection = graph.importGPUVector(`${id}-selection`, props.selectionMask);
        // Native forms bins from the CPU-filtered rows; GPU range filters only hide rows inside a bin.
        const orderMask =
          firstRows && props.binOrderMask && props.binOrderMask !== props.selectionMask
            ? props.binOrderMask
            : undefined;
        const orderKeys = orderMask ? transient('order-keys', 'uint32', rowCount) : undefined;
        const orderSelection = orderMask
          ? graph.importGPUVector(`${id}-order-selection`, orderMask)
          : undefined;
        alignViews(
          graph,
          orderSelection
            ? [positions, selection, keys, orderSelection, orderKeys]
            : [positions, selection, keys]
        ).forEach((span, spanIndex) => {
          graph.add(
            createKernelNode(graph, {
              id: `${id}-keys-${spanIndex}`,
              invocationCount: span[2].length,
              declarations: key.declarations,
              bindings: [
                {name: 'positions', view: span[0], type: 'f32', access: 'read'},
                {name: 'selection', view: span[1], type: 'u32', access: 'read'},
                {name: 'keys', view: span[2], type: 'u32', access: 'read_write'}
              ],
              body: key.body
            })
          );
          if (orderMask) {
            graph.add(
              createKernelNode(graph, {
                id: `${id}-order-keys-${spanIndex}`,
                invocationCount: span[2].length,
                declarations: key.declarations,
                bindings: [
                  {name: 'positions', view: span[0], type: 'f32', access: 'read'},
                  {name: 'selection', view: span[3], type: 'u32', access: 'read'},
                  {name: 'keys', view: span[4], type: 'u32', access: 'read_write'}
                ],
                body: key.body
              })
            );
          }
          if (firstRows) {
            graph.add(
              createKernelNode(graph, {
                id: `${id}-first-rows-${spanIndex}`,
                invocationCount: span[2].length,
                declarations: `const ROW_BASE: u32 = ${spanRowBase}u;`,
                bindings: [
                  {name: 'keys', view: orderMask ? span[4] : span[2], type: 'u32', access: 'read'},
                  {name: 'firstRows', view: firstRows, type: 'atomic<u32>', access: 'read_write'}
                ],
                body: `let cellKey = keys[keysOffset + index];
  if (cellKey != 0xffffffffu) { atomicMin(&firstRows[firstRowsOffset + cellKey], ROW_BASE + index); }`
              })
            );
          }
          spanRowBase += span[2].length;
        });
        const columnNames = [...new Set(weighted.map(channel => channel.weightColumn!))];
        const statisticColumns = columnNames.map((name, columnIndex) => {
          const column = props.columns[name];
          if (!column) throw new Error(`No resident weight column is bound for ${name}.`);
          const values = transient(`values-${columnIndex}`, 'float32', rowCount);
          const type =
            SHADER_TYPES[
              (column.values.format ??
                column.values.data[0].format ??
                'float32') as GPUResidentScalarFormat
            ];
          const operands: unknown[] = [
            graph.importGPUVector(`${id}-source-${columnIndex}`, column.values)
          ];
          if (column.validity)
            operands.push(graph.importGPUVector(`${id}-validity-${columnIndex}`, column.validity));
          operands.push(values);
          alignViews(graph, operands).forEach((span, spanIndex) => {
            const source = span[0];
            const validity = column.validity ? span[1] : undefined;
            const output = span[span.length - 1];
            graph.add(
              createKernelNode(graph, {
                id: `${id}-values-${columnIndex}-${spanIndex}`,
                invocationCount: output.length,
                bindings: [
                  {name: 'source', view: source, type, access: 'read'},
                  ...(validity
                    ? [
                        {
                          name: 'validity',
                          view: validity,
                          type: 'u32' as const,
                          access: 'read' as const
                        }
                      ]
                    : []),
                  {name: 'output', view: output, type: 'f32', access: 'read_write'}
                ],
                body: `let value = f32(source[sourceOffset + index]);
  let defined = ${validity ? 'validity[validityOffset + index] != 0u && ' : ''}isFiniteBits(value);
  output[outputOffset + index] = select(getNaN(), value + 0.0, defined);`
              })
            );
          });
          const statistics = [
            ...new Set(
              weighted
                .filter(channel => channel.weightColumn === name)
                .map(
                  channel =>
                    GROUP_STATISTIC_BY_AGGREGATION[
                      channel.aggregation as Exclude<typeof channel.aggregation, 'count'>
                    ]
                )
            )
          ];
          const output: Record<string, GraphDataView> = {};
          for (const statistic of statistics) {
            const unique = statistic === 'uniqueCount';
            const view = transient(
              `statistic-${columnIndex}-${statistic}`,
              unique ? 'uint32' : 'float32',
              cellCount
            );
            output[OUTPUT_FIELDS[statistic]] = view;
            statisticOutputs.set(`${name}:${statistic}`, view);
          }
          return {values, statistics, output};
        });
        groupCounts = transient('group-counts', 'uint32', cellCount);
        graph.add(
          new GPUGroupStatistics({
            id: `${id}-group`,
            keys,
            keyCount: cellCount,
            variance: 'sample',
            columns: statisticColumns as never,
            output: {
              keys: transient('group-keys', 'uint32', cellCount),
              counts: groupCounts,
              status: {
                count: transient('group-count', 'uint32', 1),
                overflow: transient('group-overflow', 'uint32', 1)
              }
            }
          })
        );
      }
      // Raw per-cell channel value; NaN means undefined and hides the cell.
      const assemble = (
        name: string,
        channel: AggregationChannelPlan,
        output: GraphDataView<'float32'>
      ): void => {
        if (channel.aggregation === 'count') {
          graph.add(
            createKernelNode(graph, {
              id: `${id}-${name}-assemble`,
              invocationCount: cellCount,
              bindings: [
                {name: 'counts', view: counts, type: 'u32', access: 'read'},
                {name: 'output', view: output, type: 'f32', access: 'read_write'}
              ],
              body: `let count = counts[countsOffset + index];
  output[outputOffset + index] = select(getNaN(), f32(count), count > 0u);`
            })
          );
          return;
        }
        const statistic =
          GROUP_STATISTIC_BY_AGGREGATION[
            channel.aggregation as Exclude<typeof channel.aggregation, 'count'>
          ];
        const source = statisticOutputs.get(`${channel.weightColumn}:${statistic}`)!;
        graph.add(
          createKernelNode(graph, {
            id: `${id}-${name}-assemble`,
            invocationCount: cellCount,
            bindings: [
              {name: 'groupCounts', view: groupCounts!, type: 'u32', access: 'read'},
              {
                name: 'source',
                view: source,
                type: source.format === 'uint32' ? 'u32' : 'f32',
                access: 'read'
              },
              {name: 'output', view: output, type: 'f32', access: 'read_write'}
            ],
            body: `let present = groupCounts[groupCountsOffset + index] > 0u;
  output[outputOffset + index] = select(getNaN(), f32(source[sourceOffset + index]), present);`
          })
        );
      };
      /**
       * Native wrapOrdinalAccessor: label code -> index of the label's first appearance among
       * bins in bin order. Bin order is the order of each cell's first source row, so a label's
       * index is the number of other labels whose earliest cell precedes its earliest cell.
       */
      const remapCategories = (
        name: string,
        raw: GraphDataView<'float32'>,
        categoryCount: number,
        mapped: GraphDataView<'float32'>
      ): void => {
        const codeFirstRows = transient(`${name}-code-first-rows`, 'uint32', categoryCount);
        const codeIndices = transient(`${name}-code-indices`, 'float32', categoryCount);
        const declarations = `const CATEGORY_COUNT: u32 = ${categoryCount}u;`;
        graph.add(
          createKernelNode(graph, {
            id: `${id}-${name}-code-clear`,
            invocationCount: categoryCount,
            bindings: [
              {
                name: 'codeFirstRows',
                view: codeFirstRows,
                type: 'atomic<u32>',
                access: 'read_write'
              },
              {name: 'codeIndices', view: codeIndices, type: 'f32', access: 'read_write'}
            ],
            body: `atomicStore(&codeFirstRows[codeFirstRowsOffset + index], 0xffffffffu);
  codeIndices[codeIndicesOffset + index] = getNaN();`
          })
        );
        graph.add(
          createKernelNode(graph, {
            id: `${id}-${name}-code-first`,
            invocationCount: cellCount,
            declarations,
            bindings: [
              {name: 'raw', view: raw, type: 'f32', access: 'read'},
              {name: 'firstRows', view: firstRows!, type: 'u32', access: 'read'},
              {
                name: 'codeFirstRows',
                view: codeFirstRows,
                type: 'atomic<u32>',
                access: 'read_write'
              }
            ],
            body: `let value = raw[rawOffset + index];
  if (isFiniteBits(value) && value >= 0.0 && u32(value) < CATEGORY_COUNT) {
    atomicMin(&codeFirstRows[codeFirstRowsOffset + u32(value)], firstRows[firstRowsOffset + index]);
  }`
          })
        );
        // Every label has exactly one representative cell: the cell holding its earliest source
        // row. Sorting the representatives by that row ranks the labels in O(cells log cells).
        const representativeKeys = transient(`${name}-representative-keys`, 'uint32', cellCount);
        const representativeCells = transient(`${name}-representative-cells`, 'uint32', cellCount);
        const sortedKeys = transient(`${name}-sorted-keys`, 'uint32', cellCount);
        const sortedCells = transient(`${name}-sorted-cells`, 'uint32', cellCount);
        graph.add(
          createKernelNode(graph, {
            id: `${id}-${name}-representatives`,
            invocationCount: cellCount,
            declarations,
            bindings: [
              {name: 'raw', view: raw, type: 'f32', access: 'read'},
              {name: 'firstRows', view: firstRows!, type: 'u32', access: 'read'},
              {
                name: 'codeFirstRows',
                view: codeFirstRows,
                type: 'atomic<u32>',
                access: 'read_write'
              },
              {
                name: 'representativeKeys',
                view: representativeKeys,
                type: 'u32',
                access: 'read_write'
              },
              {
                name: 'representativeCells',
                view: representativeCells,
                type: 'u32',
                access: 'read_write'
              }
            ],
            body: `let value = raw[rawOffset + index];
  var key = 0xffffffffu;
  if (isFiniteBits(value) && value >= 0.0 && u32(value) < CATEGORY_COUNT) {
    let own = firstRows[firstRowsOffset + index];
    if (own == atomicLoad(&codeFirstRows[codeFirstRowsOffset + u32(value)])) { key = own; }
  }
  representativeKeys[representativeKeysOffset + index] = key;
  representativeCells[representativeCellsOffset + index] = index;`
          })
        );
        graph.add(
          new GPUSort({
            id: `${id}-${name}-representative-sort`,
            keys: representativeKeys,
            values: representativeCells,
            outputKeys: sortedKeys,
            outputValues: sortedCells,
            keyBits: 32
          })
        );
        graph.add(
          createKernelNode(graph, {
            id: `${id}-${name}-code-rank`,
            invocationCount: cellCount,
            declarations,
            bindings: [
              {name: 'raw', view: raw, type: 'f32', access: 'read'},
              {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
              {name: 'sortedCells', view: sortedCells, type: 'u32', access: 'read'},
              {name: 'codeIndices', view: codeIndices, type: 'f32', access: 'read_write'}
            ],
            body: `if (sortedKeys[sortedKeysOffset + index] != 0xffffffffu) {
    let value = raw[rawOffset + sortedCells[sortedCellsOffset + index]];
    codeIndices[codeIndicesOffset + u32(value)] = f32(index);
  }`
          })
        );
        graph.add(
          createKernelNode(graph, {
            id: `${id}-${name}-code-map`,
            invocationCount: cellCount,
            declarations,
            bindings: [
              {name: 'raw', view: raw, type: 'f32', access: 'read'},
              {name: 'codeIndices', view: codeIndices, type: 'f32', access: 'read'},
              {name: 'mapped', view: mapped, type: 'f32', access: 'read_write'}
            ],
            body: `let value = raw[rawOffset + index];
  var result = getNaN();
  if (isFiniteBits(value) && value >= 0.0 && u32(value) < CATEGORY_COUNT) {
    result = codeIndices[codeIndicesOffset + u32(value)];
  }
  mapped[mappedOffset + index] = result;`
          })
        );
      };
      const colorValues = own('color-values', cellCount);
      this.colorValues = colorValues;
      const colorRaw = viewOf('color-values', colorValues, 'float32', cellCount);
      assemble('color', plan.color, colorRaw);
      let colorScaleInput = colorRaw;
      this.scaleInput = colorValues;
      if (isRemapped(categoryCounts.color) && firstRows) {
        this.scaleInput = own('color-category-index', cellCount);
        colorScaleInput = viewOf('color-category-index', this.scaleInput, 'float32', cellCount);
        remapCategories('color', colorRaw, categoryCounts.color, colorScaleInput);
      }
      const elevationRaw = plan.elevation
        ? transient('elevation-values', 'float32', cellCount)
        : undefined;
      if (plan.elevation && elevationRaw) assemble('elevation', plan.elevation, elevationRaw);
      let elevationScaleInput = elevationRaw;
      if (elevationRaw && isRemapped(categoryCounts.elevation) && firstRows) {
        elevationScaleInput = transient('elevation-category-index', 'float32', cellCount);
        remapCategories('elevation', elevationRaw, categoryCounts.elevation, elevationScaleInput);
      }

      const colorCount = props.palette.length;
      const createStage = (
        name: string,
        channel: AggregationChannelPlan,
        raw: GraphDataView<'float32'>
      ): ChannelStage => {
        const scaled = transient(`${name}-scaled`, 'float32', cellCount);
        const valid = transient(`${name}-valid`, 'uint32', cellCount);
        const thresholds: GraphDataView<'float32'>[] = [];
        const addQuantiles = (
          label: string,
          probabilities: readonly number[]
        ): GraphDataView<'float32'> => {
          const quantiles = transient(
            `${name}-${label}-quantiles`,
            'float32',
            probabilities.length
          );
          const parameters = Float32Array.from(
            getGPUColumnQuantilesParameterValues({
              quantiles: probabilities,
              quantileCount: probabilities.length,
              interpolation: 'linear'
            })
          );
          graph.add(
            new GPUColumnQuantiles({
              id: `${id}-${name}-${label}`,
              values: raw,
              parameters: viewOf(
                `${name}-${label}-parameters`,
                own(`${name}-${label}-parameters`, parameters.length, parameters),
                'float32',
                parameters.length
              ),
              quantileCount: probabilities.length,
              output: {
                quantiles,
                validCount: transient(`${name}-${label}-valid-count`, 'uint32', 1)
              }
            })
          );
          return quantiles;
        };
        const percentileActive = channel.lowerPercentile > 0 || channel.upperPercentile < 100;
        if (channel.scale === 'quantile') {
          thresholds.push(
            addQuantiles(
              'thresholds-a',
              Array.from({length: 64}, (_, index) => (index + 1) / 100)
            ),
            addQuantiles(
              'thresholds-b',
              Array.from({length: 35}, (_, index) => (index + 65) / 100)
            )
          );
        }
        let cutoffQuantiles: GraphDataView<'float32'> | undefined;
        const lowerIndex = Math.floor(channel.lowerPercentile);
        const upperIndex = Math.floor(channel.upperPercentile);
        const lowerFinite = lowerIndex >= 1 && lowerIndex <= 99;
        const upperFinite = upperIndex >= 1 && upperIndex <= 99;
        const classBreaksScale =
          channel.scale === 'custom' || channel.scale === 'jenks' || channel.scale === 'ordinal';
        if (percentileActive && channel.scale !== 'quantile' && !classBreaksScale)
          cutoffQuantiles = addQuantiles('cutoff', [
            lowerFinite ? lowerIndex / 100 : NaN,
            upperFinite ? upperIndex / 100 : NaN
          ]);
        const bindings: KernelBinding[] = [{name: 'raw', view: raw, type: 'f32', access: 'read'}];
        const usesLookup =
          (channel.scale === 'jenks' || channel.scale === 'ordinal') && scaleLookupView;
        if (usesLookup)
          bindings.push({name: 'scaleLookup', view: scaleLookupView, type: 'f32', access: 'read'});
        thresholds.forEach((view, index) =>
          bindings.push({name: `thresholds${index}`, view, type: 'f32', access: 'read'})
        );
        bindings.push(
          {name: 'scaled', view: scaled, type: 'f32', access: 'read_write'},
          {name: 'valid', view: valid, type: 'u32', access: 'read_write'}
        );
        let transform = 'v';
        let declarations = '';
        if (channel.scale === 'quantile') {
          declarations = `fn getPercentileIndex(v: f32) -> f32 {
  var count = 0u;
  for (var i = 0u; i < 64u; i++) { if (thresholds0[thresholds0Offset + i] <= v) { count++; } }
  for (var i = 0u; i < 35u; i++) { if (thresholds1[thresholds1Offset + i] <= v) { count++; } }
  return f32(count);
}`;
          transform = 'getPercentileIndex(v)';
        } else if (usesLookup) {
          declarations = SCALE_LOOKUP_WGSL;
          transform = channel.scale === 'jenks' ? 'getJenksIndex(v)' : 'getOrdinalIndex(v)';
        } else if (channel.scale === 'custom') {
          const list = channel.customThresholds.map(getLiteral);
          declarations = list.length
            ? `var<private> CUSTOM: array<f32, ${list.length}> = array<f32, ${list.length}>(${list.join(', ')});
fn getCustomIndex(v: f32) -> f32 {
  for (var i = 0u; i < ${list.length}u; i++) { if (v < CUSTOM[i]) { return f32(i); } }
  return ${colorCount - 1}.0;
}`
            : `fn getCustomIndex(v: f32) -> f32 { return ${colorCount - 1}.0; }`;
          transform = 'getCustomIndex(v)';
        }
        graph.add(
          createKernelNode(graph, {
            id: `${id}-${name}-scale`,
            invocationCount: cellCount,
            bindings,
            declarations,
            body: `let v = raw[rawOffset + index];
  let ok = isFiniteBits(v);
  scaled[scaledOffset + index] = select(getNaN(), ${transform}, ok);
  valid[validOffset + index] = select(0u, 1u, ok);`
          })
        );
        // Domain: native getResultDomain (min/max of every aggregated value) for raw scales.
        const rawDomain = channel.scale !== 'quantile' && !classBreaksScale;
        const extent = transient(`${name}-extent`, 'float32', 2);
        if (rawDomain)
          graph.add(
            new GPUReduction({
              id: `${id}-${name}-extent`,
              input: scaled,
              mask: valid,
              output: extent,
              operation: 'extent'
            })
          );
        const parameters = transient(`${name}-parameters`, 'float32', 4);
        const domain =
          channel.scale === 'quantile'
            ? 'let domainStart = 0.0; let domainEnd = 99.0;'
            : channel.scale === 'ordinal' && scaleLookupView
              ? 'let domainStart = 0.0; let domainEnd = max(scaleLookup[scaleLookupOffset + 1u] - 1.0, 0.0);'
              : channel.scale === 'custom' || channel.scale === 'jenks'
                ? `let domainStart = 0.0; let domainEnd = ${colorCount - 1}.0;`
                : 'let domainStart = extent[extentOffset]; let domainEnd = extent[extentOffset + 1u];';
        const cutoff =
          channel.scale === 'quantile'
            ? `let cutoffStart = ${getLiteral(channel.lowerPercentile)}; let cutoffEnd = ${getLiteral(channel.upperPercentile - 1)};`
            : cutoffQuantiles
              ? `let cutoffStart = ${lowerFinite ? `select(-${UNBOUNDED}, cutoffs[cutoffsOffset], isFiniteBits(cutoffs[cutoffsOffset]))` : `-${UNBOUNDED}`};
  let cutoffEnd = ${upperFinite ? `select(${UNBOUNDED}, cutoffs[cutoffsOffset + 1u], isFiniteBits(cutoffs[cutoffsOffset + 1u]))` : `${UNBOUNDED}`};`
              : `let cutoffStart = -${UNBOUNDED}; let cutoffEnd = ${UNBOUNDED};`;
        graph.add(
          createKernelNode(graph, {
            id: `${id}-${name}-domain`,
            invocationCount: 1,
            bindings: [
              ...(rawDomain
                ? [{name: 'extent', view: extent, type: 'f32' as const, access: 'read' as const}]
                : []),
              ...(channel.scale === 'ordinal' && scaleLookupView
                ? [
                    {
                      name: 'scaleLookup',
                      view: scaleLookupView,
                      type: 'f32' as const,
                      access: 'read' as const
                    }
                  ]
                : []),
              ...(cutoffQuantiles
                ? [
                    {
                      name: 'cutoffs',
                      view: cutoffQuantiles,
                      type: 'f32' as const,
                      access: 'read' as const
                    }
                  ]
                : []),
              {name: 'parameters', view: parameters, type: 'f32', access: 'read_write'}
            ],
            body: `${domain}
  ${cutoff}
  parameters[parametersOffset] = max(domainStart, cutoffStart);
  parameters[parametersOffset + 1u] = min(domainEnd, cutoffEnd);
  parameters[parametersOffset + 2u] = max(domainStart - 1.0, cutoffStart);
  parameters[parametersOffset + 3u] = min(domainEnd + 1.0, cutoffEnd);`
          })
        );
        return {parameters, scaled};
      };
      const colorStage = createStage('color', plan.color, colorScaleInput);
      const elevationStage =
        plan.elevation && elevationScaleInput
          ? createStage('elevation', plan.elevation, elevationScaleInput)
          : undefined;

      this.colors = own('cell-colors', cellCount);
      this.visible = own('cell-visible', cellCount);
      const palette = Uint32Array.from(props.palette);
      const paletteView = viewOf(
        'palette',
        own('palette', palette.length, palette),
        'uint32',
        palette.length
      );
      const bindings: KernelBinding[] = [
        {name: 'colorScaled', view: colorStage.scaled, type: 'f32', access: 'read'},
        {name: 'colorParameters', view: colorStage.parameters, type: 'f32', access: 'read'},
        {name: 'palette', view: paletteView, type: 'u32', access: 'read'}
      ];
      if (elevationStage)
        bindings.push(
          {name: 'elevationScaled', view: elevationStage.scaled, type: 'f32', access: 'read'},
          {
            name: 'elevationParameters',
            view: elevationStage.parameters,
            type: 'f32',
            access: 'read'
          }
        );
      bindings.push(
        {
          name: 'colors',
          view: viewOf('cell-colors', this.colors, 'uint32', cellCount),
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'visible',
          view: viewOf('cell-visible', this.visible, 'uint32', cellCount),
          type: 'u32',
          access: 'read_write'
        }
      );
      if (elevationStage) {
        this.elevations = own('cell-elevations', cellCount);
        bindings.push({
          name: 'elevations',
          view: viewOf('cell-elevations', this.elevations, 'float32', cellCount),
          type: 'f32',
          access: 'read_write'
        });
      }
      const linear = plan.interpolation === 'linear';
      graph.add(
        createKernelNode(graph, {
          id: `${id}-style`,
          invocationCount: cellCount,
          bindings,
          declarations: `const COLOR_COUNT: u32 = ${colorCount}u;
fn getPaletteColor(position: u32) -> vec4<f32> {
  return unpack4x8unorm(palette[paletteOffset + min(position, COLOR_COUNT - 1u)]);
}`,
          body: `let colorValue = colorScaled[colorScaledOffset + index];
  let domainStart = colorParameters[colorParametersOffset];
  let domainEnd = colorParameters[colorParametersOffset + 1u];
  var shown = isFiniteBits(colorValue) &&
    colorValue >= colorParameters[colorParametersOffset + 2u] &&
    colorValue <= colorParameters[colorParametersOffset + 3u];
  let span = domainEnd - domainStart;
  var fraction = 0.0;
  if (span != 0.0) { fraction = clamp((colorValue - domainStart) / span, 0.0, 1.0); }
  ${
    linear
      ? `let texel = fraction * f32(COLOR_COUNT) - 0.5;
  let base = floor(texel);
  let blend = texel - base;
  let first = getPaletteColor(u32(max(base, 0.0)));
  let second = getPaletteColor(u32(max(base + 1.0, 0.0)));
  let color = mix(first, second, blend);`
      : 'let color = getPaletteColor(u32(floor(fraction * f32(COLOR_COUNT))));'
  }
  colors[colorsOffset + index] = pack4x8unorm(color);
  ${
    elevationStage
      ? `let elevationValue = elevationScaled[elevationScaledOffset + index];
  let elevationStart = elevationParameters[elevationParametersOffset];
  let elevationEnd = elevationParameters[elevationParametersOffset + 1u];
  let elevationDefined = isFiniteBits(elevationValue);
  let elevationInside = !elevationDefined ||
    (elevationValue >= elevationParameters[elevationParametersOffset + 2u] &&
     elevationValue <= elevationParameters[elevationParametersOffset + 3u]);
  ${plan.extruded ? 'shown = shown && elevationDefined && elevationInside;' : 'shown = shown && elevationInside;'}
  var elevationFraction = 0.0;
  if (elevationEnd != elevationStart) { elevationFraction = clamp((elevationValue - elevationStart) / (elevationEnd - elevationStart), 0.0, 1.0); }
  elevations[elevationsOffset + index] = mix(${getLiteral(plan.elevationRange[0])}, ${getLiteral(plan.elevationRange[1])}, elevationFraction);`
      : ''
  }
  visible[visibleOffset + index] = select(0u, 1u, shown);`
        })
      );
      this.compiled = graph.compile();
    } catch (error) {
      for (const buffer of this.buffers) buffer.destroy();
      throw error;
    }
  }

  /** Bounded readback of the per-cell color measure (cell budget, never source rows). */
  async readColorValues(): Promise<Float32Array> {
    const bytes = await this.colorValues.readAsync(0, this.cellCount * 4);
    return new Float32Array(bytes.slice().buffer);
  }

  /** Bounded readback of the value the color scale classifies (first-appearance index if categorical). */
  async readScaleInput(): Promise<Float32Array> {
    const bytes = await this.scaleInput.readAsync(0, this.cellCount * 4);
    return new Float32Array(bytes.slice().buffer);
  }

  /** Publishes the jenks thresholds or ordinal table derived from `readColorValues`. */
  setScaleLookup(source: readonly number[] | OrdinalLookup): void {
    if (this.destroyed || !this.scaleLookup || !this.scaleLookupKind) return;
    this.scaleLookup.write(
      packScaleLookup(
        this.scaleLookupKind,
        source,
        this.scaleLookup.byteLength / Float32Array.BYTES_PER_ELEMENT
      )
    );
  }

  encode(encoder: CommandEncoder): void {
    if (!this.destroyed) this.compiled.encode(encoder, {parameters: undefined});
  }

  get accountedResidentByteLength(): number {
    return this.destroyed
      ? 0
      : this.buffers.reduce((total, buffer) => total + buffer.byteLength, 0) +
          this.compiled.stats.physicalTransientResourceBytes;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.compiled.destroy();
    for (const buffer of this.buffers) buffer.destroy();
  }
}
