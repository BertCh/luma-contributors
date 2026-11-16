// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  GraphDataView,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import type {Device} from '@luma.gl/core';
import type {GPUInvalidStatusPort} from '../../analysis-core/contracts/status';
import {validateGPUStatusPort} from '../../analysis-core/contracts/status';
import {
  FIXED_POINT_WGSL,
  FIXED_POINT_WORD_COUNT,
  createFixedPointClearNode,
  createFixedPointDecodeNode,
  getFixedPointAtomicAddWGSL,
  getFixedPointHeadroomBits
} from '../../analysis-core/kernels/fixed-point-accumulator';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength,
  getGraphViewChunks
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {
  validateGraphChunkBindingRanges,
  validateDisjointWritableViews
} from '../../analysis-core/kernels/graph-chunk-dispatch';

const OPERATION = 'GPUValueCube';
/** Cell and period indices are `uint32`, and the bin index `cell * periodCount + period` must not wrap. */
const MAXIMUM_INDEX = 0xfffffffe;
/** Row invocations and exact per-bin counters use uint32. */
const MAXIMUM_ROW_COUNT = 0xffffffff;
/** Threads of the cooperative scale pass. */
const SCALE_WORKGROUP_SIZE = 256;

/**
 * Largest `cellCount * periodCount` of {@link GPUValueCube} on `device`: the exact sums hold three
 * `uint32` words (12 bytes) per bin in one storage binding, so 11,184,810 bins on a default device
 * (128 MiB bindings).
 */
export function getGPUValueCubeMaximumBinCount(device: Device): number {
  return getStorageBindingElementLimit(device, FIXED_POINT_WORD_COUNT * 4);
}

/**
 * Caller-owned outputs of {@link GPUValueCube}. Every view holds `cellCount * periodCount` rows in
 * the cell-major layout `cell * periodCount + period`. Provide at least one of `counts`, `sums` and
 * `means`.
 */
export type GPUValueCubeOutput = {
  /** Binned rows per (cell, period): exact. */
  counts?: GraphDataView<'uint32'>;
  /**
   * Sum of the binned values per (cell, period), the exact 96-bit fixed-point sum rounded once to
   * the nearest float32 (ties to even). 0 for an empty bin.
   */
  sums?: GraphDataView<'float32'>;
  /**
   * `sums / counts` in float32, NaN for an empty bin, so the cube is directly the missing-aware
   * input of `GPUEmergingHotSpots`. The exact sum is rounded to float32 first and then divided, so
   * the mean is within a few float32 units in the last place of the true mean (WGSL division is
   * not correctly rounded, up to 2.5 ULP).
   */
  means?: GraphDataView<'float32'>;
  /** `status.invalidCount`: rows that passed the mask but were not binned, see {@link GPUValueCubeProps}. */
  status?: GPUInvalidStatusPort;
};

/**
 * Properties for {@link GPUValueCube}.
 *
 * Per-frame (no recompile): the contents of `cells`, `periods`, `values` and `mask`. Topology (needs
 * a new graph): the row count, `cellCount`, `periodCount`, which outputs and `mask` are present.
 * The cube shape is compile-time because it fixes the size of every output.
 */
export type GPUValueCubeProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'value-cube'`. */
  id?: string;
  /** Number of cells, 1 to `2^32 - 2`; `cellCount * periodCount` is limited by the device. */
  cellCount: number;
  /** Number of periods, 1 to `2^32 - 2`. */
  periodCount: number;
  /** Cell index per row, `uint32`. A row with a cell at or above `cellCount` is not binned. */
  cells: GraphDataView<'uint32'>;
  /**
   * Period per row, `uint32`: the `periods` output of `GPUPeriodIndex`. `0xffffffff`
   * (`GPU_PERIOD_INDEX_INVALID`, a row that upstream already excluded) is skipped silently; any other
   * period at or above `periodCount` (`GPU_PERIOD_INDEX_OUT_OF_RANGE` included) is not binned.
   */
  periods: GraphDataView<'uint32'>;
  /** Value per row, `float32`. A non-finite value is not binned. */
  values: GraphDataView<'float32'>;
  /** Optional `uint32` row mask, for example `outputMask` of `GPUTimeWindowFilter`; zero skips the row silently. */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUValueCubeOutput;
};

/**
 * Dense (cell, period) value cube: count, sum and mean of a value column per cell and period, in
 * the cell-major layout of the hot spot cube. Together with `GPUPeriodIndex` (the periods) and
 * `GPUTimeWindowFilter` (the mask) it turns timestamped, valued events into the cube that
 * `GPUEmergingHotSpots` reads, with no readback.
 *
 * Sums are deterministic and independent of thread order: each value is quantized to the shared
 * 96-bit fixed-point accumulator (`analysis-core/kernels/fixed-point-accumulator.ts`, library
 * decision 4) and added with integer atomics, then the exact integer is rounded once to float32.
 * There are no float atomics. The unit is data-derived: a power of two set by the largest
 * magnitude among the rows that are binned (an order-independent maximum reduced on the GPU) and
 * by the bit length of the row count as the bound on terms per bin. A value keeps all 24
 * significand bits while its magnitude is at least `max * 2^(h - 71)` where `h` is that bit length
 * (a dynamic range of about 1e15 at a million rows); a smaller value is rounded to the nearest
 * unit, so the sum of `n` terms is off by at most `n / 2` units before the final float32 rounding.
 * Because the unit follows the largest binned value, the rounding of a bin can change when a
 * different row set is binned (for example another time window), and only then.
 *
 * Not binned: masked rows (silent), rows whose period is `0xffffffff` (silent), and rows with a
 * cell or period out of range or a non-finite value (counted in `output.status.invalidCount`).
 *
 * Limits: `cellCount`, `periodCount` and the row count at most `2^32 - 2`; `cellCount * periodCount`
 * at most {@link getGPUValueCubeMaximumBinCount} (11,184,810 on a default device) because the
 * fixed-point words need 12 bytes per bin in one binding; the count of binned rows per bin is
 * bounded by the row count. Inputs and outputs are single packed views.
 *
 * For a sparse cube over a very large key space (DGGS cells) use `GPUCellPeriodKeys` and
 * `GPUGroupStatistics` instead of this dense cube.
 */
export class GPUValueCube implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUValueCubeProps;
  /** Number of input rows. */
  readonly rowCount: number;
  /** `cellCount * periodCount`. */
  readonly binCount: number;

  constructor(props: GPUValueCubeProps) {
    this.id = props.id ?? 'value-cube';
    this.props = props;
    const id = this.id;
    const {cellCount, periodCount, cells, periods, values, mask, output} = props;
    validateCapacity(id, 'cellCount', cellCount, 1, MAXIMUM_INDEX);
    validateCapacity(id, 'periodCount', periodCount, 1, MAXIMUM_INDEX);
    this.binCount = cellCount * periodCount;
    validateMaximumCount(id, 'cellCount * periodCount', this.binCount, MAXIMUM_INDEX);
    validateSinglePackedView(cells, ['uint32'], `${id} cells`);
    validateSinglePackedView(periods, ['uint32'], `${id} periods`);
    validateSinglePackedView(values, ['float32'], `${id} values`);
    this.rowCount = cells.length;
    validateCapacity(id, 'cells length', this.rowCount, 1, MAXIMUM_ROW_COUNT);
    validateViewLength(id, 'periods', periods, this.rowCount);
    validateViewLength(id, 'values', values, this.rowCount);
    if (mask) {
      validateSinglePackedView(mask, ['uint32'], `${id} mask`);
      validateViewLength(id, 'mask', mask, this.rowCount);
    }
    validateRequirement(
      id,
      Boolean(output.counts || output.sums || output.means),
      'output needs at least one of counts, sums and means'
    );
    for (const [name, view, format] of [
      ['counts', output.counts, 'uint32'],
      ['sums', output.sums, 'float32'],
      ['means', output.means, 'float32']
    ] as const) {
      if (view) {
        validateSinglePackedView(view, [format], `${id} output.${name}`);
        validateViewLength(id, `output.${name}`, view, this.binCount);
      }
    }
    validateGPUStatusPort(id, {invalidCount: output.status?.invalidCount});
    const outputs = this.getOutputViews();
    validateGraphOutputsDisjointFromInputs(id, outputs, this.getInputViews());
    {
      const writableOutputs: Readonly<Record<string, GraphDataView | GraphVectorView | undefined>> =
        {
          counts: output.counts,
          sums: output.sums,
          means: output.means,
          invalidCount: output.status?.invalidCount
        };
      const publicationPasses: readonly (readonly (
        | GraphDataView
        | GraphVectorView
        | undefined
      )[])[] = [
        [output.counts, output.status?.invalidCount],
        ...(output.means ? [[output.counts, output.sums, output.means]] : [])
      ];
      // Model all logical writes as separate chunks; only the explicit publication groups coexist.
      const writableWordChunks = Object.values(writableOutputs)
        .flatMap(view => (view ? getGraphViewChunks(view) : []))
        .map(
          view =>
            new GraphDataView(view.buffer, {
              format: 'uint32',
              length: (view.length * view.rowByteLength) / 4,
              byteOffset: view.byteOffset,
              byteStride: 4,
              rowByteLength: 4
            })
        );
      const writableWordCount = writableWordChunks.reduce(
        (total, chunk) => total + chunk.length,
        0
      );
      validateDisjointWritableViews(id, {
        outputs: new GraphVectorView({
          id: `${id}-writable-ranges`,
          name: 'outputs',
          format: 'uint32',
          length: writableWordCount,
          valueLength: writableWordCount,
          stride: 1,
          byteStride: 4,
          rowByteLength: 4,
          data: writableWordChunks
        })
      });
      for (const fields of publicationPasses) {
        validateDisjointWritableViews(
          id,
          Object.fromEntries(fields.map((view, index) => [String(index), view]))
        );
      }
    }
  }

  /**
   * Returns the clears, the scale pass, the accumulate pass, the decode and the optional mean pass.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, rowCount, binCount} = this;
    const {cells, periods, values, mask, output, cellCount, periodCount} = props;
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [...this.getInputViews(), ...this.getOutputViews()].map((view, index) => [
          `view[${index}]`,
          view
        ])
      )
    );
    validateMaximumCount(
      id,
      'fixed-point scratch byteLength',
      binCount * FIXED_POINT_WORD_COUNT * 4,
      graph.device.limits.maxBufferSize
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'fixed-point sums',
      binCount * FIXED_POINT_WORD_COUNT * 4,
      'reduce cellCount * periodCount'
    );
    const counts = output.counts ?? createTransientView(graph, `${id}-counts`, 'uint32', binCount);
    const sums =
      output.sums ??
      (output.means ? createTransientView(graph, `${id}-sums`, 'float32', binCount) : undefined);
    const scale = createTransientView(graph, `${id}-scale`, 'uint32', 1);
    const words = createTransientView(
      graph,
      `${id}-fixed-point`,
      'uint32',
      binCount * FIXED_POINT_WORD_COUNT
    );
    const invalidCount = output.status?.invalidCount;
    const constants = {CELL_COUNT: cellCount, PERIOD_COUNT: periodCount};
    // A row is binned when its cell and period are in range and its value is finite. The same
    // predicate sets the scale and accumulates, so the unit follows exactly the binned rows.
    const declarations = `override CELL_COUNT: u32;
override PERIOD_COUNT: u32;
fn isRowBinned(cell: u32, period: u32, value: f32) -> bool {
  return cell < CELL_COUNT && period < PERIOD_COUNT && ((bitcast<u32>(value) >> 23u) & 0xffu) != 0xffu;
}`;
    const maskCondition = mask ? 'mask[maskOffset + index] != 0u' : 'true';
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const inputBindings = [
      read('cells', cells, 'u32'),
      read('periods', periods, 'u32'),
      read('values', values, 'f32'),
      ...(mask ? [read('mask', mask, 'u32')] : [])
    ];
    const nodes: GPUCommandNode<Parameters>[] = [
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-scale`,
        operation: OPERATION,
        view: scale,
        type: 'u32',
        value: '0u'
      }),
      createFixedPointClearNode<Parameters>(graph, {
        id: `${id}-clear-words`,
        operation: OPERATION,
        words
      })
    ];
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-counts`,
        operation: OPERATION,
        view: counts,
        type: 'u32',
        value: '0u'
      })
    );
    if (invalidCount) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-invalid-count`,
          operation: OPERATION,
          view: invalidCount,
          type: 'u32',
          value: '0u'
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scale-pass`,
        operation: OPERATION,
        variant: 'scale',
        bindings: [
          ...inputBindings,
          {name: 'scale', view: scale, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: rowCount,
        workgroupSize: SCALE_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${declarations}
${FIXED_POINT_WGSL}
var<workgroup> workgroupExponentBits: atomic<u32>;`,
        constants,
        // No early return: every invocation of a workgroup must reach the barriers.
        body: `if (localInvocationIndex == 0u) {
    atomicStore(&workgroupExponentBits, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let period = periods[periodsOffset + index];
    let value = values[valuesOffset + index];
    if (period != 0xffffffffu && ${maskCondition} && isRowBinned(cells[cellsOffset + index], period, value)) {
      let exponentBits = getFixedPointExponentBits(value);
      if (exponentBits != 0u) {
        atomicMax(&workgroupExponentBits, exponentBits);
      }
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let workgroupMaximum = atomicLoad(&workgroupExponentBits);
    if (workgroupMaximum != 0u) {
      atomicMax(&scale[scaleOffset], workgroupMaximum);
    }
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-accumulate`,
        operation: OPERATION,
        variant: 'accumulate',
        bindings: [
          ...inputBindings,
          read('scale', scale, 'u32'),
          {name: 'countsOut', view: counts, type: 'atomic<u32>', access: 'read_write'},
          {name: 'fixedWords', view: words, type: 'atomic<u32>', access: 'read_write'},
          ...(invalidCount
            ? [
                {
                  name: 'invalidCountOut',
                  view: invalidCount,
                  type: 'atomic<u32>' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: rowCount,
        declarations: `${declarations}
override FIXED_POINT_HEADROOM_BITS: u32;
${FIXED_POINT_WGSL}
${getFixedPointAtomicAddWGSL('fixedWords')}`,
        constants: {
          ...constants,
          FIXED_POINT_HEADROOM_BITS: getFixedPointHeadroomBits(rowCount)
        },
        body: `let period = periods[periodsOffset + index];
  if (period == 0xffffffffu || !(${maskCondition})) {
    return;
  }
  let cell = cells[cellsOffset + index];
  let value = values[valuesOffset + index];
  if (!isRowBinned(cell, period, value)) {${
    invalidCount ? '\n    atomicAdd(&invalidCountOut[invalidCountOutOffset], 1u);' : ''
  }
    return;
  }
  let bin = cell * PERIOD_COUNT + period;
  let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], FIXED_POINT_HEADROOM_BITS);
  atomicAdd(&countsOut[countsOutOffset + bin], 1u);
  addFixedPointToFixedWords(bin, quantizeFixedPoint(value, unitExponent));`
      })
    );
    if (sums) {
      nodes.push(
        createFixedPointDecodeNode<Parameters>(graph, {
          id: `${id}-decode`,
          operation: OPERATION,
          words,
          scale,
          maximumTermCount: rowCount,
          output: sums
        })
      );
    }
    if (output.means && sums) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-means`,
          operation: OPERATION,
          variant: 'means',
          // Retained outputs may share one buffer with means; all co-bound ranges are writable.
          bindings: [
            {name: 'countsIn', view: counts, type: 'u32', access: 'read_write'},
            {name: 'sumsIn', view: sums, type: 'f32', access: 'read_write'},
            {name: 'meansOut', view: output.means, type: 'f32', access: 'read_write'}
          ],
          invocationCount: binCount,
          declarations: GET_NAN_WGSL,
          body: `let count = countsIn[countsInOffset + index];
  meansOut[meansOutOffset + index] = select(getNaN(), sumsIn[sumsInOffset + index] / f32(count), count > 0u);`
        })
      );
    }
    return nodes;
  }

  private getInputViews(): GraphDataView[] {
    const {cells, periods, values, mask} = this.props;
    return [cells, periods, values, mask].filter(view => Boolean(view)) as GraphDataView[];
  }

  private getOutputViews(): GraphDataView[] {
    const {output} = this.props;
    return [output.counts, output.sums, output.means, output.status?.invalidCount].filter(view =>
      Boolean(view)
    ) as GraphDataView[];
  }
}
