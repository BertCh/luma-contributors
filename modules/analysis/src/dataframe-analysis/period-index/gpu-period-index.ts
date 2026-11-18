// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGraphChunkGroupKernelNode,
  validateGraphChunkGroupKernelSupport
} from '../../analysis-core/kernels/graph-chunk-groups';

import {
  getBoundedDispatchLayout,
  GraphDataView,
  type GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  resolveGPUTemporalContext,
  type GPUTemporalCalendarStep,
  type GPUTemporalContext,
  type GPUTemporalContextSupport
} from '../../analysis-core/contracts/temporal-context';
import {
  validateGPUStatusPort,
  type GPUInvalidStatusPort
} from '../../analysis-core/contracts/status';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  getGraphViewChunks,
  validateCommandNodeBudget,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  createFillNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {TEMPORAL_REDUCTION_WORD_DIVISION_WGSL} from '../temporal-reduction/temporal-reduction-word-division';
import {TIME_WORDS_WGSL} from '../time-window-filter/time-words';
import {UINT64_WGSL} from '../../analysis-core/kernels/uint64-wgsl';
import {
  getGPUPeriodIndexStepIndex,
  getGPUPeriodIndexStepsPerPeriod,
  validateGPUPeriodIndexStepsPerPeriod,
  validateGPUPeriodIndexOriginStep,
  type GPUPeriodIndexCalendarFields
} from './period-index-calendar';

const OPERATION = 'GPUPeriodIndex';
/** Fixed workgroup size keeps all row chunks on the same pipeline specialization. */
const WORKGROUP_SIZE = 256;
/** The calendar columns are decoded from Unix-epoch Int64 words, and fixed widths count from the Unix epoch. */
const TEMPORAL_CONTEXT_SUPPORT: GPUTemporalContextSupport = Object.freeze({
  encodings: Object.freeze(['int64-words'] as const),
  unixEpoch: true
});
const CALENDAR_UNITS = ['year', 'quarter', 'month', 'week', 'day', 'hour'] as const;
const INT64_MAXIMUM = (1n << 63n) - 1n;
const INT64_MINIMUM = -(1n << 63n);

/** Period written for a row with no usable time: masked, or an invalid calendar decode. */
export const GPU_PERIOD_INDEX_INVALID = 0xffffffff;
/** Period written for a valid time before the origin or past the last period. */
export const GPU_PERIOD_INDEX_OUT_OF_RANGE = 0xfffffffe;
/** Largest `count`: the two sentinels above are not periods. */
export const GPU_PERIOD_INDEX_MAXIMUM_COUNT = 0xfffffffd;

/** Preserves packed row types by default and the caller's chunked view kind when requested. */
type GPUPeriodIndexRowView<
  Format extends GraphDataView['format'],
  Rows extends GraphDataView | GraphVectorView
> = Rows extends GraphVectorView ? GraphVectorView<Format> : GraphDataView<Format>;

/**
 * Calendar columns of the row times, the outputs of `GPUCalendarBuckets`. A step needs `year`;
 * `month` for every step but `'year'`; `dayOfMonth` for `'week'`, `'day'` and `'hour'`; `hour` for
 * `'hour'`. Extra columns are ignored.
 */
export type GPUPeriodIndexCalendarColumns<
  Rows extends GraphDataView | GraphVectorView = GraphDataView
> = {
  /** Year, `-2^31` for an invalid row. */
  year: GPUPeriodIndexRowView<'sint32', Rows>;
  month?: GPUPeriodIndexRowView<'uint32', Rows>;
  dayOfMonth?: GPUPeriodIndexRowView<'uint32', Rows>;
  hour?: GPUPeriodIndexRowView<'uint32', Rows>;
};

/**
 * Periods that are whole calendar steps. Period `floor((stepIndex(t) - stepIndex(origin)) /
 * stepsPerPeriod)` holds the local time `t` (see `getGPUPeriodIndexStepIndex`), where a `'week'` is
 * 7 days anchored to the origin day.
 */
export type GPUPeriodIndexCalendarPeriods = {
  kind: 'calendar';
  /**
   * `calendarUnit` and a positive integer `interval` (default 1). The divisor in whole steps
   * (seven days per week) must not exceed MAXIMUM_CALENDAR_STEPS_PER_PERIOD (2^31 - 1).
   */
  step: GPUTemporalCalendarStep;
  /** Start of period 0; its whole-step index must fit MINIMUM_CALENDAR_STEP_INDEX..MAXIMUM_CALENDAR_STEP_INDEX (i32). */
  origin: GPUPeriodIndexCalendarFields;
  /** Number of periods. */
  count: number;
};

/**
 * Periods of a fixed width in the ticks of `temporalContext.unit`: period
 * `floor((time - origin) / width)`, computed exactly on the Int64 words.
 */
export type GPUPeriodIndexFixedPeriods = {
  kind: 'fixed';
  /** Width in ticks, a positive integer below 2^63. */
  width: number | bigint;
  /** Start of period 0 in ticks after the Unix epoch. */
  origin: bigint;
  /** Number of periods. */
  count: number;
};

/**
 * Caller-owned outputs of {@link GPUPeriodIndex}.
 *
 * `periods` has one `uint32` row per input row: the period, `GPU_PERIOD_INDEX_OUT_OF_RANGE` for a
 * valid time outside the periods, or `GPU_PERIOD_INDEX_INVALID` for a masked or undecodable row.
 */
export type GPUPeriodIndexOutput<Rows extends GraphDataView | GraphVectorView = GraphDataView> = {
  periods: GPUPeriodIndexRowView<'uint32', Rows>;
  /**
   * Cleared on every encoding. `invalidCount` is the number of skipped rows: valid times that fall
   * before the origin or past the last period. Masked and undecodable rows are not counted, they
   * carry `GPU_PERIOD_INDEX_INVALID`.
   */
  status?: GPUInvalidStatusPort;
};

/**
 * Properties for {@link GPUPeriodIndex}.
 *
 * Per-frame (no recompile): the contents of `columns` or `timestamps` and of `mask`. Topology
 * (needs a new graph): the row count, `periods` (kind, step, interval, origin, width, count),
 * which calendar columns are present, whether `mask` and the status are present. The period axis
 * is compile-time because it fixes the size of the cube the periods index.
 */
export type GPUPeriodIndexProps<Rows extends GraphDataView | GraphVectorView = GraphDataView> = {
  /** Prefix for generated node IDs. Defaults to `'period-index'`. */
  id?: string;
  /**
   * Temporal context of the row times, for example the `temporalContext` of an Arrow upload or of
   * the `GPUCalendarBuckets` input. It must be `'int64-words'` counted from the Unix epoch. Its
   * `unit` is the unit of a fixed `width` and `origin`.
   */
  temporalContext: GPUTemporalContext;
  /** The period axis: calendar steps or a fixed width. */
  periods: GPUPeriodIndexCalendarPeriods | GPUPeriodIndexFixedPeriods;
  /** Calendar columns, required by (and only read for) `periods.kind === 'calendar'`. */
  columns?: GPUPeriodIndexCalendarColumns<Rows>;
  /** Int64 words, `(low, high)` per row, required by (and only read for) `periods.kind === 'fixed'`. */
  timestamps?: GPUPeriodIndexRowView<'uint32x2', Rows>;
  /** Optional packed `uint32` row mask; zero writes `GPU_PERIOD_INDEX_INVALID`. */
  mask?: GPUPeriodIndexRowView<'uint32', Rows>;
  /** Caller-owned outputs. */
  output: GPUPeriodIndexOutput<Rows>;
};

/**
 * Period index per row: which regular calendar period or fixed-width bin a time falls in, as a
 * `uint32` that a cell-major cube (`cell * periodCount + period`) or `GPUTemporalReduction`-style
 * binning can use directly. It is the adapter that `addSpaceTimeHotSpotsRecipe` runs between
 * `GPUCalendarBuckets` and its cube, extracted so other space-time pipelines share it.
 *
 * Calendar periods read the columns of `GPUCalendarBuckets` and count whole steps (year, quarter,
 * month, week, day, hour; a multiple with `interval`) from a calendar origin, in integer
 * arithmetic (Hinnant's `days_from_civil`), so the result is exact and identical on every adapter.
 * Fixed periods divide the Int64 word difference to the origin by the width on the words, exact for
 * every width below 2^63 and every time, including times before 1970.
 *
 * Rows outside `[0, count)` periods are skipped, not clamped: their period is
 * `GPU_PERIOD_INDEX_OUT_OF_RANGE`, and the optional `status.invalidCount` counts them with an
 * integer `atomicAdd`.
 *
 * Limits: `count` at most `GPU_PERIOD_INDEX_MAXIMUM_COUNT`; the `uint32x2` timestamps of a fixed
 * axis must fit one storage binding per chunk. Row inputs and outputs may be chunked with
 * matching ordered chunk lengths; total rows fit the shared uint32 chunk limit. Status stays packed.
 */
export class GPUPeriodIndex<
  Props extends GPUPeriodIndexProps<GraphDataView | GraphVectorView> = GPUPeriodIndexProps
> implements GPUCommandNodeProducer
{
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: Props & GPUPeriodIndexProps<Props['output']['periods']>;
  /** Number of input rows. */
  readonly rowCount: number;
  /** Validated temporal context of the row times. */
  readonly temporalContext: GPUTemporalContext;

  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;

  constructor(props: Props) {
    this.id = props.id ?? 'period-index';
    this.props = props as Props & GPUPeriodIndexProps<Props['output']['periods']>;
    const id = this.id;
    const {periods, output} = props;
    this.temporalContext = resolveGPUTemporalContext(
      id,
      props.temporalContext,
      TEMPORAL_CONTEXT_SUPPORT
    );
    validateCapacity(id, 'periods.count', periods.count, 1, GPU_PERIOD_INDEX_MAXIMUM_COUNT);
    if (periods.kind === 'calendar') {
      this.rowCount = this.validateCalendar(periods);
    } else {
      this.rowCount = this.validateFixed(periods);
    }
    if (this.rowCount < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (props.mask) {
      for (const chunk of getGraphViewChunks(props.mask))
        validateSinglePackedView(chunk, ['uint32'], `${id} mask`);
      validateViewLength(id, 'mask', props.mask, this.rowCount);
    }
    for (const chunk of getGraphViewChunks(output.periods))
      validateSinglePackedView(chunk, ['uint32'], `${id} output.periods`);
    validateRequirement(
      id,
      output.periods.length >= this.rowCount,
      'output.periods must hold one row per input row'
    );
    validateGPUStatusPort(id, {invalidCount: output.status?.invalidCount});
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
    const periodRows =
      output.periods instanceof GraphDataView && output.periods.length > this.rowCount
        ? new GraphDataView(output.periods.buffer, {...output.periods, length: this.rowCount})
        : output.periods;
    validateDisjointWritableViews(id, {
      periods: periodRows,
      invalidCount: output.status?.invalidCount
    });
    const calendarUnit = periods.kind === 'calendar' ? periods.step.calendarUnit : undefined;
    this.chunkPlan = planGraphChunkDispatch(id, {
      ...(periods.kind === 'fixed'
        ? {timestamps: props.timestamps}
        : {
            calendarYear: props.columns!.year,
            calendarMonth: calendarUnit !== 'year' ? props.columns!.month : undefined,
            calendarDay: ['week', 'day', 'hour'].includes(calendarUnit!)
              ? props.columns!.dayOfMonth
              : undefined,
            calendarHour: calendarUnit === 'hour' ? props.columns!.hour : undefined
          }),
      rowMask: props.mask,
      periodsOut: periodRows
    });
    this.estimatedCommandNodeCount =
      Number(Boolean(output.status?.invalidCount)) + this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  /** Returns the optional status clear, then the one period kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, chunkPlan} = this;
    const {periods, output} = props;
    validateGraphChunkDispatchBindings(graph, id, chunkPlan);
    validateGraphChunkBindingRanges(graph, id, {invalidCount: output.status?.invalidCount});
    validateGraphChunkGroupKernelSupport(
      graph,
      id,
      1 +
        Number(Boolean(props.mask)) +
        Number(Boolean(output.status?.invalidCount)) +
        (periods.kind === 'fixed'
          ? 1
          : 1 +
            Number(periods.step.calendarUnit !== 'year') +
            Number(['week', 'day', 'hour'].includes(periods.step.calendarUnit)) +
            Number(periods.step.calendarUnit === 'hour')),
      WORKGROUP_SIZE
    );
    for (const chunk of chunkPlan.chunks) {
      if (chunk.length > 0)
        getBoundedDispatchLayout(
          id,
          chunk.length,
          WORKGROUP_SIZE,
          graph.device.limits.maxComputeWorkgroupsPerDimension
        );
    }

    const nodes: GPUCommandNode<Parameters>[] = [];
    const invalidCount = output.status?.invalidCount;
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
    const read = (name: string, view: GraphDataView, type: 'u32' | 'i32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    for (const chunk of chunkPlan.chunks) {
      if (chunk.length === 0) continue;
      const bindings: WGSLKernelBinding[] = [];
      let declarations: string;
      let periodSource: string;
      if (periods.kind === 'calendar') {
        const {calendarUnit} = periods.step;
        const needsMonth = calendarUnit !== 'year';
        const needsDay = ['week', 'day', 'hour'].includes(calendarUnit);
        const needsHour = calendarUnit === 'hour';
        bindings.push(read('calendarYear', chunk.views['calendarYear'], 'i32'));
        if (needsMonth) {
          bindings.push(read('calendarMonth', chunk.views['calendarMonth'], 'u32'));
        }
        if (needsDay) {
          bindings.push(read('calendarDay', chunk.views['calendarDay'], 'u32'));
        }
        if (needsHour) {
          bindings.push(read('calendarHour', chunk.views['calendarHour'], 'u32'));
        }
        const stepExpression = {
          year: 'year',
          quarter: 'year * 4 + (month - 1) / 3',
          month: 'year * 12 + month - 1',
          week: 'daysFromCivil(year, month, day)',
          day: 'daysFromCivil(year, month, day)',
          hour: ''
        }[calendarUnit];
        const stepWordsExpression = needsHour
          ? 'addUint64(multiplyUint64ByUint32(calendarStepWords(daysFromCivil(year, month, day)), 24u), vec2<u32>(0u, u32(hour)))'
          : `calendarStepWords(${stepExpression})`;
        const originStep = getGPUPeriodIndexStepIndex(calendarUnit, periods.origin);
        const [originLow, originHigh] = getWords(BigInt(originStep));
        declarations = `${CALENDAR_WGSL}\n${UINT64_WGSL}
const ORIGIN_STEP_WORDS = vec2<u32>(${originHigh}u, ${originLow}u);`;
        periodSource = `let year = calendarYear[calendarYearBase + index];
  ${needsMonth ? 'let month = i32(calendarMonth[calendarMonthBase + index]);' : ''}
  ${needsDay ? 'let day = i32(calendarDay[calendarDayBase + index]);' : ''}
  ${needsHour ? 'let hour = i32(calendarHour[calendarHourBase + index]);' : ''}
  let isTimeValid = year != bitcast<i32>(0x80000000u);
  var period = ${GPU_PERIOD_INDEX_INVALID}u;
  if (isTimeValid${props.mask ? ' && rowMask[rowMaskBase + index] != 0u' : ''}) {
    let stepWords = ${stepWordsExpression};
    period = ${GPU_PERIOD_INDEX_OUT_OF_RANGE}u;
    if (compareInt64(stepWords, ORIGIN_STEP_WORDS) >= 0i) {
      let stepOffset = subtractUint64(stepWords, ORIGIN_STEP_WORDS);
      let division = divideUint64ByUint32(stepOffset, ${getGPUPeriodIndexStepsPerPeriod(periods.step)}u);
      let candidate = division.quotient;
      if (candidate.x == 0u && candidate.y < ${periods.count}u) {
        period = candidate.y;
      }
    }
  }`;
      } else {
        const [originLow, originHigh] = getWords(periods.origin);
        const [widthLow, widthHigh] = getWords(BigInt(periods.width));
        bindings.push(read('timestamps', chunk.views['timestamps'], 'u32'));
        declarations = `${TIME_WORDS_WGSL}\n${TEMPORAL_REDUCTION_WORD_DIVISION_WGSL}
const ORIGIN_WORDS = vec2<u32>(${originLow}u, ${originHigh}u);
const WIDTH_WORDS = vec2<u32>(${widthLow}u, ${widthHigh}u);`;
        periodSource = `let time = vec2<u32>(timestamps[timestampsBase + 2u * index], timestamps[timestampsBase + 2u * index + 1u]);
  var period = ${GPU_PERIOD_INDEX_INVALID}u;
  if (${props.mask ? 'rowMask[rowMaskBase + index] != 0u' : 'true'}) {
    period = ${GPU_PERIOD_INDEX_OUT_OF_RANGE}u;
    if (timeWordsCompare(time, ORIGIN_WORDS) >= 0i) {
      // time >= origin, so the unsigned 64-bit difference is exact even beyond 2^63.
      let division = divideTimeDifference(timeWordsSubtract(time, ORIGIN_WORDS), WIDTH_WORDS);
      if (!division.overflow && division.quotient < ${periods.count}u) {
        period = division.quotient;
      }
    }
  }`;
      }
      if (props.mask) {
        bindings.push(read('rowMask', chunk.views['rowMask'], 'u32'));
      }
      bindings.push({
        name: 'periodsOut',
        view: chunk.views['periodsOut'],
        type: 'u32',
        access: 'read_write'
      });
      if (invalidCount) {
        bindings.push({
          name: 'invalidCountOut',
          view: invalidCount,
          type: 'atomic<u32>',
          access: 'read_write'
        });
      }
      nodes.push(
        createGraphChunkGroupKernelNode<Parameters>(graph, {
          id: `${id}-periods-chunk-${chunk.chunkIndex}`,
          operation: OPERATION,
          variant: periods.kind,
          bindings: bindings,
          invocationCount: chunk.length,
          workgroupSize: WORKGROUP_SIZE,
          declarations,
          body: `if (index >= getChunkParameter(1u)) { return; }
  ${bindings
    .filter(binding => !['invalidCountOut', 'chunkParameters'].includes(binding.name))
    .map(
      binding =>
        `let ${binding.name}Base = arrayLength(&${binding.name}) - ${binding.name === 'timestamps' ? '2u * ' : ''}getChunkParameter(1u);`
    )
    .join('\n  ')}
  ${periodSource}
  periodsOut[periodsOutBase + index] = period;${
    invalidCount
      ? `
  if (period == ${GPU_PERIOD_INDEX_OUT_OF_RANGE}u) {
    atomicAdd(&invalidCountOut[invalidCountOutOffset], 1u);
  }`
      : ''
  }`
        })
      );
    }
    return nodes;
  }

  private validateCalendar(periods: GPUPeriodIndexCalendarPeriods): number {
    const {id} = this;
    const {columns} = this.props;
    const {calendarUnit, interval = 1} = periods.step;
    validateRequirement(
      id,
      CALENDAR_UNITS.includes(calendarUnit) && Number.isInteger(interval) && interval >= 1,
      'periods.step needs a calendar unit and a positive integer interval'
    );
    validateGPUPeriodIndexStepsPerPeriod(id, periods.step);
    validateRequirement(id, columns !== undefined, 'calendar periods need columns');
    const required = {
      year: true,
      month: calendarUnit !== 'year',
      dayOfMonth: ['week', 'day', 'hour'].includes(calendarUnit),
      hour: calendarUnit === 'hour'
    };
    for (const chunk of getGraphViewChunks(columns!.year))
      validateSinglePackedView(chunk, ['sint32'], `${id} columns.year`);
    const rowCount = columns!.year.length;
    for (const name of ['month', 'dayOfMonth', 'hour'] as const) {
      const view: GraphDataView | GraphVectorView | undefined = columns![name];
      if (required[name]) {
        validateRequirement(
          id,
          view !== undefined,
          `columns.${name} is required for a ${calendarUnit} step`
        );
        for (const chunk of getGraphViewChunks(view!))
          validateSinglePackedView(chunk, ['uint32'], `${id} columns.${name}`);
        validateViewLength(id, `columns.${name}`, view!, rowCount);
      }
    }
    const {origin} = periods;
    for (const [name, value, minimum, maximum] of [
      ['year', origin.year, -1_000_000, 1_000_000],
      ['month', origin.month ?? 1, 1, 12],
      ['day', origin.day ?? 1, 1, 31],
      ['hour', origin.hour ?? 0, 0, 23]
    ] as const) {
      validateRequirement(
        id,
        Number.isInteger(value) && value >= minimum && value <= maximum,
        `periods.origin.${name} must be an integer in ${minimum}..${maximum}`
      );
    }
    validateGPUPeriodIndexOriginStep(id, calendarUnit, origin);
    return rowCount;
  }

  private validateFixed(periods: GPUPeriodIndexFixedPeriods): number {
    const {id} = this;
    const {timestamps} = this.props;
    const {width, origin} = periods;
    validateRequirement(id, timestamps !== undefined, 'fixed periods need timestamps');
    for (const chunk of getGraphViewChunks(timestamps!))
      validateSinglePackedView(chunk, ['uint32x2'], `${id} timestamps`);
    validateRequirement(
      id,
      (typeof width === 'bigint' || Number.isSafeInteger(width)) &&
        BigInt(width) >= 1n &&
        BigInt(width) <= INT64_MAXIMUM,
      'periods.width must be a positive integer below 2^63'
    );
    validateRequirement(
      id,
      typeof origin === 'bigint' && origin >= INT64_MINIMUM && origin <= INT64_MAXIMUM,
      'periods.origin must be a signed Int64 bigint'
    );
    return timestamps!.length;
  }

  private getInputViews(): (GraphDataView | GraphVectorView)[] {
    const {props} = this;
    const {columns} = props;
    return [
      props.timestamps,
      props.mask,
      columns?.year,
      columns?.month,
      columns?.dayOfMonth,
      columns?.hour
    ].filter(view => Boolean(view)) as (GraphDataView | GraphVectorView)[];
  }

  private getOutputViews(): (GraphDataView | GraphVectorView)[] {
    const {output} = this.props;
    return [output.periods, output.status?.invalidCount].filter(view => Boolean(view)) as (
      | GraphDataView
      | GraphVectorView
    )[];
  }
}

/** `(low, high)` words of a signed Int64 as unsigned 32-bit numbers. */
function getWords(value: bigint): [number, number] {
  const unsigned = BigInt.asUintN(64, value);
  return [Number(unsigned & 0xffffffffn), Number(unsigned >> 32n)];
}

// Hinnant's days_from_civil in i32 arithmetic with truncating division.
const CALENDAR_WGSL = /* wgsl */ `fn daysFromCivil(year: i32, month: i32, day: i32) -> i32 {
  let adjustedYear = select(year, year - 1, month <= 2);
  let era = select(adjustedYear, adjustedYear - 399, adjustedYear < 0) / 400;
  let yearOfEra = adjustedYear - era * 400;
  let shiftedMonth = (month + 9) % 12;
  let dayOfYear = (153 * shiftedMonth + 2) / 5 + day - 1;
  let dayOfEra = yearOfEra * 365 + yearOfEra / 4 - yearOfEra / 100 + dayOfYear;
  return era * 146097 + dayOfEra - 719468;
}

// Shared uint64 arithmetic uses (high, low); sign-extend before multiplying hours or subtracting.
fn calendarStepWords(step: i32) -> vec2<u32> {
  return vec2<u32>(select(0u, 0xffffffffu, step < 0), bitcast<u32>(step));
}`;
