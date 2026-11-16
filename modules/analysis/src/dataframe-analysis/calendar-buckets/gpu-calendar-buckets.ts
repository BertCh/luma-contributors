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
  GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  resolveGPUTemporalContext,
  type GPUTemporalContext,
  type GPUTemporalContextSupport
} from '../../analysis-core/contracts/temporal-context';
import type {ArrowAnalyticsTimeUnit} from '../../analysis-core/graph-support/arrow-analytics-normalization';
import {
  getGraphViewChunks,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  planGraphChunkDispatch,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan,
  validateDisjointWritableViews
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  GPU_CALENDAR_BUCKETS_MATRIX_LENGTH,
  GPU_CALENDAR_BUCKETS_MAXIMUM_DAYS,
  GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH
} from './calendar-buckets-parameters';

const OPERATION = 'GPUCalendarBuckets';
/** Storage bindings a kernel may use on the WebGPU default limit. */
const MAXIMUM_BINDINGS = 8;
/** Fixed workgroup size keeps every row chunk on the same pipeline specialization. */
const WORKGROUP_SIZE = 256;
/** The decode reads Int64 words and turns them into civil dates, so it needs a Unix-epoch count. */
const TEMPORAL_CONTEXT_SUPPORT: GPUTemporalContextSupport = Object.freeze({
  encodings: Object.freeze(['int64-words'] as const),
  unixEpoch: true
});

/** Preserves packed row types by default and the caller's chunked view kind when requested. */
type GPUCalendarBucketsRowView<
  Format extends GraphDataView['format'],
  Rows extends GraphDataView | GraphVectorView
> = Rows extends GraphVectorView ? GraphVectorView<Format> : GraphDataView<Format>;

/**
 * Caller-owned output columns of {@link GPUCalendarBuckets}. Each is optional and which ones are
 * present is compile-time, but at least one must be given. Every column holds one row per input
 * row; a masked or invalid row writes `0xffffffff` to `uint32` columns and `-2^31` to `year` and
 * `isoWeekYear`.
 */
export type GPUCalendarBucketsOutput<Rows extends GraphDataView | GraphVectorView = GraphDataView> =
  {
    /** Proleptic Gregorian year of the local time, astronomical numbering (year 0 exists). */
    year?: GPUCalendarBucketsRowView<'sint32', Rows>;
    /** Month, 1 to 12. */
    month?: GPUCalendarBucketsRowView<'uint32', Rows>;
    /** Day of the month, 1 to 31. */
    dayOfMonth?: GPUCalendarBucketsRowView<'uint32', Rows>;
    /** Hour, 0 to 23. */
    hour?: GPUCalendarBucketsRowView<'uint32', Rows>;
    /** Minute, 0 to 59. */
    minute?: GPUCalendarBucketsRowView<'uint32', Rows>;
    /** Weekday, 0 to 6, where 0 is the `firstDayOfWeek` parameter (Monday by default). */
    weekday?: GPUCalendarBucketsRowView<'uint32', Rows>;
    /** Day of the year, 1 to 366. */
    dayOfYear?: GPUCalendarBucketsRowView<'uint32', Rows>;
    /** ISO 8601 week number, 1 to 53 (weeks start on Monday, week 1 contains the first Thursday). */
    isoWeek?: GPUCalendarBucketsRowView<'uint32', Rows>;
    /** ISO 8601 week-based year, which differs from `year` around January 1. */
    isoWeekYear?: GPUCalendarBucketsRowView<'sint32', Rows>;
    /** Quarter of the year, 1 to 4. */
    quarter?: GPUCalendarBucketsRowView<'uint32', Rows>;
    /**
     * Hour by weekday count matrix, `weekday * 24 + hour`, at least 168 elements. It is cleared on
     * every encoding, then every valid row adds one. The counts are exact integers.
     */
    hourWeekdayCounts?: GraphDataView<'uint32'>;
  };

/**
 * Time zone transition table read by {@link GPUCalendarBuckets}; build it with
 * `getGPUTimeZoneTransitionTable`. Its length is topology: a table of another length needs a new
 * graph, its contents can change per frame.
 */
export type GPUCalendarBucketsTimeZone = {
  /**
   * Strictly increasing Int64 epoch-millisecond instants, one `uint32x2` `(low, high)` row each
   * (the format of `timestamps`), at least one.
   */
  transitions: GraphDataView<'uint32x2'>;
  /**
   * UTC offset in seconds (`sint32`, within `[-86400, 86400]`), `transitions.length + 1` long:
   * `offsets[0]` applies before the first transition and `offsets[i + 1]` from transition `i` on.
   * Seconds keep pre-standardization local mean time such as +00:19:32 exact.
   */
  offsets: GraphDataView<'sint32'>;
};

/**
 * Properties for {@link GPUCalendarBuckets}.
 *
 * Per-frame (no recompile): the contents of `parameters` and of every input buffer. Topology
 * (needs a new graph): the row count, the unit, which outputs are present, and whether `mask`,
 * `utcOffsets` and `timeZone` are present, and the length of the `timeZone` table.
 */
export type GPUCalendarBucketsProps<Rows extends GraphDataView | GraphVectorView = GraphDataView> =
  {
    /** Prefix for generated node IDs. Defaults to `'calendar-buckets'`. */
    id?: string;
    /**
     * Epoch times as signed 64-bit two's complement words, one `uint32x2` `(low, high)` row per row
     * (for example from `getInt64TimeWords`), in the unit given by `unit`. Times before 1970 are
     * negative.
     */
    timestamps: GPUCalendarBucketsRowView<'uint32x2', Rows>;
    /**
     * Unit of `timestamps`: `'second'`, `'millisecond'`, `'microsecond'` or `'nanosecond'`, the unit
     * of an Arrow timestamp column. Defaults to `'millisecond'`. Compile-time (a new graph). Times
     * are floored to whole milliseconds before the calendar decode, so a negative sub-millisecond
     * time lands in the earlier millisecond. The time zone `transitions` stay epoch milliseconds.
     */
    unit?: ArrowAnalyticsTimeUnit;
    /**
     * Temporal context of `timestamps`, for example the `temporalContext` of an Arrow upload. Supplies
     * the unit. It must be `'int64-words'` counted from the Unix epoch, and an explicit `unit` must
     * agree with it. Its `timeZone` is not applied: local time comes from `parameters`, `utcOffsets`
     * or `timeZone` below.
     */
    temporalContext?: GPUTemporalContext;
    /** Optional packed `uint32` row mask; zero makes the row invalid. */
    mask?: GPUCalendarBucketsRowView<'uint32', Rows>;
    /**
     * Optional UTC offset in minutes per row (`sint32`, within `[-1440, 1440]`), which overrides the
     * fixed offset in `parameters`. This is the only way to express daylight saving time: the caller
     * precomputes each row's offset from its zone rules. An out-of-range offset makes the row invalid.
     */
    utcOffsets?: GPUCalendarBucketsRowView<'sint32', Rows>;
    /**
     * Optional time zone transition table. Each row binary-searches its instant in `transitions`
     * (signed 64-bit compare, the last transition at or before the row) and uses that offset, so
     * daylight saving time needs no per-row CPU offsets. Mutually exclusive with `utcOffsets`; it
     * overrides the fixed offset in `parameters`. A table offset beyond +-86400 seconds makes the
     * row invalid. Build it with `getGPUTimeZoneTransitionTable`.
     */
    timeZone?: GPUCalendarBucketsTimeZone;
    /**
     * Per-frame `sint32` view of at least 2 elements written with
     * `getGPUCalendarBucketsParameterValues`: the fixed UTC offset in minutes (a fixed offset, not a
     * zone) and the first day of the week.
     */
    parameters: GraphDataView<'sint32'>;
    /** Caller-owned outputs. At least one must be present. */
    output: GPUCalendarBucketsOutput<Rows>;
  };

type OutputSpec = {
  name: keyof GPUCalendarBucketsOutput;
  type: 'u32' | 'i32' | 'atomic<u32>';
  /** WGSL statement that writes the output for one row. */
  write: string;
};

const COLUMN_OUTPUTS: readonly {
  name: Exclude<keyof GPUCalendarBucketsOutput, 'hourWeekdayCounts'>;
  field: string;
  signed?: boolean;
}[] = [
  {name: 'year', field: 'year', signed: true},
  {name: 'month', field: 'month'},
  {name: 'dayOfMonth', field: 'day'},
  {name: 'hour', field: 'hour'},
  {name: 'minute', field: 'minute'},
  {name: 'weekday', field: 'weekday'},
  {name: 'dayOfYear', field: 'dayOfYear'},
  {name: 'isoWeek', field: 'isoWeek'},
  {name: 'isoWeekYear', field: 'isoWeekYear', signed: true},
  {name: 'quarter', field: 'quarter'}
];

/**
 * Decodes epoch timestamps (milliseconds by default, see `unit`) into calendar columns (year, month, day, hour, minute,
 * weekday, day of year, ISO week, ISO week-year, quarter) and an hour by weekday count matrix, so
 * time-of-day and weekday charts and calendar bucket keys need no per-row `Date` on the CPU.
 *
 * Time zone: local time is UTC plus an offset. The per-frame `parameters` carry one fixed offset
 * in minutes; an optional per-row `utcOffsets` column (minutes) overrides it, and an optional
 * `timeZone` transition table (offsets in seconds) is searched per row on the GPU, which is how
 * daylight saving time is supported without per-row CPU work.
 *
 * Exactness: all arithmetic is integer. The signed 64-bit millisecond count plus offset is
 * floor-divided by 86,400,000 with u32 operations (nibble-wise long division, exact for negative
 * times too), then Howard Hinnant's `civil_from_days` gives year, month and day in i32. Weekday
 * comes from the day number, day of year from the month and leap rule, and the ISO week and
 * week-year from the Thursday of the row's ISO week. Results are bit-identical on every adapter.
 * Weekdays are ISO (Monday is 0) unless `firstDayOfWeek` says otherwise; ISO week numbers always
 * start on Monday.
 *
 * Range: rows whose local day number is beyond +-1e9 days (about +-2.7 million years), whose sum with
 * the offset overflows Int64, whose per-row offset is beyond +-1440 minutes (+-86400 seconds in a table), or that are masked,
 * are invalid and get the sentinels documented on {@link GPUCalendarBucketsOutput}. The hour by
 * weekday matrix counts only valid rows, with integer `atomicAdd`, so it is deterministic.
 *
 * Because WebGPU allows 8 storage bindings per kernel, the outputs are split over as many decode
 * kernels as needed; each recomputes the (cheap) decode. Row inputs and columns may be chunked
 * with matching ordered chunk lengths. Parameters, the time zone table and the count matrix stay
 * packed. Every chunk must fit the device storage binding limit; total rows fit uint32 counts.
 */
export class GPUCalendarBuckets<
  Props extends GPUCalendarBucketsProps<GraphDataView | GraphVectorView> = GPUCalendarBucketsProps
> implements GPUCommandNodeProducer
{
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: Props & GPUCalendarBucketsProps<Props['timestamps']>;
  /** Number of input rows. */
  readonly rowCount: number;
  /** Unit of `timestamps`, from `temporalContext` or `unit`. */
  readonly unit: ArrowAnalyticsTimeUnit;

  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;

  constructor(props: Props) {
    this.id = props.id ?? 'calendar-buckets';
    this.props = props as Props & GPUCalendarBucketsProps<Props['timestamps']>;
    const id = this.id;
    const {output} = props;
    this.unit = resolveCalendarUnit(id, props);
    for (const chunk of getGraphViewChunks(props.timestamps)) {
      validateSinglePackedView(chunk, ['uint32x2'], `${id} timestamps`);
    }
    this.rowCount = props.timestamps.length;
    if (this.rowCount < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (props.mask) {
      for (const chunk of getGraphViewChunks(props.mask)) {
        validateSinglePackedView(chunk, ['uint32'], `${id} mask`);
      }
      if (props.mask.length !== this.rowCount) {
        throw new Error(`${id} mask length must equal timestamps length`);
      }
    }
    if (props.utcOffsets) {
      for (const chunk of getGraphViewChunks(props.utcOffsets)) {
        validateSinglePackedView(chunk, ['sint32'], `${id} utcOffsets`);
      }
      if (props.utcOffsets.length !== this.rowCount) {
        throw new Error(`${id} utcOffsets length must equal timestamps length`);
      }
    }
    if (props.timeZone) {
      if (props.utcOffsets) {
        throw new Error(`${id} timeZone and utcOffsets are mutually exclusive`);
      }
      validateSinglePackedView(
        props.timeZone.transitions,
        ['uint32x2'],
        `${id} timeZone.transitions`
      );
      validateSinglePackedView(props.timeZone.offsets, ['sint32'], `${id} timeZone.offsets`);
      if (props.timeZone.transitions.length < 1) {
        throw new Error(`${id} timeZone needs at least one transition`);
      }
      if (props.timeZone.offsets.length !== props.timeZone.transitions.length + 1) {
        throw new Error(`${id} timeZone.offsets length must be transitions length + 1`);
      }
    }
    validateSinglePackedView(props.parameters, ['sint32'], `${id} parameters`);
    if (props.parameters.length < GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH} sint32 values`
      );
    }
    let outputCount = 0;
    for (const column of COLUMN_OUTPUTS) {
      const view = output[column.name];
      if (!view) {
        continue;
      }
      outputCount++;
      for (const chunk of getGraphViewChunks(view)) {
        validateSinglePackedView(
          chunk,
          [column.signed ? 'sint32' : 'uint32'],
          `${id} output.${column.name}`
        );
      }
      if (view.length < this.rowCount) {
        throw new Error(`${id} output.${column.name} must hold one row per timestamp`);
      }
    }
    if (output.hourWeekdayCounts) {
      outputCount++;
      validateSinglePackedView(
        output.hourWeekdayCounts,
        ['uint32'],
        `${id} output.hourWeekdayCounts`
      );
      if (output.hourWeekdayCounts.length < GPU_CALENDAR_BUCKETS_MATRIX_LENGTH) {
        throw new Error(
          `${id} output.hourWeekdayCounts must hold ${GPU_CALENDAR_BUCKETS_MATRIX_LENGTH} elements`
        );
      }
    }
    if (outputCount === 0) {
      throw new Error(`${id} needs at least one output`);
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
    // Preserve packed output capacity: only its row prefix participates in row dispatch.
    const columns = Object.fromEntries(
      COLUMN_OUTPUTS.map(({name}) => {
        const view = output[name];
        return [
          name,
          view instanceof GraphDataView && view.length > this.rowCount
            ? new GraphDataView(view.buffer, {...view, length: this.rowCount})
            : view
        ];
      })
    );
    this.chunkPlan = planGraphChunkDispatch(id, {
      timestamps: props.timestamps,
      mask: props.mask,
      utcOffsets: props.utcOffsets,
      ...columns
    });
    const inputBindingCount =
      2 +
      Number(Boolean(props.mask)) +
      Number(Boolean(props.utcOffsets)) +
      (props.timeZone ? 2 : 0);
    const packedMatrix = output.hourWeekdayCounts;
    const matrixWrites = packedMatrix
      ? new GraphDataView(packedMatrix.buffer, {
          ...packedMatrix,
          length: GPU_CALENDAR_BUCKETS_MATRIX_LENGTH
        })
      : undefined;
    const writableFields = [
      ...COLUMN_OUTPUTS.filter(column => output[column.name]).map(column => columns[column.name]),
      ...(packedMatrix ? [packedMatrix] : [])
    ];
    const dispatches = [];
    for (
      let first = 0;
      first < writableFields.length;
      first += MAXIMUM_BINDINGS - inputBindingCount
    ) {
      dispatches.push(writableFields.slice(first, first + MAXIMUM_BINDINGS - inputBindingCount));
    }
    {
      const writableOutputs: Readonly<Record<string, GraphDataView | GraphVectorView | undefined>> =
        {...columns, hourWeekdayCounts: matrixWrites};
      const publicationPasses: readonly (readonly (
        | GraphDataView
        | GraphVectorView
        | undefined
      )[])[] = dispatches;
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
    this.estimatedCommandNodeCount =
      Number(Boolean(output.hourWeekdayCounts)) +
      this.chunkPlan.nonemptyChunkCount *
        Math.ceil(outputCount / (MAXIMUM_BINDINGS - inputBindingCount));
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  /** Returns the optional fill node of the matrix, then the decode kernels in output order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, unit, chunkPlan} = this;
    const {output} = props;
    validateGraphChunkDispatchBindings(graph, id, chunkPlan);
    validateGraphChunkBindingRanges(graph, id, {
      parameters: props.parameters,
      transitions: props.timeZone?.transitions,
      zoneOffsets: props.timeZone?.offsets,
      hourWeekdayCounts: output.hourWeekdayCounts
    });
    validateGraphChunkGroupKernelSupport(
      graph,
      id,
      Math.min(
        MAXIMUM_BINDINGS,
        2 +
          Number(Boolean(props.mask)) +
          Number(Boolean(props.utcOffsets)) +
          (props.timeZone ? 2 : 0) +
          Object.values(output).filter(Boolean).length
      ),
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
    const specs: OutputSpec[] = [];
    for (const column of COLUMN_OUTPUTS) {
      if (output[column.name]) {
        specs.push({
          name: column.name,
          type: column.signed ? 'i32' : 'u32',
          write: `${column.name}Out[${column.name}OutBase + index] = fields.${column.field};`
        });
      }
    }
    if (output.hourWeekdayCounts) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-fill-matrix`,
          operation: OPERATION,
          view: output.hourWeekdayCounts,
          type: 'u32',
          value: '0u',
          componentCount: GPU_CALENDAR_BUCKETS_MATRIX_LENGTH
        })
      );
      specs.push({
        name: 'hourWeekdayCounts',
        type: 'atomic<u32>',
        write: `if (fields.valid) {
    atomicAdd(&hourWeekdayCountsOut[hourWeekdayCountsOutOffset + fields.weekday * 24u + fields.hour], 1u);
  }`
      });
    }

    const read = (name: string, view: GraphDataView, type: 'u32' | 'i32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    for (const chunk of chunkPlan.chunks) {
      if (chunk.length === 0) continue;
      const inputBindings: WGSLKernelBinding[] = [
        read('timestamps', chunk.views['timestamps'], 'u32'),
        read('params', props.parameters, 'i32')
      ];
      if (props.mask) {
        inputBindings.push(read('rowMask', chunk.views['mask'], 'u32'));
      }
      if (props.utcOffsets) {
        inputBindings.push(read('rowOffsets', chunk.views['utcOffsets'], 'i32'));
      }
      if (props.timeZone) {
        inputBindings.push(
          read('zoneTransitions', props.timeZone.transitions, 'u32'),
          read('zoneOffsets', props.timeZone.offsets, 'i32')
        );
      }
      const perKernel = MAXIMUM_BINDINGS - inputBindings.length;
      for (let first = 0, step = 0; first < specs.length; first += perKernel, step++) {
        const group = specs.slice(first, first + perKernel);
        nodes.push(
          createGraphChunkGroupKernelNode<Parameters>(graph, {
            id: `${id}-decode-${step}-chunk-${chunk.chunkIndex}`,
            operation: OPERATION,
            variant: 'decode',
            bindings: [
              ...inputBindings,
              ...group.map(
                (spec): WGSLKernelBinding => ({
                  name: `${spec.name}Out`,
                  view:
                    spec.name === 'hourWeekdayCounts'
                      ? output.hourWeekdayCounts!
                      : chunk.views[spec.name],
                  type: spec.type,
                  access: 'read_write'
                })
              )
            ],
            invocationCount: chunk.length,
            workgroupSize: WORKGROUP_SIZE,
            declarations: `${CALENDAR_WGSL}${UNIT_WGSL[unit]}${props.timeZone ? ZONE_WGSL : ''}`,
            body: `if (index >= getChunkParameter(1u)) { return; }
  let timestampsBase = arrayLength(&timestamps) - 2u * getChunkParameter(1u);
  ${props.mask ? 'let rowMaskBase = arrayLength(&rowMask) - getChunkParameter(1u);' : ''}
  ${props.utcOffsets ? 'let rowOffsetsBase = arrayLength(&rowOffsets) - getChunkParameter(1u);' : ''}
  ${group
    .filter(spec => spec.name !== 'hourWeekdayCounts')
    .map(spec => `let ${spec.name}OutBase = arrayLength(&${spec.name}Out) - getChunkParameter(1u);`)
    .join('\n  ')}
  let time = toEpochMilliseconds(vec2<u32>(timestamps[timestampsBase + 2u * index], timestamps[timestampsBase + 2u * index + 1u]));
  let offsetMilliseconds = ${
    props.timeZone
      ? 'getZoneOffsetMilliseconds(time)'
      : `minutesToMilliseconds(${props.utcOffsets ? 'rowOffsets[rowOffsetsBase + index]' : 'params[paramsOffset]'})`
  };
  let firstDay = u32(((params[paramsOffset + 1u] % 7) + 7) % 7);
  var fields = decodeCalendar(time, offsetMilliseconds, firstDay);
  ${props.mask ? 'if (rowMask[rowMaskBase + index] == 0u) {\n    fields = invalidCalendar();\n  }' : ''}
  ${group.map(spec => spec.write).join('\n  ')}`
          })
        );
      }
    }
    return nodes;
  }

  private getInputViews(): (GraphDataView | GraphVectorView)[] {
    const {props} = this;
    return [
      props.timestamps,
      props.mask,
      props.utcOffsets,
      props.timeZone?.transitions,
      props.timeZone?.offsets,
      props.parameters
    ].filter(view => Boolean(view)) as (GraphDataView | GraphVectorView)[];
  }

  private getOutputViews(): (GraphDataView | GraphVectorView)[] {
    return Object.values(this.props.output).filter(view => Boolean(view)) as (
      | GraphDataView
      | GraphVectorView
    )[];
  }
}

function resolveCalendarUnit(
  id: string,
  {unit, temporalContext}: GPUCalendarBucketsProps
): ArrowAnalyticsTimeUnit {
  if (!temporalContext) {
    return unit ?? 'millisecond';
  }
  const contextUnit = resolveGPUTemporalContext(id, temporalContext, TEMPORAL_CONTEXT_SUPPORT).unit;
  if (unit !== undefined && unit !== contextUnit) {
    throw new Error(`${id} unit ${unit} contradicts temporalContext unit ${contextUnit}`);
  }
  return contextUnit;
}

const CALENDAR_WGSL = /* wgsl */ `
const MILLISECONDS_PER_DAY: u32 = 86400000u;
const MAXIMUM_DAYS: u32 = ${GPU_CALENDAR_BUCKETS_MAXIMUM_DAYS}u;

struct CalendarFields {
  valid: bool,
  year: i32,
  month: u32,
  day: u32,
  hour: u32,
  minute: u32,
  weekday: u32,
  dayOfYear: u32,
  isoWeek: u32,
  isoWeekYear: i32,
  quarter: u32
}

fn invalidCalendar() -> CalendarFields {
  let invalidYear = bitcast<i32>(0x80000000u);
  return CalendarFields(false, invalidYear, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
    0xffffffffu, 0xffffffffu, 0xffffffffu, invalidYear, 0xffffffffu);
}

fn isLeapYear(year: i32) -> bool {
  return (year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0));
}

// Hinnant civil_from_days: days since 1970-01-01 to (year, month, day).
fn civilFromDays(days: i32) -> vec3<i32> {
  let z = days + 719468;
  let era = select((z - 146096) / 146097, z / 146097, z >= 0);
  let dayOfEra = z - era * 146097;
  let yearOfEra = (dayOfEra - dayOfEra / 1460 + dayOfEra / 36524 - dayOfEra / 146096) / 365;
  let dayOfMarchYear = dayOfEra - (365 * yearOfEra + yearOfEra / 4 - yearOfEra / 100);
  let monthPrime = (5 * dayOfMarchYear + 2) / 153;
  let day = dayOfMarchYear - (153 * monthPrime + 2) / 5 + 1;
  let month = select(monthPrime - 9, monthPrime + 3, monthPrime < 10);
  let year = yearOfEra + era * 400 + select(0, 1, month <= 2);
  return vec3<i32>(year, month, day);
}

fn getDayOfYear(date: vec3<i32>) -> i32 {
  let adjustment = select(select(-2, -1, isLeapYear(date.x)), 0, date.y <= 2);
  return (367 * date.y - 362) / 12 + date.z + adjustment;
}

struct DivideResult {
  quotientLow: u32,
  quotientHigh: u32,
  remainder: u32
}

// Exact unsigned 64-bit division by a divisor below 2^28, one nibble at a time so that
// remainder * 16 never overflows 32 bits.
fn divideUnsigned64(value: vec2<u32>, divisor: u32) -> DivideResult {
  var remainder = value.y % divisor;
  var quotientLow = 0u;
  for (var nibble = 7i; nibble >= 0i; nibble = nibble - 1i) {
    let shift = u32(nibble) * 4u;
    let current = (remainder << 4u) | ((value.x >> shift) & 0xfu);
    quotientLow = quotientLow | ((current / divisor) << shift);
    remainder = current % divisor;
  }
  return DivideResult(quotientLow, value.y / divisor, remainder);
}

// Minutes to milliseconds; an offset beyond +-1440 minutes maps to a value the decode rejects.
fn minutesToMilliseconds(offsetMinutes: i32) -> i32 {
  return select(offsetMinutes * 60000, 0x7fffffff, offsetMinutes < -1440 || offsetMinutes > 1440);
}

fn decodeCalendar(time: vec2<u32>, offsetMilliseconds: i32, firstDay: u32) -> CalendarFields {
  let invalid = invalidCalendar();
  if (offsetMilliseconds < -86400000 || offsetMilliseconds > 86400000) {
    return invalid;
  }
  // Signed 64-bit local time = time + offset, with overflow detection.
  let offsetLow = bitcast<u32>(offsetMilliseconds);
  let offsetHigh = select(0u, 0xffffffffu, offsetMilliseconds < 0);
  let low = time.x + offsetLow;
  let high = time.y + offsetHigh + select(0u, 1u, low < time.x);
  let timeSign = time.y >> 31u;
  let sumSign = high >> 31u;
  if (timeSign == (offsetHigh >> 31u) && sumSign != timeSign) {
    return invalid;
  }
  let isNegative = sumSign != 0u;
  var magnitude = vec2<u32>(low, high);
  if (isNegative) {
    let negatedLow = ~low + 1u;
    magnitude = vec2<u32>(negatedLow, ~high + select(0u, 1u, negatedLow == 0u));
  }
  let division = divideUnsigned64(magnitude, MILLISECONDS_PER_DAY);
  if (division.quotientHigh != 0u || division.quotientLow > MAXIMUM_DAYS) {
    return invalid;
  }
  var days = i32(division.quotientLow);
  var millisecondsOfDay = division.remainder;
  if (isNegative) {
    // floor(-(q * D + r) / D) = -(q + 1) with D - r left over when r is not zero.
    if (millisecondsOfDay != 0u) {
      days = days + 1;
      millisecondsOfDay = MILLISECONDS_PER_DAY - millisecondsOfDay;
    }
    days = -days;
  }
  let date = civilFromDays(days);
  let isoWeekday = u32(((days + 3) % 7 + 7) % 7);
  let thursday = days - i32(isoWeekday) + 3;
  let thursdayDate = civilFromDays(thursday);
  var fields = invalid;
  fields.valid = true;
  fields.year = date.x;
  fields.month = u32(date.y);
  fields.day = u32(date.z);
  fields.hour = millisecondsOfDay / 3600000u;
  fields.minute = (millisecondsOfDay % 3600000u) / 60000u;
  fields.weekday = (isoWeekday + 7u - firstDay) % 7u;
  fields.dayOfYear = u32(getDayOfYear(date));
  fields.isoWeek = u32((getDayOfYear(thursdayDate) - 1) / 7 + 1);
  fields.isoWeekYear = thursdayDate.x;
  fields.quarter = u32((date.y - 1) / 3 + 1);
  return fields;
}
`;

const NEGATE_INT64_WGSL = /* wgsl */ `
fn negateInt64(value: vec2<u32>) -> vec2<u32> {
  let low = ~value.x + 1u;
  return vec2<u32>(low, ~value.y + select(0u, 1u, low == 0u));
}
`;

// Floor division of an Int64 by 1000 or 1,000,000 in unsigned magnitude words, so negative times
// round toward the earlier millisecond.
const divideToMilliseconds = (divisor: string) => /* wgsl */ `${NEGATE_INT64_WGSL}
fn toEpochMilliseconds(time: vec2<u32>) -> vec2<u32> {
  let isNegative = (time.y >> 31u) != 0u;
  let magnitude = select(time, negateInt64(time), isNegative);
  let division = divideUnsigned64(magnitude, ${divisor}u);
  var quotient = vec2<u32>(division.quotientLow, division.quotientHigh);
  if (isNegative && division.remainder != 0u) {
    quotient = vec2<u32>(quotient.x + 1u, quotient.y + select(0u, 1u, quotient.x == 0xffffffffu));
  }
  return select(quotient, negateInt64(quotient), isNegative);
}
`;

/** Per-unit `toEpochMilliseconds(time)`: converts Int64 words of that unit to epoch milliseconds. */
const UNIT_WGSL: Record<ArrowAnalyticsTimeUnit, string> = {
  millisecond: /* wgsl */ `
fn toEpochMilliseconds(time: vec2<u32>) -> vec2<u32> {
  return time;
}
`,
  microsecond: divideToMilliseconds('1000'),
  nanosecond: divideToMilliseconds('1000000'),
  // Seconds times 1000. A magnitude of 2^52 seconds or more is far beyond the +-1e9 day range, so
  // it saturates to the Int64 extreme, which the decode rejects as out of range.
  second: /* wgsl */ `${NEGATE_INT64_WGSL}
fn toEpochMilliseconds(time: vec2<u32>) -> vec2<u32> {
  let isNegative = (time.y >> 31u) != 0u;
  let magnitude = select(time, negateInt64(time), isNegative);
  if (magnitude.y >= 0x100000u) {
    return select(vec2<u32>(0xffffffffu, 0x7fffffffu), vec2<u32>(0u, 0x80000000u), isNegative);
  }
  let lowPart = (magnitude.x & 0xffffu) * 1000u;
  let highPart = (magnitude.x >> 16u) * 1000u;
  let shifted = highPart << 16u;
  let low = shifted + lowPart;
  let high = magnitude.y * 1000u + (highPart >> 16u) + select(0u, 1u, low < shifted);
  let product = vec2<u32>(low, high);
  return select(product, negateInt64(product), isNegative);
}
`
};

const ZONE_WGSL = /* wgsl */ `
// Signed 64-bit a <= b on (low, high) words.
fn lessOrEqualInt64(a: vec2<u32>, b: vec2<u32>) -> bool {
  let aHigh = bitcast<i32>(a.y);
  let bHigh = bitcast<i32>(b.y);
  if (aHigh != bHigh) {
    return aHigh < bHigh;
  }
  return a.x <= b.x;
}

// Upper-bound binary search: the number of transitions at or before the instant, which indexes the
// offset in force.
fn getZoneOffsetMilliseconds(time: vec2<u32>) -> i32 {
  var low = 0u;
  var high = (arrayLength(&zoneTransitions) - zoneTransitionsOffset) / 2u;
  loop {
    if (low >= high) {
      break;
    }
    let middle = low + (high - low) / 2u;
    let transition = vec2<u32>(
      zoneTransitions[zoneTransitionsOffset + 2u * middle],
      zoneTransitions[zoneTransitionsOffset + 2u * middle + 1u]);
    if (lessOrEqualInt64(transition, time)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let seconds = zoneOffsets[zoneOffsetsOffset + low];
  // A table offset beyond +-86400 s maps to a value the decode rejects.
  return select(seconds * 1000, 0x7fffffff, seconds < -86400 || seconds > 86400);
}
`;
