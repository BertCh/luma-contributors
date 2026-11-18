// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU oracle of GPUCellTableCompare. It mirrors the kernel's f32 steps with Math.fround and the
// fixed-order z-score reduction (strided partial sums, then a binary tree).

const f = Math.fround;
const WORKGROUP_SIZE = 256;

/** One input table row: key, count, and exact fixed-point sum. */
export type CompareInputCell = {key: bigint; count: number; sum: bigint};

/** One union row. */
export type CompareRow = {
  key: bigint;
  /** Bit 1 in before, bit 2 in after. */
  presence: number;
  before: number;
  after: number;
  delta: number;
  ratio: number;
  percentChange: number;
  zScore: number;
};

/** Oracle result: rows bounded by the capacity plus unclamped total and overflow. */
export type CompareResult = {
  rows: CompareRow[];
  total: number;
  overflow: boolean;
  candidateOverflow: boolean;
};

/** Options of {@link compareCellTablesOnCPU}. */
export type CompareOptions = {
  measure: 'count' | 'sum';
  sumScale: number;
  zScore: 'poisson' | 'standardized';
  capacity: number;
  /** Input final truncation, propagated as result candidateOverflow. */
  inputOverflow?: boolean;
  inputCandidateOverflow?: boolean;
};

/** Converts a signed integer or difference using its full uint64 magnitude, then its sign. */
export function convertInt64ToFloat32(value: bigint): number {
  const magnitude = value < 0n ? -value : value;
  const discardedCount = Math.max(0, magnitude.toString(2).length - 24);
  const shift = BigInt(discardedCount);
  let significand = magnitude >> shift;
  if (discardedCount > 0) {
    const discarded = magnitude - (significand << shift);
    const halfway = 1n << (shift - 1n);
    if (discarded > halfway || (discarded === halfway && (significand & 1n) !== 0n)) {
      significand++;
    }
  }
  const result = Number(significand) * 2 ** discardedCount;
  return value < 0n ? -result : result;
}

/** Fixed-order f32 sum of `values`: strided partials per thread, then a binary tree. */
export function sumFixedOrder(values: readonly number[]): number {
  const partials = new Array<number>(WORKGROUP_SIZE).fill(0);
  for (let thread = 0; thread < WORKGROUP_SIZE; thread++) {
    let sum = 0;
    for (let row = thread; row < values.length; row += WORKGROUP_SIZE) {
      sum = f(sum + values[row]);
    }
    partials[thread] = sum;
  }
  for (let stride = WORKGROUP_SIZE / 2; stride > 0; stride = Math.floor(stride / 2)) {
    for (let thread = 0; thread < stride; thread++) {
      partials[thread] = f(partials[thread] + partials[thread + stride]);
    }
  }
  return partials[0];
}

/** Outer-joins two ascending, distinct-key tables and derives the compare columns. */
export function compareCellTablesOnCPU(
  before: readonly CompareInputCell[],
  after: readonly CompareInputCell[],
  options: CompareOptions
): CompareResult {
  const isSum = options.measure === 'sum';
  const measureOf = (cell: CompareInputCell | undefined) =>
    cell ? (isSum ? cell.sum : BigInt(cell.count)) : 0n;
  const toFloat = (value: bigint) =>
    isSum ? f(convertInt64ToFloat32(value) / options.sumScale) : convertInt64ToFloat32(value);
  const union: {
    key: bigint;
    before?: CompareInputCell;
    after?: CompareInputCell;
  }[] = [];
  let beforeIndex = 0;
  let afterIndex = 0;
  while (beforeIndex < before.length || afterIndex < after.length) {
    const left = before[beforeIndex];
    const right = after[afterIndex];
    if (left && (!right || left.key < right.key)) {
      union.push({key: left.key, before: left});
      beforeIndex++;
    } else if (right && (!left || right.key < left.key)) {
      union.push({key: right.key, after: right});
      afterIndex++;
    } else {
      union.push({key: left.key, before: left, after: right});
      beforeIndex++;
      afterIndex++;
    }
  }
  const bounded = union.slice(0, options.capacity);
  const base = bounded.map(entry => {
    const beforeMeasure = measureOf(entry.before);
    const afterMeasure = measureOf(entry.after);
    return {
      key: entry.key,
      presence: (entry.before ? 1 : 0) | (entry.after ? 2 : 0),
      before: toFloat(beforeMeasure),
      after: toFloat(afterMeasure),
      delta: toFloat(afterMeasure - beforeMeasure)
    };
  });
  let mean = 0;
  let deviation = 0;
  let normalizationScale = 1;
  if (options.zScore === 'standardized' && base.length > 0) {
    const maximum = base.reduce((largest, row) => Math.max(largest, Math.abs(row.delta)), 0);
    const exponent = maximum === 0 ? 0 : Math.floor(Math.log2(maximum)) + 1;
    normalizationScale = 2 ** -exponent;
    const count = base.length;
    mean = f(sumFixedOrder(base.map(row => f(row.delta * normalizationScale))) / count);
    const squares = base.map(row => {
      const centered = f(f(row.delta * normalizationScale) - mean);
      return f(centered * centered);
    });
    deviation = f(Math.sqrt(f(sumFixedOrder(squares) / count)));
  }
  const rows: CompareRow[] = base.map(row => {
    let zScore: number;
    if (options.zScore === 'poisson') {
      const denominator = f(row.before + row.after);
      // Defined for a positive denominator, 0 when both values are 0, otherwise no value.
      zScore =
        denominator > 0
          ? f(row.delta / f(Math.sqrt(denominator)))
          : row.before === 0 && row.after === 0
            ? 0
            : NaN;
    } else {
      zScore = deviation === 0 ? 0 : f(f(f(row.delta * normalizationScale) - mean) / deviation);
    }
    return {
      ...row,
      ratio: row.before === 0 ? NaN : f(row.after / row.before),
      percentChange: row.before === 0 ? NaN : f((100 * row.delta) / row.before),
      zScore
    };
  });
  return {
    rows,
    total: union.length,
    overflow: union.length > options.capacity,
    candidateOverflow: Boolean(options.inputOverflow || options.inputCandidateOverflow)
  };
}
