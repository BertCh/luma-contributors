// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ResidentSqlCastTarget, ResidentSqlScalarFormat} from './resident-sql-strings';

/**
 * WGSL body for one numeric CAST kernel, and the step-for-step CPU mirror of it.
 *
 * Words are the raw 32-bit cells of the resident column (`float32`, `sint32` or `uint32`). A cast
 * that cannot represent a value (NaN, out of range, negative to unsigned) writes validity 0 where
 * DuckDB would raise an error; rounding is half away from zero like DuckDB.
 */
export function getResidentSqlCastKernelBody(
  source: ResidentSqlScalarFormat,
  target: ResidentSqlCastTarget,
  hasSourceValidity: boolean
): string {
  const header = `  let word = source[index];
  var valid: u32 = ${hasSourceValidity ? 'validityIn[index]' : '1u'};
  var result: u32 = 0u;`;
  return `${header}
${getResidentSqlCastConversionBody(source, target)}
  output[index] = select(0u, result, valid != 0u);
  validityOut[index] = valid;`;
}

/**
 * Conversion statements over `word` (raw source cell), `valid` and `result` (both declared by the
 * caller). DATE and TIMESTAMP are epoch seconds in an int32; DATE floors to UTC midnight.
 */
export function getResidentSqlCastConversionBody(
  source: ResidentSqlScalarFormat,
  target: ResidentSqlCastTarget
): string {
  if (target === 'timestamp') return getResidentSqlCastConversionBody(source, 'sint32');
  if (target === 'date')
    return `${getResidentSqlCastConversionBody(source, 'sint32')}
  if (valid != 0u) {
    let seconds = bitcast<i32>(result);
    var days = seconds / 86400;
    if (seconds % 86400 < 0) { days = days - 1; }
    result = bitcast<u32>(days * 86400);
  }`;
  let conversion: string;
  if (source === 'float32') {
    const value = '  let v = bitcast<f32>(word);';
    conversion =
      target === 'float32'
        ? '  result = word;'
        : target === 'bool'
          ? `${value}\n  result = select(0u, 1u, v != 0.0);`
          : target === 'sint32'
            ? `${value}
  if (!(v >= -2147483648.0 && v < 2147483648.0)) { valid = 0u; } else {
    let magnitude = floor(abs(v) + 0.5);
    result = bitcast<u32>(i32(select(magnitude, -magnitude, v < 0.0)));
  }`
            : `${value}
  if (!(v > -0.5 && v < 4294967296.0)) { valid = 0u; } else { result = u32(floor(max(v, 0.0) + 0.5)); }`;
  } else if (source === 'sint32') {
    const value = '  let v = bitcast<i32>(word);';
    conversion =
      target === 'float32'
        ? `${value}\n  result = bitcast<u32>(f32(v));`
        : target === 'bool'
          ? `${value}\n  result = select(0u, 1u, v != 0);`
          : target === 'sint32'
            ? '  result = word;'
            : `${value}\n  if (v < 0) { valid = 0u; } else { result = word; }`;
  } else {
    conversion =
      target === 'float32'
        ? '  result = bitcast<u32>(f32(word));'
        : target === 'bool'
          ? '  result = select(0u, 1u, word != 0u);'
          : target === 'sint32'
            ? '  if (word > 2147483647u) { valid = 0u; } else { result = word; }'
            : '  result = word;';
  }
  return conversion;
}

/** Order-preserving unsigned key of a raw cell (floats and signed integers sort as unsigned). */
export function makeResidentSqlOrderedKey(source: ResidentSqlScalarFormat, word: number): number {
  const bits = word >>> 0;
  if (source === 'float32') return (bits & 0x80000000 ? ~bits : bits | 0x80000000) >>> 0;
  if (source === 'sint32') return (bits ^ 0x80000000) >>> 0;
  return bits;
}

/** Inverse of {@link makeResidentSqlOrderedKey}. */
export function readResidentSqlOrderedKey(source: ResidentSqlScalarFormat, key: number): number {
  const bits = key >>> 0;
  if (source === 'float32') return (bits & 0x80000000 ? bits & 0x7fffffff : ~bits) >>> 0;
  if (source === 'sint32') return (bits ^ 0x80000000) >>> 0;
  return bits;
}

/** WGSL for the ordered key of `word` into `key` (declared by this snippet). */
export function getResidentSqlOrderedKeySource(source: ResidentSqlScalarFormat): string {
  return source === 'float32'
    ? '  var key: u32 = word | 0x80000000u;\n  if ((word & 0x80000000u) != 0u) { key = ~word; }'
    : source === 'sint32'
      ? '  let key: u32 = word ^ 0x80000000u;'
      : '  let key: u32 = word;';
}

/** WGSL body of the value-to-dictionary-code kernel (binary search over sorted ordered keys). */
export function getResidentSqlValueLookupBody(source: ResidentSqlScalarFormat): string {
  return `  let word = source[index];
${getResidentSqlOrderedKeySource(source)}
  var low: u32 = 0u;
  var high: u32 = KEY_COUNT;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (keysTable[middle] < key) { low = middle + 1u; } else { high = middle; }
  }
  if (low < KEY_COUNT && keysTable[low] == key) { output[index] = codesTable[low]; } else { output[index] = 0xffffffffu; }`;
}

/** CPU mirror of {@link getResidentSqlCastKernelBody} over raw words. */
export function mirrorResidentSqlCastKernel(
  source: ResidentSqlScalarFormat,
  target: ResidentSqlCastTarget,
  word: number,
  sourceValid = true
): {word: number; valid: boolean} {
  if (target === 'timestamp')
    return mirrorResidentSqlCastKernel(source, 'sint32', word, sourceValid);
  if (target === 'date') {
    const converted = mirrorResidentSqlCastKernel(source, 'sint32', word, sourceValid);
    if (!converted.valid) return converted;
    const seconds = converted.word | 0;
    return {word: (Math.floor(seconds / 86400) * 86400) >>> 0, valid: true};
  }
  const view = new DataView(new ArrayBuffer(4));
  view.setUint32(0, word >>> 0, true);
  let valid = sourceValid;
  let result = 0;
  if (source === 'float32') {
    const v = view.getFloat32(0, true);
    if (target === 'float32') result = word >>> 0;
    else if (target === 'bool') result = v !== 0 ? 1 : 0;
    else if (target === 'sint32') {
      if (!(v >= -2147483648 && v < 2147483648)) valid = false;
      else {
        const magnitude = Math.floor(Math.abs(v) + 0.5);
        view.setInt32(0, v < 0 ? -magnitude : magnitude, true);
        result = view.getUint32(0, true);
      }
    } else if (!(v > -0.5 && v < 4294967296)) valid = false;
    else result = Math.floor(Math.max(v, 0) + 0.5) >>> 0;
  } else if (source === 'sint32') {
    const v = view.getInt32(0, true);
    if (target === 'float32') {
      view.setFloat32(0, v, true);
      result = view.getUint32(0, true);
    } else if (target === 'bool') result = v !== 0 ? 1 : 0;
    else if (target === 'sint32') result = word >>> 0;
    else if (v < 0) valid = false;
    else result = word >>> 0;
  } else if (target === 'float32') {
    view.setFloat32(0, word >>> 0, true);
    result = view.getUint32(0, true);
  } else if (target === 'bool') result = word !== 0 ? 1 : 0;
  else if (target === 'sint32') {
    if (word > 2147483647) valid = false;
    else result = word >>> 0;
  } else result = word >>> 0;
  return {word: valid ? result : 0, valid};
}
