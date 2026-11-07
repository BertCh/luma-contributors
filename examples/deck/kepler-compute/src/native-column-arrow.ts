// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';

/** The subset of a Kepler field that determines its Arrow representation. */
export type NativeFieldDescription = {name: string; type: string; format: string};

const INT32_MINIMUM = -(2 ** 31);
const INT32_MAXIMUM = 2 ** 31 - 1;
const UINT32_MAXIMUM = 2 ** 32 - 1;

/** Route chosen for one native column; surfaced to the user by the auto-prepare report. */
export type NativeColumnRoute = {
  arrowType: string;
  /** Present when the Kepler analyzer type is not carried over one-to-one. */
  note?: string;
};

function isMissing(value: unknown): value is null | undefined {
  return (
    value === null || value === undefined || (typeof value === 'number' && Number.isNaN(value))
  );
}

function readNumber(value: unknown): number | null {
  if (isMissing(value) || value === '') return null;
  const number = typeof value === 'number' ? value : Number(value);
  return Number.isNaN(number) ? null : number;
}

function formatText(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value instanceof Date) return value.toISOString();
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

/**
 * Builds one exact Arrow column from the values of a Kepler field. Kepler's analyzer type, not
 * JavaScript value inference, selects the representation: integers become Int32/Uint32 when they
 * fit and otherwise exact decimal category labels (never a lossy Float32), reals stay Float64
 * (the precision policy converts them), timestamps become millisecond instants, geometry and
 * nested values become JSON text.
 */
/** Kepler's `timeToUnixMilli`, injected so this module stays free of Kepler runtime imports. */
export type NativeTimeParser = (value: string | number | Date, format: string) => number | null;

export function makeArrowColumnFromNativeField(
  field: NativeFieldDescription,
  values: readonly unknown[],
  timeToUnixMilli: NativeTimeParser
): {vector: arrow.Vector; route: NativeColumnRoute} {
  switch (field.type) {
    case 'timestamp':
    case 'date': {
      const milliseconds = values.map((value, rowIndex): number | null => {
        if (isMissing(value) || value === '') return null;
        if (typeof value !== 'string' && typeof value !== 'number' && !(value instanceof Date))
          throw new Error(`Unsupported time value in ${field.name}, row ${rowIndex}`);
        const instant = timeToUnixMilli(value, field.format);
        if (instant === null || instant === undefined) return null;
        if (!Number.isSafeInteger(instant))
          throw new Error(
            `Native time ${field.name}, row ${rowIndex} is not an exact millisecond instant`
          );
        return instant;
      });
      return {
        vector: arrow.vectorFromArray(milliseconds, new arrow.TimestampMillisecond()),
        route: {arrowType: 'timestamp[ms]'}
      };
    }
    case 'boolean':
      return {
        vector: arrow.vectorFromArray(
          values.map(value =>
            isMissing(value) || value === '' ? null : value === true || value === 'true'
          ),
          new arrow.Bool()
        ),
        route: {arrowType: 'bool'}
      };
    case 'integer': {
      const numbers = values.map(readNumber);
      let isInt32 = true;
      let isUint32 = true;
      let isIntegral = true;
      for (const number of numbers) {
        if (number === null) continue;
        if (!Number.isInteger(number)) {
          isIntegral = false;
          break;
        }
        if (number < INT32_MINIMUM || number > INT32_MAXIMUM) isInt32 = false;
        if (number < 0 || number > UINT32_MAXIMUM) isUint32 = false;
      }
      if (isIntegral && isInt32)
        return {
          vector: arrow.vectorFromArray(numbers, new arrow.Int32()),
          route: {arrowType: 'int32'}
        };
      if (isIntegral && isUint32)
        return {
          vector: arrow.vectorFromArray(numbers, new arrow.Uint32()),
          route: {arrowType: 'uint32'}
        };
      if (isIntegral)
        // Identities beyond 32 bits keep every digit as a category label instead of rounding.
        return {
          vector: arrow.vectorFromArray(
            numbers.map(number => (number === null ? null : BigInt(number).toString())),
            new arrow.Utf8()
          ),
          route: {
            arrowType: 'utf8',
            note: 'Integers beyond 32 bits are kept exactly as category labels, not numbers'
          }
        };
      return {
        vector: arrow.vectorFromArray(numbers, new arrow.Float64()),
        route: {arrowType: 'float64', note: 'Non-integral values in an integer field'}
      };
    }
    case 'real':
      return {
        vector: arrow.vectorFromArray(values.map(readNumber), new arrow.Float64()),
        route: {arrowType: 'float64'}
      };
    case 'geojson':
    case 'point':
    case 'array':
    case 'object':
      return {
        vector: arrow.vectorFromArray(
          values.map(value => (isMissing(value) || value === '' ? null : formatText(value))),
          new arrow.Utf8()
        ),
        route: {arrowType: 'utf8', note: 'Nested or geometry values are retained as JSON text'}
      };
    default:
      return {
        vector: arrow.vectorFromArray(
          values.map(value => (isMissing(value) ? null : formatText(value))),
          new arrow.Utf8()
        ),
        route: {arrowType: 'utf8'}
      };
  }
}
