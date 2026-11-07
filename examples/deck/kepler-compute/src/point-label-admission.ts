// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';
import * as arrow from 'apache-arrow';
import {preparePointLabelSource, type PointLabelScalar} from './label-source';
import {
  STREAMED_LABEL_LIMITS,
  normalizeLabelText,
  type StreamedLabelTextChunk
} from './label-streamed';

export const POINT_LABEL_LIMITS = Object.freeze({
  maximumRows: 100000,
  maximumBatches: 4096,
  maximumCharactersPerLabel: 256,
  maximumGlyphs: 250000,
  maximumSourceBytes: 8 << 20,
  workingMemoryBudgetBytes: 32 << 20
});

/** Label glyph and atlas budgets bound sources to this many rows, resident or not. */
export function assertPointLabelRowCount(rowCount: number): void {
  if (rowCount > POINT_LABEL_LIMITS.maximumRows)
    throw new Error(`Point labels support at most ${POINT_LABEL_LIMITS.maximumRows} source rows.`);
}

/** Explicit field admission reads original scalar cells once, preserving source batch identity. */
export function prepareResidentPointLabels(
  source: arrow.Table,
  batches: readonly {id: string; rowCount: number}[],
  fieldName: string,
  formatValue?: (value: PointLabelScalar) => PointLabelScalar
) {
  assertPointLabelRowCount(source.numRows);
  if (batches.length > POINT_LABEL_LIMITS.maximumBatches)
    throw new KeplerAllocationError('capacity', {
      message: 'Point labels exceed the admitted source batch count.'
    });
  if (source.batches.length !== batches.length)
    throw new Error('Point label source batches do not match the resident source.');
  const field = source.schema.fields.find(candidate => candidate.name === fieldName);
  if (!field || !isLabelScalarType(field.type))
    throw new Error('Point labels require a string, integer, float, or boolean source field.');
  const budget = {sourceBytes: 0, decodedBytes: 0, glyphs: 0};
  const chunks = source.batches.map((batch, batchIndex) => {
    const column = batch.getChild(fieldName);
    if (!column || column.length !== batches[batchIndex].rowCount)
      throw new Error('Point label source rows do not match the resident source.');
    preflightLabelStrings(column, budget);
    const values: PointLabelScalar[] = [];
    for (let rowIndex = 0; rowIndex < column.length; rowIndex++) {
      let value: unknown = column.get(rowIndex);
      if (value instanceof Date) value = value.getTime();
      if (
        value !== null &&
        value !== undefined &&
        typeof value !== 'string' &&
        typeof value !== 'number' &&
        typeof value !== 'boolean' &&
        typeof value !== 'bigint'
      )
        throw new Error('Point labels require ordinary scalar source values.');
      const formatted = formatValue ? formatValue(value) : value;
      const text = formatted === null || formatted === undefined ? '' : String(formatted);
      budget.decodedBytes += text.length * 2 + 8;
      let characters = 0;
      for (const _character of text) {
        characters++;
        budget.glyphs++;
      }
      if (
        characters > POINT_LABEL_LIMITS.maximumCharactersPerLabel ||
        budget.glyphs > POINT_LABEL_LIMITS.maximumGlyphs ||
        budget.decodedBytes > POINT_LABEL_LIMITS.maximumSourceBytes
      )
        throw new KeplerAllocationError('memory-budget', {
          message: 'Point labels exceed their admitted character or byte budget.'
        });
      values.push(formatted);
    }
    return {id: batches[batchIndex].id, values};
  });
  return preparePointLabelSource({
    fieldName,
    chunks,
    maxCharactersPerLabel: POINT_LABEL_LIMITS.maximumCharactersPerLabel,
    maxGlyphs: POINT_LABEL_LIMITS.maximumGlyphs,
    maxSourceBytes: POINT_LABEL_LIMITS.maximumSourceBytes
  });
}

/**
 * Streamed admission for GPU-collision sources above the 100,000 row cap: no glyph, byte or row
 * budget is spent here because rows are read lazily in windows (see label-streamed.ts); only the
 * collision row cap and the source shape are checked.
 */
export function prepareStreamedPointLabels(
  source: arrow.Table,
  batches: readonly {id: string; rowCount: number}[],
  fieldName: string,
  formatValue?: (value: PointLabelScalar) => PointLabelScalar
): StreamedLabelTextChunk[] {
  if (source.numRows > STREAMED_LABEL_LIMITS.maximumRows)
    throw new KeplerAllocationError('capacity', {
      message: 'Streamed point labels exceed the GPU collision row cap.'
    });
  if (batches.length > POINT_LABEL_LIMITS.maximumBatches)
    throw new KeplerAllocationError('capacity', {
      message: 'Point labels exceed the admitted source batch count.'
    });
  if (source.batches.length !== batches.length)
    throw new Error('Point label source batches do not match the resident source.');
  const field = source.schema.fields.find(candidate => candidate.name === fieldName);
  if (!field || !isLabelScalarType(field.type))
    throw new Error('Point labels require a string, integer, float, or boolean source field.');
  return source.batches.map((batch, batchIndex) => {
    const column = batch.getChild(fieldName);
    if (!column || column.length !== batches[batchIndex].rowCount)
      throw new Error('Point label source rows do not match the resident source.');
    return {
      id: batches[batchIndex].id,
      rowCount: column.length,
      getText(rowIndex: number) {
        let value: unknown = column.get(rowIndex);
        if (value instanceof Date) value = value.getTime();
        if (value !== null && value !== undefined && typeof value === 'object') return '';
        const scalar = value as PointLabelScalar;
        return normalizeLabelText(formatValue ? formatValue(scalar) : scalar);
      }
    };
  });
}

export function isLabelScalarType(type: arrow.DataType): boolean {
  if (arrow.DataType.isDictionary(type)) return isLabelScalarType(type.dictionary);
  return (
    arrow.DataType.isUtf8(type) ||
    arrow.DataType.isLargeUtf8(type) ||
    arrow.DataType.isInt(type) ||
    arrow.DataType.isFloat(type) ||
    arrow.DataType.isBool(type) ||
    // Timestamps are formatted from Unix milliseconds by the native display format.
    arrow.DataType.isTimestamp(type)
  );
}

/** Reject oversized UTF-8 cells before Arrow decodes their payloads into JavaScript strings. */
function preflightLabelStrings(column: arrow.Vector, budget: {sourceBytes: number}): void {
  if (column.length > POINT_LABEL_LIMITS.maximumRows)
    throw new KeplerAllocationError('memory-budget', {
      message: 'Point label dictionary exceeds the admitted row budget.'
    });
  if (arrow.DataType.isDictionary(column.type)) {
    for (const data of column.data) {
      if (data.dictionary) preflightLabelStrings(data.dictionary, budget);
    }
    return;
  }
  if (!arrow.DataType.isUtf8(column.type) && !arrow.DataType.isLargeUtf8(column.type)) return;
  for (const data of column.data) {
    const offsets = data.valueOffsets;
    for (let rowIndex = 0; rowIndex < data.length; rowIndex++) {
      if (!data.getValid(rowIndex)) continue;
      const start = Number(offsets[rowIndex]);
      const end = Number(offsets[rowIndex + 1]);
      const byteLength = end - start;
      if (
        !Number.isSafeInteger(start) ||
        !Number.isSafeInteger(end) ||
        start < 0 ||
        end > data.values.byteLength ||
        byteLength < 0 ||
        byteLength > POINT_LABEL_LIMITS.maximumCharactersPerLabel * 4
      )
        throw new Error('A point label exceeds the bounded UTF-8 source-cell limit.');
      budget.sourceBytes += byteLength;
      if (budget.sourceBytes > POINT_LABEL_LIMITS.maximumSourceBytes)
        throw new KeplerAllocationError('memory-budget', {
          message: 'Point label source exceeds the admitted byte budget.'
        });
    }
  }
}
