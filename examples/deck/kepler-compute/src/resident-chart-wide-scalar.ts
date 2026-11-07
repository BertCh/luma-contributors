// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import * as arrow from 'apache-arrow';
import {aggregateInteger64OnGpu} from './wide-numeric-gpu-aggregate';
import type {WideInteger64Summary} from './wide-numeric-gpu-reference';
import {makeWideNumericWordVectors} from './wide-numeric-gpu-summary';
import type {ResidentChartPlan} from './resident-chart-plan';
import type {ResidentScalarChartResult} from './resident-chart-runtime';

export type ResidentWideScalarMeasure = {
  column: string;
  signed: boolean;
  operation: 'sum' | 'mean' | 'min' | 'max';
};

/**
 * A population-wide scalar chart (Big Number) whose measure is an Int64 or Uint64 source column.
 * Resident storage narrows those columns to Float32, so these statistics run exactly on the wide
 * kernels instead. Filtered populations and ordered statistics (median, deviation) are not covered.
 */
export function planResidentWideScalarMeasure(
  plan: ResidentChartPlan,
  source: arrow.Table
): ResidentWideScalarMeasure | null {
  if (plan.kind !== 'scalar' || plan.population !== 'all' || !plan.column) return null;
  const operation = plan.operation;
  if (operation !== 'sum' && operation !== 'mean' && operation !== 'min' && operation !== 'max')
    return null;
  const type = source.schema.fields.find(field => field.name === plan.column)?.type;
  if (!type || !arrow.DataType.isInt(type) || type.bitWidth !== 64) return null;
  return {column: plan.column, signed: type.isSigned, operation};
}

export function getWideScalarValue(
  summary: WideInteger64Summary,
  operation: ResidentWideScalarMeasure['operation']
): bigint | null {
  if (!summary.count) return null;
  return operation === 'sum'
    ? summary.sum
    : operation === 'min'
      ? summary.minimum
      : operation === 'max'
        ? summary.maximum
        : null;
}

/** One exact summary per source revision and column, shared by every chart over that column. */
export class ResidentWideScalarReader {
  private readonly summaries = new Map<string, Promise<WideInteger64Summary>>();

  async read(props: {
    device: Device;
    source: arrow.Table;
    measure: ResidentWideScalarMeasure;
    sourceRevision: number;
  }): Promise<{result: ResidentScalarChartResult; note: string}> {
    const {measure} = props;
    const key = `${props.sourceRevision}:${measure.column}`;
    let pending = this.summaries.get(key);
    if (!pending) {
      pending = this.scan(props.device, props.source, measure);
      this.summaries.set(key, pending);
      pending.catch(() => this.summaries.delete(key));
    }
    const summary = await pending;
    const exact = getWideScalarValue(summary, measure.operation);
    const value =
      measure.operation === 'mean'
        ? summary.count
          ? Number(summary.sum) / summary.count
          : 0
        : exact === null
          ? 0
          : Number(exact);
    const exactText = measure.operation === 'mean' ? summary.sum.toString() : exact?.toString();
    return {
      result: {
        value,
        hasValue: summary.count > 0,
        sourceRevision: props.sourceRevision,
        selectionRevision: null,
        readbackBytes: 44
      },
      note: `Exact 64-bit GPU ${measure.operation === 'mean' ? 'sum' : measure.operation} of ${measure.column}: ${exactText ?? 'no values'} over ${summary.count} rows${
        measure.operation === 'mean' ? '; the mean is that sum divided by the count' : ''
      }.`
    };
  }

  private async scan(
    device: Device,
    source: arrow.Table,
    measure: ResidentWideScalarMeasure
  ): Promise<WideInteger64Summary> {
    const vectors = makeWideNumericWordVectors(device, source.getChild(measure.column)!);
    try {
      return await aggregateInteger64OnGpu({
        device,
        words: vectors.words,
        signed: measure.signed,
        validity: vectors.validity
      });
    } finally {
      vectors.words.destroy();
      vectors.validity?.destroy();
      vectors.unusedValidity?.destroy();
    }
  }

  clear(): void {
    this.summaries.clear();
  }
}
