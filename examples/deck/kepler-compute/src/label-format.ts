// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PointLabelScalar} from './label-source';
import {
  DEFAULT_TIMESTAMP_LABEL_FORMAT,
  getTimestampMilliseconds,
  isTimestampLabelFieldType
} from './label-timestamp';

export type PointLabelFormatConfig = {
  field?: {name?: string; type?: string; displayFormat?: unknown} | null;
  format?: unknown;
};

/** Native precedence: field display format first, then the label format. */
export function getPointLabelFormat(label: PointLabelFormatConfig): string | undefined {
  const format = label.field?.displayFormat ?? label.format;
  if (typeof format === 'string' && format) return format;
  // Native falls back to the date-time tooltip format for timestamp fields.
  return isTimestampLabelFieldType(label.field?.type) ? DEFAULT_TIMESTAMP_LABEL_FORMAT : undefined;
}

/** Stable variant key so a format change prepares a new label source. */
export function getPointLabelFormatKey(label: PointLabelFormatConfig): string {
  return getPointLabelFormat(label) ?? '';
}

type KeplerFormatterFactory = (format: string, field: unknown) => (value: unknown) => unknown;
let formatterFactory: KeplerFormatterFactory | null = null;

/** Host boundary registers Kepler's getFormatter; keeps @kepler.gl/utils out of node-tested modules. */
export function setPointLabelFormatterFactory(factory: KeplerFormatterFactory | null): void {
  formatterFactory = factory;
}

/** Native getFormatter applied once per cell during explicit admission; null stays an empty label. */
export function makePointLabelFormatter(
  label: PointLabelFormatConfig
): ((value: PointLabelScalar) => PointLabelScalar) | undefined {
  const format = getPointLabelFormat(label);
  if (!format) return undefined;
  // Formatter factory is registered by the host (label-format-host.ts) before labels are prepared.
  if (!formatterFactory) return undefined;
  const formatter = formatterFactory(format, label.field);
  const isTimestamp = isTimestampLabelFieldType(label.field?.type);
  return value => {
    if (value === null || value === undefined) return value;
    const input = isTimestamp
      ? getTimestampMilliseconds(value)
      : typeof value === 'bigint'
        ? Number(value)
        : value;
    if (input === null) return '';
    const text = formatter(input);
    return text === null || text === undefined ? '' : String(text);
  };
}
