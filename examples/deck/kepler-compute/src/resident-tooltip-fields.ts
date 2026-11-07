// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ResidentConsumerState} from './resident-consumer-state';
import {readConsumerState} from './resident-consumer-state';

import {clamp} from '@math.gl/core';

export type ResidentTooltipValue = string | number | boolean | null;
export type ResidentTooltipFieldConfig = {name: string; format?: string | null};
export type ResidentTooltipComputedProvider = (
  datasetId: string,
  layerId: string,
  sourceRowIndex: number
) => Promise<Record<string, ResidentTooltipValue> | null>;

export const MAXIMUM_TOOLTIP_FIELDS = 8;

/** Native fieldsToShow order is preserved; duplicates are dropped and the list is bounded. */
export function resolveTooltipFields(
  fieldsToShow: readonly ResidentTooltipFieldConfig[],
  maximumFields = MAXIMUM_TOOLTIP_FIELDS
): {fields: ResidentTooltipFieldConfig[]; truncated: boolean} {
  const seen = new Set<string>();
  const fields: ResidentTooltipFieldConfig[] = [];
  for (const field of fieldsToShow) {
    if (seen.has(field.name)) continue;
    seen.add(field.name);
    fields.push({name: field.name, format: field.format ?? null});
  }
  return {fields: fields.slice(0, maximumFields), truncated: fields.length > maximumFields};
}

/**
 * Approximates the d3-format specifiers Kepler offers for tooltips (',', '.1s', '.2%', '.3~f',
 * '+,.3f', '$,.2f', '.4~r', boolean yn/01). Unknown specifiers fall back to the raw value.
 */
export function formatTooltipValue(
  value: ResidentTooltipValue,
  format: string | null | undefined
): string {
  if (value === null) return 'null';
  if (!format || typeof value === 'string') return String(value);
  if (typeof value === 'boolean')
    return format === 'yn' ? (value ? 'yes' : 'no') : value ? '1' : '0';
  if (!Number.isFinite(value)) return String(value);
  if (format === '01') return value ? '1' : '0';
  const match = /^([+$]*)(,?)(?:\.(\d+))?(~?)([sf%er]?)$/.exec(format);
  if (!match) return String(value);
  const [, prefix, comma, digitsText, trim, type] = match;
  const digits = digitsText === undefined ? undefined : Number(digitsText);
  let text: string;
  if (type === 's') {
    const magnitude = Math.abs(value);
    const suffixes = ['', 'k', 'M', 'G', 'T'];
    const exponent = clamp(Math.floor(Math.log10(magnitude || 1) / 3), 0, 4);
    text = `${Number((value / 1000 ** exponent).toPrecision(digits ?? 6))}${suffixes[exponent]}`;
  } else if (type === 'e') {
    text = value.toExponential(digits ?? 6);
  } else if (type === 'r') {
    text = String(Number(value.toPrecision(digits ?? 6)));
  } else {
    const scaled = type === '%' ? value * 100 : value;
    text = scaled.toLocaleString('en-US', {
      useGrouping: comma === ',',
      minimumFractionDigits: trim ? 0 : (digits ?? 0),
      maximumFractionDigits: digits ?? (type === 'f' || type === '%' ? 6 : 20)
    });
    if (type === '%') text += '%';
  }
  if (prefix.includes('+') && value > 0) text = `+${text}`;
  if (prefix.includes('$')) text = text.startsWith('-') ? `-$${text.slice(1)}` : `$${text}`;
  return text;
}

/**
 * Original and computed (analysis output, derived) values are read separately and revalidated
 * against the displayed revision; a revision change yields a stale state with no values.
 */
export async function readTooltipValues(options: {
  getRevision(): string;
  readOriginal(): Promise<Record<string, ResidentTooltipValue> | null>;
  readComputed?: () => Promise<Record<string, ResidentTooltipValue> | null>;
}): Promise<ResidentConsumerState<Record<string, ResidentTooltipValue>>> {
  return readConsumerState(
    'tooltip',
    options.getRevision,
    async () => {
      const original = await options.readOriginal();
      if (!original) return null;
      const computed = (await options.readComputed?.()) ?? {};
      return {...original, ...computed};
    },
    values => Object.keys(values).length === 0
  );
}

const computedProviders = new Map<string, ResidentTooltipComputedProvider>();
/** Analysis panels register one provider per dataset; returns an unregister function. */
export function registerTooltipComputedProvider(
  datasetId: string,
  provider: ResidentTooltipComputedProvider
): () => void {
  computedProviders.set(datasetId, provider);
  return () => {
    if (computedProviders.get(datasetId) === provider) computedProviders.delete(datasetId);
  };
}
export function getTooltipComputedProvider(
  datasetId: string
): ResidentTooltipComputedProvider | undefined {
  return computedProviders.get(datasetId);
}
