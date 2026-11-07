// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {isWmsTimeSupported, parseWmsTimeExtent, type WmsTimeExtent} from './wms-identify';

/**
 * Finds the time dimension text of one named layer in a WMS GetCapabilities document.
 * WMS 1.3.0 uses `<Dimension name="time">`, 1.1.1 uses `<Extent name="time">`; both live inside
 * the layer's `<Layer>` element and may be inherited from a parent layer.
 */
export function findWmsTimeExtentText(
  capabilitiesXml: string,
  layerName: string
): {extentText: string; defaultValue?: string} | undefined {
  const escapedName = layerName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const nameMatch = new RegExp(`<Name>\\s*${escapedName}\\s*</Name>`).exec(capabilitiesXml);
  const searchStart = nameMatch ? capabilitiesXml.lastIndexOf('<Layer', nameMatch.index) : 0;
  const nextLayer = nameMatch
    ? capabilitiesXml.indexOf('<Layer', nameMatch.index + nameMatch[0].length)
    : -1;
  const layerBlock = capabilitiesXml.slice(searchStart, nextLayer === -1 ? undefined : nextLayer);
  const dimension =
    /<Dimension\b[^>]*\bname\s*=\s*["']time["'][^>]*>([\s\S]*?)<\/Dimension>/i.exec(layerBlock) ??
    /<Extent\b[^>]*\bname\s*=\s*["']time["'][^>]*>([\s\S]*?)<\/Extent>/i.exec(layerBlock);
  if (!dimension) return undefined;
  const defaultMatch = /\bdefault\s*=\s*["']([^"']+)["']/i.exec(dimension[0]);
  return {extentText: dimension[1].trim(), defaultValue: defaultMatch?.[1]};
}

const PERIOD_PATTERN =
  /^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$/;

function addPeriod(instant: Date, period: string, multiplier: number): Date | undefined {
  const match = PERIOD_PATTERN.exec(period);
  if (!match) return undefined;
  const [years, months, weeks, days, hours, minutes, seconds] = match
    .slice(1)
    .map(part => Number(part ?? 0));
  if (!(years || months || weeks || days || hours || minutes || seconds)) return undefined;
  const next = new Date(instant.getTime());
  next.setUTCFullYear(
    next.getUTCFullYear() + years * multiplier,
    next.getUTCMonth() + months * multiplier
  );
  next.setUTCDate(next.getUTCDate() + (weeks * 7 + days) * multiplier);
  next.setUTCHours(
    next.getUTCHours() + hours * multiplier,
    next.getUTCMinutes() + minutes * multiplier,
    next.getUTCSeconds() + seconds * multiplier
  );
  return next;
}

/**
 * Expands a parsed time extent into at most `limit` ISO instants (listed values plus interval
 * steps by ISO 8601 period). Intervals without a usable period contribute their endpoints.
 */
export function expandWmsTimeExtent(extent: WmsTimeExtent, limit = 500): string[] {
  const instants = new Set<string>(extent.values);
  for (const interval of extent.intervals) {
    const end = Date.parse(interval.end);
    let cursor: Date | undefined = new Date(Date.parse(interval.start));
    if (Number.isNaN(cursor.getTime()) || Number.isNaN(end)) continue;
    if (!interval.period) {
      instants.add(interval.start);
      instants.add(interval.end);
      continue;
    }
    while (cursor && cursor.getTime() <= end && instants.size < limit) {
      instants.add(cursor.toISOString().replace('.000Z', 'Z'));
      cursor = addPeriod(cursor, interval.period, 1);
    }
  }
  return [...instants]
    .sort((first, second) => Date.parse(first) - Date.parse(second))
    .slice(0, limit);
}

export function parseWmsExtentInstants(extentText: string, limit = 500): string[] {
  return expandWmsTimeExtent(parseWmsTimeExtent(extentText), limit);
}

export type WmsTimeResolution = Readonly<{
  /** Instant to send as TIME; undefined when the extent has no usable instants. */
  time?: string;
  /** exact: requested instant is advertised; nearest: snapped; current: keyword resolved. */
  kind: 'exact' | 'nearest' | 'current' | 'default' | 'none';
}>;

function findNearestInstant(instants: readonly string[], target: number): string | undefined {
  let best: string | undefined;
  let bestDistance = Infinity;
  for (const instant of instants) {
    const distance = Math.abs(Date.parse(instant) - target);
    if (distance < bestDistance) {
      best = instant;
      bestDistance = distance;
    }
  }
  return best;
}

/**
 * WMS 1.3.0 Annex C time semantics: `current` means the instant nearest to now (clamped to the
 * extent), `default` the capabilities default (else the latest instant), and a requested instant
 * outside the extent snaps to the nearest advertised instant when `allowNearest` is true.
 */
export function resolveWmsTimeValue(
  extentText: string,
  requested: string,
  options: {defaultValue?: string; now?: number; allowNearest?: boolean; limit?: number} = {}
): WmsTimeResolution {
  const instants = parseWmsExtentInstants(extentText, options.limit ?? 5000);
  if (instants.length === 0) return {kind: 'none'};
  const keyword = requested.trim().toLowerCase();
  if (keyword === 'current')
    return {kind: 'current', time: findNearestInstant(instants, options.now ?? Date.now())};
  if (keyword === 'default' || keyword === '')
    return {
      kind: 'default',
      time: options.defaultValue ?? instants[instants.length - 1]
    };
  const target = Date.parse(requested);
  if (Number.isNaN(target)) return {kind: 'none'};
  if (instants.some(instant => Date.parse(instant) === target))
    return {kind: 'exact', time: requested};
  if (
    isWmsTimeSupported(parseWmsTimeExtent(extentText), requested) &&
    options.allowNearest !== true
  )
    return {kind: 'exact', time: requested};
  return options.allowNearest === false
    ? {kind: 'none'}
    : {kind: 'nearest', time: findNearestInstant(instants, target)};
}
