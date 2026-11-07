// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Small formatting and numeric helpers shared by examples. */

import type {Device} from '@luma.gl/core';

/** Clamps `value` to the inclusive range [`minimum`, `maximum`]. */
export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

/** Escapes text for interpolation into HTML text or attribute values. */
export function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

/** Formats a byte count using binary units (`B`, `KiB`, `MiB`) with one fractional digit. */
export function formatBytes(byteLength: number): string {
  if (byteLength < 1024) return `${byteLength} B`;
  if (byteLength < 1024 * 1024) return `${(byteLength / 1024).toFixed(1)} KiB`;
  return `${(byteLength / 1024 / 1024).toFixed(1)} MiB`;
}

/** Reads a named device limit, returning 0 when the backend does not report it. */
export function getDeviceLimit(device: Device, limitName: string): number {
  // `DeviceLimits` only names the cross-backend limits; backend-specific limits are looked up by name.
  return (device.limits as unknown as Record<string, number | undefined>)[limitName] ?? 0;
}
