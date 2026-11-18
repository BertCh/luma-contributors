// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LensResourceBytesKind} from './types.ts';

/** Conservatively combines recorded sizes; omitted precision belongs to legacy records. */
export function getResourceBytesKind(
  resources: readonly {bytesKind?: LensResourceBytesKind}[]
): LensResourceBytesKind {
  if (resources.some(resource => !resource.bytesKind || resource.bytesKind === 'unspecified'))
    return 'unspecified';
  if (resources.some(resource => resource.bytesKind === 'unavailable')) return 'unavailable';
  if (resources.some(resource => resource.bytesKind === 'estimated')) return 'estimated';
  return resources.some(resource => resource.bytesKind === 'lower-bound')
    ? 'lower-bound'
    : 'logical';
}

export function formatResourceBytes(bytes: number, bytesKind?: LensResourceBytesKind): string {
  const formatted = formatBytes(bytes);
  switch (bytesKind) {
    case 'logical':
      return formatted;
    case 'lower-bound':
      return `≥${bytes} B`;
    case 'unavailable':
      return 'size unavailable';
    case 'estimated':
      return `~${formatted}`;
    default:
      return `${formatted} (precision unknown)`;
  }
}

export function formatResourceCount(resources: {count: number; inventoryKnown?: boolean}): string {
  return resources.inventoryKnown === false
    ? 'inventory unavailable'
    : `${formatCount(resources.count)} buffers`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${trim(bytes / 1024)} KB`;
  if (bytes < 1024 ** 3) return `${trim(bytes / 1024 ** 2)} MB`;
  return `${trim(bytes / 1024 ** 3)} GB`;
}

export function formatCount(value: number): string {
  return value.toLocaleString('en-US');
}

function trim(value: number): string {
  return value >= 100 ? value.toFixed(0) : value >= 10 ? value.toFixed(1) : value.toFixed(2);
}
