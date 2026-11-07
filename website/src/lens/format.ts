// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
