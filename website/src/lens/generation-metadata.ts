// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {isDeepStrictEqual} from 'node:util';

/** Every source must have one published model, and every model must have a source. */
export function validateLensFileNames(
  sourceFiles: readonly string[],
  modelFiles: readonly string[]
): void {
  const sources = new Set(sourceFiles);
  const models = new Set(modelFiles);
  for (const file of sourceFiles)
    if (!models.has(file))
      throw new Error(
        `${file}: source has no model; run full generation before publishing metadata`
      );
  for (const file of modelFiles)
    if (!sources.has(file))
      throw new Error(
        `${file}: model has no source; run full generation before publishing metadata`
      );
}

/** Compare JSON content, preserving harmless formatting while refusing broken or stale models. */
export function validateLensModel(scenarioId: string, text: string, expected: unknown): void {
  if (!isDeepStrictEqual(JSON.parse(text), JSON.parse(JSON.stringify(expected)))) {
    throw new Error(
      `${scenarioId}: model differs from fresh source extraction; regenerate this model before publishing metadata`
    );
  }
}

/** Ungrouped source compute counts and UTF-8 file sizes for one proposed published model. */
export type LensPageEntry = {
  id: string;
  kernels: number;
  recordBytes: number;
  viewModelBytes: number;
};

/** Calculate and validate page metadata before a writer replaces any published artifacts. */
export function createLensPageData(entries: readonly LensPageEntry[]) {
  const featured = [
    'gpu-spatial-analysis.kernel-density',
    'gpu-spatial-analysis.local-moran',
    'gpu-network.network-service-areas'
  ].map(id => {
    const entry = entries.find(entry => entry.id === id);
    if (!entry)
      throw new Error(
        `Lens page requires featured scenario ${id}; restore its source record and model before generating`
      );
    return {id, recordBytes: entry.recordBytes, viewModelBytes: entry.viewModelBytes};
  });
  const sorted = [...entries].sort((left, right) => left.kernels - right.kernels);
  const middle = sorted[Math.floor(sorted.length / 2)];
  const median = {id: middle.id, kernels: middle.kernels};
  const starts = [0, 1, 2, 5, 10, 20, 50, 100, 300];
  const bins = starts.map((start, index) => {
    const next = starts[index + 1];
    return {
      label:
        next === undefined ? `${start}+` : next - start === 1 ? `${start}` : `${start}-${next - 1}`,
      count: entries.filter(
        entry => entry.kernels >= start && (next === undefined || entry.kernels < next)
      ).length
    };
  });
  return {featured, median, bins};
}
