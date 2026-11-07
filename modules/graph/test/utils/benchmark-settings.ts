// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type BenchmarkSourceSnapshot = {
  revision: string;
  sha256: string;
  capturedAt: string;
  fileCount: number;
};

/** Sampling and provenance shared by opt-in analytical benchmarks, never shipped at runtime. */
export function getBenchmarkSettings(environment: Record<string, string | undefined> = {}) {
  const integer = (key: string, fallback: number, minimum: number) => {
    const value = Number(environment[key] ?? fallback);
    if (!Number.isInteger(value) || value < minimum || value > 1000) {
      throw new Error(`Invalid benchmark setting ${key}`);
    }
    return value;
  };
  let sourceSnapshot: BenchmarkSourceSnapshot | null = null;
  const encoded = environment['VITE_BENCHMARK_SOURCE_SNAPSHOT'];
  if (encoded) {
    const parsed = JSON.parse(encoded) as Partial<BenchmarkSourceSnapshot>;
    if (
      typeof parsed.revision !== 'string' ||
      !/^[a-f0-9]{40}$/.test(parsed.revision) ||
      typeof parsed.sha256 !== 'string' ||
      !/^[a-f0-9]{64}$/.test(parsed.sha256) ||
      typeof parsed.capturedAt !== 'string' ||
      !Number.isFinite(Date.parse(parsed.capturedAt)) ||
      !Number.isInteger(parsed.fileCount) ||
      !(parsed.fileCount! > 0)
    ) {
      throw new Error('Invalid benchmark source snapshot');
    }
    sourceSnapshot = parsed as BenchmarkSourceSnapshot;
    for (const [key, expected] of [
      ['VITE_BENCHMARK_REVISION', sourceSnapshot.revision],
      ['VITE_BENCHMARK_SOURCE_STATE', sourceSnapshot.sha256]
    ]) {
      if (environment[key] && environment[key] !== expected) {
        throw new Error(`Contradictory benchmark provenance ${key}`);
      }
    }
  }
  return {
    warmupCount: integer('VITE_BENCHMARK_WARMUPS', 1, 0),
    sampleCount: integer('VITE_BENCHMARK_SAMPLES', 5, 1),
    updateSampleCount: integer('VITE_BENCHMARK_UPDATE_SAMPLES', 3, 1),
    workloadFilter: environment['VITE_BENCHMARK_WORKLOAD'] ?? '',
    sourceSnapshot
  };
}

export function getBenchmarkDistribution(samples: readonly number[]) {
  if (!samples.length || samples.some(value => !Number.isFinite(value) || value < 0)) {
    throw new Error('Benchmark samples must be finite nonnegative durations');
  }
  const ordered = [...samples].sort((first, second) => first - second);
  const middle = Math.floor(ordered.length / 2);
  return {
    minimum: ordered[0],
    median: ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2,
    maximum: ordered[ordered.length - 1]
  };
}
