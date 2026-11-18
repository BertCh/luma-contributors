// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {execFileSync} from 'node:child_process';
import {existsSync, readdirSync, readFileSync} from 'node:fs';
import {join} from 'node:path';

/** Sample resident bytes for a process and its descendants, using ps RSS in KiB. */
export function readTreeRSS(processId) {
  const records = execFileSync('ps', ['-axo', 'pid=,ppid=,rss='], {encoding: 'utf8'})
    .trim()
    .split('\n')
    .map(line => line.trim().split(/\s+/).map(Number));
  const descendants = new Set([processId]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const [childId, parentId] of records) {
      if (descendants.has(parentId) && !descendants.has(childId)) {
        descendants.add(childId);
        changed = true;
      }
    }
  }
  return records.reduce(
    (total, [childId, , residentKilobytes]) =>
      total + (descendants.has(childId) ? residentKilobytes * 1024 : 0),
    0
  );
}

export const BENCHMARK_PATTERNS = [
  'modules/**/*-bench.spec.{ts,js}',
  'modules/**/*-benchmark.spec.{ts,js}'
];
export const isBenchmark = file => /-(?:bench|benchmark)\.spec\.(?:ts|js)$/.test(file);

const OPT_IN_TEST_JOBS = [
  {
    file: 'modules/analysis/test/gpu-spatial-analysis/delaunay-tessellation/ab/delaunay-ab.spec.ts',
    flag: 'A5_AB',
    value: '1'
  },
  {
    file: 'modules/analysis/test/gpu-spatial-analysis/polygon-overlay/ab/overlay-ab.spec.ts',
    flag: 'A5_AB',
    value: '1'
  },
  {
    file: 'modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/vecchia-float32-gpu-study.spec.ts',
    flag: 'VITE_VECCHIA_FLOAT32_STUDY'
  },
  {
    file: 'modules/lenses/test/computation-lenses/capture-records.spec.ts',
    flag: 'VITE_CAPTURE_LENS_RECORDS',
    value: '1'
  }
];

/** Eligibility mirrors each job's explicit entry gate; ordinary correctness files never opt out. */
export function getOptInTestExclusions(environment = process.env) {
  return OPT_IN_TEST_JOBS.filter(job =>
    job.value ? environment[job.flag] !== job.value : !environment[job.flag]
  ).map(job => job.file);
}

export function getOptInBrowserDefines(environment = process.env) {
  return {'import.meta.env.LUMA_A5_AB': JSON.stringify(environment.A5_AB ?? '')};
}

function readNodeHeapOptions(callerOptions) {
  const tokens = [];
  let token = '';
  let quoted = false;
  for (let index = 0; index < callerOptions.length; index++) {
    const character = callerOptions[index];
    if (character === '"') quoted = !quoted;
    else if (character === '\\' && quoted && index + 1 < callerOptions.length)
      token += callerOptions[++index];
    else if (/\s/.test(character) && !quoted) {
      if (token) tokens.push(token);
      token = '';
    } else token += character;
  }
  if (token) tokens.push(token);
  const heapOptions = [];
  for (let index = 0; index < tokens.length; index++) {
    const [rawName, inlineValue] = tokens[index].split('=');
    const name = rawName.replaceAll('_', '-');
    if (!['--max-old-space-size', '--max-old-space-size-percentage'].includes(name)) continue;
    const value = Number(inlineValue ?? tokens[index + 1]);
    heapOptions.push({name, value});
  }
  return heapOptions;
}

/** Preserve Node's original option text; normalize only for heap-policy decisions. */
export function resolveNodeHeapOptions(callerOptions = '', maximumHeapMegabytes) {
  const heapOptions = readNodeHeapOptions(callerOptions);
  for (const {name, value} of heapOptions) {
    if (
      maximumHeapMegabytes !== undefined &&
      (name.endsWith('-percentage') ||
        !Number.isSafeInteger(value) ||
        value < 1 ||
        value > maximumHeapMegabytes)
    ) {
      throw new Error(
        `verify allows a V8 heap of at most ${maximumHeapMegabytes} MiB; use an absolute heap limit in NODE_OPTIONS`
      );
    }
  }
  return heapOptions.length ? callerOptions : `${callerOptions} --max-old-space-size=8192`.trim();
}

/** Keep only a validated numeric heap argument when isolating a worker from loader options. */
export function getIsolatedNodeHeapArguments(callerOptions = '') {
  const validatedOptions = resolveNodeHeapOptions(callerOptions, 8192);
  const heap = readNodeHeapOptions(validatedOptions).at(-1);
  return [`--max-old-space-size=${heap.value}`];
}

/** Forward the effective wrapper heap, retaining environment options and CLI heap precedence. */
export function getChildNodeOptions(
  environmentOptions = process.env.NODE_OPTIONS ?? '',
  executableArguments = process.execArgv
) {
  const callerOptions = [
    environmentOptions,
    ...executableArguments.map(argument => JSON.stringify(argument))
  ].join(' ');
  return [environmentOptions, ...getIsolatedNodeHeapArguments(callerOptions)]
    .filter(Boolean)
    .join(' ');
}

/** Same exclusions as vitest.config.ts, without starting Vite or a browser. */
export function listTestFiles(rootDirectory, kind = 'headless', environment = process.env) {
  const exclusionFile = join(rootDirectory, 'test/kepler-compute-example-specs.json');
  const excluded = [
    ...getOptInTestExclusions(environment),
    ...(environment.LUMA_CONTRIBUTORS_RUN_EXCLUDED_SPECS === 'true' || !existsSync(exclusionFile)
      ? []
      : Object.keys(JSON.parse(readFileSync(exclusionFile, 'utf8')).excludedFromRuns ?? {}))
  ];
  function visit(directory) {
    if (!existsSync(join(rootDirectory, directory))) return [];
    return readdirSync(join(rootDirectory, directory), {withFileTypes: true}).flatMap(entry => {
      const path = `${directory}/${entry.name}`;
      if (['node_modules', 'dist', '.git'].includes(entry.name)) return [];
      if (entry.isDirectory()) return visit(path);
      if (
        !/\.spec\.(?:ts|js)$/.test(path) ||
        path.includes('.disabled.') ||
        excluded.includes(path)
      )
        return [];
      if (kind === 'benchmarks') return isBenchmark(path) ? [path] : [];
      if (kind === 'node')
        return path.endsWith('.node.spec.ts') || path.endsWith('.node.spec.js') ? [path] : [];
      return !/\.node\.spec\.(?:ts|js)$/.test(path) && !isBenchmark(path) ? [path] : [];
    });
  }
  return visit('modules').sort();
}

/** A fixed-size queue: at most concurrency transforms/read/write groups are in flight. */
export async function mapWithConcurrency(values, concurrency, callback) {
  if (!Number.isSafeInteger(concurrency) || concurrency < 1)
    throw new Error('concurrency must be a positive integer');
  let nextIndex = 0;
  await Promise.all(
    Array.from({length: Math.min(concurrency, values.length)}, async () => {
      while (nextIndex < values.length) {
        const index = nextIndex++;
        await callback(values[index], index);
      }
    })
  );
}
