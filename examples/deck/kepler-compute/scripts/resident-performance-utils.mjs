// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Bounded command-line options; importing this helper never starts a browser. */
export function parsePerformanceOptions(argumentsList) {
  const options = {
    rows: 100000,
    iterations: 20,
    warmup: 3,
    batchSize: 65536,
    timeoutMilliseconds: 120000,
    url: 'http://127.0.0.1:5318/',
    output: undefined,
    headed: false,
    diagnostics: false,
    interaction: 'create-or-update',
    label: null,
    help: false
  };
  const names = new Map([
    ['--rows', 'rows'],
    ['--iterations', 'iterations'],
    ['--warmup', 'warmup'],
    ['--batch-size', 'batchSize'],
    ['--timeout-ms', 'timeoutMilliseconds'],
    ['--url', 'url'],
    ['--output', 'output'],
    ['--label', 'label'],
    ['--interaction', 'interaction']
  ]);
  for (let index = 0; index < argumentsList.length; index++) {
    const argument = argumentsList[index];
    if (argument === '--headed') options.headed = true;
    else if (argument === '--diagnostics') options.diagnostics = true;
    else if (argument === '--help') options.help = true;
    else {
      const key = names.get(argument);
      const value = argumentsList[++index];
      if (!key || !value || value.startsWith('--'))
        throw new Error(`Unknown option or missing value: ${argument}`);
      options[key] = ['url', 'output', 'label', 'interaction'].includes(key)
        ? value
        : Number(value);
    }
  }
  if (!['create-or-update', 'value-only'].includes(options.interaction))
    throw new Error('--interaction must be create-or-update or value-only');
  if (![100000, 1000000].includes(options.rows))
    throw new Error('--rows must be 100000 or 1000000');
  for (const [key, minimum, maximum] of [
    ['iterations', 5, 100],
    ['warmup', 1, 20],
    ['batchSize', 1024, 1000000],
    ['timeoutMilliseconds', 1000, 300000]
  ]) {
    if (!Number.isInteger(options[key]) || options[key] < minimum || options[key] > maximum)
      throw new Error(`${key} must be an integer in [${minimum}, ${maximum}]`);
  }
  const url = new URL(options.url);
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('--url must use HTTP or HTTPS');
  // Keep the built-in fixture small and constant while measuring the separately published source.
  url.searchParams.set('rows', '256');
  options.url = url.href;
  options.output ??= `/tmp/luma-kepler-resident-performance-${options.rows}-${options.interaction}${options.diagnostics ? '-diagnostics' : ''}.json`;
  return options;
}

/** Deterministic contiguous source batches: three Float32 fields, no normalization or nulls. */
export function createPerformanceColumns(offset, length) {
  const longitude = new Float32Array(length);
  const latitude = new Float32Array(length);
  const value = new Float32Array(length);
  for (let localIndex = 0; localIndex < length; localIndex++) {
    const sourceIndex = offset + localIndex;
    longitude[localIndex] = -74.1 + (sourceIndex % 997) / 5000;
    latitude[localIndex] = 40.65 + (Math.floor(sourceIndex / 997) % 991) / 5000;
    value[localIndex] = sourceIndex % 1000;
  }
  return {longitude, latitude, value};
}

/** Inclusive native range oracle; works across partial periods and batch boundaries. */
export function expectedRangeCount(rowCount, minimum, maximum) {
  const lower = Math.max(0, Math.ceil(minimum));
  const upper = Math.min(999, Math.floor(maximum));
  if (lower > upper) return 0;
  const fullPeriods = Math.floor(rowCount / 1000);
  const remainder = rowCount % 1000;
  return (
    fullPeriods * (upper - lower + 1) + Math.max(0, Math.min(remainder - 1, upper) - lower + 1)
  );
}

/** Nearest-rank quantiles retain raw samples and do not interpolate an unobserved latency. */
export function summarizeDurations(values) {
  if (!values.length) return {available: false, reason: 'No observations'};
  if (values.some(value => !Number.isFinite(value) || value < 0))
    throw new Error('Latency samples must be finite and nonnegative');
  const sorted = [...values].sort((first, second) => first - second);
  const percentile = fraction => sorted[Math.max(0, Math.ceil(fraction * sorted.length) - 1)];
  return {
    available: true,
    unit: 'milliseconds',
    count: sorted.length,
    minimum: sorted[0],
    p50: percentile(0.5),
    p95: percentile(0.95),
    maximum: sorted.at(-1),
    mean: sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
    quantileMethod: 'nearest-rank'
  };
}

export function createInteractionRanges(count) {
  // Different cardinalities prevent an old summary from accidentally satisfying the next sample.
  return Array.from({length: count}, (_, index) => {
    const width = 100 + ((index * 137) % 800);
    const minimum = (index * 53) % (1000 - width);
    return [minimum, minimum + width - 1];
  });
}

/** Missing counters are unavailable, never treated as zero transfers. */
export function counterDelta(before, after) {
  if (typeof before === 'number' && typeof after === 'number')
    return after >= before
      ? {available: true, before, after, delta: after - before}
      : {available: false, reason: 'Counter reset during measurement', before, after};
  if (before && after && typeof before === 'object' && typeof after === 'object')
    return Object.fromEntries(
      [...new Set([...Object.keys(before), ...Object.keys(after)])].map(key => [
        key,
        counterDelta(before[key], after[key])
      ])
    );
  return {available: false, reason: 'Counter not exposed by the host'};
}

/** Leaf CPU samples identify JavaScript hotspots; counts are not GPU or wall-clock durations. */
export function summarizeCpuProfile(profile) {
  const nodes = new Map(profile.nodes.map(node => [node.id, node]));
  const counts = new Map();
  for (const identifier of profile.samples ?? []) {
    const node = nodes.get(identifier);
    if (!node) throw new Error('CPU profile references an unknown node');
    const frame = node.callFrame;
    const key = JSON.stringify([
      frame.functionName,
      frame.url,
      frame.lineNumber,
      frame.columnNumber
    ]);
    const entry = counts.get(key) ?? {
      functionName: frame.functionName || '(anonymous)',
      url: frame.url,
      lineNumber: frame.lineNumber,
      columnNumber: frame.columnNumber,
      samples: 0
    };
    entry.samples++;
    counts.set(key, entry);
  }
  return {
    sampleCount: profile.samples?.length ?? 0,
    observedProfileMilliseconds: (profile.endTime - profile.startTime) / 1000,
    method:
      'CDP V8 leaf CPU samples at requested 1000 microsecond interval; sample counts are not exact function durations. Raw profile retains stacks. Profiling perturbs timings.',
    topLeafFrames: [...counts.values()]
      .sort((first, second) => second.samples - first.samples)
      .slice(0, 40)
  };
}

/** Cumulative counters include every intermediate encode; bounded event retention is explicit. */
export function summarizeEncodeTelemetry(before, after, sourceRevision) {
  if (!before || !after)
    return {available: false, reason: 'Cumulative encode telemetry unavailable'};
  if (
    before.generation !== after.generation ||
    after.count < before.count ||
    after.totalMilliseconds < before.totalMilliseconds
  )
    return {available: false, reason: 'Cumulative encode telemetry reset during interaction'};
  const events = after.events.filter(
    event => event.sequence > before.count && event.sequence <= after.count
  );
  if (events.some(event => event.sourceRevision !== sourceRevision))
    return {available: false, reason: 'Source revision changed during encoded interval'};
  return {
    available: true,
    count: after.count - before.count,
    javascriptMilliseconds: after.totalMilliseconds - before.totalMilliseconds,
    eventsComplete: events.length === after.count - before.count,
    droppedEvents: after.droppedEvents - before.droppedEvents,
    events,
    contract:
      'Whole-workspace JavaScript encoding attributed to this source; includes intermediate revisions and overlapping work for other sources, not GPU execution.'
  };
}
