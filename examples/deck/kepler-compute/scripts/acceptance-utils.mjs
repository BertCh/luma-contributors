// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const DEFAULT_TARGETS_PATH = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  'acceptance-targets.json'
);
const METRIC_KEYS = [
  'filterLatency',
  'brushLatency',
  'histogramLatency',
  'styleLatency',
  'linkedSummaryLatency',
  'frameTime',
  'uploadTime',
  'memoryAccounted',
  'warmUploadBytes',
  'warmReadbackBytes'
];
const STATISTICS = ['p50', 'p95', 'max', 'value'];
const ORIGINS = ['plan', 'ratified', 'derived', 'provisional'];
const UNITS = ['ms', 'bytes'];

/** Bounded command-line options; importing this helper never starts a browser. */
export function parseAcceptanceOptions(argumentsList) {
  const options = {
    sizes: [1000000],
    iterations: 20,
    warmup: 3,
    frameSeconds: 5,
    timeoutMilliseconds: 300000,
    url: 'http://127.0.0.1:5318/',
    targets: DEFAULT_TARGETS_PATH,
    output: path.resolve('acceptance-report.json'),
    tierOverride: undefined,
    headed: false,
    verbose: false,
    label: null,
    help: false
  };
  const names = new Map([
    ['--sizes', 'sizes'],
    ['--iterations', 'iterations'],
    ['--warmup', 'warmup'],
    ['--frame-seconds', 'frameSeconds'],
    ['--timeout-ms', 'timeoutMilliseconds'],
    ['--url', 'url'],
    ['--targets', 'targets'],
    ['--output', 'output'],
    ['--tier', 'tierOverride'],
    ['--label', 'label']
  ]);
  const textKeys = ['url', 'targets', 'output', 'tierOverride', 'label'];
  for (let index = 0; index < argumentsList.length; index++) {
    const argument = argumentsList[index];
    if (argument === '--headed') options.headed = true;
    else if (argument === '--verbose') options.verbose = true;
    else if (argument === '--help') options.help = true;
    else {
      const key = names.get(argument);
      const value = argumentsList[++index];
      if (!key || !value || value.startsWith('--'))
        throw new Error(`Unknown option or missing value: ${argument}`);
      if (key === 'sizes') options.sizes = value.split(',').map(Number);
      else options[key] = textKeys.includes(key) ? value : Number(value);
    }
  }
  if (options.help) return options;
  for (const [key, minimum, maximum] of [
    ['iterations', 1, 100],
    ['warmup', 0, 20],
    ['frameSeconds', 1, 60],
    ['timeoutMilliseconds', 1000, 3600000]
  ]) {
    if (!Number.isInteger(options[key]) || options[key] < minimum || options[key] > maximum)
      throw new Error(`${key} must be an integer in [${minimum}, ${maximum}]`);
  }
  const url = new URL(options.url);
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('--url must use HTTP or HTTPS');
  options.url = url.href;
  const configuration = loadAcceptanceTargets(options.targets);
  if (!options.sizes.length || new Set(options.sizes).size !== options.sizes.length)
    throw new Error('--sizes must be a nonempty list without duplicates');
  for (const size of options.sizes) {
    if (!configuration.sizes.includes(size))
      throw new Error(`--sizes entries must be one of ${configuration.sizes.join(', ')}`);
  }
  if (options.tierOverride && !configuration.tiers.some(tier => tier.id === options.tierOverride))
    throw new Error(`--tier must be one of ${configuration.tiers.map(tier => tier.id).join(', ')}`);
  options.markdownOutput = options.output.replace(/\.[^./\\]*$/, '') + '.md';
  return options;
}

/** Loads and validates the acceptance target configuration. */
export function loadAcceptanceTargets(filePath) {
  const configuration = JSON.parse(readFileSync(filePath, 'utf8'));
  const fail = message => {
    throw new Error(`Invalid acceptance targets: ${message}`);
  };
  if (configuration?.version !== 1) fail('version must be 1');
  const {sizes, tiers, targets} = configuration;
  if (!Array.isArray(sizes) || !sizes.length || !sizes.every(Number.isInteger))
    fail('sizes must be a nonempty integer array');
  if (!Array.isArray(tiers) || !tiers.length) fail('tiers must be a nonempty array');
  const tierIds = new Set();
  for (const tier of tiers) {
    if (typeof tier?.id !== 'string' || !tier.id || tierIds.has(tier.id))
      fail('tier ids must be unique nonempty strings');
    if (!tier.match || typeof tier.match !== 'object') fail(`tier ${tier.id} needs a match object`);
    tierIds.add(tier.id);
  }
  const last = tiers.at(-1);
  if (last.id !== 'unclassified' || Object.keys(last.match).length)
    fail('the final tier must be unclassified with an empty match');
  if (!targets || typeof targets !== 'object') fail('targets must be an object');
  for (const [tierId, byRows] of Object.entries(targets)) {
    if (!tierIds.has(tierId)) fail(`targets reference unknown tier ${tierId}`);
    for (const [rows, byMetric] of Object.entries(byRows)) {
      if (!sizes.includes(Number(rows))) fail(`${tierId} has targets for unlisted size ${rows}`);
      for (const [metric, spec] of Object.entries(byMetric)) {
        const where = `${tierId}/${rows}/${metric}`;
        if (!METRIC_KEYS.includes(metric)) fail(`unknown metric ${where}`);
        if (!STATISTICS.includes(spec?.statistic)) fail(`bad statistic at ${where}`);
        if (!Number.isFinite(spec.maxValue) || spec.maxValue < 0) fail(`bad maxValue at ${where}`);
        if (!UNITS.includes(spec.unit)) fail(`bad unit at ${where}`);
        if (!ORIGINS.includes(spec.origin)) fail(`bad origin at ${where}`);
        if (spec.note !== undefined && typeof spec.note !== 'string') fail(`bad note at ${where}`);
      }
    }
  }
  return configuration;
}

function includesAny(text, needles) {
  return needles.some(needle => text.includes(String(needle).toLowerCase()));
}

/** Evaluates one rule; every specified key must hold. Returns reasons or null. */
function matchRule(rule, context) {
  const reasons = [];
  const arrayKeys = ['vendor', 'architecture', 'device', 'description', 'text'];
  for (const key of arrayKeys) {
    const needles = rule[key];
    if (!needles?.length) continue;
    const haystack = key === 'text' ? context.text : context.fields[key];
    if (!includesAny(haystack, needles)) return null;
    reasons.push(`${key} matches ${needles.join('|')}`);
  }
  if (rule.excludeText?.length && includesAny(context.text, rule.excludeText)) return null;
  if (rule.userAgent?.length) {
    if (!includesAny(context.userAgent, rule.userAgent)) return null;
    reasons.push(`userAgent matches ${rule.userAgent.join('|')}`);
  }
  if (rule.excludeUserAgent?.length && includesAny(context.userAgent, rule.excludeUserAgent))
    return null;
  if (rule.isFallbackAdapter !== undefined) {
    if (context.isFallbackAdapter !== rule.isFallbackAdapter) return null;
    reasons.push('fallback adapter');
  }
  for (const [key, limit] of [
    ['minMaxBufferSize', 'maxBufferSize'],
    ['minMaxStorageBufferBindingSize', 'maxStorageBufferBindingSize']
  ]) {
    if (rule[key] === undefined) continue;
    const value = context.limits[limit];
    if (!Number.isFinite(value) || value < rule[key]) return null;
    reasons.push(`${limit} ${value} >= ${rule[key]}`);
  }
  if (rule.maxMaxStorageBufferBindingSize !== undefined) {
    const value = context.limits.maxStorageBufferBindingSize;
    if (!Number.isFinite(value) || value > rule.maxMaxStorageBufferBindingSize) return null;
    reasons.push(`maxStorageBufferBindingSize ${value} <= ${rule.maxMaxStorageBufferBindingSize}`);
  }
  return reasons.length ? reasons : ['rule matched'];
}

/** First matching tier wins; tiers are ordered most capable first. */
export function classifyHardwareTier(deviceReport, tierDefinitions) {
  const info = deviceReport?.adapterInfo ?? {};
  const fields = Object.fromEntries(
    ['vendor', 'architecture', 'device', 'description'].map(key => [
      key,
      typeof info[key] === 'string' ? info[key].toLowerCase() : ''
    ])
  );
  const context = {
    fields,
    text: Object.values(fields).join(' '),
    userAgent: String(deviceReport?.userAgent ?? '').toLowerCase(),
    limits: deviceReport?.limits ?? {},
    isFallbackAdapter: info.isFallbackAdapter === true
  };
  for (const tier of tierDefinitions) {
    const match = tier.match ?? {};
    const rules = match.anyOf ?? (Object.keys(match).length ? [match] : null);
    if (!rules) return {tier: tier.id, reasons: ['no other tier matched']};
    for (const rule of rules) {
      const reasons = matchRule(rule, context);
      if (reasons) return {tier: tier.id, reasons};
    }
  }
  return {tier: 'unclassified', reasons: ['no other tier matched']};
}

/** Nearest-rank statistics; never interpolates an unobserved sample. */
export function summarizeSamples(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((first, second) => first - second);
  const percentile = fraction => sorted[Math.max(0, Math.ceil(fraction * sorted.length) - 1)];
  return {
    count: sorted.length,
    p50: percentile(0.5),
    p95: percentile(0.95),
    min: sorted[0],
    max: sorted.at(-1),
    mean: sorted.reduce((sum, value) => sum + value, 0) / sorted.length
  };
}

function readObservation(metricValue, statistic) {
  if (metricValue === null || metricValue === undefined || typeof metricValue !== 'object')
    return null;
  const candidate =
    statistic === 'value'
      ? (metricValue.milliseconds ??
        metricValue.accountedBytes ??
        metricValue.bytes ??
        metricValue.value)
      : metricValue[statistic];
  return Number.isFinite(candidate) ? candidate : null;
}

/** Compares observations to the tier targets; missing data is unavailable, never zero. */
export function evaluateAcceptance({tier, rows, metrics, targets}) {
  const tierTargets = targets?.[tier]?.[String(rows)];
  const results = [];
  const missingTargets = [];
  const blocked = metrics?.blocked;
  for (const metric of METRIC_KEYS) {
    const spec = tierTargets?.[metric];
    if (!spec) {
      missingTargets.push(metric);
      continue;
    }
    const entry = {
      metric,
      statistic: spec.statistic,
      comparator: '<=',
      target: spec.maxValue,
      unit: spec.unit,
      origin: spec.origin,
      observed: null,
      status: 'unavailable'
    };
    if (blocked) {
      entry.status = 'blocked';
      entry.note = blocked.reason;
    } else {
      const observed = readObservation(metrics?.[metric], spec.statistic);
      if (observed === null) {
        entry.note = 'No observation recorded';
      } else {
        entry.observed = observed;
        entry.status = observed <= spec.maxValue ? 'pass' : 'fail';
      }
    }
    results.push(entry);
  }
  const count = status => results.filter(result => result.status === status).length;
  const summary = {
    passed: count('pass'),
    failed: count('fail'),
    unavailable: count('unavailable'),
    blocked: count('blocked')
  };
  const verdict = summary.failed
    ? 'fail'
    : results.length && !summary.unavailable && !summary.blocked
      ? 'pass'
      : 'incomplete';
  return {tier, rows, results, summary, verdict, missingTargets};
}

/** Any failure fails; otherwise any incomplete run (or no runs) is incomplete. */
export function summarizeVerdict(runs) {
  const verdicts = runs.map(run => run.evaluation.verdict);
  if (verdicts.includes('fail')) return 'fail';
  return verdicts.length && verdicts.every(verdict => verdict === 'pass') ? 'pass' : 'incomplete';
}

function formatValue(value, unit) {
  if (value === null || value === undefined) return 'n/a';
  if (unit === 'bytes') return `${value} B`;
  return `${Math.round(value * 100) / 100} ms`;
}

/** Renders a human-readable report; the disclaimers are part of the contract. */
export function renderAcceptanceMarkdown(report) {
  const {environment, runs} = report;
  const lines = [
    `# Kepler compute acceptance report${report.label ? `: ${report.label}` : ''}`,
    '',
    `Generated: ${report.generatedAt}`,
    '',
    '## Environment',
    '',
    '| Field | Value |',
    '| --- | --- |',
    `| Browser | ${environment.browser ?? 'unknown'} |`,
    `| Hardware tier | ${environment.tier} |`,
    `| Tier reasons | ${(environment.tierReasons ?? []).join('; ')} |`,
    `| Git revision | ${environment.gitRevision ?? 'unknown'} |`,
    `| Adapter | ${JSON.stringify(environment.deviceReport?.adapterInfo ?? {})} |`,
    ''
  ];
  for (const run of runs) {
    const {evaluation} = run;
    lines.push(
      `## ${run.rows.toLocaleString('en-US')} rows (${evaluation.verdict})`,
      '',
      '| Metric | Statistic | Target | Observed | Origin | Status |',
      '| --- | --- | --- | --- | --- | --- |'
    );
    for (const result of evaluation.results) {
      const status = result.note ? `${result.status} (${result.note})` : result.status;
      lines.push(
        `| ${result.metric} | ${result.statistic} | ${result.comparator} ${formatValue(result.target, result.unit)} | ${formatValue(result.observed, result.unit)} | ${result.origin} | ${status} |`
      );
    }
    lines.push('');
    if (evaluation.missingTargets?.length)
      lines.push(`Missing targets: ${evaluation.missingTargets.join(', ')}`, '');
    for (const note of run.notes ?? []) lines.push(`- ${note}`);
    if (run.notes?.length) lines.push('');
  }
  lines.push(
    '## Overall verdict',
    '',
    `**${summarizeVerdict(runs)}**`,
    '',
    '## Disclaimers',
    '',
    '- Frame and latency observations are host-submitted-frame timings. They are not GPU completion or physical presentation latency.',
    '- Origin `plan` targets come from the integration plan; origin `ratified` targets were scaled from it and ratified by the project owner. Any `provisional` or `derived` target is not yet accepted.',
    ''
  );
  return lines.join('\n');
}
