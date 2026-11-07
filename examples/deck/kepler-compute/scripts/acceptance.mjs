// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {execFileSync} from 'node:child_process';
import {mkdir, writeFile} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';
import {
  classifyHardwareTier,
  evaluateAcceptance,
  loadAcceptanceTargets,
  parseAcceptanceOptions,
  renderAcceptanceMarkdown,
  summarizeSamples,
  summarizeVerdict
} from './acceptance-utils.mjs';

const EXIT_PASS = 0;
const EXIT_FAIL = 1;
const EXIT_INCOMPLETE = 2;
const EXIT_USAGE = 3;

const USAGE = `Usage: node examples/deck/kepler-compute/scripts/acceptance.mjs [options]
  --sizes LIST          Comma separated row counts (default from targets file)
  --iterations N        Measured samples per interaction kind
  --warmup N            Excluded warmup interactions per kind
  --frame-seconds N     Seconds of continuous pan used for frame-time recording
  --timeout-ms N        Per-stage and per-interaction timeout
  --url URL             Existing Vite host; any query is preserved and ?rows=N is set per size
  --targets PATH        Acceptance targets JSON
  --output PATH         JSON report path; the Markdown report uses the same path with .md
  --tier NAME           Override the detected hardware tier
  --verbose             Log page console output and page errors
  --headed              Use a visible browser instead of headless Chromium
  --label TEXT          Optional human-readable machine/run label
  --help                Print this message
No server is started. Memory figures are registry accounted bytes, not VRAM.
Exit codes: 0 pass, 1 fail, 2 incomplete or workflow error, 3 usage.`;

const INTERACTION_ENDPOINTS = {
  filter: 'submittedFrameMilliseconds',
  brush: 'submittedFrameMilliseconds',
  histogram: 'summaryMilliseconds',
  style: 'submittedFrameMilliseconds',
  summary: 'summaryMilliseconds'
};

/** Metric key and acceptance interaction kind, in measurement order (brush last: it enables brushing). */
const INTERACTION_PLAN = [
  {kind: 'filter', metricKey: 'filterLatency'},
  {kind: 'summary', metricKey: 'linkedSummaryLatency'},
  {kind: 'histogram', metricKey: 'histogramLatency'},
  {kind: 'style', metricKey: 'styleLatency'},
  {kind: 'brush', metricKey: 'brushLatency'}
];

let options;
try {
  options = parseAcceptanceOptions(process.argv.slice(2));
} catch (error) {
  console.error(`${error.message}\n\n${USAGE}`);
  process.exit(EXIT_USAGE);
}
if (options.help) {
  console.log(USAGE);
  process.exit(EXIT_PASS);
}

/** Preserve the existing query of --url and set rows for this size. */
function createSizeUrl(baseUrl, rows) {
  const url = new URL(baseUrl);
  url.searchParams.set('rows', String(rows));
  // Million-row fixtures exceed the 512 MiB registry default; honor an explicit query override.
  if (rows >= 1000000 && !url.searchParams.has('memoryBudgetMebibytes')) {
    url.searchParams.set('memoryBudgetMebibytes', '32768');
  }
  return url.href;
}

function readGitRevision() {
  try {
    const revision = execFileSync('git', ['rev-parse', 'HEAD'], {encoding: 'utf8'}).trim();
    const dirty = Boolean(
      execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], {
        encoding: 'utf8'
      }).trim()
    );
    return dirty ? `${revision}-dirty` : revision;
  } catch {
    return null;
  }
}

/** Deterministic, varying inputs so consecutive samples always change state. */
function createFilterRange(index) {
  const width = 20 + ((index * 13) % 60);
  const minimum = (index * 7) % (100 - width);
  return [minimum, minimum + width];
}

function createCategorySelection(index) {
  const first = index % 4;
  return [first, (first + 1) % 4];
}

/** Evaluate in page: bracket an action with beginInteraction and poll the needed endpoint. */
async function runBracketedInteraction(page, kind, actionName, argument, timeoutMilliseconds) {
  return page.evaluate(
    ({kind, endpointKey, actionName, argument, timeoutMilliseconds}) => {
      const acceptance = window.__keplerComputeAcceptance;
      const api = window.__keplerComputeTest;
      const identifier = acceptance.beginInteraction(kind);
      try {
        if (actionName === 'filter' || actionName === 'summary') api.setRange(...argument);
        else if (actionName === 'histogram') api.setCategories(argument);
        else if (actionName === 'style') api.setPointMarkerFill(argument.layerId, argument.filled);
        else throw new Error(`Unknown action ${actionName}`);
      } catch (error) {
        return {error: `Action failed: ${error.message}`};
      }
      return new Promise(resolvePromise => {
        const started = performance.now();
        const poll = () => {
          const state = api.getState();
          if (state.error) return resolvePromise({error: state.error});
          if (document.visibilityState !== 'visible')
            return resolvePromise({error: 'Benchmark page became hidden'});
          const interaction = acceptance.getInteraction(identifier);
          const value = interaction?.[endpointKey];
          if (typeof value === 'number') return resolvePromise({milliseconds: value});
          if (performance.now() - started > timeoutMilliseconds)
            return resolvePromise({error: `Interaction ${kind} timed out waiting for ${endpointKey}`});
          setTimeout(poll, 2);
        };
        poll();
      });
    },
    {kind, endpointKey: INTERACTION_ENDPOINTS[kind], actionName, argument, timeoutMilliseconds}
  );
}

/** Brush is pointer driven (render only): begin in page, move the real mouse, then poll. */
async function runBrushInteraction(page, index, timeoutMilliseconds) {
  const box = await page.evaluate(() => {
    const canvases = [...document.querySelectorAll('canvas')].map(canvas => {
      const rectangle = canvas.getBoundingClientRect();
      return {x: rectangle.x, y: rectangle.y, width: rectangle.width, height: rectangle.height};
    });
    canvases.sort((first, second) => second.width * second.height - first.width * first.height);
    return canvases[0] ?? null;
  });
  if (!box) return {error: 'No map canvas found for brush interaction'};
  const angle = index * 0.9;
  const x = box.x + box.width * (0.5 + 0.3 * Math.cos(angle));
  const y = box.y + box.height * (0.5 + 0.3 * Math.sin(angle));
  const identifier = await page.evaluate(() =>
    window.__keplerComputeAcceptance.beginInteraction('brush')
  );
  await page.mouse.move(x, y);
  return page.evaluate(
    ({identifier, timeoutMilliseconds}) =>
      new Promise(resolvePromise => {
        const started = performance.now();
        const poll = () => {
          const state = window.__keplerComputeTest.getState();
          if (state.error) return resolvePromise({error: state.error});
          if (document.visibilityState !== 'visible')
            return resolvePromise({error: 'Benchmark page became hidden'});
          const value =
            window.__keplerComputeAcceptance.getInteraction(identifier)?.submittedFrameMilliseconds;
          if (typeof value === 'number') return resolvePromise({milliseconds: value});
          if (performance.now() - started > timeoutMilliseconds)
            return resolvePromise({error: 'Interaction brush timed out waiting for submitted frame'});
          setTimeout(poll, 2);
        };
        poll();
      }),
    {identifier, timeoutMilliseconds}
  );
}

async function readMemorySnapshot(page) {
  return page.evaluate(() => window.__keplerComputeAcceptance.getMemorySnapshot());
}

/** Frame time: record submitted frames while dragging the map in circles (continuous redraws). */
async function recordFrameIntervals(page, frameSeconds) {
  const box = await page.evaluate(() => {
    const canvases = [...document.querySelectorAll('canvas')].map(canvas => {
      const rectangle = canvas.getBoundingClientRect();
      return {x: rectangle.x, y: rectangle.y, width: rectangle.width, height: rectangle.height};
    });
    canvases.sort((first, second) => second.width * second.height - first.width * first.height);
    return canvases[0] ?? null;
  });
  if (!box) return {error: 'No map canvas found for frame recording'};
  const centerX = box.x + box.width / 2;
  const centerY = box.y + box.height / 2;
  await page.evaluate(() => window.__keplerComputeAcceptance.startFrameRecording());
  await page.mouse.move(centerX, centerY);
  await page.mouse.down();
  const endTime = Date.now() + frameSeconds * 1000;
  for (let step = 0; Date.now() < endTime; step++) {
    const angle = step * 0.15;
    await page.mouse.move(
      centerX + box.width * 0.1 * Math.cos(angle),
      centerY + box.height * 0.1 * Math.sin(angle)
    );
    await page.waitForTimeout(8);
  }
  await page.mouse.up();
  const recording = await page.evaluate(() => window.__keplerComputeAcceptance.stopFrameRecording());
  return {intervals: recording.frameIntervalsMilliseconds};
}

function subtractCounters(before, after) {
  if (typeof before !== 'number' || typeof after !== 'number' || after < before) return null;
  return after - before;
}

/** Measure one size in a fresh browser context; never converts missing values to zero. */
async function measureSize(browser, rows, state) {
  const notes = [];
  const rawSamples = {};
  const metrics = {
    filterLatency: null,
    brushLatency: null,
    histogramLatency: null,
    styleLatency: null,
    linkedSummaryLatency: null,
    frameTime: null,
    uploadTime: null,
    memoryAccounted: null,
    warmUploadBytes: null,
    warmReadbackBytes: null
  };
  const context = await browser.newContext({viewport: {width: 1600, height: 1100}});
  try {
    const page = await context.newPage();
    page.setDefaultTimeout(options.timeoutMilliseconds);
    const pageErrors = [];
    page.on('pageerror', error => {
      pageErrors.push(error.stack ?? error.message);
      if (options.verbose) console.error(`[pageerror] ${error.stack ?? error.message}`);
    });
    if (options.verbose)
      page.on('console', message => console.log(`[console.${message.type()}] ${message.text()}`));
    page.on('console', message => {
      if (
        message.type() === 'error' &&
        /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error|memory budget/i.test(message.text())
      )
        pageErrors.push(message.text());
    });
    const {pinViteModules} = await import('./pin-vite-modules.mjs');
    await pinViteModules(page);
    // Long budget for the large sizes: ready includes generating and uploading all rows.
    const readyTimeout = Math.max(options.timeoutMilliseconds, rows >= 1000000 ? 600000 : 0);
    try {
      await page.goto(createSizeUrl(options.url, rows), {waitUntil: 'domcontentloaded'});
      await page.waitForFunction(
        expectedRows => {
          const overlay = document.querySelector('vite-error-overlay');
          if (overlay) throw new Error(overlay.shadowRoot?.textContent || 'Vite module error');
          const appState = window.__keplerComputeTest?.getState();
          if (appState?.error) throw new Error(appState.error);
          return (
            Boolean(window.__keplerComputeAcceptance) &&
            appState?.ready &&
            appState.selectedCount === expectedRows
          );
        },
        rows,
        {timeout: readyTimeout, polling: 250}
      );
    } catch (error) {
      metrics.blocked = {reason: `Admission or load failed: ${error.message.split('\n')[0]}`};
      if (pageErrors.length) notes.push(...pageErrors.slice(0, 3));
      return {rows, metrics, rawSamples, notes};
    }

    const uploadMeasures = await page.evaluate(() =>
      window.__keplerComputeAcceptance.getUploadMeasures()
    );
    const uploadMeasure =
      uploadMeasures.find(measure => measure.name === 'publish-source') ??
      uploadMeasures.find(measure => measure.name === 'runtime-source');
    if (uploadMeasure) {
      metrics.uploadTime = {milliseconds: uploadMeasure.durationMilliseconds};
      notes.push(`uploadTime taken from the ${uploadMeasure.name} upload measure.`);
    } else notes.push('No publish-source or runtime-source upload measure was recorded; uploadTime is unavailable.');

    if (!state.deviceReport) {
      state.deviceReport = await page.evaluate(() =>
        window.__keplerComputeAcceptance.getDeviceReport()
      );
    }

    let peakAllocatedBytes = null;
    const observeMemory = async () => {
      const snapshot = await readMemorySnapshot(page);
      if (typeof snapshot.registryAllocatedBytes === 'number')
        peakAllocatedBytes = Math.max(peakAllocatedBytes ?? 0, snapshot.registryAllocatedBytes);
      return snapshot;
    };
    await observeMemory();

    // Prepare: one point layer for style changes (native config round trip).
    const layerId = await page.evaluate(() => {
      const configuration = window.__keplerComputeTest.saveNativeConfig();
      return (
        configuration.config.visState.layers.find(layer => layer.type === 'point')?.id ?? null
      );
    });
    if (!layerId) notes.push('No native point layer found; style interaction is unavailable.');

    const runSample = (kind, index) => {
      if (kind === 'filter' || kind === 'summary')
        return runBracketedInteraction(
          page,
          kind,
          kind,
          createFilterRange(index),
          options.timeoutMilliseconds
        );
      if (kind === 'histogram')
        return runBracketedInteraction(
          page,
          kind,
          'histogram',
          createCategorySelection(index),
          options.timeoutMilliseconds
        );
      if (kind === 'style') {
        if (!layerId) return {error: 'No native point layer'};
        return runBracketedInteraction(
          page,
          kind,
          'style',
          {layerId, filled: index % 2 === 1},
          options.timeoutMilliseconds
        );
      }
      return runBrushInteraction(page, index, options.timeoutMilliseconds);
    };

    const resetFilters = () =>
      page.evaluate(() => {
        window.__keplerComputeTest.setRange(0, 100);
        window.__keplerComputeTest.setCategories([]);
      });

    const enableBrush = async () => {
      await page.evaluate(() => {
        const configuration = window.__keplerComputeTest.saveNativeConfig();
        const interaction = configuration.config.visState.interactionConfig;
        interaction.brush = {...interaction.brush, enabled: true, config: {size: 5}};
        window.__keplerComputeTest.applyNativeConfig(configuration);
      });
      await page.waitForFunction(() => window.__keplerComputeTest.getState().ready);
    };

    const failures = {};
    const recordFailure = (kind, message) => {
      failures[kind] ??= message;
    };

    // Warmups first (excluded), so the measured window below is warm for every kind.
    for (const {kind} of INTERACTION_PLAN) {
      if (kind === 'brush') await enableBrush();
      for (let index = 0; index < options.warmup; index++) {
        const result = await runSample(kind, index);
        if (result.error) recordFailure(kind, result.error);
        await observeMemory();
      }
      if (kind !== 'brush') await resetFilters();
    }

    // Measured pass; brushing is enabled by now (render-only), which also applies to the later pan.
    const before = await observeMemory();
    for (const {kind, metricKey} of INTERACTION_PLAN) {
      const samples = [];
      for (let index = 0; index < options.iterations; index++) {
        const result = await runSample(kind, options.warmup + index);
        if (typeof result.milliseconds === 'number') samples.push(result.milliseconds);
        else recordFailure(kind, result.error ?? 'No measurement');
        await observeMemory();
      }
      rawSamples[metricKey] = samples;
      metrics[metricKey] = summarizeSamples(samples);
      if (kind !== 'brush') await resetFilters();
    }
    const after = await observeMemory();
    metrics.warmUploadBytes = (() => {
      const bytes = subtractCounters(before.sourceUploadBytes, after.sourceUploadBytes);
      return bytes === null ? null : {bytes};
    })();
    // Targets are per interaction batch, so normalize the measured pass total by its interaction count.
    metrics.warmReadbackBytes = (() => {
      const totalBytes = subtractCounters(before.explorerReadbackBytes, after.explorerReadbackBytes);
      if (totalBytes === null) return null;
      const interactionCount = INTERACTION_PLAN.length * options.iterations;
      notes.push(
        `warmReadbackBytes: ${totalBytes} B read back over ${interactionCount} measured interactions (reported per interaction).`
      );
      return {bytes: Math.ceil(totalBytes / interactionCount)};
    })();

    const frames = await recordFrameIntervals(page, options.frameSeconds);
    if (frames.error) notes.push(frames.error);
    else {
      rawSamples.frameTime = frames.intervals;
      metrics.frameTime = summarizeSamples(frames.intervals);
    }
    await observeMemory();
    if (peakAllocatedBytes !== null) metrics.memoryAccounted = {accountedBytes: peakAllocatedBytes};
    for (const [kind, message] of Object.entries(failures))
      notes.push(`${kind}: first failure: ${message}`);
    if (pageErrors.length) notes.push(`Page errors: ${pageErrors.slice(0, 3).join(' | ')}`);
    return {rows, metrics, rawSamples, notes};
  } catch (error) {
    metrics.blocked ??= {reason: `Workflow error: ${error.message.split('\n')[0]}`};
    notes.push(error.stack ?? error.message);
    return {rows, metrics, rawSamples, notes};
  } finally {
    await context.close().catch(() => {});
  }
}

async function writeReports(report) {
  const jsonPath = resolve(options.output);
  const markdownPath = jsonPath.replace(/\.json$/i, '') + '.md';
  await mkdir(dirname(jsonPath), {recursive: true});
  await writeFile(jsonPath, `${JSON.stringify(report, null, 2)}\n`);
  await writeFile(markdownPath, renderAcceptanceMarkdown(report));
  return {jsonPath, markdownPath};
}

function extractVerdict(runs) {
  const summary = summarizeVerdict(runs);
  return String(typeof summary === 'object' && summary !== null ? summary.verdict : summary);
}

let exitCode = EXIT_INCOMPLETE;
const report = {
  schemaVersion: 1,
  label: options.label ?? null,
  generatedAt: new Date().toISOString(),
  environment: {
    browser: null,
    deviceReport: null,
    tier: null,
    tierReasons: [],
    gitRevision: readGitRevision()
  },
  runs: []
};
let browser;
try {
  const config = await loadAcceptanceTargets(options.targets);
  // Lazy imports: --help above never reaches playwright.
  const {chromium} = await import('playwright');
  const {getPlaywrightLaunchOptions} = await import(
    '../../../../scripts/playwright/get-playwright-launch-options.mjs'
  );
  const launchOptions = getPlaywrightLaunchOptions({
    headless: !options.headed,
    backend: 'webgpu-core'
  });
  browser = await chromium.launch(launchOptions);
  report.environment.browser = `Chromium ${browser.version()}${options.headed ? ' (headed)' : ' (headless)'}`;
  const state = {deviceReport: null};
  const measurements = [];
  for (const rows of options.sizes) {
    console.log(`Measuring ${rows} rows...`);
    measurements.push(await measureSize(browser, rows, state));
    if (!report.environment.tier && state.deviceReport) {
      const classification = options.tierOverride
        ? {tier: options.tierOverride, reasons: ['Overridden with --tier']}
        : classifyHardwareTier(state.deviceReport, config.tiers);
      report.environment.deviceReport = state.deviceReport;
      report.environment.tier = classification.tier;
      report.environment.tierReasons = classification.reasons;
    }
  }
  if (!report.environment.tier) {
    report.environment.tier = options.tierOverride ?? 'unknown';
    report.environment.tierReasons = ['No device report was available'];
    report.environment.deviceReport = state.deviceReport;
  }
  for (const measurement of measurements) {
    const evaluation = evaluateAcceptance({
      tier: report.environment.tier,
      rows: measurement.rows,
      metrics: measurement.metrics,
      targets: config.targets
    });
    report.runs.push({
      rows: measurement.rows,
      evaluation,
      rawSamples: measurement.rawSamples,
      notes: measurement.metrics.blocked
        ? [`Blocked: ${measurement.metrics.blocked.reason}`, ...measurement.notes]
        : measurement.notes
    });
  }
  const verdict = extractVerdict(report.runs);
  report.verdict = verdict;
  exitCode = verdict === 'pass' ? EXIT_PASS : verdict === 'fail' ? EXIT_FAIL : EXIT_INCOMPLETE;
} catch (error) {
  console.error(error.stack ?? error.message);
  report.workflowError = error.stack ?? error.message;
  report.verdict = 'incomplete';
  exitCode = EXIT_INCOMPLETE;
} finally {
  await browser?.close().catch(() => {});
}
try {
  const paths = await writeReports(report);
  console.log(`Wrote ${paths.jsonPath}\nWrote ${paths.markdownPath}`);
} catch (error) {
  console.error(`Failed to write reports: ${error.message}`);
  exitCode = EXIT_INCOMPLETE;
}
console.log(`Overall verdict: ${report.verdict}`);
process.exit(exitCode);
