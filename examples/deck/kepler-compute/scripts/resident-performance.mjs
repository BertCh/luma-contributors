// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdir, writeFile} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';
import {performance as nodePerformance} from 'node:perf_hooks';
import {arch, cpus, platform, release} from 'node:os';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {closeNativeDialog, openNativeData} from './native-navigation.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {
  counterDelta,
  createInteractionRanges,
  createPerformanceColumns,
  expectedRangeCount,
  parsePerformanceOptions,
  summarizeDurations,
  summarizeCpuProfile,
  summarizeEncodeTelemetry
} from './resident-performance-utils.mjs';

const options = parsePerformanceOptions(process.argv.slice(2));
if (options.help) {
  console.log(`Usage: node examples/deck/kepler-compute/scripts/resident-performance.mjs [options]
  --rows 100000|1000000   Published source rows (default 100000)
  --iterations 5..100    Measured sequential interactions (default 20)
  --warmup 1..20         Excluded warmup interactions (default 3)
  --batch-size N         1024..1000000 rows per original Arrow batch (default 65536)
  --timeout-ms N         Per-stage/interaction timeout, 1000..300000 (default 120000)
  --url URL              Existing Vite host (default http://127.0.0.1:5318/)
  --output PATH          JSON report (default filename includes rows, mode and diagnostics)
  --headed              Use a visible browser instead of headless Chromium
  --interaction MODE    create-or-update (default) or value-only native slider action
  --diagnostics         Opt-in stage marks and CPU sampling (30s maximum; perturbs timings)
  --label TEXT          Optional human-readable machine/run label
No server is started. No native CPU baseline or GPU timestamp timing is inferred.`);
  process.exit(0);
}
const unavailable = reason => ({available: false, reason});
const report = {
  schemaVersion: 2,
  status: 'running',
  startedAt: new Date().toISOString(),
  options,
  host: {
    platform: platform(),
    release: release(),
    architecture: arch(),
    cpu: cpus()[0]?.model ?? null
  },
  methodology: {
    scenario:
      'Published native Point layer + inclusive numeric range filter + native Big Number Count',
    cold: 'First source publication in a fresh browser context after the 256-row app fixture is ready; OS, driver, and Vite caches are not cleared.',
    interaction:
      options.interaction === 'value-only'
        ? 'Existing native range filter value-only action; separate from create-or-update baseline.'
        : 'Native createOrUpdateFilter updates dataset, field and value; broader than a slider value update.',
    warm: 'Sequential native filter actions after excluded warmups. Browser performance.now clock; 4ms scheduled polling can be delayed by main-thread load.',
    submittedFrame:
      'Revision-tagged host submission boundary; excludes GPU completion and physical presentation.',
    animationFrame:
      'First requested browser animation callback after observing that submitted frame; a presentation opportunity, not proof of pixels displayed.',
    summary: 'First polling observation of the current revision compact resident count.',
    linkedChart:
      'First polling observation of the native linked Count DOM matching the current resident count.',
    memory:
      'Registry allocated and reserved accounting only, plus optional Chromium JS heap; neither is total device/process memory.',
    transfers:
      'Only explicitly exposed counters. No inference from missing counters or resident buffer size.',
    quantiles: 'Nearest rank; warmup samples excluded; raw per-interaction observations retained.',
    gpuTiming: unavailable('No GPU timestamp query instrumentation'),
    nativeCpuBaseline: unavailable('This harness measures only the resident adapter')
  },
  cold: {},
  warmup: [],
  samples: [],
  errors: []
};
try {
  report.revision = execFileSync('git', ['rev-parse', 'HEAD'], {encoding: 'utf8'}).trim();
  report.workspaceDirty = Boolean(
    execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], {
      encoding: 'utf8'
    }).trim()
  );
} catch {
  report.revision = null;
  report.workspaceDirty = null;
}
const preparationStart = nodePerformance.now();
const batches = [];
for (let offset = 0; offset < options.rows; offset += options.batchSize) {
  const columns = createPerformanceColumns(
    offset,
    Math.min(options.batchSize, options.rows - offset)
  );
  batches.push(
    ...new arrow.Table({
      longitude: arrow.makeVector(columns.longitude),
      latitude: arrow.makeVector(columns.latitude),
      value: arrow.makeVector(columns.value)
    }).batches
  );
}
const source = new arrow.Table(batches);
const sourceBytes = Buffer.from(arrow.tableToIPC(source, 'stream'));
report.cold.nodeFixtureAndIPCPreparationMilliseconds = nodePerformance.now() - preparationStart;
report.source = {
  rows: source.numRows,
  batchLengths: source.batches.map(batch => batch.numRows),
  fields: source.schema.fields.map(field => ({name: field.name, type: String(field.type)})),
  logicalColumnBytes: options.rows * 3 * Float32Array.BYTES_PER_ELEMENT,
  ipcBytes: sourceBytes.byteLength,
  distribution: 'value = sourceRowIndex % 1000; deterministic NYC coordinate grid; no nulls',
  normalization: 'Source Float32 values uploaded directly; no Float64 normalization'
};
const launchOptions = getPlaywrightLaunchOptions({
  headless: !options.headed,
  backend: 'webgpu-core'
});
report.browser = {
  name: 'Chromium',
  channel: launchOptions.channel,
  headless: !options.headed,
  arguments: launchOptions.args
};
let browser;
let page;
let datasetId;
let stopCpuDiagnostics;
try {
  const startupStart = nodePerformance.now();
  browser = await chromium.launch(launchOptions);
  report.browser.version = browser.version();
  page = await browser.newPage({viewport: {width: 1600, height: 1100}});
  page.setDefaultTimeout(options.timeoutMilliseconds);
  await pinViteModules(page);
  page.on('pageerror', error => report.errors.push(error.stack ?? error.message));
  page.on('console', message => {
    if (
      message.type() === 'error' &&
      /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error/i.test(message.text())
    )
      report.errors.push(message.text());
  });
  await page.goto(options.url, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => {
      const overlay = document.querySelector('vite-error-overlay');
      if (overlay) throw new Error(overlay.shadowRoot?.textContent || 'Vite module error');
      const state = window.__keplerComputeTest?.getState();
      if (state?.error) throw new Error(state.error);
      return state?.ready && state.selectedCount === 256;
    },
    undefined,
    {timeout: options.timeoutMilliseconds}
  );
  report.cold.browserLaunchAndApplicationReadyMilliseconds = nodePerformance.now() - startupStart;
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  const fileStart = nodePerformance.now();
  await workspace.locator('input[type=file]').setInputFiles({
    name: `resident-performance-${options.rows}.arrow`,
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: sourceBytes
  });
  await workspace.getByLabel('Source longitude column').selectOption('longitude');
  await workspace.getByLabel('Source latitude column').selectOption('latitude');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).waitFor();
  report.cold.fileTransferDecodeAndRoleSelectionMilliseconds = nodePerformance.now() - fileStart;
  const checkStart = nodePerformance.now();
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await page.waitForFunction(() =>
    [...document.querySelectorAll('button')].some(
      button => button.textContent?.trim() === 'Publish selected fields' && !button.disabled
    )
  );
  report.cold.schemaAndPrecisionCheckMilliseconds = nodePerformance.now() - checkStart;
  const publicationStart = nodePerformance.now();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    expectedRows => {
      const state = window.__keplerComputeTest.getState();
      if (state.error) throw new Error(state.error);
      const publication = state.publications[0];
      if (publication?.error) throw new Error(publication.error);
      return publication?.rowCount === expectedRows && publication.selectedCount === expectedRows;
    },
    options.rows,
    {timeout: options.timeoutMilliseconds}
  );
  report.cold.publicationToCurrentSummaryMilliseconds = nodePerformance.now() - publicationStart;
  const initial = await page.evaluate(() => window.__keplerComputeTest.getState().publications[0]);
  datasetId = initial.id;
  assert.deepEqual(
    initial.sourceBatches,
    report.source.batchLengths,
    'Publication must preserve original batches'
  );
  await page.waitForFunction(
    identifier =>
      window.__keplerComputeTest
        .renderedLayers()
        .some(layer => layer.id === `${identifier}-points` && layer.loaded && layer.visible),
    datasetId
  );
  report.cold.publicationToLoadedPointLayerMilliseconds = nodePerformance.now() - publicationStart;
  await closeNativeDialog(page);
  const configuration = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  const point = configuration.config.visState.layers.find(
    layer => layer.type === 'point' && layer.config.dataId === datasetId
  );
  assert(point, 'Publication must register an actual native Point layer');
  for (const layer of configuration.config.visState.layers)
    layer.config.isVisible = layer.id === point.id;
  point.config.textLabel = [];
  point.config.visConfig = {
    ...point.config.visConfig,
    filled: true,
    outline: false,
    radius: 2,
    opacity: 0.75
  };
  configuration.config.mapState = {
    ...configuration.config.mapState,
    longitude: -74,
    latitude: 40.75,
    zoom: 10,
    pitch: 0,
    bearing: 0
  };
  await page.evaluate(value => window.__keplerComputeTest.applyNativeConfig(value), configuration);
  await page.waitForFunction(
    identifier =>
      window.__keplerComputeTest
        .renderedLayers()
        .some(layer => layer.id === identifier && layer.loaded && layer.visible),
    point.id
  );
  report.nativeLayerId = point.id;
  report.viewport = configuration.config.mapState;
  await page.getByRole('button', {name: 'Toggle charts', exact: true}).click();
  await page.locator('.chart-config__type').click();
  await page.getByText('Big Number', {exact: true}).click();
  const card = page.locator('.resident-chart-card').first();
  await card.getByLabel('Chart dataset', {exact: true}).selectOption(datasetId);
  await card.getByLabel('Chart field', {exact: true}).selectOption('');
  await card.getByLabel('Chart measure', {exact: true}).selectOption('count');
  await card.getByLabel('Apply map filters', {exact: true}).check();
  await page.waitForFunction(
    expected =>
      document.querySelector(`.resident-chart-card [aria-label="Chart value: ${expected}"]`) !==
      null,
    options.rows
  );
  const snapshot = async () =>
    page.evaluate(identifier => {
      const api = window.__keplerComputeTest;
      const publication = api.getState().publications.find(value => value.id === identifier);
      const observation = api.getPerformanceSnapshot?.(identifier) ?? null;
      const device = api.resourceIdentity().device;
      return {
        observation,
        publication,
        deviceInfo: observation?.deviceInfo ?? device?.info ?? null,
        userAgent: navigator.userAgent,
        devicePixelRatio,
        jsHeap: performance.memory
          ? {
              usedJSHeapSize: performance.memory.usedJSHeapSize,
              totalJSHeapSize: performance.memory.totalJSHeapSize,
              jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
            }
          : null
      };
    }, datasetId);
  await page.evaluate(identifier => {
    const identity = window.__keplerComputeTest.resourceIdentity();
    window.__residentPerformanceSourceIdentity = {
      device: identity.device,
      buffers: [...identity.sourceBuffers[identifier]]
    };
  }, datasetId);
  if (options.interaction === 'value-only') {
    await page.evaluate(identifier => {
      const api = window.__keplerComputeTest;
      if (!api.setPublishedRangeValue) throw new Error('Value-only native filter hook unavailable');
      api.setPublishedRange(identifier, 'value', 0, 999);
    }, datasetId);
    await page.waitForFunction(
      ({identifier, expectedRows}) => {
        const api = window.__keplerComputeTest;
        const current = api.getPerformanceSnapshot(identifier);
        const filter = api
          .getState()
          .nativeFilters.find(value => value.id === `published-${identifier}-value`);
        return (
          filter?.value?.[0] === 0 &&
          filter.value[1] === 999 &&
          current.summary?.revision === current.explorerRevision &&
          current.summary.selectedCount === expectedRows
        );
      },
      {identifier: datasetId, expectedRows: options.rows}
    );
    report.valueOnlyPreparation =
      'Created the exact native range filter once before excluded warmups; preparation is outside warm metrics.';
  }
  report.beforeWarmup = await snapshot();
  report.device = report.beforeWarmup.deviceInfo ?? unavailable('Device information not exposed');
  if (options.diagnostics) {
    report.diagnostics = {
      enabled: true,
      warning:
        'Diagnostic runs include instrumentation overhead; compare separately from baseline.',
      hostStageHook: await page.evaluate(() => {
        const enable = window.__keplerComputeTest.setPerformanceDiagnosticsEnabled;
        if (!enable) return false;
        enable(true);
        return true;
      })
    };
  }
  const ranges = createInteractionRanges(options.warmup + options.iterations);
  for (const [index, range] of ranges.entries()) {
    if (options.diagnostics && index === options.warmup)
      stopCpuDiagnostics = await startCpuDiagnostics(page);
    const sample = await measureInteraction(
      page,
      datasetId,
      range,
      expectedRangeCount(options.rows, ...range),
      options.timeoutMilliseconds,
      options.diagnostics,
      options.interaction
    );
    if (sample.diagnostics) {
      const telemetry = sample.diagnostics.encodeTelemetry;
      sample.diagnostics.encodesDuringDispatch = summarizeEncodeTelemetry(
        telemetry.before,
        telemetry.afterDispatch,
        sample.sourceRevision
      );
      sample.diagnostics.encodesThroughCompletion = summarizeEncodeTelemetry(
        telemetry.before,
        telemetry.afterCompletion,
        sample.sourceRevision
      );
    }
    if (index < options.warmup) report.warmup.push(sample);
    else report.samples.push(sample);
    if (index + 1 === options.warmup) report.beforeMeasured = await snapshot();
    if (report.errors.length) throw new Error(report.errors.join('\n'));
  }
  if (stopCpuDiagnostics) await stopCpuDiagnostics();
  report.afterMeasured = await snapshot();
  const final = report.afterMeasured.publication;
  for (const field of ['sourceRevision', 'sourceUploadBytes', 'sourceBufferIds', 'sourceBatches'])
    assert.deepEqual(
      final[field],
      initial[field],
      `${field} must remain unchanged during interaction`
    );
  report.identity = await page.evaluate(identifier => {
    const before = window.__residentPerformanceSourceIdentity;
    const after = window.__keplerComputeTest.resourceIdentity();
    const buffers = after.sourceBuffers[identifier];
    return {
      deviceUnchanged: before.device === after.device,
      sourceBuffersUnchanged:
        before.buffers.length === buffers.length &&
        before.buffers.every((buffer, index) => buffer === buffers[index])
    };
  }, datasetId);
  assert(
    report.identity.deviceUnchanged && report.identity.sourceBuffersUnchanged,
    'Warm interactions must retain source buffers and device'
  );
  report.transfers = {
    registeredSourceUploadBytes: counterDelta(
      report.beforeMeasured.publication.sourceUploadBytes,
      final.sourceUploadBytes
    ),
    exposedReadbackBytes: counterDelta(
      report.beforeMeasured.observation?.readbackBytes,
      report.afterMeasured.observation?.readbackBytes
    ),
    controlUploads: unavailable('No comprehensive control-upload counter exposed'),
    totalDeviceTransfers: unavailable('No comprehensive device transfer counter exposed')
  };
  report.memory = {
    before:
      report.beforeMeasured.observation?.memoryUsage ??
      unavailable('Registry accounting not exposed'),
    after:
      report.afterMeasured.observation?.memoryUsage ??
      unavailable('Registry accounting not exposed'),
    jsHeapBefore:
      report.beforeMeasured.jsHeap ?? unavailable('Chromium performance.memory unavailable'),
    jsHeapAfter:
      report.afterMeasured.jsHeap ?? unavailable('Chromium performance.memory unavailable')
  };
  report.metrics = Object.fromEntries(
    [
      'submittedFrameMilliseconds',
      'nextAnimationFrameMilliseconds',
      'residentSummaryMilliseconds',
      'linkedChartMilliseconds'
    ].map(key => {
      const samples = report.samples.map(sample => sample[key]);
      return [
        key,
        samples.every(value => typeof value === 'number')
          ? summarizeDurations(samples)
          : unavailable(
              'Revision-tagged host frame/summary instrumentation unavailable for one or more samples'
            )
      ];
    })
  );
  if (options.diagnostics) {
    report.diagnostics.metrics = Object.fromEntries(
      [
        [
          'synchronousDispatchMilliseconds',
          sample => sample.diagnostics.synchronousDispatchMilliseconds
        ],
        ['pollCpuMilliseconds', sample => sample.diagnostics.pollCpuMilliseconds],
        [
          'latestEncodeJavascriptMilliseconds',
          sample => sample.diagnostics.encode?.elapsedMilliseconds
        ],
        [
          'cumulativeEncodeJavascriptMilliseconds',
          sample =>
            sample.diagnostics.encodesThroughCompletion?.available
              ? sample.diagnostics.encodesThroughCompletion.javascriptMilliseconds
              : null
        ],
        [
          'linkedSummaryReadWorkflowMilliseconds',
          sample => sample.diagnostics.summaryRead?.elapsedMilliseconds
        ],
        [
          'independentMatchingCountDomMilliseconds',
          sample => sample.diagnostics.firstMatchingCountDomMilliseconds
        ]
      ].map(([name, read]) => {
        const values = report.samples.map(read);
        return [
          name,
          values.every(value => typeof value === 'number')
            ? summarizeDurations(values)
            : unavailable('Optional diagnostic stage not observed for every measured interaction')
        ];
      })
    );
  }
  assert.deepEqual(report.errors, []);
  report.status = 'passed';
} catch (error) {
  report.status = 'failed';
  report.failure = error.stack ?? String(error);
  if (page && !page.isClosed()) {
    try {
      report.finalState = await page.evaluate(() => window.__keplerComputeTest?.getState());
    } catch {}
  }
  process.exitCode = 1;
} finally {
  report.finishedAt = new Date().toISOString();
  if (stopCpuDiagnostics) {
    try {
      await stopCpuDiagnostics();
    } catch (error) {
      report.diagnostics.cpuProfile = unavailable(`Profiler failed: ${String(error)}`);
    }
  }
  await browser?.close();
  await mkdir(dirname(resolve(options.output)), {recursive: true});
  await writeFile(options.output, `${JSON.stringify(report, null, 2)}\n`);
  console.log(
    JSON.stringify(
      {
        status: report.status,
        report: resolve(options.output),
        rows: options.rows,
        metrics: report.metrics,
        failure: report.failure
      },
      null,
      2
    )
  );
}

async function measureInteraction(
  page,
  identifier,
  range,
  expectedCount,
  timeoutMilliseconds,
  diagnostics,
  interaction
) {
  return page.evaluate(
    async ({identifier, range, expectedCount, timeoutMilliseconds, diagnostics, interaction}) => {
      const api = window.__keplerComputeTest;
      const initial = api.getPerformanceSnapshot?.(identifier) ?? null;
      const started = performance.now();
      if (interaction === 'value-only')
        api.setPublishedRangeValue(identifier, 'value', range[0], range[1]);
      else api.setPublishedRange(identifier, 'value', range[0], range[1]);
      const dispatchReturnedAt = diagnostics ? performance.now() : null;
      const stages = diagnostics
        ? {
            synchronousDispatchMilliseconds: dispatchReturnedAt - started,
            revisionObservedMilliseconds: null,
            firstMatchingCountDomMilliseconds: null,
            pollCount: 0,
            pollCpuMilliseconds: 0,
            maximumPollCpuMilliseconds: 0,
            encode: null,
            summaryRead: null,
            encodeTelemetry: {
              before: initial?.encodeTelemetry ?? null,
              afterDispatch: api.getPerformanceSnapshot?.(identifier)?.encodeTelemetry ?? null,
              afterCompletion: null
            }
          }
        : null;
      const observeCount = () => {
        if (
          stages &&
          stages.firstMatchingCountDomMilliseconds === null &&
          document.querySelector(
            `.resident-chart-card [aria-label="Chart value: ${expectedCount}"]`
          )
        )
          stages.firstMatchingCountDomMilliseconds = performance.now() - started;
      };
      const observer = diagnostics ? new MutationObserver(observeCount) : null;
      observer?.observe(document.body, {
        subtree: true,
        childList: true,
        attributes: true,
        attributeFilter: ['aria-label'],
        characterData: true
      });
      observeCount();
      let targetRevision = null;
      let submittedFrameMilliseconds = null;
      let nextAnimationFrameMilliseconds = null;
      let residentSummaryMilliseconds = null;
      let linkedChartMilliseconds = null;
      let animationRequest = null;
      return new Promise((resolve, reject) => {
        const fail = error => {
          if (animationRequest !== null) cancelAnimationFrame(animationRequest);
          observer?.disconnect();
          reject(error);
        };
        const poll = () => {
          const pollStarted = diagnostics ? performance.now() : null;
          try {
            const now = performance.now();
            const state = api.getState();
            if (state.error) throw new Error(state.error);
            if (document.visibilityState !== 'visible')
              throw new Error('Benchmark page became hidden');
            const publication = state.publications.find(value => value.id === identifier);
            if (!publication || publication.error)
              throw new Error(publication?.error ?? 'Published source disappeared');
            const nativeRangeApplied = state.nativeFilters.some(
              filter =>
                filter.dataId?.includes(identifier) &&
                filter.name?.includes('value') &&
                filter.value?.[0] === range[0] &&
                filter.value?.[1] === range[1]
            );
            const snapshot = api.getPerformanceSnapshot?.(identifier) ?? null;
            if (snapshot && snapshot.sourceRevision !== initial?.sourceRevision)
              throw new Error('Source revision changed during measurement');
            if (snapshot && snapshot.explorerRevision > initial.explorerRevision) {
              if (targetRevision !== snapshot.explorerRevision) {
                targetRevision = snapshot.explorerRevision;
                if (stages) {
                  stages.revisionObservedMilliseconds = now - started;
                  stages.encode = null;
                  stages.summaryRead = null;
                }
                submittedFrameMilliseconds =
                  nextAnimationFrameMilliseconds =
                  residentSummaryMilliseconds =
                  linkedChartMilliseconds =
                    null;
                if (animationRequest !== null) cancelAnimationFrame(animationRequest);
                animationRequest = null;
              }
              if (stages) {
                for (const name of ['encode', 'summaryRead']) {
                  const timing = snapshot[name];
                  if (
                    timing?.sourceRevision === snapshot.sourceRevision &&
                    (timing.explorerRevision ?? timing.revision) === targetRevision &&
                    timing.startedAt >= started
                  )
                    stages[name] = {
                      ...timing,
                      startedMilliseconds: timing.startedAt - started,
                      finishedMilliseconds: timing.finishedAt - started,
                      elapsedMilliseconds: timing.finishedAt - timing.startedAt
                    };
                }
              }
              const frame = snapshot.frame;
              if (
                submittedFrameMilliseconds === null &&
                frame?.stage === 'submitted' &&
                frame.explorerRevision === targetRevision &&
                frame.sourceRevision === snapshot.sourceRevision &&
                frame.timestamp >= started
              ) {
                submittedFrameMilliseconds = frame.timestamp - started;
                animationRequest = requestAnimationFrame(() => {
                  nextAnimationFrameMilliseconds = performance.now() - started;
                  animationRequest = null;
                });
              }
              if (
                snapshot.summary?.revision === targetRevision &&
                snapshot.summary.selectedCount === expectedCount
              )
                residentSummaryMilliseconds ??= now - started;
            } else if (!snapshot && publication.selectedCount === expectedCount) {
              // The publication getter only exposes its current summary; revision timing is unavailable.
              residentSummaryMilliseconds ??= now - started;
            }
            if (
              residentSummaryMilliseconds !== null &&
              document.querySelector(
                `.resident-chart-card [aria-label="Chart value: ${expectedCount}"]`
              )
            )
              linkedChartMilliseconds ??= now - started;
            if (
              nativeRangeApplied &&
              residentSummaryMilliseconds !== null &&
              linkedChartMilliseconds !== null &&
              (!snapshot || nextAnimationFrameMilliseconds !== null)
            ) {
              observer?.disconnect();
              if (stages)
                stages.encodeTelemetry.afterCompletion = snapshot?.encodeTelemetry ?? null;
              resolve({
                ...(stages ? {diagnostics: stages} : {}),
                interaction,
                sourceRevision: initial?.sourceRevision ?? null,
                range,
                expectedCount,
                nativeRangeApplied,
                revision: targetRevision,
                submittedFrameMilliseconds,
                nextAnimationFrameMilliseconds,
                residentSummaryMilliseconds,
                linkedChartMilliseconds,
                timingContract: snapshot
                  ? 'revision-tagged submitted frame + next RAF + polled summary/DOM'
                  : 'current summary/DOM only; frame timing unavailable'
              });
              return;
            }
            if (now - started > timeoutMilliseconds)
              throw new Error(
                `Interaction timed out: ${JSON.stringify({range, expectedCount, targetRevision, selectedCount: publication.selectedCount, snapshot})}`
              );
            setTimeout(poll, 4);
          } catch (error) {
            fail(error);
          } finally {
            if (stages) {
              const elapsed = performance.now() - pollStarted;
              stages.pollCount++;
              stages.pollCpuMilliseconds += elapsed;
              stages.maximumPollCpuMilliseconds = Math.max(
                stages.maximumPollCpuMilliseconds,
                elapsed
              );
            }
          }
        };
        poll();
      });
    },
    {identifier, range, expectedCount, timeoutMilliseconds, diagnostics, interaction}
  );
}

async function startCpuDiagnostics(page) {
  const session = await page.context().newCDPSession(page);
  await session.send('Profiler.enable');
  await session.send('Profiler.setSamplingInterval', {interval: 1000});
  await session.send('Profiler.start');
  let completion;
  let timer;
  const stop = (boundedTimeout = false) => {
    completion ??= (async () => {
      clearTimeout(timer);
      try {
        const {profile} = await session.send('Profiler.stop');
        const profilePath = `${resolve(options.output)}.cpuprofile`;
        await mkdir(dirname(profilePath), {recursive: true});
        await writeFile(profilePath, JSON.stringify(profile));
        report.diagnostics.cpuProfile = {
          available: true,
          path: profilePath,
          maximumDurationMilliseconds: 30000,
          stoppedByDurationBound: boundedTimeout,
          ...summarizeCpuProfile(profile)
        };
      } finally {
        await session.detach();
      }
    })();
    return completion;
  };
  timer = setTimeout(() => {
    // Preserve rejection for the awaited stop in normal/failure cleanup without an unhandled promise.
    void stop(true).catch(() => {});
  }, 30000);
  return stop;
}
