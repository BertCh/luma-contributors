// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Load every public Kepler sample through Add Data > Sample maps, with its original map config.
// Raw GitHub requests are attempted at runtime; gitignored fixtures are a READ-ONLY offline fallback.
// No sample data is written by this script. No browser is launched for --download-only.
// node scripts/kepler-demo-data.mjs [url] [--only=id,id] [--max-bytes=N] [--download-only]
import assert from 'node:assert/strict';
import {existsSync, readFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {probeSampleRemoteService} from '../src/sample-map-remote.mjs';
import {
  assertSamplePresentation,
  assertSampleRenderedViewsWhenReady
} from './sample-map-presentation.mjs';

const directory = path.dirname(fileURLToPath(import.meta.url));
const cacheDirectory = path.join(directory, 'kepler-demo-cache');
const repositoryUrl = 'https://raw.githubusercontent.com/keplergl/kepler.gl-data/master/';
const argumentsList = process.argv.slice(2);
const address =
  argumentsList.find(argument => /^https?:/.test(argument)) || 'http://127.0.0.1:5320/';
const onlyIds = argumentsList
  .find(argument => argument.startsWith('--only='))
  ?.slice(7)
  .split(',');
const maximumBytes = Number(
  argumentsList.find(argument => argument.startsWith('--max-bytes='))?.slice(12) ??
    256 * 1024 * 1024
);
const resources = new Map();
const unavailable = new Map();

function findCachedResource(url) {
  if (!url.startsWith(repositoryUrl)) return undefined;
  const relative = url.slice(repositoryUrl.length);
  if (relative.includes('..')) return undefined;
  const candidates = [path.join(cacheDirectory, 'samples', relative)];
  if (relative === 'samples.json') candidates.push(path.join(cacheDirectory, 'samples.json'));
  const [identifier, filename] = relative.split('/');
  if (filename?.startsWith('data.'))
    candidates.push(
      path.join(cacheDirectory, `${identifier.replaceAll('_', '-')}${path.extname(filename)}`)
    );
  return candidates.find(candidate => existsSync(candidate));
}

async function fetchResource(url) {
  if (resources.has(url)) return resources.get(url);
  if (unavailable.has(url)) throw new Error(unavailable.get(url));
  let networkError;
  try {
    const response = await fetch(url, {signal: AbortSignal.timeout(20000)});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    if (Number(response.headers.get('content-length')) > maximumBytes)
      throw new Error('exceeds --max-bytes');
    const body = Buffer.from(await response.arrayBuffer());
    if (body.length > maximumBytes) throw new Error('exceeds --max-bytes');
    const result = {body, source: 'network'};
    resources.set(url, result);
    return result;
  } catch (error) {
    networkError = String(error);
  }
  const cached = findCachedResource(url);
  if (cached) {
    const body = readFileSync(cached);
    if (body.length <= maximumBytes) {
      const result = {body, source: 'read-only cache'};
      resources.set(url, result);
      return result;
    }
  }
  unavailable.set(url, `${networkError}; no admissible offline fixture`);
  throw new Error(unavailable.get(url));
}

let catalog;
try {
  catalog = JSON.parse((await fetchResource(`${repositoryUrl}samples.json`)).body.toString());
} catch (error) {
  console.log(JSON.stringify({skipped: 'catalog unavailable offline', reason: String(error)}));
  process.exit(0);
}
const entries = catalog.filter(sample => !onlyIds || onlyIds.includes(sample.id));
assert.ok(entries.length, 'no samples matched --only');
const available = [];
const skipped = [];
for (const sample of entries) {
  if (!sample.configUrl || (!sample.dataUrl && !sample.remoteDatasetConfigUrl)) {
    available.push({...sample, unsupported: true});
    continue;
  }
  try {
    const requested = [sample.configUrl, sample.dataUrl || sample.remoteDatasetConfigUrl];
    const results = await Promise.all(requested.map(fetchResource));
    let remoteService;
    let remoteServiceProbeError;
    if (!sample.dataUrl) {
      const descriptor = JSON.parse(results[1].body.toString());
      const saved = JSON.parse(results[0].body.toString());
      console.log(
        JSON.stringify({
          id: sample.id,
          remoteDataset: {
            descriptorId: descriptor.info?.id,
            type: descriptor.info?.type,
            configuredIds: [
              ...new Set(saved.config.visState.layers.map(layer => layer.config.dataId))
            ]
          }
        })
      );
      // The descriptor is not enough: the provider may have moved its archive since publication.
      // Probe GET access without downloading the archive, independently of the browser load.
      try {
        remoteService = await probeSampleRemoteService(descriptor);
      } catch (error) {
        remoteServiceProbeError = String(error);
      }
      console.log(JSON.stringify({id: sample.id, remoteService, remoteServiceProbeError}));
    }
    available.push({...sample, remoteService, remoteServiceProbeError});
    console.log(
      JSON.stringify({
        id: sample.id,
        resources: requested.map((url, index) => ({
          url,
          source: results[index].source,
          bytes: results[index].body.length
        }))
      })
    );
  } catch (error) {
    skipped.push({id: sample.id, reason: String(error)});
    available.push({...sample, unavailable: String(error)});
  }
}
console.log(JSON.stringify({available: available.map(sample => sample.id), skipped}));
if (argumentsList.includes('--download-only')) process.exit(0);

const {chromium} = await import('playwright');
const {getPlaywrightLaunchOptions} = await import(
  '../../../../scripts/playwright/get-playwright-launch-options.mjs'
);
const {openNativeData} = await import('./native-navigation.mjs');
const {pinViteModules} = await import('./pin-vite-modules.mjs');
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const summaries = [];

async function createPage() {
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  await pinViteModules(page);
  await page.route(`${repositoryUrl}**`, async route => {
    try {
      const result = await fetchResource(route.request().url());
      await route.fulfill({
        status: 200,
        body: result.body,
        headers: {'access-control-allow-origin': '*'}
      });
    } catch {
      await route.fulfill({
        status: 503,
        body: 'Sample unavailable (including offline script cache)'
      });
    }
  });
  return page;
}

function createUrl(parameters) {
  const url = new URL(address);
  url.searchParams.delete('rows');
  url.searchParams.delete('sample');
  for (const [name, value] of Object.entries(parameters)) url.searchParams.set(name, value);
  // Exercise larger public maps without changing the app's normal default startup budget.
  if (parameters.rows && !url.searchParams.has('memoryBudgetMebibytes'))
    url.searchParams.set('memoryBudgetMebibytes', '8192');
  return url.href;
}

try {
  for (const sample of available) {
    const page = await createPage();
    const errors = [];
    const failedRequests = [];
    const mapStyleRequests = [];
    let checkingSampleStyles = false;
    page.on('request', request => {
      if (
        checkingSampleStyles &&
        /api\.mapbox\.com\/styles\/|basemaps\.cartocdn\.com\/gl\/|\/style\.json(?:\?|$)/i.test(
          request.url()
        )
      )
        mapStyleRequests.push(request.url());
    });
    page.on('pageerror', error => errors.push(error.stack || error.message));
    page.on('requestfailed', request =>
      failedRequests.push({
        url: request.url(),
        method: request.method(),
        resourceType: request.resourceType(),
        failure: request.failure()?.errorText
      })
    );
    page.on('console', message => {
      if (/unresolved value|Compilation error|Error while parsing WGSL/.test(message.text()))
        errors.push(message.text());
    });
    try {
      await page.goto(createUrl({rows: '128'}), {waitUntil: 'domcontentloaded'});
      await page.waitForFunction(
        () => window.__keplerComputeTest?.getState().selectedCount === 128,
        undefined,
        {timeout: 60000}
      );
      checkingSampleStyles = true;
      await openNativeData(page, 'Sample maps');
      const card = page.getByTestId(`sample-map-${sample.id}`);
      await card.waitFor();
      await card.getByTestId(`sample-attribution-${sample.id}`).waitFor();
      const button = card.getByRole('button');
      if (sample.unsupported) {
        assert.equal(
          await button.isDisabled(),
          true,
          `${sample.id}: unsupported sample load must be disabled`
        );
        assert.match(
          await card.innerText(),
          /Unsupported:/,
          `${sample.id}: show unsupported reason`
        );
        assert.deepEqual(
          errors,
          [],
          `${sample.id}: unsupported listing must not raise page errors`
        );
        summaries.push({id: sample.id, passed: true, unsupported: true});
        continue;
      }
      if (sample.unavailable) {
        await button.click();
        await page.getByTestId('sample-map-error').waitFor({timeout: 90000});
        assert.equal(
          (await page.evaluate(() => window.__keplerComputeTest.getState())).error,
          '',
          `${sample.id}: fetch failure must not set GPU error`
        );
        assert.deepEqual(errors, [], `${sample.id}: fetch failure must not raise page errors`);
        summaries.push({
          id: sample.id,
          passed: true,
          skipped: sample.unavailable,
          gracefulFailure: true
        });
        continue;
      }
      assert.equal(
        sample.remoteServiceProbeError,
        undefined,
        `${sample.id}: independent remote service probe must complete (network/CORS failures are not classified as HTTP unavailability)`
      );
      if (sample.remoteService && !sample.remoteService.available) {
        await button.click();
        const notice = page.getByTestId('sample-map-error');
        await notice.waitFor({timeout: 90000});
        const support = await page.evaluate(() => window.__keplerComputeTest.getSampleMapState());
        assert.deepEqual(
          support.remoteServiceFailure,
          {
            code: 'remote-service-unavailable',
            status: sample.remoteService.status,
            url: sample.remoteService.url
          },
          `${sample.id}: UI service failure must match the independent HTTP probe`
        );
        assert.equal(
          await notice.getAttribute('data-code'),
          'remote-service-unavailable',
          `${sample.id}: visible notice must name the service condition`
        );
        assert.match(
          await notice.innerText(),
          new RegExp(`Remote service unavailable \\(HTTP ${sample.remoteService.status}\\)`),
          `${sample.id}: show HTTP status in the UI`
        );
        assert.equal(
          support.loaded?.id === sample.id,
          false,
          `${sample.id}: unavailable service must not be reported loaded`
        );
        const state = await page.evaluate(() => window.__keplerComputeTest.getState());
        assert.equal(
          state.error,
          '',
          `${sample.id}: service outage must not set a GPU runtime error`
        );
        assert.equal(
          state.selectedCount,
          128,
          `${sample.id}: service outage must retain the existing synthetic selection`
        );
        assert.deepEqual(
          errors,
          [],
          `${sample.id}: service preflight must prevent uncaught loader errors; no page error is ignored`
        );
        summaries.push({
          id: sample.id,
          passed: true,
          support: 'remote-service-unavailable',
          remoteService: sample.remoteService,
          errors
        });
        continue;
      }
      await button.click();
      await page.waitForFunction(
        id => {
          const status = window.__keplerComputeTest?.getSampleMapState();
          if (status?.error) throw new Error(status.error);
          return status?.loaded?.id === id;
        },
        sample.id,
        // Loaded now includes queued GPU preparation: retain the former 120s + 180s allowance.
        {timeout: 300000}
      );
      if (sample.dataUrl) {
        await page.waitForFunction(
          id => {
            const state = window.__keplerComputeTest.getState();
            const report = window.__keplerComputeTest
              .getSampleMapState()
              .reports.find(report => report.datasetId === id);
            if (report?.state === 'failed') throw new Error(report.error);
            return (
              report?.state === 'ready' &&
              state.publications.some(
                publication =>
                  publication.id === id &&
                  !publication.scopePending &&
                  publication.selectedCount !== undefined
              )
            );
          },
          sample.id,
          {timeout: 180000}
        );
      }
      const saved = await page.evaluate(() =>
        JSON.parse(JSON.stringify(window.__keplerComputeTest.saveNativeConfig()))
      );
      const original = JSON.parse(resources.get(sample.configUrl).body.toString());
      const expected = await page.evaluate(
        ({config, label}) => window.__keplerComputeTest.createSampleMapReference(config, label),
        {config: original, label: sample.id}
      );
      const actual = saved.config.visState;
      const support = await page.evaluate(() => window.__keplerComputeTest.getSampleMapState());
      assert.ok(
        support.datasetIds.includes('resident-observations'),
        `${sample.id}: explicit sample loading must retain the bootstrap dataset`
      );
      assert.equal(
        await page
          .getByTestId('retained-sample-attribution')
          .filter({hasText: sample.label})
          .isVisible(),
        true,
        `${sample.id}: retained dataset attribution must be visible`
      );
      if (!sample.dataUrl) {
        for (const layer of expected.config.visState.layers)
          assert.ok(
            support.datasetIds.includes(layer.config.dataId),
            `${sample.id}: configured remote dataset ${layer.config.dataId} must be registered`
          );
      }
      const expectedCode =
        sample.id === 'sftrees'
          ? 'dense-aggregation-capacity'
          : sample.id === 'la_assessorparcels'
            ? 'wide-integer-style'
            : undefined;
      assert.deepEqual(
        support.limitations.map(limitation => limitation.code),
        expectedCode ? [expectedCode] : [],
        `${sample.id}: only documented limitations are allowed`
      );
      for (const limitation of support.limitations) {
        const notice = page.getByTestId('sample-map-limitation').filter({hasText: limitation.code});
        assert.equal(
          await notice.isVisible(),
          true,
          `${sample.id}: limitation ${limitation.code} must be visible`
        );
        const original = expected.config.visState.layers.find(
          layer => layer.id === limitation.layerId
        );
        assert.ok(original, `${sample.id}: partial-support status must name an original layer`);
        const fallback = actual.layers.find(layer => layer.id === limitation.fallbackLayerId);
        assert.equal(fallback?.type, 'point', `${sample.id}: fallback layer type`);
        assert.equal(fallback.config.isVisible, true, `${sample.id}: fallback layer visibility`);
        assert.equal(
          fallback.visualChannels.colorField,
          null,
          `${sample.id}: fallback color must be constant`
        );
        assert.equal(
          fallback.visualChannels.sizeField,
          null,
          `${sample.id}: fallback size must be constant`
        );
      }
      for (const expectedLayer of expected.config.visState.layers) {
        const actualLayer = actual.layers.find(layer => layer.id === expectedLayer.id);
        assert.ok(actualLayer, `${sample.id}: missing layer ${expectedLayer.id}`);
        assert.equal(
          actualLayer.type,
          expectedLayer.type,
          `${sample.id}: layer ${expectedLayer.id} type`
        );
        assert.equal(
          actualLayer.config.dataId,
          expectedLayer.config.dataId,
          `${sample.id}: layer ${expectedLayer.id} dataset ID`
        );
        const limitation = support.limitations.find(
          limitation => limitation.layerId === expectedLayer.id
        );
        // Only these named, visible and documented fallbacks may change original layer visibility.
        assert.equal(
          actualLayer.config.isVisible,
          limitation ? false : expectedLayer.config.isVisible,
          `${sample.id}: visibility changed for ${expectedLayer.id}`
        );
      }
      for (const expectedFilter of expected.config.visState.filters || []) {
        const actualFilter = actual.filters.find(filter => filter.id === expectedFilter.id);
        assert.ok(actualFilter, `${sample.id}: missing filter ${expectedFilter.id}`);
        assert.deepEqual(
          actualFilter.name,
          Array.isArray(expectedFilter.name) ? expectedFilter.name : [expectedFilter.name],
          `${sample.id}: filter ${expectedFilter.id} fields`
        );
        assert.deepEqual(
          actualFilter.value,
          expectedFilter.value,
          `${sample.id}: filter ${expectedFilter.id} values`
        );
      }
      assert.equal(
        actual.splitMaps?.length || 0,
        expected.config.visState.splitMaps?.length || 0,
        `${sample.id}: split view count`
      );
      assertSamplePresentation(saved, expected, support.limitations, sample.id);
      await page.waitForTimeout(3000);
      const state = await page.evaluate(() => window.__keplerComputeTest.getState());
      if (!sample.dataUrl) {
        assert.equal(
          await page.getByTestId('sample-recovery-notice').isVisible(),
          true,
          `${sample.id}: remote recovery limitation must be visible`
        );
        assert.equal(
          state.recovery.hasCheckpoint,
          true,
          `${sample.id}: remote load must retain a healthy recovery checkpoint`
        );
      }
      await assertSampleRenderedViewsWhenReady(
        () => page.evaluate(() => window.__keplerComputeTest.renderedViews()),
        expected,
        support.limitations,
        sample.id
      );
      // Asynchronous rendering must not mutate the saved presentation after the first check.
      const settled = await page.evaluate(() =>
        JSON.parse(JSON.stringify(window.__keplerComputeTest.saveNativeConfig()))
      );
      assertSamplePresentation(settled, expected, support.limitations, `${sample.id} settled`);
      assert.deepEqual(
        mapStyleRequests,
        [],
        `${sample.id}: sample loading and its reference must not request a remote basemap style`
      );
      if (sample.id === 'nyc_earnings') {
        const geometry = (
          await page.evaluate(() => window.__keplerComputeTest.getState())
        ).publications.find(publication => publication.id === 'nyc_earnings')?.geometry;
        assert.equal(
          geometry?.column,
          'geometry',
          `${sample.id}: publication keeps the geometry role`
        );
        assert.equal(
          geometry?.featureTypes.polygon,
          true,
          `${sample.id}: publication contains resident polygon geometry`
        );
        assert.equal(
          geometry?.invalidRowCount,
          0,
          `${sample.id}: Arrow ingestion preserves valid multipolygons`
        );
      }
      assert.equal(state.error, '', `${sample.id}: GPU runtime must have no error`);
      assert.deepEqual(
        state.nativeLayers.filter(layer => layer.error),
        [],
        `${sample.id}: native layers must have no error`
      );
      const workflowNotices = await page.getByTestId('resident-workflow-notice').allTextContents();
      assert.deepEqual(
        workflowNotices,
        [],
        `${sample.id}: no row-preparation/workflow notice after loading: ${workflowNotices.join('; ')}`
      );
      assert.deepEqual(errors, [], `${sample.id}: no uncaught page or shader errors`);
      summaries.push({
        id: sample.id,
        passed: true,
        layers: expected.config.visState.layers.length,
        filters: expected.config.visState.filters?.length || 0,
        remote: !sample.dataUrl,
        support: expectedCode ? 'partial' : 'full',
        limitations: support.limitations
      });
    } catch (error) {
      const diagnostics = await page
        .evaluate(() => {
          const hook = window.__keplerComputeTest;
          const state = hook?.getState();
          const geometry = hook?.getNativeLayerDataSummary('geojson');
          return {
            rendered: hook?.renderedLayers(),
            sampleLoadFailure: hook?.getSampleMapState().loadFailure,
            sampleRollbackFailure: hook?.getSampleMapState().rollbackFailure,
            renderedViews: hook?.renderedViews(),
            nativeLayers: state?.nativeLayers,
            fillLegends: state?.nativeLayers?.map(layer => ({
              id: layer.id,
              style: hook?.getStyleLegend(layer.id, 'fillColor')
            })),
            geojson: geometry && {
              layerId: geometry.layerId,
              dataId: geometry.dataId,
              datasetType: geometry.datasetType,
              layerOrder: geometry.layerOrder,
              shouldRender: geometry.shouldRender,
              renderProbe: geometry.renderProbe
            }
          };
        })
        .catch(error => ({failure: String(error)}));
      summaries.push({
        id: sample.id,
        passed: false,
        error: String(error),
        errors,
        failedRequests,
        mapStyleRequests,
        diagnostics
      });
    } finally {
      await page.close();
      console.log(JSON.stringify(summaries.at(-1)));
    }
  }
  assert.deepEqual(
    summaries.map(summary => summary.id),
    available.map(sample => sample.id),
    'each catalog entry needs exactly one explicit result'
  );
  // Check real-sample startup and a visible offline fallback independently from the gallery path.
  for (const offline of [false, true]) {
    if (!offline && !available.some(sample => sample.id === 'earthquakes' && !sample.unavailable))
      continue;
    const page = await createPage();
    try {
      if (offline)
        await page.route(`${repositoryUrl}**`, route => route.abort('internetdisconnected'));
      await page.goto(createUrl(offline ? {sample: 'earthquakes'} : {}), {
        waitUntil: 'domcontentloaded'
      });
      if (offline) {
        await page
          .getByTestId('sample-map-notice')
          .filter({hasText: 'Showing synthetic squares'})
          .waitFor({timeout: 90000});
        await page.waitForFunction(
          () => window.__keplerComputeTest?.getState().selectedCount === 100000
        );
      } else {
        await page.waitForFunction(
          () =>
            window.__keplerComputeTest
              ?.getState()
              .publications.some(
                publication =>
                  publication.id === 'earthquakes' && publication.selectedCount !== undefined
              ),
          undefined,
          {timeout: 180000}
        );
        assert.equal(
          (
            await page.evaluate(() => window.__keplerComputeTest.getSampleMapState())
          ).datasetIds.includes('resident-observations'),
          false,
          'default earthquakes startup must remove synthetic bootstrap table'
        );
        // The sample's saved time filter must plot from the replaced resident dataset, not stay pending.
        await page.waitForFunction(
          () =>
            window.__keplerComputeTest
              ?.getState()
              .nativeFilters?.filter(filter => filter.type === 'timeRange')
              .every(filter => filter.lumaComputePlot?.status === 'ready'),
          undefined,
          {timeout: 30000}
        );
        // Brushing the resident time filter must move its start to a finite value and replot.
        const readTimeFilter = () =>
          page.evaluate(() => {
            const filter = window.__keplerComputeTest
              .getState()
              .nativeFilters.find(candidate => candidate.type === 'timeRange');
            return {value: filter.value, status: filter.lumaComputePlot?.status};
          });
        const timeFilterBefore = await readTimeFilter();
        const handleBox = await page
          .locator('.kg-range-slider__brush .handle--w')
          .first()
          .boundingBox();
        assert.ok(handleBox, 'time brush west handle must be drawn');
        const handleX = handleBox.x + handleBox.width / 2;
        const handleY = handleBox.y + handleBox.height / 2;
        await page.mouse.move(handleX, handleY);
        await page.mouse.down();
        await page.mouse.move(handleX + 75, handleY, {steps: 10});
        await page.mouse.up();
        await page.waitForFunction(
          () =>
            window.__keplerComputeTest
              .getState()
              .nativeFilters.find(filter => filter.type === 'timeRange')?.lumaComputePlot
              ?.status === 'ready',
          undefined,
          {timeout: 30000}
        );
        const timeFilterAfter = await readTimeFilter();
        assert.ok(
          timeFilterAfter.value.every(Number.isFinite) &&
            timeFilterAfter.value[0] > timeFilterBefore.value[0] &&
            timeFilterAfter.value[1] === timeFilterBefore.value[1],
          `time brush drag must move the start to a finite value: ${JSON.stringify(timeFilterAfter.value)}`
        );
      }
      summaries.push({startup: offline ? 'offline fallback' : 'default earthquakes', passed: true});
    } catch (error) {
      summaries.push({
        startup: offline ? 'offline fallback' : 'default earthquakes',
        passed: false,
        error: String(error)
      });
    } finally {
      await page.close();
    }
  }
} finally {
  await browser.close();
}
console.log(
  JSON.stringify({passed: summaries.every(summary => summary.passed), summaries, skipped})
);
if (summaries.some(summary => !summary.passed)) process.exitCode = 1;
