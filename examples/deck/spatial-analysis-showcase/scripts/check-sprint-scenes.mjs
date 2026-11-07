// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {mkdir, readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import {chromium} from 'playwright';
import {PNG} from 'pngjs';

import {collectPageDiagnostics, probeWebGPU} from '../../../../scripts/playwright/collect-page-diagnostics.mjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const DEFAULT_URL = 'http://127.0.0.1:5312/';
const DEFAULT_OUTPUT = '/tmp/spatial-analysis-showcase-check';
const DEFAULT_LIDAR_EXPECTED_VOXELS = 9489;
const DEFAULT_SCENES = [
  'rainfall-interpolation',
  'facility-placement',
  'habitat-connectivity',
  'street-enclosure',
  'watersheds',
  'nature-clusters-outliers',
  'election-dynamics',
  'lidar-ground'
];
const PAGE_READY_TIMEOUT_MS = 90_000;
const STEP_READY_TIMEOUT_MS = 45_000;
// Fresh browser processes release native GPU/compiler caches during a complete atlas sweep.
const SCENES_PER_BROWSER = 12;

const options = parseArguments(process.argv.slice(2));
if (options.help) {
  printHelp();
  process.exit(0);
}

const outputDirectory = path.resolve(options.output);
await mkdir(outputDirectory, {recursive: true});
const buildEvidenceResponse = await fetch(new URL('build-evidence.json', options.url));
const buildEvidence = buildEvidenceResponse.ok && buildEvidenceResponse.headers.get('content-type')?.includes('json')
  ? await buildEvidenceResponse.json()
  : null;
if (buildEvidence) {
  await writeFile(path.join(outputDirectory, 'build-evidence.json'), `${JSON.stringify(buildEvidence, null, 2)}\n`);
}
const launchOptions = getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'});
let browser = await chromium.launch(launchOptions);
const results = [];
let failed = false;

try {
  if (options.all) {
    const inventoryPage = await browser.newPage();
    try {
      const inventoryUrl = new URL(options.url);
      inventoryUrl.hash = '#/';
      inventoryUrl.searchParams.delete('scene');
      await inventoryPage.goto(inventoryUrl.toString(), {waitUntil: 'networkidle'});
      await inventoryPage.locator('.atlas-entry').first().waitFor();
      options.scenes = await inventoryPage.locator('.atlas-entry').evaluateAll(entries =>
        [...new Set(entries.map(entry => decodeURIComponent(new URL(entry.href).hash.split('/')[2])))]
      );
      if (!options.scenes.length) throw new Error('The atlas exposes no story links.');
      console.log(`[check] Discovered ${options.scenes.length} stories from the served atlas.`);
    } finally {
      await inventoryPage.close();
    }
  }
  for (const [sceneIndex, sceneId] of options.scenes.entries()) {
    if (sceneIndex > 0 && sceneIndex % SCENES_PER_BROWSER === 0) {
      await browser.close();
      browser = await chromium.launch(launchOptions);
    }
    const result = await checkScene(browser, sceneId, options.url, outputDirectory);
    results.push(result);
    const outcome = result.errors.length ? 'FAIL' : 'PASS';
    console.log(`[${outcome}] ${sceneId}: ${result.steps.length} step(s), ${result.errors.length} error(s)`);
    for (const error of result.errors) console.log(`  - ${error}`);
    failed ||= result.errors.length > 0;
    await writeFile(path.join(outputDirectory, 'report.json'), `${JSON.stringify({url: options.url, bundleSha256: buildEvidence?.bundleSha256 ?? null, scenes: options.scenes, results}, null, 2)}\n`);
  }
} finally {
  await browser.close();
}

let bundleUnchanged = null;
if (buildEvidence) {
  const finalBuildEvidence = await (await fetch(new URL('build-evidence.json', options.url))).json();
  bundleUnchanged = finalBuildEvidence.bundleSha256 === buildEvidence.bundleSha256;
  failed ||= !bundleUnchanged;
  if (!bundleUnchanged) console.error('The served bundle changed during verification; rerun against one build.');
}
const report = {url: options.url, bundleSha256: buildEvidence?.bundleSha256 ?? null, bundleUnchanged, scenes: options.scenes, results};
const reportPath = path.join(outputDirectory, 'report.json');
await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(`[check] Report: ${reportPath}`);
process.exitCode = failed ? 1 : 0;

async function checkScene(browserInstance, sceneId, baseUrl, artifactDirectory) {
  const page = await browserInstance.newPage({viewport: {width: 1440, height: 960}, deviceScaleFactor: 1});
  const {diagnostics, dispose} = collectPageDiagnostics(page);
  const webgpuErrors = [];
  const httpErrors = [];
  const sceneResult = {sceneId, steps: [], diagnostics, webgpuErrors, httpErrors, errors: []};

  page.on('response', response => {
    if (response.status() >= 400) {
      httpErrors.push({status: response.status(), url: response.url()});
    }
  });
  await page.addInitScript(() => {
    const deviceErrors = [];
    Object.defineProperty(globalThis, '__sprintWebGPUErrors', {value: deviceErrors});
    let submittedCommandBuffers = 0;
    Object.defineProperty(globalThis, '__sprintGpuSubmittedCommandBuffers', {
      get: () => submittedCommandBuffers
    });
    let computeDispatches = 0;
    Object.defineProperty(globalThis, '__sprintGpuComputeDispatches', {get: () => computeDispatches});
    const computePassPrototype = globalThis.GPUComputePassEncoder?.prototype;
    if (computePassPrototype) {
      for (const method of ['dispatchWorkgroups', 'dispatchWorkgroupsIndirect']) {
        const originalDispatch = computePassPrototype[method];
        computePassPrototype[method] = function (...dispatchArguments) {
          computeDispatches++;
          return originalDispatch.apply(this, dispatchArguments);
        };
      }
    }
    const queuePrototype = globalThis.GPUQueue?.prototype;
    if (queuePrototype && !queuePrototype.__sprintInstrumented) {
      const originalSubmit = queuePrototype.submit;
      Object.defineProperty(queuePrototype, '__sprintInstrumented', {value: true});
      queuePrototype.submit = function (commandBuffers) {
        submittedCommandBuffers += commandBuffers.length;
        return originalSubmit.call(this, commandBuffers);
      };
    }
    const gpu = navigator.gpu;
    if (!gpu) return;
    const originalRequestAdapter = gpu.requestAdapter.bind(gpu);
    gpu.requestAdapter = async (...arguments_) => {
      const adapter = await originalRequestAdapter(...arguments_);
      if (!adapter) return adapter;
      const originalRequestDevice = adapter.requestDevice.bind(adapter);
      adapter.requestDevice = async (...deviceArguments) => {
        const device = await originalRequestDevice(...deviceArguments);
        device.addEventListener('uncapturederror', event => {
          deviceErrors.push({name: event.error.name, message: event.error.message});
        });
        return device;
      };
      return adapter;
    };
  });

  try {
    const url = new URL(baseUrl);
    url.hash = `#/story/${encodeURIComponent(sceneId)}`;
    await page.goto(url.toString(), {waitUntil: 'domcontentloaded', timeout: PAGE_READY_TIMEOUT_MS});
    await page.waitForFunction(
      id => {
        const state = globalThis.spatialAnalysisShowcase;
        return state?.sceneId === id && (state.ready || state.error);
      },
      sceneId,
      {timeout: PAGE_READY_TIMEOUT_MS}
    );

    const state = await page.evaluate(() => ({
      scene: globalThis.spatialAnalysisShowcase,
      body: {...document.body.dataset},
      canvasCount: document.querySelectorAll('canvas').length,
      mapCanvasCount: document.querySelectorAll('[aria-label="Map"] canvas').length,
      gpuErrors: globalThis.__sprintWebGPUErrors ?? [],
      gpuSubmittedCommandBuffers: globalThis.__sprintGpuSubmittedCommandBuffers ?? 0,
      gpuComputeDispatches: globalThis.__sprintGpuComputeDispatches ?? 0
    }));
    sceneResult.state = {
      sceneId: state.scene?.sceneId,
      ready: state.scene?.ready,
      error: state.scene?.error,
      rebuildCount: state.scene?.rebuildCount,
      body: state.body,
      canvasCount: state.canvasCount,
      mapCanvasCount: state.mapCanvasCount,
      gpuSubmittedCommandBuffers: state.gpuSubmittedCommandBuffers,
      gpuComputeDispatches: state.gpuComputeDispatches
    };
    sceneResult.webgpuErrors.push(...state.gpuErrors);
    if (state.scene?.error) sceneResult.errors.push(`Scene initialization failed: ${state.scene.error}`);
    if (!state.scene?.ready) sceneResult.errors.push('Scene never reached the host ready state.');
    if (!state.mapCanvasCount) sceneResult.errors.push('No map canvas was mounted.');

    const stepButtons = page.locator('.step-list button.step-row');
    const stepCount = await stepButtons.count();
    if (!stepCount) sceneResult.errors.push('Story controls are missing; refusing to treat the page shell as a passing scene.');
    for (let stepIndex = 0; stepIndex < stepCount; stepIndex++) {
      if (stepIndex > 0) {
        await stepButtons.nth(stepIndex).click();
        await waitForReadyState(page, sceneId, STEP_READY_TIMEOUT_MS);
      }
      const stepTitle = (await stepButtons.nth(stepIndex).innerText()).trim();
      const mapCanvas = page.locator('[aria-label="Map"] canvas').first();
      await mapCanvas.waitFor({state: 'visible', timeout: STEP_READY_TIMEOUT_MS});
      await page.waitForTimeout(1_000);
      await page.waitForFunction(() => {
        const values = document.querySelectorAll(
          '.step-detail:not([hidden]) [data-readout-value]:not(.readout-chart), .step-detail:not([hidden]) .live-readout, .status-line'
        );
        return [...values].every(element =>
          !/^(?:building GPU\b|(?:rebuilding|recomputing|computing|compiling|loading|solving)\b)/i.test(element.textContent.trim())
        );
      }, null, {timeout: STEP_READY_TIMEOUT_MS});
      if (sceneId === 'sampling-frames') {
        await page.waitForFunction(() => {
          const status = document.querySelector('[data-readout-value="status"]')?.textContent?.trim();
          return status && !/^(?:pending(?: replay)?|building GPU design…|—)$/i.test(status);
        }, null, {timeout: STEP_READY_TIMEOUT_MS});
        const designStatus = await page.locator('[data-readout-value="status"]').first().innerText();
        if (/GPU design error/i.test(designStatus)) sceneResult.errors.push(designStatus);
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      }
      if (sceneId === 'flight-matrix') {
        await page.waitForFunction(() => {
          const active = document.querySelector('.step-detail:not([hidden])');
          const occupied = active?.querySelector('[data-readout-value="occupied"]');
          if (!occupied) return true;
          const power = Number(document.querySelector('#all-option-resolutionPower')?.getAttribute('aria-valuenow'));
          // Counts are deliberately not read back above 1024 bins. At supported resolutions,
          // wait beyond the 900ms reorder plus 300ms settling delay for actual GPU evidence.
          return power > 10 || /^\d/.test(occupied.textContent.trim());
        }, null, {timeout: STEP_READY_TIMEOUT_MS});
      }
      const screenshotPath = path.join(artifactDirectory, `${sceneId}-${String(stepIndex + 1).padStart(2, '0')}.png`);
      const screenshotBuffer = await page.screenshot({path: screenshotPath, fullPage: false});
      const renderEvidence = getCanvasPixelEvidence(screenshotBuffer, await mapCanvas.boundingBox());
      const stepState = await page.evaluate(() => ({
        ready: globalThis.spatialAnalysisShowcase?.ready,
        error: globalThis.spatialAnalysisShowcase?.error,
        gpuErrors: globalThis.__sprintWebGPUErrors ?? [],
        gpuSubmittedCommandBuffers: globalThis.__sprintGpuSubmittedCommandBuffers ?? 0,
        gpuComputeDispatches: globalThis.__sprintGpuComputeDispatches ?? 0,
        status: document.querySelector('[role="status"]')?.textContent?.trim() ?? '',
        canvasCount: document.querySelectorAll('[aria-label="Map"] canvas').length
      }));
      sceneResult.webgpuErrors.push(...stepState.gpuErrors);
      const stepResult = {index: stepIndex + 1, title: stepTitle, screenshotPath, status: stepState.status, renderEvidence};
      stepResult.furniture = await page.evaluate(collectFurnitureEvidence);
      if (stepResult.furniture.overlaps.length) sceneResult.errors.push(`Step ${stepIndex + 1} map furniture overlaps: ${stepResult.furniture.overlaps.join(', ')}.`);
      sceneResult.steps.push(stepResult);
      stepResult.gpuSubmittedCommandBuffers = stepState.gpuSubmittedCommandBuffers;
      stepResult.gpuComputeDispatches = stepState.gpuComputeDispatches;
      if (!stepState.ready || stepState.error) sceneResult.errors.push(`Step ${stepIndex + 1} lost readiness${stepState.error ? `: ${stepState.error}` : '.'}`);
      if (stepState.gpuSubmittedCommandBuffers < 1) sceneResult.errors.push(`Step ${stepIndex + 1} submitted no WebGPU command buffers.`);
      if (!renderEvidence.canvasHasArea || renderEvidence.uniqueColors < 12 || renderEvidence.opaquePixels < 1000) {
        sceneResult.errors.push(`Step ${stepIndex + 1} did not produce substantial visible canvas pixels.`);
      }
      if (stepState.status && /could not start|failed|error/i.test(stepState.status)) {
        sceneResult.errors.push(`Step ${stepIndex + 1} reports a failed computation: ${stepState.status}`);
      }
    }

    if (sceneId === 'rainfall-interpolation' && stepCount > 4) {
      await checkRainfallValidation(page, sceneResult, artifactDirectory);
    }
    if (sceneId === 'facility-placement' && stepCount > 1) {
      await checkFacilityAssignments(page, sceneResult, artifactDirectory);
    }
    if (sceneId === 'lidar-ground') {
      await checkLidarVoxels(page, sceneResult, options.expectedLidarVoxels, artifactDirectory);
    }

    if (options.mobile) {
      await page.setViewportSize({width: 390, height: 844});
      // The sheet animation is followed by the debounced bounds refit.
      await page.waitForTimeout(1000);
      const screenshotPath = path.join(artifactDirectory, `${sceneId}-mobile.png`);
      await page.screenshot({path: screenshotPath});
      sceneResult.mobile = await page.evaluate(() => {
        const map = document.querySelector('[aria-label="Map"] canvas')?.getBoundingClientRect();
        const header = document.querySelector('.site-header')?.getBoundingClientRect();
        const cartouche = document.querySelector('.map-furniture-cartouche')?.getBoundingClientRect();
        return {
          viewportWidth: innerWidth,
          documentWidth: document.documentElement.scrollWidth,
          mapWidth: map?.width ?? 0,
          mapHeight: map?.height ?? 0,
          mapTop: map?.top ?? 0,
          headerBottom: header?.bottom ?? 0,
          cartoucheTop: cartouche?.height ? cartouche.top : null
        };
      });
      sceneResult.mobile.screenshotPath = screenshotPath;
      if (sceneResult.mobile.documentWidth > sceneResult.mobile.viewportWidth) {
        sceneResult.errors.push('Mobile story overflows the viewport horizontally.');
      }
      if (sceneResult.mobile.mapWidth < 100 || sceneResult.mobile.mapHeight < 100) {
        sceneResult.errors.push('Mobile story has no usable map canvas area.');
      }
      if (sceneResult.mobile.mapTop < sceneResult.mobile.headerBottom - 1 ||
          (sceneResult.mobile.cartoucheTop !== null && sceneResult.mobile.cartoucheTop < sceneResult.mobile.headerBottom - 1)) {
        sceneResult.errors.push('Mobile map or cartouche extends beneath the navigation header.');
      }
      const legendToggle = page.locator('.map-legend-toggle');
      const panelLegendButton = page.locator('.panel-legend-button');
      const hasLegend = await page.locator('.legends').evaluate(element => element.childElementCount > 0);
      if (hasLegend && await panelLegendButton.isVisible()) {
        await panelLegendButton.click();
        await page.waitForTimeout(350);
        sceneResult.mobile.legendAccess = 'panel';
      } else if (await legendToggle.isVisible()) {
        if ((await legendToggle.getAttribute('aria-expanded')) !== 'false') {
          sceneResult.errors.push('Mobile legend did not collapse when the viewport narrowed.');
        }
        await legendToggle.click();
        await page.waitForTimeout(350);
        sceneResult.mobile.legendAccess = 'map';
      } else if (hasLegend) {
        sceneResult.errors.push('Mobile legend has no visible access control.');
      }
      if (hasLegend) {
        sceneResult.mobile.legendCanExpand = await page.locator('.map-legend-body').isVisible();
        if (!sceneResult.mobile.legendCanExpand) sceneResult.errors.push('Mobile legend cannot be opened.');
        const legendBodyBounds = await page.locator('.map-legend-body').boundingBox();
        if (legendBodyBounds && legendBodyBounds.height < 44) sceneResult.errors.push('Expanded mobile legend has less than 44 pixels of readable height.');
        sceneResult.mobile.furniture = await page.evaluate(collectFurnitureEvidence);
        if (sceneResult.mobile.furniture.overlaps.length) sceneResult.errors.push(`Expanded mobile legend overlaps: ${sceneResult.mobile.furniture.overlaps.join(', ')}.`);
        if (await legendToggle.isVisible()) await legendToggle.click();
      }
    }

    const finalGpuErrors = await page.evaluate(() => globalThis.__sprintWebGPUErrors ?? []);
    sceneResult.webgpuErrors.push(...finalGpuErrors);
    sceneResult.webgpu = await probeWebGPU(page);
    sceneResult.gpuSubmittedCommandBuffers = await page.evaluate(() => globalThis.__sprintGpuSubmittedCommandBuffers ?? 0);
    if (!sceneResult.webgpu.adapter) sceneResult.errors.push('Browser has no WebGPU adapter.');
    if (!sceneResult.webgpuErrors.length && !sceneResult.diagnostics.consoleMessages.some(message => /WebGPU|validation error/i.test(message.text))) {
      sceneResult.validation = 'No uncaptured WebGPU or console validation errors observed.';
    }
    if (sceneResult.webgpuErrors.length) sceneResult.errors.push(`${sceneResult.webgpuErrors.length} uncaptured WebGPU validation error(s).`);
    const consoleValidationErrors = sceneResult.diagnostics.consoleMessages.filter(message =>
      message.type === 'error' && /WebGPU|validation error/i.test(message.text)
    );
    if (consoleValidationErrors.length) sceneResult.errors.push(`${consoleValidationErrors.length} WebGPU validation console error(s).`);
    const browserConsoleErrors = sceneResult.diagnostics.consoleMessages.filter(message => message.type === 'error');
    if (browserConsoleErrors.length) sceneResult.errors.push(`${browserConsoleErrors.length} browser console error(s).`);
    if (sceneResult.diagnostics.pageErrors.length) sceneResult.errors.push(`${sceneResult.diagnostics.pageErrors.length} uncaught page error(s).`);
    if (httpErrors.length) sceneResult.errors.push(`${httpErrors.length} HTTP error response(s).`);

    await writeFile(path.join(artifactDirectory, `${sceneId}-diagnostics.json`), `${JSON.stringify(sceneResult, null, 2)}\n`);
    return sceneResult;
  } catch (error) {
    sceneResult.errors.push(error instanceof Error ? error.stack || error.message : String(error));
    sceneResult.state = await page.evaluate(() => ({
      scene: globalThis.spatialAnalysisShowcase
        ? {
            sceneId: globalThis.spatialAnalysisShowcase.sceneId,
            ready: globalThis.spatialAnalysisShowcase.ready,
            error: globalThis.spatialAnalysisShowcase.error
          }
        : null,
      body: {...document.body.dataset},
      status: document.querySelector('[role="status"]')?.textContent?.trim() ?? '',
      mapCanvasCount: document.querySelectorAll('[aria-label="Map"] canvas').length,
      gpuSubmittedCommandBuffers: globalThis.__sprintGpuSubmittedCommandBuffers ?? 0,
      gpuErrors: globalThis.__sprintWebGPUErrors ?? []
    })).catch(() => null);
    sceneResult.webgpuErrors.push(...(sceneResult.state?.gpuErrors ?? []));
    await page.screenshot({path: path.join(artifactDirectory, `${sceneId}-failure.png`)}).catch(() => {});
    sceneResult.diagnostics = diagnostics;
    sceneResult.webgpu = await probeWebGPU(page).catch(error => ({error: String(error)}));
    await writeFile(path.join(artifactDirectory, `${sceneId}-diagnostics.json`), `${JSON.stringify(sceneResult, null, 2)}\n`);
    return sceneResult;
  } finally {
    dispose();
    await page.close();
  }
}

/** Checks rendered boxes, including an expanded legend's scroll viewport. */
function collectFurnitureEvidence() {
  const getVisibleBox = selector => {
    const element = document.querySelector(selector);
    if (!element || !element.getClientRects().length) return null;
    const rectangle = element.getBoundingClientRect();
    return rectangle.width && rectangle.height ? rectangle : null;
  };
  const legend = getVisibleBox('.map-legend');
  const body = getVisibleBox('.map-legend-body');
  const credit = getVisibleBox('.map-credit-slot');
  const title = getVisibleBox('.map-furniture-cartouche');
  const scale = getVisibleBox('.map-furniture-scale');
  const overlaps = [];
  for (const [name, rectangle] of [['source credit', credit], ['map title', title]]) {
    if (legend && rectangle &&
      Math.min(legend.right, rectangle.right) - Math.max(legend.left, rectangle.left) > 1 &&
      Math.min(legend.bottom, rectangle.bottom) - Math.max(legend.top, rectangle.top) > 1) overlaps.push(name);
  }
  if (scale && credit &&
    Math.min(scale.right, credit.right) - Math.max(scale.left, credit.left) > 1 &&
    Math.min(scale.bottom, credit.bottom) - Math.max(scale.top, credit.top) > 1) overlaps.push('scale bar and source credit');
  if (legend && body && body.bottom > legend.bottom + 1) overlaps.push('legend content outside its frame');
  return {overlaps};
}

async function checkRainfallValidation(page, sceneResult, artifactDirectory) {
  const validationStep = page.locator('.step-list button.step-row').nth(4);
  await validationStep.click();
  await waitForReadyState(page, 'rainfall-interpolation', STEP_READY_TIMEOUT_MS);
  await page.getByRole('tab', {name: 'All controls'}).click();
  const validationSplit = page.locator('#all-option-validationSplit');
  const holdOut = page.locator('#all-option-holdOut');
  await validationSplit.waitFor({state: 'visible', timeout: STEP_READY_TIMEOUT_MS});
  if (!(await holdOut.isChecked())) await holdOut.check();

  const modes = [];
  for (const design of ['random', 'spatial']) {
    await validationSplit.selectOption(design);
    await page.waitForFunction(value => {
      const selected = document.querySelector('#all-option-validationSplit')?.value;
      const design = document.querySelector('.step-body [data-readout="validationDesign"]')?.textContent?.trim();
      const readouts = [...document.querySelectorAll('.step-readouts .readout')].map(row => ({
        label: row.querySelector('dt')?.textContent?.trim(),
        value: row.querySelector('dd')?.textContent?.trim()
      }));
      const support = readouts.find(readout => readout.label === 'Scored / unsupported')?.value;
      const verdict = readouts.find(readout => readout.label === 'Verdict')?.value;
      const designMatches = value === 'spatial' ? design?.includes('250 km tiles') : design?.includes('random gauge holdout');
      const settled = [design, support, verdict].every(text => text && !/—|assigning|waiting|pending|computing/i.test(text));
      return selected === value && designMatches && settled;
    }, design, {timeout: STEP_READY_TIMEOUT_MS});
    const values = await page.evaluate(() => {
      const getReadout = label => [...document.querySelectorAll('.step-readouts .readout')]
        .find(row => row.querySelector('dt')?.textContent?.trim() === label)
        ?.querySelector('dd')?.textContent?.trim() ?? '';
      return [
        document.querySelector('.step-body [data-readout="validationDesign"]')?.textContent?.trim() ?? '',
        getReadout('Scored / unsupported'),
        getReadout('Verdict')
      ];
    });
    const screenshotPath = path.join(artifactDirectory, `rainfall-validation-${design}.png`);
    await page.screenshot({path: screenshotPath, fullPage: false});
    modes.push({design, readouts: Object.fromEntries(['design', 'support', 'verdict'].map((key, index) => [key, values[index]])), screenshotPath});
    const evidence = {design: values[0].trim(), support: values[1].trim(), verdict: values[2].trim()};
    if (Object.values(evidence).some(value => !value || value === '—' || /pending|loading|computing/i.test(value))) {
      sceneResult.errors.push(`Rainfall ${design} validation has placeholder or pending readouts: ${JSON.stringify(evidence)}`);
    }
  }
  if (modes[0]?.readouts.design === modes[1]?.readouts.design) {
    sceneResult.errors.push('Rainfall random and spatial validation produced the same design readout.');
  }
  sceneResult.validationModes = modes;
}

async function checkFacilityAssignments(page, sceneResult, artifactDirectory) {
  await page.locator('.step-list button.step-row').nth(3).click();
  await waitForReadyState(page, 'facility-placement', STEP_READY_TIMEOUT_MS);
  await page.getByRole('tab', {name: 'All controls'}).click();
  const mapControl = page.locator('#all-option-map');
  const objectiveControl = page.locator('#all-option-objective');
  await mapControl.selectOption('assignments');
  const assignmentScreenshots = [];
  for (const objective of ['average', 'worst']) {
    await objectiveControl.selectOption(objective);
    await page.waitForFunction(value =>
      document.querySelector('#all-option-map')?.value === 'assignments' &&
      document.querySelector('#all-option-objective')?.value === value &&
      Boolean(globalThis.spatialAnalysisShowcase?.ready), objective, {timeout: STEP_READY_TIMEOUT_MS});
    await page.waitForFunction(() => {
      const value = [...document.querySelectorAll('.step-detail:not([hidden]) .readout')]
        .find(row => row.querySelector('dt')?.textContent?.trim() === 'Selected candidate sites')
        ?.querySelector('dd')?.textContent?.trim();
      return Boolean(value && !/—|pending|calculating|loading/i.test(value));
    }, undefined, {timeout: STEP_READY_TIMEOUT_MS});
    await page.waitForTimeout(750);
    const screenshotPath = path.join(artifactDirectory, `facility-placement-assignments-${objective}.png`);
    const screenshotBuffer = await page.screenshot({path: screenshotPath, fullPage: false});
    const mapCanvas = page.locator('[aria-label="Map"] canvas').first();
    const renderEvidence = getCanvasPixelEvidence(screenshotBuffer, await mapCanvas.boundingBox());
    const selectedSites = await page.evaluate(() => [...document.querySelectorAll('.step-detail:not([hidden]) .readout')]
      .find(row => row.querySelector('dt')?.textContent?.trim() === 'Selected candidate sites')
      ?.querySelector('dd')?.textContent?.trim() ?? '');
    assignmentScreenshots.push({objective, screenshotPath, renderEvidence, selectedSites});
    if (!selectedSites || /—|pending|calculating|loading/i.test(selectedSites)) {
      sceneResult.errors.push(`Facility ${objective} objective has no settled selected-site readout.`);
    }
    if (renderEvidence.selectedSitePixels < 8) {
      sceneResult.errors.push(`Facility ${objective} assignments map has no visible orange selected-site markers.`);
    }
  }
  sceneResult.facilityAssignments = assignmentScreenshots;
}

async function checkLidarVoxels(page, sceneResult, expectedVoxelCount, artifactDirectory) {
  if (!Number.isInteger(expectedVoxelCount) || expectedVoxelCount < 1) {
    sceneResult.errors.push('LiDAR validation requires --lidar-expected-voxels from the CPU reference.');
    return;
  }
  await page.waitForFunction(() => {
    const getReadout = label => [...document.querySelectorAll('.step-detail:not([hidden]) .readout')]
      .find(row => row.querySelector('dt')?.textContent?.trim() === label)
      ?.querySelector('dd')?.textContent?.trim();
    const occupied = getReadout('Occupied 2 m voxels');
    const sampled = getReadout('Retained source returns');
    return [occupied, sampled].every(value => value && /^[\d,]+$/.test(value));
  }, undefined, {timeout: STEP_READY_TIMEOUT_MS});
  const voxelCounts = await page.evaluate(() => {
    const getReadout = label => [...document.querySelectorAll('.step-detail:not([hidden]) .readout')]
      .find(row => row.querySelector('dt')?.textContent?.trim() === label)
      ?.querySelector('dd')?.textContent?.trim() ?? '';
    const occupiedText = getReadout('Occupied 2 m voxels');
    const sampledText = getReadout('Retained source returns');
    return {
      occupied: Number(occupiedText.replaceAll(',', '')),
      sampled: Number(sampledText.replaceAll(',', '')),
      occupiedText,
      sampledText
    };
  });
  sceneResult.lidarVoxelCounts = {...voxelCounts, expectedFromCpu: expectedVoxelCount};
  if (voxelCounts.occupied !== expectedVoxelCount) {
    sceneResult.errors.push(`LiDAR GPU reports ${voxelCounts.occupiedText} occupied voxels; CPU reference expects ${expectedVoxelCount}.`);
  }
  if (!Number.isInteger(voxelCounts.sampled) || voxelCounts.occupied >= voxelCounts.sampled) {
    sceneResult.errors.push(`LiDAR voxel sampling did not reduce the point count: ${voxelCounts.occupiedText} of ${voxelCounts.sampledText} returns.`);
  }
  const mapCanvas = page.locator('[aria-label="Map"] canvas').first();
  const mapBounds = await mapCanvas.boundingBox();
  const fullResolutionPath = sceneResult.steps[0]?.screenshotPath;
  const voxelSamplePath = sceneResult.steps.at(-1)?.screenshotPath;
  if (!fullResolutionPath || !voxelSamplePath || !mapBounds) {
    sceneResult.errors.push('LiDAR full-resolution and voxel-sample screenshots are unavailable for visual comparison.');
    return;
  }
  const imageDifference = await getCanvasScreenshotDifference(fullResolutionPath, voxelSamplePath, mapBounds);
  sceneResult.lidarRenderDifference = imageDifference;
  await writeFile(
    path.join(artifactDirectory, 'lidar-ground-render-difference.json'),
    `${JSON.stringify(imageDifference, null, 2)}\n`
  );
  if (imageDifference.changedPixelRatio < 0.02) {
    sceneResult.errors.push(`LiDAR voxel sample changed only ${(imageDifference.changedPixelRatio * 100).toFixed(2)}% of map pixels from the full-resolution view; the reduced draw is not visibly confirmed.`);
  }
}

async function getCanvasScreenshotDifference(firstScreenshotPath, secondScreenshotPath, canvasBounds) {
  const [firstScreenshotBuffer, secondScreenshotBuffer] = await Promise.all([
    readFile(firstScreenshotPath),
    readFile(secondScreenshotPath)
  ]);
  const firstScreenshot = PNG.sync.read(firstScreenshotBuffer);
  const secondScreenshot = PNG.sync.read(secondScreenshotBuffer);
  if (firstScreenshot.width !== secondScreenshot.width || firstScreenshot.height !== secondScreenshot.height) {
    return {changedPixels: 0, comparedPixels: 0, changedPixelRatio: 0};
  }
  const left = Math.max(0, Math.floor(canvasBounds.x));
  const top = Math.max(0, Math.floor(canvasBounds.y));
  const right = Math.min(firstScreenshot.width, Math.ceil(canvasBounds.x + canvasBounds.width));
  const bottom = Math.min(firstScreenshot.height, Math.ceil(canvasBounds.y + canvasBounds.height));
  let comparedPixels = 0;
  let changedPixels = 0;
  for (let y = top; y < bottom; y += 1) {
    for (let x = left; x < right; x += 1) {
      const offset = (y * firstScreenshot.width + x) * 4;
      const redDifference = Math.abs(firstScreenshot.data[offset] - secondScreenshot.data[offset]);
      const greenDifference = Math.abs(firstScreenshot.data[offset + 1] - secondScreenshot.data[offset + 1]);
      const blueDifference = Math.abs(firstScreenshot.data[offset + 2] - secondScreenshot.data[offset + 2]);
      if (redDifference + greenDifference + blueDifference > 80) changedPixels += 1;
      comparedPixels += 1;
    }
  }
  return {
    changedPixels,
    comparedPixels,
    changedPixelRatio: comparedPixels ? changedPixels / comparedPixels : 0
  };
}

async function waitForReadyState(page, sceneId, timeout) {
  await page.waitForFunction(
    id => {
      const state = globalThis.spatialAnalysisShowcase;
      return state?.sceneId === id && (state.ready || state.error);
    },
    sceneId,
    {timeout}
  );
  const error = await page.evaluate(() => globalThis.spatialAnalysisShowcase?.error ?? null);
  if (error) throw new Error(`Scene failed while changing story steps: ${error}`);
}

function getCanvasPixelEvidence(screenshotBuffer, canvasBounds) {
  const png = PNG.sync.read(screenshotBuffer);
  if (!canvasBounds || canvasBounds.width < 1 || canvasBounds.height < 1) {
    return {canvasHasArea: false, opaquePixels: 0, uniqueColors: 0};
  }
  const left = Math.max(0, Math.floor(canvasBounds.x));
  const top = Math.max(0, Math.floor(canvasBounds.y));
  const right = Math.min(png.width, Math.ceil(canvasBounds.x + canvasBounds.width));
  const bottom = Math.min(png.height, Math.ceil(canvasBounds.y + canvasBounds.height));
  const colors = new Set();
  let opaquePixels = 0;
  let selectedSitePixels = 0;
  for (let y = top; y < bottom; y += 1) {
    for (let x = left; x < right; x += 1) {
      const offset = (y * png.width + x) * 4;
      const red = png.data[offset];
      const green = png.data[offset + 1];
      const blue = png.data[offset + 2];
      const alpha = png.data[offset + 3];
      if (alpha > 200) opaquePixels += 1;
      if (red >= 180 && red <= 255 && green >= 100 && green <= 190 && blue >= 25 && blue <= 115) {
        selectedSitePixels += 1;
      }
      colors.add(`${red >> 3}:${green >> 3}:${blue >> 3}:${alpha >> 5}`);
    }
  }
  return {
    canvasHasArea: right > left && bottom > top,
    opaquePixels,
    selectedSitePixels,
    uniqueColors: colors.size
  };
}

function parseArguments(argumentsList) {
  const parsed = {
    url: DEFAULT_URL,
    scenes: DEFAULT_SCENES,
    output: DEFAULT_OUTPUT,
    expectedLidarVoxels: DEFAULT_LIDAR_EXPECTED_VOXELS,
    help: false,
    all: false,
    mobile: false
  };
  for (let index = 0; index < argumentsList.length; index += 1) {
    const argument = argumentsList[index];
    if (argument === '--help' || argument === '-h') parsed.help = true;
    else if (argument === '--all') parsed.all = true;
    else if (argument === '--mobile') parsed.mobile = true;
    else if (argument === '--url') parsed.url = argumentsList[++index];
    else if (argument.startsWith('--url=')) parsed.url = argument.slice('--url='.length);
    else if (argument === '--scenes') parsed.scenes = parseSceneList(argumentsList[++index]);
    else if (argument.startsWith('--scenes=')) parsed.scenes = parseSceneList(argument.slice('--scenes='.length));
    else if (argument === '--output') parsed.output = argumentsList[++index];
    else if (argument.startsWith('--output=')) parsed.output = argument.slice('--output='.length);
    else if (argument === '--lidar-expected-voxels') parsed.expectedLidarVoxels = Number(argumentsList[++index]);
    else if (argument.startsWith('--lidar-expected-voxels=')) parsed.expectedLidarVoxels = Number(argument.slice('--lidar-expected-voxels='.length));
    else throw new Error(`Unknown argument: ${argument}`);
  }
  if (!parsed.url || !parsed.scenes.length || !parsed.output) throw new Error('URL, at least one scene, and output directory are required.');
  return parsed;
}

function parseSceneList(value = '') {
  return value.split(',').map(scene => scene.trim()).filter(Boolean);
}

function printHelp() {
  console.log(`Usage: node scripts/check-sprint-scenes.mjs [options]

Options:
  --url <url>            Showcase origin or entry URL (default: ${DEFAULT_URL})
  --scenes <ids>         Comma-separated story scene ids (default: sprint scenes)
  --all                 Discover and check every story in the served atlas
  --mobile              Also capture and check a 390-pixel mobile story layout
  --output <directory>   Screenshot and JSON artifact directory (default: ${DEFAULT_OUTPUT})
  --lidar-expected-voxels <count>  CPU reference count for lidar-ground (default: ${DEFAULT_LIDAR_EXPECTED_VOXELS})

The checker uses each story's configured dataset, waits for the showcase ready hook for each step,
verifies painted map canvas pixels, records browser/WebGPU diagnostics, and exits nonzero on missing
work or failures.
`);
}
