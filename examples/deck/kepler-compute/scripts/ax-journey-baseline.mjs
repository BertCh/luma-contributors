// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// AX-70 journey baseline for the Kepler host. Drives the real application in headless WebGPU
// Chromium and records, per journey, every user action the script performs (click, select-option,
// fill, press, file selection) and the wall-clock time from the first action to a stated end
// condition. Writes journeys.json and manifest.json (and a few small screenshots) so that the
// numbers can be re-measured after each release.
//
// Usage: node scripts/ax-journey-baseline.mjs [--url http://127.0.0.1:5391/] [--out <folder>]
//          [--rows 256] [--only <journey id>,<journey id>] [--timeout-ms 30000]
//
// A missing selector never skips a journey silently: the journey is recorded as failed with the
// journey id, the step and the selector, the remaining journeys still run, and the process exits
// with a non-zero status.
import {execFileSync} from 'node:child_process';
import {existsSync, mkdirSync, readFileSync, statSync, writeFileSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const SCRIPT_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const REPOSITORY_DIRECTORY = path.resolve(SCRIPT_DIRECTORY, '../../../..');
const DEFAULT_OUTPUT_DIRECTORY = path.join(
  REPOSITORY_DIRECTORY,
  'dev-docs/research/analysis-workflow-ux-2026-11-14/baseline'
);
const VIEWPORT = {width: 1440, height: 1000};
let waitTimeoutMilliseconds = 30000;
const MAXIMUM_SCREENSHOT_BYTES = 300 * 1024;
const GALLERY_SAMPLE_IDENTIFIERS = [
  'h3-cells',
  'points-time',
  'districts',
  'connections',
  'trips',
  'trip-observations',
  'numeric-grid',
  'directed-network'
];
const GALLERY_DEFAULT_SAMPLE_IDENTIFIER = 'points-time';
const COORDINATE_FIELD_NAMES = new Set([
  'longitude',
  'latitude',
  'targetLongitude',
  'targetLatitude'
]);

function parseOptions(argumentsList) {
  const options = {
    url: 'http://127.0.0.1:5391/',
    out: DEFAULT_OUTPUT_DIRECTORY,
    rows: 256,
    only: null,
    timeoutMilliseconds: 30000,
    startupTimeoutMilliseconds: 240000
  };
  for (let index = 0; index < argumentsList.length; index++) {
    const name = argumentsList[index];
    const value = argumentsList[++index];
    if (value === undefined || value.startsWith('--'))
      throw new Error(`Missing value for ${name}`);
    if (name === '--url') options.url = value;
    else if (name === '--out') options.out = path.resolve(value);
    else if (name === '--rows') options.rows = Number(value);
    else if (name === '--only') options.only = new Set(value.split(','));
    else if (name === '--timeout-ms') options.timeoutMilliseconds = Number(value);
    else throw new Error(`Unknown option ${name}`);
  }
  const address = new URL(options.url);
  if (!address.searchParams.has('rows')) address.searchParams.set('rows', String(options.rows));
  options.url = address.href;
  return options;
}

class JourneyFailure extends Error {
  constructor(journeyIdentifier, stepDescription, cause) {
    const reason = cause instanceof Error ? cause.message.split('\n')[0] : String(cause);
    super(`Journey ${journeyIdentifier}: ${stepDescription} failed: ${reason}`);
    this.journeyIdentifier = journeyIdentifier;
  }
}

/** Records user actions and timing for one journey on its own page. */
class JourneyRecorder {
  constructor(identifier, title, page) {
    this.identifier = identifier;
    this.title = title;
    this.page = page;
    this.steps = [];
    this.phase = 'setup';
    this.startedAt = null;
    this.setupFinishedAt = null;
    this.finishedAt = null;
    this.observations = {};
    this.probes = [];
    this.notes = [];
    this.consoleErrors = [];
    page.on('pageerror', error => this.consoleErrors.push(`pageerror: ${error.message}`));
    page.on('console', message => {
      if (message.type() === 'error') this.consoleErrors.push(message.text().slice(0, 300));
    });
  }

  /** Runs one counted user action. The first action starts the clock. */
  async act(action, target, operation) {
    if (this.startedAt === null) this.startedAt = performance.now();
    this.steps.push({action, target, phase: this.phase});
    try {
      await operation();
    } catch (failure) {
      throw new JourneyFailure(this.identifier, `${action} ${target}`, failure);
    }
  }

  click(locator) {
    return this.act('click', locator.toString(), () => locator.click());
  }

  selectOption(locator, value) {
    const label = typeof value === 'string' ? value : JSON.stringify(value);
    return this.act('select-option', `${locator.toString()} = ${label}`, () =>
      locator.selectOption(value)
    );
  }

  selectFile(locator, file) {
    return this.act('select-file', `${locator.toString()} = ${file.name}`, () =>
      locator.setInputFiles(file)
    );
  }

  /** Waits for a condition that is not a user action, failing loudly with its description. */
  async waitUntil(description, pageFunction, argument, timeout) {
    try {
      return await this.page.waitForFunction(pageFunction, argument, {
        timeout: timeout ?? waitTimeoutMilliseconds
      });
    } catch (failure) {
      throw new JourneyFailure(this.identifier, `waiting for ${description}`, failure);
    }
  }

  /** Reads page state without acting; a missing selector fails the journey. */
  async observe(description, operation) {
    try {
      return await operation();
    } catch (failure) {
      throw new JourneyFailure(this.identifier, `reading ${description}`, failure);
    }
  }

  markSetupFinished() {
    this.setupFinishedAt = performance.now();
    this.phase = 'analysis';
  }

  markFinished() {
    this.finishedAt = performance.now();
  }

  toRecord(fields) {
    const setupActionCount = this.steps.filter(step => step.phase === 'setup').length;
    return {
      id: this.identifier,
      title: this.title,
      steps: this.steps,
      actionCount: this.steps.length,
      setupActionCount,
      analysisActionCount: this.steps.length - setupActionCount,
      milliseconds:
        this.startedAt !== null && this.finishedAt !== null
          ? Math.round(this.finishedAt - this.startedAt)
          : null,
      setupMilliseconds:
        this.startedAt !== null && this.setupFinishedAt !== null
          ? Math.round(this.setupFinishedAt - this.startedAt)
          : null,
      ...fields,
      observations: this.observations,
      probes: this.probes,
      notes: this.notes,
      consoleErrors: [...new Set(this.consoleErrors)].slice(0, 5)
    };
  }
}

async function startApplication(page, options) {
  await page.goto(options.url, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => {
      const state = window.__keplerComputeTest?.getState();
      if (state?.error) throw new Error(state.error);
      return state?.ready === true;
    },
    undefined,
    {timeout: options.startupTimeoutMilliseconds}
  );
}

/** Add Data, Prepare data, open the gallery, choose, prepare, check, publish. */
async function loadSampleThroughGallery(recorder, sampleIdentifier) {
  const {page} = recorder;
  await recorder.click(page.getByRole('button', {name: 'Add Data', exact: true}));
  await recorder.click(page.getByText('Prepare data', {exact: true}));
  await recorder.click(page.getByText('Try a sample dataset', {exact: true}));
  const gallerySelect = page.locator('.sample-gallery select');
  const defaultIdentifier = await recorder.observe('gallery sample select', () =>
    gallerySelect.inputValue()
  );
  if (defaultIdentifier !== sampleIdentifier) {
    await recorder.selectOption(gallerySelect, sampleIdentifier);
  } else {
    recorder.notes.push(`${sampleIdentifier} is the gallery default, so no select action is needed.`);
  }
  const title = await recorder.observe('gallery sample title', () =>
    gallerySelect.locator(`option[value="${sampleIdentifier}"]`).innerText()
  );
  const rowCountText = await recorder.observe('gallery row count text', () =>
    page.locator('.sample-gallery').innerText()
  );
  const rowCount = Number(/(\d+) rows ·/.exec(rowCountText)?.[1] ?? Number.NaN);
  await recorder.click(page.getByRole('button', {name: 'Prepare sample', exact: true}));
  await recorder.click(page.getByRole('button', {name: 'Check schema and precision', exact: true}));
  await recorder.click(page.getByRole('button', {name: 'Publish selected fields', exact: true}));
  await recorder.waitUntil(
    'the sample dataset in the analysis dataset list',
    sampleTitle =>
      Array.from(document.querySelectorAll('#analysis-dataset option')).some(
        option => option.textContent === sampleTitle
      ),
    title
  );
  await recorder.observe('closed Add Data dialog', () =>
    page.locator('.ReactModal__Overlay').waitFor({state: 'detached', timeout: 15000})
  );
  return {title, rowCount};
}

function getAnalysisSection(page) {
  return page.locator('section[aria-label="GPU analysis"]');
}

/** Chooses the loaded sample in the analysis panel when it is not already the selected dataset. */
async function chooseAnalysisDataset(recorder, title) {
  const datasetSelect = getAnalysisSection(recorder.page).locator('#analysis-dataset');
  const selectedLabel = await recorder.observe('analysis dataset select', () =>
    datasetSelect.evaluate(element => element.selectedOptions[0]?.textContent ?? '')
  );
  recorder.observations.analysisDatasetSelectedAfterLoad = selectedLabel;
  recorder.observations.analysisDatasetOptions = await recorder.observe(
    'analysis dataset options',
    () => datasetSelect.locator('option').allTextContents()
  );
  if (selectedLabel === title) {
    recorder.notes.push('The loaded sample was already selected in the analysis panel.');
    return;
  }
  await recorder.selectOption(datasetSelect, {label: title});
}

/** Text of the analysis result block the panel renders after an operation has run. */
const ANALYSIS_SUMMARY_SELECTOR = 'section[aria-label="GPU analysis"] [aria-live="polite"]';

async function waitForAnalysisSummary(recorder, pattern, description) {
  const handle = await recorder.waitUntil(
    description,
    ({selector, source}) => {
      const failure = document.querySelector(
        'section[aria-label="GPU analysis"] .error, section[aria-label="GPU analysis"] [role="alert"]'
      );
      if (failure?.textContent) return {failed: failure.textContent};
      const text = Array.from(document.querySelectorAll(selector))
        .map(element => element.innerText)
        .join('\n');
      return new RegExp(source).test(text) ? {text} : false;
    },
    {selector: ANALYSIS_SUMMARY_SELECTOR, source: pattern.source}
  );
  const outcome = await handle.jsonValue();
  if (outcome.failed)
    throw new JourneyFailure(recorder.identifier, `waiting for ${description}`, outcome.failed);
  return outcome.text;
}

function parseAnalysisSummary(text) {
  const counts = /([\d,]+) analyzed · ([\d,]+) undefined/.exec(text);
  const range = /range: (\S+) to (\S+)/.exec(text);
  return {
    analyzedCount: counts ? Number(counts[1].replace(/,/g, '')) : null,
    undefinedCount: counts ? Number(counts[2].replace(/,/g, '')) : null,
    rangeMinimum: range ? range[1] : null,
    rangeMaximum: range ? range[2] : null,
    text: text.replace(/\n/g, ' | ')
  };
}

async function readLegendText(recorder) {
  const {page} = recorder;
  const button = page.locator('.map-control-button.show-legend');
  await button.click();
  const panel = page.locator('.map-control-panel');
  await panel.waitFor({state: 'visible'});
  return panel.innerText();
}

async function saveScreenshot(page, outputDirectory, name, files) {
  const file = path.join(outputDirectory, name);
  await page.screenshot({path: file, type: 'jpeg', quality: 60});
  const bytes = statSync(file).size;
  if (bytes > MAXIMUM_SCREENSHOT_BYTES)
    throw new Error(`Screenshot ${name} is ${bytes} bytes, above the ${MAXIMUM_SCREENSHOT_BYTES} limit`);
  files.push({name, bytes});
}

/** Counts dominant colors in the map area to show whether a layer is one color or styled. */
async function countDominantMapColors(page) {
  const image = PNG.sync.read(await page.screenshot());
  const counts = new Map();
  for (let y = 40; y < image.height - 60; y++) {
    for (let x = 330; x < image.width - 80; x++) {
      const offset = (y * image.width + x) * 4;
      const key = [0, 1, 2].map(channel => image.data[offset + channel] >> 4).join(',');
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  const total = image.width * image.height;
  return [...counts.entries()]
    .filter(([, count]) => count > total * 0.002)
    .sort((first, second) => second[1] - first[1])
    .map(([key, count]) => ({quantizedRgb: key.split(',').map(value => Number(value) * 16), pixels: count}));
}

async function readFormControls(recorder, formSelector) {
  return recorder.observe(`form controls in ${formSelector}`, () =>
    recorder.page.locator(formSelector).evaluate(root => {
      const controls = Array.from(root.querySelectorAll('select, input, textarea'));
      return controls.map(control => {
        const labelElement = control.id ? root.querySelector(`label[for="${control.id}"]`) : null;
        const labelText =
          control.getAttribute('aria-label') ||
          (labelElement
            ? Array.from(labelElement.childNodes)
                .filter(node => node.nodeType === Node.TEXT_NODE)
                .map(node => node.textContent)
                .join('')
                .trim() || labelElement.textContent.trim()
            : control.id);
        return {
          id: control.id,
          label: labelText,
          value: control.type === 'checkbox' ? String(control.checked) : control.value
        };
      });
    })
  );
}

// ---------------------------------------------------------------------------------------------
// Journeys
// ---------------------------------------------------------------------------------------------

/** Journey 1: gallery sample to the first analysis result the interface offers. */
async function journeyFirstResult(page, options, sampleIdentifier, screenshotFiles) {
  const recorder = new JourneyRecorder(
    `first-result:${sampleIdentifier}`,
    `Gallery sample ${sampleIdentifier} to first analysis result`,
    page
  );
  await startApplication(page, options);
  const {title, rowCount} = await loadSampleThroughGallery(recorder, sampleIdentifier);
  recorder.markSetupFinished();
  await chooseAnalysisDataset(recorder, title);
  const analysis = getAnalysisSection(page);
  const defaultField = await recorder.observe('numeric field select', () =>
    analysis.locator('#analysis-column').inputValue()
  );
  const fieldOptions = await recorder.observe('numeric field options', () =>
    analysis.locator('#analysis-column option').allTextContents()
  );
  recorder.observations.sampleRowCount = rowCount;
  recorder.observations.numericFieldOptions = fieldOptions;
  recorder.observations.defaultNumericField = defaultField;
  recorder.observations.defaultFieldIsCoordinate = COORDINATE_FIELD_NAMES.has(defaultField);
  if (fieldOptions.length === 0) {
    recorder.markFinished();
    return recorder.toRecord({
      endCondition: 'The numeric field list of the analysis panel is empty, so no operation can run.',
      outcome: 'noAnalysisAvailable'
    });
  }
  await recorder.selectOption(analysis.locator('#analysis-operation'), 'z-score');
  const text = await waitForAnalysisSummary(
    recorder,
    /[\d,]+ analyzed · [\d,]+ undefined[\s\S]*Transformed value range/,
    'the analysis result summary (N analyzed, transformed value range)'
  );
  recorder.markFinished();
  const summary = parseAnalysisSummary(text);
  recorder.observations.resultSummary = summary;
  if (summary.rangeMinimum === summary.rangeMaximum)
    recorder.notes.push(
      'The result range is a single value, so the first result is degenerate for this sample (the default field is constant).'
    );
  if (summary.analyzedCount !== rowCount)
    recorder.notes.push(
      `${summary.analyzedCount} of ${rowCount} rows were analyzed; the rest are undefined (missing values).`
    );
  if (recorder.observations.defaultFieldIsCoordinate)
    recorder.notes.push(
      `The default numeric field is the coordinate column ${defaultField}; analysing a measurement needs one more select action.`
    );
  // Probe after the clock stopped: not a counted step.
  recorder.probes.push({
    probe: 'legend panel text after the result (opened by clicking the map legend button; not counted)',
    value: await readLegendText(recorder)
  });
  if (sampleIdentifier === 'points-time')
    await saveScreenshot(page, options.out, 'first-result-points-time.jpg', screenshotFiles);
  return recorder.toRecord({
    endCondition:
      'The GPU analysis panel shows a live result block with "N analyzed · M undefined" and a transformed value range for the chosen dataset.',
    outcome: 'result-visible'
  });
}

/** Journey 2: point hot spots on the points sample through the hand-written panel. */
async function journeyPointHotspots(page, options, useReadingField, screenshotFiles) {
  const recorder = new JourneyRecorder(
    useReadingField ? 'point-hotspots:reading-field' : 'point-hotspots:default-field',
    `Point hot spots on the points sample${useReadingField ? ' with the reading field chosen' : ' with the default numeric field'}`,
    page
  );
  await startApplication(page, options);
  const {title} = await loadSampleThroughGallery(recorder, GALLERY_DEFAULT_SAMPLE_IDENTIFIER);
  recorder.markSetupFinished();
  await chooseAnalysisDataset(recorder, title);
  const analysis = getAnalysisSection(page);
  recorder.observations.defaultNumericField = await recorder.observe('numeric field select', () =>
    analysis.locator('#analysis-column').inputValue()
  );
  await recorder.click(analysis.locator('.analysis-point-hotspot-settings > summary'));
  const controls = await readFormControls(
    recorder,
    'section[aria-label="GPU analysis"] .analysis-point-hotspot-settings form'
  );
  recorder.observations.formControlsBeforeEditing = controls;
  recorder.observations.blankFieldsBeforeEditing = controls
    .filter(control => control.value === '')
    .map(control => control.label);
  if (useReadingField)
    await recorder.selectOption(analysis.locator('#analysis-column'), 'reading');
  await recorder.selectOption(analysis.locator('#analysis-point-hotspot-longitude'), 'longitude');
  await recorder.selectOption(analysis.locator('#analysis-point-hotspot-latitude'), 'latitude');
  await recorder.click(
    analysis.getByRole('button', {name: 'Apply neighbors and run point hot spots', exact: true})
  );
  const text = await waitForAnalysisSummary(
    recorder,
    /[\d,]+ analyzed · [\d,]+ undefined[\s\S]*Hot-spot z-score range/,
    'the point hot-spot result summary'
  );
  recorder.markFinished();
  recorder.observations.resultSummary = parseAnalysisSummary(text);
  recorder.notes.push(
    'Action count excluding the dataset selection (the convention of the plan) is analysisActionCount minus one when the dataset select was a counted step.'
  );
  if (!useReadingField)
    await saveScreenshot(page, options.out, 'point-hotspots-result.jpg', screenshotFiles);
  return recorder.toRecord({
    endCondition:
      'The GPU analysis panel shows "N analyzed · M undefined" and a "Hot-spot z-score range" line for the points sample.',
    outcome: 'result-visible'
  });
}

function getCatalogPanel(page) {
  return page.getByRole('region', {name: 'Analysis catalog', exact: true});
}

async function readCatalogDatasetOptions(recorder) {
  const panel = getCatalogPanel(recorder.page);
  const sourceSelect = panel.locator('select[aria-label$=" source"]').first();
  await recorder.selectOption(sourceSelect, 'native');
  const datasetSelect = panel.locator('select[aria-label$=" dataset"]').first();
  return recorder.observe('catalog dataset select', () => datasetSelect.locator('option').allTextContents());
}

/** Journey 3a: what the catalog dataset list holds after a gallery sample with geometry is prepared. */
async function journeyCatalogAfterGallerySample(page, options) {
  const recorder = new JourneyRecorder(
    'catalog-datasets:after-gallery-sample',
    'Catalog dataset list after the districts sample is prepared through the gallery',
    page
  );
  await startApplication(page, options);
  const {title} = await loadSampleThroughGallery(recorder, 'districts');
  recorder.markSetupFinished();
  const catalogOptions = await readCatalogDatasetOptions(recorder);
  recorder.markFinished();
  recorder.observations.catalogDatasetOptions = catalogOptions;
  recorder.observations.analysisDatasetOptions = await recorder.observe(
    'analysis dataset options',
    () => getAnalysisSection(page).locator('#analysis-dataset option').allTextContents()
  );
  recorder.observations.preparedDatasetInCatalog = catalogOptions.includes(title);
  return recorder.toRecord({
    endCondition:
      'The first catalog input is switched to Native dataset and its dataset select options are read.',
    outcome: 'observed'
  });
}

/** Journey 3b: the same observation after a GeoJSON file is added through Add Data (auto-prepared). */
async function journeyCatalogAfterNativeLoad(page, options, screenshotFiles) {
  const recorder = new JourneyRecorder(
    'catalog-datasets:after-native-geojson-load',
    'Catalog dataset list after a GeoJSON file is added through Add Data and auto-prepared',
    page
  );
  await startApplication(page, options);
  const square = (west, south) => [
    [
      [west, south],
      [west + 0.1, south],
      [west + 0.1, south + 0.1],
      [west, south + 0.1],
      [west, south]
    ]
  ];
  const file = {
    name: 'ax-two-squares.geojson',
    mimeType: 'application/geo+json',
    buffer: Buffer.from(
      JSON.stringify({
        type: 'FeatureCollection',
        features: [
          {type: 'Feature', properties: {value: 1}, geometry: {type: 'Polygon', coordinates: square(-122.5, 37.7)}},
          {type: 'Feature', properties: {value: 2}, geometry: {type: 'Polygon', coordinates: square(-122.4, 37.7)}}
        ]
      })
    )
  };
  await recorder.click(page.getByRole('button', {name: 'Add Data', exact: true}));
  await recorder.selectFile(page.locator('.ReactModal__Content input[type=file]').first(), file);
  await recorder.click(
    page.locator('.ReactModal__Content').getByRole('button', {name: 'Add Data', exact: true}).last()
  );
  await recorder.observe('closed Add Data dialog', () =>
    page.locator('.ReactModal__Overlay').waitFor({state: 'detached', timeout: 30000})
  );
  let autoPrepared = true;
  try {
    await recorder.waitUntil(
      'the file in the analysis dataset list (automatic preparation)',
      name =>
        Array.from(document.querySelectorAll('#analysis-dataset option')).some(
          option => option.textContent === name
        ),
      file.name,
      30000
    );
  } catch {
    autoPrepared = false;
  }
  recorder.markSetupFinished();
  const catalogOptions = await readCatalogDatasetOptions(recorder);
  recorder.markFinished();
  recorder.observations.autoPrepared = autoPrepared;
  recorder.observations.keplerDatasetCountText = await recorder.observe('Datasets heading', () =>
    page.getByText(/^Datasets\(\d+\)$/).first().innerText()
  );
  recorder.observations.catalogDatasetOptions = catalogOptions;
  recorder.observations.analysisDatasetOptions = await recorder.observe(
    'analysis dataset options',
    () => getAnalysisSection(page).locator('#analysis-dataset option').allTextContents()
  );
  recorder.observations.preparedDatasetInCatalog = catalogOptions.includes(file.name);
  await saveScreenshot(page, options.out, 'catalog-dataset-list-after-native-load.jpg', screenshotFiles);
  return recorder.toRecord({
    endCondition:
      'The first catalog input is switched to Native dataset and its dataset select options are read after the file was auto-prepared.',
    outcome: 'observed'
  });
}

/** Journeys 3c and 4: the shortest catalog route (built-in sample input) to a published layer, and its look. */
async function journeyCatalogBuiltInSample(page, options, screenshotFiles) {
  const recorder = new JourneyRecorder(
    'catalog-builtin-sample:to-published-layer',
    'Catalog default operation on its built-in sample to a published layer',
    page
  );
  await startApplication(page, options);
  recorder.phase = 'analysis';
  const panel = getCatalogPanel(page);
  recorder.observations.defaultOperation = await recorder.observe('catalog operation select', () =>
    panel.getByLabel('Operation', {exact: true}).inputValue()
  );
  const layersBefore = await recorder.observe('native layers', () =>
    page.evaluate(() => window.__keplerComputeTest.getState().nativeLayers.length)
  );
  await recorder.click(panel.getByTestId('catalog-run'));
  await recorder.waitUntil(
    'the catalog result block',
    () => Boolean(document.querySelector('[data-testid="catalog-result"]'))
  );
  recorder.observations.catalogResultText = await recorder.observe('catalog result', () =>
    panel.getByTestId('catalog-result').innerText().then(text => text.slice(0, 600))
  );
  await recorder.click(panel.getByTestId('catalog-publish').first());
  await recorder.click(page.getByRole('button', {name: 'Add Data', exact: true}));
  await recorder.click(page.getByText('Prepare data', {exact: true}));
  await recorder.click(page.getByRole('button', {name: 'Check schema and precision', exact: true}));
  await recorder.click(page.getByRole('button', {name: 'Publish selected fields', exact: true}));
  await recorder.waitUntil(
    'a new visible valid layer in the native layer list',
    count => {
      const layers = window.__keplerComputeTest.getState().nativeLayers;
      return layers.length > count && layers.every(layer => layer.valid && !layer.error);
    },
    layersBefore
  );
  await recorder.observe('closed Add Data dialog', () =>
    page.locator('.ReactModal__Overlay').waitFor({state: 'detached', timeout: 15000})
  );
  recorder.markFinished();
  recorder.observations.nativeLayers = await page.evaluate(() =>
    window.__keplerComputeTest.getState().nativeLayers.map(layer => ({
      id: layer.id,
      type: layer.type,
      dataId: layer.dataId,
      visible: layer.visible
    }))
  );
  const published = recorder.toRecord({
    endCondition:
      'A new valid native layer for the published catalog table exists in the compute runtime state.',
    outcome: 'result-visible'
  });

  // Journey 4: how the published layer looks. One extra action: open the map legend.
  const styleRecorder = new JourneyRecorder(
    'catalog-result:published-style',
    'Look of the published catalog result (same page as catalog-builtin-sample:to-published-layer)',
    page
  );
  styleRecorder.phase = 'analysis';
  const legendButton = page.locator('.map-control-button.show-legend');
  await styleRecorder.click(legendButton);
  const legendText = await styleRecorder.observe('legend panel', () =>
    page.locator('.map-control-panel').innerText()
  );
  await styleRecorder.waitUntil('the legend panel text', () =>
    Boolean(document.querySelector('.map-control-panel'))
  );
  styleRecorder.markFinished();
  styleRecorder.observations.legendText = legendText;
  styleRecorder.observations.legendMentionsColorByField = /ramp|quantile|threshold|by /i.test(legendText);
  styleRecorder.observations.dominantMapColors = await countDominantMapColors(page);
  styleRecorder.observations.layerPanelTypes = await styleRecorder.observe('layer panels', () =>
    page.locator('.layer-panel').allInnerTexts().then(texts => texts.map(text => text.split('\n')[0]))
  );
  await saveScreenshot(page, options.out, 'catalog-published-layer-legend.jpg', screenshotFiles);
  const styleRecord = styleRecorder.toRecord({
    endCondition: 'The map legend panel is open and its text is read; a screenshot is saved.',
    outcome: 'observed'
  });
  return [published, styleRecord];
}

/** Journey 5: IDW and kriging default bounds on the points sample, beside the sample extent. */
async function journeyInterpolationDefaults(page, options, screenshotFiles) {
  const recorder = new JourneyRecorder(
    'interpolation-defaults:points-sample',
    'IDW and kriging default bounds on the points sample',
    page
  );
  await startApplication(page, options);
  const {title} = await loadSampleThroughGallery(recorder, GALLERY_DEFAULT_SAMPLE_IDENTIFIER);
  recorder.markSetupFinished();
  await chooseAnalysisDataset(recorder, title);
  const analysis = getAnalysisSection(page);
  await recorder.click(analysis.locator('.analysis-interpolation-settings > summary'));
  const idwControls = await readFormControls(
    recorder,
    'section[aria-label="GPU analysis"] .analysis-interpolation-settings form'
  );
  await recorder.click(analysis.locator('.analysis-kriging-settings > summary'));
  const krigingControls = await readFormControls(
    recorder,
    'section[aria-label="GPU analysis"] .analysis-kriging-settings form'
  );
  recorder.markFinished();
  const pick = (controls, suffix) => controls.find(control => control.id.endsWith(suffix))?.value;
  const bounds = controls => ({
    west: pick(controls, '-west'),
    south: pick(controls, '-south'),
    east: pick(controls, '-east'),
    north: pick(controls, '-north')
  });
  recorder.observations.idwDefaultBounds = bounds(idwControls);
  recorder.observations.krigingDefaultBounds = bounds(krigingControls);
  recorder.observations.idwBlankFields = idwControls
    .filter(control => control.value === '')
    .map(control => control.label);
  recorder.observations.krigingBlankFields = krigingControls
    .filter(control => control.value === '')
    .map(control => control.label);
  recorder.observations.idwControls = idwControls;
  recorder.observations.krigingControls = krigingControls;
  // Actual extents of the gallery samples that carry coordinates, from the sample generator itself.
  recorder.observations.sampleExtents = await recorder.observe('sample extents', () =>
    page.evaluate(async sampleIdentifiers => {
      const module = await import('/src/sandbox-samples.ts');
      const extents = {};
      for (const identifier of sampleIdentifiers) {
        const table = module.createSandboxSample(identifier).table;
        const longitude = table.getChild('longitude');
        const latitude = table.getChild('latitude');
        if (!longitude || !latitude) continue;
        const range = vector => {
          let minimum = Infinity;
          let maximum = -Infinity;
          for (const value of vector.toArray()) {
            if (Number.isNaN(value)) continue;
            if (value < minimum) minimum = value;
            if (value > maximum) maximum = value;
          }
          return [minimum, maximum];
        };
        const [west, east] = range(longitude);
        const [south, north] = range(latitude);
        extents[identifier] = {west, south, east, north};
      }
      return extents;
    }, ['points-time', 'connections', 'trip-observations', 'numeric-grid'])
  );
  const fixed = recorder.observations.idwDefaultBounds;
  const extent = recorder.observations.sampleExtents['points-time'];
  recorder.observations.fixedBoundsContainSampleExtent =
    Number(fixed.west) <= extent.west &&
    Number(fixed.east) >= extent.east &&
    Number(fixed.south) <= extent.south &&
    Number(fixed.north) >= extent.north;
  recorder.observations.fixedBoundsEqualSampleExtent =
    Number(fixed.west) === extent.west &&
    Number(fixed.east) === extent.east &&
    Number(fixed.south) === extent.south &&
    Number(fixed.north) === extent.north;
  await saveScreenshot(page, options.out, 'interpolation-default-bounds.jpg', screenshotFiles);
  return recorder.toRecord({
    endCondition: 'The IDW and kriging forms are open and their default bounds are read.',
    outcome: 'observed'
  });
}

// ---------------------------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------------------------

function readCommit() {
  try {
    const headText = readFileSync(path.join(REPOSITORY_DIRECTORY, '.git/HEAD'), 'utf8').trim();
    if (!headText.startsWith('ref: ')) return {head: headText, reference: null};
    const reference = headText.slice(5);
    const referenceFile = path.join(REPOSITORY_DIRECTORY, '.git', reference);
    if (existsSync(referenceFile))
      return {head: readFileSync(referenceFile, 'utf8').trim(), reference};
    const packed = readFileSync(path.join(REPOSITORY_DIRECTORY, '.git/packed-refs'), 'utf8')
      .split('\n')
      .find(line => line.endsWith(` ${reference}`));
    return {head: packed ? packed.split(' ')[0] : null, reference};
  } catch (failure) {
    return {head: null, reference: null, error: String(failure)};
  }
}

function readSysctl(name) {
  try {
    return execFileSync('sysctl', ['-n', name], {encoding: 'utf8'}).trim();
  } catch {
    return null;
  }
}

async function readAdapterInformation(page) {
  return page.evaluate(async () => {
    if (!navigator.gpu) return {error: 'navigator.gpu is not available'};
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return {error: 'navigator.gpu.requestAdapter() returned null'};
    const info = adapter.info ?? (await adapter.requestAdapterInfo?.()) ?? {};
    return {
      vendor: info.vendor ?? '',
      architecture: info.architecture ?? '',
      device: info.device ?? '',
      description: info.description ?? '',
      isFallbackAdapter: adapter.isFallbackAdapter ?? info.isFallbackAdapter ?? null,
      featureCount: adapter.features ? adapter.features.size : null,
      maxBufferSize: adapter.limits?.maxBufferSize ?? null,
      maxStorageBufferBindingSize: adapter.limits?.maxStorageBufferBindingSize ?? null,
      userAgent: navigator.userAgent
    };
  });
}

// ---------------------------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------------------------

async function main() {
  const options = parseOptions(process.argv.slice(2));
  waitTimeoutMilliseconds = options.timeoutMilliseconds;
  mkdirSync(options.out, {recursive: true});
  const launchOptions = getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'});
  const browser = await chromium.launch(launchOptions);
  const records = [];
  const failures = [];
  const screenshotFiles = [];
  const startedAt = new Date();
  let adapterInformation = null;
  const journeys = [
    ...GALLERY_SAMPLE_IDENTIFIERS.map(sampleIdentifier => ({
      id: `first-result:${sampleIdentifier}`,
      run: page => journeyFirstResult(page, options, sampleIdentifier, screenshotFiles)
    })),
    {
      id: 'point-hotspots:default-field',
      run: page => journeyPointHotspots(page, options, false, screenshotFiles)
    },
    {
      id: 'point-hotspots:reading-field',
      run: page => journeyPointHotspots(page, options, true, screenshotFiles)
    },
    {
      id: 'catalog-datasets:after-gallery-sample',
      run: page => journeyCatalogAfterGallerySample(page, options)
    },
    {
      id: 'catalog-datasets:after-native-geojson-load',
      run: page => journeyCatalogAfterNativeLoad(page, options, screenshotFiles)
    },
    {
      id: 'catalog-builtin-sample:to-published-layer',
      run: page => journeyCatalogBuiltInSample(page, options, screenshotFiles)
    },
    {
      id: 'interpolation-defaults:points-sample',
      run: page => journeyInterpolationDefaults(page, options, screenshotFiles)
    }
  ].filter(journey => !options.only || [...options.only].some(prefix => journey.id.startsWith(prefix)));
  if (journeys.length === 0) throw new Error('--only matched no journey');

  try {
    for (const journey of journeys) {
      const page = await browser.newPage({viewport: VIEWPORT});
      page.setDefaultTimeout(options.timeoutMilliseconds);
      await pinViteModules(page);
      try {
        const result = await journey.run(page);
        for (const record of Array.isArray(result) ? result : [result]) records.push(record);
        process.stdout.write(`ok ${journey.id}\n`);
        if (adapterInformation === null) adapterInformation = await readAdapterInformation(page);
      } catch (failure) {
        const message = failure instanceof Error ? failure.message : String(failure);
        failures.push({id: journey.id, message});
        records.push({id: journey.id, outcome: 'failed', error: message});
        process.stderr.write(`FAILED ${message}\n`);
        await page
          .screenshot({path: path.join(options.out, `failed-${journey.id.replace(/[^\w-]/g, '_')}.jpg`), type: 'jpeg', quality: 50})
          .catch(() => undefined);
      } finally {
        await page.close();
      }
    }
    const browserVersion = browser.version();
    const manifest = {
      date: startedAt.toISOString(),
      commit: readCommit(),
      commitNote:
        'Read from .git/HEAD without running git. The measured application is the working tree, which may hold uncommitted changes.',
      browser: {name: 'chromium', version: browserVersion, launchOptions},
      webgpuAdapter: adapterInformation,
      machine: {
        model: readSysctl('hw.model'),
        memoryBytes: Number(readSysctl('hw.memsize')),
        cpu: readSysctl('machdep.cpu.brand_string'),
        platform: `${os.platform()} ${os.release()} ${os.arch()}`,
        nodeVersion: process.version
      },
      url: options.url,
      parameters: {
        rows: options.rows,
        viewport: VIEWPORT,
        timeoutMilliseconds: options.timeoutMilliseconds,
        only: options.only ? [...options.only] : null
      },
      script: 'examples/deck/kepler-compute/scripts/ax-journey-baseline.mjs',
      journeyIds: records.map(record => record.id),
      failedJourneys: failures,
      screenshots: screenshotFiles
    };
    writeFileSync(path.join(options.out, 'journeys.json'), `${JSON.stringify(records, null, 2)}\n`);
    writeFileSync(path.join(options.out, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  } finally {
    await browser.close();
  }
  if (failures.length > 0) {
    process.stderr.write(`${failures.length} journey(s) failed; see journeys.json\n`);
    process.exitCode = 1;
  }
}

await main();
