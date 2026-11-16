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
//          [--release <version>] [--targets <json>] [--rows 256]
//          [--only <journey id>,<journey id>] [--timeout-ms 30000]
// Defaults to releases/<version>/<timestamp>; existing evidence is never overwritten.
//
// A missing selector never skips a journey silently: the journey is recorded as failed with the
// journey id, the step and the selector, the remaining journeys still run, and the process exits
// with a non-zero status. New AX surfaces are optional: absence is explicitly unqualified.
import {readAnalysisSheetField} from './analysis-sheet-controls.mjs';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {
  existsSync,
  lstatSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  readlinkSync,
  realpathSync,
  statSync,
  writeFileSync
} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {inflateSync} from 'node:zlib';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {
  AX_SURFACE_IDENTIFIERS,
  accountJourney,
  compareJourneys,
  evaluateJourneyTargets,
  GALLERY_SAMPLE_IDENTIFIERS,
  getSuggestionResultFailure,
  validateJourneyTargets
} from './ax-journey-model.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {TEST_IDS} from './workflow-utils.mjs';

const SCRIPT_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const REPOSITORY_DIRECTORY = path.resolve(SCRIPT_DIRECTORY, '../../../..');
const HISTORICAL_BASELINE_DIRECTORY = path.join(
  REPOSITORY_DIRECTORY,
  'dev-docs/research/analysis-workflow-ux-2026-11-14/baseline'
);
const VIEWPORT = {width: 1440, height: 1000};
let waitTimeoutMilliseconds = 30000;
const MAXIMUM_SCREENSHOT_BYTES = 300 * 1024;
const GALLERY_DEFAULT_SAMPLE_IDENTIFIER = 'points-time';
const COORDINATE_FIELD_NAMES = new Set([
  'longitude',
  'latitude',
  'targetLongitude',
  'targetLatitude'
]);

export function parseOptions(argumentsList) {
  const options = {
    url: 'http://127.0.0.1:5391/',
    out: null,
    release: 'working-tree',
    targets: path.join(SCRIPT_DIRECTORY, 'ax-journey-targets.json'),
    rows: 256,
    only: null,
    timeoutMilliseconds: 30000,
    startupTimeoutMilliseconds: 240000
  };
  for (let index = 0; index < argumentsList.length; index++) {
    const name = argumentsList[index];
    const value = argumentsList[++index];
    if (value === undefined || value.startsWith('--')) throw new Error(`Missing value for ${name}`);
    if (name === '--url') options.url = value;
    else if (name === '--out') options.out = path.resolve(value);
    else if (name === '--release') options.release = value;
    else if (name === '--targets') options.targets = path.resolve(value);
    else if (name === '--rows') options.rows = Number(value);
    else if (name === '--only') options.only = new Set(value.split(','));
    else if (name === '--timeout-ms') options.timeoutMilliseconds = Number(value);
    else throw new Error(`Unknown option ${name}`);
  }
  if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]*$/.test(options.release))
    throw new Error('--release must be a single path-safe release identifier');
  for (const value of [options.rows, options.timeoutMilliseconds]) {
    if (!Number.isSafeInteger(value) || value <= 0)
      throw new Error('--rows and --timeout-ms must be positive safe integers');
  }
  options.out ??= path.join(
    path.dirname(HISTORICAL_BASELINE_DIRECTORY),
    'releases',
    options.release,
    new Date().toISOString().replace(/[:.]/g, '-')
  );
  // Resolve existing ancestors as well: a symlink must not bypass baseline protection.
  let ancestor = options.out;
  while (!existsSync(ancestor)) ancestor = path.dirname(ancestor);
  const resolvedOutput = path.resolve(realpathSync(ancestor), path.relative(ancestor, options.out));
  const baseline = realpathSync(HISTORICAL_BASELINE_DIRECTORY);
  if (resolvedOutput === baseline || resolvedOutput.startsWith(`${baseline}${path.sep}`))
    throw new Error('AX-70: historical baseline output is read-only; choose a release folder');
  if (existsSync(options.out) && readdirSync(options.out).length > 0)
    throw new Error('AX-70: output folder must be empty to preserve previous run evidence');
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
export class JourneyRecorder {
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
    // Retain partial action evidence even when a selector fails.
    page.axJourneyRecorder = this;
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

  press(locator, key) {
    return this.act('press', `${locator.toString()} ${key}`, () => locator.press(key));
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
    return {
      id: this.identifier,
      title: this.title,
      steps: this.steps,
      ...accountJourney(this),
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
    recorder.notes.push(
      `${sampleIdentifier} is the gallery default, so no select action is needed.`
    );
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
    throw new Error(
      `Screenshot ${name} is ${bytes} bytes, above the ${MAXIMUM_SCREENSHOT_BYTES} limit`
    );
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
    .map(([key, count]) => ({
      quantizedRgb: key.split(',').map(value => Number(value) * 16),
      pixels: count
    }));
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
      endCondition:
        'The numeric field list of the analysis panel is empty, so no operation can run.',
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
    probe:
      'legend panel text after the result (opened by clicking the map legend button; not counted)',
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
  if (useReadingField) await recorder.selectOption(analysis.locator('#analysis-column'), 'reading');
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
  return recorder.observe('catalog dataset select', () =>
    datasetSelect.locator('option').allTextContents()
  );
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
          {
            type: 'Feature',
            properties: {value: 1},
            geometry: {type: 'Polygon', coordinates: square(-122.5, 37.7)}
          },
          {
            type: 'Feature',
            properties: {value: 2},
            geometry: {type: 'Polygon', coordinates: square(-122.4, 37.7)}
          }
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
    page
      .getByText(/^Datasets\(\d+\)$/)
      .first()
      .innerText()
  );
  recorder.observations.catalogDatasetOptions = catalogOptions;
  recorder.observations.analysisDatasetOptions = await recorder.observe(
    'analysis dataset options',
    () => getAnalysisSection(page).locator('#analysis-dataset option').allTextContents()
  );
  recorder.observations.preparedDatasetInCatalog = catalogOptions.includes(file.name);
  await saveScreenshot(
    page,
    options.out,
    'catalog-dataset-list-after-native-load.jpg',
    screenshotFiles
  );
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
  await recorder.waitUntil('the catalog result block', () =>
    Boolean(document.querySelector('[data-testid="catalog-result"]'))
  );
  recorder.observations.catalogResultText = await recorder.observe('catalog result', () =>
    panel
      .getByTestId('catalog-result')
      .innerText()
      .then(text => text.slice(0, 600))
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
  styleRecorder.observations.legendMentionsColorByField = /ramp|quantile|threshold|by /i.test(
    legendText
  );
  styleRecorder.observations.dominantMapColors = await countDominantMapColors(page);
  styleRecorder.observations.layerPanelTypes = await styleRecorder.observe('layer panels', () =>
    page
      .locator('.layer-panel')
      .allInnerTexts()
      .then(texts => texts.map(text => text.split('\n')[0]))
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
    page.evaluate(
      async sampleIdentifiers => {
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
      },
      ['points-time', 'connections', 'trip-observations', 'numeric-grid']
    )
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

/** Absence is optional; page closure, invalid selectors and application errors still fail. */
export async function waitForOptionalSurface(surface, timeoutMilliseconds) {
  try {
    await surface.waitFor({state: 'visible', timeout: timeoutMilliseconds});
    return true;
  } catch (failure) {
    if (failure.name === 'TimeoutError') return false;
    throw failure;
  }
}

/** Follow the host's supported opening path before deciding an AX surface is absent. */
export async function openWorkflowJourneySurface(recorder, surfaceIdentifier) {
  const {page} = recorder;
  const sidePanel = page.getByTestId(TEST_IDS.sidePanel).first();
  const workflowsTab = page.locator('.side-panel__tab[data-for="workflows-nav"]').first();
  if (!(await sidePanel.isVisible()) && (await workflowsTab.isVisible())) {
    await recorder.click(workflowsTab);
    await recorder.observe('Workflows side panel after navigation', () =>
      sidePanel.waitFor({state: 'visible'})
    );
  }
  const dock = page.getByTestId(TEST_IDS.dock).first();
  const toolbar = dock.getByTestId(TEST_IDS.toolbar);
  if (!(await toolbar.isVisible())) {
    const expand = dock.getByTestId(TEST_IDS.dockToggle);
    if (!(await expand.isVisible())) return false;
    await recorder.click(expand);
    await recorder.observe('expanded workbench toolbar', () => toolbar.waitFor({state: 'visible'}));
  }
  if (surfaceIdentifier === 'ax-command-palette') {
    const selectedWorkflow = toolbar.getByTestId('workflow-select');
    if (!(await selectedWorkflow.isEnabled()) || !(await selectedWorkflow.inputValue())) {
      await recorder.click(toolbar.getByTestId(TEST_IDS.workflowNew));
      await recorder.observe('new workflow selection', async () => {
        if (!(await selectedWorkflow.inputValue()))
          throw new Error('New workflow was not selected');
      });
    }
  }
  const opener = toolbar.getByTestId(
    surfaceIdentifier === 'ax-command-palette' ? TEST_IDS.addStep : TEST_IDS.templates
  );
  if (!(await opener.isVisible())) return false;
  await recorder.click(opener);
  return true;
}

/**
 * AX hosts must expose auditable result metadata; text/visibility alone cannot qualify a run.
 * Suggestion buttons carry data-sample-id and data-suggestion-id. The result card carries those
 * plus data-status="done", data-run-id, data-stale="false" and aria-busy="false".
 * Until the optional surface implements this evidence contract it fails qualification.
 */
export async function runSuggestionToResult(recorder, surface, sampleIdentifier) {
  const {page} = recorder;
  const suggestion = surface.locator('button:not(:disabled), a[href]').first();
  const suggestionIdentifier = await suggestion.getAttribute('data-suggestion-id');
  if (
    !suggestionIdentifier ||
    (await suggestion.getAttribute('data-sample-id')) !== sampleIdentifier
  )
    throw new Error(
      `AX-70: selected suggestion must identify sample ${sampleIdentifier} and its suggestion ID`
    );
  const resultCard = page.getByTestId('ax-result-card').first();
  const previousRunIdentifier = (await resultCard.count())
    ? await resultCard.getAttribute('data-run-id')
    : null;
  const expected = {sampleIdentifier, suggestionIdentifier, previousRunIdentifier};
  recorder.observations.suggestionEvidence = expected;
  await recorder.click(suggestion);
  const handle = await recorder.waitUntil(
    'a fresh completed AX suggestion result',
    previousRun => {
      const card = document.querySelector('[data-testid="ax-result-card"]');
      if (!card || !card.getClientRects().length) return false;
      const status = card.getAttribute('data-status');
      const busy = card.getAttribute('aria-busy');
      const hasError =
        Boolean(card.querySelector('[role="alert"], [data-status="error"]')) ||
        card.getAttribute('data-error') === 'true';
      const text = card.textContent ?? '';
      if (
        !hasError &&
        !/^\s*(?:error:|failed:)|GPU computation failed/i.test(text) &&
        (['running', 'loading', 'pending'].includes(status) ||
          busy === 'true' ||
          (status === 'done' && card.getAttribute('data-run-id') === previousRun))
      )
        return false;
      return {
        status,
        busy: busy === 'false' ? false : true,
        stale: card.getAttribute('data-stale') === 'false' ? false : true,
        sampleIdentifier: card.getAttribute('data-sample-id'),
        suggestionIdentifier: card.getAttribute('data-suggestion-id'),
        runIdentifier: card.getAttribute('data-run-id'),
        hasError,
        text
      };
    },
    previousRunIdentifier
  );
  let result;
  try {
    result = await handle.jsonValue();
  } finally {
    await handle.dispose();
  }
  recorder.observations.resultEvidence = result;
  const failure = getSuggestionResultFailure(result, expected);
  if (failure) throw new Error(`AX-70: ${failure}`);
  recorder.observations.resultText = result.text;
  return result;
}

/** AX-22 opening gate, independent of the later one-action-result gate. */
export async function runSuggestionOpening(recorder, surface) {
  const suggestion = surface.locator('button[data-suggestion-id]:not(:disabled)').first();
  if (!(await suggestion.count()))
    throw new Error(
      'AX-22: this loaded profile has no launchable suggestion; inspect the dataset mapping limitation.'
    );
  const expected = {
    bindingId: await suggestion.getAttribute('data-binding-id'),
    inputMappings: JSON.parse(await suggestion.getAttribute('data-input-mappings')),
    fieldBindings: JSON.parse(await suggestion.getAttribute('data-field-bindings'))
  };
  const mappingFields = ['mode', 'datasetId', 'geometry', 'longitude', 'latitude'];
  if (
    !expected.bindingId ||
    !expected.inputMappings ||
    !Object.keys(expected.inputMappings).length ||
    Object.values(expected.inputMappings).some(
      mapping =>
        !mapping ||
        mappingFields.some(field => !Object.hasOwn(mapping, field)) ||
        mapping.mode !== 'native' ||
        !mapping.datasetId ||
        !mapping.geometry ||
        ((mapping.longitude !== null || mapping.latitude !== null) &&
          (!mapping.longitude || !mapping.latitude))
    )
  )
    throw new Error(
      `AX-22 missing complete expected named input mappings: ${JSON.stringify(expected)}`
    );
  recorder.observations.openingExpectations = expected;
  await recorder.click(suggestion);
  const catalog = recorder.page.getByTestId('analysis-catalog');
  await recorder.observe('suggested catalog binding', () => catalog.waitFor({state: 'visible'}));
  await recorder.waitUntil(
    'exact suggested operation',
    bindingId => document.querySelector('#catalog-operation')?.value === bindingId,
    expected.bindingId
  );
  const operation = await catalog.locator('#catalog-operation').inputValue();
  if (operation !== expected.bindingId)
    throw new Error(`AX-22: expected ${expected.bindingId}, got ${operation}`);
  // Expectations were recorded on the suggestion before the catalog opened, from the
  // assigned profiles. Compare named ports, never an unordered multiset of dataset IDs.
  const mappings = await catalog.locator('[data-catalog-input]').evaluateAll(fieldsets =>
    Object.fromEntries(
      fieldsets.map(fieldset => [
        fieldset.getAttribute('data-catalog-input'),
        {
          mode: fieldset.querySelector('select[aria-label$=" source"]')?.value ?? null,
          datasetId: fieldset.querySelector('select[aria-label$=" dataset"]')?.value ?? null,
          geometry: fieldset.querySelector('select[aria-label$=" geometry field"]')?.value ?? null,
          longitude:
            fieldset.querySelector('select[aria-label$=" X / longitude column"]')?.value ?? null,
          latitude:
            fieldset.querySelector('select[aria-label$=" Y / latitude column"]')?.value ?? null
        }
      ])
    )
  );
  if (
    !expected.inputMappings ||
    Object.keys(mappings).length !== Object.keys(expected.inputMappings).length
  )
    throw new Error(`AX-22 named input mismatch: ${JSON.stringify({expected, mappings})}`);
  for (const [inputName, mapping] of Object.entries(expected.inputMappings)) {
    const actual = mappings[inputName];
    if (!actual || Object.entries(mapping).some(([name, value]) => actual[name] !== value))
      throw new Error(
        `AX-22 input ${inputName} mapping mismatch: ${JSON.stringify({expected: mapping, actual})}`
      );
  }
  for (const [parameter, columns] of Object.entries(expected.fieldBindings)) {
    const actual = await readAnalysisSheetField(catalog, parameter);
    const value = Array.isArray(columns) ? columns.join(',') : columns;
    if (actual !== value) throw new Error(`AX-22 ${parameter}: expected ${value}, got ${actual}`);
  }
  recorder.observations.openingEvidence = {expected, mappings};
}

async function journeySuggestionOpening(page, options, sampleIdentifier, screenshotFiles) {
  const recorder = new JourneyRecorder(
    `suggestion-open:${sampleIdentifier}`,
    `Open a suggestion for ${sampleIdentifier}`,
    page
  );
  await startApplication(page, options);
  const {title} = await loadSampleThroughGallery(recorder, sampleIdentifier);
  // Selecting the active dataset is setup; the gate starts at its loaded card.
  await chooseAnalysisDataset(recorder, title);
  const surface = page
    .getByTestId('ax-suggestions')
    .filter({has: page.getByRole('heading', {name: 'Suggested for this data', exact: true})})
    .first();
  await recorder.observe('loaded dataset suggestions', () => surface.waitFor({state: 'visible'}));
  recorder.markSetupFinished();
  await runSuggestionOpening(recorder, surface);
  recorder.markFinished();
  await saveScreenshot(
    page,
    options.out,
    `suggestion-open_${sampleIdentifier}.jpg`,
    screenshotFiles
  );
  return recorder.toRecord({
    outcome: 'observed',
    endCondition:
      'One click opened the exact binding with native datasets and suggested roles; no result gate claimed.'
  });
}

/** The existing optional-surface hook also supports the AX-22 filtered browse gate. */
async function journeySuggestionBrowse(page, options, screenshotFiles) {
  const recorder = new JourneyRecorder(
    'suggestion-browse:points-time',
    'Browse analyses for the active dataset',
    page
  );
  await startApplication(page, options);
  const {title} = await loadSampleThroughGallery(recorder, 'points-time');
  await chooseAnalysisDataset(recorder, title);
  const surface = page.getByTestId('ax-suggestions').first();
  await recorder.observe('dataset suggestion card', () => surface.waitFor({state: 'visible'}));
  recorder.markSetupFinished();
  const browse = surface
    .locator('details')
    .filter({has: page.getByText('Browse all that fit', {exact: true})});
  await recorder.click(browse.locator('summary'));
  const buttons = browse.locator('button[data-binding-id]');
  await recorder.observe('compatible browse operations', () =>
    buttons.first().waitFor({state: 'visible'})
  );
  const datasetId = await surface.getAttribute('data-dataset-id');
  const candidates = await buttons.evaluateAll(elements =>
    elements.map(element => ({
      bindingId: element.getAttribute('data-binding-id'),
      assignments: JSON.parse(element.getAttribute('data-input-assignments')),
      disabled: element.disabled
    }))
  );
  if (
    !candidates.length ||
    candidates.some(
      candidate => candidate.disabled || !Object.values(candidate.assignments).includes(datasetId)
    )
  )
    throw new Error(
      `AX-22 browse does not fit the active dataset: ${JSON.stringify({datasetId, candidates})}`
    );
  recorder.observations.browseEvidence = {datasetId, candidates};
  recorder.markFinished();
  await saveScreenshot(page, options.out, 'suggestion-browse_points-time.jpg', screenshotFiles);
  return recorder.toRecord({
    outcome: 'observed',
    endCondition:
      'Browse all shows enabled operations assigned to the active dataset; Node specs check their applicability.'
  });
}

/** Optional AX journeys become active as their host surfaces land. */
async function journeyOptionalSurface(
  page,
  options,
  surfaceIdentifier,
  sampleIdentifier,
  screenshotFiles
) {
  const identifier =
    surfaceIdentifier === 'ax-suggestions'
      ? `suggestion-result:${sampleIdentifier}`
      : `surface:${surfaceIdentifier}`;
  const recorder = new JourneyRecorder(
    identifier,
    `${surfaceIdentifier} on ${sampleIdentifier}`,
    page
  );
  await startApplication(page, options);
  const {title} = await loadSampleThroughGallery(recorder, sampleIdentifier);
  recorder.observations.sampleTitle = title;
  recorder.markSetupFinished();
  const surface = page.getByTestId(surfaceIdentifier).first();
  const absent = () => {
    recorder.markFinished();
    return recorder.toRecord({
      outcome: 'surface-not-present',
      reason: `surface not present: [data-testid="${surfaceIdentifier}"]`,
      endCondition: 'Optional AX surface unavailable; no qualification claimed.'
    });
  };
  if (!(await surface.isVisible())) {
    recorder.phase = 'setup';
    if (['ax-command-palette', 'ax-workflow-gallery'].includes(surfaceIdentifier)) {
      await openWorkflowJourneySurface(recorder, surfaceIdentifier);
    } else {
      const opener = page.getByTestId(`${surfaceIdentifier}-open`).first();
      if (await opener.isVisible()) await recorder.click(opener);
    }
    recorder.markSetupFinished();
  }
  if (
    ['ax-analysis-sheet', 'ax-result-card'].includes(surfaceIdentifier) &&
    !(await surface.isVisible())
  ) {
    const suggestion = page
      .getByTestId('ax-suggestions')
      .locator('button:not(:disabled), a[href]')
      .first();
    if (await suggestion.isVisible()) {
      recorder.phase = 'setup';
      await recorder.click(suggestion);
      recorder.markSetupFinished();
    }
  }
  // Allow React/profile publication to settle; only an optional surface timeout is a skip.
  if (!(await waitForOptionalSurface(surface, Math.min(options.timeoutMilliseconds, 3000))))
    return absent();
  recorder.observations.surfaceText = (await surface.innerText()).slice(0, 2000);
  if (surfaceIdentifier === 'ax-suggestions') {
    // Preserve selection actions in the loaded-sample budget when the legacy host needs one.
    if (await page.locator('#analysis-dataset').count())
      await chooseAnalysisDataset(recorder, title);
    await runSuggestionToResult(recorder, surface, sampleIdentifier);
  } else if (surfaceIdentifier === 'ax-command-palette') {
    const search = surface.locator('input').first();
    // Focus must already be on the search control, as the AX-60 keyboard gate requires.
    if (!(await search.evaluate(element => document.activeElement === element)))
      throw new JourneyFailure(
        identifier,
        'palette search focus',
        new Error('Search input is not focused')
      );
    for (const key of ['h', 'o', 't', 'Enter']) await recorder.press(search, key);
    await recorder.observe('mapped analysis sheet after hot + Enter', () =>
      page.getByTestId('ax-analysis-sheet').first().waitFor({state: 'visible'})
    );
    recorder.observations.analysisSheetText = await page
      .getByTestId('ax-analysis-sheet')
      .first()
      .innerText();
  }
  recorder.markFinished();
  await saveScreenshot(
    page,
    options.out,
    `${identifier.replace(/[^\w-]/g, '_')}.jpg`,
    screenshotFiles
  );
  return recorder.toRecord({
    outcome: surfaceIdentifier === 'ax-suggestions' ? 'result-visible' : 'observed',
    endCondition:
      surfaceIdentifier === 'ax-suggestions'
        ? 'Selecting the sample suggestion completes a fresh successful result with matching sample and suggestion IDs.'
        : surfaceIdentifier === 'ax-command-palette'
          ? 'Focused palette receives h, o, t, Enter and displays the analysis sheet.'
          : 'The AX surface is visible; text and screenshot recorded (presence probe only).'
  });
}

const SOURCE_EXCLUDED_DIRECTORIES = new Set([
  '.git',
  'node_modules',
  '.ax',
  '.typecheck',
  '.cache',
  '.yarn',
  'dist',
  'coverage',
  '.playwright-artifacts',
  '.docusaurus',
  '.tanstack',
  '.output',
  '.reports',
  'workflow-screenshots',
  'kepler-demo-cache',
  '__pycache__'
]);
const hashBytes = bytes => createHash('sha256').update(bytes).digest('hex');
const hashObject = (kind, bytes) =>
  createHash('sha1').update(`${kind} ${bytes.length}\0`).update(bytes).digest('hex');

/** Minimal read-only object access for loose and packed SHA-1 repositories (no git subprocess). */
function createRepositoryObjectReader(repositoryDirectory) {
  const objectDirectory = path.join(repositoryDirectory, '.git/objects');
  const objectCache = new Map();
  const packCache = new Map();
  const packDirectory = path.join(objectDirectory, 'pack');
  const indexes = existsSync(packDirectory)
    ? readdirSync(packDirectory)
        .filter(name => name.endsWith('.idx'))
        .map(name => ({name, bytes: readFileSync(path.join(packDirectory, name))}))
    : [];
  const readVariableInteger = (bytes, cursor) => {
    let value = 0;
    let multiplier = 1;
    let byte;
    do {
      byte = bytes[cursor.offset++];
      value += (byte & 0x7f) * multiplier;
      multiplier *= 128;
    } while (byte & 0x80);
    return value;
  };
  const applyDelta = (base, delta) => {
    const cursor = {offset: 0};
    const baseSize = readVariableInteger(delta, cursor);
    const resultSize = readVariableInteger(delta, cursor);
    if (baseSize !== base.bytes.length) throw new Error('AX-70: packed delta base size mismatch');
    const result = Buffer.alloc(resultSize);
    let resultOffset = 0;
    while (cursor.offset < delta.length) {
      const instruction = delta[cursor.offset++];
      if (instruction & 0x80) {
        let copyOffset = 0;
        let copySize = 0;
        for (let index = 0; index < 4; index++) {
          if (instruction & (1 << index)) copyOffset += delta[cursor.offset++] * 2 ** (index * 8);
        }
        for (let index = 0; index < 3; index++) {
          if (instruction & (1 << (index + 4)))
            copySize += delta[cursor.offset++] * 2 ** (index * 8);
        }
        copySize ||= 0x10000;
        if (copyOffset + copySize > base.bytes.length || resultOffset + copySize > result.length)
          throw new Error('AX-70: packed delta copy exceeds object size');
        base.bytes.copy(result, resultOffset, copyOffset, copyOffset + copySize);
        resultOffset += copySize;
      } else {
        if (
          !instruction ||
          cursor.offset + instruction > delta.length ||
          resultOffset + instruction > result.length
        )
          throw new Error('AX-70: invalid packed delta insertion');
        delta.copy(result, resultOffset, cursor.offset, cursor.offset + instruction);
        cursor.offset += instruction;
        resultOffset += instruction;
      }
    }
    if (resultOffset !== resultSize) throw new Error('AX-70: packed delta result size mismatch');
    return {kind: base.kind, bytes: result};
  };
  const readPackedObject = (packName, offset) => {
    const cacheKey = `${packName}:${offset}`;
    if (objectCache.has(cacheKey)) return objectCache.get(cacheKey);
    if (!packCache.has(packName))
      packCache.set(packName, readFileSync(path.join(packDirectory, packName)));
    const pack = packCache.get(packName);
    let cursor = offset;
    let byte = pack[cursor++];
    const type = (byte >> 4) & 7;
    let size = byte & 15;
    let multiplier = 16;
    while (byte & 0x80) {
      byte = pack[cursor++];
      size += (byte & 127) * multiplier;
      multiplier *= 128;
    }
    let base;
    if (type === 6) {
      byte = pack[cursor++];
      let distance = byte & 127;
      while (byte & 128) {
        byte = pack[cursor++];
        distance = (distance + 1) * 128 + (byte & 127);
      }
      base = readPackedObject(packName, offset - distance);
    } else if (type === 7) {
      base = readObject(pack.subarray(cursor, cursor + 20).toString('hex'));
      cursor += 20;
    }
    const bytes = inflateSync(pack.subarray(cursor));
    if (bytes.length !== size) throw new Error('AX-70: packed object size mismatch');
    const object = base
      ? applyDelta(base, bytes)
      : {kind: {1: 'commit', 2: 'tree', 3: 'blob', 4: 'tag'}[type], bytes};
    if (!object.kind) throw new Error('AX-70: unsupported packed object type');
    objectCache.set(cacheKey, object);
    return object;
  };
  const readObject = identifier => {
    if (objectCache.has(identifier)) return objectCache.get(identifier);
    const looseFile = path.join(objectDirectory, identifier.slice(0, 2), identifier.slice(2));
    let object;
    if (existsSync(looseFile)) {
      const content = inflateSync(readFileSync(looseFile));
      object = {
        kind: content.subarray(0, content.indexOf(32)).toString(),
        bytes: content.subarray(content.indexOf(0) + 1)
      };
    } else {
      const requested = Buffer.from(identifier, 'hex');
      for (const {name, bytes} of indexes) {
        if (bytes.readUInt32BE(0) !== 0xff744f63 || bytes.readUInt32BE(4) !== 2)
          throw new Error('AX-70: repository pack index must use version 2');
        const count = bytes.readUInt32BE(8 + 255 * 4);
        const namesOffset = 8 + 256 * 4;
        let lower = 0;
        let upper = count - 1;
        while (lower <= upper) {
          const middle = Math.floor((lower + upper) / 2);
          const comparison = Buffer.compare(
            bytes.subarray(namesOffset + middle * 20, namesOffset + (middle + 1) * 20),
            requested
          );
          if (comparison < 0) lower = middle + 1;
          else if (comparison > 0) upper = middle - 1;
          else {
            const offsetsStart = namesOffset + count * 24;
            const shortOffset = bytes.readUInt32BE(offsetsStart + middle * 4);
            const offset =
              shortOffset & 0x80000000
                ? Number(
                    bytes.readBigUInt64BE(offsetsStart + count * 4 + (shortOffset & 0x7fffffff) * 8)
                  )
                : shortOffset;
            object = readPackedObject(name.replace(/\.idx$/, '.pack'), offset);
            break;
          }
        }
        if (object) break;
      }
    }
    if (!object || hashObject(object.kind, object.bytes) !== identifier)
      throw new Error(`AX-70: repository object missing or corrupt: ${identifier}`);
    objectCache.set(identifier, object);
    return object;
  };
  return identifier => readObject(identifier).bytes;
}

/** Read-only repository identity; never invokes git or changes its metadata. */
export function readSourceIdentity(outputDirectory, repositoryDirectory = REPOSITORY_DIRECTORY) {
  const commit = readCommit(repositoryDirectory);
  const files = [];
  const excludedOutput = path.relative(repositoryDirectory, outputDirectory);
  const visit = relativeDirectory => {
    for (const entry of readdirSync(path.join(repositoryDirectory, relativeDirectory), {
      withFileTypes: true
    })) {
      const relativeName = path.join(relativeDirectory, entry.name);
      if (
        relativeName === excludedOutput ||
        relativeName === 'dev-docs/research/analysis-workflow-ux-2026-11-14/releases'
      )
        continue;
      if (entry.isDirectory()) {
        if (!SOURCE_EXCLUDED_DIRECTORIES.has(entry.name)) visit(relativeName);
      } else if (!entry.name.endsWith('.tsbuildinfo') && entry.name !== '.DS_Store') {
        const fullName = path.join(repositoryDirectory, relativeName);
        const bytes = entry.isSymbolicLink()
          ? Buffer.from(readlinkSync(fullName))
          : readFileSync(fullName);
        const mode = entry.isSymbolicLink()
          ? '120000'
          : lstatSync(fullName).mode & 0o111
            ? '100755'
            : '100644';
        files.push({
          path: relativeName,
          mode,
          sha256: hashBytes(bytes),
          blob: hashObject('blob', bytes)
        });
      }
    }
  };
  visit('');
  files.sort((first, second) => (first.path < second.path ? -1 : first.path > second.path ? 1 : 0));
  const identity = {
    commit,
    algorithm: 'sha256',
    treeHash: hashBytes(JSON.stringify(files)),
    files,
    excludedDirectories: [...SOURCE_EXCLUDED_DIRECTORIES],
    excludedOutput: [excludedOutput, 'dev-docs/research/analysis-workflow-ux-2026-11-14/releases'],
    headTree: null,
    dirty: null,
    changedPaths: [],
    untrackedPaths: [],
    dirtyError: null,
    indexHash: existsSync(path.join(repositoryDirectory, '.git/index'))
      ? hashBytes(readFileSync(path.join(repositoryDirectory, '.git/index')))
      : null
  };
  try {
    const readObject = createRepositoryObjectReader(repositoryDirectory);
    identity.headTree = /^tree ([a-f0-9]{40})/m.exec(readObject(commit.head).toString())?.[1];
    if (!identity.headTree) throw new Error('HEAD tree unavailable');
    const trackedFiles = new Map();
    const readTree = (identifier, prefix = '') => {
      const bytes = readObject(identifier);
      let offset = 0;
      while (offset < bytes.length) {
        const separator = bytes.indexOf(0, offset);
        const entry = bytes.subarray(offset, separator).toString();
        const mode = entry.slice(0, entry.indexOf(' '));
        const name = entry.slice(entry.indexOf(' ') + 1);
        const objectIdentifier = bytes.subarray(separator + 1, separator + 21).toString('hex');
        const relativeName = `${prefix}${name}`;
        if (mode === '40000') readTree(objectIdentifier, `${relativeName}/`);
        else trackedFiles.set(relativeName, {mode, blob: objectIdentifier});
        offset = separator + 21;
      }
    };
    readTree(identity.headTree);
    const currentFiles = new Map(files.map(file => [file.path, file]));
    // Include every tracked path, even those in otherwise excluded generated directories.
    for (const [relativeName, expected] of trackedFiles) {
      let current = currentFiles.get(relativeName);
      if (!current && existsSync(path.join(repositoryDirectory, relativeName))) {
        const fullName = path.join(repositoryDirectory, relativeName);
        const symbolicLink = lstatSync(fullName).isSymbolicLink();
        const bytes = symbolicLink ? Buffer.from(readlinkSync(fullName)) : readFileSync(fullName);
        current = {
          path: relativeName,
          mode: symbolicLink ? '120000' : lstatSync(fullName).mode & 0o111 ? '100755' : '100644',
          sha256: hashBytes(bytes),
          blob: hashObject('blob', bytes)
        };
        files.push(current);
      }
      if (!current || current.blob !== expected.blob || current.mode !== expected.mode)
        identity.changedPaths.push(relativeName);
    }
    identity.untrackedPaths = [...currentFiles.keys()].filter(name => !trackedFiles.has(name));
    identity.dirty = identity.changedPaths.length > 0 || identity.untrackedPaths.length > 0;
    files.sort((first, second) =>
      first.path < second.path ? -1 : first.path > second.path ? 1 : 0
    );
    identity.treeHash = hashBytes(JSON.stringify(files));
  } catch (failure) {
    // Unsupported repositories still get a source fingerprint, never a false clean claim.
    identity.dirtyError = `Cannot compare HEAD objects: ${String(failure)}`;
  }
  return identity;
}

async function readFixtureHashes(page) {
  const fixtures = await page.evaluate(async identifiers => {
    const {createSandboxSample} = await import('/src/sandbox-samples.ts');
    return identifiers.map(sampleIdentifier => {
      const {table} = createSandboxSample(sampleIdentifier);
      return {
        id: sampleIdentifier,
        rows: table.numRows,
        serialized: JSON.stringify(
          {
            fields: table.schema.fields.map(field => ({
              name: field.name,
              type: String(field.type),
              nullable: field.nullable
            })),
            rows: table.toArray().map(row => row.toJSON())
          },
          (_key, value) =>
            typeof value === 'bigint'
              ? {bigint: String(value)}
              : typeof value === 'number' && !Number.isFinite(value)
                ? {number: String(value)}
                : value
        )
      };
    });
  }, GALLERY_SAMPLE_IDENTIFIERS);
  return {
    algorithm: 'sha256',
    encoding:
      'JSON of Arrow field names/types/nullability and all rows; BigInts and nonfinite numbers tagged',
    generator: {
      path: 'examples/deck/kepler-compute/src/sandbox-samples.ts',
      sha256: hashBytes(readFileSync(path.join(SCRIPT_DIRECTORY, '../src/sandbox-samples.ts')))
    },
    samples: fixtures.map(({id: sampleIdentifier, rows, serialized}) => ({
      id: sampleIdentifier,
      rows,
      sha256: hashBytes(serialized)
    }))
  };
}

// ---------------------------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------------------------

function readCommit(repositoryDirectory = REPOSITORY_DIRECTORY) {
  try {
    const headText = readFileSync(path.join(repositoryDirectory, '.git/HEAD'), 'utf8').trim();
    if (!headText.startsWith('ref: ')) return {head: headText, reference: null};
    const reference = headText.slice(5);
    const referenceFile = path.join(repositoryDirectory, '.git', reference);
    if (existsSync(referenceFile))
      return {head: readFileSync(referenceFile, 'utf8').trim(), reference};
    const packed = readFileSync(path.join(repositoryDirectory, '.git/packed-refs'), 'utf8')
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
  const targets = validateJourneyTargets(JSON.parse(readFileSync(options.targets, 'utf8')));
  waitTimeoutMilliseconds = options.timeoutMilliseconds;
  const records = [];
  const failures = [];
  const screenshotFiles = [];
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
    },
    {
      id: 'suggestion-browse:points-time',
      run: page => journeySuggestionBrowse(page, options, screenshotFiles)
    },
    ...GALLERY_SAMPLE_IDENTIFIERS.map(sampleIdentifier => ({
      id: `suggestion-open:${sampleIdentifier}`,
      run: page => journeySuggestionOpening(page, options, sampleIdentifier, screenshotFiles)
    })),
    ...GALLERY_SAMPLE_IDENTIFIERS.map(sampleIdentifier => ({
      id: `suggestion-result:${sampleIdentifier}`,
      run: page =>
        journeyOptionalSurface(page, options, 'ax-suggestions', sampleIdentifier, screenshotFiles)
    })),
    ...AX_SURFACE_IDENTIFIERS.filter(identifier => identifier !== 'ax-suggestions').map(
      identifier => ({
        id: `surface:${identifier}`,
        run: page =>
          journeyOptionalSurface(
            page,
            options,
            identifier,
            GALLERY_DEFAULT_SAMPLE_IDENTIFIER,
            screenshotFiles
          )
      })
    )
  ].filter(
    journey => !options.only || [...options.only].some(prefix => journey.id.startsWith(prefix))
  );
  if (journeys.length === 0) throw new Error('--only matched no journey');

  const unknownTargets = targets.journeys.filter(
    target =>
      ![
        ...GALLERY_SAMPLE_IDENTIFIERS.map(identifier => `suggestion-result:${identifier}`),
        ...AX_SURFACE_IDENTIFIERS.map(identifier => `surface:${identifier}`)
      ].includes(target.id)
  );
  if (unknownTargets.length)
    throw new Error(
      `Unknown target journey: ${unknownTargets.map(target => target.id).join(', ')}`
    );
  const startedAt = new Date();
  const sourceIdentity = readSourceIdentity(options.out);
  mkdirSync(options.out, {recursive: true});
  const launchOptions = getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'});
  const manifest = {
    schemaVersion: 2,
    date: startedAt.toISOString(),
    release: options.release,
    status: 'running',
    commit: sourceIdentity.commit,
    sourceIdentity,
    identityNote:
      'Source files and HEAD objects read without git. treeHash identifies the measured source; dirty compares its contents to HEAD. indexHash separately identifies staging metadata. Unknown HEAD comparisons remain unqualified.',
    browser: {name: 'chromium', version: null, launchOptions},
    webgpuAdapter: null,
    machine: {
      model: readSysctl('hw.model'),
      memoryBytes: Number(readSysctl('hw.memsize')) || os.totalmem(),
      cpu: readSysctl('machdep.cpu.brand_string') ?? os.cpus()[0]?.model,
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
    targets: {
      path: path.relative(REPOSITORY_DIRECTORY, options.targets),
      sha256: hashBytes(readFileSync(options.targets)),
      policy: targets
    },
    latencyQualification:
      'not-run: requires the designated reference machine; recorded durations are observations',
    fixtures: null,
    journeyIds: [],
    failedJourneys: failures,
    screenshots: screenshotFiles
  };
  const persist = () => {
    manifest.journeyIds = records.map(record => record.id);
    manifest.gateResults = evaluateJourneyTargets(records, targets);
    manifest.comparisons = compareJourneys(
      records,
      JSON.parse(readFileSync(path.join(HISTORICAL_BASELINE_DIRECTORY, 'journeys.json'), 'utf8'))
    );
    writeFileSync(path.join(options.out, 'journeys.json'), `${JSON.stringify(records, null, 2)}\n`);
    writeFileSync(
      path.join(options.out, 'manifest.json'),
      `${JSON.stringify(manifest, null, 2)}\n`
    );
  };
  // Write identity before launch, fixture/adapter manifest before any timed journey.
  persist();
  let browser;
  try {
    browser = await chromium.launch(launchOptions);
    manifest.browser.version = browser.version();
    const preparationPage = await browser.newPage({viewport: VIEWPORT});
    try {
      await pinViteModules(preparationPage);
      await startApplication(preparationPage, options);
      manifest.webgpuAdapter = await readAdapterInformation(preparationPage);
      if (manifest.webgpuAdapter.error) throw new Error(manifest.webgpuAdapter.error);
      manifest.fixtures = await readFixtureHashes(preparationPage);
      manifest.referenceManifestCollectedAt = new Date().toISOString();
      persist();
    } finally {
      await preparationPage.close();
    }
    for (const journey of journeys) {
      const page = await browser.newPage({viewport: VIEWPORT});
      page.setDefaultTimeout(options.timeoutMilliseconds);
      try {
        await pinViteModules(page);
        const result = await journey.run(page);
        for (const record of Array.isArray(result) ? result : [result]) records.push(record);
        process.stdout.write(`${records.at(-1).outcome} ${journey.id}\n`);
      } catch (failure) {
        const message = failure instanceof Error ? failure.message : String(failure);
        failures.push({id: journey.id, message});
        page.axJourneyRecorder?.markFinished();
        records.push(
          page.axJourneyRecorder?.toRecord({outcome: 'failed', error: message}) ?? {
            id: journey.id,
            outcome: 'failed',
            error: message
          }
        );
        process.stderr.write(`FAILED ${message}\n`);
        await saveScreenshot(
          page,
          options.out,
          `failed-${journey.id.replace(/[^\w-]/g, '_')}.jpg`,
          screenshotFiles
        ).catch(() => undefined);
      } finally {
        await page.close();
        persist();
      }
    }
  } catch (failure) {
    const message = failure instanceof Error ? failure.message : String(failure);
    failures.push({id: 'run:environment', message});
    records.push({id: 'run:environment', outcome: 'failed', error: message});
    process.stderr.write(`FAILED ${message}\n`);
  } finally {
    try {
      await browser?.close();
    } catch (failure) {
      const message = `Browser close failed: ${String(failure)}`;
      failures.push({id: 'run:cleanup', message});
      records.push({id: 'run:cleanup', outcome: 'failed', error: message});
    }
    const finalSourceIdentity = readSourceIdentity(options.out);
    manifest.sourceStable =
      sourceIdentity.treeHash === finalSourceIdentity.treeHash &&
      sourceIdentity.indexHash === finalSourceIdentity.indexHash &&
      sourceIdentity.commit.head === finalSourceIdentity.commit.head;
    manifest.finalTreeHash = finalSourceIdentity.treeHash;
    manifest.status = 'finished';
    manifest.finishedAt = new Date().toISOString();
    persist();
    const identityGates = [
      {id: 'source-stable', required: true, status: manifest.sourceStable ? 'passed' : 'failed'},
      {
        id: 'dirty-identity',
        required: true,
        status: sourceIdentity.dirty === null ? 'not-run' : 'passed'
      }
    ];
    manifest.gateResults.gates.push(...identityGates);
    manifest.gateResults.qualified &&= identityGates.every(gate => gate.status === 'passed');
    if (identityGates.some(gate => gate.status === 'failed'))
      manifest.gateResults.status = 'failed';
    else if (!manifest.gateResults.qualified && manifest.gateResults.status === 'passed')
      manifest.gateResults.status = 'incomplete';
    writeFileSync(
      path.join(options.out, 'manifest.json'),
      `${JSON.stringify(manifest, null, 2)}\n`
    );
  }
  if (manifest.gateResults.status === 'failed') {
    process.stderr.write('AX-70 gates failed; see manifest.json and journeys.json\n');
    process.exitCode = 1;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  await main();
