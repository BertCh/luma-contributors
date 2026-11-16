// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// External driver only: sandbox Chromium is unavailable to the implementor.
// node scripts/ax-analysis-sheet-qualification.mjs <url> <evidence-directory>
// Writes every wave-one outcome before failing an incomplete AX-40 latency gate.
import {mkdir, writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {runAcceptance, screenshot} from './workflow-utils.mjs';
import {setAnalysisSheetParameter} from './analysis-sheet-controls.mjs';

await runAcceptance('ax-analysis-sheet-qualification', async (page, directory) => {
  await mkdir(directory, {recursive: true});
  const catalog = page.getByTestId('analysis-catalog');
  await catalog.waitFor();
  // Resident methods each own a sheet inside their disclosure; qualify the catalog sheet here.
  const sheet = catalog.getByTestId('ax-analysis-sheet');
  await sheet.waitFor();
  for (const section of ['Data', 'Key settings', 'Checks', 'Preview scope']) {
    if ((await sheet.getByRole('region', {name: section, exact: true}).count()) !== 1)
      throw new Error(`Analysis sheet section missing: ${section}`);
  }
  await catalog.locator('#catalog-operation').selectOption('local-moran-statistics');
  if (
    !/row-standardized.*binary.*analytic/s.test(
      await sheet.getByTestId('ax-method-difference').innerText()
    )
  )
    throw new Error('Local Moran method distinction is missing');
  await catalog.locator('#catalog-operation').selectOption('kriging-grid');
  if (
    !/without curve fitting or cross-validation/.test(
      await sheet.getByTestId('ax-method-difference').innerText()
    )
  )
    throw new Error('Kriging method distinction is missing');
  if ((await catalog.getByTestId('catalog-lens').count()) !== 1)
    throw new Error('Lens access was lost');
  if (
    (await sheet.getByTestId('ax-preview-scope').getAttribute('data-scope')) !== 'full-population'
  )
    throw new Error('Sheet must not enable sampled significance previews');

  // Regression gates use the retained point-buffer path independently of wave-one availability.
  await catalog.locator('#catalog-operation').selectOption('geometry-point-buffer');
  await catalog.getByTestId('catalog-run').click();
  await page.waitForFunction(
    () =>
      document.querySelector('[data-testid="catalog-result"][data-current="true"]') &&
      !document.querySelector('[data-testid="catalog-run"]').disabled &&
      /Engine: GPU/.test(document.querySelector('[data-testid="catalog-engine"]').textContent)
  );
  const acceptedParameters = JSON.parse(
    await catalog.getByTestId('catalog-result').getAttribute('data-parameters')
  );
  const radius = sheet.getByTestId('catalog-parameter-radius');
  await radius.fill('-1');
  await page.waitForFunction(
    () =>
      document
        .querySelector('[data-testid="catalog-parameter-radius"]')
        .getAttribute('aria-invalid') === 'true' &&
      document.querySelector('[data-testid="catalog-run"]').disabled &&
      document.querySelector('[data-testid="catalog-publish"]').disabled &&
      document.querySelector('[data-testid="catalog-result"]').getAttribute('data-current') ===
        'false' &&
      document.querySelector('[data-testid="catalog-preflight-status"]').textContent === 'blocked'
  );
  await radius.fill(String(acceptedParameters.radius));
  await radius.press('Tab');
  await page.waitForFunction(() =>
    document.querySelector('[data-testid="catalog-result"][data-current="true"]')
  );
  await catalog.getByTestId('ax-preview-cancel').click();
  await page.waitForFunction(
    () =>
      document.querySelector('[data-testid="catalog-result"][data-current="true"]') &&
      !document.querySelector('[data-testid="catalog-publish"]').disabled
  );
  const keptParameters = JSON.parse(
    await catalog.getByTestId('catalog-result').getAttribute('data-parameters')
  );
  if (JSON.stringify(keptParameters) !== JSON.stringify(acceptedParameters))
    throw new Error(
      'Correcting invalid text or cancelling a completed preview changed the accepted parameters'
    );
  const interactionChecks = {
    invalidTextBlocksRunAndStage: true,
    correctedTextRestoresAcceptedResult: true,
    completedCancelPreservesStage: true
  };
  await screenshot(page, directory, 'ax-analysis-sheet-validity-and-cancel');

  const inventory = await page.evaluate(async () => {
    const {getAnalysisSheetQualificationCases} = await import('/src/analysis-sheet-model.ts');
    return getAnalysisSheetQualificationCases();
  });
  const measurements = [];
  for (const candidate of inventory) {
    const measurement = {
      templateId: candidate.templateId,
      sampleId: candidate.sampleId,
      bindingId: candidate.bindingId,
      engine: candidate.engine,
      status: 'unavailable',
      previewLatencyMilliseconds: null,
      mapLatencyMilliseconds: null,
      reason: candidate.unavailable
    };
    measurements.push(measurement);
    if (candidate.unavailable) continue;
    try {
      await catalog.locator('#catalog-operation').selectOption(candidate.bindingId);
      for (const input of candidate.inputLabels) {
        await sheet.getByLabel(`${input.label} source`, {exact: true}).selectOption('text');
        await sheet
          .getByLabel(`${input.label} GeoJSON`, {exact: true})
          .fill(candidate.inputs[input.name]);
      }
      for (const [name, value] of Object.entries(candidate.parameters)) {
        await setAnalysisSheetParameter(sheet, name, value, {optional: true});
      }
      await catalog.getByTestId('catalog-run').click();
      await page.waitForFunction(() =>
        document.querySelector('[data-testid="catalog-result"][data-current="true"]')
      );
      const parameter = candidate.live.find(
        specification =>
          specification.kind === 'number' ||
          specification.kind === 'integer' ||
          specification.kind === 'enum' ||
          specification.kind === 'boolean'
      );
      if (!parameter) {
        measurement.reason = 'No supported live control to exercise';
        continue;
      }
      const result = catalog.getByTestId('catalog-result');
      const before = JSON.parse(await result.getAttribute('data-parameters'));
      const current = before[parameter.name];
      const next =
        parameter.kind === 'enum'
          ? parameter.options.find(value => value !== current)
          : parameter.kind === 'boolean'
            ? !current
            : Number(current) + 1 <= (parameter.maximum ?? Infinity)
              ? Number(current) + 1
              : Number(current) - 1;
      const started = await page.evaluate(() => performance.now());
      await setAnalysisSheetParameter(sheet, parameter.name, next);
      await page.waitForFunction(
        ({name, value}) => {
          const output = document.querySelector(
            '[data-testid="catalog-result"][data-current="true"]'
          );
          return output && JSON.parse(output.getAttribute('data-parameters'))[name] === value;
        },
        {name: parameter.name, value: next}
      );
      const completed = await page.evaluate(
        () => new Promise(resolve => requestAnimationFrame(() => resolve(performance.now())))
      );
      measurement.previewLatencyMilliseconds = completed - started;
      measurement.parameter = parameter.name;
      measurement.status = 'preview-measured';
      measurement.previewWithinThreshold = measurement.previewLatencyMilliseconds <= 500;
      // Stage in Sources is currently explicit. A changed table is not evidence of a changed map.
      measurement.reason =
        'Retained table preview measured; automatic map replacement is unavailable';
    } catch (failure) {
      measurement.status = 'failed';
      measurement.reason = String(failure);
    }
  }
  const report = {
    gate: 'AX-40',
    thresholdMilliseconds: 500,
    population: 'full-selected-inputs',
    qualified: false,
    interactionChecks,
    measurements
  };
  await writeFile(
    join(directory, 'ax-analysis-sheet-latency.json'),
    `${JSON.stringify(report, null, 2)}\n`
  );
  await screenshot(page, directory, 'ax-analysis-sheet');
  throw new Error(`AX-40 map-latency gate remains unqualified: ${JSON.stringify(measurements)}`);
});
