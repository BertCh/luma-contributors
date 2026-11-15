// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const TRAJECTORY_OPERATIONS = [
  ['trajectory-hampel', 'GPUTrajectoryHampel'],
  ['trajectory-outlier-flags', 'GPUTrajectoryOutlierFlags'],
  ['trajectory-split', null],
  ['trajectory-thinning', null],
  ['trajectory-track-relative-times', 'GPUTrajectoryTrackRelativeTimes'],
  ['trajectory-staypoints', null],
  ['trajectory-segments', null],
  ['trajectory-locations', null],
  ['trajectory-recursions', null],
  ['trajectory-mobility-indicators', null],
  ['trajectory-modal-split', null],
  ['trajectory-brownian-bridge', null],
  ['trajectory-isopleths', 'GPUUtilizationVolumeIsopleths']
];

export async function assertTrajectoryCatalogDiscovery(panel) {
  const operations = await panel
    .getByLabel('Operation', {exact: true})
    .locator('option')
    .evaluateAll(options => options.map(option => option.value));
  for (const [id] of TRAJECTORY_OPERATIONS)
    assert(operations.includes(id), `${id} is listed exactly once`);
  assert.equal(new Set(operations).size, operations.length);
  if (await panel.getByTestId('catalog-blocked').isVisible()) {
    const blocked = await panel.getByTestId('catalog-blocked').innerText();
    for (const [, contributor] of TRAJECTORY_OPERATIONS)
      if (contributor) assert(!blocked.includes(contributor), `${contributor} must be runnable`);
  }
}

export async function exerciseTrajectoryCatalog(page, panel) {
  await assertTrajectoryCatalogDiscovery(panel);
  for (const [id, contributor] of TRAJECTORY_OPERATIONS) {
    await panel.getByLabel('Operation', {exact: true}).selectOption(id);
    await panel.getByLabel('Engine', {exact: true}).selectOption(contributor ? 'auto' : 'cpu');
    await panel.getByTestId('catalog-run').click();
    await panel.getByTestId('catalog-result').waitFor({timeout: 30000});
    const engine = await panel.getByTestId('catalog-engine').innerText();
    assert.match(engine, contributor ? /Engine: GPU/ : /Engine: CPU reference/, id);
    if (contributor) assert(engine.includes(contributor), `${id} dispatches ${contributor}`);
    assert.match(
      await panel.getByTestId('catalog-result-summary').innerText(),
      /trackCount: 2/,
      id
    );
    assert.equal(await panel.getByTestId('catalog-error').count(), 0, id);
    assert(
      (await panel.getByRole('table', {name: 'Result preview'}).count()) > 0,
      `${id} has typed output`
    );
  }
  // Native scalar tables have no GeoJSON column: explicitly map the point coordinate roles.
  const pointDatasetId = 'trajectory-native-point-observations';
  await page.evaluate(datasetId => {
    const origin = Date.UTC(2026, 0, 1);
    window.__keplerComputeTest.addNativeCpuDataset(
      datasetId,
      [
        {name: 'track_id', type: 'string', format: '', analyzerType: 'STRING'},
        {name: 'timestamp', type: 'real', format: '', analyzerType: 'FLOAT'},
        {name: 'east', type: 'real', format: '', analyzerType: 'FLOAT'},
        {name: 'north', type: 'real', format: '', analyzerType: 'FLOAT'}
      ],
      [
        ['a', origin + 2000, -73.98, 40.74],
        ['a', origin, -74, 40.74],
        ['a', origin + 1000, -73.99, 40.74]
      ]
    );
  }, pointDatasetId);
  await panel.getByLabel('Operation', {exact: true}).selectOption('trajectory-thinning');
  await panel.getByLabel('Engine', {exact: true}).selectOption('cpu');
  await panel.getByLabel(/^Thinning tolerance/).fill('0');
  await panel.getByLabel(/^Point longitude column/).fill('east');
  await panel.getByLabel(/^Point latitude column/).fill('north');
  await panel.getByLabel('Trajectory observations source', {exact: true}).selectOption('native');
  await panel
    .getByLabel('Trajectory observations dataset', {exact: true})
    .locator(`option[value="${pointDatasetId}"]`)
    .waitFor({state: 'attached'});
  await panel
    .getByLabel('Trajectory observations dataset', {exact: true})
    .selectOption(pointDatasetId);
  await panel
    .getByLabel('Trajectory observations geometry field', {exact: true})
    .selectOption('Point rows (mapped longitude/latitude columns)');
  await panel.getByTestId('catalog-run').click();
  await panel.getByTestId('catalog-result').waitFor();
  assert.match(await panel.getByTestId('catalog-result').innerText(), /Trajectory tracks: 1 rows/);
  assert.match(await panel.getByTestId('catalog-result').innerText(), /Trajectory fixes: 3 rows/);
}

/** Runs the real imported, null-prefixed Trips source through the catalog and republishes native Trips. */
export async function exerciseTrajectoryNativeTrips(page, datasetId, geometryField) {
  await closeNativeDialog(page);
  const panel = page.getByRole('region', {name: 'Analysis catalog', exact: true});
  await panel.getByLabel('Operation', {exact: true}).selectOption('trajectory-thinning');
  await panel.getByLabel('Engine', {exact: true}).selectOption('cpu');
  await panel.getByLabel(/^Thinning tolerance/).fill('0');
  await panel.getByLabel('Trajectory observations source', {exact: true}).selectOption('native');
  await panel.getByLabel('Trajectory observations dataset', {exact: true}).selectOption(datasetId);
  await panel
    .getByLabel('Trajectory observations geometry field', {exact: true})
    .selectOption(geometryField);
  await panel.getByTestId('catalog-run').click();
  await panel.getByTestId('catalog-result').waitFor();
  assert.match(await panel.getByTestId('catalog-result').innerText(), /Trajectory tracks: 2 rows/);
  assert.match(await panel.getByTestId('catalog-result').innerText(), /trajectory-empty-geometry/);
  await panel.getByTestId('catalog-publish').first().click();
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.getByText('Trajectory tracks').first().waitFor();
  assert.equal(
    await workspace.getByLabel('Source geometry column', {exact: true}).inputValue(),
    'geometry'
  );
  assert.equal(await workspace.getByLabel('Timed paths (Trips)', {exact: true}).isChecked(), true);
  assert.equal(
    await workspace.getByLabel('Trip timestamp encoding', {exact: true}).inputValue(),
    'unix-millisecond'
  );
  // These exact epoch-millisecond metadata fields need not become resident Float32 scalars.
  // Exclusion retains them in the original Arrow source; Trips reads its own vertex timestamps.
  const preparation = workspace.getByRole('region', {
    name: 'Prepare Trajectory tracks',
    exact: true
  });
  await preparation.getByRole('checkbox', {name: 'start_time (Float64)', exact: true}).uncheck();
  await preparation.getByRole('checkbox', {name: 'end_time (Float64)', exact: true}).uncheck();
  const originalTrips = await page.evaluate(
    identifier =>
      window.__keplerComputeTest.getState().publications.find(source => source.id === identifier)
        .trips,
    datasetId
  );
  await preparation.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await preparation
    .getByRole('status')
    .filter({
      hasText:
        /^2 timed paths, 6 vertices; 0 invalid rows retained but omitted from rendering\. Maximum time rounding: 0 ms\. Ready for upload\.$/
    })
    .waitFor();
  assert.equal(
    await preparation
      .getByRole('button', {name: 'Publish selected fields', exact: true})
      .isEnabled(),
    true
  );
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    originalId =>
      window.__keplerComputeTest
        .getState()
        .publications.some(
          publication => publication.id !== originalId && publication.trips?.tripCount === 2
        ),
    datasetId,
    {timeout: 45000}
  );
  const state = await page.evaluate(() => window.__keplerComputeTest.getState());
  const publication = state.publications.find(
    item => item.id !== datasetId && item.trips?.tripCount === 2
  );
  assert.equal(publication.error, null);
  assert.deepEqual(
    publication.trips.domain,
    originalTrips.domain,
    'native Trips retains the absolute animation domain'
  );
  assert.equal(
    publication.trips.origin,
    originalTrips.origin,
    'native Trips retains the epoch origin'
  );
  assert.equal(
    publication.trips.actualTimeError,
    0,
    'scalar exclusion does not round path timestamps'
  );
  await page.waitForFunction(
    identifier => {
      const api = window.__keplerComputeTest;
      const state = api.getState();
      const published = state.publications.find(source => source.id === identifier);
      if (published?.error) throw new Error(published.error);
      const native = state.nativeLayers.find(layer => layer.id === `${identifier}-trips`);
      if (native?.valid === false) throw new Error(JSON.stringify(native));
      return (
        native?.type === 'trip' &&
        api
          .renderedLayers()
          .some(layer => layer.id === `${identifier}-trips` && layer.loaded && layer.visible)
      );
    },
    publication.id,
    {timeout: 30000}
  );
  await closeNativeDialog(page);
}
