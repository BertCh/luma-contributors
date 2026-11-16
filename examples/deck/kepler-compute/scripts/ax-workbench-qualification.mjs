// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Run against the development host; the implementor sandbox cannot launch Chromium.
// node scripts/ax-workbench-qualification.mjs <url> <evidence-directory>
import {writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {runAcceptance, screenshot, waitForWorkbench} from './workflow-utils.mjs';

await runAcceptance('ax-workbench-qualification', async (page, directory) => {
  const publication = await page.evaluate(async () => {
    const workbench = window.__keplerWorkflowTest.store;
    const {appState} = await import('/src/app-state.ts');
    const source = {
      id: 'observations',
      operationId: 'source-geojson',
      inputs: {},
      position: {x: 0, y: 0},
      parameters: {
        coordinateSystem: {from: 'static', value: 'EPSG:4326'},
        text: {
          from: 'static',
          value: JSON.stringify({
            type: 'FeatureCollection',
            features: Array.from({length: 12}, (_, index) => ({
              type: 'Feature',
              properties: {},
              geometry: {
                type: 'Point',
                coordinates: [-87.6 + (index % 4) * 0.001, 41.8 + Math.floor(index / 4) * 0.001]
              }
            }))
          })
        }
      }
    };
    const density = {
      id: 'density',
      operationId: 'kernel-density-grid',
      position: {x: 300, y: 0},
      inputs: {points: {stepId: source.id, output: 'data'}},
      parameters: {cellSize: {from: 'static', value: 100}, bandwidth: {from: 'static', value: 200}},
      display: ['density-cells']
    };
    workbench.addWorkflow({
      schemaVersion: 1,
      id: 'ax-browser-presentation',
      title: 'AX browser presentation',
      inputs: [],
      steps: [source, density],
      settings: {engine: 'auto', autoRun: false}
    });
    workbench.setDock({open: true, inspectorTab: 'result'});
    workbench.select([density.id]);
    const record = await workbench.run();
    const state = workbench.getState().runner.steps[density.id];
    const data = state.result?.outputs['density-cells'];
    const presentation = data?.analysisPresentation;
    const published = state.published?.['density-cells'];
    if (
      record?.status !== 'done' ||
      !published ||
      !presentation?.presentation ||
      !data.analysisReport ||
      state.result.engine !== 'gpu'
    )
      throw new Error(`AX browser GPU/publication failed: ${JSON.stringify({record, state})}`);
    // Read the mounted host through its existing hook. Importing app-store here can create
    // another Redux store with no registered compute instance under a different Vite URL.
    const visState = window.__keplerComputeTest.saveNativeConfig().config.visState;
    const layer = visState.layers.find(
      candidate => candidate.config.dataId === published.datasetId
    );
    // Kepler's saved schema puts field/scale in visualChannels, outside config.
    if (!layer || layer.visualChannels?.colorScale !== 'customOrdinal' || !layer.visualChannels.colorField?.name)
      throw new Error(`AX browser class scale missing: ${JSON.stringify(layer)}`);
    const field = layer.visualChannels.colorField.name;
    const mapTable = appState.runtime.publications.get(published.datasetId)?.dataset.value.source;
    const classes = mapTable?.getChild(field) ? Array.from(mapTable.getChild(field)) : undefined;
    if (!classes)
      throw new Error(`AX published class column ${field} is missing from the retained source.`);
    const colorMap = layer.config.visConfig.colorRange.colorMap;
    const toHex = color => `#${color.map(value => value.toString(16).padStart(2, '0')).join('')}`;
    const expectedMap = presentation.legend.map(entry => [entry.label, toHex(entry.color)]);
    const counts = expectedMap.map(([label]) => classes.filter(value => value === label).length);
    const expectedCounts = presentation.legend.map(entry => entry.count);
    const plan = presentation.presentation;
    const expectedClasses = data.table.columns[plan.column].map(value => {
      const index = plan.breaks.findIndex(bound => value <= bound);
      return presentation.legend[index < 0 ? plan.ranges.length - 1 : index].label;
    });
    if (
      JSON.stringify(colorMap) !== JSON.stringify(expectedMap) ||
      JSON.stringify(counts) !== JSON.stringify(expectedCounts) ||
      JSON.stringify(classes) !== JSON.stringify(expectedClasses) ||
      Object.hasOwn(data.table.columns, field)
    ) {
      throw new Error(
        `AX browser legend/map mismatch: ${JSON.stringify({colorMap, expectedMap, counts, expectedCounts, classes, expectedClasses, breaks: plan.breaks})}`
      );
    }
    return {
      datasetId: published.datasetId,
      engine: state.result.engine,
      breaks: plan.breaks,
      colorMap,
      counts,
      report: data.analysisReport
    };
  });
  await page
    .getByTestId('workflow-result-output-density-cells')
    .getByRole('list', {name: /legend/})
    .waitFor();
  // The shared result card replaces the legacy report disclosure when analysisRun is retained.
  const card = page.getByTestId('ax-result-card');
  await card.waitFor();
  if (!publication.report ||
    await card.getByTestId('ax-result-sentence').innerText() !== publication.report.sentence ||
    await card.getByTestId('ax-result-engine').innerText() !== publication.report.engine)
    throw new Error(`Workbench result card lost its run report: ${JSON.stringify(publication.report)}`);
  await card.locator('summary').filter({hasText: /^What was chosen$/}).click();
  const cardText = await card.innerText();
  for (const chosen of publication.report.chosen)
    if (!cardText.includes(chosen)) throw new Error(`Workbench result card lost chosen setting: ${chosen}`);
  await screenshot(page, directory, 'ax-workbench-presentation');

  const residentId = 'ax-workbench-resident-input';
  await page.evaluate(async datasetId => {
    const workbench = window.__keplerWorkflowTest.store;
    const {appState} = await import('/src/app-state.ts');
    const {publishSource} = await import('/src/publication.ts');
    const {makeArrowTableFromCatalogTable} = await import('/src/analysis-catalog-table.ts');
    const data = workbench.getState().runner.steps.observations.result.outputs.data;
    const table = makeArrowTableFromCatalogTable(data.table);
    const geometryColumn = data.table.geometryColumn;
    const normalization = {
      columns: table.schema.fields.map(field => field.name).filter(name => name !== geometryColumn),
      float64: {mode: 'round', maximumAbsoluteError: 1e-37, maximumRelativeError: 2 ** -23}
    };
    const owner = appState.runtime;
    const candidate = {
      table,
      label: 'AX resident qualification input',
      isCurrent: () => appState.runtime === owner
    };
    window.__axWorkbenchResidentSource = {candidate, normalization, geometryColumn};
    appState.requestedPublicationId = datasetId;
    await publishSource(candidate, normalization, undefined, geometryColumn);
  }, residentId);
  await waitForWorkbench(
    page,
    'Dedicated resident input must reach the shared profile service',
    async datasetId => {
      const {analysisDatasetContext} = await import('/src/analysis-dataset-context.ts');
      return Boolean(analysisDatasetContext.get(datasetId));
    },
    residentId
  );
  const revisions = [];
  try {
    for (const revisionKind of ['declaration', 'data']) {
      await page.evaluate(async datasetId => {
        const workbench = window.__keplerWorkflowTest.store;
        const {analysisDatasetContext} = await import('/src/analysis-dataset-context.ts');
        const dataset = workbench.host.listDatasets().find(candidate => candidate.id === datasetId);
        if (!dataset) throw new Error(`Dedicated resident input ${datasetId} is missing.`);
        const previousDeclarations = analysisDatasetContext.getDeclarations(dataset.id);
        await analysisDatasetContext.readStatistics(dataset.id);
        analysisDatasetContext.setDeclarations(dataset.id, {
          ...previousDeclarations,
          coordinateModel: 'geographic',
          rowSemantics: 'events'
        });
        const profile = workbench.host.getAnalysisProfile(dataset.id);
        const parameters = {
          dataset: {from: 'static', value: dataset.id},
          maximumRows: {from: 'static', value: 20000},
          geometryMode: {
            from: 'static',
            value: profile.pointColumns ? 'point-columns' : 'geojson-field'
          }
        };
        if (profile.pointColumns) {
          parameters.longitudeField = {from: 'static', value: profile.pointColumns.longitude};
          parameters.latitudeField = {from: 'static', value: profile.pointColumns.latitude};
        } else parameters.geometryField = {from: 'static', value: profile.geometryColumns[0]};
        const source = {
          id: 'source',
          operationId: 'source-dataset',
          inputs: {},
          parameters,
          position: {x: 0, y: 0}
        };
        const density = {
          id: 'density',
          operationId: 'kernel-density-grid',
          inputs: {points: {stepId: source.id, output: 'data'}},
          parameters: {},
          position: {x: 300, y: 0},
          display: ['density-cells']
        };
        workbench.addWorkflow({
          schemaVersion: 1,
          id: 'ax-resident-profile-revision',
          title: 'AX resident profile revision',
          inputs: [],
          steps: [source, density],
          settings: {engine: 'auto', autoRun: false}
        });
        const readDataset = workbench.host.readDataset;
        let release;
        const gate = new Promise(resolve => {
          release = resolve;
        });
        const probe = {
          entered: false,
          record: null,
          error: null,
          release,
          datasetId: dataset.id,
          previousDeclarations,
          readDataset
        };
        window.__axWorkbenchQualification = probe;
        workbench.host.readDataset = async (...arguments_) => {
          probe.entered = true;
          await gate;
          return readDataset(...arguments_);
        };
        void workbench.run().then(
          record => {
            probe.record = record;
          },
          error => {
            probe.error = String(error);
          }
        );
      }, residentId);
      await waitForWorkbench(
        page,
        'Resident source must enter extraction',
        () => window.__axWorkbenchQualification.entered
      );
      await page.evaluate(async kind => {
        const {analysisDatasetContext} = await import('/src/analysis-dataset-context.ts');
        const probe = window.__axWorkbenchQualification;
        if (kind === 'declaration')
          analysisDatasetContext.setDeclarations(probe.datasetId, {
            coordinateModel: 'planar',
            rowSemantics: 'events'
          });
        else {
          const {publishSource} = await import('/src/publication.ts');
          const source = window.__axWorkbenchResidentSource;
          await publishSource(
            {...source.candidate, table: source.candidate.table.slice(0, 11)},
            source.normalization,
            undefined,
            source.geometryColumn,
            undefined,
            undefined,
            {datasetId: probe.datasetId}
          );
        }
        window.__keplerWorkflowTest.store.notifyExternalChange();
        probe.release();
      }, revisionKind);
      await waitForWorkbench(page, 'Obsolete resident profile run must settle', () =>
        Boolean(window.__axWorkbenchQualification.record || window.__axWorkbenchQualification.error)
      );
      const revision = await page.evaluate(async kind => {
        const {analysisDatasetContext} = await import('/src/analysis-dataset-context.ts');
        const workbench = window.__keplerWorkflowTest.store;
        const probe = window.__axWorkbenchQualification;
        try {
          const state = workbench.getState().runner.steps.density;
          if (
            probe.error ||
            probe.record?.status !== 'cancelled' ||
            state.published ||
            state.result
          )
            throw new Error(
              `AX resident revision published obsolete results: ${JSON.stringify({record: probe.record, error: probe.error, state})}`
            );
          return {
            kind,
            datasetId: probe.datasetId,
            status: probe.record.status,
            obsoletePublications: 0
          };
        } finally {
          workbench.host.readDataset = probe.readDataset;
          analysisDatasetContext.setDeclarations(probe.datasetId, probe.previousDeclarations);
          delete window.__axWorkbenchQualification;
        }
      }, revisionKind);
      revisions.push(revision);
    }
  } finally {
    await page.evaluate(async datasetId => {
      const {removePublishedDataset} = await import('/src/publication.ts');
      const probe = window.__axWorkbenchQualification;
      if (probe) {
        window.__keplerWorkflowTest.store.host.readDataset = probe.readDataset;
        probe.release();
      }
      await removePublishedDataset(datasetId);
      delete window.__axWorkbenchResidentSource;
    }, residentId);
  }
  const evidence = {qualifiedAt: new Date().toISOString(), publication, revisions};
  await writeFile(
    join(directory, 'ax-workbench-qualification.json'),
    JSON.stringify(evidence, null, 2)
  );
  return evidence;
});
