// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createStore, applyMiddleware, type Middleware} from 'redux';
import {addDataToMap, createNewDatasetSuccess, removeDataset} from '@kepler.gl/actions';
import {keplerGlReducerCore, enhanceReduxMiddleware, type KeplerGlState} from '@kepler.gl/reducers';
import KeplerGlSchema, {type SavedConfigV1} from '@kepler.gl/schemas';
import KeplerTable from '@kepler.gl/table';
import type {ProtoDataset} from '@kepler.gl/types';
import * as arrow from 'apache-arrow';
import {processArrowTable} from '@kepler.gl/processors';
import {createSampleMapReference} from '../../../../examples/deck/kepler-compute/src/sample-map-config-expectation';
import {selectNativeDatasetConfig} from '../../../../examples/deck/kepler-compute/src/native-config-capture';
import {makeArrowSourceFromNativeDataset} from '../../../../examples/deck/kepler-compute/src/native-dataset-source';
import {sanitizeSampleConfig} from '../../../../examples/deck/kepler-compute/src/sample-map-catalog';

/** Bundled into a fresh Node VM: stock Kepler only, with no app factory or GPU imports. */
export async function runReferenceFixture(serialized: string): Promise<string> {
  const requests: string[] = [];
  const fetchRemote = globalThis.fetch;
  globalThis.fetch = async (input, options) => {
    requests.push(String(input));
    return fetchRemote(input, options);
  };
  try {
    const result = JSON.parse(await runFixture(serialized));
    return JSON.stringify({...result, requests});
  } finally {
    globalThis.fetch = fetchRemote;
  }
}

async function runFixture(serialized: string): Promise<string> {
  const fixture = JSON.parse(serialized) as {
    label: string;
    original: SavedConfigV1;
    datasets: ProtoDataset[];
    remote?: boolean;
    missingDataset?: boolean;
    arrow?: boolean;
    referenceRemoteUrl?: string;
    publishedTime?: boolean;
    republish?: boolean;
    inlineBackground?: boolean;
    arrowIpc?: number[];
  };
  if (fixture.inlineBackground)
    fixture.original = sanitizeSampleConfig(fixture.original) as SavedConfigV1;
  if (fixture.arrowIpc) {
    const data = processArrowTable({
      shape: 'arrow-table',
      data: arrow.tableFromIPC(Uint8Array.from(fixture.arrowIpc))
    } as unknown as Parameters<typeof processArrowTable>[0]);
    if (!data) throw new Error('Arrow ingestion fixture could not be processed');
    fixture.datasets[0]!.data = data;
  }
  if (fixture.publishedTime) {
    for (const dataset of fixture.datasets) {
      dataset.data.rows = dataset.data.rows.map(row =>
        row.map((value, index) =>
          dataset.data.fields[index]!.type === 'timestamp' ? new Date(Number(value)) : value
        )
      );
    }
  }
  if (fixture.arrow) {
    for (const dataset of fixture.datasets) {
      const columns = Object.fromEntries(
        dataset.data.fields.map((field, columnIndex) => [
          field.name,
          dataset.data.rows.map(row => row[columnIndex])
        ])
      );
      const arrowTable = arrow.tableFromArrays(columns);
      dataset.data = {
        // Node resolves the repo's Arrow types; the bundle uses Kepler's single Arrow runtime.
        ...processArrowTable({shape: 'arrow-table', data: arrowTable} as unknown as Parameters<
          typeof processArrowTable
        >[0]),
        fields: dataset.data.fields,
        rows: [],
        arrowTable
      };
    }
  }
  const options = {centerMap: false, keepExistingConfig: false, autoCreateLayers: false};
  let finish: (error?: Error) => void = () => {};
  const observer: Middleware = () => next => action => {
    const nativeAction = action as Parameters<typeof createNewDatasetSuccess.match>[0];
    const result = next(nativeAction);
    if (createNewDatasetSuccess.match(nativeAction)) {
      const failure = nativeAction.payload.results.find(
        candidate => candidate.status === 'rejected'
      );
      finish(failure?.status === 'rejected' ? new Error(String(failure.reason)) : undefined);
    }
    return result;
  };
  const store = createStore(
    keplerGlReducerCore,
    applyMiddleware(...enhanceReduxMiddleware([observer]))
  );
  if (fixture.remote || fixture.publishedTime) {
    // The network/metadata registration boundary is supplied, as it is for the page oracle.
    // No remote service, tile renderer or GPU is required to exercise the stock reducers.
    store.dispatch(
      addDataToMap({datasets: [], config: structuredClone(fixture.original), options})
    );
    const tables = await Promise.all(
      fixture.datasets.map(async dataset => {
        const table = new KeplerTable({
          info: dataset.info,
          color: [12, 34, 56],
          metadata: dataset.metadata
        });
        await table.importData({data: dataset.data});
        return table;
      })
    );
    store.dispatch(
      createNewDatasetSuccess({
        results: tables.map(value => ({status: 'fulfilled' as const, value})),
        addToMapOptions: options
      })
    );
  } else {
    // Full public addDataToMap + real stock CREATE_TABLE_TASK, not a mocked reducer boundary.
    const completion = new Promise<void>((resolve, reject) => {
      finish = error => (error ? reject(error) : resolve());
    });
    store.dispatch(
      addDataToMap({
        datasets: fixture.datasets,
        config: structuredClone(fixture.original),
        options
      })
    );
    await completion;
  }
  const baseline = KeplerGlSchema.getConfigToSave(store.getState());
  if (fixture.arrowIpc) {
    const dataset = Object.values((store.getState() as KeplerGlState).visState.datasets)[0]!;
    const containerTable = dataset.dataContainer.getTable?.() as arrow.Table;
    const source = makeArrowSourceFromNativeDataset(dataset, {
      maximumRows: 100,
      maximumCells: 10000
    });
    // Reproduce the publication's explicit native schema and retained Arrow input, including
    // the geojson role over a physically native GeoArrow column. The oracle must handle both.
    const published = new KeplerTable({
      info: {id: dataset.id, label: dataset.label},
      color: [...dataset.color]
    });
    await published.importData({
      data: {
        fields: dataset.fields.map(field => ({
          name: field.name,
          type: field.name === 'geometry' ? 'geojson' : field.type,
          format: field.format,
          analyzerType: field.analyzerType
        })),
        rows: [],
        cols: source.table.schema.fields.map(field => source.table.getChild(field.name)!),
        arrowTable: source.table
      }
    });
    const expected = await createSampleMapReference(
      fixture.original,
      {[dataset.id]: published},
      fixture.label
    );
    store.dispatch({type: '@@VERIFY_ARROW_REFERENCE_TASK_QUEUE'});
    await new Promise(resolve => setTimeout(resolve, 0));
    return JSON.stringify({
      actual: baseline,
      expected,
      fieldEncoding: dataset.fields
        .find(field => field.name === 'geometry')
        ?.metadata?.['get']('ARROW:extension:name'),
      containerEncoding: containerTable.schema.fields
        .find(field => field.name === 'geometry')
        ?.metadata.get('ARROW:extension:name'),
      restoredIpc: Array.from(arrow.tableToIPC(source.table as unknown as arrow.Table)),
      sameBuffers:
        source.table.getChild('geometry')!.data[0]!.children[0]!.valueOffsets.buffer ===
        containerTable.getChild('geometry')!.data[0]!.children[0]!.valueOffsets.buffer
    });
  }
  if (fixture.republish) {
    for (const dataset of fixture.datasets) {
      const captured = selectNativeDatasetConfig(baseline, dataset.info.id!);
      store.dispatch(removeDataset(dataset.info.id!));
      const completion = new Promise<void>((resolve, reject) => {
        finish = error => (error ? reject(error) : resolve());
      });
      store.dispatch(
        addDataToMap({
          datasets: dataset,
          config: captured!.config as SavedConfigV1,
          options: {...options, keepExistingConfig: true}
        })
      );
      await completion;
    }
  }
  const state = store.getState() as KeplerGlState;
  if (fixture.publishedTime) {
    for (const dataset of Object.values(state.visState.datasets)) {
      const columns = Object.fromEntries(
        dataset.fields.map((field, index) => [
          field.name,
          field.type === 'timestamp'
            ? arrow.vectorFromArray(
                fixture.datasets[0]!.data.rows.map(row => Number(row[index])),
                new arrow.TimestampMillisecond()
              )
            : arrow.vectorFromArray(fixture.datasets[0]!.data.rows.map(row => row[index]))
        ])
      );
      const source = new arrow.Table(columns);
      dataset.fields.forEach(field => {
        if (field.type === 'timestamp') field.format = 'x';
      });
      dataset.dataContainer.getTable = () => source;
    }
  }
  if (fixture.referenceRemoteUrl) {
    for (const dataset of Object.values(state.visState.datasets)) {
      dataset.metadata = {...dataset.metadata, metadataUrl: fixture.referenceRemoteUrl};
    }
  }
  const before = JSON.stringify(KeplerGlSchema.getConfigToSave(state));
  const originalFields = Object.values(state.visState.datasets).map(dataset => dataset.fields);
  // These canaries would fail if the oracle reused host table operations or source accessors.
  for (const dataset of Object.values(state.visState.datasets)) {
    dataset.filterTable = () => {
      throw new Error('Live table filtering called by reference');
    };
    dataset.getColumnFilterProps = () => {
      throw new Error('Live filter props called by reference');
    };
  }
  const expected = await createSampleMapReference(
    fixture.original,
    fixture.missingDataset ? {} : state.visState.datasets,
    fixture.label
  );
  // Reducer tasks share Kepler's global queue. A later host action would otherwise run
  // tasks accidentally enqueued by the oracle (including the original remote basemap).
  store.dispatch({type: '@@VERIFY_REFERENCE_TASK_QUEUE'});
  await new Promise(resolve => setTimeout(resolve, 0));
  if (before !== JSON.stringify(KeplerGlSchema.getConfigToSave(state)))
    throw new Error('Reference mutated the live presentation');
  Object.values(state.visState.datasets).forEach((dataset, index) => {
    if (dataset.fields !== originalFields[index]) throw new Error('Reference mutated live fields');
  });
  return JSON.stringify({actual: JSON.parse(before), expected, baseline});
}
