// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {addDataToMap, createNewDatasetSuccess} from '@kepler.gl/actions';
import {keplerGlReducerCore, type KeplerGlState} from '@kepler.gl/reducers';
import {processArrowTable} from '@kepler.gl/processors';
import * as arrow from 'apache-arrow';
import KeplerGlSchema, {type SavedConfigV1} from '@kepler.gl/schemas';
import KeplerTable, {type Datasets} from '@kepler.gl/table';
import {sanitizeSampleConfig} from './sample-map-catalog';

/** Stock tables, never host table subclasses, accessors, cached filter props or GPU bindings. */
async function detachReferenceDataset(dataset: KeplerTable): Promise<KeplerTable> {
  const table = new KeplerTable({
    info: {id: dataset.id, label: dataset.label, type: dataset.type},
    color: [...dataset.color],
    metadata: structuredClone(dataset.metadata),
    supportedFilterTypes: dataset.supportedFilterTypes,
    disableDataOperation: dataset.disableDataOperation
  });
  const fields = dataset.fields.map(({name, type, format, analyzerType, displayName}) => ({
    name,
    type,
    format,
    analyzerType,
    displayName
  }));
  const arrowTable = dataset.dataContainer.getTable?.() as arrow.Table | undefined;
  const hasArrowTime = arrowTable?.schema.fields.some(field =>
    arrow.DataType.isTimestamp(field.type)
  );
  // This installed Kepler's numeric timeToUnixMilli(value, 'x') multiplies by 1000.
  // Resident Arrow timestamps already yield epoch milliseconds. Give stock Kepler Date
  // cells (its public parser's identity route), keeping the live field descriptors intact.
  // Do not borrow resident field accessors or change the expected filter values/intervals.
  const data = hasArrowTime
    ? {
        fields,
        rows: Array.from({length: arrowTable!.numRows}, (_, rowIndex) =>
          fields.map((_, columnIndex) => {
            const column = arrowTable!.getChildAt(columnIndex)!;
            const value = column.get(rowIndex);
            return value !== null && arrow.DataType.isTimestamp(column.type)
              ? new Date(value)
              : value;
          })
        )
      }
    : arrowTable
      ? {
          ...processArrowTable({shape: 'arrow-table', data: arrowTable} as unknown as Parameters<
            typeof processArrowTable
          >[0]),
          fields,
          rows: [],
          arrowTable
        }
      : {
          fields,
          rows: structuredClone(
            Array.from({length: dataset.dataContainer.numRows()}, (_, rowIndex) =>
              fields.map((_, columnIndex) => dataset.dataContainer.valueAt(rowIndex, columnIndex))
            )
          )
        };
  await table.importData({data});
  return table;
}

/**
 * Independent stock Kepler reference run. Stage the original config with addDataToMap and
 * supply detached stock tables at its public async-registration boundary. This is the same
 * CREATE_NEW_DATASET_SUCCESS action as Kepler's tasks dispatch, including restored remote
 * ids/metadata. No network tasks, host table factory, host layer classes or live settings run.
 */
export async function createSampleMapReference(
  original: SavedConfigV1,
  datasets: Datasets,
  label = 'sample'
) {
  let path = 'initialState';
  try {
    let reference: KeplerGlState = keplerGlReducerCore(undefined, {type: '@@INIT'});
    const options = {centerMap: false, keepExistingConfig: false, autoCreateLayers: false};
    path = 'addDataToMap';
    reference = keplerGlReducerCore(
      reference,
      addDataToMap({
        datasets: [],
        // The host replaces the basemap only. Use that same inline background here:
        // Kepler's global task queue must not fetch the original sample style for this oracle.
        config: sanitizeSampleConfig(structuredClone(original)) as SavedConfigV1,
        options
      })
    );
    const datasetIds = new Set(original.config.visState.layers.map(layer => layer.config.dataId));
    for (const filter of original.config.visState.filters ?? []) {
      for (const datasetId of Array.isArray(filter.dataId) ? filter.dataId : [filter.dataId])
        datasetIds.add(datasetId);
    }
    const restored: KeplerTable[] = [];
    for (const datasetId of datasetIds) {
      path = `datasets.${datasetId}`;
      const dataset = datasets[datasetId];
      if (!dataset) throw new Error('Missing restored native dataset');
      restored.push(await detachReferenceDataset(dataset));
    }
    path = 'createNewDatasetSuccess';
    reference = keplerGlReducerCore(
      reference,
      createNewDatasetSuccess({
        results: restored.map(value => ({status: 'fulfilled' as const, value})),
        addToMapOptions: options
      })
    );
    path = 'remoteMetadata';
    // Stock tile formatters can eagerly initialize a data source while merging layers.
    // Observe every metadata promise immediately: a service failure must name its sample
    // and layer, and must not escape as an unrelated unhandled page rejection.
    await Promise.all(
      reference.visState.layerData.map(async (data, index) => {
        const metadata = data?.tileSource?.metadata;
        if (metadata && typeof metadata.then === 'function') {
          try {
            await metadata;
          } catch (error) {
            throw new Error(
              `visState.layers.${reference.visState.layers[index]?.id}.tileSource.metadata: ${error instanceof Error ? error.message : String(error)}`,
              {cause: error}
            );
          }
        }
      })
    );
    // Deferred layers must fail here with an identity, rather than disappear from the oracle.
    for (const layer of original.config.visState.layers) {
      path = `visState.layers.${layer.id}`;
      if (!reference.visState.layers.some(candidate => candidate.id === layer.id))
        throw new Error('Stock Kepler did not restore the original layer');
    }
    path = 'getConfigToSave';
    const saved = KeplerGlSchema.getConfigToSave(reference);
    for (const layer of saved.config.visState.layers) {
      const source = original.config.visState.layers.find(candidate => candidate.id === layer.id);
      // Absent constant colors are assigned by Kepler's module-global cycling generator.
      if (!source?.config.color) delete (layer.config as Partial<typeof layer.config>).color;
    }
    // Only supplied camera settings have a sample-defined expectation; other defaults come
    // from the host's baseline. Width/height are not part of Kepler's saved map-state schema.
    saved.config.mapState = Object.fromEntries(
      Object.entries(saved.config.mapState ?? {}).filter(
        ([key]) => key in (original.config.mapState ?? {})
      )
    ) as typeof saved.config.mapState;
    return JSON.parse(JSON.stringify(saved)) as typeof saved;
  } catch (error) {
    throw new Error(
      `${label}: reference.${path}: ${error instanceof Error ? error.message : String(error)}`,
      {cause: error}
    );
  }
}
