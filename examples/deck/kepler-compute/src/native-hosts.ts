// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerSourceLinkHost} from './share-link-sources';
import {wrapTo, loadFiles, reorderLayer} from '@kepler.gl/actions';
import {registerNativeDatasetAccess} from './native-source-access';
import {NativeAutoPrepare} from './native-auto-prepare';
import {appState} from './app-state';
import {selectNativeDatasetConfig} from './native-config-capture';
import KeplerGlSchema from '@kepler.gl/schemas';
import {fieldPreparationCoordinator} from './binding3-field-preparation';
import {categoryCoverageRegistry} from './binding3-filter-runtime';
import {type KeplerComputeTableBinding, createKeplerComputeTableClass} from './kepler-table-bridge';
import {DATASET_ID} from './sample-data';
import KeplerTable from '@kepler.gl/table';
import {createResidentInputValidator} from './resident-input-validator';
import {validateInputData, initApplicationConfig} from '@kepler.gl/utils';
import {store} from './app-store';
import {isResidentDataset, publicationsInProgress, publishSource} from './publication';
import {notify, residentEviction} from './app-services';

registerSourceLinkHost({
  loadFiles: files => void store.dispatch(wrapTo('compute', loadFiles(files)))
});

registerNativeDatasetAccess({
  getDatasets: () => store.getState().keplerGl['compute']?.visState?.datasets,
  isResidentDataset
});

/** Native Load Files, saved maps and URLs are prepared automatically; `?autoPrepare=0` restores the manual flow. */
export const nativeAutoPrepare = new NativeAutoPrepare(
  {
    getDatasets: () => store.getState().keplerGl['compute']?.visState?.datasets,
    getLayers: () => store.getState().keplerGl['compute']?.visState?.layers ?? [],
    isResidentDataset,
    getDevice: () =>
      appState.runtime && !appState.runtime.session.deviceLost
        ? appState.runtime.session.device
        : undefined,
    isReady: () =>
      Boolean(appState.runtime) &&
      !appState.runtime?.session.deviceLost &&
      !appState.runtime?.isLayerRegistrationPending &&
      appState.deviceRecovery.status.phase === 'ready' &&
      publicationsInProgress.size === 0,
    getLayerOrder() {
      const order = store.getState().keplerGl['compute']?.visState?.layerOrder ?? [];
      // Layer groups keep their own order; only flat layer stacks are restored.
      return order.every(entry => typeof entry === 'string') ? (order as string[]) : [];
    },
    setLayerOrder(layerIds) {
      const visState = store.getState().keplerGl['compute']?.visState;
      if (!visState || !layerIds.length || visState.layerOrder.some(e => typeof e !== 'string'))
        return;
      const current = visState.layerOrder as string[];
      const order = [
        ...layerIds.filter(id => current.includes(id)),
        ...current.filter(id => !layerIds.includes(id))
      ];
      if (order.join() !== current.join()) store.dispatch(reorderLayer(order));
    },
    captureNativeConfig(datasetId) {
      const captured = selectNativeDatasetConfig(
        KeplerGlSchema.getConfigToSave(store.getState().keplerGl['compute']),
        datasetId
      );
      const color = store.getState().keplerGl['compute']?.visState?.datasets[datasetId]?.color;
      return captured && color ? {...captured, datasetColor: color} : captured;
    },
    publish: request =>
      publishSource(
        request.candidate,
        request.plan.normalization,
        request.plan.coordinates,
        request.plan.geometryColumn,
        request.plan.tripOptions,
        request.plan.h3Column,
        {
          datasetId: request.datasetId,
          ...(request.nativeConfig?.hasLayers ? {nativeConfig: request.nativeConfig} : {})
        }
      )
  },
  typeof location === 'undefined' || new URLSearchParams(location.search).get('autoPrepare') !== '0'
);

nativeAutoPrepare.subscribe(() => queueMicrotask(notify));

fieldPreparationCoordinator.setHost({
  getSourceFieldNames: datasetId =>
    store
      .getState()
      .keplerGl['compute']?.visState?.datasets[datasetId]?.fields.map(field => field.name),
  getExcludedFields: datasetId => nativeAutoPrepare.reports.get(datasetId)?.excluded ?? [],
  requestRepublish: datasetId => nativeAutoPrepare.requestRepublish(datasetId)
});

categoryCoverageRegistry.subscribe(() => queueMicrotask(notify));

function resolveTableBinding(datasetId: string): KeplerComputeTableBinding | undefined {
  const activeRuntime = appState.runtime;
  residentEviction.requestResident(datasetId);
  if (datasetId !== DATASET_ID) return activeRuntime?.publications.get(datasetId)?.binding;
  if (datasetId !== DATASET_ID || !activeRuntime) return undefined;
  return activeRuntime.tableBinding.bind(activeRuntime.explorer);
}

class ResidentKeplerTable extends createKeplerComputeTableClass(KeplerTable, resolveTableBinding) {
  static getInputDataValidator() {
    return createResidentInputValidator(validateInputData);
  }
}

initApplicationConfig({
  maxArrowBatches: Number.MAX_SAFE_INTEGER,
  enableAnnotations: false,
  enableChartsPanel: true,
  // Resident point, line, arc, GeoJSON, H3, icon and trip layers render under Deck GlobeView.
  enableGlobeView: true,
  table: ResidentKeplerTable
});
