// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type Reducer, createStore, combineReducers, applyMiddleware} from 'redux';
import keplerGlReducer, {type KeplerGlState, enhanceReduxMiddleware} from '@kepler.gl/reducers';
import {getResidentSuitabilityStatus} from './resident-suitability-actions';
import {
  updateResidentGroupByStatus,
  getResidentGroupByStatus,
  getResidentActionRejection
} from './resident-action-guards';
import {getResidentJoinStatus} from './resident-join-actions';
import {appState} from './app-state';
import {DATASET_ID} from './sample-data';
import {updateKeplerComputeFilterPlot} from './kepler-table-bridge';
import {updateFilterPlot, getLineChart} from '@kepler.gl/utils';
import {createAssistantHost} from './assistant-host';
import {createAssistantNoticeCapabilities} from './assistant-host-capabilities';
import {residentLayerClasses} from './layers';
import {isResidentDataset} from './publication';
import {SUPPORTED_VIEW_COUNT} from './split-view';
import {notify} from './app-services';
import {sampleMapNativeMiddleware} from './sample-map-native-transaction';
import {
  residentGroupByMiddleware,
  residentJoinMiddleware,
  residentSuitabilityMiddleware
} from './resident-operations';
import {getActiveRuntime} from './native-config';

const reducer: Reducer<Record<string, Partial<KeplerGlState>>> = keplerGlReducer.initialState({
  mapStyle: {styleType: 'compute-background'},
  visState: {layerClasses: residentLayerClasses}
});

const residentReducer: typeof reducer = (state, action) => {
  const suitabilityStatus = getResidentSuitabilityStatus(action);
  if (suitabilityStatus && state?.['compute']?.visState) {
    const instance = state['compute'];
    const visState = instance.visState;
    if (visState)
      return {
        ...state,
        compute: {
          ...instance,
          visState: {
            ...visState,
            suitabilities: updateResidentGroupByStatus(visState.suitabilities, suitabilityStatus)
          }
        }
      };
  }
  const joinStatus = getResidentJoinStatus(action);
  if (joinStatus && state?.['compute']?.visState) {
    const instance = state['compute'];
    const visState = instance.visState;
    if (visState)
      return {
        ...state,
        compute: {
          ...instance,
          visState: {...visState, joins: updateResidentGroupByStatus(visState.joins, joinStatus)}
        }
      };
  }
  const groupStatus = getResidentGroupByStatus(action);
  if (groupStatus && state?.['compute']?.visState) {
    const instance = state['compute'];
    const visState = instance.visState;
    if (visState)
      return {
        ...state,
        compute: {
          ...instance,
          visState: {
            ...visState,
            groupBys: updateResidentGroupByStatus(visState.groupBys, groupStatus)
          }
        }
      };
  }
  const rejection = getResidentActionRejection(
    state?.['compute'],
    action,
    isResidentDataset,
    undefined,
    SUPPORTED_VIEW_COUNT
  );
  if (rejection) {
    queueMicrotask(() => {
      appState.workflowNotice = rejection.reason;
      notify();
    });
    if (!state) return reducer(undefined, {type: '@@INIT'});
    const instance = state['compute'];
    const visState = instance?.visState;
    const operation = rejection.operation;
    if (!instance || !visState || !operation) return state;
    return {
      ...state,
      compute: {
        ...instance,
        visState: {
          ...visState,
          [operation.list]: visState[operation.list].map(candidate =>
            candidate.id === operation.id ? {...candidate, error: rejection.reason} : candidate
          )
        }
      }
    };
  }
  const next = reducer(state, action);
  const instance = next['compute'];
  if (!instance?.visState) return next;
  const visState = instance.visState;
  if (action.type === 'LUMA_RENDER_ANALYSIS')
    return {
      ...next,
      compute: {...instance, visState: {...visState, layerData: [...visState.layerData]}}
    };
  if (action.type !== 'LUMA_PUBLISH_SUMMARY') return next;
  return {
    ...next,
    compute: {
      ...instance,
      visState: {
        ...visState,
        layers: visState.layers.map(layer =>
          layer.config.dataId === DATASET_ID ? layer.updateLayerDomain(visState.datasets) : layer
        ),
        filters: visState.filters.map(
          filter =>
            updateKeplerComputeFilterPlot(
              visState.datasets,
              filter,
              linked => updateFilterPlot(visState.datasets, linked),
              {
                buildLineChart: (datasets, plotFilter) =>
                  getLineChart(datasets as never, plotFilter as never)
              }
            ) ?? filter
        )
      }
    }
  };
};

export const store = createStore(
  combineReducers({keplerGl: residentReducer}),
  applyMiddleware(
    sampleMapNativeMiddleware,
    residentGroupByMiddleware,
    residentJoinMiddleware,
    residentSuitabilityMiddleware,
    ...enhanceReduxMiddleware([])
  )
);

/** Approved structured assistant operations reuse the manual UI's actions and capability routes. */
export const assistantHost = createAssistantHost({
  store,
  instanceId: 'compute',
  isResidentDataset,
  getPopulation: datasetId => {
    const population = appState.runtime?.session.registry.getDataset(datasetId)?.population;
    return population?.kind === 'stream' && population.state !== 'complete'
      ? 'streaming'
      : 'complete';
  },
  getMemory: () => {
    const usage = appState.runtime?.session.registry.memoryUsage;
    return {
      residentBytes: usage?.totalBytes ?? 0,
      budgetBytes: (usage?.totalBytes ?? 0) + (usage?.availableBytes ?? 0)
    };
  },
  limits: {
    maximumExportRows: 1_000_000,
    maximumLoadBytes: 512 * 1024 * 1024,
    maximumGroupByAggregations: 8
  },
  supportedLayerTypes: ['point', 'arc', 'line', 'grid', 'hexagon', 'heatmap', 'geojson', 'trip'],
  supportedAnalyses: [
    'quantile',
    'equal-interval',
    'natural-breaks',
    'z-score',
    'deviation-from-mean',
    'mad',
    'range-adjust'
  ],
  supportedExportFormats: ['json', 'arrow', 'csv'],
  hostCapabilities: {
    'analysis-panel': request => {
      const analysis = getActiveRuntime().analysis;
      const fields = request['fields'] as string[];
      if (request['datasetId'] !== DATASET_ID)
        throw new Error('Assistant analysis runs on the resident observations dataset.');
      if (fields[0]) analysis.setColumn(fields[0]);
      analysis.setMode(request['recipe'] as Parameters<typeof analysis.setMode>[0]);
    },
    // Export and source loading are explicit, budgeted panel actions; the assistant stages a notice.
    'export-data': request => {
      appState.workflowNotice = `Assistant requested a ${String(request['scope'])} ${String(request['format'])} export of ${String(request['datasetId'])}. Confirm it from the Export panel.`;
      notify();
    },
    'source-workspace': request => {
      appState.workflowNotice = `Assistant requested loading ${String(request['sourceName'])}. Choose it in the Sources panel.`;
      notify();
    },
    ...createAssistantNoticeCapabilities(message => {
      appState.workflowNotice = message;
      notify();
    })
  }
});
