// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type KeplerDatasetSnapshot, type KeplerArrowDataset} from '@luma-contributors/kepler';
import {ComputeRuntime} from './compute-runtime';
import {type CoordinateRoles, preparePublishedH3} from './source-publication';
import {type TripPreparationOptions} from './trip-source-options';
import {type ResidentGroupByProvenance} from './resident-group-by-plan';
import {type ResidentJoinLineage, getResidentJoinDerivedDataset} from './resident-join-lineage';
import {
  type ResidentSuitabilityLineage,
  getResidentSuitabilityDerivedDataset
} from './resident-suitability-lineage';
import {type NativeDatasetConfig, findMissingNativeConfigFields} from './native-config-capture';
import {appState} from './app-state';
import {getPopulationLabel} from './remote-stream-source';
import {getNativeCellIndex, CELL_INDEX_METADATA_KEY} from './cell-layer-types';
import {removeDataset, addDataToMap} from '@kepler.gl/actions';
import {fieldPreparationCoordinator} from './binding3-field-preparation';
import {type SourceCandidate} from './source-workspace-panel';
import {type PrepareArrowTableForGPUAnalyticsOptions} from '@luma-contributors/graph/arrow';
import {recordUploadMeasure} from './acceptance-instrumentation';
import {PREPARE_DATA_MESSAGE} from './prepared-layer-classes';
import {DatasetOperationQueue} from './binding2-dataset-operations';
import {type Device} from '@luma.gl/core';
import {residentDatasetIds, plotDiagnostics, notify, residentEviction} from './app-services';
import {store} from './app-store';
import {scheduleRecoveryCheckpoint} from './recovery';
import {getActiveRuntime} from './native-config';

/**
 * The globe basemap's resident tiles are painter-ordered (no depth), so they must draw right after
 * the Kepler globe background layers and before data layers, which they would otherwise cover.
 */
export const publicationsInProgress = new Map<string, number>();

export function beginPublication(datasetId: string): void {
  publicationsInProgress.set(datasetId, (publicationsInProgress.get(datasetId) ?? 0) + 1);
}

export function finishPublication(datasetId: string): void {
  const remaining = (publicationsInProgress.get(datasetId) ?? 1) - 1;
  if (remaining > 0) publicationsInProgress.set(datasetId, remaining);
  else publicationsInProgress.delete(datasetId);
}

export function isResidentDataset(datasetId: string): boolean {
  return residentDatasetIds.has(datasetId);
}

export function reportPlot(filterId: string, reason: string | null): void {
  if (reason) plotDiagnostics.set(filterId, reason);
  else plotDiagnostics.delete(filterId);
  queueMicrotask(notify);
}

export function getGroupByGeneration(dataset: KeplerDatasetSnapshot<KeplerArrowDataset>): string {
  return JSON.stringify([dataset.id, dataset.deviceEpoch, dataset.revision]);
}

export async function attachPublishedDataset(
  activeRuntime: ComputeRuntime,
  dataset: KeplerDatasetSnapshot<KeplerArrowDataset>,
  label: string,
  coordinates?: CoordinateRoles,
  geometryColumn?: string,
  tripOptions?: TripPreparationOptions,
  h3Column?: string,
  groupBy?: ResidentGroupByProvenance,
  isCurrent: () => boolean = () => true,
  join?: ResidentJoinLineage,
  suitability?: ResidentSuitabilityLineage,
  /** Presentation of a native dataset being replaced in place; its layers and filters are kept. */
  nativeConfig?: NativeDatasetConfig
): Promise<void> {
  if (appState.runtime !== activeRuntime || !isCurrent())
    throw new Error('Workspace or request changed before publication');
  // Every dataset/analysis/chart/export surface reads this label; partial prefixes say so.
  label = getPopulationLabel(label, dataset.population);
  const cellIndex = getNativeCellIndex(dataset.source?.metadata);
  beginPublication(dataset.id);
  try {
    if (!nativeConfig) {
      residentDatasetIds.add(dataset.id);
      store.dispatch(removeDataset(dataset.id));
    }
    const published = await activeRuntime.attachDataset(
      dataset,
      label,
      coordinates,
      reportPlot,
      () => {
        queueMicrotask(() => {
          if (appState.runtime === activeRuntime) {
            store.dispatch({type: 'LUMA_RENDER_ANALYSIS'});
            notify();
          }
        });
      },
      geometryColumn,
      tripOptions,
      h3Column,
      groupBy,
      join,
      suitability
    );
    if (
      appState.runtime !== activeRuntime ||
      !isCurrent() ||
      !published.consumer.explorer.isCurrent
    )
      throw new Error('Workspace or source changed before native publication');
    if (nativeConfig) {
      // Saved layers, channels, filters and tooltips must find every field they name; otherwise keep the
      // native dataset (its layers still render natively) and report which fields the copy lacks.
      const missingFields = findMissingNativeConfigFields(
        nativeConfig,
        published.nativeData.fields.map(field => field.name)
      );
      if (missingFields.length)
        throw new Error(
          `Saved layers reference fields not available in the prepared copy: ${missingFields.join(', ')}`
        );
      // The native table is replaced only after the resident copy exists, so a failure keeps it.
      residentDatasetIds.add(dataset.id);
      store.dispatch(removeDataset(dataset.id));
    }
    store.dispatch(
      addDataToMap({
        datasets: {
          info: {
            id: dataset.id,
            label,
            ...(nativeConfig?.datasetColor ? {color: [...nativeConfig.datasetColor]} : {})
          },
          data: published.nativeData,
          ...(groupBy
            ? {
                metadata: {
                  lumaGroupByGeneration: getGroupByGeneration(dataset),
                  derivedDataset: {
                    type: 'groupBy',
                    sourceDataIds: [groupBy.source.datasetId],
                    operationId: groupBy.operationId
                  }
                }
              }
            : join
              ? {
                  metadata: {
                    lumaGroupByGeneration: getGroupByGeneration(dataset),
                    derivedDataset: getResidentJoinDerivedDataset(join)
                  }
                }
              : suitability
                ? {
                    metadata: {
                      lumaGroupByGeneration: getGroupByGeneration(dataset),
                      derivedDataset: getResidentSuitabilityDerivedDataset(suitability)
                    }
                  }
                : {})
        },
        options: {
          centerMap: !nativeConfig && Boolean(coordinates || geometryColumn || h3Column),
          autoCreateLayers: false,
          autoCreateTooltips: false,
          keepExistingConfig: true
        },
        ...(nativeConfig
          ? {config: nativeConfig.config as never}
          : coordinates || geometryColumn || h3Column
            ? {
                config: {
                  version: 'v1',
                  config: {
                    visState: {
                      layers: [
                        {
                          id: `${dataset.id}-${cellIndex ? cellIndex.family : h3Column ? 'h3' : tripOptions ? 'trips' : geometryColumn ? 'geometry' : coordinates?.flow ? 'flow' : 'points'}`,
                          type: cellIndex
                            ? cellIndex.family
                            : h3Column
                              ? 'hexagonId'
                              : tripOptions
                                ? 'trip'
                                : geometryColumn
                                  ? 'geojson'
                                  : coordinates?.flow
                                    ? 'flow'
                                    : 'point',
                          config: {
                            dataId: dataset.id,
                            label,
                            color: [86, 203, 249],
                            isVisible: true,
                            ...(coordinates?.flow ? {columnMode: 'LAT_LNG'} : {}),
                            columns: coordinates?.flow
                              ? {
                                  lat0: coordinates.latitude,
                                  lng0: coordinates.longitude,
                                  lat1: coordinates.flow.targetLatitude,
                                  lng1: coordinates.flow.targetLongitude,
                                  ...(coordinates.flow.count
                                    ? {count: coordinates.flow.count}
                                    : {}),
                                  ...(coordinates.flow.sourceName
                                    ? {sourceName: coordinates.flow.sourceName}
                                    : {}),
                                  ...(coordinates.flow.targetName
                                    ? {targetName: coordinates.flow.targetName}
                                    : {})
                                }
                              : coordinates
                                ? {
                                    lat: coordinates.latitude,
                                    lng: coordinates.longitude,
                                    ...(coordinates.altitude
                                      ? {altitude: coordinates.altitude}
                                      : {})
                                  }
                                : cellIndex
                                  ? {token: cellIndex.column}
                                  : h3Column
                                    ? {hex_id: h3Column}
                                    : tripOptions?.grouping
                                      ? {
                                          id: geometryColumn!,
                                          lat: tripOptions.grouping.latitudeColumn,
                                          lng: tripOptions.grouping.longitudeColumn,
                                          timestamp: tripOptions.grouping.timestampColumn,
                                          ...(tripOptions.grouping.altitudeColumn
                                            ? {altitude: tripOptions.grouping.altitudeColumn}
                                            : {})
                                        }
                                      : {geojson: geometryColumn!},
                            ...(tripOptions?.grouping ? {columnMode: 'table'} : {}),
                            visConfig: {
                              radius: 10,
                              filled: true,
                              outline: false,
                              opacity: 0.8,
                              ...(tripOptions ? {thickness: 2, trailLength: 180} : {})
                            }
                          }
                        }
                      ]
                    }
                  }
                }
              }
            : {})
      })
    );
    await waitForNativeDataset(dataset.id, activeRuntime);
    if (!isCurrent()) throw new Error('Request changed during native dataset registration');
    activeRuntime.requestUpdate();
    notify();
  } finally {
    finishPublication(dataset.id);
    residentEviction.touch(dataset.id);
    scheduleRecoveryCheckpoint();
  }
}

export function waitForNativeDataset(
  datasetId: string,
  activeRuntime: ComputeRuntime
): Promise<void> {
  const isRegistered = () =>
    Boolean(store.getState().keplerGl['compute']?.visState?.datasets[datasetId]);
  if (isRegistered()) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      unsubscribe();
      reject(new Error('Native dataset registration did not complete'));
    }, 15000);
    const unsubscribe = store.subscribe(() => {
      if (appState.runtime !== activeRuntime || isRegistered()) {
        clearTimeout(timeout);
        unsubscribe();
        if (appState.runtime !== activeRuntime)
          reject(new Error('Workspace changed during native registration'));
        else resolve();
      }
    });
  });
}

/** Automatic replacements of one native dataset run through the per-dataset operation queue. */
export async function publishSource(...args: Parameters<typeof publishSourceNow>): Promise<void> {
  const nativeReplacement = args[6];
  if (!nativeReplacement) return publishSourceNow(...args);
  try {
    await datasetOperations.run(nativeReplacement.datasetId, 'replace', () =>
      publishSourceNow(...args)
    );
  } finally {
    fieldPreparationCoordinator.markSettled(nativeReplacement.datasetId);
  }
}

async function publishSourceNow(
  candidate: SourceCandidate,
  normalization: PrepareArrowTableForGPUAnalyticsOptions,
  coordinates?: CoordinateRoles,
  geometryColumn?: string,
  tripOptions?: TripPreparationOptions,
  h3Column?: string,
  /** Automatic preparation replaces a native dataset in place under its own ID and layers. */
  nativeReplacement?: {datasetId: string; nativeConfig?: NativeDatasetConfig}
): Promise<void> {
  const activeRuntime = getActiveRuntime();
  if (!candidate.isCurrent()) throw new Error('Source changed before upload');
  if (h3Column) preparePublishedH3(candidate.table, h3Column);
  const uploadStartMilliseconds = performance.now();
  const uploadBytesBefore = activeRuntime.uploadBytes ?? 0;
  const dataset = activeRuntime.session.importArrowDataset({
    id: nativeReplacement?.datasetId ?? createPublicationId(activeRuntime),
    table: candidate.table,
    normalization,
    ...(candidate.cellIndex ? {metadata: {[CELL_INDEX_METADATA_KEY]: candidate.cellIndex}} : {})
  });
  try {
    if (!candidate.isCurrent()) throw new Error('Source changed during upload');
    await attachPublishedDataset(
      activeRuntime,
      dataset,
      candidate.label,
      coordinates,
      geometryColumn,
      tripOptions,
      h3Column,
      undefined,
      undefined,
      undefined,
      undefined,
      nativeReplacement?.nativeConfig
    );
    recordUploadMeasure(
      'publish-source',
      uploadStartMilliseconds,
      (activeRuntime.uploadBytes ?? 0) - uploadBytesBefore
    );
    if (appState.workflowNotice === PREPARE_DATA_MESSAGE) appState.workflowNotice = '';
    notify();
  } catch (error) {
    // The native dataset stays in Kepler when its automatic replacement fails.
    if (nativeReplacement && !activeRuntime.publications.has(dataset.id))
      residentDatasetIds.delete(dataset.id);
    await activeRuntime.removePublication(dataset.id);
    if (activeRuntime.session.registry.getDataset(dataset.id))
      await activeRuntime.session.removeDataset(dataset.id, Promise.resolve());
    throw error;
  }
}

function createPublicationId(activeRuntime: ComputeRuntime): string {
  if (appState.requestedPublicationId) {
    const requestedId = appState.requestedPublicationId;
    appState.requestedPublicationId = undefined;
    return requestedId;
  }
  let datasetId: string;
  do {
    datasetId = `resident-published-${++appState.publicationSequence}`;
  } while (
    residentDatasetIds.has(datasetId) ||
    activeRuntime.session.registry.getDataset(datasetId)
  );
  return datasetId;
}

export const datasetOperations = new DatasetOperationQueue();

export async function removePublishedDataset(datasetId: string): Promise<void> {
  // Removals of one dataset never overlap each other or a queued replacement.
  await datasetOperations.run(datasetId, 'remove', async () => {
    residentEviction.forget(datasetId);
    fieldPreparationCoordinator.forget(datasetId);
    store.dispatch(removeDataset(datasetId));
    await appState.runtime?.removePublication(datasetId);
    residentDatasetIds.delete(datasetId);
    notify();
  });
}

export async function discardPublishedResult(
  {
    owner,
    result
  }: {
    owner: ComputeRuntime;
    result: {dataset: KeplerDatasetSnapshot<KeplerArrowDataset>; destroy(): Promise<void>};
  },
  label: string
): Promise<void> {
  const dataset = result.dataset;
  const published = owner.publications.get(dataset.id);
  const errors: unknown[] = [];
  if (
    published?.dataset === dataset &&
    published.dataset.revision === dataset.revision &&
    published.dataset.deviceEpoch === dataset.deviceEpoch
  ) {
    // Detach this exact consumer synchronously. Later cleanup must not remove a newer same-ID source.
    owner.publications.delete(dataset.id);
    const nativeDataset = store.getState().keplerGl['compute']?.visState?.datasets[dataset.id];
    if (
      appState.runtime === owner &&
      nativeDataset?.metadata?.['lumaGroupByGeneration'] === getGroupByGeneration(dataset)
    )
      store.dispatch(removeDataset(dataset.id));
    owner.requestUpdate();
    let fence: ReturnType<Device['createFence']> | undefined;
    try {
      if (!owner.session.deviceLost) fence = owner.device.createFence();
    } catch (error) {
      errors.push(error);
    }
    const completion = Promise.race([fence?.signaled ?? Promise.resolve(), owner.device.lost]);
    try {
      const outcomes = await Promise.allSettled([
        owner.styles.retireDataset(dataset.id, completion),
        owner.heatmaps.retireDataset(dataset.id, completion),
        owner.labels.retireDataset(dataset.id, completion),
        owner.pointAltitudes.retireDataset(dataset.id, completion)
      ]);
      for (const outcome of outcomes)
        if (outcome.status === 'rejected') errors.push(outcome.reason);
      try {
        await published.destroy();
      } catch (error) {
        errors.push(error);
      }
    } finally {
      try {
        fence?.destroy();
      } catch (error) {
        errors.push(error);
      }
    }
  }
  try {
    await result.destroy();
  } catch (error) {
    errors.push(error);
  }
  if (
    appState.runtime === owner &&
    !owner.publications.has(dataset.id) &&
    !owner.session.registry.getDataset(dataset.id)
  ) {
    residentDatasetIds.delete(dataset.id);
    notify();
  }
  if (errors.length) throw new AggregateError(errors, `${label} result cleanup failed.`);
}
