// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {isSampleRecord} from './sample-map-catalog';

let generation = 0;
let controller: AbortController | undefined;
let mutationCount = 0;
const idleListeners = new Set<() => void>();

export function cancelSampleMapLoads(
  reason = 'Sample loading was superseded by a workspace change.'
): void {
  generation++;
  controller?.abort(new Error(reason));
}

export function beginSampleMapRequest() {
  cancelSampleMapLoads('Sample loading was superseded by a newer request.');
  controller = new AbortController();
  const current = generation;
  const requestController = controller;
  return {
    signal: requestController.signal,
    isCurrent: () => generation === current && !requestController.signal.aborted,
    abort: (reason: Error) => requestController.abort(reason)
  };
}

export function beginSampleMapMutation(): () => void {
  mutationCount++;
  let finished = false;
  return () => {
    if (!finished) {
      finished = true;
      mutationCount--;
      if (mutationCount === 0) {
        for (const listener of idleListeners) listener();
        idleListeners.clear();
      }
    }
  };
}

export function isSampleMapMutationPending(): boolean {
  return mutationCount > 0;
}

export function whenSampleMapMutationsSettled(): Promise<void> {
  return mutationCount === 0
    ? Promise.resolve()
    : new Promise(resolve => idleListeners.add(resolve));
}

/** A deliberate checkpoint deferral must never erase a previously healthy checkpoint. */
export class SampleMapCheckpointDeferredError extends Error {}

export const REMOTE_SAMPLE_RECOVERY_NOTICE =
  'Remote sample maps are not recoverable in the current Arrow-source recovery manifest. GPU recovery restores the last healthy workspace; remote tiles and their current presentation will need to be loaded again. The healthy checkpoint is retained.';

export function assertSampleMapCheckpointSupported(
  config: unknown,
  datasets: readonly {type?: string}[] = []
): void {
  if (isSampleMapMutationPending())
    throw new SampleMapCheckpointDeferredError(
      'A sample load is still in progress; the healthy checkpoint is retained.'
    );
  const configuration = isSampleRecord(config) ? config['config'] : undefined;
  const visState = isSampleRecord(configuration) ? configuration['visState'] : undefined;
  const layers = isSampleRecord(visState) ? visState['layers'] : undefined;
  if (
    datasets.some(dataset =>
      ['vector-tile', 'raster-tile', 'wms-tile'].includes(dataset.type ?? '')
    ) ||
    (Array.isArray(layers) &&
      layers.some(
        layer =>
          isSampleRecord(layer) &&
          ['vectorTile', 'rasterTile', 'wms'].includes(String(layer['type']))
      ))
  )
    throw new SampleMapCheckpointDeferredError(REMOTE_SAMPLE_RECOVERY_NOTICE);
}
