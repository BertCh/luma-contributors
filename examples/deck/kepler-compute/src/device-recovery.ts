// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {parseKeplerSandboxState, type KeplerJSONValue} from '@luma-contributors/kepler';
import type {ComputeRuntime} from './compute-runtime';
import type {ResidentRecoverySnapshot} from './resident-device-recovery';
import type {ResidentResultRecipe} from './resident-result-recipes';
import {getSandboxBundleSources, replaySandboxBundle, saveSandbox} from './sandbox-state';

export {ResidentDeviceRecovery} from './resident-device-recovery';
export type {
  ResidentRecoverySnapshot,
  ResidentRecoveryContext,
  ResidentRecoveryStatus,
  ResidentDeviceRecoveryProps
} from './resident-device-recovery';

/** Capture while healthy: analysis recipes deliberately cannot be serialized from stale GPU handles. */
export function captureResidentRecoverySnapshot(props: {
  runtime: ComputeRuntime;
  getNativeConfig(): unknown;
  /** Lineage for exactly the given resident result IDs, ordered by dependency. */
  collectResultRecipes?(residentResultIds: readonly string[]): readonly ResidentResultRecipe[];
}): ResidentRecoverySnapshot {
  assertHealthyRuntime(props.runtime);
  const residentResultIds = Array.from(props.runtime.publications?.values() ?? [])
    .filter(published => published.dataset.value.resident)
    .map(published => published.dataset.id);
  // Without lineage a resident result cannot be recomputed, so the checkpoint must fail visibly.
  if (residentResultIds.length && !props.collectResultRecipes)
    throw new Error('Resident results require recovery lineage');
  const derived = props.collectResultRecipes?.(residentResultIds) ?? [];
  const state = parseKeplerSandboxState(
    JSON.parse(
      saveSandbox({
        ...props,
        // Saving never invokes this replay callback.
        applyNativeConfig() {}
      })
    )
  );
  const sources = getSandboxBundleSources(props.runtime, {omitResident: true});
  if (sources.some(source => !source.isCurrent()))
    throw new Error('Recovery sources changed during checkpoint capture');
  const retained = state.sources
    .filter(descriptor => !residentResultIds.includes(descriptor.dataset.datasetId))
    .map(descriptor => {
      const source = sources.find(
        candidate => candidate.datasetId === descriptor.dataset.datasetId
      );
      if (!source || source.revision !== descriptor.dataset.revision)
        throw new Error('Recovery source manifest is stale');
      return {datasetId: source.datasetId, revision: source.revision, table: source.table};
    });
  assertHealthyRuntime(props.runtime);
  return {
    version: 1,
    capturedAt: Date.now(),
    bundle: {state, sources: retained},
    ...(derived.length ? {derived} : {})
  };
}

/** Call after a fresh runtime has reimported every source under its original logical dataset ID. */
export function restoreResidentRecoveryState(
  snapshot: ResidentRecoverySnapshot,
  props: {
    runtime: ComputeRuntime;
    applyNativeConfig(config: KeplerJSONValue): void | Promise<void>;
    /** Resident results recomputed by the host before this call; defaults to the snapshot's. */
    derivedDatasetIds?: ReadonlySet<string>;
  }
): Promise<ComputeRuntime> {
  if (snapshot.version !== 1) return Promise.reject(new Error('Unsupported recovery snapshot'));
  // Reuse complete schema/role/normalization preflight and explicit source/selection revision
  // rebinding. The host already rebuilt the sources; no second device or import is created here.
  return replaySandboxBundle(snapshot.bundle, {
    ...props,
    derivedDatasetIds:
      props.derivedDatasetIds ?? new Set(snapshot.derived?.map(recipe => recipe.resultId)),
    getNativeConfig: () => snapshot.bundle.state.nativeConfig,
    restoreBundle: async () => props.runtime
  });
}

function assertHealthyRuntime(runtime: ComputeRuntime): void {
  if (runtime.device.isLost || runtime.session.deviceLost)
    throw new Error('Cannot checkpoint a lost device');
}
