// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Deck, WebMercatorViewport} from '@deck.gl/core';
import type {AutomaticRecoveryPolicy} from './automatic-recovery-policy';
import type {ComputeRuntime, ExplorerSummary} from './compute-runtime';
import type {ResidentDeviceRecovery} from './device-recovery';
import {createSampleData} from './sample-data';
import {parseStartupSample} from './sample-map-catalog';

export type SubmittedPerformanceFrame = {
  timestamp: number;
  explorerRevision: number;
  sourceRevision: number;
  stage: 'submitted';
};

export type SecondaryViewCapture = {
  layers: unknown;
  viewState: unknown;
  onViewStateChange?: (parameters: unknown) => void;
};

/**
 * The mutable state of the Kepler compute app. One explicit holder replaces the module-level
 * `let` variables that `main.tsx` used to own, so feature modules read and write the same state
 * without importing `main.tsx`.
 */
class AppState {
  rowCount = parseStartupSample(location.search).rowCount;
  source = createSampleData(this.rowCount);
  runtime: ComputeRuntime | null = null;
  submittedPerformanceFrames:
    | {owner: ComputeRuntime; datasets: Map<string, SubmittedPerformanceFrame>}
    | undefined;
  currentError = '';
  workflowNotice = '';
  latestSummary: ExplorerSummary | null = null;
  unsupportedSelection = false;
  deviceGeneration = 0;
  checkpointScheduled = false;
  pendingDeviceInitialization: {
    generation: number;
    signal: AbortSignal;
    resolve(runtime: ComputeRuntime): void;
    reject(reason: unknown): void;
  } | null = null;
  automaticRecovery: AutomaticRecoveryPolicy | undefined;
  /** Assigned by `main.tsx` right after the policy is parsed, before any other module reads it. */
  deviceRecovery!: ResidentDeviceRecovery<ComputeRuntime>;
  publicationSequence = 0;
  /** One-shot dataset id for the next publication (`?sqlFixture=` names its tables). */
  requestedPublicationId: string | undefined;
  nativeMapZoom: number | undefined;
  nativeMapViewport: WebMercatorViewport | undefined;
  previousNativeMapState: object | undefined;
  secondaryView: SecondaryViewCapture | null = null;
  secondaryMapResizeObserver: ResizeObserver | null = null;
  /** View 1's own (empty) Deck: its props carry map 1's native view-state, hover and click handlers. */
  secondaryDeck: Deck | null = null;
  /** Per-view view-states for the shared Deck while Dual or Swipe is active, otherwise null. */
  splitViewStates: Record<string, unknown> | null = null;
  /** Live swipe divider position in percent (null outside Swipe), published by the maps layout. */
  swipePercentage: number | null = null;
  /**
   * The hovered popover, the click-pinned popover and the compare-mode pinned read all start
   * together, but the resident explorer admits only a few concurrent reads and answers the rest
   * with null. Tooltip reads therefore run one at a time.
   */
  tooltipReadQueue: Promise<unknown> = Promise.resolve();
  groupByNotice = '';
  joinNotice = '';
  suitabilityNotice = '';
  previousLayerTopology = '';
  previousLayerIds = new Set<string>();
  previousHeatmapIds = new Set<string>();
  cleanupScheduled = false;
  injectedRestoreFailureIndex = -1;
}

export const appState = new AppState();
