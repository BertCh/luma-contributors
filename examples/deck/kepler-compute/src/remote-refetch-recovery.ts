// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {parseRemoteStreamUrl, type NativeStreamProvenance} from './remote-stream-source';

/**
 * Explicit, opt-in refetch recovery for remote-origin sources. Device recovery and bundle restore
 * still rehydrate from retained checksummed Arrow bytes and never fetch silently; this module only
 * runs when the user (or a host flag) asks to refresh a recorded remote origin. Each refetch is a
 * new stream added under a NEW dataset id (the referenced id is never replaced in place), and a
 * failure is recorded as an explicit failed phase for that source while the others continue.
 */
export type RefetchSource = Readonly<{
  datasetId: string;
  label: string;
  stream?: NativeStreamProvenance;
}>;

export type RefetchPlanItem = Readonly<{
  sourceDatasetId: string;
  newDatasetId: string;
  label: string;
  url: string;
}>;

export type RefetchPhase = 'pending' | 'fetching' | 'complete' | 'failed' | 'cancelled';

export type RefetchItemStatus = Readonly<{
  item: RefetchPlanItem;
  phase: RefetchPhase;
  error?: string;
}>;

/** Sources with a recorded remote origin, each assigned a fresh id not present in `existingIds`. */
export function planRemoteRefetch(
  sources: readonly RefetchSource[],
  existingIds: ReadonlySet<string>
): RefetchPlanItem[] {
  const taken = new Set(existingIds);
  const items: RefetchPlanItem[] = [];
  for (const source of sources) {
    const origin = source.stream?.origin;
    if (!origin) continue;
    // Re-validated so a tampered manifest URL (credentials, other schemes) is never fetched.
    const url = parseRemoteStreamUrl(origin.url, undefined).href;
    let sequence = 1;
    let newDatasetId = `${source.datasetId}-refetch-${sequence}`;
    while (taken.has(newDatasetId)) newDatasetId = `${source.datasetId}-refetch-${++sequence}`;
    taken.add(newDatasetId);
    items.push({
      sourceDatasetId: source.datasetId,
      newDatasetId,
      label: `${source.label} (refetched)`,
      url
    });
  }
  return items;
}

/**
 * Runs the plan one source at a time (bounded memory and bandwidth). `refetch` owns streaming,
 * resume and publication; any throw marks only that item failed. Nothing already resident changes.
 */
export async function runRemoteRefetch(props: {
  plan: readonly RefetchPlanItem[];
  signal: AbortSignal;
  refetch(item: RefetchPlanItem, signal: AbortSignal): Promise<void>;
  onStatus?(statuses: readonly RefetchItemStatus[]): void;
}): Promise<readonly RefetchItemStatus[]> {
  const statuses: RefetchItemStatus[] = props.plan.map(item => ({item, phase: 'pending'}));
  const update = (index: number, status: Omit<RefetchItemStatus, 'item'>) => {
    statuses[index] = {item: statuses[index].item, ...status};
    props.onStatus?.(statuses);
  };
  for (let index = 0; index < statuses.length; index++) {
    if (props.signal.aborted) {
      update(index, {phase: 'cancelled'});
      continue;
    }
    update(index, {phase: 'fetching'});
    try {
      await props.refetch(statuses[index].item, props.signal);
      update(index, {phase: 'complete'});
    } catch (error) {
      update(
        index,
        props.signal.aborted
          ? {phase: 'cancelled'}
          : {phase: 'failed', error: error instanceof Error ? error.message : String(error)}
      );
    }
  }
  return statuses;
}
