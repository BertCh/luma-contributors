// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useRef, useState} from 'react';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerDatasetSnapshot
} from '@luma-contributors/kepler';
import {ArrowStreamController} from './arrow-stream-controller';
import {
  planRemoteRefetch,
  runRemoteRefetch,
  type RefetchItemStatus,
  type RefetchPlanItem,
  type RefetchSource
} from './remote-refetch-recovery';
import {DEFAULT_BACKOFF_OPTIONS} from './automatic-recovery-policy';

export type RefetchHostDependencies = {
  getSession(): KeplerComputeSession;
  getCompletionFence(): PromiseLike<unknown>;
  /** Same publication path as the Stream panel's "Add completed stream". */
  attach(dataset: KeplerDatasetSnapshot<KeplerArrowDataset>, label: string): Promise<void>;
};

/**
 * Streams one recorded remote origin into a NEW dataset id and publishes it. Retry and resume are
 * always on here because the user explicitly asked to refresh; nothing resident is replaced.
 */
export async function refetchRemoteItem(
  item: RefetchPlanItem,
  signal: AbortSignal,
  dependencies: RefetchHostDependencies
): Promise<void> {
  const controller = new ArrowStreamController({
    session: dependencies.getSession(),
    id: item.newDatasetId,
    normalization: {float64: {mode: 'exact'}},
    maximumInputBytes: 128 << 20,
    maximumInputChunkBytes: 1 << 20,
    maximumQueuedBytes: 32 << 20,
    maximumBatchRows: 250000,
    getCompletionFence: dependencies.getCompletionFence,
    remoteUrl: item.url,
    retry: {options: DEFAULT_BACKOFF_OPTIONS},
    resume: {}
  });
  const cancel = () => controller.cancel();
  signal.addEventListener('abort', cancel, {once: true});
  try {
    await controller.startRemote();
    await controller.resume();
    await controller.finish();
    await controller.publish(dataset => dependencies.attach(dataset, item.label));
  } catch (error) {
    await controller.dispose().catch(() => undefined);
    throw error;
  } finally {
    signal.removeEventListener('abort', cancel);
  }
}

/** Recovery-area section: plan and run an explicit refetch of every recorded remote origin. */
export function RemoteRefetchSection({
  getSources,
  getExistingIds,
  dependencies
}: {
  getSources(): readonly RefetchSource[];
  getExistingIds(): ReadonlySet<string>;
  dependencies: RefetchHostDependencies;
}): React.JSX.Element | null {
  const [statuses, setStatuses] = useState<readonly RefetchItemStatus[]>([]);
  const [error, setError] = useState('');
  const controller = useRef<AbortController | undefined>(undefined);
  const sources = getSources();
  const plan = planRemoteRefetchSafely(sources, getExistingIds());
  if (!plan.length && !statuses.length) return null;
  const isRunning = statuses.some(
    status => status.phase === 'fetching' || status.phase === 'pending'
  );
  const start = async () => {
    setError('');
    const abort = new AbortController();
    controller.current = abort;
    const current = planRemoteRefetchSafely(getSources(), getExistingIds());
    try {
      await runRemoteRefetch({
        plan: current,
        signal: abort.signal,
        onStatus: next => setStatuses([...next]),
        refetch: (item, signal) => refetchRemoteItem(item, signal, dependencies)
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  };
  return (
    <section aria-label="Remote source refetch" data-testid="remote-refetch">
      <h2>Refetch remote sources</h2>
      <p>
        Re-download recorded remote origins as new datasets. Existing datasets are never replaced,
        and each source reports its own result.
      </p>
      <ul>
        {(statuses.length ? statuses.map(status => status.item) : plan).map(item => {
          const status = statuses.find(candidate => candidate.item === item);
          return (
            <li key={item.newDatasetId}>
              {item.label}: {status?.phase ?? 'ready'}
              {status?.error ? ` - ${status.error}` : ''}
            </li>
          );
        })}
      </ul>
      {error ? <p role="alert">{error}</p> : null}
      <button type="button" disabled={isRunning || !plan.length} onClick={() => void start()}>
        Refetch {plan.length} remote source{plan.length === 1 ? '' : 's'}
      </button>
      {isRunning ? (
        <button type="button" onClick={() => controller.current?.abort()}>
          Cancel refetch
        </button>
      ) : null}
    </section>
  );
}

function planRemoteRefetchSafely(
  sources: readonly RefetchSource[],
  existingIds: ReadonlySet<string>
): RefetchPlanItem[] {
  // A tampered recorded origin is skipped rather than fetched.
  const valid = sources.filter(source => {
    try {
      planRemoteRefetch([source], existingIds);
      return true;
    } catch {
      return false;
    }
  });
  return planRemoteRefetch(valid, existingIds);
}
