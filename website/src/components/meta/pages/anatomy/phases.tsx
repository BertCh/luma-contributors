// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The recorded kernels of the kernel-density run, grouped by the job they do. Derived from the
// labels in the view model, so a new recording regroups itself; unknown kernels land in "Other".

import {Link} from '@tanstack/react-router';
import {formatCount} from '../../../../lens/format';
import type {LensViewModel, LensViewNode} from '../../../../lens/types';
import type {TopicSlug} from '../../topic-types';

export type PhaseId = 'index' | 'sort' | 'weights' | 'gather' | 'tap' | 'other';

type Phase = {
  id: PhaseId;
  title: string;
  /** One sentence: why the run needs it. */
  why: string;
  /** The page that explains the idea. */
  topic?: TopicSlug;
  topicLabel?: string;
};

const PHASES: readonly Phase[] = [
  {
    id: 'index',
    title: 'Index the points',
    why: 'Bucket points by cell: count per cell, scan the counts to offsets, scatter the points.',
    topic: 'building-blocks',
    topicLabel: 'Grid index'
  },
  {
    id: 'sort',
    title: 'Restore row order',
    why: 'The scatter races, so each cell is sorted by row. The same input always gives the same bits.',
    topic: 'building-blocks',
    topicLabel: 'Sort'
  },
  {
    id: 'weights',
    title: 'Weigh each point',
    why: 'One invocation per point: its weight, or zero when the row is invalid.'
  },
  {
    id: 'gather',
    title: 'Gather per cell',
    why: 'One invocation per raster cell sums the kernel over points in nearby index cells only.',
    topic: 'kernels',
    topicLabel: 'Dispatch'
  },
  {
    id: 'tap',
    title: 'Copy for the record',
    why: 'Added by the recorder, not by the operation: it copies buffers out so this page can draw them.',
    topic: 'lenses',
    topicLabel: 'Snapshot tap'
  },
  {id: 'other', title: 'Other', why: 'Kernels this page does not classify.'}
];

export function getPhaseOfNode(node: LensViewNode): PhaseId {
  if (node.kind === 'copy') return 'tap';
  if (node.label.startsWith('index-')) return 'index';
  if (node.label.startsWith('sort-')) return 'sort';
  if (node.label === 'point-weights') return 'weights';
  if (node.label === 'gather') return 'gather';
  return 'other';
}

export function PhaseStrip({
  model,
  active,
  onActive,
  pinned,
  onPin
}: {
  model: LensViewModel;
  active: PhaseId | null;
  onActive: (phase: PhaseId | null) => void;
  /** Keeps a phase lit after the pointer leaves; pressing it again releases it. */
  pinned: PhaseId | null;
  onPin: (phase: PhaseId | null) => void;
}) {
  const phases = PHASES.map(phase => {
    const nodes = model.nodes.filter(node => getPhaseOfNode(node) === phase.id);
    return {
      ...phase,
      kernels: nodes.reduce((total, node) => total + node.count, 0),
      // Only when every kernel of the phase reports a workload; a partial sum would mislead.
      invocations: nodes.every(node => node.invocations !== undefined)
        ? Math.max(...nodes.map(node => node.invocations ?? 0))
        : 0
    };
  }).filter(phase => phase.kernels > 0);
  return (
    <ol className="anatomy-phases" onPointerLeave={() => onActive(null)}>
      {phases.map(phase => (
        <li
          key={phase.id}
          className="anatomy-phase"
          data-kind={phase.id === 'tap' ? 'copy' : 'compute'}
          data-active={phase.id === active}
          onPointerEnter={() => onActive(phase.id)}
        >
          <span className="anatomy-phase__count">
            {formatCount(phase.kernels)} {phase.kernels === 1 ? 'kernel' : 'kernels'}
          </span>
          <button
            type="button"
            className="anatomy-phase__title"
            aria-pressed={phase.id === pinned}
            title="Show these kernels in the graph"
            onClick={() => onPin(phase.id === pinned ? null : phase.id)}
            onFocus={() => onActive(phase.id)}
            onBlur={() => onActive(null)}
          >
            {phase.title}
          </button>
          <span className="anatomy-phase__why">{phase.why}</span>
          <span className="anatomy-phase__foot">
            {phase.invocations > 0 ? (
              <span>{formatCount(phase.invocations)} invocations</span>
            ) : (
              <span />
            )}
            {phase.topic ? (
              <Link
                to="/how-it-works/$topic"
                params={{topic: phase.topic}}
                onFocus={() => onActive(phase.id)}
                onBlur={() => onActive(null)}
              >
                {phase.topicLabel}
              </Link>
            ) : null}
          </span>
        </li>
      ))}
    </ol>
  );
}
