// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The stage stepper and the recorded-kernel strip the building-blocks figures share. The stepper
// starts on "all": every figure shows its whole idea before a step is pressed.

import type {KeyboardEvent, ReactNode} from 'react';

export type Stage = {id: string; title: string};

export const ALL_STAGES = 'all';

/** Buttons for "all" plus each stage, with previous and next; arrow keys move between stages. */
export function Stepper({
  label,
  stages,
  active,
  onActive,
  numbered = true
}: {
  label: string;
  /** Prefix each stage title with its position. */
  numbered?: boolean;
  stages: readonly Stage[];
  active: string;
  onActive: (id: string) => void;
}) {
  const ids = [ALL_STAGES, ...stages.map(stage => stage.id)];
  const index = Math.max(0, ids.indexOf(active));
  const move = (offset: number) => onActive(ids[(index + offset + ids.length) % ids.length]);
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'ArrowRight') {
      event.preventDefault();
      move(1);
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault();
      move(-1);
    }
  };
  return (
    <div
      className="meta-controls building-blocks-stepper"
      role="group"
      aria-label={label}
      onKeyDown={onKeyDown}
    >
      <button
        type="button"
        className="meta-play"
        onClick={() => move(-1)}
        aria-label="Previous stage"
      >
        ←
      </button>
      <div className="meta-toggle building-blocks-stepper__stages">
        <button
          type="button"
          aria-pressed={active === ALL_STAGES}
          onClick={() => onActive(ALL_STAGES)}
        >
          all
        </button>
        {stages.map((stage, stageIndex) => (
          <button
            key={stage.id}
            type="button"
            aria-pressed={active === stage.id}
            onClick={() => onActive(stage.id)}
          >
            {numbered ? `${stageIndex + 1} ` : ''}
            {stage.title}
          </button>
        ))}
      </div>
      <button type="button" className="meta-play" onClick={() => move(1)} aria-label="Next stage">
        →
      </button>
    </div>
  );
}

/** Whether a part of a figure is emphasised: everything is, until one stage is chosen. */
export function isLit(active: string, ...stageIds: string[]): boolean {
  return active === ALL_STAGES || stageIds.includes(active);
}

export type StripKernel = {
  id: string;
  label: string;
  /** Stage ids that light this kernel; empty for setup kernels. */
  stages: readonly string[];
  /** `readback` draws the snapshot tap in the signal colour. */
  kind?: 'kernel' | 'readback';
  detail?: ReactNode;
};

/** The recorded kernels of a run in execution order, those of the chosen stage lit. */
export function KernelStrip({
  title,
  kernels,
  active
}: {
  title: ReactNode;
  kernels: readonly StripKernel[];
  active: string;
}) {
  return (
    <div className="building-blocks-strip">
      <div className="building-blocks-strip__title">{title}</div>
      <ol className="building-blocks-strip__list">
        {kernels.map(kernel => {
          const lit = active !== ALL_STAGES && kernel.stages.includes(active);
          return (
            <li
              key={kernel.id}
              className="building-blocks-chip"
              data-kind={kernel.kind ?? 'kernel'}
              data-lit={lit ? 'true' : undefined}
              data-dim={active !== ALL_STAGES && !lit ? 'true' : undefined}
              title={kernel.id}
            >
              <span className="building-blocks-chip__label">{kernel.label}</span>
              {kernel.detail ? (
                <span className="building-blocks-chip__detail">{kernel.detail}</span>
              ) : null}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
