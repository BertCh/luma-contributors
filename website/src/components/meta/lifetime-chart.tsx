// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Buffer lifetimes of a recorded run: one row per buffer, one column per step of the execution order.
// A bar fills when the first kernel writes the buffer and fades once its last reader has run.

import {formatResourceBytes} from '../../lens/format';
import type {LensResourceLifetime, LensViewModel} from '../../lens/types';

const LABEL_WIDTH = 188;
const COLUMN_WIDTH = 30;
const ROW_HEIGHT = 17;
const TOP = 26;

export type LifetimeState = 'unborn' | 'alive' | 'dead';

/** `stepsRun` kernels have executed. A buffer is alive from the step after its first write to its last use. */
export function getLifetimeState(lifetime: LensResourceLifetime, stepsRun: number): LifetimeState {
  if (stepsRun <= lifetime.first) return 'unborn';
  return stepsRun <= lifetime.last ? 'alive' : 'dead';
}

/** Bytes alive after `stepsRun` kernels. */
export function getLiveBytes(lifetimes: readonly LensResourceLifetime[], stepsRun: number): number {
  let total = 0;
  for (const lifetime of lifetimes) {
    if (getLifetimeState(lifetime, stepsRun) === 'alive') total += lifetime.bytes;
  }
  return total;
}

export function getPeakLiveBytes(
  lifetimes: readonly LensResourceLifetime[],
  stepCount: number
): number {
  let peak = 0;
  for (let step = 1; step <= stepCount; step++)
    peak = Math.max(peak, getLiveBytes(lifetimes, step));
  return peak;
}

function shortenName(name: string, slug: string): string {
  const stem = name.startsWith(`${slug}-`) ? name.slice(slug.length + 1) : name;
  return stem.length > 27 ? `${stem.slice(0, 26)}…` : stem;
}

export function LifetimeChart({model, step}: {model: LensViewModel; step: number}) {
  const lifetimes = model.lifetimes ?? [];
  const stepCount = model.nodes.length;
  const width = LABEL_WIDTH + stepCount * COLUMN_WIDTH + 12;
  const height = TOP + lifetimes.length * ROW_HEIGHT + 10;
  const slug = model.title;
  const playheadX = LABEL_WIDTH + step * COLUMN_WIDTH;
  return (
    <>
      <div className="meta-scroll">
        <svg
          className="lifetime-chart"
          viewBox={`0 0 ${width} ${height}`}
          style={{minWidth: Math.min(width, 520), maxWidth: Math.max(720, width * 1.3)}}
          role="img"
          aria-label={`Lifetimes of ${lifetimes.length} buffers over ${stepCount} steps`}
        >
          {model.nodes.map((node, index) => (
            <g key={node.id}>
              <title>{`${index + 1}. ${node.label}`}</title>
              <rect
                className="lifetime-column"
                data-now={index === step - 1}
                x={LABEL_WIDTH + index * COLUMN_WIDTH}
                y={TOP - 4}
                width={COLUMN_WIDTH}
                height={lifetimes.length * ROW_HEIGHT + 6}
              />
              <text
                className="lifetime-tick"
                x={LABEL_WIDTH + index * COLUMN_WIDTH + COLUMN_WIDTH / 2}
                y={TOP - 10}
                textAnchor="middle"
              >
                {index + 1}
              </text>
            </g>
          ))}
          {lifetimes.map((lifetime, row) => {
            const state = getLifetimeState(lifetime, step);
            const ownership =
              lifetime.bytesKind === undefined && lifetime.ownership === 'imported'
                ? 'unknown'
                : lifetime.ownership;
            const y = TOP + row * ROW_HEIGHT;
            const x = LABEL_WIDTH + lifetime.first * COLUMN_WIDTH;
            const fullWidth = (lifetime.last - lifetime.first + 1) * COLUMN_WIDTH;
            const filledWidth =
              Math.max(0, Math.min(step - lifetime.first, lifetime.last - lifetime.first + 1)) *
              COLUMN_WIDTH;
            return (
              <g key={lifetime.id} className="lifetime-row" data-state={state}>
                <title>{`${lifetime.id}: ${formatResourceBytes(lifetime.bytes, lifetime.bytesKind)}, ownership ${ownership}, steps ${lifetime.first + 1} to ${lifetime.last + 1}`}</title>
                <text className="lifetime-name" x={LABEL_WIDTH - 8} y={y + 11} textAnchor="end">
                  {shortenName(lifetime.id, slug)}
                </text>
                <rect
                  className="lifetime-ghost"
                  x={x + 2}
                  y={y + 2}
                  width={fullWidth - 4}
                  height={ROW_HEIGHT - 6}
                  rx={3}
                />
                <rect
                  className="lifetime-bar"
                  data-ownership={ownership}
                  x={x + 2}
                  y={y + 2}
                  width={Math.max(0, filledWidth - 4)}
                  height={ROW_HEIGHT - 6}
                  rx={3}
                />
              </g>
            );
          })}
          <line
            className="lifetime-playhead"
            x1={playheadX}
            x2={playheadX}
            y1={TOP - 6}
            y2={TOP + lifetimes.length * ROW_HEIGHT + 2}
          />
        </svg>
      </div>
      <p className="meta-scroll-hint">Scroll sideways to follow every step.</p>
    </>
  );
}
