// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Least-significant-digit radix sort of twelve 4-bit keys in two passes of 2-bit digits: per pass a
// digit histogram, its exclusive scan (the first slot of each digit) and a stable scatter.

import {useState} from 'react';
import {ALL_STAGES, Stepper} from './stepper';
import {getDigit, RADIX_BUCKETS, RADIX_DIGIT_BITS, RADIX_KEYS, RADIX_PASSES} from './toy-data';

// GPUSort's radix digit width: RADIX_DIGIT_BITS = 4 in @luma.gl/graph/dist/gpu-core/gpu-sort.js:16.
const RECORDED_DIGIT_BITS = 4;
const ROW = 24;
const TOP = 40;
const KEY_WIDTH = 62;
const COLUMN_X = [44, 194, 344];
const CELL = KEY_WIDTH / RADIX_BUCKETS;

export const RADIX_STAGES = RADIX_PASSES.flatMap((_, passIndex) => [
  {id: `count-${passIndex}`, title: `count ${passIndex + 1}`},
  {id: `scan-${passIndex}`, title: `scan ${passIndex + 1}`},
  {id: `scatter-${passIndex}`, title: `scatter ${passIndex + 1}`}
]);

/** Which of histogram, scan or scatter the stage is, for lighting the recorded kernels. */
export function getRadixRole(stage: string): string {
  return stage.split('-')[0];
}

function KeyText({
  x,
  y,
  keyValue,
  highlightOffset
}: {
  x: number;
  y: number;
  keyValue: number;
  highlightOffset?: number;
}) {
  const digits = [2, 0].map(bitOffset => ({bitOffset, digit: getDigit(keyValue, bitOffset)}));
  return (
    <text x={x} y={y} className="building-blocks-text">
      <tspan className="building-blocks-text--muted">{String(keyValue).padStart(2, ' ')}</tspan>
      <tspan dx={8}> </tspan>
      {digits.map(({bitOffset, digit}) => (
        <tspan
          key={bitOffset}
          dx={4}
          className={bitOffset === highlightOffset ? 'building-blocks-text--digit' : undefined}
        >
          {digit}
        </tspan>
      ))}
    </text>
  );
}

export function RadixFigure({kernelIds}: {kernelIds: readonly string[]}) {
  const [active, setActive] = useState(ALL_STAGES);
  const rows = RADIX_KEYS.length;
  const tableTop = TOP + rows * ROW + 8;
  const height = tableTop + 3 * 18 + 6;
  const width = COLUMN_X[2] + KEY_WIDTH + 20;
  const activePass = active === ALL_STAGES ? -1 : Number(active.split('-')[1]);
  const role = getRadixRole(active);
  const keyCenterY = (slot: number) => TOP + slot * ROW;
  const sorted = RADIX_PASSES[RADIX_PASSES.length - 1].output.map(item => item.key);

  return (
    <div className="meta-panel">
      <Stepper
        label="Radix sort stages"
        stages={RADIX_STAGES}
        active={active}
        onActive={setActive}
      />
      <div className="building-blocks-figure building-blocks-figure--narrow">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="building-blocks-svg"
          role="img"
          aria-label={`Twelve keys written in base 4 are sorted in two passes. Each pass counts the keys per digit, scans the counts into first slots and scatters stably. Result: ${sorted.join(', ')}.`}
        >
          <text x={COLUMN_X[0]} y={14} className="building-blocks-text building-blocks-text--muted">
            keys
          </text>
          <text x={COLUMN_X[0]} y={27} className="building-blocks-text building-blocks-text--muted">
            dec · base 4
          </text>
          {RADIX_PASSES.map((pass, passIndex) => (
            <text
              key={pass.bitOffset}
              x={COLUMN_X[passIndex + 1]}
              y={14}
              className="building-blocks-text building-blocks-text--muted"
              data-dim={activePass >= 0 && activePass !== passIndex ? 'true' : undefined}
            >
              <tspan>pass {passIndex + 1}</tspan>
              <tspan x={COLUMN_X[passIndex + 1]} dy={13}>
                by bits {pass.bitOffset}–{pass.bitOffset + RADIX_DIGIT_BITS - 1}
              </tspan>
            </text>
          ))}
          {RADIX_PASSES.map((pass, passIndex) => {
            const passLit = activePass < 0 || activePass === passIndex;
            const fromX = COLUMN_X[passIndex] + KEY_WIDTH + 4;
            const toX = COLUMN_X[passIndex + 1] - 4;
            return (
              <g key={`pass-${pass.bitOffset}`}>
                {pass.input.map((item, index) => {
                  const fromY = keyCenterY(index);
                  const toY = keyCenterY(pass.targets[index]);
                  return (
                    <path
                      key={`line-${item.source}`}
                      d={`M${fromX} ${fromY} C${fromX + 40} ${fromY} ${toX - 40} ${toY} ${toX} ${toY}`}
                      className="building-blocks-edge"
                      data-lit={activePass === passIndex && role === 'scatter' ? 'true' : undefined}
                      data-dim={
                        !passLit || (activePass === passIndex && role !== 'scatter')
                          ? 'true'
                          : undefined
                      }
                    />
                  );
                })}
              </g>
            );
          })}
          {[RADIX_PASSES[0].input, ...RADIX_PASSES.map(pass => pass.output)].map(
            (items, columnIndex) => {
              // The digit that matters in this column: the one a pass reads (count, scan) or the
              // one the column is now ordered by (after a scatter, and in the overview).
              const readingPass = columnIndex < RADIX_PASSES.length ? columnIndex : -1;
              let highlightOffset: number | undefined;
              if (activePass < 0) {
                highlightOffset = RADIX_PASSES[Math.max(0, columnIndex - 1)].bitOffset;
              } else if (activePass === readingPass) {
                highlightOffset = RADIX_PASSES[activePass].bitOffset;
              } else if (activePass === columnIndex - 1 && role === 'scatter') {
                highlightOffset = RADIX_PASSES[activePass].bitOffset;
              }
              const columnLit =
                activePass < 0 ||
                activePass === readingPass ||
                (activePass === columnIndex - 1 && role === 'scatter');
              return (
                <g
                  key={`column-${columnIndex}`}
                  data-dim={!columnLit ? 'true' : undefined}
                  className="building-blocks-node-group"
                >
                  {items.map((item, slot) => (
                    <g key={`key-${item.source}`}>
                      <rect
                        x={COLUMN_X[columnIndex]}
                        y={keyCenterY(slot) - 10}
                        width={KEY_WIDTH}
                        height={20}
                        rx={3}
                        className="building-blocks-cell"
                      />
                      <KeyText
                        x={COLUMN_X[columnIndex] + 6}
                        y={keyCenterY(slot) + 4}
                        keyValue={item.key}
                        highlightOffset={highlightOffset}
                      />
                    </g>
                  ))}
                </g>
              );
            }
          )}
          {['digit', 'count', 'start'].map((label, rowIndex) => (
            <text
              key={label}
              x={COLUMN_X[0] + KEY_WIDTH}
              y={tableTop + rowIndex * 18 + 13}
              textAnchor="end"
              className="building-blocks-text building-blocks-text--muted"
            >
              {label}
            </text>
          ))}
          {RADIX_PASSES.map((pass, passIndex) => {
            const x = COLUMN_X[passIndex + 1];
            const countLit = activePass < 0 || (activePass === passIndex && role === 'count');
            const scanLit =
              activePass < 0 ||
              (activePass === passIndex && (role === 'scan' || role === 'scatter'));
            return (
              <g key={`table-${pass.bitOffset}`}>
                {pass.histogram.map((count, digit) => (
                  <g key={digit}>
                    <text
                      x={x + digit * CELL + CELL / 2}
                      y={tableTop + 13}
                      textAnchor="middle"
                      className="building-blocks-text building-blocks-text--muted"
                    >
                      {digit}
                    </text>
                    <g
                      data-dim={!countLit ? 'true' : undefined}
                      className="building-blocks-node-group"
                    >
                      <rect
                        x={x + digit * CELL + 1}
                        y={tableTop + 19}
                        width={CELL - 2}
                        height={16}
                        rx={2}
                        className="building-blocks-cell"
                        data-on={activePass === passIndex && role === 'count' ? 'true' : undefined}
                      />
                      <text
                        x={x + digit * CELL + CELL / 2}
                        y={tableTop + 31}
                        textAnchor="middle"
                        className="building-blocks-text"
                      >
                        {count}
                      </text>
                    </g>
                    <g
                      data-dim={!scanLit ? 'true' : undefined}
                      className="building-blocks-node-group"
                    >
                      <rect
                        x={x + digit * CELL + 1}
                        y={tableTop + 37}
                        width={CELL - 2}
                        height={16}
                        rx={2}
                        className="building-blocks-cell"
                        data-on={activePass === passIndex && role === 'scan' ? 'true' : undefined}
                      />
                      <text
                        x={x + digit * CELL + CELL / 2}
                        y={tableTop + 49}
                        textAnchor="middle"
                        className="building-blocks-text"
                      >
                        {pass.starts[digit]}
                      </text>
                    </g>
                  </g>
                ))}
              </g>
            );
          })}
        </svg>
      </div>
      <div className="building-blocks-radix-side">
        <RecordedSortStrip kernelIds={kernelIds} role={active === ALL_STAGES ? ALL_STAGES : role} />
        <p className="meta-readout building-blocks-readout" aria-live="polite">
          {RADIX_READOUTS[active === ALL_STAGES ? ALL_STAGES : role]}
        </p>
      </div>
    </div>
  );
}

const RADIX_READOUTS: Record<string, string> = {
  [ALL_STAGES]:
    'Each pass orders the keys by one digit and keeps earlier order among equal digits, so after the last pass the keys are sorted.',
  count:
    'Count: one invocation per key adds 1 to the bucket of its digit (on the GPU, per workgroup).',
  scan: 'Scan: an exclusive prefix sum of the counts gives the first slot of each digit.',
  scatter:
    "Scatter: each key writes to its digit's start plus the number of equal digits before it, so equal digits stay in order."
};

const PASS_PARTS = [
  {id: 'count', symbol: '#'},
  {id: 'scan', symbol: 'Σ'},
  {id: 'scatter', symbol: '→'}
] as const;

type SortStripItem =
  | {type: 'pass'; id: string; word: string; bitOffset: number}
  | {type: 'kernel'; id: string; label: string};

/** Recorded cell-aggregation kernels with each radix pass (histogram, scan, scatter) as one chip. */
export function getSortStripItems(kernelIds: readonly string[]): SortStripItem[] {
  const items: SortStripItem[] = [];
  for (const id of kernelIds) {
    const pass = id.match(/sort-(low|high)-radix-digit-(\d+)-histogram$/);
    if (pass) {
      items.push({type: 'pass', id, word: pass[1], bitOffset: Number(pass[2])});
    } else if (!/radix-digit-\d+-(scan|scatter)/.test(id)) {
      items.push({
        type: 'kernel',
        id,
        label: id
          .replace(/^gpu-spatial-analysis-cell-aggregation-/, '')
          .replace(/^cell-aggregation-/, '')
          .replace(/^boundaries-geometry-/, '')
      });
    }
  }
  return items;
}

function RecordedSortStrip({kernelIds, role}: {kernelIds: readonly string[]; role: string}) {
  const items = getSortStripItems(kernelIds);
  const passCount = items.filter(item => item.type === 'pass').length;
  return (
    <div className="building-blocks-strip">
      <div className="building-blocks-strip__title">
        Recorded cell aggregation: <b>{kernelIds.length}</b> kernels, <b>{passCount}</b> radix
        passes of {RECORDED_DIGIT_BITS}-bit digits, each count, scan, scatter
      </div>
      <ol className="building-blocks-strip__list">
        {items.map(item =>
          item.type === 'pass' ? (
            <li
              key={item.id}
              className="building-blocks-chip building-blocks-chip--pass"
              title={item.id}
            >
              <span className="building-blocks-chip__label">
                {item.word} {item.bitOffset}–{item.bitOffset + RECORDED_DIGIT_BITS - 1}
              </span>
              <span className="building-blocks-pass">
                {PASS_PARTS.map(part => (
                  <i key={part.id} data-lit={role === part.id ? 'true' : undefined} title={part.id}>
                    {part.symbol}
                  </i>
                ))}
              </span>
            </li>
          ) : (
            <li
              key={item.id}
              className="building-blocks-chip"
              data-kind={item.id.endsWith('snapshot-tap') ? 'readback' : 'kernel'}
              data-dim={role !== ALL_STAGES ? 'true' : undefined}
              title={item.id}
            >
              <span className="building-blocks-chip__label">{item.label}</span>
            </li>
          )
        )}
      </ol>
    </div>
  );
}
