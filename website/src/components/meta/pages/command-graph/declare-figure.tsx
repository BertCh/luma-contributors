// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The kernels of one uncollapsed recorded run in schedule order, each with the buffers it reads and
// writes, and the read-after-write edges derived from those sets alone. Selecting a kernel lights
// the buffers it reads and the earlier kernels that wrote them.

import {useMemo, useState} from 'react';
import type {LensViewModel} from '../../../../lens/types';
import {getHazardInstances, getReadSources} from './hazards';

const ROW_HEIGHT = 40;
const TOP = 30;
const GUTTER = 96;
const KERNEL_X = GUTTER + 10;
const KERNEL_WIDTH = 262;
const CHIP_HEIGHT = 22;
const CHIP_GAP = 6;
const CHARACTER_WIDTH = 6.7;
const COLUMN_GAP = 26;

/** Strips the run's own prefixes so names fit: `time-window-filter-visibility-compact-flags` → `flags`. */
export function shortenName(name: string, title: string): string {
  let short = name;
  for (const prefix of [`gpu-dataframe-${title}-`, `${title}-`, 'visibility-']) {
    if (short.startsWith(prefix)) short = short.slice(prefix.length);
  }
  // Scan scratch buffers keep only their role: `compact-scan-level-0-block-sums` → `block-sums`.
  return short.replace(/^compact-scan-level-0-(block-)/, '$1');
}

function getChipWidth(label: string): number {
  return Math.round(label.length * CHARACTER_WIDTH + 16);
}

function getChipsWidth(labels: readonly string[]): number {
  return labels.reduce((total, label) => total + getChipWidth(label) + CHIP_GAP, 0) - CHIP_GAP;
}

/** The kernel whose reads come from the most distinct earlier kernels: the best default. */
export function getMostConnectedKernel(model: LensViewModel): number {
  let best = 0;
  let bestCount = -1;
  model.nodes.forEach((node, index) => {
    if (node.kind !== 'compute') return;
    const writers = new Set(
      getReadSources(model, index)
        .map(source => source.writer)
        .filter(writer => writer !== undefined)
    );
    if (writers.size > bestCount) {
      best = index;
      bestCount = writers.size;
    }
  });
  return best;
}

type ChipState = 'lit' | 'outside' | undefined;

export function DeclareFigure({model}: {model: LensViewModel}) {
  const [selected, setSelected] = useState(() => getMostConnectedKernel(model));
  const instances = useMemo(() => getHazardInstances(model), [model]);
  const sources = useMemo(() => getReadSources(model, selected), [model, selected]);
  const title = model.title;

  const rows = model.nodes.map(node => ({
    node,
    label: shortenName(node.id, title),
    reads: (node.reads ?? []).map(buffer => ({buffer, label: shortenName(buffer, title)})),
    writes: (node.writes ?? []).map(buffer => ({buffer, label: shortenName(buffer, title)}))
  }));
  const readsWidth = Math.max(
    60,
    ...rows.map(row => getChipsWidth(row.reads.map(read => read.label)))
  );
  const writesWidth = Math.max(
    60,
    ...rows.map(row => getChipsWidth(row.writes.map(write => write.label)))
  );
  const readsX = KERNEL_X + KERNEL_WIDTH + COLUMN_GAP;
  const writesX = readsX + readsWidth + COLUMN_GAP;
  const width = writesX + writesWidth + 12;
  const height = TOP + rows.length * ROW_HEIGHT + 6;
  const rowCenter = (index: number) => TOP + index * ROW_HEIGHT + ROW_HEIGHT / 2;

  // Writer row index → buffers of the selected kernel's reads it supplied.
  const suppliedBy = new Map<number, Set<string>>();
  for (const source of sources) {
    if (source.writer === undefined) continue;
    const buffers = suppliedBy.get(source.writer) ?? new Set<string>();
    buffers.add(source.buffer);
    suppliedBy.set(source.writer, buffers);
  }
  const fromInside = sources.filter(source => source.writer !== undefined);
  const fromOutside = sources.filter(source => source.writer === undefined);
  const selectedRow = rows[selected];

  const renderChips = (
    chips: readonly {buffer: string; label: string}[],
    x: number,
    y: number,
    getState: (buffer: string) => ChipState
  ) => {
    let cursor = x;
    return chips.map(chip => {
      const chipWidth = getChipWidth(chip.label);
      const chipX = cursor;
      cursor += chipWidth + CHIP_GAP;
      return (
        <g
          key={chip.buffer}
          className="command-graph-chip"
          data-state={getState(chip.buffer)}
          transform={`translate(${chipX} ${y - CHIP_HEIGHT / 2})`}
        >
          <title>{chip.buffer}</title>
          <rect width={chipWidth} height={CHIP_HEIGHT} rx={CHIP_HEIGHT / 2} />
          <text x={chipWidth / 2} y={15} textAnchor="middle">
            {chip.label}
          </text>
        </g>
      );
    });
  };

  return (
    <div className="meta-panel">
      <div className="meta-panel__bar">
        <span className="meta-panel__title">{model.title}: declared reads and writes</span>
        <ul className="meta-stats">
          <li>
            <strong>{model.nodes.length}</strong> kernels
          </li>
          <li>
            <strong>{instances.length}</strong> derived edges
          </li>
          <li>
            <strong>0</strong> written by hand
          </li>
        </ul>
      </div>
      <div className="command-graph-picker" role="group" aria-label="Select a kernel">
        {rows.map((row, index) => (
          <button
            key={row.node.id}
            type="button"
            aria-pressed={index === selected}
            onClick={() => setSelected(index)}
          >
            <span aria-hidden="true">{index + 1}</span> {row.label}
          </button>
        ))}
      </div>
      <div className="meta-scroll">
        <svg
          className="command-graph-declare"
          viewBox={`0 0 ${width} ${height}`}
          style={{minWidth: Math.round(width * 0.9)}}
          role="img"
          aria-label={`${model.nodes.length} kernels in schedule order with their read and write buffers. ${selectedRow.label} reads ${sources.length} buffers; ${fromInside.length} were written by earlier kernels, so ${instances.filter(instance => instance.to === selected).length} edges lead into it.`}
        >
          <text className="command-graph-head" x={KERNEL_X} y={16}>
            kernel, in schedule order
          </text>
          <text className="command-graph-head" x={readsX} y={16}>
            reads
          </text>
          <text className="command-graph-head" x={writesX} y={16}>
            writes
          </text>
          <g>
            {instances.map(instance => {
              const y0 = rowCenter(instance.from);
              const y1 = rowCenter(instance.to);
              const span = instance.to - instance.from;
              const bend = GUTTER - Math.min(GUTTER - 8, 12 + span * 10);
              const isLit = instance.to === selected;
              return (
                <path
                  key={`${instance.from}>${instance.to}:${instance.buffer}`}
                  className="command-graph-edge"
                  data-lit={isLit ? 'true' : undefined}
                  d={`M${KERNEL_X} ${y0 + 4} C${bend} ${y0 + 4} ${bend} ${y1 - 4} ${KERNEL_X} ${y1 - 4}`}
                >
                  <title>{`${rows[instance.to].label} reads ${shortenName(instance.buffer, title)}, written by ${rows[instance.from].label}`}</title>
                </path>
              );
            })}
            {instances
              .filter(instance => instance.to === selected)
              .map(instance => {
                const y1 = rowCenter(instance.to) - 4;
                return (
                  <path
                    key={`head:${instance.from}:${instance.buffer}`}
                    className="command-graph-arrow"
                    d={`M${KERNEL_X} ${y1} l-7 -4 l0 8 z`}
                  />
                );
              })}
          </g>
          {rows.map((row, index) => {
            const y = rowCenter(index);
            const supplied = suppliedBy.get(index);
            const rowState = index === selected ? 'selected' : supplied ? 'source' : undefined;
            return (
              <g key={row.node.id}>
                <g
                  className="command-graph-kernel"
                  data-kind={row.node.kind}
                  data-state={rowState}
                  transform={`translate(${KERNEL_X} ${y - 14})`}
                  onClick={() => setSelected(index)}
                >
                  <title>{row.node.id}</title>
                  <rect width={KERNEL_WIDTH} height={28} rx={6} />
                  <text className="command-graph-kernel__index" x={10} y={18}>
                    {index + 1}
                  </text>
                  <text className="command-graph-kernel__label" x={28} y={18}>
                    {row.label.length > 33 ? `${row.label.slice(0, 32)}…` : row.label}
                  </text>
                </g>
                {row.reads.length === 0 ? (
                  <text className="command-graph-none" x={readsX} y={y + 4}>
                    nothing
                  </text>
                ) : (
                  renderChips(row.reads, readsX, y, buffer =>
                    index !== selected
                      ? undefined
                      : sources.find(source => source.buffer === buffer)?.writer === undefined
                        ? 'outside'
                        : 'lit'
                  )
                )}
                {row.writes.length === 0 ? (
                  <text className="command-graph-none" x={writesX} y={y + 4}>
                    nothing
                  </text>
                ) : (
                  renderChips(row.writes, writesX, y, buffer =>
                    supplied?.has(buffer) ? 'lit' : undefined
                  )
                )}
              </g>
            );
          })}
        </svg>
      </div>
      <p className="meta-scroll-hint">Scroll sideways to see every buffer.</p>
      <p className="meta-detail" aria-live="polite">
        <span>
          <b>{selectedRow.label}</b> reads {sources.length}{' '}
          {sources.length === 1 ? 'buffer' : 'buffers'}.{' '}
          {fromInside.length > 0 ? (
            <>
              {fromInside.length === 1 ? 'One was' : `${fromInside.length} were`} written earlier:{' '}
              {fromInside.map((source, index) => (
                <span key={source.buffer}>
                  {index > 0 ? ', ' : ''}
                  <b>{shortenName(source.buffer, title)}</b> by {rows[source.writer ?? 0].label}
                </span>
              ))}
              .{' '}
            </>
          ) : (
            'No earlier kernel wrote any of them. '
          )}
          {fromOutside.length > 0
            ? `${fromOutside.map(source => shortenName(source.buffer, title)).join(', ')} ${fromOutside.length === 1 ? 'comes' : 'come'} from outside the graph.`
            : null}
        </span>
      </p>
    </div>
  );
}
