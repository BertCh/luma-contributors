// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A grid index on sixty seeded points: count per cell, scan to cell offsets, scatter the ids into
// cell order, then one query cell gathers only from its neighbourhood, one contiguous range of the
// sorted ids per grid row. Beside it, the recorded kernels of kernel density that build the same.

import {type KeyboardEvent, useState} from 'react';
import {formatCount} from '../../../../lens/format';
import type {LensViewModel} from '../../../../lens/types';
import {ALL_STAGES, isLit, KernelStrip, Stepper, type StripKernel} from './stepper';
import {
  GRID_CELL_COUNT,
  GRID_COUNTS,
  GRID_OFFSETS,
  GRID_POINTS,
  GRID_SIZE,
  GRID_SORTED,
  getQueryRanges
} from './toy-data';

const STAGES = [
  {id: 'count', title: 'count'},
  {id: 'scan', title: 'scan'},
  {id: 'scatter', title: 'scatter'},
  {id: 'gather', title: 'gather'}
] as const;

const GRID_PIXELS = 260;
const CELL_PIXELS = GRID_PIXELS / GRID_SIZE;
const STRIP_LEFT = 62;
const STRIP_WIDTH = 600;
const STRIP_CELL = STRIP_WIDTH / (GRID_CELL_COUNT + 1);
const POINT_WIDTH = STRIP_WIDTH / GRID_POINTS.length;

function getKernelStage(nodeId: string): string[] {
  if (/index-count/.test(nodeId)) return ['count'];
  if (/index-(scan|finalize|terminal)/.test(nodeId)) return ['scan'];
  if (/index-(rank|scatter)|sort-cells/.test(nodeId)) return ['scatter'];
  if (nodeId.endsWith('-gather')) return ['gather'];
  return [];
}

export function getKernelDensityKernels(model: LensViewModel): StripKernel[] {
  return model.nodes.map(node => ({
    id: node.id,
    label: node.id
      .replace(/^gpu-spatial-analysis-kernel-density-/, '')
      .replace(/^kernel-density-/, ''),
    stages: getKernelStage(node.id),
    kind: node.id.endsWith('snapshot-tap') ? 'readback' : 'kernel',
    detail: node.invocations
      ? formatCount(node.invocations)
      : node.id.endsWith('snapshot-tap')
        ? 'read back'
        : undefined
  }));
}

export function GridIndexFigure({model}: {model: LensViewModel}) {
  const [active, setActive] = useState<string>(ALL_STAGES);
  const [query, setQuery] = useState({column: 1, row: 1});
  const ranges = getQueryRanges(query.column, query.row);
  const visitedIds = new Set<number>();
  for (const range of ranges) {
    for (let index = range.begin; index < range.end; index++) visitedIds.add(GRID_SORTED[index].id);
  }
  const visitedCells = ranges.reduce((total, range) => total + range.last - range.first + 1, 0);
  const inNeighbourhood = (cell: number) => {
    const column = cell % GRID_SIZE;
    const row = Math.floor(cell / GRID_SIZE);
    return Math.abs(column - query.column) <= 1 && Math.abs(row - query.row) <= 1;
  };
  const move = (columnOffset: number, rowOffset: number) =>
    setQuery(current => ({
      column: Math.min(GRID_SIZE - 1, Math.max(0, current.column + columnOffset)),
      row: Math.min(GRID_SIZE - 1, Math.max(0, current.row + rowOffset))
    }));
  const onKeyDown = (event: KeyboardEvent) => {
    const offsets: Record<string, [number, number]> = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, -1],
      ArrowDown: [0, 1]
    };
    const offset = offsets[event.key];
    if (offset) {
      event.preventDefault();
      move(...offset);
    }
  };
  const lit = {
    count: isLit(active, 'count'),
    scan: isLit(active, 'scan'),
    scatter: isLit(active, 'scatter'),
    gather: isLit(active, 'gather')
  };
  const kernels = getKernelDensityKernels(model);

  return (
    <div className="meta-panel">
      <Stepper label="Grid index stages" stages={STAGES} active={active} onActive={setActive} />
      <div className="building-blocks-grid">
        <div className="building-blocks-grid__map">
          <svg
            viewBox={`-2 -2 ${GRID_PIXELS + 4} ${GRID_PIXELS + 4}`}
            className="building-blocks-svg building-blocks-svg--map"
            role="img"
            aria-label={`Sixty points on a ${GRID_SIZE} by ${GRID_SIZE} grid with a count in each cell. The query cell at column ${query.column + 1}, row ${query.row + 1} reads ${visitedIds.size} points from ${visitedCells} cells instead of all ${GRID_POINTS.length}.`}
          >
            {GRID_COUNTS.map((count, cell) => {
              const column = cell % GRID_SIZE;
              const row = Math.floor(cell / GRID_SIZE);
              const isQuery = column === query.column && row === query.row;
              return (
                <g
                  key={cell}
                  onClick={() => setQuery({column, row})}
                  className="building-blocks-grid-cell"
                >
                  <rect
                    x={column * CELL_PIXELS}
                    y={row * CELL_PIXELS}
                    width={CELL_PIXELS}
                    height={CELL_PIXELS}
                    className="building-blocks-map-cell"
                    data-near={lit.gather && inNeighbourhood(cell) ? 'true' : undefined}
                    data-query={lit.gather && isQuery ? 'true' : undefined}
                  />
                  <text
                    x={column * CELL_PIXELS + 5}
                    y={row * CELL_PIXELS + 14}
                    className="building-blocks-text building-blocks-text--count"
                    data-dim={!lit.count ? 'true' : undefined}
                  >
                    {count}
                  </text>
                </g>
              );
            })}
            {GRID_POINTS.map(point => (
              <circle
                key={point.id}
                cx={point.x * GRID_PIXELS}
                cy={point.y * GRID_PIXELS}
                r={3.6}
                className="building-blocks-point"
                data-visited={lit.gather && visitedIds.has(point.id) ? 'true' : undefined}
              />
            ))}
          </svg>
          <div
            className="building-blocks-grid__controls"
            role="group"
            aria-label="Move the query cell"
            onKeyDown={onKeyDown}
          >
            <button
              type="button"
              className="meta-play"
              onClick={() => move(-1, 0)}
              aria-label="Query cell left"
            >
              ←
            </button>
            <button
              type="button"
              className="meta-play"
              onClick={() => move(0, -1)}
              aria-label="Query cell up"
            >
              ↑
            </button>
            <button
              type="button"
              className="meta-play"
              onClick={() => move(0, 1)}
              aria-label="Query cell down"
            >
              ↓
            </button>
            <button
              type="button"
              className="meta-play"
              onClick={() => move(1, 0)}
              aria-label="Query cell right"
            >
              →
            </button>
          </div>
          <p className="meta-readout building-blocks-readout" aria-live="polite">
            Query cell ({query.column + 1}, {query.row + 1}) visits <b>{visitedIds.size}</b> of{' '}
            {GRID_POINTS.length} points in {visitedCells} of {GRID_CELL_COUNT} cells, as{' '}
            {ranges.length} contiguous ranges of the sorted ids.
          </p>
        </div>
        <div className="building-blocks-grid__strips meta-scroll">
          <svg
            viewBox={`0 0 ${STRIP_LEFT + STRIP_WIDTH + 4} 172`}
            className="building-blocks-svg building-blocks-svg--strips"
            style={{minWidth: 560}}
            role="img"
            aria-label="Cell counts, their exclusive scan as cell offsets, and the point ids sorted by cell. Each offset points to where its cell's run of ids begins."
          >
            <text x={0} y={18} className="building-blocks-text building-blocks-text--muted">
              cell
            </text>
            <text x={0} y={40} className="building-blocks-text building-blocks-text--muted">
              count
            </text>
            <text x={0} y={68} className="building-blocks-text building-blocks-text--muted">
              offset
            </text>
            <text x={0} y={150} className="building-blocks-text building-blocks-text--muted">
              ids
            </text>
            {GRID_OFFSETS.map((offset, cell) => {
              const x = STRIP_LEFT + cell * STRIP_CELL;
              const terminal = cell === GRID_CELL_COUNT;
              const near = !terminal && lit.gather && inNeighbourhood(cell);
              return (
                <g key={cell}>
                  {!terminal ? (
                    <>
                      <text
                        x={x + STRIP_CELL / 2}
                        y={18}
                        textAnchor="middle"
                        className="building-blocks-text building-blocks-text--tiny"
                      >
                        {cell}
                      </text>
                      <g
                        data-dim={!lit.count ? 'true' : undefined}
                        className="building-blocks-node-group"
                      >
                        <rect
                          x={x + 1}
                          y={27}
                          width={STRIP_CELL - 2}
                          height={18}
                          rx={2}
                          className="building-blocks-cell"
                          data-on={active === 'count' || near ? 'true' : undefined}
                        />
                        <text
                          x={x + STRIP_CELL / 2}
                          y={40}
                          textAnchor="middle"
                          className="building-blocks-text building-blocks-text--small"
                        >
                          {GRID_COUNTS[cell]}
                        </text>
                      </g>
                    </>
                  ) : null}
                  <g
                    data-dim={!lit.scan ? 'true' : undefined}
                    className="building-blocks-node-group"
                  >
                    <rect
                      x={x + 1}
                      y={55}
                      width={STRIP_CELL - 2}
                      height={18}
                      rx={2}
                      className="building-blocks-cell"
                      data-on={active === 'scan' ? 'true' : undefined}
                      data-root={terminal ? 'true' : undefined}
                    />
                    <text
                      x={x + STRIP_CELL / 2}
                      y={68}
                      textAnchor="middle"
                      className="building-blocks-text building-blocks-text--small"
                    >
                      {offset}
                    </text>
                    <line
                      x1={x + STRIP_CELL / 2}
                      y1={74}
                      x2={STRIP_LEFT + offset * POINT_WIDTH}
                      y2={134}
                      className="building-blocks-edge building-blocks-edge--thin"
                      data-lit={active === 'scan' ? 'true' : undefined}
                    />
                  </g>
                </g>
              );
            })}
            <g
              data-dim={!lit.scatter && !lit.gather ? 'true' : undefined}
              className="building-blocks-node-group"
            >
              {GRID_SORTED.map((point, index) => (
                <rect
                  key={point.id}
                  x={STRIP_LEFT + index * POINT_WIDTH + 0.5}
                  y={136}
                  width={POINT_WIDTH - 1}
                  height={22}
                  className="building-blocks-cell building-blocks-sorted"
                  data-parity={point.cell % 2 ? 'odd' : 'even'}
                  data-on={lit.gather && visitedIds.has(point.id) ? 'true' : undefined}
                >
                  <title>{`point ${point.id}, cell ${point.cell}`}</title>
                </rect>
              ))}
            </g>
            {lit.gather
              ? ranges.map(range => (
                  <path
                    key={range.row}
                    d={`M${STRIP_LEFT + range.begin * POINT_WIDTH} 164 v4 H${STRIP_LEFT + range.end * POINT_WIDTH} v-4`}
                    className="building-blocks-bracket"
                  />
                ))
              : null}
          </svg>
        </div>
      </div>
      <div className="building-blocks-grid__recorded">
        <KernelStrip
          title={
            <>
              Recorded kernel density: <b>{model.recordedNodeCount}</b> kernels
            </>
          }
          kernels={kernels}
          active={active}
        />
      </div>
    </div>
  );
}
