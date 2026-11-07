// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A table as GPU vectors: columns of chunks, one buffer per chunk, uploaded batch by batch.

import {useMemo} from 'react';
import {GpuCanvas} from '../../gpu/gpu-canvas';
import {sampleValueRamp} from '../../lens/ramp';
import {
  createColumnsSceneLoader,
  type ColumnCell,
  type ColumnsLayout
} from './scenes/columns-scene';
import './meta.css';

type ColumnSpec = {name: string; format: string; bytes: string; variable?: boolean};

const COLUMNS: readonly ColumnSpec[] = [
  {name: 'x', format: 'float32', bytes: '4 B / row'},
  {name: 'y', format: 'float32', bytes: '4 B / row'},
  {name: 'value', format: 'float32', bytes: '4 B / row'},
  {name: 'color', format: 'unorm8x4', bytes: '4 B / row'},
  {name: 'path', format: 'vertex-list<float32x2>', bytes: '8 B / vertex', variable: true}
];
const BATCHES = [7, 5, 6];

const LEFT = 78;
const COLUMN_WIDTH = 128;
const COLUMN_GAP = 22;
const TOP = 58;
const CELL_HEIGHT = 15;
const CELL_GAP = 2;
const CHUNK_GAP = 16;
const CHUNK_PADDING = 4;

// Integer hash, so the server and the browser agree to the last bit (Math.sin may differ).
function pseudoRandom(seed: number): number {
  let state = Math.imul(seed + 1, 0x9e3779b1) >>> 0;
  state = Math.imul(state ^ (state >>> 15), 0x85ebca6b) >>> 0;
  state = Math.imul(state ^ (state >>> 13), 0xc2b2ae35) >>> 0;
  return ((state ^ (state >>> 16)) >>> 0) / 4294967296;
}

function makeLayout(): {layout: ColumnsLayout; frames: Frame[]} {
  const cells: ColumnCell[] = [];
  const frames: Frame[] = [];
  let top = TOP;
  let row = 0;
  BATCHES.forEach((rowCount, batch) => {
    const chunkHeight = rowCount * (CELL_HEIGHT + CELL_GAP) - CELL_GAP;
    COLUMNS.forEach((column, columnIndex) => {
      const left = LEFT + columnIndex * (COLUMN_WIDTH + COLUMN_GAP);
      frames.push({
        x: left - CHUNK_PADDING,
        y: top - CHUNK_PADDING,
        width: COLUMN_WIDTH + CHUNK_PADDING * 2,
        height: chunkHeight + CHUNK_PADDING * 2,
        batch
      });
      for (let index = 0; index < rowCount; index++) {
        const seed = batch * 97 + columnIndex * 31 + index;
        const fraction = column.variable ? (1 + Math.floor(pseudoRandom(seed) * 4)) / 4 : 1;
        cells.push({
          x: left,
          y: top + index * (CELL_HEIGHT + CELL_GAP),
          width: COLUMN_WIDTH * fraction,
          height: CELL_HEIGHT,
          value: 0.12 + 0.88 * pseudoRandom(seed + 5),
          chunkOrder: batch * COLUMNS.length + columnIndex,
          row: row + index
        });
      }
    });
    top += chunkHeight + CHUNK_GAP + CHUNK_PADDING * 2;
    row += rowCount;
  });
  const width = LEFT + COLUMNS.length * COLUMN_WIDTH + (COLUMNS.length - 1) * COLUMN_GAP + 22;
  return {layout: {width, height: top - CHUNK_GAP + 6, rowCount: row, cells}, frames};
}

type Frame = {x: number; y: number; width: number; height: number; batch: number};

const MODEL = makeLayout();

function ColumnsFallback({layout}: {layout: ColumnsLayout}) {
  return (
    <svg
      viewBox={`0 0 ${layout.width} ${layout.height}`}
      style={{width: '100%', height: '100%'}}
      aria-hidden="true"
    >
      {layout.cells.map((cell, index) => {
        const [red, green, blue] = sampleValueRamp(cell.value);
        return (
          <rect
            key={index}
            x={cell.x}
            y={cell.y}
            width={cell.width}
            height={cell.height}
            rx={2}
            fill={`rgb(${Math.round(red * 255)} ${Math.round(green * 255)} ${Math.round(blue * 255)})`}
            fillOpacity={0.5 + 0.5 * cell.value}
          />
        );
      })}
    </svg>
  );
}

export function ColumnsDemo() {
  const {layout, frames} = MODEL;
  const loader = useMemo(() => createColumnsSceneLoader(layout), [layout]);
  return (
    <div className="meta-canvas">
      <div className="meta-scroll">
        <div
          className="meta-stage"
          style={{minWidth: 640, aspectRatio: `${layout.width} / ${layout.height}`}}
        >
          <div className="meta-stage__canvas">
            <GpuCanvas
              scene={loader}
              fallback={<ColumnsFallback layout={layout} />}
              seed={1}
              label="A table of five columns in three batches, uploaded as chunked GPU vectors"
              aspectRatio={layout.width / layout.height}
            />
          </div>
          <svg
            className="meta-stage__svg"
            viewBox={`0 0 ${layout.width} ${layout.height}`}
            aria-hidden="true"
            style={{pointerEvents: 'none'}}
          >
            {COLUMNS.map((column, index) => (
              <g
                key={column.name}
                transform={`translate(${LEFT + index * (COLUMN_WIDTH + COLUMN_GAP)} 0)`}
              >
                <text className="column-label" x={0} y={24}>
                  {column.name}
                </text>
                <text className="column-format" x={0} y={40}>
                  {column.format}
                </text>
              </g>
            ))}
            {frames.map((frame, index) => (
              <rect
                key={index}
                className="chunk-frame"
                x={frame.x}
                y={frame.y}
                width={frame.width}
                height={frame.height}
                rx={5}
              />
            ))}
            {BATCHES.map((_, batch) => {
              const frame = frames.find(candidate => candidate.batch === batch);
              return frame ? (
                <text key={batch} className="chunk-label" x={8} y={frame.y + 14}>
                  batch {batch}
                </text>
              ) : null;
            })}
          </svg>
        </div>
      </div>
    </div>
  );
}
