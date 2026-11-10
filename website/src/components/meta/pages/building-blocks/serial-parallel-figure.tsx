// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A serial loop carries a running count from row to row; a kernel's invocations each see one row
// and cannot know where to write. Same twenty recorded rows as the compaction figure.

import {COMPACTION_ROWS} from './toy-data';

const LEFT = 78;
const CELL = 17;
const WIDTH = LEFT + COMPACTION_ROWS.length * CELL + 8;

export function SerialParallelFigure() {
  const kept = COMPACTION_ROWS.filter(row => row.flag).length;
  return (
    <div className="meta-canvas building-blocks-figure meta-scroll">
      <svg
        viewBox={`0 0 ${WIDTH} 196`}
        className="building-blocks-svg building-blocks-svg--wide"
        style={{minWidth: 480}}
        role="img"
        aria-label={`A loop keeps a running count n, so each kept row knows its output slot. A kernel's ${COMPACTION_ROWS.length} invocations each see only their own row, so the ${kept} kept rows cannot know where to write.`}
      >
        <defs>
          <marker
            id="building-blocks-arrow"
            viewBox="0 0 6 6"
            refX="5"
            refY="3"
            markerWidth="5"
            markerHeight="5"
            orient="auto"
          >
            <path d="M0,0 L6,3 L0,6 z" className="building-blocks-arrowhead" />
          </marker>
        </defs>
        <text x={0} y={22} className="building-blocks-text building-blocks-text--strong">
          loop
        </text>
        <text x={0} y={36} className="building-blocks-text building-blocks-text--muted">
          one thread
        </text>
        <text x={0} y={70} className="building-blocks-text building-blocks-text--muted">
          n before
        </text>
        {COMPACTION_ROWS.map((row, index) => {
          const x = LEFT + index * CELL;
          return (
            <g key={`loop-${row.row}`}>
              <rect
                x={x + 1}
                y={10}
                width={CELL - 2}
                height={CELL + 6}
                rx={2}
                className="building-blocks-cell"
                data-on={row.flag ? 'true' : undefined}
                data-kind="cpu"
              />
              <text x={x + CELL / 2} y={26} textAnchor="middle" className="building-blocks-text">
                {row.flag}
              </text>
              <text
                x={x + CELL / 2}
                y={70}
                textAnchor="middle"
                className="building-blocks-text building-blocks-text--cpu"
              >
                {row.position}
              </text>
              {index < COMPACTION_ROWS.length - 1 ? (
                <path
                  d={`M${x + CELL / 2 + 4} 52 Q${x + CELL} 44 ${x + CELL * 1.5 - 4} 52`}
                  className="building-blocks-carry"
                  markerEnd="url(#building-blocks-arrow)"
                />
              ) : null}
            </g>
          );
        })}
        <text x={0} y={118} className="building-blocks-text building-blocks-text--strong">
          kernel
        </text>
        <text x={0} y={132} className="building-blocks-text building-blocks-text--muted">
          one row each
        </text>
        <text x={0} y={174} className="building-blocks-text building-blocks-text--muted">
          write at
        </text>
        {COMPACTION_ROWS.map((row, index) => {
          const x = LEFT + index * CELL;
          return (
            <g key={`kernel-${row.row}`}>
              <rect
                x={x + 1}
                y={106}
                width={CELL - 2}
                height={CELL + 6}
                rx={2}
                className="building-blocks-cell"
                data-on={row.flag ? 'true' : undefined}
              />
              <text x={x + CELL / 2} y={122} textAnchor="middle" className="building-blocks-text">
                {row.flag}
              </text>
              <line
                x1={x + CELL / 2}
                x2={x + CELL / 2}
                y1={136}
                y2={152}
                className="building-blocks-lane"
              />
              <text
                x={x + CELL / 2}
                y={174}
                textAnchor="middle"
                className="building-blocks-text building-blocks-text--signal"
              >
                {row.flag ? '?' : ''}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
