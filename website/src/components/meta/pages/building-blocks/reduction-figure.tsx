// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A sum reduction over the recorded time-window mask: 16 slices of the 1,500 rows, then pairwise
// sums level by level until one value is left. The root equals the recorded `count`.

import {useState} from 'react';
import {formatCount} from '../../../../lens/format';
import {ALL_STAGES, isLit, Stepper} from './stepper';
import {RECORDED, REDUCTION_LEVELS} from './toy-data';

const ROW = 24;
const TOP = 30;
const LEFT = 96;
const COLUMN = 66;
const NODE_WIDTH = 42;
const NODE_HEIGHT = 18;

function getNodeY(level: number, index: number): number {
  const span = 2 ** level;
  return TOP + (index * span + (span - 1) / 2) * ROW;
}

export function ReductionFigure() {
  const [active, setActive] = useState(ALL_STAGES);
  const levels = REDUCTION_LEVELS;
  const total = levels[levels.length - 1][0];
  const leaves = RECORDED.reductionLeaves;
  const height = TOP + leaves.length * ROW + 4;
  const width = LEFT + (levels.length - 1) * COLUMN + NODE_WIDTH + 6;
  const stages = levels.slice(1).map((level, index) => ({
    id: `level-${index + 1}`,
    title: `${levels[index].length} → ${level.length}`
  }));
  const readout =
    active === ALL_STAGES
      ? `${levels.length - 1} steps turn ${leaves.length} partial counts into one. Each step halves the values; no invocation adds more than two.`
      : `Step ${active.slice(6)}: ${levels[Number(active.slice(6))].length} invocations each add two values from the step before.`;
  return (
    <div className="meta-panel">
      <Stepper
        label="Reduction steps"
        stages={stages}
        active={active}
        onActive={setActive}
        numbered={false}
      />
      <div className="building-blocks-figure building-blocks-figure--narrow">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="building-blocks-svg"
          role="img"
          aria-label={`A reduction tree: ${leaves.length} partial counts of kept rows are added in pairs over ${levels.length - 1} steps to ${total}, the recorded count.`}
        >
          <text x={0} y={14} className="building-blocks-text building-blocks-text--muted">
            rows
          </text>
          {levels.map((level, levelIndex) => (
            <text
              key={`head-${level.length}`}
              x={LEFT + levelIndex * COLUMN + NODE_WIDTH / 2}
              y={14}
              textAnchor="middle"
              className="building-blocks-text building-blocks-text--muted"
            >
              {level.length}
            </text>
          ))}
          {leaves.map((leaf, index) => (
            <text
              key={`range-${leaf.begin}`}
              x={0}
              y={getNodeY(0, index) + 4}
              className="building-blocks-text building-blocks-text--muted"
            >
              {leaf.begin}–{leaf.end - 1}
            </text>
          ))}
          {levels.slice(1).map((level, parentLevelIndex) =>
            level.map((_, index) => {
              const levelIndex = parentLevelIndex + 1;
              const lit = isLit(active, `level-${levelIndex}`);
              const x = LEFT + levelIndex * COLUMN;
              const y = getNodeY(levelIndex, index);
              return [0, 1].map(child => {
                const childIndex = index * 2 + child;
                if (childIndex >= levels[levelIndex - 1].length) return null;
                const childY = getNodeY(levelIndex - 1, childIndex);
                const childX = x - COLUMN + NODE_WIDTH;
                return (
                  <path
                    key={`edge-${levelIndex}-${childIndex}`}
                    d={`M${childX} ${childY} C${childX + 14} ${childY} ${x - 14} ${y} ${x} ${y}`}
                    className="building-blocks-edge"
                    data-lit={active !== ALL_STAGES && lit ? 'true' : undefined}
                    data-dim={!lit ? 'true' : undefined}
                  />
                );
              });
            })
          )}
          {levels.map((level, levelIndex) =>
            level.map((value, index) => {
              const x = LEFT + levelIndex * COLUMN;
              const y = getNodeY(levelIndex, index);
              const lit = levelIndex === 0 || isLit(active, `level-${levelIndex}`);
              const root = levelIndex === levels.length - 1;
              return (
                <g
                  key={`node-${levelIndex}-${index}`}
                  data-dim={!lit ? 'true' : undefined}
                  className="building-blocks-node-group"
                >
                  <rect
                    x={x}
                    y={y - NODE_HEIGHT / 2}
                    width={NODE_WIDTH}
                    height={NODE_HEIGHT}
                    rx={3}
                    className="building-blocks-cell"
                    data-on={
                      levelIndex > 0 && active !== ALL_STAGES && active === `level-${levelIndex}`
                        ? 'true'
                        : undefined
                    }
                    data-root={root ? 'true' : undefined}
                  />
                  <text
                    x={x + NODE_WIDTH / 2}
                    y={y + 4}
                    textAnchor="middle"
                    className="building-blocks-text"
                  >
                    {value}
                  </text>
                </g>
              );
            })
          )}
        </svg>
      </div>
      <p className="meta-readout building-blocks-readout">
        {readout}{' '}
        {total === RECORDED.timeWindow.keptCount ? (
          <>
            The root, <b>{formatCount(total)}</b>, equals the run's recorded <code>count</code>.
          </>
        ) : (
          <>
            The root is <b>{formatCount(total)}</b>; the recorded <code>count</code> is{' '}
            {formatCount(RECORDED.timeWindow.keptCount)}.
          </>
        )}
      </p>
    </div>
  );
}
