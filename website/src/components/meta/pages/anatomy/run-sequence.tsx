// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Who talks to whom during a run: your code, the command graph (both on the CPU) and the GPU. One row
// per step of the code walk; the selected step is lit. Rows are bracketed by how often they happen.

export type RunStepId = 'buffers' | 'views' | 'build' | 'compile' | 'encode' | 'readback';

type Actor = 'code' | 'graph' | 'gpu';

type SequenceRow = {
  id: RunStepId;
  from: Actor;
  to: Actor;
  label: string;
  /** What the receiving side does with it. */
  effect: string;
};

const ACTOR_X: Record<Actor, number> = {code: 130, graph: 430, gpu: 730};
const ACTOR_TITLES: Record<Actor, [string, string]> = {
  code: ['Your code', 'CPU'],
  graph: ['Command graph', 'CPU'],
  gpu: ['GPU', 'buffers and kernels']
};

const ROWS: readonly SequenceRow[] = [
  {
    id: 'buffers',
    from: 'code',
    to: 'gpu',
    label: 'upload points, allocate the raster',
    effect: 'memory you own'
  },
  {
    id: 'views',
    from: 'code',
    to: 'graph',
    label: 'import buffers, create views',
    effect: 'format and row count'
  },
  {
    id: 'build',
    from: 'code',
    to: 'graph',
    label: 'graph.add(new GPUKernelDensity(...))',
    effect: 'kernels declared'
  },
  {id: 'compile', from: 'graph', to: 'gpu', label: 'compile', effect: 'pipelines created'},
  {id: 'encode', from: 'graph', to: 'gpu', label: 'encode, submit', effect: 'every kernel runs'},
  {id: 'readback', from: 'gpu', to: 'code', label: 'read back', effect: 'the numbers, if you ask'}
];

const WIDTH = 960;
const TOP = 78;
const ROW_HEIGHT = 46;
const HEIGHT = TOP + ROWS.length * ROW_HEIGHT + 18;
const BRACKET_X = 836;

type Cadence = {label: string; first: number; last: number};
const CADENCES: readonly Cadence[] = [
  {label: 'once', first: 0, last: 3},
  {label: 'every frame', first: 4, last: 4},
  {label: 'when needed', first: 5, last: 5}
];

function getRowY(index: number): number {
  return TOP + index * ROW_HEIGHT + ROW_HEIGHT / 2;
}

export function RunSequence({
  active,
  onActive,
  kernelCount
}: {
  active: RunStepId;
  onActive: (id: RunStepId) => void;
  /** Kernels the recorded run declared, shown on the build row. */
  kernelCount?: number;
}) {
  return (
    <div className="meta-canvas">
      <div className="meta-scroll">
        <svg
          className="anatomy-sequence"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          style={{minWidth: 640}}
          role="img"
          aria-label="Sequence of a run: your code uploads buffers to the GPU and describes them to the command graph; the graph compiles once, then encodes and submits every frame; the GPU returns numbers only when asked."
        >
          <defs>
            <marker
              id="anatomy-arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M0 0 L10 5 L0 10 z" fill="context-stroke" />
            </marker>
          </defs>
          {(Object.keys(ACTOR_X) as Actor[]).map(actor => (
            <g key={actor} className="anatomy-sequence__actor" data-actor={actor}>
              <rect x={ACTOR_X[actor] - 92} y={12} width={184} height={46} rx={8} />
              <text className="anatomy-sequence__actor-title" x={ACTOR_X[actor]} y={32}>
                {ACTOR_TITLES[actor][0]}
              </text>
              <text className="anatomy-sequence__actor-meta" x={ACTOR_X[actor]} y={48}>
                {ACTOR_TITLES[actor][1]}
              </text>
              <line x1={ACTOR_X[actor]} y1={58} x2={ACTOR_X[actor]} y2={HEIGHT - 8} />
            </g>
          ))}
          {ROWS.map((row, index) => {
            const y = getRowY(index);
            const from = ACTOR_X[row.from];
            const to = ACTOR_X[row.to];
            const direction = to > from ? 1 : -1;
            const effect =
              row.id === 'build' && kernelCount ? `${kernelCount} kernels declared` : row.effect;
            return (
              <g
                key={row.id}
                className="anatomy-sequence__row"
                data-active={row.id === active}
                data-target={row.to}
                role="button"
                tabIndex={0}
                aria-pressed={row.id === active}
                aria-label={`Step ${index + 1}: ${row.label}`}
                onClick={() => onActive(row.id)}
                onPointerEnter={event => {
                  if (event.pointerType === 'mouse') onActive(row.id);
                }}
                onFocus={() => onActive(row.id)}
                onKeyDown={event => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    onActive(row.id);
                  }
                }}
              >
                <rect
                  className="anatomy-sequence__band"
                  x={16}
                  y={y - ROW_HEIGHT / 2 + 3}
                  width={BRACKET_X - 40}
                  height={ROW_HEIGHT - 6}
                  rx={6}
                />
                <text className="anatomy-sequence__number" x={34} y={y + 4}>
                  {index + 1}
                </text>
                <line
                  className="anatomy-sequence__arrow"
                  x1={from + direction * 8}
                  y1={y}
                  x2={to - direction * 8}
                  y2={y}
                  markerEnd="url(#anatomy-arrow)"
                />
                <circle className="anatomy-sequence__dot" cx={from} cy={y} r={4} />
                <text className="anatomy-sequence__label" x={(from + to) / 2} y={y - 8}>
                  {row.label}
                </text>
                <text
                  className="anatomy-sequence__effect"
                  x={to - direction * 14}
                  y={y + 16}
                  textAnchor={direction > 0 ? 'end' : 'start'}
                >
                  {effect}
                </text>
              </g>
            );
          })}
          {CADENCES.map(cadence => {
            const top = getRowY(cadence.first) - ROW_HEIGHT / 2 + 6;
            const bottom = getRowY(cadence.last) + ROW_HEIGHT / 2 - 6;
            return (
              <g key={cadence.label} className="anatomy-sequence__cadence">
                <path d={`M${BRACKET_X - 8} ${top} H${BRACKET_X} V${bottom} H${BRACKET_X - 8}`} />
                <text x={BRACKET_X + 10} y={(top + bottom) / 2 + 4}>
                  {cadence.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
