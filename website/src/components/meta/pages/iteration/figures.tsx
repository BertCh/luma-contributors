// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Figures of the iteration page. Every number is derived from the recorded reachability run
// (search-data.ts) or the generated view models; the default state of each figure makes its point.

import {Link} from '@tanstack/react-router';
import {useEffect, useId, useState} from 'react';
import {catalog} from '../../../../catalog';
import {formatCount} from '../../../../lens/format';
import {type NodeState, RoadMap} from './road-map';
import {
  DEFAULT_MAXIMUM_ITERATIONS,
  FIRST_EMPTY_ROUND,
  GATED,
  type GatedRecord,
  getBudgetStatus,
  getRoundWorkgroups,
  PEAK_ROUND,
  PEAK_SIZE,
  SEARCH,
  SETTLED_BY_ROUND,
  WORKING_ROUNDS
} from './search-data';

const [EVEN_BUFFER, ODD_BUFFER] = SEARCH.gateBuffers;

// A loop the CPU cannot run

const LOOP_WIDTH = 680;

/** The CPU loop with a readback per round, against one submission of laid-out rounds. */
export function LoopFigure() {
  const naiveRounds = [0, 1, 2];
  const cellCount = 56;
  const cellWidth = 8;
  return (
    <div className="meta-scroll">
      <svg
        className="iteration-loop"
        viewBox={`0 0 ${LOOP_WIDTH} 290`}
        style={{minWidth: 600}}
        role="img"
        aria-label="Top, not done: the CPU reads a done flag back after every round and waits before submitting the next. Bottom, what the library does: one submission, rounds run back to back on the GPU, empty rounds dispatch nothing."
      >
        <text className="iteration-loop__title" x={0} y={16}>
          Not done: ask after every round
        </text>
        <text className="iteration-loop__lane" x={0} y={50}>
          CPU
        </text>
        <text className="iteration-loop__lane" x={0} y={100}>
          GPU
        </text>
        {naiveRounds.map(round => {
          const x = 48 + round * 196;
          return (
            <g key={round}>
              <rect className="iteration-loop__cpu" x={x} y={34} width={64} height={24} rx={4} />
              <text className="iteration-loop__label" x={x + 32} y={50} textAnchor="middle">
                submit
              </text>
              <path className="iteration-loop__arrow" d={`M${x + 32} 58 L${x + 32} 84`} />
              <rect
                className="iteration-loop__gpu"
                x={x + 8}
                y={86}
                width={76}
                height={24}
                rx={4}
              />
              <text className="iteration-loop__label" x={x + 46} y={102} textAnchor="middle">
                round {round}
              </text>
              <path
                className="iteration-loop__arrow"
                data-kind="readback"
                d={`M${x + 84} 98 C${x + 110} 98 ${x + 116} 46 ${x + 132} 46`}
              />
              <rect
                className="iteration-loop__wait"
                x={x + 134}
                y={34}
                width={52}
                height={24}
                rx={4}
              />
              <text className="iteration-loop__label" x={x + 160} y={50} textAnchor="middle">
                done?
              </text>
              {round < naiveRounds.length - 1 ? (
                <path className="iteration-loop__arrow" d={`M${x + 186} 46 L${x + 196} 46`} />
              ) : null}
            </g>
          );
        })}
        <text className="iteration-loop__note" x={48} y={134}>
          each round waits for a readback of the done flag before the next submit
        </text>
        <line className="iteration-loop__rule" x1={0} x2={LOOP_WIDTH} y1={152} y2={152} />
        <text className="iteration-loop__title" x={0} y={178}>
          What runs: rounds laid out once, gated on the GPU
        </text>
        <text className="iteration-loop__lane" x={0} y={212}>
          CPU
        </text>
        <text className="iteration-loop__lane" x={0} y={262}>
          GPU
        </text>
        <rect className="iteration-loop__cpu" x={48} y={196} width={64} height={24} rx={4} />
        <text className="iteration-loop__label" x={80} y={212} textAnchor="middle">
          submit
        </text>
        <path className="iteration-loop__arrow" d="M80 220 L80 246" />
        {Array.from({length: cellCount}, (_value, round) => (
          <rect
            key={round}
            className="iteration-loop__round"
            data-empty={round >= FIRST_EMPTY_ROUND ? 'true' : undefined}
            x={56 + round * (cellWidth + 2)}
            y={248}
            width={cellWidth}
            height={24}
            rx={1.5}
          />
        ))}
        <text className="iteration-loop__label" x={56 + cellCount * (cellWidth + 2) + 6} y={264}>
          … {SEARCH.relaxRoundCount}
        </text>
        <text className="iteration-loop__note" x={56} y={286}>
          {WORKING_ROUNDS} rounds do work; the rest dispatch zero workgroups
        </text>
      </svg>
    </div>
  );
}

// A recorded search

function usePlayer(last: number, initial: number) {
  const [value, setValue] = useState(initial);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setValue(current => {
        if (current >= last) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 320);
    return () => window.clearInterval(timer);
  }, [playing, last]);
  const toggle = () => {
    if (!playing && value >= last) setValue(0);
    setPlaying(!playing);
  };
  return {value, setValue, playing, setPlaying, toggle};
}

const BAR_WIDTH = 420;
const BAR_HEIGHT = 132;

function FrontierBars({round, onRound}: {round: number; onRound: (round: number) => void}) {
  const top = 22;
  const bottom = BAR_HEIGHT - 24;
  const left = 30;
  const step = (BAR_WIDTH - left - 4) / SEARCH.frontier.length;
  const scale = (bottom - top) / PEAK_SIZE;
  return (
    <svg
      className="iteration-bars"
      viewBox={`0 0 ${BAR_WIDTH} ${BAR_HEIGHT}`}
      role="img"
      aria-label={`Nodes in each round's queue: rising to ${PEAK_SIZE} at round ${PEAK_ROUND} and falling to 0 at round ${FIRST_EMPTY_ROUND}. Round ${round} has ${SEARCH.frontier[round]}.`}
    >
      <line className="iteration-bars__axis" x1={left} x2={BAR_WIDTH} y1={bottom} y2={bottom} />
      <text className="iteration-bars__tick" x={left - 6} y={top + 4} textAnchor="end">
        {PEAK_SIZE}
      </text>
      <text className="iteration-bars__tick" x={left - 6} y={bottom} textAnchor="end">
        0
      </text>
      {SEARCH.frontier.map((size, index) => {
        const x = left + index * step;
        const height = Math.max(size * scale, 0);
        return (
          <g key={index} className="iteration-bars__bar" onClick={() => onRound(index)}>
            <rect x={x} y={top} width={step} height={bottom - top} fill="transparent" />
            {size > 0 ? (
              <rect
                data-current={index === round ? 'true' : undefined}
                x={x + 1}
                y={bottom - height}
                width={step - 2}
                height={height}
              />
            ) : (
              <rect data-empty="true" x={x + 1} y={bottom - 3} width={step - 2} height={3} />
            )}
          </g>
        );
      })}
      <text
        className="iteration-bars__value"
        x={left + (round + 0.5) * step}
        y={bottom - SEARCH.frontier[round] * scale - 5}
        textAnchor="middle"
      >
        {SEARCH.frontier[round]}
      </text>
      {[0, 10, 20, 30, FIRST_EMPTY_ROUND].map(index => (
        <text
          key={index}
          className="iteration-bars__tick"
          x={left + (index + 0.5) * step}
          y={bottom + 16}
          textAnchor="middle"
        >
          {index}
        </text>
      ))}
    </svg>
  );
}

/** The recorded search: the road network lit by round, with the queue size of every round. */
export function SearchFigure() {
  const sliderId = useId();
  const player = usePlayer(FIRST_EMPTY_ROUND, PEAK_ROUND);
  const round = player.value;
  const lit = SEARCH.lastImprovedRound.filter(stamp => stamp <= round).length;
  const getState = (node: number): NodeState => {
    const stamp = SEARCH.lastImprovedRound[node];
    return stamp < round ? 'lit' : stamp === round ? 'front' : 'dark';
  };
  return (
    <div className="meta-panel iteration-search">
      <div className="meta-controls">
        <button
          type="button"
          className="meta-play"
          aria-pressed={player.playing}
          onClick={player.toggle}
        >
          {player.playing ? 'Pause' : 'Play'}
        </button>
        <label className="meta-slider meta-slider--wide" htmlFor={sliderId}>
          Round <b>{round}</b>
          <input
            id={sliderId}
            type="range"
            min={0}
            max={FIRST_EMPTY_ROUND}
            step={1}
            value={round}
            onChange={event => {
              player.setPlaying(false);
              player.setValue(Number(event.target.value));
            }}
          />
        </label>
        <span className="meta-controls__now" aria-live="polite">
          queue <b>{SEARCH.frontier[round]}</b> · last queued here <b>{SETTLED_BY_ROUND[round]}</b>{' '}
          · final <b>{formatCount(lit)}</b> of {formatCount(SEARCH.nodeCount)}
        </span>
      </div>
      <div className="iteration-search__body">
        <div className="iteration-search__map">
          <RoadMap
            getState={getState}
            stateKey={`round-${round}`}
            label={`Chicago Loop road network at round ${round}: ${formatCount(lit)} of ${formatCount(SEARCH.nodeCount)} intersections have their final travel time; ${SETTLED_BY_ROUND[round]} join this round's queue for the last time.`}
          />
          <ul className="iteration-legend" aria-label="Legend">
            <li data-swatch="front">queued this round, last time</li>
            <li data-swatch="lit">final cost</li>
            <li data-swatch="dark">not final yet</li>
          </ul>
        </div>
        <div className="iteration-search__bars">
          <span className="iteration-search__axis-label">Nodes in each round's queue</span>
          <FrontierBars
            round={round}
            onRound={next => {
              player.setPlaying(false);
              player.setValue(next);
            }}
          />
        </div>
      </div>
    </div>
  );
}

// Zero workgroups

const STRIP_COLUMNS = 32;
const STRIP_CELL = 18;
const STRIP_GAP = 3;

/** All laid-out rounds as cells: filled by queue size where the round did work, hollow after. */
export function RoundStrip() {
  const rows = Math.ceil(SEARCH.relaxRoundCount / STRIP_COLUMNS);
  const width = STRIP_COLUMNS * (STRIP_CELL + STRIP_GAP) - STRIP_GAP;
  const height = rows * (STRIP_CELL + STRIP_GAP) - STRIP_GAP;
  return (
    <div className="meta-canvas iteration-strip">
      <svg
        viewBox={`-1 -1 ${width + 2} ${height + 2}`}
        role="img"
        aria-label={`${SEARCH.relaxRoundCount} relax rounds laid out in the graph, read left to right, top to bottom. The first ${WORKING_ROUNDS} are filled by queue size; the other ${SEARCH.relaxRoundCount - WORKING_ROUNDS} are hollow: they dispatch zero workgroups.`}
      >
        {Array.from({length: SEARCH.relaxRoundCount}, (_value, round) => {
          const size = SEARCH.frontier[round] ?? 0;
          const column = round % STRIP_COLUMNS;
          const row = Math.floor(round / STRIP_COLUMNS);
          return (
            <rect
              key={round}
              className="iteration-strip__cell"
              data-empty={size === 0 ? 'true' : undefined}
              x={column * (STRIP_CELL + STRIP_GAP)}
              y={row * (STRIP_CELL + STRIP_GAP)}
              width={STRIP_CELL}
              height={STRIP_CELL}
              rx={2}
              style={size > 0 ? {fillOpacity: 0.25 + (0.75 * size) / PEAK_SIZE} : undefined}
            >
              <title>{`round ${round}: ${size} queued, ${getRoundWorkgroups(round)} workgroups`}</title>
            </rect>
          );
        })}
      </svg>
      <ul className="iteration-legend" aria-label="Legend">
        <li data-swatch="lit">did work (darker: larger queue)</li>
        <li data-swatch="empty">in the graph, dispatched zero workgroups</li>
      </ul>
    </div>
  );
}

const GATE_WIDTH = 700;

/** Round k reads its dispatch from one buffer and writes round k + 1's into the other. */
export function GateDiagram() {
  const columns: ({kind: 'seed'} | {kind: 'round'; round: number} | {kind: 'gap'})[] = [
    {kind: 'seed'},
    {kind: 'round', round: 0},
    {kind: 'round', round: 1},
    {kind: 'round', round: 2},
    {kind: 'round', round: 3},
    {kind: 'gap'},
    {kind: 'round', round: FIRST_EMPTY_ROUND - 1},
    {kind: 'round', round: FIRST_EMPTY_ROUND},
    {kind: 'round', round: FIRST_EMPTY_ROUND + 1},
    {kind: 'gap'}
  ];
  const left = 12;
  const step = (GATE_WIDTH - left) / columns.length;
  const evenY = 44;
  const oddY = 214;
  const boxY = 112;
  const boxHeight = 44;
  const slotSize = 34;
  const getLaneY = (round: number) => (round % 2 === 0 ? evenY : oddY);
  return (
    <div className="meta-scroll">
      <svg
        className="iteration-gate"
        viewBox={`0 0 ${GATE_WIDTH} 268`}
        style={{minWidth: 620}}
        role="img"
        aria-label={`Round k reads its workgroup count x from a slot in ${EVEN_BUFFER} (even rounds) or ${ODD_BUFFER} (odd rounds) and writes round k + 1's count into the other buffer. Rounds 0 to ${FIRST_EMPTY_ROUND - 1} read x = 1; round ${FIRST_EMPTY_ROUND} reads x = 0 and dispatches nothing, and so does every later round.`}
      >
        <defs>
          <marker
            id="iteration-gate-arrow"
            viewBox="0 0 8 8"
            refX={7}
            refY={4}
            markerWidth={7}
            markerHeight={7}
            orient="auto-start-reverse"
          >
            <path d="M0 0 L8 4 L0 8 z" className="iteration-gate__head" />
          </marker>
        </defs>
        <text className="iteration-gate__buffer" x={left} y={evenY - slotSize / 2 - 8}>
          {EVEN_BUFFER}
        </text>
        <text className="iteration-gate__buffer" x={left} y={oddY + slotSize / 2 + 18}>
          {ODD_BUFFER}
        </text>
        <line className="iteration-gate__lane" x1={left} x2={GATE_WIDTH} y1={evenY} y2={evenY} />
        <line className="iteration-gate__lane" x1={left} x2={GATE_WIDTH} y1={oddY} y2={oddY} />
        {columns.map((column, index) => {
          const centre = left + (index + 0.5) * step;
          if (column.kind === 'gap') {
            return (
              <text
                key={index}
                className="iteration-gate__gap"
                x={centre}
                y={boxY + boxHeight / 2 + 5}
                textAnchor="middle"
              >
                …
              </text>
            );
          }
          if (column.kind === 'seed') {
            const nextCentre = centre + step;
            return (
              <g key={index}>
                <rect
                  className="iteration-gate__kernel"
                  data-kind="seed"
                  x={centre - 28}
                  y={boxY}
                  width={56}
                  height={boxHeight}
                  rx={5}
                />
                <text className="iteration-gate__name" x={centre} y={boxY + 27} textAnchor="middle">
                  seed
                </text>
                <path
                  className="iteration-gate__write"
                  d={`M${centre + 12} ${boxY} L${nextCentre - 8} ${evenY + slotSize / 2 + 2}`}
                  markerEnd="url(#iteration-gate-arrow)"
                />
              </g>
            );
          }
          const {round} = column;
          const workgroups = getRoundWorkgroups(round);
          const laneY = getLaneY(round);
          const readsDown = laneY === evenY;
          const nextColumn = columns[index + 1];
          const writes = nextColumn?.kind === 'round';
          const nextLaneY = getLaneY(round + 1);
          return (
            <g key={index}>
              <rect
                className="iteration-gate__slot"
                data-empty={workgroups === 0 ? 'true' : undefined}
                x={centre - slotSize / 2}
                y={laneY - slotSize / 2}
                width={slotSize}
                height={slotSize}
                rx={4}
              />
              <text className="iteration-gate__x" x={centre} y={laneY + 5} textAnchor="middle">
                x {workgroups}
              </text>
              <path
                className="iteration-gate__read"
                d={
                  readsDown
                    ? `M${centre} ${laneY + slotSize / 2} L${centre} ${boxY - 2}`
                    : `M${centre} ${laneY - slotSize / 2} L${centre} ${boxY + boxHeight + 2}`
                }
                markerEnd="url(#iteration-gate-arrow)"
              />
              <rect
                className="iteration-gate__kernel"
                data-empty={workgroups === 0 ? 'true' : undefined}
                x={centre - 31}
                y={boxY}
                width={62}
                height={boxHeight}
                rx={5}
              />
              <text className="iteration-gate__name" x={centre} y={boxY + 19} textAnchor="middle">
                relax {round}
              </text>
              <text className="iteration-gate__meta" x={centre} y={boxY + 35} textAnchor="middle">
                {workgroups === 0 ? 'idle' : `${SEARCH.frontier[round]} queued`}
              </text>
              {writes && workgroups > 0 ? (
                <path
                  className="iteration-gate__write"
                  d={
                    nextLaneY === oddY
                      ? `M${centre + 14} ${boxY + boxHeight} L${centre + step - 8} ${oddY - slotSize / 2 - 2}`
                      : `M${centre + 14} ${boxY} L${centre + step - 8} ${evenY + slotSize / 2 + 2}`
                  }
                  markerEnd="url(#iteration-gate-arrow)"
                />
              ) : null}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// When the budget runs out

/** The recorded run cut at a round budget: what status says and which nodes are not final yet. */
export function BudgetFigure() {
  const sliderId = useId();
  const [budget, setBudget] = useState(Math.round(FIRST_EMPTY_ROUND * 0.55));
  const status = getBudgetStatus(budget);
  const getState = (node: number): NodeState =>
    SEARCH.lastImprovedRound[node] > budget ? 'pending' : 'lit';
  return (
    <div className="meta-panel iteration-budget">
      <div className="meta-controls">
        <label className="meta-slider meta-slider--wide" htmlFor={sliderId}>
          Round budget <b>{budget}</b>
          <input
            id={sliderId}
            type="range"
            min={1}
            max={DEFAULT_MAXIMUM_ITERATIONS}
            step={1}
            value={budget}
            onChange={event => setBudget(Number(event.target.value))}
          />
        </label>
        <span className="meta-controls__now">replay of the recorded run, not a new GPU run</span>
      </div>
      <div className="iteration-budget__body">
        <div className="iteration-search__map">
          <RoadMap
            getState={getState}
            stateKey={`budget-${budget}`}
            label={`With a budget of ${budget} rounds, ${formatCount(status.notFinal)} of ${formatCount(SEARCH.nodeCount)} intersections are not final yet and converged is ${status.converged}.`}
          />
          <ul className="iteration-legend" aria-label="Legend">
            <li data-swatch="lit">final cost</li>
            <li data-swatch="pending">not final yet</li>
          </ul>
        </div>
        <dl className="iteration-status" aria-live="polite">
          <div>
            <dt>
              <code>converged</code>
            </dt>
            <dd data-attention={status.converged === 0 ? 'true' : undefined}>{status.converged}</dd>
          </div>
          <div>
            <dt>
              <code>iterationCount</code>
            </dt>
            <dd>{status.iterationCount}</dd>
          </div>
          <div>
            <dt>not final yet</dt>
            <dd data-attention={status.notFinal > 0 ? 'true' : undefined}>
              {formatCount(status.notFinal)}
            </dd>
          </div>
          <p className="iteration-status__note">{getBudgetNote(budget, status)}</p>
        </dl>
      </div>
    </div>
  );
}

function getBudgetNote(budget: number, status: ReturnType<typeof getBudgetStatus>): string {
  if (status.converged === 0) {
    const proof =
      status.notFinal === 0 ? ' Every cost is final already, but the search could not tell.' : '';
    return `Round ${budget - 1} still queued work for round ${budget}, past the limit, so the result is published with converged = 0.${proof}`;
  }
  if (budget === WORKING_ROUNDS) return `Exactly the ${WORKING_ROUNDS} rounds the search needed.`;
  return `The search needed ${WORKING_ROUNDS} rounds; the other ${budget - WORKING_ROUNDS} dispatch nothing.`;
}

// Where else

const ENTRY_BY_SCENARIO = new Map(
  catalog.flatMap(entry => (entry.lensScenario ? [[entry.lensScenario, entry] as const] : []))
);

function GatedRow({record, highlight}: {record: GatedRecord; highlight?: boolean}) {
  const entry = ENTRY_BY_SCENARIO.get(record.id);
  const largest = GATED.top[0]?.kernels ?? record.kernels;
  return (
    <li className="iteration-gated__row" data-highlight={highlight ? 'true' : undefined}>
      <span className="iteration-gated__name">
        {entry ? (
          <Link to="/fn/$id" params={{id: entry.id}}>
            {entry.title}
          </Link>
        ) : (
          record.title
        )}
      </span>
      <span className="iteration-gated__track" aria-hidden="true">
        <span
          className="iteration-gated__total"
          style={{width: `${(100 * record.kernels) / largest}%`}}
        />
        <span
          className="iteration-gated__gated"
          style={{width: `${(100 * record.gated) / largest}%`}}
        />
      </span>
      <span className="iteration-gated__count">
        <b>{formatCount(record.gated)}</b> of {formatCount(record.kernels)}
      </span>
    </li>
  );
}

/** Records ranked by GPU-gated kernels, with the reachability run for scale. */
export function GatedList() {
  const reachability = GATED.reachability;
  const inTop = GATED.top.some(record => record.id === reachability?.id);
  return (
    <div className="meta-panel iteration-gated">
      <div className="meta-panel__bar">
        <span className="meta-panel__title">Gated kernels of all kernels, by record</span>
      </div>
      <ol className="iteration-gated__list">
        {GATED.top.map(record => (
          <GatedRow key={record.id} record={record} />
        ))}
      </ol>
      {reachability && !inTop ? (
        <ol className="iteration-gated__list" start={GATED.reachabilityRank}>
          <GatedRow record={reachability} highlight />
        </ol>
      ) : null}
    </div>
  );
}
