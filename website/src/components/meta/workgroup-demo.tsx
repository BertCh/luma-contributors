// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// One dispatch as a picture: a grid of workgroups, each a grid of lanes. The static figure is the
// whole point and renders without a GPU: filled lanes run, hollow lanes sit past the element count
// and return at the index guard. The live canvas adds a front of workgroups sweeping the grid.

import {useId, useMemo} from 'react';
import {GpuCanvas} from '../../gpu/gpu-canvas';
import {formatCount} from '../../lens/format';
import {
  createWorkgroupSceneLoader,
  DEFAULT_WORKGROUP_GRID,
  type WorkgroupGrid
} from './scenes/workgroup-scene';
import './meta.css';
import './pages/kernels/kernels.css';

const BLOCK_SIZE = 40;
const BLOCK_PADDING = 3;

/** Lane counts of a dispatch: how many workgroups, slots and guarded lanes the grid has. */
export function getDispatchCounts(grid: WorkgroupGrid) {
  const lanes = grid.laneSide * grid.laneSide;
  const workgroups = grid.columns * grid.rows;
  const slots = workgroups * lanes;
  const guarded = Math.max(0, slots - grid.invocationCount);
  return {lanes, workgroups, slots, guarded};
}

/** Lane rectangles of one workgroup: filled when the lane's element exists, hollow when guarded. */
function WorkgroupLanes({
  grid,
  workgroupIndex,
  originX,
  originY,
  size
}: {
  grid: WorkgroupGrid;
  workgroupIndex: number;
  originX: number;
  originY: number;
  size: number;
}) {
  const {laneSide} = grid;
  const laneSize = size / laneSide;
  const lanes = [];
  for (let lane = 0; lane < laneSide * laneSide; lane++) {
    const isGuarded = workgroupIndex * laneSide * laneSide + lane >= grid.invocationCount;
    const x = originX + (lane % laneSide) * laneSize + laneSize * 0.12;
    const y = originY + Math.floor(lane / laneSide) * laneSize + laneSize * 0.12;
    const side = laneSize * 0.76;
    lanes.push(
      <rect
        key={lane}
        x={x}
        y={y}
        width={side}
        height={side}
        rx={laneSize * 0.16}
        fill={isGuarded ? 'none' : 'var(--color-accent)'}
        fillOpacity={isGuarded ? undefined : 0.78}
        stroke={isGuarded ? 'var(--color-signal)' : undefined}
        strokeWidth={isGuarded ? Math.max(laneSize * 0.1, 0.6) : undefined}
      />
    );
  }
  return <>{lanes}</>;
}

/** Static stand-in: every workgroup drawn with its lanes. Full workgroups use a lane pattern. */
function WorkgroupFallback({grid}: {grid: WorkgroupGrid}) {
  const patternId = useId();
  const workgroupCount = grid.columns * grid.rows;
  const fullCount = Math.floor(grid.invocationCount / (grid.laneSide * grid.laneSide));
  const laneSize = BLOCK_SIZE / grid.laneSide;
  const blocks = [];
  for (let index = 0; index < workgroupCount; index++) {
    const x = (index % grid.columns) * BLOCK_SIZE;
    const y = Math.floor(index / grid.columns) * BLOCK_SIZE;
    if (index < fullCount) {
      blocks.push(
        <rect
          key={index}
          x={x + BLOCK_PADDING}
          y={y + BLOCK_PADDING}
          width={BLOCK_SIZE - 2 * BLOCK_PADDING}
          height={BLOCK_SIZE - 2 * BLOCK_PADDING}
          rx={3}
          fill={`url(#${patternId})`}
        />
      );
    } else {
      blocks.push(
        <WorkgroupLanes
          key={index}
          grid={grid}
          workgroupIndex={index}
          originX={x + BLOCK_PADDING}
          originY={y + BLOCK_PADDING}
          size={BLOCK_SIZE - 2 * BLOCK_PADDING}
        />
      );
    }
  }
  return (
    <svg
      viewBox={`0 0 ${grid.columns * BLOCK_SIZE} ${grid.rows * BLOCK_SIZE}`}
      style={{width: '100%', height: '100%'}}
      aria-hidden="true"
    >
      <defs>
        <pattern
          id={patternId}
          width={laneSize}
          height={laneSize}
          patternUnits="userSpaceOnUse"
          patternTransform={`translate(${BLOCK_PADDING} ${BLOCK_PADDING})`}
        >
          <rect
            x={laneSize * 0.12}
            y={laneSize * 0.12}
            width={laneSize * 0.76}
            height={laneSize * 0.76}
            rx={laneSize * 0.16}
            fill="var(--color-accent)"
            fillOpacity={0.78}
          />
        </pattern>
      </defs>
      {blocks}
    </svg>
  );
}

/** The last workgroup, enlarged: the lanes that run and the lanes the guard returns. */
function LastWorkgroup({grid}: {grid: WorkgroupGrid}) {
  const {lanes, workgroups, guarded} = getDispatchCounts(grid);
  const last = workgroups - 1;
  const running = lanes - Math.min(guarded, lanes);
  const size = 120;
  return (
    <figure className="workgroup-zoom">
      <svg
        viewBox={`0 0 ${size} ${size}`}
        width={size}
        height={size}
        role="img"
        aria-label={`The last workgroup: ${running} of ${lanes} lanes run, ${lanes - running} are guarded out`}
      >
        <WorkgroupLanes grid={grid} workgroupIndex={last} originX={0} originY={0} size={size} />
      </svg>
      <figcaption>
        Last workgroup: <b>{running}</b> of <b>{lanes}</b> lanes run; <b>{lanes - running}</b>{' '}
        return at the guard.
      </figcaption>
    </figure>
  );
}

export function WorkgroupDemo({grid = DEFAULT_WORKGROUP_GRID}: {grid?: WorkgroupGrid}) {
  const loader = useMemo(() => createWorkgroupSceneLoader(grid), [grid]);
  const {lanes, workgroups, slots, guarded} = getDispatchCounts(grid);
  return (
    <div className="meta-canvas">
      <GpuCanvas
        scene={loader}
        fallback={<WorkgroupFallback grid={grid} />}
        seed={1}
        label={`A dispatch of ${workgroups} workgroups of ${lanes} invocations, ${guarded} lanes guarded out`}
        aspectRatio={grid.columns / grid.rows}
      />
      <div className="workgroup-foot">
        {guarded > 0 && guarded < lanes ? <LastWorkgroup grid={grid} /> : null}
        <ul className="workgroup-key">
          <li>
            <b>{formatCount(workgroups)}</b> workgroups x <b>{lanes}</b> lanes ={' '}
            <b>{formatCount(slots)}</b> slots
          </li>
          <li>
            <i data-kind="active" />
            runs an element
          </li>
          <li>
            <i data-kind="guard" />
            past <b>{formatCount(grid.invocationCount)}</b> elements: guarded out
          </li>
        </ul>
      </div>
    </div>
  );
}
