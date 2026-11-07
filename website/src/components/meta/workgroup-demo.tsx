// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useMemo} from 'react';
import {GpuCanvas} from '../../gpu/gpu-canvas';
import {formatCount} from '../../lens/format';
import {
  createWorkgroupSceneLoader,
  DEFAULT_WORKGROUP_GRID,
  type WorkgroupGrid
} from './scenes/workgroup-scene';
import './meta.css';

/** Static stand-in: the same grid with every workgroup drawn as an empty frame. */
function WorkgroupFallback({grid}: {grid: WorkgroupGrid}) {
  const size = 40;
  return (
    <svg
      viewBox={`0 0 ${grid.columns * size} ${grid.rows * size}`}
      style={{width: '100%', height: '100%'}}
      aria-hidden="true"
    >
      {Array.from({length: grid.columns * grid.rows}, (_, index) => (
        <rect
          key={index}
          x={(index % grid.columns) * size + 3}
          y={Math.floor(index / grid.columns) * size + 3}
          width={size - 6}
          height={size - 6}
          rx={4}
          fill="none"
          stroke="var(--color-accent)"
          strokeOpacity={0.4}
        />
      ))}
    </svg>
  );
}

export function WorkgroupDemo({grid = DEFAULT_WORKGROUP_GRID}: {grid?: WorkgroupGrid}) {
  const loader = useMemo(() => createWorkgroupSceneLoader(grid), [grid]);
  const workgroups = grid.columns * grid.rows;
  const lanes = grid.laneSide * grid.laneSide;
  return (
    <div className="meta-canvas">
      <GpuCanvas
        scene={loader}
        fallback={<WorkgroupFallback grid={grid} />}
        seed={1}
        label={`A dispatch of ${workgroups} workgroups of ${lanes} invocations`}
        aspectRatio={(grid.columns / grid.rows) * 1}
      />
      <ul className="workgroup-key">
        <li>
          <b>{formatCount(workgroups)}</b> workgroups x <b>{lanes}</b> lanes ={' '}
          <b>{formatCount(workgroups * lanes)}</b> slots
        </li>
        <li>
          <i data-kind="active" />
          running
        </li>
        <li>
          <i data-kind="idle" />
          waiting
        </li>
        <li>
          <i data-kind="guard" />
          past <b>{formatCount(grid.invocationCount)}</b> invocations: guarded out
        </li>
      </ul>
    </div>
  );
}
