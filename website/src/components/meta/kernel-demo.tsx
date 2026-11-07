// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The kernels page: a real WGSL kernel beside a luma.gl canvas that maps invocations to workgroups
// to data elements. The slider sets @workgroup_size and the grid re-tiles; hovering a workgroup
// lights the lines it runs and names the data it touches.

import {useEffect, useMemo, useRef, useState, type PointerEvent} from 'react';
import {GpuCanvas} from '../../gpu/gpu-canvas';
import {formatCount} from '../../lens/format';
import {sampleValueRamp} from '../../lens/ramp';
import {
  describeWorkgroup,
  getKernelLines,
  KERNEL_GRID,
  KERNEL_SOURCE_PATH,
  WORKGROUP_SIZES,
  type KernelLineTag
} from './kernel-source';
import {createTilingSceneLoader, createTilingStore} from './scenes/tiling-scene';
import './meta.css';

const DEFAULT_SIZE_INDEX = 3;

/** Lines a workgroup runs, by tag. The guard line is hot only when the workgroup has guarded lanes. */
function getHotTags(hasGuardedLanes: boolean): ReadonlySet<KernelLineTag> {
  const tags: KernelLineTag[] = ['shape', 'index', 'read', 'math', 'write'];
  if (hasGuardedLanes) tags.push('guard');
  return new Set(tags);
}

function TilingFallback({workgroupSize, hovered}: {workgroupSize: number; hovered: number}) {
  const {columns, rows, invocationCount} = KERNEL_GRID;
  const workgroupCount = Math.ceil(invocationCount / workgroupSize);
  const cells = [];
  for (let index = 0; index < columns * rows; index++) {
    const x = (index % columns) + 0.5;
    const y = Math.floor(index / columns) + 0.5;
    if (index >= invocationCount) {
      cells.push(
        <rect
          key={index}
          x={x - 0.36}
          y={y - 0.36}
          width={0.72}
          height={0.72}
          rx={0.14}
          fill="none"
          stroke="var(--color-signal)"
          strokeWidth={0.1}
        />
      );
      continue;
    }
    const workgroup = Math.floor(index / workgroupSize);
    const [red, green, blue] = sampleValueRamp(workgroup / Math.max(workgroupCount - 1, 1));
    cells.push(
      <rect
        key={index}
        x={x - 0.4}
        y={y - 0.4}
        width={0.8}
        height={0.8}
        rx={0.14}
        fill={`rgb(${Math.round(red * 255)} ${Math.round(green * 255)} ${Math.round(blue * 255)})`}
        fillOpacity={hovered < 0 ? 0.82 : hovered === workgroup ? 1 : 0.16}
      />
    );
  }
  return (
    <svg
      viewBox={`0 0 ${columns} ${rows}`}
      style={{width: '100%', height: '100%'}}
      aria-hidden="true"
    >
      {cells}
    </svg>
  );
}

export function KernelDemo() {
  const [sizeIndex, setSizeIndex] = useState(DEFAULT_SIZE_INDEX);
  const [hovered, setHovered] = useState(-1);
  const workgroupSize = WORKGROUP_SIZES[sizeIndex];
  const storeRef = useRef(createTilingStore(WORKGROUP_SIZES[DEFAULT_SIZE_INDEX]));
  const loader = useMemo(() => createTilingSceneLoader(storeRef.current), []);
  const lines = useMemo(() => getKernelLines(workgroupSize), [workgroupSize]);
  const {columns, rows, invocationCount} = KERNEL_GRID;
  const workgroupCount = Math.ceil(invocationCount / workgroupSize);
  const guardedTotal = workgroupCount * workgroupSize - invocationCount;

  useEffect(() => {
    storeRef.current.workgroupSize = workgroupSize;
    storeRef.current.hovered = -1;
    setHovered(-1);
  }, [workgroupSize]);
  useEffect(() => {
    storeRef.current.hovered = hovered;
  }, [hovered]);

  const handlePointer = (event: PointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const column = Math.floor(((event.clientX - bounds.left) / bounds.width) * columns);
    const row = Math.floor(((event.clientY - bounds.top) / bounds.height) * rows);
    if (column < 0 || row < 0 || column >= columns || row >= rows) return setHovered(-1);
    const index = row * columns + column;
    setHovered(index >= invocationCount ? workgroupCount - 1 : Math.floor(index / workgroupSize));
  };

  const detail = hovered >= 0 ? describeWorkgroup(workgroupSize, hovered) : null;
  const hotTags = detail ? getHotTags(detail.guarded > 0) : null;

  return (
    <div className="meta-panel kernel-demo">
      <div className="kernel-demo__code">
        <div className="meta-panel__bar">
          <span className="meta-panel__title">GPUTerrainRGBDecode</span>
          <span className="meta-legend" style={{marginLeft: 'auto'}}>
            WGSL
          </span>
        </div>
        <pre className="kernel-code" aria-label="WGSL source of the kernel">
          <code>
            {lines.map((line, index) => (
              <span
                key={index}
                className="kernel-line"
                data-tag={line.tag}
                data-hot={hotTags ? hotTags.has(line.tag) : undefined}
                data-guard-hit={
                  detail && detail.guarded > 0 && line.tag === 'guard' ? true : undefined
                }
              >
                {line.text || ' '}
                {'\n'}
              </span>
            ))}
          </code>
        </pre>
        <p className="kernel-source-path">{KERNEL_SOURCE_PATH}</p>
      </div>
      <div className="kernel-demo__visual">
        <div
          className="kernel-demo__canvas"
          onPointerMove={handlePointer}
          onPointerDown={handlePointer}
          onPointerLeave={() => setHovered(-1)}
        >
          <GpuCanvas
            scene={loader}
            fallback={<TilingFallback workgroupSize={workgroupSize} hovered={hovered} />}
            seed={1}
            label={`${formatCount(invocationCount)} data elements tiled into ${workgroupCount} workgroups of ${workgroupSize}`}
            aspectRatio={columns / rows}
          />
        </div>
        <label className="meta-slider">
          <span>
            workgroup_size <b>{workgroupSize}</b>
          </span>
          <input
            type="range"
            min={0}
            max={WORKGROUP_SIZES.length - 1}
            step={1}
            value={sizeIndex}
            onChange={event => setSizeIndex(Number(event.target.value))}
            aria-valuetext={`${workgroupSize} invocations per workgroup`}
          />
        </label>
        <p className="meta-readout" aria-live="polite">
          {detail ? (
            <>
              <b>workgroup {hovered}</b> runs elements <b>{detail.first}</b> to <b>{detail.last}</b>
              {detail.guarded > 0 ? (
                <>
                  , guards out <b>{detail.guarded}</b> lanes
                </>
              ) : null}
              . It reads{' '}
              <code>
                inputWords[{detail.first}..{detail.last}]
              </code>{' '}
              and writes <code>outputValues</code> and <code>outputValidity</code> at the same
              indices.
            </>
          ) : (
            <>
              <b>{workgroupCount}</b> workgroups x <b>{workgroupSize}</b> lanes ={' '}
              <b>{formatCount(workgroupCount * workgroupSize)}</b> slots for{' '}
              <b>{formatCount(invocationCount)}</b> pixels
              {guardedTotal > 0 ? (
                <>
                  ; <b>{guardedTotal}</b> guarded out (hollow)
                </>
              ) : (
                '; none wasted'
              )}
              . Hover a tile.
            </>
          )}
        </p>
      </div>
    </div>
  );
}
