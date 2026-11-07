// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// How the picture you are looking at is made: a live canvas, a HUD of numbers read from it, and the
// five steps from the prerendered SVG to a render pass, lighting up as they happen on this page.
// Steps 3 and 4 come from the wrapped scene (page-probe.ts); step 5 follows the `data-ready`
// attribute that <GpuCanvas> sets once its first frame is validated and shown.

import {useEffect, useMemo, useRef, useState, useSyncExternalStore} from 'react';
import {GpuCanvas} from '../../gpu/gpu-canvas';
import {formatBytes} from '../../lens/format';
import {PageProbe, probeScene, type PageProbeMilestone} from './page-probe';
import {createWorkgroupSceneLoader, type WorkgroupGrid} from './scenes/workgroup-scene';
import './meta.css';

const PAGE_GRID: WorkgroupGrid = {
  columns: 24,
  rows: 10,
  laneSide: 4,
  invocationCount: 24 * 10 * 16 - 37,
  concurrency: 9
};

function PageFallback() {
  const size = 30;
  return (
    <svg
      viewBox={`0 0 ${PAGE_GRID.columns * size} ${PAGE_GRID.rows * size}`}
      style={{width: '100%', height: '100%'}}
      aria-hidden="true"
    >
      {Array.from({length: PAGE_GRID.columns * PAGE_GRID.rows}, (_, index) => (
        <rect
          key={index}
          x={(index % PAGE_GRID.columns) * size + 3}
          y={Math.floor(index / PAGE_GRID.columns) * size + 3}
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

type StepSpec = {
  id: 'fallback' | PageProbeMilestone;
  title: string;
  text: string;
};

const STEPS: readonly StepSpec[] = [
  {id: 'fallback', title: 'SVG fallback', text: 'The HTML already holds the finished picture.'},
  {id: 'hydrated', title: 'Island hydrates', text: 'React wakes this component in the browser.'},
  {id: 'device', title: 'luma.gl device', text: 'WebGPU if the browser has it, else WebGL2.'},
  {id: 'scene', title: 'Scene', text: 'Buffers, shader and model are created.'},
  {id: 'ready', title: 'Render pass', text: 'A frame is drawn and fades in over the SVG.'}
];

function formatTime(milliseconds: number | undefined): string {
  return milliseconds === undefined ? 'waiting' : `${Math.round(milliseconds)} ms`;
}

export function ThisPageDemo() {
  const probe = useMemo(() => new PageProbe(), []);
  const loader = useMemo(() => probeScene(createWorkgroupSceneLoader(PAGE_GRID), probe), [probe]);
  const snapshot = useSyncExternalStore(
    probe.subscribe,
    probe.getSnapshot,
    probe.getServerSnapshot
  );
  const containerRef = useRef<HTMLDivElement>(null);
  const [isWaitingTooLong, setIsWaitingTooLong] = useState(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;
    const canvasContainer = container.querySelector<HTMLElement>('.gpu-canvas');
    probe.update({
      fallbackInHtml: Boolean(canvasContainer?.querySelector('svg')),
      devicePixelRatio: window.devicePixelRatio || 1
    });
    probe.mark('hydrated');
    if (!canvasContainer) return undefined;
    let wasReady = false;
    const observer = new MutationObserver(() => {
      const isReady = canvasContainer.dataset.ready === 'true';
      if (isReady) probe.mark('ready');
      else if (wasReady) probe.reset();
      wasReady = isReady;
    });
    observer.observe(canvasContainer, {attributes: true, attributeFilter: ['data-ready']});
    const timeout = window.setTimeout(() => setIsWaitingTooLong(true), 4000);
    return () => {
      observer.disconnect();
      window.clearTimeout(timeout);
    };
  }, [probe]);

  const {times} = snapshot;
  const isActive = (id: StepSpec['id']): boolean =>
    id === 'fallback' ? snapshot.fallbackInHtml !== false : times[id] !== undefined;
  const backendName =
    snapshot.backend === 'webgpu' ? 'WebGPU' : snapshot.backend === 'webgl' ? 'WebGL2' : 'waiting';
  const framesPerSecond = snapshot.frameMilliseconds > 0 ? 1000 / snapshot.frameMilliseconds : 0;
  const hasFrames = snapshot.frameCount > 0;

  return (
    <div className="this-page">
      <div className="meta-panel" ref={containerRef}>
        <div className="this-page__canvas">
          <GpuCanvas
            scene={loader}
            fallback={<PageFallback />}
            seed={1}
            label="A live luma.gl canvas: a grid of workgroups, measured by the numbers below"
            aspectRatio={PAGE_GRID.columns / PAGE_GRID.rows}
          />
        </div>
        <ul className="hud" aria-label="Numbers read from this canvas">
          <li>
            <span className="hud__label">Backend</span>
            <span className="hud__value">{backendName}</span>
          </li>
          <li>
            <span className="hud__label">Canvas</span>
            <span className="hud__value">
              {snapshot.canvasWidth ? `${snapshot.canvasWidth} x ${snapshot.canvasHeight}` : '-'}
              <small>device px</small>
            </span>
          </li>
          <li>
            <span className="hud__label">Pixel ratio</span>
            <span className="hud__value">{snapshot.devicePixelRatio.toFixed(2)}</span>
          </li>
          <li>
            <span className="hud__label">Frame interval</span>
            <span className="hud__value">
              {hasFrames && snapshot.frameMilliseconds > 0
                ? snapshot.frameMilliseconds.toFixed(1)
                : '-'}
              <small>ms{framesPerSecond ? ` / ${Math.round(framesPerSecond)} fps` : ''}</small>
            </span>
          </li>
          <li>
            <span className="hud__label">CPU per frame</span>
            <span className="hud__value">
              {hasFrames ? snapshot.renderMilliseconds.toFixed(2) : '-'}
              <small>ms</small>
            </span>
          </li>
          <li>
            <span className="hud__label">Draw calls</span>
            <span className="hud__value">
              {hasFrames ? snapshot.drawCalls : '-'}
              <small>{hasFrames ? `in ${snapshot.renderPasses} pass` : ''}</small>
            </span>
          </li>
          <li>
            <span className="hud__label">luma-tracked memory</span>
            <span className="hud__value">
              {snapshot.gpuMemoryBytes === null ? '-' : formatBytes(snapshot.gpuMemoryBytes)}
            </span>
          </li>
          <li>
            <span className="hud__label">Adapter</span>
            <span className="hud__value" style={{fontSize: '0.8rem', overflowWrap: 'anywhere'}}>
              {snapshot.adapter ?? '-'}
            </span>
          </li>
        </ul>
      </div>
      <ol className="pipeline-steps this-page-steps" aria-label="Steps, lit as they happen">
        {STEPS.map(step => (
          <li key={step.id} data-active={isActive(step.id)}>
            <strong>{step.title}</strong>
            {step.text}
            <small>{step.id === 'fallback' ? 'in the HTML' : formatTime(times[step.id])}</small>
          </li>
        ))}
      </ol>
      {isWaitingTooLong && times.ready === undefined ? (
        <p className="meta-figure__caption">
          No usable GPU path here, so the SVG stays. The page works the same without it.
        </p>
      ) : null}
    </div>
  );
}
