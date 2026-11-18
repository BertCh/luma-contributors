// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// <GpuCanvas>: a live luma.gl picture over a static fallback. Prerendering emits the fallback only.
// In the browser the scene starts when scrolled into view, pauses when offscreen, and the canvas
// fades in over the fallback once its first frame is drawn. Any failure keeps the fallback.
// With `controls`, a strip under the canvas holds one slider per scene parameter; the sliders are
// part of the prerendered HTML (disabled) and come alive with the canvas.

import {
  useEffect,
  useId,
  useRef,
  useState,
  type CSSProperties,
  type ReactElement,
  type ReactNode
} from 'react';
import type {Device} from '@luma.gl/core';
import {bindCanvasPointer} from './canvas-pointer';
import type {
  GpuSceneInstance,
  GpuSceneLoader,
  GpuSceneParameter,
  GpuSceneParameters,
  GpuTheme
} from './types.ts';
import './gpu-canvas.css';

export type GpuCanvasProps = {
  scene: GpuSceneLoader;
  /** Shown during prerendering, before the first frame, and when the GPU is unavailable. */
  fallback: ReactNode;
  /** Deterministic seed shared with the fallback. */
  seed?: number;
  /** Accessible description. */
  label: string;
  /** Width over height. */
  aspectRatio?: number;
  className?: string;
  /** Force a theme when the containing panel has a fixed palette. */
  theme?: GpuTheme;
  /** Forward pointer position to the scene. */
  interactive?: boolean;
  /**
   * Element that receives pointer events. `canvas` (default) listens on the canvas itself. `card`
   * listens on the enclosing `.card`, whose stretched title link covers the canvas; the position is
   * still reported in canvas pixels, and null while the pointer is over the card but off the canvas.
   * Touch is ignored on cards (a tap navigates) and keyboard focus sends the canvas center.
   */
  pointerSurface?: 'canvas' | 'card';
  /** Show the "move pointer" chip on interactive canvases. */
  hint?: boolean;
  /** Render a slider strip under the canvas, one slider per scene parameter. */
  controls?: boolean;
  /** The scene's parameters, known at render time so sliders appear in the prerendered HTML. */
  parameters?: GpuSceneParameters;
};

type ParameterValues = Record<string, number>;

function getDefaultValues(parameters: GpuSceneParameters): ParameterValues {
  const values: ParameterValues = {};
  for (const parameter of parameters) values[parameter.id] = parameter.default;
  return values;
}

function formatValue(parameter: GpuSceneParameter, value: number): string {
  const decimals = (String(parameter.step).split('.')[1] ?? '').length;
  const unit = parameter.unit ?? '';
  // Words get a space ("4 buildings"); symbols stay attached ("25%", "2°").
  return `${value.toFixed(decimals)}${/^[A-Za-z]/.test(unit) ? ' ' : ''}${unit}`;
}

function getTheme(): GpuTheme {
  const attribute = document.documentElement.dataset['theme'];
  if (attribute === 'light' || attribute === 'dark') return attribute;
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

const FADE: CSSProperties = {transition: 'opacity 400ms ease'};

export function GpuCanvas({
  scene,
  fallback,
  seed = 1,
  label,
  aspectRatio = 16 / 10,
  className,
  interactive = false,
  controls = false,
  parameters,
  pointerSurface = 'canvas',
  theme,
  hint = true
}: GpuCanvasProps): ReactElement {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasHostRef = useRef<HTMLDivElement>(null);
  const [readout, setReadout] = useState<string | null>(null);
  // The scene's own pointer gesture: its hint, whether the reader has moved yet, and pinning.
  const [gesture, setGesture] = useState<{hint?: string; hasMoved?: boolean; isPinned?: boolean}>(
    {}
  );
  const [isReady, setIsReady] = useState(false);
  // Bumped when a canvas is released offscreen, so the next start gets a fresh element.
  const [generation, setGeneration] = useState(0);
  const [backend, setBackend] = useState<string | null>(null);
  // Parameters declared by the loaded scene module; used when the page did not pass `parameters`.
  const [moduleParameters, setModuleParameters] = useState<GpuSceneParameters | null>(null);
  const activeParameters = parameters ?? moduleParameters ?? [];
  const [values, setValues] = useState<ParameterValues>(() => getDefaultValues(parameters ?? []));
  // Current slider values: read by the effect when it creates an instance, so they survive rebuilds.
  const valuesRef = useRef<ParameterValues>(values);
  const applyParametersRef = useRef<((changed: ParameterValues) => void) | null>(null);
  const resetPointerRef = useRef<(() => void) | null>(null);
  const forwardsPointer = interactive || controls;

  useEffect(() => {
    if (!containerRef.current || !canvasHostRef.current) return undefined;
    const container = containerRef.current;
    // In-flight builds may outlive their React effect. Each device gets an exclusive canvas
    // so releasing an older device cannot unconfigure the replacement's presentation context.
    const canvas = document.createElement('canvas');
    canvas.style.cssText = 'display:block;width:100%;height:100%';
    canvasHostRef.current.replaceChildren(canvas);
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    let visible = false;
    let disposed = false;
    let starting = false;
    let attempted = false;
    let frame = 0;
    let device: Device | null = null;
    let instance: GpuSceneInstance | null = null;
    let releaseSlot: (() => void) | undefined;
    setIsReady(false);
    setBackend(null);
    setReadout(null);
    setGesture({});

    function disposeResources() {
      instance?.destroy();
      instance = null;
      device?.destroy();
      device = null;
      releaseSlot?.();
      releaseSlot = undefined;
    }

    function fail(error: unknown) {
      if (!disposed) {
        // biome-ignore lint/suspicious/noConsole: diagnostic output
        console.warn('GpuCanvas could not draw, keeping the static fallback', error);
        setIsReady(false);
      }
      cancelAnimationFrame(frame);
      pointer?.destroy();
      disposeResources();
    }

    function draw(time: number) {
      try {
        instance?.render(time);
      } catch (error) {
        fail(error);
      }
    }

    function tick(time: number) {
      if (disposed || !visible || !instance) return;
      draw(time);
      if (instance && !motionQuery.matches) frame = requestAnimationFrame(tick);
    }

    function drawStill() {
      if (visible && !starting && motionQuery.matches) draw(0);
    }

    const pointer = forwardsPointer
      ? bindCanvasPointer(
          container,
          pointerSurface === 'card',
          drawStill,
          (parameter, value) => {
            const label = parameter.label.replace(/^Data: /, '');
            const formatted =
              parameter.options?.[Math.round(value)] ?? formatValue(parameter, value);
            setReadout(`${label} ${formatted}`);
          },
          state => setGesture(previous => ({...previous, ...state}))
        )
      : null;

    applyParametersRef.current = changed => {
      instance?.setParameters?.(changed);
      drawStill();
    };
    resetPointerRef.current = () => {
      instance?.resetPointer?.();
      drawStill();
    };

    function getSize(): [number, number] {
      const ratio = window.devicePixelRatio || 1;
      return [
        Math.max(1, Math.round(container.clientWidth * ratio)),
        Math.max(1, Math.round(container.clientHeight * ratio))
      ];
    }

    async function start() {
      if (attempted || starting || disposed) return;
      attempted = true;
      starting = true;
      try {
        const [{createGpuDevice, acquireGpuSlot}, loaded] = await Promise.all([
          import('./device.ts'),
          scene()
        ]);
        if (disposed || !visible) return;
        releaseSlot = await acquireGpuSlot();
        if (disposed || !visible) return;
        device = await createGpuDevice(canvas);
        if (disposed || !visible || !device) return;
        const context = device.getDefaultCanvasContext();
        await context.initialized;
        if (disposed || !visible) return;
        const [width, height] = getSize();
        context.setDrawingBufferSize(width, height);
        if (loaded.parameters) {
          valuesRef.current = {...getDefaultValues(loaded.parameters), ...valuesRef.current};
          setValues(valuesRef.current);
          setModuleParameters(loaded.parameters);
        }
        instance = await loaded.default({
          device,
          canvas,
          width,
          height,
          theme: theme ?? getTheme(),
          reducedMotion: motionQuery.matches,
          seed,
          parameters: {...valuesRef.current},
          reportReadout: text => {
            if (!disposed) setReadout(text);
          },
          // Values the pointer gesture wrote: keep the sliders on them.
          reportParameters: changed => {
            if (disposed) return;
            valuesRef.current = {...valuesRef.current, ...changed};
            setValues(valuesRef.current);
          }
        });
        if (disposed || !visible) return;
        if (!(await renderValidatedFrame(device, () => instance?.render(0)))) {
          throw new Error('First frame failed GPU validation');
        }
        if (disposed || !visible) return;
        const activeDevice = device;
        void activeDevice.lost.then(info => {
          if (!disposed && device === activeDevice) fail(info);
        });
        pointer?.attach(instance, loaded.parameters ?? []);
        setBackend(device.type === 'webgpu' ? 'WebGPU' : 'WebGL2');
        setIsReady(true);
        if (!motionQuery.matches) frame = requestAnimationFrame(tick);
      } catch (error) {
        if (!disposed) fail(error);
      } finally {
        starting = false;
        // Async device creation/compilation must finish before destroying its device.
        if (disposed || !visible || !instance) disposeResources();
      }
    }

    // A fresh canvas is needed when changing themes or when a released scene becomes visible again.
    function restart() {
      if (disposed) return;
      setIsReady(false);
      setGeneration(value => value + 1);
    }

    const intersectionObserver = new IntersectionObserver(
      entries => {
        const wasVisible = visible;
        visible = entries.some(entry => entry.isIntersecting);
        if (visible) void start();
        else if (wasVisible && attempted) restart();
      },
      {rootMargin: pointerSurface === 'card' ? '0px' : '120px'}
    );
    intersectionObserver.observe(container);
    const resizeObserver = new ResizeObserver(() => {
      if (!instance || starting) return;
      instance.resize?.(...getSize());
      drawStill();
    });
    resizeObserver.observe(container);
    const themeObserver = new MutationObserver(restart);
    if (!theme)
      themeObserver.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ['data-theme']
      });
    motionQuery.addEventListener('change', restart);

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      pointer?.destroy();
      intersectionObserver.disconnect();
      resizeObserver.disconnect();
      themeObserver.disconnect();
      motionQuery.removeEventListener('change', restart);
      applyParametersRef.current = null;
      resetPointerRef.current = null;
      canvas.remove();
      if (!starting) disposeResources();
    };
  }, [scene, seed, forwardsPointer, pointerSurface, theme, generation]);

  const changeParameter = (id: string, value: number) => {
    valuesRef.current = {...valuesRef.current, [id]: value};
    setValues(valuesRef.current);
    applyParametersRef.current?.({[id]: value});
  };

  const resetParameters = () => {
    const defaults = getDefaultValues(activeParameters);
    valuesRef.current = {...valuesRef.current, ...defaults};
    setValues(valuesRef.current);
    applyParametersRef.current?.(defaults);
    resetPointerRef.current?.();
  };

  // Before the reader's first move the chip names the gesture; after it, the scene's readout.
  const chipText = gesture.hint && !gesture.hasMoved ? gesture.hint : readout;

  const canvasBlock = (
    <div
      ref={containerRef}
      className={className ? `gpu-canvas ${className}` : 'gpu-canvas'}
      role="img"
      aria-label={readout && isReady ? `${label}. ${readout}.` : label}
      data-ready={isReady ? 'true' : 'false'}
      data-interactive={forwardsPointer ? 'true' : undefined}
      style={{
        position: 'relative',
        overflow: 'hidden',
        width: '100%',
        aspectRatio: String(aspectRatio),
        background: 'var(--preview-background, #070b14)'
      }}
    >
      <div style={{position: 'absolute', inset: 0, opacity: isReady ? 0 : 1, ...FADE}}>
        {fallback}
      </div>
      <div
        ref={canvasHostRef}
        aria-hidden="true"
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          opacity: isReady ? 1 : 0,
          ...FADE
        }}
      />
      {forwardsPointer && hint && isReady ? (
        <span
          className="gpu-canvas-hint"
          data-card={pointerSurface === 'card' || Boolean(gesture.hint) || undefined}
          data-gesture={gesture.hint ? 'true' : undefined}
          aria-hidden="true"
        >
          {pointerSurface === 'card'
            ? // Once the reader is steering, the whole chip belongs to the readout.
              (gesture.hasMoved && chipText ? [chipText] : [backend, chipText])
                .filter(Boolean)
                .join(' · ')
            : gesture.hint
              ? [chipText, gesture.isPinned ? 'pinned, click to release' : null]
                  .filter(Boolean)
                  .join(' · ')
              : 'move pointer'}
        </span>
      ) : null}
    </div>
  );

  if (!controls) return canvasBlock;
  return (
    <div className="gpu-canvas-frame">
      {canvasBlock}
      <GpuControls
        parameters={activeParameters}
        values={values}
        isLive={isReady}
        backend={backend}
        onChange={changeParameter}
        onReset={resetParameters}
      />
    </div>
  );
}

type GpuControlsProps = {
  parameters: GpuSceneParameters;
  values: ParameterValues;
  isLive: boolean;
  backend: string | null;
  onChange(id: string, value: number): void;
  onReset(): void;
};

/** Slider strip under a canvas. Disabled (but visible) until the canvas is live. */
function GpuControls({
  parameters,
  values,
  isLive,
  backend,
  onChange,
  onReset
}: GpuControlsProps): ReactElement {
  const identifier = useId();
  const isChanged = parameters.some(parameter => values[parameter.id] !== parameter.default);
  // Sliders labelled "Data: ..." vary the synthetic input, not a prop of the operation.
  const dataParameters = parameters
    .filter(parameter => parameter.label.startsWith('Data: '))
    .map(parameter => ({...parameter, label: parameter.label.slice('Data: '.length)}));
  const propParameters = parameters.filter(parameter => !parameter.label.startsWith('Data: '));
  return (
    <div className="gpu-controls" data-live={isLive ? 'true' : 'false'}>
      <div className="gpu-controls-groups">
        {[
          {title: 'Props', items: propParameters, isData: false},
          {title: 'Sample data', items: dataParameters, isData: true}
        ].map(group =>
          group.items.length === 0 ? null : (
            <div
              className="gpu-controls-group"
              data-group={group.isData ? 'data' : 'props'}
              key={group.title}
            >
              {propParameters.length > 0 && dataParameters.length > 0 ? (
                <span className="gpu-controls-title">{group.title}</span>
              ) : null}
              <div className="gpu-controls-sliders">
                {group.items.map(parameter => {
                  const value = values[parameter.id] ?? parameter.default;
                  const inputId = `${identifier}-${parameter.id}`;
                  if (parameter.options) {
                    return (
                      <div className="gpu-control gpu-control--options" key={parameter.id}>
                        <span id={inputId}>{parameter.label}</span>
                        <div className="gpu-segments" role="group" aria-labelledby={inputId}>
                          {parameter.options.map((option, index) => (
                            <button
                              key={option}
                              type="button"
                              aria-pressed={index === Math.round(value)}
                              disabled={!isLive}
                              onClick={() => onChange(parameter.id, index)}
                            >
                              {option}
                            </button>
                          ))}
                        </div>
                      </div>
                    );
                  }
                  return (
                    <div className="gpu-control" key={parameter.id}>
                      <label htmlFor={inputId}>{parameter.label}</label>
                      <output htmlFor={inputId}>{formatValue(parameter, value)}</output>
                      <input
                        id={inputId}
                        type="range"
                        min={parameter.min}
                        max={parameter.max}
                        step={parameter.step}
                        value={value}
                        disabled={!isLive}
                        onChange={event =>
                          onChange(parameter.id, Number(event.currentTarget.value))
                        }
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          )
        )}
      </div>
      <div className="gpu-controls-status">
        <span className="gpu-badge" data-live={isLive ? 'true' : 'false'}>
          {isLive ? `live · ${backend ?? 'GPU'}` : 'drawn'}
        </span>
        {parameters.length > 0 ? (
          <button
            type="button"
            className="gpu-reset"
            disabled={!isLive || !isChanged}
            onClick={onReset}
          >
            reset
          </button>
        ) : null}
      </div>
    </div>
  );
}

/**
 * Renders one frame and reports whether the GPU accepted it. On WebGPU the frame runs inside a
 * validation error scope; WebGL2 has no equivalent, so it is trusted.
 */
async function renderValidatedFrame(device: Device, renderFrame: () => void): Promise<boolean> {
  const gpuDevice = device.type === 'webgpu' ? (device.handle as GPUDevice) : null;
  if (!gpuDevice) {
    renderFrame();
    return true;
  }
  gpuDevice.pushErrorScope('validation');
  let error: GPUError | null;
  try {
    renderFrame();
  } finally {
    error = await gpuDevice.popErrorScope();
  }
  return error === null;
}
