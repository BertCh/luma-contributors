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
  /** Forward pointer position to the scene. */
  interactive?: boolean;
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
  const attribute = document.documentElement.dataset.theme;
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
  parameters
}: GpuCanvasProps): ReactElement {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
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
  const forwardsPointer = interactive || controls;

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !canvasRef.current) return undefined;
    let canvas = canvasRef.current;
    let releaseSlot: (() => void) | null = null;

    let isDestroyed = false;
    let isVisible = false;
    let isStarting = false;
    let frameHandle = 0;
    let instance: GpuSceneInstance | null = null;
    let device: Awaited<ReturnType<typeof import('./device.ts').createGpuDevice>> = null;
    let sceneFunction: Awaited<ReturnType<GpuSceneLoader>>['default'] | null = null;
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

    const releaseResources = () => {
      instance?.destroy();
      instance = null;
      device?.destroy();
      device = null;
      releaseSlot?.();
      releaseSlot = null;
    };

    const getSize = (): [number, number] => {
      const ratio = window.devicePixelRatio || 1;
      return [
        Math.round(container.clientWidth * ratio),
        Math.round(container.clientHeight * ratio)
      ];
    };

    applyParametersRef.current = changed => {
      instance?.setParameters?.(changed);
      // A still frame (reduced motion) has no loop to show the change.
      if (instance && isVisible && motionQuery.matches) renderFrame(0);
    };

    const renderFrame = (time: number) => {
      try {
        instance?.render(time);
      } catch (error) {
        // biome-ignore lint/suspicious/noConsole: diagnostic output
        console.warn('GpuCanvas render failed', error);
        stopLoop();
        instance = null;
        setIsReady(false);
      }
    };

    const stopLoop = () => {
      cancelAnimationFrame(frameHandle);
      frameHandle = 0;
    };

    const tick = (time: number) => {
      frameHandle = 0;
      if (isDestroyed || !isVisible || !instance) return;
      renderFrame(time);
      if (!motionQuery.matches) frameHandle = requestAnimationFrame(tick);
    };

    const startLoop = () => {
      if (frameHandle || !instance || !isVisible) return;
      frameHandle = requestAnimationFrame(tick);
    };

    const createInstance = async () => {
      if (!device || !sceneFunction) return;
      instance?.destroy();
      instance = null;
      const [width, height] = getSize();
      const created = await sceneFunction({
        device,
        canvas,
        width,
        height,
        theme: getTheme(),
        reducedMotion: motionQuery.matches,
        seed,
        parameters: {...valuesRef.current}
      });
      if (isDestroyed) {
        created.destroy();
        return;
      }
      instance = created;
    };

    const start = async () => {
      if (isStarting || device) return;
      isStarting = true;
      try {
        const [{createGpuDevice, acquireGpuSlot}, loaded] = await Promise.all([
          import('./device.ts'),
          scene()
        ]);
        const release = await acquireGpuSlot();
        if (isDestroyed || !isVisible) {
          release();
          return;
        }
        releaseSlot = release;
        canvas = canvasRef.current ?? canvas;
        device = await createGpuDevice(canvas);
        if (!device) {
          releaseResources();
          return;
        }
        setBackend(device.type === 'webgpu' ? 'WebGPU' : 'WebGL2');
        if (isDestroyed) {
          releaseResources();
          return;
        }
        const canvasContext = device.getDefaultCanvasContext();
        await canvasContext.initialized;
        const [initialWidth, initialHeight] = getSize();
        canvasContext.setDrawingBufferSize(Math.max(initialWidth, 1), Math.max(initialHeight, 1));
        sceneFunction = loaded.default;
        if (loaded.parameters) {
          valuesRef.current = {...getDefaultValues(loaded.parameters), ...valuesRef.current};
          setValues(valuesRef.current);
          setModuleParameters(loaded.parameters);
        }
        await createInstance();
        if (isDestroyed || !instance) return;
        if (!isVisible) {
          releaseResources();
          setGeneration(value => value + 1);
          return;
        }
        // Some adapters (software or blocklisted GPUs) accept the device but cannot present to the
        // canvas. Validate the first frame so those keep the static fallback instead of a blank canvas.
        const isFirstFrameValid = await renderValidatedFrame(device, () => renderFrame(0));
        if (!isFirstFrameValid) {
          // biome-ignore lint/suspicious/noConsole: diagnostic output
          console.warn(
            'GpuCanvas: the GPU could not present this canvas, keeping the static fallback'
          );
          releaseResources();
          return;
        }
        if (isDestroyed || !instance) return;
        requestAnimationFrame(() => {
          if (!isDestroyed && instance) setIsReady(true);
        });
        startLoop();
      } catch (error) {
        // biome-ignore lint/suspicious/noConsole: diagnostic output
        console.warn('GpuCanvas could not start, keeping the static fallback', error);
        setIsReady(false);
        releaseResources();
      } finally {
        isStarting = false;
      }
    };

    const intersectionObserver = new IntersectionObserver(
      entries => {
        isVisible = entries.some(entry => entry.isIntersecting);
        if (isVisible) {
          if (device) startLoop();
          else void start();
        } else {
          stopLoop();
          if (device) {
            releaseResources();
            setIsReady(false);
            setGeneration(value => value + 1);
          }
        }
      },
      {rootMargin: '120px'}
    );
    intersectionObserver.observe(container);

    const resizeObserver = new ResizeObserver(() => {
      if (!instance) return;
      const [width, height] = getSize();
      instance.resize?.(width, height);
      if (motionQuery.matches && isVisible) renderFrame(0);
    });
    resizeObserver.observe(container);

    const rebuild = () => {
      if (!device) return;
      void createInstance().then(() => {
        if (instance && isVisible) {
          renderFrame(0);
          startLoop();
        }
      });
    };
    const themeObserver = new MutationObserver(rebuild);
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme']
    });
    motionQuery.addEventListener('change', rebuild);

    const handlePointerMove = (event: PointerEvent) => {
      const bounds = container.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      instance?.setPointer?.([
        (event.clientX - bounds.left) * ratio,
        (event.clientY - bounds.top) * ratio
      ]);
    };
    const handlePointerLeave = () => instance?.setPointer?.(null);
    if (forwardsPointer) {
      container.addEventListener('pointermove', handlePointerMove);
      container.addEventListener('pointerdown', handlePointerMove);
      container.addEventListener('pointerleave', handlePointerLeave);
    }

    return () => {
      isDestroyed = true;
      stopLoop();
      intersectionObserver.disconnect();
      resizeObserver.disconnect();
      themeObserver.disconnect();
      motionQuery.removeEventListener('change', rebuild);
      container.removeEventListener('pointermove', handlePointerMove);
      container.removeEventListener('pointerdown', handlePointerMove);
      container.removeEventListener('pointerleave', handlePointerLeave);
      applyParametersRef.current = null;
      releaseResources();
    };
  }, [scene, seed, forwardsPointer]);

  // The backend label only makes sense while the canvas is live.
  useEffect(() => {
    if (!isReady) setBackend(null);
  }, [isReady]);

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
  };

  const canvasBlock = (
    <div
      ref={containerRef}
      className={className ? `gpu-canvas ${className}` : 'gpu-canvas'}
      role="img"
      aria-label={label}
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
      <canvas
        key={generation}
        ref={canvasRef}
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
      {forwardsPointer && isReady ? (
        <span className="gpu-canvas-hint" aria-hidden="true">
          move pointer
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
  renderFrame();
  const error = await gpuDevice.popErrorScope();
  return error === null;
}
