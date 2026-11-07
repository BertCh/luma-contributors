// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// <GpuCanvas>: a live luma.gl picture over a static fallback. Prerendering emits the fallback only.
// In the browser the scene starts when scrolled into view, pauses when offscreen, and the canvas
// fades in over the fallback once its first frame is drawn. Any failure keeps the fallback.

import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactElement,
  type ReactNode
} from 'react';
import type {Device} from '@luma.gl/core';
import type {GpuSceneInstance, GpuSceneLoader, GpuTheme} from './types.ts';

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
};

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
  interactive = false
}: GpuCanvasProps): ReactElement {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isReady, setIsReady] = useState(false);
  // Bumped when a canvas is released offscreen, so the next start gets a fresh element.
  const [generation, setGeneration] = useState(0);

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
        seed
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
        if (isDestroyed) {
          releaseResources();
          return;
        }
        const canvasContext = device.getDefaultCanvasContext();
        await canvasContext.initialized;
        const [initialWidth, initialHeight] = getSize();
        canvasContext.setDrawingBufferSize(Math.max(initialWidth, 1), Math.max(initialHeight, 1));
        sceneFunction = loaded.default;
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
    if (interactive) {
      container.addEventListener('pointermove', handlePointerMove);
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
      container.removeEventListener('pointerleave', handlePointerLeave);
      releaseResources();
    };
  }, [scene, seed, interactive]);

  return (
    <div
      ref={containerRef}
      className={className ? `gpu-canvas ${className}` : 'gpu-canvas'}
      role="img"
      aria-label={label}
      data-ready={isReady ? 'true' : 'false'}
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
