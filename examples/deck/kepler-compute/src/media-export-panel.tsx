// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useRef, useState} from 'react';
import type {SandboxBundleLimits} from './sandbox-bundle';
import type {SandboxCallbacks} from './sandbox-state';
import {serializeBundleWithNativeSources} from './bundle-native-embed';
import {buildStandaloneHtmlWithViewer} from './standalone-html-export';
import {
  captureVideo,
  detectVideoExportSupport,
  getMediaExportHost,
  type VideoKeyframe
} from './native-video-export';
import {captureVideoWithWebCodecs} from './native-webcodecs-capture';
import {createCanvasFrameCapturer} from './webgpu-canvas-capture';
import {
  captureTiledImage,
  getTileViewState,
  MAXIMUM_TILE_DIMENSION,
  type TileViewState
} from './native-tiled-capture';

const BUNDLE_LIMITS: SandboxBundleLimits = {
  maximumDocumentBytes: 96 * 1024 * 1024,
  maximumSourceBytes: 64 * 1024 * 1024,
  maximumRows: 1_000_000,
  maximumSources: 16
};

function downloadBlob(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  try {
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = fileName;
    anchor.click();
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}

export function MediaExportPanel({
  callbacks,
  disabled,
  onBusyChange
}: {
  callbacks: SandboxCallbacks;
  disabled: boolean;
  onBusyChange(busy: boolean): void;
}) {
  const [status, setStatus] = useState('');
  const [seconds, setSeconds] = useState(5);
  const [zoomChange, setZoomChange] = useState(1);
  const [bearingChange, setBearingChange] = useState(0);
  const [sweepTime, setSweepTime] = useState(false);
  const [tiledScale, setTiledScale] = useState(8);
  const [videoFormat, setVideoFormat] = useState<'auto' | 'mp4' | 'webm' | 'recorder'>('auto');
  const abort = useRef<AbortController | null>(null);
  const support = detectVideoExportSupport();
  const run = async (action: (signal: AbortSignal) => Promise<void>) => {
    if (disabled || abort.current) return;
    const controller = new AbortController();
    abort.current = controller;
    onBusyChange(true);
    setStatus('');
    try {
      await action(controller.signal);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      abort.current = null;
      onBusyChange(false);
    }
  };
  return (
    <section aria-label="Image, video and HTML export">
      <h3>Large image, video and HTML</h3>
      <p className="hint">
        Large PNG: tiles the live map up to 32,768 px per side and 128 MP. Video: {support.message}{' '}
        Capture runs in real time; the viewport and animation time are restored afterwards.
      </p>
      <div className="actions">
        <label>
          Seconds
          <input
            type="number"
            min={1}
            max={120}
            value={seconds}
            onChange={event => setSeconds(Number(event.target.value))}
          />
        </label>
        <label>
          Zoom change
          <input
            type="number"
            step={0.5}
            value={zoomChange}
            onChange={event => setZoomChange(Number(event.target.value))}
          />
        </label>
        <label>
          Bearing change
          <input
            type="number"
            step={15}
            value={bearingChange}
            onChange={event => setBearingChange(Number(event.target.value))}
          />
        </label>
        <label>
          <input
            type="checkbox"
            checked={sweepTime}
            onChange={event => setSweepTime(event.target.checked)}
          />
          Sweep time bar
        </label>
        <label>
          Video encoder
          <select
            value={videoFormat}
            onChange={event => setVideoFormat(event.target.value as typeof videoFormat)}
          >
            <option value="auto">Auto (WebCodecs, else recorder)</option>
            <option value="mp4">WebCodecs MP4 (H.264)</option>
            <option value="webm">WebCodecs WebM (VP9/VP8)</option>
            <option value="recorder">MediaRecorder (real time)</option>
          </select>
        </label>
        <button
          disabled={disabled || (!support.supported && !support.webCodecs)}
          onClick={() =>
            void run(async signal => {
              const host = getMediaExportHost();
              if (!host) throw new Error('The map is not connected to media export.');
              const canvas = host.getCanvas();
              if (!canvas) throw new Error('The map canvas is unavailable.');
              const start = host.getViewState();
              const keyframes: VideoKeyframe[] = [
                {time: 0, viewState: start},
                {
                  time: seconds,
                  viewState: {
                    ...start,
                    zoom: start.zoom + zoomChange,
                    bearing: start.bearing + bearingChange
                  }
                }
              ];
              const even = (value: number) => Math.max(2, value - (value % 2));
              const common = {
                host,
                durationSeconds: seconds,
                fps: 30,
                width: even(canvas.width),
                height: even(canvas.height),
                keyframes,
                sweepAnimation: sweepTime,
                signal,
                onProgress: (done: number, count: number) =>
                  setStatus(`Recording frame ${done} of ${count}`)
              };
              let blob: Blob | null = null;
              if (videoFormat !== 'recorder' && support.webCodecs) {
                try {
                  blob = await captureVideoWithWebCodecs({
                    ...common,
                    format: videoFormat === 'auto' ? 'auto' : videoFormat
                  });
                } catch (error) {
                  // Explicit WebCodecs choices report the failure; Auto falls through to MediaRecorder.
                  if (videoFormat !== 'auto' || signal.aborted) throw error;
                }
              }
              blob ??= await captureVideo({...common, support});
              downloadBlob(blob, blob.type === 'video/mp4' ? 'kepler-map.mp4' : 'kepler-map.webm');
              setStatus(`Video saved (${(blob.size / 1024 / 1024).toFixed(1)} MiB).`);
            })
          }
        >
          Record video
        </button>
        <label>
          PNG scale
          <input
            type="number"
            min={1}
            max={32}
            value={tiledScale}
            onChange={event => setTiledScale(Number(event.target.value))}
          />
        </label>
        <button
          disabled={disabled}
          onClick={() =>
            void run(async signal => {
              const host = getMediaExportHost();
              const canvas = host?.getCanvas();
              if (!host || !canvas) throw new Error('The map canvas is unavailable.');
              const original = host.getViewState();
              const tileWidth = Math.min(canvas.width, MAXIMUM_TILE_DIMENSION);
              const tileHeight = Math.min(canvas.height, MAXIMUM_TILE_DIMENSION);
              const width = Math.round(tileWidth * tiledScale);
              const height = Math.round(tileHeight * tiledScale);
              const full: TileViewState = {
                ...original,
                zoom: original.zoom + Math.log2(tiledScale)
              };
              const capturer = createCanvasFrameCapturer(host);
              try {
                const blob = await captureTiledImage({
                  width,
                  height,
                  tileWidth,
                  tileHeight,
                  isCurrent: () => !signal.aborted,
                  onProgress: (done, count) => setStatus(`Rendered tile ${done} of ${count}`),
                  renderTile: async tile => {
                    host.setViewState({
                      ...original,
                      ...getTileViewState(full, tile, width, height)
                    });
                    await host.waitForRender();
                    return (await capturer.capture(host.getCanvas() ?? canvas)).source;
                  }
                });
                downloadBlob(blob, 'kepler-map-large.png');
                setStatus(
                  `Saved ${width}x${height} PNG. Basemap and DOM overlays are not included.`
                );
              } finally {
                host.setViewState(original);
              }
            })
          }
        >
          Save large PNG
        </button>
        <button
          disabled={disabled}
          onClick={() =>
            void run(async () => {
              const {
                serialized: bundle,
                skippedMessage,
                embeddedMessage
              } = await serializeBundleWithNativeSources(callbacks, BUNDLE_LIMITS);
              const {html, usedViewer} = await buildStandaloneHtmlWithViewer(bundle, {
                viewerUrl: `${import.meta.env.BASE_URL}standalone-viewer.html`
              });
              downloadBlob(new Blob([html], {type: 'text/html'}), 'kepler-workspace.html');
              setStatus(
                `${
                  usedViewer
                    ? 'Standalone HTML saved with the embedded luma.gl viewer (works offline; restores layers, styles, filters and legends, no basemap tiles).'
                    : 'Standalone HTML saved with the full bundle only: run "yarn build:viewer" and serve dist/standalone-viewer.html to embed the offline viewer.'
                } ${embeddedMessage} ${skippedMessage}`
                  .replace(/\s+/g, ' ')
                  .trim()
              );
            })
          }
        >
          Save standalone HTML
        </button>
        <button disabled={!disabled} onClick={() => abort.current?.abort()}>
          Cancel
        </button>
      </div>
      {status ? <p role="status">{status}</p> : null}
    </section>
  );
}
