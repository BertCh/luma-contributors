// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useRef, useState, type ComponentType} from 'react';
import {ExportImageModalFactory} from '@kepler.gl/components';
import {
  captureTiledImage,
  getTileViewState,
  MAXIMUM_TILE_DIMENSION,
  type TileViewState
} from './native-tiled-capture';
import {
  formatByteCount,
  formatTiledProgress,
  getTiledProgressFraction,
  isTiledCaptureCancelled,
  summarizeTiledImage
} from './native-tiled-capture-progress';
import {getAllowedTiledScales} from './tiled-projection-offset';
import {getMediaExportHost} from './native-video-export';
import {createCanvasFrameCapturer} from './webgpu-canvas-capture';

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

type ExportImageModalProps = {
  mapW: number;
  mapH: number;
  exportImage: {fileName?: string};
} & Record<string, unknown>;

/**
 * Kepler's Export Image modal with an added "Large image (tiled)" section. Native export is limited
 * to 8,192 px per side (single pass); this section stitches tiles of the live map up to 32,768 px per
 * side and 128 MP and downloads the PNG with the modal's file name. It reuses the modal's map size
 * (mapW x mapH) and scales it, so the screen aspect ratio is kept. Basemap and DOM overlays are not in
 * tiles, and pitched views are refused (see tiled-projection-offset.ts).
 *
 * Progress shows tile count and an ETA; Cancel (or closing the modal) aborts between tiles and the
 * original view state is restored. Remaining limits: a tile in flight is not interrupted, rotation is
 * supported but pitch is not, and browsers may still refuse the largest stitch canvases.
 */
export function ExportImageModalTiledFactory() {
  const NativeModal = ExportImageModalFactory() as unknown as ComponentType<ExportImageModalProps>;
  return function ExportImageModalWithTiledCapture(props: ExportImageModalProps) {
    const {mapW, mapH, exportImage} = props;
    const scales = getAllowedTiledScales(mapW || 1, mapH || 1);
    const [scale, setScale] = useState(4);
    const [status, setStatus] = useState('');
    const [busy, setBusy] = useState(false);
    const [progress, setProgress] = useState(0);
    const abort = useRef<AbortController | null>(null);
    const mounted = useRef(true);
    useEffect(() => {
      mounted.current = true;
      return () => {
        // Closing the modal cancels a running export so the map view is restored.
        mounted.current = false;
        abort.current?.abort();
      };
    }, []);
    const report = (message: string) => {
      if (mounted.current) setStatus(message);
    };
    const canvasForSummary = getMediaExportHost()?.getCanvas();
    const previewTileWidth = Math.min(canvasForSummary?.width ?? 1, MAXIMUM_TILE_DIMENSION);
    const previewTileHeight = Math.min(canvasForSummary?.height ?? 1, MAXIMUM_TILE_DIMENSION);
    const preview = canvasForSummary
      ? summarizeTiledImage(
          Math.round(previewTileWidth * scale),
          Math.round(previewTileHeight * scale),
          previewTileWidth,
          previewTileHeight
        )
      : null;
    const run = async () => {
      if (abort.current) return;
      const host = getMediaExportHost();
      const canvas = host?.getCanvas();
      if (!host || !canvas) return report('The map canvas is unavailable.');
      const original = host.getViewState();
      if (original.pitch) return report('Large tiled images need a top-down view (pitch 0).');
      const controller = new AbortController();
      abort.current = controller;
      setBusy(true);
      setProgress(0);
      const startTime = Date.now();
      const tileWidth = Math.min(canvas.width, MAXIMUM_TILE_DIMENSION);
      const tileHeight = Math.min(canvas.height, MAXIMUM_TILE_DIMENSION);
      const width = Math.round(tileWidth * scale);
      const height = Math.round(tileHeight * scale);
      const full: TileViewState = {...original, zoom: original.zoom + Math.log2(scale)};
      const capturer = createCanvasFrameCapturer(host);
      try {
        const blob = await captureTiledImage({
          width,
          height,
          tileWidth,
          tileHeight,
          isCurrent: () => !controller.signal.aborted,
          signal: controller.signal,
          onProgress: (done, count) => {
            report(formatTiledProgress(done, count, Date.now() - startTime));
            if (mounted.current) setProgress(getTiledProgressFraction(done, count));
          },
          renderTile: async tile => {
            host.setViewState({...original, ...getTileViewState(full, tile, width, height)});
            await host.waitForRender();
            return (await capturer.capture(host.getCanvas() ?? canvas)).source;
          }
        });
        downloadBlob(blob, `${exportImage.fileName || 'kepler-map'}-large.png`);
        report(`Saved ${width}x${height} PNG. Basemap and DOM overlays are not included.`);
      } catch (error) {
        report(
          isTiledCaptureCancelled(error)
            ? 'Cancelled. The map view was restored.'
            : error instanceof Error
              ? error.message
              : String(error)
        );
      } finally {
        host.setViewState(original);
        abort.current = null;
        if (mounted.current) setBusy(false);
      }
    };
    return (
      <>
        <NativeModal {...props} />
        <section className="image-option-section" aria-label="Large image (tiled)">
          <div className="image-option-section-title">Large image (tiled)</div>
          <p>
            Beyond the single-pass limit: stitches live-map tiles (deck layers only), up to 32,768
            px per side.
          </p>
          {preview ? (
            <p>
              {'summary' in preview
                ? `${preview.summary.width} x ${preview.summary.height} px, ${preview.summary.megapixels} MP, ${preview.summary.tileCount} tiles (${preview.summary.columns} x ${preview.summary.rows}), about ${formatByteCount(preview.summary.stitchBytes)} for the stitched image`
                : preview.error}
            </p>
          ) : null}
          <label>
            Scale{' '}
            <select value={scale} onChange={event => setScale(Number(event.target.value))}>
              {scales.map(option => (
                <option key={option} value={option}>
                  {option}x ({Math.round(mapW * option)} x {Math.round(mapH * option)})
                </option>
              ))}
            </select>
          </label>{' '}
          <button
            disabled={!scales.length || busy || (preview !== null && 'error' in preview)}
            onClick={() => void run()}
          >
            Save large PNG
          </button>{' '}
          <button disabled={!busy} onClick={() => abort.current?.abort()}>
            Cancel
          </button>
          {busy ? <progress value={progress} max={1} aria-label="Tiled export progress" /> : null}
          {status ? <p role="status">{status}</p> : null}
        </section>
      </>
    );
  };
}
ExportImageModalTiledFactory.deps = [] as const;
