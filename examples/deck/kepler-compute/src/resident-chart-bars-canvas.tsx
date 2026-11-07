// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useMemo, useRef, useState, type ComponentProps} from 'react';
import {displayBinKey, formatNumber, type ChartBin, type ChartRenderer} from '@kepler.gl/charts';
import {
  getBarIndexAtPixel,
  getBarRangeAtPixel,
  getDominantBarIndex,
  makeBarLayout
} from './resident-chart-bars-layout';

export type ResidentChartBarsCanvasProps = {
  bins: readonly ChartBin[];
  selectedKey?: string;
  onSelect?: ComponentProps<typeof ChartRenderer>['onSelect'];
  readOnly?: boolean;
  height?: number;
};

const LEFT_PADDING = 48;
const RIGHT_PADDING = 12;
const TOP_PADDING = 10;
const BOTTOM_PADDING = 22;
const MINIMUM_BRUSH_PIXELS = 3;

/** Canvas bars for thousands of bins; hover, click and drag-brush selection mirror the SVG chart. */
export function ResidentChartBarsCanvas({
  bins,
  selectedKey,
  onSelect,
  readOnly = false,
  height = 200
}: ResidentChartBarsCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  const [hover, setHover] = useState<{index: number; x: number; y: number} | undefined>();
  const [brush, setBrush] = useState<readonly [number, number] | undefined>();
  const plotWidth = Math.max(1, width - LEFT_PADDING - RIGHT_PADDING);
  const plotHeight = height - TOP_PADDING - BOTTOM_PADDING;
  const interactive = !readOnly && Boolean(onSelect);

  const values = useMemo(() => Float64Array.from(bins, bin => bin.value), [bins]);
  const counts = useMemo(() => Float64Array.from(bins, bin => bin.count), [bins]);
  const layout = useMemo(
    () => makeBarLayout(values, counts, plotWidth),
    [values, counts, plotWidth]
  );
  const domain = useMemo(() => {
    let minimum = 0;
    let maximum = 0;
    for (const value of values) {
      if (value < minimum) minimum = value;
      if (value > maximum) maximum = value;
    }
    return [minimum, maximum === minimum ? minimum + 1 : maximum] as const;
  }, [values]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;
    const observer = new ResizeObserver(entries =>
      setWidth(Math.max(120, Math.floor(entries[0].contentRect.width)))
    );
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const [minimum, maximum] = domain;
    const toY = (value: number) =>
      TOP_PADDING + plotHeight - ((value - minimum) / (maximum - minimum)) * plotHeight;
    const baseline = toY(0);
    context.font = '10px sans-serif';
    context.fillStyle = '#c4cfdd';
    context.strokeStyle = 'rgba(102, 117, 139, 0.35)';
    context.textAlign = 'right';
    for (let tick = 0; tick <= 4; tick++) {
      const value = minimum + ((maximum - minimum) * tick) / 4;
      const y = Math.round(toY(value)) + 0.5;
      context.beginPath();
      context.moveTo(LEFT_PADDING, y);
      context.lineTo(LEFT_PADDING + plotWidth, y);
      context.stroke();
      context.fillText(formatNumber(value), LEFT_PADDING - 6, y + 3);
    }
    context.textAlign = 'center';
    context.fillText(displayBinKey(bins[0]?.key ?? ''), LEFT_PADDING + 30, height - 6);
    context.fillText(
      displayBinKey(bins[bins.length - 1]?.key ?? ''),
      LEFT_PADDING + plotWidth - 30,
      height - 6
    );
    const wide = layout.barWidth >= 1;
    const gap = layout.barWidth >= 4 ? 1 : 0;
    context.globalAlpha = 0.9;
    for (const column of layout.columns) {
      const bin =
        bins[
          wide
            ? column.firstIndex
            : getDominantBarIndex(values, [column.firstIndex, column.lastIndex])
        ];
      const top = Math.min(baseline, toY(column.value));
      const barHeight = Math.max(1, Math.abs(toY(column.value) - baseline));
      context.fillStyle = bin.color;
      context.fillRect(
        LEFT_PADDING + (wide ? column.firstIndex * layout.barWidth : column.column),
        top,
        wide ? Math.max(1, layout.barWidth - gap) : 1,
        barHeight
      );
    }
    context.globalAlpha = 1;
    context.strokeStyle = '#aab6c6';
    context.beginPath();
    context.moveTo(LEFT_PADDING, Math.round(baseline) + 0.5);
    context.lineTo(LEFT_PADDING + plotWidth, Math.round(baseline) + 0.5);
    context.stroke();
    const selectedIndex = selectedKey ? bins.findIndex(bin => bin.key === selectedKey) : -1;
    context.strokeStyle = '#ffffff';
    context.lineWidth = 2;
    if (selectedIndex >= 0) {
      const x = LEFT_PADDING + selectedIndex * layout.barWidth;
      context.strokeRect(x, TOP_PADDING, Math.max(1, layout.barWidth), plotHeight);
    }
    if (brush) {
      context.fillStyle = 'rgba(255, 255, 255, 0.2)';
      const start = Math.min(brush[0], brush[1]);
      context.fillRect(
        LEFT_PADDING + start,
        TOP_PADDING,
        Math.abs(brush[1] - brush[0]),
        plotHeight
      );
    }
    context.lineWidth = 1;
  }, [bins, values, layout, domain, width, height, plotWidth, plotHeight, selectedKey, brush]);

  const pixelFor = (clientX: number) =>
    clientX - (canvasRef.current?.getBoundingClientRect().left ?? 0) - LEFT_PADDING;
  const hoverIndex = hover ? getBarIndexAtPixel(layout, hover.x) : -1;
  const hoverRange = hover ? getBarRangeAtPixel(layout, hover.x) : undefined;
  const hoverBin = hoverRange ? bins[getDominantBarIndex(values, hoverRange)] : undefined;
  const hoverCount = hoverRange
    ? layout.columns
        .slice(
          layout.barWidth >= 1 ? hoverRange[0] : Math.floor(hover!.x),
          (layout.barWidth >= 1 ? hoverRange[0] : Math.floor(hover!.x)) + 1
        )
        .reduce((total, column) => total + column.count, 0)
    : 0;

  function finishBrush(startPixel: number, endPixel: number) {
    const first = getBarIndexAtPixel(layout, Math.min(startPixel, endPixel));
    const last = getBarIndexAtPixel(layout, Math.max(startPixel, endPixel));
    const firstBin = bins[first];
    const lastBin = bins[last];
    if (first < 0 || last < 0 || !firstBin.filterValue || !lastBin.filterValue) return;
    const low = firstBin.filterValue[0];
    const high = lastBin.filterValue[lastBin.filterValue.length - 1];
    onSelect?.(first === last ? firstBin.key : `${firstBin.key}…${lastBin.key}`, {
      filterValue: first === last ? firstBin.filterValue : [low, high]
    });
  }

  return (
    <div ref={containerRef} className="resident-bars-canvas" style={{position: 'relative'}}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={`Bar chart with ${bins.length.toLocaleString()} bars; hover to inspect${interactive ? ', drag to select a range' : ''}`}
        style={{width, height, display: 'block', cursor: interactive ? 'crosshair' : 'default'}}
        onPointerMove={event => {
          const x = pixelFor(event.clientX);
          const rect = event.currentTarget.getBoundingClientRect();
          setHover({index: -1, x, y: event.clientY - rect.top});
          if (brush) setBrush([brush[0], x]);
        }}
        onPointerLeave={() => setHover(undefined)}
        onPointerDown={event => {
          if (!interactive) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          const x = pixelFor(event.clientX);
          setBrush([x, x]);
        }}
        onPointerUp={event => {
          if (!brush) return;
          const end = pixelFor(event.clientX);
          const start = brush[0];
          setBrush(undefined);
          finishBrush(start, Math.abs(end - start) < MINIMUM_BRUSH_PIXELS ? start : end);
        }}
      />
      {hover && hoverBin && hoverIndex >= 0 && (
        <div
          role="tooltip"
          className="resident-bars-canvas-tooltip"
          style={{
            position: 'absolute',
            left: Math.min(width - 160, hover.x + LEFT_PADDING + 8),
            top: Math.max(0, hover.y - 36),
            pointerEvents: 'none',
            padding: '3px 6px',
            background: 'rgba(15, 20, 28, 0.92)',
            color: '#edf2f7',
            fontSize: 11,
            whiteSpace: 'nowrap'
          }}
        >
          {displayBinKey(hoverBin.key)} · {formatNumber(hoverBin.value)} (
          {hoverCount.toLocaleString()} {hoverCount === 1 ? 'row' : 'rows'})
        </div>
      )}
    </div>
  );
}
