// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useId, type ComponentProps, type ReactNode} from 'react';
import {displayBinKey, formatNumber, type ChartBin, type ChartRenderer} from '@kepler.gl/charts';
import {makeResidentChartSeriesLayout} from './resident-chart-series-layout';
import './resident-chart-series-view.css';

import {toRadians} from '@math.gl/core';

export type ResidentChartSeriesViewProps = {
  bins: readonly ChartBin[];
  type: 'barChart' | 'horizontalBar' | 'lineChart';
  colors?: readonly string[];
  /** Logical category/time axis, even for horizontal bars. */
  xLabel?: string;
  /** Logical measure axis, even for horizontal bars. */
  yLabel?: string;
  selectedKey?: string;
  onSelect?: ComponentProps<typeof ChartRenderer>['onSelect'];
  readOnly?: boolean;
};

const AGGREGATE_COLOR = '#aab6c6';
function shortenLabel(label: string, limit: number): string {
  return label.length > limit ? `${label.slice(0, limit - 1)}…` : label;
}

/** Separate series preserve native statistics while selecting only their parent category/range. */
export function ResidentChartSeriesView({
  bins,
  type,
  colors,
  xLabel = 'Group',
  yLabel = 'Value',
  selectedKey,
  onSelect,
  readOnly = false
}: ResidentChartSeriesViewProps) {
  const identifier = useId();
  const layout = makeResidentChartSeriesLayout(bins, colors);
  const horizontal = type === 'horizontalBar';
  const line = type === 'lineChart';
  const slotSize = line ? 44 : Math.max(48, layout.series.length * 14 + 12);
  const labels = bins.map(bin => shortenLabel(displayBinKey(bin.key), 24));
  // Reserve room for bounded labels at the existing 10px font size, including their rotated extent.
  const labelWidth = labels.reduce((maximum, label) => Math.max(maximum, label.length * 6), 0);
  const left = horizontal
    ? Math.max(102, labelWidth + 16)
    : Math.max(48, Math.ceil(labelWidth * Math.cos(toRadians(35)) - slotSize / 2 + 8));
  const bottomPadding = horizontal
    ? 44
    : Math.max(62, Math.ceil(labelWidth * Math.sin(toRadians(35)) + 34));
  const width = horizontal ? left + 298 : Math.max(320, bins.length * slotSize + left + 16);
  const height = horizontal ? Math.max(190, bins.length * slotSize + 62) : 188 + bottomPadding;
  const top = 14;
  const right = width - 16;
  const bottom = height - bottomPadding;
  const groupSpan = (horizontal ? bottom - top : right - left) / Math.max(1, bins.length);
  const seriesSpan = groupSpan / Math.max(1, layout.series.length);
  const [minimum, maximum] = layout.domain;
  const valueCoordinate = (value: number) =>
    horizontal
      ? left + ((value - minimum) / (maximum - minimum)) * (right - left)
      : bottom - ((value - minimum) / (maximum - minimum)) * (bottom - top);
  const groupCoordinate = (index: number) => (horizontal ? top : left) + (index + 0.5) * groupSpan;
  const baseline = valueCoordinate(0);
  const ticks = Array.from({length: 5}, (_, index) => minimum + ((maximum - minimum) * index) / 4);
  const hasValues =
    layout.series.some(series => series.segments.length) || layout.aggregateGroups.length > 0;

  function renderMark(
    bin: ChartBin,
    seriesKey: string,
    value: number,
    count: number,
    color: string,
    categoryCoordinate: number,
    thickness: number,
    aggregate = false
  ): ReactNode {
    const coordinate = valueCoordinate(value);
    const interactive =
      !readOnly && Boolean(onSelect) && Boolean(bin.filterValue?.length) && !aggregate;
    const selected = selectedKey === bin.key;
    const label = `${displayBinKey(bin.key)} · ${seriesKey}: ${formatNumber(value)} (${count.toLocaleString()} ${count === 1 ? 'row' : 'rows'})`;
    const select = () => onSelect?.(bin.key, {filterValue: bin.filterValue});
    const start = Math.min(baseline, coordinate);
    const length = Math.abs(coordinate - baseline);
    return (
      <g
        key={JSON.stringify([bin.key, seriesKey, aggregate])}
        className={`resident-series-mark${selected ? ' is-selected' : ''}`}
        role={interactive ? 'button' : 'img'}
        tabIndex={interactive ? 0 : undefined}
        aria-label={label}
        aria-pressed={interactive ? selected : undefined}
        onClick={interactive ? select : undefined}
        onKeyDown={
          interactive
            ? event => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  select();
                }
              }
            : undefined
        }
      >
        <title>
          {label}
          {interactive ? '\nSelect the entire group; all series share this filter.' : ''}
        </title>
        {line ? (
          <>
            <circle className="resident-series-hit" cx={categoryCoordinate} cy={coordinate} r={8} />
            <circle
              cx={categoryCoordinate}
              cy={coordinate}
              r={selected ? 4.5 : 3.5}
              fill={aggregate ? 'none' : color}
              stroke={aggregate ? color : 'currentColor'}
              strokeWidth={selected ? 2 : 0.7}
            />
          </>
        ) : horizontal ? (
          <>
            <rect
              className="resident-series-hit"
              x={Math.min(start, coordinate - 6)}
              y={categoryCoordinate - Math.max(10, thickness) / 2}
              width={Math.max(12, length)}
              height={Math.max(10, thickness)}
            />
            <rect
              x={length ? start : baseline - 0.5}
              y={categoryCoordinate - thickness / 2}
              width={Math.max(1, length)}
              height={thickness}
              fill={color}
              fillOpacity={aggregate ? 0.4 : 0.9}
              stroke={selected ? 'currentColor' : color}
              strokeWidth={selected ? 2 : 0.5}
              strokeDasharray={aggregate ? '3 2' : undefined}
            />
          </>
        ) : (
          <>
            <rect
              className="resident-series-hit"
              x={categoryCoordinate - Math.max(10, thickness) / 2}
              y={Math.min(start, coordinate - 6)}
              width={Math.max(10, thickness)}
              height={Math.max(12, length)}
            />
            <rect
              x={categoryCoordinate - thickness / 2}
              y={length ? start : baseline - 0.5}
              width={thickness}
              height={Math.max(1, length)}
              fill={color}
              fillOpacity={aggregate ? 0.4 : 0.9}
              stroke={selected ? 'currentColor' : color}
              strokeWidth={selected ? 2 : 0.5}
              strokeDasharray={aggregate ? '3 2' : undefined}
            />
          </>
        )}
      </g>
    );
  }

  if (!bins.length || !hasValues) return <p role="status">No series values in this population.</p>;
  return (
    <div className="resident-series-view" data-read-only={readOnly || undefined}>
      <ul
        className="resident-series-legend"
        aria-label="Chart series legend"
        tabIndex={readOnly ? undefined : 0}
      >
        {layout.series.map(series => (
          <li key={series.key}>
            <svg width="20" height="10" viewBox="0 0 20 10" aria-hidden="true">
              <line
                x1="1"
                y1="5"
                x2="19"
                y2="5"
                stroke={series.color}
                strokeWidth="3"
                strokeDasharray={line ? series.dash : undefined}
              />
            </svg>
            <span>{series.key}</span>
          </li>
        ))}
        {layout.aggregateGroups.length > 0 && (
          <li>
            <svg width="20" height="10" viewBox="0 0 20 10" aria-hidden="true">
              <line
                x1="1"
                y1="5"
                x2="19"
                y2="5"
                stroke={AGGREGATE_COLOR}
                strokeWidth="3"
                strokeDasharray="3 2"
              />
            </svg>
            <span>Combined groups (aggregate only)</span>
          </li>
        )}
      </ul>
      <div
        className="resident-series-scroll"
        tabIndex={readOnly ? undefined : 0}
        role="region"
        aria-label="Series plot, scroll to explore all groups"
      >
        <svg
          className="resident-series-plot"
          viewBox={`0 0 ${width} ${height}`}
          style={{
            minWidth:
              readOnly || horizontal
                ? undefined
                : Math.min(width, bins.length * slotSize + left + 16)
          }}
          role="group"
          aria-labelledby={`${identifier}-title ${identifier}-description`}
        >
          <title id={`${identifier}-title`}>
            {line ? 'Line' : horizontal ? 'Horizontal bar' : 'Bar'} chart by series
          </title>
          <desc id={`${identifier}-description`}>
            Series have labeled colors and line styles. Missing series values leave gaps.{' '}
            {readOnly || !onSelect ? '' : 'Select a mark to filter its entire group.'}
          </desc>
          {ticks.map((value, index) => {
            const coordinate = valueCoordinate(value);
            return horizontal ? (
              <g key={index} className="resident-series-axis">
                <line x1={coordinate} y1={top} x2={coordinate} y2={bottom} />
                <text x={coordinate} y={bottom + 15} textAnchor="middle">
                  {formatNumber(value)}
                </text>
              </g>
            ) : (
              <g key={index} className="resident-series-axis">
                <line x1={left} y1={coordinate} x2={right} y2={coordinate} />
                <text x={left - 6} y={coordinate + 3} textAnchor="end">
                  {formatNumber(value)}
                </text>
              </g>
            );
          })}
          {horizontal ? (
            <line
              className="resident-series-zero"
              x1={baseline}
              x2={baseline}
              y1={top}
              y2={bottom}
            />
          ) : (
            <line
              className="resident-series-zero"
              x1={left}
              x2={right}
              y1={baseline}
              y2={baseline}
            />
          )}
          {bins.map((bin, groupIndex) =>
            horizontal ? (
              <text
                key={bin.key}
                className="resident-series-label"
                x={left - 8}
                y={groupCoordinate(groupIndex) + 3}
                textAnchor="end"
              >
                <title>{displayBinKey(bin.key)}</title>
                {labels[groupIndex]}
              </text>
            ) : (
              <text
                key={bin.key}
                className="resident-series-label"
                x={groupCoordinate(groupIndex)}
                y={bottom + 14}
                textAnchor="end"
                transform={`rotate(-35,${groupCoordinate(groupIndex)},${bottom + 14})`}
              >
                <title>{displayBinKey(bin.key)}</title>
                {labels[groupIndex]}
              </text>
            )
          )}
          {layout.series.map((series, seriesIndex) => (
            <g key={series.key} aria-label={`Series ${series.key}`}>
              {line &&
                series.segments
                  .filter(segment => segment.length > 1)
                  .map((segment, segmentIndex) => (
                    <polyline
                      key={segmentIndex}
                      fill="none"
                      stroke={series.color}
                      strokeWidth="2"
                      strokeDasharray={series.dash}
                      points={segment
                        .map(
                          point =>
                            `${groupCoordinate(point.groupIndex)},${valueCoordinate(point.value)}`
                        )
                        .join(' ')}
                    />
                  ))}
              {series.values.map((value, groupIndex) =>
                value && Number.isFinite(value.value)
                  ? renderMark(
                      bins[groupIndex],
                      series.key,
                      value.value,
                      value.count,
                      series.color,
                      line
                        ? groupCoordinate(groupIndex)
                        : (horizontal ? top : left) +
                            groupIndex * groupSpan +
                            (seriesIndex + 0.5) * seriesSpan,
                      Math.max(2, seriesSpan - 4)
                    )
                  : null
              )}
            </g>
          ))}
          {layout.aggregateGroups.map(({bin, groupIndex}) =>
            renderMark(
              bin,
              'Combined groups',
              bin.value,
              bin.count,
              AGGREGATE_COLOR,
              groupCoordinate(groupIndex),
              Math.min(28, groupSpan * 0.7),
              true
            )
          )}
          <text
            className="resident-series-axis-title"
            x={(left + right) / 2}
            y={height - 3}
            textAnchor="middle"
          >
            <title>{horizontal ? yLabel : xLabel}</title>
            {shortenLabel(horizontal ? yLabel : xLabel, 48)}
          </text>
          {!horizontal && (
            <text
              className="resident-series-axis-title"
              x={11}
              y={(top + bottom) / 2}
              textAnchor="middle"
              transform={`rotate(-90,11,${(top + bottom) / 2})`}
            >
              <title>{yLabel}</title>
              {shortenLabel(yLabel, 28)}
            </text>
          )}
        </svg>
      </div>
      {!readOnly && onSelect && (
        <p className="resident-series-note">Selecting a series mark filters its entire group.</p>
      )}
      {layout.aggregateGroups.length > 0 && (
        <p className="resident-series-note">
          Combined groups have no individual series breakdown and cannot be selected.
        </p>
      )}
    </div>
  );
}
