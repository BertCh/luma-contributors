// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {memo, useEffect, useReducer} from 'react';
import {LayerColorLegendFactory} from '@kepler.gl/components';
import {getConsumerStateMessage, makeConsumerState} from './resident-consumer-state';
import type {KeplerResidentStyleLegend} from '@luma-contributors/kepler';

import {clamp} from '@math.gl/core';

export type ResidentStyleLegendSnapshot = {
  expected: Pick<
    KeplerResidentStyleLegend,
    'styleId' | 'revision' | 'source' | 'selectionRevision'
  >;
  legend?: KeplerResidentStyleLegend;
  reason?: string;
};
export type ResidentStyleLegendOptions = {
  isResidentDataset(datasetId: string): boolean;
  getLegend(
    layerId: string,
    channel: 'fillColor' | 'strokeColor'
  ): ResidentStyleLegendSnapshot | null;
  /** A completed bounded readback refreshes the memoized legend independently of config. */
  subscribe?(listener: () => void): () => void;
  maximumEntries?: number;
};
type NativeLegendProps = Parameters<ReturnType<typeof LayerColorLegendFactory>>[0];

/** Exact AX classes carry the source field in the custom range, separately from storage. */
export function getResidentStyleLegendTitle(
  fieldName: string,
  measure: string | undefined,
  colorRange?: {type?: string; name?: string}
): string {
  return fieldName.startsWith('analysis_result_class (') &&
    colorRange?.type === 'customOrdinal' &&
    colorRange.name
    ? colorRange.name
    : measure || fieldName;
}

/** Inject at LayerColorLegendFactory so fill and stroke retain independent provenance. */
export function createResidentStyleLegendFactory(options: ResidentStyleLegendOptions) {
  const maximumEntries = clamp(options.maximumEntries ?? 64, 1, 256);
  function ResidentStyleLegendFactory(...dependencies: Parameters<typeof LayerColorLegendFactory>) {
    const NativeLegend = LayerColorLegendFactory(...dependencies);
    const ResidentStyleLegend = memo(function ResidentStyleLegend(props: NativeLegendProps) {
      const [, refresh] = useReducer((revision: number) => revision + 1, 0);
      useEffect(() => {
        if (!options.subscribe) return undefined;
        const unsubscribe = options.subscribe(refresh);
        refresh();
        return unsubscribe;
      }, []);
      const datasetId = props.layer.config.dataId;
      const channel = props.colorChannel.property === 'strokeColor' ? 'strokeColor' : 'fillColor';
      const field =
        channel === 'strokeColor'
          ? 'strokeColorField' in props.layer.config
            ? props.layer.config.strokeColorField
            : null
          : props.layer.config.colorField;
      if (!datasetId || !options.isResidentDataset(datasetId) || !field)
        return <NativeLegend {...props} />;
      const snapshot = options.getLegend(props.layer.id, channel);
      const legend = snapshot && matchesResidentStyleLegend(snapshot) ? snapshot.legend : undefined;
      const title = getResidentStyleLegendTitle(
        typeof field === 'object' && 'name' in field ? String(field.name) : channel,
        props.description.measure,
        props.layer.config.visConfig?.['colorRange']
      );
      if (!legend) {
        return (
          <div
            className="legend--layer__item"
            data-resident-style-legend={channel}
            data-resident-legend-layer={props.layer.id}
          >
            <strong>{title}</strong>
            <div
              role="status"
              data-resident-consumer-state={snapshot?.legend ? 'stale' : 'pending'}
            >
              {snapshot?.reason ??
                getConsumerStateMessage(
                  snapshot?.legend
                    ? makeConsumerState('stale', 'legend', 'Updating GPU legend…')
                    : makeConsumerState('pending', 'legend', 'Computing GPU legend…')
                )}
            </div>
          </div>
        );
      }
      if (legend.domainStatus === 'invalid-log-domain') {
        return (
          <div
            className="legend--layer__item"
            data-resident-style-legend={channel}
            data-resident-legend-layer={props.layer.id}
          >
            <strong>{title}</strong>
            <div role="status">
              The logarithmic domain crosses or touches zero. Choose a domain entirely above or
              below zero.
            </div>
          </div>
        );
      }
      if (!legend.domain || legend.finiteCount === 0) {
        return (
          <div
            className="legend--layer__item"
            data-resident-style-legend={channel}
            data-resident-legend-layer={props.layer.id}
          >
            <strong>{title}</strong>
            <div>No finite values in the style domain.</div>
          </div>
        );
      }
      const palette = legend.palette ?? [];
      const continuous =
        legend.scale === 'linear' || legend.scale === 'sqrt' || legend.scale === 'log';
      const rows = getResidentStyleLegendRows(legend, maximumEntries);
      const total = legend.labels?.length ?? palette.length;
      return (
        <div
          className="legend--layer__item"
          data-resident-style-legend={channel}
          data-resident-legend-layer={props.layer.id}
        >
          <strong>{title}</strong>
          {continuous && palette.length <= maximumEntries ? (
            <div>
              <div
                role="img"
                aria-label={`${legend.scale} color scale`}
                style={{
                  height: 12,
                  background: getContinuousLegendGradient(legend)
                }}
              />
              <div style={{display: 'flex', justifyContent: 'space-between'}}>
                <span>{formatValue(legend.domain[0])}</span>
                <span>{formatValue(legend.domain[legend.domain.length - 1])}</span>
              </div>
            </div>
          ) : (
            <div>
              {rows.map((row, index) => (
                <div key={index} style={{display: 'flex', alignItems: 'center', gap: 6}}>
                  <span
                    aria-hidden="true"
                    style={{
                      display: 'inline-block',
                      width: 16,
                      height: 12,
                      background: packedColorCSS(row.color)
                    }}
                  />
                  <span>{row.label}</span>
                </div>
              ))}
            </div>
          )}
          {getResidentMissingValueEntry(legend) && (
            <div
              data-resident-legend-missing
              style={{display: 'flex', alignItems: 'center', gap: 6}}
            >
              <span
                aria-hidden="true"
                style={{
                  display: 'inline-block',
                  width: 16,
                  height: 12,
                  border: '1px dashed currentColor',
                  background: packedColorCSS(NO_VALUE_COLOR_PACKED)
                }}
              />
              <span>{getResidentMissingValueEntry(legend)}</span>
            </div>
          )}
          {total > maximumEntries && (
            <div>
              Showing {maximumEntries} of {total} legend entries.
            </div>
          )}
        </div>
      );
    });
    return ResidentStyleLegend;
  }
  ResidentStyleLegendFactory.deps = LayerColorLegendFactory.deps;
  return ResidentStyleLegendFactory;
}

/** Kepler NO_VALUE_COLOR is transparent; rows with null or NaN values get no palette color. */
const NO_VALUE_COLOR_PACKED = 0;

/**
 * Missing-value legend label. Uses missingCount when the GPU legend reports it; otherwise the
 * entry is shown without a count because null/NaN rows are excluded from finiteCount.
 */
export function getResidentMissingValueEntry(
  legend: KeplerResidentStyleLegend & {missingCount?: number | null}
): string | null {
  if (legend.missingCount === 0) return null;
  return typeof legend.missingCount === 'number'
    ? `No value (null or NaN): ${legend.missingCount.toLocaleString('en-US')} rows`
    : 'No value (null or NaN rows are not colored)';
}

/** The host may retain old legend data while a new GPU submission/readback is pending. */
export function matchesResidentStyleLegend(snapshot: ResidentStyleLegendSnapshot): boolean {
  const {legend, expected} = snapshot;
  return Boolean(
    legend &&
      legend.styleId === expected.styleId &&
      legend.revision === expected.revision &&
      legend.source.datasetId === expected.source.datasetId &&
      legend.source.revision === expected.source.revision &&
      legend.selectionRevision === expected.selectionRevision
  );
}

/** Uses GPU-produced edges directly. Quantile edges must never be treated as sample values. */
export function getResidentStyleLegendRows(
  legend: KeplerResidentStyleLegend,
  maximumEntries = 64
): {color: number; label: string}[] {
  if (!legend.domain || !legend.palette || legend.domainStatus === 'invalid-log-domain') return [];
  const {domain, palette} = legend;
  if (legend.labels)
    return legend.labels.slice(0, maximumEntries).map((label, index) => ({
      label,
      color: palette[legend.codes?.[index] ?? index] ?? 0
    }));
  const count = palette.length;
  return palette.slice(0, maximumEntries).map((color, index) => {
    if (legend.scale === 'linear' || legend.scale === 'sqrt' || legend.scale === 'log')
      return {color, label: formatValue(domain[index])};
    if (count === 1) return {color, label: 'All finite values'};
    const edge = (edgeIndex: number) =>
      legend.scale === 'equal-interval'
        ? domain[0] + ((domain[domain.length - 1] - domain[0]) * edgeIndex) / count
        : domain[edgeIndex - (legend.scale === 'threshold' ? 1 : 0)];
    const label =
      index === 0
        ? `< ${formatValue(edge(1))}`
        : index === count - 1
          ? `≥ ${formatValue(edge(index))}`
          : `${formatValue(edge(index))} ≤ value < ${formatValue(edge(index + 1))}`;
    return {color, label};
  });
}

function getContinuousLegendGradient(legend: KeplerResidentStyleLegend): string {
  const domain = legend.domain!;
  const transform = (value: number) =>
    legend.scale === 'log'
      ? Math.sign(value) * Math.log(Math.abs(value))
      : legend.scale === 'sqrt'
        ? Math.sign(value) * Math.sqrt(Math.abs(value))
        : value;
  const minimum = transform(domain[0]);
  const maximum = transform(domain[domain.length - 1]);
  const colors = legend.palette!.map((color, index) => {
    const fraction =
      maximum === minimum
        ? index / (legend.palette!.length - 1)
        : (transform(domain[index]) - minimum) / (maximum - minimum);
    return `${packedColorCSS(color)} ${fraction * 100}%`;
  });
  return `linear-gradient(to right, ${colors.join(', ')})`;
}

function formatValue(value: number): string {
  return Number(value.toPrecision(7)).toLocaleString('en-US', {maximumSignificantDigits: 7});
}

function packedColorCSS(color: number): string {
  return `rgba(${color & 255}, ${(color >>> 8) & 255}, ${(color >>> 16) & 255}, ${(color >>> 24) / 255})`;
}
