// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useId, useRef, useState, type ComponentProps, type ReactNode} from 'react';
import {ThemeProvider} from 'styled-components';
import {ChartPanelContentFactory} from '@kepler.gl/components/charts';
import {MapPopoverContentFactory} from '@kepler.gl/components';
import {
  ChartRenderer,
  displayBinKey,
  CHART_AGGREGATION_OPTIONS,
  getDefaultHeatmapColorRange,
  propsForChartDatasetChange,
  toChartableDataset
} from '@kepler.gl/charts';
import {filterResidentHoverCharts, type NativeChartConfiguration} from './resident-chart-plan';
import type {ResidentChartProvider, ResidentChartReadResult} from './resident-chart-provider';
import {makeResidentCategoryChartBins} from './resident-chart-bins';
import {makeResidentNumericChartBins} from './resident-numeric-bins';
import {makeResidentTimeChartBins} from './resident-time-bins';
import {
  getResidentGridAxisLimit,
  isResidentGridChart,
  type GridChartConfiguration,
  makeResidentGridSyntheticChart,
  makeResidentHeatmapCells,
  makeResidentPivotTable
} from './resident-chart-grid';
import {ResidentChartSeriesView} from './resident-chart-series-view';
import {ResidentChartBarsCanvas} from './resident-chart-bars-canvas';
import {shouldUseCanvasBars} from './resident-chart-bars-layout';
import {getResidentChartSeriesColor} from './resident-chart-series-layout';
import './resident-chart-panel.css';
import {ResidentConsumerNotice} from './resident-consumer-notice';
import {
  ResidentTooltip,
  ResidentHoverIdentity,
  type ResidentTooltipReader
} from './resident-tooltip';
import {ResidentLayerCharts} from './resident-layer-chart-view';

type NativePanelProps = ComponentProps<ReturnType<typeof ChartPanelContentFactory>>;
export type ResidentChartPanelOptions = {
  isResidentDataset(datasetId: string): boolean;
  resolveProvider(datasetId: string): ResidentChartProvider | undefined;
};

export function createResidentChartPopoverFactory(
  options: Pick<ResidentChartPanelOptions, 'isResidentDataset'> & {
    readTooltip?: ResidentTooltipReader;
    /** Identifies the displayed dataset/selection revision for tooltip revalidation. */
    getRevisionKey?: (datasetId: string) => string;
  }
) {
  function ResidentChartPopoverFactory(
    ...dependencies: Parameters<typeof MapPopoverContentFactory>
  ) {
    const NativePopover = MapPopoverContentFactory(...dependencies);
    return function ResidentChartPopover(props: ComponentProps<typeof NativePopover>) {
      const layer = props.layerHoverProp?.layer;
      const identity = props.layerHoverProp?.data;
      if (
        layer?.config.dataId &&
        options.isResidentDataset(layer.config.dataId) &&
        identity instanceof ResidentHoverIdentity &&
        options.readTooltip
      ) {
        const revisionKey = options.getRevisionKey?.(layer.config.dataId) ?? '';
        return (
          <>
            <ResidentTooltip
              key={`${layer.id}:${identity.sourceRowIndex}`}
              datasetId={layer.config.dataId}
              layerId={layer.id}
              sourceRowIndex={identity.sourceRowIndex}
              readTooltip={options.readTooltip}
              fieldNames={(
                props.layerHoverProp?.fieldsToShow ??
                props.layerHoverProp?.fields ??
                []
              ).map(field => field.name)}
              fields={props.layerHoverProp?.fieldsToShow?.map(field => ({
                name: field.name,
                format: (field as {format?: string | null}).format
              }))}
              revisionKey={revisionKey}
              compare={
                props.layerHoverProp?.primaryData instanceof ResidentHoverIdentity
                  ? {
                      sourceRowIndex: props.layerHoverProp.primaryData.sourceRowIndex,
                      type:
                        props.layerHoverProp.compareType === 'relative' ? 'relative' : 'absolute'
                    }
                  : undefined
              }
              geometrySelectionUnavailable={layer.type === 'geojson' || layer.type === 'hexagonId'}
            />
            <ResidentLayerCharts
              key={`chart:${layer.id}`}
              charts={props.charts ?? []}
              datasetId={layer.config.dataId}
              layerId={layer.id}
              hoveredRow={identity.sourceRowIndex}
              readTooltip={options.readTooltip}
              revisionKey={revisionKey}
            />
          </>
        );
      }
      const result = filterResidentHoverCharts(
        props.charts ?? [],
        layer?.id,
        layer?.config.dataId,
        options.isResidentDataset
      );
      return (
        <>
          <NativePopover {...props} charts={result.charts} />
          {result.blocked && (
            <p role="status">
              Resident tooltip charts need a bounded layer-summary adapter. Use the Charts panel for
              supported dataset summaries.
            </p>
          )}
        </>
      );
    };
  }
  ResidentChartPopoverFactory.deps = MapPopoverContentFactory.deps;
  return ResidentChartPopoverFactory;
}

/** Native manager/configuration/actions remain in charge; replace resident computation at the panel. */
export function createResidentChartPanelFactory(options: ResidentChartPanelOptions) {
  function ResidentChartPanelFactory(...dependencies: Parameters<typeof ChartPanelContentFactory>) {
    const NativePanel = ChartPanelContentFactory(...dependencies);
    const NativeJsonEditor = dependencies[3];
    return function ResidentChartPanel(props: NativePanelProps) {
      return (
        <>
          {(props.charts ?? []).map(chart =>
            chart.dataId && options.isResidentDataset(chart.dataId) ? (
              <ResidentChartCard
                key={chart.id}
                {...props}
                chart={chart}
                provider={options.resolveProvider(chart.dataId)}
                jsonEditor={
                  <NativeJsonEditor
                    chart={chart}
                    updateChart={props.visStateActions?.updateChart}
                  />
                }
              />
            ) : (
              <NativePanel key={chart.id} {...props} charts={[chart]} />
            )
          )}
        </>
      );
    };
  }
  ResidentChartPanelFactory.deps = ChartPanelContentFactory.deps;
  return ResidentChartPanelFactory;
}

function ResidentChartCard({
  chart,
  provider,
  datasets,
  visStateActions,
  readOnly,
  hidePinAction,
  jsonEditor
}: NativePanelProps & {
  chart: NativeChartConfiguration;
  provider?: ResidentChartProvider;
  jsonEditor: ReactNode;
}) {
  const identifier = useId();
  const [revision, setRevision] = useState(provider?.getRevision());
  const [retry, setRetry] = useState(0);
  const staleRetries = useRef(0);
  const [response, setResponse] = useState<{
    key: string;
    provider: ResidentChartProvider;
    value: ResidentChartReadResult | null;
    error?: string;
  }>();
  const chartKey = JSON.stringify(chart);
  const requestKey = `${chartKey}:${provider?.getRevision()}:${revision}:${retry}`;
  useEffect(() => provider?.subscribe(() => setRevision(provider.getRevision())), [provider]);
  useEffect(() => {
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    if (provider)
      void provider.read(chart).then(
        value => {
          if (cancelled) return;
          setResponse({key: requestKey, provider, value});
          // A null read means the revision moved or the selection is still computing: retry with
          // backoff (stale, not an error) instead of waiting for a manual refresh.
          if (value === null && staleRetries.current < 5) {
            staleRetries.current += 1;
            retryTimer = setTimeout(
              () => setRetry(count => count + 1),
              250 * 2 ** staleRetries.current
            );
          } else if (value !== null) staleRetries.current = 0;
        },
        failure => {
          if (!cancelled)
            setResponse({
              key: requestKey,
              provider,
              value: null,
              error: failure instanceof Error ? failure.message : String(failure)
            });
        }
      );
    return () => {
      cancelled = true;
      clearTimeout(retryTimer);
    };
  }, [provider, requestKey]);
  useEffect(
    () => () => {
      void provider?.remove(chart.id);
    },
    [provider, chart.id]
  );
  const current =
    response?.key === requestKey && response.provider === provider ? response : undefined;
  const scalar = chart.type === 'bigNumber';
  const grouped =
    chart.type === 'barChart' || chart.type === 'horizontalBar' || chart.type === 'lineChart';
  const seriesChartType =
    chart.type === 'barChart'
      ? 'barChart'
      : chart.type === 'horizontalBar'
        ? 'horizontalBar'
        : chart.type === 'lineChart'
          ? 'lineChart'
          : undefined;
  const measureAxisKey = scalar ? 'axis' : chart.type === 'horizontalBar' ? 'xAxis' : 'yAxis';
  const categoryAxisKey = chart.type === 'horizontalBar' ? 'yAxis' : 'xAxis';
  const axis = scalar ? chart.axis : grouped ? chart[measureAxisKey] : undefined;
  const categoryAxis = grouped ? chart[categoryAxisKey] : undefined;
  const ready = current?.value?.status === 'ready' ? current.value.result : undefined;
  const categories = ready && 'summaries' in ready ? ready : undefined;
  const scalarResult = ready && !('summaries' in ready) ? ready : undefined;
  // Heatmap and pivot charts reuse the grouped summary: X groups, Y as series, value as measure.
  const gridChart = makeResidentGridSyntheticChart(chart);
  const binChart = gridChart ?? chart;
  const chartBins = categories
    ? categories.kind === 'time'
      ? makeResidentTimeChartBins(
          binChart,
          categories.summaries,
          categories.periods,
          chart.dataId ? datasets[chart.dataId]?.color : undefined
        )
      : categories.kind === 'numeric'
        ? makeResidentNumericChartBins(
            binChart,
            categories.summaries,
            categories.boundaries,
            chart.dataId ? datasets[chart.dataId]?.color : undefined,
            categories.categoryValues
          )
        : makeResidentCategoryChartBins(
            binChart,
            categories.summaries,
            chart.dataId ? datasets[chart.dataId]?.color : undefined
          )
    : [];

  const seriesColors =
    grouped && chart.type !== 'lineChart' && chart.colorBy !== 'none'
      ? chart.chartDisplay?.['colorRange']?.colors
      : undefined;
  const fields = chart.dataId ? (datasets[chart.dataId]?.fields ?? []) : [];
  const update = (changes: Partial<NativeChartConfiguration>) =>
    visStateActions?.updateChart(chart.id, changes);
  const onSelectCategory: ComponentProps<typeof ChartRenderer>['onSelect'] = (key, extra) => {
    if (!grouped || !chart.dataId || !categoryAxis?.field) return;
    const filterId = chart.crossFilter?.filterId || `chart-${chart.id}`;
    const alreadySelected =
      chart.crossFilter?.enabled && String(chart.crossFilter.value?.['x']) === key;
    if (!alreadySelected)
      visStateActions?.createOrUpdateFilter(
        filterId,
        chart.dataId,
        categoryAxis.field.name,
        extra?.filterValue?.length ? extra.filterValue : [key]
      );
    update({
      crossFilter: {
        enabled: !alreadySelected,
        filterId,
        fieldNames: {x: categoryAxis.field.name},
        value: alreadySelected ? {} : {x: key}
      }
    });
  };
  return (
    <section className="resident-chart-card" aria-label={`Chart: ${chart.title || chart.id}`}>
      <h3>{chart.title || 'Chart'}</h3>
      {!readOnly && (
        <>
          <label htmlFor={`${identifier}-title`}>Chart title</label>
          <input
            id={`${identifier}-title`}
            value={chart.title ?? ''}
            onChange={event => update({title: event.target.value})}
          />
          <label htmlFor={`${identifier}-dataset`}>Chart dataset</label>
          <select
            id={`${identifier}-dataset`}
            value={chart.dataId ?? ''}
            onChange={event =>
              update(
                propsForChartDatasetChange(
                  chart,
                  toChartableDataset(datasets[event.target.value]),
                  {dataId: event.target.value}
                )
              )
            }
          >
            {Object.values(datasets).map(dataset => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.label}
              </option>
            ))}
          </select>
          {grouped && (
            <>
              <label htmlFor={`${identifier}-category`}>Chart categories</label>
              <select
                id={`${identifier}-category`}
                value={categoryAxis?.field?.name ?? ''}
                onChange={event => {
                  const field = fields.find(candidate => candidate.name === event.target.value);
                  update({
                    [categoryAxisKey]: {
                      field: field ? {name: field.name, type: field.type} : null,
                      aggregation:
                        field && ['timestamp', 'date', 'time'].includes(field.type)
                          ? 'timeBin'
                          : 'uniqueBin'
                    },
                    crossFilter: {
                      enabled: false,
                      filterId: chart.crossFilter?.filterId || `chart-${chart.id}`,
                      fieldNames: {},
                      value: {}
                    }
                  });
                }}
              >
                <option value="">Select categories</option>
                {fields.map(field => (
                  <option key={field.name} value={field.name}>
                    {field.displayName || field.name}
                  </option>
                ))}
              </select>
              <label htmlFor={`${identifier}-series`}>Chart series</label>
              <select
                id={`${identifier}-series`}
                value={chart.groupBy?.field?.name ?? ''}
                onChange={event => {
                  const field = fields.find(candidate => candidate.name === event.target.value);
                  update({
                    groupBy: {
                      field: field ? {name: field.name, type: field.type} : null,
                      aggregation: 'uniqueBin'
                    }
                  });
                }}
              >
                <option value="">No series</option>
                {fields.map(field => (
                  <option key={field.name} value={field.name}>
                    {field.displayName || field.name}
                  </option>
                ))}
              </select>
              <label htmlFor={`${identifier}-binning`}>Chart binning</label>
              <select
                id={`${identifier}-binning`}
                value={categoryAxis?.aggregation ?? 'uniqueBin'}
                onChange={event =>
                  update({
                    [categoryAxisKey]: {
                      ...categoryAxis,
                      field: categoryAxis?.field ?? null,
                      aggregation: event.target.value
                    }
                  })
                }
              >
                <option value="uniqueBin">Automatic by field type</option>
                <option value="numericBin">Numeric intervals</option>
                <option value="timeBin">Time intervals</option>
              </select>
              {categoryAxis?.aggregation === 'timeBin' && (
                <>
                  <label htmlFor={`${identifier}-interval`}>Calendar interval (UTC)</label>
                  <select
                    id={`${identifier}-interval`}
                    value={categoryAxis.interval ?? chart.chartDisplay?.['interval'] ?? ''}
                    onChange={event =>
                      update({
                        [categoryAxisKey]: {...categoryAxis, interval: event.target.value || null}
                      })
                    }
                  >
                    <option value="">Automatic</option>
                    <option value="hour">Hour</option>
                    <option value="day">Day</option>
                    <option value="week">Week</option>
                    <option value="month">Month</option>
                    <option value="year">Year</option>
                  </select>
                </>
              )}
              {chart.type !== 'lineChart' && (
                <>
                  <label htmlFor={`${identifier}-groups`}>Maximum displayed groups</label>
                  <input
                    id={`${identifier}-groups`}
                    type="number"
                    min="0"
                    max="512"
                    value={chart.numGroups ?? 10}
                    onChange={event => {
                      const value = Number(event.target.value);
                      if (Number.isSafeInteger(value) && value >= 0) update({numGroups: value});
                    }}
                  />
                  <label>
                    <input
                      type="checkbox"
                      checked={Boolean(chart.groupOthers)}
                      onChange={event => update({groupOthers: event.target.checked})}
                    />{' '}
                    Combine remaining groups as Other
                  </label>
                </>
              )}
              <label>
                <input
                  type="checkbox"
                  checked={Boolean(chart.crossFilter?.enabled)}
                  onChange={event =>
                    update({
                      crossFilter: {
                        enabled: event.target.checked,
                        filterId: chart.crossFilter?.filterId || `chart-${chart.id}`,
                        fieldNames: chart.crossFilter?.fieldNames ?? {},
                        value: chart.crossFilter?.value ?? {}
                      }
                    })
                  }
                />{' '}
                Crossfilter from chart
              </label>
            </>
          )}
          {(scalar || grouped) && (
            <>
              <label htmlFor={`${identifier}-field`}>Chart field</label>
              <select
                id={`${identifier}-field`}
                value={axis?.field?.name ?? ''}
                onChange={event => {
                  const field = fields.find(candidate => candidate.name === event.target.value);
                  update({
                    [measureAxisKey]: {
                      ...axis,
                      field: field ? {name: field.name, type: field.type} : null,
                      aggregation: axis?.aggregation ?? 'count'
                    }
                  });
                }}
              >
                <option value="">No field · count rows</option>
                {fields.map(field => (
                  <option key={field.name} value={field.name}>
                    {field.displayName || field.name}
                  </option>
                ))}
              </select>
              <label htmlFor={`${identifier}-measure`}>Chart measure</label>
              <select
                id={`${identifier}-measure`}
                value={axis?.aggregation ?? 'count'}
                onChange={event => {
                  const measure = CHART_AGGREGATION_OPTIONS.find(
                    candidate => candidate.id === event.target.value
                  );
                  if (measure)
                    update({
                      [measureAxisKey]: {
                        ...axis,
                        field: axis?.field ?? null,
                        aggregation: measure.id
                      }
                    });
                }}
              >
                {CHART_AGGREGATION_OPTIONS.map(measure => (
                  <option key={measure.id} value={measure.id}>
                    {measure.label}
                  </option>
                ))}
              </select>
              {scalar && (
                <label>
                  <input
                    type="checkbox"
                    checked={chart.chartDisplay?.['showCaption'] !== false}
                    onChange={event =>
                      update({
                        chartDisplay: {...chart.chartDisplay, showCaption: event.target.checked}
                      })
                    }
                  />{' '}
                  Show chart caption
                </label>
              )}
            </>
          )}
          <label>
            <input
              type="checkbox"
              checked={Boolean(chart.applyFilters)}
              onChange={event => update({applyFilters: event.target.checked})}
            />{' '}
            Apply map filters
          </label>
          {!hidePinAction && (
            <label>
              <input
                type="checkbox"
                checked={Boolean(chart.pinned)}
                onChange={event => update({pinned: event.target.checked})}
              />{' '}
              Keep chart visible
            </label>
          )}
          <button type="button" onClick={() => visStateActions?.removeChart(chart.id)}>
            Remove chart
          </button>
          <details>
            <summary>Edit native chart JSON</summary>
            {jsonEditor}
          </details>
        </>
      )}
      <div aria-busy={!current}>
        {!provider ? (
          <ResidentConsumerNotice
            consumer="chart"
            status="pending"
            reason="This resident source is being prepared."
          />
        ) : current?.error ? (
          <ResidentConsumerNotice consumer="chart" status="error" reason={current.error} />
        ) : current?.value?.status === 'unsupported' ? (
          <ResidentConsumerNotice
            consumer="chart"
            status="unsupported"
            reason={current.value.reason}
          />
        ) : current?.value?.status === 'ready' ? (
          <>
            <div
              role="status"
              aria-label={
                scalarResult ? `Chart value: ${scalarResult.value}` : 'Chart category summary'
              }
            >
              <ThemeProvider
                theme={theme => ({...theme, textColor: '#edf2f7', subtextColor: '#c4cfdd'})}
              >
                {categories && gridChart && isResidentGridChart(chart) ? (
                  <ChartRenderer
                    data={makeResidentGridChartData(
                      chart,
                      chartBins,
                      categories.kind !== 'category'
                    )}
                  />
                ) : categories && grouped && seriesChartType && chart.groupBy?.field ? (
                  <ResidentChartSeriesView
                    bins={chartBins}
                    type={seriesChartType}
                    colors={seriesColors}
                    xLabel={getChartAxisTitle(categoryAxis, 'Group')}
                    yLabel={getChartAxisTitle(axis, 'Value')}
                    selectedKey={
                      chart.crossFilter?.enabled && chart.crossFilter.value?.['x'] != null
                        ? String(chart.crossFilter.value['x'])
                        : undefined
                    }
                    onSelect={onSelectCategory}
                    readOnly={readOnly}
                  />
                ) : categories &&
                  chart.type !== 'lineChart' &&
                  chart.type !== 'horizontalBar' &&
                  shouldUseCanvasBars(chartBins.length) &&
                  !(window as {__residentChartForceSvgBars?: boolean})
                    .__residentChartForceSvgBars ? (
                  <ResidentChartBarsCanvas
                    bins={chartBins}
                    selectedKey={
                      chart.crossFilter?.enabled && chart.crossFilter.value?.['x'] != null
                        ? String(chart.crossFilter.value['x'])
                        : undefined
                    }
                    onSelect={grouped && !readOnly ? onSelectCategory : undefined}
                    readOnly={readOnly}
                  />
                ) : (
                  <ChartRenderer
                    data={
                      categories
                        ? chart.type === 'lineChart'
                          ? {
                              kind: 'line',
                              bins: chartBins,
                              xLabel: getChartAxisTitle(categoryAxis, 'X'),
                              yLabel: getChartAxisTitle(axis, 'Y')
                            }
                          : {
                              kind: 'bars',
                              horizontal: chart.type === 'horizontalBar',
                              bins: chartBins
                            }
                        : {
                            kind: 'bigNumber',
                            value: scalarResult?.value ?? 0,
                            caption: axis?.field?.name
                              ? `${axis.aggregation || 'count'} of ${axis.field.name}`
                              : 'Count of rows',
                            format: scalar ? chart.chartDisplay?.['format'] : undefined
                          }
                    }
                    selectedKey={
                      chart.crossFilter?.enabled && chart.crossFilter.value?.['x'] != null
                        ? String(chart.crossFilter.value['x'])
                        : undefined
                    }
                    onSelect={grouped && !readOnly ? onSelectCategory : undefined}
                    showCaption={scalar ? chart.chartDisplay?.['showCaption'] !== false : true}
                  />
                )}
              </ThemeProvider>
            </div>
            {grouped && categories && chart.groupBy?.field && (
              <details className="resident-chart-series" open>
                <summary>Series breakdown · {chart.groupBy.field.name}</summary>
                <p className="resident-chart-note">
                  The plot shows individual series values. Group totals sum those values. Rows
                  without a series label count toward the group but are excluded from its value.
                </p>
                <div className="resident-chart-series-scroll" tabIndex={0}>
                  <table aria-label="Chart series breakdown">
                    <thead>
                      <tr>
                        <th scope="col">Group</th>
                        <th scope="col">Series</th>
                        <th scope="col">Value</th>
                        <th scope="col">Rows</th>
                      </tr>
                    </thead>
                    <tbody>
                      {chartBins.flatMap(
                        bin =>
                          bin.series?.map(series => (
                            <tr key={JSON.stringify([bin.key, series.key])}>
                              <th scope="row">{displayBinKey(bin.key)}</th>
                              <td>
                                <span
                                  className="resident-chart-series-color"
                                  style={{
                                    backgroundColor: getResidentChartSeriesColor(
                                      series.key,
                                      seriesColors
                                    )
                                  }}
                                  aria-hidden="true"
                                />
                                {series.key}
                              </td>
                              <td>
                                {series.value.toLocaleString(undefined, {
                                  maximumSignificantDigits: 8
                                })}
                              </td>
                              <td>{series.count.toLocaleString()}</td>
                            </tr>
                          )) ?? [
                            <tr key={bin.key}>
                              <th scope="row">{displayBinKey(bin.key)}</th>
                              <td>Combined groups</td>
                              <td>{bin.value.toLocaleString()}</td>
                              <td>{bin.count.toLocaleString()}</td>
                            </tr>
                          ]
                      )}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
            {current.value.notes?.map(note => (
              <p key={note} role="status" className="resident-chart-note">
                {note}
              </p>
            ))}
            {!readOnly && (
              <p className="resident-chart-note">
                Prepared GPU values · {chart.applyFilters ? 'Current selection' : 'All source rows'}{' '}
                · source revision {current.value.result.sourceRevision} ·{' '}
                {current.value.result.readbackBytes} bytes read
              </p>
            )}
            {scalarResult && !scalarResult.hasValue && (
              <p role="status">No finite values; displaying zero as in native charts.</p>
            )}
          </>
        ) : (
          <ResidentConsumerNotice
            consumer="chart"
            status={current ? 'stale' : 'pending'}
            reason={
              current
                ? 'The source or selection changed while reading; refreshing the chart.'
                : 'Computing chart…'
            }
          />
        )}
      </div>
      {provider && !readOnly && (
        <button type="button" onClick={() => setRetry(value => value + 1)}>
          Refresh chart
        </button>
      )}
    </section>
  );
}

function getChartAxisTitle(
  axis: {field?: {name: string} | null; aggregation?: string | null} | null | undefined,
  fallback: string
): string {
  if (!axis?.field?.name) return fallback;
  return !axis.aggregation || ['uniqueBin', 'numericBin', 'timeBin'].includes(axis.aggregation)
    ? axis.field.name
    : `${axis.aggregation} of ${axis.field.name}`;
}

/** Native heatmap/pivot render data from the bounded X-group bins that carry the Y series. */
function makeResidentGridChartData(
  chart: GridChartConfiguration,
  bins: ComponentProps<typeof ChartRenderer>['data'] extends infer Data
    ? Data extends {bins: infer Bins}
      ? Bins
      : never
    : never,
  orderedX: boolean
): ComponentProps<typeof ChartRenderer>['data'] {
  const props = {limit: getResidentGridAxisLimit(chart.numGroups), orderedX};
  const xLabel = getChartAxisTitle(chart.xAxis, 'X');
  const yLabel = getChartAxisTitle(chart.yAxis, 'Y');
  if (chart.type === 'pivotTable')
    return {kind: 'pivot', table: makeResidentPivotTable(bins, props), xLabel, yLabel};
  const palette = chart.chartDisplay?.['colorRange']?.colors;
  return {
    kind: 'heatmap',
    cells: makeResidentHeatmapCells(bins, props),
    colors: palette && palette.length >= 2 ? palette : getDefaultHeatmapColorRange().colors,
    xLabel,
    yLabel
  };
}
