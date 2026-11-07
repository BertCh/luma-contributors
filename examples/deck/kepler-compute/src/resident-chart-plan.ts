// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ComponentProps} from 'react';
import type {ChartPanelContentFactory} from '@kepler.gl/components/charts';
import type {GPUResidentScalarFormat} from '@luma-contributors/kepler';
import {
  MAXIMUM_RESIDENT_GROUPS,
  clampIntegerGroupLimit,
  clampNumericGroupBudget,
  describeCategoryOverflow,
  describeCategoryTopGroups
} from './resident-chart-group-limits';
import {
  MAXIMUM_RESIDENT_TOP_DICTIONARY,
  getResidentTopGroupLimit
} from './resident-chart-top-groups';

export type NativeChartConfiguration = NonNullable<
  ComponentProps<ReturnType<typeof ChartPanelContentFactory>>['charts']
>[number];

export type ResidentChartField =
  | {
      kind?: 'scalar';
      name: string;
      format: GPUResidentScalarFormat;
      /** Dictionary codes are not numeric measurements. */
      categorical?: boolean;
      categoryCount?: number;
    }
  | {
      kind: 'time';
      name: string;
      unit: string;
      sourceType: string;
      format?: never;
      categorical?: never;
      categoryCount?: never;
    };

/** Series dictionaries stay at 512 entries; group axes accept up to MAXIMUM_RESIDENT_GROUPS. */
export const MAXIMUM_RESIDENT_CHART_CATEGORIES = 512;
export const MAXIMUM_RESIDENT_CHART_CELLS = 4096;
/** Native MAX_CHART_POINTS: calendar charts show up to this many periods, then use equal-width bins. */
export const MAXIMUM_RESIDENT_TIME_GROUPS = 1000;

/**
 * Calendar period capacity for the cell budget: the native 1,000 whenever the series cross-product
 * fits, otherwise the historical 512-period floor that every accepted plan can afford.
 */
export function getResidentTimeGroupCapacity(seriesCategoryCount?: number): number {
  if (seriesCategoryCount === undefined) return MAXIMUM_RESIDENT_TIME_GROUPS;
  return Math.max(
    MAXIMUM_RESIDENT_CHART_CATEGORIES,
    Math.min(
      MAXIMUM_RESIDENT_TIME_GROUPS,
      Math.floor(MAXIMUM_RESIDENT_CHART_CELLS / (seriesCategoryCount + 1))
    )
  );
}

/** Median and sample deviation are exact ordered statistics rather than streaming reductions. */
export type ResidentChartOperation =
  | 'count'
  | 'sum'
  | 'mean'
  | 'min'
  | 'max'
  | 'median'
  | 'standardDeviation';

/** Whether an operation needs the per-key ordered-statistics adapter. */
export function isResidentChartOrderOperation(
  operation: ResidentChartOperation
): operation is 'median' | 'standardDeviation' {
  return operation === 'median' || operation === 'standardDeviation';
}

/**
 * Whether a numeric source column must first be widened to exact Float32. Native scalar minimum
 * and maximum stay in the integer domain; every other measure runs on Float32 values.
 */
export function needsResidentChartMeasureWidening(
  kind: ResidentChartPlan['kind'],
  operation: ResidentChartOperation,
  format: GPUResidentScalarFormat | undefined
): boolean {
  if (!format || format === 'float32' || operation === 'count') return false;
  return kind !== 'scalar' || (operation !== 'min' && operation !== 'max');
}

export type ResidentScalarChartPlan = {
  kind: 'scalar';
  chartId: string;
  datasetId: string;
  population: 'all' | 'selected';
  operation: ResidentChartOperation;
  column?: string;
};

export type ResidentCategoryChartPlan = Omit<ResidentScalarChartPlan, 'kind'> & {
  kind: 'category';
  categoryColumn: string;
  categoryCount: number;
  /** Dictionary above the 1,000-group cap: keep this many top groups plus one Other bucket. */
  topGroups?: {limit: number};
  series?: {column: string; categoryCount: number};
  /** Series on a numeric field: equal-width bins of the observed range (categoryCount = binCount). */
  numericSeries?: {binFormat: GPUResidentScalarFormat};
  excludedCpuFilterIds: readonly string[];
  excludedGpuFieldNames: readonly string[];
};

export type ResidentNumericChartPlan = Omit<
  ResidentCategoryChartPlan,
  'kind' | 'categoryColumn' | 'categoryCount'
> & {
  kind: 'numeric';
  binColumn: string;
  binCount: number;
  binFormat: GPUResidentScalarFormat;
  /** Native integer auto: unique groups up to this limit, then intervals; zero means no top-N. */
  integerGroupLimit?: number;
};

export type ResidentTimeChartPlan = Omit<
  ResidentCategoryChartPlan,
  'kind' | 'categoryColumn' | 'categoryCount'
> & {
  kind: 'time';
  timeColumn: string;
  /** `auto` is native's unset interval: the runtime chooses hour to year from the population span. */
  interval: ResidentCalendarInterval | 'auto' | 'equal-width';
  /** Native bar charts split the observed Date extent into this many equal-width bins. */
  binCount?: number;
  /** A literal native "auto" string groups by UTC day but formats each label as date and time. */
  dateTimeLabels?: boolean;
};

export type ResidentCalendarInterval = 'hour' | 'day' | 'week' | 'month' | 'year';

export type ResidentChartPlan =
  | ResidentScalarChartPlan
  | ResidentCategoryChartPlan
  | ResidentNumericChartPlan
  | ResidentTimeChartPlan;

export type ResidentChartPlanResult =
  | {supported: true; plan: ResidentChartPlan; notes?: readonly string[]}
  | {supported: false; reason: string};

/** Native configuration is the source of truth; unsupported controls never produce an empty plot. */
export function planResidentChart(
  chart: NativeChartConfiguration,
  datasetId: string,
  fields: readonly ResidentChartField[]
): ResidentChartPlanResult {
  if (!chart.dataId || chart.dataId !== datasetId)
    return {supported: false, reason: 'Select an available resident dataset for this chart.'};
  if (chart.type === 'barChart' || chart.type === 'horizontalBar') {
    const binAxis = chart.type === 'horizontalBar' ? chart.yAxis : chart.xAxis;
    if (
      binAxis?.aggregation !== 'uniqueBin' &&
      binAxis?.aggregation !== 'numericBin' &&
      (binAxis?.aggregation === 'timeBin' ||
        fields.some(field => field.name === binAxis?.field?.name && field.kind === 'time'))
    )
      return planResidentTimeChart(chart, datasetId, fields);
    return planResidentCategoryChart(chart, datasetId, fields);
  }
  if (
    chart.type === 'lineChart' &&
    (chart.xAxis?.aggregation === 'timeBin' ||
      fields.some(field => field.name === chart.xAxis?.field?.name && field.kind === 'time'))
  )
    return planResidentTimeChart(chart, datasetId, fields);
  if (chart.type === 'lineChart')
    return planResidentCategoryChart(
      {...chart, type: 'barChart', numGroups: 0, groupOthers: false},
      datasetId,
      fields
    );
  if (chart.type !== 'bigNumber')
    return {
      supported: false,
      reason: 'This chart needs a resident grouped-summary or layer-summary adapter.'
    };
  const aggregation = chart.axis?.aggregation || 'count';
  const column = chart.axis?.field?.name;
  const field = fields.find(candidate => candidate.name === column);
  const operation =
    aggregation === 'count'
      ? 'count'
      : aggregation === 'sum'
        ? 'sum'
        : aggregation === 'average'
          ? 'mean'
          : aggregation === 'minimum'
            ? 'min'
            : aggregation === 'maximum'
              ? 'max'
              : aggregation === 'median'
                ? 'median'
                : aggregation === 'stdev'
                  ? 'standardDeviation'
                  : undefined;
  // Native aggregateIndexes counts rows when no value field is selected. A removed field instead
  // receives an explicit error: do not relabel a row count as a missing measurement's statistic.
  if (column && !field)
    return {supported: false, reason: `Chart field "${column}" is unavailable in this source.`};
  if (column && aggregation !== 'count' && !operation)
    return {
      supported: false,
      reason: `The ${aggregation} measure needs a resident statistical-summary adapter.`
    };
  if (column && aggregation !== 'count' && field?.kind === 'time')
    return {
      supported: false,
      reason: 'Choose a numeric measurement; exact timestamps are grouping fields.'
    };
  if (column && aggregation !== 'count' && field?.categorical)
    return {supported: false, reason: 'Choose a numeric measurement rather than dictionary codes.'};
  return {
    supported: true,
    plan: {
      kind: 'scalar',
      chartId: chart.id,
      datasetId,
      population: chart.applyFilters ? 'selected' : 'all',
      operation: column ? (operation ?? 'count') : 'count',
      ...(column && operation !== 'count' ? {column} : {})
    }
  };
}

/** Cosmetic chart changes do not rebuild a GPU program; source revision is checked by its owner. */
export function getResidentChartPlanKey(plan: ResidentChartPlan): string {
  return JSON.stringify([
    plan.kind,
    plan.datasetId,
    plan.population,
    plan.operation,
    plan.column ?? null,
    plan.kind === 'scalar' ? null : (plan.series ?? null),
    plan.kind === 'category' ? (plan.numericSeries ?? null) : null,
    ...(plan.kind === 'category'
      ? [
          plan.categoryColumn,
          plan.categoryCount,
          plan.topGroups?.limit ?? null,
          plan.excludedCpuFilterIds,
          plan.excludedGpuFieldNames
        ]
      : plan.kind === 'numeric'
        ? [
            plan.binColumn,
            plan.binCount,
            plan.binFormat,
            plan.integerGroupLimit ?? null,
            plan.excludedCpuFilterIds,
            plan.excludedGpuFieldNames
          ]
        : plan.kind === 'time'
          ? [
              plan.timeColumn,
              plan.interval,
              plan.binCount ?? null,
              plan.dateTimeLabels ?? false,
              plan.excludedCpuFilterIds,
              plan.excludedGpuFieldNames
            ]
          : [])
  ]);
}

function planResidentCategoryChart(
  chart: NativeChartConfiguration,
  datasetId: string,
  fields: readonly ResidentChartField[]
): ResidentChartPlanResult {
  if (chart.type !== 'barChart' && chart.type !== 'horizontalBar')
    return {supported: false, reason: 'Select a categorical bar chart.'};
  const categoryAxis = chart.type === 'horizontalBar' ? chart.yAxis : chart.xAxis;
  const measureAxis = chart.type === 'horizontalBar' ? chart.xAxis : chart.yAxis;
  const categoryColumn = categoryAxis?.field?.name;
  const category = fields.find(field => field.name === categoryColumn);
  if (!categoryColumn || !category)
    return {
      supported: false,
      reason: 'Choose an available category field for the grouped-summary chart.'
    };
  const seriesColumn = chart.groupBy?.field?.name;
  const seriesField = fields.find(field => field.name === seriesColumn);
  const numericSeriesField =
    seriesColumn &&
    seriesField &&
    seriesField.kind !== 'time' &&
    !seriesField.categorical &&
    (chart.groupBy?.field?.type === 'real' || chart.groupBy?.field?.type === 'integer')
      ? seriesField
      : undefined;
  if (
    seriesColumn &&
    !numericSeriesField &&
    (!seriesField?.categorical ||
      chart.groupBy?.field?.type !== 'string' ||
      !seriesField.categoryCount ||
      seriesField.categoryCount > MAXIMUM_RESIDENT_CHART_CATEGORIES)
  )
    return {
      supported: false,
      reason:
        'Chart series require an available prepared string dictionary with at most 512 entries.'
    };
  if (category.kind === 'time')
    return {
      supported: false,
      reason:
        'Use a line chart with a UTC hour, day, week, month or year interval for exact timestamps; equal-width time bars need their own adapter.'
    };
  const numeric =
    !category.categorical &&
    categoryAxis?.aggregation !== 'timeBin' &&
    (categoryAxis?.field?.type === 'real' ||
      categoryAxis?.field?.type === 'integer' ||
      categoryAxis?.aggregation === 'numericBin');
  const integerAuto =
    numeric &&
    categoryAxis?.field?.type === 'integer' &&
    categoryAxis?.aggregation !== 'numericBin';
  if (
    !numeric &&
    (!category.categorical ||
      categoryAxis?.field?.type !== 'string' ||
      categoryAxis.aggregation === 'numericBin' ||
      categoryAxis.aggregation === 'timeBin')
  )
    return {
      supported: false,
      reason:
        'Choose a prepared string dictionary or numeric field. Time intervals need their own adapter.'
    };
  const requestedGroups = (chart.numGroups ?? 10) > 0 ? (chart.numGroups ?? 10) : 10;
  const requestedIntegerLimit = Math.max(0, chart.numGroups ?? 10);
  if (integerAuto && !Number.isSafeInteger(requestedIntegerLimit))
    return {supported: false, reason: 'Integer charts require a whole-number group budget.'};
  if (numeric && !Number.isSafeInteger(requestedGroups))
    return {supported: false, reason: 'Numeric charts require a whole-number group budget.'};
  // Above the native 1,000-point cap numeric charts fall back to 1,000 equal-width bins (with a note).
  const binBudget = clampNumericGroupBudget(requestedGroups);
  const integerBudget = clampIntegerGroupLimit(requestedIntegerLimit);
  const binCount = binBudget.groups;
  const integerGroupLimit = integerBudget.groups;
  const notes: string[] = numeric
    ? [binBudget.note, integerAuto ? integerBudget.note : undefined].filter(
        (note): note is string => Boolean(note)
      )
    : [];
  if (!numeric && !category.categoryCount)
    return {
      supported: false,
      reason: 'Category charts require a prepared dictionary with at least one entry.'
    };
  // Above the 1,000-group cap: top groups ranked on the GPU plus an Other bucket, up to a bounded dictionary.
  const overflow = !numeric && category.categoryCount! > MAXIMUM_RESIDENT_GROUPS;
  if (overflow && category.categoryCount! > MAXIMUM_RESIDENT_TOP_DICTIONARY)
    return {supported: false, reason: describeCategoryOverflow(category.categoryCount!)};
  const topLimit = overflow
    ? getResidentTopGroupLimit(seriesField?.categoryCount, MAXIMUM_RESIDENT_CHART_CELLS)
    : undefined;
  if (topLimit !== undefined)
    notes.push(describeCategoryTopGroups(category.categoryCount!, topLimit));
  const groupCount = numeric
    ? getResidentNumericGroupCount({
        binCount,
        integerGroupLimit: integerAuto ? integerGroupLimit : undefined
      })
    : topLimit !== undefined
      ? topLimit + 1
      : category.categoryCount!;
  // A numeric series is binned into equal-width intervals that fit the cell budget beside the groups.
  let seriesCategoryCount = seriesField?.categoryCount;
  if (numericSeriesField) {
    if (numeric)
      return {
        supported: false,
        reason:
          'A numeric series needs a category axis; numeric-bin axes with a numeric series are not supported.'
      };
    const available = Math.floor(MAXIMUM_RESIDENT_CHART_CELLS / (groupCount + 1)) - 1;
    const requestedSeriesBins = Math.max(2, requestedGroups);
    seriesCategoryCount = Math.min(requestedSeriesBins, available, MAXIMUM_RESIDENT_GROUPS);
    if (seriesCategoryCount < 2)
      return {
        supported: false,
        reason: `Too many groups (${groupCount.toLocaleString()}) to bin a numeric series within the ${MAXIMUM_RESIDENT_CHART_CELLS}-cell summary budget.`
      };
    notes.push(
      `Series "${seriesColumn}" is numeric: grouped into ${seriesCategoryCount} equal-width bins of its range when the chart was prepared${seriesCategoryCount < requestedSeriesBins ? ` (${requestedSeriesBins} requested, limited by the ${MAXIMUM_RESIDENT_CHART_CELLS}-cell budget)` : ''}.`
    );
  }
  if (seriesColumn && groupCount * (seriesCategoryCount! + 1) > MAXIMUM_RESIDENT_CHART_CELLS)
    return {
      supported: false,
      reason: `Chart groups and series exceed the ${MAXIMUM_RESIDENT_CHART_CELLS}-cell summary budget. Choose fewer numeric bins or a smaller dictionary.`
    };
  const scalar = planResidentChart(
    {...chart, type: 'bigNumber', axis: measureAxis},
    datasetId,
    fields
  );
  if (!scalar.supported) return scalar;
  const filterId = chart.crossFilter?.enabled ? chart.crossFilter.filterId : undefined;
  const common = {
    chartId: chart.id,
    datasetId,
    population: scalar.plan.population,
    operation: scalar.plan.operation,
    ...(seriesColumn ? {series: {column: seriesColumn, categoryCount: seriesCategoryCount!}} : {}),
    ...(numericSeriesField ? {numericSeries: {binFormat: numericSeriesField.format!}} : {}),
    ...(scalar.plan.column ? {column: scalar.plan.column} : {}),
    excludedCpuFilterIds: filterId ? [filterId, `${filterId}-x`, `${filterId}-y`] : [],
    excludedGpuFieldNames: chart.crossFilter?.enabled ? [categoryColumn] : []
  };
  if (numeric)
    return {
      supported: true,
      ...(notes.length ? {notes} : {}),
      plan: {
        ...common,
        kind: 'numeric',
        binColumn: categoryColumn,
        binCount,
        binFormat: category.format,
        ...(integerAuto ? {integerGroupLimit} : {})
      }
    };
  return {
    supported: true,
    ...(notes.length ? {notes} : {}),
    plan: {
      ...common,
      kind: 'category',
      categoryColumn,
      categoryCount: category.categoryCount!,
      ...(topLimit !== undefined ? {topGroups: {limit: topLimit}} : {})
    }
  };
}

/** Hover charts have a separate synchronous CPU consumer, outside the native chart panel. */
export function filterResidentHoverCharts(
  charts: readonly NativeChartConfiguration[],
  layerId: string | undefined,
  datasetId: string | undefined,
  isResidentDataset: (datasetId: string) => boolean
): {charts: NativeChartConfiguration[]; blocked: boolean} {
  let blocked = false;
  const allowed = charts.filter(chart => {
    if (!layerId || !('layerId' in chart) || chart.layerId !== layerId) return true;
    if (
      (datasetId && isResidentDataset(datasetId)) ||
      (chart.dataId && isResidentDataset(chart.dataId))
    ) {
      blocked = true;
      return false;
    }
    return true;
  });
  return {charts: allowed, blocked};
}

/** Positive integer auto limits need only enough groups for either unique values or fallback bins. */
export function getResidentNumericGroupCount(
  plan: Pick<ResidentNumericChartPlan, 'binCount' | 'integerGroupLimit'>
): number {
  return plan.integerGroupLimit === undefined
    ? plan.binCount
    : plan.integerGroupLimit > 0
      ? Math.max(plan.binCount, plan.integerGroupLimit)
      : MAXIMUM_RESIDENT_GROUPS;
}

function planResidentTimeChart(
  chart: NativeChartConfiguration,
  datasetId: string,
  fields: readonly ResidentChartField[]
): ResidentChartPlanResult {
  if (chart.type !== 'lineChart' && chart.type !== 'barChart' && chart.type !== 'horizontalBar')
    return {supported: false, reason: 'Choose a calendar line or time bar chart.'};
  const horizontal = chart.type === 'horizontalBar';
  const binAxis = horizontal ? chart.yAxis : chart.xAxis;
  const valueAxis = horizontal ? chart.xAxis : chart.yAxis;
  const timeColumn = binAxis?.field?.name;
  const field = fields.find(candidate => candidate.name === timeColumn);
  if (
    !timeColumn ||
    field?.kind !== 'time' ||
    (field.unit !== 'millisecond' &&
      field.unit !== 'second' &&
      field.unit !== 'microsecond' &&
      field.unit !== 'nanosecond') ||
    !/^(Timestamp|Date)/.test(field.sourceType)
  )
    return {
      supported: false,
      reason:
        'Calendar charts require a prepared second, millisecond, microsecond or nanosecond Arrow Timestamp or Date field.'
    };
  if (binAxis?.aggregation === 'uniqueBin' || binAxis?.aggregation === 'numericBin')
    return {supported: false, reason: 'Choose Time intervals for an exact timestamp chart.'};
  // Native bars pass numGroups (default 10) as an equal-width bin count and ignore the interval;
  // lines and non-positive bar group counts use calendar periods.
  // A non-positive bar group count takes the same native calendar-period path as a line.
  const equalWidthCount = chart.type === 'lineChart' ? 0 : (chart.numGroups ?? 10);
  const isEqualWidth = chart.type !== 'lineChart' && equalWidthCount > 0;
  const binCount = Math.max(2, equalWidthCount);
  if (isEqualWidth && (!Number.isSafeInteger(binCount) || binCount > MAXIMUM_RESIDENT_TIME_GROUPS))
    return {
      supported: false,
      reason: `Time bar charts require at most ${MAXIMUM_RESIDENT_TIME_GROUPS} equal-width bins within their bounded summary budget.`
    };
  const requestedInterval = binAxis?.interval ?? chart.chartDisplay?.['interval'];
  // Native passes `interval || undefined`: unset, null and empty strings select automatic periods.
  // A literal "auto" is truthy, so native floors to UTC days but formats labels with the time.
  const literalAuto = !isEqualWidth && requestedInterval === 'auto';
  const interval = isEqualWidth
    ? 'equal-width'
    : literalAuto
      ? 'day'
      : requestedInterval
        ? requestedInterval
        : 'auto';
  if (
    interval !== 'equal-width' &&
    interval !== 'auto' &&
    interval !== 'hour' &&
    interval !== 'day' &&
    interval !== 'week' &&
    interval !== 'month' &&
    interval !== 'year'
  )
    return {
      supported: false,
      reason:
        'Choose UTC hour, day, week, month or year, or leave the interval unset for automatic periods.'
    };
  const seriesColumn = chart.groupBy?.field?.name;
  const series = fields.find(candidate => candidate.name === seriesColumn);
  if (
    seriesColumn &&
    (!series?.categorical || chart.groupBy?.field?.type !== 'string' || !series.categoryCount)
  )
    return {
      supported: false,
      reason: 'Chart series require an available prepared string dictionary.'
    };
  if (
    seriesColumn &&
    (isEqualWidth ? binCount : MAXIMUM_RESIDENT_CHART_CATEGORIES) * (series!.categoryCount! + 1) >
      MAXIMUM_RESIDENT_CHART_CELLS
  )
    return {
      supported: false,
      reason: 'Calendar groups and series exceed the 4,096-cell summary budget.'
    };
  const scalar = planResidentChart(
    {...chart, type: 'bigNumber', axis: valueAxis},
    datasetId,
    fields
  );
  if (!scalar.supported) return scalar;
  const filterId = chart.crossFilter?.enabled ? chart.crossFilter.filterId : undefined;
  return {
    supported: true,
    plan: {
      kind: 'time',
      chartId: chart.id,
      datasetId,
      timeColumn,
      interval,
      ...(isEqualWidth ? {binCount} : {}),
      ...(literalAuto ? {dateTimeLabels: true} : {}),
      population: scalar.plan.population,
      operation: scalar.plan.operation,
      ...(scalar.plan.column ? {column: scalar.plan.column} : {}),
      ...(seriesColumn
        ? {series: {column: seriesColumn, categoryCount: series!.categoryCount!}}
        : {}),
      excludedCpuFilterIds: filterId ? [filterId, `${filterId}-x`, `${filterId}-y`] : [],
      excludedGpuFieldNames: chart.crossFilter?.enabled ? [timeColumn] : []
    }
  };
}

/** Chart-origin policy comes from native configuration, never an arbitrary filter-ID prefix. */
export function getResidentChartTimeNullEpochFilters(
  charts: readonly NativeChartConfiguration[],
  datasetId: string,
  timeFields: readonly string[]
): {filterId: string; field: string}[] {
  const fields = new Set(timeFields);
  return charts.flatMap(chart => {
    if (
      chart.dataId !== datasetId ||
      !chart.crossFilter?.enabled ||
      !chart.crossFilter.filterId ||
      (chart.type !== 'lineChart' && chart.type !== 'barChart' && chart.type !== 'horizontalBar')
    )
      return [];
    const axis = chart.type === 'horizontalBar' ? chart.yAxis : chart.xAxis;
    const field = axis?.field?.name;
    if (
      !field ||
      !fields.has(field) ||
      chart.crossFilter.fieldNames?.['x'] !== field ||
      axis.aggregation === 'uniqueBin' ||
      axis.aggregation === 'numericBin'
    )
      return [];
    return [{filterId: chart.crossFilter.filterId, field}];
  });
}
