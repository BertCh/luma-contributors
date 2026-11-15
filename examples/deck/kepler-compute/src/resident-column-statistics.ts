// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ColumnStats} from '@kepler.gl/utils';
import type * as arrow from 'apache-arrow';
import {getOriginalFieldType, readOriginalColumnValues} from './original-column-values';
import {MAXIMUM_ORIGINAL_SORT_ROWS} from './original-table-sort';
import type {ResidentTableProvider} from './resident-data-table';
import {makeNumericColumnStatsFromGPU} from './resident-gpu-column-statistics';
import {chooseColumnStatisticsRoute} from './resident-table-population-routes';
import type {GPUColumnStatistics, GPUColumnStatisticsRoute} from '@luma-contributors/kepler';
import type {
  ResidentTableWideStatistics,
  ResidentWideColumnStatistics
} from './resident-table-wide-statistics';

type ColumnStatsComputer = (values: unknown[], fieldType: string) => ColumnStats;
let computeColumnStats: ColumnStatsComputer | null = null;

/** Installs native `computeColumnStats` from @kepler.gl/utils; called once by `native-utils-host`. */
export function setColumnStatsComputer(nextComputer: ColumnStatsComputer): void {
  computeColumnStats = nextComputer;
}

export type ResidentColumnValuesRequest = {
  column: string;
  population: 'all' | 'selected';
};
export type ResidentColumnStatistics = {
  column: string;
  population: 'all' | 'selected';
  fieldType: string;
  rowCount: number;
  nullCount: number;
  /** Native Kepler column statistics (histogram, quantiles, mean, deviation, unique values, range). */
  stats: ColumnStats;
  /** Where the reduction ran. GPU results have histogram-estimated interior quantiles. */
  source?: 'gpu' | 'cpu';
  /** Stated route reason (why the GPU or CPU path was chosen). */
  routeReason?: string;
};
/** Optional GPU reduction supplied by the table provider (Float32 columns, any row count up to 2^24). */
export type ResidentGPUColumnStatistics = {
  plan(column: string): GPUColumnStatisticsRoute;
  read(request: ResidentColumnValuesRequest): Promise<GPUColumnStatistics | null>;
};
export type ResidentColumnValues = {
  values: unknown[];
  fieldType: string;
};

/**
 * Bounded asynchronous column values and Kepler column statistics for the current population.
 * Original Arrow columns read only the population's rows; derived and prepared GPU columns return
 * through bounded gather pages. Populations above MAXIMUM_ORIGINAL_SORT_ROWS are refused so no
 * unbounded CPU scan or readback ever happens.
 */
export function createColumnStatisticsReader(props: {
  source: arrow.Table;
  provider: ResidentTableProvider;
  readPopulationRows(population: 'all' | 'selected'): Promise<Uint32Array | null>;
  isCurrent(): boolean;
  /** Prefer the GPU profile for Float32 columns; it has no 262,144-row bound. */
  gpu?: ResidentGPUColumnStatistics;
  /** GPU profile for dictionary strings, Int64, timestamp and Float64 above the CPU bound. */
  wide?: Pick<ResidentTableWideStatistics, 'plan' | 'read'>;
  /** Total rows of the source table, for route decisions when the population is not yet read. */
  getPopulationRows?(population: 'all' | 'selected'): number | Promise<number>;
}) {
  const {source, provider} = props;
  const readValues = async (
    request: ResidentColumnValuesRequest,
    exact: boolean
  ): Promise<ResidentColumnValues | null> => {
    const revision = provider.getRevision();
    const vector = source.getChild(request.column);
    const field = provider.fields.find(candidate => candidate.name === request.column);
    if (!vector && !field) throw new Error('Unknown statistics column');
    const rows = await props.readPopulationRows(request.population);
    if (!rows || provider.getRevision() !== revision || !props.isCurrent()) return null;
    if (vector) {
      return {
        values: readOriginalColumnValues(vector, rows, exact),
        fieldType: getOriginalFieldType(vector.type)
      };
    }
    if (!provider.readRows) throw new Error('This table cannot read derived columns by row');
    const values = new Array<unknown>(rows.length);
    for (let start = 0; start < rows.length; start += provider.maximumPageSize) {
      const chunk = Array.from(rows.subarray(start, start + provider.maximumPageSize));
      const page = await provider.readRows(chunk, [request.column]);
      if (!page || provider.getRevision() !== revision || !props.isCurrent()) return null;
      const numbers = page.columns[request.column];
      const validity = page.validity[request.column];
      for (let index = 0; index < chunk.length; index++)
        values[start + index] =
          validity && validity[index] === 0 ? null : (numbers[index] as number | string);
    }
    return {values, fieldType: 'real'};
  };
  return {
    readColumnValues(request: ResidentColumnValuesRequest): Promise<ResidentColumnValues | null> {
      return readValues(request, true);
    },
    async readColumnStatistics(
      request: ResidentColumnValuesRequest
    ): Promise<ResidentColumnStatistics | null> {
      const revision = provider.getRevision();
      const gpuPlan = props.gpu?.plan(request.column);
      const isFloat32Source =
        !source.getChild(request.column) ||
        String(source.getChild(request.column)!.type) === 'Float32';
      const populationRows = props.getPopulationRows
        ? await props.getPopulationRows(request.population)
        : source.numRows;
      const widePlan =
        source.getChild(request.column) && !(gpuPlan?.supported && isFloat32Source)
          ? props.wide?.plan(request.column)
          : undefined;
      const route = chooseColumnStatisticsRoute({
        gpuSupported: Boolean(gpuPlan?.supported && isFloat32Source),
        gpuReason:
          gpuPlan && !gpuPlan.supported
            ? gpuPlan.reason
            : isFloat32Source
              ? undefined
              : 'GPU statistics reduce Float32 columns; this source type uses the bounded CPU path.',
        populationRows,
        wideSupported: widePlan?.supported,
        wideReason: widePlan?.reason
      });
      if (route.route === 'refused') throw new Error(route.reason);
      // A prepared Float32 copy can be profileable even when the original Int64/Float64 source
      // needs relative-to-minimum profiling. Route on the original source eligibility above.
      if (route.route === 'gpu' && widePlan?.supported) {
        const profile: ResidentWideColumnStatistics | null = await props.wide!.read(request);
        if (!profile || provider.getRevision() !== revision || !props.isCurrent()) return null;
        return {
          column: request.column,
          population: request.population,
          fieldType: profile.fieldType,
          rowCount: profile.rowCount,
          nullCount: profile.nullCount,
          stats: profile.stats,
          source: 'gpu',
          routeReason: profile.routeReason
        };
      }
      if (route.route === 'gpu') {
        const profile = await props.gpu!.read(request);
        if (!profile || provider.getRevision() !== revision || !props.isCurrent()) return null;
        return {
          column: request.column,
          population: request.population,
          fieldType: 'real',
          rowCount: profile.populationRows,
          nullCount: profile.nullCount,
          stats: makeNumericColumnStatsFromGPU(profile),
          source: 'gpu',
          routeReason: route.reason
        };
      }
      const result = await readValues(request, false);
      if (!result) return null;
      const nullCount = result.values.reduce<number>(
        (count, value) =>
          count + (value === null || value === undefined || Number.isNaN(value) ? 1 : 0),
        0
      );
      return {
        column: request.column,
        population: request.population,
        fieldType: result.fieldType,
        rowCount: result.values.length,
        nullCount,
        // Host has not installed computeColumnStats: import './native-utils-host' from the browser entry.
        stats: computeColumnStats!(result.values, result.fieldType),
        source: 'cpu',
        routeReason: route.reason
      };
    }
  };
}

export {MAXIMUM_ORIGINAL_SORT_ROWS as MAXIMUM_COLUMN_STATISTICS_ROWS};
