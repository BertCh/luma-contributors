// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {COORDINATE_SYSTEM} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUDataFrame, column, literal, parameter} from '@luma-contributors/graph/gpu-dataframe';
import {GPURecordBatch, GPUTable} from '@luma-contributors/graph/gpu-tables';
import type {GPUDataFrameQueryParameters} from '@luma-contributors/graph/gpu-dataframe';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';

const DATASET_ID = 'poopdeck-earthquakes';
const MINIMUM_YEAR = 2020;
const MAXIMUM_YEAR = 2024;
const HISTOGRAM_BINS = 20;
const TOP_K_LIMITS = [50, 250] as const;
const SORT_DIRECTIONS = ['ascending', 'descending'] as const;

export type EarthquakeQueryLabOptions = {
  minimumMagnitude: number;
  minimumDepth: number;
  maximumDepth: number;
  startYear: number;
  endYear: number;
  rankCount: '50' | '250';
  rankDirection: 'ascending' | 'descending';
  includeOtherTypes: boolean;
  showContext: boolean;
};

type EarthquakeColumns = {
  position: 'float32x2';
  magnitude: 'float32';
  depth: 'float32';
  year: 'float32';
  eventType: 'uint32';
};

type EventTypeColumns = {typeCode: 'uint32'};

function physicalBuffer(data: GPUData): Buffer {
  return data.buffer instanceof Buffer ? data.buffer : data.buffer.buffer;
}

function vectorFromBuffer<Format extends 'float32' | 'float32x2' | 'uint32'>(
  name: string,
  format: Format,
  buffer: Buffer,
  rowCount: number,
  byteOffset: number
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    name,
    format,
    data: [new GPUData<Format>({buffer, format, length: rowCount, byteOffset, ownsBuffer: false})],
    ownsData: false
  });
}

function makeYearPartitions(
  timestamps: Uint32Array,
  timeOriginMs: number
): Array<{year: number; start: number; count: number}> {
  const starts = new Array<number>(MAXIMUM_YEAR - MINIMUM_YEAR + 2).fill(timestamps.length);
  for (let row = 0; row < timestamps.length; row++) {
    const year = new Date(timeOriginMs + timestamps[row] * 1000).getUTCFullYear();
    if (year >= MINIMUM_YEAR && year <= MAXIMUM_YEAR) {
      const yearIndex = year - MINIMUM_YEAR;
      if (starts[yearIndex] === timestamps.length) starts[yearIndex] = row;
      starts[yearIndex + 1] = row + 1;
    }
  }
  return Array.from({length: MAXIMUM_YEAR - MINIMUM_YEAR + 1}, (_, index) => {
    const start = starts[index] === timestamps.length ? 0 : starts[index];
    const end = starts[index + 1] === timestamps.length ? timestamps.length : starts[index + 1];
    return {year: MINIMUM_YEAR + index, start, count: Math.max(0, end - start)};
  });
}

export async function createEarthquakeQueryLab(
  ctx: SceneContext<EarthquakeQueryLabOptions>
): Promise<SceneInstance<EarthquakeQueryLabOptions>> {
  const {device} = ctx;
  const dataset = ctx.datasets.get(DATASET_ID);
  const resources = new SpatialAnalysisResources(device, 'earthquake-query-lab');
  const rowCount = dataset.count;
  const positions = dataset.column<Float32Array>('position');
  const magnitudes = dataset.column<Float32Array>('magnitude');
  const depths = dataset.column<Float32Array>('depth');
  const timestamps = dataset.column<Uint32Array>('timestamp');
  const sourceTypes = dataset.column<Uint8Array>('type');
  const eventTypes = Uint32Array.from(sourceTypes);
  const sourceYears = new Float32Array(rowCount);
  const timeOriginMs = Number(dataset.properties.timeOriginMs);
  for (let row = 0; row < rowCount; row++) {
    sourceYears[row] = new Date(timeOriginMs + timestamps[row] * 1000).getUTCFullYear();
  }

  const positionBuffer = resources.createBuffer('positions-lon-lat', positions);
  const magnitudeBuffer = resources.createBuffer('magnitude', magnitudes);
  const depthBuffer = resources.createBuffer('depth', depths);
  const yearBuffer = resources.createBuffer('year', sourceYears);
  const eventTypeBuffer = resources.createBuffer('event-type', eventTypes);
  const partitions = makeYearPartitions(timestamps, timeOriginMs);
  const fields = [
    {name: 'position', format: 'float32x2' as const, nullable: false},
    {name: 'magnitude', format: 'float32' as const, nullable: false},
    {name: 'depth', format: 'float32' as const, nullable: false},
    {name: 'year', format: 'float32' as const, nullable: false},
    {name: 'eventType', format: 'uint32' as const, nullable: false}
  ];
  const sourceBuffers = {
    position: positionBuffer,
    magnitude: magnitudeBuffer,
    depth: depthBuffer,
    year: yearBuffer,
    eventType: eventTypeBuffer
  };
  const createBatch = (start: number, count: number, sourceBatchIndex: number) => {
    const sourceInfo = {sourceBatchIndex, sourceRowIndexOffset: start, sourceRowCount: count};
    const gpuData = {
      position: new GPUData<'float32x2'>({
        buffer: sourceBuffers.position,
        format: 'float32x2',
        length: count,
        byteOffset: start * 8,
        ownsBuffer: false
      }),
      magnitude: new GPUData<'float32'>({
        buffer: sourceBuffers.magnitude,
        format: 'float32',
        length: count,
        byteOffset: start * 4,
        ownsBuffer: false
      }),
      depth: new GPUData<'float32'>({
        buffer: sourceBuffers.depth,
        format: 'float32',
        length: count,
        byteOffset: start * 4,
        ownsBuffer: false
      }),
      year: new GPUData<'float32'>({
        buffer: sourceBuffers.year,
        format: 'float32',
        length: count,
        byteOffset: start * 4,
        ownsBuffer: false
      }),
      eventType: new GPUData<'uint32'>({
        buffer: sourceBuffers.eventType,
        format: 'uint32',
        length: count,
        byteOffset: start * 4,
        ownsBuffer: false
      })
    };
    return new GPURecordBatch<EarthquakeColumns>({gpuData, fields, numRows: count, sourceInfo});
  };
  const yearBatches = partitions.map(({start, count}, sourceBatchIndex) =>
    createBatch(start, count, sourceBatchIndex)
  );
  const eventTypeNames = dataset.categories('type');
  const earthquakeTypeCode = Math.max(0, eventTypeNames.indexOf('earthquake'));
  const frame = new GPUDataFrame<EarthquakeColumns>({
    table: new GPUTable<EarthquakeColumns>({batches: [createBatch(0, rowCount, 0)]}),
    dictionaries: {eventType: eventTypeNames},
    ownership: 'owned'
  });
  const yearFrame = new GPUDataFrame<EarthquakeColumns>({
    table: new GPUTable<EarthquakeColumns>({batches: yearBatches}),
    dictionaries: {eventType: eventTypeNames},
    ownership: 'owned'
  });
  const selectedFrame = frame.select(['position', 'magnitude', 'depth', 'year', 'eventType']);
  const typeCodeBuffer = resources.createBuffer(
    'event-type-dimension',
    Uint32Array.from({length: eventTypeNames.length}, (_, index) => index)
  );
  const typeDimension = new GPUDataFrame<EventTypeColumns>({
    table: new GPUTable<EventTypeColumns>({
      vectors: {
        typeCode: vectorFromBuffer('typeCode', 'uint32', typeCodeBuffer, eventTypeNames.length, 0)
      }
    }),
    dictionaries: {typeCode: eventTypeNames},
    ownership: 'borrowed'
  });
  resources.track({destroy: () => typeDimension.destroy()});

  const predicate = selectedFrame
    .withColumn('yearsFromStart', column('year').subtract(parameter('startYear', MINIMUM_YEAR)))
    .filter(column('magnitude').greaterThanOrEqual(parameter('minimumMagnitude', 4)))
    .filter(
      column('yearsFromStart')
        .greaterThanOrEqual(literal(0))
        .and(column('year').lessThanOrEqual(parameter('endYear', MAXIMUM_YEAR)))
    )
    .filter(
      column('depth')
        .greaterThanOrEqual(parameter('minimumDepth', -100))
        .and(column('depth').lessThanOrEqual(parameter('maximumDepth', 700)))
    )
    .filter(
      column('eventType')
        .equal(parameter('earthquakeType', earthquakeTypeCode))
        .or(parameter('includeOtherTypes', 0).greaterThan(literal(0)))
    );
  const queryParameters = (): GPUDataFrameQueryParameters => ({
    minimumMagnitude: ctx.options.minimumMagnitude,
    minimumDepth: ctx.options.minimumDepth,
    maximumDepth: ctx.options.maximumDepth,
    startYear: ctx.options.startYear,
    endYear: ctx.options.endYear,
    earthquakeType: earthquakeTypeCode,
    includeOtherTypes: ctx.options.includeOtherTypes ? 1 : 0
  });

  const summaryQuery = predicate
    .aggregate({
      events: 'count',
      meanMagnitude: {mean: 'magnitude'},
      maximumMagnitude: {max: 'magnitude'},
      minimumDepth: {min: 'depth'}
    })
    .compile(new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'quake-query-summary'}));
  const histogramQuery = predicate
    .histogram('magnitude', {bins: HISTOGRAM_BINS, domain: [4, 9]})
    .compile(
      new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {
        id: 'quake-query-magnitude-histogram'
      })
    );
  const globalTopQueries = TOP_K_LIMITS.flatMap(limit =>
    SORT_DIRECTIONS.map(direction => ({
      limit,
      direction,
      query: predicate.topKGlobal('magnitude', limit, {direction}).compile(
        new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {
          id: `quake-query-global-top-${direction}-${limit}`
        })
      )
    }))
  );
  const batchPredicate = yearFrame
    .withColumn('yearsFromStart', column('year').subtract(parameter('startYear', MINIMUM_YEAR)))
    .filter(column('magnitude').greaterThanOrEqual(parameter('minimumMagnitude', 4)))
    .filter(
      column('yearsFromStart')
        .greaterThanOrEqual(literal(0))
        .and(column('year').lessThanOrEqual(parameter('endYear', MAXIMUM_YEAR)))
    )
    .filter(
      column('depth')
        .greaterThanOrEqual(parameter('minimumDepth', -100))
        .and(column('depth').lessThanOrEqual(parameter('maximumDepth', 700)))
    )
    .filter(
      column('eventType')
        .equal(parameter('earthquakeType', earthquakeTypeCode))
        .or(parameter('includeOtherTypes', 0).greaterThan(literal(0)))
    );
  const batchTopQueries = TOP_K_LIMITS.flatMap(limit =>
    SORT_DIRECTIONS.map(direction => ({
      limit,
      direction,
      query: batchPredicate.topK('magnitude', limit, {direction}).compile(
        new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {
          id: `quake-query-batch-top-${direction}-${limit}`
        })
      )
    }))
  );
  // Probe the complete archive so filtered-out rows are not counted as missing keys.
  const typeLookupQuery = selectedFrame
    .lookup(typeDimension, {leftOn: 'eventType', rightOn: 'typeCode'})
    .compile(
      new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'quake-query-type-lookup'})
    );
  const earthquakeTypes = new GPUDataFrame<EventTypeColumns>({
    table: new GPUTable<EventTypeColumns>({
      vectors: {
        typeCode: vectorFromBuffer('typeCode', 'uint32', typeCodeBuffer, 1, earthquakeTypeCode * 4)
      }
    }),
    dictionaries: {typeCode: eventTypeNames},
    ownership: 'borrowed'
  });
  resources.track({destroy: () => earthquakeTypes.destroy()});
  const otherTypeQuery = frame
    .antiJoin(earthquakeTypes, {leftOn: 'eventType', rightOn: 'typeCode'})
    .compile(
      new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'quake-query-other-types'})
    );

  const summaryData = summaryQuery.table.gpuVectors;
  const histogramData = histogramQuery.table.gpuVectors.count;
  const summaryBuffers = [
    summaryData.events.data[0],
    summaryData.meanMagnitude.data[0],
    summaryData.maximumMagnitude.data[0],
    summaryData.minimumDepth.data[0]
  ];
  const reader = new SummaryReader(
    resources,
    'earthquake-query-results',
    [
      ...summaryBuffers.map(data => ({buffer: physicalBuffer(data), size: 4})),
      {buffer: physicalBuffer(histogramData.data[0]), size: HISTOGRAM_BINS * 4},
      ...globalTopQueries.flatMap(({query}) =>
        query.globalSelectedCount.data.map(data => ({buffer: physicalBuffer(data), size: 4}))
      ),
      ...batchTopQueries.flatMap(({query}) =>
        query.selectedCounts.data.map(data => ({buffer: physicalBuffer(data), size: 4}))
      ),
      {buffer: physicalBuffer(typeLookupQuery.lookupStatistics.data[0]), size: 16},
      {buffer: physicalBuffer(otherTypeQuery.selectedCounts.data[0]), size: 4},
      {buffer: physicalBuffer(otherTypeQuery.overflows.data[0]), size: 4}
    ],
    bytes => {
      if (readbackSignature !== latestQuerySignature) return;
      const view = new DataView(bytes);
      let offset = 0;
      const eventTotal = view.getUint32(offset, true);
      offset += 4;
      const meanMagnitude = view.getFloat32(offset, true);
      offset += 4;
      const maximumMagnitude = view.getFloat32(offset, true);
      offset += 4;
      const minimumDepth = view.getFloat32(offset, true);
      offset += 4;
      const histogram = Float32Array.from(
        new Uint32Array(bytes.slice(offset, offset + HISTOGRAM_BINS * 4))
      );
      offset += HISTOGRAM_BINS * 4;
      const globalTopCounts = new Map<string, number>();
      for (const {limit, direction} of globalTopQueries) {
        globalTopCounts.set(`${direction}:${limit}`, view.getUint32(offset, true));
        offset += 4;
      }
      const batchTopCounts = new Map<string, number>();
      for (const {limit, direction} of batchTopQueries) {
        let count = 0;
        for (let batch = 0; batch < partitions.length; batch++) {
          count += view.getUint32(offset, true);
          offset += 4;
        }
        batchTopCounts.set(`${direction}:${limit}`, count);
      }
      const typeMatches = view.getUint32(offset, true);
      const typeMisses = view.getUint32(offset + 4, true);
      const typeProbes = view.getUint32(offset + 8, true);
      const maximumTypeProbes = view.getUint32(offset + 12, true);
      offset += 16;
      const otherTypeRows = view.getUint32(offset, true);
      offset += 4;
      const joinOverflow = view.getUint32(offset, true) !== 0;
      ctx.setReadout('filteredRows', formatCount(eventTotal));
      ctx.setReadout('meanMagnitude', eventTotal ? meanMagnitude.toFixed(2) : '—');
      ctx.setReadout('maximumMagnitude', eventTotal ? maximumMagnitude.toFixed(1) : '—');
      ctx.setReadout('minimumDepth', eventTotal ? `${minimumDepth.toFixed(1)} km` : '—');
      const selectedRank = `${ctx.options.rankDirection}:${ctx.options.rankCount}`;
      activeRankCount = globalTopCounts.get(selectedRank) ?? 0;
      ctx.requestLayers();
      ctx.setReadout(
        'topKRows',
        `${formatCount(globalTopCounts.get(selectedRank) ?? 0)} global rows`
      );
      ctx.setReadout(
        'batchTopKRows',
        `${formatCount(batchTopCounts.get(selectedRank) ?? 0)} rows across ${partitions.length} year batches`
      );
      ctx.setReadout(
        'typeJoin',
        `${formatCount(typeMatches)} found, ${formatCount(typeMisses)} missing; ${formatCount(typeProbes)} archive-wide probes, max ${maximumTypeProbes}`
      );
      ctx.setReadout(
        'otherTypeRows',
        `${formatCount(otherTypeRows)} archive-wide non-earthquake events`
      );
      ctx.setReadout(
        'joinCapacity',
        joinOverflow ? 'join publication overflow' : 'no join overflow'
      );
      ctx.setChart('magnitudeHistogram', {
        kind: 'histogram',
        title: 'Filtered event magnitudes',
        xLabel: 'Magnitude',
        yLabel: 'Events',
        values: histogram,
        xDomain: [4, 9],
        description:
          'GPU histogram of rows that satisfy the current magnitude, year and event-type filters. Values outside the fixed 4–9 display domain do not appear in bins.'
      });
      if (eventTotal === 0) ctx.setReadout('filteredRows', '0 (no rows match this query)');
      ctx.requestLayers();
    }
  );

  let latestQuerySignature = '';
  let readbackSignature = '';
  let activeRankCount = 0;
  ctx.setReadout('catalogRows', `${formatCount(rowCount)} archived rows`);
  ctx.setReadout('catalogTypes', `${eventTypeNames.length} event-type categories`);
  let encodedOnce = false;
  let queryDirty = true;

  return {
    // GPUDataFrame.compile owns its compiled graph privately; expose no placeholder graph here.
    getCompiledGraphs: () => [],
    encode(commandEncoder) {
      const currentParameters = queryParameters();
      const signature = JSON.stringify({
        ...currentParameters,
        rankCount: ctx.options.rankCount,
        rankDirection: ctx.options.rankDirection
      });
      latestQuerySignature = signature;
      if (!encodedOnce || queryDirty) {
        summaryQuery.encode(commandEncoder, currentParameters);
        histogramQuery.encode(commandEncoder, currentParameters);
        for (const {query} of globalTopQueries) query.encode(commandEncoder, currentParameters);
        for (const {query} of batchTopQueries) query.encode(commandEncoder, currentParameters);
        typeLookupQuery.encode(commandEncoder, {});
        otherTypeQuery.encode(commandEncoder, {});
        encodedOnce = true;
        queryDirty = false;
        if (!reader.isPending) readbackSignature = signature;
        reader.request(commandEncoder);
      } else {
        if (!reader.isPending) readbackSignature = signature;
        reader.flush(commandEncoder);
      }
    },
    getLayers() {
      const layers: Layer[] = [];
      if (ctx.options.showContext) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'earthquake-query-context',
            positions: positionBuffer,
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            instanceCount: rowCount,
            values: magnitudeBuffer,
            valueFormat: 'float32',
            colormap: 'inferno',
            valueRange: [4, 8],
            radiusPixels: 1.6,
            opacity: 0.12
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'earthquake-query-filtered-output',
          positions: positionBuffer,
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          instanceCount: rowCount,
          values: physicalBuffer(summaryQuery.selectionMask.data[0]),
          valueFormat: 'uint32',
          colormap: 'mask',
          color: [255, 178, 78, 240],
          noDataColor: [0, 0, 0, 0],
          radiusPixels: 2.1,
          opacity: 0.92
        })
      );
      const selectedRankQuery = globalTopQueries.find(
        ({limit, direction}) =>
          `${direction}:${limit}` === `${ctx.options.rankDirection}:${ctx.options.rankCount}`
      )?.query;
      if (selectedRankQuery && activeRankCount > 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'earthquake-query-ranked-output',
            positions: positionBuffer,
            ids: physicalBuffer(selectedRankQuery.globalRowIndices.data[0]),
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            instanceCount: activeRankCount,
            shape: 'star',
            color: [255, 255, 255, 255],
            outlineColor: [42, 30, 20, 255],
            outlineWidthPixels: 1,
            radiusPixels: 6.5,
            opacity: 1
          })
        );
      }
      return layers;
    },
    setOption() {
      // Reject a pending summary immediately, before the next frame encodes new parameters.
      latestQuerySignature = JSON.stringify({
        ...queryParameters(),
        rankCount: ctx.options.rankCount,
        rankDirection: ctx.options.rankDirection
      });
      activeRankCount = 0;
      queryDirty = true;
      reader.markStale();
      ctx.requestLayers();
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    destroy() {
      reader.stop();
      summaryQuery.destroy();
      histogramQuery.destroy();
      for (const {query} of globalTopQueries) query.destroy();
      for (const {query} of batchTopQueries) query.destroy();
      typeLookupQuery.destroy();
      otherTypeQuery.destroy();
      selectedFrame.destroy();
      frame.destroy();
      yearFrame.destroy();
      resources.destroy();
    }
  };
}
