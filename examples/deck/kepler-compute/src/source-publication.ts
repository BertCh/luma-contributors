// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getTimeIntervalDeclaration} from './time-contract-settings';
import * as arrow from 'apache-arrow';
import type {CommandEncoder, Fence} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerDatasetSnapshot,
  KeplerMemoryReservation
} from '@luma-contributors/kepler';
import {
  prepareKeplerGeometrySource,
  prepareKeplerGeometrySourceFromArrow,
  makeKeplerGeometryGPUVectors,
  prepareKeplerGeoJSONTripsFromArrow,
  makeKeplerGeoJSONTripGPUVectors,
  makeKeplerTabularTripGPUVectors,
  prepareKeplerTabularTripsFromArrow,
  isKeplerTabularTrips,
  type KeplerGeoJSONTripGPUVectors,
  type KeplerGlobeGeometryOptions,
  type PreparedKeplerGeometryGPUVectors,
  type PreparedKeplerGeoJSONTrips
} from './geometry-preparation';
import {type PrepareArrowTableForGPUAnalyticsOptions} from '@luma-contributors/analysis/analysis-core';
import {getGeographicViewportBounds} from '@luma-contributors/analysis/geospatial-analysis';
import {GeometryRuntime} from './geometry-runtime';
import {GlobeGeometryVariants, prepareGlobeGeometryIncrementally} from './globe-geometry-variants';
import {PlanarGeometryHibernation} from './planar-geometry-hibernation';
import {TripRuntime} from './trip-runtime';
import {FlowRuntime} from './flow-runtime';
import type {TripPreparationOptions} from './trip-source-options';
import {ResidentChartProvider} from './resident-chart-provider';
import {AggregationRuntime} from './aggregation-runtime';
import {prepareH3SourceFromArrow} from './h3-source';
import {makeH3GPUVectors, type H3GPUVectors} from './h3-runtime';
import {getNativeCellIndex, type NativeCellIndex} from './cell-layer-types';
import {AnalysisRuntime} from './analysis-runtime';
import type {PolygonAnalysisSource} from './analysis-polygon-weights-cache';
import {getPolygonAnalysisSource} from './polygon-publication-source';
import {PublishedDatasetRuntime} from './published-dataset-runtime';
import {getPublishedTableSortMemoryBudgetBytes} from './resident-table-sort-keys';
import {TimeRuntime} from './time-runtime';
import {assertNotResidentSource, selectRangeDomain} from './resident-source-guards';
import {ResidentScopeBinding} from './resident-scope-binding';
import type {KeplerComputeTableBinding, KeplerComputeTableState} from './kepler-table-bridge';
import {markResidentArrowInput} from './resident-input-validator';
import type {ResidentDatasetBinding} from './resident-layers';
import {
  parseResidentGroupByProvenance,
  type ResidentGroupByProvenance
} from './resident-group-by-plan';
import {parseResidentJoinLineage, type ResidentJoinLineage} from './resident-join-lineage';
import {
  parseResidentSuitabilityLineage,
  type ResidentSuitabilityLineage
} from './resident-suitability-lineage';

import {
  estimateTabularPointPositionBytes,
  prepareTabularPointPositions,
  makeGPUVectorsFromTabularPointPositions,
  type TabularPointGPUVectors,
  type CoordinateRoles
} from './point-position-source';
const TIME_FILTER_STEP_MILLISECONDS = 1000;

export type {CoordinateRoles} from './point-position-source';
export type PublishedSource = {
  id: string;
  label: string;
  source: arrow.Table;
  normalization: PrepareArrowTableForGPUAnalyticsOptions;
  coordinates?: CoordinateRoles;
  isCurrent(): boolean;
};

/** A resident consumer with explicit field roles; source Arrow and its batches stay unchanged. */
export class PublishedSourceRuntime {
  readonly consumer: PublishedDatasetRuntime;
  readonly binding: KeplerComputeTableBinding;
  readonly scopes: ResidentScopeBinding;
  readonly nativeData;
  readonly groupBy?: ResidentGroupByProvenance;
  readonly join?: ResidentJoinLineage;
  readonly suitability?: ResidentSuitabilityLineage;
  readonly positions?: GPUVector<'float32x2'>;
  readonly bounds?: [number, number, number, number];
  private readonly times: TimeRuntime[] = [];
  readonly pointPositions?: TabularPointGPUVectors;
  /** Arc and line endpoints: flow target coordinates uploaded like the source positions. */
  readonly targetPointPositions?: TabularPointGPUVectors;
  private geometryReservation?: KeplerMemoryReservation;
  private geometryMappingReservation?: KeplerMemoryReservation;
  private retainedPolygonSource?: PolygonAnalysisSource;
  private polygonUnavailableReason?: string;
  private globeVariants?: GlobeGeometryVariants;
  /** Planar geometry runtime; undefined while released for a globe view (see `planarSummary`). */
  geometry?: GeometryRuntime;
  /** Lightweight metadata kept while the planar copy is released, so dependents keep working. */
  private planarSummary?: {
    invalidRowCount: number;
    byteLength: number;
    budget: number;
    featureTypes: {line: boolean; polygon: boolean; point: boolean};
    fixedRadius?: boolean;
    representativeUnsupportedReason?: string;
  };
  private planarHibernation?: PlanarGeometryHibernation;
  readonly trip?: TripRuntime;
  flow?: FlowRuntime;
  readonly tripGeometry?: KeplerGeoJSONTripGPUVectors;
  readonly h3?: H3GPUVectors;
  private tripReservation?: KeplerMemoryReservation;
  private tripSourceReservation?: KeplerMemoryReservation;
  analysis?: AnalysisRuntime;
  private chartProvider?: ResidentChartProvider;
  private aggregationRuntime?: AggregationRuntime;
  private aggregationSourceReservation?: KeplerMemoryReservation;
  private valid = true;
  private destroyed = false;
  private destruction?: Promise<void>;
  private notice: string | null = null;

  constructor(
    readonly dataset: KeplerDatasetSnapshot<KeplerArrowDataset>,
    readonly label: string,
    readonly coordinates: CoordinateRoles | undefined,
    private readonly session: KeplerComputeSession,
    private readonly requestUpdate: () => void,
    reportPlot: (filterId: string, reason: string | null) => void,
    private readonly changed: () => void,
    readonly geometryColumn?: string,
    readonly tripOptions?: TripPreparationOptions,
    readonly h3Column?: string,
    groupBy?: ResidentGroupByProvenance,
    join?: ResidentJoinLineage,
    suitability?: ResidentSuitabilityLineage
  ) {
    if (groupBy) {
      this.groupBy = parseResidentGroupByProvenance(groupBy);
      if (
        this.groupBy.resultId !== dataset.id ||
        coordinates ||
        geometryColumn ||
        tripOptions ||
        h3Column ||
        this.groupBy.fields.length !== dataset.value.source.schema.fields.length ||
        this.groupBy.fields.some(
          (field, index) => field.name !== dataset.value.source.schema.fields[index]?.name
        )
      )
        throw new Error('Group By publication requires its matching table-only result schema');
    }
    if (join) {
      this.join = parseResidentJoinLineage(join);
      if (
        groupBy ||
        this.join.resultId !== dataset.id ||
        coordinates ||
        geometryColumn ||
        tripOptions ||
        h3Column ||
        this.join.fields.length !== dataset.value.source.schema.fields.length ||
        this.join.fields.some(
          (field, index) => field.name !== dataset.value.source.schema.fields[index]?.name
        )
      )
        throw new Error('Join publication requires its matching table-only result schema');
    }
    if (suitability) {
      this.suitability = parseResidentSuitabilityLineage(suitability);
      if (
        groupBy ||
        join ||
        this.suitability.resultId !== dataset.id ||
        coordinates ||
        geometryColumn ||
        tripOptions ||
        h3Column ||
        this.suitability.fields.length !== dataset.value.source.schema.fields.length ||
        this.suitability.fields.some(
          (field, index) => field.name !== dataset.value.source.schema.fields[index]?.name
        )
      )
        throw new Error('Suitability publication requires its matching table-only result schema');
    }
    if (h3Column && (coordinates || geometryColumn || tripOptions))
      throw new Error('H3 cells require an exclusive cell ID column role');
    if (tripOptions && (!geometryColumn || coordinates))
      throw new Error('Trips require a timed geometry column and no point coordinate roles');
    if (geometryColumn || h3Column || tripOptions || coordinates)
      // Geometry, H3, trip and coordinate roles scan CPU source cells that resident results lack.
      assertNotResidentSource(dataset.value, 'publication with geometry or coordinate roles');
    let consumer: PublishedDatasetRuntime | undefined;
    let scopes: ResidentScopeBinding | undefined;
    const prepared = dataset.value.prepared;
    const fields: Record<string, KeplerComputeTableBinding['fields'][string]> = {};
    const derivedFields = this.groupBy?.fields ?? this.join?.fields ?? this.suitability?.fields;
    const names = derivedFields
      ? derivedFields.map(field => field.name)
      : [...new Set([...Object.keys(dataset.value.columns), ...Object.keys(prepared.timeColumns)])];
    // Resident SQL results keep time columns as sint32 epoch seconds (`luma.gl.time.format`) beside
    // their exact time words, so their native field shows seconds ('X') rather than milliseconds.
    const getTimeFormat = (name: string): string =>
      dataset.value.resident && dataset.value.columns[name]
        ? (dataset.value.source.schema.fields
            .find(field => field.name === name)
            ?.metadata.get('luma.gl.time.format') ?? 'x')
        : 'x';
    if (geometryColumn && !names.includes(geometryColumn)) names.push(geometryColumn);
    if (h3Column && !names.includes(h3Column)) names.push(h3Column);
    // The DGGS token column is source-only; saved cell layers still name it as their `token`.
    const cellColumn = getNativeCellIndex(dataset.source?.metadata)?.column;
    if (cellColumn && !names.includes(cellColumn)) names.push(cellColumn);
    for (const name of Object.values(tripOptions?.grouping ?? {}))
      if (!names.includes(name)) names.push(name);
    // For resident datasets this is the lazy table: cells read null until ensureRows() pages them,
    // so the native data table only receives schema, row count and batch layout, not a CPU scan.
    const source = dataset.value.source.select(names);
    this.nativeData = markResidentArrowInput({
      fields: derivedFields
        ? derivedFields.map(field => ({...field, format: ''}))
        : names.map(name => ({
            name,
            format: prepared.timeColumns[name] ? getTimeFormat(name) : '',
            type:
              name === h3Column
                ? 'h3'
                : name === geometryColumn
                  ? 'geojson'
                  : name === cellColumn
                    ? 'string'
                    : prepared.timeColumns[name]
                      ? 'timestamp'
                      : prepared.dictionaries[name]
                        ? 'string'
                        : // Trip role columns excluded from upload are retained source-only columns.
                          dataset.value.columns[name]?.format === 'float32' ||
                            (!dataset.value.columns[name] &&
                              arrow.DataType.isFloat(source.getChild(name)!.type))
                          ? 'real'
                          : 'integer'
          })),
      rows: [],
      cols: names.map(name => source.getChild(name)!),
      arrowSchema: source.schema,
      arrowTable: source
    });
    try {
      for (const name of Object.keys(dataset.value.columns)) {
        const dictionary = prepared.dictionaries[name];
        if (dictionary) {
          fields[name] = {
            dimension: name,
            kind: 'category',
            domain: dictionary.values,
            categories: dictionary.values
          };
        } else {
          const domain = selectRangeDomain(dataset.value, name, () => {
            const vector = prepared.table.getChild(name)!;
            let minimum = Infinity;
            let maximum = -Infinity;
            for (let index = 0; index < vector.length; index++) {
              const value: unknown = vector.get(index);
              if (typeof value === 'number' && Number.isFinite(value)) {
                minimum = Math.min(minimum, value);
                maximum = Math.max(maximum, value);
              }
            }
            return minimum <= maximum ? [minimum, maximum] : [0, 0];
          });
          fields[name] = {
            dimension: name,
            kind: 'range',
            domain,
            step: dataset.value.columns[name].format === 'float32' ? 0.01 : 1
          };
        }
      }
      for (const name of Object.keys(prepared.timeColumns)) {
        const interval = getTimeIntervalDeclaration(dataset.id, name);
        const time = new TimeRuntime({
          session,
          datasetId: dataset.id,
          column: name,
          dimension: name,
          interval
        });
        this.times.push(time);
        fields[name] = {
          dimension: name,
          kind: 'time',
          domain: time.domainMilliseconds ?? [0, 0],
          // Native time filters step by one second. Without a step Kepler's brush rounds to a zero
          // step, divides by zero and reports [NaN, NaN].
          step: TIME_FILTER_STEP_MILLISECONDS,
          validateRange: time.validateRange,
          ...(interval ? {intervalEndColumn: interval.endColumn} : {})
        };
      }
      if (coordinates) {
        const estimate = estimateTabularPointPositionBytes(
          prepared.sourceBatches.map(batch => batch.sourceRowCount),
          coordinates.altitude !== undefined
        );
        const preparationReservation = session.registry.reserveMemory(
          estimate.cpuByteLength,
          'cpu',
          'scratch'
        );
        try {
          const pointSource = prepareTabularPointPositions(prepared, coordinates);
          this.invalidCoordinateRowCount = pointSource.invalidRowCount;
          this.pointPositions = makeGPUVectorsFromTabularPointPositions(
            session.device,
            pointSource,
            {
              memoryRegistry: session.registry
            }
          );
          this.positions = this.pointPositions.positions;
          this.bounds = this.pointPositions.bounds;
          const target = coordinates.flow
            ? {
                longitude: coordinates.flow.targetLongitude,
                latitude: coordinates.flow.targetLatitude
              }
            : coordinates.target;
          if (target) {
            // The second CPU preparation pass (target coordinates) is reserved on top of the
            // source pass, whose typed storage stays alive until this block ends.
            preparationReservation.resize(
              preparationReservation.byteLength +
                estimateTabularPointPositionBytes(
                  prepared.sourceBatches.map(batch => batch.sourceRowCount),
                  'altitude' in target && target.altitude !== undefined
                ).cpuByteLength
            );
            this.targetPointPositions = makeGPUVectorsFromTabularPointPositions(
              session.device,
              prepareTabularPointPositions(prepared, target),
              {memoryRegistry: session.registry}
            );
          }
        } finally {
          preparationReservation.release();
        }
      }
      scopes = new ResidentScopeBinding({
        residentStyleDomains: true,
        session,
        datasetId: dataset.id,
        fields,
        layers: this.positions
          ? [
              {
                id: `${dataset.id}-points`,
                geometryRevision: dataset.revision,
                geometry: {kind: 'point', positions: this.positions}
              }
            ]
          : [],
        onUpdate: () => {
          consumer?.invalidate();
          this.valid = true;
          this.notice = null;
          changed();
          requestUpdate();
        },
        onUnsupported: reason => {
          this.valid = false;
          this.notice = reason;
          changed();
          requestUpdate();
        },
        onPlotUnsupported: reportPlot
      });
      this.scopes = scopes;
      this.consumer = new PublishedDatasetRuntime({
        session,
        datasetId: dataset.id,
        explorer: {
          dimensions: [scopes.externalDimension],
          workingMemoryBudgetBytes: 8 * 1024 * 1024 + dataset.rowCount * 128,
          maximumPageSize: 256,
          maximumReadbackBytes: 65536
        },
        sortMemoryBudgetBytes: getPublishedTableSortMemoryBudgetBytes(dataset.rowCount),
        requestUpdate,
        getDerivedColumns: () => this.analysis?.getTableColumns(),
        destroyDerived: async completion => {
          await this.analysis?.destroy(completion);
        }
      });
      consumer = this.consumer;
      // A GeoJSON trip source can also back ordinary GeoJSON layers (world_flights saves
      // both animated Flights and static Paths). Admit both representations through their
      // existing budgets. A grouped tabular trip ID column is not a geometry column.
      if (geometryColumn && !tripOptions?.grouping) {
        const geometry = preparePublishedGeometry(dataset.value.source, geometryColumn, {
          timedLineStrings: Boolean(tripOptions)
        });
        const polygonAnalysis = getPolygonAnalysisSource(geometry, geometryColumn);
        this.polygonUnavailableReason = polygonAnalysis.reason;
        const sourceBytes = geometry.batches.reduce(
          (bytes, batch) =>
            bytes +
            batch.points.positions.byteLength +
            batch.paths.positions.byteLength +
            batch.polygonOutlines.positions.byteLength +
            (batch.polygonOutlines.sideOrientations?.byteLength ?? 0) +
            (batch.representative?.positions.byteLength ?? 0) +
            (batch.representative?.validity.byteLength ?? 0) +
            Object.values(batch.properties ?? {}).reduce(
              (propertyBytes, property) =>
                propertyBytes + property.values.byteLength + property.validity.byteLength,
              0
            ) +
            batch.polygons.data.reduce((sum, data) => sum + getArrowDataBytes(data), 0),
          0
        );
        const budget = 8 * 1024 * 1024 + sourceBytes * 64 + dataset.rowCount * 128;
        this.geometryReservation = session.registry.reserveMemory(budget, 'gpu', 'source');
        const mappingBudget =
          sourceBytes * 4 + dataset.rowCount * 64 + 4096 + polygonAnalysis.byteLength;
        this.geometryMappingReservation = session.registry.reserveMemory(
          mappingBudget,
          'cpu',
          'source'
        );
        this.retainedPolygonSource = polygonAnalysis.source;
        const vectors = makeKeplerGeometryGPUVectors(session.device, geometry);
        try {
          this.geometry = new GeometryRuntime({
            device: session.device,
            column: geometryColumn,
            geometry: vectors,
            selectionMask: this.consumer.explorer.selectionMask,
            ownsGeometry: true
          });
          this.geometry.prepare();
          if (
            vectors.gpuByteLength + this.geometry.accountedResidentByteLength > budget ||
            vectors.mappingByteLength + polygonAnalysis.byteLength > mappingBudget
          )
            throw new Error('Prepared geometry exceeds the admitted geometry budget');
          if (geometry.metadata.bounds) this.bounds = [...geometry.metadata.bounds];
          this.planarSummary = {
            invalidRowCount: vectors.metadata.invalidRows.length,
            byteLength: vectors.gpuByteLength + vectors.mappingByteLength,
            budget,
            featureTypes: {
              line: Boolean(vectors.metadata.featureTypes.line),
              polygon: Boolean(vectors.metadata.featureTypes.polygon),
              point: Boolean(vectors.metadata.featureTypes.point)
            },
            fixedRadius: vectors.metadata.fixedRadius,
            representativeUnsupportedReason: vectors.representativeUnsupportedReason
          };
        } catch (error) {
          this.geometry?.destroy();
          vectors.destroy();
          throw error;
        }
      }
      if (geometryColumn && tripOptions) {
        const preparedTrips = preparePublishedTrips(
          dataset.value.source,
          geometryColumn,
          tripOptions
        );
        const sourceBytes =
          preparedTrips.typedStorageByteLength +
          preparedTrips.batches.reduce(
            (bytes, batch) =>
              bytes +
              batch.sourceTimestamps.reduce<number>(
                (timestampBytes, timestamp) =>
                  timestampBytes + 8 + (typeof timestamp === 'string' ? timestamp.length * 2 : 0),
                0
              ),
            0
          );
        this.tripSourceReservation = session.registry.reserveMemory(sourceBytes, 'cpu', 'source');
        const gpuBytes = preparedTrips.batches.reduce(
          (bytes, batch, index) =>
            bytes +
            Math.max(4, batch.positions.byteLength) +
            Math.max(4, batch.timestamps.byteLength) +
            Math.max(4, batch.sourceRows.byteLength) +
            (isKeplerTabularTrips(preparedTrips)
              ? Math.max(4, preparedTrips.batches[index].memberOffsets.byteLength) +
                Math.max(4, preparedTrips.batches[index].memberRows.byteLength)
              : 0) +
            Object.values(batch.properties ?? {}).reduce(
              (bytes, property) =>
                bytes +
                Math.max(4, property.values.byteLength) +
                Math.max(4, property.validity.byteLength),
              0
            ),
          0
        );
        this.tripReservation = session.registry.reserveMemory(gpuBytes, 'gpu', 'source');
        this.tripGeometry = isKeplerTabularTrips(preparedTrips)
          ? makeKeplerTabularTripGPUVectors(session.device, preparedTrips)
          : makeKeplerGeoJSONTripGPUVectors(session.device, preparedTrips);
        if (this.tripGeometry.gpuByteLength > gpuBytes)
          throw new Error('Prepared Trips exceed the admitted source budget');
        this.trip = new TripRuntime({
          device: session.device,
          column: geometryColumn,
          trips: this.tripGeometry,
          grouping: tripOptions?.grouping,
          selectionMask: this.consumer.explorer.selectionMask,
          memoryRegistry: session.registry,
          workingMemoryBudgetBytes:
            4 * 1024 * 1024 + preparedTrips.tripCount * 64 + preparedTrips.vertexCount * 4
        });
        this.trip.prepare();
        // Share all-part fitting bounds with native metadata and zoom-to-layer. A tight Float32
        // envelope can put an original boundary coordinate just outside the fitted viewport.
        this.bounds =
          getGeographicViewportBounds(preparedTrips.batches.map(batch => batch.positions)) ??
          undefined;
      }
      if (h3Column) {
        const preparedCells = preparePublishedH3(
          dataset.value.source,
          h3Column,
          dataset.batches.map(batch => batch.id)
        );
        this.h3 = makeH3GPUVectors(session.device, preparedCells, {
          memoryRegistry: session.registry,
          workingMemoryBudgetBytes:
            4 * 1024 * 1024 + preparedCells.typedStorageByteLength * 64 + dataset.rowCount * 128
        });
        const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
        for (const batch of preparedCells.batches)
          for (let row = 0; row < batch.rowCount; row++) {
            if (!batch.validity[row]) continue;
            const longitude = batch.centers[row * 2];
            const latitude = batch.centers[row * 2 + 1];
            bounds[0] = Math.min(bounds[0], longitude);
            bounds[1] = Math.min(bounds[1], latitude);
            bounds[2] = Math.max(bounds[2], longitude);
            bounds[3] = Math.max(bounds[3], latitude);
          }
        if (bounds.every(Number.isFinite)) this.bounds = bounds;
      }
      this.binding = scopes.bind(this.consumer.explorer);
    } catch (error) {
      if (consumer) void consumer.destroy();
      if (scopes) void scopes.destroy(Promise.resolve());
      if (this.trip) void this.trip.destroy(Promise.resolve());
      if (this.h3) void this.h3.destroy(Promise.resolve());
      this.flow?.destroy();
      this.tripGeometry?.destroy();
      this.tripReservation?.release();
      this.tripSourceReservation?.release();
      this.geometry?.destroy();
      this.geometryReservation?.release();
      this.geometryMappingReservation?.release();
      this.retainedPolygonSource = undefined;
      if (this.pointPositions) void this.pointPositions.destroy(Promise.resolve());
      if (this.targetPointPositions) void this.targetPointPositions.destroy(Promise.resolve());
      for (const time of this.times) void time.destroy(Promise.resolve());
      throw error;
    }
  }

  /** Rows with null or out-of-range coordinates: retained in the table, hidden on the map. */
  invalidCoordinateRowCount = 0;

  get error(): string | null {
    const invalidRows =
      this.planarSummary?.invalidRowCount ??
      this.tripGeometry?.source.invalidRows.length ??
      this.h3?.source.invalidRows.length ??
      0;
    return (
      this.notice ??
      (invalidRows
        ? `${invalidRows} source rows have invalid geometry; their table rows are retained and their geometry is omitted.`
        : null)
    );
  }
  get polygonSource(): PolygonAnalysisSource | undefined {
    return this.destroyed ? undefined : this.retainedPolygonSource;
  }
  get polygonAnalysisUnavailableReason(): string | undefined {
    return this.retainedPolygonSource
      ? undefined
      : (this.polygonUnavailableReason ??
          'Polygon analysis requires an explicitly prepared polygon source.');
  }

  getAnalysis(): AnalysisRuntime {
    if (this.destroyed || !this.consumer.explorer.isCurrent)
      throw new Error('Published source is no longer current');
    if (!this.analysis) {
      this.analysis = new AnalysisRuntime({
        session: this.session,
        explorer: this.consumer.explorer,
        requestUpdate: this.requestUpdate,
        polygonSource: this.retainedPolygonSource
      });
      this.analysis.subscribe(() => {
        this.consumer.invalidateDerived();
        this.changed();
        this.requestUpdate();
      });
    }
    return this.analysis;
  }
  getInterpolationLayer() {
    return this.analysis?.getInterpolationLayer() ?? null;
  }

  getAnalysisResultLayers() {
    return this.analysis?.getResultLayers() ?? [];
  }

  getCharts(): ResidentChartProvider | undefined {
    if (this.destroyed || !this.consumer.explorer.isCurrent) return undefined;
    if (!this.chartProvider)
      this.chartProvider = new ResidentChartProvider({
        session: this.session,
        explorer: this.consumer.explorer,
        selection: this.scopes,
        subscribe: listener => this.consumer.subscribe(listener)
      });
    return this.chartProvider;
  }

  get aggregation(): AggregationRuntime | undefined {
    return this.aggregationRuntime;
  }

  /** Explicit one-time source projection; subsequent controls only encode resident masks. */
  getAggregation(): AggregationRuntime {
    if (this.destroyed || !this.consumer.explorer.isCurrent)
      throw new Error('The published aggregation source is stale or closed.');
    if (!this.coordinates || !this.positions)
      throw new Error(
        'Resident Grid and Hexagon require published longitude/latitude point roles.'
      );
    if (this.aggregationRuntime) return this.aggregationRuntime;
    const reservation = this.session.registry.reserveMemory(
      this.dataset.rowCount * 16,
      'cpu',
      'scratch'
    );
    try {
      const longitudeSource = this.dataset.value.prepared.table.getChild(
        this.coordinates.longitude
      );
      const latitudeSource = this.dataset.value.prepared.table.getChild(this.coordinates.latitude);
      if (!longitudeSource || !latitudeSource)
        throw new Error('Published coordinate columns are unavailable.');
      const longitude = new Float64Array(this.dataset.rowCount);
      const latitude = new Float64Array(this.dataset.rowCount);
      for (let row = 0; row < this.dataset.rowCount; row++) {
        const longitudeValue: unknown = longitudeSource.get(row);
        const latitudeValue: unknown = latitudeSource.get(row);
        longitude[row] = typeof longitudeValue === 'number' ? longitudeValue : NaN;
        latitude[row] = typeof latitudeValue === 'number' ? latitudeValue : NaN;
      }
      this.aggregationRuntime = new AggregationRuntime({
        device: this.session.device,
        longitude,
        latitude,
        sourceChunkLengths: this.dataset.batches.map(batch => batch.rowCount),
        coordinateColumns: {lat: this.coordinates.latitude, lng: this.coordinates.longitude},
        selectionMask: this.consumer.explorer.selectionMask,
        columns: Object.fromEntries(
          Object.entries(this.dataset.value.columns).map(([name, values]) => [
            name,
            {
              values,
              validity: this.dataset.value.analytics.validity[name]
            }
          ])
        ),
        memoryRegistry: this.session.registry,
        requestUpdate: this.requestUpdate,
        getCategoryLabels: column => {
          // Globally reconciled dictionary (all batches); codes in the GPU column index it.
          const reconciled = this.dataset.value.prepared.dictionaries[column]?.values;
          if (reconciled) return Array.from(reconciled, label => String(label));
          const dictionary =
            this.dataset.value.prepared.table.getChild(column)?.data[0]?.dictionary;
          return dictionary ? Array.from(dictionary, label => String(label)) : undefined;
        }
      });
      this.aggregationSourceReservation = reservation;
      return this.aggregationRuntime;
    } catch (error) {
      reservation.release();
      throw error;
    }
  }
  /** Recorded DGGS family and native `token` column; footprints are the geometry column. */
  get cellIndex(): NativeCellIndex | undefined {
    return getNativeCellIndex(this.dataset.source?.metadata);
  }

  /**
   * GlobeView variant of the prepared geometry: longitudes unwrapped across the antimeridian,
   * paths and outlines subdivided and polygons cut on a grid of 5 degrees (finer at higher zoom
   * bands), as Deck does for a globe. It is derived from the retained source (never read back from
   * the GPU) in time-sliced steps, so this returns undefined (planar geometry draws) until the
   * variant is ready and requests an update when it is. One variant is resident at a time, it is
   * accounted in the memory registry and released by `releaseGlobeGeometry` or with the publication.
   * A source that is not geometry, or a band over budget, keeps the planar geometry.
   */
  getGlobeGeometry(zoom?: number): PreparedKeplerGeometryGPUVectors | undefined {
    const column = this.geometryColumn;
    const summary = this.planarSummary;
    if (!column || !summary || this.destroyed) return undefined;
    this.globeVariants ??= new GlobeGeometryVariants({
      createPreparation: resolutionDegrees =>
        prepareGlobeGeometryIncrementally(
          this.dataset.value.source,
          column,
          resolutionDegrees,
          (source, name, options) =>
            preparePublishedGeometry(source, name, {
              ...options,
              timedLineStrings: Boolean(this.tripOptions)
            })
        ),
      makeVectors: prepared => makeKeplerGeometryGPUVectors(this.session.device, prepared),
      reserve: byteLength => this.session.registry.reserveMemory(byteLength, 'gpu', 'source'),
      getPlanarByteLength: () => summary.byteLength,
      requestUpdate: () => this.requestUpdate()
    });
    const variant = this.globeVariants.getVariant(zoom);
    // The planar copy is released (after a grace period) once the globe variant can stand in for it.
    this.getPlanarHibernation()?.noteGlobeFrame(Boolean(variant));
    return variant;
  }

  /**
   * Flat view: returns undefined when the planar geometry can draw. While the released planar copy
   * is rebuilt (time-sliced) the resident globe variant keeps drawing; it is freed once planar is back.
   */
  releaseGlobeGeometry(): PreparedKeplerGeometryGPUVectors | undefined {
    const state = this.getPlanarHibernation()?.noteFlatFrame() ?? 'resident';
    if (state === 'resident') {
      this.globeVariants?.release();
      return undefined;
    }
    return this.globeVariants?.peekVariant();
  }

  /** Planar copy state, for diagnostics. */
  get planarGeometryStatus() {
    return (
      this.planarHibernation?.status ?? (this.geometry ? {state: 'resident' as const} : undefined)
    );
  }

  /** Representative points for polygon filters; while planar is released the globe variant's (antimeridian-unwrapped). */
  get representativeGeometry() {
    return (this.geometry?.geometry ?? this.globeVariants?.peekVariant())?.representative;
  }
  get representativeUnsupportedReason(): string | undefined {
    return this.geometry
      ? this.geometry.geometry.representativeUnsupportedReason
      : this.planarSummary?.representativeUnsupportedReason;
  }

  private getPlanarHibernation(): PlanarGeometryHibernation | undefined {
    const column = this.geometryColumn;
    const summary = this.planarSummary;
    if (!column || !summary || this.destroyed) return undefined;
    this.planarHibernation ??= new PlanarGeometryHibernation({
      releasePlanar: () => {
        this.geometry?.destroy();
        this.geometry = undefined;
        this.geometryReservation?.release();
        this.geometryReservation = undefined;
      },
      createPreparation: () =>
        prepareGlobeGeometryIncrementally(
          this.dataset.value.source,
          column,
          undefined,
          (source, name, options) =>
            preparePublishedGeometry(source, name, {
              ...options,
              timedLineStrings: Boolean(this.tripOptions)
            })
        ),
      installPlanar: prepared => {
        const reservation = this.session.registry.reserveMemory(summary.budget, 'gpu', 'scratch');
        let vectors: PreparedKeplerGeometryGPUVectors | undefined;
        try {
          vectors = makeKeplerGeometryGPUVectors(this.session.device, prepared as never);
          const runtime = new GeometryRuntime({
            device: this.session.device,
            column,
            geometry: vectors,
            selectionMask: this.consumer.explorer.selectionMask,
            ownsGeometry: true
          });
          runtime.prepare();
          if (vectors.gpuByteLength + runtime.accountedResidentByteLength > summary.budget) {
            runtime.destroy();
            throw new Error('Rebuilt planar geometry exceeds the admitted geometry budget');
          }
          this.geometry = runtime;
          this.geometryReservation = reservation;
        } catch (error) {
          vectors?.destroy();
          reservation.release();
          throw error;
        }
      },
      requestUpdate: () => this.requestUpdate()
    });
    return this.planarHibernation;
  }

  get globeGeometryStatus() {
    return this.globeVariants
      ? {...this.globeVariants.status, planar: this.planarGeometryStatus}
      : undefined;
  }

  get renderBinding(): ResidentDatasetBinding | undefined {
    if (this.h3)
      return {
        selectionMask: this.consumer.explorer.selectionMask,
        colorOverride: this.analysis?.active ? this.analysis.colors : undefined,
        enabled: this.valid && this.consumer.explorer.isCurrent && this.h3.isCurrent(),
        bounds: this.bounds
      };
    if (this.trip && this.geometryColumn)
      return {
        trip: this.trip.resolve(this.geometryColumn),
        selectionMask: this.consumer.explorer.selectionMask,
        colorOverride: this.analysis?.active ? this.analysis.colors : undefined,
        enabled: this.valid && this.consumer.explorer.isCurrent,
        bounds: this.bounds
      };
    if ((this.geometry || this.planarSummary) && this.geometryColumn)
      return {
        // Undefined while the planar copy is released: the style host expands the globe variant.
        geojson: this.geometry?.resolve(this.geometryColumn),
        geojsonFeatureTypes: this.geometry ? undefined : this.planarSummary?.featureTypes,
        ...(this.cellIndex ? {cellColumn: this.cellIndex.column} : {}),
        selectionMask: this.consumer.explorer.selectionMask,
        colorOverride: this.analysis?.active ? this.analysis.colors : undefined,
        enabled: this.valid && this.consumer.explorer.isCurrent,
        bounds: this.bounds
      };
    return this.positions && this.coordinates
      ? {
          point: {
            positions: this.positions,
            columns: {lng: this.coordinates.longitude, lat: this.coordinates.latitude}
          },
          ...(this.flow ? {flow: this.flow.binding} : {}),
          ...(this.targetPointPositions && (this.coordinates.flow || this.coordinates.target)
            ? {
                connections: {
                  // Native Line alt0/alt1 bind float32x3 endpoints; otherwise the 2D vectors.
                  sourcePositions:
                    (this.coordinates.altitude !== undefined
                      ? this.pointPositions?.positions3D
                      : undefined) ?? this.positions,
                  targetPositions:
                    (this.coordinates.target?.altitude !== undefined
                      ? this.targetPointPositions.positions3D
                      : undefined) ?? this.targetPointPositions.positions,
                  columns: {
                    ...(this.coordinates.altitude !== undefined && this.pointPositions?.positions3D
                      ? {alt0: this.coordinates.altitude}
                      : {}),
                    ...(this.coordinates.target?.altitude !== undefined &&
                    this.targetPointPositions.positions3D
                      ? {alt1: this.coordinates.target.altitude}
                      : {}),
                    lat0: this.coordinates.latitude,
                    lng0: this.coordinates.longitude,
                    lat1:
                      this.coordinates.flow?.targetLatitude ?? this.coordinates.target!.latitude,
                    lng1:
                      this.coordinates.flow?.targetLongitude ?? this.coordinates.target!.longitude
                  }
                }
              }
            : {}),
          selectionMask: this.consumer.explorer.selectionMask,
          colorOverride: this.analysis?.active ? this.analysis.colors : undefined,
          enabled: this.valid && this.consumer.explorer.isCurrent,
          bounds: this.bounds
        }
      : undefined;
  }
  /** Flow clustering uses the pinned flowmap library, which loads asynchronously. */
  async prepareFlow(): Promise<void> {
    if (!this.coordinates?.flow || this.flow) return;
    this.flow = await FlowRuntime.create(this.dataset.value.source, {
      coordinates: this.coordinates,
      roles: this.coordinates.flow,
      requestUpdate: this.requestUpdate,
      getRevision: () => this.scopes.currentRevision,
      memoryRegistry: this.session.registry,
      device: this.session.device
    });
  }
  encode(encoder: CommandEncoder): void {
    this.consumer.encode(encoder);
    this.aggregationRuntime?.encode(encoder);
    this.analysis?.encode(encoder);
  }
  async readSummary(bridge?: KeplerComputeTableState): Promise<void> {
    await this.scopes.readSummary();
    await this.consumer.readSummary();
    await this.analysis?.readSummary();
    if (bridge) this.scopes.publishHistograms(bridge);
  }
  destroy(): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.destruction = this.destroyResources();
    return this.destruction;
  }
  private async destroyResources(): Promise<void> {
    const chartResults = await Promise.allSettled([
      this.chartProvider?.destroy(),
      this.aggregationRuntime?.destroy()
    ]);
    const results = [...chartResults, ...(await Promise.allSettled([this.consumer.destroy()]))];
    const errors: unknown[] = results.flatMap(result =>
      result.status === 'rejected' ? [result.reason] : []
    );
    let fence: Fence | undefined;
    try {
      if (!this.session.deviceLost) fence = this.session.device.createFence();
    } catch (error) {
      errors.push(error);
    }
    const completion = fence
      ? Promise.race([fence.signaled, this.session.device.lost])
      : Promise.resolve();
    try {
      const outcomes = await Promise.allSettled([
        this.scopes.destroy(completion),
        this.trip?.destroy(completion),
        Promise.resolve(this.flow?.destroy()),
        this.h3?.destroy(completion),
        this.pointPositions?.destroy(completion),
        this.targetPointPositions?.destroy(completion),
        ...this.times.map(time => time.destroy(completion)),
        completion
      ]);
      for (const outcome of outcomes)
        if (outcome.status === 'rejected') errors.push(outcome.reason);
    } finally {
      for (const cleanup of [
        () => fence?.destroy(),
        () => this.tripGeometry?.destroy(),
        () => this.tripReservation?.release(),
        () => this.tripSourceReservation?.release(),
        () => this.planarHibernation?.destroy(),
        () => this.geometry?.destroy(),
        () => this.globeVariants?.destroy(),
        () => this.geometryReservation?.release(),
        () => {
          this.retainedPolygonSource = undefined;
          this.geometryMappingReservation?.release();
        },
        () => this.aggregationSourceReservation?.release()
      ]) {
        try {
          cleanup();
        } catch (error) {
          errors.push(error);
        }
      }
    }
    if (errors.length) throw new AggregateError(errors, 'Published source cleanup failed');
  }
}

function getArrowDataBytes(data: arrow.Data): number {
  return (
    (data.values?.byteLength ?? 0) +
    (data.valueOffsets?.byteLength ?? 0) +
    (data.nullBitmap?.byteLength ?? 0) +
    (data.children ?? []).reduce((bytes, child) => bytes + getArrowDataBytes(child), 0)
  );
}

/** Field metadata belongs to the source schema, so pass it across the vector-only adapter. */
export function preparePublishedGeometry(
  source: arrow.Table,
  name: string,
  options: {globe?: KeplerGlobeGeometryOptions; timedLineStrings?: boolean} = {}
) {
  const field = source.schema.fields.find(candidate => candidate.name === name);
  const column = source.getChild(name);
  if (!field || !column) throw new Error('Geometry column is missing from the retained source');
  const metadataText = field.metadata.get('ARROW:extension:metadata');
  const metadata: unknown = metadataText ? JSON.parse(metadataText) : undefined;
  if (
    metadata !== undefined &&
    (!metadata || typeof metadata !== 'object' || Array.isArray(metadata))
  )
    throw new Error('Geometry extension metadata must be an object');
  const crs =
    metadata && typeof metadata === 'object' && 'crs' in metadata ? metadata.crs : undefined;
  const encoding = field.metadata.get('ARROW:extension:name');
  switch (encoding) {
    case 'geoarrow.wkb':
    case 'geoarrow.point':
    case 'geoarrow.multipoint':
    case 'geoarrow.linestring':
    case 'geoarrow.multilinestring':
    case 'geoarrow.polygon':
    case 'geoarrow.multipolygon':
    case 'geoarrow.geometry':
      return prepareKeplerGeometrySourceFromArrow(column, {
        encoding,
        crs,
        globe: options.globe,
        // Low parts are retained only when Float32 cannot represent the coordinates exactly.
        fp64: !options.globe
      });
    case undefined:
      return prepareKeplerGeometrySource(
        source.batches.map((batch, index) => ({
          id: `geometry-${index}`,
          rows: Array.from({length: batch.numRows}, (_, row): unknown =>
            batch.getChild(name)?.get(row)
          )
        })),
        {
          crs,
          globe: options.globe,
          fp64: !options.globe,
          timedLineStrings: options.timedLineStrings
        }
      );
    default:
      throw new Error(
        `Unsupported geometry encoding ${encoding}; convert it explicitly before publication`
      );
  }
}

/** Explicit timed-LineString source preparation, reusable by restore preflight. */
export function preparePublishedTrips(
  source: arrow.Table,
  name: string,
  options: TripPreparationOptions
): PreparedKeplerGeoJSONTrips {
  if (options.grouping) {
    const {grouping, ...preparationOptions} = options;
    const crs = preparationOptions.crs;
    if (crs !== undefined && crs !== null && crs !== 'EPSG:4326' && crs !== 'OGC:CRS84')
      throw new Error('Trips require longitude/latitude coordinates in EPSG:4326 or OGC:CRS84');
    const columns = {
      tripId: name,
      timestamp: grouping.timestampColumn,
      longitude: grouping.longitudeColumn,
      latitude: grouping.latitudeColumn,
      ...(grouping.altitudeColumn ? {altitude: grouping.altitudeColumn} : {}),
      ...(grouping.partColumn ? {partId: grouping.partColumn} : {})
    };
    // Columns are read by name from the retained source, so a missing role fails before upload.
    for (const column of Object.values(columns))
      if (!source.getChild(column))
        throw new Error('Trip column is missing from the retained source');
    const grouped = prepareKeplerTabularTripsFromArrow(source, columns, preparationOptions);
    if (!grouped.tripCount) throw new Error('The grouped rows contain no valid Trips');
    return grouped;
  }
  const column = source.getChild(name);
  const field = source.schema.fields.find(candidate => candidate.name === name);
  if (!column || !field)
    throw new Error('Timed geometry column is missing from the retained source');
  const metadataText = field.metadata.get('ARROW:extension:metadata');
  const metadata: unknown = metadataText ? JSON.parse(metadataText) : undefined;
  const crs =
    metadata && typeof metadata === 'object' && 'crs' in metadata ? metadata.crs : undefined;
  if (
    crs !== undefined &&
    options.crs !== undefined &&
    JSON.stringify(crs) !== JSON.stringify(options.crs)
  )
    throw new Error('Trip CRS settings must match the retained source metadata');
  const retainedCrs = crs ?? options.crs;
  if (
    retainedCrs !== undefined &&
    retainedCrs !== null &&
    retainedCrs !== 'EPSG:4326' &&
    retainedCrs !== 'OGC:CRS84'
  )
    throw new Error('Trips require longitude/latitude coordinates in EPSG:4326 or OGC:CRS84');
  const prepared = prepareKeplerGeoJSONTripsFromArrow(column, {...options, crs: retainedCrs});
  if (!prepared.tripCount) throw new Error('The timed geometry column contains no valid Trips');
  return prepared;
}

/** Validate lossless IDs against the complete retained source before any GPU publication. */
export function preparePublishedH3(
  source: arrow.Table,
  name: string,
  batchIds?: readonly string[]
) {
  const column = source.getChild(name);
  if (!column) throw new Error('H3 cell ID column is missing from the retained source');
  const prepared = prepareH3SourceFromArrow(column, {accessor: 'decimal-or-hex', batchIds});
  const precisionFailure = prepared.invalidRows.find(row => row.code === 'number-identity');
  if (precisionFailure) throw new Error(precisionFailure.reason);
  // Float32-collapsed cells are admitted: makeH3GPUVectors uploads hi/lo centers and hi/lo
  // vertex streams, so both the column and the polygon branch render them.
  if (!prepared.validRowCount) throw new Error('The H3 cell ID column contains no valid cells');
  return prepared;
}
