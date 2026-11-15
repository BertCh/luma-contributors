// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useMemo, useRef, useState} from 'react';
import * as arrow from 'apache-arrow';
import type KeplerTable from '@kepler.gl/table';
import type {PrepareArrowTableForGPUAnalyticsOptions} from '@luma-contributors/analysis/analysis-core';
import {SourcePreviewPanel} from './source-preview-panel';
import {SampleGalleryPanel} from './sample-gallery-panel';
import {RemoteSourceControl} from './remote-source-control';
import {loadLocalSource, REMOTE_SOURCE_FILE_ACCEPT} from './remote-source-loader';
import {restoreArrowSourceRoleColumns} from './source-type-routes';
import {
  prepareSandboxTrips,
  readSandboxTripOptions,
  type SandboxTripOptions
} from './trip-source-options';
import {makeArrowSourceFromNativeDataset} from './native-dataset-source';
import {preparePublishedH3, type CoordinateRoles} from './source-publication';
import type {PreparedH3Source} from './h3-source';
import type {FlowRoles} from './flow-source';
import type {NativeCellIndex} from './cell-layer-types';
import {
  DGGS_FOOTPRINT_COLUMN,
  appendDGGSFootprintColumn,
  type DGGSFamily,
  type DGGSFootprintSummary
} from './dggs-source';

export type SourceCandidate = {
  table: arrow.Table;
  /** Decoded columns before scalar routing; explicit source roles retain these exact types. */
  originalTable?: arrow.Table;
  label: string;
  isCurrent(): boolean;
  suggestedCoordinates?: CoordinateRoles;
  suggestedGeometryColumn?: string;
  suggestedTripOptions?: SandboxTripOptions;
  suggestedH3Column?: string;
  /** Recorded in the dataset source metadata and the saved source manifest. */
  cellIndex?: NativeCellIndex;
};

export function SourceWorkspacePanel({
  datasets,
  resolveNative,
  candidate,
  publications,
  onPublish,
  onRemove
}: {
  datasets: readonly KeplerTable[];
  resolveNative(datasetId: string): KeplerTable | undefined;
  candidate: SourceCandidate | null;
  publications: readonly {
    id: string;
    label: string;
    rowCount: number;
    selectedCount?: number;
    error: string | null;
  }[];
  onPublish(
    candidate: SourceCandidate,
    normalization: PrepareArrowTableForGPUAnalyticsOptions,
    coordinates?: CoordinateRoles,
    geometryColumn?: string,
    tripOptions?: SandboxTripOptions,
    h3Column?: string
  ): Promise<void>;
  onRemove(datasetId: string): Promise<void>;
}) {
  const [selected, setSelected] = useState<SourceCandidate | null>(candidate);
  const [longitude, setLongitude] = useState('');
  const [latitude, setLatitude] = useState('');
  const [altitude, setAltitude] = useState('');
  const [geometryColumn, setGeometryColumn] = useState('');
  const [h3Column, setH3Column] = useState('');
  const [cellFamily, setCellFamily] = useState<DGGSFamily | ''>('');
  const [cellColumn, setCellColumn] = useState('');
  const roleSource = useMemo(
    () =>
      selected &&
      restoreArrowSourceRoleColumns(
        selected.table,
        selected.originalTable,
        [h3Column, cellColumn].filter(Boolean)
      ),
    [selected, h3Column, cellColumn]
  );
  const [timedPaths, setTimedPaths] = useState(false);
  const [timestampEncoding, setTimestampEncoding] = useState('');
  const [timeOrigin, setTimeOrigin] = useState('');
  const [timeError, setTimeError] = useState('0');
  const [flowColumns, setFlowColumns] = useState({
    targetLongitude: '',
    targetLatitude: '',
    count: '',
    sourceName: '',
    targetName: ''
  });
  const getFlowRoles = (): FlowRoles | undefined => {
    if (!flowColumns.targetLongitude && !flowColumns.targetLatitude) return undefined;
    if (!flowColumns.targetLongitude || !flowColumns.targetLatitude)
      throw new Error('Choose both flow target coordinate columns.');
    if (!longitude || !latitude) throw new Error('Flow maps require source coordinate roles.');
    return {
      targetLongitude: flowColumns.targetLongitude,
      targetLatitude: flowColumns.targetLatitude,
      ...(flowColumns.count ? {count: flowColumns.count} : {}),
      ...(flowColumns.sourceName ? {sourceName: flowColumns.sourceName} : {}),
      ...(flowColumns.targetName ? {targetName: flowColumns.targetName} : {})
    };
  };
  const [groupedTrips, setGroupedTrips] = useState(false);
  const [tripGroupColumns, setTripGroupColumns] = useState({
    id: '',
    timestamp: '',
    longitude: '',
    latitude: '',
    altitude: '',
    part: ''
  });
  /** Grouped trips use the ID column as the layer's geometry-role column. */
  // Cell IDs are decoded once here; publication reuses this exact derived table.
  const dggs = useMemo<
    {table: arrow.Table; summary: DGGSFootprintSummary} | {error: string} | undefined
  >(() => {
    if (!roleSource || !cellFamily || !cellColumn) return undefined;
    try {
      return appendDGGSFootprintColumn(roleSource, cellColumn, cellFamily);
    } catch (failure) {
      return {error: failure instanceof Error ? failure.message : String(failure)};
    }
  }, [roleSource, cellFamily, cellColumn]);
  const dggsTable = dggs && 'table' in dggs ? dggs.table : undefined;
  const effectiveGeometryColumn = dggsTable
    ? DGGS_FOOTPRINT_COLUMN
    : groupedTrips
      ? tripGroupColumns.id
      : geometryColumn;
  const getTripOptions = () => {
    if (groupedTrips) {
      if (!tripGroupColumns.id) throw new Error('Choose a trip ID column');
    } else if (!geometryColumn || !timedPaths) return undefined;
    if (!timeError.trim()) throw new Error('Enter a trip time error budget');
    return readSandboxTripOptions({
      timestampEncoding,
      maximumTimeErrorMilliseconds: Number(timeError),
      ...(timeOrigin.trim() ? {originMilliseconds: Number(timeOrigin)} : {}),
      ...(groupedTrips
        ? {
            grouping: {
              timestampColumn: tripGroupColumns.timestamp,
              longitudeColumn: tripGroupColumns.longitude,
              latitudeColumn: tripGroupColumns.latitude,
              ...(tripGroupColumns.altitude ? {altitudeColumn: tripGroupColumns.altitude} : {}),
              ...(tripGroupColumns.part ? {partColumn: tripGroupColumns.part} : {})
            }
          }
        : {})
    });
  };
  const [error, setError] = useState('');
  const [preserveArrowTypes, setPreserveArrowTypes] = useState(false);
  const sourceReadGeneration = useRef(0);
  const selectSource = (source: SourceCandidate | null) => {
    setSelected(source);
    setLongitude(source?.suggestedH3Column ? '' : (source?.suggestedCoordinates?.longitude ?? ''));
    setLatitude(source?.suggestedH3Column ? '' : (source?.suggestedCoordinates?.latitude ?? ''));
    setAltitude(source?.suggestedH3Column ? '' : (source?.suggestedCoordinates?.altitude ?? ''));
    const flow = source?.suggestedCoordinates?.flow;
    setFlowColumns({
      targetLongitude: flow?.targetLongitude ?? '',
      targetLatitude: flow?.targetLatitude ?? '',
      count: flow?.count ?? '',
      sourceName: flow?.sourceName ?? '',
      targetName: flow?.targetName ?? ''
    });
    const grouping = source?.suggestedTripOptions?.grouping;
    setGroupedTrips(Boolean(grouping));
    setTripGroupColumns({
      id: grouping ? (source?.suggestedGeometryColumn ?? '') : '',
      timestamp: grouping?.timestampColumn ?? '',
      longitude: grouping?.longitudeColumn ?? '',
      latitude: grouping?.latitudeColumn ?? '',
      altitude: grouping?.altitudeColumn ?? '',
      part: grouping?.partColumn ?? ''
    });
    setGeometryColumn(
      source?.suggestedH3Column || grouping ? '' : (source?.suggestedGeometryColumn ?? '')
    );
    setH3Column(source?.suggestedH3Column ?? '');
    setTimedPaths(Boolean(!source?.suggestedH3Column && !grouping && source?.suggestedTripOptions));
    setTimestampEncoding(source?.suggestedTripOptions?.timestampEncoding ?? '');
    setTimeOrigin(source?.suggestedTripOptions?.originMilliseconds?.toString() ?? '');
    setTimeError(String(source?.suggestedTripOptions?.maximumTimeErrorMilliseconds ?? 0));
  };
  useEffect(
    () => () => {
      sourceReadGeneration.current++;
    },
    []
  );
  useEffect(() => {
    if (candidate) {
      sourceReadGeneration.current++;
      selectSource(candidate);
    }
  }, [candidate]);
  useEffect(() => {
    if (selected && !selected.isCurrent()) selectSource(null);
  }, [selected, datasets]);
  const prepareNative = (dataset: KeplerTable) => {
    sourceReadGeneration.current++;
    try {
      const source = makeArrowSourceFromNativeDataset(dataset, {
        maximumRows: 1000000,
        maximumCells: 16000000
      });
      selectSource({
        table: source.table,
        label: source.label,
        isCurrent: () => resolveNative(dataset.id) === dataset && source.isCurrent()
      });
      setError('');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  };
  return (
    <section aria-label="Resident datasets">
      <h2>Resident datasets</h2>
      <p>
        Datasets opened with Kepler Add Data are prepared automatically (see Automatic preparation
        in the sidebar header). Use this panel to prepare a dataset with other roles or precision
        choices, or open Arrow IPC files directly. Publication preserves source batches and field
        names.
      </p>
      <p>
        Preparing an existing dataset adds a resident copy. Its original dataset and layers remain
        available in Kepler.
      </p>
      <SampleGalleryPanel
        onPrepare={sample => {
          sourceReadGeneration.current++;
          selectSource({
            table: sample.table,
            label: sample.description.title,
            isCurrent: () => true,
            suggestedCoordinates: sample.coordinates,
            suggestedGeometryColumn: sample.geometryColumn,
            suggestedTripOptions: sample.tripOptions,
            suggestedH3Column: sample.h3Column
          });
          setError('');
        }}
      />
      <label>
        <input
          type="checkbox"
          checked={preserveArrowTypes}
          onChange={event => {
            sourceReadGeneration.current++;
            selectSource(null);
            setPreserveArrowTypes(event.target.checked);
          }}
        />
        Preserve original Arrow types
      </label>
      <p>
        Applies to the next Arrow IPC file opened. Wide integers stay numeric and require an
        explicit precision choice below; other unsupported types must be excluded. By default, wide
        integers become exact decimal category labels.
      </p>
      <label>
        Open Arrow IPC file
        <input
          type="file"
          accept={REMOTE_SOURCE_FILE_ACCEPT}
          onChange={event => {
            const file = event.target.files?.[0];
            const generation = ++sourceReadGeneration.current;
            if (!file) return;
            if (file.size > 128 * 1024 * 1024) {
              setError('Arrow file exceeds the 128 MiB import limit.');
              return;
            }
            void loadLocalSource(file, {preserveArrowTypes})
              .then(source => {
                if (sourceReadGeneration.current !== generation) return;
                selectSource(source);
                setError('');
              })
              .catch(failure => {
                if (sourceReadGeneration.current === generation) setError(String(failure));
              });
          }}
        />
      </label>
      <RemoteSourceControl
        onSource={source => {
          sourceReadGeneration.current++;
          selectSource(source);
          setError('');
        }}
      />
      {datasets.map(dataset => (
        <button key={dataset.id} onClick={() => prepareNative(dataset)}>
          Prepare {dataset.label}
        </button>
      ))}
      {selected && (
        <>
          <fieldset>
            <legend>Map coordinate roles (optional)</legend>
            <label>
              GeoJSON geometry column
              <select
                aria-label="Source geometry column"
                value={geometryColumn}
                disabled={groupedTrips}
                onChange={event => {
                  setGeometryColumn(event.target.value);
                  if (event.target.value) setH3Column('');
                  if (!event.target.value) setTimedPaths(false);
                  setLongitude('');
                  setLatitude('');
                  setAltitude('');
                }}
              >
                <option value="">No geometry column</option>
                {selected.table.schema.fields.map(field => (
                  <option key={field.name} value={field.name}>
                    {field.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              H3 index column
              <select
                aria-label="Source H3 column"
                value={h3Column}
                disabled={groupedTrips}
                onChange={event => {
                  setH3Column(event.target.value);
                  if (event.target.value) {
                    setGeometryColumn('');
                    setTimedPaths(false);
                    setLongitude('');
                    setLatitude('');
                    setAltitude('');
                  }
                }}
              >
                <option value="">No H3 column</option>
                {selected.table.schema.fields.map(field => (
                  <option key={field.name} value={field.name}>
                    {field.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Cell index family (S2, Geohash, A5)
              <select
                aria-label="Source cell index family"
                value={cellFamily}
                disabled={groupedTrips}
                onChange={event => {
                  setCellFamily(event.target.value as DGGSFamily | '');
                  if (event.target.value) {
                    setGeometryColumn('');
                    setH3Column('');
                    setTimedPaths(false);
                    setLongitude('');
                    setLatitude('');
                    setAltitude('');
                  }
                }}
              >
                <option value="">No cell index</option>
                <option value="s2">S2 (token or Uint64)</option>
                <option value="geohash">Geohash (string)</option>
                <option value="a5">A5 (hex token or Uint64)</option>
              </select>
            </label>
            {cellFamily && (
              <label>
                Cell ID column
                <select
                  aria-label="Source cell ID column"
                  value={cellColumn}
                  onChange={event => setCellColumn(event.target.value)}
                >
                  <option value="">Choose a column</option>
                  {selected.table.schema.fields.map(field => (
                    <option key={field.name} value={field.name}>
                      {field.name}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {dggs && 'error' in dggs && <p role="alert">{dggs.error}</p>}
            {dggs && 'summary' in dggs && (
              <p>
                {dggs.summary.validRowCount} valid {dggs.summary.family.toUpperCase()} cells;{' '}
                {dggs.summary.nullRowCount} missing and {dggs.summary.invalidRows.length} invalid
                IDs keep their source rows without footprints.{' '}
                {dggs.summary.datelineCrossingRowCount} footprints cross the antimeridian and use
                continuous longitudes. IDs stay exact in the original field; footprints are derived
                polygons rendered with Float32 coordinates.
              </p>
            )}
            {h3Column && (
              <p>
                H3 IDs stay in their original Arrow field and are excluded from scalar
                normalization. Check schema validates every cell before upload; numeric rounding is
                never used for H3 IDs. Geometry uses Float32 longitude/latitude; cells whose
                boundaries collapse at this precision also upload fp64 low parts.
              </p>
            )}
            {geometryColumn && (
              <p>
                This column is retained as geometry and excluded from scalar normalization below.
                Rendering and spatial selection use Float32 positions; original geometry and feature
                rows are retained.
              </p>
            )}
            {geometryColumn && (
              <fieldset>
                <legend>Path timing</legend>
                <label>
                  <input
                    type="checkbox"
                    checked={timedPaths}
                    onChange={event => setTimedPaths(event.target.checked)}
                  />
                  Timed paths (Trips)
                </label>
                {timedPaths && (
                  <>
                    <p>
                      One GeoJSON LineString per row, with longitude, latitude, altitude and time at
                      every vertex. Original times are retained; rendering uses relative Float32
                      milliseconds.
                    </p>
                    <label>
                      Trip timestamp encoding
                      <select
                        aria-label="Trip timestamp encoding"
                        value={timestampEncoding}
                        onChange={event => setTimestampEncoding(event.target.value)}
                      >
                        <option value="">Choose timestamp encoding</option>
                        <option value="unix-second">Unix seconds</option>
                        <option value="unix-millisecond">Unix milliseconds</option>
                        <option value="iso8601">ISO 8601 with timezone</option>
                      </select>
                    </label>
                    <label>
                      Trip time origin (epoch milliseconds)
                      <input
                        aria-label="Trip time origin (epoch milliseconds)"
                        value={timeOrigin}
                        placeholder="Use earliest accepted vertex"
                        onChange={event => setTimeOrigin(event.target.value)}
                      />
                    </label>
                    <label>
                      Maximum trip time error (milliseconds)
                      <input
                        aria-label="Maximum trip time error (milliseconds)"
                        type="number"
                        min="0"
                        step="any"
                        value={timeError}
                        onChange={event => setTimeError(event.target.value)}
                      />
                    </label>
                    <p className="hint">
                      Zero requires exact relative times. A positive budget explicitly allows
                      measured rounding. Check schema to review timed-path acceptance before
                      uploading.
                    </p>
                  </>
                )}
              </fieldset>
            )}
            <fieldset>
              <legend>Grouped trips</legend>
              <label>
                <input
                  type="checkbox"
                  aria-label="Grouped timed rows (Trips)"
                  checked={groupedTrips}
                  disabled={Boolean(geometryColumn || h3Column)}
                  onChange={event => {
                    setGroupedTrips(event.target.checked);
                    if (event.target.checked) {
                      setTimedPaths(false);
                      // Suggested point coordinates are replaced by the grouped trip columns.
                      setLongitude('');
                      setLatitude('');
                      setAltitude('');
                    }
                  }}
                />
                Grouped timed rows (Trips)
              </label>
              {groupedTrips && (
                <>
                  <p>
                    Rows are grouped by trip ID and ordered by time. A trip is shown while any of
                    its rows is selected, and unselected rows are omitted from its path like native
                    Kepler tabular trips. Rows with a missing ID, time or position are skipped.
                  </p>
                  {(
                    [
                      ['id', 'Trip ID column'],
                      ['timestamp', 'Trip time column'],
                      ['longitude', 'Trip longitude column'],
                      ['latitude', 'Trip latitude column'],
                      ['altitude', 'Trip altitude column'],
                      ['part', 'Trip part column']
                    ] as const
                  ).map(([role, label]) => (
                    <label key={role}>
                      {label}
                      <select
                        aria-label={label}
                        value={tripGroupColumns[role]}
                        onChange={event =>
                          setTripGroupColumns({...tripGroupColumns, [role]: event.target.value})
                        }
                      >
                        <option value="">
                          {role === 'altitude'
                            ? 'No altitude'
                            : role === 'part'
                              ? 'Single part'
                              : 'Choose column'}
                        </option>
                        {selected.table.schema.fields.map(field => (
                          <option key={field.name} value={field.name}>
                            {field.name}
                          </option>
                        ))}
                      </select>
                    </label>
                  ))}
                  <label>
                    Trip timestamp encoding
                    <select
                      aria-label="Trip timestamp encoding"
                      value={timestampEncoding}
                      onChange={event => setTimestampEncoding(event.target.value)}
                    >
                      <option value="">Choose timestamp encoding</option>
                      <option value="unix-second">Unix seconds</option>
                      <option value="unix-millisecond">Unix milliseconds</option>
                      <option value="iso8601">ISO 8601 with timezone</option>
                    </select>
                  </label>
                  <label>
                    Trip time origin (epoch milliseconds)
                    <input
                      aria-label="Trip time origin (epoch milliseconds)"
                      value={timeOrigin}
                      placeholder="Use earliest accepted vertex"
                      onChange={event => setTimeOrigin(event.target.value)}
                    />
                  </label>
                  <label>
                    Maximum trip time error (milliseconds)
                    <input
                      aria-label="Maximum trip time error (milliseconds)"
                      type="number"
                      min="0"
                      step="any"
                      value={timeError}
                      onChange={event => setTimeError(event.target.value)}
                    />
                  </label>
                </>
              )}
            </fieldset>
            {(['Longitude', 'Latitude'] as const).map(role => (
              <label key={role}>
                {role}
                <select
                  aria-label={`Source ${role.toLowerCase()} column`}
                  value={role === 'Longitude' ? longitude : latitude}
                  disabled={Boolean(geometryColumn || h3Column || groupedTrips)}
                  onChange={event =>
                    role === 'Longitude'
                      ? setLongitude(event.target.value)
                      : setLatitude(event.target.value)
                  }
                >
                  <option value="">Table only</option>
                  {selected.table.schema.fields.map(field => (
                    <option key={field.name} value={field.name}>
                      {field.name}
                    </option>
                  ))}
                </select>
              </label>
            ))}
            <label>
              Altitude (meters, optional)
              <select
                aria-label="Source altitude column"
                value={altitude}
                disabled={
                  Boolean(geometryColumn || h3Column || groupedTrips) || !longitude || !latitude
                }
                onChange={event => setAltitude(event.target.value)}
              >
                <option value="">No altitude</option>
                {selected.table.schema.fields.map(field => (
                  <option key={field.name} value={field.name}>
                    {field.name}
                  </option>
                ))}
              </select>
            </label>
            <fieldset>
              <legend>Flow map (optional)</legend>
              {(
                [
                  ['targetLongitude', 'Flow target longitude column'],
                  ['targetLatitude', 'Flow target latitude column'],
                  ['count', 'Flow count column'],
                  ['sourceName', 'Flow source name column'],
                  ['targetName', 'Flow target name column']
                ] as const
              ).map(([role, label]) => (
                <label key={role}>
                  {label}
                  <select
                    aria-label={label}
                    value={flowColumns[role]}
                    disabled={
                      Boolean(geometryColumn || h3Column || groupedTrips) ||
                      !longitude ||
                      !latitude ||
                      Boolean(altitude)
                    }
                    onChange={event => setFlowColumns({...flowColumns, [role]: event.target.value})}
                  >
                    <option value="">{role === 'count' ? 'Count each row once' : 'None'}</option>
                    {selected.table.schema.fields.map(field => (
                      <option key={field.name} value={field.name}>
                        {field.name}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
              <p>
                Source coordinates above are flow origins. Locations are unique coordinate pairs;
                clustering, totals and top flows use the native flowmap library on selected rows.
              </p>
            </fieldset>
            <p>
              Coordinates use longitude/latitude degrees. Both numeric fields must be selected in
              the precision preview. Empty or invalid geometry can be published as a table.
            </p>
          </fieldset>
          <SourcePreviewPanel
            key={JSON.stringify([
              selected.label,
              geometryColumn,
              h3Column,
              cellFamily,
              cellColumn,
              groupedTrips,
              tripGroupColumns,
              flowColumns,
              timedPaths,
              timestampEncoding,
              timeOrigin,
              timeError
            ])}
            source={dggsTable ?? roleSource!}
            sourceOnlyColumns={[effectiveGeometryColumn, h3Column, cellColumn].filter(Boolean)}
            label={selected.label}
            validateSource={
              cellFamily && !dggsTable
                ? () => {
                    throw new Error(
                      dggs && 'error' in dggs ? dggs.error : 'Choose a cell ID column'
                    );
                  }
                : h3Column
                  ? () => {
                      const prepared = preparePublishedH3(roleSource!, h3Column);
                      return formatH3SourcePreview(prepared);
                    }
                  : timedPaths || groupedTrips
                    ? () => {
                        const options = getTripOptions();
                        if (!options)
                          throw new Error('Choose trip geometry and timestamp encoding');
                        const prepared = prepareSandboxTrips(
                          selected.table,
                          effectiveGeometryColumn,
                          options
                        );
                        return `${prepared.tripCount} timed ${groupedTrips ? 'grouped trips' : 'paths'}, ${prepared.vertexCount} vertices; ${prepared.invalidRows.length} invalid rows retained but omitted from rendering. Maximum time rounding: ${prepared.actualTimeErrorMilliseconds} ms. Ready for upload.`;
                      }
                    : undefined
            }
            onPublish={async (_source, normalization) => {
              if (!selected.isCurrent())
                throw new Error('This source changed; prepare its current revision again.');
              if (Boolean(longitude) !== Boolean(latitude))
                throw new Error('Choose both coordinate roles, or leave both empty for a table.');
              await onPublish(
                dggsTable && cellFamily
                  ? {
                      ...selected,
                      table: dggsTable,
                      cellIndex: {family: cellFamily, column: cellColumn}
                    }
                  : {...selected, table: roleSource!},
                normalization,
                longitude && latitude
                  ? {
                      longitude,
                      latitude,
                      ...(altitude ? {altitude} : {}),
                      ...(getFlowRoles() ? {flow: getFlowRoles()} : {})
                    }
                  : undefined,
                effectiveGeometryColumn || undefined,
                getTripOptions(),
                h3Column || undefined
              );
            }}
          />
        </>
      )}
      {publications.map(publication => (
        <div key={publication.id}>
          <strong>{publication.label}</strong>
          <p>
            {publication.selectedCount ?? 'Preparing'} / {publication.rowCount} rows selected.
            Native Kepler table and layer controls use this dataset.
          </p>
          {publication.error && <p role="alert">{publication.error}</p>}
          <button
            onClick={() =>
              void onRemove(publication.id).catch(failure => setError(String(failure)))
            }
          >
            Remove {publication.label}
          </button>
        </div>
      ))}
      {publications.length > 0 && (
        <p>
          Saved state records every resident source. Use a portable bundle to include uploaded data
          for another session; reference-only URL links reconstruct only the fixture.
        </p>
      )}
      {error && <p role="alert">{error}</p>}
    </section>
  );
}

/** Source identity and measured coordinate quantization are deliberately reported separately. */
export function formatH3SourcePreview(
  prepared: Pick<PreparedH3Source, 'validRowCount' | 'nullRowCount' | 'invalidRows' | 'precision'>
): string {
  const reasons = prepared.invalidRows
    .slice(0, 3)
    .map(row => `row ${row.sourceRowIndex}: ${row.reason}`)
    .join('; ');
  const boundaryError = prepared.precision.maximumCoordinateErrorDegrees.toExponential(3);
  const centerError = prepared.precision.maximumCenterErrorDegrees.toExponential(3);
  return `${prepared.validRowCount} valid H3 cells; ${prepared.nullRowCount} missing and ${prepared.invalidRows.length} invalid IDs retain source rows and omit geometry.${reasons ? ` ${reasons}.` : ''} Exact source IDs and batches retained. Geometry uses Float32 longitude/latitude. Measured maximum rounding: boundary approximately ${boundaryError} degrees; center approximately ${centerError} degrees. Cells with Float32-collapsed boundaries use the fp64 hi/lo center path for hexagon columns and hi/lo vertex streams for mixed-resolution or pentagon polygons. Ready for upload.`;
}
