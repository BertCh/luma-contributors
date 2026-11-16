// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {
  createSandboxSample,
  type SandboxSampleId
} from '../../../../examples/deck/kepler-compute/src/sandbox-samples';
import {prepareH3SourceFromArrow} from '../../../../examples/deck/kepler-compute/src/h3-source';
import {
  profileKeplerColumnTable,
  withKeplerDatasetNetworkProfile,
  type KeplerColumnTableProfileOptions,
  type KeplerProfileColumnTable
} from '../../src/profile/dataset-profile';
import {
  profileKeplerNativeDataset,
  type KeplerNativeDatasetShape
} from '../../src/profile/native-dataset-profile';
import {getCatalogBinding, getCatalogDefaultParameters} from '../../src/analysis/catalog-bindings';
import {profileKeplerDatasetNetwork} from '../../src/profile/dataset-network-profile';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';

/** Pure adapter over real sample cells and source roles, using the publisher's exact H3 decoder. */
export function readGallerySample(sampleId: SandboxSampleId) {
  const sample = createSandboxSample(sampleId);
  const grouping = sample.tripOptions?.grouping;
  const table: KeplerProfileColumnTable = {
    rowCount: sample.table.numRows,
    columns: sample.table.schema.fields.map(field => ({
      name: field.name,
      ...(arrow.DataType.isTimestamp(field.type) ? {type: 'time' as const} : {}),
      values: Array.from(sample.table.getChild(field.name)!, value =>
        typeof value === 'bigint' ? String(value) : value
      )
    }))
  };
  let shape: KeplerNativeDatasetShape | undefined;
  let geometryColumn = grouping ? undefined : sample.geometryColumn;
  if (sample.h3Column) {
    const prepared = prepareH3SourceFromArrow(sample.table.getChild(sample.h3Column)!, {
      accessor: 'decimal-or-hex'
    });
    geometryColumn = 'published_cell_boundary';
    table.columns = [
      ...table.columns,
      {
        name: geometryColumn,
        values: prepared.batches.flatMap(batch =>
          Array.from({length: batch.rowCount}, (_, row) =>
            batch.validity[row]
              ? JSON.stringify({
                  type: 'Polygon',
                  coordinates: [
                    Array.from(
                      {length: batch.boundaryOffsets[row + 1] - batch.boundaryOffsets[row]},
                      (_, vertex) => [
                        batch.boundaryPositions[(batch.boundaryOffsets[row] + vertex) * 2],
                        batch.boundaryPositions[(batch.boundaryOffsets[row] + vertex) * 2 + 1]
                      ]
                    )
                  ]
                })
              : null
          )
        )
      }
    ];
    shape = {kind: 'cells'};
  } else if (sample.tripOptions) {
    shape = {
      kind: 'trajectories',
      timestampEncoding: sample.tripOptions.timestampEncoding as 'unix-second' | 'unix-millisecond',
      ...(grouping
        ? {trackColumn: sample.geometryColumn, timeColumn: grouping.timestampColumn}
        : {})
    };
  }
  const metadata = JSON.parse(sample.table.schema.metadata.get('luma:sandbox-sample')!);
  if (metadata.grid) shape = {kind: 'scalar-grid', ...metadata.grid, valueColumn: 'value'};
  const options: KeplerColumnTableProfileOptions = {
    datasetId: sampleId,
    label: sample.description.title,
    revision: 1,
    coordinateSystem: 'EPSG:4326',
    ...(geometryColumn
      ? {geometryColumns: [geometryColumn]}
      : {
          pointColumns: grouping
            ? {longitude: grouping.longitudeColumn, latitude: grouping.latitudeColumn}
            : sample.coordinates
        })
  };
  return {table, options, shape};
}

export function profileGallerySample(sampleId: SandboxSampleId) {
  const {table, options, shape} = readGallerySample(sampleId);
  const profile = shape
    ? profileKeplerNativeDataset(table, options, shape)
    : profileKeplerColumnTable(table, options);
  if (profile.catalogGeometry !== 'line' || profile.trajectory) return profile;
  // S08 owns topology evidence. Reuse its exact builder/key rather than guessing from row counts.
  const binding = getCatalogBinding('network-structure')!;
  return withKeplerDatasetNetworkProfile(
    profile,
    profileKeplerDatasetNetwork(
      profile,
      readGalleryFeatures(sampleId),
      getCatalogDefaultParameters(binding),
      'gpu',
      binding.id
    )
  );
}

export function readGalleryFeatures(sampleId: SandboxSampleId): CatalogFeatureCollection {
  const {table, options} = readGallerySample(sampleId);
  return {
    type: 'FeatureCollection',
    features: Array.from({length: table.rowCount}, (_, row) => ({
      id: row,
      properties: Object.fromEntries(
        table.columns
          .filter(column => !options.geometryColumns?.includes(column.name))
          .map(column => [column.name, column.values[row]])
      ),
      geometry: options.geometryColumns?.length
        ? JSON.parse(
            String(
              table.columns.find(column => column.name === options.geometryColumns![0])!.values[row]
            )
          )
        : {
            type: 'Point',
            coordinates: [
              table.columns.find(column => column.name === options.pointColumns!.longitude)!.values[
                row
              ],
              table.columns.find(column => column.name === options.pointColumns!.latitude)!.values[
                row
              ]
            ]
          }
    }))
  };
}
