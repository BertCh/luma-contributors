// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  computeTerrainGridResident,
  estimateKeplerResidentRowCacheByteLength,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetSnapshot,
  type KeplerMemoryReservation,
  type KeplerResidentColumnSchema
} from '@luma-contributors/kepler';
import type {RasterTerrainOperation} from './analysis2-raster-terrain';
import {padRasterTileWithHalo} from './analysis3-raster-terrain-gpu';
import type {NumericRasterSource} from './raster-tile-ingest';

import {toDegrees} from '@math.gl/core';
import {WGS84_SEMI_MAJOR_AXIS_METERS} from './earth-constants';

const WEB_MERCATOR_RADIUS = WGS84_SEMI_MAJOR_AXIS_METERS;
const WORD_BYTES = 4;
const USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | Buffer.VERTEX;
/** Row-sized working words alive at once while one tile is computed, scaled and cropped. */
const TRANSIENT_WORDS_PER_CELL = 8;
const SCRATCH_MARGIN_BYTES = 4 * 1024 * 1024;

type ResidentFormat = 'float32' | 'sint32' | 'uint32';

export type TerrainResidentPublicationResult = Readonly<{
  dataset: KeplerDatasetSnapshot<KeplerArrowDataset>;
  /** Cells in the dataset, including cells that are null because the terrain is nodata there. */
  rowCount: number;
  /** Tiles, which are the dataset's source batches. */
  chunkCount: number;
  contributors: readonly string[];
  /** Bytes read back from the GPU: the min/max summary per numeric column, never cells. */
  readbackBytes: number;
  isCurrent(): boolean;
  /** Transfer ownership after the host attached the dataset to the native workspace. */
  transfer(): void;
  destroy(): Promise<void>;
}>;

function getCellCoordinates(
  crs: 'EPSG:4326' | 'EPSG:3857',
  extent: readonly [number, number, number, number],
  width: number,
  height: number
): {longitude: Float32Array; latitude: Float32Array; column: Int32Array; row: Int32Array} {
  const [west, south, east, north] = extent;
  const pixelWidth = (east - west) / width;
  const pixelHeight = (north - south) / height;
  const longitude = new Float32Array(width * height);
  const latitude = new Float32Array(width * height);
  const column = new Int32Array(width * height);
  const row = new Int32Array(width * height);
  for (let rowIndex = 0; rowIndex < height; rowIndex++) {
    const y = north - (rowIndex + 0.5) * pixelHeight;
    const rowLatitude =
      crs === 'EPSG:4326'
        ? y
        : toDegrees(2 * Math.atan(Math.exp(y / WEB_MERCATOR_RADIUS)) - Math.PI / 2);
    for (let columnIndex = 0; columnIndex < width; columnIndex++) {
      const x = west + (columnIndex + 0.5) * pixelWidth;
      const index = rowIndex * width + columnIndex;
      longitude[index] = crs === 'EPSG:4326' ? x : toDegrees(x / WEB_MERCATOR_RADIUS);
      latitude[index] = rowLatitude;
      column[index] = columnIndex;
      row[index] = rowIndex;
    }
  }
  return {longitude, latitude, column, row};
}

/**
 * Publishes terrain cells as a GPU-resident dataset with no cell readback. Each tile runs the
 * terrain contributors with a neighbor halo; the halo is cropped and the output unit scale applied
 * on the GPU, and the tile's value buffer becomes one chunk of the `<operation>` column directly.
 * Cell centers (`longitude`, `latitude`, `column`, `row`, `tile_index`) are generated from the
 * tile extents and uploaded; rows where the terrain is nodata are null in every column through GPU
 * copies of the validity words. Only the 12-byte min/max summary per column returns (domains).
 *
 * Differences from the table publication: cells are points (float32 centers), not polygons;
 * nodata cells stay as null rows instead of being dropped (the valid count is not known without a
 * readback); the dataset is not device-recoverable; GeoTIFF export must read the result back.
 */
export async function executeTerrainResidentPublication(props: {
  session: KeplerComputeSession;
  source: NumericRasterSource;
  operation: RasterTerrainOperation;
  bandIndex?: number;
  zFactor?: number;
  azimuthDegrees?: number;
  altitudeDegrees?: number;
  resultId: string;
  signal?: AbortSignal;
  isCurrent(): boolean;
}): Promise<TerrainResidentPublicationResult> {
  const {session, source, operation} = props;
  const device = session.device;
  const bandIndex = props.bandIndex ?? 1;
  const tiles = source.tiles.filter(tile => tile.width * tile.height > 0);
  // Usage error: nothing to compute without tiles.
  if (!tiles.length) throw new Error('Terrain needs at least one raster tile');
  if (session.registry.getDataset(props.resultId)) throw new Error('Terrain result ID is occupied');
  const chunkLengths = tiles.map(tile => tile.width * tile.height);
  const rowCount = chunkLengths.reduce((total, length) => total + length, 0);
  const schema: KeplerResidentColumnSchema[] = [
    {name: 'tile_index', format: 'uint32'},
    {name: 'column', format: 'sint32'},
    {name: 'row', format: 'sint32'},
    {name: 'longitude', format: 'float32'},
    {name: 'latitude', format: 'float32'},
    {name: operation, format: 'float32'}
  ];
  const request = session.registry.beginRequest(`terrain-resident:${props.resultId}`, []);
  const current = () =>
    !request.signal.aborted &&
    !props.signal?.aborted &&
    !session.deviceLost &&
    session.device === device &&
    props.isCurrent();
  const buffers: Buffer[] = [];
  let gpuReservation: KeplerMemoryReservation | undefined;
  let cpuReservation: KeplerMemoryReservation | undefined;
  let adopted: KeplerDatasetSnapshot<KeplerArrowDataset> | undefined;
  let readbackBytes = 0;
  try {
    gpuReservation = session.registry.reserveMemory(
      rowCount * WORD_BYTES * (schema.length * 2 + TRANSIENT_WORDS_PER_CELL) + SCRATCH_MARGIN_BYTES,
      'gpu',
      'scratch'
    );
    cpuReservation = session.registry.reserveMemory(
      estimateKeplerResidentRowCacheByteLength(schema, chunkLengths),
      'cpu',
      'readback'
    );
    const chunkBuffers: Record<string, Buffer[]> = Object.fromEntries(
      schema.map(column => [column.name, []])
    );
    const validityBuffers: Record<string, Buffer[]> = Object.fromEntries(
      schema.map(column => [column.name, []])
    );
    const allocate = (name: string, length: number, data?: ArrayBufferView): Buffer => {
      const buffer = device.createBuffer({
        id: `terrain-resident-${name}`,
        byteLength: Math.max(WORD_BYTES, length * WORD_BYTES),
        usage: USAGE,
        ...(data ? {data} : {})
      });
      buffers.push(buffer);
      return buffer;
    };
    const contributors = new Set<string>();
    for (const [tileIndex, tile] of tiles.entries()) {
      if (!current()) throw new Error('Terrain publication was cancelled or its source changed');
      const padded = padRasterTileWithHalo(source.tiles, tile, bandIndex);
      const result = await computeTerrainGridResident(
        device,
        {
          ...padded,
          crs: source.crs,
          operation,
          zFactor: props.zFactor,
          azimuthDegrees: props.azimuthDegrees,
          altitudeDegrees: props.altitudeDegrees
        },
        {cropHalo: true}
      );
      buffers.push(result.values, result.validity);
      for (const name of result.contributors) contributors.add(name);
      const length = tile.width * tile.height;
      const coordinates = getCellCoordinates(source.crs, tile.extent, tile.width, tile.height);
      chunkBuffers['tile_index'].push(
        allocate(`tile-${tileIndex}`, length, new Uint32Array(length).fill(tileIndex))
      );
      chunkBuffers['column'].push(allocate(`column-${tileIndex}`, length, coordinates.column));
      chunkBuffers['row'].push(allocate(`row-${tileIndex}`, length, coordinates.row));
      chunkBuffers['longitude'].push(
        allocate(`longitude-${tileIndex}`, length, coordinates.longitude)
      );
      chunkBuffers['latitude'].push(
        allocate(`latitude-${tileIndex}`, length, coordinates.latitude)
      );
      chunkBuffers[operation].push(result.values);
      // Every column shares the terrain validity: a nodata cell is a null row.
      const copyEncoder = device.createCommandEncoder({id: 'terrain-resident-validity'});
      for (const column of schema) {
        const copy =
          column.name === operation
            ? result.validity
            : allocate(`validity-${column.name}-${tileIndex}`, length);
        if (column.name !== operation)
          copyEncoder.copyBufferToBuffer({
            sourceBuffer: result.validity,
            destinationBuffer: copy,
            size: length * WORD_BYTES
          });
        validityBuffers[column.name].push(copy);
      }
      device.submit(copyEncoder.finish());
    }
    if (!current()) throw new Error('Terrain publication was cancelled or its source changed');
    const makeVector = <Format extends ResidentFormat>(
      name: string,
      format: Format,
      chunks: readonly Buffer[]
    ): GPUVector<Format> =>
      new GPUVector({
        name,
        format,
        type: 'data',
        ownsData: true,
        data: chunks.map(
          (buffer, index) =>
            new GPUData({format, buffer, length: chunkLengths[index], ownsBuffer: false})
        )
      });
    const columns: Record<string, GPUVector<ResidentFormat>> = Object.create(null);
    const validity: Record<string, GPUVector<'uint32'>> = Object.create(null);
    for (const column of schema) {
      columns[column.name] = makeVector(
        column.name,
        column.format as ResidentFormat,
        chunkBuffers[column.name]
      );
      validity[column.name] = makeVector(
        `${column.name}-validity`,
        'uint32',
        validityBuffers[column.name]
      );
    }
    gpuReservation.resize(buffers.reduce((total, buffer) => total + buffer.byteLength, 0));
    adopted =
      session.adoptResidentDataset({
        id: props.resultId,
        schema,
        columns,
        validity,
        schemaMetadata: {
          'luma.gl.terrain': JSON.stringify({
            operation,
            crs: source.crs,
            tiles: tiles.length,
            contributors: [...contributors]
          })
        },
        request,
        cpuReservation,
        gpuReservation,
        destroy: () => {
          for (const buffer of buffers) buffer.destroy();
        }
      }) ?? undefined;
    if (!adopted) throw new Error('Terrain result was superseded before publication');
    const snapshot = adopted;
    try {
      const domains = await session.readResidentDomains(snapshot, {
        workingMemoryBudgetBytes: Math.max(SCRATCH_MARGIN_BYTES, rowCount * WORD_BYTES * 4)
      });
      if (!domains) throw new Error('Terrain result was superseded before publication');
      readbackBytes = schema.length * 3 * WORD_BYTES;
    } catch (failure) {
      await removeDatasetAfterFence(session, snapshot.id);
      throw failure;
    }
    let transferred = false;
    return {
      dataset: snapshot,
      rowCount,
      chunkCount: tiles.length,
      contributors: [...contributors],
      readbackBytes,
      // The adopted dataset consumed its request (the signal is aborted once adoption completes),
      // so currency afterwards is the registry revision plus device and workspace identity.
      isCurrent: () =>
        session.registry.getDataset(snapshot.id)?.revision === snapshot.revision &&
        (transferred ||
          (!props.signal?.aborted &&
            !session.deviceLost &&
            session.device === device &&
            props.isCurrent())),
      transfer() {
        transferred = true;
      },
      async destroy() {
        if (transferred) return;
        if (session.registry.getDataset(snapshot.id)?.revision === snapshot.revision)
          await removeDatasetAfterFence(session, snapshot.id);
      }
    };
  } finally {
    try {
      // Adopted buffers belong to the dataset; anything else was never handed over.
      if (!adopted) for (const buffer of buffers) buffer.destroy();
      gpuReservation?.release();
      cpuReservation?.release();
    } finally {
      await session.registry.finishRequest(request);
    }
  }
}

/** Removes a dataset once submitted GPU work that may still read it has completed. */
async function removeDatasetAfterFence(session: KeplerComputeSession, id: string): Promise<void> {
  const fence = session.device.createFence();
  try {
    await session.removeDataset(id, Promise.race([fence.signaled, session.device.lost]));
  } finally {
    fence.destroy();
  }
}
