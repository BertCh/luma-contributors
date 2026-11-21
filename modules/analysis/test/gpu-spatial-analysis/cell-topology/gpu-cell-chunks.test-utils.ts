// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {expect} from 'vitest';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {GPUCellGridPath} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-grid-path';
import {GPUCellTopology} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-topology';
import {GPUH3CellIdentities} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-h3-cell-identities';
import {GPUS2CellHierarchy} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-hierarchy';
import {GPUS2CellMeasures} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-measures';
import {GPUHealpixPointToCell} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell';
import {createVectorView} from '../../utils/gpu-contributor-test-utils';

export type CellChunkFormat = 'uint32x2' | 'float32x2' | 'uint32';
export type CellChunkViewProps<Format extends CellChunkFormat> = {
  name: string;
  format: Format;
  length: number;
  chunkIndex: number;
  rowBase: number;
  multiplier: number;
};
export type CellChunkViewFactory = <Format extends CellChunkFormat>(
  props: CellChunkViewProps<Format>
) => GraphDataView<Format>;

export function createCellChunkFields(
  graph: GPUCommandGraph,
  lengths: number[],
  stride: number,
  createChunkView?: CellChunkViewFactory,
  distanceStride = 1
) {
  const createField = <Format extends 'uint32x2' | 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    multiplier = 1
  ) =>
    createVectorView(
      name,
      format,
      lengths.map((length, chunkIndex) => {
        if (createChunkView)
          return createChunkView({
            name,
            format,
            length: length * multiplier,
            chunkIndex,
            rowBase: lengths.slice(0, chunkIndex).reduce((total, rows) => total + rows, 0),
            multiplier
          });
        const backing = createTransientView(
          graph,
          `${name}-${chunkIndex}`,
          'uint32',
          length * multiplier * (format === 'uint32' ? 1 : 2) + 67
        );
        return graph.createDataView(backing.buffer, {
          format,
          length: length * multiplier,
          byteOffset: [4, 260, 12][chunkIndex % 3]
        });
      })
    );
  return {
    cells: createField('cells', 'uint32x2'),
    destinations: createField('destinations', 'uint32x2'),
    positions: createField('positions', 'float32x2'),
    mask: createField('mask', 'uint32'),
    outputCells: createField('outputCells', 'uint32x2', stride),
    distances: createField('distances', 'uint32', distanceStride),
    counts: createField('counts', 'uint32'),
    status: createField('status', 'uint32'),
    measures: createField('measures', 'float32x2'),
    validity: createField('validity', 'uint32')
  };
}
export type CellChunkFields = ReturnType<typeof createCellChunkFields>;
export type CellChunkViewFields = {
  [Name in keyof CellChunkFields]: CellChunkFields[Name] | CellChunkFields[Name]['data'][number];
};
export type CellChunkOutputName =
  | 'outputCells'
  | 'distances'
  | 'counts'
  | 'status'
  | 'measures'
  | 'validity';
export type CellChunkCase = {
  name: string;
  input: 'cells' | 'positions';
  stride: number;
  distanceStride?: number;
  outputs: CellChunkOutputName[];
  create: (fields: CellChunkViewFields, id?: string) => GPUCommandNodeProducer;
};
export const CELL_CHUNK_CASES: CellChunkCase[] = [
  {
    name: 'GPUCellGridPath',
    input: 'cells',
    stride: 4,
    outputs: ['outputCells', 'distances', 'counts'],
    create: (fields, id) =>
      new GPUCellGridPath({
        id,
        family: 'h3',
        origins: fields.cells,
        destinations: fields.destinations,
        mask: fields.mask,
        maximumPathLength: 4,
        output: {cells: fields.outputCells, distances: fields.distances, counts: fields.counts}
      })
  },
  {
    name: 'GPUCellTopology',
    input: 'cells',
    stride: 1,
    outputs: ['outputCells', 'counts'],
    create: (fields, id) =>
      new GPUCellTopology({
        id,
        family: 's2',
        operation: {type: 'parent', resolution: 0},
        cells: fields.cells,
        mask: fields.mask,
        output: {cells: fields.outputCells, counts: fields.counts}
      })
  },
  {
    name: 'GPUCellTopology children',
    input: 'cells',
    stride: 4,
    outputs: ['outputCells', 'counts'],
    create: (fields, id) =>
      new GPUCellTopology({
        id,
        family: 's2',
        operation: {type: 'children', inputResolution: 0, resolution: 1},
        cells: fields.cells,
        mask: fields.mask,
        output: {cells: fields.outputCells, counts: fields.counts}
      })
  },
  {
    name: 'GPUH3CellIdentities directed-edges',
    input: 'cells',
    stride: 6,
    outputs: ['outputCells'],
    create: (fields, id) =>
      new GPUH3CellIdentities({
        id,
        cells: fields.cells,
        operation: 'directed-edges',
        output: fields.outputCells
      })
  },
  {
    name: 'GPUH3CellIdentities vertices',
    input: 'cells',
    stride: 6,
    outputs: ['outputCells'],
    create: (fields, id) =>
      new GPUH3CellIdentities({
        id,
        cells: fields.cells,
        operation: 'vertices',
        output: fields.outputCells
      })
  },
  {
    name: 'GPUS2CellHierarchy',
    input: 'cells',
    stride: 2,
    outputs: ['outputCells', 'status'],
    create: (fields, id) =>
      new GPUS2CellHierarchy({
        id,
        cells: fields.cells,
        operation: 'range',
        output: {cells: fields.outputCells, status: fields.status}
      })
  },
  {
    name: 'GPUS2CellMeasures float32',
    input: 'cells',
    stride: 1,
    outputs: ['measures', 'status'],
    create: (fields, id) =>
      new GPUS2CellMeasures({
        id,
        cells: fields.cells,
        output: {measures: fields.measures, status: fields.status}
      })
  },
  {
    name: 'GPUS2CellMeasures fp64',
    input: 'cells',
    stride: 1,
    outputs: ['measures', 'status'],
    create: (fields, id) =>
      new GPUS2CellMeasures({
        id,
        cells: fields.cells,
        precision: 'double-single',
        output: {measures: fields.measures, status: fields.status}
      })
  },
  {
    name: 'GPUHealpixPointToCell',
    input: 'positions',
    stride: 1,
    outputs: ['outputCells', 'validity'],
    create: (fields, id) =>
      new GPUHealpixPointToCell({
        id,
        order: 0,
        positions: fields.positions,
        mask: fields.mask,
        output: {cells: fields.outputCells, validity: fields.validity}
      })
  }
];

/** Add every returned node so graph validation catches undefined or malformed resources. */
export function addCellChunkNodes(
  graph: GPUCommandGraph,
  contributor: GPUCommandNodeProducer
): readonly GPUCommandNode[] {
  const nodes = contributor.getCommandNodes(graph);
  for (const node of nodes) graph.add(node);
  return nodes;
}

/** Bound staging allocations while checking outputs larger than one storage binding. */
const CELL_CHUNK_READBACK_MAXIMUM_BYTE_LENGTH = 4 * 1024 * 1024;

/** Probe both sides of workgroup boundaries and the first/last records of every chunk. */
export function getCellChunkProbeRows(length: number): number[] {
  return [...new Set([0, 1, 254, 255, 256, 257, 510, 511, 512, 513, length - 2, length - 1])]
    .filter(row => row >= 0 && row < length)
    .sort((first, second) => first - second);
}

/** Initialize the complete primary output without a data-sized CPU staging array. */
export function fillCellChunkOutput(
  buffer: Buffer,
  format: 'float32' | 'uint32',
  value: number
): void {
  const wordCount = CELL_CHUNK_READBACK_MAXIMUM_BYTE_LENGTH / 4;
  const values =
    format === 'float32'
      ? new Float32Array(wordCount).fill(value)
      : new Uint32Array(wordCount).fill(value);
  for (let byteOffset = 0; byteOffset < buffer.byteLength; byteOffset += values.byteLength) {
    buffer.write(
      values.subarray(0, Math.min(values.length, (buffer.byteLength - byteOffset) / 4)),
      byteOffset
    );
  }
}

/** Check every component, including all padding and all invalid rows, against a sparse CPU oracle. */
export async function expectCellChunkOutputRecords(
  buffer: Buffer,
  format: 'float32' | 'uint32',
  rowCount: number,
  componentCount: number,
  records: ReadonlyMap<number, readonly number[]>,
  tolerance: {absolute?: number; relative?: number} = {}
): Promise<void> {
  const rowsPerRead = Math.max(
    1,
    Math.floor(CELL_CHUNK_READBACK_MAXIMUM_BYTE_LENGTH / (componentCount * 4))
  );
  for (let rowBase = 0; rowBase < rowCount; rowBase += rowsPerRead) {
    const length = Math.min(rowsPerRead, rowCount - rowBase);
    const bytes = await buffer.readAsync(rowBase * componentCount * 4, length * componentCount * 4);
    const values =
      format === 'float32'
        ? new Float32Array(bytes.buffer, bytes.byteOffset, length * componentCount)
        : new Uint32Array(bytes.buffer, bytes.byteOffset, length * componentCount);
    let mismatch: string | undefined;
    for (let row = 0; row < length && mismatch === undefined; row++) {
      const expected = records.get(rowBase + row);
      for (let component = 0; component < componentCount; component++) {
        const actual = values[row * componentCount + component];
        const expectedValue = expected?.[component] ?? 0;
        const allowed =
          (tolerance.absolute ?? 0) + (tolerance.relative ?? 0) * Math.abs(expectedValue);
        if (!Number.isFinite(actual) || Math.abs(actual - expectedValue) > allowed) {
          mismatch = `row ${rowBase + row}, component ${component}: ${actual}, expected ${expectedValue} (tolerance ${allowed})`;
          break;
        }
      }
    }
    expect(mismatch).toBeUndefined();
  }
}
