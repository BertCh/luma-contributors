// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GraphVectorView,
  validatePackedView,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  isGraphVectorView,
  getGraphViewChunks,
  validateGraphPorts,
  type GPUGraphPortDescriptor,
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** A stable diagnostic for a boundary through the middle of one cell's record. @internal */
export const RASTER_TIME_SERIES_SPLIT_CELL_RECORD = 'split cell record';

/** Required absence wins over every format error, independent of descriptor insertion order. @internal */
export function validateRasterTimeSeriesPorts(
  id: string,
  ports: Readonly<Record<string, unknown>>,
  descriptors: Readonly<Record<string, GPUGraphPortDescriptor>>
): void {
  // Delegate diagnostics to core. Core currently checks descriptor paths in insertion order;
  // this presence-only selection prevents optional format failures from hiding missing inputs.
  const missing = Object.fromEntries(
    Object.entries(descriptors).filter(([path, descriptor]) => {
      if (!descriptor.required) return false;
      const value = Object.prototype.hasOwnProperty.call(ports, path)
        ? ports[path]
        : path
            .split('.')
            .reduce<unknown>(
              (parent, name) =>
                parent && typeof parent === 'object'
                  ? (parent as Record<string, unknown>)[name]
                  : undefined,
              ports
            );
      return value === undefined || value === null;
    })
  );
  validateGraphPorts(id, ports, missing);
  validateGraphPorts(id, ports, descriptors);
}

/** Cell record layouts shared by both time-series lowering routes. @internal */
export function planRasterTimeSeriesCellTopology(
  id: string,
  cellCount: number,
  fields: Readonly<
    Record<string, {view: GraphDataView | GraphVectorView | undefined; entriesPerCell: number}>
  >
): {cellCounts: number[]; chunked: boolean} {
  const present = Object.entries(fields).filter(
    (entry): entry is [string, {view: GraphDataView | GraphVectorView; entriesPerCell: number}] =>
      entry[1].view !== undefined
  );
  validateRequirement(id, present.length > 0, 'cell fields must contain at least one view');
  for (const [name, field] of present)
    validateCapacity(id, `${name} entriesPerCell`, field.entriesPerCell);
  const template = present.find(([, field]) => isGraphVectorView(field.view))?.[1];
  const cellCounts = template
    ? getGraphViewChunks(template.view).map(chunk => chunk.length / template.entriesPerCell)
    : [cellCount];
  for (const [name, field] of present) {
    const chunks = getGraphViewChunks(field.view);
    validateRequirement(
      id,
      chunks.length === cellCounts.length,
      `${name} must match the cell chunk topology`
    );
    for (const [chunkIndex, chunk] of chunks.entries()) {
      validatePackedView(
        chunk,
        ['float32', 'uint32', 'sint32'],
        `${id} ${name} chunk ${chunkIndex}`
      );
      validateRequirement(
        id,
        !template ||
          (chunk.length % field.entriesPerCell === 0 &&
            chunk.length / field.entriesPerCell === cellCounts[chunkIndex]),
        `${name} chunk ${chunkIndex} ${RASTER_TIME_SERIES_SPLIT_CELL_RECORD}: length must be a multiple of ${field.entriesPerCell}, contain complete cell records and match the cell chunk topology`
      );
    }
    if (template) {
      validateRequirement(
        id,
        chunks.reduce((total, chunk) => total + chunk.length, 0) === field.view.length &&
          field.view.length === cellCount * field.entriesPerCell,
        `${name} chunk total must equal cellCount times entriesPerCell`
      );
    }
  }
  return {cellCounts, chunked: Boolean(template)};
}

/** Resolves every cell-field port to its original packed chunk without changing offsets or strides. @internal */
export function lowerRasterTimeSeriesCellChunks(
  cellCounts: readonly number[],
  views: readonly (GraphDataView | GraphVectorView)[]
): {cellCount: number; views: ReadonlyMap<GraphDataView | GraphVectorView, GraphDataView>}[] {
  const fields = views.map(view => ({view, chunks: getGraphViewChunks(view)}));
  return cellCounts.map((cellCount, chunkIndex) => ({
    cellCount,
    views: new Map(fields.map(({view, chunks}) => [view, chunks[chunkIndex]]))
  }));
}
