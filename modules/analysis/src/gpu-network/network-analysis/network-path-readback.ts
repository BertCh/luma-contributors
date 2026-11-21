// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getPackedUint32Values} from '../../analysis-core/kernels/packed-readback';
import {
  getTypedArrayName,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Uint32 payload or raw bytes from a buffer read, respecting the byte window. */
export type NetworkPathReadbackWords = Uint32Array | Uint8Array | ArrayBuffer;
export type NetworkPathReadbackInput = {
  ids: NetworkPathReadbackWords;
  nodeRows?: NetworkPathReadbackWords;
  nodeDomain?: 'ids' | 'rows';
  pathFound: NetworkPathReadbackWords;
  lineOffsets: NetworkPathReadbackWords;
  count: NetworkPathReadbackWords;
  edges?: {
    ids: NetworkPathReadbackWords;
    /** Required witness: application ID 0xffffffff is a valid identity. */
    edgeRows: NetworkPathReadbackWords;
    lineOffsets: NetworkPathReadbackWords;
    count: NetworkPathReadbackWords;
  };
};
export type NetworkPathReadbackPath = {
  found: boolean;
  fullyStored: boolean;
  nodeDomain: 'ids' | 'rows';
  /** Empty unless the entire required node interval fits the written prefix. */
  nodes: Uint32Array;
  edges: null | {
    fullyStored: boolean;
    ids: Uint32Array;
    edgeRows: Uint32Array;
    /** Nonzero exactly for resolved CSR slots, independent of the application IDs. */
    resolutionValidity: Uint32Array;
  };
};

/**
 * Reads complete path intervals without mistaking a truncated prefix for a shorter path.
 * Request edges.output.edgeRows on the contributor to distinguish unresolved edges from IDs.
 * Required offsets remain useful for sizing even when fullyStored is false.
 */
export function decodeGPUNetworkPathExtractionOutput(
  input: NetworkPathReadbackInput
): NetworkPathReadbackPath[] {
  const id = 'decodeGPUNetworkPathExtractionOutput';
  const words = (value: NetworkPathReadbackWords, name: string): Uint32Array => {
    validateRequirement(
      id,
      getTypedArrayName(value) === 'Uint32Array' ||
        getTypedArrayName(value) === 'Uint8Array' ||
        Object.prototype.toString.call(value) === '[object ArrayBuffer]',
      `${name} must be Uint32Array, Uint8Array or ArrayBuffer`
    );
    validateRequirement(
      id,
      value.byteLength % 4 === 0,
      `${name} byteLength must be a whole number of uint32 elements; received ${value.byteLength}`
    );
    return getPackedUint32Values(value);
  };
  const nodeDomain = input.nodeDomain ?? 'ids';
  validateRequirement(
    id,
    nodeDomain === 'ids' || nodeDomain === 'rows',
    `nodeDomain must be ids or rows; received ${nodeDomain}`
  );
  validateRequirement(
    id,
    nodeDomain !== 'rows' || input.nodeRows !== undefined,
    'nodeDomain rows requires nodeRows'
  );
  const nodes = words(nodeDomain === 'rows' ? input.nodeRows! : input.ids, 'nodes');
  const found = words(input.pathFound, 'pathFound');
  const offsets = words(input.lineOffsets, 'lineOffsets');
  const validatePrefix = (countInput: NetworkPathReadbackWords, capacity: number, name: string) => {
    const count = words(countInput, name);
    validateRequirement(
      id,
      count.length === 1 && count[0] <= capacity,
      `${name} must contain one written count at most capacity ${capacity}`
    );
    return count[0];
  };
  const validateOffsets = (values: Uint32Array, name: string) => {
    validateRequirement(
      id,
      values.length === found.length + 1,
      `${name} must contain exactly pathFound.length + 1 elements`
    );
    validateRequirement(id, values[0] === 0, `${name} must begin at zero`);
    for (let row = 0; row < found.length; row++)
      validateRequirement(
        id,
        values[row + 1] >= values[row],
        `${name} must be nondecreasing at row ${row}`
      );
  };
  const count = validatePrefix(input.count, nodes.length, 'count');
  validateOffsets(offsets, 'lineOffsets');
  validateRequirement(
    id,
    count <= offsets[found.length],
    'count must not exceed the terminal required lineOffset'
  );
  const edges = input.edges && {
    ids: words(input.edges.ids, 'edges.ids'),
    rows: words(input.edges.edgeRows, 'edges.edgeRows'),
    offsets: words(input.edges.lineOffsets, 'edges.lineOffsets'),
    count: 0
  };
  if (edges && input.edges) {
    edges.count = validatePrefix(
      input.edges.count,
      Math.min(edges.ids.length, edges.rows.length),
      'edges.count'
    );
    validateOffsets(edges.offsets, 'edges.lineOffsets');
    validateRequirement(
      id,
      edges.count <= edges.offsets[found.length],
      'edges.count must not exceed the terminal required edge lineOffset'
    );
    for (let row = 0; row < found.length; row++)
      validateRequirement(
        id,
        edges.offsets[row + 1] - edges.offsets[row] ===
          Math.max(offsets[row + 1] - offsets[row] - 1, 0),
        `edges.lineOffsets must hold max(nodeLength - 1, 0) rows at target ${row}`
      );
  }
  return Array.from(found, (flag, row) => {
    const fullyStored = flag !== 0 && offsets[row + 1] > offsets[row] && offsets[row + 1] <= count;
    const edgeStart = edges?.offsets[row] ?? 0;
    const edgeEnd = edges?.offsets[row + 1] ?? 0;
    const edgesFullyStored = flag !== 0 && edges !== undefined && edgeEnd <= edges.count;
    const edgeRows = edgesFullyStored ? edges!.rows.slice(edgeStart, edgeEnd) : new Uint32Array();
    return {
      found: flag !== 0,
      fullyStored,
      nodeDomain,
      nodes: fullyStored ? nodes.slice(offsets[row], offsets[row + 1]) : new Uint32Array(),
      edges: edges
        ? {
            fullyStored: edgesFullyStored,
            ids: edgesFullyStored ? edges.ids.slice(edgeStart, edgeEnd) : new Uint32Array(),
            edgeRows,
            resolutionValidity: Uint32Array.from(edgeRows, edge => Number(edge !== 0xffffffff))
          }
        : null
    };
  });
}
