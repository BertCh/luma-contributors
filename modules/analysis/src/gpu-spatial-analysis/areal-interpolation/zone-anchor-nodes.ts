// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getSortKeyBits} from '../../analysis-core/kernels/sorted-segment-sums';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validateCapacity,
  validateViewLength,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** The invalid sort key is zoneCount + 1 and must remain representable as uint32. */
const MAXIMUM_ANCHOR_ZONE_COUNT = 0xfffffffe;

/** Deterministic parallel anchor claims: lowest zone wins each cell, then each donor keeps one cell. */
export function getZoneAnchorNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    zones: GraphDataView<'uint32'>;
    zoneCount: number;
    anchorCells: GraphDataView<'uint32'>;
  }
): {zones: GraphDataView<'uint32'>; nodes: GPUCommandNode<Parameters>[]} {
  const {id, operation, zones, zoneCount, anchorCells} = props;
  const cellCount = zones.length;
  validateCapacity(id, 'zoneCount', zoneCount, 1, MAXIMUM_ANCHOR_ZONE_COUNT);
  validateViewLength(id, 'anchorCells', anchorCells, zoneCount);
  validateGraphChunkBindingRanges(graph, id, {zones, anchorCells});
  for (const [name, length] of Object.entries({
    effectiveZones: cellCount,
    counts: zoneCount,
    claims: cellCount,
    keys: zoneCount,
    rows: zoneCount,
    sortedKeys: zoneCount,
    sortedRows: zoneCount
  })) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      `anchor ${name}`,
      length * Uint32Array.BYTES_PER_ELEMENT,
      'reduce the grid size or zoneCount'
    );
  }
  const view = (name: string, length: number) =>
    createTransientView(graph, `${id}-anchor-${name}`, 'uint32', length);
  // The returned raster composes with graph copy passes as well as storage readers.
  const effectiveZones = createTransientView(
    graph,
    `${id}-anchor-zones`,
    'uint32',
    cellCount,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const counts = view('counts', zoneCount);
  const claims = view('claims', cellCount);
  const keys = view('keys', zoneCount);
  const rows = view('rows', zoneCount);
  const sortedKeys = view('sorted-keys', zoneCount);
  const sortedRows = view('sorted-rows', zoneCount);
  const declarations = `const ZONE_COUNT: u32 = ${zoneCount}u;
const CELL_COUNT: u32 = ${cellCount}u;`;
  const nodes: GPUCommandNode<Parameters>[] = [
    createFillNode(graph, {
      id: `${id}-anchor-counts-clear`,
      operation,
      view: counts,
      type: 'u32',
      value: '0u'
    }),
    createFillNode(graph, {
      id: `${id}-anchor-claims-clear`,
      operation,
      view: claims,
      type: 'u32',
      value: '0xffffffffu'
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-anchor-copy`,
      operation,
      variant: 'anchor-copy',
      invocationCount: cellCount,
      declarations,
      bindings: [
        {name: 'zones', view: zones, type: 'u32', access: 'read'},
        {name: 'effectiveZones', view: effectiveZones, type: 'u32', access: 'read_write'},
        {name: 'counts', view: counts, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `let zone = zones[zonesOffset + index];
      effectiveZones[effectiveZonesOffset + index] = zone;
      if (zone < ZONE_COUNT) { atomicAdd(&counts[countsOffset + zone], 1u); }`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-anchor-propose`,
      operation,
      variant: 'anchor-propose',
      invocationCount: zoneCount,
      declarations,
      bindings: [
        {name: 'zones', view: zones, type: 'u32', access: 'read'},
        {name: 'anchorCells', view: anchorCells, type: 'u32', access: 'read'},
        {name: 'counts', view: counts, type: 'u32', access: 'read'},
        {name: 'claims', view: claims, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `if (counts[countsOffset + index] != 0u) { return; }
      let anchor = anchorCells[anchorCellsOffset + index];
      if (anchor >= CELL_COUNT) { return; }
      let owner = zones[zonesOffset + anchor];
      if (owner < ZONE_COUNT && counts[countsOffset + owner] < 2u) { return; }
      atomicMin(&claims[claimsOffset + anchor], index);`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-anchor-keys`,
      operation,
      variant: 'anchor-keys',
      invocationCount: zoneCount,
      declarations,
      bindings: [
        {name: 'zones', view: zones, type: 'u32', access: 'read'},
        {name: 'anchorCells', view: anchorCells, type: 'u32', access: 'read'},
        {name: 'claims', view: claims, type: 'u32', access: 'read'},
        {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
        {name: 'rows', view: rows, type: 'u32', access: 'read_write'}
      ],
      body: `var key = ZONE_COUNT + 1u;
      let anchor = anchorCells[anchorCellsOffset + index];
      if (anchor < CELL_COUNT && claims[claimsOffset + anchor] == index) { key = min(zones[zonesOffset + anchor], ZONE_COUNT); }
      keys[keysOffset + index] = key;
      rows[rowsOffset + index] = index;`
    }),
    ...new GPUSort({
      id: `${id}-anchor-sort`,
      keys,
      values: rows,
      outputKeys: sortedKeys,
      outputValues: sortedRows,
      keyBits: getSortKeyBits(zoneCount + 1)
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-anchor-claim`,
      operation,
      variant: 'anchor-claim',
      invocationCount: zoneCount,
      declarations,
      bindings: [
        {name: 'anchorCells', view: anchorCells, type: 'u32', access: 'read'},
        {name: 'counts', view: counts, type: 'u32', access: 'read'},
        {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
        {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'effectiveZones', view: effectiveZones, type: 'u32', access: 'read_write'}
      ],
      body: `let owner = sortedKeys[sortedKeysOffset + index];
      if (owner > ZONE_COUNT) { return; }
      var first = 0u; var end = index;
      while (first < end) {
        let middle = first + (end - first) / 2u;
        if (sortedKeys[sortedKeysOffset + middle] < owner) { first = middle + 1u; } else { end = middle; }
      }
      // Stable owner sort retains ascending zone order, exactly the sequential donor quota.
      if (owner < ZONE_COUNT && index - first >= counts[countsOffset + owner] - 1u) { return; }
      let zone = sortedRows[sortedRowsOffset + index];
      effectiveZones[effectiveZonesOffset + anchorCells[anchorCellsOffset + zone]] = zone;`
    })
  ];
  return {zones: effectiveZones, nodes};
}
