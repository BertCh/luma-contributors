// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createSaturatedScanTotalNodes} from '../../analysis-core/kernels/scan-total';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {FIND_LINE_WGSL} from '../line-segmentize/line-segmentize-kernels';
import type {LineDensityGridConstants} from './line-density-kernels';

/** Counts literal geographic longitude jumps without changing the segment-cell length route. */
export function createLineDensitySeamNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; constants: LineDensityGridConstants; output: GraphDataView<'uint32'>}
): GPUCommandNode<Parameters>[] {
  const {id, constants, output} = props;
  const rowCount = constants.positions.length;
  const counts = createTransientView(graph, `${id}-seam-counts`, 'uint32', rowCount);
  const offsets = createTransientView(graph, `${id}-seam-offsets`, 'uint32', rowCount);
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-seam-flags`,
      operation: 'GPULineDensity',
      variant: 'seam-flags',
      invocationCount: rowCount,
      bindings: [
        {name: 'positions', view: constants.positions, type: 'f32', access: 'read'},
        {name: 'lineOffsets', view: constants.lineOffsets, type: 'u32', access: 'read'},
        {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
      ],
      declarations: `const LINE_COUNT: u32 = ${constants.lineOffsets.length - 1}u;
const ROW_COUNT: u32 = ${rowCount}u;
${FIND_LINE_WGSL}
// Compare against a half turn before adding opposite-sign magnitudes. For max in [90,180],
// 180-max is exact by Sterbenz; a rounded full separation can erase a genuine seam crossing.
fn hasLongitudeJump(first: f32, second: f32) -> bool {
  let maximum = max(abs(first), abs(second));
  let minimum = min(abs(first), abs(second));
  if ((first < 0.0) != (second < 0.0)) {
    if (maximum > 180.0) { return true; }
    if (maximum < 90.0) { return false; }
    return bitcast<u32>(minimum) > bitcast<u32>(180.0 - maximum);
  }
  if (maximum <= 180.0) { return false; }
  if (maximum <= 360.0) { return bitcast<u32>(minimum) < bitcast<u32>(maximum - 180.0); }
  if (minimum < maximum * 0.5) { return true; }
  let differenceBits = bitcast<u32>(maximum - minimum);
  return differenceBits < 0x7f800000u && differenceBits > bitcast<u32>(180.0);
}`,
      body: `counts[countsOffset + index] = 0u;
  let line = findLine(index);
  if (line == NO_LINE) { return; }
  if (index + 1u >= min(lineOffsets[lineOffsetsOffset + line + 1u], ROW_COUNT)) { return; }
  let first = positions[positionsOffset + 2u * index];
  let second = positions[positionsOffset + 2u * (index + 1u)];
  counts[countsOffset + index] = select(0u, 1u, ${constants.coordinateSystem === 'spherical'} && hasLongitudeJump(first, second));`
    }),
    ...new GPUScan({
      id: `${id}-seam-scan`,
      input: counts,
      output: offsets,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    ...createSaturatedScanTotalNodes(graph, {
      id: `${id}-seam-total`,
      operation: 'GPULineDensity',
      counts,
      offsets,
      rowCount,
      total: output
    })
  ];
}
