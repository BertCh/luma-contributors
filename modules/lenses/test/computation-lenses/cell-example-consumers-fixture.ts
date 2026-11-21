// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCellCompaction,
  GPUCellSetOutline
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createGlobalCoordinateFixture} from '../../../analysis/test/gpu-spatial-analysis/global-coordinates/global-coordinate-fixture';
import showcaseSource from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/cells/h3-aggregation.compute.ts?raw';
import explorerSource from '../../../../examples/deck/spatial-analysis-explorer/modes/cells-mode.ts?raw';
import {addKernelPass} from '../../../../examples/deck/spatial-analysis-showcase/src/engine/mode-kernels';

// Execute the actual constructor declarations, with small caller-owned resources substituted.
// This exercises the example wiring rather than a second handwritten copy of its source maps.
export function addExampleCellConsumers(
  fixture: ReturnType<typeof createGlobalCoordinateFixture>,
  example: 'showcase' | 'explorer',
  compactCapacity = 16,
  withRings = false
) {
  const source = example === 'showcase' ? showcaseSource : explorerSource;
  const sourceCells = fixture.output('uint32x2', 4);
  const sourceCount = fixture.output('uint32', 1);
  const sourceOverflow = fixture.output('uint32', 1);
  const sourceCandidateOverflow = fixture.output('uint32', 1);
  const compactCells = fixture.output('uint32x2', compactCapacity);
  const compactCount = fixture.output('uint32', 1);
  const compactOverflow = fixture.output('uint32', 1);
  const compactCandidateOverflow = fixture.output('uint32', 1);
  const ringOffsets = fixture.output('uint32', 9);
  const ringVertexCounts = fixture.output('uint32', 8);
  const ringPositions = fixture.output('float32x4', 16);
  const ringCount = fixture.output('uint32', 1);
  const ringOverflow = fixture.output('uint32', 1);
  const ringOpen = fixture.output('uint32', 1);
  type Output = ReturnType<typeof fixture.output>;
  const outputs = new Map<string, Output>([
    ['table-overflow', sourceOverflow],
    ['cover-overflow', sourceOverflow],
    ['cover-candidate-overflow', sourceCandidateOverflow],
    ['compact-count', compactCount],
    ['compact-overflow', compactOverflow],
    ['compact-candidate-overflow', compactCandidateOverflow],
    ['ring-vertex-counts', ringVertexCounts]
  ]);
  const getOutput = (
    name: string,
    format: Parameters<typeof fixture.output>[0],
    length: number
  ) => {
    let output = outputs.get(name);
    if (!output) {
      output = fixture.output(format, length);
      outputs.set(name, output);
    }
    return output.view;
  };
  const scope: Record<string, unknown> = {
    GPUCellSetOutline,
    GPUCellCompaction,
    graph: fixture.graph,
    name: 'example-outline',
    family: 'quadbin',
    selectedFamily: 'quadbin',
    tabularFamily: 'quadbin',
    resolution: 2,
    compactDepth: 1,
    OUTLINE_CAPACITY: 16,
    COVER_CAPACITY: 16,
    RING_CAPACITY: 8,
    tableCellsView: sourceCells.view,
    tableCountView: sourceCount.view,
    tableOverflow: 'table-overflow',
    tableGroupsView: undefined,
    ringOptions: withRings
      ? {
          normalizeWinding: true,
          output: {
            ringOffsets: ringOffsets.view,
            positions: fixture.graph.createDataView(ringPositions.view.buffer, {
              format: 'float32x2',
              length: 16
            }),
            status: {
              count: ringCount.view,
              overflow: ringOverflow.view,
              openSegmentCount: ringOpen.view
            }
          }
        }
      : undefined,
    ringOffsetsView: ringOffsets.view,
    ringCounts: 'ring-vertex-counts',
    endpoints: 'outline-endpoints',
    groups: undefined,
    outlineCount: 'outline-count',
    outlineOverflow: 'outline-overflow',
    coverCellsView: sourceCells.view,
    coverCountView: sourceCount.view,
    coverOverflow: 'cover-overflow',
    coverCandidateOverflow: 'cover-candidate-overflow',
    compactCellsView: compactCells.view,
    compactCountView: compactCount.view,
    compactCount: 'compact-count',
    compactOverflow: 'compact-overflow',
    compactCandidateOverflow: 'compact-candidate-overflow',
    coverOverflowView: sourceOverflow.view,
    coverCandidateOverflowView: sourceCandidateOverflow.view,
    compactOverflowView: compactOverflow.view,
    compactCandidateOverflowView: compactCandidateOverflow.view,
    uncompactCells: 'uncompact-cells',
    uncompactCount: 'uncompact-count',
    uncompactOverflow: 'uncompact-overflow',
    uncompactDropped: 'uncompact-dropped',
    create: (name: string) => name,
    view: (
      _name: string,
      buffer: string,
      format: Parameters<typeof fixture.output>[0],
      length: number
    ) => getOutput(buffer, format, length),
    importCellBuffer: (
      _graph: unknown,
      _name: string,
      buffer: string,
      format: Parameters<typeof fixture.output>[0],
      length: number
    ) => getOutput(buffer, format, length)
  };
  const contributors: (GPUCellSetOutline | GPUCellCompaction)[] = [];
  for (const className of ['GPUCellSetOutline', 'GPUCellCompaction']) {
    let offset = 0;
    while (true) {
      const start = source.indexOf(`new ${className}({`, offset);
      if (start < 0) break;
      const firstBrace = source.indexOf('{', start);
      let depth = 1;
      let end = firstBrace + 1;
      while (depth && end < source.length) {
        if (source[end] === '{') depth++;
        if (source[end] === '}') depth--;
        end++;
      }
      const declaration = source.slice(start, end).replace(/ as 'h3' \| 'quadbin'/g, '') + ')';
      const contributor: unknown = new Function('scope', `with (scope) {return ${declaration};}`)(
        scope
      );
      if (!(contributor instanceof GPUCellSetOutline || contributor instanceof GPUCellCompaction))
        throw new Error('unexpected example consumer');
      fixture.graph.add(contributor);
      if (withRings && contributor instanceof GPUCellSetOutline) {
        const passStart = source.indexOf('ringCountsPass = {');
        const first = source.indexOf('{', passStart);
        let last = first + 1;
        let passDepth = 1;
        while (passDepth && last < source.length) {
          if (source[last] === '{') passDepth++;
          if (source[last] === '}') passDepth--;
          last++;
        }
        const pass: Parameters<typeof addKernelPass>[1] = new Function(
          'scope',
          `with(scope){return (${source.slice(first, last)});}`
        )(scope);
        addKernelPass(fixture.graph, pass);
      }
      contributors.push(contributor);
      offset = end;
    }
  }
  return {
    sourceCells,
    sourceCount,
    sourceOverflow,
    sourceCandidateOverflow,
    compactCount,
    compactOverflow,
    compactCandidateOverflow,
    outputs,
    contributors,
    ringVertexCounts,
    ringCount
  };
}
