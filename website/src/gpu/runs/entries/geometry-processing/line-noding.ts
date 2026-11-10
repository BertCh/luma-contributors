// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineNoding splits every line at each crossing into atomic segments. Each noded segment is
// drawn colored by the source line it came from, up to the count the operation wrote.

import {GPULineNoding} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createCrossingLines, pickCrossingLines} from './shared/polygon-set';
import {addKernel, addMaskedPoints} from './shared/shape-segments';
import {parameters} from './line-noding.parameters';

const CAPACITY = 512;

export default defineOperationRun({
  parameters,
  caption: 'Lines split at every crossing, colored by their source line.',
  createInputs: ({seed, world}) => ({
    seed,
    world,
    lines: createCrossingLines(makeRandom(seed), world, 3, 4, 12, 0.06),
    shifts: [] as Array<{line: number; offset: number}>
  }),
  build(builder, inputs, values) {
    const lines = createCrossingLines(
      makeRandom(inputs.seed),
      inputs.world,
      values.firstCount,
      values.secondCount,
      12,
      0.06
    );
    inputs.lines = lines;
    inputs.shifts = [];
    const positions = builder.upload('positions', 'float32x2', lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', lines.lineOffsets);
    const endpoints = builder.allocate('endpoints', 'float32x4', CAPACITY);
    const sourceFeatureIds = builder.allocate('source-feature-ids', 'uint32', CAPACITY);
    const count = builder.allocate('count', 'uint32', 1);
    builder.graph.add(
      new GPULineNoding({
        lines: {kind: 'lines', positions: positions.view, lineOffsets: lineOffsets.view},
        intersectionCapacity: 512,
        output: {
          endpoints: endpoints.view,
          sourceFeatureIds: sourceFeatureIds.view,
          sourceRingIds: builder.allocate('source-ring-ids', 'uint32', CAPACITY).view,
          sourceEdgeIds: builder.allocate('source-edge-ids', 'uint32', CAPACITY).view,
          sourceStartParameters: builder.allocate('source-start', 'float32', CAPACITY).view,
          sourceEndParameters: builder.allocate('source-end', 'float32', CAPACITY).view,
          status: {
            count: count.view,
            overflow: builder.allocate('overflow', 'uint32', 1).view,
            requiredCount: builder.allocate('required-count', 'uint32', 1).view,
            candidateOverflow: builder.allocate('candidate-overflow', 'uint32', 1).view
          }
        }
      })
    );
    // Each source line gets two alternating colors, so neighboring noded pieces read as separate.
    const categories = builder.allocate('piece-categories', 'uint32', CAPACITY * 2);
    addKernel(builder, {
      id: 'piece-categories',
      invocationCount: CAPACITY * 2,
      bindings: [
        {name: 'sources', buffer: sourceFeatureIds, access: 'read'},
        {name: 'count', buffer: count, access: 'read'},
        {name: 'categories', buffer: categories, access: 'read_write'}
      ],
      body: `let row = index / 2u;
  categories[categoriesOffset + index] = select(0xffffffffu, (sources[sourcesOffset + row] % 4u) * 2u + row % 2u, row < count[countOffset]);`
    });
    return {
      update: () => {
        lines.regenerate(inputs.shifts);
        positions.write(lines.positions);
      },
      outputs: {
        count,
        nodes: endpoints,
        edges: builder.upload(
          'edges',
          'uint32x2',
          Uint32Array.from({length: CAPACITY * 2}, (_, index) => index)
        ),
        categories,
        splits: addMaskedPoints(builder, 'splits', endpoints, count)
      }
    };
  },
  rebuildOn: ['firstCount', 'secondCount'],
  // The pointer is where one west-east and one south-north line cross: both follow it.
  interaction: {
    kind: 'drag',
    hint: 'move the crossing',
    rest: ({inputs, toUnit}) => {
      const {lines} = inputs;
      return toUnit([
        lines.lineBases[
          lines.firstLineCount + Math.floor((lines.lineCount - lines.firstLineCount) / 2)
        ],
        lines.lineBases[Math.floor(lines.firstLineCount / 2)]
      ]);
    },
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.shifts = pickCrossingLines(inputs.lines, pointer.x, pointer.y, inputs.shifts);
      const count = peek(outputs.count)?.[0];
      return {
        rerun: true,
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout:
          count === undefined
            ? 'noded segments ...'
            : `${count} noded segments · ${count - inputs.lines.lineCount} splits`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {
      layer: 'segments',
      nodes: outputs.nodes,
      edges: outputs.edges,
      values: outputs.categories,
      radius: 2.2
    },
    {layer: 'dots', positions: outputs.splits, radius: 3.2, color: [...theme.ink, 1]}
  ]
});
