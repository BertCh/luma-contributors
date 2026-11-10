// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineMerge joins fragments that meet end to end into chains. Fragments are cut from a few long
// lines, shuffled and partly reversed; muted lines are the fragments, colors are the merged chains.
// With `directed`, a reversed fragment can no longer be joined, so more chains remain.

import {createRiverLine} from '../../../data/hydrography';
import {GPULineMerge} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {
  addPackedSegments,
  createPackedSegments,
  findNearestLine,
  getBendingReaches
} from './shared/lines';
import {parameters} from './line-merge.parameters';

const LINE_COUNT = 4;
const FRAGMENTS_PER_LINE = 4;
const VERTICES_PER_FRAGMENT = 5;

function getHash(index: number): number {
  const value = Math.sin(index * 78.233 + 1.7) * 43758.5453;
  return value - Math.floor(value);
}

export default defineOperationRun({
  parameters,
  caption: 'Fragments (muted) merged into continuous chains, one color per chain.',
  createInputs: ({world}) => {
    const fragments: number[][][] = [];
    const reaches = getBendingReaches(0, LINE_COUNT, world);
    for (let line = 0; line < LINE_COUNT; line++) {
      const total = FRAGMENTS_PER_LINE * (VERTICES_PER_FRAGMENT - 1);
      const points = createRiverLine(reaches[line], total + 1, world, line, LINE_COUNT);
      for (let fragment = 0; fragment < FRAGMENTS_PER_LINE; fragment++) {
        const first = fragment * (VERTICES_PER_FRAGMENT - 1);
        fragments.push(points.slice(first, first + VERTICES_PER_FRAGMENT));
      }
    }
    // Deterministic shuffle so fragment order says nothing about connectivity.
    const order = fragments.map((_, index) => index).sort((a, b) => getHash(a) - getHash(b));
    const shuffled = order.map(index => fragments[index]);
    const lineOffsets = Uint32Array.from(
      shuffled
        .map((_, index) => index * VERTICES_PER_FRAGMENT)
        .concat(shuffled.length * VERTICES_PER_FRAGMENT)
    );
    const getPositions = (reversedShare: number) =>
      Float32Array.from(
        shuffled.flatMap((fragment, index) =>
          (getHash(index + 11) < reversedShare ? [...fragment].reverse() : fragment).flat()
        )
      );
    return {fragmentCount: shuffled.length, lineOffsets, getPositions};
  },
  build(builder, inputs, values) {
    const initial = inputs.getPositions(values.reversed);
    const positions = builder.upload('positions', 'float32x2', initial);
    const lineOffsets = builder.upload('line-offsets', 'uint32', inputs.lineOffsets);
    const chainOffsets = builder.allocate('chain-offsets', 'uint32', inputs.fragmentCount + 1);
    const merged = builder.allocate('merged-positions', 'float32x2', initial.length / 2);
    const count = builder.allocate('chain-count', 'uint32', 1);
    builder.graph.add(
      new GPULineMerge({
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        directed: values.directed === 1,
        output: {
          chainOffsets: chainOffsets.view,
          positions: merged.view,
          status: {count: count.view}
        }
      })
    );
    const drawing = addPackedSegments(builder, 'chains', merged, chainOffsets);
    const original = builder.allocate(
      'original',
      'float32x4',
      initial.length / 2 - inputs.fragmentCount
    );
    const endpoints = builder.allocate('endpoints', 'float32x2', inputs.fragmentCount * 2);
    return {
      outputs: {original, endpoints, merged, chainOffsets, count, ...drawing},
      update: current => {
        const nextPositions = inputs.getPositions(current.reversed);
        positions.write(nextPositions);
        original.write(createPackedSegments(nextPositions, inputs.lineOffsets, false));
        endpoints.write(
          Float32Array.from({length: inputs.fragmentCount * 4}, (_, slot) => {
            const fragment = Math.floor(slot / 4);
            const row =
              fragment * VERTICES_PER_FRAGMENT + (slot % 4 < 2 ? 0 : VERTICES_PER_FRAGMENT - 1);
            return nextPositions[row * 2 + (slot % 2)];
          })
        );
      }
    };
  },
  rebuildOn: ['directed'],
  // The pointer picks the chain nearest to it from the operation's own output: the chain is drawn
  // bright with its ends ringed, and the readout counts the fragments that were joined into it.
  interaction: {
    kind: 'probe',
    hint: 'hover a chain',
    resolve(pointer, {outputs, peek}) {
      const merged = peek(outputs.merged);
      const offsets = peek(outputs.chainOffsets);
      const count = peek(outputs.count)?.[0];
      if (!merged || !offsets || count === undefined || count === 0) return {};
      const chains = {positions: merged, lineOffsets: offsets.subarray(0, count + 1)};
      const hit = findNearestLine(chains, pointer.x, pointer.y);
      if (!hit) return {};
      const first = offsets[hit.line];
      const end = offsets[hit.line + 1];
      const fragmentCount = Math.round((end - first - 1) / (VERTICES_PER_FRAGMENT - 1));
      const glyphs: RunGlyph[] = [
        {shape: 'polyline', points: Array.from(merged.subarray(first * 2, end * 2))},
        {shape: 'ring', at: [merged[first * 2], merged[first * 2 + 1]], radius: 0.02},
        {shape: 'ring', at: [merged[(end - 1) * 2], merged[(end - 1) * 2 + 1]], radius: 0.02}
      ];
      return {
        focus: hit.line,
        glyphs,
        readout: `chain ${hit.line} · ${fragmentCount} fragments · ${count} chains`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 4.5},
    {layer: 'dots', context: true, positions: outputs.endpoints, radius: 6},
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 2
    },
    {layer: 'dots', positions: outputs.points, values: outputs.lineIds, radius: 3}
  ]
});
