// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUIslandAttachment on a distance-band graph. Points with no neighbor in the band (islands) get
// an edge to their nearest neighbor. Band links are muted; the added links are drawn bright, and
// points are colored by their neighbor count after attachment.

import {GPUIslandAttachment} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints, createDistanceBandWeights} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {addKernelPass} from './shared/kernel';
import {parameters} from './island-attachment.parameters';
import {
  addCardinality,
  addLinkSegments,
  allocateWeights,
  getRowLinkGlyphs,
  getSegmentTargets,
  stretchToWorld,
  uploadWeightsBuffers
} from './shared/weights';

export default defineOperationRun({
  parameters,
  caption: 'Isolated points are attached to their nearest neighbor.',
  createInputs: ({random, world}) => {
    const points = createClusteredPoints(random, world, 200, 4, 0.18);
    const positions = stretchToWorld(points.positions, world);
    return {...points, positions, index: createPointIndex(positions), focus: {row: -1}};
  },
  build(builder, points, values) {
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const band = uploadWeightsBuffers(
      builder,
      'band',
      createDistanceBandWeights(points.positions, values.radius)
    );
    const attached = allocateWeights(
      builder,
      'attached',
      points.count,
      band.capacity + points.count * 2
    );
    const attachment = builder.allocate('attachment', 'uint32', points.count);
    const added = builder.allocate('added', 'uint32', attached.capacity);
    builder.graph.add(
      new GPUIslandAttachment({
        weights: band.input,
        positions: positions.view,
        reciprocal: values.reciprocal === 1,
        output: {
          ...attached.port,
          attachment: attachment.view,
          added: added.view
        }
      })
    );
    const cardinality = addCardinality(builder, 'attached', attached);
    // Links each point gained from the attachment: an island always gains one, its target only when
    // `reciprocal` is on.
    const bandCardinality = addCardinality(builder, 'band', band);
    const gained = builder.allocate('gained-links', 'float32', points.count);
    addKernelPass(builder, {
      id: 'gained-links',
      invocationCount: points.count,
      bindings: [
        {name: 'after', buffer: cardinality, access: 'read'},
        {name: 'before', buffer: bandCardinality, access: 'read'},
        {name: 'gained', buffer: gained, access: 'read_write'}
      ],
      body: 'gained[gainedOffset + index] = after[afterOffset + index] - before[beforeOffset + index];'
    });
    return {
      outputs: {
        positions,
        attachment,
        cardinality,
        gained,
        bandLinks: addLinkSegments(builder, 'band', band, positions),
        addedLinks: addLinkSegments(builder, 'added', attached, positions, {onlyWhere: added})
      }
    };
  },
  // The pointer picks a point: an island shows the link the operation added; others show their band.
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const attachment = peek(outputs.attachment);
      if (!attachment) return {};
      // An island near the pointer wins over the nearest point, so the added links are easy to find.
      let row = -1;
      let nearest = 0.3;
      for (let candidate = 0; candidate < attachment.length; candidate++) {
        if (attachment[candidate] === 0xffffffff) continue;
        const distance = inputs.index.distanceTo(candidate, pointer.x, pointer.y);
        if (distance < nearest) {
          row = candidate;
          nearest = distance;
        }
      }
      if (row < 0) row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const from = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      const glyphs: RunGlyph[] = [{shape: 'ring', at: from, radius: values.radius, dashed: true}];
      const target = attachment[row];
      if (target !== 0xffffffff) {
        const to = [inputs.positions[target * 2], inputs.positions[target * 2 + 1]] as const;
        glyphs.push({shape: 'line', from, to}, {shape: 'dot', at: to});
        const distance = Math.hypot(to[0] - from[0], to[1] - from[1]);
        return {
          focus: row,
          glyphs,
          readout: `island ${row} → point ${target} at ${distance.toFixed(3)} (band ${values.radius})`
        };
      }
      const band = getSegmentTargets(peek(outputs.bandLinks), from);
      glyphs.push(...getRowLinkGlyphs(from, band, {dashed: true}));
      return {
        focus: row,
        glyphs,
        readout: `point ${row} · ${band.length} band neighbors, no island`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.bandLinks},
    {
      layer: 'lines',
      segments: outputs.addedLinks,
      color: [1, 0.45, 0.2, 0.95],
      radius: 1.8
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.cardinality,
      radius: 3.4,
      range: [0, 8]
    },
    // Points that gained links through the attachment, larger the more they gained.
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.gained,
      ramp: 'sequential',
      range: [0, 4],
      fade: true,
      sizeByValue: true,
      radius: 10
    }
  ]
});
