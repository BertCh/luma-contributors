// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addChangeOfSupportRecipe: values on coarse source zones (outlined, dots) transferred to a
// different target zone system by area overlap. Targets are filled by the transferred value;
// mass that falls in gaps of the target system is lost (extensive, denominator 'zone').

import {addChangeOfSupportRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {createBuiltUpZones} from './lib/built-up-zones';
import {createQuadZones, getPolygonPort, getQuadFill, uploadQuadZones} from './lib/quads';
import {createWorldGrid} from './lib/rasterize';
import {findZoneAt, getZoneRing} from './lib/zone-lookup';
import {parameters} from './change-of-support-recipe.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Source zone values redistributed to target zones by overlapping area.',
  createInputs: ({random, world}) => ({
    source: createBuiltUpZones(random, world, 5, 3),
    target: createQuadZones(random, world, 9, 5, 0.14),
    grid: createWorldGrid(world)
  }),
  build(builder, {source, target, grid}, values) {
    const sourceMesh = uploadQuadZones(builder, source, 'source');
    const targetMesh = uploadQuadZones(builder, target, 'target');
    const sourceValues = builder.allocate('source-values', 'float32', source.count);
    const capacity = 2048;
    const offsets = builder.allocate('offsets', 'uint32', target.count + 1);
    const neighbors = builder.allocate('neighbors', 'uint32', capacity);
    const weights = builder.allocate('weights', 'float32', capacity);
    const intensiveWeightValues = builder.allocate('intensive-weights', 'float32', capacity);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const requiredCount = builder.allocate('required-count', 'uint32', 1);
    const invalidCount = builder.allocate('invalid-count', 'uint32', 1);
    const extensiveValues = builder.allocate('extensive', 'float32', target.count);
    const intensiveValues = builder.allocate('intensive', 'float32', target.count);
    addChangeOfSupportRecipe(builder.graph, {
      grid: grid.grid,
      parameters: builder.upload('parameters', 'float32', grid.parameters).view,
      source: {...getPolygonPort(sourceMesh), crossingCapacity: 4096},
      target: {...getPolygonPort(targetMesh), crossingCapacity: 4096},
      denominator: {extensive: values.denominator === 1 ? 'overlap' : 'zone'},
      pairCapacity: capacity,
      sourceValues: sourceValues.view,
      outputs: {
        extensiveWeights: {offsets: offsets.view, neighbors: neighbors.view, weights: weights.view},
        intensiveWeightValues: intensiveWeightValues.view,
        extensiveValues: extensiveValues.view,
        intensiveValues: intensiveValues.view,
        status: {
          overflow: overflow.view,
          requiredCount: requiredCount.view,
          invalidCount: invalidCount.view
        }
      }
    });
    return {
      outputs: {
        sourceValues,
        offsets,
        neighbors,
        weights,
        shown: values.show === 1 ? intensiveValues : extensiveValues,
        sourceMesh,
        targetMesh
      },
      update: current =>
        sourceValues.write(
          Float32Array.from(source.values, value => 20 + 180 * current.contrast * value ** 2)
        )
    };
  },
  rebuildOn: ['denominator', 'show'],
  // The pointer picks a target zone: the source zones it overlaps are outlined, joined by their weight.
  interaction: {
    kind: 'probe',
    hint: 'hover a target zone',
    rest: ({toUnit, inputs}) =>
      toUnit([inputs.target.centers[2 * 22], inputs.target.centers[2 * 22 + 1]]),
    resolve(pointer, {inputs, outputs, peek}) {
      const zone = findZoneAt(inputs.target, pointer.x, pointer.y);
      const glyphs: RunGlyph[] = [
        {shape: 'polyline', points: getZoneRing(inputs.target, zone), closed: true}
      ];
      const offsets = peek(outputs.offsets);
      const neighbors = peek(outputs.neighbors);
      const weights = peek(outputs.weights);
      const shown = peek(outputs.shown);
      let sourceCount = 0;
      if (offsets && neighbors && weights) {
        const center: [number, number] = [
          inputs.target.centers[zone * 2],
          inputs.target.centers[zone * 2 + 1]
        ];
        for (let slot = offsets[zone]; slot < offsets[zone + 1]; slot++) {
          const source = neighbors[slot];
          sourceCount++;
          glyphs.push(
            {
              shape: 'polyline',
              points: getZoneRing(inputs.source, source),
              closed: true,
              dashed: true
            },
            {
              shape: 'line',
              from: center,
              to: [inputs.source.centers[source * 2], inputs.source.centers[source * 2 + 1]]
            }
          );
        }
      }
      return {
        glyphs,
        readout: `target ${zone} · from ${sourceCount} sources · value ${shown ? shown[zone].toFixed(0) : '...'}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    getQuadFill(outputs.targetMesh, inputs.target, outputs.shown, {ramp: 'sequential'}),
    {
      layer: 'lines',
      context: true,
      segments: outputs.sourceMesh.outline,
      count: inputs.source.outlineCount,
      radius: 2.5
    },
    {
      layer: 'dots',
      positions: outputs.sourceMesh.centers,
      values: outputs.sourceValues,
      radius: 9,
      ramp: 'sequential'
    }
  ]
});
