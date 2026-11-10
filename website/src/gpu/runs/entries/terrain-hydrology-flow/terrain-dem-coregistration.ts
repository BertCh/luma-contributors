// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainDEMCoregistration between a reference DEM and a second epoch that is the same terrain
// plus a constant vertical bias, plus real change (a thinning glacier) on unstable ground. The
// median shift is fitted on the stable mask only, and the corrected difference is drawn: the bias
// is gone, the glacier loss remains. A curtain swipes between the raw difference of the epochs
// (bias and glacier) and the corrected one. The inset bars are the fitted shift and the true bias.
// (The operation interleaves difference and standard deviation; the difference is copied out.)

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import {GPUTerrainDEMCoregistration} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {parameters} from './terrain-dem-coregistration.parameters';
import {createTerrainInputs, getTerrainContextLayer} from './shared/terrain';

const NOISE_STANDARD_DEVIATION = 0.002;

export default defineOperationRun({
  parameters,
  caption:
    'Corrected difference between two epochs: the constant bias is removed, the glacier loss stays.',
  createInputs: context => {
    const terrain = createTerrainInputs(context, 36);
    const unstable = new Uint32Array(terrain.cellCount);
    const glacierChange = new Float32Array(terrain.cellCount);
    const noise = new Float32Array(terrain.cellCount);
    for (let cell = 0; cell < terrain.cellCount; cell++) {
      const x = ((cell % terrain.columns) + 0.5) / terrain.columns;
      const y = (Math.floor(cell / terrain.columns) + 0.5) / terrain.rows;
      const distance = Math.hypot((x - 0.3) * (terrain.columns / terrain.rows), y - 0.45);
      const glacier = Math.exp(-((distance / 0.22) ** 2));
      glacierChange[cell] = -glacier;
      unstable[cell] = glacier > 0.1 ? 1 : 0;
      noise[cell] = (context.random() - 0.5) * 2 * NOISE_STANDARD_DEVIATION;
    }
    return {
      ...terrain,
      stableMask: unstable.map(value => 1 - value),
      glacierChange,
      noise,
      trueBias: parameters[0].default
    };
  },
  build(builder, terrain) {
    const reference = builder.upload('reference', 'float32', terrain.elevation);
    const moving = builder.allocate('moving', 'float32', terrain.cellCount);
    const stableMask = builder.upload('stable-mask', 'uint32', terrain.stableMask);
    const differenceVariance = builder.upload(
      'difference-variance',
      'float32',
      new Float32Array(terrain.cellCount).fill(NOISE_STANDARD_DEVIATION ** 2)
    );
    const fit = builder.allocate('fit', 'float32', 4);
    // The fitted vertical shift (fit row 0), copied so it can be drawn as one bar.
    // Row 0 is the fitted shift (written by the kernel below), row 1 the bias that was applied.
    const shift = builder.allocate('shift', 'float32', 2);
    const differences = builder.allocate('differences', 'float32', terrain.cellCount * 2);
    const corrected = builder.allocate('corrected', 'float32', terrain.cellCount);
    const raw = builder.allocate('raw-difference', 'float32', terrain.cellCount);
    builder.graph.add(
      new GPUTerrainDEMCoregistration({
        grid: terrain.grid,
        reference: reference.view,
        moving: moving.view,
        stableMask: stableMask.view,
        differenceVariance: differenceVariance.view,
        output: {
          fit: fit.view,
          differences: differences.view,
          validity: builder.allocate('validity', 'uint32', terrain.cellCount).view,
          volume: builder.allocate('volume', 'float32', 3).view
        }
      })
    );
    builder.graph.add({
      getCommandNodes: graph => [
        createWGSLKernelNode(graph, {
          id: 'fitted-shift',
          operation: 'fit-shift',
          bindings: [
            {name: 'fitSource', view: fit.view, type: 'f32', access: 'read'},
            {name: 'shiftTarget', view: shift.view, type: 'f32', access: 'read_write'}
          ],
          invocationCount: 1,
          body: 'shiftTarget[shiftTargetOffset] = fitSource[fitSourceOffset];'
        }),
        // The operation interleaves difference and standard deviation: keep the difference.
        createWGSLKernelNode(graph, {
          id: 'corrected-difference',
          operation: 'corrected-difference',
          bindings: [
            {name: 'interleaved', view: differences.view, type: 'f32', access: 'read'},
            {name: 'difference', view: corrected.view, type: 'f32', access: 'read_write'}
          ],
          invocationCount: terrain.cellCount,
          body: 'difference[differenceOffset + index] = interleaved[interleavedOffset + index * 2u];'
        })
      ]
    });
    return {
      outputs: {reference, corrected, raw, shift},
      update: current => {
        const epoch = new Float32Array(terrain.cellCount);
        const rawDifference = new Float32Array(terrain.cellCount);
        for (let cell = 0; cell < terrain.cellCount; cell++) {
          rawDifference[cell] =
            current.bias + current.thinning * terrain.glacierChange[cell] + terrain.noise[cell];
          epoch[cell] = terrain.elevation[cell] + rawDifference[cell];
        }
        moving.write(epoch);
        // What the bars compare: the shift the fit finds (row 0, overwritten by the GPU) and the bias.
        terrain.trueBias = current.bias;
        shift.write(Float32Array.of(0, current.bias));
        raw.write(rawDifference);
      }
    };
  },
  // x draws the curtain: the raw difference of the two epochs on the left, the corrected one on the
  // right, both on one color range.
  interaction: {
    kind: 'swipe',
    hint: 'swipe raw | corrected',
    rest: [0.5, 0.5],
    resolve(pointer, {outputs, inputs, peek}) {
      const shift = peek(outputs.shift);
      return {
        curtain: pointer.x,
        glyphs: [{shape: 'curtain', x: pointer.x}],
        readout: `raw | corrected · fitted shift ${shift ? formatNumber(shift[0], 3) : '...'} · true bias ${formatNumber(inputs.trueBias, 3)}`
      };
    }
  },
  draw: ({inputs, outputs, interaction}) => [
    getTerrainContextLayer(inputs, outputs.reference),
    {
      layer: 'raster',
      clip: {side: 'left', x: interaction.curtain ?? 0},
      values: outputs.raw,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'diverging',
      range: [-0.1, 0.1],
      alpha: 0.85
    },
    {
      layer: 'raster',
      clip: {side: 'right', x: interaction.curtain ?? 0},
      values: outputs.corrected,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'diverging',
      range: [-0.1, 0.1],
      alpha: 0.85
    },
    {
      // The shift the operation fitted on stable ground, beside the bias that was applied.
      layer: 'chart',
      kind: 'bars',
      values: outputs.shift,
      range: [-0.12, 0.12],
      rectangle: [0.8, 0.04, 0.97, 0.24],
      ramp: 'diverging'
    }
  ]
});
