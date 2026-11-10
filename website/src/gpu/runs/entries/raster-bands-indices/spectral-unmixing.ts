// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterSpectralUnmixing: every cell holds a four-band spectrum that is fitted to three reference
// materials (water, soil, vegetation); the operation estimates their fractions (abundances).
// A patch of an unknown fourth material does not fit any mixture, so it lights up in the residual.
// The three fractions of a cell are stored side by side, so GPURasterSampling reads the wide
// abundance raster at the right column to pull each fraction out as a plain per-cell raster.
// `constraint` is a construction prop: unconstrained fractions can go negative or exceed one,
// simplex fractions are nonnegative and sum to one. The pointer probes a cell: its three
// fractions and the residual are bars, with the values in the readout.

import {
  getGPURasterSamplingParameterValues,
  GPURasterSampling,
  GPURasterSpectralUnmixing,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {sampleReflectance} from '../../../data/reflectance';
import {formatNumber} from '../../interaction';
import {parameters} from './spectral-unmixing.parameters';
import {getCellAt, getCellBox} from './support/gestures';
import {readFloat32} from './support/labels';

const CONSTRAINTS = ['unconstrained', 'nonnegative', 'simplex'] as const;
const ENDMEMBERS = [
  [0.08, 0.06, 0.03, 0.02], // water: green, red, near-infrared, shortwave infrared
  [0.2, 0.28, 0.34, 0.42], // bare soil
  [0.1, 0.06, 0.5, 0.22] // vegetation
];
const UNKNOWN_MATERIAL = [0.25, 0.02, 0.01, 0.3];
const BAND_COUNT = 4;
const MEMBER_COUNT = 3;
const INSET = [0.03, 0.05, 0.3, 0.3] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['constraint'],
  caption: 'Four-band spectra in; material fractions and the misfit of each cell out.',
  createInputs: ({random, aspect}) => {
    const rows = 36;
    const columns = Math.max(8, Math.round((rows * aspect) / MEMBER_COUNT));
    const bands = sampleReflectance(columns, rows);
    const spectra = [bands.green, bands.red, bands.nearInfrared, bands.shortwaveInfrared];
    const noise = Float32Array.from({length: columns * rows * BAND_COUNT}, () => random() - 0.5);
    return {columns, rows, cellCount: columns * rows, spectra, noise, focus: {cell: -1}};
  },
  build(builder, inputs, values) {
    const {cellCount} = inputs;
    const spectra = builder.allocate('spectra', 'float32', cellCount * BAND_COUNT);
    const endmembers = builder.upload(
      'endmembers',
      'float32',
      Float32Array.from(ENDMEMBERS.flat())
    );
    const output = {
      abundances: builder.allocate('abundances', 'float32', cellCount * MEMBER_COUNT),
      residual: builder.allocate('residual', 'float32', cellCount),
      status: builder.allocate('status', 'uint32', cellCount),
      iterationCounts: builder.allocate('iteration-counts', 'uint32', cellCount)
    };
    builder.graph.add(
      new GPURasterSpectralUnmixing({
        cellCount,
        bandCount: BAND_COUNT,
        endmemberCount: MEMBER_COUNT,
        values: spectra.view,
        endmembers: endmembers.view,
        constraint: CONSTRAINTS[Math.round(values.constraint)],
        output: {
          abundances: output.abundances.view,
          residual: output.residual.view,
          status: output.status.view,
          iterationCounts: output.iterationCounts.view
        }
      })
    );
    // Abundance m of cell (column, row) is pixel (column * 3 + m, row) of a three-times-wide raster.
    const wideGrid = makeGPURasterMetadata({
      width: inputs.columns * MEMBER_COUNT,
      height: inputs.rows,
      origin: [0, inputs.rows]
    });
    const samplingParameters = builder.upload(
      'sampling-parameters',
      'float32',
      getGPURasterSamplingParameterValues({method: 'nearest'}, wideGrid)
    );
    const fractions = ENDMEMBERS.map((_, member) => {
      const positions = new Float32Array(cellCount * 2);
      for (let cell = 0; cell < cellCount; cell++) {
        positions[cell * 2] = (cell % inputs.columns) * MEMBER_COUNT + member + 0.5;
        positions[cell * 2 + 1] = inputs.rows - Math.floor(cell / inputs.columns) - 0.5;
      }
      const fraction = builder.allocate(`fraction-${member}`, 'float32', cellCount);
      builder.graph.add(
        new GPURasterSampling({
          id: `gather-${member}`,
          grid: wideGrid,
          values: output.abundances.view,
          positions: builder.upload(`positions-${member}`, 'float32x2', positions).view,
          parameters: samplingParameters.view,
          output: {values: fraction.view}
        })
      );
      return fraction;
    });
    return {
      outputs: {
        fractions,
        residual: output.residual,
        inset: builder.allocate('inset', 'float32', MEMBER_COUNT + 1)
      },
      update: current => {
        const data = new Float32Array(cellCount * BAND_COUNT);
        for (let cell = 0; cell < cellCount; cell++) {
          for (let band = 0; band < BAND_COUNT; band++) {
            const value = inputs.spectra[band][cell];
            const foreign =
              inputs.spectra[2][cell] > 0.25 ? current.unknown * UNKNOWN_MATERIAL[band] : 0;
            data[cell * BAND_COUNT + band] =
              value + foreign + current.noise * inputs.noise[cell * BAND_COUNT + band];
          }
        }
        spectra.write(data);
      }
    };
  },
  // A rebuild clears the inset bars; rewrite them from the last probed cell.
  async afterRun({inputs, outputs}) {
    const cell = inputs.focus.cell;
    if (cell < 0) return;
    const fractions = await Promise.all(outputs.fractions.map(readFloat32));
    const residual = await readFloat32(outputs.residual);
    outputs.inset.write(
      Float32Array.from([
        ...fractions.map(values => values[cell]),
        Math.min(1, residual[cell] * 10)
      ])
    );
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a cell',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, world, peek}) => {
      const {column, row, index} = getCellAt(inputs, world, pointer);
      inputs.focus.cell = index;
      const fractions = outputs.fractions.map(fraction => peek(fraction)?.[index] ?? 0);
      const residual = peek(outputs.residual)?.[index] ?? 0;
      outputs.inset.write(Float32Array.from([...fractions, Math.min(1, residual * 10)]));
      return {
        glyphs: [getCellBox(inputs, world, column, row)],
        readout: `water ${formatNumber(fractions[0])} | soil ${formatNumber(fractions[1])} | veg ${formatNumber(fractions[2])} | misfit ${residual.toFixed(3)}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => {
    const shown = Math.round(values.shown);
    return [
      {
        layer: 'raster',
        values: shown === 3 ? outputs.residual : outputs.fractions[shown],
        columns: inputs.columns,
        rows: inputs.rows,
        ramp: 'sequential',
        range: shown === 3 ? [0, 0.1] : [0, 1],
        smoothing: true
      },
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.inset,
        range: [0, 1],
        color: [1, 1, 1, 0.9],
        rectangle: INSET
      }
    ];
  }
});
