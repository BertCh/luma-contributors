// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterSpectralUnmixing: every cell holds a four-band spectrum that is fitted to three reference
// materials (water, soil, vegetation); the operation estimates their fractions (abundances).
// A patch of an unknown fourth material does not fit any mixture, so it lights up in the residual.
// The three fractions of a cell are stored side by side, so one GPURasterSampling pass reads the
// wide abundance raster at the right column and lays the fractions out as three panels (water,
// soil, vegetation) of one raster, each panel the scene on the same 0 to 1 ramp; reference spectra are pure pixels of the scene.
// `constraint` is a construction prop: unconstrained fractions can go negative or exceed one,
// simplex fractions are nonnegative and sum to one. The pointer probes a cell (outlined in all
// three panels): its three fractions and the residual are bars, with the values in the readout.

import {
  getGPURasterSamplingParameterValues,
  GPURasterSampling,
  GPURasterSpectralUnmixing,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {sampleReflectance} from '../../../data/reflectance';
import {formatNumber} from '../../interaction';
import {createChartAxes, getChartRectangle, mergeText, type RunLabel} from '../../text';
import {parameters} from './spectral-unmixing.parameters';
import {getCellAt, getCellBox} from './support/gestures';
import {readFloat32} from './support/labels';

const CONSTRAINTS = ['unconstrained', 'nonnegative', 'simplex'] as const;
const UNKNOWN_MATERIAL = [0.25, 0.02, 0.01, 0.3];
const BAND_COUNT = 4;
const MEMBER_COUNT = 3;
const INSET = [0.03, 0.05, 0.3, 0.3] as const;

/**
 * Reference spectra taken from the scene itself, as pure-pixel endmembers: the darkest pixel
 * (water or shade), the pixel with the largest shortwave over near infrared (bare or burned ground) and the
 * pixel with the largest near over red infrared (vegetation). Order: dark, bare, vegetation.
 */
function pickEndmembers(spectra: ArrayLike<number>[], cellCount: number): number[][] {
  const pick = (score: (cell: number) => number) => {
    let best = 0;
    for (let cell = 1; cell < cellCount; cell++) if (score(cell) > score(best)) best = cell;
    return spectra.map(band => band[best]);
  };
  // Band order of `spectra`: green, red, near infrared, shortwave infrared.
  return [
    pick(cell => -(spectra[0][cell] + spectra[1][cell] + spectra[2][cell] + spectra[3][cell])),
    pick(cell => spectra[3][cell] - spectra[2][cell]),
    pick(cell => spectra[2][cell] - spectra[1][cell])
  ];
}

/** The three fractions of a scene cell, read from the triptych raster. */
function getCellFractions(
  inputs: {columns: number},
  triptych: ArrayLike<number>,
  cell: number
): number[] {
  const column = cell % inputs.columns;
  const row = Math.floor(cell / inputs.columns);
  return Array.from(
    {length: MEMBER_COUNT},
    (_, member) => triptych[row * inputs.columns * MEMBER_COUNT + member * inputs.columns + column]
  );
}

const PANEL_TITLES = ['water fraction', 'soil fraction', 'vegetation fraction'];

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
    return {
      columns,
      rows,
      cellCount: columns * rows,
      spectra,
      noise,
      endmembers: pickEndmembers(spectra, columns * rows),
      focus: {cell: -1}
    };
  },
  build(builder, inputs, values) {
    const {cellCount} = inputs;
    const spectra = builder.allocate('spectra', 'float32', cellCount * BAND_COUNT);
    const endmembers = builder.upload(
      'endmembers',
      'float32',
      Float32Array.from(inputs.endmembers.flat())
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
    // Panel m of the triptych raster, cell (column, row), reads abundance m of that scene cell.
    const panelWidth = inputs.columns;
    const positions = new Float32Array(cellCount * MEMBER_COUNT * 2);
    for (let row = 0; row < inputs.rows; row++) {
      for (let member = 0; member < MEMBER_COUNT; member++) {
        for (let column = 0; column < panelWidth; column++) {
          const target = row * panelWidth * MEMBER_COUNT + member * panelWidth + column;
          positions[target * 2] = column * MEMBER_COUNT + member + 0.5;
          positions[target * 2 + 1] = inputs.rows - row - 0.5;
        }
      }
    }
    const fractions = builder.allocate('fractions', 'float32', cellCount * MEMBER_COUNT);
    builder.graph.add(
      new GPURasterSampling({
        id: 'gather-fractions',
        grid: wideGrid,
        values: output.abundances.view,
        positions: builder.upload('positions', 'float32x2', positions).view,
        parameters: samplingParameters.view,
        output: {values: fractions.view}
      })
    );
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
    const fractions = await readFloat32(outputs.fractions);
    const residual = await readFloat32(outputs.residual);
    outputs.inset.write(
      Float32Array.from([
        ...getCellFractions(inputs, fractions, cell),
        Math.min(1, residual[cell] * 10)
      ])
    );
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a cell',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, values, world, peek}) => {
      // Three panels side by side show the same scene, each a third of the frame wide.
      const isTriptych = Math.round(values.shown) === 0;
      const panelWorld = isTriptych ? {width: world.width / MEMBER_COUNT, height: 1} : world;
      const {column, row, index} = getCellAt(inputs, panelWorld, {
        x: isTriptych ? pointer.x % panelWorld.width : pointer.x,
        y: pointer.y
      });
      inputs.focus.cell = index;
      const triptych = peek(outputs.fractions);
      const fractions = triptych
        ? getCellFractions(inputs, triptych, index)
        : new Array<number>(MEMBER_COUNT).fill(0);
      const residual = peek(outputs.residual)?.[index] ?? 0;
      outputs.inset.write(Float32Array.from([...fractions, Math.min(1, residual * 10)]));
      const box = getCellBox(inputs, panelWorld, column, row);
      return {
        glyphs:
          isTriptych && box.shape === 'box'
            ? PANEL_TITLES.map((_, member) => ({
                ...box,
                minimum: [box.minimum[0] + member * panelWorld.width, box.minimum[1]] as const,
                maximum: [box.maximum[0] + member * panelWorld.width, box.maximum[1]] as const
              }))
            : [box],
        readout: `water ${formatNumber(fractions[0])} | soil ${formatNumber(fractions[1])} | veg ${formatNumber(fractions[2])} | misfit ${residual.toFixed(3)}`
      };
    }
  },
  draw: ({inputs, outputs, values, world, theme}) => {
    const isTriptych = Math.round(values.shown) === 0;
    const largerIs = theme.background[0] < 0.5 ? 'bright' : 'dark';
    const panelWidth = world.width / MEMBER_COUNT;
    const barRectangle = getChartRectangle(world, INSET);
    const labels: RunLabel[] = isTriptych
      ? [
          ...PANEL_TITLES.map((text, member) => ({
            text,
            at: [(member + 0.5) * panelWidth, 0.97] as const,
            anchor: 'top' as const,
            role: 'title' as const
          })),
          {
            text: `${largerIs} = larger fraction, 0 to 1 in every panel`,
            at: [world.width - 0.02, 0.03] as const,
            anchor: 'bottom-right' as const,
            role: 'label' as const
          }
        ]
      : [
          {
            text: `misfit (residual), ${largerIs} = poor fit of the three materials, 0 to 0.1`,
            at: [world.width / 2, 0.97] as const,
            anchor: 'top' as const,
            role: 'title' as const
          }
        ];
    const bars = createChartAxes({
      rectangle: barRectangle,
      x: {categories: ['water', 'soil', 'veg', 'misfit']}
    });
    return [
      isTriptych
        ? {
            layer: 'raster',
            values: outputs.fractions,
            columns: inputs.columns * MEMBER_COUNT,
            rows: inputs.rows,
            ramp: 'sequential',
            range: [0, 1],
            smoothing: false
          }
        : {
            layer: 'raster',
            values: outputs.residual,
            columns: inputs.columns,
            rows: inputs.rows,
            ramp: 'sequential',
            range: [0, 0.1],
            smoothing: true
          },
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.inset,
        range: [0, 1],
        color: [...theme.ink, 0.9],
        rectangle: INSET
      },
      {layer: 'text', ...mergeText({labels}, bars)}
    ];
  }
});
