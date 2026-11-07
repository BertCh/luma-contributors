// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {S2CaptureSpec, S2Uploads} from './s2-lens-run';

type StatisticsLensFormat = 'float32' | 'float32x2' | 'uint32';

/** Explicit synthetic input construction shared by the small statistics examples. */
export function createStatisticsLensInputs(element: string, uploads: S2Uploads) {
  const captures: Record<string, S2CaptureSpec> = {};
  let serial = 0;
  const input = <Format extends StatisticsLensFormat>(
    name: string,
    format: Format,
    length: number
  ) => {
    const components = format === 'float32x2' ? 2 : 1;
    const data =
      format === 'uint32' ? new Uint32Array(length) : new Float32Array(length * components);
    for (let index = 0; index < data.length; index++) {
      const row = index % 24;
      let value = Math.sin(index * 0.7) + index / 24;
      if (name.endsWith('mask') || name === 'pairState') value = 1;
      else if (name.endsWith('blocks')) value = Math.floor(index / 2);
      else if (name === 'regimes') value = index % 2;
      else if (name === 'origins') value = Math.floor(index / 6);
      else if (name === 'destinations') value = index % 6;
      else if (name === 'flows') value = 2 + (index % 5);
      else if (name.endsWith('featureOffsets') || name.endsWith('polygonOffsets')) value = index;
      else if (name.endsWith('ringOffsets')) value = index * 4;
      else if (name.endsWith('positions')) {
        const point = Math.floor(index / 2);
        value = index % 2 === 0 ? point % 6 : Math.floor(point / 6);
        if (element === 'GPUFuzzyContiguity') {
          const feature = Math.floor(point / 4);
          const corner = point % 4;
          value =
            index % 2 === 0
              ? [0, 1.05, 3, 4][feature] + Number(corner === 1 || corner === 2)
              : Number(corner >= 2);
        }
      } else if (name === 'predictors' || name === 'design' || name === 'instruments') {
        const columns =
          element === 'GPUSpatialPanelErrorMaximumLikelihood' ||
          element === 'GPUSpatialPanelLagMaximumLikelihood' ||
          element === 'GPUSeeminglyUnrelatedRegression'
            ? 1
            : length / 24;
        const observation = Math.floor(index / columns);
        const column = index % columns;
        value = column === 0 ? observation / 24 : Math.sin(observation * (column + 1));
        if (
          element === 'GPUSpatialPanelErrorMaximumLikelihood' ||
          element === 'GPUSpatialPanelLagMaximumLikelihood'
        )
          value = Math.sin(index * 0.7) + row / 24;
        if (element === 'GPUSeeminglyUnrelatedRegression') value = Math.sin(index * 0.7) + row / 24;
      } else if (name === 'response' || name === 'responses') {
        value = 2 + row / 12 + 0.3 * Math.sin(index * 1.7);
        if (element === 'GPUProbitRegression')
          value = Number(index % 5 < 2 + Math.floor(index / 12));
        if (element === 'GPUSeeminglyUnrelatedRegression') value += Math.sin(index * 0.7);
        if (
          element === 'GPUSpatialPanelErrorMaximumLikelihood' ||
          element === 'GPUSpatialPanelLagMaximumLikelihood'
        )
          value += 2 * Math.sin(index * 0.7);
      } else if (name === 'residuals') value = 0.3 * Math.sin(index * 1.7);
      data[index] = value;
    }
    const view = uploads.view(`statistics-input-${serial++}`, data, format, length);
    captures[`input-${name}`] = {view, role: 'input'};
    return view;
  };
  const output = <Format extends StatisticsLensFormat>(
    name: string,
    format: Format,
    length: number
  ) => {
    const data =
      format === 'uint32'
        ? new Uint32Array(length)
        : new Float32Array(length * (format === 'float32x2' ? 2 : 1));
    const view = uploads.view(`statistics-output-${serial++}`, data, format, length);
    captures[name] = {view, role: 'output'};
    return view;
  };
  const offsets = new Uint32Array(25);
  const neighbors: number[] = [];
  const values: number[] = [];
  for (let row = 0; row < 24; row++) {
    offsets[row] = neighbors.length;
    if (element === 'GPUInstrumentalVariablesCovariance') {
      neighbors.push(row);
      values.push(1);
    } else if (element !== 'GPUIslandAttachment' || row < 23) {
      const population = element === 'GPUIslandAttachment' ? 23 : 24;
      const selected = [1, 2, 3, 4]
        .map(offset => (row + offset) % population)
        .sort((left, right) => left - right);
      neighbors.push(...selected);
      values.push(0.25, 0.25, 0.25, 0.25);
    }
  }
  offsets[24] = neighbors.length;
  const weights = {
    offsets: uploads.view('statistics-weights-offsets', offsets, 'uint32'),
    neighbors: uploads.view('statistics-weights-neighbors', new Uint32Array(neighbors), 'uint32'),
    weights: uploads.view('statistics-weights-values', new Float32Array(values), 'float32')
  };
  return {input, output, weights, captures};
}
