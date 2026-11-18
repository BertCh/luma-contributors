// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ChartData} from '../chart-types';

/** Readback fields used by the selected-cell chart. */
type MissingObservationsStructureSummary = {
  morphology: Record<'dilate' | 'erode' | 'opening' | 'closing', {values: Float32Array}>;
  reconstruction: {values: Float32Array; converged: number};
};

/** Updates the scene's selected-cell chart using the latest operation summaries. */
export function updateMissingObservationsStructureChart(
  context: {setChart(id: string, chart: ChartData | null): void},
  selectedCell: number,
  julyValues: Float32Array,
  summary: MissingObservationsStructureSummary,
  maximumReflectance: number
): void {
  const sampleSeries = [
    ['measured July', julyValues[selectedCell]],
    ['dilate', summary.morphology.dilate.values[selectedCell]],
    ['erode', summary.morphology.erode.values[selectedCell]],
    ['opening', summary.morphology.opening.values[selectedCell]],
    ['closing', summary.morphology.closing.values[selectedCell]],
    ...(summary.reconstruction.converged
      ? [['reconstruction', summary.reconstruction.values[selectedCell]] as const]
      : [])
  ] as const;
  context.setChart('structureChart', {
    kind: 'bars',
    title: 'NIR reflectance at selected cell',
    labels: sampleSeries.map(([label]) => label),
    values: sampleSeries.map(([, value]) => (Number.isFinite(value) ? value : Number.NaN)),
    horizontal: true,
    xDomain: [0, maximumReflectance],
    formatX: value => value.toFixed(3),
    description:
      'The map and chart share the selected cell. Morphology and grayscale reconstruction operate on measured July NIR; bars are GPU outputs except the directly loaded July reference. The reconstruction bar is omitted until convergence, matching the map.'
  });
}
