// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterFractionalZonalStatistics behind GPURasterFractionalCoverage: each polygon aggregates
// the raster cells it overlaps, weighted by the exact covered fraction of every cell
// (extensive quantity, so totals scale with coverage). The pointer probes a tract: the cells it
// overlaps are shaded by covered fraction over the grid, edge cells partial.

import {
  GPURasterFractionalZonalStatistics,
  makeGPURasterBand
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './fractional-zonal-statistics.parameters';
import {drawRaster} from './support/bands';
import {
  addFractionalCoverage,
  createGridLines,
  createZoneInputs,
  drawZones,
  getTractAt,
  getTractRing,
  readZoneCoverage,
  sampleField,
  uploadZoneMesh
} from './support/zones';
import {readFloat32, readUint32} from './support/labels';

const RANGES = {
  means: [0, 1],
  totals: [0, 0.1],
  coveredArea: [0, 0.05],
  validArea: [0, 0.05]
} as const;

const STATISTICS = ['means', 'totals', 'coveredArea', 'validArea'] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['gridWidth'],
  caption: 'A value raster and polygons in; coverage-weighted statistics per polygon out.',
  createInputs: ({random, world}) => createZoneInputs(random, world),
  build(builder, inputs, values) {
    const {features} = inputs;
    const result = addFractionalCoverage(builder, inputs, values['gridWidth']);
    const cellValues = builder.upload(
      'values',
      'float32',
      sampleField(inputs, result.columns, result.rows)
    );
    const zoneCount = features.featureCount;
    const output = {
      totals: builder.allocate('totals', 'float32', zoneCount),
      means: builder.allocate('means', 'float32', zoneCount),
      coveredArea: builder.allocate('covered-area', 'float32', zoneCount),
      validArea: builder.allocate('valid-area', 'float32', zoneCount),
      validity: builder.allocate('validity', 'uint32', zoneCount)
    };
    builder.graph.add(
      new GPURasterFractionalZonalStatistics({
        grid: result.grid,
        input: makeGPURasterBand(cellValues.view, result.grid, {id: 'values'}),
        zoneCount,
        coverage: result.output,
        quantity: 'extensive',
        output: {
          totals: output.totals.view,
          means: output.means.view,
          coveredArea: output.coveredArea.view,
          validArea: output.validArea.view,
          validity: output.validity.view
        }
      })
    );
    return {
      outputs: {
        ...output,
        cellValues,
        zoneIds: result.coverage.zoneIds,
        cellIds: result.coverage.cellIds,
        fractions: result.coverage.fractions,
        tupleCount: result.coverage.count,
        requiredCount: result.coverage.requiredCount,
        coverageOverflow: result.coverage.overflow,
        candidateOverflow: result.coverage.candidateOverflow,
        zoneValidity: result.coverage.zoneValidity,
        hover: builder.allocate('hover', 'float32', result.cellCount),
        grid: builder.upload(
          'grid-lines',
          'float32x4',
          createGridLines(inputs.world, result.columns, result.rows)
        ),
        cellArea: result.cellSize[0] * result.cellSize[1],
        columns: result.columns,
        rows: result.rows,
        mesh: uploadZoneMesh(builder, features)
      }
    };
  },
  // A rebuild clears the highlight buffer; rewrite it from the last probed tract.
  async afterRun({inputs, outputs}) {
    if (inputs.focus.tract < 0) return;
    const hover = new Float32Array(outputs.hover.length);
    const [
      zoneIds,
      cellIds,
      fractions,
      count,
      requiredCount,
      overflow,
      candidateOverflow,
      zoneValidity
    ] = await Promise.all([
      readUint32(outputs.zoneIds),
      readUint32(outputs.cellIds),
      readFloat32(outputs.fractions),
      readUint32(outputs.tupleCount),
      readUint32(outputs.requiredCount),
      readUint32(outputs.coverageOverflow),
      readUint32(outputs.candidateOverflow),
      readUint32(outputs.zoneValidity)
    ]);
    readZoneCoverage(
      hover,
      {
        zoneIds,
        cellIds,
        fractions,
        count,
        requiredCount,
        overflow,
        candidateOverflow,
        zoneValidity
      },
      inputs.focus.tract
    );
    outputs.hover.write(hover);
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a tract',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, peek}) => {
      const tract = getTractAt(inputs.features, pointer.x, pointer.y);
      inputs.focus.tract = tract;
      const hover = new Float32Array(outputs.hover.length);
      const summary = readZoneCoverage(
        hover,
        {
          zoneIds: peek(outputs.zoneIds),
          cellIds: peek(outputs.cellIds),
          fractions: peek(outputs.fractions),
          count: peek(outputs.tupleCount),
          requiredCount: peek(outputs.requiredCount),
          overflow: peek(outputs.coverageOverflow),
          candidateOverflow: peek(outputs.candidateOverflow),
          zoneValidity: peek(outputs.zoneValidity)
        },
        tract
      );
      outputs.hover.write(hover);
      if (tract < 0) return {readout: 'outside the tracts'};
      const means = peek(outputs.means);
      return {
        glyphs: [{shape: 'polyline', closed: true, points: getTractRing(inputs.features, tract)}],
        readout: `tract ${inputs.features.geoids[tract].slice(-6)} | ${summary.status !== 'complete' ? `coverage ${summary.status}` : `mean ${means && Number.isFinite(means[tract]) ? means[tract].toFixed(2) : 'unavailable'} | ${summary.area.toFixed(1)} cells`}`
      };
    }
  },
  draw: ({inputs, outputs, values, theme}) => {
    const statistic = STATISTICS[Math.round(values['statistic'])];
    return [
      drawRaster(outputs, outputs.cellValues, {context: true}),
      drawZones(inputs.features, outputs.mesh, outputs[statistic], {
        ramp: 'sequential',
        range: RANGES[statistic],
        alpha: 0.8
      }),
      {layer: 'lines', context: true, segments: outputs.grid, color: [...theme.ink, 0.25]},
      {layer: 'lines', context: true, segments: outputs.mesh.outline, color: [...theme.ink, 0.8]},
      drawRaster(outputs, outputs.hover, {ramp: 'sequential', range: [0, 1], fade: true})
    ];
  }
});
