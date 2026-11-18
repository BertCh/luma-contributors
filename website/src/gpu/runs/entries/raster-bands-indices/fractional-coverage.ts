// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterFractionalCoverage: for every polygon and every grid cell it touches, the exact share
// of the cell the polygon covers, as (zone, cell, fraction) tuples. The pointer probes a tract:
// the cells it touches are shaded by that fraction over the grid, full cells bright and edge
// cells partial.

import {defineOperationRun} from '../../define-run';
import {parameters} from './fractional-coverage.parameters';
import {drawRaster} from './support/bands';
import {
  addFractionalCoverage,
  createGridLines,
  createZoneInputs,
  getTractAt,
  getTractRing,
  readZoneCoverage,
  uploadZoneMesh
} from './support/zones';
import {readFloat32, readUint32} from './support/labels';

export default defineOperationRun({
  parameters,
  rebuildOn: ['gridWidth'],
  caption: 'Polygons and a grid in; one (zone, cell, covered fraction) tuple per overlap out.',
  createInputs: ({random, world}) => createZoneInputs(random, world),
  build(builder, inputs, values) {
    const {features, world} = inputs;
    const result = addFractionalCoverage(builder, inputs, values['gridWidth']);
    return {
      outputs: {
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
          createGridLines(world, result.columns, result.rows)
        ),
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
      return {
        glyphs: [{shape: 'polyline', closed: true, points: getTractRing(inputs.features, tract)}],
        readout: `tract ${inputs.features.geoids[tract].slice(-6)} | ${summary.status === 'complete' ? `${summary.cells} cells | ${summary.partial} partial | ${summary.area.toFixed(1)} area` : `coverage ${summary.status}`}`
      };
    }
  },
  draw: ({inputs, outputs, theme}) => [
    {layer: 'lines', context: true, segments: outputs.mesh.outline, color: [...theme.ink, 0.5]},
    {layer: 'lines', context: true, segments: outputs.grid, color: [...theme.ink, 0.25]},
    drawRaster(outputs, outputs.hover, {ramp: 'sequential', range: [0, 1], alpha: 0.9})
  ]
});
