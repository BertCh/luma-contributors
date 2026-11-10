// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterJoin behind GPUPolygonRasterization: every point looks up the zone of its raster cell
// and job counts are summed per tract. The boundary mask identifies assignments sensitive
// to the grid resolution. The pointer probes a tract: its points are enlarged and counted.

import {GPURasterJoin} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {getWorkplaceRows} from '../../../data/workplace-rows';
import {createTractFeatures} from './support/tract-zones';
import {parameters} from './raster-join.parameters';
import {addZoneRaster, drawZones, getTractAt, getTractRing, uploadZoneMesh} from './support/zones';

export default defineOperationRun({
  parameters,
  rebuildOn: ['gridWidth'],
  caption:
    'Jobs by Chicago tract on a log scale. Teal dots mark workplace blocks in boundary cells.',
  createInputs: ({random, world}) => {
    const {tracts, positions, jobs, keys} = getWorkplaceRows(random, world);
    return {world, features: createTractFeatures(tracts), positions, jobs, keys};
  },
  build(builder, inputs, values) {
    const {features, positions, jobs} = inputs;
    const zoneRaster = addZoneRaster(builder, inputs, values.gridWidth);
    const pointPositions = builder.upload('points', 'float32x2', positions);
    const pointValues = builder.upload('point-values', 'float32', jobs);
    const zoneCount = features.featureCount;
    const sums = builder.allocate('sums', 'float32', zoneCount);
    const boundaryMask = builder.allocate('boundary-mask', 'uint32', jobs.length);
    builder.graph.add(
      new GPURasterJoin({
        grid: zoneRaster.grid,
        parameters: zoneRaster.extent.view,
        points: pointPositions.view,
        values: pointValues.view,
        zones: zoneRaster.zones.view,
        boundary: zoneRaster.boundary.view,
        zoneCount,
        output: {sums: sums.view, pointBoundaryMask: boundaryMask.view}
      })
    );
    const hoverPoints = builder.allocate('hover-points', 'float32x2', jobs.length);
    return {
      outputs: {
        sums,
        boundaryMask,
        pointPositions,
        hoverPoints,
        mesh: uploadZoneMesh(builder, features)
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a tract',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, peek}) => {
      const tract = getTractAt(inputs.features, pointer.x, pointer.y);
      if (tract < 0) return {focus: 0, readout: 'outside the tracts'};
      const boundaryMask = peek(outputs.boundaryMask);
      const sums = peek(outputs.sums);
      const points: number[] = [];
      let edge = 0;
      inputs.keys.forEach((key, row) => {
        if (key !== tract) return;
        points.push(inputs.positions[row * 2], inputs.positions[row * 2 + 1]);
        if (boundaryMask?.[row]) edge++;
      });
      outputs.hoverPoints.write(
        Float32Array.from({length: inputs.jobs.length * 2}, (_, index) => points[index] ?? 0)
      );
      return {
        focus: points.length / 2,
        glyphs: [{shape: 'polyline', closed: true, points: getTractRing(inputs.features, tract)}],
        readout: `tract ${inputs.features.geoids[tract].slice(-6)} | ${sums ? Math.round(sums[tract]).toLocaleString() : '-'} jobs | ${edge} boundary pts`
      };
    }
  },
  draw: ({inputs, outputs, interaction, theme}) => [
    drawZones(inputs.features, outputs.mesh, outputs.sums, {
      ramp: 'sequential',
      logarithmic: true,
      range: [0, 19],
      alpha: 0.85
    }),
    {
      layer: 'dots',
      positions: outputs.pointPositions,
      values: outputs.boundaryMask,
      radius: 2.3,
      ramp: 'categorical',
      categories: [theme.muted, [0, 0.85, 0.8]]
    },
    {layer: 'lines', context: true, segments: outputs.mesh.outline, color: [...theme.ink, 0.7]},
    {
      layer: 'dots',
      positions: outputs.hoverPoints,
      count: typeof interaction.focus === 'number' ? interaction.focus : 0,
      radius: 4.5,
      color: [1, 1, 1, 1]
    }
  ]
});
