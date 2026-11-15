// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUAdaptiveBandwidthKernelDensity,
  GPUSpaceTimeKernelDensity,
  GPUSpaceTimeRelativeRisk,
  getGPUAdaptiveBandwidthKernelDensityParameterValues,
  getGPUSpaceTimeKernelDensityParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import {appendCatalogRow, makeCatalogTable} from './catalog-geojson';
import {withCatalogGpuSession} from './catalog-gpu-session';
import type {Inputs, Values} from './catalog-platform3-b-common';
import {
  finiteSpacetimeValue,
  makeSpacetimeGrid,
  makeSpacetimePointTable,
  readSpacetimePoints,
  type SpacetimePoints
} from './catalog-spacetime-common';

/** Slice placement follows the binary64 observed range; a zero-duration population has one slice. */
export function getSpacetimeDensityTimeSlices(
  points: SpacetimePoints,
  requestedSliceCount: number
) {
  const lastTime = Math.max(...points.absoluteTimes);
  const duration = (lastTime - points.timeOrigin) / points.timeScale;
  const sliceCount = duration > 0 ? requestedSliceCount : 1;
  // The contributor requires a positive interval, even when there is only one evaluated slice.
  const sliceInterval = sliceCount > 1 ? duration / (sliceCount - 1) : 1;
  const times = Array.from({length: sliceCount}, (_, slice) =>
    slice > 0 && slice === sliceCount - 1
      ? lastTime
      : points.timeOrigin + slice * sliceInterval * points.timeScale
  );
  return {duration, sliceCount, sliceInterval, times};
}

export async function runSpacetimeDensity(
  kind: 'density' | 'risk' | 'adaptive',
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const points = readSpacetimePoints(inputs['points'], values, kind !== 'adaptive');
  const {grid, bounds} = makeSpacetimeGrid(points, values);
  const requestedSliceCount = kind === 'adaptive' ? 1 : Number(values['sliceCount']);
  const slices = getSpacetimeDensityTimeSlices(points, requestedSliceCount);
  const {duration, sliceCount, sliceInterval} = slices;
  const voxelCount = grid.width * grid.height * sliceCount;
  const spatialBandwidth = Number(values['spatialBandwidth']);
  const temporalBandwidth = Number(values['temporalBandwidth'] ?? 0);
  const caseIndices: number[] = [];
  const controlIndices: number[] = [];
  if (kind === 'risk') {
    points.collection.features.forEach((feature, index) => {
      const indicator = feature.properties?.[String(values['caseProperty'])];
      if (indicator !== 0 && indicator !== 1)
        throw new Error('Space-time risk requires binary 0/1 case indicator');
      (indicator === 1 ? caseIndices : controlIndices).push(index);
    });
    if (!caseIndices.length || !controlIndices.length)
      throw new Error('Space-time risk needs both cases and controls');
  }
  const result: {
    density: Float32Array;
    bandwidths?: Float32Array;
    pilot?: Float32Array;
    caseDensity?: Float32Array;
    controlDensity?: Float32Array;
  } = await withCatalogGpuSession(context.device, `catalog-spacetime-${kind}`, async session => {
    const output = session.output('values', 'float32', voxelCount);
    const common = {grid, indexGridSize: [4, 4] as const, indexBounds: bounds};
    if (kind === 'adaptive') {
      const pointBandwidths = session.output('bandwidths', 'float32', points.count);
      const pilot = session.output('pilot', 'float32', points.count);
      session.graph.add(
        new GPUAdaptiveBandwidthKernelDensity({
          ...common,
          positions: session.input('positions', points.positions, 'float32x2'),
          parameters: session.input(
            'parameters',
            getGPUAdaptiveBandwidthKernelDensityParameterValues(
              {bandwidth: spatialBandwidth, sensitivity: Number(values['sensitivity'])},
              grid
            ),
            'float32'
          ),
          bandwidthFactorLimit: Number(values['factorLimit']),
          normalization: 'intensity',
          output: {
            values: output.view,
            pointBandwidths: pointBandwidths.view,
            pilotDensities: pilot.view
          }
        })
      );
      session.run();
      return {
        density: await session.readFloat32(output),
        bandwidths: await session.readFloat32(pointBandwidths),
        pilot: await session.readFloat32(pilot)
      };
    }
    const parameters = session.input(
      'parameters',
      getGPUSpaceTimeKernelDensityParameterValues(
        {
          bandwidth: spatialBandwidth,
          temporalBandwidth,
          firstSliceTime: 0,
          sliceInterval,
          timeWindow: [-1, duration + 1]
        },
        grid,
        sliceCount
      ),
      'float32'
    );
    const temporal = {
      ...common,
      parameters,
      sliceCount,
      spatialEdgeCorrection: 'none' as const,
      temporalEdgeCorrection: 'none' as const
    };
    const caseDensity =
      kind === 'risk' ? session.output('case-density', 'float32', voxelCount) : undefined;
    const controlDensity =
      kind === 'risk' ? session.output('control-density', 'float32', voxelCount) : undefined;
    if (kind === 'risk') {
      const pickPositions = (indices: number[]) =>
        Float32Array.from(
          indices.flatMap(index => [points.positions[index * 2], points.positions[index * 2 + 1]])
        );
      session.graph.add(
        new GPUSpaceTimeRelativeRisk({
          ...temporal,
          casePositions: session.input('case-positions', pickPositions(caseIndices), 'float32x2'),
          controlPositions: session.input(
            'control-positions',
            pickPositions(controlIndices),
            'float32x2'
          ),
          caseTimes: session.input(
            'case-times',
            Float32Array.from(caseIndices.map(index => points.times[index])),
            'float32'
          ),
          controlTimes: session.input(
            'control-times',
            Float32Array.from(controlIndices.map(index => points.times[index])),
            'float32'
          ),
          scale: 'ratio',
          output: {
            values: output.view,
            caseDensity: caseDensity!.view,
            controlDensity: controlDensity!.view
          }
        })
      );
    } else {
      session.graph.add(
        new GPUSpaceTimeKernelDensity({
          ...temporal,
          positions: session.input('positions', points.positions, 'float32x2'),
          times: session.input('times', points.times, 'float32'),
          normalization: 'intensity',
          output: {values: output.view}
        })
      );
    }
    session.run();
    return {
      density: await session.readFloat32(output),
      ...(caseDensity && controlDensity
        ? {
            caseDensity: await session.readFloat32(caseDensity),
            controlDensity: await session.readFloat32(controlDensity)
          }
        : {})
    };
  });
  const gridTable = makeCatalogTable(
    'grid',
    'Spacetime density grid',
    {
      voxel_index: 'int32',
      cell_index: 'int32',
      slice_index: 'int32',
      time: 'float64',
      value: 'float64',
      case_density: 'float64',
      control_density: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  for (let cell = 0; cell < grid.width * grid.height; cell++) {
    const column = cell % grid.width;
    const row = Math.floor(cell / grid.width);
    const position = points.frame.inverse(
      bounds[0] + ((column + 0.5) * (bounds[2] - bounds[0])) / grid.width,
      bounds[1] + ((row + 0.5) * (bounds[3] - bounds[1])) / grid.height
    );
    for (let slice = 0; slice < sliceCount; slice++) {
      const voxel = cell * sliceCount + slice;
      appendCatalogRow(gridTable, {
        voxel_index: voxel,
        cell_index: cell,
        slice_index: slice,
        time: slices.times[slice],
        value: finiteSpacetimeValue(result.density[voxel]),
        case_density: result.caseDensity ? finiteSpacetimeValue(result.caseDensity[voxel]) : null,
        control_density: result.controlDensity
          ? finiteSpacetimeValue(result.controlDensity[voxel])
          : null,
        geometry: JSON.stringify({type: 'Point', coordinates: position})
      });
    }
  }
  const contributor =
    kind === 'adaptive'
      ? 'GPUAdaptiveBandwidthKernelDensity'
      : kind === 'risk'
        ? 'GPUSpaceTimeRelativeRisk'
        : 'GPUSpaceTimeKernelDensity';
  return {
    tables: [
      gridTable,
      ...(result.bandwidths && result.pilot
        ? [
            makeSpacetimePointTable(points, {
              bandwidth: Array.from(result.bandwidths, finiteSpacetimeValue),
              pilot_density: Array.from(result.pilot, finiteSpacetimeValue)
            })
          ]
        : [])
    ],
    diagnostics: [],
    dispatchedContributors: [contributor],
    summary: {
      featureCount: points.count,
      voxelCount,
      spatialBandwidth,
      temporalBandwidth,
      timeOrigin: points.timeOrigin,
      timeUnit: String(values['timeUnit']),
      sliceIntervalSeconds: sliceCount > 1 ? sliceInterval : 0,
      requestedSliceCount,
      sliceCount,
      caseCount: caseIndices.length,
      controlCount: controlIndices.length,
      factorLimit: Number(values['factorLimit'] ?? 1),
      sensitivity: Number(values['sensitivity'] ?? 0),
      maximumFeatures: 512,
      maximumVoxels: 16384,
      edgeCorrection: kind === 'adaptive' ? 'adaptive contributor default' : 'none'
    }
  };
}
