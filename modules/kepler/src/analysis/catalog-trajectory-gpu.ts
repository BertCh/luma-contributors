// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUTrajectoryHampel,
  GPUTrajectoryOutlierFlags,
  GPUTrajectoryTrackRelativeTimes,
  GPUUtilizationVolumeIsopleths,
  getGPUTrajectoryHampelParameterValues,
  getGPUTrajectoryOutlierFlagsParameterValues,
  getGPUUtilizationVolumeIsoplethsParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogGpuContext} from './catalog-binding-types';
import type {CatalogPosition} from './catalog-geojson';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {makeTrajectoryHomeRange, prepareTrajectoryHomeRange} from './catalog-trajectory-cpu';
import {
  makeTrajectoryFixes,
  makeTrajectoryTracks,
  makeTrajectoryOutput,
  getTrajectoryTrackRows,
  type CatalogTrajectory,
  type TrajectoryValues
} from './catalog-trajectory-common';

export async function runTrajectoryHampel(
  context: CatalogGpuContext,
  trajectory: CatalogTrajectory,
  values: TrajectoryValues
) {
  return withCatalogGpuSession(context.device, 'catalog-trajectory-hampel', async session => {
    const target = values['target'];
    const input = Float32Array.from(trajectory.fixes, (fix, row) => {
      const value =
        target === 'x'
          ? trajectory.positions[row * 2]
          : target === 'y'
            ? trajectory.positions[row * 2 + 1]
            : fix.properties[values['valueProperty'] as string];
      if (
        typeof value !== 'number' ||
        !Number.isFinite(value) ||
        !Number.isFinite(Math.fround(value))
      )
        throw new Error('Hampel requires a finite Float32 scalar on every fix.');
      return value;
    });
    const filtered = session.output('filtered', 'float32', input.length);
    const flags = session.output('flags', 'uint32', input.length);
    session.graph.add(
      new GPUTrajectoryHampel({
        values: session.input('values', input, 'float32'),
        trackOffsets: session.input('tracks', trajectory.trackOffsets, 'uint32'),
        windowSize: values['windowSize'] as number,
        parameters: session.input(
          'parameters',
          getGPUTrajectoryHampelParameterValues({
            numberOfSigmas: values['numberOfSigmas'] as number
          }),
          'float32'
        ),
        output: {filtered: filtered.view, flags: flags.view}
      })
    );
    session.run();
    const filteredValues = await session.readFloat32(filtered);
    const flagValues = await session.readUint32(flags);
    const positions: CatalogPosition[] = trajectory.fixes.map((fix, row) =>
      target === 'property'
        ? fix.position
        : trajectory.frame.inverse(
            target === 'x' ? filteredValues[row] : trajectory.positions[row * 2],
            target === 'y' ? filteredValues[row] : trajectory.positions[row * 2 + 1]
          )
    );
    return {
      ...makeTrajectoryOutput(
        trajectory,
        [
          makeTrajectoryFixes(
            trajectory,
            {filtered_value: 'float64', outlier: 'boolean'},
            undefined,
            row => ({filtered_value: filteredValues[row], outlier: Boolean(flagValues[row])}),
            positions
          ),
          makeTrajectoryTracks(trajectory, undefined, positions)
        ],
        {outlierCount: flagValues.reduce((sum, flag) => sum + flag, 0)}
      ),
      dispatchedContributors: ['GPUTrajectoryHampel']
    };
  });
}

export async function runTrajectoryOutlierFlags(
  context: CatalogGpuContext,
  trajectory: CatalogTrajectory,
  values: TrajectoryValues
) {
  return withCatalogGpuSession(context.device, 'catalog-trajectory-outliers', async session => {
    const flags = session.output('flags', 'uint32', trajectory.fixes.length);
    session.graph.add(
      new GPUTrajectoryOutlierFlags({
        positions: session.input('positions', trajectory.positions, 'float32x2'),
        timestamps: session.input('timestamps', trajectory.timestamps, 'uint32x2'),
        trackOffsets: session.input('tracks', trajectory.trackOffsets, 'uint32'),
        parameters: session.input(
          'parameters',
          getGPUTrajectoryOutlierFlagsParameterValues({
            maximumSpeed: (values['maximumSpeed'] as number) / 1000,
            maximumAcceleration: (values['maximumAcceleration'] as number) / 1e6
          }),
          'float32'
        ),
        output: {flags: flags.view}
      })
    );
    session.run();
    const flagValues = await session.readUint32(flags);
    const groups = getTrajectoryTrackRows(trajectory).map(group =>
      group.filter(row => flagValues[row] === 0)
    );
    return {
      ...makeTrajectoryOutput(
        trajectory,
        [
          makeTrajectoryFixes(trajectory, {outlier_flags: 'int32'}, undefined, row => ({
            outlier_flags: flagValues[row]
          })),
          makeTrajectoryTracks(trajectory, groups)
        ],
        {outlierCount: flagValues.filter(flag => flag !== 0).length}
      ),
      dispatchedContributors: ['GPUTrajectoryOutlierFlags']
    };
  });
}

export async function runTrajectoryRelativeTimes(
  context: CatalogGpuContext,
  trajectory: CatalogTrajectory,
  values: TrajectoryValues
) {
  return withCatalogGpuSession(
    context.device,
    'catalog-trajectory-relative-times',
    async session => {
      const relative = session.output('relative', 'float32', trajectory.fixes.length);
      session.graph.add(
        new GPUTrajectoryTrackRelativeTimes({
          timestamps: session.input('timestamps', trajectory.timestamps, 'uint32x2'),
          trackOffsets: session.input('tracks', trajectory.trackOffsets, 'uint32'),
          output: relative.view
        })
      );
      session.run();
      const times = await session.readFloat32(relative);
      const maximumError = values['maximumTimeErrorMilliseconds'] as number;
      trajectory.fixes.forEach((fix, row) => {
        const first = trajectory.fixes[trajectory.trackOffsets[fix.trackIndex]];
        if (Math.abs(times[row] - (fix.timestamp - first.timestamp)) > maximumError)
          throw new Error(
            'Trajectory relative time exceeds maximumTimeErrorMilliseconds; no rounded result published.'
          );
      });
      return {
        ...makeTrajectoryOutput(trajectory, [
          makeTrajectoryFixes(trajectory, {relative_milliseconds: 'float64'}, undefined, row => ({
            relative_milliseconds: times[row]
          })),
          makeTrajectoryTracks(trajectory)
        ]),
        dispatchedContributors: ['GPUTrajectoryTrackRelativeTimes']
      };
    }
  );
}

export async function runTrajectoryIsopleths(
  context: CatalogGpuContext,
  trajectory: CatalogTrajectory,
  values: TrajectoryValues
) {
  const grid = prepareTrajectoryHomeRange(trajectory, values);
  return withCatalogGpuSession(context.device, 'catalog-trajectory-isopleths', async session => {
    const volume = session.output('volume', 'float32', grid.density.length);
    const inside = session.output('inside', 'uint32', grid.density.length);
    session.graph.add(
      new GPUUtilizationVolumeIsopleths({
        density: session.input('density', grid.density, 'float32'),
        parameters: session.input(
          'parameters',
          getGPUUtilizationVolumeIsoplethsParameterValues({level: values['level'] as number}),
          'float32'
        ),
        output: {volume: volume.view, inside: inside.view}
      })
    );
    session.run();
    return {
      ...makeTrajectoryHomeRange(
        trajectory,
        grid,
        await session.readFloat32(volume),
        await session.readUint32(inside)
      ),
      dispatchedContributors: ['GPUUtilizationVolumeIsopleths']
    };
  });
}
