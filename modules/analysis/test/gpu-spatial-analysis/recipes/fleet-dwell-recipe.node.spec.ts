// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  addFleetDwellRecipe,
  addFleetDwellZoneEventsRecipe,
  type GPUFleetDwellRecipeProps
} from '../../../src/gpu-spatial-analysis/recipes/fleet-dwell-recipe';
import {GPUTrajectoryMetrics} from '../../../src/gpu-spatial-analysis/trajectory-analysis/index';
import {GPUZoneEvents} from '../../../src/gpu-spatial-analysis/trajectory-zones/index';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  findNonFiniteBitcasts,
  findReservedIdentifiers
} from '../../../../../examples/deck/spatial-analysis-showcase/scripts/check-wgsl-identifiers.mjs';

function createProps(graph: GPUCommandGraph): GPUFleetDwellRecipeProps {
  return {
    trajectories: {
      kind: 'trajectories',
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      timestamps: createTransientView(graph, 'timestamps', 'float32', 2),
      trackOffsets: createTransientView(graph, 'track-offsets', 'uint32', 2)
    },
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    stopCapacity: 64,
    zones: {
      kind: 'polygons',
      positions: createTransientView(graph, 'zone-positions', 'float32x2', 4),
      featureOffsets: createTransientView(graph, 'feature-offsets', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygon-offsets', 'uint32', 2),
      ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 2),
      candidateCapacity: 1
    }
  };
}

it('forwards the complete declared double-single time column into stop metrics', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    props.trajectories.timestampsLow = createTransientView(graph, 'timestamps-low', 'float32', 2);
    props.trajectories.temporalContext = {
      encoding: 'double-single',
      unit: 'second',
      epoch: 0n,
      axis: {kind: 'irregular'}
    };
    const result = addFleetDwellRecipe(graph, props);
    const metrics = result.contributors.find(
      contributor => contributor instanceof GPUTrajectoryMetrics
    );
    expect(metrics).toBeInstanceOf(GPUTrajectoryMetrics);
    // Recipe configuration is captured before declaration; its values, rather than the
    // caller's mutable metadata object identity, must reach the contributor.
    expect((metrics as GPUTrajectoryMetrics).props.temporalContext).toEqual(
      props.trajectories.temporalContext
    );
    expect((metrics as GPUTrajectoryMetrics).props.timestampsLow).toBe(
      props.trajectories.timestampsLow
    );
  } finally {
    device.destroy();
  }
});

it.each([
  {name: 'stops', recipe: addFleetDwellRecipe},
  {name: 'zone events', recipe: addFleetDwellZoneEventsRecipe}
])('$name rejects a mismatched time declaration before adding nodes', ({recipe}) => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    props.trajectories.temporalContext = {
      encoding: 'int64-words',
      unit: 'millisecond',
      epoch: 'unix',
      axis: {kind: 'irregular'}
    };
    const add = vi.spyOn(graph, 'add');
    expect(() =>
      recipe(graph, {...props, candidateCapacity: 4, trackEventCapacity: 4} as never)
    ).toThrow(/does not support int64-words times/);
    expect(add).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('zone events rejects a low time part and forwards a supported declaration', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    props.trajectories.timestampsLow = createTransientView(graph, 'timestamps-low', 'float32', 2);
    const add = vi.spyOn(graph, 'add');
    expect(() =>
      addFleetDwellZoneEventsRecipe(graph, {
        ...props,
        candidateCapacity: 4,
        trackEventCapacity: 4
      } as never)
    ).toThrow(/timestampsLow/);
    expect(add).not.toHaveBeenCalled();
    delete props.trajectories.timestampsLow;
    props.trajectories.temporalContext = {
      encoding: 'float32-relative',
      unit: 'second',
      epoch: 0n,
      axis: {kind: 'irregular'}
    };
    const result = addFleetDwellZoneEventsRecipe(graph, {
      zones: props.zones,
      trajectories: {
        kind: 'trajectories',
        positions: props.trajectories.positions,
        timestamps: props.trajectories.timestamps,
        trackOffsets: props.trajectories.trackOffsets,
        temporalContext: {...props.trajectories.temporalContext, encoding: 'float32-relative'}
      },
      candidateCapacity: 4,
      trackEventCapacity: 4
    });
    const events = result.contributors.find(contributor => contributor instanceof GPUZoneEvents);
    expect((events as GPUZoneEvents).props.tracks.temporalContext).toEqual(
      props.trajectories.temporalContext
    );
  } finally {
    device.destroy();
  }
});

it('excludes spare stop slots before the zone join generates candidates', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const result = addFleetDwellRecipe(graph, createProps(graph));
    const join = result.contributors.find(
      contributor => contributor instanceof GPUPointInPolygonJoin
    );
    expect(join).toBeInstanceOf(GPUPointInPolygonJoin);
    expect((join as GPUPointInPolygonJoin).props.points.positions).not.toBe(result.stops.centroids);
    const joinIndex = result.contributors.indexOf(join!);
    const adapter = result.contributors[joinIndex - 1].getCommandNodes(graph);
    expect(adapter).toHaveLength(1);
    expect(adapter[0].id).toBe('fleet-dwell-stop-mask-kernel');
  } finally {
    device.destroy();
  }
});

it.each([
  {name: 'reserved identifiers', inspectSource: findReservedIdentifiers},
  {name: 'constant nonfinite bitcasts', inspectSource: findNonFiniteBitcasts}
])('the stop adapter generates WGSL without $name', ({inspectSource}) => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const result = addFleetDwellRecipe(graph, createProps(graph));
    const joinIndex = result.contributors.findIndex(
      contributor => contributor instanceof GPUPointInPolygonJoin
    );
    expect(joinIndex).toBeGreaterThan(0);
    const adapterNodes = result.contributors[joinIndex - 1].getCommandNodes(graph);
    const kernels = getGPUCommandNodeKernelSources(adapterNodes);
    expect(kernels).toHaveLength(1);
    for (const kernel of kernels) {
      expect(inspectSource(kernel.source), kernel.id).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});
