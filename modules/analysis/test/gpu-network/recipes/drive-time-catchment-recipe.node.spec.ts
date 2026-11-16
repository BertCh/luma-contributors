// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkSnapping,
  GPU_NETWORK_SNAPPING_MAXIMUM_UNINDEXED_WORK
} from '../../../src/gpu-network/network-accessibility/index';
import {createGPULocalFrame} from '../../../src/analysis-core/contracts/local-frame';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {addDriveTimeCatchmentRecipe} from '../../../src/gpu-network/recipes/drive-time-catchment-recipe';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {GPUNetworkIsochrones} from '../../../src/gpu-network/network-isochrones/index';
import {buildCSR, snapOracle} from '../network-accessibility/network-accessibility-oracle';
import {getDriveTimeOracle} from './drive-time-oracle';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

let serial = 0;

function buildRecipe(spatialContext: GPUSpatialContext | undefined, candidateCapacity?: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `drive-time-${serial++}`});
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `drive-time-view-${serial++}`, format, length);
  const recipe = addDriveTimeCatchmentRecipe(graph, {
    spatialContext,
    topology: {
      offsets: view('uint32', 5),
      neighbors: view('uint32', 8),
      weights: view('float32', 8),
      nodePositions: view('float32x2', 4)
    },
    facilities: view('float32x2', 2),
    demand: view('float32x2', 6),
    maximumSnapDistance: view('float32', 1),
    candidateCapacity,
    costBreaks: view('float32', 3)
  });
  const snappings = recipe.contributors.filter(
    (contributor): contributor is GPUNetworkSnapping => contributor instanceof GPUNetworkSnapping
  );
  return {recipe, snappings};
}

it('drive-time recipe snaps longitude/latitude positions great-circle, in meters', () => {
  // The explorer reported raw-degree snapping: Euclidean on lon/lat, about a 24% east-west bias at
  // its latitude. Either lon/lat metric now snaps great-circle; planar stays planar.
  for (const metric of ['none', 'great-circle'] as const) {
    const {recipe, snappings} = buildRecipe({coordinateSpace: 'longitude-latitude', metric});
    expect(recipe.spatialContext.coordinateSpace).toBe('longitude-latitude');
    expect(snappings).toHaveLength(2);
    for (const snapping of snappings) {
      expect(snapping.spatialContext).toMatchObject({
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle'
      });
    }
  }
  const {snappings: planar} = buildRecipe(undefined);
  for (const snapping of planar) {
    expect(snapping.spatialContext.coordinateSpace).toBe('planar');
  }
  // A great-circle context keeps its sphere radius.
  const {snappings: custom} = buildRecipe({
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    sphereRadius: 6378137
  });
  expect(custom[0].spatialContext.sphereRadius).toBe(6378137);
  expect(() => buildRecipe({coordinateSpace: 'longitude-latitude', metric: 'rhumb'})).toThrow(
    /does not support the rhumb metric/
  );
});

it('drive-time recipe reports BVH snapping overflow per stage', () => {
  // Review B-08: both snappings took candidateCapacity but no overflow view, so a candidate
  // overflow mis-snapped facilities or demand with no signal.
  const {recipe, snappings} = buildRecipe(undefined, 16);
  expect(recipe.snapping).toBeDefined();
  const [facilities, demand] = snappings;
  expect(facilities.props.output.status?.overflow).toBe(recipe.snapping!.status.facilitiesOverflow);
  expect(demand.props.output.status?.overflow).toBe(recipe.snapping!.status.demandOverflow);
  expect(recipe.snapping!.status.facilitiesOverflow).not.toBe(
    recipe.snapping!.status.demandOverflow
  );
  const stages = Object.fromEntries(recipe.status.stages.map(stage => [stage.stage, stage.status]));
  expect(stages['snap-facilities']).toEqual({
    overflow: recipe.snapping!.status.facilitiesOverflow
  });
  expect(stages['snap-demand']).toEqual({overflow: recipe.snapping!.status.demandOverflow});

  // The exact all-edges scan cannot overflow, so it reports no snapping stage.
  const exact = buildRecipe(undefined);
  expect(exact.recipe.snapping).toBeUndefined();
  expect(exact.recipe.status.stages.map(stage => stage.stage)).not.toContain('snap-demand');
});

it('drive-time recipe feeds one topology to both snappings', () => {
  // Each snapping expands its own edge sources from the topology; the recipe keeps no private
  // edge-source input on the snapping props.
  const {snappings} = buildRecipe(undefined);
  const [facilities, demand] = snappings;
  expect(facilities.props.topology).toBe(demand.props.topology);
  expect(facilities.edgeCount).toBe(8);
  expect(facilities.nodeCount).toBe(4);
});

it('drive-time recipe validates the topology', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `drive-time-${serial++}`});
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `drive-time-view-${serial++}`, format, length);
  expect(() =>
    addDriveTimeCatchmentRecipe(graph, {
      topology: {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 8),
        weights: view('float32', 8),
        nodePositions: view('float32x2', 3)
      },
      facilities: view('float32x2', 2),
      demand: view('float32x2', 6),
      costBreaks: view('float32', 3)
    })
  ).toThrow(/topology.nodePositions must contain one row per node/);
});

it('drive-time recipe accepts distance frames and rejects equal-area frames', () => {
  const equidistant = createGPULocalFrame({origin: [10, 60]}).spatialContext;
  const {recipe, snappings} = buildRecipe(equidistant);
  expect(recipe.spatialContext).toBe(equidistant);
  expect(equidistant.units).toBe('meters');
  for (const snapping of snappings) {
    expect(snapping.spatialContext).toBe(equidistant);
  }
  const equalArea = createGPULocalFrame({
    origin: [10, 60],
    kind: 'lambert-equal-area'
  }).spatialContext;
  expect(() => buildRecipe(equalArea)).toThrow(
    'drive-time-catchment measures distance; project into an azimuthal-equidistant frame, not lambert-equal-area'
  );
});

it('drive-time recipe uses kilometer radii for both longitude/latitude metrics', () => {
  for (const metric of ['none', 'great-circle'] as const) {
    const context: GPUSpatialContext = {
      coordinateSpace: 'longitude-latitude',
      metric,
      units: 'kilometers'
    };
    const {recipe, snappings} = buildRecipe(context, 4);
    expect(recipe.spatialContext).toBe(context);
    for (const snapping of snappings) {
      expect(snapping.spatialContext).toMatchObject({
        metric: 'great-circle',
        units: 'kilometers',
        sphereRadius: 6371.0088
      });
    }
  }
  const {snappings} = buildRecipe({
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'kilometers',
    sphereRadius: 6400
  });
  expect(snappings[0].spatialContext.sphereRadius).toBe(6400);
});

it('drive-time recipe still schedules the exact scan above the unindexed snapping limit', () => {
  // 2^24 demand points on 16 edges are 2^28 point-edge tests, above the 2^27 a direct
  // GPUNetworkSnapping accepts without candidateCapacity. This built before the limit existed.
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'drive-time-large'});
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `drive-time-large-${serial++}`, format, length);
  const recipe = addDriveTimeCatchmentRecipe(graph, {
    topology: {
      offsets: view('uint32', 5),
      neighbors: view('uint32', 16),
      weights: view('float32', 16),
      nodePositions: view('float32x2', 4)
    },
    facilities: view('float32x2', 2),
    demand: view('float32x2', 2 ** 24),
    costBreaks: view('float32', 3)
  });
  const snappings = recipe.contributors.filter(
    (contributor): contributor is GPUNetworkSnapping => contributor instanceof GPUNetworkSnapping
  );
  expect(snappings).toHaveLength(2);
  expect(snappings[1].pointCount * snappings[1].edgeCount).toBeGreaterThan(
    GPU_NETWORK_SNAPPING_MAXIMUM_UNINDEXED_WORK
  );
});

it('drive-time oracle preserves zero-cost junction access independently of one-way restrictions', () => {
  for (const fixture of [
    {
      positions: [0, 0, 10, 0, 0, 10],
      edges: [
        [0, 1, 10],
        [0, 2, 10]
      ],
      facilities: [0, 0],
      demand: [0, 5, 0, 0],
      expected: [5, 0]
    },
    {
      positions: [10, 0, 0, 0, 0, 10],
      edges: [
        [0, 1, 10],
        [2, 1, 10]
      ],
      facilities: [0, 5],
      demand: [0, 0, 5, 0],
      expected: [5, Infinity]
    }
  ]) {
    const topology = buildCSR(3, fixture.edges as [number, number, number][]);
    const options = {edgeCosts: topology.weights};
    const facilities = snapOracle(
      fixture.facilities,
      fixture.positions,
      topology.sources,
      topology.neighbors,
      options
    );
    const demand = snapOracle(
      fixture.demand,
      fixture.positions,
      topology.sources,
      topology.neighbors,
      options
    );
    expect(getDriveTimeOracle(topology, 3, facilities, demand).demandTimes).toEqual(
      fixture.expected
    );
  }
});

it.each([
  false,
  true
])('drive-time raster source paths pass shader validation (labelled=%s)', labelled => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'drive-time-source-raster'});
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `source-raster-${serial++}`, format, length);
  try {
    const recipe = addDriveTimeCatchmentRecipe(graph, {
      topology: {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 2),
        weights: view('float32', 2),
        nodePositions: view('float32x2', 2)
      },
      facilities: view('float32x2', 2),
      demand: view('float32x2', 2),
      costBreaks: view('float32', 3),
      isochrones: {
        costBreaks: view('float32', 3),
        parameters: view('float32', 12),
        raster: {
          width: 16,
          height: 16,
          output: {
            values: view('float32', 256),
            bandClasses: view('uint32', 256),
            ...(labelled ? {pixelFacilities: view('uint32', 256)} : {})
          }
        }
      }
    });
    const isochrones = recipe.contributors.find(
      contributor => contributor instanceof GPUNetworkIsochrones
    ) as GPUNetworkIsochrones;
    const paths = isochrones.props.raster!.sourcePaths!;
    expect(paths.costs.length).toBe(6);
    expect(paths.topology.offsets.length).toBe(7);
    expect(paths.topology.neighbors.length).toBe(4);
    const kernels = getParsedKernelSources((graph as unknown as {nodes: unknown[]}).nodes);
    const identifiers = [...kernels.keys()];
    expect(identifiers.some(identifier => identifier.endsWith('-source-path-geometry'))).toBe(true);
    const splat = identifiers.findIndex(identifier => identifier.endsWith('-source-paths-splat'));
    const decode = identifiers.findIndex(identifier => identifier.endsWith('-decode'));
    expect(splat).toBeGreaterThanOrEqual(0);
    expect(decode).toBeGreaterThan(splat);
    // wgsl_reflect parses reserved identifiers and constant non-finite bitcasts that Dawn
    // rejects. Either error invalidates the submission, including the earlier routing outputs.
    const sourcePathKernels = [...kernels].filter(([identifier]) =>
      /-source-path(?:s-splat|-geometry|-topology)$/.test(identifier)
    );
    expect(sourcePathKernels).toHaveLength(3);
    for (const [identifier, source] of sourcePathKernels) {
      expect(auditWGSLModule(source), identifier).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});
