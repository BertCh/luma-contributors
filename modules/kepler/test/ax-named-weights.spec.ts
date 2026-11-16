// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {test} from 'vitest';
import * as arrow from 'apache-arrow';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {prepareKeplerGeometrySource} from '../../../examples/deck/kepler-compute/src/geometry-preparation/index';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

function checkEqual(actual: unknown, expected: unknown, label: string): void {
  if (JSON.stringify(actual) !== JSON.stringify(expected))
    throw new Error(
      `AX-43 ${label}\nExpected: ${JSON.stringify(expected)}\nReceived: ${JSON.stringify(actual)}`
    );
}

async function checkReject(
  promise: Promise<unknown>,
  expectedMessage: string,
  label: string
): Promise<void> {
  let message = '';
  try {
    await promise;
  } catch (failure) {
    message = failure instanceof Error ? failure.message : String(failure);
  }
  if (!message.includes(expectedMessage))
    throw new Error(
      `AX-43 ${label}: expected failure containing “${expectedMessage}”; received “${message}”`
    );
}

function square(left: number, bottom: number) {
  return {
    type: 'Polygon',
    coordinates: [
      [
        [left, bottom],
        [left + 1, bottom],
        [left + 1, bottom + 1],
        [left, bottom + 1],
        [left, bottom]
      ]
    ]
  };
}

for (const family of ['point', 'polygon'] as const) {
  test(`AX-43 ${family}: two actual analyses share one build, inspector matches hand-counted topology, edits stale both`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
    // Point distance band: 0—1—2; row 3 is isolated, row 4 has invalid coordinates.
    // Polygon queen: 0—1—2 (the last touch is a corner); row 3 isolated, row 4 null.
    const geometries = [square(0, 0), square(1, 0), square(2, 1), square(10, 10), null];
    const dataset = session.importArrowDataset({
      id: `named-${family}`,
      table: arrow.tableFromArrays({
        events: new Float32Array([10, 20, 60, 90, 1000]),
        population: new Float32Array([100, 100, 100, 100, 100]),
        row: new Float32Array([0, 1, 2, 3, 4]),
        longitude: new Float32Array([0, 1, 2, 10, 181]),
        latitude: new Float32Array([0, 0, 0, 0, 0]),
        geometry: geometries.map(value => value && JSON.stringify(value))
      })
    });
    const explorer = session.openExplorer(dataset.id, {
      workingMemoryBudgetBytes: 8 * 1024 * 1024,
      dimensions: [{id: 'row', column: 'row', kind: 'range'}]
    });
    const polygonSource =
      family === 'polygon'
        ? {
            column: 'geometry',
            source: prepareKeplerGeometrySource([{id: 'source', rows: geometries}])
          }
        : undefined;
    const first = new AnalysisRuntime({session, explorer, requestUpdate() {}, polygonSource});
    const second = new AnalysisRuntime({session, explorer, requestUpdate() {}, polygonSource});
    const render = async () => {
      const encoder = device.createCommandEncoder();
      explorer.encode(encoder);
      first.encode(encoder);
      second.encode(encoder);
      device.submit(encoder.finish());
      await Promise.all([first.readSummary(), second.readSummary()]);
      checkEqual([first.error, second.error], ['', ''], 'both analysis runs succeeded');
    };
    try {
      if (family === 'point') {
        const weights = {
          longitudeColumn: 'longitude',
          latitudeColumn: 'latitude',
          kind: 'distance-band' as const,
          radiusMeters: 120000,
          neighborCapacity: 8
        };
        first.configurePointWeights({...weights, scale: 100, islandPolicy: 'undefined'});
        second.configurePointHotspots({
          ...weights,
          minimumNeighbors: 1,
          selfWeight: 1,
          multipleTesting: 'none',
          islands: 'exclude'
        });
      } else {
        const weights = {
          kind: 'contiguity' as const,
          geometryColumn: 'geometry',
          criterion: 'queen' as const,
          equality: 'exact-source-xy' as const,
          neighborCapacity: 8,
          pairCapacity: 64
        };
        first.configurePolygon({
          kind: 'spatial-rate-polygons',
          events: 'events',
          populations: 'population',
          weights,
          scale: 100,
          islandPolicy: 'undefined'
        });
        second.configurePolygon({
          kind: 'hot-spots-polygons',
          column: 'events',
          weights,
          minimumNeighbors: 1,
          selfWeight: 1,
          multipleTesting: 'none',
          islands: 'exclude',
          inference: 'analytic-two-sided'
        });
      }
      const weightsId = first.weightsReference!.weightsId;
      checkEqual(second.weightsReference!.weightsId, weightsId, 'shared named object identity');
      const store = first.weightsStore;
      const definition = store.get(weightsId);
      // These imports must refer to the very same cache entry, not equivalent settings copies.
      const firstGraph = new GPUCommandGraph(device);
      const secondGraph = new GPUCommandGraph(device);
      const firstLease =
        definition.settings.kind === 'contiguity'
          ? store.polygonCache(dataset).acquire(firstGraph, definition.settings)
          : store.pointCache(dataset).acquire(firstGraph, definition.settings);
      const secondLease =
        definition.settings.kind === 'contiguity'
          ? store.polygonCache(dataset).acquire(secondGraph, definition.settings)
          : store.pointCache(dataset).acquire(secondGraph, definition.settings);
      if (!firstLease.topology || firstLease.topology !== secondLease.topology)
        throw new Error('AX-43 consumer graphs did not import one topology object');
      firstLease.release();
      secondLease.release();
      await render();
      const stats = () =>
        family === 'point' ? first.pointWeightsCacheStats : first.polygonWeightsCacheStats;
      checkEqual(
        [stats().buildCount, stats().encodeCount],
        [1, 1],
        'one submitted build shared by rates and hot spots'
      );
      const inspection = await store.inspect(weightsId);
      checkEqual(
        {
          islands: inspection.islands,
          minimum: inspection.minimumNeighbors,
          maximum: inspection.maximumNeighbors,
          bins: inspection.histogram.map(bin => bin.count)
        },
        {islands: 2, minimum: 0, maximum: 2, bins: [2, 2, 1]},
        'hand-counted full-source histogram and islands'
      );
      checkEqual([stats().buildCount, stats().encodeCount], [1, 1], 'inspection reused topology');
      const inspectionBytes = session.registry.memoryUsage.totalBytes;
      const cancelled = store.inspect(weightsId);
      const request = session.registry
        .getActiveRequests()
        .find(candidate => candidate.slot === `named-weights-inspector:${weightsId}`);
      if (!request)
        throw new Error('AX-43 inspector did not register a cancellable source request');
      request.cancel();
      await checkReject(cancelled, 'cancelled', 'cancelled inspection cannot publish');
      checkEqual(
        session.registry.memoryUsage.totalBytes,
        inspectionBytes,
        'cancelled inspection releases its reservation'
      );
      checkEqual(
        session.registry.getActiveRequests().length,
        0,
        'cancelled inspection releases its source request'
      );
      explorer.setRange('row', 0, 1);
      first.setPopulation('selected');
      await render();
      checkEqual(
        (await store.inspect(weightsId)).histogram.map(bin => bin.count),
        [2, 2, 1],
        'selection did not alter the source topology'
      );
      store.rename(weightsId, 'Shared neighbors');
      checkEqual(
        [first.weightsStaleReason, second.weightsStaleReason],
        [undefined, undefined],
        'rename leaves consumers current'
      );
      if (definition.settings.kind === 'knn')
        throw new Error('AX-43 fixture requires distance-band weights');
      const superseded = store.inspect(weightsId);
      store.edit(
        weightsId,
        definition.settings.kind === 'contiguity'
          ? {...definition.settings, criterion: 'rook'}
          : {...definition.settings, radiusMeters: 250000}
      );
      await checkReject(
        superseded,
        'version must match',
        'version edit cannot publish an old inspection'
      );
      checkEqual(
        [first.active, second.active, first.summary, second.summary],
        [false, false, null, null],
        'both dependent results stale after edit'
      );
      first.selectNamedWeights(weightsId);
      checkEqual(
        Boolean(second.weightsStaleReason),
        true,
        'second stays stale until accepting the version'
      );
      second.selectNamedWeights(weightsId);
      await render();
      checkEqual(
        [stats().buildCount, stats().encodeCount],
        [2, 2],
        'one new build for the edited topology'
      );
      const edited = await store.inspect(weightsId);
      checkEqual(
        edited.histogram.map(bin => bin.count),
        family === 'polygon' ? [3, 2] : [2, 0, 3],
        'edited topology histogram'
      );
      checkEqual(edited.islands, family === 'polygon' ? 3 : 2, 'edited topology islands');
      const insufficient = store.create(
        first.namedWeightsSource,
        {...definition.settings, neighborCapacity: 1},
        'Too few neighbor slots'
      );
      await checkReject(
        store.inspect(insufficient.id),
        'no topology or candidate overflow',
        'inspector refuses incomplete topology'
      );
      const bytes = session.registry.memoryUsage.totalBytes;
      await first.destroy(Promise.resolve());
      await store.inspect(weightsId);
      if (session.registry.memoryUsage.totalBytes >= bytes)
        throw new Error('AX-43 destroying one consumer did not release its owned outputs');
    } finally {
      const fence = device.createFence();
      try {
        await first.destroy(fence.signaled);
        await second.destroy(fence.signaled);
        await session.destroy(fence.signaled);
      } finally {
        fence.destroy();
      }
    }
  });
}

test('AX-43 selecting geographic then planar weights changes the executed neighbors in both directions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'coordinate-transitions',
    table: arrow.tableFromArrays({
      events: new Float32Array([10, 20, 90]),
      population: new Float32Array([100, 100, 100]),
      longitude: new Float32Array([0, 1, 3]),
      latitude: new Float32Array([0, 0, 0]),
      x: new Float32Array([0, 1000, 1]),
      y: new Float32Array([0, 0, 0]),
      row: new Float32Array([0, 1, 2])
    })
  });
  const explorer = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'row', column: 'row', kind: 'range'}]
  });
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate() {}});
  const renderAndCheckRates = async (expected: readonly number[], label: string) => {
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    analysis.encode(encoder);
    device.submit(encoder.finish());
    await analysis.readSummary();
    checkEqual(analysis.error, '', `${label} run succeeded`);
    for (const [row, expectedValue] of expected.entries()) {
      const result = await analysis.readRow(row);
      if (!result || Math.abs(result.value - expectedValue) > 1e-5)
        throw new Error(
          `AX-43 ${label}, row ${row}: expected rate ${expectedValue}, received ${result?.value}`
        );
    }
  };
  try {
    analysis.setColumn('events');
    analysis.setDenominatorColumn('population');
    analysis.configurePointWeights({
      coordinateSpace: 'planar',
      xColumn: 'x',
      yColumn: 'y',
      crs: 'EPSG:3857',
      bounds: [0, -1, 1001, 1],
      k: 1,
      scale: 100,
      islandPolicy: 'undefined'
    });
    const planarWeightsId = analysis.weightsReference!.weightsId;
    const geographic = analysis.weightsStore.create(
      analysis.namedWeightsSource,
      {
        kind: 'knn',
        coordinateSpace: 'longitude-latitude',
        longitudeColumn: 'longitude',
        latitudeColumn: 'latitude',
        metric: 'great-circle',
        units: 'meters',
        k: 1
      },
      'Geographic nearest neighbors'
    );
    // Projected X order is rows 0, 2, 1: links 0→2, 1→2, 2→0.
    await renderAndCheckRates([50, 55, 50], 'initial planar topology');
    analysis.selectNamedWeights(geographic.id);
    checkEqual(
      JSON.parse(analysis.serializeRecipe()).operation.weights,
      geographic.settings,
      'geographic reference matches execution'
    );
    // Longitude order is rows 0, 1, 2, with unequal gaps: links 0→1, 1→0, 2→1.
    await renderAndCheckRates([15, 15, 55], 'planar to geographic selection');
    analysis.selectNamedWeights(planarWeightsId);
    checkEqual(
      JSON.parse(analysis.serializeRecipe()).operation.weights,
      analysis.weightsStore.get(planarWeightsId).settings,
      'planar reference matches execution'
    );
    await renderAndCheckRates([50, 55, 50], 'geographic to planar selection');
  } finally {
    const fence = device.createFence();
    try {
      await analysis.destroy(fence.signaled);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
});
