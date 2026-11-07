// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import {
  GPUColumnProfile,
  GPU_COLUMN_PROFILE_STATISTIC
} from '@luma-contributors/graph/gpu-dataframe';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUSinusoidalProjection} from '@luma-contributors/graph/geospatial';
import {GraphSwap} from '../../engine/graph-swap';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';

export type SinusoidalProjectionOptions = {
  hub: 'ORD' | 'HND' | 'SYD';
  wrapLongitude: boolean;
  axis: 'x' | 'y';
};

type ProjectionVariant = {
  hub: SinusoidalProjectionOptions['hub'];
  compiled: CompiledGPUCommandGraph<void>;
  raw: {
    x: ReturnType<SpatialAnalysisResources['createBuffer']>;
    y: ReturnType<SpatialAnalysisResources['createBuffer']>;
  };
  wrapped: {
    x: ReturnType<SpatialAnalysisResources['createBuffer']>;
    y: ReturnType<SpatialAnalysisResources['createBuffer']>;
  };
  stats: ReturnType<SpatialAnalysisResources['createBuffer']>;
  probe: ReturnType<SpatialAnalysisResources['createBuffer']>;
  encoded: boolean;
};

const AIRPORT_COUNT = 3257;
const STATISTIC_COUNT = 11;
const PROBE_COUNT = 8;
const HUB_ROWS = {ORD: 2033, HND: 1071, SYD: 2589} as const;
const LOS_ANGELES_ROW = 1501;

/** GPU coordinate-contract diagnostic on the frozen OpenFlights airport coordinates. */
export async function createSinusoidalProjection(
  ctx: SceneContext<SinusoidalProjectionOptions>
): Promise<SceneInstance<SinusoidalProjectionOptions>> {
  const {device} = ctx;
  const dataset = ctx.datasets.get('openflights');
  const positions = dataset.column<Float32Array>('locations');
  const airportCount = positions.length / 2;
  if (airportCount !== AIRPORT_COUNT) throw new Error(`Expected ${AIRPORT_COUNT} airport rows`);
  const resources = new SpatialAnalysisResources(device, 'sinusoidal-projection');
  const positionBuffer = resources.createBuffer('airport-positions', positions);
  const hubIdBuffers = new Map(
    Object.entries(HUB_ROWS).map(([hub, row]) => [
      hub,
      resources.createBuffer(`hub-${hub}-id`, new Uint32Array([row]))
    ])
  );
  const laxIdBuffer = resources.createBuffer('lax-id', new Uint32Array([LOS_ANGELES_ROW]));
  const swap = resources.track(new GraphSwap());
  const cache = new Map<string, ProjectionVariant>();
  let active: ProjectionVariant | null = null;
  let latestSignature = '';
  let reader: SummaryReader | null = null;
  let requested = true;
  let viewAxis = ctx.options.axis;
  let wrappedView = ctx.options.wrapLongitude;

  function clearReadouts(): void {
    ctx.setReadout('projectionExtent', 'Updating GPU projection…');
    ctx.setReadout('projectionProbe', 'Updating GPU projection…');
  }

  async function buildVariant(hub: SinusoidalProjectionOptions['hub']): Promise<ProjectionVariant> {
    const cached = cache.get(hub);
    if (cached) return cached;
    const raw = resources.createBuffer(`raw-${hub}`, airportCount * 8);
    const wrapped = resources.createBuffer(`wrapped-${hub}`, airportCount * 8);
    const rawX = resources.createBuffer(`raw-x-${hub}`, airportCount * 4);
    const rawY = resources.createBuffer(`raw-y-${hub}`, airportCount * 4);
    const wrappedX = resources.createBuffer(`wrapped-x-${hub}`, airportCount * 4);
    const wrappedY = resources.createBuffer(`wrapped-y-${hub}`, airportCount * 4);
    const stats = resources.createBuffer(`statistics-${hub}`, 4 * STATISTIC_COUNT * 4);
    const probe = resources.createBuffer(`probe-${hub}`, PROBE_COUNT * 4);
    const probeIndices = resources.createBuffer(
      `probe-indices-${hub}`,
      new Uint32Array([HUB_ROWS[hub], LOS_ANGELES_ROW])
    );
    const graph = new GPUCommandGraph<void>(device, {id: `sinusoidal-${hub}`});
    const importBuffer = createGraphBufferImporter(graph);
    const positionsView = importBuffer(positionBuffer, 'float32x2', airportCount);
    const rawView = importBuffer(raw, 'float32x2', airportCount);
    const wrappedView = importBuffer(wrapped, 'float32x2', airportCount);
    graph.add(
      new GPUSinusoidalProjection({
        id: 'projection-raw',
        positions: positionsView,
        output: rawView,
        origin: [positions[HUB_ROWS[hub] * 2], positions[HUB_ROWS[hub] * 2 + 1]]
      })
    );
    graph.add(
      new GPUSinusoidalProjection({
        id: 'projection-wrapped',
        positions: positionsView,
        output: wrappedView,
        origin: [positions[HUB_ROWS[hub] * 2], positions[HUB_ROWS[hub] * 2 + 1]],
        wrapLongitude: true
      })
    );
    const rawScalarView = importBuffer(raw, 'float32', airportCount * 2);
    const wrappedScalarView = importBuffer(wrapped, 'float32', airportCount * 2);
    const rawXView = importBuffer(rawX, 'float32', airportCount);
    const rawYView = importBuffer(rawY, 'float32', airportCount);
    const wrappedXView = importBuffer(wrappedX, 'float32', airportCount);
    const wrappedYView = importBuffer(wrappedY, 'float32', airportCount);
    for (const [id, source, x, y] of [
      ['split-raw', rawScalarView, rawXView, rawYView],
      ['split-wrapped', wrappedScalarView, wrappedXView, wrappedYView]
    ] as const) {
      addKernelPass(graph, {
        id,
        invocationCount: airportCount,
        bindings: [
          {name: 'source', view: source, type: 'f32', access: 'read'},
          {name: 'xOutput', view: x, type: 'f32', access: 'read_write'},
          {name: 'yOutput', view: y, type: 'f32', access: 'read_write'}
        ],
        body: 'xOutput[xOutputOffset + index] = source[sourceOffset + index * 2u]; yOutput[yOutputOffset + index] = source[sourceOffset + index * 2u + 1u];'
      });
    }
    const statsView = importBuffer(stats, 'float32', 4 * STATISTIC_COUNT);
    graph.add(
      new GPUColumnProfile({
        id: 'profile-projected-coordinates',
        columns: [rawXView, rawYView, wrappedXView, wrappedYView].map(values => ({values})),
        histogramBinCount: 1,
        output: {statistics: statsView}
      })
    );
    const probeView = importBuffer(probe, 'float32', PROBE_COUNT);
    const probeRowView = importBuffer(probeIndices, 'uint32', 2);
    addKernelPass(graph, {
      id: 'probe-hub-and-lax',
      invocationCount: 1,
      bindings: [
        {name: 'rawX', view: rawXView, type: 'f32', access: 'read'},
        {name: 'rawY', view: rawYView, type: 'f32', access: 'read'},
        {name: 'wrappedX', view: wrappedXView, type: 'f32', access: 'read'},
        {name: 'wrappedY', view: wrappedYView, type: 'f32', access: 'read'},
        {name: 'rows', view: probeRowView, type: 'u32', access: 'read'},
        {name: 'output', view: probeView, type: 'f32', access: 'read_write'}
      ],
      body: 'let hub = rows[rowsOffset]; let lax = rows[rowsOffset + 1u]; output[outputOffset] = rawX[rawXOffset + hub]; output[outputOffset + 1u] = rawY[rawYOffset + hub]; output[outputOffset + 2u] = wrappedX[wrappedXOffset + hub]; output[outputOffset + 3u] = wrappedY[wrappedYOffset + hub]; output[outputOffset + 4u] = rawX[rawXOffset + lax]; output[outputOffset + 5u] = rawY[rawYOffset + lax]; output[outputOffset + 6u] = wrappedX[wrappedXOffset + lax]; output[outputOffset + 7u] = wrappedY[wrappedYOffset + lax];'
    });
    const compiled = resources.track(await graph.compileAsync());
    const variant = {
      hub,
      compiled,
      raw: {x: rawX, y: rawY},
      wrapped: {x: wrappedX, y: wrappedY},
      stats,
      probe,
      encoded: false
    };
    cache.set(hub, variant);
    return variant;
  }

  function requestHub(hub: SinusoidalProjectionOptions['hub']): void {
    const signature = `${hub}|${viewAxis}|${wrappedView}`;
    latestSignature = signature;
    requested = true;
    clearReadouts();
    ctx.setStatus(`Compiling the ${hub} projection variant`);
    void swap
      .request({
        build: () => buildVariant(hub),
        commit: variant => {
          active = variant;
          reader = makeReader(variant);
          requested = true;
          ctx.setStatus('');
          ctx.requestLayers();
        }
      })
      .catch(error => {
        ctx.setReadout('projectionExtent', 'GPU projection unavailable');
        ctx.setReadout('projectionProbe', String(error));
        ctx.setStatus('Projection graph failed to compile');
      });
  }

  clearReadouts();
  requestHub(ctx.options.hub);
  const makeReader = (variant: ProjectionVariant) => {
    reader?.release();
    const ownerSignature = `${variant.hub}|${viewAxis}|${wrappedView}`;
    return new SummaryReader(
      resources,
      'projection-summary',
      [
        {buffer: variant.stats, size: 4 * STATISTIC_COUNT * 4},
        {buffer: variant.probe, size: PROBE_COUNT * 4}
      ],
      bytes => {
        if (latestSignature !== ownerSignature || active !== variant) return;
        const values = new Float32Array(bytes);
        const statIndex = (column: number, statistic: number) =>
          values[column * STATISTIC_COUNT + statistic];
        const minimum = statIndex(
          viewAxis === 'x' ? (wrappedView ? 2 : 0) : wrappedView ? 3 : 1,
          GPU_COLUMN_PROFILE_STATISTIC.minimum
        );
        const maximum = statIndex(
          viewAxis === 'x' ? (wrappedView ? 2 : 0) : wrappedView ? 3 : 1,
          GPU_COLUMN_PROFILE_STATISTIC.maximum
        );
        const finiteCount = statIndex(
          viewAxis === 'x' ? (wrappedView ? 2 : 0) : wrappedView ? 3 : 1,
          GPU_COLUMN_PROFILE_STATISTIC.count
        );
        const probe = new Float32Array(bytes, 4 * STATISTIC_COUNT * 4, PROBE_COUNT);
        ctx.setReadout(
          'projectionExtent',
          `${formatCount(finiteCount)} finite GPU rows · ${minimum.toFixed(3)} to ${maximum.toFixed(3)} km on ${viewAxis}`
        );
        ctx.setReadout(
          'projectionProbe',
          `${variant.hub} raw → wrapped (x, y km): ${probe[0].toFixed(3)}, ${probe[1].toFixed(3)} → ${probe[2].toFixed(3)}, ${probe[3].toFixed(3)}\nLAX raw → wrapped (x, y km): ${probe[4].toFixed(3)}, ${probe[5].toFixed(3)} → ${probe[6].toFixed(3)}, ${probe[7].toFixed(3)}`
        );
      }
    );
  };

  return {
    getCompiledGraphs: () => (active ? [active.compiled] : []),
    setOption(id, value) {
      if (id === 'hub') requestHub(value as SinusoidalProjectionOptions['hub']);
      if (id === 'axis' || id === 'wrapLongitude') {
        viewAxis = id === 'axis' ? (value as SinusoidalProjectionOptions['axis']) : viewAxis;
        wrappedView = id === 'wrapLongitude' ? Boolean(value) : wrappedView;
        latestSignature = `${ctx.options.hub}|${viewAxis}|${wrappedView}`;
        if (active) reader = makeReader(active);
        requested = true;
        clearReadouts();
      }
      ctx.requestLayers();
    },
    onThemeChange: () => ctx.requestLayers(),
    encode(commandEncoder) {
      const variant = active;
      if (!variant) return;
      if (!variant.encoded) {
        variant.compiled.encode(commandEncoder, {parameters: undefined});
        variant.encoded = true;
        reader = makeReader(variant);
        requested = true;
      }
      if (requested && reader) {
        reader.request(commandEncoder);
        requested = false;
      }
      reader?.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      const variant = active;
      if (!variant) return [];
      const selected = wrappedView ? variant.wrapped : variant.raw;
      const valueBuffer = viewAxis === 'x' ? selected.x : selected.y;
      const range: readonly [number, number] = viewAxis === 'x' ? [-40000, 40000] : [-20000, 20000];
      ctx.setLegendExtent('projected-coordinate', range);
      return [
        new SpatialAnalysisPointLayer({
          id: 'sinusoidal-airports',
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          positions: positionBuffer,
          values: valueBuffer,
          valueFormat: 'float32',
          instanceCount: airportCount,
          radiusPixels: 3,
          colormap: 'diverging',
          valueRange: range,
          color: [255, 255, 255, 255]
        }),
        new SpatialAnalysisPointLayer({
          id: 'sinusoidal-origin',
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          positions: positionBuffer,
          ids: hubIdBuffers.get(variant.hub)!,
          instanceCount: 1,
          radiusPixels: 9,
          shape: 'ring',
          color: [182, 133, 255, 255],
          outlineWidthPixels: 2
        }),
        new SpatialAnalysisPointLayer({
          id: 'sinusoidal-lax-probe',
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          positions: positionBuffer,
          ids: laxIdBuffer,
          instanceCount: 1,
          radiusPixels: 7,
          shape: 'ring',
          color: [255, 157, 53, 255],
          outlineWidthPixels: 2
        })
      ];
    },
    destroy: () => {
      reader?.release();
      swap.destroy();
      resources.destroy();
    }
  };
}
