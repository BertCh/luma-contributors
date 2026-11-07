// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getGPURasterConditionalParameterValues, GPURasterConditional, GPURasterTemporal} from '@luma-contributors/analysis/raster-analysis';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import type {RampName} from '../../engine/ramps';
import {SpatialAnalysisResources, formatCount} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {createGraphViewer, createUtmFrame, UtmRasterLayer} from './b16-common';
import {loadNdviStack} from './vegetation-trends.compute';
import {SOURCE_DATE_COLORS} from './composite-evidence-data';

export type CompositeEvidenceOptions = {
  operation: 'mean' | 'minimum' | 'maximum' | 'median';
  layer: 'composite' | 'support' | 'source-date';
  dateWindow: readonly [number, number];
  minimumCount: number;
  opacity: number;
  ramp: RampName;
};

const OPERATIONS = ['mean', 'minimum', 'maximum', 'median'] as const;

type CompositeBuffers = {
  value: Buffer;
  count: Buffer;
  source: Buffer;
  status: Buffer;
  visible: Buffer;
};

/** A direct GPU mean/extreme/median over the valid Sentinel-2 acquisitions in a chosen window. */
export async function createCompositeEvidence(
  ctx: SceneContext<CompositeEvidenceOptions>
): Promise<SceneInstance<CompositeEvidenceOptions>> {
  ctx.setStatus('Loading the Sentinel-2 NDVI time stack');
  const ndvi = await loadNdviStack(ctx.signal);
  ctx.signal.throwIfAborted();
  const {width, height, cellCount, sliceCount, manifest} = ndvi;
  const frame = createUtmFrame(
    [(manifest.bbox[0] + manifest.bbox[2]) / 2, (manifest.bbox[1] + manifest.bbox[3]) / 2],
    {
      boundsProjected: manifest.raster.boundsProjected,
      cellSizeM: manifest.raster.cellSizeM,
      width,
      height
    }
  );
  const resources = new SpatialAnalysisResources(ctx.device, 'composite');
  const stack = new Float32Array(ndvi.cellMajor);
  const stackBuffer = resources.createBuffer('stack', stack);
  const minimumMask = resources.createBuffer(
    'minimum-count-mask',
    new Uint32Array(cellCount).fill(1)
  );
  const minimumParameters = resources.createParameterBuffer(
    'minimum-count-parameters',
    'float32',
    8,
    getGPURasterConditionalParameterValues({constantB: Number.NaN})
  );
  const marker = resources.createBuffer('selected-cell', new Float32Array(2));
  const outputs = {} as Record<(typeof OPERATIONS)[number], CompositeBuffers>;
  const compiled = {} as Record<(typeof OPERATIONS)[number], CompiledGPUCommandGraph<void>>;

  for (const operation of OPERATIONS) {
    const buffers: CompositeBuffers = {
      value: resources.createBuffer(`${operation}-value`, cellCount * 4),
      count: resources.createBuffer(`${operation}-count`, cellCount * 4),
      source: resources.createBuffer(`${operation}-source-date`, cellCount * 4),
      status: resources.createBuffer(`${operation}-status`, cellCount * 4),
      visible: resources.createBuffer(`${operation}-minimum-count-value`, cellCount * 4)
    };
    outputs[operation] = buffers;
    const graph = new GPUCommandGraph<void>(ctx.device, {id: `composite-${operation}`});
    const view = createGraphViewer(graph);
    const floatView = (
      name: string,
      buffer: CompositeBuffers[keyof CompositeBuffers] | typeof stackBuffer,
      length = cellCount
    ) => view(name, buffer, 'float32', length);
    const wordView = (
      name: string,
      buffer: CompositeBuffers[keyof CompositeBuffers] | typeof minimumMask,
      length = cellCount
    ) => view(name, buffer, 'uint32', length);
    graph.add(
      new GPURasterTemporal({
        id: `temporal-${operation}`,
        cellCount,
        sliceCount,
        values: floatView('ndvi', stackBuffer, cellCount * sliceCount),
        operation,
        output: {
          values: floatView('raw-composite', buffers.value),
          counts: wordView('valid-count', buffers.count),
          chosenSlice: wordView('chosen-date', buffers.source),
          status: wordView('status', buffers.status)
        }
      })
    );
    graph.add(
      new GPURasterConditional({
        id: `support-mask-${operation}`,
        cellCount,
        mask: wordView('minimum-count-mask', minimumMask),
        a: floatView('raw-composite', buffers.value),
        parameters: minimumParameters.importToGraph(graph),
        output: {values: floatView('display-composite', buffers.visible)}
      })
    );
    compiled[operation] = resources.track(await graph.compileAsync());
  }

  ctx.setLegendData('dates', [...ndvi.dates]);
  let changed = true;
  let currentWindow = '';
  let selectedCell = 13449;
  marker.write(
    Float32Array.of(...frame.cellToMeters(selectedCell % width, Math.floor(selectedCell / width)))
  );
  let dirtyMask = true;
  let appliedMaskKey = '';
  let revision = 0;
  const summaries: Partial<
    Record<
      (typeof OPERATIONS)[number],
      {values: Float32Array; counts: Uint32Array; sources: Uint32Array; statuses: Uint32Array}
    >
  > = {};
  const readers = {} as Record<(typeof OPERATIONS)[number], SummaryReader>;
  const requestedRevision = {} as Record<(typeof OPERATIONS)[number], number>;
  const acceptedRevision = {} as Record<(typeof OPERATIONS)[number], number>;

  const publishSummary = (operation: (typeof OPERATIONS)[number]) => {
    const summary = summaries[operation];
    if (!summary) return;
    let included = 0;
    let total = 0;
    let lowSupport = 0;
    let maximumSupport = 0;
    for (let cell = 0; cell < cellCount; cell++) {
      const count = summary.counts[cell];
      total += count;
      maximumSupport = Math.max(maximumSupport, count);
      if (count < ctx.options.minimumCount) lowSupport++;
      else if (Number.isFinite(summary.values[cell])) included++;
    }
    const windowLength = Math.abs(ctx.options.dateWindow[1] - ctx.options.dateWindow[0]) + 1;
    ctx.setReadout(
      'support',
      `${formatCount(included)} cells meet the ${ctx.options.minimumCount}-of-${windowLength} minimum · mean support ${(total / cellCount).toFixed(1)} · maximum ${maximumSupport}`
    );
    ctx.setReadout(
      'masked',
      `${formatCount(lowSupport)} cells below minimum support (shown transparent)`
    );
    ctx.setReadout(
      'window',
      `${ndvi.dates[Math.min(...ctx.options.dateWindow)]} to ${ndvi.dates[Math.max(...ctx.options.dateWindow)]} · ${Math.abs(ctx.options.dateWindow[1] - ctx.options.dateWindow[0]) + 1} sampled acquisitions`
    );
    if (selectedCell >= 0) publishPixel(summary);
  };

  for (const operation of OPERATIONS) {
    const buffers = outputs[operation];
    readers[operation] = new SummaryReader(
      resources,
      `composite-read-${operation}`,
      [
        {buffer: buffers.value, size: cellCount * 4},
        {buffer: buffers.count, size: cellCount * 4},
        {buffer: buffers.source, size: cellCount * 4},
        {buffer: buffers.status, size: cellCount * 4}
      ],
      bytes => {
        if (requestedRevision[operation] !== revision || ctx.options.operation !== operation)
          return;
        const fieldByteLength = cellCount * 4;
        summaries[operation] = {
          values: new Float32Array(bytes, 0, cellCount),
          counts: new Uint32Array(bytes, fieldByteLength, cellCount),
          sources: new Uint32Array(bytes, fieldByteLength * 2, cellCount),
          statuses: new Uint32Array(bytes, fieldByteLength * 3, cellCount)
        };
        const summary = summaries[operation]!;
        const maskKey = `${currentWindow}:${ctx.options.minimumCount}`;
        if (appliedMaskKey !== maskKey) {
          const mask = new Uint32Array(cellCount);
          for (let cell = 0; cell < cellCount; cell++)
            mask[cell] = summary.counts[cell] >= ctx.options.minimumCount ? 1 : 0;
          minimumMask.write(mask);
          appliedMaskKey = maskKey;
          changed = true;
          ctx.requestLayers();
        }
        dirtyMask = false;
        acceptedRevision[operation] = revision;
        publishSummary(operation);
        if (selectedCell >= 0) ctx.refreshTooltip();
      }
    );
  }

  function publishPixel(summary = summaries[ctx.options.operation]): void {
    if (!summary || selectedCell < 0 || acceptedRevision[ctx.options.operation] !== revision) {
      ctx.setReadout('pixel', 'click the map');
      ctx.setChart('pixelChart', null);
      return;
    }
    const column = selectedCell % width;
    const row = Math.floor(selectedCell / width);
    const dateIndex = summary.sources[selectedCell];
    const sourceDate =
      dateIndex === 0xffffffff
        ? 'no unique source date (mean or even-sample median)'
        : (ndvi.dates[dateIndex] ?? 'unknown source date');
    const series = ndvi.cellMajor.subarray(
      selectedCell * sliceCount,
      (selectedCell + 1) * sliceCount
    );
    const acquisitionDays = ndvi.dates.map(date => Date.parse(date) / 86400000);
    ctx.setReadout(
      'pixel',
      `${ctx.options.operation}: ${Number.isFinite(summary.values[selectedCell]) ? summary.values[selectedCell].toFixed(3) : 'no valid observation'} · support ${summary.counts[selectedCell]} · source ${sourceDate} · status ${summary.statuses[selectedCell]} · column ${column}, row ${row}`
    );
    ctx.setChart('pixelChart', {
      kind: 'line',
      title: 'Observed NDVI at this 100 m cell',
      xLabel: 'acquisition date',
      yLabel: 'NDVI',
      xDomain: [acquisitionDays[0], acquisitionDays[acquisitionDays.length - 1]],
      yDomain: [-1, 1],
      formatX: value => new Date(value * 86400000).toISOString().slice(0, 10),
      formatY: value => value.toFixed(2),
      description:
        'The clicked cell’s actual NDVI at each irregular Sentinel-2 acquisition; gaps remain missing.',
      series: [{label: 'Observed', x: acquisitionDays, y: series, points: true}]
    });
  }

  function updateWindow(): void {
    const first = Math.min(...ctx.options.dateWindow);
    const last = Math.max(...ctx.options.dateWindow);
    const key = `${first}-${last}`;
    if (key === currentWindow) return;
    currentWindow = key;
    for (let cell = 0; cell < cellCount; cell++) {
      const base = cell * sliceCount;
      for (let slice = 0; slice < sliceCount; slice++) {
        if (slice < first || slice > last) stack[base + slice] = Number.NaN;
        else stack[base + slice] = ndvi.cellMajor[base + slice];
      }
    }
    stackBuffer.write(stack);
    dirtyMask = true;
  }

  const refresh = () => {
    changed = true;
    ctx.requestLayers();
  };
  const getCell = (coordinate: readonly [number, number] | null): number => {
    if (!coordinate) return -1;
    const [column, row] = frame.lngLatToCell(coordinate[0], coordinate[1]);
    const cellColumn = Math.floor(column);
    const cellRow = Math.floor(row);
    return cellColumn < 0 || cellRow < 0 || cellColumn >= width || cellRow >= height
      ? -1
      : cellRow * width + cellColumn;
  };
  ctx.setReadout('pixel', 'click the map');
  ctx.setReadout(
    'screening',
    `${sliceCount} Sentinel-2 acquisitions · ${ndvi.dates[0]} to ${ndvi.dates[sliceCount - 1]} · 100 m cells`
  );
  ctx.setStatus('');

  return {
    getCompiledGraphs: () => OPERATIONS.map(operation => compiled[operation]),
    setOption(id) {
      revision++;
      for (const operation of OPERATIONS) delete summaries[operation];
      if (id === 'dateWindow') currentWindow = '';
      if (id === 'minimumCount') dirtyMask = true;
      refresh();
    },
    encode(commandEncoder, frameInfo) {
      updateWindow();
      const operation = ctx.options.operation;
      const shouldRead = changed || dirtyMask || frameInfo.frameIndex < 2;
      if (changed || frameInfo.frameIndex < 2) {
        compiled[operation].encode(commandEncoder, {parameters: undefined});
        changed = false;
      }
      const reader = readers[operation];
      if (
        (shouldRead || acceptedRevision[operation] !== revision) &&
        acceptedRevision[operation] !== revision &&
        !reader.isPending &&
        requestedRevision[operation] !== revision
      ) {
        requestedRevision[operation] = revision;
        reader.request(commandEncoder);
      }
      reader.flush(commandEncoder);
    },
    getLayers() {
      const state = ctx.options;
      const buffers = outputs[state.operation];
      const base = {
        id: `composite-${state.layer}-${state.operation}`,
        frame,
        opacity: state.opacity,
        noDataColor: [0, 0, 0, 0] as const
      };
      let raster: Layer;
      if (state.layer === 'support') {
        raster = new UtmRasterLayer({
          ...base,
          values: buffers.count,
          valueFormat: 'uint32',
          colormap: state.ramp,
          valueRange: [0, Math.max(1, Math.abs(state.dateWindow[1] - state.dateWindow[0]) + 1)]
        });
      } else if (state.layer === 'source-date') {
        raster = new UtmRasterLayer({
          ...base,
          values: buffers.source,
          valueFormat: 'uint32',
          noDataValue: 0xffffffff,
          colormap: 'category',
          palette: SOURCE_DATE_COLORS
        });
      } else {
        raster = new UtmRasterLayer({
          ...base,
          values: buffers.visible,
          valueFormat: 'float32',
          colormap: state.ramp,
          valueRange: [-0.1, 0.9]
        });
      }
      const layers: Layer[] = [raster];
      if (selectedCell >= 0)
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'composite-selected-cell',
            coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
            positions: marker,
            instanceCount: 1,
            radiusPixels: 7,
            color: ctx.theme() === 'dark' ? [255, 255, 255, 255] : [20, 20, 20, 255]
          })
        );
      return layers;
    },
    getTooltip(event) {
      const cell = getCell(event.coordinate);
      if (cell < 0) return null;
      const summary = summaries[ctx.options.operation];
      if (!summary || acceptedRevision[ctx.options.operation] !== revision)
        return `Loading ${ctx.options.operation} composite…`;
      const dateIndex = summary.sources[cell];
      const sourceDate =
        dateIndex === 0xffffffff ? 'no unique source date' : (ndvi.dates[dateIndex] ?? 'unknown');
      return `NDVI ${Number.isFinite(summary.values[cell]) ? summary.values[cell].toFixed(3) : 'no data'}\nValid observations ${summary.counts[cell]} / ${Math.abs(ctx.options.dateWindow[1] - ctx.options.dateWindow[0]) + 1}\nSource ${sourceDate}\n${summary.statuses[cell] === 0 ? 'status: supported' : 'status: no valid observation'}`;
    },
    onClick(event) {
      const cell = getCell(event.coordinate);
      if (cell < 0) return false;
      selectedCell = cell;
      marker.write(Float32Array.of(...frame.cellToMeters(cell % width, Math.floor(cell / width))));
      publishPixel();
      ctx.requestLayers();
      return true;
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    destroy() {
      for (const reader of Object.values(readers)) reader.stop();
      resources.destroy();
    }
  };
}
