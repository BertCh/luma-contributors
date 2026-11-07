// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import {
  GPUColumnProfile,
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT,
  GPUVariableStandardization
} from '@luma-contributors/graph/gpu-dataframe';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';

const DATASET_ID = 'poopdeck-earthquakes';
const HISTOGRAM_BINS = 28;
const METHOD_IDS = [
  'deviation-from-mean',
  'z-score',
  'mad',
  'range-adjust',
  'range-standardize'
] as const;
const OUTPUT_METHOD_IDS = [...METHOD_IDS, 'z-score-population'] as const;
const PROFILE_COLUMN_COUNT = 1 + OUTPUT_METHOD_IDS.length;
const REFERENCE_POPULATIONS = ['all-earthquakes', 'magnitude-6-plus', 'shallow-70-km'] as const;

type StandardizationMethod = (typeof METHOD_IDS)[number];
export type EarthquakeStandardizationOptions = {
  field: 'magnitude' | 'depth';
  referencePopulation: (typeof REFERENCE_POPULATIONS)[number];
  method: StandardizationMethod;
  variance: 'sample' | 'population';
};

function getOutputIndex(method: StandardizationMethod, variance: 'sample' | 'population'): number {
  if (method !== 'z-score') return METHOD_IDS.indexOf(method) + 1;
  return variance === 'sample' ? METHOD_IDS.indexOf('z-score') + 1 : OUTPUT_METHOD_IDS.length;
}

function getMethodLabel(method: StandardizationMethod, variance: 'sample' | 'population'): string {
  switch (method) {
    case 'deviation-from-mean':
      return 'Deviation from mean · x − mean';
    case 'z-score':
      return `Z-score · ${variance} standard deviation`;
    case 'mad':
      return 'Mean absolute deviation · (x − mean) / mean(|x − mean|)';
    case 'range-adjust':
      return 'Range-adjust · x / (max − min)';
    case 'range-standardize':
      return 'Range-standardize · (x − min) / (max − min)';
  }
}

function getReferenceLabel(
  referencePopulation: EarthquakeStandardizationOptions['referencePopulation']
): string {
  switch (referencePopulation) {
    case 'all-earthquakes':
      return 'All catalog records typed as earthquake';
    case 'magnitude-6-plus':
      return 'Earthquakes with magnitude ≥ 6';
    case 'shallow-70-km':
      return 'Earthquakes with reported depth ≤ 70 km';
  }
}

/** Standardization controls only choose a field, source-derived population and output view. */
export async function createEarthquakeStandardization(
  ctx: SceneContext<EarthquakeStandardizationOptions>
): Promise<SceneInstance<EarthquakeStandardizationOptions>> {
  const {device} = ctx;
  const dataset = ctx.datasets.get(DATASET_ID);
  const rowCount = dataset.count;
  const positions = dataset.column<Float32Array>('position');
  const magnitude = dataset.column<Float32Array>('magnitude');
  const depth = dataset.column<Float32Array>('depth');
  const eventType = dataset.column<Uint8Array>('type');
  const eventTypeNames = dataset.categories('type');
  const earthquakeTypeCode = eventTypeNames.indexOf('earthquake');
  const resources = new SpatialAnalysisResources(device, 'earthquake-standardization');
  const positionBuffer = resources.createBuffer('event-positions', positions);
  const valuesBuffer = resources.createBuffer('selected-values', magnitude);
  const maskBuffer = resources.createBuffer('reference-mask', new Uint32Array(rowCount));
  const probeIndexBuffer = resources.createBuffer('probe-row-index', new Uint32Array(1));
  const probeValuesBuffer = resources.createBuffer(
    'probe-values',
    (OUTPUT_METHOD_IDS.length + 1) * 4
  );
  const outputBuffers = OUTPUT_METHOD_IDS.map(method =>
    resources.createBuffer(`standardized-${method}`, rowCount * 4)
  );
  const statisticsBuffer = resources.createBuffer(
    'profile-statistics',
    PROFILE_COLUMN_COUNT * GPU_COLUMN_PROFILE_STATISTIC_COUNT * 4
  );
  const histogramsBuffer = resources.createBuffer(
    'profile-histograms',
    PROFILE_COLUMN_COUNT * HISTOGRAM_BINS * 4
  );

  const graph = new GPUCommandGraph<void>(device, {id: 'earthquake-standardization'});
  const values = importGraphBuffer(graph, 'selected-values', valuesBuffer, 'float32', rowCount);
  const mask = importGraphBuffer(graph, 'reference-mask', maskBuffer, 'uint32', rowCount);
  const probeIndex = importGraphBuffer(graph, 'probe-row-index', probeIndexBuffer, 'uint32', 1);
  const probeValues = importGraphBuffer(
    graph,
    'probe-values',
    probeValuesBuffer,
    'float32',
    OUTPUT_METHOD_IDS.length + 1
  );
  const outputViews = outputBuffers.map((buffer, index) =>
    importGraphBuffer(
      graph,
      `standardized-${OUTPUT_METHOD_IDS[index]}`,
      buffer,
      'float32',
      rowCount
    )
  );
  for (const method of METHOD_IDS) {
    graph.add(
      new GPUVariableStandardization({
        id: `standardization-${method}`,
        values,
        mask,
        method,
        ...(method === 'z-score' ? {variance: 'sample' as const} : {}),
        output: outputViews[OUTPUT_METHOD_IDS.indexOf(method)]
      })
    );
  }
  graph.add(
    new GPUVariableStandardization({
      id: 'standardization-z-score-population',
      values,
      mask,
      method: 'z-score',
      variance: 'population',
      output: outputViews[OUTPUT_METHOD_IDS.length - 1]
    })
  );
  addKernelPass(graph, {
    id: 'earthquake-standardization-probe',
    invocationCount: 1,
    bindings: [
      {name: 'values', view: values, type: 'f32', access: 'read'},
      {name: 'probeIndex', view: probeIndex, type: 'u32', access: 'read'},
      ...outputViews.map((view, index) => ({
        name: `output${index}`,
        view,
        type: 'f32' as const,
        access: 'read' as const
      })),
      {name: 'probeValues', view: probeValues, type: 'f32', access: 'read_write'}
    ],
    body: `
      let row = probeIndex[probeIndexOffset];
      probeValues[probeValuesOffset] = values[row + valuesOffset];
      ${outputViews
        .map(
          (_, index) =>
            `probeValues[probeValuesOffset + ${index + 1}u] = output${index}[row + output${index}Offset];`
        )
        .join('\n      ')}
    `
  });
  graph.add(
    new GPUColumnProfile({
      id: 'earthquake-standardization-profiles',
      columns: [{values}, ...outputViews.map(output => ({values: output}))],
      mask,
      histogramBinCount: HISTOGRAM_BINS,
      hyperLogLogPrecision: 4,
      topCategoryCount: 1,
      output: {
        statistics: importGraphBuffer(
          graph,
          'profile-statistics',
          statisticsBuffer,
          'float32',
          PROFILE_COLUMN_COUNT * GPU_COLUMN_PROFILE_STATISTIC_COUNT
        ),
        histograms: importGraphBuffer(
          graph,
          'profile-histograms',
          histogramsBuffer,
          'uint32',
          PROFILE_COLUMN_COUNT * HISTOGRAM_BINS
        )
      }
    })
  );
  const compiled = resources.track(await graph.compileAsync());

  let selectedValues = magnitude;
  let latestSignature = '';
  let readbackSignature = '';
  let dirty = true;
  let destroyed = false;
  let resultsCurrent = false;
  let activeValueRange: readonly [number, number] = [-1, 1];
  const summaryReader = new SummaryReader(
    resources,
    'standardization-summary',
    [
      {buffer: statisticsBuffer, size: statisticsBuffer.byteLength},
      {buffer: histogramsBuffer, size: histogramsBuffer.byteLength},
      {buffer: probeValuesBuffer, size: probeValuesBuffer.byteLength}
    ],
    bytes => {
      if (destroyed || readbackSignature !== latestSignature) return;
      const view = new DataView(bytes);
      const activeIndex = getOutputIndex(ctx.options.method, ctx.options.variance);
      const rawBase = 0;
      const outputBase = activeIndex * GPU_COLUMN_PROFILE_STATISTIC_COUNT * 4;
      const rawCount = view.getFloat32(rawBase + GPU_COLUMN_PROFILE_STATISTIC.count * 4, true);
      const rawNullCount = view.getFloat32(
        rawBase + GPU_COLUMN_PROFILE_STATISTIC.nullCount * 4,
        true
      );
      const rawMinimum = view.getFloat32(rawBase + GPU_COLUMN_PROFILE_STATISTIC.minimum * 4, true);
      const rawMaximum = view.getFloat32(rawBase + GPU_COLUMN_PROFILE_STATISTIC.maximum * 4, true);
      const rawMean = view.getFloat32(rawBase + GPU_COLUMN_PROFILE_STATISTIC.mean * 4, true);
      const rawDeviation = view.getFloat32(
        rawBase + GPU_COLUMN_PROFILE_STATISTIC.standardDeviation * 4,
        true
      );
      const outputCount = view.getFloat32(
        outputBase + GPU_COLUMN_PROFILE_STATISTIC.count * 4,
        true
      );
      const outputNullCount = view.getFloat32(
        outputBase + GPU_COLUMN_PROFILE_STATISTIC.nullCount * 4,
        true
      );
      const outputMinimum = view.getFloat32(
        outputBase + GPU_COLUMN_PROFILE_STATISTIC.minimum * 4,
        true
      );
      const outputMaximum = view.getFloat32(
        outputBase + GPU_COLUMN_PROFILE_STATISTIC.maximum * 4,
        true
      );
      const outputMean = view.getFloat32(outputBase + GPU_COLUMN_PROFILE_STATISTIC.mean * 4, true);
      const outputDeviation = view.getFloat32(
        outputBase + GPU_COLUMN_PROFILE_STATISTIC.standardDeviation * 4,
        true
      );
      const histogramOffset = PROFILE_COLUMN_COUNT * GPU_COLUMN_PROFILE_STATISTIC_COUNT * 4;
      const rawHistogram = Float32Array.from(
        new Uint32Array(bytes.slice(histogramOffset, histogramOffset + HISTOGRAM_BINS * 4))
      );
      const transformedHistogramOffset = histogramOffset + activeIndex * HISTOGRAM_BINS * 4;
      const transformedHistogram = Float32Array.from(
        new Uint32Array(
          bytes.slice(transformedHistogramOffset, transformedHistogramOffset + HISTOGRAM_BINS * 4)
        )
      );
      const probeOffset =
        PROFILE_COLUMN_COUNT * GPU_COLUMN_PROFILE_STATISTIC_COUNT * 4 +
        PROFILE_COLUMN_COUNT * HISTOGRAM_BINS * 4;
      const probeOriginal = view.getFloat32(probeOffset, true);
      const probeTransformed = view.getFloat32(probeOffset + activeIndex * 4, true);
      const outputHistogramRange: readonly [number, number] =
        Number.isFinite(outputMinimum) && Number.isFinite(outputMaximum)
          ? [
              outputMinimum,
              outputMaximum > outputMinimum
                ? outputMaximum
                : outputMinimum + Math.max(Math.abs(outputMinimum), 1) * 1e-6
            ]
          : [0, 1];
      const isCentered =
        ctx.options.method === 'deviation-from-mean' ||
        ctx.options.method === 'z-score' ||
        ctx.options.method === 'mad';
      const outputExtent =
        Number.isFinite(outputMinimum) && Number.isFinite(outputMaximum)
          ? Math.max(Math.abs(outputMinimum), Math.abs(outputMaximum))
          : 1;
      activeValueRange = isCentered
        ? [-Math.max(outputExtent, 1e-6), Math.max(outputExtent, 1e-6)]
        : Number.isFinite(outputMinimum) && Number.isFinite(outputMaximum)
          ? [outputMinimum, outputMaximum > outputMinimum ? outputMaximum : outputMinimum + 1]
          : [0, 1];
      ctx.setLegendExtent('standardized-values', activeValueRange);
      const fieldLabel = ctx.options.field === 'magnitude' ? 'Magnitude' : 'Reported depth';
      const fieldUnit = ctx.options.field === 'magnitude' ? 'M' : 'km';
      const formatOriginalValue = (value: number) =>
        Number.isFinite(value) ? value.toFixed(ctx.options.field === 'magnitude' ? 2 : 1) : '—';
      const formatTransformedValue = (value: number) =>
        Number.isFinite(value) ? value.toFixed(6) : '—';
      ctx.setReadout(
        'referenceCount',
        `${formatCount(rawCount)} finite of ${formatCount(rowCount)} archived rows · ${formatCount(rowCount - rawCount - rawNullCount)} outside the reference · ${getReferenceLabel(ctx.options.referencePopulation)}`
      );
      ctx.setReadout(
        'originalSummary',
        `${formatCount(rawCount)} finite · ${formatOriginalValue(rawMinimum)} to ${formatOriginalValue(rawMaximum)} ${fieldUnit} · mean ${formatOriginalValue(rawMean)}, population SD ${formatOriginalValue(rawDeviation)} · ${formatCount(rawNullCount)} null within population`
      );
      ctx.setReadout(
        'transformedSummary',
        `${formatCount(outputCount)} finite · ${formatTransformedValue(outputMinimum)} to ${formatTransformedValue(outputMaximum)} · mean ${formatTransformedValue(outputMean)}, population SD ${formatTransformedValue(outputDeviation)} · ${formatCount(outputNullCount)} null within population`
      );
      ctx.setReadout('activeFormula', getMethodLabel(ctx.options.method, ctx.options.variance));
      ctx.setReadout(
        'probeEvent',
        `Source row ${formatCount(probeRow)} · ${formatOriginalValue(probeOriginal)} ${fieldUnit} → ${formatTransformedValue(probeTransformed)} · ${getMethodLabel(ctx.options.method, ctx.options.variance)}`
      );
      ctx.setChart('originalDistribution', {
        kind: 'histogram',
        title: `Original ${fieldLabel.toLowerCase()}`,
        values: rawHistogram,
        xDomain: [rawMinimum, rawMaximum > rawMinimum ? rawMaximum : rawMinimum + 1],
        xLabel: `${fieldLabel} (${fieldUnit})`,
        yLabel: 'Earthquakes',
        height: 110,
        description: `GPU profile histogram of the original ${fieldLabel.toLowerCase()} values for the selected reference population.`
      });
      ctx.setChart('standardizedDistribution', {
        kind: 'histogram',
        title: 'Transformed values',
        values: transformedHistogram,
        xDomain: outputHistogramRange,
        xLabel: getMethodLabel(ctx.options.method, ctx.options.variance),
        yLabel: 'Earthquakes',
        height: 110,
        description: `GPU profile histogram of ${getMethodLabel(ctx.options.method, ctx.options.variance).toLowerCase()} values. Rows outside the selected reference population are excluded.`
      });
      resultsCurrent = true;
      ctx.requestLayers();
    }
  );

  const makeReferenceMask = (): Uint32Array => {
    const options = ctx.options;
    const referenceMask = new Uint32Array(rowCount);
    for (let row = 0; row < rowCount; row++) {
      const isEarthquake = eventType[row] === earthquakeTypeCode;
      // The reference population is a transparent, deterministic mask derived from source columns.
      const included =
        isEarthquake &&
        (options.referencePopulation === 'all-earthquakes' ||
          (options.referencePopulation === 'magnitude-6-plus' && magnitude[row] >= 6) ||
          (options.referencePopulation === 'shallow-70-km' && depth[row] <= 70));
      referenceMask[row] = included ? 1 : 0;
    }
    return referenceMask;
  };

  const probeRow = eventType.findIndex(
    (typeCode, row) => typeCode === earthquakeTypeCode && magnitude[row] >= 6 && depth[row] <= 70
  );
  // The shared probe belongs to every reference population and remains the same event as controls change.
  probeIndexBuffer.write(new Uint32Array([probeRow]));
  const writeInputs = () => {
    selectedValues = ctx.options.field === 'magnitude' ? magnitude : depth;
    valuesBuffer.write(selectedValues);
    const referenceMask = makeReferenceMask();
    maskBuffer.write(referenceMask);
  };
  writeInputs();
  const signatureForOptions = () => JSON.stringify(ctx.options);
  latestSignature = signatureForOptions();
  ctx.setReadout('referenceCount', 'GPU profile loading');
  ctx.setReadout('originalSummary', 'GPU profile loading');
  ctx.setReadout('transformedSummary', 'GPU standardization loading');

  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder) {
      const signature = signatureForOptions();
      latestSignature = signature;
      if (dirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        if (!summaryReader.isPending) readbackSignature = signature;
        summaryReader.request(commandEncoder);
      } else {
        if (!summaryReader.isPending) readbackSignature = signature;
        summaryReader.flush(commandEncoder);
      }
    },
    getLayers(): Layer[] {
      const layers: Layer[] = [
        new SpatialAnalysisPointLayer({
          id: 'earthquake-standardization-context',
          positions: positionBuffer,
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          instanceCount: rowCount,
          color: [110, 115, 128, 125],
          colormap: 'uniform',
          radiusPixels: 1.4,
          opacity: 0.1
        })
      ];
      if (resultsCurrent) {
        const outputIndex = getOutputIndex(ctx.options.method, ctx.options.variance);
        const centered =
          ctx.options.method === 'deviation-from-mean' ||
          ctx.options.method === 'z-score' ||
          ctx.options.method === 'mad';
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'earthquake-standardized-output',
            positions: positionBuffer,
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            instanceCount: rowCount,
            values: outputBuffers[outputIndex - 1],
            valueFormat: 'float32',
            colormap: centered ? 'diverging' : 'viridis',
            valueRange: activeValueRange,
            noDataColor: [0, 0, 0, 0],
            radiusPixels: 2.2,
            opacity: 0.9
          })
        );
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'earthquake-standardization-probe',
            positions: positionBuffer,
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            ids: probeIndexBuffer,
            instanceCount: 1,
            color: [17, 37, 74, 255],
            colormap: 'uniform',
            radiusPixels: 8,
            opacity: 1
          })
        );
      }
      return layers;
    },
    setOption() {
      latestSignature = signatureForOptions();
      resultsCurrent = false;
      writeInputs();
      dirty = true;
      summaryReader.markStale();
      ctx.setReadout('transformedSummary', 'Updating GPU result…');
      ctx.setReadout('probeEvent', 'Updating GPU probe…');
      ctx.setChart('originalDistribution', null);
      ctx.setChart('standardizedDistribution', null);
      ctx.requestLayers();
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    destroy() {
      destroyed = true;
      summaryReader.stop();
      resources.destroy();
    }
  };
}
