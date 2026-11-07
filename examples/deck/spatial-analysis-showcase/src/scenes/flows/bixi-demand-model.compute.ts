// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer, type CommandEncoder} from '@luma.gl/core';
import {
  GPUSpatialInteraction,
  type GPUSpatialInteractionProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  BIXI_MODEL_ORIGIN,
  findBixiStationAtPixel,
  makeBixiModelData,
  type BixiModelData
} from './bixi-model-data';
import {readBixiFlows} from './bixi-data';

const CONSTRAINTS = ['unconstrained', 'origin', 'destination', 'doubly'] as const;
const COEFFICIENTS = [-1.5, -1.3125, -1.125, -0.9375, -0.75, -0.5625, -0.375, -0.1875, 0];
const CALIBRATIONS = ['grid', 'bounded'] as const;
const MODEL_KEYS = CONSTRAINTS.flatMap(constraint =>
  CALIBRATIONS.map(calibration => `${constraint}:${calibration}` as const)
);

export type BixiDemandModelOptions = {
  constraint: (typeof CONSTRAINTS)[number];
  calibration: (typeof CALIBRATIONS)[number];
  display: 'observed' | 'predicted' | 'residual';
};

type ModelBuffers = {
  predictions: Buffer;
  summary: Buffer;
  status: Buffer;
  candidates: Buffer;
  calibration?: Buffer;
  originResiduals: Buffer;
  destinationResiduals: Buffer;
};

type FitSummary = {
  coefficient: number;
  deviance: number;
  marginalError: number;
  balancingIterations: number;
  status: number;
  bracketWidth: number | null;
  originResiduals: Float32Array;
  destinationResiduals: Float32Array;
  candidateDeviances: Float32Array;
};

const CONSTRAINT_LABELS: Record<BixiDemandModelOptions['constraint'], string> = {
  unconstrained: 'unconstrained',
  origin: 'origin constrained',
  destination: 'destination constrained',
  doubly: 'doubly constrained'
};

/** Four constraint forms, with grid and bounded profile calibration, over observed monthly pairs. */
export async function createBixiDemandModel(
  context: SceneContext<BixiDemandModelOptions>
): Promise<SceneInstance<BixiDemandModelOptions>> {
  const dataset = context.datasets.get('bixi-flows');
  const data = makeBixiModelData(readBixiFlows(dataset));
  const resources = new SpatialAnalysisResources(context.device, 'bixi-demand-model');
  const input = {
    origins: resources.createBuffer('origins', data.origins),
    destinations: resources.createBuffer('destinations', data.destinations),
    flows: resources.createBuffer('observed-flows', data.flows, Buffer.VERTEX),
    covariate: resources.createBuffer('distance-km', data.distancesKm),
    pairState: resources.createBuffer('pair-state', data.pairState)
  };
  const outputs = new Map<string, ModelBuffers>();
  const graph = new GPUCommandGraph<void>(context.device, {id: 'bixi-demand-model'});
  const importBuffer = createGraphBufferImporter(graph);
  const inputViews = {
    origins: importBuffer(input.origins, 'uint32', data.flows.length),
    destinations: importBuffer(input.destinations, 'uint32', data.flows.length),
    flows: importBuffer(input.flows, 'float32', data.flows.length),
    covariate: importBuffer(input.covariate, 'float32', data.flows.length),
    pairState: importBuffer(input.pairState, 'uint32', data.flows.length)
  };
  for (const constraint of CONSTRAINTS) {
    for (const calibration of CALIBRATIONS) {
      const key = `${constraint}:${calibration}`;
      const fitBuffers: ModelBuffers = {
        predictions: resources.createBuffer(
          `${key}-predictions`,
          data.flows.length * 4,
          Buffer.VERTEX
        ),
        summary: resources.createBuffer(`${key}-summary`, new Float32Array(4)),
        status: resources.createBuffer(`${key}-status`, new Uint32Array(1)),
        candidates: resources.createBuffer(
          `${key}-candidates`,
          new Float32Array((calibration === 'grid' ? COEFFICIENTS.length : 2) * 4)
        ),
        ...(calibration === 'bounded'
          ? {calibration: resources.createBuffer(`${key}-bracket`, new Float32Array(4))}
          : {}),
        originResiduals: resources.createBuffer(
          `${key}-origin-residuals`,
          new Float32Array(data.stationIds.length),
          Buffer.VERTEX
        ),
        destinationResiduals: resources.createBuffer(
          `${key}-destination-residuals`,
          new Float32Array(data.stationIds.length),
          Buffer.VERTEX
        )
      };
      outputs.set(key, fitBuffers);
      const view = <Format extends GPUVectorFormat>(
        buffer: Buffer,
        format: Format,
        length: number
      ) => importBuffer(buffer, format, length);
      const props: GPUSpatialInteractionProps = {
        id: `bixi-${key}`,
        ...inputViews,
        originCount: data.stationIds.length,
        destinationCount: data.stationIds.length,
        constraint,
        coefficients: calibration === 'grid' ? COEFFICIENTS : [-1.5, 0],
        calibration,
        maximumCalibrationIterations: 32,
        coefficientTolerance: 1e-5,
        maximumIterations: 512,
        tolerance: 1e-5,
        output: {
          predictions: view(fitBuffers.predictions, 'float32', data.flows.length),
          summary: view(fitBuffers.summary, 'float32', 4),
          status: view(fitBuffers.status, 'uint32', 1),
          candidates: view(
            fitBuffers.candidates,
            'float32',
            (calibration === 'grid' ? COEFFICIENTS.length : 2) * 4
          ),
          ...(fitBuffers.calibration
            ? {calibration: view(fitBuffers.calibration, 'float32', 4)}
            : {}),
          originResiduals: view(fitBuffers.originResiduals, 'float32', data.stationIds.length),
          destinationResiduals: view(
            fitBuffers.destinationResiduals,
            'float32',
            data.stationIds.length
          )
        }
      };
      graph.add(new GPUSpatialInteraction(props));
    }
  }
  const compiled = resources.track(await graph.compileAsync());
  const segmentBuffer = resources.createBuffer('od-segments', createSegments(data), Buffer.VERTEX);
  const stationBuffer = resources.createBuffer('station-positions', data.positions, Buffer.VERTEX);
  const segmentCount = data.origins.length;
  const stationCount = data.stationIds.length;
  const summaries = new Map<string, FitSummary>();
  let dirty = true;
  let destroyed = false;
  let latestModelData = new Map<string, {predictions: Float32Array; candidates: Float32Array}>();
  let flowMaximum = Math.max(1, ...data.flows);
  let residualMaximum = 1;
  const readerSources = MODEL_KEYS.flatMap(key => {
    const output = outputs.get(key)!;
    return [
      {buffer: output.summary, size: 16},
      {buffer: output.status, size: 4},
      ...(output.calibration ? [{buffer: output.calibration, size: 16}] : []),
      {buffer: output.originResiduals, size: data.stationIds.length * 4},
      {buffer: output.destinationResiduals, size: data.stationIds.length * 4},
      {buffer: output.candidates, size: output.candidates.byteLength},
      {buffer: output.predictions, size: data.flows.length * 4}
    ];
  });
  const reader = new SummaryReader(resources, 'bixi-demand-model', readerSources, bytes => {
    if (destroyed) return;
    const floats = new Float32Array(bytes);
    const words = new Uint32Array(bytes);
    let offset = 0;
    const nextModelData = new Map<string, {predictions: Float32Array; candidates: Float32Array}>();
    for (const key of MODEL_KEYS) {
      const hasBracket = key.endsWith(':bounded');
      const candidateCount = hasBracket ? 2 : COEFFICIENTS.length;
      const summary = floats.slice(offset / 4, offset / 4 + 4);
      const status = words[offset / 4 + 4];
      offset += 20;
      let bracketWidth: number | null = null;
      if (hasBracket) {
        bracketWidth = floats[offset / 4 + 2];
        offset += 16;
      }
      const originResiduals = floats.slice(offset / 4, offset / 4 + data.stationIds.length);
      offset += data.stationIds.length * 4;
      const destinationResiduals = floats.slice(offset / 4, offset / 4 + data.stationIds.length);
      offset += data.stationIds.length * 4;
      const candidateValues = floats.slice(offset / 4, offset / 4 + candidateCount * 4);
      offset += candidateCount * 16;
      const predictions = floats.slice(offset / 4, offset / 4 + data.flows.length);
      offset += data.flows.length * 4;
      const fitSummary: FitSummary = {
        coefficient: summary[0],
        deviance: summary[1],
        marginalError: summary[2],
        balancingIterations: summary[3],
        status,
        bracketWidth,
        originResiduals,
        destinationResiduals,
        candidateDeviances: Float32Array.from(
          {length: candidateCount},
          (_, index) => candidateValues[index * 4]
        )
      };
      summaries.set(key, fitSummary);
      nextModelData.set(key, {predictions, candidates: candidateValues});
    }
    latestModelData = nextModelData;
    updateCharts(
      latestModelData.get(selectedKey(context.options)),
      summaries.get(selectedKey(context.options))
    );
    updateReadouts();
    context.requestLayers();
  });
  const selectedKey = (options: BixiDemandModelOptions) =>
    `${options.constraint}:${options.calibration}`;
  const updateReadouts = () => {
    context.setReadout(
      'support',
      `${data.observedPairCount} observed pairs of ${data.flows.length} candidate cells`
    );
    context.setReadout('missingPairs', data.missingPairCount);
    const key = selectedKey(context.options);
    const fitSummary = summaries.get(key);
    if (!fitSummary) return;
    context.setReadout('fitStatus', statusLabel(fitSummary.status));
    context.setReadout('constraintLabel', CONSTRAINT_LABELS[context.options.constraint]);
    if (fitSummary.status !== 0) {
      clearFitReadouts();
      context.setReadout('coefficient', 'withheld');
      context.setReadout('deviance', 'withheld');
      context.setReadout('marginalError', 'not converged');
      return;
    }
    context.setReadout('coefficient', `${fitSummary.coefficient.toFixed(3)} / km`);
    context.setReadout('deviance', fitSummary.deviance.toFixed(1));
    context.setReadout('marginalError', fitSummary.marginalError.toExponential(2));
    context.setReadout('balancingIterations', Math.round(fitSummary.balancingIterations));
    context.setReadout(
      'boundedWidth',
      fitSummary.bracketWidth === null
        ? 'grid candidates'
        : `${fitSummary.bracketWidth.toExponential(2)} / km`
    );
  };
  const clearFitReadouts = () => {
    for (const readout of [
      'coefficient',
      'deviance',
      'marginalError',
      'balancingIterations',
      'boundedWidth'
    ]) {
      context.setReadout(readout, 'withheld');
    }
  };
  const updateCharts = (
    fitData: {predictions: Float32Array; candidates: Float32Array} | undefined,
    fitSummary: FitSummary | undefined
  ) => {
    if (!fitData || !fitSummary || fitSummary.status !== 0) {
      for (const chart of [
        'pairCounts',
        'marginalResiduals',
        'marginalFit',
        'coefficientProfile'
      ]) {
        context.setChart(chart, null);
      }
      return;
    }
    const observed = Float32Array.from(data.flows, (value, pair) =>
      data.pairState[pair] === 1 ? value : Number.NaN
    );
    const predicted = Float32Array.from(fitData.predictions, (value, pair) =>
      data.pairState[pair] === 1 ? value : Number.NaN
    );
    const selectedFlowValues =
      context.options.display === 'observed' ? data.flows : fitData.predictions;
    flowMaximum = Math.max(
      1,
      ...Array.from(selectedFlowValues, (value, pair) =>
        data.pairState[pair] === 1 && Number.isFinite(value) ? value : 0
      )
    );
    const residuals =
      context.options.constraint === 'destination'
        ? fitSummary.destinationResiduals
        : fitSummary.originResiduals;
    const observedMargins = new Float32Array(data.stationIds.length);
    const fittedMargins = new Float32Array(data.stationIds.length);
    for (let pair = 0; pair < data.flows.length; pair++) {
      if (data.pairState[pair] !== 1) continue;
      const station =
        context.options.constraint === 'destination' ? data.destinations[pair] : data.origins[pair];
      observedMargins[station] += data.flows[pair];
      fittedMargins[station] += fitData.predictions[pair];
    }
    residualMaximum = Math.max(1, ...Array.from(residuals, Math.abs));
    context.setLegendExtent('bixi-model-flows', [0, flowMaximum]);
    context.setLegendExtent('bixi-model-residuals', [-residualMaximum, residualMaximum]);
    const pairIndexes = Float32Array.from({length: data.flows.length}, (_, index) => index);
    context.setChart('pairCounts', {
      kind: 'line',
      xLabel: 'observed pair index',
      yLabel: 'rides in August',
      series: [
        {label: 'observed', x: pairIndexes, y: observed},
        {label: 'fitted', x: pairIndexes, y: predicted}
      ]
    });
    context.setChart('marginalResiduals', {
      kind: 'bars',
      values: Array.from(
        context.options.constraint === 'destination'
          ? fitSummary.destinationResiduals
          : fitSummary.originResiduals
      ),
      labels: data.stationNames
    });
    context.setChart('marginalFit', {
      kind: 'line',
      xLabel: 'sample station index',
      yLabel: 'rides in August',
      series: [
        {label: 'observed margin', y: observedMargins},
        {label: 'fitted margin', y: fittedMargins}
      ]
    });
    if (context.options.calibration === 'grid') {
      context.setChart('coefficientProfile', {
        kind: 'line',
        xLabel: 'distance coefficient / km',
        yLabel: 'Poisson deviance',
        series: [{label: 'profile deviance', x: COEFFICIENTS, y: fitSummary.candidateDeviances}]
      });
    } else context.setChart('coefficientProfile', null);
  };
  const selectedOutput = () => outputs.get(selectedKey(context.options))!;
  const coordinateOrigin: [number, number, number] = [...BIXI_MODEL_ORIGIN, 0];
  const layers = (): Layer[] => {
    const selectedSummary = summaries.get(selectedKey(context.options));
    if (context.options.display !== 'observed' && selectedSummary?.status !== 0) return [];
    const showLinks = context.options.display !== 'residual';
    const residualRange: [number, number] = [-residualMaximum, residualMaximum];
    const stationValues =
      context.options.constraint === 'destination'
        ? selectedOutput().destinationResiduals
        : selectedOutput().originResiduals;
    return [
      ...(showLinks
        ? [
            new SpatialAnalysisSegmentLayer({
              id: 'bixi-model-pairs',
              coordinateOrigin,
              segments: segmentBuffer,
              instanceCount: segmentCount,
              values:
                context.options.display === 'observed' ? input.flows : selectedOutput().predictions,
              valueFormat: 'float32',
              colormap: context.options.display === 'observed' ? 'cividis' : 'blues',
              valueRange: [0, flowMaximum],
              noDataColor: [0, 0, 0, 0],
              discardAtOrBelow: 0,
              widthPixels: 1.3,
              opacity: context.options.display === 'observed' ? 0.64 : 0.8
            })
          ]
        : []),
      new SpatialAnalysisPointLayer({
        id: 'bixi-model-stations',
        coordinateOrigin,
        positions: stationBuffer,
        instanceCount: stationCount,
        values: context.options.display === 'residual' ? stationValues : undefined,
        valueFormat: context.options.display === 'residual' ? 'float32' : 'none',
        colormap: context.options.display === 'residual' ? 'diverging' : 'uniform',
        ...(context.options.display === 'residual'
          ? {valueRange: residualRange}
          : {color: [34, 95, 142, 255]}),
        radiusPixels: 5,
        outlineWidthPixels: 1
      })
    ];
  };
  function requestSummary(commandEncoder: CommandEncoder) {
    if (dirty) {
      compiled.encode(commandEncoder, {parameters: undefined});
      reader.request(commandEncoder);
      dirty = false;
    } else reader.flush(commandEncoder);
  }
  return {
    getCompiledGraphs: () => [compiled],
    encode: commandEncoder => requestSummary(commandEncoder),
    getLayers: layers,
    getTooltip(event) {
      const viewport = context.getViewport();
      if (!viewport) return null;
      const station = findBixiStationAtPixel(data, viewport, event.pixel);
      if (station < 0) return null;
      const fit = summaries.get(selectedKey(context.options));
      return {
        title: data.stationNames[station],
        subtitle: 'Observed BIXI station · conditional OD model',
        rows: [
          {label: 'August departures + arrivals', value: data.activity[station].toLocaleString()},
          {
            label: 'Origin margin residual',
            value: fit?.status === 0 ? fit.originResiduals[station].toFixed(2) : 'withheld'
          },
          {
            label: 'Destination margin residual',
            value: fit?.status === 0 ? fit.destinationResiduals[station].toFixed(2) : 'withheld'
          }
        ],
        note: 'Model margins use only retained pairs inside this 16-station sample. Station activity also includes trips to other stations.'
      };
    },
    setOption: () => {
      context.requestLayers();
      updateCharts(
        latestModelData.get(selectedKey(context.options)),
        summaries.get(selectedKey(context.options))
      );
      updateReadouts();
    },
    destroy: () => {
      destroyed = true;
      reader.release();
      resources.destroy();
    }
  };
}

function createSegments(data: BixiModelData): Float32Array {
  const segments = new Float32Array(data.flows.length * 4);
  for (let pair = 0; pair < data.flows.length; pair++) {
    const origin = data.origins[pair];
    const destination = data.destinations[pair];
    segments.set(
      [
        data.positions[origin * 2],
        data.positions[origin * 2 + 1],
        data.positions[destination * 2],
        data.positions[destination * 2 + 1]
      ],
      pair * 4
    );
  }
  return segments;
}

function statusLabel(status: number): string {
  return (
    [
      'converged',
      'no observed mass',
      'invalid input',
      'no converged coefficient',
      'bounded bracket exhausted'
    ][status] ?? `unknown status ${status}`
  );
}
