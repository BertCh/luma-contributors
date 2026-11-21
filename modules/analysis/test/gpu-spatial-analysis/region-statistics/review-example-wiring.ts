// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
/// <reference types="vite/client" />

// Execute the examples' actual constructor/selection expressions without loading their UI,
// datasets or rendering passes. Node and headless tests supply tiny physical buffer fixtures.
import explorerLasso from '../../../../../examples/deck/spatial-analysis-explorer/modes/lasso-mode.ts?raw';
import showcaseLasso from '../../../../../examples/deck/spatial-analysis-showcase/src/scenes/points/lasso-explorer.compute.ts?raw';
import explorerSegregation from '../../../../../examples/deck/spatial-analysis-explorer/modes/segregation-mode.ts?raw';
import showcaseSegregation from '../../../../../examples/deck/spatial-analysis-showcase/src/scenes/statistics/segregation.compute.ts?raw';
import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUPickRegionMask} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-pick-region-mask';
import {GPURegionMask} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-region-mask';
import type {
  GPURegionMaskSelection,
  GPURegionShape
} from '../../../src/gpu-spatial-analysis/region-statistics/region-statistics-types';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';

export const LASSO_EXAMPLE_SOURCES = [
  {name: 'explorer', source: explorerLasso},
  {name: 'showcase', source: showcaseLasso}
];
export const SEGREGATION_EXAMPLE_SOURCES = [
  {name: 'explorer', source: explorerSegregation},
  {name: 'showcase', source: showcaseSegregation}
];

function getObjectExpression(
  source: string,
  marker: string,
  start = 0
): {expression: string; end: number} {
  const markerStart = source.indexOf(marker, start);
  if (markerStart < 0) throw new Error(`Example construction marker missing: ${marker}`);
  const opening = source.indexOf('{', markerStart);
  let depth = 0;
  for (let index = opening; index < source.length; index++) {
    if (source[index] === '{') depth++;
    if (source[index] === '}' && --depth === 0)
      return {expression: source.slice(opening, index + 1), end: index + 1};
  }
  throw new Error('Unclosed example construction expression');
}

function evaluateExpression<Result>(expression: string, scope: Record<string, unknown>): Result {
  // All evaluated text is checked-in example code; fixture values are explicit parameters.
  return new Function(...Object.keys(scope), `return (${expression});`)(...Object.values(scope));
}

export function createExampleMaskSelection(
  device: Device,
  graph: GPUCommandGraph,
  source: string,
  picks: GraphDataView<'uint32'>,
  shape?: {positions: GraphDataView<'float32x2'>; region: GPURegionShape}
) {
  const maskBuffer = device.createBuffer({
    byteLength: 8,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const overflowBuffer = device.createBuffer({
    byteLength: 4,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const mask = importGraphBuffer(graph, 'example-mask', maskBuffer, 'uint32', 2);
  const scope = {
    graph,
    result: picks,
    pointCount: 2,
    extraMask: mask,
    mask,
    importGraphBuffer,
    pick: {mask: maskBuffer, overflow: overflowBuffer},
    set: {
      count: 2,
      pick: {mask: maskBuffer, overflow: overflowBuffer},
      regionMaskOverflow: overflowBuffer
    },
    positions: shape?.positions,
    positionsView: shape?.positions,
    bikeParking: {frame: {spatialContext: undefined}},
    projection: {spatialContext: undefined},
    graphKey: 'mask-polygon',
    polygonShape: () => shape!.region,
    regionMaskOverflow: overflowBuffer
  };
  const argument = getObjectExpression(
    source,
    shape ? 'new GPURegionMask(' : 'new GPUPickRegionMask('
  );
  const producer = shape
    ? new GPURegionMask(evaluateExpression(argument.expression, scope))
    : new GPUPickRegionMask(evaluateExpression(argument.expression, scope));
  const selection = getObjectExpression(source, 'selection =', argument.end);
  const region = evaluateExpression(selection.expression, {
    ...scope,
    maskProducer: producer
  }) as GPURegionMaskSelection;
  return {
    producer,
    region,
    overflowBuffer,
    destroy() {
      maskBuffer.destroy();
      overflowBuffer.destroy();
    }
  };
}

export function createExampleWeights(
  graph: GPUCommandGraph,
  source: string,
  buffers: {
    offsets: Buffer;
    neighbors: Buffer;
    weights: Buffer;
    distances: Buffer;
    overflow: Buffer;
    capacity: number;
  }
): GPUSpatialWeights {
  const scope = {
    graph,
    buffers,
    scaleBuffers: [buffers],
    scale: 0,
    unitCount: 2,
    n: 2,
    importGraphBuffer,
    projection: {spatialContext: undefined}
  };
  const expression = source.includes('const importWeights =')
    ? getObjectExpression(source, 'return', source.indexOf('const importWeights =')).expression
    : getObjectExpression(source, 'const weights: SearchWeights =').expression;
  return evaluateExpression(expression, scope);
}
