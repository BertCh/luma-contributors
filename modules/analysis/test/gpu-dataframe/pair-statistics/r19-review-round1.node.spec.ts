// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {EXACT_HANISCH_SEARCH_WGSL} from '../../../src/dataframe-analysis/pair-statistics/exact-hanisch-search';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUPointPatternIndices} from '../../../src/dataframe-analysis/pair-statistics/gpu-point-pattern-indices';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createRipleyTestWindow} from './ripley-envelope-fixture';
import {GPUSpaceTimeKFunction} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-k-function';

function getIndicesSources(polygon = false) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `review-${serial++}`, format, length);
  const nodes = new GPUPointPatternIndices({
    positions: view('float32x2', 2),
    parameters: view('float32', 8),
    gridSize: [4, 4],
    quadratGrid: [7, 1],
    studyWindow: polygon ? createRipleyTestWindow(graph) : undefined,
    output: {
      quadratCounts: view('uint32', 7),
      ...(polygon ? {quadratStatistics: view('float32', 7)} : {})
    }
  }).getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  device.destroy();
  return sources;
}

it('earlier quadrat fix preserves the immediate predecessor of a representable edge', () => {
  const source = getIndicesSources().find(kernel => kernel.id.endsWith('-quadrat-counts'))!.source;
  const start = source.indexOf('fn getQuadratIndex(');
  let end = source.indexOf('{', start) + 1;
  let depth = 1;
  while (depth) {
    if (source[end] === '{') depth++;
    if (source[end] === '}') depth--;
    end++;
  }
  const helpers = source.slice(
    source.indexOf('// QUADRAT_PRODUCT_HELPERS') >= 0
      ? source.indexOf('// QUADRAT_PRODUCT_HELPERS')
      : start,
    end
  );
  const actual: number[] = [];
  for (const offset of [2396754.75]) {
    const probe = `${helpers}\n@group(0) @binding(0) var<storage,read_write> result:array<u32>;@compute @workgroup_size(1) fn main(){result[0]=getQuadratIndex(${offset.toFixed(2)},3355457.0,7u);}`;
    const result = new Uint32Array(1);
    new WgslExec(new WgslParser().parse(probe)).dispatchWorkgroups('main', [1, 1, 1], {
      0: {0: result}
    });
    actual.push(result[0]);
  }
  // Exact integer products: 2396754.75*7=16777283.25 < 3355457*5=16777285.
  // The edge itself is exactly 3355457*5/7=2396755.
  expect(actual).toEqual([4]);
});

it('earlier quadrat invalidation has one statistics writer after an atomic flag reduction', () => {
  const sources = getIndicesSources(true);
  const parallel = sources.find(kernel => kernel.id.endsWith('-quadrat-unsampled'))!.source;
  expect(parallel).not.toContain('quadratStatistics');
  expect(parallel).toContain('atomicOr');
  const finish = sources.find(kernel => kernel.id.endsWith('-quadrat-invalid-finish'))!;
  expect(finish.constants['INVOCATION_COUNT']).toBe(1);
  expect(finish.source).toContain('quadratStatistics');
});

it('earlier space-time fixes keep PCF scale distinct and propagate unsupported intensity scales', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `scale-review-${serial++}`, format, length);
  const sources = getGPUCommandNodeKernelSources(
    new GPUSpaceTimeKFunction({
      positions: view('float32x2', 2),
      times: view('float32', 2),
      intensities: view('float32', 2),
      parameters: view('float32', 11),
      gridSize: [4, 4],
      distanceClassCount: 1,
      timeClassCount: 1,
      edgeCorrection: 'none',
      pairCorrelationKernel: 'epanechnikov',
      output: {
        k: view('float32', 1),
        pairCorrelation: view('float32', 1),
        status: view('uint32', 1)
      }
    }).getCommandNodes(graph)
  );
  const pairs = sources.find(kernel => kernel.id.endsWith('-pairs'))!.source;
  expect(pairs).toContain('PAIR_CORRELATION_SCALE');
  expect(pairs).toContain('scaledCorrelation < 0.5');
  expect(sources.find(kernel => kernel.id.endsWith('-row-table'))!.source).toContain(
    'inverseMinimumOffset + 1u'
  );
  expect(sources.find(kernel => kernel.id.endsWith('-finish'))!.source).toContain('inputLoss');
  device.destroy();
});

it('B1 exact tail search includes a neighbor assigned across a rounded cell boundary', () => {
  const offsets = new Uint32Array(257);
  offsets.fill(1, 128);
  offsets.fill(4, 129);
  const bits = new Uint32Array(
    Float32Array.from([-1 / 8, 15 / 128, -1 / 64, 15 / 128, 3, 15 / 128, 3 + 1 / 256, 15 / 128])
      .buffer
  );
  const points = Uint32Array.from([
    bits[0],
    bits[1],
    0,
    bits[2],
    bits[3],
    1,
    bits[4],
    bits[5],
    2,
    bits[6],
    bits[7],
    3
  ]);
  const result = new Float32Array(2);
  const source = `
const sortedPointsOffset:u32=0u;const cellOffsetsOffset:u32=0u;
@group(0) @binding(0) var<storage,read> sortedPoints:array<u32>;
@group(0) @binding(1) var<storage,read> cellOffsets:array<u32>;
@group(0) @binding(2) var<storage,read_write> result:array<f32>;
struct Lattice{minimumX:f32,minimumY:f32,maximumX:f32,maximumY:f32,cellWidth:f32,cellHeight:f32,columns:u32,rows:u32}
fn getCellColumn(lattice:Lattice,x:f32)->u32{return 127u;}
fn getCellRow(lattice:Lattice,y:f32)->u32{return 0u;}
${EXACT_HANISCH_SEARCH_WGSL}
@compute @workgroup_size(1) fn main(){
 let lattice=Lattice(-1048576.0,0.0,1048576.0,1024.0,8192.0,1024.0,256u,1u);
 let nearest=getExactHanischDistance(lattice,vec2f(-0.125,0.1171875),0u,0.1171875);
 result[0]=nearest.x;result[1]=nearest.y;
}`;
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [1, 1, 1], {
    0: {0: points, 1: offsets, 2: result}
  });
  expect(Array.from(result)).toEqual([0.109375, 1]);
});

it('S1 exact rings enumerate four sides without rescanning interior rows', () => {
  expect(EXACT_HANISCH_SEARCH_WGSL).toContain('side < 4u');
  expect(EXACT_HANISCH_SEARCH_WGSL).not.toContain('cellRow = firstRow; cellRow <= lastRow');
});
