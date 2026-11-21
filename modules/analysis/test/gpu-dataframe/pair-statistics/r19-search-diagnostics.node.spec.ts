// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPURipleyDistanceFunctions} from '../../../src/dataframe-analysis/pair-statistics/gpu-ripley-distance-functions';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  fitVariogramModel,
  fitVariogramModelWithDiagnostics
} from '../../../src/dataframe-analysis/pair-statistics/variogram-model';
import {
  evaluateSpaceTimeVariogramModel,
  fitSpaceTimeVariogramModel
} from '../../../src/dataframe-analysis/pair-statistics/space-time-variogram-model';
import {GPUVariogramModelFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit';
import {GPUSpaceTimeVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-variogram';
import {WgslReflect, WgslExec, WgslParser} from 'wgsl_reflect';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {EXACT_HANISCH_SEARCH_WGSL} from '../../../src/dataframe-analysis/pair-statistics/exact-hanisch-search';

it('PTS-138 exact Hanisch searches beyond the reported radii with a geometric bound', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPURipleyDistanceFunctions({
      positions: createTransientView(graph, 'points', 'float32x2', 4),
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      gridSize: [10, 10],
      referenceGrid: [3, 3],
      radiusCount: 2,
      hanischSearch: 'exact',
      output: {g: createTransientView(graph, 'g', 'float32', 2)}
    }).getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    for (const kernel of sources) {
      expect(new WgslReflect(kernel.source).storage.length).toBeLessThanOrEqual(8);
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    }
    for (const suffix of ['-g-pairs', '-f-reference']) {
      const source = sources.find(kernel => kernel.id.endsWith(suffix))!.source;
      expect(source).toContain('getExactHanischDistance');
      expect(source).toContain('unvisitedDistance');
    }
  } finally {
    device.destroy();
  }
});

it('PTS-138 exact ring helper finds uncensored tails and excludes self on a hand-indexed lattice', () => {
  // Independent hand-index: [0,12]^2, cell width 1+2^-10, points (2,6),(3,6)
  // in cells 61,62. Supply the lattice directly so the test isolates the search helper.
  const offsets = new Uint32Array(145);
  offsets[62] = 1;
  offsets.fill(2, 63);
  const positionBits = new Uint32Array(Float32Array.from([2, 6, 3, 6]).buffer);
  const points = Uint32Array.from([
    positionBits[0],
    positionBits[1],
    0,
    positionBits[2],
    positionBits[3],
    1
  ]);
  for (const [positionX, excludedRow, border, expected] of [
    [6, 0xffffffff, 6, [3, 1]],
    [2, 0, 2, [1, 1]],
    [10, 0xffffffff, 2, [2, 0]]
  ] as const) {
    const result = new Float32Array(2);
    const source = `
const sortedPointsOffset:u32=0u;const cellOffsetsOffset:u32=0u;
@group(0) @binding(0) var<storage,read> sortedPoints:array<u32>;
@group(0) @binding(1) var<storage,read> cellOffsets:array<u32>;
@group(0) @binding(2) var<storage,read_write> result:array<f32>;
struct Lattice{minimumX:f32,minimumY:f32,cellWidth:f32,cellHeight:f32,columns:u32,rows:u32}
fn getCellColumn(lattice:Lattice,x:f32)->u32{return min(u32(floor(x/lattice.cellWidth)),11u);}
fn getCellRow(lattice:Lattice,y:f32)->u32{return min(u32(floor(y/lattice.cellHeight)),11u);}
${EXACT_HANISCH_SEARCH_WGSL}
@compute @workgroup_size(1) fn main(){
 let lattice=Lattice(0.0,0.0,1.0009765625,1.0009765625,12u,12u);
 let nearest=getExactHanischDistance(lattice,vec2f(${positionX}.0,6.0),${excludedRow}u,${border}.0);
 result[0]=nearest.x;result[1]=nearest.y;
}`;
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [1, 1, 1], {
      0: {0: points, 1: offsets, 2: result}
    });
    expect(Array.from(result)).toEqual(expected);
  }
});

it('scale eligibility includes finite positions inside the window before quantiles (R19)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `eligibility-${serial++}`, format, length);
    const nodes = new GPUSpaceTimeVariogram({
      positions: view('float32x2', 16),
      values: view('float32', 16),
      secondValues: view('float32', 16),
      times: view('float32', 16),
      mask: view('uint32', 16),
      parameters: view('float32', 8),
      lagCount: 1,
      temporalLagCount: 1,
      gridSize: [4, 4],
      output: {
        semivariances: view('float32', 4),
        pairCounts: view('uint32', 4),
        status: view('uint32', 4)
      }
    }).getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    expect(sources.find(kernel => kernel.id.endsWith('-complete-case'))!.source).toContain(
      'isPointValid(readLattice(), point.x, point.y)'
    );
    for (const kernel of sources) {
      expect(new WgslReflect(kernel.source).storage.length, kernel.id).toBeLessThanOrEqual(8);
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});

it('GPU fit topology errors identify the view, actual and expected bin counts', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const name of ['semivariances', 'pairCounts'] as const) {
      const graph = new GPUCommandGraph(device);
      const props = {
        id: 'fit-topology',
        distances: createTransientView(graph, 'distances', 'float32', 3),
        semivariances: createTransientView(
          graph,
          'semivariances',
          'float32',
          name === 'semivariances' ? 2 : 3
        ),
        pairCounts: createTransientView(graph, 'counts', 'uint32', name === 'pairCounts' ? 2 : 3),
        output: {
          parameters: createTransientView(graph, 'model', 'float32x4', 3),
          diagnostics: createTransientView(graph, 'diagnostics', 'float32x4', 3),
          status: createTransientView(graph, 'status', 'uint32', 3)
        }
      };
      expect(() => new GPUVariogramModelFit(props)).toThrow(
        `fit-topology invalid model or bin topology: ${name} bin count 2 must equal distances bin count 3`
      );
    }
  } finally {
    device.destroy();
  }
});

it('PTS-113 public fit diagnostics distinguish boundary, unrefined and interior fits', () => {
  const input = {
    distances: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    semivariances: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    pairCounts: Array<number>(10).fill(1)
  };
  const boundary = fitVariogramModelWithDiagnostics(input, {
    model: 'spherical',
    weighting: 'none'
  });
  expect(boundary.atBoundary).toBe(true);
  expect(boundary.converged).toBe(false);
  expect(boundary.model.range).toBeCloseTo(30, 8);
  // Exponential gamma = 0.2 + 2*(1-exp(-h)), effective range 3; literals generated
  // independently with /usr/bin/python3 math.exp, h=1..10.
  const interiorInput = {
    ...input,
    semivariances: [
      1.4642411176571153, 1.9293294335267746, 2.100425863264272, 2.1633687222225317,
      2.186524106001829, 2.195042495646667, 2.198176236068891, 2.199329074744195, 2.199753180391827,
      2.199909200140475
    ]
  };
  const coarse = fitVariogramModelWithDiagnostics(interiorInput, {
    model: 'exponential',
    weighting: 'none',
    maximumIterations: 0
  });
  expect(coarse.atBoundary).toBe(false);
  expect(coarse.logRangeBracket).toBeGreaterThan(1e-5);
  expect(coarse.converged).toBe(false);
  const interior = fitVariogramModelWithDiagnostics(interiorInput, {
    model: 'exponential',
    weighting: 'none'
  });
  expect(interior.atBoundary).toBe(false);
  expect(interior.converged).toBe(true);
  expect(Number.isFinite(interior.model.residual)).toBe(true);
  expect(interior.model.range).toBeCloseTo(3, 5);
  expect(fitVariogramModel(interiorInput, {model: 'exponential', weighting: 'none'})).toEqual(
    interior.model
  );
});

it('PTS-116 reports actual cycles and convergence independently of search boundaries', () => {
  const flat = fitSpaceTimeVariogramModel(
    {distances: [1, 2, 3], timeLags: [0, 1, 2], semivariances: [1, 1, 1], pairCounts: [1, 1, 1]},
    {type: 'metric', weighting: 'none'}
  );
  expect(flat.cycleCount).toBe(1);
  expect(flat.converged).toBe(true);
  // Identifiable separable surface gamma(h,u)=3*(1-exp(-h-u/2)).
  // Values from Python math.exp; separate spatial and temporal axes identify both ranges.
  const input = {
    distances: [1, 2, 4, 0, 0, 0, 1, 2, 4],
    timeLags: [0, 0, 0, 1, 2, 4, 1, 2, 4],
    semivariances: [
      1.896361676485673, 2.593994150290162, 2.9450530833337973, 1.1804080208620997,
      1.896361676485673, 2.593994150290162, 2.3306095195547107, 2.8506387948964083,
      2.992563743470001
    ],
    pairCounts: Array<number>(9).fill(1)
  };
  const limited = fitSpaceTimeVariogramModel(input, {
    type: 'metric',
    weighting: 'none',
    maximumCycles: 1
  });
  expect(limited.cycleCount).toBe(1);
  expect(limited.converged).toBe(false);
  const fitted = fitSpaceTimeVariogramModel(input, {type: 'separable', weighting: 'none'});
  const exhausted = fitSpaceTimeVariogramModel(input, {type: 'sum-metric', weighting: 'none'});
  expect(exhausted.cycleCount).toBe(24);
  expect(exhausted.converged).toBe(false);
  expect(fitted.cycleCount).toBeLessThanOrEqual(24);
  for (let bin = 0; bin < input.distances.length; bin++) {
    const actual = evaluateSpaceTimeVariogramModel(
      fitted.model,
      input.distances[bin],
      input.timeLags[bin]
    );
    expect(Number.isFinite(actual)).toBe(true);
    expect(actual).toBeCloseTo(input.semivariances[bin], 3);
  }
});
