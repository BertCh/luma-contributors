// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {VertexFormat} from '@luma.gl/core';
import typescript from 'typescript';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/gpu-raster-grid';
import {GPUSpaceTimeKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-space-time-kernel-density';
import {getGPUSpaceTimeKernelDensityParameterValues} from '../../../src/gpu-spatial-analysis/point-density/space-time-kernel-density-math';
import {GPUSpaceTimeKFunction} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-k-function';
import {getGPUSpaceTimeKFunctionParameterValues} from '../../../src/dataframe-analysis/pair-statistics/space-time-k-function-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const GUIDE_PATH = 'docs/api/gpu-spatial-analysis/space-time-density-and-k-function.md';
const readGuide = () => readFileSync(GUIDE_PATH, 'utf8');

/** Execute the actual fence with typed declarations supplied by the fixture; no typecheck/build. */
function executeGuideFence(marker: string, density: boolean): Float32Array {
  const code = [...readGuide().matchAll(/```ts\n([\s\S]*?)\n```/g)].find(match =>
    match[1].includes(marker)
  )![1];
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters = createTransientView(graph, 'parameters', 'float32', density ? 14 : 11);
  let packed: Float32Array = new Float32Array();
  const inputs: Record<string, unknown> = {
    graph,
    parameterBuffer: {
      importToGraph: () => parameters,
      write: (values: Float32Array) => {
        packed = values;
      }
    }
  };
  const lengths: Record<string, number> = {
    positions: 2,
    times: 2,
    weights: 2,
    intensities: 2,
    mask: 2,
    values: 256 * 256 * 64,
    weightSum: 1,
    invalidCount: 1,
    k: 56,
    spatialK: 8,
    temporalK: 7,
    d: 56,
    pairCorrelation: 56,
    pairCounts: 56,
    local: 112,
    statusView: 1
  };
  for (const match of code.matchAll(/declare const (\w+): GraphDataView<'([^']+)'>;/g)) {
    inputs[match[1]] = createTransientView(
      graph,
      match[1],
      match[2] as VertexFormat,
      lengths[match[1]]
    );
  }
  const exports = {
    makeGPURasterMetadata,
    GPUSpaceTimeKernelDensity,
    getGPUSpaceTimeKernelDensityParameterValues,
    GPUSpaceTimeKFunction,
    getGPUSpaceTimeKFunctionParameterValues
  };
  const executable = typescript.transpileModule(code, {
    compilerOptions: {
      target: typescript.ScriptTarget.ES2022,
      module: typescript.ModuleKind.CommonJS
    }
  }).outputText;
  try {
    new Function('require', 'exports', ...Object.keys(inputs), executable)(
      () => exports,
      {},
      ...Object.values(inputs)
    );
    return packed;
  } finally {
    device.destroy();
  }
}

it('R1 B3 homogeneous K guide fence constructs with all advertised marginal outputs', () => {
  expect(executeGuideFence('new GPUSpaceTimeKFunction', false)).toHaveLength(11);
});

it('R1 B4 KDE guide placement covers its advertised positive sample domain', () => {
  const packed = executeGuideFence('new GPUSpaceTimeKernelDensity', true);
  // Independent intended rectangle from indexBounds in the fence: [0,0,256000,256000].
  expect(packed).toHaveLength(14);
  expect([
    Math.min(packed[0], packed[2]),
    Math.min(packed[1], packed[3]),
    Math.max(packed[0], packed[2]),
    Math.max(packed[1], packed[3])
  ]).toEqual([0, 0, 256000, 256000]);
});

it('R1 B8 guide distinguishes public count saturation from weighted loss', () => {
  // Ordered nonself pairs of 65,537 events exceed a uint32 output by 65,537.
  expect(65_537 * 65_536).toBe(4_295_032_832);
  expect(readGuide().includes('saturates at 4,294,967,295')).toBe(true);
  expect(readGuide().includes('Count-port saturation alone does not set status 3')).toBe(true);
});

it('R1 S4 guide states row admission separately from the minimum-derived scale', () => {
  // The minimum inverse 1 chooses exponent 16; another included inverse 2^59 needs -102.
  expect(16 - 2 * 59).toBe(-102);
  expect(readGuide().includes('every included row')).toBe(true);
  expect(readGuide().includes('separate from the minimum')).toBe(true);
});

it('R1 earlier-fix note qualifies full parallel OPTICS traversal by its command-node budget', () => {
  const guide = readFileSync('docs/api/gpu-spatial-analysis/space-time-clustering.md', 'utf8');
  expect(guide.includes('full parallel traversal')).toBe(true);
  expect(guide.includes('command-node budget')).toBe(true);
});
