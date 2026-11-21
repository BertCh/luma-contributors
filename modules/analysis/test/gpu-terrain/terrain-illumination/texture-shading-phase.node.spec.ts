// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {GPUTextureShading} from '../../../src/gpu-terrain/terrain-illumination/gpu-texture-shading';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createBand} from '../terrain-test-utils';

it('RT-136 explicit tile route checks both expanded-origin coordinates', () => {
  for (const downsampleLevels of [true, false]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const stage = new GPUTextureShading({
      grid: makeGPURasterMetadata({width: 65, height: 65}),
      levelCount: 3,
      baseSigma: 2,
      downsampleLevels,
      elevation: createBand(graph, 'elevation', 65 * 65),
      parameters: createTransientView(graph, 'parameters', 'float32', 12),
      output: {textureShade: createTransientView(graph, 'shade', 'float32', 65 * 65)}
    });
    if (downsampleLevels) {
      for (const origin of [
        [1, 0],
        [0, 1]
      ] as const) {
        expect(() => stage.getCommandNodesForTile(graph, origin)).toThrow(/expandedOrigin.*2/);
      }
    }
    expect(() =>
      stage.getCommandNodesForTile(graph, downsampleLevels ? [-4, 6] : [1, 3])
    ).not.toThrow();
  }
});

it('S2 rejects missing, sparse and extra expanded-origin coordinates at scale eight', () => {
  for (const malformed of [[], [0], new Array(2), [0, ,], [0, 0, 0], [NaN, 0], [0, Infinity]]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const stage = new GPUTextureShading({
      grid: makeGPURasterMetadata({width: 9, height: 9}),
      levelCount: 6,
      baseSigma: 1,
      elevation: createBand(graph, 'elevation', 81),
      parameters: createTransientView(graph, 'parameters', 'float32', 12),
      output: {textureShade: createTransientView(graph, 'shade', 'float32', 81)}
    });
    expect(stage.levelScaleLog2.at(-1)).toBe(3);
    expect(() =>
      stage.getCommandNodesForTile(graph, malformed as unknown as readonly [number, number])
    ).toThrow(/expandedOrigin.*8/);
  }
});

it('S1 reconstruction interpolates numerator and denominator at odd coordinates before division', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const stage = new GPUTextureShading({
    grid: makeGPURasterMetadata({width: 3, height: 3}),
    levelCount: 3,
    baseSigma: 2,
    elevation: createBand(graph, 'elevation', 9),
    parameters: createTransientView(graph, 'parameters', 'float32', 12),
    output: {textureShade: createTransientView(graph, 'shade', 'float32', 9)}
  });
  const kernel = getGPUCommandNodeKernelSources(stage.getCommandNodes(graph)).find(kernel =>
    kernel.id.endsWith('-level-2-accumulate')
  )!;
  const check = (source: string) => {
    const settings = new Float32Array(12);
    settings[3] = 1;
    const result = new Float32Array(9);
    new WgslExec(
      new WgslParser().parse(source.replace(/^\s*_ = [^;]*;$/gm, ''))
    ).dispatchWorkgroups(
      'main',
      9,
      {
        0: {
          0: new Float32Array(9),
          1: new Float32Array(9).fill(1),
          2: new Float32Array([0, 8, 4, 0]),
          3: new Float32Array([1, 1, 1, 0]),
          4: settings,
          5: result
        }
      },
      {constants: kernel.constants}
    );
    // At local (1,1), four weights are 1/4. The current quotient is (0+8+4+0)/(1+1+1+0)
    // = 4; the previous value is zero. Quotient-first interpolation would give -3; nearest 0.
    expect(Number.isFinite(result[4])).toBe(true);
    expect(result[4]).toBe(-4);
  };
  check(kernel.source);
  const cornerQuotients = ['00', '10', '01', '11'].map(
    corner =>
      `getBlurred(currentNumerator[currentNumeratorOffset + currentIndex${corner}], currentDenominator[currentDenominatorOffset + currentIndex${corner}])`
  );
  expect(() =>
    check(
      kernel.source.replace(
        'getBlurred(currentNumerator_, currentDenominator_)',
        `0.25 * (${cornerQuotients.join(' + ')})`
      )
    )
  ).toThrow();
  expect(() =>
    check(
      kernel.source.replace(
        'getBlurred(currentNumerator_, currentDenominator_)',
        cornerQuotients[0]
      )
    )
  ).toThrow();
});
