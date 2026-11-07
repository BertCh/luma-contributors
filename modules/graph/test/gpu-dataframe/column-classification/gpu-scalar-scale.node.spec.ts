// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {existsSync, readFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {runInNewContext} from 'node:vm';
import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {scaleLinear, scaleLog, scalePoint, scaleSqrt} from 'd3-scale';
import {expect, it} from 'vitest';
import {GPUScalarScale} from '../../../src/gpu-dataframe/column-classification/gpu-scalar-scale';
import {
  getGPUScalarScaleParameterValues,
  prepareGPUScalarScaleControls
} from '../../../src/gpu-dataframe/column-classification/scalar-scale-parameters';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {scalarScaleFixtures} from './scalar-scale-fixtures';

it('checks scalar fixtures against the pinned d3-scale oracle', () => {
  const require = createRequire(import.meta.url);
  const requireKepler = createRequire(require.resolve('@kepler.gl/constants'));
  expect(require.resolve('d3-scale')).toBe(requireKepler.resolve('d3-scale'));
  // d3-scale does not export package.json; walk up from its resolved entry point.
  let packageDirectory = dirname(requireKepler.resolve('d3-scale'));
  while (!existsSync(join(packageDirectory, 'package.json'))) {
    packageDirectory = dirname(packageDirectory);
  }
  const packagePath = join(packageDirectory, 'package.json');
  expect(JSON.parse(readFileSync(packagePath, 'utf8')).version).toBe('4.0.3');
  for (const fixture of scalarScaleFixtures) {
    const {scale, domain, range = [0, 1], invert = false} = fixture.controls;
    const fixedRange = invert ? [...domain].reverse() : [...domain];
    const native =
      scale === 'point'
        ? scalePoint<number>()
            .domain([...domain])
            .range([...range])
        : (scale === 'sqrt' ? scaleSqrt() : scale === 'log' ? scaleLog() : scaleLinear())
            .domain([...domain])
            .range(scale === 'identity' ? fixedRange : [...range]);
    const controls = prepareGPUScalarScaleControls(fixture.controls);
    expect(controls.parameters[1]).toBe(domain.length);
    fixture.values.forEach((value, index) => {
      const mapped = value === null ? undefined : native(value);
      const valid =
        value !== null &&
        Number.isFinite(value) &&
        fixture.validity?.[index] !== 0 &&
        mapped !== undefined &&
        Number.isFinite(Math.fround(mapped));
      const expected = fixture.expected[index];
      expect(valid, `${fixture.name}: row ${index}`).toBe(expected !== null);
      if (expected !== null && mapped !== undefined) {
        expect(Math.abs(Math.fround(mapped) - expected)).toBeLessThanOrEqual(
          2e-6 * Math.max(1, Math.abs(expected))
        );
      }
    });
  }
});

it('matches the installed native fixed-height inversion accessor independently', () => {
  const require = createRequire(import.meta.url);
  const source = readFileSync(
    join(dirname(require.resolve('@kepler.gl/layers')), 'base-layer.js'),
    'utf8'
  );
  const functionSource = source.match(
    /function getVisChannelScale\(scale, domain, range, fixed, invert\) \{[\s\S]*?return applyInvert\(scaleFunc\);\s*\}/
  )?.[0];
  expect(functionSource).toBeDefined();
  // Execute the installed native method; these fixtures use plain domains, not zoom stops.
  const getNativeScale = runInNewContext(`(${functionSource})`, {
    _constants: {SCALE_FUNC: {linear: scaleLinear}},
    _utils: {isDomainQuantile: () => false, isDomainStops: () => false}
  });
  for (const fixture of scalarScaleFixtures.filter(entry => entry.controls.invert)) {
    const native = getNativeScale('sqrt', fixture.controls.domain, [1, 99], true, true);
    for (const [index, value] of fixture.values.entries()) {
      const expected = fixture.expected[index];
      if (value === null || expected === null) continue;
      expect(Math.abs(Math.fround(native(value)) - expected), fixture.name).toBeLessThanOrEqual(
        2e-6 * Math.max(1, Math.abs(expected))
      );
    }
  }
});

it('rejects lossy controls and preserves reusable parameter targets', () => {
  for (const domain of [
    [0, 100],
    [-1, 1],
    [1, 0]
  ])
    expect(() => prepareGPUScalarScaleControls({scale: 'log', domain})).toThrow();
  for (const domain of [
    [0, 0],
    [-0, 0],
    [1, 2, 1]
  ])
    expect(() => prepareGPUScalarScaleControls({scale: 'point', domain})).toThrow();
  for (const domain of [[0, 0.1], [NaN, 10], [0, Infinity], [0], [0, 1, 2]])
    expect(() => prepareGPUScalarScaleControls({scale: 'linear', domain})).toThrow();
  expect(() => getGPUScalarScaleParameterValues({scale: 'point', domainCount: 257})).toThrow();
  expect(() =>
    getGPUScalarScaleParameterValues({scale: 'linear', domainCount: 2, nullValue: Infinity})
  ).toThrow();
  expect(() =>
    getGPUScalarScaleParameterValues({scale: 'linear', domainCount: 2}, new Float32Array(3))
  ).toThrow();
  expect(() =>
    prepareGPUScalarScaleControls({scale: 'linear', domain: [0, 1], invert: true})
  ).toThrow();
  const target = new Float32Array(4);
  getGPUScalarScaleParameterValues({scale: 'identity', domainCount: 2, invert: true}, target);
  expect([...target]).toEqual([3, 2, 0, 1]);
  expect(getGPUScalarScaleParameterValues({scale: 'sqrt', domainCount: 2}, target)).toBe(target);
  expect([...target]).toEqual([1, 2, 0, 0]);
  getGPUScalarScaleParameterValues({scale: 'point', domainCount: 3, nullValue: -99}, target);
  expect([...target]).toEqual([4, 3, -99, 0]);
  for (const domain of [[], [5], [5, 3, 7]]) {
    const controls = prepareGPUScalarScaleControls({scale: 'point', domain});
    const allocatedDomain = new Float32Array(Math.max(2, domain.length));
    allocatedDomain.set(controls.domain);
    expect([...allocatedDomain.slice(0, domain.length)]).toEqual(domain);
    expect(controls.domain.length).toBe(domain.length);
  }
});

it('validates view topology and intentional separate-output-buffer restriction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const shared = createTransientView(graph, 'shared', 'float32', 32);
  const values = graph.createDataView(shared.buffer, {format: 'float32', length: 4});
  const domain = graph.createDataView(shared.buffer, {
    format: 'float32',
    length: 8,
    byteOffset: 16
  });
  const props = {
    values,
    domain,
    range: createTransientView(graph, 'range', 'float32', 2),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    maximumDomainCount: 8,
    validity: createTransientView(graph, 'validity', 'uint32', 4),
    output: {
      values: createTransientView(graph, 'output', 'float32', 4),
      validity: createTransientView(graph, 'output-validity', 'uint32', 4)
    }
  };
  try {
    const contributor = new GPUScalarScale(props);
    expect(getParsedKernelSources(contributor.getCommandNodes(graph)).size).toBe(1);
    expect(contributor.props.values).toBe(values);
    const disjointOutput = graph.createDataView(shared.buffer, {
      format: 'float32',
      length: 4,
      byteOffset: 64
    });
    const disjointValidity = graph.createDataView(shared.buffer, {
      format: 'uint32',
      length: 4,
      byteOffset: 80
    });
    expect(() => new GPUScalarScale({...props, output: {values: disjointOutput}})).toThrow();
    const separateInputs = {
      ...props,
      values: props.output.values,
      domain: createTransientView(graph, 'separate-domain', 'float32', 8)
    };
    expect(
      () =>
        new GPUScalarScale({
          ...separateInputs,
          output: {values: disjointOutput, validity: disjointValidity}
        })
    ).toThrow();
    expect(() => new GPUScalarScale({...props, maximumDomainCount: 9})).toThrow();
    expect(
      () =>
        new GPUScalarScale({
          ...props,
          validity: createTransientView(graph, 'short-validity', 'uint32', 3)
        })
    ).toThrow();
    expect(() => contributor.getCommandNodes(new GPUCommandGraph(device))).toThrow();
    const empty = createTransientView(graph, 'empty', 'float32', 0);
    expect(
      new GPUScalarScale({...props, values: empty, validity: undefined}).getCommandNodes(graph)
    ).toEqual([]);
  } finally {
    device.destroy();
  }
});
