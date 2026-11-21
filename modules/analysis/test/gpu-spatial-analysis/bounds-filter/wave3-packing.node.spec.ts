// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUBoundsFilterParameterValues} from '../../../src/gpu-spatial-analysis/bounds-filter/gpu-bounds-filter';
import {
  GPUBoundsFilter,
  type GPUBoundsFilterProps
} from '../../../src/gpu-spatial-analysis/bounds-filter/gpu-bounds-filter';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('PCK-03 refuses wrapped inversion lost to float32 before touching the target', () => {
  const target = Float32Array.of(42, 42, 42, 42);
  expect(() =>
    getGPUBoundsFilterParameterValues(
      {minX: 10.0000001, maxX: 10, minY: -1, maxY: 1, wrapX: true},
      target
    )
  ).toThrow(/minX.*10\.0000001.*10.*maxX.*10/);
  expect(Array.from(target)).toEqual([42, 42, 42, 42]);
  expect(
    Array.from(getGPUBoundsFilterParameterValues({minX: 10, maxX: 10, minY: -1, maxY: 1}))
  ).toEqual([10, -1, 10, 1]);
  expect(
    Array.from(
      getGPUBoundsFilterParameterValues({minX: 170, maxX: -170, minY: -1, maxY: 1, wrapX: true})
    )
  ).toEqual([170, -1, -170, 1]);
  expect(() =>
    getGPUBoundsFilterParameterValues({minX: 1e40, maxX: 2e40, minY: -1, maxY: 1})
  ).toThrow(/encoded/);
});

it('VAL-09 identifies each missing required compact scalar before declaration', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props: GPUBoundsFilterProps = {
    id: 'required-bounds',
    bounds: createTransientView(graph, 'bounds', 'float32x4', 1),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {
      ids: createTransientView(graph, 'ids', 'uint32', 1),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      }
    }
  };
  try {
    expect(() => new GPUBoundsFilter(props)).not.toThrow();
    for (const path of ['status', 'count', 'overflow']) {
      const output = {...props.output, status: {...props.output.status}};
      if (path === 'status') Reflect.deleteProperty(output, path);
      else Reflect.deleteProperty(output.status, path);
      expect(() => new GPUBoundsFilter({...props, output})).toThrow(
        new RegExp(`required-bounds.*output\\.status${path === 'status' ? ' ' : `\\.${path} `}`)
      );
    }
  } finally {
    device.destroy();
  }
});
