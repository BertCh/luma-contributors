// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {BinaryOperator, ScalarData, WgslExec} from 'wgsl_reflect';
import {expect, it, vi} from 'vitest';
import {
  createCompactnessNode,
  createMomentsNode,
  createOrientationNode
} from '../../../src/gpu-spatial-analysis/shape-descriptors/shape-descriptors-kernels';
import {getShapeMomentTileNodes} from '../../../src/gpu-spatial-analysis/shape-descriptors/shape-moment-tiles';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {createCPUGraphMemory} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {executeGeometryNodesOnCPU} from '../polygon-triangulation/geometry-cpu-executor';

useCPUExecutorTestHooks();

/** Model the permitted reassociations that the plain interpreter misses, with flushed reciprocals. */
function executeWithFastMath(
  nodes: readonly unknown[],
  memory: ReturnType<typeof createCPUGraphMemory>
) {
  for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
    const code = source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '');
    expect(
      (code.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(word => WGSL_RESERVED_WORDS.has(word))
    ).toEqual([]);
    expect(code).not.toMatch(/(?:^|[=,(])\s*(?:NaN|Infinity|inf)\b/m);
  }
  const descriptor = Object.getOwnPropertyDescriptor(WgslExec.prototype, '_evalBinaryOp')!;
  const original = WgslExec.prototype._evalBinaryOp;
  const divide = (first: number, second: number) => {
    const reciprocal = Math.fround(1 / second);
    return Math.fround(first * (Math.abs(reciprocal) < 2 ** -126 ? 0 : reciprocal));
  };
  Object.defineProperty(WgslExec.prototype, '_evalBinaryOp', {
    ...descriptor,
    value: function (
      this: WgslExec,
      node: Parameters<typeof original>[0],
      context: Parameters<typeof original>[1]
    ) {
      if (node.operator === '/' || node.operator === '*') {
        // Resolve a chain of divisions before its trailing constant multiplier. This models
        // area / (perimeter * perimeter) * 4*pi, the zero observed on Metal for a huge square.
        if (
          node.operator === '*' &&
          node.left instanceof BinaryOperator &&
          node.left.operator === '/' &&
          node.left.left instanceof BinaryOperator &&
          node.left.left.operator === '/'
        ) {
          return original.call(this, node, context);
        }
        if (node.left instanceof BinaryOperator && node.left.operator === '/') {
          const first = this.evalExpression(node.left.left, context);
          const second = this.evalExpression(node.left.right, context);
          const third = this.evalExpression(node.right, context);
          if (
            first instanceof ScalarData &&
            second instanceof ScalarData &&
            third instanceof ScalarData &&
            first.typeInfo.name === 'f32'
          ) {
            // (a / b) / c -> a / (b*c); (a / b) * c -> (a*c) / b.
            const result =
              node.operator === '/'
                ? divide(first.value, Math.fround(second.value * third.value))
                : divide(Math.fround(first.value * third.value), second.value);
            return new ScalarData(result, first.typeInfo);
          }
        }
        if (node.operator === '/') {
          const first = this.evalExpression(node.left, context);
          const second = this.evalExpression(node.right, context);
          if (
            first instanceof ScalarData &&
            second instanceof ScalarData &&
            first.typeInfo.name === 'f32'
          ) {
            return new ScalarData(divide(first.value, second.value), first.typeInfo);
          }
        }
      }
      return original.call(this, node, context);
    }
  });
  try {
    executeGeometryNodesOnCPU(nodes, memory);
  } finally {
    Object.defineProperty(WgslExec.prototype, '_evalBinaryOp', descriptor);
  }
}

it('reproduces the unsafe compactness, reciprocal and edge-product lowerings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const values = createTransientView(graph, 'values', 'float32', 5);
  const rawResult = createTransientView(graph, 'raw-result', 'float32', 3);
  const memory = createCPUGraphMemory();
  memory.write(values, new Float32Array([1e38, 4e19, 1.7e38, 8e18, 4.8e19]));
  executeWithFastMath(
    [
      createWGSLKernelNode(graph, {
        id: 'unsafe-range-control',
        operation: 'GPUShapeDescriptors',
        invocationCount: 1,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'rawResult', view: rawResult, type: 'f32', access: 'read_write'}
        ],
        body: `let area = values[valuesOffset]; let perimeter = values[valuesOffset + 1u];
let moment = values[valuesOffset + 2u]; let cross = values[valuesOffset + 3u]; let polynomial = values[valuesOffset + 4u];
rawResult[rawResultOffset] = (area / perimeter) / perimeter * (4.0 * 3.14159265358979);
rawResult[rawResultOffset + 1u] = moment / moment;
rawResult[rawResultOffset + 2u] = (cross / 12.0) * polynomial;`
      })
    ],
    memory
  );
  const actual = memory.readFloat32(rawResult);
  expect(actual[0]).toBe(0);
  expect(actual[1]).toBe(0);
  expect(Number.isFinite(actual[2])).toBe(false);
  device.destroy();
});

it('executes compactness under reassociated division against the square closed form', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const scales = [1, 1e38, 1e-30, 1e-37];
  const areas = createTransientView(graph, 'areas', 'float32', scales.length);
  const perimeters = createTransientView(graph, 'perimeters', 'float32', scales.length);
  const parameters = createTransientView(graph, 'parameters', 'float32', 4);
  const polsbyPopper = createTransientView(graph, 'compactness', 'float32', scales.length);
  const schwartzberg = createTransientView(graph, 'schwartzberg', 'float32', scales.length);
  const memory = createCPUGraphMemory();
  memory.write(areas, Float32Array.from(scales));
  memory.write(
    perimeters,
    Float32Array.from(scales, area => 4 * Math.sqrt(area))
  );
  memory.write(parameters, new Float32Array([0.05, 0, 0, 0]));
  executeWithFastMath(
    [
      createCompactnessNode(graph, {
        id: 'compactness',
        operation: 'GPUShapeDescriptors',
        featureCount: scales.length,
        areas,
        perimeters,
        parameters,
        polsbyPopper,
        schwartzberg
      })
    ],
    memory
  );
  for (const value of memory.readFloat32(polsbyPopper)) expect(value).toBeCloseTo(Math.PI / 4, 5);
  for (const value of memory.readFloat32(schwartzberg))
    expect(value).toBeCloseTo(Math.sqrt(Math.PI / 4), 5);
  device.destroy();
});

it('executes orientation near both normal f32 bounds without a flushed reciprocal', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const scales = [1, 4e37, 1e-37];
  const moments = createTransientView(graph, 'moments', 'float32x4', scales.length);
  const orientation = createTransientView(graph, 'orientation', 'float32', scales.length);
  const elongation = createTransientView(graph, 'elongation', 'float32', scales.length);
  const memory = createCPUGraphMemory();
  memory.write(moments, Float32Array.from(scales.flatMap(scale => [4 * scale, scale, 0, 1])));
  executeWithFastMath(
    [
      createOrientationNode(graph, {
        id: 'orientation',
        operation: 'GPUShapeDescriptors',
        featureCount: scales.length,
        moments,
        orientation,
        elongation
      })
    ],
    memory
  );
  for (const value of memory.readFloat32(orientation)) expect(value).toBeCloseTo(0, 5);
  for (const value of memory.readFloat32(elongation)) expect(value).toBeCloseTo(0.5, 5);
  device.destroy();
});

for (const method of ['serial', 'tiled'] as const) {
  it(`executes ${method} edge integrals near the f32 maximum under reassociation`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const scales = [1e-7, 1, 4e9, 4.2e9];
    const featureCount = scales.length;
    const props = {
      id: 'moments',
      operation: 'GPUShapeDescriptors',
      featureCount,
      positions: createTransientView(graph, 'positions', 'float32x2', 4 * featureCount),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', featureCount + 1),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', featureCount + 1),
      featureOffsets: createTransientView(graph, 'features', 'uint32', featureCount + 1),
      centroids: createTransientView(graph, 'centroids', 'float32x2', featureCount),
      moments: createTransientView(graph, 'moments', 'float32x4', featureCount)
    };
    const memory = createCPUGraphMemory();
    memory.write(
      props.positions,
      Float32Array.from(scales.flatMap(scale => [0, 0, 2 * scale, 0, 2 * scale, scale, 0, scale]))
    );
    memory.write(props.centroids, Float32Array.from(scales.flatMap(scale => [scale, scale / 2])));
    memory.write(
      props.ringOffsets,
      Uint32Array.from({length: featureCount + 1}, (_, row) => 4 * row)
    );
    memory.write(
      props.polygonOffsets,
      Uint32Array.from({length: featureCount + 1}, (_, row) => row)
    );
    memory.write(
      props.featureOffsets,
      Uint32Array.from({length: featureCount + 1}, (_, row) => row)
    );
    const createView = vi.spyOn(graph, 'createDataView');
    const nodes =
      method === 'serial'
        ? [createMomentsNode(graph, props)]
        : getShapeMomentTileNodes(graph, props);
    // The cooperative shared reductions need a GPU. Each ring here owns one tile, so inspect
    // actual tile results before reduction, against the same independent physical integrals.
    executeWithFastMath(
      method === 'serial'
        ? nodes
        : nodes.slice(0, nodes.findIndex(node => node.id.endsWith('-tiles')) + 1),
      memory
    );
    const components =
      method === 'serial'
        ? [0, 1, 2].map(component =>
            scales.map((_, row) => memory.readFloat32(props.moments)[4 * row + component])
          )
        : ['xx', 'yy', 'xy'].map(name => {
            const view = createView.mock.results.find(
              result =>
                result.type === 'return' && result.value.buffer.id === `moments-tile-${name}`
            )!.value;
            return Array.from(memory.readFloat32(view));
          });
    for (let feature = 0; feature < featureCount; feature++) {
      expect(components[0][feature] / ((2 / 3) * scales[feature] ** 4)).toBeCloseTo(1, 5);
      expect(components[1][feature] / ((1 / 6) * scales[feature] ** 4)).toBeCloseTo(1, 5);
      expect(Math.abs(components[2][feature]) / ((2 / 3) * scales[feature] ** 4)).toBeLessThan(
        2e-6
      );
    }
    vi.restoreAllMocks();
    device.destroy();
  });
}
