// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
import {GPUAdamStep} from '../../src/analysis-core/kernels/gpu-adam-step';
import {GPUFittingGradientSums} from '../../src/analysis-core/kernels/gpu-fitting-gradient-sums';
import {GPUFittingCoordinateEncoder} from '../../src/analysis-core/kernels/gpu-fitting-coordinate-encoder';
import {createKernelAuditContext} from './kernel-audit/kernel-audit-context';
import {createAdamAuditProps} from './kernel-audit/cases-fitting-kernels';
import {ExampleChunkContributor} from './graph-chunk-dispatch.test-utils';
import {GPUFittingMinibatchOrder} from '../../src/analysis-core/kernels/gpu-fitting-minibatch-order';
import {GPUPolygonTriangulation} from '../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import {GPUSegmentRingAssembly} from '../../src/analysis-core/ring-assembly/gpu-segment-ring-assembly';
import {GPUSpatialJoinPrepared} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {GPUStudyWindowPreparation} from '../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';

it('R2 authoring fixture captures inherited option getters', () => {
  const context = createKernelAuditContext('getter-fixture');
  let id = 'original';
  class Options {
    get id() {
      return id;
    }
    get maximumWorkgroupsPerDimension() {
      return 2;
    }
  }
  try {
    const input = createVectorView('input', 'uint32', [context.view('uint32', 3)]);
    const output = createVectorView('output', 'uint32', [context.view('uint32', 3)]);
    const producer = new ExampleChunkContributor(input, output, new Options());
    id = 'mutated';
    expect(producer.options).toEqual({id: 'original', maximumWorkgroupsPerDimension: 2});
    expect(producer.getCommandNodes(context.graph).length).toBeGreaterThan(0);
  } finally {
    context.device.destroy();
  }
});

function inheritProperties<Props extends object>(props: Props): Props {
  const prototype = {};
  for (const key of Object.keys(props) as (keyof Props)[]) {
    Object.defineProperty(prototype, key, {get: () => props[key]});
  }
  return Object.create(prototype) as Props;
}

it('R2 preserves inherited scalar, view and nested status getters in every other core producer', () => {
  const context = createKernelAuditContext('getter-props');
  const {graph, view, device} = context;
  try {
    const gradients = new GPUFittingGradientSums(
      inheritProperties({
        rowCount: 2,
        parameterCount: 1,
        gradients: view('float32', 2),
        output: inheritProperties({sums: view('float32', 1)})
      })
    );
    const adamProps = createAdamAuditProps(context, 2, 3);
    const adam = new GPUAdamStep(
      inheritProperties({
        ...adamProps,
        output: inheritProperties({
          ...adamProps.output,
          status: inheritProperties(adamProps.output.status)
        })
      })
    );
    const encoder = new GPUFittingCoordinateEncoder(
      inheritProperties({
        encoding: 'fourier' as const,
        dimensions: 2,
        frequencyCount: 1,
        rowCount: 1,
        coordinates: view('float32', 2),
        frequencies: view('float32', 2),
        spatialContext: inheritProperties({
          coordinateSpace: 'planar' as const,
          metric: 'native' as const
        }),
        output: inheritProperties({features: view('float32', 2), validity: view('uint32', 1)})
      })
    );
    const ringCount = view('uint32', 1);
    const rings = new GPUSegmentRingAssembly(
      inheritProperties({
        endpoints: view('float32x4', 4),
        output: inheritProperties({
          positions: view('float32x2', 5),
          ringOffsets: view('uint32', 2),
          status: inheritProperties({count: ringCount, overflow: view('uint32', 1)})
        })
      })
    );
    expect(gradients.getCommandNodes(graph)).toHaveLength(1);
    expect(adam.getCommandNodes(graph)).toHaveLength(2);
    expect(encoder.getCommandNodes(graph)).toHaveLength(1);
    expect(rings.getCommandNodes(graph).length).toBeGreaterThan(0);
    expect(adam.props.output.status.iterationCount).toBe(adamProps.output.status.iterationCount);
    expect(rings.props.output.status.count).toBe(ringCount);
    expect(Reflect.set(encoder.spatialContext, 'coordinateSpace', 'longitude-latitude')).toBe(
      false
    );
  } finally {
    device.destroy();
  }
});

it('R5 catalog callback types and descriptions are regenerated from the declared props', () => {
  const inventory = JSON.parse(
    readFileSync('website/src/catalog/generated/inventory.json', 'utf8')
  ) as {
    symbols: {symbol: string; props: {name: string; type: string; jsdoc: string}[]}[];
  };
  for (const [symbol, file] of [
    ['GPUSpatialJoinPrepared', 'spatial-join/gpu-spatial-join-prepared.ts'],
    ['GPUStudyWindowPreparation', 'study-window/gpu-study-window-preparation.ts']
  ]) {
    const source = ts.createSourceFile(
      file,
      readFileSync(`modules/analysis/src/gpu-spatial-analysis/${file}`, 'utf8'),
      ts.ScriptTarget.Latest,
      true
    );
    let callback: ts.PropertySignature | undefined;
    const visit = (node: ts.Node) => {
      if (ts.isPropertySignature(node) && node.name.getText(source) === 'rebuildWhen')
        callback = node;
      ts.forEachChild(node, visit);
    };
    visit(source);
    expect(callback?.type).toBeDefined();
    const property = inventory.symbols
      .find(entry => entry.symbol === symbol)!
      .props.find(entry => entry.name === 'rebuildWhen')!;
    const functionType = callback!.type!.getText(source);
    expect(property.type).toBe(`(${functionType}) | undefined`);
    const comment = ts.getJSDocCommentsAndTags(callback!).find(ts.isJSDoc)?.comment;
    const description = ts.getTextOfJSDocComment(comment)!.replace(/\s+/g, ' ').trim();
    expect(property.jsdoc).toBe(description.split(/(?<=\.)\s/)[0]);
    if (symbol === 'GPUSpatialJoinPrepared') {
      expect(readFileSync('website/src/catalog/entries/cells-index-join.ts', 'utf8')).toContain(
        `type: '(${functionType})'`
      );
    }
  }
});

it('R2 preserves a prototype rowCount getter and freezes its captured value', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let rowCount = 2;
  class Props {
    get rowCount() {
      return rowCount;
    }
    parameters = createTransientView(graph, 'parameters', 'uint32', 3);
    output = {order: createTransientView(graph, 'order', 'uint32', 2)};
  }
  try {
    const props = new Props();
    const producer = new GPUFittingMinibatchOrder(props);
    rowCount = 0;
    expect(producer.props.rowCount).toBe(2);
    expect(producer.props.output.order).toBe(props.output.order);
    expect(producer.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('R2 retains a prototype triangleCount port and its generated writer', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const triangleCount = createTransientView(graph, 'triangle-count', 'uint32', 1);
  class Output {
    indices = createTransientView(graph, 'indices', 'uint32', 3);
    valid = createTransientView(graph, 'valid', 'uint32', 1);
    get triangleCount() {
      return triangleCount;
    }
  }
  try {
    const producer = new GPUPolygonTriangulation({
      positions: createTransientView(graph, 'positions', 'float32x2', 3),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
      largePolygonMethod: 'none',
      output: new Output()
    });
    expect(producer.props.output.triangleCount).toBe(triangleCount);
    const nodes = producer.getCommandNodes(graph);
    expect(Reflect.set(producer.spatialContext, 'coordinateSpace', 'longitude-latitude')).toBe(
      false
    );
    expect(
      nodes.some(node =>
        node.resources?.some(
          resource =>
            'buffer' in resource &&
            resource.buffer === triangleCount &&
            resource.usage === 'storage-read-write'
        )
      )
    ).toBe(true);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    expect(
      kernels.some(kernel =>
        kernel.source.includes('triangleCount[triangleCountOffset + polygon] = written;')
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('R4 freezes exposed spatial context metadata and retains the planar shader', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const producer = new GPUSegmentRingAssembly({
      spatialContext: {
        coordinateSpace: 'planar',
        metric: 'native',
        frame: {id: 'frame', kind: 'crs', origin: [0, 0], unit: {name: 'meters', metersPerUnit: 1}}
      },
      endpoints: createTransientView(graph, 'endpoints', 'float32x4', 4),
      output: {
        positions: createTransientView(graph, 'positions', 'float32x2', 5),
        ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 2),
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        }
      }
    });
    expect(
      Reflect.set(producer.props.spatialContext!, 'coordinateSpace', 'longitude-latitude')
    ).toBe(false);
    expect(Reflect.set(producer.spatialContext.frame!, 'id', 'changed')).toBe(false);
    expect(Reflect.set(producer.spatialContext.frame!.origin!, '0', 10)).toBe(false);
    expect(Reflect.set(producer.spatialContext.frame!.unit!, 'metersPerUnit', 2)).toBe(false);
    const sources = getGPUCommandNodeKernelSources(producer.getCommandNodes(graph));
    expect(sources.some(kernel => kernel.source.includes('return delta;'))).toBe(true);
    expect(
      sources.some(kernel => kernel.source.includes('delta.x * max(cos(radians(latitude)), 0.01)'))
    ).toBe(false);
  } finally {
    device.destroy();
  }
});

for (const kind of ['join', 'window'] as const) {
  function createFixture() {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 3);
    const rebuildWhen = vi.fn(() => false);
    const producer =
      kind === 'join'
        ? new GPUSpatialJoinPrepared({geometry: {kind: 'points', positions}, rebuildWhen})
        : new GPUStudyWindowPreparation({
            positions,
            featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
            polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
            ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
            samplingGrid: [2, 2],
            rebuildWhen
          });
    const nodes = producer.getCommandNodes(graph);
    const evaluate = (index: number) => {
      const condition = nodes[index].condition;
      // GPU conditions record an indirect dispatch; their outcome is decided only on the GPU.
      return condition?.source === 'cpu' ? condition.evaluate(undefined) : true;
    };
    return {
      producer,
      device,
      nodes,
      rebuildWhen,
      evaluate,
      encode: () => nodes.map((_node, index) => evaluate(index)),
      destroy() {
        producer.destroy();
        device.destroy();
      }
    };
  }

  it(`R1 ${kind} restarts the latch after abandoning a clean one-node prefix`, () => {
    const fixture = createFixture();
    try {
      fixture.encode();
      expect(fixture.evaluate(0)).toBe(true);
      fixture.producer.invalidate();
      expect(fixture.encode()).toEqual(fixture.nodes.map(() => true));
    } finally {
      fixture.destroy();
    }
  });

  it(`R5 ${kind} bypasses the predicate when initially or explicitly dirty`, () => {
    const fixture = createFixture();
    try {
      fixture.encode();
      expect(fixture.rebuildWhen).toHaveBeenCalledTimes(0);
      fixture.encode();
      expect(fixture.rebuildWhen).toHaveBeenCalledTimes(1);
      fixture.producer.invalidate();
      fixture.encode();
      expect(fixture.rebuildWhen).toHaveBeenCalledTimes(1);
      fixture.encode();
      expect(fixture.rebuildWhen).toHaveBeenCalledTimes(2);
    } finally {
      fixture.destroy();
    }
  });
}
