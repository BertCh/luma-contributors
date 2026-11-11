// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNormalizeGeometryOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-normalize-geometry-operation';
import {
  GPUNormalizeGeometryOperation,
  type NormalizeGeometryOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-geometry-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {getNormalizeGeometryScene} from './normalize-geometry-scene';

for (const leafOnly of [true, false])
  it(`normalizes ${leafOnly ? 'a leaf-only tree without children' : 'an entirely empty tree'} on both operations`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: NormalizeGeometryOperationProps = {
      positions: 'positions',
      nodes: 'nodes',
      children: 'children',
      rowCount: leafOnly ? 1 : 0,
      nodeCount: leafOnly ? 1 : 0,
      childCount: 0,
      output: {positions: 'normalized', children: 'ordered', status: 'status'}
    };
    for (const Operation of [GPUNormalizeGeometryOperation, CPUNormalizeGeometryOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: new Float32Array(leafOnly ? [1, 2] : [])},
          nodes: {format: 'uint32x4', values: new Uint32Array(leafOnly ? [0, 0, 1, 0] : [])},
          children: {format: 'uint32', values: new Uint32Array(0)}
        },
        operations: [new Operation(props)],
        outputs: ['normalized', 'ordered', 'status']
      });
      // A point is copied with status 0; an empty tree emits no coordinates or status rows.
      expect(Array.from(result['normalized'])).toEqual(leafOnly ? [1, 2] : []);
      expect(result['ordered'].length).toBe(0);
      expect(Array.from(result['status'])).toEqual(leafOnly ? [0] : []);
    }
  });

for (const invalid of [false, true])
  it(`compares every tree-normalization output (${invalid ? 'nonfinite leaf and invalid parent' : 'nested heterogeneous collections'})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {input} = getNormalizeGeometryScene();
    if (invalid) input.positions[2] = NaN;
    const props: NormalizeGeometryOperationProps = {
      positions: 'positions',
      nodes: 'nodes',
      children: 'children',
      rowCount: input.positions.length / 2,
      nodeCount: input.nodes.length / 4,
      childCount: input.children.length,
      maximumNestingDepth: 8,
      output: {positions: 'normalized', children: 'ordered', status: 'status'}
    };
    const outputs = Object.values(props.output);
    const results = [];
    for (const Operation of [GPUNormalizeGeometryOperation, CPUNormalizeGeometryOperation])
      results.push(
        await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: input.positions},
            nodes: {format: 'uint32x4', values: input.nodes},
            children: {format: 'uint32', values: input.children}
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    // Coordinates are copied and all other fields are integers; tolerance is zero, with
    // NaNs matched explicitly and signed zeros compared numerically.
    for (const port of outputs) {
      expect(results[0][port].length).toBe(results[1][port].length);
      for (let index = 0; index < results[1][port].length; index++) {
        const expected = results[1][port][index];
        const actual = results[0][port][index];
        expect(Number.isNaN(expected) ? Number.isNaN(actual) : actual === expected).toBe(true);
      }
    }
  });
