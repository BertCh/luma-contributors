// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  createTransientView,
  getViewBindingRange,
  type GraphBufferHandle
} from '@luma.gl/graph/gpu-core';
import {ScalarData, WgslExec, WgslParser} from 'wgsl_reflect';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createMultiscaleGlobalFit} from '../../../src/gpu-spatial-analysis/spatial-regression/multiscale-global-fit';
import {
  getRegressionBindings,
  getRegressionWGSL
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import {expect, it, vi} from 'vitest';
import {GPUMultiscaleGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

for (const predictorCount of [1, 24]) {
  it(`MGWR tiles the global fit at the indexed row cap with ${predictorCount} predictors`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const rowCount = 1048576;
      let serial = 0;
      const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
        format: Format,
        length: number
      ) => createTransientView(graph, `global-${serial++}`, format, length);
      const contributor = new GPUMultiscaleGeographicallyWeightedRegression({
        positions: view('float32x2', rowCount),
        predictors: view('float32', rowCount * predictorCount),
        response: view('float32', rowCount),
        mask: view('uint32', rowCount),
        parameters: view('float32', predictorCount + 2),
        predictorCount,
        kernel: 'bisquare',
        maximumIterations: 1,
        output: {
          coefficients: view('float32', rowCount * (predictorCount + 1)),
          fitted: view('float32', rowCount),
          status: {codes: view('uint32', rowCount), iterationCount: view('uint32', 1)}
        }
      });
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      const sources = getGPUCommandNodeKernelSources(nodes).filter(kernel =>
        kernel.id.includes('-global-')
      );
      expect(sources.map(kernel => kernel.id.split('-global-')[1])).toEqual([
        'sample-tiles',
        'sample-merge',
        'deviations-tiles',
        'deviations-merge',
        'moments-tiles',
        'moments-merge',
        'fit'
      ]);
      for (const kernel of sources) {
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
        expect(kernel.source).not.toMatch(/for \(var row = 0u; row < 1048576u;/);
      }
      Object.assign(device.limits, {maxStorageBufferBindingSize: 4});
      const allocations = vi.spyOn(graph, 'createTransientBuffer');
      const subviews = vi.spyOn(graph, 'createDataView');
      const nodesAdded = vi.spyOn(graph, 'add');
      // Caller bindings are now preflighted before any nested scratch or node is declared.
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        /positions chunk 0 needs 8388608 bytes.*limit 4/
      );
      expect(allocations).not.toHaveBeenCalled();
      expect(subviews).not.toHaveBeenCalled();
      expect(nodesAdded).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  });
}

for (const scenario of ['outlier', 'opposite-sign', 'small-response', 'zero-response'] as const) {
  for (const responseSign of [1, -1]) {
    it(`emitted MGWR centering WGSL retains finite moments for ${scenario}/${responseSign}`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        recordWGSLKernelNodes(graph);
        const rowCount = scenario === 'outlier' ? 129 : 3;
        const magnitude =
          scenario === 'small-response' ? 2 ** -120 : scenario === 'zero-response' ? 0 : 2 ** 127;
        const props = {
          positions: createTransientView(graph, 'positions', 'float32x2', rowCount),
          predictors: createTransientView(graph, 'predictors', 'float32', rowCount),
          response: createTransientView(graph, 'response', 'float32', rowCount),
          parameters: createTransientView(graph, 'parameters', 'float32', 3),
          predictorCount: 1
        };
        const output = createTransientView(graph, 'coefficients', 'float32', 2);
        const nodes = createMultiscaleGlobalFit(graph, {
          id: 'outlier-fit',
          rowCount,
          predictorCount: 1,
          bindings: getRegressionBindings(props).filter(binding => binding.name !== 'parameters'),
          declarations: getRegressionWGSL(props),
          output
        });
        const storage = new Map<GraphBufferHandle, ArrayBuffer>();
        const getStorage = (buffer: GraphBufferHandle) => {
          if (!storage.has(buffer)) storage.set(buffer, new ArrayBuffer(buffer.byteLength));
          return storage.get(buffer)!;
        };
        const predictors = new Float32Array(getStorage(props.predictors.buffer));
        const response = new Float32Array(getStorage(props.response.buffer));
        if (scenario !== 'outlier') {
          predictors.set([0.5, -0.5, 0]);
          response.set([-responseSign * magnitude, 0, responseSign * magnitude]);
        } else {
          for (let row = 0; row < rowCount - 1; row++) predictors[row] = row % 2 ? 1 : -1;
          response[rowCount - 1] = responseSign * 1e37;
        }
        for (const node of nodes) {
          const record = getWGSLKernelNodeRecord(graph, node.id)!;
          const bindings = Object.fromEntries(
            record.bindings.map((binding, index) => {
              const {offset, size} = getViewBindingRange(binding.view);
              return [index, new Float32Array(getStorage(binding.view.buffer), offset, size / 4)];
            })
          );
          // Probe the emitted mean helper and reduced cross term; the headless companion
          // runs the full dense solve as well. No reduction code is replaced.
          const source = record.id.endsWith('-global-fit')
            ? record.source.replace(
                record.body,
                `globalFit[globalFitOffset] = getGlobalMean(0u) * getGlobalResponseScale();
            globalFit[globalFitOffset + 1u] = moments[momentsOffset] * getGlobalResponseScale();`
              )
            : record.source;
          const interpreter = new WgslExec(
            new WgslParser().parse(source.replace(/^\s*_\s*=[^;]*;/gm, ''))
          );
          // wgsl_reflect 1.2.3 leaves ldexp unimplemented. Supply scalar IEEE arithmetic;
          // the headless companion uses the GPU builtin without this interpreter shim.
          interpreter.builtins.Ldexp = (call, context) => {
            const [valueArgument, exponentArgument] = call.args ?? [];
            if (!valueArgument || !exponentArgument)
              throw new Error('ldexp requires two arguments');
            const value = interpreter.evalExpression(valueArgument, context);
            const exponent = interpreter.evalExpression(exponentArgument, context);
            if (!(value instanceof ScalarData) || !(exponent instanceof ScalarData))
              throw new Error('ldexp probe requires scalar arguments');
            return new ScalarData(value.value * 2 ** exponent.value, value.typeInfo);
          };
          interpreter.dispatchWorkgroups(
            'main',
            [...record.workgroups],
            {0: bindings},
            {constants: record.constants}
          );
        }
        const actual = new Float32Array(getStorage(output.buffer));
        // Closed forms: sum(x) = 0. Symmetric responses have sum(y) = 0 and
        // sum(x*y) = -sign*magnitude/2; the outlier scene has sum(x*y) = 0.
        const expectedIntercept = scenario === 'outlier' ? response[rowCount - 1] / rowCount : 0;
        const expectedCross = scenario === 'outlier' ? 0 : (-responseSign * magnitude) / 2;
        expect(Number.isFinite(actual[0])).toBe(true);
        expect(Number.isFinite(actual[1])).toBe(true);
        if (expectedIntercept === 0) expect(Math.abs(actual[0])).toBe(0);
        else
          expect(Math.abs(actual[0] - expectedIntercept)).toBeLessThan(
            Math.abs(expectedIntercept) * 2e-5
          );
        if (expectedCross === 0) expect(Math.abs(actual[1])).toBe(0);
        else
          expect(Math.abs(actual[1] - expectedCross)).toBeLessThan(Math.abs(expectedCross) * 2e-6);
      } finally {
        device.destroy();
      }
    });
  }
}
