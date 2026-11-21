// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {addSpatialRegressionRecipe} from '../../../src/gpu-spatial-analysis/recipes/spatial-regression-recipe';
import {
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_EXCLUDED_ROW_COUNT,
  GPUOrdinaryLeastSquares,
  GPUSpatialRegressionDiagnostics
} from '../../../src/gpu-spatial-analysis/spatial-regression/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPULocalMoran} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it.each([
  8, 11, 12
])('S1 local fits infer the documented eleven-predictor limit: %s', predictorCount => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: 64
  });
  try {
    const props = {
      predictors: createTransientView(graph, 'predictors', 'float32', 16 * predictorCount),
      response: createTransientView(graph, 'response', 'float32', 16),
      parameters,
      weights: {
        offsets: createTransientView(graph, 'offsets', 'uint32', 17),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 16),
        weights: createTransientView(graph, 'weights', 'float32', 16)
      },
      localFits: {
        positions: createTransientView(graph, 'positions', 'float32x2', 16),
        parameters: createTransientView(graph, 'local-parameters', 'float32', 64),
        maximumBandwidthCount: 1
      }
    };
    if (predictorCount === 12)
      expect(() => addSpatialRegressionRecipe(graph, props)).toThrow(/predictorCount.*11/);
    else
      expect(addSpatialRegressionRecipe(graph, props).localFits?.coefficients.length).toBe(
        16 * (predictorCount + 1)
      );
  } finally {
    parameters.destroy();
    device.destroy();
  }
});

it('B5 folds a live weights certificate before both spatial consumers and rejects a destructive alias', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: 64
  });
  const complete = createTransientView(graph, 'complete', 'uint32', 1);
  const requiredCount = createTransientView(graph, 'required', 'uint32', 1);
  const weights = {
    offsets: createTransientView(graph, 'offsets', 'uint32', 8),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 7),
    weights: createTransientView(graph, 'weights', 'float32', 7),
    status: {complete, requiredCount}
  };
  const props = {
    id: 'cert',
    predictors: createTransientView(graph, 'predictors', 'float32', 7),
    response: createTransientView(graph, 'response', 'float32', 7),
    parameters,
    weights
  };
  try {
    const add = vi.spyOn(graph, 'add');
    expect(() =>
      addSpatialRegressionRecipe(graph, {...props, output: {ols: {status: {code: complete}}}})
    ).toThrow(/overlap/);
    expect(add).not.toHaveBeenCalled();
    const result = addSpatialRegressionRecipe(graph, props);
    const diagnostics = result.contributors.find(
      contributor => contributor instanceof GPUSpatialRegressionDiagnostics
    ) as GPUSpatialRegressionDiagnostics;
    const loss = diagnostics.props.weights.status?.overflow;
    expect(loss).toBeDefined();
    if (!loss) throw new Error('missing certificate gate');
    const moran = result.contributors.find(
      contributor => contributor instanceof GPULocalMoran
    ) as GPULocalMoran;
    expect(moran.props.weights.status?.overflow === loss).toBe(true);
    const memory = createCPUGraphMemory();
    memory.write(weights.offsets, Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7));
    memory.write(requiredCount, Uint32Array.of(7));
    const nodes = (graph as unknown as {nodes: GPUCommandNode[]}).nodes.filter(node =>
      node.id.startsWith('cert-weights-input')
    );
    for (const certificate of [1, 0, 1]) {
      memory.write(complete, Uint32Array.of(certificate));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(loss))).toEqual([certificate === 0 ? 1 : 0]);
    }
    expect(result.ols.status.code.buffer).not.toBe(complete.buffer);
  } finally {
    parameters.destroy();
    device.destroy();
  }
});

it('S3 compares the actual scalar OLS status range and retains the oversized output view', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: 64
  });
  const status = createTransientView(graph, 'status', 'uint32', 128);
  const requiredCount = graph.createDataView(status.buffer, {
    format: 'uint32',
    byteOffset: 256,
    length: 1
  });
  try {
    const result = addSpatialRegressionRecipe(graph, {
      predictors: createTransientView(graph, 'predictors', 'float32', 7),
      response: createTransientView(graph, 'response', 'float32', 7),
      parameters,
      weights: {
        offsets: createTransientView(graph, 'offsets', 'uint32', 8),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 7),
        weights: createTransientView(graph, 'weights', 'float32', 7),
        status: {requiredCount}
      },
      output: {ols: {status: {code: status}}}
    });
    expect(result.ols.status.code).toBe(status);
    expect(result.status.stages[0].code?.length).toBe(1);
  } finally {
    parameters.destroy();
    device.destroy();
  }
});

it.each([
  undefined,
  2
])('DEF-11 resolves fourteen predictor values / seven responses with count %s', predictorCount => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: 64
  });
  const props = {
    id: 'infer',
    predictorCount,
    predictors: createTransientView(graph, 'predictors', 'float32', 14),
    response: createTransientView(graph, 'response', 'float32', 7),
    parameters,
    weights: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 8),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 8),
      weights: createTransientView(graph, 'weights', 'float32', 8)
    }
  };
  try {
    const result = addSpatialRegressionRecipe(graph, props);
    expect(result.ols.coefficients.length).toBe(3);
    for (const contributor of result.contributors) {
      if (
        contributor instanceof GPUOrdinaryLeastSquares ||
        contributor instanceof GPUSpatialRegressionDiagnostics
      )
        expect(contributor.props.predictorCount).toBe(2);
    }
  } finally {
    parameters.destroy();
    device.destroy();
  }
});

it.each([
  undefined,
  2
])('DEF-11 rejects an inconsistent design before scheduling: %s', predictorCount => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: 64
  });
  const props = {
    id: 'shape',
    predictorCount,
    predictors: createTransientView(graph, 'predictors', 'float32', 13),
    response: createTransientView(graph, 'response', 'float32', 7),
    parameters,
    weights: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 8),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 8),
      weights: createTransientView(graph, 'weights', 'float32', 8)
    }
  };
  try {
    const add = vi.spyOn(graph, 'add');
    expect(() => addSpatialRegressionRecipe(graph, props)).toThrow(/predictor/);
    expect(add).not.toHaveBeenCalled();
    expect(() =>
      addSpatialRegressionRecipe(graph, {
        ...props,
        predictors: createTransientView(graph, 'fixed-predictors', 'float32', 14)
      })
    ).not.toThrow();
  } finally {
    parameters.destroy();
    device.destroy();
  }
});

it.each([
  'count',
  'requiredCount'
] as const)('W5-16 rejects an OLS status alias of weights.status.%s before mutation and allows same-ID retry', name => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: 64
  });
  const evidence = createTransientView(graph, 'evidence', 'uint32', 1);
  const props = {
    id: 'retry',
    predictorCount: 1,
    predictors: createTransientView(graph, 'predictors', 'float32', 7),
    response: createTransientView(graph, 'response', 'float32', 7),
    parameters,
    weights: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 8),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 8),
      weights: createTransientView(graph, 'weights', 'float32', 8),
      status: {[name]: evidence}
    }
  };
  try {
    const add = vi.spyOn(graph, 'add');
    expect(() =>
      addSpatialRegressionRecipe(graph, {...props, output: {ols: {status: {code: evidence}}}})
    ).toThrow(/overlap/);
    expect(add).not.toHaveBeenCalled();
    const result = addSpatialRegressionRecipe(graph, props);
    expect(result.ols.status.code.buffer).not.toBe(evidence.buffer);
  } finally {
    parameters.destroy();
    device.destroy();
  }
});

it.each([false, true])('RCP-104 local fit stage is present only when enabled: %s', enabled => {
  const device = createNullWebGPUDevice();
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: 64
  });
  try {
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    const recipe = addSpatialRegressionRecipe(graph, {
      predictorCount: 1,
      predictors: view('predictors', 'float32', 4),
      response: view('response', 'float32', 4),
      parameters,
      weights: {
        offsets: view('offsets', 'uint32', 5),
        neighbors: view('neighbors', 'uint32', 4),
        weights: view('weights', 'float32', 4)
      },
      localFits: enabled
        ? {
            positions: view('positions', 'float32x2', 4),
            parameters: view('gwr', 'float32', 64),
            maximumBandwidthCount: 1
          }
        : undefined
    });
    const stage = recipe.status.stages.find(
      entry => entry.stage === 'geographically-weighted-regression'
    );
    if (enabled) {
      expect(stage !== undefined).toBe(true);
      expect(stage?.rowCodes === recipe.localFits?.localStatus).toBe(true);
      expect(stage?.rowCodes?.length).toBe(4);
      expect(stage?.code).toBeUndefined();
    } else expect(stage).toBeUndefined();
  } finally {
    parameters.destroy();
    device.destroy();
  }
});

it('B1 and earlier scalar-code note bound stage views without changing oversized overrides', () => {
  const device = createNullWebGPUDevice();
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: 64
  });
  try {
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    const localBuffer = view('local-buffer', 'uint32', 12);
    const localStatus = graph.createDataView(localBuffer.buffer, {
      format: 'uint32',
      byteOffset: 8,
      length: 10
    });
    const olsCode = view('ols-code', 'uint32', 3);
    const diagnosticCode = view('diagnostic-code', 'uint32', 2);
    const recipe = addSpatialRegressionRecipe(graph, {
      predictorCount: 1,
      predictors: view('predictors', 'float32', 6),
      response: view('response', 'float32', 6),
      parameters,
      weights: {
        offsets: view('offsets', 'uint32', 7),
        neighbors: view('neighbors', 'uint32', 6),
        weights: view('weights', 'float32', 6)
      },
      localFits: {
        positions: view('positions', 'float32x2', 6),
        parameters: view('gwr', 'float32', 64),
        maximumBandwidthCount: 1
      },
      output: {
        ols: {status: {code: olsCode}},
        diagnostics: {status: {code: diagnosticCode}},
        localFits: {localStatus}
      }
    });
    const stages = recipe.status.stages;
    const local = stages.find(stage => stage.stage === 'geographically-weighted-regression')!;
    expect(local.rowCodes?.length).toBe(6);
    expect(local.rowCodes?.buffer === localStatus.buffer).toBe(true);
    expect(local.rowCodes?.byteOffset).toBe(8);
    expect(recipe.localFits?.localStatus === localStatus).toBe(true);
    for (const [name, original] of [
      ['ordinary-least-squares', olsCode],
      ['spatial-diagnostics', diagnosticCode]
    ] as const) {
      const stage = stages.find(entry => entry.stage === name)!;
      expect(stage.code?.length).toBe(1);
      expect(stage.code?.buffer === original.buffer).toBe(true);
      expect(stage.status.invalidCount?.length).toBe(1);
      expect(stage.status.invalidCount?.buffer === original.buffer).toBe(false);
    }
    expect(recipe.ols.status.code === olsCode).toBe(true);
    expect(recipe.diagnostics.status.code === diagnosticCode).toBe(true);
    // Six input rows, four admitted to OLS and one remaining usable in diagnostics.
    // Independent counts must be published even when the model status is nonzero.
    const memory = createCPUGraphMemory();
    memory.write(recipe.ols.summary, Float32Array.of(4));
    const diagnosticsSummary = new Float32Array(recipe.diagnostics.summary.length);
    diagnosticsSummary[GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_EXCLUDED_ROW_COUNT] = 5;
    memory.write(recipe.diagnostics.summary, diagnosticsSummary);
    memory.write(olsCode, Uint32Array.of(1, 77, 77));
    memory.write(diagnosticCode, Uint32Array.of(2, 77));
    const countNodes = (graph as unknown as {nodes: GPUCommandNode[]}).nodes.filter(node =>
      node.id.endsWith('-excluded-row-counts')
    );
    expect(countNodes).toHaveLength(1);
    executeCommandNodesOnCPU(countNodes, memory);
    expect(Array.from(memory.readUint32(stages[0].status.invalidCount!))).toEqual([2]);
    expect(Array.from(memory.readUint32(stages[1].status.invalidCount!))).toEqual([5]);
    expect(Array.from(memory.readUint32(olsCode))).toEqual([1, 77, 77]);
    expect(Array.from(memory.readUint32(diagnosticCode))).toEqual([2, 77]);
  } finally {
    parameters.destroy();
    device.destroy();
  }
});
