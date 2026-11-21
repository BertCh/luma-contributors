// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import typescript from 'typescript';
import {Buffer} from '@luma.gl/core';
import {DynamicBuffer} from '@luma.gl/core/dynamic';
import * as validationLeaf from '../../src/analysis-core/kernels/analysis-validation-error';
import * as contributorUtils from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {getCoreDefaultBuffer} from '../../src/analysis-core/contracts/network-topology';
import {planGraphChunkGroups} from '../../src/analysis-core/kernels/graph-chunk-groups';
import {validateDisjointWritableViews} from '../../src/analysis-core/kernels/graph-chunk-dispatch';
import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {
  AnalysisValidationError,
  validateSinglePackedView
} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../../src/testing/gpu-contributor-test-utils';

it('TSC-01 preserves packed data acceptance and structured vector/format refusals across module conditions', () => {
  const device = createNullWebGPUDevice();
  try {
    const {GPUCommandGraph: ForeignGraph, GraphVectorView: ForeignVector} = createRequire(
      import.meta.url
    )('@luma.gl/graph/gpu-core') as {
      GPUCommandGraph: typeof GPUCommandGraph;
      GraphVectorView: typeof GraphVectorView;
    };
    const identity = {id: 'caller with spaces', contributor: 'PackedConsumer'};
    for (const [Graph, Vector] of [
      [GPUCommandGraph, GraphVectorView],
      [ForeignGraph, ForeignVector]
    ] as const) {
      const graph = new Graph(device);
      const data = createTransientView(graph, 'data', 'float32', 1);
      expect(() =>
        validateSinglePackedView(data, ['float32'], identity, 'input values')
      ).not.toThrow();
      const vector = new Vector({
        id: 'singleton',
        name: 'singleton',
        format: 'float32',
        length: 1,
        valueLength: 1,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data: [data]
      });
      for (const [view, received, message] of [
        [vector, 'vector', 'must be one packed view'],
        [
          createTransientView(graph, 'wrong', 'uint32', 1),
          'uint32',
          'must be packed, uint32-aligned float32 GPU data; received uint32'
        ]
      ] as const) {
        try {
          validateSinglePackedView(view, ['float32'], identity, 'input values');
          expect.fail('expected a structured refusal');
        } catch (error) {
          expect(error).toBeInstanceOf(AnalysisValidationError);
          expect(error).toMatchObject({
            id: identity.id,
            contributor: identity.contributor,
            code: 'VIEW_FORMAT',
            path: 'input values',
            received,
            message: `${identity.id} input values ${message}`
          });
        }
      }
      // Retain the legacy combined-label overload as well.
      expect(() => validateSinglePackedView(data, ['float32'], 'legacy values')).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

// Execute the owned resolver source under real CommonJS dependency resolution without a build.
function loadCommonJSBufferResolver(): typeof getCoreDefaultBuffer {
  const source = readFileSync(
    new URL('../../src/analysis-core/contracts/network-topology.ts', import.meta.url),
    'utf8'
  );
  const exports = {} as {getCoreDefaultBuffer: typeof getCoreDefaultBuffer};
  const require = createRequire(import.meta.url);
  runInNewContext(
    typescript.transpileModule(source, {
      compilerOptions: {
        module: typescript.ModuleKind.CommonJS,
        target: typescript.ScriptTarget.ES2022
      }
    }).outputText,
    {
      exports,
      require: (specifier: string) =>
        specifier === '../kernels/gpu-contributor-utils' ? contributorUtils : require(specifier)
    }
  );
  return exports.getCoreDefaultBuffer;
}

it('W5-01 resolves genuine ESM/CommonJS dynamic buffers and rejects physical write aliases', () => {
  const device = createNullWebGPUDevice();
  const {DynamicBuffer: ForeignDynamicBuffer} = createRequire(import.meta.url)(
    '@luma.gl/core/dynamic'
  ) as {
    DynamicBuffer: typeof DynamicBuffer;
  };
  expect(ForeignDynamicBuffer).not.toBe(DynamicBuffer);
  const resolveCommonJS = loadCommonJSBufferResolver();
  const failures: string[] = [];
  try {
    for (const [condition, Dynamic] of [
      ['esm', DynamicBuffer],
      ['commonjs', ForeignDynamicBuffer]
    ] as const) {
      const dynamic = new Dynamic(device, {id: condition, byteLength: 16, usage: Buffer.STORAGE});
      const distinct = device.createBuffer({byteLength: 16, usage: Buffer.STORAGE});
      try {
        // One 16-byte allocation, not two allocations, is the independent identity expectation.
        for (const [resolverName, resolve] of [
          ['esm', getCoreDefaultBuffer],
          ['commonjs', resolveCommonJS]
        ] as const) {
          if (resolve(dynamic) !== dynamic.buffer) failures.push(`${resolverName} <- ${condition}`);
          expect(resolve(dynamic.buffer)).toBe(dynamic.buffer);
        }
        const graph = new GPUCommandGraph(device);
        const importView = (name: string, buffer: Buffer | DynamicBuffer, byteOffset = 0) =>
          graph.createDataView(
            graph.importBuffer({id: name, byteLength: 16, usage: Buffer.STORAGE}, buffer),
            {format: 'uint32', length: 1, byteOffset}
          );
        const wrapped = importView('wrapped', dynamic);
        const backing = importView('backing', dynamic.buffer);
        const other = importView('distinct', distinct);
        try {
          validateDisjointWritableViews(condition, {wrapped, backing});
          failures.push(`${condition} overlapping writes accepted`);
        } catch (error) {
          expect(String(error)).toMatch(/overlap/);
        }
        expect(() => validateDisjointWritableViews(condition, {wrapped, other})).not.toThrow();
        const adjacent = importView('adjacent', dynamic.buffer, 4);
        expect(() =>
          validateDisjointWritableViews(condition, {wrapped, adjacent}, {separateDispatches: true})
        ).not.toThrow();
        try {
          validateDisjointWritableViews(
            condition,
            {wrapped, adjacent},
            {strictAlignedBindings: true}
          );
          failures.push(`${condition} co-bound aliases accepted`);
        } catch (error) {
          expect(String(error)).toMatch(/aligned binding/);
        }
      } finally {
        distinct.destroy();
        dynamic.destroy();
      }
    }
    expect(failures).toEqual([]);
  } finally {
    device.destroy();
  }
});

it('VAL-01 checks every captured required path before reading any earlier view format', () => {
  const descriptor = {required: true, viewType: 'data', formats: ['float32']} as const;
  for (const missingPath of ['missing', 'output.status.count']) {
    for (const reverse of [false, true]) {
      for (const malformed of [
        {},
        {
          get format(): never {
            throw new Error('format getter visited');
          }
        }
      ]) {
        let captures = 0;
        const ports = {
          get earlier() {
            captures++;
            return malformed;
          },
          output: {status: {}}
        };
        const entries = [
          ['earlier', descriptor],
          [missingPath, descriptor],
          ['optional', {...descriptor, required: false}]
        ] as const;
        const descriptors = Object.fromEntries(reverse ? [...entries].reverse() : entries);
        expect(() =>
          contributorUtils.validateGraphPorts('required-first', ports, descriptors)
        ).toThrowError(
          expect.objectContaining({code: 'MISSING_PROP', path: missingPath, received: undefined})
        );
        expect(captures).toBe(1);
      }
    }
  }
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const earlier = createTransientView(graph, 'valid', 'float32', 1);
    let captures = 0;
    const ports = {
      get earlier() {
        captures++;
        return earlier;
      }
    };
    expect(() =>
      contributorUtils.validateGraphPorts('optional', ports, {
        earlier: descriptor,
        'omitted.nested': {...descriptor, required: false}
      })
    ).not.toThrow();
    expect(captures).toBe(1);
  } finally {
    device.destroy();
  }
});

it('VAL-12 retains caller identity and complete spaced chunk paths for template and field refusals', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const storage = createTransientView(graph, 'group-storage', 'float32', 4);
    const packed = graph.createDataView(storage.buffer, {format: 'float32', length: 2});
    const strided = graph.createDataView(storage.buffer, {
      format: 'float32',
      length: 2,
      byteStride: 8
    });
    const vector = (name: string, data: (typeof packed)[]) =>
      new GraphVectorView({
        id: name,
        name,
        format: 'float32',
        length: 4,
        valueLength: 4,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data
      });
    for (const explicit of [false, true]) {
      const id = 'caller ID with spaces';
      const identity = explicit ? {id, contributor: 'GroupConsumer'} : id;
      for (const templateFailure of [true, false]) {
        const template = vector('queries', templateFailure ? [packed, strided] : [packed, packed]);
        const field = vector('field', [packed, strided]);
        const path = templateFailure ? 'template chunk 1' : 'output values chunk 1';
        expect(() =>
          planGraphChunkGroups(identity, {view: template}, {'output values': {view: field}})
        ).toThrowError(
          expect.objectContaining({
            id,
            contributor: explicit ? 'GroupConsumer' : id,
            path,
            message: `${id} ${path} must be packed, uint32-aligned float32 GPU data; received float32`
          })
        );
      }
    }
  } finally {
    device.destroy();
  }
});

it('W5-02 reports raw and differing float32 values without changing structured received', () => {
  // Math.fround(1e40) is Infinity; 2 + 2^-23 is the ties-to-even midpoint above 2.
  for (const [value, minimum, maximum, suffix] of [
    [3, 0, 2, '; received 3'],
    [1e40, -Infinity, Infinity, '; received 1e+40; encoded Infinity'],
    [2 + 2 ** -23, 2 + 2 ** -23, 3, '; received 2.0000001192092896; encoded 2']
  ] as const) {
    expect(() =>
      contributorUtils.validateFiniteNumber('number-check', 'threshold', value, minimum, maximum)
    ).toThrowError(
      expect.objectContaining({
        code: 'NUMBER_RANGE',
        path: 'threshold',
        received: value,
        message: `number-check threshold must be a finite number in [${minimum}, ${maximum}]${suffix}`
      })
    );
  }
});

it('W5-03 exposes pure metadata validators with the same functions and structured error class', () => {
  expect(validationLeaf.validateOption).toBe(contributorUtils.validateOption);
  expect(validationLeaf.validateRequirement).toBe(contributorUtils.validateRequirement);
  expect(validationLeaf.AnalysisValidationError).toBe(AnalysisValidationError);
  const source = readFileSync(
    new URL('../../src/analysis-core/kernels/analysis-validation-error.ts', import.meta.url),
    'utf8'
  );
  const exports = {} as typeof validationLeaf;
  // Metadata-only loading must succeed with no graph, device or other runtime imports available.
  runInNewContext(
    typescript.transpileModule(source, {
      compilerOptions: {
        module: typescript.ModuleKind.CommonJS,
        target: typescript.ScriptTarget.ES2022
      }
    }).outputText,
    {
      exports,
      require: (specifier: string): never => {
        throw new Error(`unexpected runtime import ${specifier}`);
      }
    }
  );
  for (const validators of [validationLeaf, exports]) {
    const validateMetadataOption: typeof validationLeaf.validateOption = validators.validateOption;
    const validateMetadataRequirement: typeof validationLeaf.validateRequirement =
      validators.validateRequirement;
    const identity = {id: 'metadata-descriptor', contributor: 'Descriptor'};
    expect(() =>
      validateMetadataOption(identity, 'mode', 'unknown', ['exact', 'approximate'])
    ).toThrowError(
      expect.objectContaining({
        name: 'AnalysisValidationError',
        code: 'INVALID_OPTION',
        id: identity.id,
        contributor: identity.contributor,
        path: 'mode',
        received: 'unknown',
        message: 'metadata-descriptor mode must be exact or approximate; received unknown'
      })
    );
    try {
      validateMetadataRequirement(identity, false, 'size must not exceed capacity', {
        code: 'INTEGER_RANGE',
        path: 'size',
        received: 5,
        expected: {maximum: 4}
      });
      expect.fail('expected requirement refusal');
    } catch (error) {
      expect(error).toBeInstanceOf(validators.AnalysisValidationError);
      expect(error).toMatchObject({
        code: 'INTEGER_RANGE',
        id: identity.id,
        contributor: identity.contributor,
        path: 'size',
        received: 5,
        expected: {maximum: 4},
        message: 'metadata-descriptor size must not exceed capacity'
      });
    }
    expect(() =>
      validateMetadataOption(identity, 'mode', 'exact', ['exact', 'approximate'])
    ).not.toThrow();
    expect(() => validateMetadataRequirement(identity, true, 'size must fit')).not.toThrow();
  }
});

it.each([
  [Symbol('unset'), 'Symbol(unset)'],
  [Object.create(null), '<unprintable>']
] as const)('REVIEW B2 preserves structured numeric errors for %s', (received, rendered) => {
  let failure: unknown;
  try {
    contributorUtils.validateFiniteNumber('model', 'size', received as unknown as number, 1, 4);
  } catch (error) {
    failure = error;
  }
  expect(failure).toBeInstanceOf(AnalysisValidationError);
  expect(failure).toMatchObject({code: 'NUMBER_RANGE', path: 'size'});
  expect((failure as AnalysisValidationError).received).toBe(received);
  expect((failure as Error).message).toBe(
    `model size must be a finite number in [1, 4]; received ${rendered}`
  );
});

it.each([
  [-1e-52, 'received -1e-52; encoded -0'],
  [-0, 'received -0']
] as const)('REVIEW S2 preserves signed zero: %s', (received, suffix) => {
  // -1e-52 is below half the smallest float32 subnormal (2^-150); rounding retains its sign.
  expect(Object.is(Math.fround(-1e-52), -0)).toBe(true);
  let failure: unknown;
  try {
    contributorUtils.validateFiniteNumber('model', 'size', received, 1, 4);
  } catch (error) {
    failure = error;
  }
  expect(failure).toBeInstanceOf(AnalysisValidationError);
  expect((failure as AnalysisValidationError).received).toBe(received);
  expect((failure as Error).message).toBe(
    `model size must be a finite number in [1, 4]; ${suffix}`
  );
});

it('REVIEW earlier-fix note retains the full chunk-length path and actual received length', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const vector = (name: string, lengths: number[]) =>
      new GraphVectorView({
        id: name,
        name,
        format: 'float32',
        length: 3,
        valueLength: 3,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data: lengths.map((length, index) =>
          createTransientView(graph, `${name}-${index}`, 'float32', length)
        )
      });
    // Both domains have three rows, but the first query chunk needs exactly one weight row.
    const template = vector('queries', [1, 2]);
    const weights = vector('weights', [2, 1]);
    expect(() =>
      planGraphChunkGroups(
        {id: 'test west', contributor: 'GroupConsumer'},
        {view: template},
        {'weight rows': {view: weights}}
      )
    ).toThrowError(
      expect.objectContaining({
        code: 'VIEW_LENGTH',
        id: 'test west',
        contributor: 'GroupConsumer',
        path: 'weight rows chunk 0',
        received: 2,
        expected: {rows: 1},
        message:
          'test west weight rows chunk 0 must hold exactly 1 rows for the query chunk topology'
      })
    );
  } finally {
    device.destroy();
  }
});

// REVIEW B1 reproduction; this test is handed off with the outside-owned implementation fix.
it('REVIEW B1 public interpolation constructors preserve caller and chunk identity', async () => {
  const {GPUNaturalNeighbor, GPULocalKriging, GPULocalRBF, GPUConditionalGaussianSimulation} =
    await import('@luma-contributors/analysis/gpu-spatial-analysis');
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const packed = createTransientView(graph, 'first-query', 'float32x2', 2);
    const storage = createTransientView(graph, 'second-query', 'float32x2', 4);
    const strided = graph.createDataView(storage.buffer, {
      format: 'float32x2',
      length: 2,
      byteStride: 16
    });
    const queries = new GraphVectorView({
      id: 'query-chunks',
      name: 'query-chunks',
      format: 'float32x2',
      length: 4,
      valueLength: 8,
      stride: 2,
      byteStride: 8,
      rowByteLength: 8,
      data: [packed, strided]
    });
    const base = {
      id: 'test west',
      positions: createTransientView(graph, 'positions', 'float32x2', 3),
      output: {
        values: createTransientView(graph, 'output-values', 'float32x4', 4),
        status: createTransientView(graph, 'output-status', 'uint32', 4)
      }
    };
    const actual: unknown[] = [];
    for (const construct of [
      () =>
        new GPUNaturalNeighbor({
          ...base,
          queries,
          values: createTransientView(graph, 'samples', 'float32', 3),
          maximumNeighborCount: 3
        }),
      () =>
        new GPULocalKriging({
          ...base,
          supports: queries,
          mode: 'ordinary',
          neighborCount: 3,
          observations: createTransientView(graph, 'observations', 'float32x4', 3),
          covariance: {model: 'exponential', range: 1, sill: 1}
        }),
      () =>
        new GPULocalRBF({
          ...base,
          queries,
          values: createTransientView(graph, 'rbf-samples', 'float32', 3),
          neighborCount: 3
        }),
      () =>
        new GPUConditionalGaussianSimulation({
          id: base.id,
          positions: strided,
          queries: packed,
          seed: 1,
          replicateCount: 1,
          output: {
            values: createTransientView(graph, 'simulation-values', 'float32', 2),
            moments: createTransientView(graph, 'simulation-moments', 'float32x2', 2),
            status: createTransientView(graph, 'simulation-status', 'uint32', 1)
          },
          observations: createTransientView(graph, 'simulation-observations', 'float32x4', 3),
          covariance: {model: 'exponential', range: 1, sill: 1}
        })
    ]) {
      try {
        construct();
        expect.fail('expected strided chunk refusal');
      } catch (error) {
        expect(error).toBeInstanceOf(AnalysisValidationError);
        const failure = error as AnalysisValidationError;
        actual.push({id: failure.id, contributor: failure.contributor, path: failure.path});
      }
    }
    expect(actual).toEqual([
      {id: 'test west', contributor: 'GPUNaturalNeighbor', path: 'queries chunk 1'},
      {id: 'test west', contributor: 'GPULocalKriging', path: 'supports chunk 1'},
      {id: 'test west', contributor: 'GPULocalRBF', path: 'queries chunk 1'},
      {id: 'test west', contributor: 'GPUConditionalGaussianSimulation', path: 'positions chunk 0'}
    ]);
  } finally {
    device.destroy();
  }
});
