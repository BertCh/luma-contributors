// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCPUVector, GPUGraphVectorFormat} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import {runInNewContext} from 'node:vm';
import {
  createLanguageService,
  displayPartsToString,
  getDefaultLibFilePath,
  ModuleKind,
  ModuleResolutionKind,
  ScriptSnapshot,
  ScriptTarget,
  sys,
  transpileModule
} from 'typescript';
import {expect, expectTypeOf, it} from 'vitest';
import * as contributorUtils from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationInt32Rows,
  getCPUOperationUint32Rows,
  type CPUOperationInputs
} from '../../../src/analysis-core/operations/cpu-operation-rows';
import {
  GPUContributorOperation,
  GPU_CONTRIBUTOR_OPERATION_LOWERING,
  GPU_CONTRIBUTOR_OPERATION_TYPE,
  validateOperationGraphNames,
  type GPUContributorOperationViewGetter
} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {GPUVariogramModelFitOperation} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit-operation';

const NO_CONTRIBUTOR = () => ({getCommandNodes: () => []});

// Inspect the actual public types through a virtual consumer. This asks for quick-info only;
// it neither runs the repository's typecheck projects nor requests program diagnostics/emit.
function inspectOperationTypes(): Record<string, string> {
  const filename = fileURLToPath(new URL('./operation-type-probe.ts', import.meta.url));
  const source = `
    import type {GPUContributorOperation, GPUContributorOperationViewGetter,
      GPUContributorOperationOutput} from '../../../src/analysis-core/operations/gpu-contributor-operation';
    type IsAny<Value> = 0 extends (1 & Value) ? true : false;
    const getter: GPUContributorOperationViewGetter = port => {
      const contextualPortIsAny = null! as IsAny<typeof port>;
      throw new Error();
    };
    type Descriptor = {value: 'rows'; format: 'uint32'; length: 1};
    declare const selected: GPUContributorOperation<{alpha: Descriptor} | {beta: Descriptor}>;
    const alpha = selected.getOutput('alpha');
    const alphaCanBePresent = null! as (Exclude<typeof alpha, undefined> extends never ? false : true);
    const alphaCanBeAbsent = null! as (undefined extends typeof alpha ? true : false);
    declare const patterned: GPUContributorOperation<Record<\`band\${number}\`, Descriptor>>;
    const band = patterned.getOutput('band99');
    const bandCanBeAbsent = null! as (undefined extends typeof band ? true : false);
    declare const overlap: \`band\${string}\`;
    const overlappingBand = patterned.getOutput(overlap);
    const overlapCanBePresent = null! as (Exclude<typeof overlappingBand, undefined> extends never ? false : true);
    const overlapCanBeAbsent = null! as (undefined extends typeof overlappingBand ? true : false);
    declare const optional: GPUContributorOperation<{required: Descriptor; optional?: Descriptor}>;
    const optionalValue = optional.getOutput('optional');
    const requiredValue = optional.getOutput('required');
    const optionalCanBeAbsent = null! as (undefined extends typeof optionalValue ? true : false);
    const requiredCanBeAbsent = null! as (undefined extends typeof requiredValue ? true : false);
    declare const subclass: GPUContributorOperation<{labels: Descriptor & {byteCount(): number}}>;
    const labels = subclass.getOutput('labels');
    const methodIsPromised = null! as ('byteCount' extends keyof typeof labels ? true : false);
  `;
  const options = {
    strict: true,
    skipLibCheck: true,
    types: [],
    target: ScriptTarget.ES2022,
    module: ModuleKind.ESNext,
    moduleResolution: ModuleResolutionKind.Bundler
  };
  const service = createLanguageService({
    getCompilationSettings: () => options,
    getScriptFileNames: () => [filename],
    getScriptVersion: () => '0',
    getScriptSnapshot: path => {
      const text = path === filename ? source : sys.readFile(path);
      return text === undefined ? undefined : ScriptSnapshot.fromString(text);
    },
    getCurrentDirectory: () => process.cwd(),
    getDefaultLibFileName: getDefaultLibFilePath,
    fileExists: path => path === filename || sys.fileExists(path),
    readFile: path => (path === filename ? source : sys.readFile(path)),
    directoryExists: sys.directoryExists
  });
  try {
    return Object.fromEntries(
      [
        'contextualPortIsAny',
        'alphaCanBePresent',
        'alphaCanBeAbsent',
        'bandCanBeAbsent',
        'overlapCanBePresent',
        'overlapCanBeAbsent',
        'methodIsPromised',
        'optionalCanBeAbsent',
        'requiredCanBeAbsent'
      ].map(name => {
        const info = service.getQuickInfoAtPosition(filename, source.indexOf(name));
        return [name, displayPartsToString(info?.displayParts)];
      })
    );
  } finally {
    service.dispose();
  }
}

it('contextually types an unannotated getter implementation parameter as string', () => {
  const getter: GPUContributorOperationViewGetter = port => {
    expectTypeOf(port).toEqualTypeOf<string>();
    throw new Error(`fixture ${port}`);
  };
  expect(() => getter('mask')).toThrow('fixture mask');
  expect(inspectOperationTypes()['contextualPortIsAny']).toBe('const contextualPortIsAny: false');
});

it('types conditional selections and patterned output keys with possible absence', () => {
  const types = inspectOperationTypes();
  for (const name of [
    'alphaCanBePresent',
    'alphaCanBeAbsent',
    'bandCanBeAbsent',
    'overlapCanBePresent',
    'overlapCanBeAbsent'
  ]) {
    expect(types[name]).toBe(`const ${name}: true`);
  }
  expect(types['optionalCanBeAbsent']).toBe('const optionalCanBeAbsent: true');
  expect(types['requiredCanBeAbsent']).toBe('const requiredCanBeAbsent: false');
  const descriptor = {value: 'rows', format: 'uint32', length: 1} as const;
  const selections: ({alpha: typeof descriptor} | {beta: typeof descriptor})[] = [
    {alpha: descriptor},
    {beta: descriptor}
  ];
  for (const outputs of selections) {
    const operation = new GPUContributorOperation<typeof outputs>({
      id: 'conditional',
      inputs: {},
      outputs,
      createContributor: NO_CONTRIBUTOR
    });
    expectTypeOf(operation.getOutput('alpha')).toEqualTypeOf<typeof descriptor | undefined>();
    expect(operation.getOutput('alpha')).toEqual('alpha' in outputs ? descriptor : undefined);
  }
  const patterned = new GPUContributorOperation<Record<`band${number}`, typeof descriptor>>({
    id: 'pattern',
    inputs: {},
    outputs: {band3: descriptor},
    createContributor: NO_CONTRIBUTOR
  });
  expect(patterned.getOutput('band3')).toEqual(descriptor);
  expect(patterned.getOutput('band99')).toBeUndefined();
  expectTypeOf(patterned.getOutput('band99')).toEqualTypeOf<typeof descriptor | undefined>();
});

it('only promises the descriptor fields preserved by a frozen output snapshot', () => {
  class Descriptor {
    readonly value = 'rows';
    readonly format = 'uint32';
    readonly length = 1;
    byteCount() {
      return this.length * 4;
    }
  }
  const operation = new GPUContributorOperation({
    id: 'subtype',
    inputs: {},
    outputs: {labels: new Descriptor()},
    createContributor: NO_CONTRIBUTOR
  });
  const labels = operation.getOutput('labels');
  expect(labels).toEqual({value: 'rows', format: 'uint32', length: 1});
  expect(Object.isFrozen(labels)).toBe(true);
  expectTypeOf(labels).toEqualTypeOf<Readonly<{value: 'rows'; format: 'uint32'; length: 1}>>();
  expect(inspectOperationTypes()['methodIsPromised']).toBe('const methodIsPromised: false');
});

it('getOutput preserves selected descriptors and safely reports absent or dynamic ports', () => {
  const outputs = {mask: {value: 'graph-mask', format: 'uint32', length: 3}} as const;
  const operation = new GPUContributorOperation({
    id: 'selected',
    inputs: {},
    outputs,
    createContributor: NO_CONTRIBUTOR
  });
  expect(operation.getOutput('mask')).toEqual(outputs.mask);
  expectTypeOf(operation.getOutput('mask')).toEqualTypeOf<typeof outputs.mask>();
  expect(operation.getOutput('absent')).toBeUndefined();
  expectTypeOf(operation.getOutput('absent')).toEqualTypeOf<undefined>();
  const dynamic: string = 'absent';
  expect(operation.getOutput(dynamic)).toBeUndefined();
  expectTypeOf(operation.getOutput(dynamic)).toEqualTypeOf<typeof outputs.mask | undefined>();
  expect(operation.getOutput('constructor')).toBeUndefined();
  class SelectedOperation extends GPUContributorOperation<typeof outputs> {}
  const selected = new SelectedOperation({
    id: 'wrapper',
    inputs: {},
    outputs,
    createContributor: NO_CONTRIBUTOR
  });
  expectTypeOf(selected.getOutput('mask')).toEqualTypeOf<typeof outputs.mask>();
  const unselected: GPUContributorOperation = selected;
  expectTypeOf(unselected.getOutput(dynamic)).toEqualTypeOf<
    | Readonly<
        import('../../../src/analysis-core/operations/gpu-contributor-operation').GPUContributorOperationOutput
      >
    | undefined
  >();
});

it('preflights original logical ports before CPU destination keys can overwrite results', () => {
  // Distinct logical results would collapse to {count: ...} if keyed by destination first.
  expect(() =>
    validateOperationGraphNames(
      'adjust',
      {pValues: 'pValues'},
      {
        adjustedPValues: 'count',
        effectiveFamilySize: 'count'
      }
    )
  ).toThrow(
    'adjust outputs.effectiveFamilySize must name a graph value ("count") that no other port uses'
  );
  expect(() => validateOperationGraphNames('adjust', {}, {result: ''})).toThrow(
    'adjust outputs.result.value must be a non-empty graph value name'
  );
  expect(() => validateOperationGraphNames('adjust', {values: ''}, {})).toThrow(
    'adjust inputs.values must be a non-empty graph value name'
  );
  expect(() => validateOperationGraphNames('adjust', {values: 'rows'}, {result: 'rows'})).toThrow(
    'adjust outputs.result must name a graph value ("rows") that no other port uses'
  );
  expect(() => validateOperationGraphNames('adjust', {values: 'rows'}, {values: 'other'})).toThrow(
    'adjust port "values" must not be both an input and an output'
  );
  expect(() => validateOperationGraphNames('adjust', {'': 'rows'}, {})).toThrow(
    'adjust input port name must be non-empty'
  );
  expect(() => validateOperationGraphNames('adjust', {}, {'': 'rows'})).toThrow(
    'adjust output port name must be non-empty'
  );
  expect(() =>
    validateOperationGraphNames(
      'adjust',
      {pValues: 'pValues'},
      {
        adjustedPValues: 'adjusted',
        effectiveFamilySize: ''
      }
    )
  ).toThrow('adjust outputs.effectiveFamilySize.value must be a non-empty graph value name');
  // Names alone: no GPU allocation/row-count limits, and repeated input reads are valid.
  expect(() =>
    validateOperationGraphNames('cpu', {left: 'rows', right: 'rows'}, {sum: 'total'})
  ).not.toThrow();
});

it('the public CPU raster-medoid control preserves one selected value', async () => {
  const {CPURasterMedoidCompositeOperation} = await import(
    '../../../src/raster-analysis/raster-time-series/cpu-raster-medoid-composite-operation'
  );
  const operation = new CPURasterMedoidCompositeOperation({
    id: 'medoid-control',
    cellCount: 1,
    bandCount: 1,
    sliceCount: 1,
    values: 'values',
    output: {composite: 'composite', sliceIndex: 'sliceIndex', status: 'status'}
  });
  const result = await operation.execute({
    inputs: {
      values: makeRows('float32', [Float32Array.of(37.5)])
    }
  });
  // A singleton's medoid is itself: no distance algorithm or CPU twin supplies this expectation.
  expect(result['composite']).toBeInstanceOf(Float32Array);
  expect(Number.isFinite((result['composite'] as Float32Array)[0])).toBe(true);
  expect(result['composite']).toEqual(Float32Array.of(37.5));
  expect(result['sliceIndex']).toEqual(Uint32Array.of(0));
  expect(result['status']).toEqual(Uint32Array.of(0));
});

function makeRows(
  format: GPUCPUVector['format'],
  chunks: readonly (Float32Array | Uint32Array | Int32Array)[]
): GPUCPUVector {
  // `length` and each chunk's `length` count rows; the arrays hold `components` values per row.
  const {components} = getGPUVectorFormatInfo(format);
  return {
    format,
    length: chunks.reduce((total, chunk) => total + chunk.length / components, 0),
    chunks: chunks.map(values => ({values, length: values.length / components})),
    pack() {
      const packed = new (chunks[0].constructor as typeof Float32Array)(this.length * components);
      let offset = 0;
      for (const chunk of chunks) {
        packed.set(chunk, offset);
        offset += chunk.length;
      }
      return packed;
    }
  };
}

it('a contributor operation declares its ports as a contract over graph values', () => {
  const operation = new GPUContributorOperation({
    id: 'fit',
    // Two ports may read the same graph value; the contract lists it once.
    inputs: {distances: 'lags', weights: 'lags', counts: 'pairs'},
    outputs: {
      parameters: {value: 'fitted', format: 'float32x4', length: 3},
      status: {value: 'flags', format: 'uint32', length: 3}
    },
    createContributor: NO_CONTRIBUTOR
  });
  expect(operation.type).toBe(GPU_CONTRIBUTOR_OPERATION_TYPE);
  expect(GPU_CONTRIBUTOR_OPERATION_LOWERING.type).toBe(GPU_CONTRIBUTOR_OPERATION_TYPE);
  const {contract} = operation;
  expect(contract.values).toEqual([
    {value: 'lags', access: 'read'},
    {value: 'pairs', access: 'read'},
    {value: 'fitted', access: 'write', shape: {length: 3}},
    {value: 'flags', access: 'write', shape: {length: 3}}
  ]);
  expect(contract.outputs).toEqual({
    fitted: {type: 'vector', format: 'float32x4', length: 3},
    flags: {type: 'vector', format: 'uint32', length: 3}
  });
});

it('a contributor operation rejects ports that cannot be planned, naming the port', () => {
  const output = {value: 'result', format: 'float32', length: 4} as const;
  expect(
    () =>
      new GPUContributorOperation({
        id: 'smooth',
        inputs: {values: 'values'},
        outputs: {values: output},
        createContributor: NO_CONTRIBUTOR
      })
  ).toThrow('smooth port "values" must not be both an input and an output');
  // In place: the published mixed executor has no read-write values.
  expect(
    () =>
      new GPUContributorOperation({
        id: 'smooth',
        inputs: {values: 'values'},
        outputs: {result: {...output, value: 'values'}},
        createContributor: NO_CONTRIBUTOR
      })
  ).toThrow('smooth outputs.result must name a graph value ("values") that no other port uses');
  expect(
    () =>
      new GPUContributorOperation({
        id: 'smooth',
        inputs: {values: 'values'},
        outputs: {result: output, copy: output},
        createContributor: NO_CONTRIBUTOR
      })
  ).toThrow('smooth outputs.copy must name a graph value ("result") that no other port uses');
  expect(
    () =>
      new GPUContributorOperation({
        id: 'smooth',
        inputs: {values: 'values'},
        outputs: {result: {...output, length: 0}},
        createContributor: NO_CONTRIBUTOR
      })
  ).toThrow('smooth outputs.result.length must be an integer in [1, 4294967295]');
  // A port the operation names as one that may be empty accepts zero rows, and only that port.
  const empty = new GPUContributorOperation({
    id: 'smooth',
    inputs: {values: 'values'},
    outputs: {result: {...output, length: 0}, count: {value: 'count', format: 'uint32', length: 1}},
    emptyOutputPorts: ['result'],
    createContributor: NO_CONTRIBUTOR
  });
  expect(empty.contract.outputs).toEqual({
    result: {type: 'vector', format: 'float32', length: 0},
    count: {type: 'vector', format: 'uint32', length: 1}
  });
  expect(
    () =>
      new GPUContributorOperation({
        id: 'smooth',
        inputs: {values: 'values'},
        outputs: {result: {...output, length: -1}},
        emptyOutputPorts: ['result'],
        createContributor: NO_CONTRIBUTOR
      })
  ).toThrow('smooth outputs.result.length must be an integer in [0, 4294967295]');
  // A listed name that is not an output port is ignored: optional ports are listed unconditionally.
  expect(
    () =>
      new GPUContributorOperation({
        id: 'smooth',
        inputs: {values: 'values'},
        outputs: {result: output},
        emptyOutputPorts: ['optional'],
        createContributor: NO_CONTRIBUTOR
      })
  ).not.toThrow();
});

it('rejects invalid graph value names at construction with the contributor ID and prop', () => {
  // JavaScript callers and deserialized props can bypass the string types. In particular, a
  // declaration object is not a name, even though luma.gl accepts it elsewhere as a reference.
  for (const value of ['', undefined, null, 0, false, {id: 'values'}, ['values']]) {
    expect(
      () =>
        new GPUContributorOperation({
          id: 'named-input',
          inputs: {positions: value as string},
          outputs: {},
          createContributor: NO_CONTRIBUTOR
        })
    ).toThrow('named-input inputs.positions must be a non-empty graph value name');
    expect(
      () =>
        new GPUContributorOperation({
          id: 'named-output',
          inputs: {},
          outputs: {result: {value: value as string, format: 'float32', length: 1}},
          createContributor: NO_CONTRIBUTOR
        })
    ).toThrow('named-output outputs.result.value must be a non-empty graph value name');
  }
});

it('rejects output formats outside the portable graph vector contract at construction', () => {
  for (const format of [
    '',
    undefined,
    null,
    32,
    'float16',
    'unorm8x4',
    'float32x1',
    'float32x5',
    'float32\n',
    'vertex-list<float32x2>',
    'fixed-size-list<float32,4>',
    {toString: () => 'float32'}
  ]) {
    expect(
      () =>
        new GPUContributorOperation({
          id: 'portable-output',
          inputs: {},
          outputs: {
            result: {value: 'result', format: format as GPUGraphVectorFormat, length: 1}
          },
          createContributor: NO_CONTRIBUTOR
        })
    ).toThrow(
      'portable-output outputs.result.format must be float32, sint32 or uint32 with one to four components'
    );
  }
});

it('accepts every portable output format without changing the declared shape', () => {
  // Enumerated independently of the validator: three scalar types, each with 1–4 components.
  const formats = [
    'float32',
    'float32x2',
    'float32x3',
    'float32x4',
    'sint32',
    'sint32x2',
    'sint32x3',
    'sint32x4',
    'uint32',
    'uint32x2',
    'uint32x3',
    'uint32x4'
  ] as const;
  for (const format of formats) {
    const operation = new GPUContributorOperation({
      id: 'portable-output',
      inputs: {},
      outputs: {result: {value: 'result', format, length: 4}},
      createContributor: NO_CONTRIBUTOR
    });
    expect(operation.contract.outputs).toEqual({result: {type: 'vector', format, length: 4}});
  }
});

it('validates output length bounds at construction, including explicitly empty ports', () => {
  for (const minimum of [0, 1]) {
    for (const length of [-1, 0.5, Number.NaN, Number.POSITIVE_INFINITY, 0x100000000]) {
      expect(
        () =>
          new GPUContributorOperation({
            id: 'bounded-output',
            inputs: {},
            outputs: {result: {value: 'result', format: 'float32', length}},
            emptyOutputPorts: minimum === 0 ? ['result'] : [],
            createContributor: NO_CONTRIBUTOR
          })
      ).toThrow(
        `bounded-output outputs.result.length must be an integer in [${minimum}, 4294967295]`
      );
    }
    for (const length of [minimum, 0xffffffff]) {
      const operation = new GPUContributorOperation({
        id: 'bounded-output',
        inputs: {},
        outputs: {result: {value: 'result', format: 'float32', length}},
        emptyOutputPorts: minimum === 0 ? ['result'] : [],
        createContributor: NO_CONTRIBUTOR
      });
      expect(operation.contract.outputs).toEqual({
        result: {type: 'vector', format: 'float32', length}
      });
    }
  }
});

it('rejects a prop-derived zero length before the contributor is lowered', () => {
  // No views or graph are needed to reject models=[]: the operation derives three zero lengths.
  expect(
    () =>
      new GPUVariogramModelFitOperation({
        id: 'empty-fit',
        distances: 'distances',
        semivariances: 'semivariances',
        pairCounts: 'pairCounts',
        models: [],
        output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'}
      })
  ).toThrow('empty-fit outputs.parameters.length must be an integer in [1, 4294967295]');
});

it('an operation destroys the resource-owning contributor of the previous run', () => {
  const destroyed: number[] = [];
  let built = 0;
  const operation = new GPUContributorOperation({
    id: 'project',
    inputs: {positions: 'positions'},
    outputs: {result: {value: 'projected', format: 'float32x2', length: 4}},
    createContributor: () => {
      const serial = built++;
      return {getCommandNodes: () => [], destroy: () => destroyed.push(serial)};
    }
  });
  const getView = (() => {
    throw new Error('not resolved in this test');
  }) as Parameters<typeof operation.createContributor>[0];
  operation.createContributor(getView);
  expect(destroyed).toEqual([]);
  // The second run replaces the first run's contributor.
  operation.createContributor(getView);
  expect(destroyed).toEqual([0]);
  operation.destroy();
  operation.destroy();
  expect(destroyed).toEqual([0, 1]);
  // A contributor without resources is not tracked.
  const plain = new GPUContributorOperation({
    id: 'plain',
    inputs: {},
    outputs: {result: {value: 'result', format: 'uint32', length: 1}},
    createContributor: NO_CONTRIBUTOR
  });
  plain.createContributor(getView);
  expect(() => plain.destroy()).not.toThrow();
});

it('the lowering refuses an operation of another class that carries its type', () => {
  expect(() =>
    GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(
      {id: 'impostor', type: GPU_CONTRIBUTOR_OPERATION_TYPE},
      {} as Parameters<typeof GPU_CONTRIBUTOR_OPERATION_LOWERING.lower>[1]
    )
  ).toThrow('impostor is not a GPUContributorOperation');
});

it('lowers import-to-require and require-to-import operation identities', () => {
  // Targeted source fixture, not a package build. The tooling gate must also exercise packed
  // conditional exports. Only the relative validation dependency is supplied by the host here.
  const source = readFileSync(
    new URL('../../../src/analysis-core/operations/gpu-contributor-operation.ts', import.meta.url),
    'utf8'
  );
  const {outputText} = transpileModule(source, {
    compilerOptions: {module: ModuleKind.CommonJS, target: ScriptTarget.ES2022}
  });
  const commonjs = {
    exports: {} as typeof import('../../../src/analysis-core/operations/gpu-contributor-operation')
  };
  const require = createRequire(import.meta.url);
  runInNewContext(outputText, {
    module: commonjs,
    exports: commonjs.exports,
    require: (name: string) =>
      name === '../kernels/gpu-contributor-utils' ? contributorUtils : require(name)
  });
  expect(commonjs.exports.GPUContributorOperation).not.toBe(GPUContributorOperation);
  const imported = new GPUContributorOperation({
    id: 'imported',
    inputs: {source: 'values'},
    outputs: {result: {value: 'flags', format: 'uint32', length: 1}},
    createContributor: getView => {
      expect(getView('source', 'float32').format).toBe('float32');
      expect(getView('result', 'uint32').format).toBe('uint32');
      return NO_CONTRIBUTOR();
    }
  });
  const required = new commonjs.exports.GPUContributorOperation({
    id: 'required',
    inputs: {source: 'values'},
    outputs: {result: {value: 'flags', format: 'uint32', length: 1}},
    createContributor: getView => {
      expect(getView('source', 'float32').format).toBe('float32');
      expect(getView('result', 'uint32').format).toBe('uint32');
      return NO_CONTRIBUTOR();
    }
  });
  let lowered = 0;
  const resolved: string[] = [];
  const context = {
    resolveVector: (value: string) => {
      resolved.push(value);
      return {data: [{format: value === 'values' ? 'float32' : 'uint32'}]};
    },
    lower: () => {
      lowered++;
    }
  } as unknown as Parameters<typeof GPU_CONTRIBUTOR_OPERATION_LOWERING.lower>[1];
  commonjs.exports.GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(imported, context);
  GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(required, context);
  expect(lowered).toBe(2);
  expect(resolved).toEqual(['values', 'flags', 'values', 'flags']);
  // A copied brand does not excuse an invalid protocol implementation or output descriptor.
  const malformedFactory = {...required, createContributor: undefined, destroy: () => {}};
  expect(typeof malformedFactory.destroy).toBe('function');
  expect(() => GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(malformedFactory, context)).toThrow(
    'required is not a GPUContributorOperation'
  );
  const malformedOutput = {
    ...required,
    createContributor: NO_CONTRIBUTOR,
    destroy: () => {},
    outputs: {result: {value: 'result', format: 'uint32', length: -1}}
  };
  expect(() => GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(malformedOutput, context)).toThrow(
    'required is not a GPUContributorOperation'
  );
});

it('port names are own keys: a name on Object.prototype is an ordinary port', () => {
  // `constructor` is an output port only, not also an input inherited from Object.prototype.
  const operation = new GPUContributorOperation({
    id: 'prototype',
    inputs: {values: 'values'},
    // TypeScript widens the format of a `constructor` key, hence `as const`.
    outputs: {constructor: {value: 'result', format: 'float32' as const, length: 1}},
    createContributor: getView => {
      getView('constructor');
      getView('valueOf');
      return NO_CONTRIBUTOR();
    }
  });
  const resolved: unknown[] = [];
  const context = {
    resolveVector: (value: unknown) => {
      resolved.push(value);
      return {data: [{format: 'float32'}]};
    },
    lower: () => {}
  } as unknown as Parameters<typeof GPU_CONTRIBUTOR_OPERATION_LOWERING.lower>[1];
  // `valueOf` is no port of the operation, so it is refused by name, not passed on as a function.
  expect(() => GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(operation, context)).toThrow(
    'prototype has no port "valueOf"'
  );
  expect(resolved).toEqual(['result']);
});

it('checks requested formats and declared output descriptors using logical ports', () => {
  const run = (
    createContributor: ConstructorParameters<
      typeof GPUContributorOperation
    >[0]['createContributor'],
    resolvedFormat: GPUGraphVectorFormat = 'uint32'
  ) => {
    const operation = new GPUContributorOperation({
      id: 'checked',
      inputs: {mask: 'graph-mask'},
      outputs: {result: {value: 'graph-result', format: 'float32', length: 1}},
      createContributor
    });
    const context = {
      resolveVector: (value: string) => {
        expect(['graph-mask', 'graph-result']).toContain(value);
        return {data: [{format: resolvedFormat}]};
      },
      lower: () => {}
    } as unknown as Parameters<typeof GPU_CONTRIBUTOR_OPERATION_LOWERING.lower>[1];
    GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(operation, context);
  };
  expect(() =>
    run(getView => {
      getView('mask', 'float32');
      return NO_CONTRIBUTOR();
    })
  ).toThrow('checked port "mask" (graph value "graph-mask") must have format float32, not uint32');
  run(getView => {
    const view = getView('mask', 'uint32');
    expectTypeOf(view.format).toEqualTypeOf<'uint32'>();
    expect(view.format).toBe('uint32');
    const either = getView('mask', ['float32', 'uint32'] as const);
    expectTypeOf(either.format).toEqualTypeOf<'float32' | 'uint32'>();
    expect(either.format).toBe('uint32');
    return NO_CONTRIBUTOR();
  });
  expect(() =>
    run(getView => {
      getView('mask', ['float32', 'sint32'] as const);
      return NO_CONTRIBUTOR();
    })
  ).toThrow('must have format float32 or sint32, not uint32');
  expect(() =>
    run(getView => {
      getView('result'); // Even the legacy signature checks declared outputs.
      return NO_CONTRIBUTOR();
    })
  ).toThrow(
    'checked port "result" (graph value "graph-result") must have format float32, not uint32'
  );
  run(getView => {
    expect(getView('result', 'float32').format).toBe('float32');
    return NO_CONTRIBUTOR();
  }, 'float32');
});

it('CPU operation row helpers return packed rows and name the port when the input is wrong', () => {
  const single = new Float32Array([1, 2, 3, 4]);
  const inputs = {
    positions: makeRows('float32x2', [single]),
    chunked: makeRows('uint32', [new Uint32Array([1, 2]), new Uint32Array([3])]),
    offsets: makeRows('sint32', [new Int32Array([-1, 5])]),
    chunkedPositions: makeRows('float32x2', [
      new Float32Array([1, 2]),
      new Float32Array([3, 4, 5, 6])
    ]),
    scale: 2
  };
  // One chunk is returned as is, without a second copy.
  expect(getCPUOperationFloat32Rows('twin', inputs, 'positions', 'float32x2')).toBe(single);
  expect(Array.from(getCPUOperationUint32Rows('twin', inputs, 'chunked'))).toEqual([1, 2, 3]);
  // Several chunks of a multi-component format are packed row after row.
  expect(inputs.chunkedPositions.length).toBe(3);
  expect(
    Array.from(getCPUOperationFloat32Rows('twin', inputs, 'chunkedPositions', 'float32x2'))
  ).toEqual([1, 2, 3, 4, 5, 6]);
  expect(Array.from(getCPUOperationInt32Rows('twin', inputs, 'offsets'))).toEqual([-1, 5]);
  expect(() => getCPUOperationFloat32Rows('twin', inputs, 'positions')).toThrow(
    'twin input "positions" must be float32, not float32x2'
  );
  expect(() => getCPUOperationFloat32Rows('twin', inputs, 'scale')).toThrow(
    'twin input "scale" must be a float32 vector'
  );
  expect(() => getCPUOperationUint32Rows('twin', inputs, 'missing')).toThrow(
    'twin input "missing" must be a uint32 vector'
  );
});

it('CPU row readers verify actual element classes on single and packed paths', () => {
  const cases = [
    {
      format: 'float32',
      read: getCPUOperationFloat32Rows,
      expected: Float32Array,
      wrong: Uint32Array
    },
    {format: 'uint32', read: getCPUOperationUint32Rows, expected: Uint32Array, wrong: Float32Array},
    {format: 'sint32', read: getCPUOperationInt32Rows, expected: Int32Array, wrong: Float32Array}
  ] as const;
  for (const {format, read, expected, wrong} of cases) {
    const valid = expected.of(1);
    expect(read('rows', {values: makeRows(format, [valid])}, 'values')).toBe(valid);
    const invalid = wrong.of(1);
    for (const chunks of [[invalid], [valid, valid]]) {
      const input = {...makeRows(format, chunks), pack: () => invalid};
      expect(() => read('rows', {values: input}, 'values')).toThrow(
        `rows input "values" must contain ${expected.name}, not ${wrong.name}`
      );
    }
  }
});

it('CPU row readers accept foreign-realm typed arrays without copying single chunks', () => {
  const cases = [
    {
      format: 'float32x2',
      read: (inputs: CPUOperationInputs) =>
        getCPUOperationFloat32Rows('foreign', inputs, 'values', 'float32x2'),
      elementClass: Float32Array
    },
    {
      format: 'uint32x2',
      read: (inputs: CPUOperationInputs) =>
        getCPUOperationUint32Rows('foreign', inputs, 'values', 'uint32x2'),
      elementClass: Uint32Array
    },
    {
      format: 'sint32x2',
      read: (inputs: CPUOperationInputs) =>
        getCPUOperationInt32Rows('foreign', inputs, 'values', 'sint32x2'),
      elementClass: Int32Array
    }
  ] as const;
  for (const {format, read, elementClass} of cases) {
    const foreign = runInNewContext(`new ${elementClass.name}([1, 2])`) as
      | Float32Array
      | Uint32Array
      | Int32Array;
    expect(foreign instanceof elementClass).toBe(false);
    const single = makeRows(format, [foreign]);
    // One row has two components; preserving the foreign array also proves zero-copy success.
    expect(single.length).toBe(1);
    const rows = read({values: single});
    expect(rows).toBe(foreign);
    expect(Array.from(rows)).toEqual([1, 2]);
    const foreignPacked = runInNewContext(`new ${elementClass.name}([1, 2, 3, 4])`) as
      | Float32Array
      | Uint32Array
      | Int32Array;
    const packed = {...makeRows(format, [foreign, foreign]), pack: () => foreignPacked};
    expect(read({values: packed})).toBe(foreignPacked);
    expect(Array.from(read({values: packed}))).toEqual([1, 2, 3, 4]);
  }
});

it('CPU row readers inspect the intrinsic element brand rather than spoofable tags', () => {
  const disguised = Uint32Array.of(1);
  Object.defineProperty(disguised, Symbol.toStringTag, {value: 'Float32Array'});
  expect(Object.prototype.toString.call(disguised)).toBe('[object Float32Array]');
  expect(() =>
    getCPUOperationFloat32Rows(
      'spoof',
      {
        values: makeRows('float32', [disguised])
      },
      'values'
    )
  ).toThrow('spoof input "values" must contain Float32Array, not Uint32Array');
});
