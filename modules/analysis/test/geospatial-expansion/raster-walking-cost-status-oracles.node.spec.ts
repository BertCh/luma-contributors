// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterWalkingCostOperation,
  type RasterWalkingCostOperationProps
} from '../../src/raster-analysis/cost-distance/gpu-raster-walking-cost-operation';
import {GPURasterWalkingCost} from '../../src/raster-analysis/gpu-raster-walking-cost';
import {createKernelAuditContext} from '../analysis-core/kernel-audit/kernel-audit-context';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {
  computeRasterWalkingCostOnCPU,
  CPURasterWalkingCostOperation
} from '../../src/raster-analysis/cost-distance/cpu-raster-walking-cost-operation';
import {getGPURasterWalkingCostParameterValues} from '../../src/raster-analysis/gpu-raster-walking-cost';

for (const Operation of [CPURasterWalkingCostOperation, GPURasterWalkingCostOperation]) {
  it.each([
    'data-status-collision',
    'data-data-collision',
    'status-status-collision',
    'empty-limit',
    'empty-count',
    'empty-invalid',
    'empty-costs',
    'empty-validity',
    'empty-converged',
    'input-output-collision'
  ] as const)(`R9 ${Operation.name} rejects invalid logical output names (%s)`, problem => {
    const props: RasterWalkingCostOperationProps = {
      id: 'R9-walking',
      grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: 'elevation'}
      },
      friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: 'friction'}},
      sources: 'sources',
      parameters: 'parameters',
      output: {
        costs:
          problem === 'empty-costs'
            ? ''
            : problem === 'input-output-collision'
              ? 'elevation'
              : 'data',
        validity:
          problem === 'empty-validity'
            ? ''
            : problem === 'data-data-collision'
              ? 'data'
              : 'validity',
        status: {
          converged:
            problem === 'data-status-collision'
              ? 'data'
              : problem === 'empty-converged'
                ? ''
                : 'done',
          iterationLimitReached:
            problem === 'status-status-collision'
              ? 'done'
              : problem === 'empty-limit'
                ? ''
                : 'limit',
          iterationCount: problem === 'empty-count' ? '' : undefined,
          invalidCount: problem === 'empty-invalid' ? '' : undefined
        }
      }
    };
    expect(() => new Operation(props)).toThrow(/R9-walking.*(name|port|graph value)/);
  });
}

it.each([
  'absent',
  'wrong-format',
  'aliased',
  'valid'
] as const)('R8 direct walking validates inherited convergence (%s)', problem => {
  const context = createKernelAuditContext('R8-walking');
  try {
    const costs = context.view('float32', 2);
    const validity = context.view('uint32', 2);
    // Alias validity so the view has the correct uint32 format but still overwrites raster data.
    const converged =
      problem === 'aliased'
        ? validity
        : context.view(problem === 'wrong-format' ? 'float32' : 'uint32', 1);
    const status = problem === 'absent' ? {} : Object.create({converged});
    const band = (id: string) => ({
      id,
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: context.view('float32', 2)}
    });
    const construct = () =>
      new GPURasterWalkingCost({
        id: 'R8-walking',
        grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        elevation: band('elevation'),
        friction: band('friction'),
        sources: context.view('uint32', 2),
        parameters: context.view('float32', 16),
        maximumIterations: 1,
        output: {costs, validity, status: status as never}
      });
    if (problem === 'valid') {
      const contributor = construct();
      expect(contributor.props.output.status.converged).toBe(converged);
      expect(contributor.getCommandNodes(context.graph).length).toBeGreaterThan(0);
    } else {
      expect(construct).toThrow(
        problem === 'absent'
          ? /R8-walking.*output.status.converged/
          : problem === 'wrong-format'
            ? /R8-walking.*converged.*uint32/
            : /R8-walking.*(overlap|distinct)/
      );
    }
  } finally {
    context.device.destroy();
  }
});

it('R7 walking forms retain the two-cell GRASS model and grid after nested edits', async () => {
  const context = createKernelAuditContext('R7-walking');
  const affine: [number, number, number, number, number, number] = [1, 0, 0, 0, 1, 0];
  const grid = {width: 2, height: 1, affine, pixelInterpretation: 'area' as const};
  const model = {
    kind: 'grass' as 'grass' | 'tobler',
    coefficients: [1, 1, -1, -1] as [number, number, number, number],
    downhillSlopeThreshold: -0.5
  };
  const props: RasterWalkingCostOperationProps = {
    grid,
    model,
    elevation: {id: 'elevation', format: 'float32', storage: {kind: 'buffer', values: 'elevation'}},
    friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: 'friction'}},
    sources: 'sources',
    parameters: 'parameters',
    output: {
      costs: 'costs',
      validity: 'validity',
      status: {converged: 'done', invalidCount: 'invalid'}
    }
  };
  const parameters = getGPURasterWalkingCostParameterValues({model}, grid);
  const cpu = new CPURasterWalkingCostOperation(props);
  const gpu = new GPURasterWalkingCostOperation(props);
  const elevation = {
    id: 'elevation',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: context.view('float32', 2)}
  };
  const friction = {
    id: 'friction',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: context.view('float32', 2)}
  };
  const output = {
    costs: context.view('float32', 2),
    validity: context.view('uint32', 2),
    status: {converged: context.view('uint32', 1)}
  };
  const originalCosts = output.costs;
  const originalElevation = elevation.storage.values;
  const direct = new GPURasterWalkingCost({
    grid,
    model,
    elevation,
    friction,
    sources: context.view('uint32', 2),
    parameters: context.view('float32', 16),
    maximumIterations: 1,
    output
  });
  model.kind = 'tobler';
  model.coefficients[0] = 9;
  model.downhillSlopeThreshold = -2;
  affine[0] = 9;
  elevation.storage.values = context.view('float32', 2);
  output.costs = context.view('float32', 2);
  const wrap = <Values extends Float32Array | Uint32Array>(values: Values) => ({
    format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  try {
    // One flat GRASS unit edge has cost 1; the original parameter layout remains valid.
    const result = await cpu.execute({
      inputs: {
        elevationValues: wrap(Float32Array.of(0, 0)),
        frictionValues: wrap(Float32Array.of(0, 0)),
        sources: wrap(Uint32Array.of(1, 0)),
        parameters: wrap(parameters)
      }
    });
    expect.soft(result['costs']).toEqual(Float32Array.of(0, 1));
    expect.soft(result['invalid']).toEqual(Uint32Array.of(0));
    grid.width = 1;
    const lowered = gpu.createContributor(
      port =>
        context.view(
          port === 'sources' ||
            port === 'validity' ||
            port === 'converged' ||
            port === 'invalidCount'
            ? 'uint32'
            : 'float32',
          port === 'parameters' ? 16 : port === 'converged' || port === 'invalidCount' ? 1 : 2
        ) as never
    ) as GPURasterWalkingCost;
    for (const contributor of [direct, lowered]) {
      expect.soft(contributor.props.grid.width).toBe(2);
      expect.soft(contributor.props.grid.affine).toEqual([1, 0, 0, 0, 1, 0]);
      expect
        .soft(contributor.props.model)
        .toEqual({kind: 'grass', coefficients: [1, 1, -1, -1], downhillSlopeThreshold: -0.5});
    }
    const elevationStorage = direct.props.elevation.storage;
    if (elevationStorage.kind !== 'buffer') throw new Error('expected captured buffer elevation');
    expect.soft(elevationStorage.values).toBe(originalElevation);
    expect.soft(direct.props.output.costs).toBe(originalCosts);
  } finally {
    gpu.destroy();
    context.device.destroy();
  }
});

it('R2 walking declares inherited stopping ports and retains original nested names', async () => {
  const status = Object.create({converged: 'done', iterationLimitReached: 'limit'});
  const output = {costs: 'costs', validity: 'validity', status};
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const props = {
    grid,
    elevation: {
      id: 'elevation',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: 'elevation'}
    },
    friction: {
      id: 'friction',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: 'friction'}
    },
    sources: 'sources',
    parameters: 'parameters',
    output
  };
  const cpuOperation = new CPURasterWalkingCostOperation(props);
  const gpuOperation = new GPURasterWalkingCostOperation(props);
  expect
    .soft(Object.keys(cpuOperation.outputs).sort())
    .toEqual(['costs', 'done', 'limit', 'validity']);
  expect
    .soft(Object.keys(gpuOperation.outputs).sort())
    .toEqual(['converged', 'costs', 'iterationLimitReached', 'validity']);
  status.converged = 'renamed';
  status.iterationLimitReached = 'renamed-limit';
  output.costs = 'renamed-costs';
  const wrap = <Values extends Float32Array | Uint32Array>(values: Values) => ({
    format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const result = await cpuOperation.execute({
    inputs: {
      elevationValues: wrap(Float32Array.of(0)),
      frictionValues: wrap(Float32Array.of(0)),
      sources: wrap(Uint32Array.of(1)),
      parameters: wrap(getGPURasterWalkingCostParameterValues({}, grid))
    }
  });
  expect.soft(Object.keys(result).sort()).toEqual(['costs', 'done', 'limit', 'validity']);
  expect.soft(result['done']).toEqual(Uint32Array.of(1));
  expect.soft(result['limit']).toEqual(Uint32Array.of(0));
  expect.soft(gpuOperation.outputs['converged']?.value).toBe('done');
  gpuOperation.destroy();
});

it('R3 direct walking keeps two relaxation nodes after caller budget mutation', () => {
  const context = createKernelAuditContext('R3-walking');
  try {
    const grid = {
      width: 3,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const band = (id: string) => ({
      id,
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: context.view('float32', 3)}
    });
    const props = {
      id: 'direct-walking',
      grid,
      elevation: band('elevation'),
      friction: band('friction'),
      sources: context.view('uint32', 3),
      parameters: context.view('float32', 16),
      maximumIterations: 1,
      output: {
        costs: context.view('float32', 3),
        validity: context.view('uint32', 3),
        status: {converged: context.view('uint32', 1)}
      }
    };
    const contributor = new GPURasterWalkingCost(props);
    props.maximumIterations = 8;
    const nodes = contributor.getCommandNodes(context.graph);
    expect(nodes.filter(node => /^direct-walking-iterate-round-\d+$/.test(node.id)).length).toBe(2);
  } finally {
    context.device.destroy();
  }
});

it.each([
  'own',
  'inherited'
] as const)('walking operation forms capture the explicit iteration budget at construction (%s)', async representation => {
  const grid = {
    width: 6,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const model = {
    kind: 'grass' as const,
    coefficients: [1, 1, -1, -1] as const,
    downhillSlopeThreshold: -0.5
  };
  const props: RasterWalkingCostOperationProps = {
    grid,
    model,
    maximumIterations: 1,
    elevation: {id: 'elevation', format: 'float32', storage: {kind: 'buffer', values: 'elevation'}},
    friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: 'friction'}},
    sources: 'sources',
    parameters: 'parameters',
    output: {
      costs: 'costs',
      validity: 'validity',
      status: {converged: 'done', iterationCount: 'count', iterationLimitReached: 'limit'}
    }
  };
  const suppliedProps =
    representation === 'inherited'
      ? (Object.create(props) as RasterWalkingCostOperationProps)
      : props;
  const cpu = new CPURasterWalkingCostOperation(suppliedProps);
  const gpu = new GPURasterWalkingCostOperation(suppliedProps);
  props.maximumIterations = 8;
  const wrap = <Values extends Float32Array | Uint32Array>(values: Values) => ({
    format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const result = await cpu.execute({
    inputs: {
      elevationValues: wrap(new Float32Array(6)),
      frictionValues: wrap(new Float32Array(6)),
      sources: wrap(Uint32Array.of(1, 0, 0, 0, 0, 0)),
      parameters: wrap(getGPURasterWalkingCostParameterValues({model}, grid))
    }
  });
  expect
    .soft(['count', 'done', 'limit'].map(name => (result[name] as Uint32Array)[0]))
    .toEqual([2, 0, 1]);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const views = Object.fromEntries(
      Object.entries({
        elevationValues: {format: 'float32' as const, length: 6},
        frictionValues: {format: 'float32' as const, length: 6},
        sources: {format: 'uint32' as const, length: 6},
        parameters: {format: 'float32' as const, length: 16},
        ...gpu.outputs
      }).map(([name, descriptor]) => [
        name,
        createTransientView(graph, name, descriptor.format, descriptor.length)
      ])
    );
    const contributor = gpu.createContributor(name => views[name] as never) as GPURasterWalkingCost;
    expect(contributor.props.maximumIterations).toBe(1);
  } finally {
    gpu.destroy();
    device.destroy();
  }
});

for (const scene of ['partial', 'complete', 'mismatch'] as const) {
  it(`walking stopping and invalid words have a hand-derived ${scene} oracle`, async () => {
    const width = scene === 'mismatch' ? 1 : 6;
    const grid = {
      width,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const model = {
      kind: 'grass' as const,
      coefficients: [1, 1, -1, -1] as const,
      downhillSlopeThreshold: -0.5
    };
    const parameters = getGPURasterWalkingCostParameterValues(
      {model: scene === 'mismatch' ? {kind: 'tobler'} : model},
      grid
    );
    const sources = Uint32Array.from({length: width}, (_, cell) => Number(cell === 0));
    const values = new Float32Array(width);
    const options = {grid, model, maximumIterations: scene === 'partial' ? 1 : 8};
    const result = computeRasterWalkingCostOnCPU(
      {
        elevation: {id: 'elevation', format: 'float32', values},
        friction: {id: 'friction', format: 'float32', values},
        sources,
        parameters
      },
      options
    );
    // Flat GRASS unit edges advance one cell per synchronous round. Budget one allows two
    // published rounds; cell 3 is still pending. Five advances plus verification finish in six.
    // A Tobler parameter block is exactly one layout mismatch and permits no transition.
    const expected =
      scene === 'partial' ? [2, 0, 1, 0] : scene === 'complete' ? [6, 1, 0, 0] : [1, 1, 0, 1];
    expect([
      result.iterationCount[0],
      result.converged[0],
      result.iterationLimitReached[0],
      result.invalidCount[0]
    ]).toEqual(expected);
    expect(Array.from(result.validity)).toEqual(new Array(width).fill(1));
    expect(Array.from(result.costs)).toEqual(
      scene === 'partial'
        ? [0, 1, 2, Infinity, Infinity, Infinity]
        : scene === 'complete'
          ? [0, 1, 2, 3, 4, 5]
          : [0]
    );
    const operation = new CPURasterWalkingCostOperation({
      ...options,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: 'elevation'}
      },
      friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: 'friction'}},
      sources: 'sources',
      parameters: 'parameters',
      output: {
        costs: 'costs',
        validity: 'validity',
        status: {
          iterationCount: 'count',
          converged: 'done',
          iterationLimitReached: 'limit',
          invalidCount: 'invalid'
        }
      }
    });
    const wrap = <Values extends Float32Array | Uint32Array>(rows: Values) => ({
      format: rows instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
      length: rows.length,
      chunks: [{values: rows, length: rows.length}],
      pack: () => rows
    });
    const outputs = await operation.execute({
      inputs: {
        elevationValues: wrap(values),
        frictionValues: wrap(values),
        sources: wrap(sources),
        parameters: wrap(parameters)
      }
    });
    expect(
      ['count', 'done', 'limit', 'invalid'].map(name => (outputs[name] as Uint32Array)[0])
    ).toEqual(expected);
  });
}

for (const Operation of [CPURasterWalkingCostOperation, GPURasterWalkingCostOperation]) {
  for (const selection of ['omitted', 'tobler', 'grass', 'false', 'misspelled'] as const) {
    it(`R7 ${Operation.name} validates supplied model before capture (${selection})`, () => {
      const props: RasterWalkingCostOperationProps = {
        id: 'R7-walking',
        grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {kind: 'buffer', values: 'elevation'}
        },
        friction: {
          id: 'friction',
          format: 'float32',
          storage: {kind: 'buffer', values: 'friction'}
        },
        sources: 'sources',
        parameters: 'parameters',
        output: {costs: 'costs', validity: 'validity', status: {converged: 'done'}}
      };
      if (selection === 'tobler') props.model = {kind: 'tobler'};
      if (selection === 'grass')
        props.model = {kind: 'grass', coefficients: [1, 1, -1, -1], downhillSlopeThreshold: -0.5};
      // JavaScript boundary: keep the well-typed fixture and inject malformed runtime input.
      if (selection === 'false') Reflect.set(props, 'model', false);
      if (selection === 'misspelled') Reflect.set(props, 'model', {kind: 'tolber'});
      if (selection === 'false' || selection === 'misspelled')
        expect(() => new Operation(props)).toThrow(
          /R7-walking.*model.kind.*(grass.*tobler|tobler.*grass)/
        );
      else {
        const operation = new Operation(props);
        if (operation instanceof GPURasterWalkingCostOperation) operation.destroy();
      }
    });
  }
}

for (const Operation of [CPURasterWalkingCostOperation, GPURasterWalkingCostOperation]) {
  it.each([
    'elevation',
    'friction',
    'elevation-validity',
    'friction-validity',
    'sources',
    'parameters',
    'omitted-validity',
    'repeated-inputs'
  ] as const)(`FRM-01 ${Operation.name} validates selected input names (%s)`, selection => {
    const props: RasterWalkingCostOperationProps = {
      id: 'FRM-01-walking',
      grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: 'elevation'}
      },
      friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: 'friction'}},
      sources: 'sources',
      parameters: 'parameters',
      output: {costs: 'costs', validity: 'validity', status: {converged: 'done'}}
    };
    if (selection === 'elevation') props.elevation.storage.values = '';
    if (selection === 'friction') props.friction.storage.values = '';
    if (selection === 'elevation-validity') props.elevation.validity = '';
    if (selection === 'friction-validity') props.friction.validity = '';
    if (selection === 'sources') props.sources = '';
    if (selection === 'parameters') props.parameters = '';
    if (selection === 'repeated-inputs')
      props.friction.storage.values = props.elevation.storage.values;
    if (selection === 'omitted-validity' || selection === 'repeated-inputs')
      expect(() => new Operation(props)).not.toThrow();
    else expect(() => new Operation(props)).toThrow(/FRM-01-walking.*non-empty graph value/);
  });
}

it('FRM-01 CPU walking name admission stays independent of GPU iteration budgets', () => {
  expect(
    () =>
      new CPURasterWalkingCostOperation({
        grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        maximumIterations: 30000,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {kind: 'buffer', values: 'samples'}
        },
        friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: 'samples'}},
        sources: 'sources',
        parameters: 'parameters',
        output: {costs: 'costs', validity: 'validity', status: {converged: 'done'}}
      })
  ).not.toThrow();
});

for (const bandName of ['elevation', 'friction'] as const) {
  it(`URT-13 walking identifies mismatched ${bandName} storage when the other band is valid`, () => {
    const grid = {
      width: 2,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const input: Parameters<typeof computeRasterWalkingCostOnCPU>[0] = {
      elevation: {id: 'height', format: 'float32', values: Float32Array.of(0, 0)},
      friction: {id: 'roughness', format: 'float32', values: Float32Array.of(0, 0)},
      sources: Uint32Array.of(1, 0),
      parameters: getGPURasterWalkingCostParameterValues({}, grid)
    };
    Reflect.set(input[bandName], 'values', Int32Array.of(0, 0));
    expect(() => computeRasterWalkingCostOnCPU(input, {grid})).toThrow(
      new RegExp(
        `^computeRasterWalkingCostOnCPU ${bandName}\\.values must be Float32Array.*Int32Array`
      )
    );
  });
}

for (const Operation of [CPURasterWalkingCostOperation, GPURasterWalkingCostOperation]) {
  it(`B1 ${Operation.name} accepts a null-prototype Tobler model`, () => {
    const model = {kind: 'tobler' as const};
    Object.setPrototypeOf(model, null);
    const props: RasterWalkingCostOperationProps = {
      id: 'review-walk',
      grid: {width: 1, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      model,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: 'elevation'}
      },
      friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: 'friction'}},
      sources: 'sources',
      parameters: 'parameters',
      output: {costs: 'costs', validity: 'validity', status: {converged: 'done'}}
    };
    const parameters = getGPURasterWalkingCostParameterValues({model}, props.grid);
    const result = computeRasterWalkingCostOnCPU(
      {
        elevation: {id: 'elevation', format: 'float32', values: Float32Array.of(0)},
        friction: {id: 'friction', format: 'float32', values: Float32Array.of(0)},
        sources: Uint32Array.of(1),
        parameters
      },
      props
    );
    expect(result.costs).toEqual(Float32Array.of(0));
    expect(() => new Operation(props)).not.toThrow();
  });

  it.each([
    'missing',
    'short',
    'nonfinite'
  ] as const)(`S3 ${Operation.name} diagnoses malformed captured GRASS coefficients (%s)`, problem => {
    const props: RasterWalkingCostOperationProps = {
      id: 'review-walk',
      grid: {width: 1, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: 'elevation'}
      },
      friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: 'friction'}},
      sources: 'sources',
      parameters: 'parameters',
      output: {costs: 'costs', validity: 'validity', status: {converged: 'done'}}
    };
    Reflect.set(props, 'model', {
      kind: 'grass',
      downhillSlopeThreshold: -0.5,
      ...(problem === 'missing'
        ? {}
        : {coefficients: problem === 'short' ? [1, 2] : [1, 2, 3, Infinity]})
    });
    expect(() => new Operation(props)).toThrow(/review-walk.*model.coefficients.*four finite/);
  });
}

for (const bandName of ['elevation', 'friction'] as const) {
  it(`S1 executing walking identifies wrong ${bandName} row storage`, () => {
    const props: RasterWalkingCostOperationProps = {
      id: 'review-walk',
      grid: {width: 1, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: 'elevation'}
      },
      friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: 'friction'}},
      sources: 'sources',
      parameters: 'parameters',
      output: {costs: 'costs', validity: 'validity', status: {converged: 'done'}}
    };
    const wrap = (format: 'float32' | 'uint32', values: Float32Array | Uint32Array) => ({
      format,
      length: values.length,
      chunks: [{values, length: values.length}],
      pack: () => values
    });
    const good = wrap('float32', Float32Array.of(0));
    const bad = wrap('float32', Uint32Array.of(0));
    const operation = new CPURasterWalkingCostOperation(props);
    expect(() =>
      operation.execute({
        inputs: {
          elevationValues: bandName === 'elevation' ? bad : good,
          frictionValues: bandName === 'friction' ? bad : good,
          sources: wrap('uint32', Uint32Array.of(1)),
          parameters: wrap('float32', getGPURasterWalkingCostParameterValues({}, props.grid))
        }
      })
    ).toThrow(
      new RegExp(
        `^review-walk input "${bandName}\\.values" must contain Float32Array, not Uint32Array`
      )
    );
  });
}

it('B1 direct walking also accepts the null-prototype Tobler model through the shared capture', () => {
  const context = createKernelAuditContext('B1-direct-walking');
  try {
    const model = {kind: 'tobler' as const};
    Object.setPrototypeOf(model, null);
    const contributor = new GPURasterWalkingCost({
      id: 'review-direct-walking',
      grid: {width: 1, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      model,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: context.view('float32', 1)}
      },
      friction: {
        id: 'friction',
        format: 'float32',
        storage: {kind: 'buffer', values: context.view('float32', 1)}
      },
      sources: context.view('uint32', 1),
      parameters: context.view('float32', 16),
      output: {
        costs: context.view('float32', 1),
        validity: context.view('uint32', 1),
        status: {converged: context.view('uint32', 1)}
      }
    });
    expect(contributor.props.model).toEqual({kind: 'tobler'});
  } finally {
    context.device.destroy();
  }
});
