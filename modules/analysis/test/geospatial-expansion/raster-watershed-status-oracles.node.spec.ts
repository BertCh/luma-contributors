// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterWatershedSegmentationOperation,
  type RasterWatershedSegmentationOperationProps
} from '../../src/raster-analysis/raster-watershed/gpu-raster-watershed-segmentation-operation';
import {GPURasterWatershedSegmentation} from '../../src/raster-analysis/gpu-raster-watershed-segmentation';
import {
  createKernelAuditContext,
  recordShaderModules
} from '../analysis-core/kernel-audit/kernel-audit-context';
import {
  readRasterOperationBand,
  resolveRasterOperationBand
} from '../../src/raster-analysis/raster-grid/raster-operation-band';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {
  computeRasterWatershedSegmentationOnCPU,
  CPURasterWatershedSegmentationOperation
} from '../../src/raster-analysis/raster-watershed/cpu-raster-watershed-segmentation-operation';

function wrapRows<Values extends Float32Array | Uint32Array>(values: Values) {
  return {
    format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  };
}

for (const Operation of [
  CPURasterWatershedSegmentationOperation,
  GPURasterWatershedSegmentationOperation
]) {
  it.each([
    'data-status-collision',
    'status-status-collision',
    'empty-limit',
    'empty-labels',
    'empty-converged',
    'input-output-collision'
  ] as const)(`R9 ${Operation.name} rejects invalid logical output names (%s)`, problem => {
    const props: RasterWatershedSegmentationOperationProps = {
      id: 'R9-watershed',
      grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      input: {id: 'surface', format: 'float32', storage: {kind: 'buffer', values: 'surface'}},
      markers: 'markers',
      output: {
        labels:
          problem === 'empty-labels'
            ? ''
            : problem === 'input-output-collision'
              ? 'surface'
              : 'data',
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
                : 'limit'
        }
      }
    };
    expect(() => new Operation(props)).toThrow(/R9-watershed.*(name|port|graph value)/);
  });
}

for (const method of ['fifo', 'minimum-spanning-forest'] as const) {
  it.each([
    'absent',
    'wrong-format',
    'aliased',
    'valid'
  ] as const)(`R8 direct ${method} watershed validates inherited convergence (%s)`, problem => {
    const context = createKernelAuditContext('R8-watershed');
    try {
      const labels = context.view('uint32', 2);
      const converged =
        problem === 'aliased'
          ? labels
          : context.view(problem === 'wrong-format' ? 'float32' : 'uint32', 1);
      const status = problem === 'absent' ? {} : Object.create({converged});
      const construct = () =>
        new GPURasterWatershedSegmentation({
          id: 'R8-watershed',
          method,
          grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
          input: {
            id: 'surface',
            format: 'float32',
            storage: {kind: 'buffer', values: context.view('float32', 2)}
          },
          markers: context.view('uint32', 2),
          output: {labels, status: status as never}
        });
      if (problem === 'valid') {
        const contributor = construct();
        expect(contributor.props.output.status.converged).toBe(converged);
        expect(contributor.getCommandNodes(context.graph).length).toBeGreaterThan(0);
      } else {
        expect(construct).toThrow(
          problem === 'absent'
            ? /R8-watershed.*output.status.converged/
            : problem === 'wrong-format'
              ? /R8-watershed.*converged.*uint32/
              : /R8-watershed.*(overlap|distinct)/
        );
      }
    } finally {
      context.device.destroy();
    }
  });
}

it('R7 watershed operations retain grid dimensions and affine after nested edits', async () => {
  const context = createKernelAuditContext('R7-watershed-operation');
  const affine: [number, number, number, number, number, number] = [1, 0, 0, 0, 1, 0];
  const grid = {width: 3, height: 1, affine, pixelInterpretation: 'area' as const};
  const props: RasterWatershedSegmentationOperationProps = {
    grid,
    input: {id: 'surface', format: 'float32', storage: {kind: 'buffer', values: 'surface'}},
    markers: 'markers',
    output: {labels: 'labels', status: {converged: 'done'}}
  };
  const cpu = new CPURasterWatershedSegmentationOperation(props);
  const gpu = new GPURasterWatershedSegmentationOperation(props);
  grid.width = 2;
  affine[0] = 9;
  try {
    // A three-cell flat component containing seed 7 has labels [7,7,7].
    const result = await cpu.execute({
      inputs: {
        values: wrapRows(Float32Array.of(0, 0, 0)),
        markers: wrapRows(Uint32Array.of(7, 0, 0))
      }
    });
    expect(result['labels']).toEqual(Uint32Array.of(7, 7, 7));
    const contributor = gpu.createContributor(
      port =>
        context.view(
          port === 'values' ? 'float32' : 'uint32',
          port === 'converged' ? 1 : 3
        ) as never
    ) as GPURasterWatershedSegmentation;
    expect(contributor.props.grid.width).toBe(3);
    expect(contributor.props.grid.affine).toEqual([1, 0, 0, 0, 1, 0]);
  } finally {
    gpu.destroy();
    context.device.destroy();
  }
});

it('R7 direct watershed snapshots inherited band metadata and output descriptors', () => {
  const context = createKernelAuditContext('R7-direct-watershed');
  try {
    const metadata = {noDataValue: -9};
    const values = context.view('float32', 3);
    const labels = context.view('uint32', 3);
    const converged = context.view('uint32', 1);
    const affine: [number, number, number, number, number, number] = [1, 0, 0, 0, 1, 0];
    const grid = {width: 3, height: 1, affine, pixelInterpretation: 'area' as const};
    const input = Object.assign(Object.create(metadata), {
      id: 'surface',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values}
    });
    const output = {labels, status: {converged}};
    const contributor = new GPURasterWatershedSegmentation({
      grid,
      input,
      markers: context.view('uint32', 3),
      maximumSteps: 1,
      output
    });
    metadata.noDataValue = -8;
    input.storage.values = context.view('float32', 3);
    output.labels = context.view('uint32', 3);
    output.status.converged = context.view('uint32', 1);
    const sources: string[] = [];
    recordShaderModules([...contributor.getCommandNodes(context.graph)], module =>
      sources.push(module.source)
    );
    // The raw middle cell -9 must remain a barrier in generated code after mutation.
    expect.soft(sources.join('\n')).toContain('rawSample != -9.0');
    expect.soft(sources.join('\n')).not.toContain('rawSample != -8.0');
    grid.width = 2;
    affine[0] = 9;
    expect.soft(contributor.props.grid.width).toBe(3);
    expect.soft(contributor.props.grid.affine).toEqual([1, 0, 0, 0, 1, 0]);
    const inputStorage = contributor.props.input.storage;
    if (inputStorage.kind !== 'buffer') throw new Error('expected captured buffer input');
    expect.soft(inputStorage.values).toBe(values);
    expect.soft(contributor.props.output.labels).toBe(labels);
    expect.soft(contributor.props.output.status.converged).toBe(converged);
  } finally {
    context.device.destroy();
  }
});

it('R4 one flat-row forest round resolves all parents with linear indexed traversal', () => {
  const width = 1024;
  const markers = new Uint32Array(width);
  markers[0] = 7;
  let indexedReads = 0;
  const originalFrom = Uint32Array.from.bind(Uint32Array);
  // Count actual indexed reads in the solver's typed-array state. Bound native methods to
  // their target so the instrumentation preserves typed-array iteration and subarray behavior.
  const fromSpy = vi
    .spyOn(Uint32Array, 'from')
    .mockImplementation((values, mapFunction?, thisArgument?) => {
      const array = originalFrom(values, mapFunction, thisArgument);
      return new Proxy(array, {
        get(target, property) {
          if (typeof property === 'string' && /^\d+$/.test(property)) indexedReads++;
          const value = Reflect.get(target, property, target);
          return typeof value === 'function' ? value.bind(target) : value;
        }
      });
    });
  try {
    const result = computeRasterWatershedSegmentationOnCPU(
      {id: 'flat', format: 'float32', values: new Float32Array(width)},
      markers,
      {
        grid: {width, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        method: 'minimum-spanning-forest',
        maximumIterations: 1
      }
    );
    // Every equal-cost component chooses its left edge; all roots are seed 0 in one round.
    // Without compression the root walks alone take n(n-1)/2 hops. Compression traverses
    // each original parent edge once plus constant reads per cell/edge; 24*n also budgets
    // the edge selection, final convergence scan and output publication, independent of time.
    expect(result.labels.every(label => label === 7)).toBe(true);
    expect(Array.from(result.converged)).toEqual([1]);
    expect(Array.from(result.iterationLimitReached)).toEqual([0]);
    expect(indexedReads).toBeLessThan(24 * width);
  } finally {
    fromSpy.mockRestore();
  }
});

it('R1 watershed operations retain inherited raw nodata as a barrier', async () => {
  const input = Object.assign(Object.create({noDataValue: -9}), {
    id: 'surface',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: 'surface'}
  });
  const props: RasterWatershedSegmentationOperationProps = {
    grid: {width: 3, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
    input,
    markers: 'markers',
    maximumSteps: 1,
    output: {labels: 'labels', status: {converged: 'done', iterationLimitReached: 'limit'}}
  };
  // Seed 0 cannot cross nodata cell 1: its sole queue pop leaves no pending cell.
  const direct = computeRasterWatershedSegmentationOnCPU(
    Object.assign(Object.create({noDataValue: -9}), {
      id: 'surface',
      format: 'float32' as const,
      values: Float32Array.of(0, -9, 0)
    }),
    Uint32Array.of(7, 0, 0),
    props
  );
  expect(Array.from(direct.labels)).toEqual([7, 0, 0]);
  const cpuOperation = new CPURasterWatershedSegmentationOperation(props);
  const result = await cpuOperation.execute({
    inputs: {
      values: wrapRows(Float32Array.of(0, -9, 0)),
      markers: wrapRows(Uint32Array.of(7, 0, 0))
    }
  });
  expect.soft(Array.from(result['labels'] as Uint32Array)).toEqual([7, 0, 0]);
  expect
    .soft([(result['done'] as Uint32Array)[0], (result['limit'] as Uint32Array)[0]])
    .toEqual([1, 0]);
  const gpuOperation = new GPURasterWatershedSegmentationOperation(props);
  const context = createKernelAuditContext('R1-watershed');
  try {
    expect(
      readRasterOperationBand('R1-reader', {values: wrapRows(Float32Array.of(0, -9, 0))}, input)
        .noDataValue
    ).toBe(-9);
    expect(
      resolveRasterOperationBand(input, () => context.view('float32', 3) as never).noDataValue
    ).toBe(-9);
    const views = Object.fromEntries(
      Object.entries({
        values: {format: 'float32' as const, length: 3},
        markers: {format: 'uint32' as const, length: 3},
        ...gpuOperation.outputs
      }).map(([name, descriptor]) => [name, context.view(descriptor.format, descriptor.length)])
    );
    const contributor = gpuOperation.createContributor(
      name => views[name] as never
    ) as GPURasterWatershedSegmentation;
    expect(contributor.props.input.noDataValue).toBe(-9);
  } finally {
    gpuOperation.destroy();
    context.device.destroy();
  }
});

it('R2 watershed returns the originally declared nested output names after mutation', async () => {
  const status = Object.create({converged: 'done', iterationLimitReached: 'limit'});
  const output = {labels: 'labels', status};
  const operation = new CPURasterWatershedSegmentationOperation({
    grid: {width: 1, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
    input: {id: 'surface', format: 'float32', storage: {kind: 'buffer', values: 'surface'}},
    markers: 'markers',
    output
  });
  status.converged = 'renamed';
  status.iterationLimitReached = 'renamed-limit';
  output.labels = 'renamed-labels';
  const result = await operation.execute({
    inputs: {values: wrapRows(Float32Array.of(0)), markers: wrapRows(Uint32Array.of(7))}
  });
  expect(Object.keys(result).sort()).toEqual(['done', 'labels', 'limit']);
  expect(result['done']).toEqual(Uint32Array.of(1));
  expect(result['limit']).toEqual(Uint32Array.of(0));
});

it('R3 direct watershed keeps one forest contraction after caller budget mutation', () => {
  const context = createKernelAuditContext('R3-watershed');
  try {
    const props = {
      id: 'direct-watershed',
      grid: {
        width: 3,
        height: 1,
        affine: [1, 0, 0, 0, 1, 0] as const,
        pixelInterpretation: 'area' as const
      },
      input: {
        id: 'surface',
        format: 'float32' as const,
        storage: {kind: 'buffer' as const, values: context.view('float32', 3)}
      },
      markers: context.view('uint32', 3),
      method: 'minimum-spanning-forest' as const,
      maximumIterations: 1,
      output: {labels: context.view('uint32', 3), status: {converged: context.view('uint32', 1)}}
    };
    const contributor = new GPURasterWatershedSegmentation(props);
    props.maximumIterations = 8;
    const nodes = contributor.getCommandNodes(context.graph);
    expect(
      nodes.filter(node => /^direct-watershed-forest-round-\d+-cost$/.test(node.id)).length
    ).toBe(1);
  } finally {
    context.device.destroy();
  }
});

for (const method of ['fifo', 'minimum-spanning-forest'] as const) {
  it.each([
    'own',
    'inherited'
  ] as const)(`watershed operation forms capture their ${method} budget at construction (%s)`, async representation => {
    const props: RasterWatershedSegmentationOperationProps = {
      grid: {width: 3, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      method,
      ...(method === 'fifo' ? {maximumSteps: 1} : {maximumIterations: 1}),
      input: {id: 'surface', format: 'float32', storage: {kind: 'buffer', values: 'surface'}},
      markers: 'markers',
      output: {labels: 'labels', status: {converged: 'done', iterationLimitReached: 'limit'}}
    };
    const suppliedProps =
      representation === 'inherited'
        ? (Object.create(props) as RasterWatershedSegmentationOperationProps)
        : props;
    const cpu = new CPURasterWatershedSegmentationOperation(suppliedProps);
    const gpu = new GPURasterWatershedSegmentationOperation(suppliedProps);
    if (method === 'fifo') props.maximumSteps = 3;
    else props.maximumIterations = 2;
    const result = await cpu.execute({
      inputs: {
        values: wrapRows(Float32Array.of(5, 0, 0)),
        markers: wrapRows(Uint32Array.of(7, 0, 0))
      }
    });
    expect.soft(Array.from(result['done'] as Uint32Array)).toEqual([0]);
    expect.soft(Array.from(result['limit'] as Uint32Array)).toEqual([1]);
    expect.soft(Array.from(result['labels'] as Uint32Array)).toEqual([7, 0xfffffffe, 0xfffffffe]);
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const views = Object.fromEntries(
        Object.entries({
          values: {format: 'float32' as const, length: 3},
          markers: {format: 'uint32' as const, length: 3},
          ...gpu.outputs
        }).map(([name, descriptor]) => [
          name,
          createTransientView(graph, name, descriptor.format, descriptor.length)
        ])
      );
      const contributor = gpu.createContributor(
        name => views[name] as never
      ) as GPURasterWatershedSegmentation;
      expect(
        method === 'fifo' ? contributor.props.maximumSteps : contributor.props.maximumIterations
      ).toBe(1);
    } finally {
      gpu.destroy();
      device.destroy();
    }
  });
}

for (const method of ['fifo', 'minimum-spanning-forest'] as const) {
  for (const complete of [false, true]) {
    it(`watershed CPU operation publishes binding ${method} budget words (complete=${complete})`, async () => {
      const operation = new CPURasterWatershedSegmentationOperation({
        grid: {width: 3, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        input: {id: 'surface', format: 'float32', storage: {kind: 'buffer', values: 'surface'}},
        markers: 'markers',
        method,
        ...(method === 'fifo'
          ? {maximumSteps: complete ? 3 : 1}
          : {maximumIterations: complete ? 2 : 1}),
        output: {labels: 'labels', status: {converged: 'done', iterationLimitReached: 'limit'}}
      });
      const result = await operation.execute({
        inputs: {
          values: wrapRows(Float32Array.of(5, 0, 0)),
          markers: wrapRows(Uint32Array.of(7, 0, 0))
        }
      });
      // FIFO needs three pops; rooted forest first joins the two low cells, then the seed.
      expect(Array.from(result['labels'] as Uint32Array)).toEqual(
        complete ? [7, 7, 7] : [7, 0xfffffffe, 0xfffffffe]
      );
      expect(Array.from(result['done'] as Uint32Array)).toEqual(complete ? [1] : [0]);
      expect(Array.from(result['limit'] as Uint32Array)).toEqual(complete ? [0] : [1]);
    });
  }
}

for (const maximumIterations of [1, 2]) {
  it(`watershed forest obeys ${maximumIterations} component-contraction rounds`, () => {
    // Edge (1,2) has level 0 and edge (0,1) level 5. In round one both unseeded
    // cells choose (1,2); only round two can join their component to seed 0.
    const result = computeRasterWatershedSegmentationOnCPU(
      {id: 'surface', format: 'float32', values: Float32Array.of(5, 0, 0)},
      Uint32Array.of(7, 0, 0),
      {
        grid: {width: 3, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        method: 'minimum-spanning-forest',
        maximumIterations
      }
    );
    expect.soft(Array.from(result.converged)).toEqual(maximumIterations === 1 ? [0] : [1]);
    expect
      .soft(Array.from(result.iterationLimitReached))
      .toEqual(maximumIterations === 1 ? [1] : [0]);
    expect
      .soft(Array.from(result.labels))
      .toEqual(maximumIterations === 1 ? [7, 0xfffffffe, 0xfffffffe] : [7, 7, 7]);
  });
}

for (const connected of [true, false]) {
  it(`watershed CPU twin honors the one-pop FIFO stopping oracle (connected=${connected})`, () => {
    // By hand: popping seed 0 enqueues cell 1 when it is valid. Cell 2 still awaits discovery.
    // With cells 1..4 masked, pop 0 exhausts the queue and the same budget is sufficient.
    const result = computeRasterWatershedSegmentationOnCPU(
      {
        id: 'surface',
        format: 'float32',
        values: Float32Array.of(0, 0, 0, 0, 0),
        validity: Uint32Array.from(connected ? [1, 1, 1, 1, 1] : [1, 0, 0, 0, 0])
      },
      Uint32Array.of(7, 0, 0, 0, 0),
      {
        grid: {
          width: 5,
          height: 1,
          affine: [1, 0, 0, 0, 1, 0],
          pixelInterpretation: 'area'
        },
        maximumSteps: 1
      }
    );
    // Check stopping words first so the regression reports the caller-visible false completion.
    expect.soft(Array.from(result.converged)).toEqual(connected ? [0] : [1]);
    expect.soft(Array.from(result.iterationLimitReached)).toEqual(connected ? [1] : [0]);
    expect
      .soft(Array.from(result.labels))
      .toEqual(connected ? [7, 0xfffffffe, 0xfffffffe, 0xfffffffe, 0xfffffffe] : [7, 0, 0, 0, 0]);
  });
}

for (const Operation of [
  CPURasterWatershedSegmentationOperation,
  GPURasterWatershedSegmentationOperation
]) {
  it.each([
    'values',
    'validity',
    'markers',
    'omitted-validity',
    'repeated-inputs'
  ] as const)(`FRM-01 ${Operation.name} validates selected input names (%s)`, selection => {
    const props: RasterWatershedSegmentationOperationProps = {
      id: 'FRM-01-watershed',
      grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      input: {id: 'surface', format: 'uint32', storage: {kind: 'buffer', values: 'surface'}},
      markers: 'markers',
      output: {labels: 'labels', status: {converged: 'done'}}
    };
    if (selection === 'values') props.input.storage.values = '';
    if (selection === 'validity') props.input.validity = '';
    if (selection === 'markers') props.markers = '';
    if (selection === 'repeated-inputs') props.markers = props.input.storage.values;
    if (selection === 'omitted-validity' || selection === 'repeated-inputs')
      expect(() => new Operation(props)).not.toThrow();
    else expect(() => new Operation(props)).toThrow(/FRM-01-watershed.*non-empty graph value/);
  });
}

it('FRM-01 CPU FIFO watershed name admission stays independent of GPU cell capacity', () => {
  expect(
    () =>
      new CPURasterWatershedSegmentationOperation({
        grid: {width: 100000, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        method: 'fifo',
        input: {id: 'surface', format: 'uint32', storage: {kind: 'buffer', values: 'samples'}},
        markers: 'samples',
        output: {labels: 'labels', status: {converged: 'done'}}
      })
  ).not.toThrow();
});
