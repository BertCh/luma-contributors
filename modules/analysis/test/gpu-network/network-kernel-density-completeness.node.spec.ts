// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUNetworkInhomogeneousK} from '../../src/gpu-network/gpu-network-inhomogeneous-k';
import {GPUNetworkKernelDensity} from '../../src/gpu-network/gpu-network-kernel-density';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';

it('Decision 26 rejects both legacy KDE output spellings with migration paths', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'float32' | 'uint32'>(name: string, format: Format) =>
    createTransientView(graph, name, format, 1);
  const props = {
    id: 'density-migration',
    distances: view('distances', 'float32'),
    eventCount: 1,
    quadratureLengths: view('lengths', 'float32'),
    bandwidth: 2
  };
  const output = {
    intensity: view('intensity', 'float32'),
    kernelMass: view('mass', 'float32'),
    validity: view('validity', 'uint32')
  };
  try {
    expect(
      () =>
        new GPUNetworkKernelDensity({
          ...props,
          output: {
            ...output,
            // @ts-expect-error Named status ports belong directly under output.status.
            ports: {status: {invalidCount: view('invalid-count', 'uint32')}}
          }
        })
    ).toThrow(/^density-migration .*use output\.status\.invalidCount/);
    expect(
      () =>
        new GPUNetworkKernelDensity({
          ...props,
          output: {
            ...output,
            // @ts-expect-error The per-query view belongs at output.validity.
            status: output.validity
          }
        })
    ).toThrow(/^density-migration .*use output\.validity/);
  } finally {
    device.destroy();
  }
});

it('GPUNetworkKernelDensity status.complete needs a complete matrix whose limit reaches the bandwidth (NET-101)', () => {
  // Bandwidth 4, two queries of length 1 at distances 0 and 2 from one event. A matrix built with
  // costLimit 1 stores [0, inf]: the second query reads as disconnected although it lies inside the
  // kernel support, so complete must be 0. Truth table by hand:
  // complete = costsComplete != 0 && 4 <= costLimit (NaN limit fails).
  const cases = [
    [1, 4, 1],
    [1, 1, 0],
    [0, 4, 0],
    [1, Infinity, 1],
    [1, Number.NaN, 0]
  ] as const;
  for (const [costsCompleteWord, matrixCostLimit, expected] of cases) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const floats = (name: string, values: number[]) => {
      const view = createTransientView(graph, name, 'float32', values.length);
      memory.write(view, Float32Array.from(values));
      return view;
    };
    const integers = (name: string, values: number[]) => {
      const view = createTransientView(graph, name, 'uint32', values.length);
      memory.write(view, Uint32Array.from(values));
      return view;
    };
    try {
      const complete = integers('complete', [99]);
      const intensity = floats('intensity', [0, 0]);
      const kernelMass = floats('mass', [0]);
      const validity = integers('validity', [9, 9]);
      const nodes = new GPUNetworkKernelDensity({
        distances: floats('distances', [0, matrixCostLimit >= 2 ? 2 : Infinity]),
        eventCount: 1,
        quadratureLengths: floats('lengths', [1, 1]),
        bandwidth: 4,
        costsComplete: integers('costs-complete', [costsCompleteWord]),
        costLimit: floats('cost-limit', [matrixCostLimit]),
        output: {
          intensity,
          kernelMass,
          validity,
          status: {complete}
        }
      }).getCommandNodes(graph);
      executeCommandNodesOnCPU(nodes, memory);
      expect(memory.readUint32(complete)[0], `${costsCompleteWord} ${matrixCostLimit}`).toBe(
        expected
      );
      // Requesting a certificate retains incumbents even when that certificate is zero.
      expect(Array.from(memory.readUint32(validity))).toEqual([1, 1]);
      const values = memory.readFloat32(intensity);
      expect(Array.from(values).every(Number.isFinite)).toBe(true);
      expect(values[0]).toBeCloseTo(matrixCostLimit >= 2 ? 2 / 3 : 1);
      expect(values[1]).toBeCloseTo(matrixCostLimit >= 2 ? 1 / 3 : 0);
      expect(memory.readFloat32(kernelMass)[0]).toBe(matrixCostLimit >= 2 ? 0.375 : 0.25);
    } finally {
      device.destroy();
    }
  }
});

it('GPUNetworkKernelDensity validates the completeness ports (NET-101)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `kde-${serial++}`, format, length);
  const create =
    (overrides: Record<string, unknown>, complete = true) =>
    () =>
      new GPUNetworkKernelDensity({
        distances: view('float32', 2),
        eventCount: 1,
        quadratureLengths: view('float32', 2),
        bandwidth: 4,
        output: {
          intensity: view('float32', 2),
          kernelMass: view('float32', 1),
          validity: view('uint32', 2),
          ...(complete ? {status: {complete: view('uint32', 1)}} : {})
        },
        ...overrides
      });
  expect(create({})).toThrow(/output\.status\.complete requires costsComplete/);
  // The scalar certificate inputs must have one row; assert structured diagnostics rather
  // than coupling this refusal to the validator's wording (which now includes "exactly").
  for (const [overrides, path] of [
    [{costsComplete: view('uint32', 2)}, 'costsComplete'],
    [{costsComplete: view('uint32', 1), costLimit: view('float32', 2)}, 'costLimit']
  ] as const) {
    let error: unknown;
    try {
      create(overrides)();
    } catch (caught) {
      error = caught;
    }
    expect(error).toMatchObject({
      name: 'AnalysisValidationError',
      id: 'network-kernel-density',
      code: 'VIEW_LENGTH',
      path,
      received: 2,
      expected: {rows: 1}
    });
  }
  expect(create({costsComplete: view('uint32', 1), costLimit: view('float32', 1)})).not.toThrow();
  expect(create({}, false)().props.output).not.toHaveProperty('status');
  expect(create({costsComplete: null}, false)).toThrow(
    /^network-kernel-density costsComplete.*null/
  );
  expect(create({costLimit: null}, false)).toThrow(/^network-kernel-density costLimit.*null/);
});

it('CPL-11 suppresses uncertified KDE without a certificate destination and recovers on replay', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const distances = createTransientView(graph, 'distances', 'float32', 2);
  const lengths = createTransientView(graph, 'lengths', 'float32', 2);
  const costsComplete = createTransientView(graph, 'costs-complete', 'uint32', 1);
  const costLimit = createTransientView(graph, 'cost-limit', 'float32', 1);
  const intensity = createTransientView(graph, 'intensity', 'float32', 2);
  const kernelMass = createTransientView(graph, 'mass', 'float32', 1);
  const validity = createTransientView(graph, 'validity', 'uint32', 2);
  const nodes = new GPUNetworkKernelDensity({
    distances,
    quadratureLengths: lengths,
    costsComplete,
    costLimit,
    eventCount: 1,
    bandwidth: 2,
    output: {intensity, kernelMass, validity}
  }).getCommandNodes(graph);
  memory.write(lengths, Float32Array.of(1, 1));
  try {
    for (const [complete, limit, secondDistance, valid] of [
      [0, 2, Infinity, false],
      [1, 1, Infinity, false],
      [1, 2, 1, true],
      [1, Infinity, 1, true],
      [1, Number.NaN, 1, false],
      [1, 2, 1, true]
    ] as const) {
      memory.write(distances, Float32Array.of(0, secondDistance));
      memory.write(costsComplete, Uint32Array.of(complete));
      memory.write(costLimit, Float32Array.of(limit));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(validity))).toEqual(valid ? [1, 1] : [0, 0]);
      if (valid) {
        // Shapes [1, 1/2], total 3/2: intensity [2/3, 1/3], mass (3/2)/2.
        const values = memory.readFloat32(intensity);
        expect(values).toHaveLength(2);
        expect(Number.isFinite(values[0]) && Number.isFinite(values[1])).toBe(true);
        expect(values[0]).toBeCloseTo(2 / 3);
        expect(values[1]).toBeCloseTo(1 / 3);
        expect(memory.readFloat32(kernelMass)[0]).toBe(0.75);
      } else {
        expect(Array.from(memory.readFloat32(intensity))).toEqual([NaN, NaN]);
        expect(Array.from(memory.readFloat32(kernelMass))).toEqual([NaN]);
      }
    }
  } finally {
    device.destroy();
  }
});

it('VAL-11 names custom density and inhomogeneous K instances and their invalid bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'uint32x2'>(format: Format, length = 1) =>
    createTransientView(graph, `validation-${serial++}`, format, length);
  try {
    const density = {
      id: 'my-density',
      distances: view('float32'),
      eventCount: 1,
      quadratureLengths: view('float32'),
      bandwidth: 1,
      output: {
        intensity: view('float32'),
        kernelMass: view('float32'),
        validity: view('uint32')
      }
    };
    expect(
      () =>
        new GPUNetworkKernelDensity({
          ...density,
          distances: view('uint32') as unknown as typeof density.distances
        })
    ).toThrow(/my-density.*distances/);
    const inhomogeneous = {
      id: 'my-inhomogeneous',
      eventNodeIds: view('uint32'),
      nodeCount: 1,
      intensity: view('float32'),
      nodeDistances: view('float32'),
      edgeLengths: view('float32'),
      edgeNodes: view('uint32x2'),
      radii: view('float32'),
      output: {
        k: view('float32'),
        numerator: view('float32'),
        normalization: view('float32', 3),
        status: view('uint32')
      }
    };
    expect(
      () =>
        new GPUNetworkInhomogeneousK({
          ...inhomogeneous,
          intensity: view('uint32') as unknown as typeof inhomogeneous.intensity
        })
    ).toThrow(/my-inhomogeneous.*intensity/);
  } finally {
    device.destroy();
  }
});

it('CPL-11 reads either supplied loss input even without the other input', () => {
  for (const evidence of ['costsComplete', 'costLimit'] as const) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const distances = createTransientView(graph, 'distances', 'float32', 1);
    const lengths = createTransientView(graph, 'lengths', 'float32', 1);
    const costsComplete = createTransientView(graph, 'complete', 'uint32', 1);
    const costLimit = createTransientView(graph, 'limit', 'float32', 1);
    const intensity = createTransientView(graph, 'intensity', 'float32', 1);
    const kernelMass = createTransientView(graph, 'mass', 'float32', 1);
    const validity = createTransientView(graph, 'validity', 'uint32', 1);
    memory.write(distances, Float32Array.of(0));
    memory.write(lengths, Float32Array.of(1));
    memory.write(costsComplete, Uint32Array.of(0));
    memory.write(costLimit, Float32Array.of(1));
    try {
      executeCommandNodesOnCPU(
        new GPUNetworkKernelDensity({
          distances,
          quadratureLengths: lengths,
          eventCount: 1,
          bandwidth: 2,
          [evidence]: evidence === 'costsComplete' ? costsComplete : costLimit,
          output: {intensity, kernelMass, validity}
        }).getCommandNodes(graph),
        memory
      );
      expect(memory.readUint32(validity)[0]).toBe(0);
      expect(Number.isNaN(memory.readFloat32(intensity)[0])).toBe(true);
      expect(Number.isNaN(memory.readFloat32(kernelMass)[0])).toBe(true);
    } finally {
      device.destroy();
    }
  }
});

it('B1 captures inherited density evidence and nested destinations once before validation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const distances = createTransientView(graph, 'distances', 'float32', 2);
  const lengths = createTransientView(graph, 'lengths', 'float32', 2);
  const costsComplete = createTransientView(graph, 'costsComplete', 'uint32', 1);
  const costLimit = createTransientView(graph, 'costLimit', 'float32', 1);
  const intensity = createTransientView(graph, 'intensity', 'float32', 2);
  const kernelMass = createTransientView(graph, 'mass', 'float32', 1);
  const validity = createTransientView(graph, 'validity', 'uint32', 2);
  let horizonReads = 0;
  let completeReads = 0;
  const status = Object.create({
    get complete() {
      completeReads++;
      return undefined;
    }
  });
  const props = Object.assign(
    Object.create({
      get costLimit() {
        horizonReads++;
        return costLimit;
      }
    }),
    {
      distances,
      quadratureLengths: lengths,
      costsComplete,
      eventCount: 1,
      bandwidth: 3,
      output: Object.create({intensity, kernelMass, validity, status})
    }
  );
  memory.write(distances, Float32Array.of(0, 1));
  memory.write(lengths, Float32Array.of(1, 1));
  memory.write(costsComplete, Uint32Array.of(7));
  memory.write(costLimit, Float32Array.of(2));
  try {
    const contributor = new GPUNetworkKernelDensity(props);
    executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
    expect(Array.from(memory.readFloat32(intensity))).toEqual([NaN, NaN]);
    expect(Array.from(memory.readUint32(validity))).toEqual([0, 0]);
    expect(horizonReads).toBe(1);
    expect(completeReads).toBe(1);
    expect(contributor.props.costLimit).toBe(costLimit);
    expect(contributor.props.output.intensity).toBe(intensity);
    expect(contributor.props.output.validity).toBe(validity);
    expect(contributor.props.output.status).not.toBe(status);
  } finally {
    device.destroy();
  }
});

it('B1 captures inherited inhomogeneous K intensity once and executes the validated snapshot', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'uint32x2'>(format: Format, length = 1) =>
    createTransientView(graph, `inhomogeneous-${serial++}`, format, length);
  const intensity = view('float32');
  let intensityReads = 0;
  const props = Object.assign(
    Object.create({
      get intensity() {
        intensityReads++;
        return intensity;
      }
    }),
    {
      eventNodeIds: view('uint32'),
      nodeCount: 1,
      nodeDistances: view('float32'),
      edgeLengths: view('float32'),
      edgeNodes: view('uint32x2'),
      radii: view('float32'),
      output: Object.create({
        k: view('float32'),
        numerator: view('float32'),
        normalization: view('float32', 3),
        status: view('uint32')
      })
    }
  );
  try {
    const contributor = new GPUNetworkInhomogeneousK(props);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    expect(contributor.props.intensity).toBe(intensity);
    expect(intensityReads).toBe(1);
  } finally {
    device.destroy();
  }
});

it('S1 uses identical horizon evidence under flush-to-zero for suppression and certification', async () => {
  const {WgslExec, WgslParser} = await import('wgsl_reflect');
  const bandwidth = new Float32Array(Uint32Array.of(16).buffer)[0];
  // Positive subnormals are ordered by their unsigned IEEE words. 8 < 16 is insufficient,
  // regardless of arithmetic flushing both operands to zero. Infinity is a sufficient horizon.
  for (const [horizonBits, expected] of [
    [8, 0],
    [16, 1],
    [32, 1],
    [0, 0],
    [0x80000000, 0],
    [0x7f800000, 1],
    [0x7fc00000, 0],
    [0xffc00000, 0],
    [0xbf800000, 0]
  ] as const) {
    for (const certificate of [false, true]) {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format) =>
        createTransientView(graph, `horizon-${serial++}`, format, 1);
      try {
        new GPUNetworkKernelDensity({
          distances: view('float32'),
          eventCount: 1,
          quadratureLengths: view('float32'),
          bandwidth,
          costsComplete: view('uint32'),
          costLimit: view('float32'),
          output: {
            intensity: view('float32'),
            kernelMass: view('float32'),
            validity: view('uint32'),
            ...(certificate ? {status: {complete: view('uint32')}} : {})
          }
        }).getCommandNodes(graph);
        for (const suffix of certificate ? ['complete'] : ['suppress-mass', 'suppress-intensity']) {
          const record = getWGSLKernelNodeRecord(graph, `network-kernel-density-${suffix}`)!;
          // Model permitted GPU FTZ behavior only at floating horizon comparisons. Integer bit
          // comparisons and bitcasts retain their input words. This fails for the old <= route.
          const source =
            `fn flushHorizon(value:f32)->f32 { if (abs(value) < 1.1754943508222875e-38f) { return 0.0; } return value; }\n` +
            record.source.replace(
              /([\d.e+-]+f) <= (costLimit\[costLimitOffset\])/g,
              'flushHorizon($1) <= flushHorizon($2)'
            );
          const storage: Record<string, Float32Array | Uint32Array> = {
            costsComplete: Uint32Array.of(7),
            costLimit: new Float32Array(Uint32Array.of(horizonBits).buffer),
            mass: Float32Array.of(123),
            intensity: Float32Array.of(123),
            validity: Uint32Array.of(1),
            complete: Uint32Array.of(99)
          };
          const bindings = Object.fromEntries(
            record.bindings.map((binding, index) => [index, storage[binding.name]])
          );
          const interpreter = new WgslExec(
            new WgslParser().parse(source.replace(/^\s*_\s*=[^;]*;/gm, ''))
          );
          interpreter.dispatchWorkgroups(
            'main',
            [1, 1, 1],
            {0: bindings},
            {constants: record.constants}
          );
          if (certificate) expect(storage['complete'][0], `${horizonBits}`).toBe(expected);
          else if (suffix === 'suppress-mass')
            expect(Array.from(storage['mass'])).toEqual(expected ? [123] : [NaN]);
          else {
            expect(Array.from(storage['intensity'])).toEqual(expected ? [123] : [NaN]);
            expect(Array.from(storage['validity'])).toEqual(expected ? [1] : [0]);
          }
        }
      } finally {
        device.destroy();
      }
    }
  }
});

it('VAL-11 structured diagnostics preserve IDs, paths, formats and received row counts', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'uint32x2'>(format: Format, length = 1) =>
    createTransientView(graph, `structured-${serial++}`, format, length);
  try {
    for (const invalid of [view('uint32'), view('float32', 2)]) {
      for (const producer of ['density', 'inhomogeneous'] as const) {
        let error: unknown;
        try {
          if (producer === 'density')
            new GPUNetworkKernelDensity({
              id: 'custom-density',
              distances: invalid as GraphDataView<'float32'>,
              eventCount: 1,
              quadratureLengths: view('float32'),
              bandwidth: 1,
              output: {
                intensity: view('float32'),
                kernelMass: view('float32'),
                validity: view('uint32')
              }
            });
          else
            new GPUNetworkInhomogeneousK({
              id: 'custom-inhomogeneous',
              eventNodeIds: view('uint32'),
              nodeCount: 1,
              intensity: invalid as GraphDataView<'float32'>,
              nodeDistances: view('float32'),
              edgeLengths: view('float32'),
              edgeNodes: view('uint32x2'),
              radii: view('float32'),
              output: {
                k: view('float32'),
                numerator: view('float32'),
                normalization: view('float32', 3),
                status: view('uint32')
              }
            });
        } catch (caught) {
          error = caught;
        }
        expect(error).toMatchObject({
          id: `custom-${producer}`,
          path: producer === 'density' ? 'distances' : 'intensity',
          received: invalid.format === 'uint32' ? 'uint32' : 2
        });
        if (invalid.format === 'float32')
          expect(error).toMatchObject({code: 'VIEW_LENGTH', expected: {rows: 1}});
        else expect(error).toMatchObject({code: 'VIEW_FORMAT', expected: {formats: ['float32']}});
      }
    }
  } finally {
    device.destroy();
  }
});

it('S3 schedules invalid-count reduction after incomplete intensity suppression', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length = 1) =>
    createTransientView(graph, `reduction-order-${serial++}`, format, length);
  const validity = view('uint32', 2);
  try {
    const nodes = new GPUNetworkKernelDensity({
      distances: view('float32', 2),
      quadratureLengths: view('float32', 2),
      eventCount: 1,
      costsComplete: view('uint32'),
      bandwidth: 2,
      output: {
        intensity: view('float32', 2),
        kernelMass: view('float32'),
        validity,
        status: {invalidCount: view('uint32')}
      }
    }).getCommandNodes(graph);
    const suppression = nodes.findIndex(node => node.id.endsWith('-suppress-intensity'));
    const markInvalid = nodes.findIndex(node => node.id.endsWith('-mark-invalid-queries'));
    const reduction = nodes.findIndex(node => node.id.includes('-invalid-count'));
    expect(suppression).toBeGreaterThanOrEqual(0);
    expect(markInvalid).toBeGreaterThan(suppression);
    expect(reduction).toBeGreaterThan(markInvalid);

    const marker = nodes[markInvalid];
    const invalidQueries = getWGSLKernelNodeRecord(graph, marker.id)!.bindings[1].view;
    const memory = createCPUGraphMemory();
    // A zero row contributes 1 to the sum. Any nonzero word contributes 0.
    memory.write(validity, Uint32Array.of(0, 7));
    executeCommandNodesOnCPU([marker], memory);
    expect(Array.from(memory.readUint32(invalidQueries))).toEqual([1, 0]);
  } finally {
    device.destroy();
  }
});
