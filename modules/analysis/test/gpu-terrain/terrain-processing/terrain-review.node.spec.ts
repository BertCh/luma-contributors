// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUTerrainBreach} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-breach';
import {GPUTerrainBranchingFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {GPUTerrainCatchmentSummary} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary';
import {GPUTerrainDepressionHierarchy} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy';
import {GPUTerrainDEMCoregistration} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration';
import {GPUTerrainFlowFractions} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-flow-fractions';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createTerrainReviewFixture,
  createTerrainReviewPackedDispatch
} from './terrain-review-fixture';

const GRID = {
  width: 1,
  height: 1,
  affine: [1, 0, 0, 0, 1, 0] as const,
  pixelInterpretation: 'area' as const
};

for (const kind of ['tiled', 'catchment'] as const) {
  it(`rejects mixed storage usage before graph mutation and accepts aligned writable packing (${kind})`, () => {
    const device = createNullWebGPUDevice();
    const fixture = createTerrainReviewFixture(device);
    try {
      for (const byteOffset of [4, 256]) {
        const scenario = createTerrainReviewPackedDispatch(fixture, kind, byteOffset, 'mixed');
        const resources = vi.spyOn(fixture.graph, 'createTransientBuffer');
        const declarations = vi.spyOn(fixture.graph, 'add');
        expect(scenario.create).toThrow(/input and output aligned binding usage scopes/);
        expect(resources).not.toHaveBeenCalled();
        expect(declarations).not.toHaveBeenCalled();
        resources.mockRestore();
        declarations.mockRestore();
      }
      const legal = createTerrainReviewPackedDispatch(fixture, kind, 256, 'writable');
      expect(() => fixture.graph.add(legal.create())).not.toThrow();
    } finally {
      fixture.destroy();
      device.destroy();
    }
  });
}

for (const execution of ['resident', 'auto'] as const) {
  it(`accepts shared stream/output storage when separate dispatches use it (${execution})`, () => {
    const device = createNullWebGPUDevice();
    const fixture = createTerrainReviewFixture(device);
    try {
      const scenario = createTerrainReviewPackedDispatch(fixture, 'tiled', 256, 'mixed', execution);
      expect(() => fixture.graph.add(scenario.create())).not.toThrow();
    } finally {
      fixture.destroy();
      device.destroy();
    }
  });
}

it('accepts valid packed output layouts throughout the terrain processing validator callers', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let identifier = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length = 1) =>
      createTransientView(graph, `field-${identifier++}`, format, length);
    const packed = view('uint32', 128);
    const slice = <Format extends 'float32' | 'uint32'>(
      format: Format,
      length: number,
      byteOffset: number
    ) => graph.createDataView(packed.buffer, {format, length, byteOffset});
    const contributors = [
      new GPUTerrainCatchmentSummary({
        grid: GRID,
        elevation: view('float32'),
        basinIds: view('uint32'),
        streams: view('uint32'),
        flowDirections: view('uint32'),
        basinCount: 1,
        histogramBins: 1,
        output: {
          statistics: slice('float32', 6, 0),
          hypsometry: slice('float32', 1, 24),
          profiles: view('float32', 3),
          profileStatus: view('uint32')
        }
      }),
      new GPUTerrainDepressionHierarchy({
        grid: GRID,
        elevation: view('float32'),
        outlets: view('uint32'),
        stages: view('float32'),
        nodeCapacity: 1,
        output: {
          nodes: slice('uint32', 4, 0),
          cellNodes: slice('uint32', 1, 16),
          levels: view('float32', 2),
          curves: view('float32', 2),
          status: view('uint32', 3)
        }
      }),
      new GPUTerrainDEMCoregistration({
        grid: GRID,
        reference: view('float32'),
        moving: view('float32'),
        stableMask: view('uint32'),
        differenceVariance: view('float32'),
        output: {
          differences: slice('float32', 2, 0),
          volume: slice('float32', 3, 8),
          fit: view('float32', 4),
          validity: view('uint32')
        }
      }),
      new GPUTerrainFlowFractions({
        grid: GRID,
        elevation: view('float32'),
        flowDirections: view('uint32'),
        parameters: view('float32', 8),
        routing: 'mfd-freeman',
        output: {fractions: slice('float32', 8, 0), edgeLengths: slice('float32', 8, 256)}
      })
    ];
    for (const contributor of contributors) {
      expect(() => contributor.getCommandNodes(graph), contributor.id).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('rejects separately imported physical breach input/output aliases at construction', () => {
  const device = createNullWebGPUDevice();
  const buffer = device.createBuffer({byteLength: 4, usage: Buffer.STORAGE});
  try {
    const graph = new GPUCommandGraph(device);
    const output = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      length = 1
    ) => createTransientView(graph, name, format, length);
    expect(
      () =>
        new GPUTerrainBreach({
          grid: GRID,
          elevation: importGraphBuffer(graph, 'source', buffer, 'float32', 1),
          pitCells: output('pits', 'uint32'),
          maximumSteps: 1,
          maximumCutDepth: 1,
          elevationDecrement: 0.25,
          output: {
            elevation: importGraphBuffer(graph, 'edited', buffer, 'float32', 1),
            cutDepth: output('depth', 'float32'),
            modificationMask: output('mask', 'uint32'),
            pathCells: output('paths', 'uint32', 2),
            pathLengths: output('lengths', 'uint32'),
            pitStatus: output('status', 'uint32'),
            unresolvedCount: output('unresolved', 'uint32')
          }
        })
    ).toThrow(/terrain-breach.*input and output.*overlap/);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('accepts genuine zero-byte pit/path views sharing one imported buffer', () => {
  const device = createNullWebGPUDevice();
  const empty = device.createBuffer({byteLength: 0, usage: Buffer.COPY_SRC});
  try {
    const graph = new GPUCommandGraph(device);
    const output = <Format extends 'float32' | 'uint32'>(name: string, format: Format) =>
      createTransientView(graph, name, format, 1);
    const backing = importGraphBuffer(graph, 'unused', empty, 'uint32', 0);
    const unused = () => graph.createDataView(backing.buffer, {format: 'uint32', length: 0});
    const contributor = new GPUTerrainBreach({
      grid: GRID,
      elevation: output('source', 'float32'),
      pitCells: unused(),
      maximumSteps: 0xffffffff,
      maximumCutDepth: 1,
      elevationDecrement: 0.25,
      burnDepth: output('burn', 'float32'),
      output: {
        elevation: output('edited', 'float32'),
        cutDepth: output('depth', 'float32'),
        modificationMask: output('mask', 'uint32'),
        pathCells: unused(),
        pathLengths: unused(),
        pitStatus: unused(),
        unresolvedCount: output('unresolved', 'uint32')
      }
    });
    const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
    for (const [name, source] of sources) expect(auditWGSLModule(source), name).toEqual([]);
  } finally {
    empty.destroy();
    device.destroy();
  }
});

it('accepts adjacent tile output and aggregate status ranges in one packed buffer', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const backing = createTransientView(graph, 'packed', 'float32', 2);
    const view = <Format extends 'float32' | 'uint32'>(name: string, format: Format) =>
      createTransientView(graph, name, format, 1);
    const contributor = new GPUTerrainTiledFlow({
      grid: GRID,
      maximumIterations: 1,
      tiles: [
        {
          column: 0,
          row: 0,
          width: 1,
          height: 1,
          owned: [0, 0, 1, 1],
          flowDirections: view('directions', 'uint32'),
          streams: view('streams', 'uint32'),
          runoff: view('runoff', 'float32'),
          output: {
            accumulation: graph.createDataView(backing.buffer, {format: 'float32', length: 1}),
            status: view('state', 'uint32'),
            reachIds: view('reach', 'uint32')
          }
        }
      ],
      output: {
        status: {
          invalidCount: graph.createDataView(backing.buffer, {
            format: 'uint32',
            byteOffset: 4,
            length: 1
          })
        }
      }
    });
    expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

for (const coefficient of [1e20, 1e40]) {
  it(`validates distance-only affine conversions before graph mutation (${coefficient})`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const view = <Format extends 'float32' | 'uint32'>(
        name: string,
        format: Format,
        length = 1
      ) => createTransientView(graph, name, format, length);
      const props = {
        grid: {...GRID, affine: [coefficient, 0, 0, 0, coefficient, 0] as const},
        fractions: view('fractions', 'float32', 8),
        values: view('source', 'float32'),
        mode: 'expected-distance' as const,
        maximumIterations: 1,
        output: {
          values: view('distances', 'float32'),
          status: view('status', 'uint32'),
          unresolvedCount: view('unresolved', 'uint32')
        }
      };
      const declarations = vi.spyOn(graph, 'createTransientBuffer');
      if (coefficient === 1e40) {
        expect(() => new GPUTerrainBranchingFlow(props)).toThrow(
          /terrain-branching-flow grid distance parameters/
        );
        expect(declarations).not.toHaveBeenCalled();
      } else {
        const contributor = new GPUTerrainBranchingFlow(props);
        for (const [name, source] of getParsedKernelSources(contributor.getCommandNodes(graph))) {
          expect(source).not.toContain('fn getGroundCellArea');
          expect(auditWGSLModule(source), name).toEqual([]);
        }
      }
      declarations.mockRestore();
    } finally {
      device.destroy();
    }
  });
}

it('checks STORAGE usage before declaring nested terrain scratch', () => {
  const device = createNullWebGPUDevice();
  const buffer = device.createBuffer({byteLength: 32, usage: Buffer.COPY_SRC});
  try {
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32' | 'uint32'>(name: string, format: Format) =>
      createTransientView(graph, name, format, 1);
    const contributor = new GPUTerrainBranchingFlow({
      grid: GRID,
      fractions: importGraphBuffer(graph, 'fractions', buffer, 'float32', 8),
      values: view('source', 'float32'),
      mode: 'transport',
      maximumIterations: 1,
      output: {
        values: view('values', 'float32'),
        status: view('status', 'uint32'),
        unresolvedCount: view('unresolved', 'uint32')
      }
    });
    const declarations = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/Buffer.STORAGE/);
    expect(declarations).not.toHaveBeenCalled();
    declarations.mockRestore();
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('allows packed branching values and count in separate dispatches but rejects co-bound status/count prefixes', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const coBound of [false, true]) {
      const graph = new GPUCommandGraph(device);
      const backing = createTransientView(graph, 'packed', 'float32', 2);
      const values = coBound
        ? createTransientView(graph, 'values', 'float32', 1)
        : graph.createDataView(backing.buffer, {format: 'float32', length: 1});
      const status = coBound
        ? graph.createDataView(backing.buffer, {format: 'uint32', length: 1})
        : createTransientView(graph, 'status', 'uint32', 1);
      const props = {
        grid: GRID,
        mode: 'transport' as const,
        maximumIterations: 1,
        fractions: createTransientView(graph, 'fractions', 'float32', 8),
        values: createTransientView(graph, 'source', 'float32', 1),
        output: {
          values,
          status,
          unresolvedCount: graph.createDataView(backing.buffer, {
            format: 'uint32',
            byteOffset: 4,
            length: 1
          })
        }
      };
      if (coBound) expect(() => new GPUTerrainBranchingFlow(props)).toThrow(/aligned binding/);
      else expect(() => new GPUTerrainBranchingFlow(props).getCommandNodes(graph)).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('does not bind or alias-check an unused optional distance plane in transport', () => {
  const device = createNullWebGPUDevice();
  const buffer = device.createBuffer({byteLength: 32, usage: Buffer.COPY_SRC});
  try {
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32' | 'uint32'>(name: string, format: Format) =>
      createTransientView(graph, name, format, 1);
    const contributor = new GPUTerrainBranchingFlow({
      grid: GRID,
      fractions: createTransientView(graph, 'fractions', 'float32', 8),
      values: view('source', 'float32'),
      edgeLengths: importGraphBuffer(graph, 'unused', buffer, 'float32', 8),
      mode: 'transport',
      maximumIterations: 1,
      output: {
        values: view('values', 'float32'),
        status: view('status', 'uint32'),
        unresolvedCount: view('unresolved', 'uint32')
      }
    });
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('centres Horn samples before weighting vector-ruggedness normals', async () => {
  const {GPUTerrainVectorRuggedness} = await import(
    '../../../src/gpu-terrain/topographic-position/gpu-terrain-vector-ruggedness'
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const source = createTransientView(graph, 'source', 'float32', 49);
    const contributor = new GPUTerrainVectorRuggedness({
      grid: {...GRID, width: 7, height: 7},
      elevation: {id: 'source', format: 'float32', storage: {kind: 'buffer', values: source}},
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      output: {vectorRuggedness: createTransientView(graph, 'ruggedness', 'float32', 49)}
    });
    const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
    expect(
      [...sources.values()].some(source =>
        source.includes('windowValues[k] = sampleValue - centreElevation')
      )
    ).toBe(true);
    for (const [name, source] of sources) expect(auditWGSLModule(source), name).toEqual([]);
  } finally {
    device.destroy();
  }
});
