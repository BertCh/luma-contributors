// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPURasterFractionalCoverage,
  GPURasterFractionalZonalStatistics
} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import {GPURasterFractionalCategoricalStatistics} from '../../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createScalabilityProps} from './raster-fractional-scalability-fixture';

it('indexes 100k polygons over a 4096 square raster and retains the named brute-force cap', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const {coverage, categorical} = createScalabilityProps(graph, 100000, 1100000, {
    width: 4096,
    height: 4096,
    affine: [1, 0, 0, 0, 1, 0],
    pixelInterpretation: 'area'
  });
  const indexed = new GPURasterFractionalCoverage(coverage);
  const indexedNodes = indexed.getCommandNodes(graph);
  expect(indexedNodes.length).toBeLessThanOrEqual(indexed.estimatedCommandNodeCount);
  expect(
    () => new GPURasterFractionalCoverage({...coverage, searchStructure: 'brute-force'})
  ).toThrow(/zones times grid cells.*4194304/);
  expect(
    () => new GPURasterFractionalCategoricalStatistics({...categorical, mode: 'exact'})
  ).toThrow(/exact mode/);
  const statistics = new GPURasterFractionalCategoricalStatistics(categorical);
  expect(statistics.getCommandNodes(graph).length).toBeLessThanOrEqual(
    statistics.estimatedCommandNodeCount
  );
});

it('accepts a 30M-row tuple capacity without allocating dense zone/cell scratch', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const {coverage, categorical} = createScalabilityProps(graph, 100000, 30000000, {
    width: 4096,
    height: 4096,
    affine: [1, 0, 0, 0, 1, 0],
    pixelInterpretation: 'area'
  });
  const indexed = new GPURasterFractionalCoverage(coverage);
  const statistics = new GPURasterFractionalCategoricalStatistics(categorical);
  expect(indexed.candidateCapacity).toBe(120000000);
  expect(indexed.estimatedCommandNodeCount).toBeLessThanOrEqual(32768);
  expect(statistics.estimatedCommandNodeCount).toBeLessThanOrEqual(32768);
});

it('keeps the number of shader modules and pipelines constant as candidate and tuple tiles grow', () => {
  for (const kind of ['coverage', 'categorical'] as const) {
    const shapes = [2, 5].map(tileCount => {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      const props = createScalabilityProps(graph, 2, tileCount * 512 + 17);
      const contributor =
        kind === 'coverage'
          ? new GPURasterFractionalCoverage({
              ...props.coverage,
              candidateCapacity: tileCount * 262144 + 17
            })
          : new GPURasterFractionalCategoricalStatistics({
              ...props.categorical,
              tupleTileSize: 512
            });
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      const tileNodes = nodes.filter(node =>
        kind === 'coverage' ? node.id.endsWith('-lookup') : node.id.endsWith('-prepare-tuples')
      );
      expect(tileNodes.at(-1)?.workload?.maximumInvocationCount).toBe(17);
      expect(tileNodes.at(-1)?.workload?.maximumWorkgroupCount).toBe(1);
      expect(tileNodes[0].workload?.maximumInvocationCount).toBe(
        kind === 'coverage' ? 262144 : 512
      );
      // Inspect complete generated sources, including the published scan/sort/compaction kernels.
      const sources = getGPUCommandNodeKernelSources(nodes);
      for (const kernel of sources) {
        expect(findReservedIdentifiers(stripWGSLComments(kernel.source)), kernel.id).toEqual([]);
        const tokens = new Set(
          stripWGSLComments(kernel.source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []
        );
        // Full W3C reserved-word list, independently of the node interpreter.
        expect(
          [...WGSL_RESERVED_WORDS].filter(word => tokens.has(word)),
          kernel.id
        ).toEqual([]);
      }
      expect(getParsedKernelSources(nodes).size).toBeGreaterThan(5);
      return {
        modules: new Set(sources.map(kernel => kernel.source)).size,
        pipelines: getGPUCommandNodePipelineKeys(nodes).size
      };
    });
    expect(shapes[1], kind).toEqual(shapes[0]);
  }
});

it('rejects overlapping output fields and status words at construction', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const {coverage, categorical} = createScalabilityProps(graph);
  expect(
    () =>
      new GPURasterFractionalCoverage({
        ...coverage,
        output: {...coverage.output, cellIds: coverage.output.zoneIds}
      })
  ).toThrow(/^raster-fractional-coverage .*logical byte ranges must not overlap/);
  expect(
    () =>
      new GPURasterFractionalCoverage({
        ...coverage,
        output: {
          ...coverage.output,
          status: {...coverage.output.status, candidateOverflow: coverage.output.status.count}
        }
      })
  ).toThrow(/^raster-fractional-coverage .*logical byte ranges must not overlap/);
  expect(
    () =>
      new GPURasterFractionalCategoricalStatistics({
        ...categorical,
        output: {...categorical.output, proportions: categorical.output.classAreas}
      })
  ).toThrow(/^fractional-categorical-statistics .*logical byte ranges must not overlap/);
});

it('preflights aligned input ranges and derived scratch before declaring any graph resource', () => {
  for (const kind of ['coverage', 'categorical', 'zonal'] as const) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const props = createScalabilityProps(graph, 2, 16);
    // This view has 32 logical bytes and a 252-byte alignment prefix.
    const buffer = graph.createTransientBuffer({
      id: 'unaligned',
      byteLength: 512,
      usage: Buffer.STORAGE
    });
    const parameters = graph.createDataView(buffer, {
      format: 'float32',
      byteOffset: 252,
      length: 8
    });
    const contributor =
      kind === 'coverage'
        ? new GPURasterFractionalCoverage({...props.coverage, parameters})
        : kind === 'categorical'
          ? new GPURasterFractionalCategoricalStatistics({...props.categorical, parameters})
          : new GPURasterFractionalZonalStatistics({
              grid: props.coverage.grid,
              input: props.categorical.input,
              zoneCount: 2,
              quantity: 'extensive',
              coverage: props.coverage.output,
              parameters,
              output: {
                totals: props.categorical.output.coverageSums,
                means: props.categorical.output.classAreas,
                coveredArea: props.categorical.output.coveredArea,
                validArea: props.categorical.output.validArea,
                validity: props.categorical.output.validity
              }
            });
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 256}
    });
    const resources = vi.spyOn(graph, 'createTransientBuffer');
    const views = vi.spyOn(graph, 'createDataView');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/284 bytes/);
    expect(resources).not.toHaveBeenCalled();
    expect(views).not.toHaveBeenCalled();
    resources.mockRestore();
    views.mockRestore();
  }
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {coverage} = createScalabilityProps(graph);
  const contributor = new GPURasterFractionalCoverage(coverage);
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 4096}
  });
  const resources = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/candidate tile/);
  expect(resources).not.toHaveBeenCalled();
  resources.mockRestore();
});

it('preflights categorical sort scratch even when every caller view fits', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {categorical} = createScalabilityProps(graph);
  const contributor = new GPURasterFractionalCategoricalStatistics({
    ...categorical,
    tupleTileSize: 256
  });
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 256}
  });
  const resources = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/sort tiles needs 1024 bytes/);
  expect(resources).not.toHaveBeenCalled();
  resources.mockRestore();
});

it('rejects separately imported writable aliases, including disjoint rows with aligned overlap', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {coverage, categorical} = createScalabilityProps(graph);
  const buffer = device.createBuffer({byteLength: 1024, usage: Buffer.STORAGE});
  const first = importGraphBuffer(graph, 'first-handle', buffer, 'float32', 256);
  const second = importGraphBuffer(graph, 'second-handle', buffer, 'float32', 256);
  const firstOutput = graph.createDataView(first.buffer, {
    format: 'float32',
    byteOffset: 256,
    length: 4
  });
  const secondOutput = graph.createDataView(second.buffer, {
    format: 'float32',
    byteOffset: 272,
    length: 4
  });
  // Logical rows are disjoint, but the second aligned binding starts at byte 256 too.
  expect(
    () =>
      new GPURasterFractionalCategoricalStatistics({
        ...categorical,
        output: {...categorical.output, coverageSums: firstOutput, classAreas: secondOutput}
      })
  ).toThrow(/^fractional-categorical-statistics .*physical buffer.*write.*one imported handle/);
  const firstIds = graph.createDataView(first.buffer, {
    format: 'uint32',
    byteOffset: 256,
    length: 16
  });
  const secondIds = graph.createDataView(second.buffer, {
    format: 'uint32',
    byteOffset: 320,
    length: 16
  });
  expect(
    () =>
      new GPURasterFractionalCoverage({
        ...coverage,
        output: {...coverage.output, zoneIds: firstIds, cellIds: secondIds}
      })
  ).toThrow(/^raster-fractional-coverage .*physical buffer.*write.*one imported handle/);
  buffer.destroy();
});

it('keeps uint32 dimensions in conservative bounds and saturates rectangle counts before scanning', () => {
  for (const width of [16777217, 1431655765]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const {coverage} = createScalabilityProps(graph);
    const nodes = new GPURasterFractionalCoverage({
      ...coverage,
      grid: {...coverage.grid, width, height: width === 16777217 ? 1 : 3},
      candidateCapacity: 128
    }).getCommandNodes(graph);
    const source = getParsedKernelSources(nodes).get('raster-fractional-coverage-bounds')!;
    expect(source).toContain(`${width}u`);
    expect(source).not.toContain(`${width}.0`);
    expect(source).toContain('getSaturatedScanProduct(right-left,top-bottom)');
  }
});

it('uses the complete W3C reserved-word guard including the review regression identifiers', () => {
  for (const word of ['target', 'active', 'filter', 'module', 'meta'])
    expect(WGSL_RESERVED_WORDS.has(word)).toBe(true);
  expect(WGSL_RESERVED_WORDS.has('sample')).toBe(false);
  for (const word of WGSL_RESERVED_WORDS) {
    const identifiers = new Set(`fn ${word}(){}`.match(/[A-Za-z_][A-Za-z_0-9]*/g));
    expect(
      [...WGSL_RESERVED_WORDS].filter(reservedWord => identifiers.has(reservedWord))
    ).toContain(word);
  }
});

it('audits generated categorical variants and the zonal consumer against the full reserved-word set', () => {
  for (const cellSizeMode of ['uniform', 'geographic', 'web-mercator'] as const) {
    for (const withParameters of [false, true]) {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      const {coverage, categorical} = createScalabilityProps(graph);
      const contributor = new GPURasterFractionalCategoricalStatistics({
        ...categorical,
        cellSizeMode,
        parameters: withParameters ? coverage.parameters : undefined,
        mode: 'exact',
        input: {...categorical.input, noDataValue: 0xffffffff, validity: coverage.output.zoneIds}
      });
      for (const kernel of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
        const code = stripWGSLComments(kernel.source);
        expect(findReservedIdentifiers(code), kernel.id).toEqual([]);
        const identifiers = new Set(code.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []);
        expect(
          [...WGSL_RESERVED_WORDS].filter(word => identifiers.has(word)),
          kernel.id
        ).toEqual([]);
      }
    }
  }
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const {coverage, categorical} = createScalabilityProps(graph);
  const zonal = new GPURasterFractionalZonalStatistics({
    grid: coverage.grid,
    input: categorical.input,
    zoneCount: 2,
    quantity: 'extensive',
    coverage: coverage.output,
    parameters: coverage.parameters,
    output: {
      totals: categorical.output.coverageSums,
      means: categorical.output.classAreas,
      coveredArea: categorical.output.coveredArea,
      validArea: categorical.output.validArea,
      validity: categorical.output.validity
    }
  });
  for (const kernel of getGPUCommandNodeKernelSources(zonal.getCommandNodes(graph))) {
    const code = stripWGSLComments(kernel.source);
    expect(findReservedIdentifiers(code), kernel.id).toEqual([]);
    const identifiers = new Set(code.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []);
    expect(
      [...WGSL_RESERVED_WORDS].filter(word => identifiers.has(word)),
      kernel.id
    ).toEqual([]);
  }
});

it('preflights backing buffers and derived scratch against maxBufferSize before graph mutation', () => {
  for (const kind of ['coverage', 'categorical', 'zonal'] as const) {
    for (const backingBuffer of [false, true]) {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const {coverage, categorical} = createScalabilityProps(graph, kind === 'zonal' ? 8 : 2);
      const parameters = backingBuffer
        ? graph.createDataView(
            graph.createTransientBuffer({
              id: 'large-backing',
              byteLength: 4096,
              usage: Buffer.STORAGE
            }),
            {format: 'float32', length: 8}
          )
        : coverage.parameters;
      const contributor =
        kind === 'coverage'
          ? new GPURasterFractionalCoverage({...coverage, parameters})
          : kind === 'categorical'
            ? new GPURasterFractionalCategoricalStatistics({
                ...categorical,
                parameters,
                tupleTileSize: 256
              })
            : new GPURasterFractionalZonalStatistics({
                grid: coverage.grid,
                input: categorical.input,
                zoneCount: 8,
                quantity: 'extensive',
                coverage: coverage.output,
                parameters,
                output: {
                  totals: categorical.output.coverageSums,
                  means: categorical.output.classAreas,
                  coveredArea: categorical.output.coveredArea,
                  validArea: categorical.output.validArea,
                  validity: categorical.output.validity
                }
              });
      Object.defineProperty(device, 'limits', {
        value: {...device.limits, maxBufferSize: backingBuffer ? 1024 : kind === 'zonal' ? 96 : 512}
      });
      const resources = vi.spyOn(graph, 'createTransientBuffer');
      const views = vi.spyOn(graph, 'createDataView');
      const passes = vi.spyOn(graph, 'addComputePass');
      const imports = vi.spyOn(graph, 'importBuffer');
      const allocation = backingBuffer
        ? 'large-backing'
        : kind === 'categorical'
          ? 'sort tiles'
          : 'zone summary';
      // The nested triangulation preflight now rejects its scratch first. Preserve the exact
      // contributor, allocation and device budget instead of requiring the outer helper's wording.
      const expectedRefusal =
        kind === 'coverage' && !backingBuffer
          ? /^raster-fractional-coverage scratch byteLength must be an integer in \[0, 512\]$/
          : new RegExp(
              `^${contributor.id} ${allocation} buffer byteLength \\(device\\.limits\\.maxBufferSize\\) must be at most ${
                backingBuffer ? 1024 : kind === 'zonal' ? 96 : 512
              }$`
            );
      expect(() => contributor.getCommandNodes(graph)).toThrow(expectedRefusal);
      expect(resources).not.toHaveBeenCalled();
      expect(views).not.toHaveBeenCalled();
      expect(passes).not.toHaveBeenCalled();
      expect(imports).not.toHaveBeenCalled();
      resources.mockRestore();
      views.mockRestore();
      passes.mockRestore();
      imports.mockRestore();
      device.destroy();
    }
  }
});

it('uses floating literals for large finite cell areas and refuses unrepresentable numeric props at construction', () => {
  for (const cellArea of [1e20, Math.fround(3.4028234663852886e38)]) {
    for (const kind of ['categorical', 'zonal'] as const) {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      const {coverage, categorical} = createScalabilityProps(graph, 2, 16, {
        width: 4,
        height: 4,
        affine: [cellArea, 0, 0, 0, 1, 0],
        pixelInterpretation: 'area'
      });
      const zonalProps = {
        grid: coverage.grid,
        input: categorical.input,
        zoneCount: 2,
        quantity: 'extensive' as const,
        coverage: coverage.output,
        output: {
          totals: categorical.output.coverageSums,
          means: categorical.output.classAreas,
          coveredArea: categorical.output.coveredArea,
          validArea: categorical.output.validArea,
          validity: categorical.output.validity
        }
      };
      const contributor =
        kind === 'categorical'
          ? new GPURasterFractionalCategoricalStatistics({...categorical, parameters: undefined})
          : new GPURasterFractionalZonalStatistics(zonalProps);
      const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      const literal = String(Math.fround(cellArea));
      const floatingLiteral = /[.eE]/.test(literal) ? literal : `${literal}.0`;
      expect(
        sources.some(
          kernel =>
            kernel.source.includes(`* ${floatingLiteral}`) ||
            kernel.source.includes(`*${floatingLiteral}`) ||
            kernel.source.includes(`let area=${floatingLiteral};`)
        )
      ).toBe(true);
      for (const kernel of sources) {
        // An integral decimal expansion without a decimal point/exponent is an abstract-int
        // before conversion, and must fit signed 64-bit even when multiplied by an f32.
        const tokens =
          stripWGSLComments(kernel.source).match(
            /\b[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?[fhiu]?\b/g
          ) ?? [];
        for (const token of tokens) {
          if (/^[0-9]+$/.test(token))
            expect(BigInt(token), kernel.id).toBeLessThanOrEqual(9223372036854775807n);
        }
      }
      expect(
        () =>
          new GPURasterFractionalZonalStatistics({
            ...zonalProps,
            zoneCount: 0x100000000
          })
      ).toThrow(/zoneCount.*4294967295/);
      for (const value of [Infinity, NaN, 1e40]) {
        expect(
          () =>
            new GPURasterFractionalZonalStatistics({
              ...zonalProps,
              input: {...zonalProps.input, scale: value}
            })
        ).toThrow(/input.scale.*finite float32/);
        expect(
          () =>
            new GPURasterFractionalZonalStatistics({
              ...zonalProps,
              input: {...zonalProps.input, offset: value}
            })
        ).toThrow(/input.offset.*finite float32/);
      }
      expect(
        () =>
          new GPURasterFractionalZonalStatistics({
            ...zonalProps,
            grid: {...zonalProps.grid, affine: [1e40, 0, 0, 0, 1, 0]}
          })
      ).toThrow(/grid.affine determinant.*finite float32/);
      for (const noDataValue of [-1, 0x100000000, 1.5]) {
        expect(
          () =>
            new GPURasterFractionalZonalStatistics({
              ...zonalProps,
              input: {...zonalProps.input, noDataValue}
            })
        ).toThrow(/input.noDataValue.*uint32/);
      }
      graph.device.destroy();
    }
  }
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const {categorical} = createScalabilityProps(graph);
  expect(
    () =>
      new GPURasterFractionalCategoricalStatistics({
        ...categorical,
        parameters: undefined,
        cellSizeMode: 'geographic',
        grid: {...categorical.grid, affine: [1, 0, 0, 0, 1, 1e40]}
      })
  ).toThrow(/grid.affine row edges.*finite float32/);
  graph.device.destroy();
});

it('prefixes every resource and node when two contributors share a graph', () => {
  for (const kind of ['coverage', 'categorical'] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const {coverage, categorical} = createScalabilityProps(graph);
    const identifiers = new Set<string>();
    for (const id of ['first-contributor', 'second-contributor']) {
      const resources = vi.spyOn(graph, 'createTransientBuffer');
      const contributor =
        kind === 'coverage'
          ? new GPURasterFractionalCoverage({...coverage, id})
          : new GPURasterFractionalCategoricalStatistics({...categorical, id});
      const nodes = contributor.getCommandNodes(graph);
      for (const [props] of resources.mock.calls) expect(props.id).toMatch(new RegExp(`^${id}-`));
      for (const node of nodes) {
        expect(node.id).toMatch(new RegExp(`^${id}-`));
        expect(identifiers.has(node.id)).toBe(false);
        identifiers.add(node.id);
      }
      resources.mockRestore();
    }
    graph.device.destroy();
  }
});
