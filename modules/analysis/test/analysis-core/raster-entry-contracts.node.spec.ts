// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import typescript, {
  createSourceFile,
  isExportDeclaration,
  isNamedExports,
  isTypeAliasDeclaration,
  ScriptTarget,
  SyntaxKind
} from 'typescript';
import {expect, expectTypeOf, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import * as pointcloud from '@luma-contributors/analysis/pointcloud-analysis';
import type {GPUNetworkIsochronesOutput} from '@luma-contributors/analysis/gpu-network';
import type {GPUSpatialScanStatisticOutput} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  getGPULineIntegralConvolutionParameterValues,
  getGPUStreamlinesParameterValues,
  getGPUParticleAdvectionParameterValues,
  getGPULagrangianFlowMapParameterValues,
  MAXIMUM_COST_DISTANCE_PATH_REFERENCE_CELLS,
  type GPULineIntegralConvolutionParameterSettings,
  type GPUStreamlinesParameterSettings,
  type GPUParticleAdvectionParameterSettings,
  type GPULagrangianFlowMapParameterSettings,
  type GPULineIntegralConvolutionSettings,
  type GPUStreamlinesSettings,
  type GPUParticleAdvectionSettings,
  type GPULagrangianFlowMapSettings,
  computeRasterMedoidCompositeOnCPU,
  decodeGPUIsolinesPolylines,
  getGPURasterFixedPointValues,
  prepareRasterStackInput,
  type CPUCompositeDimensions,
  type CPURasterOperationBand,
  type GPURasterPatchLabelsStatus,
  type GPUIsolinesPolylineReadback,
  type GPUIsolinesDecodedPolylines,
  type PrepareRasterStackInputProps,
  type PreparedRasterStackInput,
  type RasterStackSourceLayout
} from '@luma-contributors/analysis/raster-analysis';
import {
  MAXIMUM_GPU_SOLAR_IRRADIANCE_SAMPLE_COUNT,
  unpackGPUTerrainHorizonUnorm16,
  type GPUTerrainHorizonUnorm16Input,
  type CPUTerrainCatchmentSummaryOperationProps,
  type CPURasterOperationBand as TerrainCPURasterOperationBand
} from '@luma-contributors/analysis/gpu-terrain';
import {MAXIMUM_COST_DISTANCE_PATH_REFERENCE_CELLS as producerReferenceCells} from '../../src/raster-analysis/cost-distance/gpu-cost-distance-path';
import {MAXIMUM_GPU_SOLAR_IRRADIANCE_SAMPLE_COUNT as producerSampleCount} from '../../src/gpu-terrain/terrain-illumination/gpu-solar-irradiance';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

it('FLT-10 exposes origin-aware parameter settings while retaining legacy settings', () => {
  for (const [path, names] of [
    [
      'flow-texture',
      ['GPULineIntegralConvolutionParameterSettings', 'GPUStreamlinesParameterSettings']
    ],
    ['particle-advection', ['GPUParticleAdvectionParameterSettings']],
    ['lagrangian-diagnostics', ['GPULagrangianFlowMapParameterSettings']]
  ] as const) {
    expect(getNamedExports(`../../src/raster-analysis/${path}/index.ts`)).toEqual(
      expect.arrayContaining([...names])
    );
  }
  const placement = {origin: [100, 220] as const, cellSize: [10, -10] as const};
  const common = {fieldPlacement: placement, localOrigin: [100, 220] as const};
  const convolution: GPULineIntegralConvolutionParameterSettings = {
    ...common,
    outputPlacement: {origin: [110, 210], cellSize: [5, -5]},
    stepLength: 0.25,
    minimumSpeed: 0.5,
    phase: 0.75,
    period: 2
  };
  const streamlines: GPUStreamlinesParameterSettings = {
    ...common,
    gridPlacement: {origin: [100, 200], cellSize: [8, 4]},
    stepLength: 1,
    minimumSpeed: 0.5
  };
  const particles: GPUParticleAdvectionParameterSettings = {
    ...common,
    timeStep: 0.25,
    speedScale: 2,
    dropRate: 0.125,
    minimumSpeed: 0.5,
    spawnBounds: [101, 202, 118, 219]
  };
  const flowMap: GPULagrangianFlowMapParameterSettings = {
    ...common,
    startTime: 1700000000,
    timeStep: 0.25,
    sliceStartTime: 1700000000,
    sliceEndTime: 1700000010,
    timeOrigin: 1700000000
  };
  // Subtract [100,220] in binary64: field [0,0], output [10,-10], grid [0,-20],
  // spawn [1,-18,18,-1]. Grid reciprocal cells 1/8,1/4 are exactly representable.
  const convolutionPacket = getGPULineIntegralConvolutionParameterValues(convolution);
  expect([...convolutionPacket]).toEqual([0, 0, 10, -10, 10, -10, 5, -5, 0.25, 0.5, 0.75, 2]);
  expect([...getGPUStreamlinesParameterValues(streamlines)]).toEqual([
    0, 0, 10, -10, 0, -20, 8, 4, 0.125, 0.25, 1, 0.5
  ]);
  expect([...getGPUParticleAdvectionParameterValues(particles, [2, 2])]).toEqual([
    0, 0, 10, -10, 0.25, 2, 0.125, 0.5, 1, -18, 18, -1
  ]);
  expect([...getGPULagrangianFlowMapParameterValues(flowMap)]).toEqual([
    0, 0, 10, -10, 0, 0.25, 0, 10
  ]);
  type Placement = {origin: readonly [number, number]; cellSize: readonly [number, number]};
  expectTypeOf<GPULineIntegralConvolutionSettings['fieldPlacement']>().toEqualTypeOf<Placement>();
  expectTypeOf<GPUStreamlinesSettings['gridPlacement']>().toEqualTypeOf<Placement>();
  expectTypeOf<GPUParticleAdvectionSettings['fieldPlacement']>().toEqualTypeOf<Placement>();
  expectTypeOf<GPULagrangianFlowMapSettings['fieldPlacement']>().toEqualTypeOf<Placement>();
});

it('W5-06 exposes the existing cost-path and solar producer route bounds', () => {
  expect
    .soft(
      getNamedReExportSource(
        '../../src/raster-analysis/cost-distance/index.ts',
        'MAXIMUM_COST_DISTANCE_PATH_REFERENCE_CELLS'
      )
    )
    .toBe('./gpu-cost-distance-path');
  expect
    .soft(
      getNamedReExportSource(
        '../../src/gpu-terrain/terrain-illumination/index.ts',
        'MAXIMUM_GPU_SOLAR_IRRADIANCE_SAMPLE_COUNT'
      )
    )
    .toBe('./gpu-solar-irradiance');
  expect(MAXIMUM_COST_DISTANCE_PATH_REFERENCE_CELLS).toBe(256);
  expect(MAXIMUM_COST_DISTANCE_PATH_REFERENCE_CELLS).toBe(producerReferenceCells);
  expect(MAXIMUM_GPU_SOLAR_IRRADIANCE_SAMPLE_COUNT).toBe(1073741824);
  expect(MAXIMUM_GPU_SOLAR_IRRADIANCE_SAMPLE_COUNT).toBe(producerSampleCount);
});

function getNamedReExportSource(relativePath: string, name: string): string | undefined {
  const source = createSourceFile(
    relativePath,
    readFileSync(new URL(relativePath, import.meta.url), 'utf8'),
    ScriptTarget.Latest
  );
  for (const statement of source.statements) {
    if (
      isExportDeclaration(statement) &&
      statement.exportClause &&
      isNamedExports(statement.exportClause) &&
      statement.moduleSpecifier &&
      typescript.isStringLiteral(statement.moduleSpecifier) &&
      statement.exportClause.elements.some(
        element =>
          !element.isTypeOnly &&
          (element.propertyName ?? element.name).text === name &&
          element.name.text === name
      )
    ) {
      return statement.moduleSpecifier.text;
    }
  }
  return undefined;
}

/** Padded, deliberately unaligned readback, with unrelated bytes on both sides. */
function createReadbackBytes(values: Float32Array | Uint32Array): Uint8Array {
  const allocation = new Uint8Array(values.byteLength + 5).fill(0xff);
  allocation.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength), 1);
  return allocation.subarray(1, 1 + values.byteLength);
}

it('UNI-01 distinguishes the any-point coverage bit from ground coverage', () => {
  const coverage = Uint32Array.of(0, 1, 3);
  expect(
    [...coverage].map(value =>
      Number((value & pointcloud.GPU_POINT_CLOUD_COVERAGE_ANY_POINT) !== 0)
    )
  ).toEqual([0, 1, 1]);
  expect(
    [...coverage].map(value => Number((value & pointcloud.GPU_POINT_CLOUD_COVERAGE_GROUND) !== 0))
  ).toEqual([0, 0, 1]);
});

it('UNI-01 exposes the delivered coverage conversion and its types through the package entry', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const output: pointcloud.GPUPointCloudCoverageValidityOutput = {
      surfaceValidity: createTransientView(graph, 'surface-validity', 'uint32', 2),
      terrainValidity: createTransientView(graph, 'terrain-validity', 'uint32', 2),
      canopyValidity: createTransientView(graph, 'canopy-validity', 'uint32', 2)
    };
    const props: pointcloud.GPUPointCloudCoverageValidityProps = {
      coverage: createTransientView(graph, 'coverage', 'uint32', 2),
      output
    };
    const converter = new pointcloud.GPUPointCloudCoverageValidity(props);
    expect(converter.getCommandNodes(graph)).toHaveLength(1);
    expect(converter.props.output.terrainValidity).toBe(output.terrainValidity);
  } finally {
    device.destroy();
  }
});

it('UNI-04 exposes point-cloud grid alignment with non-unit cells through the package entry', () => {
  const grid = {width: 2, height: 2, cellSize: 10, origin: [100, 200] as const};
  const metadata = pointcloud.getRasterMetadataFromPointCloudGrid(grid);
  expect(metadata.affine).toEqual([10, 0, 100, 0, -10, 220]);
  const centres = Array.from({length: 4}, (_, index) => {
    const column = (index % 2) + 0.5;
    const row = Math.floor(index / 2) + 0.5;
    const [stepX, skewX, originX, skewY, stepY, originY] = metadata.affine;
    return [originX + column * stepX + row * skewX, originY + column * skewY + row * stepY];
  });
  // Hand-worked 100..120 by 200..220 footprint: centres are five units inside each edge.
  expect(centres).toEqual([
    [105, 215],
    [115, 215],
    [105, 205],
    [115, 205]
  ]);
  expect(
    pointcloud.getRasterMetadataFromPointCloudGrid({...grid, rowDirection: 'north'}).affine
  ).toEqual([10, 0, 100, 0, 10, 200]);
});

it('EGR-13 publicly decodes retained geometry while preserving loss from byte readbacks', () => {
  const readback: GPUIsolinesPolylineReadback = {
    polylines: {
      vertices: createReadbackBytes(Float32Array.of(1, 2.5, 1, 1.5, 1, 0.5, 99, 99)),
      polylineOffsets: createReadbackBytes(Uint32Array.of(0, 3, 999)),
      polylineLevels: createReadbackBytes(Uint32Array.of(0, 999)),
      polylineClosed: createReadbackBytes(Uint32Array.of(0, 999)),
      polylineCount: createReadbackBytes(Uint32Array.of(1)),
      vertexCount: createReadbackBytes(Uint32Array.of(3))
    },
    status: {
      overflow: createReadbackBytes(Uint32Array.of(0)),
      polylineOverflow: createReadbackBytes(Uint32Array.of(1))
    }
  };
  const result: GPUIsolinesDecodedPolylines = decodeGPUIsolinesPolylines(readback, {
    levels: createReadbackBytes(Float32Array.of(0.5)),
    localOrigin: [100, 200]
  });
  // The x=1 contour at level 0.5 is translated by the local origin; spare rows are ignored.
  expect(result).toEqual({
    polylines: [
      {
        levelIndex: 0,
        levelValue: 0.5,
        closed: false,
        coordinates: [
          [101, 202.5],
          [101, 201.5],
          [101, 200.5]
        ]
      }
    ],
    segmentComplete: true,
    polylineComplete: false,
    complete: false
  });
  expect(() =>
    decodeGPUIsolinesPolylines(
      {
        ...readback,
        status: {...readback.status, polylineOverflow: new Uint8Array(3)}
      },
      {levels: Float32Array.of(0.5)}
    )
  ).toThrow(/multiple of 4/);
});

it('ING-17 keeps the existing star-export route for checked acquisition packing and its types', () => {
  const sourceLayout: RasterStackSourceLayout = 'acquisition-major';
  const props: PrepareRasterStackInputProps = {
    grid: {
      width: 2,
      height: 1,
      affine: [10, 0, 100, 0, -10, 220],
      pixelInterpretation: 'area'
    },
    bandCount: 1,
    sourceLayout,
    slices: [Float32Array.of(0, 10), Float32Array.of(100, 20), Float32Array.of(200, 30)],
    maskLayout: sourceLayout,
    mask: [Uint32Array.of(0, 2), Uint32Array.of(3, 0), Uint32Array.of(0, 0)]
  };
  const prepared: PreparedRasterStackInput = prepareRasterStackInput(props);
  expect([...prepared.values]).toEqual([0, 100, 200, 10, 20, 30]);
  expect(prepared.mask && [...prepared.mask]).toEqual([0, 3, 0, 2, 0, 0]);
  expect([prepared.cellCount, prepared.sliceCount, prepared.bandCount]).toEqual([2, 3, 1]);
  const dimensions: CPUCompositeDimensions = prepared;
  // Cell 0 retains only slice 1 (100), cell 1 only slice 0 (10). Nonzero masks 2 and 3
  // are usable; dropping the mask would instead return the unmasked medoids [100, 20].
  expect([
    ...computeRasterMedoidCompositeOnCPU(prepared.values, dimensions, prepared.mask).composite
  ]).toEqual([100, 10]);
});

it('B1 keeps the horizon input type resolvable in an ESM declaration after the production transform', () => {
  const source = createSourceFile(
    'index.d.ts',
    readFileSync(
      new URL('../../src/gpu-terrain/terrain-illumination/index.ts', import.meta.url),
      'utf8'
    ),
    ScriptTarget.Latest
  );
  const configuration = typescript.parseConfigFileTextToJson(
    'tsconfig.json',
    readFileSync(new URL('../../../../tsconfig.json', import.meta.url), 'utf8')
  ).config;
  const appendExtension = createRequire(import.meta.url)(
    '@vis.gl/ts-plugins/ts-transform-append-extension'
  ).default;
  // This transform does not use its Program argument. Parse/transform only; no compiler program,
  // emit, build or typecheck is launched. ImportTypeNode is unchanged by the production plugin.
  const transformed = typescript.transform(source, [
    appendExtension(
      undefined,
      configuration.compilerOptions.plugins.find(
        (plugin: {afterDeclarations?: boolean}) => plugin.afterDeclarations
      ),
      {ts: typescript}
    )
  ]);
  try {
    const imports: string[] = [];
    const visit = (node: typescript.Node) => {
      if (
        typescript.isImportTypeNode(node) &&
        typescript.isLiteralTypeNode(node.argument) &&
        typescript.isStringLiteral(node.argument.literal)
      ) {
        imports.push(node.argument.literal.text);
      }
      typescript.forEachChild(node, visit);
    };
    visit(transformed.transformed[0]);
    expect(imports).toHaveLength(1);
    const resolved = imports.map(
      specifier =>
        typescript.resolveModuleName(
          specifier,
          '/packed/index.d.ts',
          {
            module: typescript.ModuleKind.NodeNext,
            moduleResolution: typescript.ModuleResolutionKind.NodeNext
          },
          {
            fileExists: path => path === '/packed/gpu-terrain-horizon.d.ts',
            readFile: () => undefined
          },
          undefined,
          undefined,
          typescript.ModuleKind.ESNext
        ).resolvedModule?.resolvedFileName
    );
    expect(resolved).toEqual(['/packed/gpu-terrain-horizon.d.ts']);
  } finally {
    transformed.dispose();
  }
});

it('EGR-01 publicly reads bounded horizon and streaming-accumulator bytes', () => {
  expect(getNamedExports('../../src/gpu-terrain/terrain-illumination/index.ts')).toContain(
    'GPUTerrainHorizonUnorm16Input'
  );
  const input: GPUTerrainHorizonUnorm16Input = {
    bytes: createReadbackBytes(Uint32Array.of(0xffff0001))
  };
  expect([...unpackGPUTerrainHorizonUnorm16(input, 2)]).toEqual([-90, 90]);
  // Low half 1 maps to -90 degrees; high half 65535 maps to +90 degrees.
  expect([
    ...unpackGPUTerrainHorizonUnorm16(createReadbackBytes(Uint32Array.of(0xffff0001)), 2)
  ]).toEqual([-90, 90]);
  expect([...unpackGPUTerrainHorizonUnorm16(Uint32Array.of(0xffff0001).buffer, 2)]).toEqual([
    -90, 90
  ]);
  // Three little-endian words represent the 96-bit integer 1; exponent 0 leaves it unchanged.
  expect([
    ...getGPURasterFixedPointValues(createReadbackBytes(Uint32Array.of(1, 0, 0)), 0)
  ]).toEqual([1]);
  expect(() => unpackGPUTerrainHorizonUnorm16(new Uint8Array(3), 2)).toThrow(/whole words/);
  expect(() => getGPURasterFixedPointValues(new Uint8Array(11), 0)).toThrow();
});

// Runtime tests erase import type declarations. Inspect explicit barrel bindings as well, so a
// missing re-export fails this focused Node run before the lead's final test typecheck.
function getNamedExports(relativePath: string): string[] {
  const source = createSourceFile(
    relativePath,
    readFileSync(new URL(relativePath, import.meta.url), 'utf8'),
    ScriptTarget.Latest
  );
  return source.statements.flatMap(statement => {
    if (
      isTypeAliasDeclaration(statement) &&
      statement.modifiers?.some(modifier => modifier.kind === SyntaxKind.ExportKeyword)
    ) {
      return [statement.name.text];
    }
    return isExportDeclaration(statement) &&
      statement.exportClause &&
      isNamedExports(statement.exportClause)
      ? statement.exportClause.elements.map(element => element.name.text)
      : [];
  });
}

it('ENT-07 and TYP-16 expose the CPU catchment props and raster label status contracts', () => {
  expect(getNamedExports('../../src/gpu-terrain/terrain-processing/index.ts')).toContain(
    'CPUTerrainCatchmentSummaryOperationProps'
  );
  expect(getNamedExports('../../src/raster-analysis/raster-patches/index.ts')).toContain(
    'GPURasterPatchLabelsStatus'
  );
  expectTypeOf<CPUTerrainCatchmentSummaryOperationProps>().toHaveProperty('output');
  expectTypeOf<GPURasterPatchLabelsStatus>().toHaveProperty('converged');
  expect(getNamedExports('../../src/gpu-network/network-isochrones/index.ts')).toContain(
    'GPUNetworkIsochronesOutput'
  );
  expect(getNamedExports('../../src/gpu-spatial-analysis/scan-statistics/index.ts')).toContain(
    'GPUSpatialScanStatisticOutput'
  );
  expectTypeOf<GPUNetworkIsochronesOutput>().toHaveProperty('status');
  expectTypeOf<GPUSpatialScanStatisticOutput>().toHaveProperty('clusterIndices');
});

it('ENT-08 exports shared raster input and composite dimensions types from their owning entries', () => {
  expect(getNamedExports('../../src/raster-analysis/raster-grid/index.ts')).toContain(
    'CPURasterOperationBand'
  );
  expect(getNamedExports('../../src/gpu-terrain/terrain-processing/index.ts')).toContain(
    'CPURasterOperationBand'
  );
  expect(getNamedExports('../../src/gpu-terrain/index.ts')).not.toContain('CPURasterOperationBand');
  const terrainEntry = createSourceFile(
    'index.ts',
    readFileSync(new URL('../../src/gpu-terrain/index.ts', import.meta.url), 'utf8'),
    ScriptTarget.Latest
  );
  for (const statement of terrainEntry.statements) {
    if (
      isExportDeclaration(statement) &&
      statement.exportClause &&
      statement.moduleSpecifier &&
      typescript.isStringLiteral(statement.moduleSpecifier)
    ) {
      // Match the existing analysis-layering contract: explicit exports are local top-level
      // declarations or analysis-core; every family subdirectory is forwarded whole.
      expect(statement.moduleSpecifier.text).toMatch(/^\.\/[a-z0-9-]+$|^\.\.\/analysis-core\//);
    }
  }
  expect(getNamedExports('../../src/raster-analysis/raster-time-series/index.ts')).toContain(
    'CPUCompositeDimensions'
  );
  expectTypeOf<CPURasterOperationBand>().toEqualTypeOf<TerrainCPURasterOperationBand>();
  expectTypeOf<CPUCompositeDimensions>().toEqualTypeOf<{
    cellCount: number;
    sliceCount: number;
    bandCount: number;
  }>();
});
