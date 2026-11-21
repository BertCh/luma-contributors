// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {getBenchmarkSettings, getBenchmarkDistribution} from '../../utils/benchmark-settings';
import {GPUDatumGridTransform} from '../../../src/geospatial-analysis/datum-grid-transform/index';
import {GPUS2CellHierarchy} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-hierarchy';
import {GPUS2CapCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering';
import {GPUSphericalConvexIntersection} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-intersection';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GPUS2CellMeasures} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-measures';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {GPUSphericalArcOverlay} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay';
import reference from './fixtures/global-coordinates.json';
import orderedReference from './fixtures/s2-covering-ordered-parity.json';
import {
  ARC_SCALING_CASES,
  getArcScalingOperand,
  DATUM_SCALING_GRID_COUNTS,
  getDatumScalingGrids
} from './global-coordinate-scaling-fixtures';
import {
  expectArcScalingOutput,
  expectDatumGridCountPositions,
  expectDatumSelectionPositions
} from './global-coordinate-output-assertions';

type Format = 'float32x2' | 'float32x3' | 'float32x4' | 'uint32' | 'uint32x2';
type Values = Float32Array | Uint32Array;
type Output = Record<string, number[]>;
type Write = {buffer: Buffer; values: Values};
const environment = (import.meta as unknown as {env?: Record<string, string | undefined>}).env;
const RUN_BENCHMARK = environment?.['VITE_GLOBAL_COORDINATES_BENCHMARK'] === 'true';
const getComponents = (format: Format) =>
  format.endsWith('x4') ? 4 : format.endsWith('x3') ? 3 : format.endsWith('x2') ? 2 : 1;

/** Same phase separation as the geospatial backlog harness; no pipeline cache clearing. */
class GlobalBenchmarkScope {
  readonly graph: GPUCommandGraph;
  readonly buffers: Buffer[] = [];
  readonly initialWrites: Write[] = [];
  readonly updateWrites: Write[] = [];
  readonly outputs: {name: string; buffer: Buffer; format: Format; length: number}[] = [];
  declarationMilliseconds: number | null = null;
  constructor(
    readonly device: Device,
    readonly name: string
  ) {
    this.graph = new GPUCommandGraph(device, {id: name});
  }
  allocate(length: number) {
    const buffer = this.device.createBuffer({
      byteLength: Math.max(length, 4),
      usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
    });
    this.buffers.push(buffer);
    return buffer;
  }
  /** Expand a contributor into command nodes, excluding argument evaluation and GPU compile. */
  declare(contributor: GPUCommandNodeProducer) {
    const start = performance.now();
    this.graph.add(contributor);
    this.declarationMilliseconds = (this.declarationMilliseconds ?? 0) + performance.now() - start;
  }
  input<DataFormat extends Format>(
    name: string,
    values: Values,
    format: DataFormat,
    updated?: Values
  ) {
    const buffer = this.allocate(values.byteLength);
    this.initialWrites.push({buffer, values});
    if (updated) this.updateWrites.push({buffer, values: updated});
    return importGraphBuffer(
      this.graph,
      `input-${name}`,
      buffer,
      format,
      values.length / getComponents(format)
    );
  }
  output<DataFormat extends Format>(name: string, rows: number, format: DataFormat) {
    const length = rows * getComponents(format);
    const buffer = this.allocate(length * 4);
    this.outputs.push({name, buffer, length, format});
    return importGraphBuffer(this.graph, `output-${name}`, buffer, format, rows);
  }
  async read(): Promise<Output> {
    const output: Output = {};
    for (const field of this.outputs)
      output[field.name] = await (field.format.startsWith('float') ? readFloat32 : readUint32)(
        field.buffer,
        field.length
      );
    return output;
  }
  destroy() {
    this.buffers.forEach(buffer => buffer.destroy());
  }
}

type Workload = {
  name: string;
  shape: Record<string, number | string>;
  updateKind: string;
  construct: (scope: GlobalBenchmarkScope) => (output: Output, updated: boolean) => void;
};

function createWorkloads(): Workload[] {
  const neighbors = [166, 16384].map(rows => ({
    name: `s2-neighbors-${rows}`,
    shape: {rows, levels: '0,1,2,15,30', distinctOracleCells: reference.cells.length},
    updateKind: 'reverse the pinned exact-key corpus; same graph',
    construct(scope: GlobalBenchmarkScope) {
      const identifiers = (updated: boolean) =>
        Array.from(
          {length: rows},
          (_, row) =>
            reference.cells[
              updated
                ? reference.cells.length - 1 - (row % reference.cells.length)
                : row % reference.cells.length
            ]
        );
      scope.graph.add(
        new GPUS2CellHierarchy({
          cells: scope.input(
            'cells',
            Uint32Array.from(identifiers(false).flatMap(row => row.cell)),
            'uint32x2',
            Uint32Array.from(identifiers(true).flatMap(row => row.cell))
          ),
          operation: 'neighbors',
          output: {
            cells: scope.output('cells', rows * 4, 'uint32x2'),
            status: scope.output('status', rows, 'uint32')
          }
        })
      );
      return (output: Output, updated: boolean) => {
        expect(output['status']).toEqual(new Array(rows).fill(0));
        for (const [row, expected] of identifiers(updated).entries()) {
          const actual = Array.from({length: 4}, (_, slot) =>
            output['cells'].slice(row * 8 + slot * 2, row * 8 + slot * 2 + 2).join(':')
          ).sort();
          expect(actual).toEqual(expected.neighbors.map(key => key.join(':')).sort());
        }
      };
    }
  }));
  const datum = [256, 16384].map(rows => ({
    name: `datum-grid-${rows}`,
    shape: {rows, gridWidth: 2, gridHeight: 2},
    updateKind: 'replace resident signed xyz grid shifts; same positions and graph',
    construct(scope: GlobalBenchmarkScope) {
      const positions = Float32Array.from(
        Array.from({length: rows}, (_, row) => [
          (row % 101) / 100,
          (row % 83) / 82,
          100 + (row % 7)
        ]).flat()
      );
      scope.graph.add(
        new GPUDatumGridTransform({
          positions: scope.input('positions', positions, 'float32x3'),
          samples: scope.input(
            'samples',
            Float32Array.from(Array(4).fill([0.1, 0.2, -10]).flat()),
            'float32x3',
            Float32Array.from(Array(4).fill([-0.1, -0.2, 20]).flat())
          ),
          mask: scope.input('validity', new Uint32Array(4).fill(1), 'uint32'),
          grids: [{width: 2, height: 2, sampleOffset: 0, origin: [0, 0], spacing: [1, 1]}],
          operation: 'combined',
          sourceVerticalReference: 'ellipsoid',
          targetVerticalReference: 'orthometric',
          output: {
            positions: scope.output('positions', rows, 'float32x3'),
            status: scope.output('status', rows, 'uint32')
          }
        })
      );
      return (output: Output, updated: boolean) => {
        expect(output['status']).toEqual(new Array(rows).fill(0));
        const shift = updated ? [-0.1, -0.2, 20] : [0.1, 0.2, -10];
        let maximumAbsoluteError = 0;
        for (let index = 0; index < positions.length; index++)
          maximumAbsoluteError = Math.max(
            maximumAbsoluteError,
            Math.abs(output['positions'][index] - positions[index] - shift[index % 3])
          );
        expect(maximumAbsoluteError).toBeLessThan(2e-5);
      };
    }
  }));
  const caps = [1, 32].map(rows => ({
    name: `s2-cap-cover-${rows}`,
    shape: {rows, level: 4, candidateCellsPerCap: 1536},
    updateKind: 'increase resident cap radius0.17 to0.31 radians; same graph and capacity',
    construct(scope: GlobalBenchmarkScope) {
      const capacity = 1536;
      const caps = (radius: number) =>
        Float32Array.from(Array(rows).fill([1, 0, 0, radius]).flat());
      scope.graph.add(
        new GPUS2CapCovering({
          caps: scope.input(
            'caps',
            caps(reference.covering.initialRadius),
            'float32x4',
            caps(reference.covering.updatedRadius)
          ),
          level: 4,
          capacity,
          output: {
            cells: scope.output('cells', rows * capacity, 'uint32x2'),
            counts: scope.output('counts', rows, 'uint32x2'),
            status: scope.output('status', rows, 'uint32')
          }
        })
      );
      return (output: Output, updated: boolean) => {
        const expected = updated
          ? reference.covering.updatedCells
          : reference.covering.initialCells;
        expect(output['status']).toEqual(new Array(rows).fill(0));
        expect(output['counts']).toEqual(
          Array(rows).fill([expected.length, expected.length]).flat()
        );
        for (let row = 0; row < rows; row++) {
          expect(
            output['cells'].slice(row * capacity * 2, row * capacity * 2 + expected.length * 2)
          ).toEqual(expected.flat());
          expect(
            output['cells']
              .slice(row * capacity * 2 + expected.length * 2, (row + 1) * capacity * 2)
              .every(word => word === 0)
          ).toBe(true);
        }
      };
    }
  }));
  const overlap = [1, 256].map(rows => ({
    name: `spherical-overlap-${rows}`,
    shape: {rows, verticesPerInput: 4, capacityPerRow: 16},
    updateKind:
      'replace second polygons with first polygons; same graph; intersection changes8 to4 vertices',
    construct(scope: GlobalBenchmarkScope) {
      const repeated = (positions: number[][]) =>
        Float32Array.from(Array(rows).fill(positions.flat()).flat());
      const offsets = Uint32Array.from({length: rows + 1}, (_, row) => row * 4);
      scope.graph.add(
        new GPUSphericalConvexIntersection({
          firstPositions: scope.input(
            'firstPositions',
            repeated(reference.overlap.first),
            'float32x3'
          ),
          secondPositions: scope.input(
            'secondPositions',
            repeated(reference.overlap.second),
            'float32x3',
            repeated(reference.overlap.first)
          ),
          firstOffsets: scope.input('firstOffsets', offsets, 'uint32'),
          secondOffsets: scope.input('secondOffsets', offsets, 'uint32'),
          capacity: 16,
          output: {
            positions: scope.output('positions', rows * 16, 'float32x3'),
            counts: scope.output('counts', rows, 'uint32x2'),
            status: scope.output('status', rows, 'uint32'),
            measures: scope.output('measures', rows, 'float32x4')
          }
        })
      );
      return (output: Output, updated: boolean) => {
        const expected = updated ? reference.overlap.first : reference.overlap.intersection;
        const firstArea = reference.overlap.area / reference.overlap.sourceFraction;
        const secondArea = reference.overlap.area / reference.overlap.targetFraction;
        const measures = updated
          ? [firstArea, 1, 1, firstArea]
          : [
              reference.overlap.area,
              reference.overlap.sourceFraction,
              reference.overlap.targetFraction,
              firstArea + secondArea - reference.overlap.area
            ];
        expect(output['status']).toEqual(new Array(rows).fill(0));
        expect(output['counts']).toEqual(
          Array(rows).fill([expected.length, expected.length]).flat()
        );
        let maximumError = 0;
        for (let row = 0; row < rows; row++) {
          for (let component = 0; component < 4; component++)
            maximumError = Math.max(
              maximumError,
              Math.abs(output['measures'][row * 4 + component] - measures[component])
            );
          for (let vertex = 0; vertex < expected.length; vertex++) {
            const actual = output['positions'].slice(
              (row * 16 + vertex) * 3,
              (row * 16 + vertex + 1) * 3
            );
            maximumError = Math.max(
              maximumError,
              Math.min(
                ...expected.map(point =>
                  Math.hypot(...point.map((value, component) => value - actual[component]))
                )
              )
            );
          }
          expect(
            output['positions']
              .slice((row * 16 + expected.length) * 3, (row + 1) * 48)
              .every(value => value === 0)
          ).toBe(true);
        }
        expect(maximumError).toBeLessThan(2e-5);
      };
    }
  }));
  const fineMeasures: Workload[] = [166, 16384].map(rows => ({
    name: `s2-stable-measures-${rows}`,
    shape: {rows, maximumLevel: 30},
    updateKind: 'reverse exact-key oracle rows',
    construct(scope) {
      const indices = (updated: boolean) =>
        Array.from(
          {length: rows},
          (_, index) =>
            reference.cells[
              updated
                ? reference.cells.length - 1 - (index % reference.cells.length)
                : index % reference.cells.length
            ]
        );
      scope.graph.add(
        new GPUS2CellMeasures({
          cells: scope.input(
            'cells',
            Uint32Array.from(indices(false).flatMap(row => row.cell)),
            'uint32x2',
            Uint32Array.from(indices(true).flatMap(row => row.cell))
          ),
          output: {
            measures: scope.output('measures', rows, 'float32x2'),
            status: scope.output('status', rows, 'uint32')
          }
        })
      );
      return (output, updated) => {
        expect(output['status'].every(value => value === 0)).toBe(true);
        for (const [index, row] of indices(updated).entries()) {
          expect(Math.abs(output['measures'][index * 2] - row.area) / row.area).toBeLessThan(2e-5);
          expect(
            Math.abs(output['measures'][index * 2 + 1] - row.perimeter) / row.perimeter
          ).toBeLessThan(2e-5);
        }
      };
    }
  }));
  const adaptiveCovers: Workload[] = [4, 6].map(maximumLevel => ({
    name: `s2-adaptive-cover-level${maximumLevel}`,
    shape: {maximumLevel, inputVertices: 3, capacity: 4096},
    updateKind: 'rotate octant across a cube face',
    construct(scope) {
      const vertices = scope.input(
        'vertices',
        Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]),
        'float32x3',
        Float32Array.from([0, 1, 0, -1, 0, 0, 0, 0, 1])
      );
      const cells = scope.output('cells', 4096, 'uint32x2');
      scope.graph.add(
        new GPUS2PolygonCovering({
          positions: vertices,
          ringOffsets: scope.input('rings', Uint32Array.from([0, 3]), 'uint32'),
          maximumLevel,
          capacity: 4096,
          output: {
            cells,
            counts: scope.output('counts', 1, 'uint32x2'),
            status: scope.output('status', 1, 'uint32'),
            visitedCount: scope.output('visited', 1, 'uint32')
          }
        })
      );
      scope.graph.add(
        new GPUS2CellMeasures({
          cells,
          output: {
            measures: scope.output('measures', 4096, 'float32x2'),
            status: scope.output('measureStatus', 4096, 'uint32')
          }
        })
      );
      return (output, updated) => {
        const expected = orderedReference.cases.find(
          value => value.name === `benchmark-level${maximumLevel}`
        )!.snapshots[updated ? 1 : 0];
        expect(output['cells'].slice(0, expected.cells.length)).toEqual(expected.cells);
        expect(output['cells'].slice(expected.cells.length).every(value => value === 0)).toBe(true);
        expect(output['counts']).toEqual(expected.counts);
        expect(output['status']).toEqual(expected.status);
        expect(output['visited']).toEqual(expected.visited);
        expect(output['status']).toEqual([0]);
        expect(output['counts'][0]).toBe(output['counts'][1]);
        expect(output['counts'][0]).toBeGreaterThan(0);
        expect(output['visited'][0]).toBeLessThan(6 * 4 ** maximumLevel);
        let area = 0;
        for (let row = 0; row < output['counts'][0]; row++) {
          expect(output['measureStatus'][row]).toBe(0);
          area += output['measures'][row * 2];
        }
        expect(area).toBeGreaterThanOrEqual(Math.PI / 2 - 1e-5);
      };
    }
  }));
  const intrinsicOverlay: Workload[] = ARC_SCALING_CASES.map(reference => ({
    name: `spherical-intrinsic-overlay-${reference.totalVertices}`,
    shape: {
      inputVertices: reference.totalVertices,
      verticesPerOperand: reference.totalVertices / 2,
      capacity: 256
    },
    updateKind: 'rotate both spherical regions about the north reference',
    construct(scope) {
      const expand = (vertices: number[][], updated: boolean) =>
        vertices
          .map(point =>
            updated
              ? [
                  point[0] * Math.cos(0.1) - point[1] * Math.sin(0.1),
                  point[0] * Math.sin(0.1) + point[1] * Math.cos(0.1),
                  point[2]
                ]
              : point
          )
          .flat();
      const operand = (name: string, vertices: number[][]) => ({
        positions: scope.input(
          name,
          Float32Array.from(expand(vertices, false)),
          'float32x3',
          Float32Array.from(expand(vertices, true))
        ),
        ringOffsets: scope.input(name + 'Rings', Uint32Array.from([0, vertices.length]), 'uint32'),
        referenceInside: true
      });
      scope.declare(
        new GPUSphericalArcOverlay({
          left: operand('left', getArcScalingOperand(reference.totalVertices, false)),
          right: operand('right', getArcScalingOperand(reference.totalVertices, true)),
          referencePoint: [0, 0, 1],
          operation: 'intersection',
          output: {
            positions: scope.output('positions', 256, 'float32x3'),
            ringOffsets: scope.output('rings', 65, 'uint32'),
            counts: scope.output('counts', 4, 'uint32'),
            status: scope.output('status', 1, 'uint32')
          }
        })
      );
      return (output, updated) => expectArcScalingOutput(output, reference, updated);
    }
  }));
  const datumGridCounts: Workload[] = DATUM_SCALING_GRID_COUNTS.map(gridCount => ({
    name: `datum-grid-count-${gridCount}`,
    shape: {gridCount, sampleCount: 4, queryCount: 4},
    updateKind: 'move queries from the last covering grid to no coverage; all metadata is fixed',
    construct(scope) {
      const positions = (covered: boolean) =>
        Array.from({length: 4}, () => [covered ? gridCount + 9.5 : 0, 0.5, 10]).flat();
      scope.declare(
        new GPUDatumGridTransform({
          positions: scope.input(
            'positions',
            Float32Array.from(positions(true)),
            'float32x3',
            Float32Array.from(positions(false))
          ),
          samples: scope.input(
            'samples',
            Float32Array.from(Array(4).fill([0.25, -0.125, 2]).flat()),
            'float32x3'
          ),
          mask: scope.input('mask', new Uint32Array(4).fill(1), 'uint32'),
          grids: getDatumScalingGrids(gridCount),
          operation: 'horizontal',
          output: {
            positions: scope.output('positions', 4, 'float32x3'),
            status: scope.output('status', 4, 'uint32')
          }
        })
      );
      return (output, updated) => {
        expect(output['status']).toEqual(new Array(4).fill(updated ? 2 : 0));
        expectDatumGridCountPositions(output['positions'], gridCount, updated);
      };
    }
  }));
  // Large ordered/nodata fixtures are opt-in compilation workloads, not ordinary correctness runs.
  const datumSelection: Workload[] = DATUM_SCALING_GRID_COUNTS.flatMap(gridCount =>
    [false, true].flatMap(reverse =>
      [false, true].map(nodata => ({
        name: `datum-grid-selection-${gridCount}-reverse-${reverse}-nodata-${nodata}`,
        shape: {gridCount, sampleCount: 8, queryCount: 2},
        updateKind:
          'toggle first-covering-grid nodata, retaining selection order and no-coverage row',
        construct(scope: GlobalBenchmarkScope) {
          const covering = [0, 4].map(sampleOffset => ({
            width: 2,
            height: 2,
            sampleOffset,
            origin: [0, 0] as const,
            spacing: [1, 1] as const
          }));
          const mask = (invalid: boolean) => {
            const values = new Uint32Array(8).fill(1);
            if (invalid) values[reverse && gridCount > 1 ? 4 : 0] = 0;
            return values;
          };
          scope.declare(
            new GPUDatumGridTransform({
              positions: scope.input(
                'positions',
                Float32Array.from([0.5, 0.5, 10, -2, -2, 10]),
                'float32x3'
              ),
              samples: scope.input(
                'samples',
                Float32Array.from([
                  ...Array(4).fill([0.25, -0.125, 2]).flat(),
                  ...Array(4).fill([-0.5, 0.5, 9]).flat()
                ]),
                'float32x3'
              ),
              mask: scope.input('mask', mask(nodata), 'uint32', mask(!nodata)),
              grids: [
                ...getDatumScalingGrids(Math.max(0, gridCount - 2)),
                ...(gridCount === 1
                  ? covering.slice(0, 1)
                  : reverse
                    ? covering.reverse()
                    : covering)
              ],
              operation: 'horizontal',
              output: {
                positions: scope.output('positions', 2, 'float32x3'),
                status: scope.output('status', 2, 'uint32')
              }
            })
          );
          return (output: Output, updated: boolean) => {
            const invalid = updated ? !nodata : nodata;
            expect(output['status']).toEqual([invalid ? 4 : 0, 2]);
            expectDatumSelectionPositions(output['positions'], reverse && gridCount > 1, invalid);
          };
        }
      }))
    )
  );
  return [
    ...neighbors,
    ...datum,
    ...caps,
    ...overlap,
    ...fineMeasures,
    ...adaptiveCovers,
    ...intrinsicOverlay,
    ...datumGridCounts,
    ...datumSelection
  ];
}

async function waitForQueue(device: Device) {
  await (device.handle as GPUDevice).queue.onSubmittedWorkDone();
}
async function measure(
  device: Device,
  workload: Workload,
  settings: ReturnType<typeof getBenchmarkSettings>
) {
  await waitForQueue(device);
  const scope = new GlobalBenchmarkScope(device, workload.name);
  let compiled: CompiledGPUCommandGraph | undefined;
  const setupStart = performance.now();
  try {
    const check = workload.construct(scope);
    const setupMilliseconds = performance.now() - setupStart;
    const upload = async (writes: Write[]) => {
      const start = performance.now();
      writes.forEach(write => write.buffer.write(write.values));
      await waitForQueue(device);
      return performance.now() - start;
    };
    const uploadMilliseconds = await upload(scope.initialWrites);
    const compileStart = performance.now();
    compiled = scope.graph.compile();
    const compileHostMilliseconds = performance.now() - compileStart;
    const run = async () => {
      const start = performance.now();
      const encoding = submitGraph(device, compiled!, undefined);
      await waitForQueue(device);
      const encodeSubmitCompletionMilliseconds = performance.now() - start;
      const timings = encoding.canReadGPUTimings ? await encoding.readTimings() : undefined;
      return {
        encodeSubmitCompletionMilliseconds,
        cpuEncodeMilliseconds: encoding.stats.cpuEncodeTimeMilliseconds,
        gpuTimestampMilliseconds: timings?.gpuTimeMilliseconds ?? null
      };
    };
    const validate = async (updated: boolean) => {
      const readStart = performance.now();
      const output = await scope.read();
      const readbackMilliseconds = performance.now() - readStart;
      const checkStart = performance.now();
      check(output, updated);
      return {readbackMilliseconds, referenceCheckMilliseconds: performance.now() - checkStart};
    };
    const firstExecution = await run();
    const initialValidation = await validate(false);
    for (let warmup = 0; warmup < settings.warmupCount; warmup++) await run();
    const repeatedExecution = [];
    for (let sample = 0; sample < settings.sampleCount; sample++)
      repeatedExecution.push(await run());
    const repeatedValidation = await validate(false);
    const updateUploadMilliseconds = await upload(scope.updateWrites);
    const updatedExecution = [];
    for (let sample = 0; sample < settings.updateSampleCount; sample++)
      updatedExecution.push(await run());
    const updatedValidation = await validate(true);
    const sorted = repeatedExecution
      .map(sample => sample.encodeSubmitCompletionMilliseconds)
      .sort((first, second) => first - second);
    return {
      workload: workload.name,
      shape: workload.shape,
      updateKind: workload.updateKind,
      phases: {
        setupMilliseconds,
        declarationMilliseconds: scope.declarationMilliseconds,
        uploadMilliseconds,
        compileHostMilliseconds,
        firstExecution,
        repeatedExecution,
        repeatedWallClockMedianMilliseconds: getBenchmarkDistribution(sorted).median,
        initialValidation,
        repeatedValidation,
        updateUploadMilliseconds,
        updatedExecution,
        updatedValidation
      },
      memory: {
        inputBytes: scope.initialWrites.reduce((sum, write) => sum + write.values.byteLength, 0),
        updateBytes: scope.updateWrites.reduce((sum, write) => sum + write.values.byteLength, 0),
        outputBytes: scope.outputs.reduce((sum, output) => sum + output['length'] * 4, 0),
        graphPhysicalTransientBytes: compiled.stats.physicalTransientResourceBytes,
        graphLogicalTransientBytes: compiled.stats.logicalTransientBytes,
        driverPeakBytes: null
      },
      graphNodes: compiled.stats.nodeOrder.length,
      correctness:
        'initial, repeated and updated outputs checked against pinned direct/composed references or analytic affine shifts'
    };
  } finally {
    compiled?.destroy();
    scope.destroy();
  }
}

it.skipIf(!RUN_BENCHMARK)(
  'bounded global-coordinate performance observations',
  {timeout: 300000},
  async () => {
    const settings = getBenchmarkSettings(environment);
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    console.log(
      `GLOBAL_COORDINATES_BENCHMARK_METADATA ${JSON.stringify({schemaVersion: 1, recordedAt: new Date().toISOString(), sourceRevision: settings.sourceSnapshot?.revision ?? environment?.['VITE_BENCHMARK_REVISION'] ?? null, sourceState: settings.sourceSnapshot?.sha256 ?? environment?.['VITE_BENCHMARK_SOURCE_STATE'] ?? null, sourceSnapshot: settings.sourceSnapshot, sampling: settings, userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency, limits: {maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage, maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize}, device: device.info, timestampQueriesAvailable: device.features.has('timestamp-query'), method: {executionClock: 'CPU encode+submit+queue completion; readback and assertion checks excluded', compileClock: 'synchronous graph.compile; deferred work may appear in first execution', uploadClock: 'buffer.write+queue completion; allocation excluded', warmups: settings.warmupCount, repeatedSamples: settings.sampleCount, updateSamples: settings.updateSampleCount, memory: 'explicit buffers and graph-owned transient allocation; not driver peak', cache: 'shared device; caches not cleared', envelope: 'bounded workload observations; no speed advantage, cold-driver guarantee or latency threshold'}})}`
    );
    let measuredWorkloads = 0;
    for (const workload of createWorkloads()) {
      if (!workload.name.includes(settings.workloadFilter)) continue;
      measuredWorkloads++;
      console.log(
        `GLOBAL_COORDINATES_BENCHMARK_MEASUREMENT ${JSON.stringify(await measure(device, workload, settings))}`
      );
    }
    if (!measuredWorkloads) throw new Error('No global-coordinate workloads matched the filter');
  }
);
