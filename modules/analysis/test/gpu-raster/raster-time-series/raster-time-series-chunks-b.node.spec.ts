// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  GPUCommandGraph,
  GraphVectorView,
  type GraphDataView,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {describe, expect, it, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  GPURasterStandardizedIndex,
  MAXIMUM_STANDARDIZED_INDEX_CHUNK_SLICE_COUNT
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index';
import {GPURasterStandardizedIndexCalibration} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index-calibration';
import {GPURasterStreamingAccumulator} from '../../../src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator';
import {GPURasterWhittakerSmoother} from '../../../src/raster-analysis/raster-time-series/gpu-raster-whittaker-smoother';
import {GPURasterTemporalResample} from '../../../src/raster-analysis/raster-time-series/gpu-raster-temporal-resample';
import {GPURasterPhenologyMetrics} from '../../../src/raster-analysis/raster-time-series/gpu-raster-phenology-metrics';
import {GPURasterTemporal} from '../../../src/raster-analysis/remote-sensing/gpu-raster-temporal';
import {GPURasterSeasonalBreak} from '../../../src/raster-analysis/remote-sensing/gpu-raster-seasonal-break';
import type {GPURasterIrregularDateAxis} from '../../../src/raster-analysis/raster-time-series/raster-irregular-date-axis';
import {GPUChangeDetection} from '../../../src/raster-analysis/change-detection/gpu-change-detection';
import {GPURasterMedoidComposite} from '../../../src/raster-analysis/raster-time-series/gpu-raster-medoid-composite';
import {planRasterTimeSeriesChunks} from '../../../src/raster-analysis/raster-time-series/raster-time-series-chunks-b';
import {
  createRasterTimeSeriesChunkCase,
  RASTER_TIME_SERIES_CHUNK_CASE_NAMES,
  type ChunkCaseHarness
} from './raster-time-series-chunks-b-cases';

let serial = 0;
function vector<Format extends 'float32' | 'uint32'>(
  chunks: GraphDataView<Format>[]
): GraphVectorView<Format> {
  return new GraphVectorView({
    id: `vector-${serial++}`,
    name: 'cell-chunks',
    format: chunks[0].format,
    length: chunks.reduce((total, chunk) => total + chunk.length, 0),
    valueLength: chunks.reduce((total, chunk) => total + chunk.length, 0),
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data: chunks
  });
}

function createHarness(
  graph: GPUCommandGraph,
  cellCounts: number[],
  packed = false,
  options: {
    isolated?: boolean;
    dateCount?: number;
    badBinding?: string;
    capture?: Map<string, GraphDataView>;
  } = {}
): ChunkCaseHarness {
  let sharedOrdinal = 0;
  const create = <Format extends 'float32' | 'uint32' | 'uint32x2'>(
    key: string,
    format: Format,
    length: number,
    defaultPrefix: number
  ) => {
    const prefix = options.isolated ? (options.badBinding === key ? 49 : 0) : defaultPrefix;
    const width = format === 'uint32x2' ? 2 : 1;
    const backing = createTransientView(
      graph,
      `binding-${serial++}`,
      'uint32',
      prefix + Math.max(length, 1) * width
    );
    const view = graph.createDataView(backing.buffer, {format, byteOffset: prefix * 4, length});
    options.capture?.set(key, view);
    return view;
  };
  return {
    cellCount: cellCounts.reduce((total, count) => total + count, 0),
    field(name, format, entriesPerCell) {
      const chunks = cellCounts.map((count, index) =>
        create(
          `field:${name}:${index}`,
          format,
          count * entriesPerCell,
          [1, 65, 3, 17, 7][index % 5]
        )
      );
      return packed ? chunks[0] : vector(chunks);
    },
    shared(values) {
      return create(`shared:${sharedOrdinal++}`, 'float32', values.length, 0);
    },
    dates() {
      return {
        acquisitionTimes: create('dateAxis', 'uint32x2', options.dateCount ?? 5, 0),
        temporalContext: {
          unit: 'second',
          epoch: 'unix',
          encoding: 'int64-words',
          axis: {kind: 'irregular'}
        }
      };
    }
  };
}

function computeNodes(nodes: readonly GPUCommandNode[]): GPUCommandNode[] {
  return nodes.filter(node => node.type === 'compute');
}

describe('SC3-RTS-B complete cell chunks', () => {
  for (const name of RASTER_TIME_SERIES_CHUNK_CASE_NAMES) {
    it(`${name}: one pipeline per statistic as unequal chunk count grows, valid WGSL and exact node budgets`, () => {
      const device = createNullWebGPUDevice();
      try {
        const pipelineKeys: Set<string>[] = [];
        for (const counts of [
          [1, 65, 129],
          [2, 33, 256, 0, 71]
        ]) {
          const graph = new GPUCommandGraph(device);
          const {contributor} = createRasterTimeSeriesChunkCase(name, createHarness(graph, counts));
          const nodes = contributor.getCommandNodes(graph);
          expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
          const kernels = getGPUCommandNodeKernelSources(computeNodes(nodes));
          const keys = new Set(
            kernels.map(
              kernel =>
                `${kernel.source}\n${JSON.stringify(kernel.constants)}\n${JSON.stringify(kernel.shaderLayout)}`
            )
          );
          pipelineKeys.push(keys);
          expect(kernels).toHaveLength(
            (name === 'change' ? 4 : 1) * counts.filter(count => count > 0).length
          );
          for (const kernel of kernels) {
            expect(() => new WgslReflect(kernel.source)).not.toThrow();
            expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
            expect(kernel.constants).toEqual({});
          }
          // Actual dispatch workgroups must be sized independently for each chunk.
          const firstStatistic = computeNodes(nodes).slice(
            0,
            counts.filter(count => count > 0).length
          );
          const workgroupSize = Number(/@workgroup_size\((\d+)\)/.exec(kernels[0].source)![1]);
          const invocationsPerCell = name === 'standardized' || name === 'temporal-sg' ? 5 : 1;
          expect(firstStatistic.map(node => node.workload!.maximumInvocationCount)).toEqual(
            counts.filter(count => count > 0).map(count => count * invocationsPerCell)
          );
          expect(firstStatistic.map(node => node.workload!.maximumWorkgroupCount)).toEqual(
            counts
              .filter(count => count > 0)
              .map(count => Math.ceil((count * invocationsPerCell) / workgroupSize))
          );
        }
        expect([...pipelineKeys[0]].sort()).toEqual([...pipelineKeys[1]].sort());
        expect(pipelineKeys[0].size).toBe(name === 'change' ? 4 : 1);
      } finally {
        device.destroy();
      }
    });

    it(`${name}: isolates every input, output and shared-table aligned binding failure`, () => {
      const device = createNullWebGPUDevice();
      try {
        Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 192});
        const baseline = new Map<string, GraphDataView>();
        createRasterTimeSeriesChunkCase(
          name,
          createHarness(new GPUCommandGraph(device), [1, 1, 1], false, {
            isolated: true,
            capture: baseline
          })
        );
        for (const badBinding of baseline.keys()) {
          const graph = new GPUCommandGraph(device);
          const capture = new Map<string, GraphDataView>();
          const {contributor} = createRasterTimeSeriesChunkCase(
            name,
            createHarness(graph, [1, 1, 1], false, {isolated: true, badBinding, capture})
          );
          for (const [key, view] of capture) {
            if (key === badBinding)
              expect(getViewBindingRange(view).size, key).toBeGreaterThan(192);
            else expect(getViewBindingRange(view).size, key).toBeLessThanOrEqual(192);
          }
          const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
          const createView = vi.spyOn(graph, 'createDataView');
          const addCompute = vi.spyOn(graph, 'addComputePass');
          const addCopy = vi.spyOn(graph, 'addCopyPass');
          expect(() => contributor.getCommandNodes(graph), badBinding).toThrow(/limit 192/);
          expect(createBuffer, badBinding).not.toHaveBeenCalled();
          expect(createView, badBinding).not.toHaveBeenCalled();
          expect(addCompute, badBinding).not.toHaveBeenCalled();
          expect(addCopy, badBinding).not.toHaveBeenCalled();
        }
      } finally {
        device.destroy();
      }
    });
  }

  for (const variant of ['multiband', 'multi-slot', 'stream-updates', 'iterative'] as const) {
    it(`${variant}: audits the full WGSL reserved list and keeps pipelines constant across chunks and temporal updates`, () => {
      const device = createNullWebGPUDevice();
      try {
        const keysByTopology: Set<string>[] = [];
        for (const counts of [
          [1, 65, 129],
          [2, 33, 256, 0, 71]
        ]) {
          const graph = new GPUCommandGraph(device);
          const harness = createHarness(graph, counts);
          const field = harness.field;
          const cellCount = harness.cellCount;
          const nodes: GPUCommandNode[] = [];
          if (variant === 'multiband') {
            nodes.push(
              ...new GPUChangeDetection({
                cellCount,
                sliceCount: 2,
                bandCount: 3,
                slices: field('slices', 'float32', 6),
                mask: field('mask', 'uint32', 1),
                observationMask: field('observations', 'uint32', 2),
                parameters: harness.shared([0, 1, 0, 0.05, 1]),
                output: {
                  difference: field('difference', 'float32', 3),
                  logRatio: field('ratio', 'float32', 3),
                  percentChange: field('percent', 'float32', 3),
                  changeMagnitude: field('magnitude', 'float32', 1),
                  changeDirection: field('direction', 'uint32', 1)
                }
              }).getCommandNodes(graph)
            );
          } else if (variant === 'multi-slot') {
            const state = {
              counts: field('counts', 'uint32', 6),
              moments: field('moments', 'uint32', 18)
            };
            const fit = field('fit', 'float32', 6);
            for (const update of [0, 1])
              nodes.push(
                ...new GPURasterStandardizedIndexCalibration({
                  id: `calibration-${update}`,
                  cellCount,
                  chunkSliceCount: 4,
                  slotCount: 2,
                  distribution: 'fisk',
                  zeroInflated: true,
                  values: field('values', 'float32', 4),
                  mask: field('mask', 'uint32', 4),
                  slotIndices: createTransientView(graph, `slots-${update}`, 'uint32', 4),
                  parameters: harness.shared([update === 0 ? 1 : 0, 4, 0, 0, 0, 0, 128]),
                  state,
                  fit
                }).getCommandNodes(graph)
              );
          } else if (variant === 'stream-updates') {
            const state = {
              counts: field('counts', 'uint32', 1),
              sums: field('sums', 'uint32', 3),
              moments: field('moments', 'uint32', 3),
              extrema: field('extrema', 'float32', 2)
            };
            for (const update of [0, 1])
              nodes.push(
                ...new GPURasterStreamingAccumulator({
                  id: `stream-${update}`,
                  cellCount,
                  chunkSliceCount: 3,
                  values: field('values', 'float32', 3),
                  mask: field('mask', 'uint32', 3),
                  parameters: harness.shared([update === 0 ? 1 : 0, 3, 0, 0, 128]),
                  state
                }).getCommandNodes(graph)
              );
          } else {
            nodes.push(
              ...new GPURasterSeasonalBreak({
                cellCount,
                sliceCount: 12,
                values: field('values', 'float32', 12),
                period: 4,
                minimumSegmentLength: 3,
                fitMethod: 'iterative',
                maximumIterations: 1,
                tolerance: 1e-20,
                minimumBICGain: 1e6,
                output: {
                  coefficients: field('coefficients', 'float32', 5),
                  components: field('components', 'float32', 36),
                  breakIndex: field('break', 'uint32', 1),
                  improvement: field('gain', 'float32', 1),
                  status: field('status', 'uint32', 1),
                  diagnostics: field('diagnostics', 'float32', 6)
                }
              }).getCommandNodes(graph)
            );
          }
          const kernels = getGPUCommandNodeKernelSources(nodes);
          for (const kernel of kernels) {
            expect(() => new WgslReflect(kernel.source)).not.toThrow();
            expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
          }
          keysByTopology.push(
            new Set(
              kernels.map(
                kernel =>
                  `${kernel.source}\n${JSON.stringify(kernel.constants)}\n${JSON.stringify(kernel.shaderLayout)}`
              )
            )
          );
        }
        expect([...keysByTopology[0]].sort()).toEqual([...keysByTopology[1]].sort());
        expect(keysByTopology[0].size).toBe(variant === 'multiband' ? 2 : 1);
      } finally {
        device.destroy();
      }
    });
  }

  it('preflights date axes for every date-aware family with every other range below the limit', () => {
    const device = createNullWebGPUDevice();
    try {
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 192});
      for (const family of [
        'whittaker',
        'resample',
        'phenology',
        'temporal',
        'seasonal',
        'change'
      ]) {
        const graph = new GPUCommandGraph(device);
        const capture = new Map<string, GraphDataView>();
        const harness = createHarness(graph, [1, 1, 1], false, {
          isolated: true,
          badBinding: 'dateAxis',
          dateCount: 12,
          capture
        });
        const field = harness.field;
        const values = field('values', 'float32', 12);
        const parameters = harness.shared([1, 12, 0, 0, 128, 0, 0]);
        const dateAxis = harness.dates();
        const status = field('status', 'uint32', 1);
        const contributor =
          family === 'whittaker'
            ? new GPURasterWhittakerSmoother({
                cellCount: 3,
                sliceCount: 12,
                values,
                parameters,
                dateAxis,
                output: {smoothed: field('smoothed', 'float32', 12), status}
              })
            : family === 'resample'
              ? new GPURasterTemporalResample({
                  cellCount: 3,
                  sliceCount: 12,
                  gridCount: 1,
                  values,
                  parameters,
                  dateAxis,
                  output: {values: field('resampled', 'float32', 1), status}
                })
              : family === 'phenology'
                ? new GPURasterPhenologyMetrics({
                    cellCount: 3,
                    sliceCount: 12,
                    maximumSeasonCount: 1,
                    smoothed: values,
                    parameters,
                    dateAxis,
                    output: {
                      metrics: field('metrics', 'float32', 10),
                      flags: field('flags', 'uint32', 1),
                      seasonCount: field('seasons', 'uint32', 1),
                      status
                    }
                  })
                : family === 'temporal'
                  ? new GPURasterTemporal({
                      cellCount: 3,
                      sliceCount: 12,
                      values,
                      operation: 'savitzky-golay',
                      windowLength: 3,
                      dateAxis,
                      output: {
                        values: field('mean', 'float32', 12),
                        counts: field('counts', 'uint32', 12),
                        chosenSlice: field('chosen', 'uint32', 12),
                        status: field('sg-status', 'uint32', 12)
                      }
                    })
                  : family === 'seasonal'
                    ? new GPURasterSeasonalBreak({
                        cellCount: 3,
                        sliceCount: 12,
                        values,
                        period: 1,
                        minimumSegmentLength: 3,
                        dateAxis,
                        output: {
                          coefficients: field('coefficients', 'float32', 5),
                          components: field('components', 'float32', 36),
                          breakIndex: field('break', 'uint32', 1),
                          improvement: field('gain', 'float32', 1),
                          status
                        }
                      })
                    : new GPUChangeDetection({
                        cellCount: 3,
                        sliceCount: 12,
                        slices: values,
                        parameters,
                        dateAxis,
                        output: {senSlope: field('senSlope', 'float32', 1)}
                      });
        for (const [key, view] of capture) {
          if (key === 'dateAxis') expect(getViewBindingRange(view).size).toBeGreaterThan(192);
          else expect(getViewBindingRange(view).size, key).toBeLessThanOrEqual(192);
        }
        const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
        const createView = vi.spyOn(graph, 'createDataView');
        expect(() => contributor.getCommandNodes(graph), family).toThrow(/limit 192/);
        expect(createBuffer).not.toHaveBeenCalled();
        expect(createView).not.toHaveBeenCalled();
      }
    } finally {
      device.destroy();
    }
  });

  it('isolates packed seasonal-slot tables from valid cell inputs and outputs', () => {
    const device = createNullWebGPUDevice();
    try {
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 192});
      for (const family of ['streaming', 'calibration', 'standardized']) {
        const graph = new GPUCommandGraph(device);
        const capture = new Map<string, GraphDataView>();
        const harness = createHarness(graph, [1, 1, 1], false, {isolated: true, capture});
        const field = harness.field;
        const backing = createTransientView(graph, 'slot-backing', 'uint32', 53);
        const slotIndices = graph.createDataView(backing.buffer, {
          format: 'uint32',
          byteOffset: 196,
          length: 4
        });
        const common = {
          cellCount: 3,
          chunkSliceCount: 4,
          slotCount: 2,
          values: field('values', 'float32', 4),
          slotIndices,
          parameters: harness.shared([1, 4, 0, 0, 0, 0, 128])
        };
        const contributor =
          family === 'streaming'
            ? new GPURasterStreamingAccumulator({
                ...common,
                state: {counts: field('counts', 'uint32', 2), sums: field('sums', 'uint32', 6)}
              })
            : family === 'calibration'
              ? new GPURasterStandardizedIndexCalibration({
                  ...common,
                  distribution: 'fisk',
                  state: {
                    counts: field('counts', 'uint32', 6),
                    moments: field('moments', 'uint32', 18)
                  },
                  fit: field('fit', 'float32', 6)
                })
              : new GPURasterStandardizedIndex({
                  ...common,
                  distribution: 'fisk',
                  fit: field('fit', 'float32', 6),
                  output: {
                    index: field('index', 'float32', 4),
                    status: field('status', 'uint32', 4)
                  }
                });
        for (const view of capture.values())
          expect(getViewBindingRange(view).size).toBeLessThanOrEqual(192);
        expect(getViewBindingRange(slotIndices).size).toBe(212);
        const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
        const createView = vi.spyOn(graph, 'createDataView');
        expect(() => contributor.getCommandNodes(graph), family).toThrow(/limit 192/);
        expect(createBuffer).not.toHaveBeenCalled();
        expect(createView).not.toHaveBeenCalled();
      }
    } finally {
      device.destroy();
    }
  });

  it('keeps the same pipeline beyond the first 65535 workgroups while sizing smaller chunks independently', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const {contributor} = createRasterTimeSeriesChunkCase(
        'temporal-mean',
        createHarness(graph, [4194241, 63, 65])
      );
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.map(node => node.workload!.maximumWorkgroupCount)).toEqual([131070, 1, 2]);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      expect(
        new Set(kernels.map(kernel => `${kernel.source}\n${JSON.stringify(kernel.constants)}`)).size
      ).toBe(1);
      for (const kernel of kernels)
        expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    } finally {
      device.destroy();
    }
  });

  it('rejects date-axis/output aliasing in every owned date-aware contributor', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const integers = (id: string, length: number) =>
        createTransientView(graph, id, 'uint32', length);
      const floats = (id: string, length: number) =>
        createTransientView(graph, id, 'float32', length);
      const backing = integers('aliased-date-output', 128);
      const dateAxis: GPURasterIrregularDateAxis = {
        acquisitionTimes: graph.createDataView(backing.buffer, {format: 'uint32x2', length: 12}),
        temporalContext: {
          unit: 'second',
          epoch: 'unix',
          encoding: 'int64-words',
          axis: {kind: 'irregular'}
        }
      };
      const output = (length: number) =>
        graph.createDataView(backing.buffer, {format: 'float32', length});
      const values = floats('values', 12);
      const parameters = floats('parameters', 7);
      const status = integers('status', 1);
      const constructors = [
        () =>
          new GPURasterWhittakerSmoother({
            cellCount: 1,
            sliceCount: 12,
            values,
            parameters,
            dateAxis,
            output: {smoothed: output(12), status}
          }),
        () =>
          new GPURasterTemporalResample({
            cellCount: 1,
            sliceCount: 12,
            gridCount: 1,
            values,
            parameters,
            dateAxis,
            output: {values: output(1), status}
          }),
        () =>
          new GPURasterPhenologyMetrics({
            cellCount: 1,
            sliceCount: 12,
            maximumSeasonCount: 1,
            smoothed: values,
            parameters,
            dateAxis,
            output: {
              metrics: output(10),
              flags: integers('flags', 1),
              seasonCount: integers('seasonCount', 1),
              status
            }
          }),
        () =>
          new GPURasterTemporal({
            cellCount: 1,
            sliceCount: 12,
            values,
            operation: 'savitzky-golay',
            windowLength: 3,
            dateAxis,
            output: {
              values: output(12),
              counts: integers('counts', 12),
              chosenSlice: integers('chosen', 12),
              status: integers('sg-status', 12)
            }
          }),
        () =>
          new GPURasterSeasonalBreak({
            cellCount: 1,
            sliceCount: 12,
            values,
            period: 1,
            minimumSegmentLength: 3,
            dateAxis,
            output: {
              coefficients: output(5),
              components: floats('components', 36),
              breakIndex: integers('break', 1),
              improvement: floats('gain', 1),
              status
            }
          }),
        () =>
          new GPUChangeDetection({
            cellCount: 1,
            sliceCount: 12,
            slices: values,
            parameters,
            dateAxis,
            output: {senSlope: output(1)}
          })
      ];
      for (const construct of constructors) expect(construct).toThrow(/logical.*overlap/);
    } finally {
      device.destroy();
    }
  });

  it('rejects a source chunk splitting a cell before dispatch planning', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const values = vector([
        createTransientView(graph, 'first', 'float32', 3),
        createTransientView(graph, 'second', 'float32', 1)
      ]);
      expect(() =>
        planRasterTimeSeriesChunks('split-record', 2, {values: {view: values, entriesPerCell: 2}})
      ).toThrow(/split-record.*split cell record.*complete cell records/);
    } finally {
      device.destroy();
    }
  });

  it('rejects mismatched cell topology and malformed vector totals', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const values = vector([
        createTransientView(graph, 'v0', 'float32', 2),
        createTransientView(graph, 'v1', 'float32', 4)
      ]);
      const mask = vector([
        createTransientView(graph, 'm0', 'uint32', 4),
        createTransientView(graph, 'm1', 'uint32', 2)
      ]);
      expect(() =>
        planRasterTimeSeriesChunks('mismatch', 3, {
          values: {view: values, entriesPerCell: 2},
          mask: {view: mask, entriesPerCell: 2}
        })
      ).toThrow(/mask.*cell chunk topology/);
      expect(() =>
        planRasterTimeSeriesChunks('total', 4, {values: {view: values, entriesPerCell: 2}})
      ).toThrow(/chunk total/);
    } finally {
      device.destroy();
    }
  });

  it('rejects overlapping aligned output fields and overlapping logical output chunks', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const values = createTransientView(graph, 'values', 'float32', 6);
      const backing = createTransientView(graph, 'outputs', 'uint32', 128);
      const view = <Format extends 'float32' | 'uint32'>(
        format: Format,
        offset: number,
        length: number
      ) => graph.createDataView(backing.buffer, {format, byteOffset: offset, length});
      expect(
        () =>
          new GPURasterMedoidComposite({
            cellCount: 3,
            sliceCount: 2,
            bandCount: 1,
            values,
            output: {
              composite: view('float32', 0, 3),
              sliceIndex: view('uint32', 12, 3),
              status: view('uint32', 24, 3)
            }
          })
      ).toThrow(/aligned binding/);
      const input = vector([
        graph.createDataView(values.buffer, {format: 'float32', length: 2}),
        graph.createDataView(values.buffer, {format: 'float32', byteOffset: 8, length: 2}),
        graph.createDataView(values.buffer, {format: 'float32', byteOffset: 16, length: 2})
      ]);
      const output = vector([view('float32', 0, 1), view('float32', 4, 1), view('float32', 8, 1)]);
      const sliceIndex = vector([
        createTransientView(graph, 'i0', 'uint32', 1),
        createTransientView(graph, 'i1', 'uint32', 1),
        createTransientView(graph, 'i2', 'uint32', 1)
      ]);
      const status = vector([
        createTransientView(graph, 's0', 'uint32', 1),
        createTransientView(graph, 's1', 'uint32', 1),
        createTransientView(graph, 's2', 'uint32', 1)
      ]);
      expect(
        () =>
          new GPURasterMedoidComposite({
            cellCount: 3,
            sliceCount: 2,
            bandCount: 1,
            values: input,
            output: {composite: output, sliceIndex, status}
          })
      ).not.toThrow();
      const overlapping = vector([
        view('float32', 0, 1),
        view('float32', 4, 1),
        view('float32', 0, 1)
      ]);
      expect(
        () =>
          new GPURasterMedoidComposite({
            cellCount: 3,
            sliceCount: 2,
            bandCount: 1,
            values: input,
            output: {composite: overlapping, sliceIndex, status}
          })
      ).toThrow(/logical/);
    } finally {
      device.destroy();
    }
  });

  it('rejects change detection outputs that overlap, even across different statistics', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const shared = createTransientView(graph, 'aliased-outputs', 'float32', 3);
      expect(
        () =>
          new GPUChangeDetection({
            cellCount: 3,
            sliceCount: 5,
            slices: createTransientView(graph, 'slices', 'float32', 15),
            parameters: createTransientView(graph, 'parameters', 'float32', 5),
            output: {difference: shared, tStatistic: shared}
          })
      ).toThrow(/logical.*overlap/);
    } finally {
      device.destroy();
    }
  });

  it('accepts independently evaluated standardized-index samples past 256 and enforces the float32 count limit', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const make = (count: number) =>
        new GPURasterStandardizedIndex({
          cellCount: 1,
          chunkSliceCount: count,
          distribution: 'fisk',
          values: createTransientView(graph, `values-${count}`, 'float32', count),
          fit: createTransientView(graph, `fit-${count}`, 'float32', 3),
          parameters: createTransientView(graph, `parameters-${count}`, 'float32', 2),
          output: {
            index: createTransientView(graph, `index-${count}`, 'float32', count),
            status: createTransientView(graph, `status-${count}`, 'uint32', count)
          }
        });
      expect(make(513).getCommandNodes(graph)).toHaveLength(1);
      expect(() => make(MAXIMUM_STANDARDIZED_INDEX_CHUNK_SLICE_COUNT)).not.toThrow();
      expect(() => make(MAXIMUM_STANDARDIZED_INDEX_CHUNK_SLICE_COUNT + 1)).toThrow(
        /chunkSliceCount.*16777216/
      );
    } finally {
      device.destroy();
    }
  });

  it('keeps packed props inferred as packed views and packed nodes unchanged', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const output = {
        composite: createTransientView(graph, 'composite', 'float32', 1),
        sliceIndex: createTransientView(graph, 'sliceIndex', 'uint32', 1),
        status: createTransientView(graph, 'status', 'uint32', 1)
      };
      const contributor = new GPURasterMedoidComposite({
        cellCount: 1,
        sliceCount: 2,
        bandCount: 1,
        values: createTransientView(graph, 'values', 'float32', 2),
        output
      });
      const packed: GraphDataView<'float32'> = contributor.props.output.composite;
      expect(packed).toBe(output.composite);
      expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual([
        'raster-medoid-composite'
      ]);
    } finally {
      device.destroy();
    }
  });
});

it('accepts packed change statistics sharing an alignment window only across separate dispatches', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const slices = createTransientView(graph, 'review-slices', 'float32', 12);
    const parameters = createTransientView(graph, 'review-parameters', 'float32', 5);
    const backing = createTransientView(graph, 'review-output', 'float32', 6);
    const difference = graph.createDataView(backing.buffer, {format: 'float32', length: 3});
    const tStatistic = graph.createDataView(backing.buffer, {
      format: 'float32',
      byteOffset: 12,
      length: 3
    });
    const props = {
      cellCount: 3,
      sliceCount: 4,
      slices,
      parameters,
      output: {difference, tStatistic}
    };
    const contributor = new GPUChangeDetection(props);
    expect(contributor.getCommandNodes(graph)).toHaveLength(2);
    const overlap = graph.createDataView(backing.buffer, {
      format: 'float32',
      byteOffset: 8,
      length: 3
    });
    expect(
      () => new GPUChangeDetection({...props, output: {difference, tStatistic: overlap}})
    ).toThrow(/logical.*overlap/);
    expect(
      () => new GPUChangeDetection({...props, output: {difference, logRatio: tStatistic}})
    ).toThrow(/aligned binding.*overlap/);
  } finally {
    device.destroy();
  }
});

it('emits seasonal floating-point props without oversized WGSL abstract-integer literals', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const extended of [false, true]) {
      for (const scalar of [2 ** -149, 2 ** 63, 1e20, Math.fround(3.402823466e38)]) {
        const graph = new GPUCommandGraph(device);
        let ordinal = 0;
        const field = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
          createTransientView(graph, `literal-${ordinal++}`, format, length);
        const contributor = new GPURasterSeasonalBreak({
          cellCount: 1,
          sliceCount: 16,
          values: field('float32', 16),
          period: Math.max(4, scalar),
          minimumSegmentLength: 3,
          minimumBICGain: scalar,
          ...(extended
            ? {
                fitMethod: 'iterative' as const,
                maximumIterations: 1,
                tolerance: scalar,
                monitoring: {historyLength: 12, threshold: scalar, minimumConsecutive: 1}
              }
            : {}),
          output: {
            coefficients: field('float32', 5),
            components: field('float32', 48),
            breakIndex: field('uint32', 1),
            improvement: field('float32', 1),
            status: field('uint32', 1),
            ...(extended ? {diagnostics: field('float32', 6)} : {})
          }
        });
        for (const kernel of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
          expect(() => new WgslReflect(kernel.source)).not.toThrow();
          expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
          const source = kernel.source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
          for (const [token] of source.matchAll(/\b\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[fhiu]?\b/g)) {
            if (/^\d+$/.test(token))
              expect(BigInt(token), token).toBeLessThanOrEqual(0x7fffffffffffffffn);
            else if (/^\d+u$/.test(token))
              expect(BigInt(token.slice(0, -1)), token).toBeLessThanOrEqual(0xffffffffn);
            else if (/[.eEf]/.test(token))
              expect(Number.isFinite(Math.fround(Number(token.replace(/f$/, '')))), token).toBe(
                true
              );
          }
        }
      }
    }
  } finally {
    device.destroy();
  }
});

for (const name of RASTER_TIME_SERIES_CHUNK_CASE_NAMES) {
  it(`${name}: accepts disjoint aligned packed outputs with separate input storage`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const backing = createTransientView(graph, 'packed-fields', 'uint32', 4096);
      let byteOffset = 0;
      let inputSerial = 0;
      const allocate = <Format extends 'float32' | 'uint32' | 'uint32x2'>(
        format: Format,
        length: number
      ) => {
        const view = graph.createDataView(backing.buffer, {format, length, byteOffset});
        byteOffset += Math.ceil((length * (format === 'uint32x2' ? 8 : 4)) / 256) * 256;
        return view;
      };
      const fixture: ChunkCaseHarness = {
        cellCount: 3,
        field: (fieldName, format, width, value) =>
          value
            ? createTransientView(
                graph,
                `packed-input-${fieldName}-${inputSerial++}`,
                format,
                3 * width
              )
            : allocate(format, 3 * width),
        shared: values =>
          createTransientView(graph, `packed-shared-${inputSerial++}`, 'float32', values.length),
        dates: () => ({
          acquisitionTimes: createTransientView(
            graph,
            `packed-dates-${inputSerial++}`,
            'uint32x2',
            5
          ),
          temporalContext: {
            unit: 'second',
            epoch: 'unix',
            encoding: 'int64-words',
            axis: {kind: 'irregular'}
          }
        })
      };
      const {contributor} = createRasterTimeSeriesChunkCase(name, fixture);
      expect(contributor.getCommandNodes(graph)).toHaveLength(
        contributor.estimatedCommandNodeCount
      );
    } finally {
      device.destroy();
    }
  });

  it(`${name}: rejects missing STORAGE usage for every actual field before graph mutation`, () => {
    const device = createNullWebGPUDevice();
    try {
      const baseline = new Map<string, GraphDataView>();
      createRasterTimeSeriesChunkCase(
        name,
        createHarness(new GPUCommandGraph(device), [1, 1, 1], false, {
          isolated: true,
          capture: baseline
        })
      );
      for (const field of baseline.keys()) {
        const graph = new GPUCommandGraph(device);
        const capture = new Map<string, GraphDataView>();
        const {contributor} = createRasterTimeSeriesChunkCase(
          name,
          createHarness(graph, [1, 1, 1], false, {isolated: true, capture})
        );
        Object.defineProperty(capture.get(field)!.buffer, 'usage', {value: Buffer.COPY_SRC});
        const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
        const createView = vi.spyOn(graph, 'createDataView');
        expect(() => contributor.getCommandNodes(graph), field).toThrow(/Buffer.STORAGE/);
        expect(createBuffer).not.toHaveBeenCalled();
        expect(createView).not.toHaveBeenCalled();
      }
    } finally {
      device.destroy();
    }
  });
}

it('allows read-only chunks to reuse bytes across separate cell dispatches', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const sample = createTransientView(graph, 'repeated-samples', 'float32', 5);
    const fields = createHarness(graph, [1, 1, 1]);
    const contributor = new GPURasterTemporal({
      cellCount: 3,
      sliceCount: 5,
      values: vector([sample, sample, sample]),
      operation: 'mean',
      output: {
        values: fields.field('mean', 'float32', 1),
        counts: fields.field('counts', 'uint32', 1),
        chosenSlice: fields.field('chosen', 'uint32', 1),
        status: fields.field('status', 'uint32', 1)
      }
    });
    expect(contributor.getCommandNodes(graph)).toHaveLength(3);
  } finally {
    device.destroy();
  }
});

it('ignores unused dates for composites and change statistics without Sen slope', () => {
  const device = createNullWebGPUDevice();
  try {
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 192});
    const foreign = new GPUCommandGraph(device);
    const backing = createTransientView(foreign, 'unused-dates', 'uint32', 100);
    Object.defineProperty(backing.buffer, 'usage', {value: Buffer.COPY_SRC});
    const dateAxis: GPURasterIrregularDateAxis = {
      acquisitionTimes: foreign.createDataView(backing.buffer, {
        format: 'uint32x2',
        byteOffset: 196,
        length: 12
      }),
      temporalContext: {
        unit: 'second',
        epoch: 'unix',
        encoding: 'int64-words',
        axis: {kind: 'irregular'}
      }
    };
    for (const operation of ['mean', 'difference'] as const) {
      const graph = new GPUCommandGraph(device);
      const fields = createHarness(graph, [1], true, {isolated: true});
      const values = fields.field('values', 'float32', 12);
      const contributor =
        operation === 'mean'
          ? new GPURasterTemporal({
              cellCount: 1,
              sliceCount: 12,
              values,
              operation,
              dateAxis,
              output: {
                values: fields.field('mean', 'float32', 1),
                counts: fields.field('counts', 'uint32', 1),
                chosenSlice: fields.field('chosen', 'uint32', 1),
                status: fields.field('status', 'uint32', 1)
              }
            })
          : new GPUChangeDetection({
              cellCount: 1,
              sliceCount: 12,
              slices: values,
              dateAxis,
              parameters: fields.shared([0, 1, 0, 0.05, 6]),
              output: {difference: fields.field('difference', 'float32', 1)}
            });
      expect(contributor.getCommandNodes(graph)).toHaveLength(1);
    }
  } finally {
    device.destroy();
  }
});

it('ignores unused single-slot tables even when they cannot be storage-bound', () => {
  const device = createNullWebGPUDevice();
  try {
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 192});
    const foreign = new GPUCommandGraph(device);
    const backing = createTransientView(foreign, 'unused-slots', 'uint32', 100);
    Object.defineProperty(backing.buffer, 'usage', {value: Buffer.COPY_SRC});
    const slotIndices = foreign.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: 196,
      length: 4
    });
    for (const family of ['streaming', 'calibration', 'standardized']) {
      const graph = new GPUCommandGraph(device);
      const harness = createHarness(graph, [1, 1, 1], false, {isolated: true});
      const field = harness.field;
      const common = {
        cellCount: 3,
        chunkSliceCount: 4,
        slotCount: 1,
        slotIndices,
        values: field('values', 'float32', 4),
        parameters: harness.shared([1, 4, 0, 0, 0, 0, 128])
      };
      const contributor =
        family === 'streaming'
          ? new GPURasterStreamingAccumulator({
              ...common,
              state: {counts: field('counts', 'uint32', 1), sums: field('sums', 'uint32', 3)}
            })
          : family === 'calibration'
            ? new GPURasterStandardizedIndexCalibration({
                ...common,
                distribution: 'fisk',
                state: {
                  counts: field('counts', 'uint32', 3),
                  moments: field('moments', 'uint32', 9)
                },
                fit: field('fit', 'float32', 3)
              })
            : new GPURasterStandardizedIndex({
                ...common,
                distribution: 'fisk',
                fit: field('fit', 'float32', 3),
                output: {index: field('index', 'float32', 4), status: field('status', 'uint32', 4)}
              });
      expect(contributor.getCommandNodes(graph)).toHaveLength(3);
    }
  } finally {
    device.destroy();
  }
});

it('rejects an irregular seasonal period that rounds to zero in float32', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const harness = createHarness(graph, [1], true, {dateCount: 12});
    const field = harness.field;
    expect(
      () =>
        new GPURasterSeasonalBreak({
          cellCount: 1,
          sliceCount: 12,
          values: field('values', 'float32', 12),
          dateAxis: harness.dates(),
          period: 1e-50,
          minimumSegmentLength: 3,
          output: {
            coefficients: field('coefficients', 'float32', 5),
            components: field('components', 'float32', 36),
            breakIndex: field('break', 'uint32', 1),
            improvement: field('gain', 'float32', 1),
            status: field('status', 'uint32', 1)
          }
        })
    ).toThrow(/raster-seasonal-break period/);
  } finally {
    device.destroy();
  }
});

it('allows unused date/slot views to alias outputs that actually write their bytes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const harness = createHarness(graph, [1], true, {isolated: true});
    const backing = createTransientView(graph, 'unused-alias', 'uint32', 24);
    const dateAxis: GPURasterIrregularDateAxis = {
      acquisitionTimes: graph.createDataView(backing.buffer, {format: 'uint32x2', length: 12}),
      temporalContext: {
        unit: 'second',
        epoch: 'unix',
        encoding: 'int64-words',
        axis: {kind: 'irregular'}
      }
    };
    const contributor = new GPURasterTemporal({
      cellCount: 1,
      sliceCount: 12,
      values: harness.field('values', 'float32', 12),
      operation: 'mean',
      dateAxis,
      output: {
        values: graph.createDataView(backing.buffer, {format: 'float32', length: 1}),
        counts: harness.field('counts', 'uint32', 1),
        chosenSlice: harness.field('chosen', 'uint32', 1),
        status: harness.field('status', 'uint32', 1)
      }
    });
    expect(contributor.getCommandNodes(graph)).toHaveLength(1);
    const counts = createTransientView(graph, 'slot-count-alias', 'uint32', 4);
    const streaming = new GPURasterStreamingAccumulator({
      cellCount: 1,
      chunkSliceCount: 4,
      slotCount: 1,
      slotIndices: counts,
      values: harness.field('stream-values', 'float32', 4),
      parameters: harness.shared([1, 4, 0, 0, 0]),
      state: {counts, sums: harness.field('sums', 'uint32', 3)}
    });
    expect(streaming.getCommandNodes(graph)).toHaveLength(1);
  } finally {
    device.destroy();
  }
});

it('ignores explicitly undefined optional seasonal outputs in both routes', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const fitMethod of [undefined, 'joint'] as const) {
      const graph = new GPUCommandGraph(device);
      const harness = createHarness(graph, [1], true, {isolated: true});
      const field = harness.field;
      const contributor = new GPURasterSeasonalBreak({
        cellCount: 1,
        sliceCount: 12,
        period: 4,
        values: field('values', 'float32', 12),
        minimumSegmentLength: 3,
        fitMethod,
        output: {
          coefficients: field('coefficients', 'float32', 5),
          components: field('components', 'float32', 36),
          breakIndex: field('break', 'uint32', 1),
          improvement: field('gain', 'float32', 1),
          status: field('status', 'uint32', 1),
          diagnostics: undefined
        }
      });
      expect(contributor.getCommandNodes(graph)).toHaveLength(1);
    }
  } finally {
    device.destroy();
  }
});

for (const name of RASTER_TIME_SERIES_CHUNK_CASE_NAMES) {
  it(`${name}: preflights chunk uniform allocation before graph mutation and leaves packed calls independent of it`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const {contributor} = createRasterTimeSeriesChunkCase(
        name,
        createHarness(graph, [1, 1, 1], false, {isolated: true})
      );
      const packedGraph = new GPUCommandGraph(device);
      const packed = createRasterTimeSeriesChunkCase(
        name,
        createHarness(packedGraph, [3], true, {isolated: true})
      ).contributor;
      Object.defineProperty(device.limits, 'maxBufferSize', {value: 32});
      const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
      const createView = vi.spyOn(graph, 'createDataView');
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        /chunk dispatch uniform.*maxBufferSize/
      );
      expect(createBuffer).not.toHaveBeenCalled();
      expect(createView).not.toHaveBeenCalled();
      expect(packed.getCommandNodes(packedGraph).length).toBeGreaterThan(0);
    } finally {
      device.destroy();
    }
  });
  it(`${name}: rejects a physical buffer used for read-only input and writable output in the same dispatch`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const backing = createTransientView(graph, `mixed-usage-${serial++}`, 'uint32', 256);
      let inputUsed = false;
      let outputUsed = false;
      const harness = createHarness(graph, [3], true, {isolated: true});
      const field = harness.field;
      harness.field = (fieldName, format, width, value) => {
        const read = Boolean(value);
        if (read ? inputUsed : outputUsed) return field(fieldName, format, width, value);
        if (read) inputUsed = true;
        else outputUsed = true;
        return graph.createDataView(backing.buffer, {
          format,
          length: 3 * width,
          byteOffset: read ? 0 : 256
        });
      };
      expect(() => createRasterTimeSeriesChunkCase(name, harness)).toThrow(/same physical buffer/);
    } finally {
      device.destroy();
    }
  });
}

it('normalizes standardized-index observations with exponent-scaled division on both distributions', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const distribution of ['gamma', 'fisk'] as const) {
      const graph = new GPUCommandGraph(device);
      const values = createTransientView(graph, 'scaled-values', 'float32', 3);
      const contributor = new GPURasterStandardizedIndex({
        cellCount: 1,
        chunkSliceCount: 3,
        distribution,
        values,
        fit: createTransientView(graph, 'scaled-fit', 'float32', 3),
        parameters: createTransientView(graph, 'scaled-parameters', 'float32', 2),
        output: {
          index: createTransientView(graph, 'scaled-index', 'float32', 3),
          status: createTransientView(graph, 'scaled-status', 'uint32', 3)
        }
      });
      const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      expect(kernels).toHaveLength(1);
      expect(kernels[0].source).toContain('getScaledQuotient(positive, fitScale, 1.0, 0)');
      expect(kernels[0].source).not.toContain('positive / fitScale');
      expect(kernels[0].source).toContain('fitShape > 0.0 && fitScale > 0.0');
      expect(auditWGSLModule(kernels[0].source, kernels[0].constants)).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});
