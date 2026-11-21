// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  getViewBindingRange,
  createTransientView,
  GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUDorlingCartogram} from '../../../src/gpu-spatial-analysis/dorling-cartogram/gpu-dorling-cartogram';
import {GPUSpatialKMeans} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-spatial-kmeans';
import {GPUDBSCAN} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-spatial-clustering';
import {GPUHDBSCAN} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-hierarchical-density-clustering';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {GPUInhomogeneousPointPattern} from '../../../src/gpu-spatial-analysis/point-process/gpu-inhomogeneous-point-pattern';
import {GPUPointPatternEnvelope} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope';
import {GPURegionPartitionEvaluation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation';
import {
  GPUSpatialWeightsMinimumSpanningTree,
  GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree';
import {GPURedcapSingleLinkage} from '../../../src/gpu-spatial-analysis/redcap-regions/gpu-redcap-single-linkage';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {validateSpatialOutputViews} from '../../../src/gpu-spatial-analysis/spatial-clustering/spatial-output-validation';

it('rejects output aliases in DBSCAN, k-means, HDBSCAN, REDCAP, regionalization and point envelopes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  try {
    const labels = view('uint32', 8);
    const positions = view('float32x2', 8);
    expect(
      () =>
        new GPUDBSCAN({
          positions,
          parameters: view('float32', 10),
          gridSize: [2, 2],
          output: {labels, rootRows: labels}
        })
    ).toThrow(/overlap/);
    expect(
      () =>
        new GPUSpatialKMeans({
          positions,
          clusterCount: 2,
          maximumIterations: 2,
          parameters: view('uint32', 2),
          output: {
            labels,
            centers: graph.createDataView(labels.buffer, {format: 'float32x2', length: 2})
          }
        })
    ).toThrow(/overlap/);
    const combined = view('uint32', 64);
    expect(
      () =>
        new GPUHDBSCAN({
          positions,
          minimumClusterSize: 2,
          output: {
            labels: graph.createDataView(combined.buffer, {format: 'uint32', length: 8}),
            coreDistances: graph.createDataView(combined.buffer, {
              format: 'float32',
              length: 8,
              byteOffset: 28
            })
          }
        })
    ).toThrow(/overlap/);
    const weights = {
      offsets: view('uint32', 9),
      neighbors: view('uint32', 8),
      weights: view('float32', 8)
    };
    expect(
      () =>
        new GPURedcapSingleLinkage({
          weights,
          values: view('float32', 8),
          parameters: view('uint32', 1),
          output: {
            labels,
            regionCount: graph.createDataView(labels.buffer, {format: 'uint32', length: 1})
          }
        })
    ).toThrow(/overlap/);
    const summary = view('float32', 9);
    expect(
      () =>
        new GPURegionPartitionEvaluation({
          values: view('float32', 8),
          labels,
          output: {
            summary,
            regionSizes: graph.createDataView(summary.buffer, {format: 'uint32', length: 8})
          }
        })
    ).toThrow(/overlap/);
    const envelope = view('float32', 8);
    const studyWindow = new GPUStudyWindowPreparation({
      positions,
      featureOffsets: view('uint32', 2),
      polygonOffsets: view('uint32', 2),
      ringOffsets: view('uint32', 2)
    });
    expect(
      () =>
        new GPUInhomogeneousPointPattern({
          positions,
          studyWindow,
          radii: view('float32', 8),
          suppliedIntensity: view('float32', 8),
          output: {
            intensity: envelope,
            k: envelope,
            pairCorrelation: view('float32', 8),
            status: view('uint32', 8)
          }
        })
    ).toThrow(/overlap/);

    expect(
      () =>
        new GPUPointPatternEnvelope({
          observed: view('float32', 8),
          simulations: view('float32', 16),
          reference: view('float32', 8),
          integrationWeights: view('float32', 8),
          simulationCount: 2,
          nullModel: 'independent',
          fittedParameters: 'none',
          output: {
            lower: envelope,
            upper: envelope,
            statistics: view('float32', 4),
            status: view('uint32', 1)
          }
        })
    ).toThrow(/overlap/);
  } finally {
    device.destroy();
  }
});

it('preflights k-means aligned inputs and its command node budget before scratch allocation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const positions = createTransientView(graph, 'positions', 'float32x2', 12);
    const props = {
      positions: graph.createDataView(positions.buffer, {
        format: 'float32x2' as const,
        length: 4,
        byteOffset: 64
      }),
      clusterCount: 2,
      maximumIterations: 2,
      parameters: createTransientView(graph, 'parameters', 'uint32', 2),
      output: {
        labels: createTransientView(graph, 'labels', 'uint32', 4),
        centers: createTransientView(graph, 'centers', 'float32x2', 2)
      }
    };
    const contributor = new GPUSpatialKMeans(props);
    Object.assign(device.limits, {maxStorageBufferBindingSize: 64});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/positions.*96.*64/);
    expect(allocate).not.toHaveBeenCalled();
    const big = {
      ...props,
      positions: createTransientView(graph, 'many-positions', 'float32x2', 256),
      clusterCount: 256,
      maximumIterations: 64,
      restartCount: 32,
      initialization: 'kmeans++' as const,
      output: {
        labels: createTransientView(graph, 'many-labels', 'uint32', 256),
        centers: createTransientView(graph, 'many-centers', 'float32x2', 256)
      }
    };
    expect(new GPUSpatialKMeans({...big, restartCount: 23}).estimatedCommandNodeCount).toBe(32640);
    // The next restart is refused at construction, before any command nodes or scratch exist.
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    allocate.mockClear();
    for (const restartCount of [24, 32]) {
      expect(() => new GPUSpatialKMeans({...big, restartCount})).toThrow(
        /^kmeans estimated command node count must be at most 32768$/
      );
    }
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('failed study-window binding preflights leave graph, ownership and persistent buffers untouched', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const geometry = {
      positions: createTransientView(graph, 'positions', 'float32x2', 8),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2)
    };
    const producer = new GPUStudyWindowPreparation({
      ...geometry,
      samplingGrid: [2, 2],
      slabEntryCapacity: 1
    });
    Object.assign(device.limits, {maxStorageBufferBindingSize: 128});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createBuffer = vi.spyOn(device, 'createBuffer');
    expect(() => producer.getCommandNodes(graph)).toThrow(
      /study-window.*(ring terms|packed|edges)/
    );
    expect(allocate).not.toHaveBeenCalled();
    expect(createBuffer).not.toHaveBeenCalled();
    expect(producer.isDeclaredIn(graph)).toBe(false);
    Object.assign(device.limits, {maxStorageBufferBindingSize: 134217728});
    expect(() => producer.getCommandNodes(graph)).not.toThrow();
    expect(producer.isDeclaredIn(graph)).toBe(true);
    producer.destroy();
  } finally {
    device.destroy();
  }
});

it('keeps k-means graph estimates above real node counts for seeds, iterations and restarts', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const [clusters, iterations, restarts] of [
      [3, 2, 1],
      [32, 4, 3],
      [256, 2, 4]
    ]) {
      const graph = new GPUCommandGraph(device);
      const contributor = new GPUSpatialKMeans({
        positions: createTransientView(graph, 'positions', 'float32x2', 513),
        clusterCount: clusters,
        maximumIterations: iterations,
        restartCount: restarts,
        initialization: 'kmeans++',
        parameters: createTransientView(graph, 'parameters', 'uint32', 2),
        output: {
          labels: createTransientView(graph, 'labels', 'uint32', 513),
          centers: createTransientView(graph, 'centers', 'float32x2', clusters)
        }
      });
      expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
        contributor.estimatedCommandNodeCount
      );
    }
  } finally {
    device.destroy();
  }
});

it('preflights envelope subviews before recording deviations and rejects raw status aliases', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const storage = createTransientView(graph, 'storage', 'float32', 20);
    const output = createTransientView(graph, 'output', 'float32', 4);
    const props = {
      observed: createTransientView(graph, 'observed', 'float32', 4),
      simulations: graph.createDataView(storage.buffer, {
        format: 'float32' as const,
        length: 8,
        byteOffset: 48
      }),
      reference: createTransientView(graph, 'reference', 'float32', 4),
      integrationWeights: createTransientView(graph, 'weights', 'float32', 4),
      simulationCount: 2,
      nullModel: 'independent',
      fittedParameters: 'none' as const,
      output: {
        lower: output,
        upper: createTransientView(graph, 'upper', 'float32', 4),
        statistics: createTransientView(graph, 'statistics', 'float32', 4),
        status: createTransientView(graph, 'status', 'uint32', 1)
      }
    };
    expect(
      () =>
        new GPUPointPatternEnvelope({
          ...props,
          output: {
            ...props.output,
            status: graph.createDataView(output.buffer, {format: 'uint32', length: 1})
          }
        })
    ).toThrow(/overlap/);
    const contributor = new GPUPointPatternEnvelope(props);
    Object.assign(device.limits, {maxStorageBufferBindingSize: 64});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/simulations.*80.*64/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('rejects MST output aliases and validates aligned inputs before standardization scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const values = createTransientView(graph, 'values-storage', 'float32', 24);
    const labels = createTransientView(graph, 'labels', 'uint32', 8);
    const props = {
      values: graph.createDataView(values.buffer, {
        format: 'float32' as const,
        length: 8,
        byteOffset: 64
      }),
      weights: {
        offsets: createTransientView(graph, 'offsets', 'uint32', 9),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 8),
        weights: createTransientView(graph, 'weights', 'float32', 8)
      },
      output: {
        treeEdgeFlags: createTransientView(graph, 'flags', 'uint32', 8),
        componentLabels: labels
      }
    };
    expect(
      () =>
        new GPUSpatialWeightsMinimumSpanningTree({
          ...props,
          output: {treeEdgeFlags: labels, componentLabels: labels}
        })
    ).toThrow(/overlap/);
    const contributor = new GPUSpatialWeightsMinimumSpanningTree(props);
    Object.assign(device.limits, {maxStorageBufferBindingSize: 64});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/values.*96.*64/);
    expect(allocate).not.toHaveBeenCalled();
    Object.assign(device.limits, {maxStorageBufferBindingSize: 134217728});
    expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
      contributor.estimatedCommandNodeCount
    );
  } finally {
    device.destroy();
  }
});

it('checks the combined packed window before allocating and rejects fractional slab capacities at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = {
      positions: createTransientView(graph, 'positions', 'float32x2', 8),
      featureOffsets: createTransientView(graph, 'feature-offsets', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygon-offsets', 'uint32', 2),
      ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 2),
      samplingGrid: [2, 2] as const
    };
    expect(() => new GPUStudyWindowPreparation({...props, slabEntryCapacity: 1.5})).toThrow(
      /slabEntryCapacity.*integer/
    );
    const producer = new GPUStudyWindowPreparation(props);
    Object.assign(device.limits, {maxStorageBufferBindingSize: 512});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createBuffer = vi.spyOn(device, 'createBuffer');
    expect(() => producer.getCommandNodes(graph)).toThrow(/packed window/);
    expect(allocate).not.toHaveBeenCalled();
    expect(createBuffer).not.toHaveBeenCalled();
    expect(producer.isDeclaredIn(graph)).toBe(false);
  } finally {
    device.destroy();
  }
});

it('accepts disjoint REDCAP and HDBSCAN outputs sharing an aligned binding prefix across separate passes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `packed-${serial++}`, format, length);
  try {
    const combined = view('uint32', 16);
    const labels = graph.createDataView(combined.buffer, {format: 'uint32', length: 8});
    const regionCount = graph.createDataView(combined.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 32
    });
    const redcap = new GPURedcapSingleLinkage({
      weights: {
        offsets: view('uint32', 9),
        neighbors: view('uint32', 8),
        weights: view('float32', 8)
      },
      values: view('float32', 8),
      parameters: view('uint32', 1),
      output: {
        labels,
        regionCount,
        // iterationCount is published in its own pass, but remains a retained logical output.
        status: {iterationCount: view('uint32', 1)}
      }
    });
    expect(
      () =>
        new GPURedcapSingleLinkage({
          ...redcap.props,
          output: {
            ...redcap.props.output,
            status: {
              iterationCount: graph.createDataView(labels.buffer, {format: 'uint32', length: 1})
            }
          }
        })
    ).toThrow(/must not overlap writable byte ranges/);
    const density = new GPUHDBSCAN({
      positions: view('float32x2', 8),
      minimumClusterSize: 2,
      output: {
        labels,
        coreDistances: graph.createDataView(combined.buffer, {
          format: 'float32',
          length: 8,
          byteOffset: 32
        })
      }
    });
    for (const contributor of [redcap, density]) {
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeGreaterThan(0);
      if (contributor === redcap)
        expect(nodes.some(node => node.id.endsWith('-iteration-count'))).toBe(true);
      for (const node of nodes) {
        const boundOutputs =
          node.resources?.filter(
            resource =>
              'buffer' in resource &&
              'buffer' in resource.buffer &&
              resource.buffer.buffer === combined.buffer
          ) ?? [];
        expect(boundOutputs.length).toBeLessThanOrEqual(1);
      }
    }
  } finally {
    device.destroy();
  }
});

it('preflights REDCAP and partition input STORAGE usage and aligned ranges before graph mutation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `guard-${serial++}`, format, length);
  try {
    const storage = view('float32', 24);
    const values = graph.createDataView(storage.buffer, {
      format: 'float32',
      length: 8,
      byteOffset: 64
    });
    const contributors = [
      new GPURedcapSingleLinkage({
        values,
        parameters: view('uint32', 1),
        weights: {
          offsets: view('uint32', 9),
          neighbors: view('uint32', 8),
          weights: view('float32', 8)
        },
        output: {labels: view('uint32', 8), regionCount: view('uint32', 1)}
      }),
      new GPURegionPartitionEvaluation({
        values,
        labels: view('uint32', 8),
        output: {summary: view('float32', 9)}
      })
    ];
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    Object.assign(device.limits, {maxStorageBufferBindingSize: 64});
    for (const contributor of contributors) {
      expect(() => contributor.getCommandNodes(graph)).toThrow(/values.*96.*64/);
      expect(allocate).not.toHaveBeenCalled();
    }
    Object.assign(device.limits, {maxStorageBufferBindingSize: 134217728});
    Object.defineProperty(storage.buffer, 'usage', {value: Buffer.COPY_DST});
    for (const contributor of contributors) {
      expect(() => contributor.getCommandNodes(graph)).toThrow(/STORAGE/);
      expect(allocate).not.toHaveBeenCalled();
    }
  } finally {
    device.destroy();
  }
});

it('preflights border terms and prepared window STORAGE usage before intensity nodes or scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `pattern-${serial++}`, format, length);
  const studyWindow = new GPUStudyWindowPreparation({
    positions: view('float32x2', 8),
    featureOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    ringOffsets: view('uint32', 2),
    samplingGrid: [2, 2],
    slabEntryCapacity: 1
  });
  try {
    studyWindow.getCommandNodes(graph);
    const windowView = studyWindow.getViews(graph).packed;
    const contributor = new GPUInhomogeneousPointPattern({
      studyWindow,
      positions: view('float32x2', 32),
      radii: view('float32', 256),
      suppliedIntensity: view('float32', 32),
      output: {
        intensity: view('float32', 32),
        k: view('float32', 256),
        pairCorrelation: view('float32', 256),
        status: view('uint32', 256)
      }
    });
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    Object.assign(device.limits, {
      maxStorageBufferBindingSize: Math.max(1024, getViewBindingRange(windowView).size)
    });
    expect(() => contributor.getCommandNodes(graph)).toThrow(/border terms/);
    expect(allocate).not.toHaveBeenCalled();
    Object.assign(device.limits, {maxStorageBufferBindingSize: 134217728});
    Object.defineProperty(windowView.buffer, 'usage', {value: Buffer.COPY_DST});
    expect(() => contributor.getCommandNodes(graph)).toThrow(/STORAGE/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    studyWindow.destroy();
    device.destroy();
  }
});

it('accepts packed MST outputs and separate-pass k-means status/inertia without weakening co-bound checks', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const combined = createTransientView(graph, 'packed-outputs', 'uint32', 256);
  const unsigned = (byteOffset: number, length: number) =>
    graph.createDataView(combined.buffer, {format: 'uint32', length, byteOffset});
  const floating = (byteOffset: number, length: number) =>
    graph.createDataView(combined.buffer, {format: 'float32', length, byteOffset});
  try {
    const spanningTree = new GPUSpatialWeightsMinimumSpanningTree({
      values: createTransientView(graph, 'values', 'float32', 8),
      weights: {
        offsets: createTransientView(graph, 'offsets', 'uint32', 9),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 8),
        weights: createTransientView(graph, 'weights', 'float32', 8)
      },
      output: {
        treeEdgeFlags: unsigned(0, 8),
        componentLabels: unsigned(32, 8),
        standardizedValues: floating(64, 8),
        status: unsigned(96, GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.length)
      }
    });
    expect(() => spanningTree.getCommandNodes(graph)).not.toThrow();
    const kmeans = new GPUSpatialKMeans({
      positions: createTransientView(graph, 'positions', 'float32x2', 8),
      parameters: createTransientView(graph, 'parameters', 'uint32', 2),
      clusterCount: 2,
      maximumIterations: 2,
      output: {
        labels: unsigned(0, 8),
        // Centers are read while labels are written: they need a separate physical buffer.
        centers: createTransientView(graph, 'packed-centers', 'float32x2', 2),
        squaredDistances: floating(512, 8),
        inertia: floating(544, 1),
        status: {iterationCount: unsigned(548, 1), converged: unsigned(552, 1)}
      }
    });
    expect(() => kmeans.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('accepts separate-pass packed Dorling metrics and preflights STORAGE usage before scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const combined = createTransientView(graph, 'dorling-packed', 'uint32', 256);
    const positions = createTransientView(graph, 'dorling-positions', 'float32x2', 8);
    const contributor = new GPUDorlingCartogram({
      positions,
      values: createTransientView(graph, 'dorling-values', 'float32', 8),
      maximumIterations: 0,
      output: {
        positions: graph.createDataView(combined.buffer, {format: 'float32x2', length: 8}),
        radii: graph.createDataView(combined.buffer, {
          format: 'float32',
          length: 8,
          byteOffset: 256
        }),
        sourceIds: graph.createDataView(combined.buffer, {
          format: 'uint32',
          length: 8,
          byteOffset: 512
        }),
        metrics: graph.createDataView(combined.buffer, {
          format: 'float32x2',
          length: 1,
          byteOffset: 544
        }),
        status: graph.createDataView(combined.buffer, {
          format: 'uint32',
          length: 4,
          byteOffset: 768
        })
      }
    });
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    Object.defineProperty(positions.buffer, 'usage', {value: Buffer.COPY_DST});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/STORAGE/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('rejects non-STORAGE DBSCAN and HDBSCAN inputs before their nested index builders allocate', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const positions = createTransientView(graph, 'non-storage-positions', 'float32x2', 8);
    const labels = createTransientView(graph, 'non-storage-labels', 'uint32', 8);
    const contributors = [
      new GPUDBSCAN({
        positions,
        parameters: createTransientView(graph, 'non-storage-parameters', 'float32', 10),
        gridSize: [2, 2],
        output: {labels}
      }),
      new GPUHDBSCAN({positions, minimumClusterSize: 2, output: {labels}})
    ];
    Object.defineProperty(positions.buffer, 'usage', {value: Buffer.COPY_DST});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    for (const contributor of contributors) {
      expect(() => contributor.getCommandNodes(graph)).toThrow(/STORAGE/);
      expect(allocate).not.toHaveBeenCalled();
    }
  } finally {
    device.destroy();
  }
});

it('accepts valid packed layouts for partition evaluation, envelopes and border point patterns', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `valid-${serial++}`, format, length);
  const storage = view('uint32', 256);
  const floating = (byteOffset: number, length: number) =>
    graph.createDataView(storage.buffer, {format: 'float32', byteOffset, length});
  const unsigned = (byteOffset: number, length: number) =>
    graph.createDataView(storage.buffer, {format: 'uint32', byteOffset, length});
  const studyWindow = new GPUStudyWindowPreparation({
    id: 'valid-window',
    positions: view('float32x2', 8),
    featureOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    ringOffsets: view('uint32', 2),
    samplingGrid: [2, 2],
    slabEntryCapacity: 1
  });
  try {
    const partition = new GPURegionPartitionEvaluation({
      values: view('float32', 8),
      labels: view('uint32', 8),
      output: {
        regionSizes: unsigned(0, 8),
        regionWithinSumOfSquares: floating(32, 8),
        summary: floating(64, 9),
        regionMeans: floating(256, 8),
        invalidRowCount: unsigned(512, 1)
      }
    });
    expect(() => partition.getCommandNodes(graph)).not.toThrow();
    const envelope = new GPUPointPatternEnvelope({
      observed: view('float32', 4),
      simulations: view('float32', 8),
      reference: view('float32', 4),
      integrationWeights: view('float32', 4),
      simulationCount: 2,
      nullModel: 'independent',
      fittedParameters: 'none',
      output: {
        lower: floating(0, 4),
        statistics: floating(16, 4),
        upper: floating(256, 4),
        status: view('uint32', 1)
      }
    });
    expect(() => envelope.getCommandNodes(graph)).not.toThrow();
    studyWindow.getCommandNodes(graph);
    const pattern = new GPUInhomogeneousPointPattern({
      positions: view('float32x2', 8),
      radii: view('float32', 4),
      suppliedIntensity: view('float32', 8),
      studyWindow,
      output: {
        intensity: view('float32', 8),
        k: floating(256, 4),
        status: unsigned(512, 4),
        pairCorrelation: view('float32', 4)
      }
    });
    expect(() => pattern.getCommandNodes(graph)).not.toThrow();
  } finally {
    studyWindow.destroy();
    device.destroy();
  }
});

it('distinguishes writable packing, separate dispatches and whole-buffer read/write usage scopes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const storage = createTransientView(graph, 'usage-scope-storage', 'uint32', 256);
    const first = graph.createDataView(storage.buffer, {format: 'uint32', length: 8});
    const adjacent = graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: 8,
      byteOffset: 32
    });
    const aligned = graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: 8,
      byteOffset: 256
    });
    expect(() =>
      validateSpatialOutputViews('packing', {first, aligned}, [[first, aligned]])
    ).not.toThrow();
    expect(() =>
      validateSpatialOutputViews('packing', {first, adjacent}, [[first], [adjacent]])
    ).not.toThrow();
    expect(() =>
      validateSpatialOutputViews('packing', {first, adjacent}, [[first, adjacent]])
    ).toThrow(/aligned binding/);
    expect(() =>
      validateSpatialOutputViews('packing', {first, aligned}, [{read: [first], write: [aligned]}])
    ).toThrow(/physical buffers/);
    expect(() =>
      validateSpatialOutputViews('packing', {first, adjacent}, [
        {read: [first, adjacent], write: []}
      ])
    ).not.toThrow();

    const positions = createTransientView(graph, 'usage-positions', 'float32x2', 8);
    const sharedCenters = graph.createDataView(storage.buffer, {
      format: 'float32x2',
      length: 2,
      byteOffset: 256
    });
    const parameters = createTransientView(graph, 'usage-parameters', 'uint32', 2);
    // The historical uncentered route assigns against the caller's centers while writing labels.
    expect(
      () =>
        new GPUSpatialKMeans({
          positions,
          parameters,
          clusterCount: 2,
          maximumIterations: 2,
          centerCoordinates: false,
          output: {labels: first, centers: sharedCenters}
        })
    ).toThrow(/physical buffers/);
    // The default centered route assigns against scratch centers and only writes the caller's
    // centers in a later dispatch, so the same layout is accepted.
    expect(
      () =>
        new GPUSpatialKMeans({
          positions,
          parameters,
          clusterCount: 2,
          maximumIterations: 2,
          output: {labels: first, centers: sharedCenters}
        })
    ).not.toThrow();
    expect(
      () =>
        new GPUDBSCAN({
          positions,
          parameters: createTransientView(graph, 'usage-density-parameters', 'float32', 10),
          gridSize: [2, 2],
          output: {
            coreFlags: first,
            rootRows: aligned,
            labels: createTransientView(graph, 'usage-labels', 'uint32', 8)
          }
        })
    ).toThrow(/physical buffers/);
    // HDBSCAN publishes its count while selecting clusters, then writes labels in a separate pass.
    expect(
      () =>
        new GPUHDBSCAN({
          positions,
          minimumClusterSize: 2,
          output: {
            labels: first,
            status: {
              count: graph.createDataView(storage.buffer, {
                format: 'uint32',
                length: 1,
                byteOffset: 256
              })
            }
          }
        })
    ).not.toThrow();
    const dorlingOutput = {
      positions: graph.createDataView(storage.buffer, {format: 'float32x2' as const, length: 8}),
      radii: createTransientView(graph, 'usage-radii', 'float32', 8),
      sourceIds: createTransientView(graph, 'usage-source-ids', 'uint32', 8),
      metrics: createTransientView(graph, 'usage-metrics', 'float32x2', 1),
      status: graph.createDataView(storage.buffer, {
        format: 'uint32' as const,
        length: 4,
        byteOffset: 256
      })
    };
    const dorlingProps = {
      positions,
      values: createTransientView(graph, 'usage-values', 'float32', 8),
      output: dorlingOutput
    };
    // Zero iterations only initializes/diagnoses these outputs. A relaxation commits positions
    // while reading status, making the same physical packing invalid even at aligned offsets.
    expect(() => new GPUDorlingCartogram({...dorlingProps, maximumIterations: 0})).not.toThrow();
    expect(() => new GPUDorlingCartogram({...dorlingProps, maximumIterations: 1})).toThrow(
      /physical buffers/
    );
  } finally {
    device.destroy();
  }
});

it('accepts zero-byte unused DBSCAN views while preflighting every bound empty-route output', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `empty-${serial++}`, format, length);
  const empty = <Format extends 'uint32' | 'float32x2'>(format: Format) =>
    graph.createDataView(
      graph.createTransientBuffer({id: `zero-${serial++}`, byteLength: 0, usage: Buffer.STORAGE}),
      {format, length: 0}
    );
  try {
    const parameters = view('float32', 10);
    Object.defineProperty(parameters.buffer, 'usage', {value: Buffer.COPY_DST});
    const positions = empty('float32x2');
    const emptyRows = graph.createDataView(positions.buffer, {format: 'uint32', length: 0});
    const packed = view('uint32', 12);
    const output = {
      labels: emptyRows,
      rootRows: emptyRows,
      coreFlags: emptyRows,
      clusterCount: view('uint32', 1),
      clusters: {
        // ids is unused on this route and may alias a written field.
        ids: graph.createDataView(packed.buffer, {format: 'uint32', length: 4}),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1),
          requiredCount: view('uint32', 1)
        }
      },
      clusterSizes: graph.createDataView(packed.buffer, {format: 'uint32', length: 4}),
      clusterCentroids: graph.createDataView(packed.buffer, {
        format: 'float32x2',
        length: 4,
        byteOffset: 16
      })
    };
    const props = {
      positions,
      sourceIds: emptyRows,
      parameters,
      gridSize: [2, 2] as const,
      output
    };
    const nodes = new GPUDBSCAN(props).getCommandNodes(graph);
    expect(nodes.map(node => node.id)).toEqual([
      'spatial-clustering-cluster-count',
      'spatial-clustering-cluster-sizes',
      'spatial-clustering-cluster-centroids',
      'spatial-clustering-publish'
    ]);
    const unused = new Set<GraphDataView>([
      props.positions,
      props.sourceIds,
      parameters,
      output.labels,
      output.rootRows,
      output.coreFlags,
      output.clusters.ids
    ]);
    for (const node of nodes)
      for (const resource of node.resources ?? []) {
        if ('buffer' in resource) expect(unused.has(resource.buffer as GraphDataView)).toBe(false);
      }
    const memory = new Map<typeof output.clusterCount.buffer, ArrayBuffer>();
    for (const node of nodes) {
      const record = getWGSLKernelNodeRecord(graph, node.id)!;
      expect(auditWGSLModule(record.source, record.constants)).toEqual([]);
      const bindings = Object.fromEntries(
        record.bindings.map((binding, index) => {
          const {view} = binding;
          let bytes = memory.get(view.buffer);
          if (!bytes) {
            bytes = new ArrayBuffer(view.buffer.byteLength);
            new Uint32Array(bytes).fill(0x7f7f7f7f);
            memory.set(view.buffer, bytes);
          }
          const range = getViewBindingRange(view);
          return [
            index,
            binding.type === 'f32'
              ? new Float32Array(bytes, range.offset, range.size / 4)
              : new Uint32Array(bytes, range.offset, range.size / 4)
          ];
        })
      );
      // WgslExec needs the equivalent three-axis attribute; it treats a scalar string as characters.
      new WgslExec(
        new WgslParser().parse(
          record.source
            .replace(/^\s*_\s*=[^;]*;/gm, '')
            .replace(/@workgroup_size\((\d+)\)/g, '@workgroup_size($1, 1, 1)')
        )
      ).dispatchWorkgroups(
        'main',
        [...record.workgroups],
        {0: bindings},
        {constants: {...record.constants}}
      );
    }
    for (const field of [
      output.clusterCount,
      output.clusterSizes,
      output.clusterCentroids,
      output.clusters.status.count,
      output.clusters.status.overflow,
      output.clusters.status.requiredCount
    ]) {
      expect(
        [
          ...new Uint32Array(
            memory.get(field.buffer)!,
            field.byteOffset,
            (field.length * field.rowByteLength) / 4
          )
        ],
        field.buffer.id
      ).toEqual(new Array((field.length * field.rowByteLength) / 4).fill(0));
    }
    for (const field of [
      output.clusterCount,
      output.clusterSizes,
      output.clusterCentroids,
      output.clusters.status.count,
      output.clusters.status.overflow,
      output.clusters.status.requiredCount
    ]) {
      const usage = field.buffer.usage;
      Object.defineProperty(field.buffer, 'usage', {value: Buffer.COPY_DST, configurable: true});
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const createView = vi.spyOn(graph, 'createDataView');
      expect(() => new GPUDBSCAN(props).getCommandNodes(graph)).toThrow(/Buffer.STORAGE/);
      expect(allocate).not.toHaveBeenCalled();
      expect(createView).not.toHaveBeenCalled();
      allocate.mockRestore();
      createView.mockRestore();
      Object.defineProperty(field.buffer, 'usage', {value: usage, configurable: true});
    }
  } finally {
    device.destroy();
  }
});

it('does not bind or preflight DBSCAN sourceIds when compact outputs are absent', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const sourceIds = createTransientView(graph, 'unused-source-ids', 'uint32', 1);
    Object.defineProperty(sourceIds.buffer, 'usage', {value: Buffer.COPY_DST});
    const nodes = new GPUDBSCAN({
      positions: createTransientView(graph, 'one-position', 'float32x2', 1),
      parameters: createTransientView(graph, 'one-parameters', 'float32', 10),
      sourceIds,
      gridSize: [2, 2],
      output: {labels: createTransientView(graph, 'one-label', 'uint32', 1)}
    }).getCommandNodes(graph);
    expect(nodes.length).toBeGreaterThan(0);
    for (const node of nodes)
      for (const resource of node.resources ?? []) {
        if ('buffer' in resource) expect(resource.buffer).not.toBe(sourceIds);
      }
  } finally {
    device.destroy();
  }
});
