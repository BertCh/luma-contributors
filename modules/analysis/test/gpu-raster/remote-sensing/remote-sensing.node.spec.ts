// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';

import {expect, it} from 'vitest';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPURasterConservativeRemapping,
  GPURasterTexture,
  GPURasterTemporal,
  GPURasterRegistration,
  GPURasterRegionGraph,
  GPURasterSpectralUnmixing,
  GPURasterSeasonalBreak,
  addRasterRegionGraphTiles
} from '../../../src/raster-analysis/remote-sensing/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

it('RS03/04/06/07/08/10/11 construct native graph kernels within eight storage bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(format: Format, length = 256) =>
    createTransientView(graph, `remote-${serial++}`, format, length);
  const values = view('float32');
  const validity = view('uint32');
  const contributors = [
    new GPURasterConservativeRemapping({
      source: {width: 4, height: 4, origin: [0, 0], spacing: [1, 1]},
      destination: {width: 2, height: 2, origin: [0.25, 0.25], spacing: [2, 2]},
      values,
      mask: validity,
      normalization: 'extensive',
      output: {values: view('float32'), coveredArea: view('float32'), status: view('uint32')}
    }),
    new GPURasterTexture({
      grid: makeGPURasterMetadata({width: 4, height: 4}),
      values,
      mask: validity,
      levels: 32,
      range: [0, 1],
      radius: 8,
      offset: [-1, 1],
      symmetric: true,
      features: view('float32'),
      pairCount: view('uint32')
    }),
    ...(['mean', 'minimum', 'maximum', 'median', 'savitzky-golay'] as const).map(
      operation =>
        new GPURasterTemporal({
          id: operation,
          cellCount: 4,
          sliceCount: 16,
          values,
          mask: validity,
          operation,
          output: {
            values: view('float32'),
            counts: view('uint32'),
            chosenSlice: view('uint32'),
            status: view('uint32')
          }
        })
    ),
    new GPURasterRegistration({
      grid: makeGPURasterMetadata({width: 4, height: 4}),
      reference: values,
      moving: view('float32'),
      output: {shift: view('float32x2'), peak: view('float32'), status: view('uint32')}
    }),
    new GPURasterRegistration({
      id: 'masked',
      grid: makeGPURasterMetadata({width: 4, height: 3}),
      reference: values,
      moving: view('float32'),
      mode: 'masked-linear',
      referenceMask: validity,
      movingMask: view('uint32'),
      maximumShift: [2, 1],
      output: {
        shift: view('float32x2'),
        peak: view('float32'),
        status: view('uint32'),
        overlap: view('uint32')
      }
    }),
    new GPURasterRegionGraph({
      id: 'adaptive',
      grid: makeGPURasterMetadata({width: 4, height: 4}),
      values,
      labels: validity,
      regionCount: 4,
      mergeThreshold: 1,
      maximumIterations: 3,
      mergeMode: 'adaptive',
      output: {
        adjacency: view('uint32'),
        weights: view('float32'),
        mapping: view('uint32'),
        status: {iterationLimitReached: view('uint32')}
      }
    }),
    new GPURasterRegionGraph({
      grid: makeGPURasterMetadata({width: 4, height: 4}),
      values,
      labels: validity,
      regionCount: 4,
      mergeThreshold: 1,
      maximumIterations: 3,
      output: {
        adjacency: view('uint32'),
        weights: view('float32'),
        mapping: view('uint32'),
        status: {iterationLimitReached: view('uint32')}
      }
    }),
    ...(['unconstrained', 'nonnegative', 'simplex'] as const).map(
      constraint =>
        new GPURasterSpectralUnmixing({
          id: constraint,
          cellCount: 4,
          bandCount: 4,
          endmemberCount: 3,
          values,
          mask: validity,
          endmembers: view('float32'),
          constraint,
          output: {
            abundances: view('float32'),
            residual: view('float32'),
            status: view('uint32'),
            iterationCounts: view('uint32')
          }
        })
    ),
    new GPURasterSeasonalBreak({
      cellCount: 4,
      sliceCount: 16,
      values,
      mask: validity,
      period: 6,
      minimumSegmentLength: 4,
      output: {
        coefficients: view('float32'),
        components: view('float32'),
        breakIndex: view('uint32'),
        improvement: view('float32'),
        status: view('uint32')
      }
    })
  ];
  try {
    for (const contributor of contributors) {
      const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
      expect(sources.size, contributor.id).toBeGreaterThan(0);
      for (const source of sources.values()) {
        expect(getStorageBindingCount(source), contributor.id).toBeLessThanOrEqual(8);
        expect(auditWGSLModule(source), contributor.id).toEqual([]);
      }
    }
    expect(
      () =>
        new GPURasterTexture({...contributors[1].props, levels: 33} as ConstructorParameters<
          typeof GPURasterTexture
        >[0])
    ).toThrow();
  } finally {
    device.destroy();
  }
});

it('remote-sensing rejects unsupported topology, insufficient capacities, aliases and foreign graphs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const foreignGraph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(format: Format, length = 256) =>
    createTransientView(graph, `validation-${serial++}`, format, length);
  const values = view('float32');
  const validity = view('uint32');
  const texture = {
    grid: makeGPURasterMetadata({width: 4, height: 4}),
    values,
    mask: validity,
    levels: 2,
    range: [0, 1] as const,
    radius: 1,
    offset: [1, 0] as const,
    features: view('float32'),
    pairCount: view('uint32')
  };
  const temporal = {
    cellCount: 4,
    sliceCount: 16,
    values,
    mask: validity,
    operation: 'savitzky-golay' as const,
    output: {
      values: view('float32'),
      counts: view('uint32'),
      chosenSlice: view('uint32'),
      status: view('uint32')
    }
  };
  const unmixing = {
    cellCount: 4,
    bandCount: 4,
    endmemberCount: 3,
    values,
    mask: validity,
    endmembers: view('float32'),
    output: {
      abundances: view('float32'),
      residual: view('float32'),
      status: view('uint32'),
      iterationCounts: view('uint32')
    }
  };
  const registration = {
    grid: makeGPURasterMetadata({width: 4, height: 4}),
    reference: values,
    moving: view('float32'),
    output: {shift: view('float32x2'), peak: view('float32'), status: view('uint32')}
  };
  const regionGraph = {
    grid: makeGPURasterMetadata({width: 4, height: 4}),
    values,
    labels: validity,
    regionCount: 4,
    mergeThreshold: 1,
    maximumIterations: 3,
    output: {
      adjacency: view('uint32'),
      weights: view('float32'),
      mapping: view('uint32'),
      status: {iterationLimitReached: view('uint32')}
    }
  };
  const seasonal = {
    cellCount: 4,
    sliceCount: 16,
    values,
    mask: validity,
    period: 6,
    minimumSegmentLength: 4,
    output: {
      coefficients: view('float32'),
      components: view('float32'),
      breakIndex: view('uint32'),
      improvement: view('float32'),
      status: view('uint32')
    }
  };
  const remapping = {
    source: {width: 4, height: 4, origin: [0, 0] as const, spacing: [1, 1] as const},
    destination: {width: 2, height: 2, origin: [0, 0] as const, spacing: [2, 2] as const},
    values,
    mask: validity,
    normalization: 'covered-area' as const,
    output: {values: view('float32'), coveredArea: view('float32'), status: view('uint32')}
  };
  try {
    for (const levels of [0, 1, 33, 2.5])
      expect(() => new GPURasterTexture({...texture, levels})).toThrow();
    expect(() => new GPURasterTexture({...texture, offset: [0, 0]})).toThrow();
    expect(() => new GPURasterTexture({...texture, features: values})).toThrow();
    expect(() => new GPURasterTexture({...texture, features: view('float32', 3)})).toThrow();
    for (const windowLength of [2, 4, 17])
      expect(() => new GPURasterTemporal({...temporal, windowLength})).toThrow();
    expect(() => new GPURasterTemporal({...temporal, polynomialOrder: 4})).toThrow();
    expect(
      () => new GPURasterTemporal({...temporal, output: {...temporal.output, values}})
    ).toThrow();
    expect(() => new GPURasterTemporal({...temporal, values: view('float32', 3)})).toThrow();
    expect(() => new GPURasterSpectralUnmixing({...unmixing, endmemberCount: 5})).toThrow();
    expect(() => new GPURasterSpectralUnmixing({...unmixing, tolerance: 0})).toThrow();
    expect(() => new GPURasterSpectralUnmixing({...unmixing, maximumIterations: 4097})).toThrow();
    expect(
      () =>
        new GPURasterRegistration({
          ...registration,
          grid: makeGPURasterMetadata({width: 3, height: 4})
        })
    ).toThrow();
    expect(() => new GPURasterRegistration({...registration, upsampleFactor: 33})).toThrow();
    expect(
      () =>
        new GPURasterRegistration({...registration, mode: 'masked-linear', maximumShift: [4, 1]})
    ).toThrow();
    expect(
      () =>
        new GPURasterRegistration({
          ...registration,
          mode: 'masked-linear',
          minimumOverlapFraction: 1.1
        })
    ).toThrow();
    expect(
      () =>
        new GPURasterRegistration({...registration, mode: 'masked-linear', minimumOverlapCount: 2})
    ).toThrow();
    expect(
      () =>
        new GPURasterRegistration({...registration, mode: 'masked-linear', minimumCorrelation: -2})
    ).toThrow();
    expect(
      () =>
        new GPURasterRegistration({
          ...registration,
          mode: 'masked-linear',
          grid: makeGPURasterMetadata({width: 513, height: 4})
        })
    ).toThrow();

    expect(
      () =>
        new GPURasterRegistration({...registration, output: {...registration.output, peak: values}})
    ).toThrow();
    expect(() => new GPURasterRegionGraph({...regionGraph, maximumIterations: 0})).toThrow();
    expect(() => new GPURasterRegionGraph({...regionGraph, mergeThreshold: -1})).toThrow();
    expect(() => new GPURasterSeasonalBreak({...seasonal, minimumSegmentLength: 2})).toThrow();
    expect(() => new GPURasterSeasonalBreak({...seasonal, period: 2})).toThrow();
    expect(
      () =>
        new GPURasterConservativeRemapping({
          ...remapping,
          destination: {...remapping.destination, spacing: [256, 256]}
        })
    ).toThrow();
    expect(
      () =>
        new GPURasterConservativeRemapping({
          ...remapping,
          source: {...remapping.source, spacing: [-1, 1]}
        })
    ).toThrow();
    expect(
      () =>
        new GPURasterConservativeRemapping({
          ...remapping,
          source: {...remapping.source, spacing: [1e-50, 1e20]},
          destination: {...remapping.destination, spacing: [1e-50, 1e20]}
        })
    ).toThrow();
    for (const contributor of [
      new GPURasterTexture(texture),
      new GPURasterTemporal(temporal),
      new GPURasterSpectralUnmixing(unmixing),
      new GPURasterRegistration(registration),
      new GPURasterRegionGraph(regionGraph),
      new GPURasterSeasonalBreak(seasonal),
      new GPURasterConservativeRemapping(remapping)
    ])
      expect(() => contributor.getCommandNodes(foreignGraph)).toThrow();
  } finally {
    device.destroy();
  }
});

it('RS03 explicit prepared weights enforce shape, ownership, and nonaliasing', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `prepared-${serial++}`, format, length);
  const props = {
    source: {width: 4, height: 4, origin: [0, 0] as const, spacing: [1, 1] as const},
    destination: {width: 2, height: 2, origin: [0, 0] as const, spacing: [2, 2] as const},
    values: view('float32', 16),
    normalization: 'extensive' as const,
    preparedWeights: {indices: view('uint32', 36), areas: view('float32', 36)},
    output: {values: view('float32', 4), coveredArea: view('float32', 4), status: view('uint32', 4)}
  };
  expect(
    new GPURasterConservativeRemapping({...props, weightMode: 'prepare'}).getCommandNodes(graph)
  ).toHaveLength(2);
  expect(
    new GPURasterConservativeRemapping({...props, weightMode: 'reuse'}).getCommandNodes(graph)
  ).toHaveLength(1);
  expect(
    () =>
      new GPURasterConservativeRemapping({
        ...props,
        preparedWeights: undefined,
        weightMode: 'reuse'
      })
  ).toThrow();
  expect(
    () =>
      new GPURasterConservativeRemapping({
        ...props,
        preparedWeights: {...props.preparedWeights, areas: view('float32', 35)}
      })
  ).toThrow();
  expect(() =>
    new GPURasterConservativeRemapping({
      ...props,
      values: props.preparedWeights.areas
    }).getCommandNodes(graph)
  ).toThrow();
});

it('RS08 tile assembly rejects overlap and incomplete core partitions', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `tile-${serial++}`, format, length);
  const props = {
    grid: makeGPURasterMetadata({width: 4, height: 2}),
    regionCount: 2,
    mergeThreshold: 1,
    maximumIterations: 2,
    output: {
      adjacency: view('uint32', 4),
      weights: view('float32', 4),
      mapping: view('uint32', 2),
      status: {iterationLimitReached: view('uint32', 1)}
    }
  };
  const tile = {
    grid: makeGPURasterMetadata({width: 2, height: 2}),
    origin: [0, 0] as const,
    labels: view('uint32', 4),
    values: view('float32', 4)
  };
  expect(() => addRasterRegionGraphTiles(graph, {...props, tiles: [tile]})).toThrow();
  expect(() => addRasterRegionGraphTiles(graph, {...props, tiles: [tile, tile]})).toThrow();
  expect(() =>
    addRasterRegionGraphTiles(graph, {...props, tiles: [tile, {...tile, origin: [3, 0]}]})
  ).toThrow();
  expect(() =>
    addRasterRegionGraphTiles(graph, {...props, tiles: [tile, {...tile, origin: [2, 0]}]})
  ).not.toThrow();
});

it('RS11 extended contracts reject missing diagnostics and excessive solver workspace/work', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `extended-${serial++}`, format, length);
  const props = {
    cellCount: 1,
    sliceCount: 256,
    period: 12,
    minimumSegmentLength: 32,
    seasonality: 'dummy' as const,
    monitoring: {historyLength: 128, threshold: 4, minimumConsecutive: 2},
    values: view('float32', 256),
    output: {
      coefficients: view('float32', 14),
      components: view('float32', 768),
      breakIndex: view('uint32', 1),
      improvement: view('float32', 1),
      status: view('uint32', 1),
      diagnostics: view('float32', 6)
    }
  };
  try {
    expect(() => new GPURasterSeasonalBreak(props).getCommandNodes(graph)).not.toThrow();
    const {diagnostics, ...withoutDiagnostics} = props.output;
    expect(diagnostics.length).toBe(6);
    expect(() => new GPURasterSeasonalBreak({...props, output: withoutDiagnostics})).toThrow();
    expect(() => new GPURasterSeasonalBreak({...props, monitoring: undefined})).toThrow();
    expect(
      () => new GPURasterSeasonalBreak({...props, fitMethod: 'iterative', maximumIterations: 256})
    ).toThrow();
    expect(() => new GPURasterSeasonalBreak({...props, period: 3.5})).toThrow();
    expect(
      () =>
        new GPURasterSeasonalBreak({
          ...props,
          monitoring: {...props.monitoring, historyLength: 256}
        })
    ).toThrow();
  } finally {
    device.destroy();
  }
});

it('STD-04 and STD-07 name the region graph status port and the input masks', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `std-${serial++}`, format, length);
  const regionGraph = {
    id: 'region-graph',
    grid: makeGPURasterMetadata({width: 2, height: 2}),
    values: view('float32', 4),
    labels: view('uint32', 4),
    regionCount: 2,
    mergeThreshold: 1,
    maximumIterations: 2,
    output: {
      adjacency: view('uint32', 4),
      weights: view('float32', 4),
      mapping: view('uint32', 2),
      status: {iterationLimitReached: view('uint32', 1)}
    }
  };
  expect(() => new GPURasterRegionGraph(regionGraph)).not.toThrow();
  expect(
    () =>
      new GPURasterRegionGraph({
        ...regionGraph,
        output: {
          ...regionGraph.output,
          status: {iterationLimitReached: view('float32', 1) as never}
        }
      })
  ).toThrow(/^region-graph iterationLimitReached must contain one uint32 row/);
  expect(
    () =>
      new GPURasterTexture({
        id: 'texture',
        grid: makeGPURasterMetadata({width: 2, height: 2}),
        values: view('float32', 4),
        mask: view('uint32', 3),
        levels: 2,
        range: [0, 1],
        radius: 1,
        offset: [1, 0],
        features: view('float32', 4 * 7),
        pairCount: view('uint32', 4)
      })
  ).toThrow(/^texture values and mask must hold/);
});
