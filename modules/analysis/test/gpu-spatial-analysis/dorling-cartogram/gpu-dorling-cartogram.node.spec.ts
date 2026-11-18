// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type ComputePass, type ComputePipeline} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, getViewBindingRange} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  GPUDorlingCartogram,
  MAXIMUM_GPU_DORLING_CARTOGRAM_ROWS,
  DORLING_NEIGHBOR_GRID_MINIMUM_ROWS
} from '../../../src/gpu-spatial-analysis/dorling-cartogram/gpu-dorling-cartogram';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getGPUCommandNodePipelineKeys,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function makeProps(rows = 2) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    graph,
    props: {
      positions: view('float32x2', rows),
      values: view('float32', rows),
      sourceIds: view('uint32', rows),
      output: {
        positions: view('float32x2', rows),
        radii: view('float32', rows),
        sourceIds: view('uint32', rows),
        metrics: view('float32x2', 1),
        status: view('uint32', 4)
      }
    }
  };
}

it('bounds rows and iteration budget and rejects mismatched or aliased views', () => {
  const {props} = makeProps();
  expect(() => new GPUDorlingCartogram({...props, maximumIterations: 257})).toThrow();
  expect(() => new GPUDorlingCartogram({...props, areaScale: Infinity})).toThrow();
  expect(() => new GPUDorlingCartogram({...props, overlapTolerance: -1})).toThrow();
  expect(
    () => new GPUDorlingCartogram({...props, output: {...props.output, positions: props.positions}})
  ).toThrow();
  const large = makeProps(65537);
  expect(() => new GPUDorlingCartogram({...large.props, neighborGrid: false})).toThrow(
    /positions rows.*65536/
  );
  expect(new GPUDorlingCartogram(large.props).neighborGrid).toBe(true);
});

it('parses all kernels within the eight storage binding device default', () => {
  for (const maximumIterations of [0, 1, 2]) {
    const {graph, props} = makeProps();
    const contributor = new GPUDorlingCartogram({...props, maximumIterations});
    const nodes = contributor.getCommandNodes(graph);
    const kernels = getParsedKernelSources(nodes);
    expect(kernels.size).toBeGreaterThan(0);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    for (const {id: name, source} of kernels.configurations) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
    }
  }
});

it('splits relax and diagnose kernels into row tiles that share one shader per pass', () => {
  const first = makeProps(10);
  const untiled = new GPUDorlingCartogram({...first.props, maximumIterations: 1}).getCommandNodes(
    first.graph
  );
  const second = makeProps(10);
  const tiled = new GPUDorlingCartogram({
    ...second.props,
    maximumIterations: 1,
    rowTileSize: 4
  }).getCommandNodes(second.graph);
  // Two tiled passes (one relax, one final diagnose) each gain two row dispatches and two GPU tile advances (10 rows in tiles of 4).
  expect(tiled.length).toBe(untiled.length + 2 * 4);
});

it('accepts empty topology and maximum supported row and iteration counts', () => {
  const empty = makeProps(0);
  expect(() =>
    new GPUDorlingCartogram({...empty.props, maximumIterations: 0}).getCommandNodes(empty.graph)
  ).not.toThrow();
  const maximum = makeProps(65536);
  expect(() => new GPUDorlingCartogram({...maximum.props, maximumIterations: 256})).not.toThrow();
});

it('selects the named grid cut-over and retains explicit reference selection', () => {
  for (const rows of [DORLING_NEIGHBOR_GRID_MINIMUM_ROWS - 1, DORLING_NEIGHBOR_GRID_MINIMUM_ROWS]) {
    const {props} = makeProps(rows);
    expect(new GPUDorlingCartogram(props).neighborGrid).toBe(
      rows >= DORLING_NEIGHBOR_GRID_MINIMUM_ROWS
    );
    expect(new GPUDorlingCartogram({...props, neighborGrid: false}).neighborGrid).toBe(false);
    expect(new GPUDorlingCartogram({...props, neighborGrid: true}).neighborGrid).toBe(true);
  }
});

it('checks device binding size before creating any scratch or command nodes', () => {
  const {graph, props} = makeProps(8388608);
  const contributor = new GPUDorlingCartogram(props);
  const createScratch = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /dorling-cartogram positions rows metric scratch.*134217728/
  );
  expect(createScratch).not.toHaveBeenCalled();
});

it('checks row tiling, command budget and uint32 metric addressing at construction', () => {
  const {props} = makeProps();
  for (const rowTileSize of [0, -1, NaN, 1.5]) {
    expect(() => new GPUDorlingCartogram({...props, rowTileSize})).toThrow(/rowTileSize/);
  }
  const large = makeProps(1000000);
  expect(() => new GPUDorlingCartogram({...large.props, rowTileSize: 1})).toThrow(
    /estimated command node count/
  );
  const tooLarge = makeProps(MAXIMUM_GPU_DORLING_CARTOGRAM_ROWS + 1);
  expect(() => new GPUDorlingCartogram(tooLarge.props)).toThrow(/positions rows/);
});

it('bounds actual indexed node counts and parses the shared grid kernels, including a million rows', () => {
  for (const rows of [1, 120, 400, 65536, 1000000]) {
    const {graph, props} = makeProps(rows);
    const contributor = new GPUDorlingCartogram({
      ...props,
      maximumIterations: 1,
      neighborGrid: true
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    const kernels = getParsedKernelSources(nodes);
    for (const {id: name, source} of kernels.configurations) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
    }
    expect(nodes.some(node => node.id.includes('cell-sort'))).toBe(true);
    expect(nodes.some(node => node.id.includes('grid-sort'))).toBe(false);
  }
});

it('admits thirty million packed rows when the device binding limit fits the metric layout', () => {
  const {graph, props} = makeProps(30000000);
  graph.device.limits.maxStorageBufferBindingSize = 536870912;
  const contributor = new GPUDorlingCartogram({...props, maximumIterations: 0});
  expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
    contributor.estimatedCommandNodeCount
  );
});

it('registers indexed graph bindings with reused reduction scratch and packed scalar outputs', () => {
  const {graph, props} = makeProps(400);
  expect(() =>
    graph.add(new GPUDorlingCartogram({...props, neighborGrid: true, maximumIterations: 2}))
  ).not.toThrow();
});

it('keeps shader and pipeline counts constant as row tile and round counts grow', () => {
  const counts: number[] = [];
  const shaders: number[] = [];
  for (const maximumIterations of [1, 3, 6]) {
    for (const rowTileSize of [64, 32, 16]) {
      const {graph, props} = makeProps(257);
      const nodes = new GPUDorlingCartogram({
        ...props,
        neighborGrid: true,
        maximumIterations,
        rowTileSize
      }).getCommandNodes(graph);
      counts.push(getGPUCommandNodePipelineKeys(nodes).size);
      shaders.push(
        new Set(getGPUCommandNodeKernelSources(nodes).map(kernel => kernel.source)).size
      );
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expect(
          Object.keys(kernel.constants).some(name => /TILE_ROW_BASE|ROUND|LEVEL/.test(name))
        ).toBe(false);
      }
    }
  }
  expect(new Set(counts).size).toBe(1);
  expect(new Set(shaders).size).toBe(1);
});

it('rejects aliases between every pair of writable output fields at construction', () => {
  const {graph, props} = makeProps(4);
  const names = ['positions', 'radii', 'sourceIds', 'metrics', 'status'] as const;
  for (let first = 0; first < names.length; first++) {
    for (let second = first + 1; second < names.length; second++) {
      const firstView = props.output[names[first]];
      const secondView = props.output[names[second]];
      const shared = graph.createTransientBuffer({
        id: `alias-${first}-${second}`,
        byteLength: Math.max(
          firstView.length * firstView.rowByteLength,
          secondView.length * secondView.rowByteLength
        ),
        usage: Buffer.STORAGE
      });
      const overlappingFirst = graph.createDataView(shared, {
        format: firstView.format,
        length: firstView.length
      });
      const overlapping = graph.createDataView(shared, {
        format: secondView.format,
        length: secondView.length,
        byteOffset: 0
      });
      expect(
        () =>
          new GPUDorlingCartogram({
            ...props,
            output: {
              ...props.output,
              [names[first]]: overlappingFirst,
              [names[second]]: overlapping
            }
          })
      ).toThrow(/must not overlap writable byte ranges/);
    }
  }
});

it('checks aligned ranges of every caller field before allocating scratch', () => {
  const names = [
    'positions',
    'values',
    'sourceIds',
    'output.positions',
    'output.radii',
    'output.sourceIds',
    'output.metrics',
    'output.status'
  ];
  for (const name of names) {
    const {graph, props} = makeProps(2);
    // A 256-byte alignment prefix makes the binding exceed 256 bytes even for tiny views.
    graph.device.limits.maxStorageBufferBindingSize = 256;
    const original = name.startsWith('output.')
      ? props.output[name.slice(7) as keyof typeof props.output]
      : props[name as 'positions' | 'values' | 'sourceIds'];
    const buffer = graph.createTransientBuffer({
      id: 'offset-buffer',
      byteLength: 512,
      usage: Buffer.STORAGE
    });
    const shifted = graph.createDataView(buffer, {
      format: original.format,
      length: original.length,
      byteOffset: 252
    });
    const shiftedProps = name.startsWith('output.')
      ? {...props, output: {...props.output, [name.slice(7)]: shifted}}
      : {...props, [name]: shifted};
    const contributor = new GPUDorlingCartogram(shiftedProps);
    const allocation = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    const createBuffer = vi.spyOn(graph.device, 'createBuffer');
    const fieldName =
      (
        {'output.positions': 'positionsOutput', 'output.sourceIds': 'sourceIdsOutput'} as Record<
          string,
          string
        >
      )[name] ?? name.replace('output.', '');
    const byteLength = 252 + original.length * original.rowByteLength;
    expect(getViewBindingRange(shifted).size).toBe(byteLength);
    expect(() => contributor.getCommandNodes(graph), name).toThrow(
      new RegExp(
        `dorling-cartogram ${fieldName} chunk 0 needs ${byteLength} bytes in one binding \\(limit 256\\)`
      )
    );
    expect(allocation).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    expect(createBuffer).not.toHaveBeenCalled();
  }
});

it('uses no WGSL reserved identifier in any generated shader', () => {
  for (const rows of [1, 255, 256, 257, 65536]) {
    const {graph, props} = makeProps(rows);
    const nodes = new GPUDorlingCartogram({
      ...props,
      neighborGrid: true,
      maximumIterations: 2,
      rowTileSize: Math.min(rows, 4096)
    }).getCommandNodes(graph);
    for (const [name, source] of getParsedKernelSources(nodes)) {
      const tokens =
        source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(
        tokens.filter(token => WGSL_RESERVED_WORDS.has(token)),
        name
      ).toEqual([]);
    }
  }
});

it('matches every latched dispatch triple to the original encode geometry, including two-dimensional commits', () => {
  for (const [rows, maximumDimension] of [
    [255, 65535],
    [256, 65535],
    [257, 65535],
    [65535, 128],
    [65536, 128],
    [1000000, 128],
    [8388480, 65535],
    [8388481, 65535]
  ]) {
    const {graph, props} = makeProps(rows);
    graph.device.limits.maxComputeWorkgroupsPerDimension = maximumDimension;
    const nodes = new GPUDorlingCartogram({
      ...props,
      neighborGrid: true,
      maximumIterations: 1
    }).getCommandNodes(graph);
    for (const node of nodes) {
      if (
        node.id.includes('round-') ||
        node.id.includes('relax-') ||
        node.id.includes('diagnose-') ||
        node.id.includes('reduce-metric-') ||
        node.id.includes('extract-metric-') ||
        node.id.includes('commit-')
      ) {
        if (!node.id.endsWith('-phase-gate')) expect(node.condition?.source, node.id).toBe('gpu');
      }
    }
    const sources = getParsedKernelSources(nodes);
    const triples = new Map<string, Map<number, number>>();
    for (const node of nodes.filter(node => node.id.endsWith('-phase-gate'))) {
      const values = new Map<number, number>();
      for (const match of sources
        .get(node.id)!
        .matchAll(/dispatch\[dispatchOffset \+ (\d+)u\] = (\d+)u;/g)) {
        if (!values.has(Number(match[1]))) values.set(Number(match[1]), Number(match[2]));
      }
      triples.set(node.id.replace('-phase-gate', '-dispatch'), values);
    }
    const device = createNullWebGPUDevice();
    Object.defineProperty(device, 'createComputePipeline', {
      value: () => ({hash: '', destroy() {}}) as unknown as ComputePipeline
    });
    for (const node of nodes) {
      if (node.type !== 'compute' || node.condition?.source !== 'gpu') continue;
      const dispatched: number[][] = [];
      const executable = node.compile({device});
      const context = {
        computePass: {
          setPipeline() {},
          setBindings() {},
          dispatch(x: number, y: number, z: number) {
            dispatched.push([x, y, z]);
          }
        } as unknown as ComputePass,
        getBuffer: () => ({}) as Buffer
      } as unknown as Parameters<typeof executable.encode>[0];
      executable.encode(context);
      const values = triples.get(node.condition.buffer.id)!;
      const base = (node.condition.byteOffset ?? 0) / 4;
      expect(dispatched, node.id).toEqual([
        [values.get(base), values.get(base + 1), values.get(base + 2)]
      ]);
    }
    device.destroy();
  }
});

it('gates all million-row iterative work and keeps the default budget within its estimate', () => {
  const {graph, props} = makeProps(1000000);
  const contributor = new GPUDorlingCartogram(props);
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  const phaseGates = nodes.filter(node => node.id.endsWith('-phase-gate'));
  expect(phaseGates).toHaveLength(101);
  const firstPhase = nodes.indexOf(phaseGates[0]);
  for (const node of nodes.slice(firstPhase)) {
    if (node.id.endsWith('-phase-gate')) continue;
    expect(node.condition?.source, node.id).toBe('gpu');
  }
});

it('dispatches the partial tail tile at its own workgroup count without a new pipeline', () => {
  const {graph, props} = makeProps(4097);
  const nodes = new GPUDorlingCartogram({
    ...props,
    neighborGrid: true,
    maximumIterations: 1,
    rowTileSize: 1024
  }).getCommandNodes(graph);
  const rowNodes = nodes.filter(node => /-relax-1-tile-\d+$/.test(node.id));
  expect(rowNodes.map(node => node.workload?.maximumInvocationCount)).toEqual([
    1024, 1024, 1024, 1024, 1
  ]);
  expect(rowNodes.map(node => node.workload?.maximumWorkgroupCount)).toEqual([4, 4, 4, 4, 1]);
  expect(getGPUCommandNodePipelineKeys(rowNodes).size).toBe(1);
  for (const kernel of getGPUCommandNodeKernelSources(rowNodes))
    expect(kernel.constants).toEqual({});
});

it('rejects aligned writable prefixes and physical aliases, and accepts separated aligned outputs', () => {
  const {graph, props} = makeProps(4);
  const shared = graph.createTransientBuffer({
    id: 'shared-output',
    byteLength: 512,
    usage: Buffer.STORAGE
  });
  const metrics = graph.createDataView(shared, {format: 'float32x2', length: 1});
  const statusWithOverlappingPrefix = graph.createDataView(shared, {
    format: 'uint32',
    length: 4,
    byteOffset: 8
  });
  expect(
    () =>
      new GPUDorlingCartogram({
        ...props,
        output: {...props.output, metrics, status: statusWithOverlappingPrefix}
      })
  ).toThrow(/aligned binding byte ranges/);
  const separatedStatus = graph.createDataView(shared, {
    format: 'uint32',
    length: 4,
    byteOffset: 256
  });
  expect(() =>
    graph.add(
      new GPUDorlingCartogram({
        ...props,
        maximumIterations: 1,
        output: {...props.output, metrics, status: separatedStatus}
      })
    )
  ).not.toThrow();

  const physical = graph.device.createBuffer({byteLength: 512, usage: Buffer.STORAGE});
  try {
    const importedMetrics = importGraphBuffer(graph, 'physical-metrics', physical, 'float32x2', 1);
    const importedStatus = graph.createDataView(importedMetrics.buffer, {
      format: 'uint32',
      length: 4
    });
    expect(
      () =>
        new GPUDorlingCartogram({
          ...props,
          output: {...props.output, metrics: importedMetrics, status: importedStatus}
        })
    ).toThrow(/metrics and status must not overlap writable byte ranges/);
    const importedSeparatedStatus = graph.createDataView(importedMetrics.buffer, {
      format: 'uint32',
      length: 4,
      byteOffset: 256
    });
    expect(() =>
      graph.add(
        new GPUDorlingCartogram({
          ...props,
          id: 'physical-packed',
          maximumIterations: 1,
          output: {...props.output, metrics: importedMetrics, status: importedSeparatedStatus}
        })
      )
    ).not.toThrow();
  } finally {
    physical.destroy();
  }
});

it('rejects nonfinite and out-of-range literal inputs before generating shaders', () => {
  const {props} = makeProps();
  for (const name of [
    'areaScale',
    'overlapTolerance',
    'attraction',
    'maximumIterations',
    'rowTileSize'
  ]) {
    for (const value of [NaN, Infinity, -Infinity, Number.MAX_VALUE]) {
      expect(() => new GPUDorlingCartogram({...props, [name]: value}), `${name}=${value}`).toThrow(
        new RegExp(name)
      );
    }
  }
  for (const value of [0, 2 ** -149, 1e12]) {
    const {graph, props: validProps} = makeProps();
    const nodes = new GPUDorlingCartogram({
      ...validProps,
      areaScale: value,
      overlapTolerance: value,
      attraction: 1,
      maximumIterations: 1
    }).getCommandNodes(graph);
    for (const source of getParsedKernelSources(nodes).values()) {
      for (const literal of source.matchAll(/:\s*f32\s*=\s*([-+0-9.eE]+)\s*;/g)) {
        expect(Number.isFinite(Math.fround(Number(literal[1]))), literal[0]).toBe(true);
      }
    }
  }
});

it('keeps the fixed million-row default-budget GPU allocation plan below 512 MiB', () => {
  const {graph, props} = makeProps(1000000);
  let allocatedBytes = 0;
  Object.defineProperty(graph.device, 'props', {
    value: {...graph.device.props, _cachePipelines: false, _cacheShaders: false}
  });
  // Compile the real allocation plan against inert buffers; never allocate row-sized host/GPU data.
  vi.spyOn(graph.device, 'createBuffer').mockImplementation(bufferProps => {
    const rawData = bufferProps instanceof ArrayBuffer || ArrayBuffer.isView(bufferProps);
    const byteLength = rawData
      ? bufferProps.byteLength
      : (bufferProps.byteLength ?? bufferProps.data?.byteLength ?? 0);
    allocatedBytes += byteLength;
    return {
      byteLength,
      usage: rawData ? Buffer.STORAGE : bufferProps.usage,
      write() {},
      destroy() {}
    } as unknown as Buffer;
  });
  vi.spyOn(graph.device, 'createComputePipeline').mockImplementation(
    () => ({hash: '', destroy() {}}) as unknown as ComputePipeline
  );
  graph.add(new GPUDorlingCartogram(props));
  const compiled = graph.compile();
  try {
    expect(allocatedBytes).toBeLessThan(512 * 1024 * 1024);
  } finally {
    compiled.destroy();
  }
});
