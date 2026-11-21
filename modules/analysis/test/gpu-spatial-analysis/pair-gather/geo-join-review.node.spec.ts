// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {readFileSync} from 'node:fs';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getStorageTopKNodeRecord} from '../../../src/analysis-core/kernels/storage-backed-top-k';
import {GPUPairGather} from '../../../src/gpu-spatial-analysis/pair-gather/index';
import {GPUBufferSelection} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import {
  GPUNearestFeatureJoin,
  GPUNearestFeatureWeights,
  GPUSpatialJoinPrepared,
  GPUSpatialPredicateJoin
} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const integers = (length = 1) =>
    createTransientView(graph, `integer-${serial++}`, 'uint32', length);
  const positions = (length = 1) =>
    createTransientView(graph, `positions-${serial++}`, 'float32x2', length);
  return {device, graph, integers, positions};
}

function execute(
  source: string,
  constants: Readonly<Record<string, number>>,
  buffers: Record<string, Uint32Array | Float32Array>
) {
  // Single-row, zero-offset fixtures; adapt only uniform layout for wgsl_reflect.
  source = source
    .replace(
      'var<uniform> chunkSettings: array<vec4<u32>, 4>',
      'var<storage, read> chunkSettings: array<u32>'
    )
    .replace(
      /chunkSettings\[(\d+)\]\[(\d+)\]/g,
      (_, row, column) => `chunkSettings[${Number(row) * 4 + Number(column)}]`
    )
    .replace(
      'var<uniform> graphChunkGroupParameters: GraphChunkGroupParameters',
      'var<storage, read> graphChunkGroupParameters: array<u32>'
    )
    .replace(
      'graphChunkGroupParameters.words[word / 4u][word % 4u]',
      'graphChunkGroupParameters[word]'
    );
  const parameters = new Uint32Array(256);
  parameters[1] = 1;
  parameters[2] = 1;
  const chunkSettings = new Uint32Array(16);
  chunkSettings[1] = 1;
  buffers = {chunkSettings, graphChunkGroupParameters: parameters, ...buffers};
  const bindings = Object.fromEntries(
    new WgslReflect(source).storage.map(binding => {
      const values = buffers[binding.name];
      return [
        binding.name,
        {
          buffer: {backing: values.buffer, byteLength: values.byteLength},
          offset: values.byteOffset,
          size: values.byteLength
        }
      ];
    })
  );
  type Pass = {
    setPipeline(value: object): void;
    setBindings(value: object): void;
    dispatch(count: number): void;
  };
  executeCommandNodesOnCPU(
    [
      {
        id: 'review-scalar',
        compile: () => ({
          encode: ({computePass}: {computePass: Pass}) => {
            computePass.setPipeline({source: source.replace(/^\s*_ = &\w+;$/gm, ''), constants});
            computePass.setBindings(bindings);
            computePass.dispatch(1);
          }
        })
      }
    ],
    createCPUGraphMemory()
  );
}

for (const kind of ['points', 'lines'] as const) {
  for (const side of ['left', 'right'] as const) {
    it(`R2 supports multipart contains and retains feature rows for nearest/prepared ${kind} on ${side}`, () => {
      const {device, graph, positions, integers} = createFixture();
      const geometry =
        kind === 'lines'
          ? {kind, positions: positions(4), lineOffsets: integers(3), featureOffsets: integers(2)}
          : {kind, positions: positions(2), featureOffsets: integers(2)};
      const point = {kind: 'points' as const, positions: positions()};
      try {
        const join = new GPUSpatialPredicateJoin({
          left: point,
          right: point,
          [side]: geometry,
          predicate: 'contains',
          output: {
            pairs: {
              leftRows: integers(2),
              rightRows: integers(2),
              status: {count: integers(), overflow: integers()}
            }
          }
        });
        expect(join.leftCount).toBe(1);
        expect(join.rightCount).toBe(1);
        const kernels = getGPUCommandNodeKernelSources(
          join.getCommandNodes(graph).filter(node => node.id === `${join.id}-exact`)
        );
        expect(kernels).toHaveLength(1);
        // Hand fixture: the multipoint {(0,0),(1,0)} contains (1,0). The two line parts
        // [(0,0),(1,0)] and [(1,0),(2,0)] contain their shared endpoint because its mod-2
        // boundary cancels. Conversely, the singleton (1,0) contains neither whole feature.
        const multipartPositions =
          kind === 'points' ? Float32Array.of(0, 0, 1, 0) : Float32Array.of(0, 0, 1, 0, 1, 0, 2, 0);
        const flags = new Uint32Array(2);
        execute(kernels[0].source, kernels[0].constants, {
          leftPositions: side === 'left' ? multipartPositions : Float32Array.of(1, 0),
          rightPositions: side === 'right' ? multipartPositions : Float32Array.of(1, 0),
          [`${side}FeatureOffsets`]: Uint32Array.of(0, 2),
          ...(kind === 'lines' ? {[`${side}RingOffsets`]: Uint32Array.of(0, 2, 4)} : {}),
          candidatePairs: Uint32Array.of(0, 0),
          flags,
          state: new Uint32Array(6)
        });
        expect(flags[0]).toBe(side === 'left' ? 1 : 0);
        const nearest = new GPUNearestFeatureJoin({
          queries: side === 'left' ? geometry : point,
          features: side === 'right' ? geometry : point,
          output: {
            neighborIds: integers(1),
            neighborCounts: integers(1),
            status: {overflow: integers()}
          }
        });
        expect(nearest.queryCount).toBe(1);
        expect(nearest.featureCount).toBe(1);
        expect(new GPUSpatialJoinPrepared({geometry}).featureCount).toBe(1);
      } finally {
        device.destroy();
      }
    });
  }
}

for (const byteOffset of [4, 256]) {
  for (const duplicateImport of [false, true]) {
    it(`R3 rejects read/write physical aliases at ${byteOffset}, duplicate import=${duplicateImport}`, () => {
      const {device, graph, positions, integers} = createFixture();
      const backing = device.createBuffer({byteLength: 512, usage: Buffer.STORAGE});
      const sourceIds = importGraphBuffer(graph, 'source-ids', backing, 'uint32', 1);
      const destination = duplicateImport
        ? importGraphBuffer(graph, 'row-destination', backing, 'uint32', 128)
        : graph.createDataView(sourceIds.buffer, {format: 'uint32', length: 128});
      const neighborRows = graph.createDataView(destination.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset
      });
      const props = {
        queries: {kind: 'points' as const, positions: positions()},
        features: {kind: 'points' as const, positions: positions(), sourceIds},
        output: {
          neighborIds: integers(),
          neighborRows,
          neighborCounts: integers(),
          status: {overflow: integers()}
        }
      };
      const declaration = vi.spyOn(graph, 'createTransientBuffer');
      try {
        expect(() => new GPUNearestFeatureJoin(props).getCommandNodes(graph)).toThrow(
          /physical buffer.*read-only and writable/
        );
        expect(declaration).not.toHaveBeenCalled();
      } finally {
        declaration.mockRestore();
        backing.destroy();
        device.destroy();
      }
    });
  }
}

it('R3 accepts separate source-ID and row-output buffers', () => {
  const {device, graph, positions, integers} = createFixture();
  try {
    expect(() =>
      new GPUNearestFeatureJoin({
        queries: {kind: 'points', positions: positions()},
        features: {kind: 'points', positions: positions(), sourceIds: integers()},
        output: {
          neighborIds: integers(),
          neighborRows: integers(),
          neighborCounts: integers(),
          status: {overflow: integers()}
        }
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('R4 gather retains its validated neighbor map and capacity after caller mutation', () => {
  const {device, graph, integers} = createFixture();
  const rows = integers();
  const replacementIds = integers();
  const neighbors = {rows, counts: integers(), capacity: 1};
  const props = {
    neighbors,
    output: {
      leftRows: integers(),
      rightRows: integers(),
      status: {count: integers(), overflow: integers()}
    }
  };
  const gather = new GPUPairGather(props);
  neighbors.capacity = 0;
  neighbors.rows = replacementIds;
  props.neighbors = {rows: replacementIds, counts: integers(), capacity: 1};
  try {
    const nodes = gather.getCommandNodes(graph).filter(node => /-(slot-mask|rows)$/.test(node.id));
    for (const node of nodes) {
      const bindings = (node.resources ?? [])
        .filter(resource => 'buffer' in resource)
        .map(resource => resource.buffer);
      expect(bindings).toContain(rows);
      expect(bindings).not.toContain(replacementIds);
    }
    expect(gather.props.neighbors?.capacity).toBe(1);
    expect(
      getGPUCommandNodeKernelSources(nodes)
        .map(kernel => kernel.source)
        .join('\n')
    ).not.toContain('CAPACITY: u32 = 0u');
  } finally {
    device.destroy();
  }
});

for (const sourceId of [100, 0xffffffff]) {
  it(`R5 selection publishes feature rows and keeps genuine matches for application ID ${sourceId}`, () => {
    const {device, graph, positions, integers} = createFixture();
    const selection = new GPUBufferSelection({
      points: {kind: 'points', positions: positions()},
      features: {kind: 'points', positions: positions(), sourceIds: integers()},
      radius: createTransientView(graph, 'radius', 'float32', 1),
      candidateCapacity: 1,
      output: {nearestFeatureRows: integers(), outputMask: integers()}
    });
    try {
      const legacyProps = {
        ...selection.props,
        output: {...selection.props.output, nearestFeatureIds: integers()}
      };
      expect(() => new GPUBufferSelection(legacyProps)).toThrow(
        /nearestFeatureIds is unsupported.*nearestFeatureRows/
      );
      const kernels = getGPUCommandNodeKernelSources(
        selection.getCommandNodes(graph).filter(node => /-(assign|mask)$/.test(node.id))
      );
      const published = new Uint32Array(1);
      const mask = new Uint32Array(1);
      for (const kernel of kernels)
        execute(kernel.source, kernel.constants, {
          assignment: Uint32Array.of(0),
          pointFeatureIds: published,
          featureIds: kernel.id.endsWith('-mask') ? published : Uint32Array.of(sourceId),
          maskOut: mask
        });
      expect(Array.from(published)).toEqual([0]);
      expect(Array.from(mask)).toEqual([1]);
    } finally {
      device.destroy();
    }
  });
}

it('R6 row-indexed weights preserve distinct segment rows despite equal application IDs', () => {
  const {device, graph, integers, positions} = createFixture();
  try {
    const neighborIds = integers(2);
    const neighborRows = integers(2);
    const neighborCounts = integers();
    const join = new GPUNearestFeatureJoin({
      queries: {kind: 'points', positions: positions()},
      features: {
        kind: 'segments',
        starts: positions(2),
        ends: positions(2),
        sourceIds: integers(2)
      },
      neighborCount: 2,
      output: {neighborIds, neighborRows, neighborCounts, status: {overflow: integers()}}
    });
    const publisher = join
      .getCommandNodes(graph)
      .map(getStorageTopKNodeRecord)
      .find(record => record?.source.includes('neighborRows['))!;
    const rows = new Uint32Array(2);
    const ids = new Uint32Array(2);
    const counts = new Uint32Array(1);
    const adapt = (source: string) =>
      source
        .replace(
          'var<uniform> selectionSettings: array<vec4<u32>, 3>',
          'var<storage, read> selectionSettings: array<u32>'
        )
        .replace(
          /selectionSettings\[(\d+)\]\.([xyzw])/g,
          (_, row, component) => `selectionSettings[${Number(row) * 4 + 'xyzw'.indexOf(component)}]`
        );
    // Two equal-distance winners in hand-derived traversal scratch: rows 0 and 1.
    execute(
      adapt(publisher.source),
      {},
      {
        topk: Uint32Array.of(2, 0, 0, 0, 0, 1, 0, 0, 0, 1),
        featureIds: Uint32Array.of(7, 7),
        neighborIds: ids,
        neighborRows: rows,
        neighborCounts: counts,
        overflow: new Uint32Array(1),
        selectionSettings: publisher.settings
      }
    );
    expect(Array.from(ids)).toEqual([7, 7]);
    expect(Array.from(rows)).toEqual([0, 1]);
    const adapter = new GPUNearestFeatureWeights({
      neighborRows,
      neighborCounts,
      slotCapacity: 2,
      output: {
        offsets: integers(2),
        neighbors: integers(2),
        weights: createTransientView(graph, 'weights', 'float32', 2),
        status: {overflow: integers()}
      }
    });
    const legacyProps = {...adapter.props, neighborIds};
    expect(() => new GPUNearestFeatureWeights(legacyProps)).toThrow(
      /neighborRows is required and neighborIds is unsupported/
    );
    const record = adapter
      .getCommandNodes(graph)
      .map(getStorageTopKNodeRecord)
      .find(record => record?.source.includes('Insertion sort'))!;
    expect(record.bindings.find(binding => binding.name === 'ids')?.view).toBe(neighborRows);
    const source = adapt(record.source);
    const neighbors = new Uint32Array(2);
    execute(
      source,
      {},
      {
        ids: rows,
        filled: counts,
        offsets: Uint32Array.of(0, 2),
        neighbors,
        values: new Float32Array(2),
        selectionSettings: record.settings
      }
    );
    expect(Array.from(neighbors)).toEqual([0, 1]);
  } finally {
    device.destroy();
  }
});

it('R7 guide distinguishes the 64-slot storage threshold from the packed public maximum', () => {
  const guide = readFileSync('docs/api/gpu-spatial-analysis/operators-joins.md', 'utf8');
  expect(guide.includes('0x7ffffffe')).toBe(true);
  expect(guide.includes('`k` is 1 to 32')).toBe(false);
});

it('final1 preserves inherited gather candidate status ports', () => {
  const {device, graph, integers} = createFixture();
  try {
    const candidateOverflow = integers();
    const status = Object.assign(
      Object.create({
        get candidateOverflow() {
          return candidateOverflow;
        }
      }),
      {
        count: integers(),
        overflow: integers()
      }
    );
    const gather = new GPUPairGather({
      pairs: {leftRows: integers(), rightRows: integers(), status},
      output: {
        leftRows: integers(),
        rightRows: integers(),
        status: {count: integers(), overflow: integers(), candidateOverflow: integers()}
      }
    });
    expect(gather.props.pairs?.status.candidateOverflow === candidateOverflow).toBe(true);
    expect(
      gather
        .getCommandNodes(graph)
        .some(node =>
          node.resources?.some(
            resource => 'buffer' in resource && resource.buffer === candidateOverflow
          )
        )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('final1 preserves inherited point identities in buffer selection', () => {
  const {device, graph, integers, positions} = createFixture();
  try {
    const sourceIds = integers();
    const points = Object.assign(
      Object.create({
        get sourceIds() {
          return sourceIds;
        }
      }),
      {
        kind: 'points' as const,
        positions: positions()
      }
    );
    const selection = new GPUBufferSelection({
      points,
      candidateCapacity: 1,
      features: {kind: 'points', positions: positions()},
      radius: createTransientView(graph, 'distance', 'float32', 1),
      output: {
        outputMask: integers(),
        matches: {ids: integers(), status: {count: integers(), overflow: integers()}}
      }
    });
    expect(selection.props.points.sourceIds === sourceIds).toBe(true);
    expect(
      selection
        .getCommandNodes(graph)
        .some(node =>
          node.resources?.some(resource => 'buffer' in resource && resource.buffer === sourceIds)
        )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('final1 accepts and propagates nearest join loss separately from CSR overflow', () => {
  const {device, graph, integers} = createFixture();
  try {
    const upstream = integers();
    const upstreamCandidate = integers();
    const candidateOverflow = integers();
    const adapter = new GPUNearestFeatureWeights({
      neighborRows: integers(2),
      neighborCounts: integers(),
      slotCapacity: 2,
      neighborStatus: {overflow: upstream, candidateOverflow: upstreamCandidate},
      output: {
        offsets: integers(2),
        neighbors: integers(2),
        weights: createTransientView(graph, 'final1-weights', 'float32', 2),
        status: {overflow: integers(), candidateOverflow}
      }
    });
    expect(adapter.outputWeights.status?.candidateOverflow).toBe(candidateOverflow);
    const records = adapter.getCommandNodes(graph).map(getStorageTopKNodeRecord);
    const record = records.find(record =>
      record?.bindings.some(binding => binding.view === upstream)
    );
    expect(record).toBeDefined();
    expect(
      record!.bindings.some(
        binding => binding.view === candidateOverflow && binding.access === 'read_write'
      )
    ).toBe(true);
    const adapt = (source: string) =>
      source
        .replace(
          'var<uniform> selectionSettings: array<vec4<u32>, 3>',
          'var<storage, read> selectionSettings: array<u32>'
        )
        .replace(
          /selectionSettings\[(\d+)\]\.([xyzw])/g,
          (_, row, component) => `selectionSettings[${Number(row) * 4 + 'xyzw'.indexOf(component)}]`
        );
    const loss = Uint32Array.of(1);
    const candidateLoss = Uint32Array.of(0);
    const published = Uint32Array.of(37);
    for (const [finalLoss, intermediateLoss] of [
      [1, 0],
      [0, 1],
      [0, 0],
      [1, 1],
      [0, 0]
    ]) {
      loss[0] = finalLoss;
      candidateLoss[0] = intermediateLoss;
      execute(
        adapt(record!.source),
        {},
        {
          upstreamOverflow: loss,
          upstreamCandidateOverflow: candidateLoss,
          candidateOverflow: published,
          selectionSettings: record!.settings
        }
      );
      expect(published[0]).toBe(Number(Boolean(finalLoss || intermediateLoss)));
    }
    const offsetsRecord = records.find(record => record?.source.includes('let total = starts'))!;
    const finalOverflow = Uint32Array.of(91);
    const offsets = new Uint32Array(2);
    execute(
      adapt(offsetsRecord.source),
      {},
      {
        counts: Uint32Array.of(2),
        starts: Uint32Array.of(0),
        offsets,
        overflow: finalOverflow,
        selectionSettings: offsetsRecord.settings
      }
    );
    expect(Array.from(offsets)).toEqual([0, 2]);
    expect(finalOverflow[0]).toBe(0);
  } finally {
    device.destroy();
  }
});

it('final1 compact pairs reject ambiguous application-ID names', () => {
  const {device, integers} = createFixture();
  try {
    expect(
      () =>
        new GPUPairGather({
          pairs: {
            leftIds: integers(),
            rightIds: integers(),
            status: {count: integers(), overflow: integers()}
          } as unknown as NonNullable<ConstructorParameters<typeof GPUPairGather>[0]['pairs']>,
          output: {
            leftRows: integers(),
            rightRows: integers(),
            status: {count: integers(), overflow: integers()}
          }
        })
    ).toThrow(/leftRows.*rightRows/);
  } finally {
    device.destroy();
  }
});

it('final1 compact pair rows gather attributes while caller IDs remain permuted', () => {
  const {device, graph, integers} = createFixture();
  try {
    const gather = new GPUPairGather({
      pairs: {
        leftRows: integers(2),
        rightRows: integers(2),
        status: {count: integers(), overflow: integers()}
      },
      rightColumns: [{source: integers(2), output: integers(2)}],
      output: {
        leftRows: integers(2),
        rightRows: integers(2),
        status: {count: integers(), overflow: integers()}
      }
    });
    const nodes = gather.getCommandNodes(graph);
    const rowKernel = getGPUCommandNodeKernelSources(
      nodes.filter(node => node.id.endsWith('-rows'))
    )[0];
    const columnKernel = getGPUCommandNodeKernelSources(
      nodes.filter(node => node.id.endsWith('-right-column-0'))
    )[0];
    const left = new Uint32Array(2),
      right = new Uint32Array(2),
      attributes = new Uint32Array(2);
    const settings = new Uint32Array(16);
    settings[1] = 2;
    execute(rowKernel.source, rowKernel.constants, {
      matchedIn: Uint32Array.of(2),
      pairLeft: Uint32Array.of(0, 1),
      pairRight: Uint32Array.of(0, 1),
      leftOut: left,
      rightOut: right,
      chunkSettings: settings
    });
    execute(columnKernel.source, columnKernel.constants, {
      rowsIn: right,
      sourceIn: Uint32Array.of(10, 20),
      columnOut: attributes,
      chunkSettings: settings
    });
    expect(Array.from(attributes)).toEqual([10, 20]);
    execute(columnKernel.source, columnKernel.constants, {
      rowsIn: right,
      sourceIn: Uint32Array.of(1, 0),
      columnOut: attributes,
      chunkSettings: settings
    });
    expect(Array.from(attributes)).toEqual([1, 0]);
  } finally {
    device.destroy();
  }
});
