// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUTrackSimilarity,
  GPUTrackSimilaritySearch
} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import fixtureJson from '../../oracles/fixtures/track-similarity-edit-measures.json';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

type TrackSet = readonly (readonly (readonly number[])[])[];

/** Packs tracks into positions and CSR offsets. */
function packTracks(tracks: TrackSet) {
  const positions: number[] = [];
  const offsets = [0];
  for (const track of tracks) {
    for (const [x, y] of track) {
      positions.push(x, y);
    }
    offsets.push(positions.length / 2);
  }
  return {positions: Float32Array.from(positions), offsets: Uint32Array.from(offsets)};
}

export type MeasureName =
  | 'hausdorff'
  | 'dynamicTimeWarping'
  | 'editDistanceOnRealSequence'
  | 'editDistanceWithRealPenalty'
  | 'symmetricSegmentPathDistance';

/** Runs the requested measures over pairs `(a[i], b[i])` in one `GPUTrackSimilarity`. */
export async function runMeasures(
  device: Device,
  setA: TrackSet,
  setB: TrackSet,
  measures: readonly MeasureName[],
  options: {
    spatialContext?: GPUSpatialContext;
    editDistanceTolerance?: number;
    editDistanceGapPoint?: readonly [number, number];
    maximumFrechetVertexCount?: number;
  } = {}
): Promise<Record<MeasureName, Float32Array> & {status: Uint32Array}> {
  const graph = new GPUCommandGraph(device, {id: 'edit-measures'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const pairCount = setA.length;
  const [packedA, packedB] = [packTracks(setA), packTracks(setB)];
  const pairs = Uint32Array.from({length: pairCount}, (_, index) => index);
  const outputBuffers = new Map<string, Buffer>();
  const outputBuffer = (name: string) => {
    const buffer = createOutputBuffer(device, pairCount);
    buffers.push(buffer);
    outputBuffers.set(name, buffer);
    return buffer;
  };
  const output = (name: string) =>
    importGraphBuffer(graph, `o-${name}`, outputBuffer(name), 'float32', pairCount);
  graph.add(
    new GPUTrackSimilarity({
      spatialContext: options.spatialContext,
      linesA: {
        kind: 'lines',
        positions: importGraphBuffer(graph, 'positions-a', input(packedA.positions), 'float32x2'),
        lineOffsets: importGraphBuffer(graph, 'offsets-a', input(packedA.offsets), 'uint32')
      },
      linesB: {
        kind: 'lines',
        positions: importGraphBuffer(graph, 'positions-b', input(packedB.positions), 'float32x2'),
        lineOffsets: importGraphBuffer(graph, 'offsets-b', input(packedB.offsets), 'uint32')
      },
      pairA: importGraphBuffer(graph, 'pair-a', input(pairs), 'uint32'),
      pairB: importGraphBuffer(graph, 'pair-b', input(pairs), 'uint32'),
      editDistanceTolerance: options.editDistanceTolerance,
      editDistanceGapPoint: options.editDistanceGapPoint,
      maximumFrechetVertexCount: options.maximumFrechetVertexCount,
      output: {
        ...Object.fromEntries(measures.map(name => [name, output(name)])),
        pairStatus: importGraphBuffer(
          graph,
          'o-status',
          outputBuffer('status'),
          'uint32',
          pairCount
        )
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const result: Record<string, Float32Array | Uint32Array> = {};
    for (const name of measures) {
      result[name] = Float32Array.from(await readFloat32(outputBuffers.get(name)!, pairCount));
    }
    result['status'] = Uint32Array.from(await readUint32(outputBuffers.get('status')!, pairCount));
    return result as Record<MeasureName, Float32Array> & {status: Uint32Array};
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

export type SearchMetric = ConstructorParameters<typeof GPUTrackSimilaritySearch>[0]['metric'];

/** Everything `GPUTrackSimilaritySearch` writes, read back. */
export type SearchResult = {
  distance: Float32Array;
  lowerBound: Float32Array;
  matched: Uint32Array;
  status: Uint32Array;
  topKIndices: Uint32Array;
  topKSummary: Uint32Array;
};

/** Runs one threshold search of query tracks `setA` against candidates `setB`, all pairs. */
export async function runSearch(
  device: Device,
  setA: TrackSet,
  setB: TrackSet,
  pairA: Uint32Array,
  pairB: Uint32Array,
  options: {
    metric: SearchMetric;
    threshold: number;
    topK: number;
    editDistanceTolerance?: number;
    editDistanceGapPoint?: readonly [number, number];
  }
): Promise<SearchResult> {
  const graph = new GPUCommandGraph(device, {id: 'edit-search'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const pairCount = pairA.length;
  const [packedA, packedB] = [packTracks(setA), packTracks(setB)];
  const outputs = {
    distance: createOutputBuffer(device, pairCount),
    lowerBound: createOutputBuffer(device, pairCount),
    matched: createOutputBuffer(device, pairCount),
    status: createOutputBuffer(device, pairCount),
    topKIndices: createOutputBuffer(device, setA.length * options.topK),
    topKSummary: createOutputBuffer(device, setA.length * 2)
  };
  buffers.push(...Object.values(outputs));
  graph.add(
    new GPUTrackSimilaritySearch({
      metric: options.metric,
      editDistanceTolerance: options.editDistanceTolerance,
      editDistanceGapPoint: options.editDistanceGapPoint,
      topK: options.topK,
      linesA: {
        kind: 'lines',
        positions: importGraphBuffer(graph, 'positions-a', input(packedA.positions), 'float32x2'),
        lineOffsets: importGraphBuffer(graph, 'offsets-a', input(packedA.offsets), 'uint32')
      },
      linesB: {
        kind: 'lines',
        positions: importGraphBuffer(graph, 'positions-b', input(packedB.positions), 'float32x2'),
        lineOffsets: importGraphBuffer(graph, 'offsets-b', input(packedB.offsets), 'uint32')
      },
      pairA: importGraphBuffer(graph, 'pair-a', input(pairA), 'uint32'),
      pairB: importGraphBuffer(graph, 'pair-b', input(pairB), 'uint32'),
      threshold: importGraphBuffer(
        graph,
        'threshold',
        input(new Float32Array([options.threshold])),
        'float32'
      ),
      output: {
        distance: importGraphBuffer(graph, 'o-distance', outputs.distance, 'float32', pairCount),
        lowerBound: importGraphBuffer(graph, 'o-bound', outputs.lowerBound, 'float32', pairCount),
        matched: importGraphBuffer(graph, 'o-matched', outputs.matched, 'uint32', pairCount),
        status: importGraphBuffer(graph, 'o-status', outputs.status, 'uint32', pairCount),
        topKIndices: importGraphBuffer(
          graph,
          'o-top-k',
          outputs.topKIndices,
          'uint32',
          setA.length * options.topK
        ),
        topKSummary: importGraphBuffer(
          graph,
          'o-top-k-summary',
          outputs.topKSummary,
          'uint32x2',
          setA.length
        )
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      distance: Float32Array.from(await readFloat32(outputs.distance, pairCount)),
      lowerBound: Float32Array.from(await readFloat32(outputs.lowerBound, pairCount)),
      matched: Uint32Array.from(await readUint32(outputs.matched, pairCount)),
      status: Uint32Array.from(await readUint32(outputs.status, pairCount)),
      topKIndices: Uint32Array.from(
        await readUint32(outputs.topKIndices, setA.length * options.topK)
      ),
      topKSummary: Uint32Array.from(await readUint32(outputs.topKSummary, setA.length * 2))
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

export type EditMeasureCase = {
  name: string;
  a: [number, number][];
  b: [number, number][];
  hausdorff?: number;
  dynamicTimeWarping?: number;
  sspd?: number;
  edr: {tolerance: number; distance: number}[];
  erp: {gap: [number, number]; distance: number}[];
};

/** The decoded oracle fixture of `test/oracles/track-similarity-edit-measures.py`. */
export const EDIT_MEASURE_FIXTURE = decodeOracleFixture<{
  versions: Record<string, string>;
  tolerances: {
    hausdorffRelative: number;
    hausdorffAbsolutePlanar: number;
    dynamicTimeWarpingRelativePerStep: number;
    dynamicTimeWarpingAbsolutePlanar: number;
    sspdRelative: number;
    sspdAbsolutePlanar: number;
    erpRelativePerStep: number;
    erpAbsolutePlanar: number;
    erpAbsoluteMetersPerStep: number;
  };
  planar: EditMeasureCase[];
  greatCircle: EditMeasureCase[];
}>(fixtureJson);
