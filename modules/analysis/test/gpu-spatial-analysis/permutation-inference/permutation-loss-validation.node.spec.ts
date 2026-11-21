// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  GPUGlobalPermutationTest,
  GPULocalPermutationTest,
  GPU_GLOBAL_PERMUTATION_SUMMARY
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import type {GPUChunkedSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

type Placement = 'whole' | 'outer' | 'chunk';
type LossName = 'overflow' | 'candidateOverflow' | 'complete';

function createFixture(rowCount = 4) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `loss-validation-${serial++}`, format, length);
  const weights = (rows = 4): GPUSpatialWeights => ({
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', rows),
    weights: view('float32', rows)
  });
  const place = (placement: Placement, name: LossName, loss: GraphDataView<'uint32'>) => {
    if (placement === 'whole') return {...weights(), status: {[name]: loss}};
    const chunked: GPUChunkedSpatialWeights = {
      rowCount: 4,
      chunks: [0, 2].map(rowStart => ({rowStart, rowCount: 2, weights: weights(2)}))
    };
    if (placement === 'outer') chunked.status = {[name]: loss};
    else chunked.chunks[1].weights.status = {[name]: loss};
    return chunked;
  };
  const construct = (
    kind: 'local' | 'global',
    input: GPUSpatialWeights | GPUChunkedSpatialWeights
  ) => {
    const common = {
      id: `loss-${kind}`,
      weights: input,
      values: view('float32', rowCount),
      parameters: view('uint32', 4),
      maximumPermutations: 5
    };
    return kind === 'local'
      ? new GPULocalPermutationTest({
          ...common,
          statistic: 'localMoran',
          output: {
            exceedances: view('uint32', rowCount),
            pseudoPValues: view('float32', rowCount),
            status: {overflow: view('uint32', 1)}
          }
        })
      : new GPUGlobalPermutationTest({
          ...common,
          statistic: 'moran',
          output: {summary: view('float32', GPU_GLOBAL_PERMUTATION_SUMMARY.length)}
        });
  };
  return {device, graph, view, place, construct};
}

it('S4 local inference counts the invalidInput writer at the 32768-node boundary', () => {
  const rows = 32509;
  const fixture = createFixture(rows);
  try {
    const {view} = fixture;
    const props = {
      statistic: 'localMoran' as const,
      rowTileSize: 1,
      maximumPermutations: 1,
      values: view('float32', rows),
      parameters: view('uint32', 4),
      weights: {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', rows),
        weights: view('float32', rows)
      },
      output: {
        exceedances: view('uint32', rows),
        pseudoPValues: view('float32', rows),
        status: {overflow: view('uint32', 1)}
      }
    };
    expect(new GPULocalPermutationTest(props).estimatedCommandNodeCount).toBe(32768);
    const construct = () =>
      new GPULocalPermutationTest({
        ...props,
        output: {...props.output, status: {...props.output.status, invalidInput: view('uint32', 1)}}
      });
    expect(construct).toThrow(/estimated command node count must be at most 32768/);
    try {
      construct();
    } catch (error) {
      expect(error).toMatchObject({received: 32769});
    }
  } finally {
    fixture.device.destroy();
  }
});

for (const kind of ['local', 'global'] as const) {
  it(`W5-11 ${kind} counts each seven-input fold without merging final and candidate causes`, () => {
    const fixture = createFixture(8);
    try {
      const {view} = fixture;
      const weights: GPUChunkedSpatialWeights = {
        rowCount: 8,
        chunks: Array.from({length: 8}, (_, rowStart) => ({
          rowStart,
          rowCount: 1,
          weights: {
            offsets: view('uint32', 2),
            neighbors: view('uint32', 1),
            weights: view('float32', 1),
            status: {
              overflow: view('uint32', 1),
              candidateOverflow: view('uint32', 1),
              complete: view('uint32', 1)
            }
          }
        }))
      };
      const plain = fixture.construct(kind, weights);
      const causes = {weightsOverflow: view('uint32', 1), candidateOverflow: view('uint32', 1)};
      const folded =
        plain instanceof GPULocalPermutationTest
          ? new GPULocalPermutationTest({
              ...plain.props,
              id: 'folded-local',
              output: {...plain.props.output, status: {...plain.props.output.status, ...causes}}
            })
          : new GPUGlobalPermutationTest({
              ...plain.props,
              id: 'folded-global',
              output: {...plain.props.output, status: causes}
            });
      const sources = getGPUCommandNodeKernelSources(folded.getCommandNodes(fixture.graph));
      // Eight same-cause sources require two batches, independently for each negative flag.
      expect(sources.filter(({id}) => id.includes('-cause-weights-'))).toHaveLength(2);
      expect(sources.filter(({id}) => id.includes('-cause-candidates-'))).toHaveLength(2);
      expect(folded.estimatedCommandNodeCount - plain.estimatedCommandNodeCount).toBe(4);
    } finally {
      fixture.device.destroy();
    }
  });
  for (const placement of ['whole', 'outer', 'chunk'] as const) {
    it(`S1 ${kind} rejects a two-word ${placement} complete certificate`, () => {
      const fixture = createFixture();
      try {
        expect(() =>
          fixture.construct(kind, fixture.place(placement, 'complete', fixture.view('uint32', 2)))
        ).toThrow(/complete.*(one uint32 row|length.*1)/);
      } finally {
        fixture.device.destroy();
      }
    });
    it(`S1 ${kind} accepts a one-word offset ${placement} complete certificate`, () => {
      const fixture = createFixture();
      try {
        const backing = fixture.view('uint32', 5);
        const complete = fixture.graph.createDataView(backing.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: 12
        });
        expect(() =>
          fixture
            .construct(kind, fixture.place(placement, 'complete', complete))
            .getCommandNodes(fixture.graph)
        ).not.toThrow();
      } finally {
        fixture.device.destroy();
      }
    });
    it(`W5-11 ${kind} preserves and inverts ${placement} complete certificates`, () => {
      const fixture = createFixture();
      try {
        const complete = fixture.view('uint32', 1);
        const contributor = fixture.construct(kind, fixture.place(placement, 'complete', complete));
        const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(fixture.graph));
        expect(sources.some(({source}) => source.includes('loss[lossOffset] == 0u'))).toBe(true);
        expect(
          sources.some(({source}) =>
            source.includes('analysisFlags[analysisFlagsOffset + index] = 0u')
          )
        ).toBe(true);
        expect(contributor.estimatedCommandNodeCount).toBeGreaterThanOrEqual(sources.length);
      } finally {
        fixture.device.destroy();
      }
    });
  }
  describe(`${kind} permutation loss metadata (SSR-130)`, () => {
    for (const name of ['overflow', 'candidateOverflow'] as const) {
      it(`rejects oversized second-chunk ${name} before binding preflight`, () => {
        const fixture = createFixture();
        try {
          const length = fixture.device.limits.maxStorageBufferBindingSize / 4;
          const backing = fixture.view('uint32', length + 3);
          const loss = fixture.graph.createDataView(backing.buffer, {
            format: 'uint32',
            length,
            byteOffset: 12
          });
          expect(() => fixture.construct(kind, fixture.place('chunk', name, loss))).toThrow(
            new RegExp(`${name}.*one uint32 row`)
          );
        } finally {
          fixture.device.destroy();
        }
      });
    }
    for (const placement of ['whole', 'outer', 'chunk'] as const) {
      for (const name of ['overflow', 'candidateOverflow'] as const) {
        for (const malformed of ['format', 'nonpacked', 'empty', 'too-long'] as const) {
          it(`rejects ${placement} ${name} with ${malformed} view at construction`, () => {
            const fixture = createFixture();
            try {
              const valid = fixture.view('uint32', 1);
              // Metadata corruption is intentional: callers can arrive through JavaScript.
              const invalid = {
                ...valid,
                ...(malformed === 'format' ? {format: 'float32'} : {}),
                ...(malformed === 'nonpacked' ? {stride: 2, byteStride: 8} : {}),
                ...(malformed === 'empty' ? {length: 0, byteLength: 0} : {}),
                ...(malformed === 'too-long' ? {length: 2 ** 32, byteLength: 2 ** 34} : {})
              } as GraphDataView<'uint32'>;
              expect(() =>
                fixture.construct(kind, fixture.place(placement, name, invalid))
              ).toThrow(new RegExp(`loss-${kind}.*${name}`));
            } finally {
              fixture.device.destroy();
            }
          });
        }

        it(`accepts a nonzero-offset ${placement} ${name} view`, () => {
          const fixture = createFixture();
          try {
            const backing = fixture.view('uint32', 5);
            const loss = fixture.graph.createDataView(backing.buffer, {
              format: 'uint32',
              length: 1,
              byteOffset: 12
            });
            const contributor = fixture.construct(kind, fixture.place(placement, name, loss));
            expect(() => contributor.getCommandNodes(fixture.graph)).not.toThrow();
          } finally {
            fixture.device.destroy();
          }
        });

        it(`rejects the oversized ${placement} ${name} status view`, () => {
          const fixture = createFixture();
          try {
            // Canonical status fields are scalar views, even when backed by a larger allocation.
            const length = fixture.device.limits.maxStorageBufferBindingSize / 4 + 1;
            const loss = fixture.view('uint32', length);
            const construct = () => fixture.construct(kind, fixture.place(placement, name, loss));
            if (placement === 'outer') {
              // Root chunked metadata still reaches the full binding-range preflight.
              expect(() => construct().getCommandNodes(fixture.graph)).toThrow(
                /binding.*needs 134217732 bytes.*limit 134217728/
              );
            } else expect(construct).toThrow(new RegExp(`${name}.*one uint32 row`));
          } finally {
            fixture.device.destroy();
          }
        });
      }
    }
  });
}
