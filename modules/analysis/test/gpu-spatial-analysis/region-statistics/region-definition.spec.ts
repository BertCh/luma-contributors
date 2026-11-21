// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUGridIndex, GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, test} from 'vitest';
import {GPURegionStatistics} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-region-statistics';
import {GPURegionMask} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-region-mask';
import {GPUPickRegionMask} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-pick-region-mask';
import type {GPURegionSelection} from '../../../src/gpu-spatial-analysis/region-statistics/region-statistics-types';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {
  DEFINITION_IDS,
  DEFINITION_POSITIONS,
  DEFINITION_VALUES,
  REGION_DEFINITION,
  REGION_MASK_COMPOSITION_GRID
} from './region-definition-fixtures';
import {expectRegionDefinitionSummary} from './region-definition-assertions';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';
import {decodeGPURegionStatistics} from '../../../src/gpu-spatial-analysis/region-statistics/region-statistics-readback';
import {createExampleMaskSelection, LASSO_EXAMPLE_SOURCES} from './review-example-wiring';

for (const {name, source} of LASSO_EXAMPLE_SOURCES) {
  for (const kind of ['pick', 'shape']) {
    test(`R19: ${name} actual ${kind} construction retains one row and propagates loss on both submissions`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new GraphRig(device);
      const picks = rig.output('uint32', 4);
      picks.buffer.write(Uint32Array.of(1, 1, 1, 0));
      const vertexCount = rig.output('uint32', 1);
      vertexCount.buffer.write(Uint32Array.of(4));
      const fixture = createExampleMaskSelection(
        device,
        rig.graph,
        source,
        picks.view,
        kind === 'shape'
          ? {
              positions: rig.input(Float32Array.of(0.25, 0.25, 5, 5), 'float32x2', 2),
              region: {
                kind: 'polygon',
                vertices: rig.input(Float32Array.of(0, 0, 1, 0, 0, 1), 'float32x2', 3),
                vertexCount: vertexCount.view
              }
            }
          : undefined
      );
      try {
        const summary = rig.output('uint32', 8);
        rig.run(
          fixture.producer,
          new GPURegionStatistics({
            region: fixture.region,
            values: rig.input(Float32Array.of(3, 7), 'float32'),
            output: {summary: summary.view}
          })
        );
        for (const regionTruncated of [true, false]) {
          if (!regionTruncated) {
            if (kind === 'shape') vertexCount.buffer.write(Uint32Array.of(3));
            else picks.buffer.write(Uint32Array.of(1, 0, 0, 0));
            rig.resubmit();
          }
          const value = kind === 'pick' && regionTruncated ? 7 : 3;
          expect(
            decodeGPURegionStatistics(Uint32Array.from(await summary.readUint32()))
          ).toMatchObject({
            selectedCount: 1,
            valueCount: 1,
            sum: value,
            mean: value,
            minimum: value,
            maximum: value,
            selectionTruncated: false,
            regionTruncated,
            candidatesTruncated: false
          });
        }
      } finally {
        rig.destroy();
        fixture.destroy();
      }
    });
  }
}

for (const limited of ['candidates', 'ids'] as const) {
  test(`R14: composed grid mask ${limited === 'ids' ? 'ignores compact-ID loss' : 'propagates candidate loss'} and recovers`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new GraphRig(device);
    try {
      const fixture = REGION_MASK_COMPOSITION_GRID;
      const positions = rig.input(fixture.positions, 'float32x2', 2),
        values = rig.input(Float32Array.of(1, 1), 'float32');
      const bounds = rig.output('float32', 4);
      bounds.buffer.write(fixture.initialQuery);
      const grid = new GPUGridIndex({
        positions,
        gridSize: [fixture.width, 1],
        bounds: fixture.bounds,
        cellOffsets: rig.output('uint32', fixture.width + 1).view,
        objectIds: rig.output('uint32', 2).view,
        count: rig.output('uint32', 1).view,
        overflow: rig.output('uint32', 1).view
      });
      const mask = rig.output('uint32', 2),
        summary = rig.output('uint32', 8),
        overflow = rig.output('uint32', 1),
        candidateOverflow = rig.output('uint32', 1);
      const selectedCount = rig.output('uint32', 1),
        selectedOverflow = rig.output('uint32', 1),
        requiredCount = rig.output('uint32', 1);
      const producerOutput = {
        mask: mask.view,
        summary: summary.view,
        status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view},
        selectedRows: {
          ids: rig.output('uint32', limited === 'ids' ? 1 : 2).view,
          status: {
            count: selectedCount.view,
            overflow: selectedOverflow.view,
            requiredCount: requiredCount.view
          }
        }
      };
      const downstream = rig.output('uint32', 8);
      rig.run(
        grid,
        new GPURegionStatistics({
          id: 'mask-producer',
          positions,
          values,
          region: {kind: 'rectangle', bounds: bounds.view},
          spatialIndex: {
            kind: 'grid',
            index: grid,
            candidateCapacity: limited === 'candidates' ? 1 : 2
          },
          output: producerOutput
        }),
        new GPURegionStatistics({
          id: 'mask-consumer',
          values,
          region: {kind: 'mask', ...producerOutput},
          output: {summary: downstream.view}
        })
      );
      for (const complete of [false, true]) {
        if (complete) {
          bounds.buffer.write(fixture.completeQuery);
          rig.resubmit();
        }
        const expectedCount = complete || limited === 'candidates' ? 1 : 2;
        expect(
          decodeGPURegionStatistics(Uint32Array.from(await downstream.readUint32()))
        ).toMatchObject({
          selectedCount: expectedCount,
          valueCount: expectedCount,
          sum: expectedCount,
          mean: 1,
          minimum: 1,
          maximum: 1,
          selectionTruncated: false,
          regionTruncated: false,
          candidatesTruncated: !complete && limited === 'candidates'
        });
        expect(await mask.readUint32()).toEqual(expectedCount === 2 ? [1, 1] : [1, 0]);
        expect(await overflow.readUint32()).toEqual([Number(!complete && limited === 'ids')]);
        expect(await candidateOverflow.readUint32()).toEqual([
          Number(!complete && limited === 'candidates')
        ]);
        expect(await selectedCount.readUint32()).toEqual([1]);
        expect(await selectedOverflow.readUint32()).toEqual([
          Number(!complete && limited === 'ids')
        ]);
        expect(await requiredCount.readUint32()).toEqual([expectedCount]);
      }
    } finally {
      rig.destroy();
    }
  });
}

test('R3: composed pick mask and direct pick region preserve loss and clear on the same graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  try {
    const picks = rig.output('uint32', 4),
      mask = rig.output('uint32', 3),
      overflow = rig.output('uint32', 1);
    const composed = rig.output('uint32', 8),
      direct = rig.output('uint32', 8);
    picks.buffer.write(Uint32Array.of(1, 1, 2, 0));
    const pickOutput = {mask: mask.view, status: {overflow: overflow.view}};
    const values = rig.input(Float32Array.of(2, 3, 5), 'float32');
    rig.run(
      new GPUPickRegionMask({picks: picks.view, output: pickOutput}),
      new GPURegionStatistics({
        id: 'composed',
        region: {kind: 'mask', ...pickOutput},
        values,
        output: {summary: composed.view}
      }),
      new GPURegionStatistics({
        id: 'direct',
        region: {kind: 'pick-region', picks: picks.view},
        values,
        output: {summary: direct.view}
      })
    );
    for (const regionTruncated of [true, false]) {
      if (!regionTruncated) {
        picks.buffer.write(Uint32Array.of(1, 0, 2, 0));
        rig.resubmit();
      }
      for (const summary of [composed, direct]) {
        expect(
          decodeGPURegionStatistics(Uint32Array.from(await summary.readUint32()))
        ).toMatchObject({
          selectedCount: 1,
          valueCount: 1,
          sum: 5,
          mean: 5,
          minimum: 5,
          maximum: 5,
          selectionTruncated: false,
          regionTruncated,
          candidatesTruncated: false
        });
      }
      expect(await mask.readUint32()).toEqual([0, 0, 1]);
      expect(await overflow.readUint32()).toEqual([Number(regionTruncated)]);
    }
  } finally {
    rig.destroy();
  }
});

test('R4: exact interleaved mask is refused and an offset packed mask selects rows zero and two', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  try {
    const backing = rig.input(Uint32Array.of(99, 1, 0, 0, 0, 1), 'uint32');
    const mask = new GraphDataView(backing.buffer, {
      format: 'uint32',
      length: 3,
      byteOffset: 4,
      byteStride: 8,
      rowByteLength: 4
    });
    const summary = rig.output('uint32', 8);
    expect(
      () => new GPURegionStatistics({region: {kind: 'mask', mask}, output: {summary: summary.view}})
    ).toThrow(/mask.*packed/);
    const packed = rig.input(Uint32Array.of(99, 1, 0, 1), 'uint32');
    const packedMask = new GraphDataView(packed.buffer, {
      format: 'uint32',
      length: 3,
      byteOffset: 4,
      byteStride: 4,
      rowByteLength: 4
    });
    const ids = rig.output('uint32', 3),
      count = rig.output('uint32', 1),
      overflow = rig.output('uint32', 1);
    rig.run(
      new GPURegionStatistics({
        region: {kind: 'mask', mask: packedMask},
        values: rig.input(Float32Array.of(2, 3, 5), 'float32'),
        output: {
          summary: summary.view,
          selectedRows: {ids: ids.view, status: {count: count.view, overflow: overflow.view}}
        }
      })
    );
    expect(decodeGPURegionStatistics(Uint32Array.from(await summary.readUint32()))).toMatchObject({
      selectedCount: 2,
      valueCount: 2,
      sum: 7,
      mean: 3.5,
      minimum: 2,
      maximum: 5,
      regionTruncated: false
    });
    expect((await ids.readUint32()).slice(0, 2)).toEqual([0, 2]);
    expect(await count.readUint32()).toEqual([2]);
    expect(await overflow.readUint32()).toEqual([0]);
  } finally {
    rig.destroy();
  }
});

for (const indexed of [false, true]) {
  for (const shape of ['rectangle', 'polygon', 'radius', 'empty'] as const) {
    test(`region definition oracle: ${shape}, ${indexed ? 'grid' : 'direct'}, every output`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const transformed of [false, true]) {
        const fixture =
          REGION_DEFINITION[shape === 'radius' ? 'radius' : shape === 'empty' ? 'empty' : 'square'];
        const coordinate = (value: number) => (transformed ? 100 + 4 * value : value);
        const rig = new AnalysisRig(device);
        try {
          const positions = rig.input(
            Float32Array.from(DEFINITION_POSITIONS, coordinate),
            'float32x2',
            12
          );
          const region: GPURegionSelection =
            shape === 'polygon'
              ? {
                  kind: 'polygon',
                  vertices: rig.input(
                    Float32Array.from([0, 0, 2, 0, 2, 2, 0, 2], coordinate),
                    'float32x2',
                    4
                  ),
                  vertexCount: rig.input(Uint32Array.of(4), 'uint32')
                }
              : shape === 'radius'
                ? {
                    kind: 'radius',
                    circle: rig.input(
                      Float32Array.of(coordinate(1), coordinate(1), transformed ? 4 : 1),
                      'float32'
                    )
                  }
                : {
                    kind: 'rectangle',
                    bounds: rig.input(
                      Float32Array.from(
                        shape === 'empty' ? [20, 20, 21, 21] : [0, 0, 2, 2],
                        coordinate
                      ),
                      'float32'
                    )
                  };
          const summary = rig.output('uint32', 12);
          const mask = rig.output('uint32', 12);
          const ids = rig.output('uint32', 5);
          const count = rig.output('uint32', 1);
          const requiredCount = rig.output('uint32', 1);
          const selectedOverflow = rig.output('uint32', 1);
          const selectedCandidateOverflow = rig.output('uint32', 1);
          const overflow = rig.output('uint32', 1);
          const candidateOverflow = rig.output('uint32', 1);
          const drawInstanceCount = rig.output('uint32', 1);
          const grid = indexed
            ? new GPUGridIndex({
                positions,
                gridSize: [4, 4],
                bounds: [coordinate(-2), coordinate(-2), coordinate(5), coordinate(5)],
                cellOffsets: rig.output('uint32', 17),
                objectIds: rig.output('uint32', 12),
                count: rig.output('uint32', 1),
                overflow: rig.output('uint32', 1)
              })
            : undefined;
          const constructionProps = {
            region,
            positions,
            values: rig.input(DEFINITION_VALUES, 'float32'),
            sourceIds: rig.input(DEFINITION_IDS, 'uint32'),
            histogram: {binCount: 4, domain: [0, 8] as [number, number]},
            ...(grid
              ? {spatialIndex: {kind: 'grid' as const, index: grid, candidateCapacity: 12}}
              : {}),
            output: {
              summary,
              mask,
              drawInstanceCount,
              selectedRows: {
                ids,
                status: {
                  count,
                  requiredCount,
                  overflow: selectedOverflow,
                  candidateOverflow: selectedCandidateOverflow
                }
              },
              status: {overflow, candidateOverflow}
            }
          };
          const contributor = new GPURegionStatistics(constructionProps);
          constructionProps.histogram.binCount = 1;
          constructionProps.histogram.domain[0] = 100;
          constructionProps.output.summary = rig.output('uint32', 9);
          constructionProps.output.selectedRows.status.count = rig.output('uint32', 1);
          if (constructionProps.spatialIndex) constructionProps.spatialIndex.candidateCapacity = 1;
          rig.run(...(grid ? [grid, contributor] : [contributor]));
          expectRegionDefinitionSummary(await rig.readUint(summary), fixture);
          expect(await rig.readUint(mask)).toEqual(fixture.mask);
          expect((await rig.readUint(ids)).slice(0, fixture.count)).toEqual(fixture.ids);
          for (const [view, expected] of [
            [count, fixture.count],
            [drawInstanceCount, fixture.count],
            [requiredCount, fixture.requiredCount],
            [selectedOverflow, fixture.overflow],
            [overflow, fixture.overflow],
            [candidateOverflow, 0],
            [selectedCandidateOverflow, 0]
          ] as const) {
            expect(await rig.readUint(view)).toEqual([expected]);
          }
        } finally {
          rig.destroy();
        }
      }
    });
  }
}

test('region mask and pick mask match independent inclusive-boundary and set-membership literals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const mask = rig.output('uint32', 12);
    const overflow = rig.output('uint32', 1);
    const pickMask = rig.output('uint32', 12);
    const pickOverflow = rig.output('uint32', 1);
    const maskProps = {
      positions: rig.input(DEFINITION_POSITIONS, 'float32x2', 12),
      region: {
        kind: 'polygon' as const,
        vertices: rig.input(Float32Array.of(0, 0, 2, 0, 2, 2, 0, 2), 'float32x2', 4),
        vertexCount: rig.input(Uint32Array.of(5), 'uint32')
      },
      output: {mask, status: {overflow}}
    };
    const pickProps = {
      // Duplicate, wrong batch, sentinel and out-of-range object identities.
      picks: rig.input(
        Uint32Array.of(7, 1, 4, 2, 7, 2, 4, 2, 9, 3, 11, 2, 0xffffffff, 2, 99, 2),
        'uint32'
      ),
      batchIndex: 2,
      output: {mask: pickMask, status: {overflow: pickOverflow}}
    };
    const maskContributor = new GPURegionMask(maskProps),
      pickContributor = new GPUPickRegionMask(pickProps);
    maskProps.region.vertexCount = rig.input(Uint32Array.of(0), 'uint32');
    maskProps.output.status.overflow = rig.output('uint32', 1);
    pickProps.batchIndex = 3;
    pickProps.output.mask = rig.output('uint32', 12);
    rig.run(maskContributor, pickContributor);
    expect(await rig.readUint(mask)).toEqual(REGION_DEFINITION.square.mask);
    expect(await rig.readUint(overflow)).toEqual([1]);
    // Deduplicated batch-2 membership is exactly {4, 7, 11}.
    expect(await rig.readUint(pickMask)).toEqual([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]);
    expect(await rig.readUint(pickOverflow)).toEqual([1]);
  } finally {
    rig.destroy();
  }
});
