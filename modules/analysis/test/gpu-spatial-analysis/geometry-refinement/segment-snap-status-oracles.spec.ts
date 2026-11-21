// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSegmentSnap} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap';
import {GPUSegmentSnapOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap-operation';
import {CPUSegmentSnapOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-segment-snap-operation';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Line (0,0)->(3,0), references at x=1,2: both references must be inserted,
// yielding four vertices. Grid capacity 1 loses candidates and uses the
// default complete fallback; capacity 16 fits both. Thus counts=[4,4,0,0],
// validity=1 and candidateOverflow=1/0 independently of candidate order.
for (const nested of [false, true])
  for (const candidateCapacity of [1, 16]) {
    it(`snap status observes candidate loss in every spelling and form (nested ${nested}, capacity ${candidateCapacity})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createOutputOracleFixture(device);
      const {graph, input, output} = fixture;
      const flag = output('candidate', 'uint32', 1),
        counts = output('counts', 'uint32x4', 1),
        validity = output('validity', 'uint32', 1);
      const statusOutput = nested ? {status: {candidateOverflow: flag}} : {candidateOverflow: flag};
      graph.add(
        new GPUSegmentSnap({
          tolerance: 0,
          maximumVerticesPerLine: 4,
          candidateCapacity,
          searchStructure: 'grid',
          positions: input('positions', 'float32x2', Float32Array.of(0, 0, 3, 0)),
          lineOffsets: input('offsets', 'uint32', Uint32Array.of(0, 2)),
          referencePositions: input('references', 'float32x2', Float32Array.of(1, 0, 2, 0)),
          output: {
            positions: output('snapped', 'float32x2', 4),
            sourceEdges: output('edges', 'uint32', 4),
            counts,
            validity,
            ...statusOutput
          }
        })
      );
      const compiled = graph.compile();
      try {
        expect(
          await fixture.submitTwice(compiled, () => fixture.readRows([counts, validity, flag]))
        ).toEqual([4, 4, 0, 0, 1, candidateCapacity === 1 ? 1 : 0]);
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
      for (const Operation of [GPUSegmentSnapOperation, CPUSegmentSnapOperation]) {
        const result = await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: Float32Array.of(0, 0, 3, 0)},
            offsets: {format: 'uint32', values: Uint32Array.of(0, 2)},
            references: {format: 'float32x2', values: Float32Array.of(1, 0, 2, 0)}
          },
          operations: [
            new Operation({
              positions: 'positions',
              lineOffsets: 'offsets',
              referencePositions: 'references',
              rowCount: 2,
              lineCount: 1,
              referenceCount: 2,
              tolerance: 0,
              maximumVerticesPerLine: 4,
              candidateCapacity,
              searchStructure: 'grid',
              output: {
                positions: 'snapped',
                sourceEdges: 'edges',
                counts: 'counts',
                validity: 'validity',
                ...(nested
                  ? {status: {candidateOverflow: 'candidate'}}
                  : {candidateOverflow: 'candidate'})
              }
            })
          ],
          outputs: ['counts', 'validity', 'candidate']
        });
        expect(Array.from(result['counts'])).toEqual([4, 4, 0, 0]);
        expect(Array.from(result['validity'])).toEqual([1]);
        expect(Array.from(result['candidate'])).toEqual([candidateCapacity === 1 ? 1 : 0]);
      }
    });
  }
