// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialPredicateJoin} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {RecipeTestFixture} from '../gpu-spatial-analysis/recipes/recipe-harness';
import {requireTestDevice} from './require-test-device';

it('spatial-3:F6 implicit weights diagnostics preserve safe adjacent scalar words in separate passes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'predicate-scalar-plan');
  try {
    const words = fixture.output('words', 'uint32', 65);
    const first = fixture.graph.createDataView(words.view.buffer, {format: 'uint32', length: 1});
    const second = fixture.graph.createDataView(words.view.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    const scalar = (name: string) => fixture.output(name, 'uint32', 1).view;
    const count = fixture.output('pair-count', 'uint32', 1);
    fixture.graph.add(
      new GPUSpatialPredicateJoin({
        left: {
          kind: 'points',
          positions: fixture.input('left', new Float32Array([0, 0, 0, 0]), 'float32x2', 2)
        },
        right: {
          kind: 'points',
          positions: fixture.input('right', new Float32Array([0, 0]), 'float32x2', 1)
        },
        predicate: 'intersects',
        candidateCapacity: 4,
        output: {
          pairs: {
            leftRows: fixture.output('left-rows', 'uint32', 1).view,
            rightRows: fixture.output('right-rows', 'uint32', 1).view,
            status: {
              count: count.view,
              overflow: second,
              candidateOverflow: scalar('pair-candidate')
            }
          },
          weights: {
            offsets: fixture.output('offsets', 'uint32', 3).view,
            neighbors: fixture.output('neighbors', 'uint32', 1).view,
            weights: fixture.output('weights', 'float32', 1).view,
            status: {
              count: scalar('weights-count'),
              requiredCount: scalar('weights-required'),
              uncertainCount: first
            }
          },
          status: {overflow: scalar('overall')}
        }
      })
    );
    fixture.run();
    expect(await fixture.readUint32(words, 2)).toEqual([0, 1]);
    expect(await fixture.readUint32(count, 1)).toEqual([1]);
  } finally {
    fixture.destroy();
  }
});

it('spatial-3:F5 complete anti join retains unmatched row 1 and clears its separate candidate word', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'anti-candidate-row');
  try {
    const ids = fixture.output('ids', 'uint32', 2),
      count = fixture.output('count', 'uint32', 1),
      overflow = fixture.output('overflow', 'uint32', 1),
      candidate = fixture.output('candidate', 'uint32', 1);
    candidate.buffer.write(new Uint32Array([37]));
    fixture.graph.add(
      new GPUSpatialPredicateJoin({
        left: {
          kind: 'points',
          positions: fixture.input('left', new Float32Array([0, 0, 10, 10]), 'float32x2', 2)
        },
        right: {
          kind: 'points',
          positions: fixture.input('right', new Float32Array([0, 0]), 'float32x2', 1)
        },
        predicate: 'intersects',
        how: 'anti',
        candidateCapacity: 4,
        output: {
          unmatched: {
            ids: ids.view,
            status: {count: count.view, overflow: overflow.view, candidateOverflow: candidate.view}
          }
        }
      })
    );
    fixture.run();
    expect(await fixture.readUint32(count, 1)).toEqual([1]);
    expect(await fixture.readUint32(ids, 1)).toEqual([1]);
    expect(await fixture.readUint32(overflow, 1)).toEqual([0]);
    expect(await fixture.readUint32(candidate, 1)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});
