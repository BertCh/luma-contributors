// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  castData,
  extent,
  gather,
  GPUDataEvaluator,
  GPUEvaluationScope,
  segmentedMap,
  sequence
} from '@luma-contributors/graph/gpu-simple-graph';
import {expect, it, vi} from 'vitest';

it('scope-capable handlers record gather, sequence, segmentedMap and extent into one submission', async ({
  skip
}) => {
  const device = await getWebGPUTestDevice();
  if (!device) return skip();
  const sourceValues = GPUDataEvaluator.fromArray([10, 11, 20, 21, 30, 31, 40, 41], {size: 2});
  const ids = GPUDataEvaluator.fromArray([2, 0, 1, 9], {type: 'uint32', size: 1});
  const gathered = gather(ids, sourceValues);
  const counted = sequence(6, 3, -2);
  const segments = GPUDataEvaluator.fromArray([0, 2, 5], {type: 'uint32', size: 1});
  const mapped = segmentedMap(segments, 8);
  // A three-pass reduction: 300 rows, 64 per group, then 5 groups.
  const ramp = GPUDataEvaluator.fromArray(
    Float32Array.from({length: 300}, (_, index) => index - 100),
    {size: 1}
  );
  const bounds = extent(ramp);
  const scope = new GPUEvaluationScope(device);
  const submit = vi.spyOn(device, 'submit');
  try {
    await scope.evaluateAll([gathered, counted, mapped, bounds]);
    expect(submit, 'a scope records every operation before submitting once').not.toHaveBeenCalled();
    scope.submit();
    expect(submit).toHaveBeenCalledTimes(1);
    expect(Array.from(await gathered.readValue())).toEqual([30, 31, 10, 11, 20, 21, 0, 0]);
    expect(Array.from(await counted.readValue())).toEqual([3, 1, -1, -3, -5, -7]);
    expect(Array.from(await mapped.readValue())).toEqual([
      0, 0, 0, 1, 1, 0, 1, 1, 1, 2, 2, 0, 2, 1, 2, 2
    ]);
    expect(Array.from(await bounds.readValue())).toEqual([-100, 199]);
  } finally {
    submit.mockRestore();
    for (const evaluator of [
      gathered,
      counted,
      mapped,
      bounds,
      ids,
      sourceValues,
      segments,
      ramp
    ]) {
      evaluator.destroy();
    }
  }
});

it('scope-capable cast-data records into the shared encoder', async ({skip}) => {
  const device = await getWebGPUTestDevice();
  if (!device) return skip();
  const input = GPUDataEvaluator.fromArray(new Float32Array([0, 0.5, 1, 2]), {size: 2});
  const cast = castData(input, {inputFormat: 'float32x2', outputFormat: 'unorm8x2'});
  const scope = new GPUEvaluationScope(device);
  const submit = vi.spyOn(device, 'submit');
  try {
    await scope.evaluate(cast);
    expect(submit).not.toHaveBeenCalled();
    scope.submit();
    expect(submit).toHaveBeenCalledTimes(1);
    expect((await cast.readValue()).length).toBeGreaterThan(0);
  } finally {
    submit.mockRestore();
    cast.destroy();
    input.destroy();
  }
});

it('handlers share one shader module across row counts and view offsets', async ({skip}) => {
  const device = await getWebGPUTestDevice();
  if (!device) return skip();
  const createShader = vi.spyOn(device, 'createShader');
  const evaluators: GPUDataEvaluator[] = [];
  try {
    const run = async (rowCount: number) => {
      const ids = GPUDataEvaluator.fromArray(
        Uint32Array.from({length: rowCount}, (_, index) => rowCount - 1 - index),
        {type: 'uint32', size: 1}
      );
      const values = GPUDataEvaluator.fromArray(
        Float32Array.from({length: rowCount}, (_, index) => index),
        {size: 1}
      );
      const gathered = gather(ids, values);
      evaluators.push(ids, values, gathered);
      await gathered.evaluate(device);
      const reversed = Array.from(await gathered.readValue());
      expect(reversed).toEqual(Array.from({length: rowCount}, (_, index) => rowCount - 1 - index));
      const sequenced = sequence(rowCount, rowCount, 1);
      evaluators.push(sequenced);
      await sequenced.evaluate(device);
      expect(Array.from(await sequenced.readValue())[rowCount - 1]).toBe(2 * rowCount - 1);
    };
    await run(5);
    const shaderCountAfterFirstSize = createShader.mock.calls.length;
    await run(9);
    await run(130);
    expect(
      createShader.mock.calls.length,
      'new sizes reuse the compiled gather and sequence shaders'
    ).toBe(shaderCountAfterFirstSize);
  } finally {
    createShader.mockRestore();
    for (const evaluator of evaluators) evaluator.destroy();
  }
});
