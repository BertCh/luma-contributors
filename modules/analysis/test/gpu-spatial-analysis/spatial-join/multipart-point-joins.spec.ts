// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it, expect} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {createBufferSelectionOperation} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection-operation';
import {createCPUBufferSelectionOperation} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-buffer-selection-operation';
import {bufferOperationProps, getOperationResult} from './multipart-point-operations-fixture';
import {pointJoinOperationProps} from './multipart-point-operations-fixture';
import {createPointInPolygonJoinOperation} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join-operation';
import {createCPUPointInPolygonJoinOperation} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-point-in-polygon-join-operation';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {createPointsInPolygonsChoroplethOperation} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-operation';
import {createCPUPointsInPolygonsChoroplethOperation} from '../../../src/gpu-spatial-analysis/recipes/cpu-points-in-polygons-choropleth-operation';
import {choroplethOperationProps} from './multipart-point-operations-fixture';
import {createLineLengthPerPolygonOperation} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-length-per-polygon-operation';
import {createCPULineLengthPerPolygonOperation} from '../../../src/gpu-spatial-analysis/line-density/cpu-line-length-per-polygon-operation';
import {lineOperationProps} from './multipart-point-operations-fixture';
import {
  multipartInputs,
  joinOutputs,
  bufferOutputs,
  lineOutputs,
  recipeOutputs,
  createMultipartJoin,
  createMultipartBuffer,
  createMultipartLines,
  createMultipartRecipe,
  assertMultipartJoin,
  assertMultipartBuffer,
  assertMultipartLines,
  assertMultipartRecipe
} from './multipart-point-joins-fixture';

for (const candidateCapacity of [1, 128])
  for (const boundaryRule of ['closed', 'open', 'half-open'] as const)
    it(`WebGPU mixed polygon operations match Shapely ${boundaryRule}, candidate capacity ${candidateCapacity}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const props = {
        ...pointJoinOperationProps,
        boundaryRule,
        candidateCapacity,
        candidateTileCapacity: Math.min(3, candidateCapacity)
      };
      for (const operation of [
        createPointInPolygonJoinOperation(props),
        createCPUPointInPolygonJoinOperation(props)
      ]) {
        const result = getOperationResult(
          await runOperationGraph(device, {
            inputs: multipartInputs,
            operations: [operation],
            outputs: Object.values(props.output)
          })
        );
        if (candidateCapacity === 128) assertMultipartJoin(result, boundaryRule);
        else {
          expect(result['truncated']).toEqual([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
          expect(result['counts']).toEqual([1, 0, 0, 0, 0, 0]);
          expect(result['candidateCount']).toEqual([25]);
          expect(result['candidateOverflow']).toEqual([1]);
          expect(result['overflow']).toEqual([1]);
        }
      }
    });
it('WebGPU mixed polygon operations report final compact overflow without candidate loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = {...pointJoinOperationProps, matchCapacity: 2};
  for (const operation of [
    createPointInPolygonJoinOperation(props),
    createCPUPointInPolygonJoinOperation(props)
  ]) {
    const result = getOperationResult(
      await runOperationGraph(device, {
        inputs: multipartInputs,
        operations: [operation],
        outputs: Object.values(props.output)
      })
    );
    expect(result['ids']).toEqual([90, 80]);
    expect(result['count']).toEqual([2]);
    expect(result['requiredCount']).toEqual([11]);
    expect(result['counts']).toEqual([7, 4, 2, 2, 2, 3]);
    expect(result['matchOverflow']).toEqual([1]);
    expect(result['overflow']).toEqual([1]);
    expect(result['candidateOverflow']).toEqual([0]);
    expect(result['matchCandidateOverflow']).toEqual([0]);
  }
});
for (const pairCapacity of [1, 128])
  it(`WebGPU mixed line operations match Shapely with pair capacity ${pairCapacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props = {...lineOperationProps, pairCapacity};
    for (const operation of [
      createLineLengthPerPolygonOperation(props),
      createCPULineLengthPerPolygonOperation(props)
    ]) {
      const result = getOperationResult(
        await runOperationGraph(device, {
          inputs: multipartInputs,
          operations: [operation],
          outputs: Object.values(props.output).filter((name): name is string => Boolean(name))
        })
      );
      if (pairCapacity === 128) assertMultipartLines(result);
      else {
        expect(result['lengths']).toEqual([4, 0, 0, 0, 0, 0]);
        expect(result['weighted']).toEqual([8, 0, 0, 0, 0, 0]);
        expect(result['segments']).toEqual([1, 0, 0, 0, 0, 0]);
        expect(result['overflow']).toEqual([1]);
        expect(result['candidateOverflow']).toEqual([1]);
      }
    }
  });
for (const backend of ['group', 'zonal'] as const)
  for (const statistic of ['count', 'sum', 'mean', 'minimum', 'maximum'] as const)
    it(`WebGPU mixed choropleth ${backend} ${statistic} matches Shapely`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const props = {
        ...choroplethOperationProps,
        backend,
        statistic,
        output: {
          counts: 'counts',
          ...(statistic === 'count' ? {} : {featureValues: 'sums'}),
          overflow: 'overflow',
          uncertainCount: 'uncertainCount'
        }
      };
      for (const operation of [
        createPointsInPolygonsChoroplethOperation(props),
        createCPUPointsInPolygonsChoroplethOperation(props)
      ]) {
        const result = getOperationResult(
          await runOperationGraph(device, {
            inputs: multipartInputs,
            operations: [operation],
            outputs: Object.values(props.output)
          })
        );
        assertMultipartRecipe(result, statistic);
      }
    });
it('WebGPU mixed choropleth propagates candidate loss in dependent counts/sums', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = {...choroplethOperationProps, candidateCapacity: 1};
  for (const operation of [
    createPointsInPolygonsChoroplethOperation(props),
    createCPUPointsInPolygonsChoroplethOperation(props)
  ]) {
    const result = getOperationResult(
      await runOperationGraph(device, {
        inputs: multipartInputs,
        operations: [operation],
        outputs: Object.values(props.output).filter((name): name is string => Boolean(name))
      })
    );
    expect(result['counts']).toEqual([1, 0, 0, 0, 0, 0]);
    expect(result['sums']).toEqual([1, 0, 0, 0, 0, 0]);
    expect(result['overflow']).toEqual([1]);
    expect(result['uncertainCount']).toEqual([0]);
  }
});
for (const matchCapacity of [2, 13])
  it(`WebGPU mixed buffer operations match the oracle at capacity ${matchCapacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props = {...bufferOperationProps, matchCapacity};
    for (const operation of [
      createBufferSelectionOperation(props),
      createCPUBufferSelectionOperation(props)
    ]) {
      const result = getOperationResult(
        await runOperationGraph(device, {
          inputs: multipartInputs,
          operations: [operation],
          outputs: Object.values(props.output)
        })
      );
      if (matchCapacity === 13) assertMultipartBuffer(result);
      else {
        expect(result['ids']).toEqual([90, 30]);
        expect(result['count']).toEqual([2]);
        expect(result['requiredCount']).toEqual([5]);
        expect(result['draw']).toEqual([2]);
        expect(result['matchOverflow']).toEqual([1]);
        expect(result['candidateOverflow']).toEqual([0]);
        expect(result['overflow']).toEqual([0]);
      }
    }
  });

for (const rule of ['closed', 'open', 'half-open'] as const)
  it(`WebGPU multipart point ${rule} matches Shapely on all ports`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGeometryFixture(device, {
      inputs: multipartInputs,
      outputs: joinOutputs,
      create: ({inputs, outputs}) => createMultipartJoin(inputs, outputs, rule)
    });
    try {
      assertMultipartJoin(await fixture.run(), rule);
    } finally {
      fixture.destroy();
    }
  });
it('WebGPU multipart buffer selects any part with stable feature identities', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: multipartInputs,
    outputs: bufferOutputs,
    create: ({inputs, outputs}) => createMultipartBuffer(inputs, outputs)
  });
  try {
    assertMultipartBuffer(await fixture.run());
  } finally {
    fixture.destroy();
  }
});
it('WebGPU multipart line length and feature weights match Shapely', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: multipartInputs,
    outputs: lineOutputs,
    create: ({inputs, outputs}) => createMultipartLines(inputs, outputs)
  });
  try {
    assertMultipartLines(await fixture.run());
  } finally {
    fixture.destroy();
  }
});
for (const backend of ['zonal', 'group'] as const)
  for (const statistic of ['count', 'sum', 'mean', 'minimum', 'maximum'] as const)
    it(`WebGPU multipart choropleth ${backend} ${statistic} counts once per polygon`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createGeometryFixture(device, {
        inputs: multipartInputs,
        outputs: recipeOutputs,
        create: ({inputs, outputs}) => ({
          getCommandNodes: graph => {
            createMultipartRecipe(graph, inputs, outputs, backend, statistic);
            return [];
          }
        })
      });
      try {
        assertMultipartRecipe(await fixture.run(), statistic);
      } finally {
        fixture.destroy();
      }
    });
it('WebGPU candidate loss on later parts flags the owning feature', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: multipartInputs,
    outputs: joinOutputs,
    create: ({inputs, outputs}) => createMultipartJoin(inputs, outputs, 'closed', 1)
  });
  try {
    const result = await fixture.run();
    expect(result['overflow']).toEqual([1]);
    expect(result['candidateOverflow']).toEqual([1]);
    expect(result['truncated']).toEqual([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
  } finally {
    fixture.destroy();
  }
});

it('WebGPU multipart membership clears on a later encoding with no matches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: multipartInputs,
    outputs: joinOutputs,
    create: ({inputs, outputs}) => createMultipartJoin(inputs, outputs)
  });
  try {
    assertMultipartJoin(await fixture.run(), 'closed');
    fixture.writeInput(
      'points',
      new Float32Array(multipartInputs['points'].values.length).fill(20)
    );
    const result = await fixture.run();
    expect(result['membership']).toEqual(Array(78).fill(0));
    expect(result['assignment']).toEqual(Array(13).fill(4294967295));
    expect(result['counts']).toEqual(Array(6).fill(0));
    expect(result['truncated']).toEqual(Array(13).fill(0));
    for (const field of [
      'count',
      'requiredCount',
      'matchOverflow',
      'matchCandidateOverflow',
      'overflow',
      'candidateOverflow',
      'candidateCount',
      'uncertainCount'
    ])
      expect(result[field], field).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it('WebGPU multipart buffer clears feature outputs when radius becomes invalid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: multipartInputs,
    outputs: bufferOutputs,
    create: ({inputs, outputs}) => createMultipartBuffer(inputs, outputs)
  });
  try {
    assertMultipartBuffer(await fixture.run());
    fixture.writeInput('radius', Float32Array.of(-1));
    const result = await fixture.run();
    expect(result['mask']).toEqual(Array(13).fill(0));
    expect(result['rows']).toEqual(Array(13).fill(4294967295));
    expect(result['distances']).toEqual(Array(13).fill(-1));
    for (const field of [
      'count',
      'requiredCount',
      'draw',
      'matchOverflow',
      'matchCandidateOverflow',
      'overflow',
      'candidateOverflow'
    ])
      expect(result[field], field).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});
