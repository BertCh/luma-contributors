// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {Buffer} from '@luma.gl/core';
import {expect, test} from 'vitest';
import {GPUSegregation} from '../../../src/gpu-spatial-analysis/segregation/gpu-segregation';
import {getGPUSegregationParameterValues} from '../../../src/gpu-spatial-analysis/segregation/segregation-parameters';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';
import {
  createExampleWeights,
  SEGREGATION_EXAMPLE_SOURCES
} from '../region-statistics/review-example-wiring';
import {
  DEFINITION_COUNTS,
  DEFINITION_WEIGHTS,
  getSegregationDefinition
} from './segregation-definition-fixtures';

for (const {name, source} of SEGREGATION_EXAMPLE_SOURCES) {
  test(`R19: ${name} actual weights construction consumes missing-edge loss and recovers`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new GraphRig(device);
    const allocate = (data: Float32Array | Uint32Array) =>
      device.createBuffer({data, usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST});
    const buffers = {
      offsets: allocate(Uint32Array.of(0, 1, 2)),
      neighbors: allocate(Uint32Array.of(1, 0)),
      weights: allocate(Float32Array.of(1, 0)),
      distances: allocate(Float32Array.of(1, 1)),
      overflow: allocate(Uint32Array.of(1)),
      capacity: 2
    };
    try {
      const indices = rig.output('float32', 13);
      rig.run(
        new GPUSegregation({
          groupCount: 2,
          groupCounts: rig.input(Float32Array.of(10, 0, 0, 10), 'float32'),
          selfWeight: 1,
          scales: [createExampleWeights(rig.graph, source, buffers)],
          output: {indices: indices.view}
        })
      );
      (await indices.readFloat32()).forEach((value, row) => {
        if (row === 2) expect(value).toBeCloseTo(Math.log(2), 6);
        else expect(value).toBeNaN();
      });
      buffers.weights.write(Float32Array.of(1, 1));
      buffers.overflow.write(Uint32Array.of(0));
      rig.resubmit();
      const expected = [0, 0, Math.log(2), 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0.5, 0.5];
      (await indices.readFloat32()).forEach((value, row) =>
        expect(value).toBeCloseTo(expected[row], 5)
      );
    } finally {
      rig.destroy();
      for (const buffer of [
        buffers.offsets,
        buffers.neighbors,
        buffers.weights,
        buffers.distances,
        buffers.overflow
      ])
        buffer.destroy();
    }
  });
}

for (const spatialForm of ['environment', 'smoothed-population'] as const) {
  for (const loss of ['overflow', 'candidateOverflow'] as const) {
    test(`R2/R18: ${spatialForm} ${loss} suppresses dependent outputs and clears on resubmission`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new GraphRig(device);
      try {
        const flag = rig.output('uint32', 1),
          weights = rig.output('float32', 2);
        flag.buffer.write(Uint32Array.of(1));
        weights.buffer.write(Float32Array.of(1, 0));
        const outputs = {
          indices: rig.output('float32', 26),
          localEnvironment: rig.output('float32', 8),
          localEntropy: rig.output('float32', 4),
          localDissimilarity: rig.output('float32', 8),
          localTheil: rig.output('float32', 4)
        };
        rig.run(
          new GPUSegregation({
            groupCount: 2,
            spatialForm,
            groupCounts: rig.input(Float32Array.of(10, 0, 0, 10), 'float32'),
            selfWeight: 1,
            scales: [
              null,
              {
                offsets: rig.input(Uint32Array.of(0, 1, 2), 'uint32'),
                neighbors: rig.input(Uint32Array.of(1, 0), 'uint32'),
                weights: weights.view,
                status: {count: rig.input(Uint32Array.of(2), 'uint32'), [loss]: flag.view}
              }
            ],
            output: {
              indices: outputs.indices.view,
              localEnvironment: outputs.localEnvironment.view,
              localEntropy: outputs.localEntropy.view,
              localDissimilarity: outputs.localDissimilarity.view,
              localTheil: outputs.localTheil.view
            }
          })
        );
        for (const [name, output] of Object.entries(outputs)) {
          const actual = await output.readFloat32();
          actual.slice(actual.length / 2).forEach((value, row) => {
            if (name === 'indices' && row === 2 && spatialForm === 'environment')
              expect(value).toBeCloseTo(Math.log(2), 6);
            else expect(value).toBeNaN();
          });
          expect(actual.slice(0, actual.length / 2).every(Number.isFinite)).toBe(true);
        }
        weights.buffer.write(Float32Array.of(1, 1));
        flag.buffer.write(Uint32Array.of(0));
        rig.resubmit();
        const indices = await outputs.indices.readFloat32();
        // Both environments are (1/2,1/2): H=D=D_g=A_g=0, isolation/exposure=1/2.
        const expected = [0, 0, Math.log(2), 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0.5, 0.5];
        expected.forEach((value, row) => expect(indices[13 + row]).toBeCloseTo(value, 5));
        expect(await outputs.localEnvironment.readFloat32()).toEqual([
          1, 0, 0, 1, 0.5, 0.5, 0.5, 0.5
        ]);
        (await outputs.localEntropy.readFloat32())
          .slice(2)
          .forEach(value => expect(value).toBeCloseTo(Math.log(2), 6));
        expect((await outputs.localDissimilarity.readFloat32()).slice(4)).toEqual([0, 0, 0, 0]);
        (await outputs.localTheil.readFloat32())
          .slice(2)
          .forEach(value => expect(value).toBeCloseTo(0, 6));
      } finally {
        rig.destroy();
      }
    });
  }
}

test('R7: identical mixed populations have zero Atkinson at scales 1 and 1e-32', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scale of [1, 1e-32]) {
    const rig = new AnalysisRig(device);
    try {
      const indices = rig.output('float32', 13);
      rig.run(
        new GPUSegregation({
          groupCount: 2,
          groupCounts: rig.input(new Float32Array(4).fill(scale), 'float32'),
          atkinsonInequalityAversion: 0.5,
          output: {indices}
        })
      );
      const actual = await rig.readFloat(indices);
      // Both groups have the same 1/2 share in each unit. The formula is scale invariant.
      actual.slice(7, 9).forEach(value => expect(value).toBeCloseTo(0, 5));
    } finally {
      rig.destroy();
    }
  }
});

test('R8: explicit diagonal weights leave the reciprocal mixed environments unchanged', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const diagonal of [false, true]) {
    const rig = new AnalysisRig(device);
    try {
      const indices = rig.output('float32', 13),
        localEnvironment = rig.output('float32', 4),
        localTheil = rig.output('float32', 2);
      rig.run(
        new GPUSegregation({
          groupCount: 2,
          groupCounts: rig.input(Float32Array.of(10, 0, 0, 10), 'float32'),
          selfWeight: 1,
          scales: [
            rig.weights(
              diagonal
                ? {offsets: [0, 2, 4], neighbors: [0, 1, 0, 1], weights: [1, 1, 1, 1]}
                : {offsets: [0, 1, 2], neighbors: [1, 0], weights: [1, 1]}
            )
          ],
          output: {indices, localEnvironment, localTheil}
        })
      );
      const actual = await rig.readFloat(indices);
      actual.slice(0, 2).forEach(value => expect(value).toBeCloseTo(0, 6));
      expect(await rig.readFloat(localEnvironment)).toEqual([0.5, 0.5, 0.5, 0.5]);
      (await rig.readFloat(localTheil)).forEach(value => expect(value).toBeCloseTo(0, 6));
    } finally {
      rig.destroy();
    }
  }
});

for (const spatialForm of ['environment', 'smoothed-population'] as const) {
  test(`segregation independent NumPy oracle: ${spatialForm}, every output`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const expected = getSegregationDefinition(spatialForm);
    for (const countScale of [1, 8]) {
      const rig = new AnalysisRig(device);
      try {
        const output = {
          indices: rig.output('float32', 42),
          localEnvironment: rig.output('float32', 48),
          localEntropy: rig.output('float32', 16),
          localDissimilarity: rig.output('float32', 48),
          localTheil: rig.output('float32', 16)
        };
        const constructionProps = {
          groupCount: 3,
          groupCounts: rig.input(
            Float32Array.from(DEFINITION_COUNTS, value => value * countScale),
            'float32'
          ),
          scales: [null, rig.weights(DEFINITION_WEIGHTS)],
          spatialForm,
          parameters: rig.input(
            getGPUSegregationParameterValues({selfWeight: 0, atkinsonInequalityAversion: 0.3}),
            'float32'
          ),
          output: {...output}
        };
        const contributor = new GPUSegregation(constructionProps);
        constructionProps.groupCount = 2;
        constructionProps.scales[1] = null;
        constructionProps.output.indices = rig.output('float32', 42);
        constructionProps.output.localTheil = rig.output('float32', 16);
        rig.run(contributor);
        for (const port of Object.keys(expected) as (keyof typeof expected)[]) {
          const actual = await rig.readFloat(output[port]);
          expected[port].forEach((value, row) => {
            if (Number.isNaN(value)) {
              expect(actual[row], `${port}[${row}]`).toBeNaN();
            } else {
              expect(Number.isFinite(actual[row]), `${port}[${row}]`).toBe(true);
              // Small populations, three groups and b=0.3: allow f32 reductions,
              // logarithms and exponentiation; an island error here is 1/7.
              expect(Math.abs(actual[row] - value), `${port}[${row}]`).toBeLessThan(2e-5);
            }
          });
        }
      } finally {
        rig.destroy();
      }
    }
  });
}

test('populated islands do not add entropy segregation when all defined environments match the population', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const indices = rig.output('float32', 13);
    const localTheil = rig.output('float32', 8);
    rig.run(
      new GPUSegregation({
        groupCount: 2,
        groupCounts: rig.input(new Float32Array(16).fill(1), 'float32'),
        scales: [
          rig.weights({offsets: [0, 1, 2, 2, 2, 2, 2, 2, 2], neighbors: [1, 0], weights: [1, 1]})
        ],
        selfWeight: 0,
        output: {indices, localTheil}
      })
    );
    expect((await rig.readFloat(indices))[0]).toBeCloseTo(0, 6);
    const local = await rig.readFloat(localTheil);
    expect(local.slice(0, 2)).toEqual([0, 0]);
    expect(local.slice(2).every(Number.isNaN)).toBe(true);
  } finally {
    rig.destroy();
  }
});
