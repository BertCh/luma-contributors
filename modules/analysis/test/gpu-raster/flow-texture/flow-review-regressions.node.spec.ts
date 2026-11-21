// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import * as typescript from 'typescript';
import {Buffer} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCoordinateParticleState,
  createCoordinateOffsets,
  expectCoordinateStreamline
} from './flow-coordinate-fixtures';
import {getGPUParticleAdvectionParameterValues} from '../../../src/raster-analysis/particle-advection/particle-advection-parameters';
import {getGPULagrangianFlowMapParameterValues} from '../../../src/raster-analysis/lagrangian-diagnostics/gpu-lagrangian-flow-map';
import {getGPUStreamlinesParameterValues} from '../../../src/raster-analysis/flow-texture/streamlines-parameters';
import {getGPULineIntegralConvolutionParameterValues} from '../../../src/raster-analysis/flow-texture/line-integral-convolution-parameters';

it('B1 particle state and sentinel offsets support staging readback', () => {
  const rig = createParityGraph(createNullWebGPUDevice());
  try {
    const state = createCoordinateParticleState(rig);
    const offsets = createCoordinateOffsets(rig);
    for (const port of [...Object.values(state), offsets]) {
      expect(port.buffer.usage & Buffer.COPY_SRC).toBe(Buffer.COPY_SRC);
      expect(port.buffer.usage & Buffer.COPY_DST).toBe(Buffer.COPY_DST);
    }
  } finally {
    rig.destroy();
  }
});

it('B3 the twelve-word LIC packet retains world-frame origins', () => {
  expect(
    Array.from(
      getGPULineIntegralConvolutionParameterValues({
        fieldPlacement: {origin: [100, 200], cellSize: [2, 3]},
        outputPlacement: {origin: [102, 203], cellSize: [2, 3]}
      })
    )
  ).toEqual([100, 200, 2, 3, 102, 203, 2, 3, 0.5, 0, 0, 0]);
});

it('S1 default spawn far corners are computed in the rebased frame', () => {
  for (const cellHeight of [1, -1]) {
    const values = getGPUParticleAdvectionParameterValues(
      {
        fieldPlacement: {origin: [2 ** 53, 2 ** 53], cellSize: [1, cellHeight]},
        localOrigin: [2 ** 53, 2 ** 53],
        timeStep: 1
      },
      [1, 1]
    );
    expect(Array.from(values.subarray(8))).toEqual([0, 0, 1, cellHeight]);
  }
});

it('S3 the coordinate oracle rejects incorrect fractional seeds, missing Y and NaN Y', () => {
  // Independent integer Philox calculation: counter [0,0,0,0], key [8,2], ten rounds gives
  // words [1295583915,1141723941,473967449,1081980456]. Top 24 bits / 2^24 * 100:
  const values = Float32Array.from({length: 34}, (_, index) =>
    index % 2 ? 26.582831144332886 : 30.165159702301025 + (index / 2 - 8) * 0.25
  );
  expectCoordinateStreamline(values);
  const shifted = values.map((value, index) => (index % 2 ? value : value + 0.125));
  expect(() => expectCoordinateStreamline(shifted)).toThrow();
  const missing = values.subarray(0, 33);
  expect(() => expectCoordinateStreamline(missing)).toThrow();
  const invalid = values.map((value, index) => (index % 2 ? NaN : value));
  expect(() => expectCoordinateStreamline(invalid)).toThrow();
});

it('S2 exported settings require named placements', () => {
  for (const [file, typeName, fields] of [
    [
      'particle-advection/particle-advection-parameters.ts',
      'GPUParticleAdvectionSettings',
      ['fieldPlacement']
    ],
    [
      'lagrangian-diagnostics/gpu-lagrangian-flow-map.ts',
      'GPULagrangianFlowMapSettings',
      ['fieldPlacement']
    ],
    [
      'flow-texture/line-integral-convolution-parameters.ts',
      'GPULineIntegralConvolutionSettings',
      ['fieldPlacement', 'outputPlacement']
    ],
    [
      'flow-texture/streamlines-parameters.ts',
      'GPUStreamlinesSettings',
      ['fieldPlacement', 'gridPlacement']
    ]
  ] as const) {
    const source = typescript.createSourceFile(
      file,
      readFileSync(new URL('../../../src/raster-analysis/' + file, import.meta.url), 'utf8'),
      typescript.ScriptTarget.Latest,
      true
    );
    const declaration = source.statements.find(
      statement => typescript.isTypeAliasDeclaration(statement) && statement.name.text === typeName
    );
    if (
      !declaration ||
      !typescript.isTypeAliasDeclaration(declaration) ||
      !typescript.isTypeLiteralNode(declaration.type)
    )
      throw new Error('Missing settings declaration');
    for (const field of fields) {
      const property = declaration.type.members.find(
        member => member.name?.getText(source) === field
      );
      if (!property || !typescript.isPropertySignature(property))
        throw new Error('Missing placement');
      expect(Boolean(property.questionToken), typeName + '.' + field).toBe(false);
    }
  }
});

it('S2 packer input types require every placement and reader types expose origins', () => {
  // These checks also enter the lead's test typecheck; no compiler is launched in this stream.
  const checkRejectedInputs = () => {
    // @ts-expect-error field and output placement are mandatory.
    getGPULineIntegralConvolutionParameterValues({});
    // @ts-expect-error a particle field placement is mandatory.
    getGPUParticleAdvectionParameterValues({timeStep: 1}, [1, 1]);
    // @ts-expect-error a flow-map field placement is mandatory.
    getGPULagrangianFlowMapParameterValues({
      startTime: 0,
      timeStep: 1,
      sliceStartTime: 0,
      sliceEndTime: 1
    });
    // @ts-expect-error both a field and occupancy placement are mandatory.
    getGPUStreamlinesParameterValues({stepLength: 1});
    // @ts-expect-error specifying only the field leaves the output placement absent.
    getGPULineIntegralConvolutionParameterValues({
      fieldPlacement: {origin: [0, 0], cellSize: [1, 1]}
    });
    // @ts-expect-error specifying only the field leaves the occupancy placement absent.
    getGPUStreamlinesParameterValues({
      stepLength: 1,
      fieldPlacement: {origin: [0, 0], cellSize: [1, 1]}
    });
  };
  void checkRejectedInputs;
  const readPlacementOrigins = (
    settings: import('../../../src/raster-analysis/flow-texture/line-integral-convolution-parameters').GPULineIntegralConvolutionSettings
  ) => settings.fieldPlacement.origin[0] + settings.outputPlacement.origin[0];
  expect(
    readPlacementOrigins({
      fieldPlacement: {origin: [100, 200], cellSize: [2, 3]},
      outputPlacement: {origin: [102, 203], cellSize: [2, 3]}
    })
  ).toBe(202);
});
