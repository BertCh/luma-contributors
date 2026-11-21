// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import {readUint32} from '../../utils/gpu-contributor-test-utils';

type OPTICSStatusFields<Value> = {
  iterationCount: Value;
  converged: Value;
  invalidCount: Value;
};

export function createOPTICSStatusFields<Value>(
  create: (name: keyof OPTICSStatusFields<Value>) => Value
): OPTICSStatusFields<Value> {
  return {
    iterationCount: create('iterationCount'),
    converged: create('converged'),
    invalidCount: create('invalidCount')
  };
}

export function getOPTICSStatusViews(fields: OPTICSStatusFields<{view: GraphDataView<'uint32'>}>) {
  return createOPTICSStatusFields(name => fields[name].view);
}

export async function readOPTICSStatus(fields: OPTICSStatusFields<{buffer: Buffer}>) {
  const values = await Promise.all(Object.values(fields).map(field => readUint32(field.buffer, 1)));
  return values.flat();
}

export function getOPTICSStatusValues(fields: OPTICSStatusFields<ArrayLike<number>>) {
  return (['iterationCount', 'converged', 'invalidCount'] as const).map(name => {
    const values = fields[name];
    expect(values, name).toHaveLength(1);
    return values[0];
  });
}
