// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

type CompactPublisherFixture = {
  writeInput(name: string, values: Uint32Array): void;
  run(): Promise<Record<string, number[]>>;
};

/** Shared assertion sequence for the device fixture and host mutation regression. */
export async function expectCompactPublicationReuse(
  fixture: CompactPublisherFixture
): Promise<void> {
  // Saturated requiredCount is a lower bound. Both causes must publish and clear.
  const readStatus = (result: Record<string, number[]>) =>
    ['count', 'requiredCount', 'overflow', 'candidateOverflow'].map(name => result[name]);
  fixture.writeInput('total', new Uint32Array([1]));
  fixture.writeInput('candidateFlag', new Uint32Array([0]));
  expect(readStatus(await fixture.run())).toEqual([[1], [1], [0], [0]]);
  fixture.writeInput('total', new Uint32Array([0xffffffff]));
  fixture.writeInput('candidateFlag', new Uint32Array([1]));
  expect(readStatus(await fixture.run())).toEqual([[2], [0xffffffff], [1], [1]]);
  fixture.writeInput('total', new Uint32Array([1]));
  fixture.writeInput('candidateFlag', new Uint32Array([0]));
  expect(readStatus(await fixture.run())).toEqual([[1], [1], [0], [0]]);
}
