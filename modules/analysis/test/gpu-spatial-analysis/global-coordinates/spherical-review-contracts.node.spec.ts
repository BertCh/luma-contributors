// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';

it('B5 poisons every containment result and status buffer before device execution', () => {
  const source = readFileSync(new URL('./global-coordinates.spec.ts', import.meta.url), 'utf8');
  const fixture = source.slice(source.indexOf("it('ENT-02 spherical containment preserves"));
  // Host audit only: four sentinel writes are required so skipped GPU writers cannot pass as 0.
  expect(fixture).toMatch(
    /for \(const field of \[\.\.\.contains, \.\.\.status\]\)\s*field\.buffer\.write\(Uint32Array\.of\(0xffffffff\)\)/
  );
});

it('S4 labels the 257-word byte probe as synthetic and keeps separate valid layout controls', () => {
  const source = readFileSync(
    new URL('./spherical-status-and-tolerances.node.spec.ts', import.meta.url),
    'utf8'
  );
  expect(source).toContain('Synthetic endianness probe');
  expect(source).toContain('valid producer count layouts');
});
