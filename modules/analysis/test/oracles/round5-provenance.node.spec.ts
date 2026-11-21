// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync, readdirSync} from 'node:fs';
import {expect, it} from 'vitest';

it('B2 Round-5 fixtures retain verifiable whole-script provenance', () => {
  const generator = readFileSync(new URL('./parity-wp5-round5.py', import.meta.url));
  const generatorHash = createHash('sha256').update(generator).digest('hex');
  const directory = new URL('./fixtures/', import.meta.url);
  const files = readdirSync(directory).filter(name => /^parity-wp5-round5-.*\.json$/.test(name));
  expect(files).toHaveLength(11);
  for (const name of files) {
    const fixture: {generatorSha256: string} = JSON.parse(
      readFileSync(new URL(name, directory), 'utf8')
    );
    expect(fixture.generatorSha256, name).toBe(generatorHash);
  }
});
