// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';

test.each([
  '~result.json',
  '~result~thumbnail.json'
])('H1 committed intersection twin %s headlines its active count', suffix => {
  const twin = JSON.parse(
    readFileSync(
      new URL(
        `../../../../../docs/lens-records/twins/gpu-spatial-analysis.segment-intersection${suffix}`,
        import.meta.url
      ),
      'utf8'
    )
  );
  expect(twin.keyNumber).toEqual({label: 'points', value: '1,802'});
});
