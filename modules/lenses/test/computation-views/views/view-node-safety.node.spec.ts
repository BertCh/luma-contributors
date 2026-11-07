// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {h, viewNodeToHtml} from '../../../src/computation-views/view-node';
import {sampleEvenly} from '../../../src/computation-views/views/result-view';

test('sampleEvenly handles degenerate limits without producing undefined', () => {
  const values = [1, 2, 3, 4, 5];
  expect(sampleEvenly(values, 1)).toEqual([5]);
  expect(sampleEvenly(values, 0)).toEqual([]);
  expect(sampleEvenly(values, 2)).toEqual([1, 5]);
  expect(sampleEvenly(values, 9)).toBe(values);
});

test('h rejects attribute names that could break serialised markup', () => {
  expect(viewNodeToHtml(h('div', {'data-key': 'a"b', 'aria-label': 'x'}))).toBe(
    '<div data-key="a&quot;b" aria-label="x"></div>'
  );
  expect(() => h('div', {'x onload="1"': 'y'})).toThrow();
});
