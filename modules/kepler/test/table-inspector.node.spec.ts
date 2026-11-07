// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  validateTablePageRequest,
  type KeplerTablePageRequest
} from '../src/exploration/gpu-table-inspector';

const request: KeplerTablePageRequest = {
  offset: 0,
  limit: 25,
  columns: ['value'],
  population: 'selected'
};
describe('resident native table request admission', () => {
  test('rejects unbounded and malformed inspection requests before allocation', () => {
    for (const change of [
      {offset: -1},
      {offset: Infinity},
      {limit: 0},
      {limit: 26},
      {columns: ['value', 'value']},
      {columns: ['missing']}
    ]) {
      expect(() => validateTablePageRequest({...request, ...change}, ['value'], 25)).toThrow();
    }
    expect(() => validateTablePageRequest(request, ['value'], 25)).not.toThrow();
  });
  test('accepts bounded identity-only pages without synthetic source fields', () => {
    expect(() => validateTablePageRequest({...request, columns: []}, [], 25)).not.toThrow();
  });
  test('accepts sorting hidden source fields but rejects unknown sort columns', () => {
    expect(() =>
      validateTablePageRequest(
        {...request, sort: {column: 'hidden', direction: 'descending'}},
        ['value', 'hidden'],
        25
      )
    ).not.toThrow();
    expect(() =>
      validateTablePageRequest(
        {...request, sort: {column: 'missing', direction: 'ascending'}},
        ['value'],
        25
      )
    ).toThrow();
  });
});
