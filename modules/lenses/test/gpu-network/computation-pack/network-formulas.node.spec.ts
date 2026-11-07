// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {NETWORK_FORMULA_REFS} from '../../../src/gpu-network/computation-pack/network-formulas';
import {describeStructure} from '../../../src/computation-lenses/scenarios/gpu-network/network-reachability.lens';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';

test('reachability formula refs are recorded with wgsl-bound symbols', async () => {
  const device = createNullWebGPUDevice();
  try {
    const record = await describeStructure(device);
    // Generated WGSL text is not recorded: published luma.gl exposes no kernel source on the graph,
    // so only the binding declaration of each symbol is checked here.
    for (const formula of Object.values(NETWORK_FORMULA_REFS).filter(reference =>
      reference.id.startsWith('gpu-network.reachability.')
    )) {
      expect(record.refs.formulas.some(reference => reference.id === formula.id)).toBe(true);
      for (const symbol of formula.symbols) {
        expect(symbol.binding).toMatch(/^wgsl:/);
      }
    }
  } finally {
    device.destroy();
  }
});
