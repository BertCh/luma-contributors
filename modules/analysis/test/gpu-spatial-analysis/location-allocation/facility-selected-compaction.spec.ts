// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createFacilitySelectedCompaction} from '../../../src/gpu-spatial-analysis/location-allocation/facility-selected-compaction';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {getFacilityCompactionRoundInputs} from './facility-compaction-fixtures';

for (const facilityCount of [1, 257, 65537]) {
  it(`compacts ${facilityCount} facilities in ascending order and clears inactive rounds`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    // Hand-picked boundaries cover a partial block and all three hierarchy levels.
    const facilities = [0, 255, 256, 65535, 65536].filter(facility => facility < facilityCount);
    const selected = new Uint32Array(facilityCount);
    for (const facility of facilities) selected[facility] = 1;
    const slots = Math.min(7, facilityCount);
    const list = rig.output('uint32', slots);
    const inverse = rig.output('uint32', facilityCount);
    const status = rig.input(new Uint32Array(7), 'uint32', 7);
    const state = rig.input(new Uint32Array(3), 'uint32', 3);
    const dispatch = rig.input(Uint32Array.of(1, 1, 1), 'uint32', 3);
    const nodes = createFacilitySelectedCompaction(rig.graph, {
      id: 'compaction',
      facilityCount,
      slots,
      selected: rig.input(selected, 'uint32', facilityCount),
      status,
      swapState: state,
      swapDispatch: dispatch,
      selectedList: list.view,
      slotOfFacility: inverse.view
    })(0);
    for (const node of nodes) rig.graph.add(node);
    const compiled = await rig.graph.compileAsync();
    try {
      for (const mode of [
        'active',
        'invalid',
        'active',
        'stopped',
        'active',
        'disarmed'
      ] as const) {
        const inputs = getFacilityCompactionRoundInputs(mode);
        rig.bufferOf(status).write(inputs.status);
        rig.bufferOf(state).write(inputs.state);
        rig.bufferOf(dispatch).write(inputs.dispatch);
        submitGraph(device, compiled, undefined);
        // Disarmed also sets invalid: an accidentally armed publisher would clear these outputs.
        // A disarmed outer gate preserves the preceding active result; invalid/stopped
        // rounds with an armed gate clear every slot and inverse entry to NONE.
        const isActive = mode === 'active' || mode === 'disarmed';
        const expectedList = new Array<number>(slots).fill(0xffffffff);
        const expectedInverse = new Array<number>(facilityCount).fill(0xffffffff);
        if (isActive) {
          facilities.forEach((facility, slot) => {
            expectedList[slot] = facility;
            expectedInverse[facility] = slot;
          });
        }
        expect(await readUint32(list.buffer, slots)).toEqual(expectedList);
        expect(await readUint32(inverse.buffer, facilityCount)).toEqual(expectedInverse);
      }
    } finally {
      compiled.destroy();
      rig.destroy();
      device.destroy();
    }
  });
}
