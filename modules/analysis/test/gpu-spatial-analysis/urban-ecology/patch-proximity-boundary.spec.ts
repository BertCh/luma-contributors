// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addPatchProximityRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/patch-proximity';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {publishBuildingTessellationOutput as publish} from './building-tessellation-test-utils';

it.each([
  [1e20, 1],
  [1e-20, 1],
  [1.2 * 2 ** 126, 2 ** -125]
] as const)('both proximity routes preserve representable distances and inverse-square terms at cell size (%s,%s)', async (horizontalSize, verticalSize) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'proximity-numeric-scale');
  try {
    const horizontal = Math.fround(horizontalSize);
    const vertical = Math.fround(verticalSize);
    const props = {
      patches: fixture.input('patches', new Uint32Array([0, 0xffffffff, 1]), 'uint32', 3),
      patchClasses: fixture.input('classes', new Uint32Array(2), 'uint32', 2),
      width: 3,
      height: 1,
      cellSize: [horizontal, vertical] as const,
      edgeDepth: 0,
      searchRadius: horizontal
    };
    const boundary = addPatchProximityRecipe(fixture.graph, {
      ...props,
      id: 'boundary',
      mode: 'boundary'
    });
    const reference = addPatchProximityRecipe(fixture.graph, {
      ...props,
      id: 'reference',
      mode: 'reference'
    });
    const outputs = [
      publish(fixture, 'boundary-metrics', boundary.outputs.metrics),
      publish(fixture, 'reference-metrics', reference.outputs.metrics)
    ];
    fixture.run();
    // One empty cell separates two one-cell patches: rectangle gap=cell width, patch area=width*height.
    const area = Math.fround(horizontal * vertical);
    const expected = [area, area, horizontal, Math.fround(area / (horizontal * horizontal))];
    for (const output of outputs) {
      const values = await fixture.readFloat32(output, 8);
      for (let patch = 0; patch < 2; patch++)
        for (const [field, value] of expected.entries()) {
          const actual = values[4 * patch + field];
          if (value === 0) expect(actual).toBe(0);
          else expect(actual / value).toBeCloseTo(1, 5);
        }
    }
  } finally {
    fixture.destroy();
  }
});

it('boundary proximity publishes exact metrics and reductions into valid packed output ranges', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'boundary-packed');
  try {
    const packed = fixture.output('packed', 'uint32', 256);
    const words = (byteOffset: number, length = 1) =>
      fixture.graph.createDataView(packed.view.buffer, {format: 'uint32', byteOffset, length});
    addPatchProximityRecipe(fixture.graph, {
      mode: 'boundary',
      patches: fixture.input('patches', new Uint32Array([0, 0xffffffff, 1]), 'uint32', 3),
      patchClasses: fixture.input('classes', new Uint32Array([0, 0]), 'uint32', 2),
      width: 3,
      height: 1,
      cellSize: [1, 1],
      edgeDepth: 0,
      searchRadius: 1,
      output: {
        core: words(0, 3),
        metrics: fixture.graph.createDataView(packed.view.buffer, {
          format: 'float32x4',
          byteOffset: 12,
          length: 2
        }),
        validity: words(256, 2),
        status: {
          count: words(512),
          requiredCount: words(516),
          overflow: words(768),
          candidateOverflow: words(520)
        }
      }
    });
    fixture.run();
    const values = await fixture.readFloat32(packed, 256);
    const flags = await fixture.readUint32(packed, 256);
    expect(flags.slice(0, 3)).toEqual([1, 0, 1]);
    expect(values.slice(3, 11)).toEqual([1, 1, 1, 1, 1, 1, 1, 1]);
    expect(flags.slice(64, 66)).toEqual([1, 1]);
    expect(flags.slice(128, 131)).toEqual([2, 2, 0]);
    expect(flags[192]).toBe(0);
  } finally {
    fixture.destroy();
  }
});

/** Exhaustive binary64 cell-pair oracle, deliberately independent of boundary sorting. */
function oracle(
  labels: number[],
  classes: number[],
  width: number,
  height: number,
  cellSize: readonly [number, number],
  edgeDepth: number,
  searchRadius: number,
  center: boolean,
  erosion: boolean,
  open: boolean
) {
  const count = classes.length;
  const core = labels.map((owner, cell) => {
    if (owner >= count) return 0;
    const column = cell % width;
    const row = Math.floor(cell / width);
    if (erosion) {
      for (let vertical = -edgeDepth; vertical <= edgeDepth; vertical++)
        for (let horizontal = -edgeDepth; horizontal <= edgeDepth; horizontal++) {
          if (Math.abs(horizontal) + Math.abs(vertical) > edgeDepth) continue;
          const otherColumn = column + horizontal;
          const otherRow = row + vertical;
          if (otherColumn < 0 || otherColumn >= width || otherRow < 0 || otherRow >= height) {
            if (!open) return 0;
            continue;
          }
          if (labels[otherRow * width + otherColumn] !== owner) return 0;
        }
      return 1;
    }
    let closest = open
      ? Infinity
      : Math.min(
          (column + 0.5) * cellSize[0],
          (width - column - 0.5) * cellSize[0],
          (row + 0.5) * cellSize[1],
          (height - row - 0.5) * cellSize[1]
        );
    for (let other = 0; other < labels.length; other++)
      if (labels[other] !== owner) {
        const horizontal = Math.max(0, (Math.abs(column - (other % width)) - 0.5) * cellSize[0]);
        const vertical = Math.max(
          0,
          (Math.abs(row - Math.floor(other / width)) - 0.5) * cellSize[1]
        );
        closest = Math.min(closest, Math.hypot(horizontal, vertical));
      }
    return Number(closest > edgeDepth);
  });
  const distances = Array.from({length: count}, () => new Array<number>(count).fill(Infinity));
  for (let first = 0; first < labels.length; first++)
    for (let second = 0; second < labels.length; second++) {
      const source = labels[first];
      const partner = labels[second];
      if (source >= count || partner >= count || source === partner) continue;
      const horizontal =
        Math.max(0, Math.abs((first % width) - (second % width)) - (center ? 0 : 1)) * cellSize[0];
      const vertical =
        Math.max(
          0,
          Math.abs(Math.floor(first / width) - Math.floor(second / width)) - (center ? 0 : 1)
        ) * cellSize[1];
      distances[source][partner] = Math.min(
        distances[source][partner],
        Math.hypot(horizontal, vertical)
      );
    }
  const area = classes.map(
    (_, owner) => labels.filter(label => label === owner).length * cellSize[0] * cellSize[1]
  );
  const metrics = classes.flatMap((category, owner) => {
    let nearest = Infinity;
    let proximity = 0;
    for (let partner = 0; partner < count; partner++)
      if (partner !== owner && classes[partner] === category) {
        nearest = Math.min(nearest, distances[owner][partner]);
        if (distances[owner][partner] <= searchRadius)
          proximity += area[partner] / distances[owner][partner] ** 2;
      }
    const coreArea = labels.reduce(
      (sum, label, cell) => sum + Number(label === owner) * core[cell] * cellSize[0] * cellSize[1],
      0
    );
    return [area[owner], coreArea, center && nearest === Infinity ? NaN : nearest, proximity];
  });
  return {core, metrics};
}

it.each([
  {center: false, erosion: false, open: false},
  {center: true, erosion: false, open: false},
  {center: false, erosion: false, open: true},
  {center: true, erosion: true, open: true}
])('boundary proximity matches the CPU oracle and dense reference ($center,$erosion,$open)', async ({
  center,
  erosion,
  open
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'boundary-proximity-oracle');
  try {
    const background = 0xffffffff;
    const labels = [
      0,
      0,
      background,
      background,
      1,
      0,
      0,
      background,
      background,
      1,
      background,
      background,
      2,
      background,
      background
    ];
    const classes = [0, 0, 1, 0];
    const props = {
      patches: fixture.input('patches', Uint32Array.from(labels), 'uint32', 15),
      patchClasses: fixture.input('classes', Uint32Array.from(classes), 'uint32', 4),
      width: 5,
      height: 3,
      cellSize: [2, 3] as const,
      edgeDepth: erosion ? 1 : 1.25,
      searchRadius: 8,
      distanceModel: center ? ('cell-center' as const) : ('cell-boundary' as const),
      coreModel: erosion ? ('raster-erosion' as const) : ('physical' as const),
      coreBoundary: open ? ('open' as const) : ('edge' as const)
    };
    const boundary = addPatchProximityRecipe(fixture.graph, {
      ...props,
      id: 'boundary',
      mode: 'boundary',
      candidateRadius: 20,
      candidateCapacity: 225
    });
    const reference = addPatchProximityRecipe(fixture.graph, {...props, id: 'reference'});
    const core = publish(fixture, 'core', boundary.outputs.core);
    const metrics = publish(fixture, 'metrics', boundary.outputs.metrics);
    const referenceMetrics = publish(fixture, 'reference-metrics', reference.outputs.metrics);
    const validity = publish(fixture, 'validity', boundary.outputs.validity);
    const status = publish(fixture, 'status', boundary.status.candidateOverflow);
    fixture.run();
    const expected = oracle(
      labels,
      classes,
      5,
      3,
      props.cellSize,
      props.edgeDepth,
      8,
      center,
      erosion,
      open
    );
    expect(await fixture.readUint32(core, 15)).toEqual(expected.core);
    expect(await fixture.readUint32(validity, 4)).toEqual([1, 1, 1, 1]);
    expect(await fixture.readUint32(status, 1)).toEqual([0]);
    const values = await fixture.readFloat32(metrics, 16);
    const exact = await fixture.readFloat32(referenceMetrics, 16);
    for (const [index, value] of expected.metrics.entries()) {
      if (Number.isNaN(value)) {
        expect(values[index]).toBeNaN();
        expect(exact[index]).toBeNaN();
      } else if (!Number.isFinite(value)) {
        expect(values[index]).toBe(value);
        expect(exact[index]).toBe(value);
      } else {
        expect(values[index]).toBeCloseTo(value, 5);
        expect(exact[index]).toBeCloseTo(value, 5);
      }
    }
  } finally {
    fixture.destroy();
  }
});

it('8,193 separated patches have exact nearest distances and closed-form proximity without dense buffers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'boundary-proximity-many-patches');
  try {
    const patchCount = 8193;
    const width = 2 * patchCount - 1;
    const labels = Uint32Array.from({length: width}, (_, cell) =>
      cell % 2 === 0 ? cell / 2 : 0xffffffff
    );
    const result = addPatchProximityRecipe(fixture.graph, {
      mode: 'boundary',
      patches: fixture.input('patches', labels, 'uint32', width),
      patchClasses: fixture.input('classes', new Uint32Array(patchCount), 'uint32', patchCount),
      width,
      height: 1,
      cellSize: [1, 1],
      edgeDepth: 0,
      searchRadius: 1
    });
    const metrics = publish(fixture, 'metrics', result.outputs.metrics);
    const validity = publish(fixture, 'validity', result.outputs.validity);
    const candidateOverflow = publish(
      fixture,
      'candidate-overflow',
      result.status.candidateOverflow
    );
    const count = publish(fixture, 'count', result.status.count);
    fixture.run();
    expect(await fixture.readUint32(candidateOverflow, 1)).toEqual([0]);
    expect(await fixture.readUint32(count, 1)).toEqual([patchCount]);
    expect(await fixture.readUint32(validity, patchCount)).toEqual(new Array(patchCount).fill(1));
    const values = await fixture.readFloat32(metrics, 4 * patchCount);
    for (let patch = 0; patch < patchCount; patch++)
      expect(values.slice(4 * patch, 4 * patch + 4)).toEqual([
        1,
        1,
        1,
        patch === 0 || patch === patchCount - 1 ? 1 : 2
      ]);
  } finally {
    fixture.destroy();
  }
}, 120000);

it.each([
  'radius',
  'capacity'
] as const)('insufficient boundary %s reports candidateOverflow and never publishes an uncertified nearest distance', async reason => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'boundary-proximity-budget');
  try {
    const result = addPatchProximityRecipe(fixture.graph, {
      mode: 'boundary',
      patches: fixture.input(
        'patches',
        new Uint32Array([0, 0xffffffff, 1, 0xffffffff, 2]),
        'uint32',
        5
      ),
      patchClasses: fixture.input('classes', new Uint32Array([0, 0, 1]), 'uint32', 3),
      width: 5,
      height: 1,
      cellSize: [1, 1],
      edgeDepth: 0,
      searchRadius: reason === 'radius' ? 0 : 1,
      candidateCapacity: reason === 'capacity' ? 1 : 8
    });
    const metrics = publish(fixture, 'metrics', result.outputs.metrics);
    const flags = publish(fixture, 'validity', result.outputs.validity);
    const status = publish(fixture, 'status', result.status.candidateOverflow);
    fixture.run();
    expect(await fixture.readUint32(status, 1)).toEqual([1]);
    expect(await fixture.readUint32(flags, 3)).toEqual(reason === 'radius' ? [0, 0, 1] : [0, 0, 0]);
    const values = await fixture.readFloat32(metrics, 12);
    for (const owner of reason === 'radius' ? [0, 1] : [0, 1, 2]) {
      expect(values[4 * owner]).toBe(1);
      expect(values[4 * owner + 2]).toBeNaN();
      expect(values[4 * owner + 3]).toBeNaN();
    }
  } finally {
    fixture.destroy();
  }
});
