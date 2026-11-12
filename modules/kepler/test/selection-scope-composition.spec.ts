// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {
  GPUKeplerSelectionScopes,
  type GPUKeplerBrushTargets
} from '../src/selection/gpu-selection-scopes';
import {
  planKeplerConsumerFilterIds,
  type KeplerConsumerSelectionScope,
  type KeplerPolygonMaskFilter,
  type KeplerScalarMaskFilter,
  type KeplerSelectionLayer
} from '../src/selection/selection-scope-plan';

const CHUNK_LENGTHS = [7, 0, 9];
const ROW_COUNT = CHUNK_LENGTHS.reduce((total, length) => total + length, 0);

type Rectangle = readonly [number, number, number, number];
type LayerFixture = {id: string; xs: number[]; ys: number[]; validity?: number[]};
type PolygonFixture = {id: string; layerIds: string[]; bounds: Rectangle; inverted?: boolean};
type Scenario = {
  name: string;
  live?: number[];
  scalarCount: number;
  polygons: PolygonFixture[];
  brush?: {values: number[]; targets: GPUKeplerBrushTargets};
};

test('composed scope masks match a CPU composition for every operation and input count', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required for selection scope verification');
  // More conjunction inputs than one GPUMask pass can bind, so composition chains groups.
  const groupCapacity = device.limits.maxStorageBuffersPerShaderStage - 1;
  const manyFilters = 3 * groupCapacity;
  const layers: LayerFixture[] = [
    {id: 'first', xs: coordinates(1), ys: coordinates(2)},
    {id: 'second', xs: coordinates(3), ys: coordinates(4), validity: values(5, 2)},
    {id: 'untargeted', xs: coordinates(6), ys: coordinates(7)}
  ];
  const scenarios: Scenario[] = [
    {
      name: 'many scalar filters, mixed polygon inversion, brush everywhere',
      live: values(11, 2),
      scalarCount: manyFilters,
      polygons: [
        {id: 'drawn-a', layerIds: ['first', 'second'], bounds: [1, 1, 8, 9]},
        {id: 'drawn-b', layerIds: ['first'], bounds: [0, 2, 6, 10]},
        {id: 'drawn-c', layerIds: ['first'], bounds: [3, 0, 5, 10], inverted: true},
        {id: 'drawn-d', layerIds: ['second'], bounds: [2, 2, 4, 4], inverted: true}
      ],
      brush: {values: values(12, 3), targets: {table: true, charts: true, legends: true}}
    },
    {
      name: 'only inverted polygons and no live mask',
      scalarCount: 3,
      polygons: [
        {id: 'drawn-a', layerIds: ['first'], bounds: [1, 1, 8, 9], inverted: true},
        {id: 'drawn-b', layerIds: ['first'], bounds: [0, 2, 6, 10], inverted: true},
        {id: 'drawn-c', layerIds: ['second'], bounds: [3, 0, 5, 10], inverted: true}
      ],
      brush: {values: values(13, 2), targets: {charts: true}}
    },
    {name: 'no predicates at all', scalarCount: 0, polygons: []},
    {
      name: 'live mask with one polygon',
      live: values(14, 3),
      scalarCount: groupCapacity,
      polygons: [{id: 'drawn-a', layerIds: ['second'], bounds: [1, 1, 8, 9]}]
    }
  ];
  const owned: GPUVector[] = [];
  const pointVectors = new Map<string, GPUVector<'float32x2'>>();
  for (const layer of layers) {
    const positions = pointVector(device, layer.xs, layer.ys);
    owned.push(positions);
    pointVectors.set(layer.id, positions);
  }
  try {
    for (const scenario of scenarios) {
      const scalarValues = Array.from({length: scenario.scalarCount}, (_, index) =>
        values(100 + index, index % 3 === 0 ? 4 : 2)
      );
      const scalarFilters: KeplerScalarMaskFilter[] = scalarValues.map((rowValues, index) => {
        const mask = maskVector(device, rowValues);
        owned.push(mask);
        return {
          filterId: `scalar-${index}`,
          mask,
          dynamicDomain: index % 2 === 0,
          nativeTopology: index % 5 === 4 ? false : undefined,
          nativeFieldName: `field-${index % 4}`,
          inverted: index % 3 === 1,
          enabled: index % 7 !== 6
        };
      });
      const selectionLayers: KeplerSelectionLayer[] = layers.map(layer => {
        const validity = layer.validity ? maskVector(device, layer.validity) : undefined;
        if (validity) owned.push(validity);
        return {
          id: layer.id,
          geometryRevision: 1,
          geometry: {kind: 'point', positions: pointVectors.get(layer.id)!, validity}
        };
      });
      const polygonFilters: KeplerPolygonMaskFilter[] = scenario.polygons.map(polygon => ({
        id: polygon.id,
        layerIds: polygon.layerIds,
        inverted: polygon.inverted,
        selection: {kind: 'rectangle', bounds: polygon.bounds}
      }));
      const liveMask = scenario.live ? maskVector(device, scenario.live) : undefined;
      if (liveMask) owned.push(liveMask);
      const brushMask = scenario.brush ? maskVector(device, scenario.brush.values) : undefined;
      if (brushMask) owned.push(brushMask);
      const scopes = new GPUKeplerSelectionScopes({
        device,
        sourceChunkLengths: CHUNK_LENGTHS,
        liveMask,
        layers: selectionLayers,
        scalarFilters,
        polygonFilters,
        coordinateSpace: 'planar'
      });
      try {
        if (brushMask) scopes.setBrushMask(brushMask, scenario.brush!.targets);
        const plotIds = scalarFilters.slice(0, 2).map(filter => filter.filterId);
        const plots = plotIds.map(id => scopes.getPlotMask(id));
        const consumerScopes: KeplerConsumerSelectionScope[] = [
          {id: 'all'},
          {id: 'excluding', excludedCpuFilterIds: ['scalar-0'], excludedGpuFieldNames: ['field-0']}
        ];
        const consumers = consumerScopes.map(scope => scopes.getConsumerMask(scope));
        const encoder = device.createCommandEncoder();
        scopes.encode(encoder);
        device.submit(encoder.finish());
        expect(await scopes.readStatus(), scenario.name).toMatchObject({complete: true});

        // CPU reference over the same plan.
        const plan = scopes.currentPlan;
        if (scenario.scalarCount === manyFilters)
          expect(plan.scalarFilterIds.length + 1).toBeGreaterThan(2 * groupCapacity);
        const truth = (rowValues: number[] | undefined) =>
          rowValues ? rowValues.map(value => value !== 0) : new Array(ROW_COUNT).fill(true);
        const live = truth(scenario.live);
        const scalar = new Map(
          scalarFilters.map((filter, index) => [
            filter.filterId,
            truth(scalarValues[index]).map(value => value !== Boolean(filter.inverted))
          ])
        );
        const brush = (target: keyof GPUKeplerBrushTargets) =>
          scenario.brush?.targets[target] ? truth(scenario.brush.values) : truth(undefined);
        const all = (masks: readonly boolean[][]) =>
          Array.from({length: ROW_COUNT}, (_, row) => masks.every(mask => mask[row]));
        const any = (masks: readonly boolean[][]) =>
          Array.from({length: ROW_COUNT}, (_, row) => masks.some(mask => mask[row]));
        const display = all([live, ...plan.scalarFilterIds.map(id => scalar.get(id)!)]);
        const nativeTopology = all([
          live,
          ...plan.nativeTopologyFilterIds.map(id => scalar.get(id)!)
        ]);
        const domain = all([
          live,
          ...plan.domainFilterIds.map(id => scalar.get(id)!),
          brush('legends')
        ]);
        const polygonMasks = new Map<string, boolean[]>();
        for (const layer of layers) {
          const predicates = plan.layerFilterIds[layer.id].map(id => {
            const polygon = scenario.polygons.find(candidate => candidate.id === id)!;
            return layer.xs.map((x, row) => {
              const [minimumX, minimumY, maximumX, maximumY] = polygon.bounds;
              const y = layer.ys[row];
              const inside =
                x > minimumX &&
                x < maximumX &&
                y > minimumY &&
                y < maximumY &&
                (layer.validity ? layer.validity[row] !== 0 : true);
              return inside !== Boolean(polygon.inverted);
            });
          });
          polygonMasks.set(layer.id, all(predicates));
        }
        const nativeSpatial = plan.targetedLayerIds.length
          ? any(plan.targetedLayerIds.map(id => polygonMasks.get(id)!))
          : truth(undefined);
        const expected: [string, GPUVector<'uint32'>, boolean[]][] = [
          ['display', scopes.displayMask, display],
          ['native topology', scopes.nativeTopologyMask, nativeTopology],
          ['domain', scopes.domainMask, domain],
          [
            'native filtered',
            scopes.nativeFilteredMask,
            all([display, nativeSpatial, brush('table')])
          ]
        ];
        for (const layer of layers) {
          const polygon = polygonMasks.get(layer.id)!;
          expected.push([
            `layer ${layer.id}`,
            scopes.getLayerMask(layer.id),
            all([display, polygon])
          ]);
          expected.push([
            `layer topology ${layer.id}`,
            scopes.getLayerTopologyMask(layer.id),
            all([nativeTopology, polygon])
          ]);
        }
        for (const [index, filterId] of plotIds.entries()) {
          expected.push([
            `plot ${filterId}`,
            plots[index],
            all([
              live,
              ...plan.scalarFilterIds.filter(id => id !== filterId).map(id => scalar.get(id)!),
              nativeSpatial,
              brush('charts')
            ])
          ]);
        }
        for (const [index, scope] of consumerScopes.entries()) {
          expected.push([
            `consumer ${scope.id}`,
            consumers[index],
            all([
              live,
              ...planKeplerConsumerFilterIds(scalarFilters, scope).map(id => scalar.get(id)!),
              nativeSpatial,
              brush('charts')
            ])
          ]);
        }
        for (const [name, mask, rows] of expected) {
          expect(await readMask(mask), `${scenario.name}: ${name}`).toEqual(
            rows.map(value => (value ? 1 : 0))
          );
        }
      } finally {
        const fence = device.createFence();
        try {
          await scopes.destroy(fence.signaled);
        } finally {
          fence.destroy();
        }
      }
    }
  } finally {
    for (const vector of owned) vector.destroy();
  }
});

test.each([
  'or',
  'difference',
  'and'
] as const)('grouped %s preserves rows decided by early and late operands', async operation => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required for selection scope verification');
  const groupCapacity = device.limits.maxStorageBuffersPerShaderStage - 1;
  // Later groups reserve one input binding for the accumulator. This exceeds two groups
  // even on devices exposing more storage bindings than the WebGPU default.
  const operandCount = 3 * groupCapacity;
  const lateOperand = 2 * groupCapacity;
  const rowCount = operation === 'and' ? 5 : 3;
  const chunkLengths = [rowCount];
  const owned: GPUVector[] = [];
  const layers: KeplerSelectionLayer[] = [];
  const scalarFilters: KeplerScalarMaskFilter[] = [];
  const polygonFilters: KeplerPolygonMaskFilter[] = [];
  const predicates: number[][] = [];
  const xCoordinates = Array.from({length: rowCount}, (_, row) => row + 0.5);
  const yCoordinates = xCoordinates.map(() => 0.5);
  const positions = pointVector(device, xCoordinates, yCoordinates, chunkLengths);
  owned.push(positions);
  if (operation === 'difference') {
    layers.push({
      id: 'difference-layer',
      geometryRevision: 1,
      geometry: {kind: 'point', positions}
    });
  }
  for (let operand = 0; operand < operandCount; operand++) {
    if (operation === 'and') {
      // Rows 0 and 4 survive. Rows 2 and 3 pass the first group and are rejected later.
      const rejectedRow =
        operand === 0 ? 1 : operand === groupCapacity ? 2 : operand === lateOperand ? 3 : -1;
      const rows = xCoordinates.map((_, row) => Number(row !== rejectedRow));
      predicates.push(rows);
      const mask = maskVector(device, rows, chunkLengths);
      owned.push(mask);
      scalarFilters.push({filterId: `scalar-${operand}`, mask, dynamicDomain: true});
    } else {
      const selectedRow = operand === 0 ? 0 : operand === lateOperand ? 1 : -1;
      const bounds: Rectangle =
        selectedRow < 0 ? [10, 10, 11, 11] : [selectedRow, 0, selectedRow + 1, 1];
      predicates.push(
        xCoordinates.map((x, row) =>
          Number(
            x > bounds[0] &&
              x < bounds[2] &&
              yCoordinates[row] > bounds[1] &&
              yCoordinates[row] < bounds[3]
          )
        )
      );
      const layerId = operation === 'or' ? `or-layer-${operand}` : 'difference-layer';
      if (operation === 'or') {
        layers.push({id: layerId, geometryRevision: 1, geometry: {kind: 'point', positions}});
      }
      polygonFilters.push({
        id: `polygon-${operand}`,
        layerIds: [layerId],
        selection: {kind: 'rectangle', bounds},
        inverted: operation === 'difference'
      });
    }
  }
  const expected = xCoordinates.map((_, row) =>
    Number(
      operation === 'or'
        ? predicates.some(predicate => predicate[row] !== 0)
        : operation === 'and'
          ? predicates.every(predicate => predicate[row] !== 0)
          : predicates.every(predicate => predicate[row] === 0)
    )
  );
  expect(expected).toEqual(
    operation === 'or' ? [1, 1, 0] : operation === 'and' ? [1, 0, 0, 0, 1] : [0, 0, 1]
  );
  const scopes = new GPUKeplerSelectionScopes({
    device,
    sourceChunkLengths: chunkLengths,
    maximumSpatialPredicates: operandCount,
    layers,
    scalarFilters,
    polygonFilters,
    coordinateSpace: 'planar'
  });
  try {
    const plan = scopes.currentPlan;
    const plannedOperandCount =
      operation === 'or'
        ? plan.targetedLayerIds.length
        : operation === 'and'
          ? plan.scalarFilterIds.length
          : 1 + plan.layerFilterIds['difference-layer'].length;
    expect(plannedOperandCount).toBeGreaterThan(2 * groupCapacity);
    const encoder = device.createCommandEncoder();
    scopes.encode(encoder);
    device.submit(encoder.finish());
    expect(await scopes.readStatus()).toMatchObject({complete: true});
    const output =
      operation === 'and'
        ? scopes.displayMask
        : operation === 'difference'
          ? scopes.getLayerMask('difference-layer')
          : scopes.nativeFilteredMask;
    expect(await readMask(output)).toEqual(expected);
  } finally {
    const fence = device.createFence();
    try {
      await scopes.destroy(fence.signaled);
    } finally {
      fence.destroy();
      for (const vector of owned) vector.destroy();
    }
  }
});

/** Deterministic row values in [0, modulus); values above one check nonzero-is-true inputs. */
function values(seed: number, modulus: number): number[] {
  let state = seed * 2654435761;
  return Array.from({length: ROW_COUNT}, () => {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    return (state >>> 16) % modulus;
  });
}

/** Coordinates in (0, 10) that never fall on an integer rectangle edge. */
function coordinates(seed: number): number[] {
  return values(seed, 20).map(value => value / 2 + 0.25);
}

function split<T>(rows: readonly T[], chunkLengths = CHUNK_LENGTHS): T[][] {
  const chunks: T[][] = [];
  let offset = 0;
  for (const length of chunkLengths) {
    chunks.push(rows.slice(offset, offset + length));
    offset += length;
  }
  return chunks;
}

function pointVector(
  device: Device,
  xs: number[],
  ys: number[],
  chunkLengths = CHUNK_LENGTHS
): GPUVector<'float32x2'> {
  return new GPUVector({
    name: 'points',
    type: 'data',
    format: 'float32x2',
    ownsData: true,
    data: split(
      xs.map((x, row) => [x, ys[row]]),
      chunkLengths
    ).map(
      chunk =>
        new GPUData({
          buffer: device.createBuffer({
            data: chunk.length ? new Float32Array(chunk.flat()) : undefined,
            byteLength: Math.max(8, chunk.length * 8),
            usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
          }),
          format: 'float32x2',
          length: chunk.length,
          ownsBuffer: true
        })
    )
  });
}

function maskVector(
  device: Device,
  rows: number[],
  chunkLengths = CHUNK_LENGTHS
): GPUVector<'uint32'> {
  return new GPUVector({
    name: 'mask',
    type: 'data',
    format: 'uint32',
    ownsData: true,
    data: split(rows, chunkLengths).map(
      chunk =>
        new GPUData({
          buffer: device.createBuffer({
            data: chunk.length ? new Uint32Array(chunk) : undefined,
            byteLength: Math.max(4, chunk.length * 4),
            usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
          }),
          format: 'uint32',
          length: chunk.length,
          ownsBuffer: true
        })
    )
  });
}

async function readMask(mask: GPUVector<'uint32'>): Promise<number[]> {
  const result: number[] = [];
  for (const chunk of mask.data) {
    if (!chunk.length) continue;
    const bytes = await chunk.buffer.readAsync(0, chunk.length * 4);
    result.push(...new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length));
  }
  return result;
}
