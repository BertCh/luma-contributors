// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it, vi} from 'vitest';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {POINT_CLOUD_BINDINGS} from '../../src/analysis/catalog-pointcloud-bindings';
import {POINT_CLOUD_OPERATION_DESCRIPTORS} from '../../src/descriptors/descriptors-pointcloud';
import {resolveKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';
import type {CatalogOutputTable} from '../../src/analysis/catalog-geojson';
import {
  makePointCloudCatalogGrid,
  preparePointCloudInputs
} from '../../src/analysis/catalog-pointcloud-input';
import {makePointCloudFixture, POINT_CLOUD_PLANE} from '../catalog-pointcloud-fixtures';

function getDescriptor(bindingId: string) {
  const descriptor = POINT_CLOUD_OPERATION_DESCRIPTORS.find(
    candidate => candidate.bindingId === bindingId
  );
  if (!descriptor) throw new Error(`Missing point-cloud descriptor ${bindingId}`);
  return descriptor;
}

function getBinding(bindingId: string) {
  const binding = POINT_CLOUD_BINDINGS.find(candidate => candidate.id === bindingId);
  if (!binding) throw new Error(`Missing point-cloud binding ${bindingId}`);
  return binding;
}

function makeProfile(collection = POINT_CLOUD_PLANE, datasetId = 'cloud') {
  return profileKeplerFeatureCollection(collection, {
    datasetId,
    label: datasetId,
    coordinateSystem: 'planar'
  });
}

function makeVoxelParts(stringKeys: boolean) {
  const makePart = (offset: number) => {
    const collection = makePointCloudFixture(
      Array.from({length: 4}, (_, row) => [row % 2, Math.floor(row / 2), offset])
    );
    collection.features.forEach((feature, row) => {
      feature.properties = {
        voxel_id: stringKeys ? `cell-${row}` : row,
        count: 1,
        representative: row
      };
    });
    return collection;
  };
  return {left: makePart(0), right: makePart(0.2)};
}

const VOXEL_FIELD_BINDINGS = {
  voxelProperty: 'voxel_id',
  countProperty: 'count',
  representativeProperty: 'representative'
};

function makeAllocationTrap() {
  const device = createNullWebGPUDevice();
  const createBuffer = vi.fn(() => {
    throw new Error('GPU allocation reached');
  });
  vi.spyOn(device, 'createBuffer').mockImplementation(createBuffer);
  return {device, createBuffer};
}

describe('point-cloud descriptors', () => {
  it.each([
    {
      bindingId: 'pointcloud-height',
      column: 'height_above_ground',
      values: [0, 1, 3, null],
      scale: 'sequential',
      domain: [0, 3]
    },
    {
      bindingId: 'pointcloud-height',
      column: 'height_above_ground',
      values: [-2, 0, 3, null],
      scale: 'sequential',
      domain: [-2, 3]
    },
    {
      bindingId: 'pointcloud-change',
      column: 'change',
      values: [0.2, 0.2, null],
      scale: 'sequential',
      domain: [0.2, 0.2]
    },
    {
      bindingId: 'pointcloud-change',
      column: 'change',
      values: [-0.4, -0.2, null],
      scale: 'sequential',
      domain: [-0.4, -0.2]
    },
    {
      bindingId: 'pointcloud-change',
      column: 'change',
      values: [-0.2, 0, 0.4, null],
      scale: 'diverging',
      domain: [-0.4, 0.4]
    }
  ])('$bindingId presents $values with a $scale scale', ({
    bindingId,
    column,
    values,
    scale,
    domain
  }) => {
    const output = getDescriptor(bindingId).outputs.find(
      candidate => candidate.tableId === 'points'
    )!;
    expect(output.presentation).toMatchObject({
      column,
      scale: bindingId === 'pointcloud-height' ? 'sequential' : 'diverging'
    });
    const table: CatalogOutputTable = {
      id: 'points',
      label: bindingId,
      rowCount: values.length,
      columns: {[column]: values},
      columnTypes: {[column]: 'float64'}
    };
    const plan = planKeplerResultPresentation(table, output)!;
    expect(plan.column).toBe(column);
    expect(plan.scale).toBe(scale);
    expect(plan.domain).toEqual(domain);
    expect(plan.missingCount).toBe(1);
    expect(plan.ranges.reduce((total, range) => total + range.count, 0)).toBe(values.length - 1);
    if (bindingId === 'pointcloud-change') {
      expect(output.presentation?.center).toBe(0);
      if (scale === 'diverging') {
        expect(plan.center).toBe(0);
        expect(plan.ranges[0].color[2]).toBeGreaterThan(plan.ranges[0].color[0]);
        const lastColor = plan.ranges[plan.ranges.length - 1].color;
        expect(lastColor[0]).toBeGreaterThan(lastColor[2]);
      } else {
        expect(plan.notes.join(' ')).toContain('lies on one side of 0');
      }
    }
  });

  it('accepts unique text voxel identifiers with explicit field mappings and reaches runtime allocation', async () => {
    const binding = getBinding('pointcloud-voxel-merge');
    const descriptor = getDescriptor(binding.id);
    const parts = makeVoxelParts(true);
    const profiles = [makeProfile(parts.left, 'left'), makeProfile(parts.right, 'right')];
    expect(profiles[0].columns.find(column => column.name === 'voxel_id')?.type).toBe('text');
    expect(
      descriptor.inputs[0].fields.find(field => field.role === 'voxel')?.measurement
    ).toContain('identifier');
    const result = evaluateKeplerOperationApplicability(binding, descriptor, profiles, {
      inputAssignments: {left: 'left', right: 'right'},
      fieldBindings: VOXEL_FIELD_BINDINGS,
      overrides: {disjointOwnedRows: true}
    });
    expect(result.status, JSON.stringify(result.missing)).toBe('ready');
    expect(result.missing).toEqual([]);
    const trap = makeAllocationTrap();
    await expect(binding.runGpu!({device: trap.device}, parts, result.parameters)).rejects.toThrow(
      'GPU allocation reached'
    );
    expect(trap.createBuffer).toHaveBeenCalledOnce();
  });

  it.each([
    'pointcloud-change',
    'pointcloud-voxel-merge'
  ])('%s requires certification before readiness and runtime allocation', async bindingId => {
    const binding = getBinding(bindingId);
    const descriptor = getDescriptor(bindingId);
    const comparison = makePointCloudFixture(
      POINT_CLOUD_PLANE.features.map(feature => {
        const coordinates = feature.geometry!.coordinates as number[];
        return [coordinates[0], coordinates[1], coordinates[2] + 0.2];
      })
    );
    const inputs =
      bindingId === 'pointcloud-change'
        ? {reference: POINT_CLOUD_PLANE, comparison}
        : makeVoxelParts(false);
    const profiles = Object.entries(inputs).map(([name, collection]) =>
      makeProfile(collection, name)
    );
    const options = {
      inputAssignments: Object.fromEntries(binding.inputs.map(input => [input.name, input.name])),
      fieldBindings: bindingId === 'pointcloud-voxel-merge' ? VOXEL_FIELD_BINDINGS : {}
    };
    const certification =
      bindingId === 'pointcloud-change' ? 'alreadyAligned' : 'disjointOwnedRows';
    for (const overrides of [{}, {[certification]: false}]) {
      const result = evaluateKeplerOperationApplicability(binding, descriptor, profiles, {
        ...options,
        overrides
      });
      expect(result.status, JSON.stringify(result.missing)).toBe('needs-confirmation');
      expect(result.missing).toContainEqual(
        expect.objectContaining({
          what: expect.stringContaining(certification),
          remedy: expect.stringContaining(certification)
        })
      );
      const trap = makeAllocationTrap();
      await expect(
        binding.runGpu!({device: trap.device}, inputs, result.parameters)
      ).rejects.toThrow(certification);
      expect(trap.createBuffer).not.toHaveBeenCalled();
      const subject = evaluateKeplerOperationApplicability(binding, descriptor, profiles, {
        ...options,
        overrides,
        subjectOnly: true
      });
      expect(subject.status).toBe('needs-confirmation');
    }
    const confirmed = evaluateKeplerOperationApplicability(binding, descriptor, profiles, {
      ...options,
      overrides: {[certification]: true}
    });
    expect(confirmed.status, JSON.stringify(confirmed.missing)).toBe('ready');
    expect(confirmed.missing).toEqual([]);
    const trap = makeAllocationTrap();
    await expect(
      binding.runGpu!({device: trap.device}, inputs, confirmed.parameters)
    ).rejects.toThrow('GPU allocation reached');
    expect(trap.createBuffer).toHaveBeenCalledOnce();
  });

  it.each([
    'ground',
    'smrf',
    'surfaces',
    'height'
  ])('%s checks mapped LAS classification bounds before runtime allocation', operation => {
    const binding = getBinding(`pointcloud-${operation}`);
    const descriptor = getDescriptor(binding.id);
    for (const value of [-1, 0, 255, 256]) {
      const collection = makePointCloudFixture(
        [
          [0, 0, 100],
          [1, 1, 101]
        ],
        [value, 2]
      );
      const result = evaluateKeplerOperationApplicability(
        binding,
        descriptor,
        [makeProfile(collection)],
        {
          fieldBindings: {classificationProperty: 'classification'}
        }
      );
      if (value < 0 || value > 255) {
        expect(result.status).toBe('blocked');
        expect(result.requirements).toContainEqual(
          expect.objectContaining({
            role: 'classification',
            id: 'numeric-range',
            state: 'failed',
            evidence: 'profile'
          })
        );
        expect(result.missing.some(missing => missing.remedy?.includes('[0,255]'))).toBe(true);
        expect(() => preparePointCloudInputs({points: collection}, result.parameters)).toThrow(
          /classification must be an integer in \[0,255\]/
        );
      } else {
        expect(result.status, JSON.stringify(result.missing)).toBe('ready');
        expect(() =>
          preparePointCloudInputs({points: collection}, result.parameters)
        ).not.toThrow();
      }
    }
  });

  it.each([
    'ground',
    'smrf'
  ])('%s preserves optional absent classification and its class-1 runtime default', operation => {
    const binding = getBinding(`pointcloud-${operation}`);
    const collection = makePointCloudFixture([
      [0, 0, 100],
      [1, 1, 101]
    ]);
    collection.features.forEach(feature => {
      feature.properties = {};
    });
    const result = evaluateKeplerOperationApplicability(binding, getDescriptor(binding.id), [
      makeProfile(collection)
    ]);
    expect(result.status, JSON.stringify(result.missing)).toBe('ready');
    expect(result.fieldBindings['classificationProperty']).toBeUndefined();
    expect(result.requirements.some(requirement => requirement.role === 'classification')).toBe(
      false
    );
    expect(
      Array.from(
        preparePointCloudInputs({points: collection}, result.parameters).clouds['points']
          .classifications
      )
    ).toEqual([1, 1]);
  });
  it('registers exactly the eleven bindings with complete roles, policies and output metadata', () => {
    expect(
      POINT_CLOUD_OPERATION_DESCRIPTORS.map(descriptor => descriptor.bindingId).sort()
    ).toEqual(POINT_CLOUD_BINDINGS.map(binding => binding.id).sort());
    expect(POINT_CLOUD_OPERATION_DESCRIPTORS).toHaveLength(11);
    for (const binding of POINT_CLOUD_BINDINGS) {
      const descriptor = getDescriptor(binding.id);
      expect(resolveKeplerOperationDescriptor(binding)).toEqual({
        source: 'hand-written',
        descriptor
      });
      expect(validateKeplerOperationDescriptor(descriptor, binding), binding.id).toEqual([]);
      expect(descriptor.questions.length).toBeGreaterThan(0);
      expect(descriptor.outputs.map(output => output.tableId)).toEqual(
        binding.outputs.map(output => output.tableId)
      );
      const fields = descriptor.inputs.flatMap(input => input.fields);
      expect(
        binding.parameters
          .filter(parameter => parameter.kind === 'string')
          .map(parameter => parameter.name)
          .sort()
      ).toEqual([...new Set(fields.map(field => field.parameter))].sort());
      for (const input of descriptor.inputs) {
        expect(input.kind).toBe('points');
        expect(descriptor.requirements).toContainEqual(
          expect.objectContaining({
            input: input.name,
            id: 'maximum-rows',
            limit: 2048,
            severity: 'block'
          })
        );
        expect(descriptor.requirements).toContainEqual(
          expect.objectContaining({
            input: input.name,
            id: 'maximum-vertices',
            limit: 2048,
            severity: 'block'
          })
        );
      }
      for (const parameter of descriptor.parameters) {
        expect(parameter.defaultPolicy.from).toMatch(/constant|data/);
        expect(parameter.apply).toBe(
          parameter.name === 'minimumSupport' && binding.id === 'pointcloud-normal-scale'
            ? 'live'
            : 'rebuild'
        );
      }
      for (const output of descriptor.outputs) {
        expect(output.interpretation?.summary).toBeTruthy();
        expect(output.interpretation?.cautions).toBeDefined();
        if (
          binding.outputs.find(candidate => candidate.tableId === output.tableId)?.geometry !==
          'none'
        ) {
          expect(output.presentation?.column).toBeTruthy();
          expect(output.presentation?.scale).toMatch(/categorical|sequential|diverging/);
        }
      }
      expect(descriptor.notes?.join(' ')).toMatch(/Elevation is the third coordinate/);
      expect(fields.some(field => field.role === 'elevation')).toBe(false);
    }
  });

  it('maps LAS classification as nominal integer data, required for terrain but optional for ground classification', () => {
    for (const operation of ['ground', 'smrf', 'surfaces', 'height']) {
      const field = getDescriptor(`pointcloud-${operation}`).inputs[0].fields[0];
      expect(field).toMatchObject({
        role: 'classification',
        parameter: 'classificationProperty',
        accepts: ['integer'],
        measurement: ['nominal'],
        required: ['surfaces', 'height'].includes(operation)
      });
    }
    const collection = makePointCloudFixture([
      [0, 0, 1],
      [1, 1, 2]
    ]);
    for (const feature of collection.features) feature.properties = {};
    const result = evaluateKeplerOperationApplicability(
      getBinding('pointcloud-height'),
      getDescriptor('pointcloud-height'),
      [makeProfile(collection)]
    );
    expect(result.fieldBindings['classificationProperty']).toBeUndefined();
    expect(result.status).not.toBe('ready');
    expect(result.notes.join(' ')).toMatch(/class 2/);
  });

  it('reproduces empty and oversized cloud rejections from profile facts before allocation', () => {
    const binding = getBinding('pointcloud-neighborhoods');
    const descriptor = getDescriptor(binding.id);
    for (const rowCount of [0, 2049]) {
      const profile = {...makeProfile(), rowCount, vertexCount: rowCount};
      const result = evaluateKeplerOperationApplicability(binding, descriptor, [profile]);
      expect(result.status).toBe('blocked');
      const failed = result.requirements.find(
        requirement =>
          requirement.state === 'failed' &&
          requirement.id === (rowCount === 0 ? 'minimum-rows' : 'maximum-rows')
      );
      expect(failed?.message).toMatch(rowCount === 0 ? /1/ : /2,?048/);
      expect(failed?.remedy).toBeTruthy();
      const collection = makePointCloudFixture(Array.from({length: rowCount}, () => [0, 0, 1]));
      expect(() =>
        preparePointCloudInputs({points: collection}, {coordinateSystem: 'planar'})
      ).toThrow(/catalog-pointcloud points.rows/);
    }
  });

  it('keeps unsafe guessed coordinates unresolved and exposes uncheckable XYZ and alignment limitations', () => {
    const binding = getBinding('pointcloud-change');
    const descriptor = getDescriptor(binding.id);
    const profile = {
      ...makeProfile(),
      coordinates: {model: 'planar' as const, source: 'guess' as const}
    };
    const result = evaluateKeplerOperationApplicability(binding, descriptor, [profile], {
      inputAssignments: {reference: 'cloud', comparison: 'cloud'}
    });
    expect(result.status).not.toBe('ready');
    expect(
      result.requirements
        .filter(requirement => requirement.id === 'declared-coordinates')
        .every(requirement => requirement.state !== 'passed')
    ).toBe(true);
    expect(result.parameters['alreadyAligned']).toBe(false);
    expect(result.notes.join(' ')).toMatch(/profile cannot verify alignment/);
    expect(getDescriptor('pointcloud-voxel-merge').notes?.join(' ')).toMatch(
      /profile cannot verify ownership/
    );
  });

  it('uses shared voxel/count/representative roles on both parts and rejects negative owned counts', () => {
    const binding = getBinding('pointcloud-voxel-merge');
    const descriptor = getDescriptor(binding.id);
    expect(descriptor.inputs.map(input => input.fields.map(field => field.role))).toEqual([
      ['voxel', 'count', 'representative'],
      ['voxel', 'count', 'representative']
    ]);
    const collection = makePointCloudFixture([
      [0, 0, 0],
      [1, 1, 1]
    ]);
    collection.features.forEach((feature, row) => {
      feature.properties = {voxel_id: `cell-${row}`, count: -1, representative: row};
    });
    const result = evaluateKeplerOperationApplicability(
      binding,
      descriptor,
      [makeProfile(collection, 'left'), makeProfile(collection, 'right')],
      {
        inputAssignments: {left: 'left', right: 'right'},
        fieldBindings: {
          voxelProperty: 'voxel_id',
          countProperty: 'count',
          representativeProperty: 'representative'
        },
        overrides: {disjointOwnedRows: true}
      }
    );
    expect(result.status).toBe('blocked');
    expect(
      result.requirements.filter(
        requirement => requirement.id === 'non-negative-values' && requirement.state === 'failed'
      )
    ).toHaveLength(2);
  });

  it('resolves a hand-computed XY distance proxy while keeping capacities and vertical assumptions explicit', () => {
    const profile = makeProfile(
      makePointCloudFixture([
        [0, 0, 0],
        [3, 4, 12],
        [6, 8, 24]
      ])
    );
    const binding = getBinding('pointcloud-features');
    const result = resolveKeplerOperationParameters(binding, getDescriptor(binding.id), {
      points: profile
    });
    expect(result.parameters['radius']).toBe(15); // XY nearest distance 5, not XYZ distance 13.
    expect(result.parameters['capacity']).toBe(32);
    expect(result.resolutions.find(resolution => resolution.parameter === 'radius')?.source).toBe(
      'data'
    );
    const withoutSpacing = {...profile, spacing: undefined};
    expect(
      resolveKeplerOperationParameters(binding, getDescriptor(binding.id), {points: withoutSpacing})
        .parameters['radius']
    ).toBe(1); // diagonal 10 / 10
    expect(
      getDescriptor('pointcloud-change').parameters.find(
        parameter => parameter.name === 'registrationError'
      )?.defaultPolicy
    ).toEqual({from: 'constant'});
  });

  it('keeps user-selected radius and capacity and reports fallback for a coincident cloud', () => {
    const binding = getBinding('pointcloud-features');
    const descriptor = getDescriptor(binding.id);
    const profile = makeProfile(
      makePointCloudFixture([
        [0, 0, 0],
        [0, 0, 20]
      ])
    );
    const fallback = resolveKeplerOperationParameters(binding, descriptor, {points: profile});
    expect(fallback.parameters['radius']).toBe(3);
    expect(fallback.resolutions.find(resolution => resolution.parameter === 'radius')?.source).toBe(
      'fallback'
    );
    const selected = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: profile},
      {overrides: {radius: 25, capacity: 64}}
    );
    expect(selected.parameters['radius']).toBe(25);
    expect(selected.parameters['capacity']).toBe(64);
  });

  it('derives metre cell sizes on geographic clouds, including a narrow antimeridian crossing', () => {
    for (const coordinates of [
      [
        [-87.65, 41.88, 100],
        [-87.649, 41.881, 101]
      ],
      [
        [179.999, 10, 100],
        [-179.999, 10.001, 101]
      ]
    ]) {
      const collection = makePointCloudFixture(coordinates);
      const profile = profileKeplerFeatureCollection(collection, {
        datasetId: 'geographic',
        label: 'Geographic cloud',
        coordinateSystem: 'EPSG:4326'
      });
      const binding = getBinding('pointcloud-surfaces');
      const resolved = resolveKeplerOperationParameters(binding, getDescriptor(binding.id), {
        points: profile
      });
      const cellSize = Number(resolved.parameters['cellSize']);
      expect(cellSize).toBeGreaterThan(1);
      expect(cellSize).toBeLessThan(10);
      const cloud = preparePointCloudInputs({points: collection}, {coordinateSystem: 'EPSG:4326'})
        .clouds['points'];
      const grid = makePointCloudCatalogGrid(cloud, cellSize);
      expect(grid.width * grid.height).toBeLessThanOrEqual(4096);
    }
  });

  for (const [width, height] of [
    [320, 240],
    [100000, 1],
    [0, 100000],
    [100000, 0]
  ]) {
    it(`derives cell size for ${width} by ${height} and fits the actual bounded raster`, () => {
      const collection = makePointCloudFixture([
        [0, 0, 100],
        [width, height, 101]
      ]);
      const profile = makeProfile(collection);
      const prepared = preparePointCloudInputs({points: collection}, {coordinateSystem: 'planar'});
      for (const operation of ['ground', 'smrf', 'surfaces', 'height']) {
        const binding = getBinding(`pointcloud-${operation}`);
        const result = resolveKeplerOperationParameters(binding, getDescriptor(binding.id), {
          points: profile
        });
        const cellSize = Number(result.parameters['cellSize']);
        if (width === 320) expect(cellSize).toBe(13); // hypot(320,240)/32 = 12.5, rounded to two significant digits.
        const grid = makePointCloudCatalogGrid(prepared.clouds['points'], cellSize);
        expect(grid.width * grid.height).toBeLessThanOrEqual(4096);
        expect(grid.width).toBeGreaterThanOrEqual(2);
        expect(grid.height).toBeGreaterThanOrEqual(2);
      }
    });
  }
});
