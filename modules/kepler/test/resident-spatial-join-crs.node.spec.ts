// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import proj4 from 'proj4';
import {describe, expect, it} from 'vitest';
import {
  normalizeProjectionLabel,
  parseProjectionDescriptor,
  projectDescriptor,
  unprojectDescriptor
} from '../../../examples/deck/kepler-compute/src/resident-spatial-join-crs';

const UTM_33N = '+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs';
const UTM_55S = '+proj=utm +zone=55 +south +datum=WGS84 +units=m +no_defs';
const TMERC_ORIGIN =
  '+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 +x_0=400000 +y_0=-100000 +ellps=GRS80 +units=m +no_defs';

describe('spatial join CRS declarations', () => {
  it('normalizes the base systems and aliases', () => {
    expect(normalizeProjectionLabel('epsg:4326')).toBe('EPSG:4326');
    expect(normalizeProjectionLabel('CRS:84')).toBe('EPSG:4326');
    expect(normalizeProjectionLabel('3857')).toBe('EPSG:3857');
    expect(normalizeProjectionLabel('+proj=longlat +datum=WGS84 +no_defs')).toBe('EPSG:4326');
  });

  it('maps UTM EPSG codes and proj4 UTM parameters to the same canonical definition', () => {
    expect(normalizeProjectionLabel('EPSG:32633')).toBe(normalizeProjectionLabel(UTM_33N));
    expect(normalizeProjectionLabel('EPSG:32755')).toBe(normalizeProjectionLabel(UTM_55S));
    expect(normalizeProjectionLabel('EPSG:32633')).not.toBe(normalizeProjectionLabel('EPSG:32634'));
    const label = normalizeProjectionLabel('EPSG:25832')!;
    expect(parseProjectionDescriptor(label)?.label).toBe(label);
  });

  it('refuses what the GPU cannot reproject faithfully', () => {
    expect(normalizeProjectionLabel('+proj=lcc +lat_1=33 +lat_2=45 +ellps=WGS84')).toBeUndefined();
    expect(
      normalizeProjectionLabel('+proj=utm +zone=33 +ellps=intl +towgs84=-87,-98,-121')
    ).toBeUndefined();
    expect(normalizeProjectionLabel('+proj=utm +zone=33 +units=us-ft')).toBeUndefined();
    expect(normalizeProjectionLabel('EPSG:27700')).toBeUndefined();
    expect(normalizeProjectionLabel('nonsense')).toBeUndefined();
  });
});

describe('transverse Mercator reference against proj4', () => {
  const cases: [string, [number, number][]][] = [
    [
      UTM_33N,
      [
        [15, 52],
        [10.7, 59.9],
        [12, 0.5],
        [17.9, 70.1],
        [9.1, 40.2],
        [11.99, 80]
      ]
    ],
    [
      UTM_55S,
      [
        [147, -37.8],
        [150.5, -33.9],
        [144.2, -10.7]
      ]
    ],
    [
      TMERC_ORIGIN,
      [
        [-2, 49],
        [-1.5, 51.5],
        [-4.5, 53.9]
      ]
    ]
  ];
  for (const [definition, points] of cases) {
    it(`forward and inverse match proj4 within millimeters: ${definition.slice(0, 40)}`, () => {
      const descriptor = parseProjectionDescriptor(definition)!;
      expect(descriptor.kind).toBe('tmerc');
      const converter = proj4('EPSG:4326', definition);
      for (const [longitude, latitude] of points) {
        const [x, y] = projectDescriptor(descriptor, longitude, latitude);
        const [expectedX, expectedY] = converter.forward([longitude, latitude]);
        expect(Math.abs(x - expectedX), `x ${longitude},${latitude}`).toBeLessThan(0.005);
        expect(Math.abs(y - expectedY), `y ${longitude},${latitude}`).toBeLessThan(0.005);
        const [backLongitude, backLatitude] = unprojectDescriptor(descriptor, x, y);
        expect(Math.abs(backLongitude - longitude)).toBeLessThan(1e-8);
        expect(Math.abs(backLatitude - latitude)).toBeLessThan(1e-8);
      }
    });
  }
});
