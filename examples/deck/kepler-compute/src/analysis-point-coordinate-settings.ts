// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  parseKeplerPointWeightCoordinates,
  type KeplerPointWeightCoordinates
} from '@luma-contributors/kepler';
import type {PointCoordinateDraft} from './analysis-point-coordinate-controls';

/** Geographic roles preserve the legacy host shape; planar roles declare meter coordinates. */
export type PointCoordinateSettings =
  | {
      coordinateSpace?: 'longitude-latitude';
      longitudeColumn: string;
      latitudeColumn: string;
    }
  | {
      coordinateSpace: 'planar';
      xColumn: string;
      yColumn: string;
      crs: string;
      bounds: readonly [number, number, number, number];
    };

/** Select only coordinate metadata from host settings before strict recipe validation. */
export function makePointCoordinateRecipe(
  settings: PointCoordinateSettings
): KeplerPointWeightCoordinates {
  return parseKeplerPointWeightCoordinates(
    settings.coordinateSpace === 'planar'
      ? {
          coordinateSpace: 'planar',
          xColumn: settings.xColumn,
          yColumn: settings.yColumn,
          crs: settings.crs,
          bounds: [...settings.bounds],
          metric: 'native',
          units: 'meters'
        }
      : {
          longitudeColumn: settings.longitudeColumn,
          latitudeColumn: settings.latitudeColumn,
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'meters'
        }
  );
}

/** Restore the geographic host shape without introducing new serialized default fields. */
export function getPointCoordinateSettings(
  settings: KeplerPointWeightCoordinates
): PointCoordinateSettings {
  return settings.coordinateSpace === 'planar'
    ? {
        coordinateSpace: 'planar',
        xColumn: settings.xColumn,
        yColumn: settings.yColumn,
        crs: settings.crs,
        bounds: [...settings.bounds]
      }
    : {longitudeColumn: settings.longitudeColumn, latitudeColumn: settings.latitudeColumn};
}

export function getPointCoordinateColumns(
  settings: PointCoordinateSettings
): readonly [string, string] {
  return settings.coordinateSpace === 'planar'
    ? [settings.xColumn, settings.yColumn]
    : [settings.longitudeColumn, settings.latitudeColumn];
}

/** The available role list must contain prepared Float32 fields, not arbitrary source columns. */
export function validatePointCoordinateColumns(
  settings: PointCoordinateSettings,
  columns: readonly string[]
): void {
  if (getPointCoordinateColumns(settings).some(column => !column || !columns.includes(column)))
    throw new Error('Choose prepared Float32 fields for both coordinate roles');
}

export function makePointCoordinateDraft(settings?: PointCoordinateSettings): PointCoordinateDraft {
  const projected = settings?.coordinateSpace === 'planar' ? settings : undefined;
  const geographic = settings && settings.coordinateSpace !== 'planar' ? settings : undefined;
  return {
    coordinateSpace: projected ? 'planar' : 'longitude-latitude',
    longitudeColumn: geographic?.longitudeColumn ?? '',
    latitudeColumn: geographic?.latitudeColumn ?? '',
    xColumn: projected?.xColumn ?? '',
    yColumn: projected?.yColumn ?? '',
    crs: projected?.crs ?? '',
    minX: projected ? String(projected.bounds[0]) : '',
    minY: projected ? String(projected.bounds[1]) : '',
    maxX: projected ? String(projected.bounds[2]) : '',
    maxY: projected ? String(projected.bounds[3]) : ''
  };
}

/** Blank inputs remain invalid instead of silently becoming zero-valued bounds. */
export function getPointCoordinateDraftSettings(
  draft: PointCoordinateDraft
): PointCoordinateSettings {
  if (
    draft.coordinateSpace === 'planar' &&
    [draft.minX, draft.minY, draft.maxX, draft.maxY].some(bound => !bound.trim())
  )
    throw new Error('Declare all four projected coordinate bounds in meters');
  const settings: PointCoordinateSettings =
    draft.coordinateSpace === 'planar'
      ? {
          coordinateSpace: 'planar',
          xColumn: draft.xColumn,
          yColumn: draft.yColumn,
          crs: draft.crs,
          bounds: [Number(draft.minX), Number(draft.minY), Number(draft.maxX), Number(draft.maxY)]
        }
      : {longitudeColumn: draft.longitudeColumn, latitudeColumn: draft.latitudeColumn};
  return getPointCoordinateSettings(makePointCoordinateRecipe(settings));
}
