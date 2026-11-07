// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {parse} from '@loaders.gl/core';
import {GLTFLoader} from '@loaders.gl/scene/gltf';
import {postProcessGLTF} from '@luma.gl/engine/gltf';
import type {GLTFPostprocessed} from '@luma.gl/engine/gltf';
import {CUSTOM_SCENEGRAPH_MODEL_ID, TRIP_LAYER_SCENEGRAPH_MODELS} from '@kepler.gl/constants';

import {toRadians} from '@math.gl/core';

export type TripMarkerModelConfiguration = {
  scenegraph?: string;
  scenegraphCustomModelUrl?: string;
  sizeScale?: number;
  adjustRoll?: number;
  adjustPitch?: number;
  adjustYaw?: number;
};

export type TripMarkerModelResolution =
  | {status: 'loading'}
  | {status: 'failed'; reason: string}
  | {
      status: 'ready';
      /**
       * Post-processed glTF with decoded images; Deck's ScenegraphLayer instantiates its meshes,
       * textures, PBR materials and animations.
       */
      scenegraph: GLTFPostprocessed;
      /** Native `model.scale * 2^sizeScale`, applied to mesh meters. */
      sizeScale: number;
      /** Column-major 3x3 from native adjust angles and model angles. */
      baseRotation: number[];
    };

type Matrix3 = number[];

/** Rotation `Rz(yaw) * Ry(pitch) * Rx(roll)` in degrees, as a row-major 3x3. */
function makeEulerRotation(roll: number, pitch: number, yaw: number): Matrix3 {
  const [cr, sr] = [Math.cos(toRadians(roll)), Math.sin(toRadians(roll))];
  const [cp, sp] = [Math.cos(toRadians(pitch)), Math.sin(toRadians(pitch))];
  const [cy, sy] = [Math.cos(toRadians(yaw)), Math.sin(toRadians(yaw))];
  return [
    cy * cp,
    cy * sp * sr - sy * cr,
    cy * sp * cr + sy * sr,
    sy * cp,
    sy * sp * sr + cy * cr,
    sy * sp * cr - cy * sr,
    -sp,
    cp * sr,
    cp * cr
  ];
}

function multiplyRowMajor3(a: Matrix3, b: Matrix3): Matrix3 {
  const result = new Array<number>(9).fill(0);
  for (let row = 0; row < 3; row++)
    for (let column = 0; column < 3; column++)
      for (let index = 0; index < 3; index++)
        result[row * 3 + column] += a[row * 3 + index] * b[index * 3 + column];
  return result;
}

/**
 * Native `_getTransformMatrixGetter` base rotation: `toQuaternion(adjustRoll, -adjustPitch,
 * -adjustYaw) * toQuaternion(...model.angles)`; the per-trip heading yaw multiplies on the left.
 * Returned column-major for the marker layer.
 */
export function makeTripMarkerBaseRotation(
  configuration: Pick<TripMarkerModelConfiguration, 'adjustRoll' | 'adjustPitch' | 'adjustYaw'>,
  modelAngles: readonly number[]
): number[] {
  const adjust = makeEulerRotation(
    configuration.adjustRoll ?? 0,
    -(configuration.adjustPitch ?? 0),
    -(configuration.adjustYaw ?? 0)
  );
  const rowMajor = multiplyRowMajor3(
    adjust,
    makeEulerRotation(modelAngles[0] ?? 0, modelAngles[1] ?? 0, modelAngles[2] ?? 0)
  );
  const columnMajor: number[] = [];
  for (let column = 0; column < 3; column++)
    for (let row = 0; row < 3; row++) columnMajor.push(rowMajor[row * 3 + column]);
  return columnMajor;
}

/** Resolves the native gallery or custom model of a Trips layer, or null when none is selected. */
export function findTripMarkerModel(
  configuration: TripMarkerModelConfiguration
): {url: string; scale: number; angles: readonly number[]} | null {
  const model = TRIP_LAYER_SCENEGRAPH_MODELS.find(
    candidate => candidate.id === configuration.scenegraph
  );
  if (!model) return null;
  const url =
    model.id === CUSTOM_SCENEGRAPH_MODEL_ID ? configuration.scenegraphCustomModelUrl : model.url;
  return url ? {url, scale: model.scale, angles: model.angles} : null;
}

/** Loads each model URL once through loaders.gl and luma glTF postprocessing. */
export class TripMarkerModelCache {
  private readonly entries = new Map<
    string,
    {scenegraph?: GLTFPostprocessed; reason?: string; promise: Promise<void>}
  >();

  constructor(
    private readonly onChange: () => void,
    private readonly loadBuffer: (url: string) => Promise<ArrayBuffer> = async url => {
      const response = await fetch(url);
      // Mirrors native fetch failure; the reason reaches the layer diagnostics.
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.arrayBuffer();
    }
  ) {}

  resolve(configuration: TripMarkerModelConfiguration): TripMarkerModelResolution {
    const model = findTripMarkerModel(configuration);
    if (!model) return {status: 'failed', reason: 'No scenegraph model is selected.'};
    let entry = this.entries.get(model.url);
    if (!entry) {
      const created: NonNullable<ReturnType<typeof this.entries.get>> = {
        promise: Promise.resolve()
      };
      created.promise = this.loadBuffer(model.url)
        .then(buffer => parse(buffer, GLTFLoader, {gltf: {loadImages: true}}))
        .then(gltf => {
          created.scenegraph = postProcessGLTF(gltf);
        })
        .catch((error: unknown) => {
          created.reason = `Scenegraph model failed to load: ${
            error instanceof Error ? error.message : String(error)
          }`;
        })
        .finally(this.onChange);
      entry = created;
      this.entries.set(model.url, entry);
    }
    if (entry.reason) return {status: 'failed', reason: entry.reason};
    if (!entry.scenegraph) return {status: 'loading'};
    const sizeScale = configuration.sizeScale ?? 1; // The native transform getter's default, not the renderLayer local.
    return {
      status: 'ready',
      scenegraph: entry.scenegraph,
      sizeScale: model.scale * 2 ** sizeScale,
      baseRotation: makeTripMarkerBaseRotation(configuration, model.angles)
    };
  }
}
