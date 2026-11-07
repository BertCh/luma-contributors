// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ScenegraphLayer} from '@deck.gl/mesh-layers';
import {areaLighting} from '@luma.gl/shaders';
import {installScenegraphShadowPatch} from '@deck.gl-community/gpu-layers';

/**
 * Host-side workaround for a Deck 9.4 and luma 10 mismatch. Deck's `scenegraphPbrMaterial` reuses
 * luma's `pbrMaterial` WGSL, which now loops over area lights (`areaLighting_getLightCount`), but
 * declares only the `lighting` dependency, so every glTF scenegraph shader (3D tiles, trip
 * markers) fails to compile on WebGPU with "unresolved call target". This wraps
 * `ScenegraphLayer.getShaders` on WebGPU to add the `areaLighting` module (zero lights by default).
 */
let isInstalled = false;

export function installScenegraphAreaLightingPatch(): void {
  if (isInstalled) return;
  isInstalled = true;
  const originalGetShaders = ScenegraphLayer.prototype.getShaders;
  ScenegraphLayer.prototype.getShaders = function getShaders(this: ScenegraphLayer) {
    const shaders = originalGetShaders.call(this) as {modules?: {name: string}[]};
    const modules = shaders.modules;
    if (
      this.context.device.type === 'webgpu' &&
      modules &&
      !modules.some(module => module.name === 'areaLighting')
    ) {
      shaders.modules = [...modules, areaLighting];
    }
    return shaders;
  } as typeof ScenegraphLayer.prototype.getShaders;
}

installScenegraphAreaLightingPatch();
// Scenegraph layers (3D tiles, trip markers) also cast and receive WGSL shadow maps.
installScenegraphShadowPatch();
