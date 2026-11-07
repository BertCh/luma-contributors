// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PathLayer} from '@deck.gl/layers';

/**
 * Host-side workaround for Deck 9.4's `path-layer.wgsl`, which never writes the picking color nor
 * the highlight color: its vertex shader computes `geometry.pickingColor` but drops it, and the
 * `// DECKGL_FILTER_COLOR` anchor has no arguments so the picking module injection never applies.
 * Stroke pixels therefore land in the picking buffer as their draw color. This wraps
 * `PathLayer.getShaders` on WebGPU to carry the picking color in a varying and apply the same
 * picking and highlight logic as the solid polygon layer.
 */

const PICKING_VARYINGS_ANCHOR = '@location(7) clipCoordinates: vec2<f32>,';
const PICKING_VERTEX_ANCHOR = 'varyings.clipCoordinates = geometry.position.xy;';
const FRAGMENT_RETURN_ANCHORS = [
  'return deckgl_premultiplied_alpha(varyings.vColor);',
  'return deckgl_premultiplied_alpha(color);'
];
const PICKING_MARKER = 'varyings.pathPickingColor';

function getPickingFragment(colorExpression: string): string {
  return `if (picking.isActive > 0.5) {
    if (!picking_isColorValid(${PICKING_MARKER})) {
      discard;
    }
    return vec4<f32>(${PICKING_MARKER}, 1.0);
  }
  var pathFragmentColor = ${colorExpression};
  if (picking.isHighlightActive > 0.5) {
    let highlightedObjectColor = picking_normalizeColor(picking.highlightedObjectColor);
    if (picking_isColorZero(abs(${PICKING_MARKER} - highlightedObjectColor))) {
      let highlightAlpha = picking.highlightColor.a;
      let blendedAlpha = highlightAlpha + pathFragmentColor.a * (1.0 - highlightAlpha);
      if (blendedAlpha > 0.0) {
        pathFragmentColor = vec4<f32>(
          mix(pathFragmentColor.rgb, picking.highlightColor.rgb, highlightAlpha / blendedAlpha),
          blendedAlpha
        );
      } else {
        pathFragmentColor = vec4<f32>(pathFragmentColor.rgb, 0.0);
      }
    }
  }
  return deckgl_premultiplied_alpha(pathFragmentColor);`;
}

/** Add picking and highlight output to Deck's path-layer WGSL; returns the input if not applicable. */
export function patchPathLayerPickingSource(source: string): string {
  if (
    source.includes(PICKING_MARKER) ||
    !source.includes(PICKING_VARYINGS_ANCHOR) ||
    !source.includes(PICKING_VERTEX_ANCHOR)
  ) {
    return source;
  }
  const fragmentAnchor = FRAGMENT_RETURN_ANCHORS.find(anchor => source.includes(anchor));
  if (!fragmentAnchor) return source;
  const colorExpression = fragmentAnchor.includes('varyings.vColor') ? 'varyings.vColor' : 'color';
  return source
    .replace(
      PICKING_VARYINGS_ANCHOR,
      `${PICKING_VARYINGS_ANCHOR}\n  @location(10) pathPickingColor: vec3<f32>,`
    )
    .replace(
      PICKING_VERTEX_ANCHOR,
      `${PICKING_VERTEX_ANCHOR}\n  ${PICKING_MARKER} = geometry.pickingColor;`
    )
    .replace(fragmentAnchor, getPickingFragment(colorExpression));
}

let isInstalled = false;

export function installPathLayerPickingPatch(): void {
  if (isInstalled) return;
  isInstalled = true;
  const originalGetShaders = PathLayer.prototype.getShaders;
  PathLayer.prototype.getShaders = function getShaders(this: PathLayer) {
    const shaders = originalGetShaders.call(this) as {source?: string};
    if (this.context.device.type === 'webgpu' && typeof shaders.source === 'string') {
      shaders.source = patchPathLayerPickingSource(shaders.source);
    }
    return shaders;
  } as typeof PathLayer.prototype.getShaders;
}

installPathLayerPickingPatch();
