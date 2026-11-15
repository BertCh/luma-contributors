// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {LayerExtension, type Layer} from '@deck.gl/core';
import type {ShaderPlugin} from '@luma.gl/shaders';

// Deck 10 alpha.1's PathLayer WGSL computes geometry.pickingColor, but its generic picking
// plugin only contributes GLSL, and the path does not call the vertex FILTER_COLOR target.
// Carry that already-encoded feature identity through the public path target instead. This
// adapter is restricted to editor strokes; it adds no uniforms or alternative color decoder.
export const RESIDENT_EDITOR_PICKING_PLUGIN: ShaderPlugin = {
  name: 'residentEditorPicking',
  interStageVariables: {
    residentEditorPickingColor: {type: 'vec3<f32>', interpolation: 'flat'},
    residentEditorPickingDepth: {type: 'f32'}
  },
  injections: [
    {
      stage: 'vertex',
      target: 'DECKGL_FILTER_PATH_COLOR_VERTEX',
      wgsl: 'residentEditorPickingColor = geometry.pickingColor;'
    },
    {
      stage: 'vertex',
      target: 'DECKGL_FILTER_GL_POSITION',
      wgsl: 'residentEditorPickingDepth = (*position).z / (*position).w;'
    },
    {
      stage: 'fragment',
      target: 'DECKGL_FILTER_COLOR',
      order: 100,
      wgsl: `
  if (picking.isActive > 0.5) {
    if (!picking_isColorValid(residentEditorPickingColor)) { discard; }
    if (picking.isAttribute > 0.5) {
      *color = vec4<f32>(residentEditorPickingDepth, 0.0, 0.0, 1.0);
    } else {
      *color = vec4<f32>(residentEditorPickingColor, 1.0);
    }
  } else if (picking.isHighlightActive > 0.5) {
    let highlightedColor = picking_normalizeColor(picking.highlightedObjectColor);
    if (picking_isColorZero(abs(residentEditorPickingColor - highlightedColor))) {
      let highlightAlpha = picking.highlightColor.a;
      let blendedAlpha = highlightAlpha + (*color).a * (1.0 - highlightAlpha);
      if (blendedAlpha > 0.0) {
        *color = vec4<f32>(
          mix((*color).rgb, picking.highlightColor.rgb, vec3<f32>(highlightAlpha / blendedAlpha)),
          blendedAlpha
        );
      } else {
        *color = vec4<f32>((*color).rgb, 0.0);
      }
    }
  }
`
    }
  ]
};

/** Restore source-feature picking and hover highlighting on native editor WGSL strokes. */
export class ResidentEditorPickingExtension extends LayerExtension {
  static override extensionName = 'ResidentEditorPickingExtension';

  override getShaders(this: Layer): {plugins: ShaderPlugin[]} | null {
    return this.context.device.type === 'webgpu' &&
      this.id.startsWith('kepler_editor_layer-') &&
      this.state &&
      'pathTesselator' in this.state
      ? {plugins: [RESIDENT_EDITOR_PICKING_PLUGIN]}
      : null;
  }
}
