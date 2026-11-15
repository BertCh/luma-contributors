// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Model} from '@luma.gl/engine';
import type {ShaderModule} from '@luma.gl/shaders';

/**
 * WGSL helper shared by GPUVector layers. It compares the instance's picking color with Deck's
 * highlighted object color in the fragment stage, so hover highlighting needs no CPU geometry,
 * no readback and no extra draw. Requires the Deck `picking` shader module.
 */
export const GPU_HIGHLIGHT_WGSL = /* wgsl */ `
fn gpuHighlightBlend(color: vec4<f32>) -> vec4<f32> {
  let highlightColor = picking_normalizeColor4(picking.highlightColor);
  let blendedAlpha = highlightColor.a + color.a * (1.0 - highlightColor.a);
  if (blendedAlpha <= 0.0) { return color; }
  return vec4<f32>(mix(color.rgb, highlightColor.rgb, highlightColor.a / blendedAlpha), blendedAlpha);
}
fn gpuHighlight(color: vec4<f32>, pickingColor: vec3<f32>) -> vec4<f32> {
  if (picking.isHighlightActive < 0.5) { return color; }
  let highlightedColor = picking_normalizeColor(picking.highlightedObjectColor);
  if (distance(pickingColor, highlightedColor) >= 0.00001) { return color; }
  return gpuHighlightBlend(color);
}
`;

/**
 * Whole-feature highlight for layers whose primitives (parts, rings, vertices) share one source
 * feature. The primitive row range of the hovered feature is a render-only uniform; when it is
 * disabled the layer falls back to Deck's single picked-id highlight.
 */
export const gpuHighlightRange = {
  name: 'highlightRange',
  uniformTypes: {
    enabled: 'f32',
    firstRow: 'f32',
    lastRow: 'f32'
  },
  source: /* wgsl */ `
struct highlightRangeUniforms {
  enabled: f32,
  firstRow: f32,
  lastRow: f32,
};
@group(0) @binding(auto) var<uniform> highlightRange: highlightRangeUniforms;
`
} as const satisfies ShaderModule;

export const GPU_HIGHLIGHT_RANGE_WGSL = /* wgsl */ `
${GPU_HIGHLIGHT_WGSL}
fn gpuHighlightRows(color: vec4<f32>, pickingColor: vec3<f32>) -> vec4<f32> {
  if (highlightRange.enabled < 0.5) { return gpuHighlight(color, pickingColor); }
  // Picking colors store row + 1 in little-endian bytes; float32 is exact below 2^24 rows.
  let bytes = round(pickingColor * 255.0);
  let row = bytes.x + bytes.y * 256.0 + bytes.z * 65536.0 - 1.0;
  if (row < highlightRange.firstRow || row > highlightRange.lastRow) { return color; }
  return gpuHighlightBlend(color);
}
`;

export type GPUHighlightRangeProps = {
  /** Inclusive primitive row range of the hovered source feature; null or omitted disables it. */
  highlightedPrimitiveRange?: readonly [number, number] | null;
};

/** Updates every model's range uniform; call from draw so rebuilt models never keep stale values. */
export function setGPUHighlightRangeProps(
  models: readonly Pick<Model, 'shaderInputs' | 'userData'>[],
  props: GPUHighlightRangeProps
): void {
  const range = props.highlightedPrimitiveRange;
  for (const model of models) {
    // The module's default is disabled, so an idle layer never touches its uniforms.
    if (!range && !model.userData?.['highlightRangeActive']) continue;
    if (model.userData) model.userData['highlightRangeActive'] = Boolean(range);
    model.shaderInputs.setProps({
      highlightRange: {
        enabled: range ? 1 : 0,
        firstRow: range?.[0] ?? 0,
        lastRow: range?.[1] ?? 0
      }
    });
  }
}
