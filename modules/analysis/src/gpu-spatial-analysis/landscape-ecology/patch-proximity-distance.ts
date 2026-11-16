// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';

/** Raster-cell distances must survive squared overflow and Metal's flushed scale reciprocals. @internal */
export const PATCH_PROXIMITY_DISTANCE_WGSL = /* wgsl */ `
${SCALED_QUOTIENT_WGSL}
fn getPatchProximityDistance(displacement:vec2<f32>)->f32 {
  let scale=max(abs(displacement.x),abs(displacement.y));
  // The dense reference uses infinity for an absent patch in a raster row.
  if(scale==0.0||(bitcast<u32>(scale)&0x7f800000u)==0x7f800000u){return scale;}
  let normalized=vec2<f32>(getScaledQuotient(displacement.x,scale,1.0,0),getScaledQuotient(displacement.y,scale,1.0,0));
  return scale*length(normalized);
}`;
