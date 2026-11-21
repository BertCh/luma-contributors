// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Signed gnomonic ring area from local differences; avoids cancelling O(1) cross products. @internal */
export function getSphericalRingOrientationWGSL(readPosition: string): string {
  return `
fn sphericalRingProjectedDelta(point:vec3f,anchor:vec3f,pole:vec3f)->vec3f {
  // Subtract the packed vectors first (exact for nearby f32 components), before projection.
  let difference=point-anchor;
  let correction=dot(pole,difference)/dot(pole,anchor);
  return (difference-anchor*correction)/dot(pole,point);
}
fn sphericalRingSignedArea(begin:u32,end:u32,pole:vec3f)->f32 {
  let anchor=${readPosition}(begin);var area=0.0;
  if(dot(pole,anchor)<=0.0){return 0.0;}
  for(var vertex=begin+1u;vertex+1u<end;vertex++) {
    let first=${readPosition}(vertex);let second=${readPosition}(vertex+1u);
    if(dot(pole,first)>0.0&&dot(pole,second)>0.0) {
      let firstDelta=sphericalRingProjectedDelta(first,anchor,pole);
      let secondDelta=sphericalRingProjectedDelta(second,anchor,pole);
      area+=dot(pole,cross(firstDelta,secondDelta));
    }
  }
  return area;
}`;
}
