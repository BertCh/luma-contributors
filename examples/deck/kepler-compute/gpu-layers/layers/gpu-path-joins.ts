// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Deck PathLayer's join geometry, evaluated against GPU-resident path neighbors. */
export const GPU_PATH_JOINS_SHADER = /* wgsl */ `
struct PathJoin {
  offset: vec3<f32>,
  cornerOffset: vec2<f32>,
  miterLength: f32,
  pathPosition: vec2<f32>,
  pathLength: f32,
  jointType: f32,
};

// The native four-triangle template: bevel start, two body triangles, bevel end.
fn getPathCorner(vertexIndex: u32) -> vec2<f32> {
  let corners = array<vec2<f32>, 12>(
    vec2<f32>(0.0, 0.0), vec2<f32>(0.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
  );
  return corners[vertexIndex];
}

fn clipPathLine(position: vec4<f32>, reference: vec4<f32>) -> vec4<f32> {
  if (position.w < 0.001) {
    let ratio = (0.001 - reference.w) / (position.w - reference.w);
    return reference + (position - reference) * ratio;
  }
  return position;
}

fn getPathJoin(
  previous: vec3<f32>, current: vec3<f32>, next: vec3<f32>,
  width: vec2<f32>, coverageScale: f32, corner: vec2<f32>,
  startCap: bool, endCap: bool
) -> PathJoin {
  let isEnd = corner.x > 0.0;
  let side = corner.y;
  let isJoint = select(0.0, 1.0, side == 0.0);
  var deltaPrevious = current - previous;
  var deltaNext = next - current;
  let rotation = project_needs_rotation(current);
  if (pathLayerStyle.billboard == 0u && rotation.needsRotation) {
    // Deck's GLSL multiplies deltas from the left (transpose): common space to the tangent frame.
    let toTangent = transpose(rotation.transform);
    deltaPrevious = toTangent * deltaPrevious;
    deltaNext = toTangent * deltaNext;
  }
  let deltaA = deltaPrevious.xy / width;
  let deltaB = deltaNext.xy / width;
  let lengthA = length(deltaA);
  let lengthB = length(deltaB);
  let directionA = select(vec2<f32>(0.0), normalize(deltaA), lengthA > 0.0);
  let directionB = select(vec2<f32>(0.0), normalize(deltaB), lengthB > 0.0);
  let perpendicularA = vec2<f32>(-directionA.y, directionA.x);
  let perpendicularB = vec2<f32>(-directionB.y, directionB.x);
  let tangentSum = directionA + directionB;
  let tangent = select(perpendicularA, normalize(tangentSum), length(tangentSum) > 0.0);
  let miter = vec2<f32>(-tangent.y, tangent.x);
  let direction = select(directionB, directionA, isEnd);
  let perpendicular = select(perpendicularB, perpendicularA, isEnd);
  let pathLength = select(lengthB, lengthA, isEnd);
  let sineHalfAngle = abs(dot(miter, perpendicular));
  let cosineHalfAngle = abs(dot(directionA, miter));
  let turnDirection = select(1.0, -1.0, directionA.x * directionB.y >= directionA.y * directionB.x);
  let cornerPosition = side * turnDirection;
  var miterSize = 1.0 / max(sineHalfAngle, 0.001);
  miterSize = mix(min(miterSize, max(lengthA, lengthB) / max(cosineHalfAngle, 0.001)), miterSize, step(0.0, cornerPosition));
  var offset = mix(miter * miterSize, perpendicular, step(0.5, cornerPosition)) * (side + isJoint * turnDirection);
  let isStartCap = lengthA == 0.0 || (!isEnd && startCap);
  let isEndCap = lengthB == 0.0 || (isEnd && endCap);
  let isCap = isStartCap || isEndCap;
  var jointType = f32(pathLayerStyle.jointRounded);
  if (isCap) {
    offset = mix(perpendicular * side, direction * f32(pathLayerStyle.capRounded) * 4.0 * select(1.0, -1.0, isStartCap), isJoint);
    jointType = f32(pathLayerStyle.capRounded);
  }
  let coverageOffset = offset * coverageScale;
  let miterLength = select(dot(coverageOffset, miter * turnDirection), isJoint, isCap);
  let fromStart = coverageOffset + deltaA * corner.x;
  let pathPosition = vec2<f32>(dot(fromStart, perpendicular), dot(fromStart, direction));
  var extrusion = vec3<f32>(coverageOffset * width, 0.0);
  if (pathLayerStyle.billboard == 0u && rotation.needsRotation) {
    extrusion = rotation.transform * extrusion;
  }
  return PathJoin(extrusion, coverageOffset, miterLength, pathPosition, pathLength, jointType);
}
`;
