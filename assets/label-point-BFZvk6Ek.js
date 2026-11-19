import{s as e,v as t}from"./spatial-context-DwZEd4EZ.js";import{E as n,f as r,y as i}from"./gpu-contributor-utils-BIwLQON5.js";import{o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as o}from"./graph-chunk-dispatch-zBt8k2-o.js";import{A as s,l as c,s as l,u}from"./pointer-overlay-Qh8qflcA.js";import{t as d}from"./define-run-B-brkCe4.js";import{n as f}from"./pick-Bo3t3hjv.js";import{s as p,v as m,y as h}from"./polygon-set-CHja50mr.js";import{i as g,p as _}from"./pointer-support-C2rPRVFO.js";import{go as v}from"./entry-preview-D70TdSbK.js";var y=`GPULabelPoint`,b=Object.freeze({...e,measure:`distance`}),x=64,S=Math.floor(4294967295/x),C=Math.floor(Math.sqrt(4294967295)),w=class{id;props;spatialContext;featureCount;initialGridSize;refinementGridSize;refinementRounds;refinementCandidates;smallFeatureVertexLimit;constructor(e){this.id=e.id??`label-point`,this.props=e,this.spatialContext=t(this.id,e.spatialContext,b),this.initialGridSize=e.initialGridSize??16,this.refinementGridSize=e.refinementGridSize??8,this.refinementRounds=e.refinementRounds??6,this.refinementCandidates=e.refinementCandidates??4,this.smallFeatureVertexLimit=e.smallFeatureVertexLimit??32;let{id:a}=this;if(r(a,`initialGridSize`,this.initialGridSize,2,C),r(a,`refinementGridSize`,this.refinementGridSize,3,C),r(a,`refinementRounds`,this.refinementRounds,0),r(a,`refinementCandidates`,this.refinementCandidates,1,16),r(a,`smallFeatureVertexLimit`,this.smallFeatureVertexLimit,0),n(e.positions,[`float32x2`],`${a} positions`),e.positions.length<1)throw Error(`${a} needs at least one position`);for(let t of[`ringOffsets`,`polygonOffsets`,`featureOffsets`])if(n(e[t],[`uint32`],`${a} ${t}`),e[t].length<2)throw Error(`${a} ${t} must contain at least two rows`);this.featureCount=e.featureOffsets.length-1,r(a,`feature count`,this.featureCount,1,S);let{points:o,distances:s,degenerate:c}=e.output;if(n(o,[`float32x2`],`${a} output.points`),o.length!==this.featureCount)throw Error(`${a} output.points must hold ${this.featureCount} rows`);if(s&&(n(s,[`float32`],`${a} output.distances`),s.length!==this.featureCount))throw Error(`${a} output.distances must hold ${this.featureCount} rows`);if(c&&(n(c,[`uint32`],`${a} output.degenerate`),c.length!==this.featureCount))throw Error(`${a} output.degenerate must hold ${this.featureCount} rows`);i(a,[o,s,c],[e.positions,e.ringOffsets,e.featureOffsets,e.polygonOffsets])}getCommandNodes(e){let{props:t,id:n}=this,{points:r,distances:i,degenerate:s}=t.output;o(e,n,{positions:t.positions,ringOffsets:t.ringOffsets,featureOffsets:t.featureOffsets,polygonOffsets:t.polygonOffsets,"output.points":r,"output.distances":i,"output.degenerate":s});let c=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:t.ringOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:t.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:t.polygonOffsets,type:`u32`,access:`read`}];c.push({name:`pointsOut`,view:r,type:`f32`,access:`read_write`}),i&&c.push({name:`distancesOut`,view:i,type:`f32`,access:`read_write`}),s&&c.push({name:`degenerateOut`,view:s,type:`u32`,access:`read_write`});let l=`const ROW_COUNT: u32 = ${t.positions.length}u;
const RING_COUNT: u32 = ${t.ringOffsets.length-1}u;
const POLYGON_COUNT: u32 = ${t.polygonOffsets.length-1}u;
const FEATURE_COUNT: u32 = ${this.featureCount}u;
const INITIAL_GRID_SIZE: u32 = ${this.initialGridSize}u;
const REFINEMENT_GRID_SIZE: u32 = ${this.refinementGridSize}u;
const REFINEMENT_ROUNDS: u32 = ${this.refinementRounds}u;
const CANDIDATE_COUNT: u32 = ${this.refinementCandidates}u;
const SMALL_FEATURE_VERTEX_LIMIT: u32 = ${this.smallFeatureVertexLimit}u;
const WORKGROUP_SIZE: u32 = ${x}u;
const LARGE: f32 = 3.0e38;
${T}
// Feature -> polygons -> rings.
fn getRingStart(feature: u32) -> u32 {
  let polygon = min(featureOffsets[featureOffsetsOffset + feature], POLYGON_COUNT);
  return min(polygonOffsets[polygonOffsetsOffset + polygon], RING_COUNT);
}
fn getRingEnd(feature: u32) -> u32 {
  let polygon = min(featureOffsets[featureOffsetsOffset + feature + 1u], POLYGON_COUNT);
  return min(polygonOffsets[polygonOffsetsOffset + polygon], RING_COUNT);
}`,u=(e,t)=>`
  pointsOut[pointsOutOffset + 2u * ${e}] = result.x;
  pointsOut[pointsOutOffset + 2u * ${e} + 1u] = result.y;
  ${i?`distancesOut[distancesOutOffset + ${e}] = select(max(bestDistance, 0.0), nan, ${t});`:``}
  ${s?`degenerateOut[degenerateOutOffset + ${e}] = isDegenerate;`:``}`;return[a(e,{id:`${n}-search`,operation:y,variant:`small`,bindings:c,invocationCount:this.featureCount,declarations:l,body:`let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  let ringStart = getRingStart(index);
  let ringEnd = max(getRingEnd(index), ringStart);
  let rowStart = getRingRowStart(ringStart);
  let rowEnd = max(getRingRowStart(ringEnd), rowStart);
  if (rowEnd - rowStart > SMALL_FEATURE_VERTEX_LIMIT) {
    return;
  }
  var result = vec2<f32>(nan);
  var bestDistance = nan;
  var isDegenerate = 1u;
  if (rowEnd > rowStart) {
    var minimum = getPosition(rowStart);
    var maximum = minimum;
    for (var row = rowStart + 1u; row < rowEnd; row++) {
      let p = getPosition(row);
      minimum = min(minimum, p);
      maximum = max(maximum, p);
    }
    let origin = 0.5 * (minimum + maximum);
    let size = maximum - minimum;
    var halfSize = 0.5 * max(size.x, size.y);
    // Seeds: first vertex (on the boundary, the fallback), then scanline midpoints.
    var best = vec3<f32>(getPosition(rowStart) - origin, 0.0);
    var bestIsSet = false;
    for (var seedIndex = 0u; seedIndex < 3u; seedIndex++) {
      let fraction = select(select(0.0, -0.2, seedIndex == 1u), 0.2, seedIndex == 2u);
      let seed = getScanlineSeed(ringStart, ringEnd, origin, fraction * size.y);
      if (seed.z > 0.0) {
        let candidate = vec3<f32>(seed.xy, getSignedDistance(seed.xy, ringStart, ringEnd, origin));
        if (!bestIsSet || candidate.z > best.z) {
          best = candidate;
          bestIsSet = true;
        }
      }
    }
    if (!bestIsSet) {
      // Archipelago fallback: every part missed the three feature scanlines, so repeat them through
      // the bounds of the largest ring instead of settling for a boundary vertex.
      let part = getLargestRingBounds(ringStart, ringEnd, origin);
      for (var seedIndex = 0u; seedIndex < 3u; seedIndex++) {
        let fraction = select(select(0.0, -0.2, seedIndex == 1u), 0.2, seedIndex == 2u);
        let seed = getScanlineSeed(ringStart, ringEnd, origin, 0.5 * (part.y + part.w) + fraction * (part.w - part.y));
        if (seed.z > 0.0) {
          let candidate = vec3<f32>(seed.xy, getSignedDistance(seed.xy, ringStart, ringEnd, origin));
          if (!bestIsSet || candidate.z > best.z) {
            best = candidate;
            bestIsSet = true;
          }
        }
      }
    }
    if (!bestIsSet) {
      best.z = getSignedDistance(best.xy, ringStart, ringEnd, origin);
    }
    if (halfSize > 0.0) {
      var top = array<vec3<f32>, CANDIDATE_COUNT>();
      for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
        top[slot] = vec3<f32>(0.0, 0.0, -LARGE);
      }
      scoreGrid(vec2<f32>(0.0), halfSize, INITIAL_GRID_SIZE, ringStart, ringEnd, origin, &top);
      let cellHalfSize = halfSize / f32(INITIAL_GRID_SIZE);
      for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
        var local = top[slot];
        // Candidates are sorted by distance, so once the cell bound cannot beat the best, stop.
        if (local.z + 1.4142135 * cellHalfSize <= best.z) {
          break;
        }
        if (local.z > best.z) {
          best = local;
        }
        var windowHalfSize = 2.0 * cellHalfSize;
        for (var round = 0u; round < REFINEMENT_ROUNDS; round++) {
          local = searchWindow(local.xy, windowHalfSize, REFINEMENT_GRID_SIZE, ringStart, ringEnd, origin, local);
          windowHalfSize = 2.0 * windowHalfSize / f32(REFINEMENT_GRID_SIZE);
        }
        if (local.z > best.z) {
          best = local;
        }
      }
    }
    result = origin + best.xy;
    bestDistance = best.z;
    isDegenerate = select(0u, 1u, best.z <= 0.0);
  }${u(`index`,`rowEnd <= rowStart`)}`}),a(e,{id:`${n}-search-workgroup`,operation:y,variant:`workgroup`,bindings:c,invocationCount:this.featureCount*x,workgroupSize:x,guardIndex:!1,declarations:l,body:`let feature = workgroupIndex;
  let lane = localInvocationIndex;
  if (feature >= FEATURE_COUNT) {
    return;
  }
  let nan = bitcast<f32>(0x7fc00000u | (feature & 0u));
  let ringStart = getRingStart(feature);
  let ringEnd = max(getRingEnd(feature), ringStart);
  let rowStart = getRingRowStart(ringStart);
  let rowEnd = max(getRingRowStart(ringEnd), rowStart);
  if (lane == 0u) {
    sharedUint[0] = select(0u, 1u, rowEnd - rowStart > SMALL_FEATURE_VERTEX_LIMIT);
  }
  if (workgroupUniformLoad(&sharedUint[0]) == 0u) {
    return;
  }
  var localMinimum = vec2<f32>(LARGE);
  var localNegativeMaximum = vec2<f32>(LARGE);
  for (var row = rowStart + lane; row < rowEnd; row += WORKGROUP_SIZE) {
    let p = getPosition(row);
    localMinimum = min(localMinimum, p);
    localNegativeMaximum = min(localNegativeMaximum, -p);
  }
  let minimum = vec2<f32>(reduceMinFloat(lane, localMinimum.x), reduceMinFloat(lane, localMinimum.y));
  let maximum = -vec2<f32>(reduceMinFloat(lane, localNegativeMaximum.x), reduceMinFloat(lane, localNegativeMaximum.y));
  let origin = 0.5 * (minimum + maximum);
  let size = maximum - minimum;
  let halfSize = 0.5 * max(size.x, size.y);
  var best = vec3<f32>(getPosition(rowStart) - origin, 0.0);
  var bestIsSet = false;
  for (var seedIndex = 0u; seedIndex < 3u; seedIndex++) {
    let fraction = select(select(0.0, -0.2, seedIndex == 1u), 0.2, seedIndex == 2u);
    let seed = getScanlineSeedCooperative(lane, ringStart, ringEnd, origin, fraction * size.y);
    if (seed.z > 0.0) {
      let candidate = vec3<f32>(seed.xy, getSignedDistanceCooperative(lane, seed.xy, ringStart, ringEnd, origin));
      if (!bestIsSet || candidate.z > best.z) {
        best = candidate;
        bestIsSet = true;
      }
    }
  }
  if (!bestIsSet) {
    // Archipelago fallback (see the serial kernel): scanlines through the largest ring's bounds.
    let part = getLargestRingBoundsCooperative(lane, ringStart, ringEnd, origin);
    for (var seedIndex = 0u; seedIndex < 3u; seedIndex++) {
      let fraction = select(select(0.0, -0.2, seedIndex == 1u), 0.2, seedIndex == 2u);
      let seed = getScanlineSeedCooperative(lane, ringStart, ringEnd, origin, 0.5 * (part.y + part.w) + fraction * (part.w - part.y));
      if (seed.z > 0.0) {
        let candidate = vec3<f32>(seed.xy, getSignedDistanceCooperative(lane, seed.xy, ringStart, ringEnd, origin));
        if (!bestIsSet || candidate.z > best.z) {
          best = candidate;
          bestIsSet = true;
        }
      }
    }
  }
  if (!bestIsSet) {
    best.z = getSignedDistanceCooperative(lane, best.xy, ringStart, ringEnd, origin);
  }
  if (halfSize > 0.0) {
    var topDistance = array<f32, CANDIDATE_COUNT>();
    var topCell = array<u32, CANDIDATE_COUNT>();
    for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
      topDistance[slot] = -LARGE;
      topCell[slot] = NO_CELL;
    }
    // Each lane scores the first-grid cells lane, lane + 64, ... in order and keeps its best few.
    let initialCellSize = 2.0 * halfSize / f32(INITIAL_GRID_SIZE);
    for (var cellIndex = lane; cellIndex < INITIAL_GRID_SIZE * INITIAL_GRID_SIZE; cellIndex += WORKGROUP_SIZE) {
      let p = getGridPoint(vec2<f32>(0.0), halfSize, initialCellSize, INITIAL_GRID_SIZE, cellIndex);
      var entryDistance = getSignedDistance(p, ringStart, ringEnd, origin);
      var entryCell = cellIndex;
      for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
        if (entryDistance > topDistance[slot]) {
          let displacedDistance = topDistance[slot];
          let displacedCell = topCell[slot];
          topDistance[slot] = entryDistance;
          topCell[slot] = entryCell;
          entryDistance = displacedDistance;
          entryCell = displacedCell;
        }
      }
    }
    let cellHalfSize = halfSize / f32(INITIAL_GRID_SIZE);
    for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
      // Next best first-grid cell over all lanes: highest distance, earliest cell on ties.
      let winner = reduceArgMax(lane, topDistance[0], topCell[0]);
      if (winner.cell != NO_CELL && lane == winner.cell % WORKGROUP_SIZE) {
        for (var shift = 0u; shift + 1u < CANDIDATE_COUNT; shift++) {
          topDistance[shift] = topDistance[shift + 1u];
          topCell[shift] = topCell[shift + 1u];
        }
        topDistance[CANDIDATE_COUNT - 1u] = -LARGE;
        topCell[CANDIDATE_COUNT - 1u] = NO_CELL;
      }
      var local = vec3<f32>(0.0, 0.0, -LARGE);
      if (winner.cell != NO_CELL) {
        local = vec3<f32>(getGridPoint(vec2<f32>(0.0), halfSize, initialCellSize, INITIAL_GRID_SIZE, winner.cell), winner.distance);
      }
      if (local.z + 1.4142135 * cellHalfSize <= best.z) {
        break;
      }
      if (local.z > best.z) {
        best = local;
      }
      var windowHalfSize = 2.0 * cellHalfSize;
      for (var round = 0u; round < REFINEMENT_ROUNDS; round++) {
        local = searchWindowCooperative(lane, local.xy, windowHalfSize, REFINEMENT_GRID_SIZE, ringStart, ringEnd, origin, local);
        windowHalfSize = 2.0 * windowHalfSize / f32(REFINEMENT_GRID_SIZE);
      }
      if (local.z > best.z) {
        best = local;
      }
    }
  }
  if (lane == 0u) {
    let result = origin + best.xy;
    let bestDistance = best.z;
    let isDegenerate = select(0u, 1u, best.z <= 0.0);${u(`feature`,`false`)}
  }`})]}},T=`
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getRingRowStart(ring: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + ring], ROW_COUNT);
}

fn getRingRowEnd(ring: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + ring + 1u], ROW_COUNT);
}

fn getSegmentDistanceSquared(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  let edge = b - a;
  let lengthSquared = dot(edge, edge);
  var t = 0.0;
  if (lengthSquared > 0.0) {
    t = clamp(dot(p - a, edge) / lengthSquared, 0.0, 1.0);
  }
  let offset = p - (a + t * edge);
  return dot(offset, offset);
}

// Signed distance to the boundary of all rings: positive inside under the even-odd rule.
fn getSignedDistance(p: vec2<f32>, ringStart: u32, ringEnd: u32, origin: vec2<f32>) -> f32 {
  var minimumSquared = LARGE;
  var inside = false;
  for (var ring = ringStart; ring < ringEnd; ring++) {
    let rowStart = getRingRowStart(ring);
    let rowEnd = getRingRowEnd(ring);
    if (rowEnd > rowStart) {
      var previous = getPosition(rowEnd - 1u) - origin;
      for (var row = rowStart; row < rowEnd; row++) {
        let current = getPosition(row) - origin;
        minimumSquared = min(minimumSquared, getSegmentDistanceSquared(p, previous, current));
        // f32 sampling test: it only places the candidate label point; topology uses utils/exact-geometry-wgsl.ts.
        if ((current.y > p.y) != (previous.y > p.y)) {
          let crossingX = previous.x + (p.y - previous.y) * (current.x - previous.x) / (current.y - previous.y);
          if (p.x < crossingX) {
            inside = !inside;
          }
        }
        previous = current;
      }
    }
  }
  let distance = sqrt(minimumSquared);
  return select(-distance, distance, inside);
}

// Bounds (minimum.xy, maximum.xy) relative to origin of ring 'ring'.
fn getRingBounds(ring: u32, origin: vec2<f32>) -> vec4<f32> {
  let rowStart = getRingRowStart(ring);
  let rowEnd = getRingRowEnd(ring);
  var minimum = vec2<f32>(LARGE);
  var maximum = vec2<f32>(-LARGE);
  for (var row = rowStart; row < rowEnd; row++) {
    let p = getPosition(row) - origin;
    minimum = min(minimum, p);
    maximum = max(maximum, p);
  }
  return vec4<f32>(minimum, maximum);
}

// Bounds of the ring with the largest extent (first on ties), used only when the feature-wide
// scanlines found no interior seed.
fn getLargestRingBounds(ringStart: u32, ringEnd: u32, origin: vec2<f32>) -> vec4<f32> {
  var largest = vec4<f32>(0.0);
  var largestExtent = -LARGE;
  for (var ring = ringStart; ring < ringEnd; ring++) {
    let bounds = getRingBounds(ring, origin);
    let extent = max(bounds.z - bounds.x, bounds.w - bounds.y);
    if (extent > largestExtent) {
      largestExtent = extent;
      largest = bounds;
    }
  }
  return largest;
}

// Midpoint of the leftmost interior interval on the horizontal line y = lineY (relative to the
// bounds center). f32 sampling code like getSignedDistance (see utils/exact-geometry-wgsl.ts for topology). Returns (x, y, 1) or z = 0 when there is none.
fn getScanlineSeed(ringStart: u32, ringEnd: u32, origin: vec2<f32>, lineY: f32) -> vec3<f32> {
  var firstX = LARGE;
  var firstCount = 0u;
  for (var scanPass = 0u; scanPass < 2u; scanPass++) {
    var secondX = LARGE;
    for (var ring = ringStart; ring < ringEnd; ring++) {
      let rowStart = getRingRowStart(ring);
      let rowEnd = getRingRowEnd(ring);
      if (rowEnd > rowStart) {
        var previous = getPosition(rowEnd - 1u) - origin;
        for (var row = rowStart; row < rowEnd; row++) {
          let current = getPosition(row) - origin;
          if ((current.y > lineY) != (previous.y > lineY)) {
            let crossingX = previous.x + (lineY - previous.y) * (current.x - previous.x) / (current.y - previous.y);
            if (scanPass == 0u) {
              if (crossingX < firstX) {
                firstX = crossingX;
                firstCount = 1u;
              } else if (crossingX == firstX) {
                firstCount += 1u;
              }
            } else if (crossingX > firstX) {
              secondX = min(secondX, crossingX);
            }
          }
          previous = current;
        }
      }
    }
    if (scanPass == 1u) {
      if (firstCount % 2u == 1u && secondX < LARGE) {
        return vec3<f32>(0.5 * (firstX + secondX), lineY, 1.0);
      }
    }
  }
  return vec3<f32>(0.0);
}

// Scores an n x n grid like searchWindow but keeps the best CANDIDATE_COUNT cells, sorted by
// signed distance descending (ties keep the earlier cell).
fn scoreGrid(
  center: vec2<f32>,
  halfSize: f32,
  gridSize: u32,
  ringStart: u32,
  ringEnd: u32,
  origin: vec2<f32>,
  top: ptr<function, array<vec3<f32>, CANDIDATE_COUNT>>
) {
  let cell = 2.0 * halfSize / f32(gridSize);
  for (var j = 0u; j < gridSize; j++) {
    for (var i = 0u; i < gridSize; i++) {
      let p = center - vec2<f32>(halfSize) + cell * (vec2<f32>(f32(i), f32(j)) + vec2<f32>(0.5));
      var entry = vec3<f32>(p, getSignedDistance(p, ringStart, ringEnd, origin));
      for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
        if (entry.z > (*top)[slot].z) {
          let displaced = (*top)[slot];
          (*top)[slot] = entry;
          entry = displaced;
        }
      }
    }
  }
}

// Scores an n x n grid of cell centers over [center - halfSize, center + halfSize]^2 and returns
// the better of the grid winner and \`best\` (ties keep the earlier candidate).
fn searchWindow(
  center: vec2<f32>,
  halfSize: f32,
  gridSize: u32,
  ringStart: u32,
  ringEnd: u32,
  origin: vec2<f32>,
  bestIn: vec3<f32>
) -> vec3<f32> {
  var best = bestIn;
  let cell = 2.0 * halfSize / f32(gridSize);
  for (var j = 0u; j < gridSize; j++) {
    for (var i = 0u; i < gridSize; i++) {
      let p = center - vec2<f32>(halfSize) + cell * (vec2<f32>(f32(i), f32(j)) + vec2<f32>(0.5));
      let distance = getSignedDistance(p, ringStart, ringEnd, origin);
      if (distance > best.z) {
        best = vec3<f32>(p, distance);
      }
    }
  }
  return best;
}

// Workgroup-cooperative variants used by the one-workgroup-per-feature kernel. Reductions end in
// workgroupUniformLoad so their results are workgroup-uniform, which keeps later barriers legal.
const NO_CELL: u32 = 0xffffffffu;
var<workgroup> sharedUint: array<u32, 1>;
var<workgroup> reduceFloat: array<f32, WORKGROUP_SIZE>;
var<workgroup> reduceUint: array<u32, WORKGROUP_SIZE>;

struct ArgMax {
  distance: f32,
  cell: u32
}

fn reduceMinFloat(lane: u32, value: f32) -> f32 {
  reduceFloat[lane] = value;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      reduceFloat[lane] = min(reduceFloat[lane], reduceFloat[lane + stride]);
    }
    workgroupBarrier();
  }
  return workgroupUniformLoad(&reduceFloat[0]);
}

fn reduceSumUint(lane: u32, value: u32) -> u32 {
  reduceUint[lane] = value;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      reduceUint[lane] = reduceUint[lane] + reduceUint[lane + stride];
    }
    workgroupBarrier();
  }
  return workgroupUniformLoad(&reduceUint[0]);
}

// Highest distance over the workgroup, the lowest cell index among equal distances.
fn reduceArgMax(lane: u32, distance: f32, cell: u32) -> ArgMax {
  reduceFloat[lane] = distance;
  reduceUint[lane] = cell;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      let otherDistance = reduceFloat[lane + stride];
      let otherCell = reduceUint[lane + stride];
      if (otherDistance > reduceFloat[lane] || (otherDistance == reduceFloat[lane] && otherCell < reduceUint[lane])) {
        reduceFloat[lane] = otherDistance;
        reduceUint[lane] = otherCell;
      }
    }
    workgroupBarrier();
  }
  let winnerDistance = workgroupUniformLoad(&reduceFloat[0]);
  let winnerCell = workgroupUniformLoad(&reduceUint[0]);
  return ArgMax(winnerDistance, winnerCell);
}

fn getGridPoint(center: vec2<f32>, halfSize: f32, cell: f32, gridSize: u32, cellIndex: u32) -> vec2<f32> {
  let i = cellIndex % gridSize;
  let j = cellIndex / gridSize;
  return center - vec2<f32>(halfSize) + cell * (vec2<f32>(f32(i), f32(j)) + vec2<f32>(0.5));
}

// Same value as getSignedDistance: lanes stride over the segments of every ring.
fn getSignedDistanceCooperative(lane: u32, p: vec2<f32>, ringStart: u32, ringEnd: u32, origin: vec2<f32>) -> f32 {
  var minimumSquared = LARGE;
  var crossings = 0u;
  for (var ring = ringStart; ring < ringEnd; ring++) {
    let ringRowStart = getRingRowStart(ring);
    let ringRowEnd = getRingRowEnd(ring);
    for (var row = ringRowStart + lane; row < ringRowEnd; row += WORKGROUP_SIZE) {
      let current = getPosition(row) - origin;
      let previous = getPosition(select(row - 1u, ringRowEnd - 1u, row == ringRowStart)) - origin;
      minimumSquared = min(minimumSquared, getSegmentDistanceSquared(p, previous, current));
      // f32 sampling test: it only places the candidate label point; topology uses utils/exact-geometry-wgsl.ts.
      if ((current.y > p.y) != (previous.y > p.y)) {
        let crossingX = previous.x + (p.y - previous.y) * (current.x - previous.x) / (current.y - previous.y);
        if (p.x < crossingX) {
          crossings += 1u;
        }
      }
    }
  }
  let distance = sqrt(reduceMinFloat(lane, minimumSquared));
  return select(-distance, distance, (reduceSumUint(lane, crossings) & 1u) == 1u);
}

// Same ring as getLargestRingBounds: lanes stride over the rings, the largest extent wins (lowest
// ring on ties), and every lane then reads that ring's bounds. Workgroup-uniform result.
fn getLargestRingBoundsCooperative(lane: u32, ringStart: u32, ringEnd: u32, origin: vec2<f32>) -> vec4<f32> {
  var laneExtent = -LARGE;
  var laneRing = NO_CELL;
  for (var ring = ringStart + lane; ring < ringEnd; ring += WORKGROUP_SIZE) {
    let bounds = getRingBounds(ring, origin);
    let extent = max(bounds.z - bounds.x, bounds.w - bounds.y);
    if (extent > laneExtent) {
      laneExtent = extent;
      laneRing = ring;
    }
  }
  let winner = reduceArgMax(lane, laneExtent, laneRing);
  if (winner.cell == NO_CELL) { return vec4<f32>(0.0); }
  return getRingBounds(winner.cell, origin);
}

// Same value as getScanlineSeed. Pass 0 finds the leftmost crossing, pass 1 counts the crossings
// sharing it, pass 2 finds the next crossing to its right.
fn getScanlineSeedCooperative(lane: u32, ringStart: u32, ringEnd: u32, origin: vec2<f32>, lineY: f32) -> vec3<f32> {
  var firstX = LARGE;
  var firstCount = 0u;
  for (var scanPass = 0u; scanPass < 3u; scanPass++) {
    var localX = LARGE;
    var localCount = 0u;
    for (var ring = ringStart; ring < ringEnd; ring++) {
      let ringRowStart = getRingRowStart(ring);
      let ringRowEnd = getRingRowEnd(ring);
      for (var row = ringRowStart + lane; row < ringRowEnd; row += WORKGROUP_SIZE) {
        let current = getPosition(row) - origin;
        let previous = getPosition(select(row - 1u, ringRowEnd - 1u, row == ringRowStart)) - origin;
        if ((current.y > lineY) != (previous.y > lineY)) {
          let crossingX = previous.x + (lineY - previous.y) * (current.x - previous.x) / (current.y - previous.y);
          if (scanPass == 0u) {
            localX = min(localX, crossingX);
          } else if (scanPass == 1u) {
            if (crossingX == firstX) {
              localCount += 1u;
            }
          } else if (crossingX > firstX) {
            localX = min(localX, crossingX);
          }
        }
      }
    }
    if (scanPass == 0u) {
      firstX = reduceMinFloat(lane, localX);
    } else if (scanPass == 1u) {
      firstCount = reduceSumUint(lane, localCount);
    } else {
      let secondX = reduceMinFloat(lane, localX);
      if (firstCount % 2u == 1u && secondX < LARGE) {
        return vec3<f32>(0.5 * (firstX + secondX), lineY, 1.0);
      }
    }
  }
  return vec3<f32>(0.0);
}

// Same result as searchWindow: every lane scores cells lane, lane + 64, ... and the best cell
// (earliest on ties) replaces bestIn only by a strictly larger distance.
fn searchWindowCooperative(
  lane: u32,
  center: vec2<f32>,
  halfSize: f32,
  gridSize: u32,
  ringStart: u32,
  ringEnd: u32,
  origin: vec2<f32>,
  bestIn: vec3<f32>
) -> vec3<f32> {
  let cell = 2.0 * halfSize / f32(gridSize);
  var laneDistance = -LARGE;
  var laneCell = NO_CELL;
  for (var cellIndex = lane; cellIndex < gridSize * gridSize; cellIndex += WORKGROUP_SIZE) {
    let distance = getSignedDistance(getGridPoint(center, halfSize, cell, gridSize, cellIndex), ringStart, ringEnd, origin);
    if (distance > laneDistance) {
      laneDistance = distance;
      laneCell = cellIndex;
    }
  }
  let winner = reduceArgMax(lane, laneDistance, laneCell);
  if (winner.cell != NO_CELL && winner.distance > bestIn.z) {
    return vec3<f32>(getGridPoint(center, halfSize, cell, gridSize, winner.cell), winner.distance);
  }
  return bestIn;
}
`,E=l(v,`refinementRounds`),D=d({parameters:v,caption:`Label anchors at the widest part of each polygon.`,buildCache:5,createInputs:({seed:e,world:t})=>{let n=p(s(e),t,{irregularity:.2,bend:.6});return{seed:e,world:t,set:n,index:f(n.centers),pointGuard:g(),distanceGuard:g()}},build(e,t,n){let r=m(e,t.set),i=e.allocate(`points`,`float32x2`,t.set.count),a=e.allocate(`distances`,`float32`,t.set.count);return e.graph.add(new w({positions:r.positions.view,featureOffsets:r.featureOffsets.view,polygonOffsets:r.polygonOffsets.view,ringOffsets:r.ringOffsets.view,initialGridSize:n.initialGridSize,refinementRounds:n.refinementRounds,output:{points:i.view,distances:a.view}})),{outputs:{...r,points:i,distances:a},update:e=>h(r,p(s(t.seed),t.world,{irregularity:.2,bend:e.bend}))}},rebuildOn:[`initialGridSize`,`refinementRounds`],interaction:{kind:`time`,hint:`refine the search`,rest:[c(E,E.default),.5],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=u(E,e.unitX),o=t.index.nearest(e.x,e.y),s=`${r.initialGridSize}/${r.refinementRounds}`,c=_(t.pointGuard,s,i(n.points)),l=_(t.distanceGuard,s,i(n.distances)),d=c&&l&&o>=0;return{settle:{refinementRounds:a},glyphs:d?[{shape:`ring`,at:[c[o*2],c[o*2+1]],radius:l[o]},{shape:`dot`,at:[c[o*2],c[o*2+1]]}]:[],readout:a===r.refinementRounds?`refinementRounds ${a} · distance ${d?l[o].toFixed(3):`...`}`:`refinementRounds ${r.refinementRounds} -> ${a}`}}},draw:({inputs:e,outputs:t,theme:n})=>[{layer:`polygons`,positions:t.fillVertices,indices:t.fillIndices,owners:t.fillOwners,values:t.distances,triangleCount:e.set.fillTriangleCount,range:[0,.1]},{layer:`lines`,context:!0,segments:t.outline,count:e.set.outlineCount},{layer:`dots`,positions:t.points,radius:5,color:[...n.ink,1]}]});export{D as default};