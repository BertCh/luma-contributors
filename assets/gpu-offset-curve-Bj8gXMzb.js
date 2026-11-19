import{s as e,v as t}from"./spatial-context-DwZEd4EZ.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{E as i,S as a,T as o,f as s,g as c,p as l,v as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{o as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as p}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as m}from"./exact-geometry-wgsl-y7TUnqOh.js";import{t as h}from"./validate-outline-output-passes-DIcqH6Pw.js";import{t as g}from"./gpu-outline-geometry-DoqtUFti.js";import{a as _}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as v,r as y,t as b}from"./line-geometry-output-Eyy3fjKL.js";var x=Object.freeze({...e,measure:`distance`}),S=`GPUOffsetCurve`,C=2147483647,w=4294967295,T=34028234663852886e22;function E(e,t=new Float32Array(4)){if(t.length<4)throw Error(`Offset curve target must hold 4 elements`);let n=e.mitreLimit??5;if(!Number.isFinite(e.distance))throw Error(`Offset curve distance must be a finite number`);if(!Number.isFinite(n)||n<1)throw Error(`Offset curve mitreLimit must be a finite number of at least 1`);return c(S,`distance`,Math.fround(e.distance),-34028234663852886e22,T),c(S,`mitreLimit`,Math.fround(n),1,T),t.set([e.distance,n,0,0]),t}function D(e,t,n=`intersection`){let r=n===`buffer`?3:2;return Math.max(e===`round`?2*t+1:2,r)}var O=class{id;props;spatialContext;geometryType;boundaryOffsets;joinStyle;quadSegments;rowsPerVertex;insideTurns;estimatedCommandNodeCount;constructor(e){this.id=e.id??`offset-curve`,this.props=e,this.estimatedCommandNodeCount=e.output.layout===`compact`?11:1,l(this.id,this.estimatedCommandNodeCount),this.spatialContext=t(this.id,e.spatialContext,x),this.geometryType=e.geometryType??`lines`,this.boundaryOffsets=e.geometryType===`polygons`?e.ringOffsets:e.lineOffsets,this.joinStyle=e.joinStyle??`round`,this.quadSegments=e.quadSegments??8,this.insideTurns=e.insideTurns??`intersection`,this.rowsPerVertex=D(this.joinStyle,this.quadSegments,this.insideTurns);let{id:n}=this;if(this.insideTurns!==`intersection`&&this.insideTurns!==`buffer`)throw Error(`${n} insideTurns must be 'intersection' or 'buffer'`);if(this.geometryType!==`lines`&&this.geometryType!==`polygons`)throw Error(`${n} geometryType must be 'lines' or 'polygons'`);if(![`round`,`mitre`,`bevel`].includes(this.joinStyle))throw Error(`${n} joinStyle must be 'round', 'mitre' or 'bevel'`);if(s(n,`quadSegments`,this.quadSegments,1,256),i(e.positions,[`float32x2`],`${n} positions`),e.positions.length<1)throw Error(`${n} needs at least one position`);let r=e.geometryType===`polygons`?`ringOffsets`:`lineOffsets`;if(i(this.boundaryOffsets,[`uint32`],`${n} ${r}`),this.boundaryOffsets.length<2)throw Error(`${n} ${r} needs two rows`);if(e.geometryType===`polygons`&&(i(e.featureOffsets,[`uint32`],`${n} featureOffsets`),i(e.polygonOffsets,[`uint32`],`${n} polygonOffsets`)),i(e.parameters,[`float32`],`${n} parameters`),e.parameters.length<4)throw Error(`${n} parameters must hold 4 float32 values`);let c=e.positions.length*this.rowsPerVertex;a(n,`positions.length * rowsPerVertex`,c,w),s(n,`${r}.length`,this.boundaryOffsets.length,2),e.output.layout===`compact`?(y(n,e.output,this.boundaryOffsets.length-1),o(n,!(`measures`in e.output&&e.output.measures),`output.measures is unsupported for offset corners`)):(o(n,e.output.layout===void 0||e.output.layout===`fixed-stride`,`output.layout must be 'fixed-stride' or 'compact'`),i(e.output.positions,[`float32x2`],`${n} output.positions`),o(n,e.output.positions.length===c,`output.positions must hold ${c} rows`),e.output.counts&&(i(e.output.counts,[`uint32`],`${n} output.counts`),o(n,e.output.counts.length===e.positions.length,`output.counts must hold ${e.positions.length} rows`))),a(n,`output.positions.length`,e.output.positions.length,C);let u=this.getOutputViews(),f=e.output;f.layout===`compact`?h(n,{positions:f.positions,sourceRows:f.sourceRows,lineOffsets:f.lineOffsets,sourceLineIndices:f.sourceLineIndices,...f.status},[[`positions`,`sourceRows`],[`lineOffsets`,`overflow`,`vertexCount`,`requiredVertexCount`,`sourceLineIndices`],[`count`,`requiredCount`]]):h(n,{positions:f.positions,counts:f.counts},[[`positions`,`counts`]]),d(n,u,[e.positions,this.boundaryOffsets,e.parameters])}getOutputViews(){let{output:e}=this.props;return e.layout===`compact`?v(e):[e.positions,e.counts]}getCommandNodes(e){let{props:t,id:i}=this,a=t.output,o=a.layout===`compact`?a:void 0,c=a.layout===`compact`?void 0:a.counts;if(_(e,i,{positions:t.positions,boundaryOffsets:this.boundaryOffsets,parameters:t.parameters,...Object.fromEntries(this.getOutputViews().map((e,t)=>[`output.${t}`,e]))}),o){let n=(t.positions.length+1)*Uint32Array.BYTES_PER_ELEMENT;u(e.device,i,`compact count scratch`,n,`reduce the input vertex count`),s(i,`compact count scratch byteLength`,n,0,e.device.limits.maxBufferSize)}let l=o?n(e,`${i}-counts`,`uint32`,t.positions.length+1):void 0,d=o?n(e,`${i}-starts`,`uint32`,t.positions.length+1):void 0,p=[];return l&&d&&(p.push(this.createGenerationNode(e,{counts:l,countOnly:!0})),p.push(...new r({id:`${i}-scan`,input:l,output:d}).getCommandNodes(e))),p.push(this.createGenerationNode(e,{counts:c,starts:d,compactOutput:o})),o&&d&&(p.push(f(e,{id:`${i}-publish`,operation:S,variant:`compact-publish`,invocationCount:this.boundaryOffsets.length,bindings:[{name:`starts`,view:d,type:`u32`,access:`read`},{name:`boundaries`,view:this.boundaryOffsets,type:`u32`,access:`read`},{name:`offsets`,view:o.lineOffsets,type:`u32`,access:`read_write`},{name:`overflow`,view:o.status.overflow,type:`u32`,access:`read_write`},...[[`vertexCount`,o.status.vertexCount],[`requiredVertexCount`,o.status.requiredVertexCount],[`sourceLineIndices`,o.sourceLineIndices]].flatMap(([e,t])=>t?[{name:e,view:t,type:`u32`,access:`read_write`}]:[])],declarations:`const ROW_COUNT: u32 = ${t.positions.length}u;
const LINE_COUNT: u32 = ${this.boundaryOffsets.length-1}u;
const CAPACITY: u32 = ${o.positions.length}u;`,body:`let row = min(boundaries[boundariesOffset + index], ROW_COUNT);
offsets[offsetsOffset + index] = min(starts[startsOffset + row], CAPACITY);
${o.sourceLineIndices?`if (index < LINE_COUNT) { sourceLineIndices[sourceLineIndicesOffset + index] = index; }`:``}
if (index == 0u) {
  let required = starts[startsOffset + ROW_COUNT];
  overflow[overflowOffset] = select(0u, 1u, required > CAPACITY);
  ${o.status.vertexCount?`vertexCount[vertexCountOffset] = min(required, CAPACITY);`:``}
  ${o.status.requiredVertexCount?`requiredVertexCount[requiredVertexCountOffset] = required;`:``}
}`})),p.push(b(e,{id:i,operation:S,output:o,lineCapacity:this.boundaryOffsets.length-1}))),p}createGenerationNode(e,t){let{props:n,id:r}=this,{counts:i,countOnly:a,starts:o,compactOutput:s}=t,c=[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:this.boundaryOffsets,type:`u32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`}];return a||c.push({name:`outputPositions`,view:n.output.positions,type:`f32`,access:`read_write`}),i&&c.push({name:`outputCounts`,view:i,type:`u32`,access:`read_write`}),o&&c.push({name:`starts`,view:o,type:`u32`,access:`read`}),s?.sourceRows&&c.push({name:`sourceRows`,view:s.sourceRows,type:`u32`,access:`read_write`}),f(e,{id:`${r}-${a?`count`:`generate`}`,operation:S,variant:`${this.geometryType}-${this.joinStyle}-${this.insideTurns}`,bindings:c,invocationCount:n.positions.length+ +!!a,declarations:`
const LINE_COUNT: u32 = ${this.boundaryOffsets.length-1}u;
const ROW_COUNT: u32 = ${n.positions.length}u;
const ROWS_PER_VERTEX: u32 = ${this.rowsPerVertex}u;
const QUAD_SEGMENTS: f32 = ${this.quadSegments}.0;
const IS_RING: bool = ${this.geometryType===`polygons`};
const HALF_PI: f32 = 1.5707963267948966;

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getLeftNormal(direction: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(-direction.y, direction.x);
}

fn rotateVector(v: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}
${g}
${p}
${m}`,body:`${a?`if (index == ROW_COUNT) { outputCounts[outputCountsOffset + index] = 0u; return; }`:``}
  let offsetDistance = parameters[parametersOffset];
  let mitreLimit = max(parameters[parametersOffset + 1u], 1.0);
  let magnitude = abs(offsetDistance);
  let center = getPosition(index);
  var points = array<vec2<f32>, ${this.rowsPerVertex}>();
  var count = 1u;
  points[0] = center;

  var hasIncoming = false;
  var hasOutgoing = false;
  var incoming = vec2<f32>(0.0);
  var outgoing = vec2<f32>(0.0);
  var incomingLength = 0.0;
  var outgoingLength = 0.0;
  var previousPoint = center;
  var nextPoint = center;
  let line = findLine(index);
  if (line != NO_LINE) {
    let lineStart = lineOffsets[lineOffsetsOffset + line];
    let lineEnd = min(lineOffsets[lineOffsetsOffset + line + 1u], ROW_COUNT);
    let lineLength = lineEnd - lineStart;
    var previousRow = ROW_COUNT;
    var nextRow = ROW_COUNT;
    if (index > lineStart) {
      previousRow = index - 1u;
    } else if (IS_RING && lineLength >= 3u) {
      previousRow = lineEnd - 1u;
    }
    if (index + 1u < lineEnd) {
      nextRow = index + 1u;
    } else if (IS_RING && lineLength >= 3u) {
      nextRow = lineStart;
    }
    if (previousRow < ROW_COUNT) {
      previousPoint = getPosition(previousRow);
      let delta = center - previousPoint;
      let deltaLength = length(delta);
      if (deltaLength > 0.0) {
        incoming = delta / deltaLength;
        incomingLength = deltaLength;
        hasIncoming = true;
      }
    }
    if (nextRow < ROW_COUNT) {
      nextPoint = getPosition(nextRow);
      let delta = nextPoint - center;
      let deltaLength = length(delta);
      if (deltaLength > 0.0) {
        outgoing = delta / deltaLength;
        outgoingLength = deltaLength;
        hasOutgoing = true;
      }
    }
  }

  if (hasIncoming && !hasOutgoing) {
    points[0] = center + getLeftNormal(incoming) * offsetDistance;
  } else if (!hasIncoming && hasOutgoing) {
    points[0] = center + getLeftNormal(outgoing) * offsetDistance;
  } else if (hasIncoming && hasOutgoing) {
    let normalIn = getLeftNormal(incoming);
    let normalOut = getLeftNormal(outgoing);
    let a = normalIn * offsetDistance;
    let b = normalOut * offsetDistance;
    // Which side is the inside of the turn is an exact orientation sign of the three vertices.
    // The f32 cross product of the two unit directions loses that sign where the edges nearly
    // reverse (the tip of a sliver thinner than one f32 step): the join then went to the wrong
    // side and a buffer lost its end cap there. The f32 sign is kept only for input the exact
    // predicate cannot certify (code 2: non-finite coordinates).
    let exactTurn = exactOrientSigns(
      1u,
      array<vec2f, 4>(previousPoint, vec2f(0.0), vec2f(0.0), vec2f(0.0)),
      array<vec2f, 4>(center, vec2f(0.0), vec2f(0.0), vec2f(0.0)),
      array<vec2f, 4>(nextPoint, vec2f(0.0), vec2f(0.0), vec2f(0.0))
    );
    let turn = select(f32(exactTurn.signs.x), sign(incoming.x * outgoing.y - incoming.y * outgoing.x), exactTurn.uncertain);
    let cosTurn = clamp(dot(incoming, outgoing), -1.0, 1.0);
    if (turn * offsetDistance > 0.0 && 1.0 + cosTurn > 1e-6) {
      // Inside of the turn: intersection of the two offset segments.
      ${k(this.insideTurns)}${this.insideTurns===`buffer`?`
    } else if (turn * offsetDistance > 0.0) {
      // A full reversal on the inside: route through the vertex.
      points[0] = center + a;
      points[1] = center;
      points[2] = center + b;
      count = 3u;`:``}
    } else if (length(a - b) < magnitude * 1e-3) {
      points[0] = center + a;
    } else {
      ${A(this.joinStyle)}
    }
  }

  ${a?``:`let base = ${o?`starts[startsOffset + index]`:`index * ROWS_PER_VERTEX`};
  for (var slot = 0u; slot < ${o?`count`:`ROWS_PER_VERTEX`}; slot++) {
    ${o?`if (base + slot >= ${n.output.positions.length}u) { break; }`:``}
    let point = points[min(slot, count - 1u)];
    let destination = base + slot;
    let coordinateOffset = outputPositionsOffset + 2u * destination;
    outputPositions[coordinateOffset] = point.x;
    outputPositions[coordinateOffset + 1u] = point.y;
    ${s?.sourceRows?`sourceRows[sourceRowsOffset + destination] = index;`:``}
  }`}
  ${i?`outputCounts[outputCountsOffset + index] = count;`:``}`})}};function k(e){return e===`intersection`?`points[0] = center + (a + b) / (1.0 + cosTurn);`:`let inset = magnitude * sqrt(max(1.0 - cosTurn, 0.0) / (1.0 + cosTurn));
      if (inset > incomingLength || inset > outgoingLength) {
        points[0] = center + a;
        points[1] = center;
        points[2] = center + b;
        count = 3u;
      } else {
        points[0] = center + (a + b) / (1.0 + cosTurn);
      }`}function A(e){return e===`bevel`?`points[0] = center + a;
      points[1] = center + b;
      count = 2u;`:e===`round`?`let turnAngle = acos(cosTurn);
      let steps = u32(floor(turnAngle * QUAD_SEGMENTS / HALF_PI + 0.5));
      let direction = select(1.0, -1.0, offsetDistance > 0.0);
      points[0] = center + a;
      if (steps >= 1u) {
        for (var arcStep = 1u; arcStep < steps; arcStep++) {
          points[arcStep] = center + rotateVector(a, direction * turnAngle * f32(arcStep) / f32(steps));
        }
        points[steps] = center + b;
        count = steps + 1u;
      } else {
        points[1] = center + b;
        count = 2u;
      }`:`let cosHalf = sqrt(max(0.5 * (1.0 + cosTurn), 0.0));
      if (cosHalf > 0.0 && 1.0 / cosHalf <= mitreLimit) {
        points[0] = center + (a + b) / (1.0 + cosTurn);
      } else {
        // Clipped mitre: the offset lines cut by the line perpendicular to the bisector at mitreLimit * distance.
        var bisector = a + b;
        if (length(bisector) < 1e-6 * magnitude) {
          bisector = incoming * magnitude;
        }
        let tipDirection = normalize(bisector);
        let clipMiddle = center + tipDirection * (mitreLimit * magnitude);
        let halfLength = magnitude * (1.0 - mitreLimit * cosHalf) / sqrt(max(1.0 - cosHalf * cosHalf, 1e-12));
        let perpendicular = getLeftNormal(tipDirection) * select(-1.0, 1.0, dot(a, getLeftNormal(tipDirection)) >= 0.0);
        points[0] = clipMiddle + perpendicular * halfLength;
        points[1] = clipMiddle - perpendicular * halfLength;
        count = 2u;
      }`}export{E as n,D as r,O as t};