import{C as e,T as t,m as n,s as r,x as i,y as a}from"./gpu-contributor-utils-BIwLQON5.js";import{n as o,t as s}from"./philox-random-DfVxWNPn.js";import{n as c}from"./raster-storage-preflight-CIxFc-0g.js";import{d as l,i as u,r as d,u as f}from"./kernel-C3tfQOn4.js";import{t as p}from"./gpu-scan-B4NNTfF9.js";import{o as m,t as h}from"./wgsl-kernel-nodes-DagJklH2.js";import{c as g,f as _,n as v,o as y,s as b}from"./raster-relaxation-Di9wjtxp.js";import{t as x}from"./vector-field-sampling-KzXxZj8z.js";import{t as S}from"./define-run-B-brkCe4.js";import{d as C,o as w,s as T}from"./pointer-overlay-Qh8qflcA.js";import{lt as E}from"./entry-preview-D70TdSbK.js";import{n as D,t as O}from"./vector-field-DwVuDUyq.js";var k=s.streamlinesJitter.id;function A(t,n=new Float32Array(12)){if(n.length<12)throw Error(`Streamlines parameter target must hold 12 elements`);let[,,r,i]=t.gridExtent;for(let e of[Math.fround(r),Math.fround(i)])if(!(e>0)||!Number.isFinite(e))throw Error(`GPUStreamlines gridExtent cells must have a positive size that is finite in float32`);return e(`GPUStreamlines`,`fieldExtent cellWidth`,t.fieldExtent[2]),e(`GPUStreamlines`,`fieldExtent cellHeight`,t.fieldExtent[3]),n.set([...t.fieldExtent,...t.gridExtent,1/Math.fround(r),1/Math.fround(i),t.stepLength,t.minimumSpeed??0]),n}function j(e,t=new Uint32Array(4)){if(t.length<4)throw Error(`Streamlines uint32 parameter target must hold 4 elements`);let n=e.seed??0,r=e.minimumPoints??2;for(let[e,t]of[[`seed`,n],[`minimumPoints`,r]])if(!Number.isInteger(t)||t<0||t>4294967295)throw Error(`Streamlines ${e} must be a uint32 integer`);return t.set([n,r,0,0]),t}var M=`GPUStreamlines`,N=65535,P=0,F=1,I=2,L=class{id;seedCount;maximumIterations;props;constructor(e){let r=e.id??`streamlines`;for(let[t,n]of[[`fieldWidth`,e.fieldWidth],[`fieldHeight`,e.fieldHeight],[`gridWidth`,e.gridWidth],[`gridHeight`,e.gridHeight],[`seedColumns`,e.seedColumns],[`seedRows`,e.seedRows]])if(!Number.isInteger(n)||n<1||n>65535)throw Error(`${r} ${t} must be an integer in [1, 65535]`);let i=e.seedColumns*e.seedRows;if(i>N)throw Error(`${r} seedColumns * seedRows must be at most ${N}`);if(!Number.isInteger(e.stepsPerDirection)||e.stepsPerDirection<1)throw Error(`${r} stepsPerDirection must be a positive integer`);let o=e.maximumIterations??32;if(!Number.isInteger(o)||o<1)throw Error(`${r} maximumIterations must be a positive integer`);if(l(e.velocities,[`float32x2`],`${r} velocities`),e.velocities.length!==e.fieldWidth*e.fieldHeight)throw Error(`${r} velocities must have fieldWidth * fieldHeight rows`);if(l(e.parameters,[`float32`],`${r} parameters`),e.parameters.length<12)throw Error(`${r} parameters must contain 12 float32 rows`);if(f(e.uint32Parameters,`${r} uint32Parameters`),e.uint32Parameters.length<4)throw Error(`${r} uint32Parameters must contain 4 uint32 rows`);let{output:s}=e;n(r,s.lines);let c=s.lines.status.candidateOverflow;if(t(r,!c||![s.lines.ids,s.lines.status.count,s.lines.status.overflow,s.lines.status.requiredCount,s.lineOffsets,s.positions,s.positionCount,s.status?.iterationLimitReached,s.candidates?.positions,s.candidates?.spans].some(e=>e&&u(e,c)),`output.lines.status.candidateOverflow must not overlap other outputs`),s.lines.ids.length<1)throw Error(`${r} output.lines.ids must hold at least one line`);if(f(s.lineOffsets,`${r} output.lineOffsets`),s.lineOffsets.length!==s.lines.ids.length+1)throw Error(`${r} output.lineOffsets must have lines.ids.length + 1 rows`);if(l(s.positions,[`float32x2`],`${r} output.positions`),s.positions.length<1)throw Error(`${r} output.positions must hold at least one point`);for(let[e,t]of[[`positionCount`,s.positionCount],[`status.iterationLimitReached`,s.status?.iterationLimitReached]])if(t&&(f(t,`${r} output.${e}`),t.length<1))throw Error(`${r} output.${e} must contain one uint32 row`);if(s.candidates&&(l(s.candidates.positions,[`float32x2`],`${r} candidates.positions`),f(s.candidates.spans,`${r} candidates.spans`),s.candidates.positions.length!==i*(2*e.stepsPerDirection+1)||s.candidates.spans.length!==i*4))throw Error(`${r} candidates must have seedCount * (2L + 1) points and seedCount * 4 spans`);a(r,[s.lines.ids,s.lines.status.count,s.lines.status.overflow,s.lines.status.candidateOverflow,s.lines.status.requiredCount,s.lineOffsets,s.positions,s.positionCount,s.status?.iterationLimitReached,s.candidates?.positions,s.candidates?.spans],[e.velocities,e.parameters,e.uint32Parameters]),this.id=r,this.seedCount=i,this.maximumIterations=o,this.props=e}getCommandNodes(e){let t=this.props;c(e,this.id,[t.velocities,t.parameters,t.uint32Parameters,...r(t.output),...r(t.output.lines),...Object.values(t.output.candidates??{})],{"candidate positions":t.output.candidates?.positions?0:this.seedCount*(2*t.stepsPerDirection+1)*8,"candidate spans":t.output.candidates?.spans?0:this.seedCount*16,"seed scalar planes":this.seedCount*4,"occupancy planes":t.gridWidth*t.gridHeight*4,"iteration state":8});let{id:n,props:a,seedCount:s,maximumIterations:l}=this,{output:u}=a;i(n,e,[a.velocities,a.parameters,a.uint32Parameters,u.lines.ids,u.lines.status.count,u.lines.status.overflow,u.lines.status.candidateOverflow,u.lines.status.requiredCount,u.lineOffsets,u.positions,u.positionCount,u.status?.iterationLimitReached,u.candidates?.positions,u.candidates?.spans]);let f=a.stepsPerDirection,S=2*f+1,C=a.gridWidth*a.gridHeight,w=u.lines.ids.length,T=u.positions.length,E=u.candidates?.positions??d(e,`${n}-candidates`,`float32x2`,s*S),D=u.candidates?.spans??d(e,`${n}-spans`,`uint32`,s*4),O=d(e,`${n}-keys`,`uint32`,s),A=d(e,`${n}-status`,`uint32`,s),j=d(e,`${n}-undecided`,`uint32`,C),N=d(e,`${n}-accepted`,`uint32`,C),L=d(e,`${n}-point-counts`,`uint32`,s),R=d(e,`${n}-point-offsets`,`uint32`,s),z=d(e,`${n}-line-flags`,`uint32`,s),B=d(e,`${n}-line-ranks`,`uint32`,s),V=d(e,`${n}-state`,`uint32`,2),H=`
const GRID_WIDTH: u32 = ${a.gridWidth}u;
const GRID_HEIGHT: u32 = ${a.gridHeight}u;
const STEPS: u32 = ${f}u;
const STRIDE: u32 = ${S}u;
fn readParameter(slot: u32) -> f32 { return parameters[parametersOffset + slot]; }
fn getGridLocal(position: vec2<f32>) -> vec2<f32> {
  return (position - vec2<f32>(readParameter(4u), readParameter(5u))) *
    vec2<f32>(readParameter(8u), readParameter(9u));
}
fn isInsideGrid(local: vec2<f32>) -> bool {
  return local.x >= 0.0 && local.y >= 0.0 && local.x < f32(GRID_WIDTH) && local.y < f32(GRID_HEIGHT);
}
fn getGridCell(position: vec2<f32>) -> u32 {
  let local = getGridLocal(position);
  let column = min(u32(max(local.x, 0.0)), GRID_WIDTH - 1u);
  let row = min(u32(max(local.y, 0.0)), GRID_HEIGHT - 1u);
  return row * GRID_WIDTH + column;
}
fn readCandidate(slot: u32) -> vec2<f32> {
  return vec2<f32>(candidates[candidatesOffset + 2u * slot], candidates[candidatesOffset + 2u * slot + 1u]);
}`,U=[];U.push(m(e,{id:`${n}-trace`,operation:M,variant:`trace`,bindings:[{name:`velocities`,view:a.velocities,type:`f32`,access:`read`},{name:`parameters`,view:a.parameters,type:`f32`,access:`read`},{name:`words`,view:a.uint32Parameters,type:`u32`,access:`read`},{name:`candidates`,view:E,type:`f32`,access:`read_write`},{name:`spans`,view:D,type:`u32`,access:`read_write`},{name:`keys`,view:O,type:`u32`,access:`read_write`},{name:`status`,view:A,type:`u32`,access:`read_write`}],invocationCount:s,declarations:`${o}
${x(`velocities`,a.fieldWidth,a.fieldHeight)}
${H}
const SEED_COLUMNS: u32 = ${a.seedColumns}u;
const SEED_ROWS: u32 = ${a.seedRows}u;
fn writeCandidate(slot: u32, position: vec2<f32>) {
  candidates[candidatesOffset + 2u * slot] = position.x;
  candidates[candidatesOffset + 2u * slot + 1u] = position.y;
}
fn getDirection(position: vec2<f32>, fieldExtent: vec4<f32>, directionSign: f32, minimumSpeed: f32) -> vec3<f32> {
  let sample = sampleField(position, fieldExtent);
  if (sample.z == 0.0) {
    return vec3<f32>(0.0);
  }
  let speed = length(sample.xy);
  if (!(speed > 0.0) || speed < minimumSpeed) {
    return vec3<f32>(0.0);
  }
  return vec3<f32>(directionSign * sample.xy / speed, 1.0);
}`,body:`
  let fieldExtent = vec4<f32>(readParameter(0u), readParameter(1u), readParameter(2u), readParameter(3u));
  let gridOrigin = vec2<f32>(readParameter(4u), readParameter(5u));
  let seedCell = vec2<f32>(readParameter(6u), readParameter(7u)) *
    vec2<f32>(f32(GRID_WIDTH), f32(GRID_HEIGHT)) / vec2<f32>(f32(SEED_COLUMNS), f32(SEED_ROWS));
  let stepLength = readParameter(10u);
  let minimumSpeed = readParameter(11u);
  let seed = words[wordsOffset];
  let random = philox4x32(vec4<u32>(index, 0u, 0u, 0u), vec2<u32>(seed, ${k}u));
  keys[keysOffset + index] = ((random.z >> 16u) << 16u) | (0xffffu - index);
  let lattice = vec2<f32>(f32(index % SEED_COLUMNS), f32(index / SEED_COLUMNS));
  let start = gridOrigin + (lattice + vec2<f32>(philoxUnitFloat(random.x), philoxUnitFloat(random.y))) * seedCell;
  let base = index * STRIDE + STEPS;
  var nanBits = 0x7fc00000u;
  let nanPoint = vec2<f32>(bitcast<f32>(nanBits));
  for (var slot = 0u; slot < STRIDE; slot = slot + 1u) {
    writeCandidate(index * STRIDE + slot, nanPoint);
  }
  spans[spansOffset + 4u * index] = 0u;
  spans[spansOffset + 4u * index + 1u] = 0u;
  spans[spansOffset + 4u * index + 2u] = 0u;
  spans[spansOffset + 4u * index + 3u] = 0u;
  if (getDirection(start, fieldExtent, 1.0, minimumSpeed).z == 0.0 || !isInsideGrid(getGridLocal(start))) {
    status[statusOffset + index] = ${I}u;
    return;
  }
  status[statusOffset + index] = ${P}u;
  writeCandidate(base, start);
  for (var direction = 0u; direction < 2u; direction = direction + 1u) {
    let directionSign = select(-1.0, 1.0, direction == 1u);
    var position = start;
    var steps = 0u;
    for (var stepIndex = 1u; stepIndex <= STEPS; stepIndex = stepIndex + 1u) {
      let first = getDirection(position, fieldExtent, directionSign, minimumSpeed);
      if (first.z == 0.0) {
        break;
      }
      let middle = getDirection(position + 0.5 * stepLength * first.xy, fieldExtent, directionSign, minimumSpeed);
      if (middle.z == 0.0) {
        break;
      }
      let next = position + stepLength * middle.xy;
      if (!isInsideGrid(getGridLocal(next))) {
        break;
      }
      position = next;
      steps = stepIndex;
      if (direction == 1u) {
        writeCandidate(base + stepIndex, position);
      } else {
        writeCandidate(base - stepIndex, position);
      }
    }
    spans[spansOffset + 4u * index + direction] = steps;
  }`})),U.push(h(e,{id:`${n}-clear-accepted`,operation:M,view:N,type:`u32`,value:`0u`}));let W=g(e,`${n}-rounds`,M,s,n);U.push(b(e,{id:`${n}-rounds-reset`,operation:M,state:W}));let G=v(e,{id:`${n}-round-gate`,operation:M,contributorId:n,groupSize:1});U.push(G.writeNode);let K=Math.ceil(s/256)*256;for(let t=0;t<l;t++){let r=_(W,`${n}-round-${t}-claim`),i=_(W,`${n}-round-${t}-decide`),o=_(W,`${n}-round-${t}-clear`);U.push(m(e,{id:`${n}-round-${t}-clear`,operation:M,variant:`clear-grid`,bindings:[{name:`undecidedMaximum`,view:j,type:`u32`,access:`read_write`}],invocationCount:s,guardIndex:!1,condition:o.condition,extraResources:o.extraResources,body:`for (var cell = index; cell < ${C}u; cell = cell + ${K}u) {
    undecidedMaximum[undecidedMaximumOffset + cell] = 0u;
  }`}),m(e,{id:`${n}-round-${t}-claim`,operation:M,variant:`claim`,condition:r.condition,extraResources:r.extraResources,bindings:[{name:`parameters`,view:a.parameters,type:`f32`,access:`read`},{name:`words`,view:a.uint32Parameters,type:`u32`,access:`read`},{name:`candidates`,view:E,type:`f32`,access:`read`},{name:`spans`,view:D,type:`u32`,access:`read_write`},{name:`keys`,view:O,type:`u32`,access:`read`},{name:`status`,view:A,type:`u32`,access:`read_write`},{name:`accepted`,view:N,type:`u32`,access:`read`},{name:`undecidedMaximum`,view:j,type:`atomic<u32>`,access:`read_write`}],invocationCount:s,declarations:H,body:`
  if (status[statusOffset + index] != ${P}u) {
    return;
  }
  // Occupancy only grows, so the line can keep at most the points before its first occupied
  // cell in each direction. Reject now when that cannot reach minimumPoints.
  let base = index * STRIDE + STEPS;
  if (accepted[acceptedOffset + getGridCell(readCandidate(base))] != 0u) {
    status[statusOffset + index] = ${I}u;
    return;
  }
  let backward = spans[spansOffset + 4u * index];
  let forward = spans[spansOffset + 4u * index + 1u];
  var keptBackward = 0u;
  while (keptBackward < backward &&
      accepted[acceptedOffset + getGridCell(readCandidate(base - keptBackward - 1u))] == 0u) {
    keptBackward = keptBackward + 1u;
  }
  var keptForward = 0u;
  while (keptForward < forward &&
      accepted[acceptedOffset + getGridCell(readCandidate(base + keptForward + 1u))] == 0u) {
    keptForward = keptForward + 1u;
  }
  if (1u + keptBackward + keptForward < words[wordsOffset + 1u]) {
    status[statusOffset + index] = ${I}u;
    return;
  }
  spans[spansOffset + 4u * index + 2u] = keptBackward;
  spans[spansOffset + 4u * index + 3u] = keptForward;
  let key = keys[keysOffset + index];
  for (var slot = base - keptBackward; slot <= base + keptForward; slot = slot + 1u) {
    atomicMax(&undecidedMaximum[undecidedMaximumOffset + getGridCell(readCandidate(slot))], key);
  }`}),m(e,{id:`${n}-round-${t}-decide`,operation:M,variant:`decide`,condition:i.condition,extraResources:i.extraResources,bindings:[{name:`parameters`,view:a.parameters,type:`f32`,access:`read`},{name:`candidates`,view:E,type:`f32`,access:`read`},{name:`spans`,view:D,type:`u32`,access:`read`},{name:`keys`,view:O,type:`u32`,access:`read`},{name:`status`,view:A,type:`u32`,access:`read_write`},{name:`undecidedMaximum`,view:j,type:`u32`,access:`read`},{name:`accepted`,view:N,type:`u32`,access:`read_write`},{name:`iterationStatus`,view:W.status,type:`atomic<u32>`,access:`read_write`}],invocationCount:s,declarations:H,body:`
  if (status[statusOffset + index] != ${P}u) {
    return;
  }
  let key = keys[keysOffset + index];
  let base = index * STRIDE + STEPS;
  let first = base - spans[spansOffset + 4u * index + 2u];
  let last = base + spans[spansOffset + 4u * index + 3u];
  for (var slot = first; slot <= last; slot = slot + 1u) {
    if (undecidedMaximum[undecidedMaximumOffset + getGridCell(readCandidate(slot))] != key) {
      return;
    }
  }
  // Ready: no undecided line with a higher key can still occupy these cells, every decided one
  // already has, and no line decided in this dispatch touches them, so the claimed span is the
  // greedy trim.
  status[statusOffset + index] = ${F}u;
  atomicOr(&iterationStatus[iterationStatusOffset], 1u);
  for (var slot = first; slot <= last; slot = slot + 1u) {
    accepted[acceptedOffset + getGridCell(readCandidate(slot))] = index + 1u;
  }`})),U.push(y(e,{id:`${n}-round-gate-${t}`,operation:M,state:W,maxIterations:l,groupSize:G.getSlot(1)}))}U.push(h(e,{id:`${n}-clear-state`,operation:M,view:V,type:`u32`,value:`0u`}),m(e,{id:`${n}-counts`,operation:M,variant:`counts`,bindings:[{name:`spans`,view:D,type:`u32`,access:`read`},{name:`status`,view:A,type:`u32`,access:`read`},{name:`pointCounts`,view:L,type:`u32`,access:`read_write`},{name:`lineFlags`,view:z,type:`u32`,access:`read_write`},{name:`state`,view:V,type:`u32`,access:`read_write`}],invocationCount:s,body:`
  let lineStatus = status[statusOffset + index];
  let isAccepted = lineStatus == ${F}u;
  pointCounts[pointCountsOffset + index] = select(
    0u,
    1u + spans[spansOffset + 4u * index + 2u] + spans[spansOffset + 4u * index + 3u],
    isAccepted
  );
  lineFlags[lineFlagsOffset + index] = select(0u, 1u, isAccepted);
  if (lineStatus == ${P}u) {
    state[stateOffset + 1u] = 1u;
  }`}),...new p({id:`${n}-point-scan`,input:L,output:R,mode:`exclusive`}).getCommandNodes(e),...new p({id:`${n}-line-scan`,input:z,output:B,mode:`inclusive`}).getCommandNodes(e));let q=`
const LINE_CAPACITY: u32 = ${w}u;
const POINT_CAPACITY: u32 = ${T}u;
fn getFittingRank(index: u32) -> u32 {
  if (lineFlags[lineFlagsOffset + index] == 0u) {
    return 0xffffffffu;
  }
  let rank = lineRanks[lineRanksOffset + index] - 1u;
  let end = pointOffsets[pointOffsetsOffset + index] + pointCounts[pointCountsOffset + index];
  if (rank >= LINE_CAPACITY || end > POINT_CAPACITY) {
    return 0xffffffffu;
  }
  return rank;
}`;U.push(m(e,{id:`${n}-emit-points`,operation:M,variant:`emit-points`,bindings:[{name:`candidates`,view:E,type:`f32`,access:`read`},{name:`spans`,view:D,type:`u32`,access:`read`},{name:`lineFlags`,view:z,type:`u32`,access:`read`},{name:`lineRanks`,view:B,type:`u32`,access:`read`},{name:`pointCounts`,view:L,type:`u32`,access:`read`},{name:`pointOffsets`,view:R,type:`u32`,access:`read`},{name:`points`,view:u.positions,type:`f32`,access:`read_write`}],invocationCount:s,declarations:`const STEPS: u32 = ${f}u;
const STRIDE: u32 = ${S}u;
${q}`,body:`
  if (getFittingRank(index) == 0xffffffffu) {
    return;
  }
  let first = index * STRIDE + STEPS - spans[spansOffset + 4u * index + 2u];
  let offset = pointOffsets[pointOffsetsOffset + index];
  let count = pointCounts[pointCountsOffset + index];
  for (var point = 0u; point < count; point = point + 1u) {
    points[pointsOffset + 2u * (offset + point)] = candidates[candidatesOffset + 2u * (first + point)];
    points[pointsOffset + 2u * (offset + point) + 1u] = candidates[candidatesOffset + 2u * (first + point) + 1u];
  }`}),m(e,{id:`${n}-emit-lines`,operation:M,variant:`emit-lines`,bindings:[{name:`lineFlags`,view:z,type:`u32`,access:`read`},{name:`lineRanks`,view:B,type:`u32`,access:`read`},{name:`pointCounts`,view:L,type:`u32`,access:`read`},{name:`pointOffsets`,view:R,type:`u32`,access:`read`},{name:`ids`,view:u.lines.ids,type:`u32`,access:`read_write`},{name:`pathOffsets`,view:u.lineOffsets,type:`u32`,access:`read_write`},{name:`state`,view:V,type:`atomic<u32>`,access:`read_write`}],invocationCount:s,declarations:q,body:`
  let rank = getFittingRank(index);
  if (rank == 0xffffffffu) {
    return;
  }
  ids[idsOffset + rank] = index;
  pathOffsets[pathOffsetsOffset + rank + 1u] =
    pointOffsets[pointOffsetsOffset + index] + pointCounts[pointCountsOffset + index];
  atomicMax(&state[stateOffset], rank + 1u);`}));let J=[{name:`lineRanks`,view:B,type:`u32`,access:`read`},{name:`state`,view:V,type:`u32`,access:`read`},{name:`pathOffsets`,view:u.lineOffsets,type:`u32`,access:`read_write`},{name:`countOut`,view:u.lines.status.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:u.lines.status.overflow,type:`u32`,access:`read_write`},{name:`pointCountOut`,view:u.positionCount,type:`u32`,access:`read_write`}],Y=[...u.lines.status.requiredCount?[{name:`totalOut`,view:u.lines.status.requiredCount,type:`u32`,access:`read_write`}]:[],...u.status?.iterationLimitReached?[{name:`unconvergedOut`,view:u.status?.iterationLimitReached,type:`u32`,access:`read_write`}]:[]];return U.push(m(e,{id:`${n}-finalize`,operation:M,variant:`finalize`,bindings:[...J,...Y],invocationCount:1,body:`
  let total = lineRanks[lineRanksOffset + ${s-1}u];
  let count = state[stateOffset];
  pathOffsets[pathOffsetsOffset] = 0u;
  countOut[countOutOffset] = count;
  overflowOut[overflowOutOffset] = select(0u, 1u, count < total);
  pointCountOut[pointCountOutOffset] = select(0u, pathOffsets[pathOffsetsOffset + count], count > 0u);
  ${u.lines.status.requiredCount?`totalOut[totalOutOffset] = total;`:``}
  ${u.status?.iterationLimitReached?`unconvergedOut[unconvergedOutOffset] = state[stateOffset + 1u];`:``}`})),u.lines.status.candidateOverflow&&U.push(m(e,{id:`${n}-candidate-status`,operation:M,invocationCount:1,bindings:[{name:`state`,view:V,type:`u32`,access:`read`},{name:`candidateOverflowOut`,view:u.lines.status.candidateOverflow,type:`u32`,access:`read_write`}],body:`candidateOverflowOut[candidateOverflowOutOffset] = select(0u, 1u, state[stateOffset + 1u] != 0u);`})),U}},R=40,z=22,B=48,V=880,H=V*97,U=S({parameters:E,caption:`Evenly spaced lines that follow the vector field.`,createInputs:e=>O(e),build(e,t,n){let r=t.columns*t.cellSize,i=e.upload(`velocities`,`float32x2`,t.velocities),a=e.allocate(`parameters`,`float32`,12),o=e.allocate(`words`,`uint32`,4),s=e.upload(`speeds`,`float32`,t.speeds),c=e.allocate(`positions`,`float32x2`,H);e.graph.add(new L({velocities:i.view,fieldWidth:t.columns,fieldHeight:t.rows,gridWidth:80,gridHeight:44,seedColumns:R,seedRows:z,stepsPerDirection:B,parameters:a.view,uint32Parameters:o.view,output:{lines:{ids:e.allocate(`line-ids`,`uint32`,V).view,status:{count:e.allocate(`line-count`,`uint32`,1).view,overflow:e.allocate(`overflow`,`uint32`,1).view}},lineOffsets:e.allocate(`line-offsets`,`uint32`,881).view,positions:c.view,positionCount:e.allocate(`position-count`,`uint32`,1).view}}));let l=e.allocate(`slow-cells`,`uint32`,t.columns*t.rows);return{outputs:{speeds:s,positions:c,slowCells:l},update:e=>{let n={fieldExtent:[0,0,t.cellSize,t.cellSize],gridExtent:[0,0,r/80,1/44],stepLength:e.stepLength,minimumSpeed:e.minimumSpeed,seed:Math.round(e.seed),minimumPoints:8};a.write(A(n)),o.write(j(n)),c.write(new Float32Array(H*2).fill(NaN)),l.write(Uint32Array.from(t.operationSpeeds,t=>+(t<e.minimumSpeed)))}}},interaction:{kind:`level`,hint:`pick a speed`,rest:({inputs:e,toUnit:t})=>{let n=T(E,`minimumSpeed`),r=[e.columns/2,e.rows/2],i=1/0;for(let t=0;t<e.operationSpeeds.length;t++){if(C(n,e.operationSpeeds[t])!==n.default)continue;let a=t%e.columns,o=Math.floor(t/e.columns),s=Math.hypot(a-e.columns/2,o-e.rows/2);s<i&&(i=s,r=[a+.5,o+.5])}return t([r[0]*e.cellSize,(e.rows-r[1])*e.cellSize])},resolve(e,{inputs:t}){let n=T(E,`minimumSpeed`),r=t.operationSpeeds[D(t,e.x,e.y)],i=w(C(n,r),n.min,n.max);return{values:{minimumSpeed:i},glyphs:[{shape:`crosshair`,at:[e.x,e.y]}],readout:`minimumSpeed ${i.toFixed(2)} (wind here)`}}},draw:({inputs:e,outputs:t})=>[{layer:`raster`,context:!0,values:t.speeds,columns:e.columns,rows:e.rows,smoothing:!0},{layer:`raster`,values:t.slowCells,columns:e.columns,rows:e.rows,categories:[[0,0,0,0],[0,0,0,.55]],ramp:`categorical`},{layer:`dots`,positions:t.positions,radius:1.7,color:[.92,.97,1,.95]}]});export{U as default};