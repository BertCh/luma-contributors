import{i as e,r as t}from"./weights-construction-snapshot-CikLt8kl.js";import{i as n,t as r}from"./spatial-weights-loss-u59PcHkg.js";import{r as i}from"./kernel-C3tfQOn4.js";import{t as a}from"./addition-rounding-error-wgsl-DVwPAqKC.js";import{a as o,o as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as c,S as l,f as u,p as d,u as f,x as p,y as m}from"./gpu-contributor-utils-BIwLQON5.js";import{c as h}from"./spatial-weights-CqWysstN.js";import{i as g,n as _,r as v,t as y}from"./gpu-spatial-weights-minimum-spanning-tree-DUMWjHV2.js";import{t as b}from"./define-run-B-brkCe4.js";import{A as x,a as S}from"./pointer-overlay-Qh8qflcA.js";import{m as C}from"./inputs-CJKKIRKK.js";import{Os as w}from"./entry-preview-D70TdSbK.js";import{c as T}from"./geometry-Dz9W1tyx.js";import{c as E,n as D,s as O,t as k}from"./regions-CO_uxNyo.js";var A=`GPUSkaterRegions`;function j(e){return f(e,4)}var M=0,N=1;function P(e){let t=e.minimumRegionSize??1;u(`GPUSkaterRegions`,`regionCount`,e.regionCount),u(`GPUSkaterRegions`,`minimumRegionSize`,t);let n=new Uint32Array(2);return n[M]=e.regionCount,n[N]=t,n}var ee=0,te=1,ne=2,re=3,ie=4,ae=5,oe=6,se=7,ce=8,F=9,I=class{id;props;estimatedCommandNodeCount;stepCount;constructor(r){let i=t(r,[`id`,`weights`,`treeEdgeFlags`,`componentLabels`,`treeStatus`,`values`,`columnCount`,`maximumRegionCount`,`parameters`,`output`]);r=Object.freeze({...i,weights:e(i.weights),output:t(i.output,[`labels`,`regionCount`,`cutEdges`,`cutGains`])});let a=r.id??`skater-regions`;this.id=a,this.props=r;let o=h(a,r.weights),s=r.columnCount??1,{output:l}=r;u(a,`columnCount`,s,1,16),u(a,`maximumRegionCount`,r.maximumRegionCount,2),this.stepCount=Math.min(r.maximumRegionCount-1,Math.max(0,o-1));let f=n(a,[r.weights]),p=f.length+Number(!!r.treeStatus),g=p?Math.ceil(p/7)+1+Number(!!l.regionCount)+Number(!!l.cutEdges)+Number(!!l.cutGains):0;if(this.estimatedCommandNodeCount=11+Math.max(1,Math.ceil(Math.log2(Math.max(2,2*o))))+8*this.stepCount+g,d(a,this.estimatedCommandNodeCount),c(r.values,[`float32`],`${a} values`),r.values.length!==o*s)throw Error(`${a} values length must equal rows * columnCount`);if(c(r.treeEdgeFlags,[`uint32`],`${a} treeEdgeFlags`),r.treeEdgeFlags.length!==r.weights.neighbors.length)throw Error(`${a} treeEdgeFlags length must equal the weights slot capacity`);if(c(r.componentLabels,[`uint32`],`${a} componentLabels`),r.componentLabels.length!==o)throw Error(`${a} componentLabels length must equal the weights row count`);if(r.treeStatus&&(c(r.treeStatus,[`uint32`],`${a} treeStatus`),r.treeStatus.length<_.length))throw Error(`${a} treeStatus must hold ${_.length} word`);if(c(r.parameters,[`uint32`],`${a} parameters`),r.parameters.length<2)throw Error(`${a} parameters must hold 2 words`);if(c(l.labels,[`uint32`],`${a} output.labels`),l.labels.length!==o)throw Error(`${a} output.labels length must equal the weights row count`);l.regionCount&&c(l.regionCount,[`uint32`],`${a} output.regionCount`);let v=r.maximumRegionCount-1;if(l.cutEdges&&(c(l.cutEdges,[`uint32`],`${a} output.cutEdges`),l.cutEdges.length<v))throw Error(`${a} output.cutEdges must hold maximumRegionCount - 1 entries`);if(l.cutGains&&(c(l.cutGains,[`float32`],`${a} output.cutGains`),l.cutGains.length<v))throw Error(`${a} output.cutGains must hold maximumRegionCount - 1 entries`);m(a,[l.labels,l.regionCount,l.cutEdges,l.cutGains],[r.values,r.parameters,r.treeStatus,r.treeEdgeFlags,r.componentLabels,r.weights.offsets,r.weights.neighbors,...f])}getCommandNodes(e){let{id:t,props:c}=this,{weights:u,values:d,output:f}=c,{labels:m}=f;p(t,e,[d,c.treeEdgeFlags,c.componentLabels,c.parameters,c.treeStatus,m,f.regionCount,f.cutEdges,f.cutGains,u.offsets,u.neighbors,...n(t,[u])]);let h=u.offsets.length-1,y=u.neighbors.length,b=c.columnCount??1,x=j(e.device);l(t,`weights.neighbors rows x (2 + columnCount)`,y*(2+b),x),l(t,`weights rows x (2 + columnCount)`,h*(2+b),x);let S=this.stepCount,C=2+b,w=F+2*S,T=c.treeStatus?e.createDataView(c.treeStatus.buffer,{format:`uint32`,byteOffset:c.treeStatus.byteOffset+4*_.candidateOverflow,length:1}):void 0,E=r(e,t,[u,{status:{candidateOverflow:T}}]),D=i(e,`${t}-twin`,`uint32`,y),O=i(e,`${t}-arc-flags`,`uint32`,y),k=i(e,`${t}-arc-info`,`uint32`,y),P=i(e,`${t}-gain-key`,`uint32`,y),I=i(e,`${t}-rank-next-a`,`uint32`,y),L=i(e,`${t}-rank-next-b`,`uint32`,y),R=i(e,`${t}-rank-distance-a`,`uint32`,y),le=i(e,`${t}-rank-distance-b`,`uint32`,y),z=i(e,`${t}-arc-stats`,`float32`,y*C),B=i(e,`${t}-row-down-arc`,`uint32`,h),V=i(e,`${t}-region-stats`,`float32`,h*C),H=i(e,`${t}-state`,`uint32`,w),U=o(e,{id:`${t}-step-dispatch`,operation:A,shapes:[[y],[h]]}),W=[{name:`offsets`,view:u.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:u.neighbors,type:`u32`,access:`read`}],G=`const ROWS: u32 = ${h}u;
const COLUMNS: u32 = ${b}u;
const STRIDE: u32 = ${C}u;
const STEP_COUNT: u32 = ${S}u;
const MAXIMUM_REGIONS: u32 = ${c.maximumRegionCount}u;
const STATE_BEST_KEY: u32 = ${ee}u;
const STATE_BEST_EDGE: u32 = ${te}u;
const STATE_REGION_COUNT: u32 = ${ne}u;
const STATE_APPLIED: u32 = ${re}u;
const STATE_TARGET: u32 = ${ie}u;
const STATE_MINIMUM_SIZE: u32 = ${ae}u;
const STATE_OLD_REGION: u32 = ${oe}u;
const STATE_CHILD: u32 = ${se}u;
const STATE_STEP: u32 = ${ce}u;
const STATE_LOG: u32 = ${F}u;
const SLOT_MASK: u32 = 0x3fffffffu;
${g}`,ue=`${v}
fn isLive(flags: u32) -> bool { return (flags & 1u) != 0u; }

// Successor of an arc in the Euler tour of the live arcs: the next live arc after the twin of
// the arc around the arc's head, cyclically.
fn getSuccessor(arc: u32) -> u32 {
  let head = neighbors[neighborsOffset + arc];
  let twinArc = twin[twinOffset + arc];
  let rowStart = offsets[offsetsOffset + head];
  let degree = offsets[offsetsOffset + head + 1u] - rowStart;
  for (var step = 1u; step <= degree; step++) {
    let slot = rowStart + (twinArc - rowStart + step) % degree;
    if (isLive(arcFlags[arcFlagsOffset + slot])) {
      return slot;
    }
  }
  return twinArc;
}

// First live arc of a row, or NONE.
fn getFirstLiveArc(row: u32) -> u32 {
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    if (isLive(arcFlags[arcFlagsOffset + slot])) {
      return slot;
    }
  }
  return NONE;
}`,de=`fn getInfoSuccessor(info: u32) -> u32 { return info & SLOT_MASK; }
fn isInfoLive(info: u32) -> bool { return ((info >> 30u) & 1u) != 0u; }
fn isInfoDown(info: u32) -> bool { return ((info >> 31u) & 1u) != 0u; }
fn isInfoLiveFlag(flags: u32) -> bool { return (flags & 1u) != 0u; }
${a}`,K=G,q=`${G}\n${ue}`,J=`${G}\n${de}`,Y=[...E.nodes];Y.push(s(e,{id:`${t}-clear-arcs`,operation:A,variant:`clear-arcs`,bindings:[{name:`arcFlags`,view:O,type:`u32`,access:`read_write`}],invocationCount:y,body:`arcFlags[arcFlagsOffset + index] = 0u;`})),Y.push(s(e,{id:`${t}-twins`,operation:A,variant:`twins`,bindings:[...W,{name:`treeFlags`,view:c.treeEdgeFlags,type:`u32`,access:`read`},{name:`twin`,view:D,type:`u32`,access:`read_write`},{name:`arcFlags`,view:O,type:`u32`,access:`read_write`}],invocationCount:h,declarations:`${G}\n${v}`,body:`for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let twinSlot = findSlot(neighbor, index);
    twin[twinOffset + slot] = twinSlot;
    var edge = slot;
    if (twinSlot != NONE && twinSlot < slot) {
      edge = twinSlot;
    }
    arcFlags[arcFlagsOffset + slot] =
      select(0u, 1u, twinSlot != NONE && treeFlags[treeFlagsOffset + edge] != 0u);
  }`})),Y.push(s(e,{id:`${t}-rank-init`,operation:A,variant:`rank-init`,bindings:[...W,{name:`twin`,view:D,type:`u32`,access:`read`},{name:`arcFlags`,view:O,type:`u32`,access:`read`},{name:`treeRoots`,view:c.componentLabels,type:`u32`,access:`read`},{name:`nextOut`,view:I,type:`u32`,access:`read_write`},{name:`distanceOut`,view:R,type:`u32`,access:`read_write`}],invocationCount:y,declarations:q,body:`if (!isLive(arcFlags[arcFlagsOffset + index])) {
    nextOut[nextOutOffset + index] = index;
    distanceOut[distanceOutOffset + index] = 0u;
    return;
  }
  let successor = getSuccessor(index);
  let root = treeRoots[treeRootsOffset + neighbors[neighborsOffset + index]];
  let isLast = successor == getFirstLiveArc(root);
  nextOut[nextOutOffset + index] = select(successor, index, isLast);
  distanceOut[distanceOutOffset + index] = select(1u, 0u, isLast);`}));let X={next:I,distance:R},Z={next:L,distance:le},fe=Math.max(1,Math.ceil(Math.log2(Math.max(2,2*h))));for(let n=0;n<fe;n++)Y.push(s(e,{id:`${t}-rank-jump-${n}`,operation:A,variant:`rank-jump`,bindings:[{name:`nextIn`,view:X.next,type:`u32`,access:`read`},{name:`distanceIn`,view:X.distance,type:`u32`,access:`read`},{name:`nextOut`,view:Z.next,type:`u32`,access:`read_write`},{name:`distanceOut`,view:Z.distance,type:`u32`,access:`read_write`}],invocationCount:y,body:`let successor = nextIn[nextInOffset + index];
  distanceOut[distanceOutOffset + index] =
    distanceIn[distanceInOffset + index] + distanceIn[distanceInOffset + successor];
  nextOut[nextOutOffset + index] = nextIn[nextInOffset + successor];`})),[X,Z]=[Z,X];Y.push(s(e,{id:`${t}-orient`,operation:A,variant:`orient`,bindings:[{name:`twin`,view:D,type:`u32`,access:`read`},{name:`distance`,view:X.distance,type:`u32`,access:`read`},{name:`arcFlags`,view:O,type:`u32`,access:`read_write`}],invocationCount:y,body:`let flags = arcFlags[arcFlagsOffset + index];
  if ((flags & 1u) != 0u &&
      distance[distanceOffset + index] > distance[distanceOffset + twin[twinOffset + index]]) {
    arcFlags[arcFlagsOffset + index] = flags | 2u;
  }`})),Y.push(s(e,{id:`${t}-init-state`,operation:A,variant:`init-state`,bindings:[{name:`parameters`,view:c.parameters,type:`u32`,access:`read`},{name:`state`,view:H,type:`u32`,access:`read_write`},...E.view?[{name:`weightsLoss`,view:E.view,type:`u32`,access:`read`}]:[],...c.treeStatus?[{name:`treeStatus`,view:c.treeStatus,type:`u32`,access:`read`}]:[],U.binding],invocationCount:w,declarations:K,body:`if (index == 0u) {
    ${U.armWGSL}
  }
  var word = 0u;
  if (index == STATE_TARGET) {
    // A nonzero invalid count from the tree sets the target to 0, so no step finds a candidate.
    word = ${c.treeStatus?`select(min(parameters[parametersOffset + ${M}u], MAXIMUM_REGIONS), 0u, treeStatus[treeStatusOffset + ${_.invalidCount}u] > 0u)`:`min(parameters[parametersOffset + ${M}u], MAXIMUM_REGIONS)`};
    ${E.view?`if (weightsLoss[weightsLossOffset] != 0u) { word = 0u; }`:``}
  } else if (index == STATE_MINIMUM_SIZE) {
    word = max(parameters[parametersOffset + ${N}u], 1u);
  } else if (index >= STATE_LOG && ((index - STATE_LOG) % 2u) == 0u) {
    word = NONE;
  }
  state[stateOffset + index] = word;`}),s(e,{id:`${t}-init-labels`,operation:A,variant:`init-labels`,bindings:[{name:`treeRoots`,view:c.componentLabels,type:`u32`,access:`read`},{name:`labels`,view:m,type:`u32`,access:`read_write`},{name:`state`,view:H,type:`atomic<u32>`,access:`read_write`}],invocationCount:h,declarations:K,body:`let root = treeRoots[treeRootsOffset + index];
  labels[labelsOffset + index] = root;
  if (root == index) {
    atomicAdd(&state[stateOffset + STATE_REGION_COUNT], 1u);
  }`}));let Q=X.distance;Y.push(s(e,{id:`${t}-successors`,operation:A,variant:`successors`,bindings:[...W,{name:`twin`,view:D,type:`u32`,access:`read`},{name:`arcFlags`,view:O,type:`u32`,access:`read`},{name:`arcInfo`,view:k,type:`u32`,access:`read_write`}],invocationCount:y,declarations:q,body:`let flags = arcFlags[arcFlagsOffset + index];
  var info = (flags & 3u) << 30u;
  if (isLive(flags)) {
    info = info | getSuccessor(index);
  }
  arcInfo[arcInfoOffset + index] = info;`}),s(e,{id:`${t}-row-down-arcs`,operation:A,variant:`row-down-arcs`,bindings:[...W,{name:`twin`,view:D,type:`u32`,access:`read`},{name:`arcFlags`,view:O,type:`u32`,access:`read`},{name:`rowDownArc`,view:B,type:`u32`,access:`read_write`}],invocationCount:h,declarations:J,body:`var downArc = NONE;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let flags = arcFlags[arcFlagsOffset + slot];
    if (isInfoLiveFlag(flags) && (flags & 2u) == 0u) {
      downArc = twin[twinOffset + slot];
    }
  }
  rowDownArc[rowDownArcOffset + index] = downArc;`}),s(e,{id:`${t}-tree-stats`,operation:A,variant:`tree-stats`,bindings:[...W,{name:`arcInfo`,view:k,type:`u32`,access:`read`},{name:`values`,view:d,type:`f32`,access:`read`},{name:`labels`,view:m,type:`u32`,access:`read`},{name:`regionStats`,view:V,type:`f32`,access:`read_write`}],invocationCount:h,declarations:J,body:`if (labels[labelsOffset + index] != index) {
    return;
  }
  var count = 1.0;
  var sums: array<f32, ${b}>;
  // Rounding error of every addition into sums, so a sum over a whole tree is right to its last
  // place (a plain serial float32 sum of n rows can be off by n / 2 last places).
  var roundingErrors: array<f32, ${b}>;
  for (var column = 0u; column < COLUMNS; column++) {
    sums[column] = values[valuesOffset + index * COLUMNS + column];
  }
  var head = NONE;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (isInfoLive(arcInfo[arcInfoOffset + slot])) {
      head = slot;
      break;
    }
  }
  if (head != NONE) {
    var arc = head;
    for (var guard = 0u; guard < 2u * ROWS; guard++) {
      let info = arcInfo[arcInfoOffset + arc];
      if (isInfoDown(info)) {
        let row = neighbors[neighborsOffset + arc];
        count += 1.0;
        for (var column = 0u; column < COLUMNS; column++) {
          let value = values[valuesOffset + row * COLUMNS + column];
          let next = sums[column] + value;
          roundingErrors[column] += getAdditionRoundingError(sums[column], value, next);
          sums[column] = next;
        }
      }
      arc = getInfoSuccessor(info);
      if (arc == head) {
        break;
      }
    }
  }
  let base = regionStatsOffset + index * STRIDE;
  regionStats[base] = count;
  // Word 1 is unused (it held the sum of squares the gain no longer needs); the stride is kept.
  regionStats[base + 1u] = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    regionStats[base + 2u + column] = sums[column] + roundingErrors[column];
  }`}),s(e,{id:`${t}-arc-stats`,operation:A,variant:`arc-stats`,bindings:[{name:`neighbors`,view:u.neighbors,type:`u32`,access:`read`},{name:`twin`,view:D,type:`u32`,access:`read`},{name:`arcInfo`,view:k,type:`u32`,access:`read`},{name:`values`,view:d,type:`f32`,access:`read`},{name:`arcStats`,view:z,type:`f32`,access:`read_write`}],invocationCount:y,declarations:J,body:`let startInfo = arcInfo[arcInfoOffset + index];
  if (!isInfoLive(startInfo) || !isInfoDown(startInfo)) {
    return;
  }
  let child = neighbors[neighborsOffset + index];
  let end = twin[twinOffset + index];
  var count = 1.0;
  var sums: array<f32, ${b}>;
  var roundingErrors: array<f32, ${b}>;
  for (var column = 0u; column < COLUMNS; column++) {
    sums[column] = values[valuesOffset + child * COLUMNS + column];
  }
  var arc = getInfoSuccessor(startInfo);
  for (var guard = 0u; guard < 2u * ROWS && arc != end; guard++) {
    let info = arcInfo[arcInfoOffset + arc];
    if (isInfoDown(info)) {
      let row = neighbors[neighborsOffset + arc];
      count += 1.0;
      for (var column = 0u; column < COLUMNS; column++) {
        let value = values[valuesOffset + row * COLUMNS + column];
        let next = sums[column] + value;
        roundingErrors[column] += getAdditionRoundingError(sums[column], value, next);
        sums[column] = next;
      }
    }
    arc = getInfoSuccessor(info);
  }
  let base = arcStatsOffset + index * STRIDE;
  arcStats[base] = count;
  arcStats[base + 1u] = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    arcStats[base + 2u + column] = sums[column] + roundingErrors[column];
  }`}));for(let n=0;n<S;n++){let r=`${t}-step-${n}`;Y.push(s(e,{id:`${r}-candidates`,operation:A,variant:`candidates`,...U.getGate(0),bindings:[{name:`neighbors`,view:u.neighbors,type:`u32`,access:`read`},{name:`arcFlags`,view:O,type:`u32`,access:`read`},{name:`arcStats`,view:z,type:`f32`,access:`read`},{name:`labels`,view:m,type:`u32`,access:`read`},{name:`regionStats`,view:V,type:`f32`,access:`read`},{name:`gainKey`,view:P,type:`u32`,access:`read_write`},{name:`state`,view:H,type:`u32`,access:`read`}],invocationCount:y,declarations:J,body:`gainKey[gainKeyOffset + index] = 0u;
  let flags = arcFlags[arcFlagsOffset + index];
  if (!isInfoLiveFlag(flags) || (flags & 2u) == 0u ||
      state[stateOffset + STATE_REGION_COUNT] >= state[stateOffset + STATE_TARGET]) {
    return;
  }
  let child = neighbors[neighborsOffset + index];
  let childBase = arcStatsOffset + index * STRIDE;
  let count = arcStats[childBase];
  let base = regionStatsOffset + labels[labelsOffset + child] * STRIDE;
  let regionCount = regionStats[base];
  let remainder = regionCount - count;
  let minimum = f32(state[stateOffset + STATE_MINIMUM_SIZE]);
  if (count < minimum || remainder < minimum) {
    return;
  }
  // SSD(region) - SSD(side A) - SSD(side B) in its direct form, the between-side sum of squares
  // nA nB / (nA + nB) |mean A - mean B|^2 (the Ward increase). The three-term difference loses
  // the gain below the last place of the region's sum of squares; this has no such cancellation.
  var squaredDifference = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    let regionSum = regionStats[base + 2u + column];
    let childSum = arcStats[childBase + 2u + column];
    let difference = childSum / count - (regionSum - childSum) / remainder;
    squaredDifference += difference * difference;
  }
  // The division comes last so that small integer inputs give the exact gain.
  let gain = count * remainder * squaredDifference / regionCount;
  // A NaN or infinite gain (a non-finite attribute in the region, when no treeStatus stopped the
  // run) is not a candidate: as a key it would outrank every finite gain.
  if ((bitcast<u32>(gain) & 0x7fffffffu) >= 0x7f800000u) {
    return;
  }
  gainKey[gainKeyOffset + index] = bitcast<u32>(max(gain, 0.0)) + 1u;`}),s(e,{id:`${r}-reset-best`,operation:A,variant:`reset-best`,bindings:[{name:`state`,view:H,type:`u32`,access:`read_write`}],invocationCount:1,declarations:K,body:`state[stateOffset + STATE_BEST_KEY] = 0u;
  state[stateOffset + STATE_BEST_EDGE] = NONE;
  state[stateOffset + STATE_APPLIED] = 0u;
  state[stateOffset + STATE_STEP] = state[stateOffset + STATE_STEP] + 1u;`}),s(e,{id:`${r}-best-gain`,operation:A,variant:`best-gain`,...U.getGate(0),bindings:[{name:`gainKey`,view:P,type:`u32`,access:`read`},{name:`state`,view:H,type:`atomic<u32>`,access:`read_write`}],invocationCount:y,declarations:K,body:`let key = gainKey[gainKeyOffset + index];
  if (key != 0u) {
    atomicMax(&state[stateOffset + STATE_BEST_KEY], key);
  }`}),s(e,{id:`${r}-best-edge`,operation:A,variant:`best-edge`,...U.getGate(0),bindings:[{name:`gainKey`,view:P,type:`u32`,access:`read`},{name:`twin`,view:D,type:`u32`,access:`read`},{name:`state`,view:H,type:`atomic<u32>`,access:`read_write`}],invocationCount:y,declarations:K,body:`let key = gainKey[gainKeyOffset + index];
  if (key != 0u && key == atomicLoad(&state[stateOffset + STATE_BEST_KEY])) {
    atomicMin(&state[stateOffset + STATE_BEST_EDGE], min(index, twin[twinOffset + index]));
  }`}),s(e,{id:`${r}-apply`,operation:A,variant:`apply`,bindings:[{name:`neighbors`,view:u.neighbors,type:`u32`,access:`read`},{name:`twin`,view:D,type:`u32`,access:`read`},{name:`arcFlags`,view:O,type:`u32`,access:`read`},{name:`arcStats`,view:z,type:`f32`,access:`read`},{name:`labels`,view:m,type:`u32`,access:`read`},{name:`regionStats`,view:V,type:`f32`,access:`read_write`},{name:`state`,view:H,type:`u32`,access:`read_write`},U.binding],invocationCount:1,declarations:J,body:`let edge = state[stateOffset + STATE_BEST_EDGE];
  if (state[stateOffset + STATE_BEST_KEY] == 0u || edge == NONE) {
    ${U.disarmWGSL}
    return;
  }
  let other = twin[twinOffset + edge];
  let down = select(other, edge, (arcFlags[arcFlagsOffset + edge] & 2u) != 0u);
  let child = neighbors[neighborsOffset + down];
  let region = labels[labelsOffset + child];
  let regionBase = regionStatsOffset + region * STRIDE;
  let childBase = regionStatsOffset + child * STRIDE;
  let removedBase = arcStatsOffset + down * STRIDE;
  regionStats[childBase] = arcStats[removedBase];
  regionStats[childBase + 1u] = 0.0;
  regionStats[regionBase] = regionStats[regionBase] - arcStats[removedBase];
  for (var column = 0u; column < COLUMNS; column++) {
    let sum = arcStats[removedBase + 2u + column];
    regionStats[childBase + 2u + column] = sum;
    regionStats[regionBase + 2u + column] = regionStats[regionBase + 2u + column] - sum;
  }
  state[stateOffset + STATE_REGION_COUNT] = state[stateOffset + STATE_REGION_COUNT] + 1u;
  state[stateOffset + STATE_APPLIED] = down + 1u;
  state[stateOffset + STATE_OLD_REGION] = region;
  state[stateOffset + STATE_CHILD] = child;
  // The step index comes from the cursor, so every step shares one pipeline (rule 12).
  let logSlot = STATE_LOG + 2u * (state[stateOffset + STATE_STEP] - 1u);
  state[stateOffset + logSlot] = min(edge, other);
  state[stateOffset + logSlot + 1u] = state[stateOffset + STATE_BEST_KEY];`}),s(e,{id:`${r}-relabel`,operation:A,variant:`relabel`,...U.getGate(1),bindings:[{name:`rowDownArc`,view:B,type:`u32`,access:`read`},{name:`twin`,view:D,type:`u32`,access:`read`},{name:`tourDistance`,view:Q,type:`u32`,access:`read`},{name:`labels`,view:m,type:`u32`,access:`read_write`},{name:`state`,view:H,type:`u32`,access:`read`}],invocationCount:h,declarations:K,body:`let applied = state[stateOffset + STATE_APPLIED];
  if (applied == 0u || labels[labelsOffset + index] != state[stateOffset + STATE_OLD_REGION]) {
    return;
  }
  let down = applied - 1u;
  let child = state[stateOffset + STATE_CHILD];
  var isBelow = index == child;
  let parentArc = rowDownArc[rowDownArcOffset + index];
  if (parentArc != NONE) {
    let distance = tourDistance[tourDistanceOffset + parentArc];
    isBelow = isBelow || (distance < tourDistance[tourDistanceOffset + down] &&
      distance > tourDistance[tourDistanceOffset + twin[twinOffset + down]]);
  }
  if (isBelow) {
    labels[labelsOffset + index] = child;
  }`}),s(e,{id:`${r}-update-ancestors`,operation:A,variant:`update-ancestors`,...U.getGate(0),bindings:[{name:`neighbors`,view:u.neighbors,type:`u32`,access:`read`},{name:`twin`,view:D,type:`u32`,access:`read`},{name:`arcFlags`,view:O,type:`u32`,access:`read`},{name:`tourDistance`,view:Q,type:`u32`,access:`read`},{name:`arcStats`,view:z,type:`f32`,access:`read_write`},{name:`labels`,view:m,type:`u32`,access:`read`},{name:`state`,view:H,type:`u32`,access:`read`}],invocationCount:y,declarations:J,body:`let applied = state[stateOffset + STATE_APPLIED];
  let flags = arcFlags[arcFlagsOffset + index];
  if (applied == 0u || applied - 1u == index || !isInfoLiveFlag(flags) || (flags & 2u) == 0u) {
    return;
  }
  let down = applied - 1u;
  if (labels[labelsOffset + neighbors[neighborsOffset + index]] != state[stateOffset + STATE_OLD_REGION]) {
    return;
  }
  if (!(tourDistance[tourDistanceOffset + index] > tourDistance[tourDistanceOffset + down] &&
      tourDistance[tourDistanceOffset + twin[twinOffset + index]] < tourDistance[tourDistanceOffset + twin[twinOffset + down]])) {
    return;
  }
  let base = arcStatsOffset + index * STRIDE;
  let removedBase = arcStatsOffset + down * STRIDE;
  for (var word = 0u; word < STRIDE; word++) {
    arcStats[base + word] = arcStats[base + word] - arcStats[removedBase + word];
  }`}),s(e,{id:`${r}-deactivate`,operation:A,variant:`deactivate`,bindings:[{name:`twin`,view:D,type:`u32`,access:`read`},{name:`arcFlags`,view:O,type:`u32`,access:`read_write`},{name:`state`,view:H,type:`u32`,access:`read`},U.binding],invocationCount:1,declarations:K,body:`if (state[stateOffset + STATE_REGION_COUNT] >= state[stateOffset + STATE_TARGET]) {
    ${U.disarmWGSL}
  }
  let applied = state[stateOffset + STATE_APPLIED];
  if (applied == 0u) {
    return;
  }
  let down = applied - 1u;
  arcFlags[arcFlagsOffset + down] = arcFlags[arcFlagsOffset + down] & ~1u;
  let up = twin[twinOffset + down];
  arcFlags[arcFlagsOffset + up] = arcFlags[arcFlagsOffset + up] & ~1u;`}))}let $=[{name:`state`,view:H,type:`u32`,access:`read`},...f.regionCount?[{name:`regionCountOut`,view:f.regionCount,type:`u32`,access:`read_write`}]:[],...f.cutEdges?[{name:`cutEdgesOut`,view:f.cutEdges,type:`u32`,access:`read_write`}]:[],...f.cutGains?[{name:`cutGainsOut`,view:f.cutGains,type:`f32`,access:`read_write`}]:[]];if($.length>1&&Y.push(s(e,{id:`${t}-publish`,operation:A,variant:`publish`,bindings:$,invocationCount:f.cutEdges||f.cutGains?c.maximumRegionCount-1:1,declarations:K,body:`${f.regionCount?`if (index == 0u) { regionCountOut[regionCountOutOffset] = state[stateOffset + STATE_REGION_COUNT]; }`:``}
  if (index < STEP_COUNT) {
    let edge = state[stateOffset + STATE_LOG + 2u * index];
    let key = state[stateOffset + STATE_LOG + 2u * index + 1u];
    ${f.cutEdges?`cutEdgesOut[cutEdgesOutOffset + index] = edge;`:``}
    ${f.cutGains?`cutGainsOut[cutGainsOutOffset + index] = select(0.0, bitcast<f32>(key - 1u), key != 0u);`:``}
  } else {
    ${f.cutEdges?`cutEdgesOut[cutEdgesOutOffset + index] = NONE;`:``}
    ${f.cutGains?`cutGainsOut[cutGainsOutOffset + index] = 0.0;`:``}
  }`})),E.view)for(let[n,r,i,a,o]of[[`labels`,f.labels,`u32`,`0xffffffffu`,h],[`region-count`,f.regionCount,`u32`,`0u`,1],[`cut-edges`,f.cutEdges,`u32`,`0xffffffffu`,c.maximumRegionCount-1],[`cut-gains`,f.cutGains,`f32`,`bitcast<f32>(nanBits)`,c.maximumRegionCount-1]])r&&Y.push(s(e,{id:`${t}-incomplete-${n}`,operation:A,invocationCount:o,bindings:[{name:`loss`,view:E.view,type:`u32`,access:`read`},{name:`result`,view:r,type:i,access:`read_write`}],body:`if (loss[lossOffset] != 0u) {
                ${i===`f32`?`var nanBits = 0x7fc00000u;`:``}
                result[resultOffset + index] = ${a};
              }`}));return Y}},L=12,R=b({parameters:w,caption:`Contiguous regions from cutting the spanning tree (white), cut where the red edges fall.`,createInputs:({world:e})=>O(x(7),e,8,2),build(e,t){let{tracts:n,attributes:r,columnCount:i}=t,a=C(e,n.contiguity),o=e.upload(`attributes`,`float32`,r),s=e.allocate(`flags`,`uint32`,n.contiguity.neighbors.length),c=e.allocate(`components`,`uint32`,n.count),l=e.allocate(`standardized`,`float32`,r.length);e.graph.add(new y({weights:a,values:o.view,columnCount:i,standardize:!0,output:{treeEdgeFlags:s.view,componentLabels:c.view,standardizedValues:l.view}}));let u=e.allocate(`parameters`,`uint32`,2),d=e.allocate(`labels`,`uint32`,n.count),f=e.allocate(`cut-edges`,`uint32`,11);e.graph.add(new I({weights:a,treeEdgeFlags:s.view,componentLabels:c.view,values:l.view,columnCount:i,maximumRegionCount:L,parameters:u.view,output:{labels:d.view,regionCount:e.allocate(`region-count`,`uint32`,1).view,cutEdges:f.view,cutGains:e.allocate(`cut-gains`,`float32`,11).view}}));let p=new Float32Array(n.contiguity.neighbors.length*4);for(let e=0;e<n.count;e++)for(let t=n.contiguity.offsets[e];t<n.contiguity.offsets[e+1];t++){let r=n.contiguity.neighbors[t];p.set([n.centers[e*2],n.centers[e*2+1],n.centers[r*2],n.centers[r*2+1]],t*4)}let m=e.upload(`slot-segments`,`float32x4`,p);return{outputs:{labels:d,tree:k(e,s,m,`tree`),cuts:D(e,f,m,`cuts`),...T(e,n)},update:e=>u.write(P({regionCount:e.regionCount,minimumRegionSize:e.minimumRegionSize}))}},interaction:S(w,{kind:`pad`,hint:`cut the tree`,x:`regionCount`,y:`minimumRegionSize`}),draw:({inputs:e,outputs:t})=>[...E(e,t,t.labels),{layer:`lines`,segments:t.tree,count:e.tracts.contiguity.neighbors.length,radius:1.4,color:[1,1,1,.9]},{layer:`lines`,segments:t.cuts,radius:3.2,color:[.95,.2,.2,1]}]});export{R as default};