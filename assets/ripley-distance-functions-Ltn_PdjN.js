import{r as e}from"./kernel-C3tfQOn4.js";import{o as t,t as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,f as i,x as a,y as o}from"./gpu-contributor-utils-BIwLQON5.js";import{i as s,n as c}from"./pair-histogram-COfc4qPq.js";import{a as l,c as u,i as d,l as f,t as p,u as m}from"./pair-statistics-grid-L8bKgCyF.js";import{t as h}from"./pair-statistics-parameters-BSUD2FEq.js";import{v as g}from"./spatial-context-DwZEd4EZ.js";import{t as _}from"./study-window-BK-B07m5.js";import{t as v}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as y}from"./exact-geometry-wgsl-y7TUnqOh.js";import{a as b,n as x,o as S}from"./study-window-wgsl-B3mOUanG.js";import{t as C}from"./define-run-B-brkCe4.js";import{a as w,c as T,d as E,f as D,i as O,l as k,o as A,s as j,t as M}from"./distance-probe-BxIfRi4a.js";import{c as N,v as P}from"./helpers-DPAd8TjN.js";import{vc as F}from"./entry-preview-D70TdSbK.js";var I={none:0,border:1,"kaplan-meier":2,hanisch:3};function L(e,t=new Float32Array(8)){if(t.length<8)throw Error(`GPURipleyDistanceFunctions target must hold 8 elements`);h(`GPURipleyDistanceFunctions`,e.bounds,e.maximumDistance);let n=e.edgeCorrection??`border`;if(!Object.prototype.hasOwnProperty.call(I,n))throw Error(`GPURipleyDistanceFunctions edgeCorrection must be 'none', 'border', 'kaplan-meier' or 'hanisch'`);return t.set([...e.bounds===`study-window`?[NaN,NaN,NaN,NaN]:e.bounds,e.maximumDistance,I[n],0,0]),t}var R=`GPURipleyDistanceFunctions`,z=256,B=2**24,V=8,H=64,U=class{id;props;spatialContext;studyWindow;constructor(e){this.id=e.id??`ripley-distance-functions`,this.props=e;let t=this.id;this.spatialContext=g(t,e.spatialContext),this.studyWindow=_(t,e.studyWindow,this.spatialContext),f(t,{...e,parameterLength:8});let{radiusCount:n,referenceGrid:a,output:c}=e;if(i(t,`radiusCount`,n,1,z),s(t,n,V),a.length!==2||!Number.isInteger(a[0])||!Number.isInteger(a[1])||a[0]<1||a[1]<1||a[0]*a[1]>B)throw Error(`${t} referenceGrid must be two positive integers with a product of at most ${B}`);if(!c.g&&!c.f&&!c.j)throw Error(`${t} output needs at least one of g, f or j`);for(let[e,i]of[[`output.g`,c.g],[`output.f`,c.f],[`output.j`,c.j],[`output.radii`,c.radii]])if(i&&(r(i,[`float32`],`${t} ${e}`),i.length<n))throw Error(`${t} ${e} must hold radiusCount rows`);o(t,[c.g,c.f,c.j,c.radii],[e.positions,e.parameters,e.mask])}getCommandNodes(r){let{id:i,props:o}=this,{positions:s,parameters:f,gridSize:h,radiusCount:g,referenceGrid:_,output:C}=o;m(i,r.device,s.length),a(i,r,[s,f,o.mask,C.g,C.f,C.j,C.radii]);let w=this.studyWindow===`rectangle`?void 0:this.studyWindow.getViews(r);w&&a(i,r,[w.edges,w.summary,w.slabs.params,w.slabs.slabOffsets,w.slabs.slabEntries,w.sampleCells,w.sampleCellCount,w.packed]);let T=w?u(r,{id:i,operation:R,parameters:f,parameterLength:f.length,window:w}):void 0,E=T?.parameters??f,D=w?S(w):void 0,O=D?`${v}
${y}
${b(D)}`:``,k=_[0]*_[1],A=d(r,{id:i,operation:R,positions:s,parameters:E,gridSize:h,mask:o.mask,studyWindow:w}),j=[...T?.nodes??[],...A.nodes],M=`
const RADIUS_COUNT: u32 = ${g}u;
const PI: f32 = 3.14159265358979;
`,N=w?e(r,`${i}-hanisch-counts`,`uint32`,4*g+1):void 0,P=w?[x(w),{name:`erosionCounts`,view:N,type:`u32`,access:`read`}]:[];if(w){let a=e(r,`${i}-hanisch-histogram`,`uint32`,4*g+1);j.push(n(r,{id:`${i}-hanisch-clear`,operation:R,view:a,type:`u32`,value:`0u`}),t(r,{id:`${i}-hanisch-reference`,operation:R,variant:`hanisch-reference`,bindings:[x(w),{name:`parameters`,view:E,type:`f32`,access:`read`},{name:`borderHistogram`,view:a,type:`atomic<u32>`,access:`read_write`}],invocationCount:k,declarations:`${l(h)}${O}`,body:`let lattice = readLattice();
  if (!lattice.valid || u32(parameters[parametersOffset + 5u] + 0.5) != 3u) { return; }
  let point = vec2f(lattice.minimumX, lattice.minimumY) +
    (vec2f(f32(index % ${_[0]}u), f32(index / ${_[0]}u)) + vec2f(0.5)) *
    vec2f(lattice.maximumX - lattice.minimumX, lattice.maximumY - lattice.minimumY) /
    vec2f(${_[0]}.0, ${_[1]}.0);
  if (isInsideStudyWindow(point)) {
    let distance = min(getStudyWindowBoundaryDistance(point, lattice.maximumDistance), lattice.maximumDistance);
    let slot = min(u32(floor(distance / lattice.maximumDistance * ${4*g}.0)), ${4*g}u);
    atomicAdd(&borderHistogram[borderHistogramOffset + slot], 1u);
  }`}),t(r,{id:`${i}-hanisch-suffix`,operation:R,variant:`hanisch-suffix`,bindings:[{name:`borderHistogram`,view:a,type:`u32`,access:`read`},{name:`erosionCounts`,view:N,type:`u32`,access:`read_write`}],invocationCount:4*g+1,body:`var total = 0u;
  for (var slot = index; slot <= ${4*g}u; slot++) { total += borderHistogram[borderHistogramOffset + slot]; }
  erosionCounts[erosionCountsOffset + index] = total;`}))}let F=`${M}${O}
const HANISCH_SCALE: f32 = 16384.0;
const HANISCH_CAP: f32 = 1024.0;

// Fixed-point Hanisch weight |W| / |W eroded by distance| of one observation; at most 2^24.
fn getHanischAmount(lattice: Lattice, distance: f32) -> u32 {
  ${w?`let scaled = clamp(distance / lattice.maximumDistance * ${4*g}.0, 0.0, ${4*g}.0);
  let lower = u32(floor(scaled));
  let upper = min(lower + 1u, ${4*g}u);
  let remaining = mix(f32(erosionCounts[erosionCountsOffset + lower]), f32(erosionCounts[erosionCountsOffset + upper]), scaled - f32(lower));
  let total = f32(erosionCounts[erosionCountsOffset]);
  // A cancels in |W| / (A * remaining / total).
  let ratio = select(HANISCH_CAP, min(total / remaining, HANISCH_CAP), remaining > 0.0);`:`let width = lattice.maximumX - lattice.minimumX;
  let height = lattice.maximumY - lattice.minimumY;
  let eroded = max(width - 2.0 * distance, 0.0) * max(height - 2.0 * distance, 0.0);
  let ratio = select(HANISCH_CAP, min(width * height / eroded, HANISCH_CAP), eroded > 0.0);`}
  return u32(ratio * HANISCH_SCALE + 0.5);
}

// Slot of a censoring distance; slots at or above RADIUS_COUNT lie beyond every radius.
fn getCensorSlot(distance: f32, step: f32) -> u32 {
  return min(u32(max(ceil(distance / step) - 1.0, 0.0)), RADIUS_COUNT);
}
`,I=c(r,{id:`${i}-g`,operation:R,positions:s,parameters:E,gridSize:h,sortedRows:A.sortedRows,cellOffsets:A.cellOffsets,slotCount:g,channelCount:V,pairOrder:`ordered`,extraBindings:P,declarations:F,focusPrologue:`var nearestDistance = lattice.maximumDistance;
    var hasNeighbor = false;
    let borderDistance = ${w?`min(getStudyWindowBoundaryDistance(vec2f(x, y), lattice.maximumDistance), lattice.maximumDistance)`:`min(min(x - lattice.minimumX, lattice.maximumX - x), min(y - lattice.minimumY, lattice.maximumY - y))`};
    let borderCount = min(u32(floor(borderDistance / lattice.maximumDistance * f32(RADIUS_COUNT))), RADIUS_COUNT);`,pairAction:`if (!hasNeighbor || pairDistance < nearestDistance) {
              nearestDistance = pairDistance;
            }
            hasNeighbor = true;`,focusEpilogue:`accumulate(borderCount, 3u, 1u);
    var annulus = 0u;
    if (hasNeighbor) {
      let scaled = nearestDistance / lattice.maximumDistance * f32(RADIUS_COUNT);
      annulus = min(u32(max(ceil(scaled) - 1.0, 0.0)), RADIUS_COUNT - 1u);
      accumulate(annulus, 0u, 1u);
      if (annulus < borderCount) {
        accumulate(annulus, 1u, 1u);
        accumulate(borderCount, 2u, 1u);
      }
    }
    // Kaplan-Meier and Hanisch: observed when the neighbor is no farther than the border.
    if (hasNeighbor && nearestDistance <= borderDistance) {
      accumulate(annulus, 4u, 1u);
      accumulate(annulus, 6u, getHanischAmount(lattice, nearestDistance));
    } else {
      accumulate(getCensorSlot(borderDistance, lattice.maximumDistance / f32(RADIUS_COUNT)), 5u, 1u);
      if (!hasNeighbor && borderDistance >= lattice.maximumDistance) {
        accumulate(0u, 7u, getHanischAmount(lattice, lattice.maximumDistance));
      }
    }`});j.push(...I.nodes);let L=e(r,`${i}-f-accumulators`,`uint32`,(g+1)*V*2);j.push(n(r,{id:`${i}-f-clear`,operation:R,view:L,type:`u32`,value:`0u`}),t(r,{id:`${i}-f-reference`,operation:R,variant:`f-reference`,bindings:[...P,{name:`parameters`,view:E,type:`f32`,access:`read`},{name:`sortedPoints`,view:I.sortedPoints,type:`u32`,access:`read`},{name:`cellOffsets`,view:A.cellOffsets,type:`u32`,access:`read`},{name:`referenceAccumulators`,view:L,type:`atomic<u32>`,access:`read_write`}],invocationCount:k,workgroupSize:H,guardIndex:!1,declarations:`${l(h)}
${F}
const REFERENCE_COLUMNS: u32 = ${_[0]}u;
const REFERENCE_ROWS: u32 = ${_[1]}u;
const CHANNELS: u32 = ${V}u;

const ACCUMULATOR_COUNT: u32 = ${(g+1)*V}u;
var<workgroup> localReferences: array<atomic<u32>, ${(g+1)*V}>;

// Workgroup partial sums (they fit 32 bits, see REFERENCE_WORKGROUP_SIZE).
fn addReference(slot: u32, channel: u32, amount: u32) {
  atomicAdd(&localReferences[slot * CHANNELS + channel], amount);
}`,body:`for (var word = localInvocationIndex; word < ACCUMULATOR_COUNT; word += ${H}u) {
    atomicStore(&localReferences[word], 0u);
  }
  workgroupBarrier();
  let lattice = readLattice();
  if (index < INVOCATION_COUNT && lattice.valid) {
    let referenceColumn = index % REFERENCE_COLUMNS;
    let referenceRow = index / REFERENCE_COLUMNS;
    let x = lattice.minimumX + (f32(referenceColumn) + 0.5) * (lattice.maximumX - lattice.minimumX) / f32(REFERENCE_COLUMNS);
    let y = lattice.minimumY + (f32(referenceRow) + 0.5) * (lattice.maximumY - lattice.minimumY) / f32(REFERENCE_ROWS);
    ${w?`if (isInsideStudyWindow(vec2f(x, y))) {
    `:``}let borderDistance = ${w?`min(getStudyWindowBoundaryDistance(vec2f(x, y), lattice.maximumDistance), lattice.maximumDistance)`:`min(min(x - lattice.minimumX, lattice.maximumX - x), min(y - lattice.minimumY, lattice.maximumY - y))`};
    let borderCount = min(u32(floor(borderDistance / lattice.maximumDistance * f32(RADIUS_COUNT))), RADIUS_COUNT);
    var nearestSquared = lattice.radiusSquared;
    var hasEvent = false;
    let column = getCellColumn(lattice, x);
    let row = getCellRow(lattice, y);
    let firstColumn = max(column, 1u) - 1u;
    let lastColumn = min(column + 1u, lattice.columns - 1u);
    let firstRow = max(row, 1u) - 1u;
    let lastRow = min(row + 1u, lattice.rows - 1u);
    for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
      let rowBase = cellRow * lattice.columns;
      let begin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
      let end = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
      for (var candidateSlot = begin; candidateSlot < end; candidateSlot++) {
        let deltaX = bitcast<f32>(sortedPoints[sortedPointsOffset + candidateSlot * 3u]) - x;
        let deltaY = bitcast<f32>(sortedPoints[sortedPointsOffset + candidateSlot * 3u + 1u]) - y;
        let distanceSquared = deltaX * deltaX + deltaY * deltaY;
        if (distanceSquared <= lattice.radiusSquared && (!hasEvent || distanceSquared < nearestSquared)) {
          nearestSquared = distanceSquared;
          hasEvent = true;
        }
      }
    }
    addReference(borderCount, 3u, 1u);
    let nearestDistance = sqrt(nearestSquared);
    var annulus = 0u;
    if (hasEvent) {
      let scaled = nearestDistance / lattice.maximumDistance * f32(RADIUS_COUNT);
      annulus = min(u32(max(ceil(scaled) - 1.0, 0.0)), RADIUS_COUNT - 1u);
      addReference(annulus, 0u, 1u);
      if (annulus < borderCount) {
        addReference(annulus, 1u, 1u);
        addReference(borderCount, 2u, 1u);
      }
    }
    if (hasEvent && nearestDistance <= borderDistance) {
      addReference(annulus, 4u, 1u);
      addReference(annulus, 6u, getHanischAmount(lattice, nearestDistance));
    } else {
      addReference(getCensorSlot(borderDistance, lattice.maximumDistance / f32(RADIUS_COUNT)), 5u, 1u);
      if (!hasEvent && borderDistance >= lattice.maximumDistance) {
        addReference(0u, 7u, getHanischAmount(lattice, lattice.maximumDistance));
      }
    }
${w?`  }
`:``}  }
  workgroupBarrier();
  // One global 64-bit add per nonzero accumulator and workgroup: exact, so order-independent.
  for (var word = localInvocationIndex; word < ACCUMULATOR_COUNT; word += ${H}u) {
    let amount = atomicLoad(&localReferences[word]);
    if (amount != 0u) {
      let globalWord = referenceAccumulatorsOffset + word * 2u;
      let previous = atomicAdd(&referenceAccumulators[globalWord], amount);
      if (previous > 0xffffffffu - amount) {
        atomicAdd(&referenceAccumulators[globalWord + 1u], 1u);
      }
    }
  }`}));let z=[{name:`accumulators`,view:I.accumulators,type:`u32`,access:`read`},{name:`referenceAccumulators`,view:L,type:`u32`,access:`read`},{name:`cellOffsets`,view:A.cellOffsets,type:`u32`,access:`read`},{name:`parameters`,view:E,type:`f32`,access:`read`}];for(let[e,t]of[[`g`,C.g],[`f`,C.f],[`j`,C.j],[`radii`,C.radii]])t&&z.push({name:e,view:t,type:`f32`,access:`read_write`});return j.push(t(r,{id:`${i}-finish`,operation:R,variant:`finish`,bindings:z,invocationCount:g,declarations:`${p}
${M}
const CELL_COUNT: u32 = ${h[0]*h[1]}u;
${w?``:`const REFERENCE_COUNT: f32 = ${k}.0;`}
const CHANNELS: u32 = ${V}u;

// Event counts stay below 2^24, so the low word of each 64-bit accumulator is exact.
fn readEvents(slot: u32, channel: u32) -> u32 {
  return accumulators[accumulatorsOffset + (slot * CHANNELS + channel) * 2u];
}

fn readReferences(slot: u32, channel: u32) -> u32 {
  return referenceAccumulators[referenceAccumulatorsOffset + (slot * CHANNELS + channel) * 2u];
}

// Full 64-bit sum as f32, for the fixed-point Hanisch weights.
fn readWeight(reference: bool, slot: u32, channel: u32) -> f32 {
  let word = (slot * CHANNELS + channel) * 2u;
  if (reference) {
    return f32(referenceAccumulators[referenceAccumulatorsOffset + word + 1u]) * 4294967296.0 +
      f32(referenceAccumulators[referenceAccumulatorsOffset + word]);
  }
  return f32(accumulators[accumulatorsOffset + word + 1u]) * 4294967296.0 +
    f32(accumulators[accumulatorsOffset + word]);
}

fn readObserved(reference: bool, slot: u32, channel: u32) -> f32 {
  return select(f32(readEvents(slot, channel)), f32(readReferences(slot, channel)), reference);
}

// Return CDF and survival together. Preserve exact empty/full probabilities even on devices
// whose floating-point division uses an approximate reciprocal. J needs the surviving mass,
// not a subtraction from an already rounded CDF.
fn getDistribution(observed: f32, remaining: f32, slot: u32) -> vec2f {
  let total = observed + remaining;
  if (!(total > 0.0)) { return vec2f(getQuietNaN(slot)); }
  if (observed == 0.0) { return vec2f(0.0, 1.0); }
  if (remaining == 0.0) { return vec2f(1.0, 0.0); }
  return vec2f(observed / total, remaining / total);
}

// Kaplan-Meier estimate of the distribution function at radius slot last, with hazards
// binned on the radius grid (events before censoring within a slot), as spatstat does.
fn getKaplanMeier(reference: bool, total: f32, last: u32) -> vec2f {
  var survival = 1.0;
  var atRisk = total;
  for (var slot = 0u; slot <= last; slot++) {
    let observed = readObserved(reference, slot, 4u);
    if (atRisk >= 1.0 && observed > 0.0) {
      survival = survival * getDistribution(observed, atRisk - observed, last).y;
    }
    atRisk = atRisk - observed - readObserved(reference, slot, 5u);
  }
  return vec2f(1.0 - survival, survival);
}

// Hanisch estimate: weighted observed fraction up to last over the total observed weight.
fn getHanisch(reference: bool, last: u32) -> vec2f {
  var observed = 0.0;
  var remaining = readWeight(reference, 0u, 7u);
  for (var slot = 0u; slot < RADIUS_COUNT; slot++) {
    let weight = readWeight(reference, slot, 6u);
    if (slot <= last) {
      observed = observed + weight;
    } else {
      remaining = remaining + weight;
    }
  }
  return getDistribution(observed, remaining, last);
}

fn getEventPrefix(channel: u32, last: u32) -> u32 {
  var sum = 0u;
  for (var slot = 0u; slot <= last; slot++) {
    sum += readEvents(slot, channel);
  }
  return sum;
}

fn getReferencePrefix(channel: u32, last: u32) -> u32 {
  var sum = 0u;
  for (var slot = 0u; slot <= last; slot++) {
    sum += readReferences(slot, channel);
  }
  return sum;
}`,body:`${w?`let REFERENCE_COUNT = f32(getReferencePrefix(3u, RADIUS_COUNT));
  `:``}let nan = getQuietNaN(index);
  let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let mode = u32(parameters[parametersOffset + 5u] + 0.5);
  let border = mode == 1u;
  let maximumDistance = parameters[parametersOffset + 4u];
  let valid = isFiniteValue(parameters[parametersOffset]) && isFiniteValue(maximumDistance) && maximumDistance > 0.0;
  var gDistribution = vec2f(nan);
  var fDistribution = vec2f(nan);
  if (valid && n >= 2.0) {
    if (border) {
      let denominator = n - f32(getEventPrefix(3u, index));
      if (denominator >= 1.0) {
        let observed = f32(getEventPrefix(1u, index) - getEventPrefix(2u, index));
        gDistribution = getDistribution(observed, denominator - observed, index);
      }
    } else {
      let observed = f32(getEventPrefix(0u, index));
      gDistribution = getDistribution(observed, n - observed, index);
    }
    if (mode == 2u) {
      gDistribution = getKaplanMeier(false, n, index);
    } else if (mode == 3u) {
      gDistribution = getHanisch(false, index);
    }
  }
  if (valid && n >= 1.0${w?` && REFERENCE_COUNT > 0.0`:``}) {
    if (border) {
      let denominator = REFERENCE_COUNT - f32(getReferencePrefix(3u, index));
      if (denominator >= 1.0) {
        let observed = f32(getReferencePrefix(1u, index) - getReferencePrefix(2u, index));
        fDistribution = getDistribution(observed, denominator - observed, index);
      }
    } else {
      let observed = f32(getReferencePrefix(0u, index));
      fDistribution = getDistribution(observed, REFERENCE_COUNT - observed, index);
    }
    if (mode == 2u) {
      fDistribution = getKaplanMeier(true, REFERENCE_COUNT, index);
    } else if (mode == 3u) {
      fDistribution = getHanisch(true, index);
    }
  }
  ${C.g?`g[gOffset + index] = gDistribution.x;`:``}
  ${C.f?`f[fOffset + index] = fDistribution.x;`:``}
  ${C.j?`j[jOffset + index] = nan;
  if (isFiniteValue(gDistribution.y) && isFiniteValue(fDistribution.y) && fDistribution.y > 0.0) {
    j[jOffset + index] = gDistribution.y / fDistribution.y;
  }`:``}
  ${C.radii?`radii[radiiOffset + index] = maximumDistance * f32(index + 1u) / f32(RADIUS_COUNT);`:``}`})),j}},W=500,G=M,K=D(W,6),q=.3,J=[`none`,`border`,`kaplan-meier`,`hanisch`],Y=C({parameters:F,caption:`G (cyan), F (orange) and J (white) against distance; the faint rules are the random-pattern G = F curve and J = 1.`,createInputs:({random:e,world:t})=>({source:N(e,W),windowWidth:t.width*.5,world:t,nearest:{share:-1,distances:new Float32Array(W)}}),build(e,t,n){let r=e.allocate(`positions`,`float32x2`,W),i=e.allocate(`parameters`,`float32`,8),a=e.allocate(`g`,`float32`,n.radiusCount),o=e.allocate(`f`,`float32`,n.radiusCount),s=e.allocate(`j`,`float32`,n.radiusCount);e.graph.add(new U({positions:r.view,parameters:i.view,gridSize:[32,32],referenceGrid:[32,32],radiusCount:n.radiusCount,output:{g:a.view,f:o.view,j:s.view}}));let c=e.allocate(`highlight`,`float32x2`,300),l=e.allocate(`references`,`float32`,n.radiusCount);return{outputs:{positions:r,g:a,f:o,j:s,highlight:c,references:l,jReference:e.upload(`j-reference`,`float32x4`,O(G,t.world,[0,2],[1]))},update:e=>{let a=W/t.windowWidth;l.write(Float32Array.from({length:n.radiusCount},(t,r)=>{let i=e.maximumDistance*(r+1)/n.radiusCount;return 1-Math.exp(-a*Math.PI*i*i)})),r.write(P(t.source,t.windowWidth,1,e.clustering)),i.write(L({bounds:[0,0,t.windowWidth,1],maximumDistance:e.maximumDistance,edgeCorrection:J[e.edgeCorrection]}))}}},rebuildOn:[`radiusCount`],interaction:{kind:`reach`,hint:`hover a radius`,rest:[j(q),.5],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=r.radiusCount,o=k(A(e.unitX),a),s=r.maximumDistance*(o+1)/a,c=P(t.source,t.windowWidth,1,r.clustering);if(t.nearest.share!==r.clustering){t.nearest.share=r.clustering;for(let e=0;e<W;e++){let n=1/0;for(let t=0;t<W;t++)t!==e&&(n=Math.min(n,Math.hypot(c[e*2]-c[t*2],c[e*2+1]-c[t*2+1])));t.nearest.distances[e]=n}}let l=w(c,K,s,n.highlight).glyphs,u=new Float32Array(600).fill(-10),d=0;for(let e=0;e<W&&d<300;e++)t.nearest.distances[e]>s||(u.set([c[e*2],c[e*2+1]],d*2),d++);n.highlight.write(u);let f=e=>{let t=i(e)?.[o];return t===void 0?`…`:t.toFixed(2)};return{spotlight:null,focus:d,glyphs:[T(G,t.world,o,a),...l],readout:`r ${s.toFixed(2)} · G ${f(n.g)} · F ${f(n.f)} · J ${f(n.j)}`}}},draw:({outputs:e,interaction:t})=>[{layer:`dots`,context:!0,positions:e.positions,color:[.75,.8,.9,.8],radius:2.2},{layer:`dots`,positions:e.highlight,count:E(t.focus),radius:3.4,color:[1,.75,.2,1]},{layer:`chart`,kind:`line`,context:!0,values:e.references,range:[0,1],rectangle:G,radius:1.4,color:[1,1,1,.5]},{layer:`lines`,context:!0,segments:e.jReference,count:1,color:[1,1,1,.4]},{layer:`chart`,kind:`line`,values:e.g,range:[0,1],rectangle:G,radius:2.6,color:[.3,.75,1,1]},{layer:`chart`,kind:`line`,values:e.f,range:[0,1],rectangle:G,radius:2.6,color:[1,.6,.2,1]},{layer:`chart`,kind:`line`,values:e.j,range:[0,2],rectangle:G,radius:1.4,color:[1,1,1,.8]}]});export{Y as default};