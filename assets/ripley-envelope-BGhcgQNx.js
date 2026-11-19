import{n as e,t}from"./gpu-ripley-Bagfeczg.js";import{r as n}from"./kernel-C3tfQOn4.js";import{v as r}from"./spatial-context-DwZEd4EZ.js";import{n as i,t as a}from"./philox-random-DfVxWNPn.js";import{o,t as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as c,f as l,x as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{a as f,c as p,l as m,t as h,u as g}from"./pair-statistics-grid-L8bKgCyF.js";import{t as _}from"./study-window-BK-B07m5.js";import{t as v}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as y}from"./exact-geometry-wgsl-y7TUnqOh.js";import{a as b,n as x,o as S,t as C}from"./study-window-wgsl-B3mOUanG.js";import{t as w}from"./define-run-B-brkCe4.js";import{a as T,c as E,d as D,f as O,i as k,l as A,o as j,s as M,t as N,u as P}from"./distance-probe-BxIfRi4a.js";import{c as F,v as I}from"./helpers-DPAd8TjN.js";import{_c as L}from"./entry-preview-D70TdSbK.js";var R=`GPURipleyEnvelope`,z=256,B={k:0,l:1,"l-minus-r":2},V=class{id;props;spatialContext;studyWindow;rank;statistic;constructor(e){this.id=e.id??`ripley-envelope`,this.props=e;let t=this.id;this.spatialContext=r(t,e.spatialContext),this.studyWindow=_(t,e.studyWindow,this.spatialContext),m(t,{...e,parameterLength:8});let{radiusCount:n,simulationCount:i,output:a}=e;l(t,`radiusCount`,n,1,z);let o=e.maximumAttempts??32;if(!Number.isInteger(o)||o<1||o>256)throw Error(`${t} maximumAttempts must be an integer in [1, 256]`);if(a.unplacedCount&&(c(a.unplacedCount,[`uint32`],`${t} output.unplacedCount`),a.unplacedCount.length<1))throw Error(`${t} output.unplacedCount must hold 1 row`);if(this.rank=e.rank??1,l(t,`rank`,this.rank),l(t,`simulationCount`,i,2*this.rank,199),this.statistic=e.statistic??`l-minus-r`,!Object.prototype.hasOwnProperty.call(B,this.statistic))throw Error(`${t} statistic must be 'k', 'l' or 'l-minus-r'`);let s=[[`observedK`,e.observedK],[`output.lower`,a.lower],[`output.upper`,a.upper],[`output.globalLower`,a.globalLower],[`output.globalUpper`,a.globalUpper],[`output.theoretical`,a.theoretical]];for(let[e,r]of s)if(r&&(c(r,[`float32`],`${t} ${e}`),r.length<n))throw Error(`${t} ${e} must hold radiusCount rows`);if(a.statistics&&(c(a.statistics,[`float32`],`${t} output.statistics`),a.statistics.length<6))throw Error(`${t} output.statistics must hold 6 rows`);if(!a.lower&&!a.upper&&!a.globalLower&&!a.globalUpper&&!a.theoretical&&!a.statistics&&!a.unplacedCount)throw Error(`${t} output needs at least one view`);d(t,[a.lower,a.upper,a.globalLower,a.globalUpper,a.theoretical,a.statistics,a.unplacedCount],[e.positions,e.parameters,e.mask,e.observedK])}getCommandNodes(e){let{id:r,props:c,rank:l,statistic:d}=this,{positions:m,parameters:_,gridSize:w,radiusCount:T,simulationCount:E,output:D}=c;g(r,e.device,m.length),u(r,e,[m,_,c.mask,c.observedK,D.lower,D.upper,D.globalLower,D.globalUpper,D.theoretical,D.statistics,D.unplacedCount]);let O=this.studyWindow===`rectangle`?void 0:this.studyWindow.getViews(e);O&&u(r,e,[O.edges,O.summary,O.slabs.params,O.slabs.slabOffsets,O.slabs.slabEntries,O.sampleCells,O.sampleCellCount,O.packed]);let k=O?p(e,{id:r,operation:R,parameters:_,parameterLength:_.length,window:O}):void 0,A=k?.parameters??_,j=O?S(O):void 0,M=j?`${v}
${y}
${b(j)}`:``,N=m.length,P=[...k?.nodes??[]],F=D.unplacedCount??(O?n(e,`${r}-unplaced-count`,`uint32`,1):void 0);F&&P.push(s(e,{id:`${r}-unplaced-clear`,operation:R,view:F,type:`u32`,value:`0u`}));let I=n(e,`${r}-count`,`uint32`,1);P.push(s(e,{id:`${r}-count-clear`,operation:R,view:I,type:`u32`,value:`0u`}),o(e,{id:`${r}-count`,operation:R,variant:`count`,bindings:[...O?[x(O)]:[],{name:`positions`,view:m,type:`f32`,access:`read`},{name:`parameters`,view:A,type:`f32`,access:`read`},...c.mask?[{name:`mask`,view:c.mask,type:`u32`,access:`read`}]:[],{name:`count`,view:I,type:`atomic<u32>`,access:`read_write`}],invocationCount:N,declarations:f(w)+M,body:`let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let included = ${c.mask?`mask[maskOffset + index] != 0u`:`true`};
  if (included && isPointValid(lattice, x, y)${O?` && isInsideStudyWindow(vec2f(x, y))`:``}) {
    atomicAdd(&count[countOffset], 1u);
  }`}));let L=n(e,`${r}-all-k`,`float32`,E*T),V=n(e,`${r}-simulation-index`,`uint32`,1);P.push(s(e,{id:`${r}-simulation-index-clear`,operation:R,view:V,type:`u32`,value:`0u`}));for(let s=0;s<E;s++){let l=n(e,`${r}-sim-${s}-positions`,`float32x2`,N),u=n(e,`${r}-sim-${s}-mask`,`uint32`,N),d=n(e,`${r}-sim-${s}-k`,`float32`,T);P.push(o(e,{id:`${r}-sim-${s}-points`,operation:R,variant:`simulate`,bindings:[...O?[x(O),{name:`unplacedCount`,view:F,type:`atomic<u32>`,access:`read_write`}]:[],{name:`parameters`,view:A,type:`u32`,access:`read`},{name:`count`,view:I,type:`u32`,access:`read`},{name:`simulationIndex`,view:V,type:`u32`,access:`read`},{name:`positions`,view:l,type:`f32`,access:`read_write`},{name:`mask`,view:u,type:`u32`,access:`read_write`}],invocationCount:N,declarations:`${i}${M}
const SEED_SLOT: u32 = 6u;
const PURPOSE: u32 = ${a.ripleyEnvelopeSimulation.id}u;`,body:O?`let cellCount = studyWindow[studyWindowOffset + ${j.layout.sampleCellCount}u];
  let usable = isStudyWindowUsable() && cellCount > 0u;
  let inside = index < count[countOffset] || !usable;
  var point = vec2f(0.0);
  var placed = false;
  if (inside && usable) {
    for (var attempt = 0u; attempt < ${c.maximumAttempts??32}u; attempt++) {
      let words = philox4x32(vec4<u32>(index, simulationIndex[simulationIndexOffset], attempt, 0u),
        vec2<u32>(parameters[parametersOffset + SEED_SLOT], PURPOSE));
      // Integer modulo: bias below cellCount / 2^32 (<= 2^-12), where a 24-bit float draw would
      // give cells unequal shares of up to 1 / 256 at 2^16 cells.
      let selected = words.x % cellCount;
      let cell = studyWindow[studyWindowOffset + ${j.layout.sampleCells}u + selected];
      let placement = getStudyWindowSampleCell(cell);
      point = placement.xy + vec2f(philoxUnitFloat(words.y), philoxUnitFloat(words.z)) * placement.zw;
      placed = (cell & ${C}u) == 0u;
      if (!placed) { placed = isInsideStudyWindow(point); }
      if (placed) { break; }
    }
  }
  if (inside && !placed) { atomicAdd(&unplacedCount[unplacedCountOffset], 1u); }
  positions[positionsOffset + index * 2u] = point.x;
  positions[positionsOffset + index * 2u + 1u] = point.y;
  mask[maskOffset + index] = select(0u, 1u, placed);`:`let inside = index < count[countOffset];
  var x = 0.0;
  var y = 0.0;
  if (inside) {
    let words = philox4x32(
      vec4<u32>(index, simulationIndex[simulationIndexOffset], 0u, 0u),
      vec2<u32>(parameters[parametersOffset + SEED_SLOT], PURPOSE)
    );
    let minimumX = bitcast<f32>(parameters[parametersOffset]);
    let minimumY = bitcast<f32>(parameters[parametersOffset + 1u]);
    let maximumX = bitcast<f32>(parameters[parametersOffset + 2u]);
    let maximumY = bitcast<f32>(parameters[parametersOffset + 3u]);
    x = minimumX + philoxUnitFloat(words.x) * (maximumX - minimumX);
    y = minimumY + philoxUnitFloat(words.y) * (maximumY - minimumY);
  }
  positions[positionsOffset + index * 2u] = x;
  positions[positionsOffset + index * 2u + 1u] = y;
  mask[maskOffset + index] = select(0u, 1u, inside);`}),...new t({id:`${r}-sim-${s}`,positions:l,spatialContext:this.spatialContext,studyWindow:this.studyWindow,parameters:A,gridSize:w,radiusCount:T,mask:u,output:{k:d}}).getCommandNodes(e),o(e,{id:`${r}-sim-${s}-gather`,operation:R,variant:`gather`,bindings:[{name:`simulatedK`,view:d,type:`f32`,access:`read`},{name:`simulationIndex`,view:V,type:`u32`,access:`read_write`},{name:`allK`,view:L,type:`f32`,access:`read_write`}],invocationCount:z,workgroupSize:z,guardIndex:!1,declarations:`const RADIUS_COUNT: u32 = ${T}u;
var<workgroup> sharedSimulationIndex: u32;`,body:`if (localInvocationIndex == 0u) {
    sharedSimulationIndex = simulationIndex[simulationIndexOffset];
  }
  let currentSimulation = workgroupUniformLoad(&sharedSimulationIndex);
  if (index < RADIUS_COUNT) {
    allK[allKOffset + currentSimulation * RADIUS_COUNT + index] = simulatedK[simulatedKOffset + index];
  }
  if (localInvocationIndex == 0u) {
    simulationIndex[simulationIndexOffset] = currentSimulation + 1u;
  }`}))}let H=`${h}
const RADIUS_COUNT: u32 = ${T}u;
const SIMULATION_COUNT: u32 = ${E}u;
const RANK: u32 = ${l}u;
const STATISTIC: u32 = ${B[d]}u;
const PI: f32 = 3.14159265358979;

fn getRadius(radiusIndex: u32) -> f32 {
  return parameters[parametersOffset + 4u] * f32(radiusIndex + 1u) / f32(RADIUS_COUNT);
}

fn getLDeviation(kValue: f32, radius: f32) -> f32 {
  return abs(sqrt(max(kValue, 0.0) / PI) - radius);
}

// Converts K to the reported statistic.
fn fromK(kValue: f32, radius: f32) -> f32 {
  if (STATISTIC == 0u) { return kValue; }
  let lValue = sqrt(max(kValue, 0.0) / PI);
  return select(lValue, lValue - radius, STATISTIC == 2u);
}

// Converts L to the reported statistic.
fn fromL(lValue: f32, radius: f32) -> f32 {
  if (STATISTIC == 0u) { return PI * max(lValue, 0.0) * max(lValue, 0.0); }
  return select(lValue, lValue - radius, STATISTIC == 2u);
}`,U=n(e,`${r}-deviations`,`float32`,E);P.push(o(e,{id:`${r}-deviations`,operation:R,variant:`deviations`,bindings:[{name:`parameters`,view:A,type:`f32`,access:`read`},{name:`allK`,view:L,type:`f32`,access:`read`},{name:`deviations`,view:U,type:`f32`,access:`read_write`}],invocationCount:E,declarations:H,body:`var deviation = 0.0;
  var anyFinite = false;
  for (var radiusIndex = 0u; radiusIndex < RADIUS_COUNT; radiusIndex++) {
    let kValue = allK[allKOffset + index * RADIUS_COUNT + radiusIndex];
    if (isFiniteValue(kValue)) {
      deviation = max(deviation, getLDeviation(kValue, getRadius(radiusIndex)));
      anyFinite = true;
    }
  }
  deviations[deviationsOffset + index] = select(getQuietNaN(index), deviation, anyFinite);`}));let W=D.statistics??n(e,`${r}-summary`,`float32`,6);P.push(o(e,{id:`${r}-summary`,operation:R,variant:`summary`,bindings:[{name:`parameters`,view:A,type:`f32`,access:`read`},{name:`deviations`,view:U,type:`f32`,access:`read`},{name:`count`,view:I,type:`u32`,access:`read`},...c.observedK?[{name:`observedK`,view:c.observedK,type:`f32`,access:`read`}]:[],{name:`summary`,view:W,type:`f32`,access:`read_write`}],invocationCount:1,declarations:H,body:`let nan = getQuietNaN(index);
  let n = count[countOffset];
  var validCount = 0u;
  for (var simulation = 0u; simulation < SIMULATION_COUNT; simulation++) {
    if (isFiniteValue(deviations[deviationsOffset + simulation])) { validCount++; }
  }
  // rank-th largest by (value, index): the total order breaks ties without double counting.
  var critical = nan;
  for (var candidate = 0u; candidate < SIMULATION_COUNT; candidate++) {
    let value = deviations[deviationsOffset + candidate];
    if (!isFiniteValue(value)) { continue; }
    var greater = 0u;
    for (var other = 0u; other < SIMULATION_COUNT; other++) {
      let otherValue = deviations[deviationsOffset + other];
      if (isFiniteValue(otherValue) && (otherValue > value || (otherValue == value && other < candidate))) {
        greater++;
      }
    }
    if (greater == RANK - 1u) { critical = value; }
  }
  var observed = nan;
  var pValue = nan;
  ${c.observedK?`var observedDeviation = 0.0;
  var observedFinite = false;
  for (var radiusIndex = 0u; radiusIndex < RADIUS_COUNT; radiusIndex++) {
    let kValue = observedK[observedKOffset + radiusIndex];
    if (isFiniteValue(kValue)) {
      observedDeviation = max(observedDeviation, getLDeviation(kValue, getRadius(radiusIndex)));
      observedFinite = true;
    }
  }
  if (observedFinite) {
    observed = observedDeviation;
    var atLeast = 0u;
    for (var simulation = 0u; simulation < SIMULATION_COUNT; simulation++) {
      let value = deviations[deviationsOffset + simulation];
      if (isFiniteValue(value) && value >= observedDeviation) { atLeast++; }
    }
    pValue = f32(1u + atLeast) / f32(validCount + 1u);
  }`:``}
  summary[summaryOffset] = f32(n);
  summary[summaryOffset + 1u] = f32(validCount);
  summary[summaryOffset + 2u] = observed;
  summary[summaryOffset + 3u] = select(nan, critical, validCount >= RANK);
  summary[summaryOffset + 4u] = pValue;
  summary[summaryOffset + 5u] = f32(RANK);`}));let G=[[`lower`,D.lower],[`upper`,D.upper],[`globalLower`,D.globalLower],[`globalUpper`,D.globalUpper],[`theoretical`,D.theoretical]];return G.some(([,e])=>e)&&P.push(o(e,{id:`${r}-envelope`,operation:R,variant:`envelope`,bindings:[{name:`parameters`,view:A,type:`f32`,access:`read`},{name:`allK`,view:L,type:`f32`,access:`read`},{name:`summary`,view:W,type:`f32`,access:`read`},...G.flatMap(([e,t])=>t?[{name:e,view:t,type:`f32`,access:`read_write`}]:[])],invocationCount:T,declarations:H,body:`let nan = getQuietNaN(index);
  let radius = getRadius(index);
  let critical = summary[summaryOffset + 3u];
  ${D.theoretical?`theoretical[theoreticalOffset + index] = fromL(radius, radius);`:``}
  ${D.globalLower?`globalLower[globalLowerOffset + index] = select(nan, fromL(radius - critical, radius), isFiniteValue(critical));`:``}
  ${D.globalUpper?`globalUpper[globalUpperOffset + index] = select(nan, fromL(radius + critical, radius), isFiniteValue(critical));`:``}
  ${D.lower||D.upper?`var validCount = 0u;
  for (var simulation = 0u; simulation < SIMULATION_COUNT; simulation++) {
    if (isFiniteValue(allK[allKOffset + simulation * RADIUS_COUNT + index])) { validCount++; }
  }
  var lowerValue = nan;
  var upperValue = nan;
  if (validCount >= 2u * RANK) {
    for (var candidate = 0u; candidate < SIMULATION_COUNT; candidate++) {
      let value = allK[allKOffset + candidate * RADIUS_COUNT + index];
      if (!isFiniteValue(value)) { continue; }
      // Position of this value in the ascending order of finite values, ties by simulation index.
      var below = 0u;
      for (var other = 0u; other < SIMULATION_COUNT; other++) {
        let otherValue = allK[allKOffset + other * RADIUS_COUNT + index];
        if (isFiniteValue(otherValue) && (otherValue < value || (otherValue == value && other < candidate))) {
          below++;
        }
      }
      if (below == RANK - 1u) { lowerValue = value; }
      if (below == validCount - RANK) { upperValue = value; }
    }
  }
  ${D.lower?`lower[lowerOffset + index] = select(nan, fromK(lowerValue, radius), isFiniteValue(lowerValue));`:``}
  ${D.upper?`upper[upperOffset + index] = select(nan, fromK(upperValue, radius), isFiniteValue(upperValue));`:``}`:``}`})),P}},H=400,U=24,W=N,G=[-.03,.1],K=.2,q=O(H,6),J=w({parameters:L,caption:`Observed L(r) - r against the band (grey) that random patterns reach; the white rule is zero.`,createInputs:({random:e,world:t})=>({source:F(e,H),windowWidth:t.width*.5,world:t}),build(n,r,i){let a=n.allocate(`positions`,`float32x2`,H),o=n.allocate(`parameters`,`float32`,8),s=n.allocate(`k`,`float32`,U),c=n.allocate(`observed`,`float32`,U);n.graph.add(new t({positions:a.view,parameters:o.view,gridSize:[32,32],radiusCount:U,output:{k:s.view,lMinusR:c.view}}));let l=n.allocate(`lower`,`float32`,U),u=n.allocate(`upper`,`float32`,U);return n.graph.add(new V({positions:a.view,parameters:o.view,gridSize:[32,32],radiusCount:U,simulationCount:i.simulationCount,rank:i.rank,statistic:`l-minus-r`,observedK:s.view,output:{lower:l.view,upper:u.view}})),{outputs:{positions:a,observed:c,lower:l,upper:u,highlight:n.allocate(`highlight`,`float32x2`,300),zero:n.upload(`zero`,`float32x4`,k(W,r.world,G,[0]))},update:t=>{a.write(I(r.source,r.windowWidth,1,t.clustering)),o.write(e({bounds:[0,0,r.windowWidth,1],maximumDistance:K,edgeCorrection:`isotropic`}))}}},rebuildOn:[`simulationCount`,`rank`],interaction:{kind:`reach`,hint:`hover a radius`,rest:[M(.3),.5],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=A(j(e.unitX),U),o=K*(a+1)/U,s=I(t.source,t.windowWidth,1,r.clustering),{glyphs:c,found:l}=T(s,q,o,n.highlight),u=i(n.observed)?.[a],d=i(n.lower)?.[a],f=i(n.upper)?.[a],p=u!==void 0&&d!==void 0&&f!==void 0,m=p?u>f?` · clustered`:u<d?` · regular`:` · random`:``;return{spotlight:null,focus:l,glyphs:[E(W,t.world,a,U),...c,...u===void 0?[]:[{shape:`dot`,at:[(W[0]+(W[2]-W[0])*(a+.5)/U)*t.world.width,P(W,t.world,G,u)]}]],readout:p?`r ${o.toFixed(2)} · L-r ${u.toFixed(3)} in [${d.toFixed(3)}, ${f.toFixed(3)}]${m}`:`r ${o.toFixed(2)}`}}},draw:({outputs:e,interaction:t,theme:n})=>[{layer:`dots`,context:!0,positions:e.positions,color:[.75,.8,.9,.8],radius:2.2},{layer:`dots`,positions:e.highlight,count:D(t.focus),radius:3.4,color:[1,.75,.2,1]},{layer:`chart`,kind:`bars`,values:e.upper,range:G,rectangle:W,color:[.55,.6,.7,.45]},{layer:`chart`,kind:`bars`,values:e.lower,range:G,rectangle:W,color:[n.background[0],n.background[1],n.background[2],1]},{layer:`lines`,context:!0,segments:e.zero,count:1,color:[1,1,1,.7]},{layer:`chart`,kind:`line`,values:e.observed,range:G,rectangle:W,radius:2.6,color:[.3,.75,1,1]}]});export{J as default};