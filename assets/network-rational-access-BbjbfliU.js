import{n as e,t}from"./network-optimization-bindings-B3Zg-1cS.js";import{n,r,t as i}from"./network-optimization-status-D0dFmcvb.js";import{u as a}from"./wgsl-common-DfyXiYB0.js";import{o}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as s,T as c,f as l}from"./gpu-contributor-utils-BIwLQON5.js";import{t as u}from"./define-run-B-brkCe4.js";import{a as d,c as f,i as p,l as m,n as h,s as g}from"./access-Co69Wfld.js";import{p as _,v}from"./roads-DzovamHj.js";import{ha as y}from"./entry-preview-D70TdSbK.js";function b(e){return{declarations:`const ORIGIN_COUNT: u32 = ${e.demand.length}u;
const FACILITY_COUNT: u32 = ${e.supply.length}u;
${a}
var<workgroup> capacities: array<f32, ${e.supply.length}>;
var<workgroup> loads: array<f32, ${e.supply.length}>;
var<workgroup> incoming: array<f32, ${e.supply.length}>;
var<workgroup> minima: array<u32, ${e.demand.length}>;
var<workgroup> maxima: array<u32, ${e.demand.length}>;
var<workgroup> shifts: array<f32, ${e.demand.length}>;
var<workgroup> gaps: array<f32, ${e.demand.length}>;
var<workgroup> nextScores: array<f32, ${e.demand.length}>;
// 0: no score (unserved/zero demand); 1: score ready; 2: nonfinite arithmetic.
var<workgroup> scoreStates: array<u32, ${e.demand.length}>;
var<workgroup> continueCycles: u32;
fn roundEven(value: f32) -> f32 {
  let lower = floor(value); let fraction = value - lower;
  return lower + select(0.0, 1.0, fraction > 0.5 || (fraction == 0.5 && u32(lower) % 2u == 1u));
}`,body:`let origin = localInvocationIndex;
let travelScale = parameters[parametersOffset];
let supplyScale = parameters[parametersOffset + 1u];
let initialStep = parameters[parametersOffset + 2u];
let minimumStep = parameters[parametersOffset + 3u];
let halfLife = parameters[parametersOffset + 4u];
let tolerance = parameters[parametersOffset + 5u];
if (localInvocationIndex == 0u) { continueCycles = summary[summaryOffset]; }
if (workgroupUniformLoad(&continueCycles) == 0u) { return; }
if (localInvocationIndex < FACILITY_COUNT) {
  capacities[localInvocationIndex] = supply[supplyOffset + localInvocationIndex] * supplyScale;
}
workgroupBarrier();
for (var cycle = 0u; cycle <= ${e.maximumCycles??150}u; cycle++) {
  // Each facility sums in ascending origin order, exactly as before. There are no float atomics.
  if (localInvocationIndex < FACILITY_COUNT) {
    var load = 0.0;
    for (var row = 0u; row < ORIGIN_COUNT; row++) {
      load += assignments[assignmentsOffset + row * FACILITY_COUNT + localInvocationIndex];
    }
    loads[localInvocationIndex] = load;
  }
  workgroupBarrier();
  if (origin < ORIGIN_COUNT) {
    shifts[origin] = 0.0; gaps[origin] = 0.0; scoreStates[origin] = 0u;
    minima[origin] = 0u; maxima[origin] = 0u;
    var infinityBits = 0x7f800000u; let infinity = bitcast<f32>(infinityBits);
    var lowest = infinity; var highest = -1.0;
    var minimum = 0u; var maximum = 0u; var weightedCost = 0.0;
    for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
      let travel = costs[costsOffset + origin * FACILITY_COUNT + facility];
      if (!isFiniteValue(travel)) { continue; }
      let cost = travel / travelScale + loads[facility] / capacities[facility];
      if (!isFiniteValue(cost)) { scoreStates[origin] = 2u; break; }
      let amount = assignments[assignmentsOffset + origin * FACILITY_COUNT + facility];
      if (cost < lowest) { lowest = cost; minimum = facility; }
      if (amount > 0.0 && cost > highest) { highest = cost; maximum = facility; }
      weightedCost += cost * amount;
    }
    let amount = demand[demandOffset + origin];
    if (scoreStates[origin] != 2u && highest >= 0.0 && amount != 0.0) {
      if (!isFiniteValue(weightedCost)) { scoreStates[origin] = 2u; }
      else {
        nextScores[origin] = weightedCost / amount; scoreStates[origin] = 1u;
        gaps[origin] = highest - lowest;
        minima[origin] = minimum; maxima[origin] = maximum;
        if (minimum != maximum) {
          let held = assignments[assignmentsOffset + origin * FACILITY_COUNT + maximum];
          let equalizingShift = (highest - lowest) / (1.0 / capacities[minimum] + 1.0 / capacities[maximum]);
          let step = max(minimumStep, initialStep * exp2(-f32(cycle) / halfLife));
          let boundedShift = max(0.0, min(held, min(equalizingShift, ${e.stepMode===`absolute`?`step`:`step * amount`})));
          shifts[origin] = ${e.demandMode===`fractional`?`boundedShift`:`floor(boundedShift)`};
        }
      }
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    var maximumGap = 0.0;
    // Preserve partial scores through the first invalid origin, including on later cycles.
    for (var row = 0u; row < ORIGIN_COUNT; row++) {
      if (scoreStates[row] == 2u) { summary[summaryOffset] = 0u; continueCycles = 0u; break; }
      if (scoreStates[row] == 1u) { scores[scoresOffset + row] = nextScores[row]; }
      maximumGap = max(maximumGap, gaps[row]);
    }
    if (continueCycles != 0u) {
      summary[summaryOffset + 3u] = bitcast<u32>(maximumGap);
      if (maximumGap <= tolerance) { summary[summaryOffset + 1u] = 1u; continueCycles = 0u; }
      if (cycle == ${e.maximumCycles??150}u) { continueCycles = 0u; }
    }
  }
  if (workgroupUniformLoad(&continueCycles) == 0u) { break; }
  // Facility inflow uses the same ascending-origin sum as the old simultaneous update.
  if (localInvocationIndex < FACILITY_COUNT) {
    var total = 0.0;
    for (var row = 0u; row < ORIGIN_COUNT; row++) {
      if (minima[row] == localInvocationIndex) { total += shifts[row]; }
    }
    incoming[localInvocationIndex] = total;
  }
  workgroupBarrier();
  if (origin < ORIGIN_COUNT) {
    var shift = shifts[origin];
    if (shift != 0.0) {
      if (cycle < ${e.initialLimitCycles??20}u) {
        let limited = shift / max(1.0, incoming[minima[origin]] / capacities[minima[origin]]);
        shift = ${e.demandMode===`fractional`?`limited`:`roundEven(limited)`};
      }
      assignments[assignmentsOffset + origin * FACILITY_COUNT + minima[origin]] += shift;
      assignments[assignmentsOffset + origin * FACILITY_COUNT + maxima[origin]] -= shift;
    }
  }
  // Every origin finishes its stores before any facility reads the next cycle's allocation.
  storageBarrier();
  if (localInvocationIndex == 0u) { summary[summaryOffset + 2u] = cycle + 1u; }
}`}}var x=64,S=32,C=4096,w=150,T=20,E=6,D=4,O=class{id;props;constructor(t){this.id=t.id??`network-rational-access`,this.props=t;let{id:n}=this;l(n,`demand rows`,t.demand.length,1,x),l(n,`supply rows`,t.supply.length,1,S),l(n,`maximumCycles`,t.maximumCycles??w,0,C),l(n,`initialLimitCycles`,t.initialLimitCycles??T,0,C);for(let e of[t.costs,t.demand,t.supply,t.parameters,t.output.assignments,t.output.scores,t.output.unservedDemand])s(e,[`float32`],this.id);s(t.output.summary,[`uint32`],this.id),c(n,t.costs.length===t.demand.length*t.supply.length,`costs must hold demand rows times supply rows`),c(n,t.parameters.length===E,`parameters must hold ${E} rows`),c(n,t.output.assignments.length===t.costs.length,`output.assignments must hold as many rows as costs`),c(n,t.output.scores.length===t.demand.length,`output.scores must hold one row per demand row`),c(n,t.output.unservedDemand.length===t.demand.length,`output.unservedDemand must hold one row per demand row`),c(n,t.output.summary.length===D,`output.summary must hold ${D} rows`),r(n,[t.output.assignments,t.output.scores,t.output.unservedDemand,t.output.summary],t.output.status),e(this.id,k(t.output),[t.costs,t.demand,t.supply,t.parameters])}getCommandNodes(e){let{id:n,props:r}=this;t(e,n,Object.fromEntries([r.costs,r.demand,r.supply,r.parameters,...k(r.output)].map((e,t)=>[`view ${t}`,e])));let s=[o(e,{id:n,operation:`GPUNetworkRationalAccess`,variant:`initialize`,invocationCount:1,workgroupSize:2,bindings:[{name:`costs`,view:r.costs,type:`f32`,access:`read`},{name:`demand`,view:r.demand,type:`f32`,access:`read`},{name:`supply`,view:r.supply,type:`f32`,access:`read`},{name:`parameters`,view:r.parameters,type:`f32`,access:`read`},{name:`assignments`,view:r.output.assignments,type:`f32`,access:`read_write`},{name:`scores`,view:r.output.scores,type:`f32`,access:`read_write`},{name:`unserved`,view:r.output.unservedDemand,type:`f32`,access:`read_write`},{name:`summary`,view:r.output.summary,type:`u32`,access:`read_write`}],declarations:`const ORIGIN_COUNT: u32 = ${r.demand.length}u; const FACILITY_COUNT: u32 = ${r.supply.length}u;
${a}
`,body:`var nanBits = 0x7fc00000u; let missing = bitcast<f32>(nanBits); var infinityBits = 0x7f800000u; let infinity = bitcast<f32>(infinityBits);
for (var word = 0u; word < 4u; word++) { summary[summaryOffset + word] = 0u; }
for (var entry = 0u; entry < ORIGIN_COUNT * FACILITY_COUNT; entry++) { assignments[assignmentsOffset + entry] = 0.0; }
for (var origin = 0u; origin < ORIGIN_COUNT; origin++) { scores[scoresOffset + origin] = missing; unserved[unservedOffset + origin] = demand[demandOffset + origin]; }
let travelScale = parameters[parametersOffset]; let supplyScale = parameters[parametersOffset + 1u];
let initialStep = parameters[parametersOffset + 2u]; let minimumStep = parameters[parametersOffset + 3u]; let halfLife = parameters[parametersOffset + 4u]; let tolerance = parameters[parametersOffset + 5u];
if (!isFiniteValue(travelScale) || travelScale <= 0.0 || !isFiniteValue(supplyScale) || supplyScale <= 0.0 || !isFiniteValue(initialStep) || initialStep < 0.0 || ${r.stepMode===`absolute`?`false`:`initialStep > 1.0`} || !isFiniteValue(minimumStep) || minimumStep < 0.0 || ${r.stepMode===`absolute`?`false`:`minimumStep > 1.0`} || !isFiniteValue(halfLife) || halfLife <= 0.0 || !isFiniteValue(tolerance) || tolerance < 0.0) { return; }
var capacities: array<f32, ${r.supply.length}>;
for (var facility = 0u; facility < FACILITY_COUNT; facility++) { capacities[facility] = supply[supplyOffset + facility] * supplyScale; if (!isFiniteValue(capacities[facility]) || capacities[facility] <= 0.0) { return; } }
var totalDemand = 0.0;
for (var origin = 0u; origin < ORIGIN_COUNT; origin++) {
  let amount = demand[demandOffset + origin]; if (!isFiniteValue(amount) || amount < 0.0 || ${r.demandMode===`fractional`?`false`:`floor(amount) != amount`} || amount > 16777216.0 - totalDemand) { return; } totalDemand += amount;
  var nearest = 0xffffffffu; var nearestCost = infinity;
  for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
    let cost = costs[costsOffset + origin * FACILITY_COUNT + facility];
    if (cost < 0.0 || (bitcast<u32>(cost) & 0x7fffffffu) > 0x7f800000u) { return; }
    if (cost < nearestCost) { nearest = facility; nearestCost = cost; }
  }
  if (nearest != 0xffffffffu) { assignments[assignmentsOffset + origin * FACILITY_COUNT + nearest] = amount; unserved[unservedOffset + origin] = 0.0; }
}
if (totalDemand > 16777216.0) { return; }
summary[summaryOffset] = 1u;
`}),o(e,{id:`${n}-cycles`,operation:`GPUNetworkRationalAccess`,variant:`cooperative-cycles`,invocationCount:64,workgroupSize:64,guardIndex:!1,bindings:[{name:`costs`,view:r.costs,type:`f32`,access:`read`},{name:`demand`,view:r.demand,type:`f32`,access:`read`},{name:`supply`,view:r.supply,type:`f32`,access:`read`},{name:`parameters`,view:r.parameters,type:`f32`,access:`read`},{name:`assignments`,view:r.output.assignments,type:`f32`,access:`read_write`},{name:`scores`,view:r.output.scores,type:`f32`,access:`read_write`},{name:`unserved`,view:r.output.unservedDemand,type:`f32`,access:`read_write`},{name:`summary`,view:r.output.summary,type:`u32`,access:`read_write`}],...b(r)})];return s.push(...i(e,n,`GPUNetworkRationalAccess`,r.output.summary,r.output.status,{invalidCount:`1u - summary[summaryOffset]`,iterationCount:`summary[summaryOffset + 2u]`,converged:`summary[summaryOffset] * summary[summaryOffset + 1u]`,iterationLimitReached:`summary[summaryOffset] * (1u - summary[summaryOffset + 1u])`})),s}};function k(e){return[e.assignments,e.scores,e.unservedDemand,e.summary,...n(e.status)]}var A=48,j=8,M=u({parameters:y,caption:`Access score per demand location after rational reallocation.`,createInputs:({random:e,world:t})=>{let n=p(e,t,j,A);return{...n,drag:d(n)}},build(e,t,n){let{network:r}=t,i=m(e,r),a=e.upload(`facility-nodes`,`uint32`,t.facilityNodes),o=e.upload(`facilities`,`float32x2`,t.facilityPositions),s=h(e,i,t.demandNodes,r.nodeCount,a),c=e.allocate(`parameters`,`float32`,6),l=e.allocate(`assignments`,`float32`,384),u=e.allocate(`scores`,`float32`,A),d=e.allocate(`unserved`,`float32`,A),f=e.allocate(`status`,`uint32`,4);return e.graph.add(new O({costs:s.view,demand:e.upload(`demand`,`float32`,Float32Array.from(t.population,e=>Math.round(e))).view,supply:e.upload(`supply`,`float32`,Float32Array.from(t.supply,e=>40*e)).view,parameters:c.view,maximumCycles:Math.round(n.maximumCycles),demandMode:`fractional`,output:{assignments:l.view,scores:u.view,unservedDemand:d.view,summary:f.view}})),{outputs:{...v(e,r),scores:u,unserved:d,demands:e.upload(`demands`,`float32x2`,t.demandPositions),facilities:o},update:e=>{a.write(t.facilityNodes),o.write(t.facilityPositions),i.costLimit.write(Float32Array.of(4)),c.write(Float32Array.of(e.tau,e.rho,.5,.01,8,1e-4))}}},rebuildOn:[`maximumCycles`],interaction:{kind:`drag`,hint:`drag a facility`,rest:({inputs:e,toUnit:t})=>t([e.drag.defaultPositions[0],e.drag.defaultPositions[1]]),resolve(e,{inputs:t,values:n}){let r=f(t,t.drag,e.x,e.y);return{rerun:r.changed,glyphs:[{shape:`crosshair`,at:[t.network.nodes[r.node*2],t.network.nodes[r.node*2+1]],size:14}],readout:`facility ${r.facility+1} · tau ${n.tau.toFixed(1)} · rho ${n.rho.toFixed(1)}`}}},draw:({outputs:e,theme:t})=>[_(e),{layer:`dots`,positions:e.demands,values:e.scores,radius:9},...g(e.facilities,t,{radius:11})]});export{M as default};