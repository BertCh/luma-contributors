import{n as e,t}from"./network-optimization-bindings-B3Zg-1cS.js";import{r as n}from"./kernel-C3tfQOn4.js";import{n as r,r as i,t as a}from"./network-optimization-status-D0dFmcvb.js";import{o}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as s,T as c,f as l,v as u}from"./gpu-contributor-utils-BIwLQON5.js";import{u as d}from"./wgsl-common-DfyXiYB0.js";var f=256,p=4096,m=256,h=65536,g=2,_=4,v=class{id;props;constructor(t){this.id=t.id??`network-time-routing`,this.props=t;let{id:n}=this;l(n,`stateVertices rows`,t.stateVertices.length,1,f),l(n,`transitions rows`,t.transitions.length,0,p),t.costModel===`piecewise-arrival`?l(n,`samplesPerTransition`,t.samplesPerTransition,2,m):c(n,t.samplesPerTransition===void 0&&t.waitingPolicy===void 0,`samplesPerTransition and waitingPolicy require costModel piecewise-arrival`),l(n,`maximumRelaxations`,t.maximumRelaxations??t.stateVertices.length,0,h);for(let e of[t.stateVertices,t.endpoints,t.output.predecessorTransitions,t.output.summary])s(e,[`uint32`],this.id);s(t.transitions,[`uint32x2`],this.id),s(t.costs,[`float32x2`],this.id);for(let e of[t.departure,t.output.arrivals])s(e,[`float32`],this.id);c(n,t.costs.length===t.transitions.length*(t.costModel===`piecewise-arrival`?t.samplesPerTransition:1),`costs must hold samplesPerTransition rows per transition (one for the affine model)`),c(n,t.endpoints.length===g,`endpoints must hold ${g} rows: initial state and destination vertex`),c(n,t.departure.length===1,`departure must hold one row`),c(n,t.output.arrivals.length===t.stateVertices.length,`output.arrivals must hold one row per state`),c(n,t.output.predecessorTransitions.length===t.stateVertices.length,`output.predecessorTransitions must hold one row per state`),c(n,t.output.summary.length===_,`output.summary must hold ${_} rows`),i(n,[t.output.arrivals,t.output.predecessorTransitions,t.output.summary],t.output.status),e(this.id,y(t.output),[t.stateVertices,t.transitions,t.costs,t.endpoints,t.departure])}getCommandNodes(e){let{id:r,props:i}=this;t(e,r,Object.fromEntries([i.stateVertices,i.transitions,i.costs,i.endpoints,i.departure,...y(i.output)].map((e,t)=>[`view ${t}`,e])));let s=!!i.output.status?.iterationCount;s&&u(e.device,r,`summary with iteration count`,20,`omit output.status.iterationCount or request a larger binding limit`);let c=s?n(e,`${r}-summary-with-iterations`,`uint32`,5):i.output.summary,l=[o(e,{id:r,operation:`GPUNetworkTimeRouting`,variant:i.costModel===`piecewise-arrival`?`piecewise-fifo-wait`:`affine-fifo`,invocationCount:1,workgroupSize:2,bindings:[{name:`vertices`,view:i.stateVertices,type:`u32`,access:`read`},{name:`transitions`,view:i.transitions,type:`u32`,access:`read`},{name:`costs`,view:i.costs,type:`f32`,access:`read`},{name:`endpoints`,view:i.endpoints,type:`u32`,access:`read`},{name:`departure`,view:i.departure,type:`f32`,access:`read`},{name:`arrivals`,view:i.output.arrivals,type:`f32`,access:`read_write`},{name:`predecessors`,view:i.output.predecessorTransitions,type:`u32`,access:`read_write`},{name:`summary`,view:c,type:`u32`,access:`read_write`}],declarations:`const STATE_COUNT: u32 = ${i.stateVertices.length}u;
const TRANSITION_COUNT: u32 = ${i.transitions.length}u;
${d}
fn arrival(edge: u32, time: f32) -> f32 {
${i.costModel===`piecewise-arrival`?`
  let base = costsOffset + 2u * edge * ${i.samplesPerTransition}u;
  let firstTime = costs[base]; let firstArrival = costs[base + 1u];
  if (time <= firstTime) { return firstArrival; }
  for (var sample = 1u; sample < ${i.samplesPerTransition}u; sample++) {
    let rightTime = costs[base + 2u * sample]; let rightArrival = costs[base + 2u * sample + 1u];
    if (time <= rightTime) {
      let leftTime = costs[base + 2u * (sample - 1u)]; let leftArrival = costs[base + 2u * (sample - 1u) + 1u];
      return mix(leftArrival, rightArrival, (time - leftTime) / (rightTime - leftTime));
    }
  }
  // Preserve the nonnegative held duration before addition; max also prevents fast-math
  // reassociation into time + absoluteArrival - departure, whose partial sum can overflow.
  let duration = max(0.0, costs[base + 2u * (${i.samplesPerTransition}u - 1u) + 1u] - costs[base + 2u * (${i.samplesPerTransition}u - 1u)]);
  return time + duration;
`:`return time + costs[costsOffset + 2u * edge] + costs[costsOffset + 2u * edge + 1u] * time;`}
}`,body:`${s?`summary[summaryOffset + 4u] = 0u;`:``}
var infinityBits = 0x7f800000u; let infinity = bitcast<f32>(infinityBits);
summary[summaryOffset] = 0u; summary[summaryOffset + 1u] = 0u; summary[summaryOffset + 2u] = 0xffffffffu; summary[summaryOffset + 3u] = bitcast<u32>(infinity);
for (var state = 0u; state < STATE_COUNT; state++) { arrivals[arrivalsOffset + state] = infinity; predecessors[predecessorsOffset + state] = 0xffffffffu; }
let initial = endpoints[endpointsOffset];
let destination = endpoints[endpointsOffset + 1u];
let startTime = departure[departureOffset];
if (initial >= STATE_COUNT || !isFiniteValue(startTime) || startTime < 0.0) { return; }
for (var edge = 0u; edge < TRANSITION_COUNT; edge++) {
  let source = transitions[transitionsOffset + 2u * edge]; let destination = transitions[transitionsOffset + 2u * edge + 1u];
  if (source >= STATE_COUNT || destination >= STATE_COUNT) { return; }
  ${i.costModel===`piecewise-arrival`?`
  var previousDeparture = -1.0; var previousArrival = -1.0;
  for (var sample = 0u; sample < ${i.samplesPerTransition}u; sample++) {
    let offset = costsOffset + 2u * (edge * ${i.samplesPerTransition}u + sample);
    let sampleDeparture = costs[offset]; let sampleArrival = costs[offset + 1u];
    if (!isFiniteValue(sampleDeparture) || !isFiniteValue(sampleArrival) || sampleDeparture < 0.0 || sampleDeparture <= previousDeparture || sampleArrival < sampleDeparture || sampleArrival < previousArrival) { return; }
    previousDeparture = sampleDeparture; previousArrival = sampleArrival;
  }`:`let base = costs[costsOffset + 2u * edge]; let slope = costs[costsOffset + 2u * edge + 1u];
  if (!isFiniteValue(base) || !isFiniteValue(slope) || base < 0.0 || slope < 0.0) { return; }`}
}
summary[summaryOffset] = 1u;
arrivals[arrivalsOffset + initial] = startTime;
for (var round = 0u; round < ${i.maximumRelaxations??i.stateVertices.length}u; round++) {
  ${s?`summary[summaryOffset + 4u]++;`:``}
  var changed = false;
  for (var edge = 0u; edge < TRANSITION_COUNT; edge++) {
    let source = transitions[transitionsOffset + 2u * edge]; let destination = transitions[transitionsOffset + 2u * edge + 1u];
    let time = arrivals[arrivalsOffset + source];
    if (!isFiniteValue(time)) { continue; }
    let candidate = arrival(edge, time);
    if (!isFiniteValue(candidate)) { summary[summaryOffset] = 0u; return; }
    if (candidate < arrivals[arrivalsOffset + destination]) {
      arrivals[arrivalsOffset + destination] = candidate; predecessors[predecessorsOffset + destination] = edge; changed = true;
    }
  }
  if (!changed) { summary[summaryOffset + 1u] = 1u; break; }
}
var best = infinity;
var bestState = 0xffffffffu;
for (var state = 0u; state < STATE_COUNT; state++) {
  if (vertices[verticesOffset + state] == destination && arrivals[arrivalsOffset + state] < best) { best = arrivals[arrivalsOffset + state]; bestState = state; }
}
summary[summaryOffset + 2u] = bestState;
summary[summaryOffset + 3u] = bitcast<u32>(best);`})];return s&&l.push(o(e,{id:`${r}-publish-summary`,operation:`GPUNetworkTimeRouting`,variant:`publish-summary`,invocationCount:_,bindings:[{name:`source`,view:c,type:`u32`,access:`read`},{name:`destination`,view:i.output.summary,type:`u32`,access:`read_write`}],body:`destination[destinationOffset + index] = source[sourceOffset + index];`})),l.push(...a(e,r,`GPUNetworkTimeRouting`,c,i.output.status,{invalidCount:`1u - summary[summaryOffset]`,iterationCount:`summary[summaryOffset + 4u]`,converged:`summary[summaryOffset] * summary[summaryOffset + 1u]`,iterationLimitReached:`summary[summaryOffset] * (1u - summary[summaryOffset + 1u])`})),l}};function y(e){return[e.arrivals,e.predecessorTransitions,e.summary,...r(e.status)]}export{v as t};