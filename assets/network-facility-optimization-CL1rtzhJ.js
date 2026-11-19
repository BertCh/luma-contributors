import{n as e,t}from"./network-optimization-bindings-B3Zg-1cS.js";import{r as n,s as r}from"./kernel-C3tfQOn4.js";import{o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as a,r as o,t as s}from"./network-optimization-status-D0dFmcvb.js";import{n as c,t as l}from"./facility-score-reduction-BRFyW8Cg.js";import{E as u,S as d,T as f,f as p,p as m,v as h}from"./gpu-contributor-utils-BIwLQON5.js";import{u as g}from"./wgsl-common-DfyXiYB0.js";import{t as _}from"./define-run-B-brkCe4.js";import{a as v,c as y,i as b,l as x,n as S,s as C}from"./access-Co69Wfld.js";import{r as w}from"./display-Bu20g6o5.js";import{p as T,v as E}from"./roads-DzovamHj.js";import{Ea as D}from"./entry-preview-D70TdSbK.js";var O=16,k=64,A=1048576,j=4,M=65536,N=7,P=class{id;props;candidateCount;estimatedCommandNodeCount;constructor(t){this.id=t.id??`network-facility-optimization`,this.props=t;let{id:n}=this;p(n,`facilityCount`,t.facilityCount,1,O),t.selectCount===void 0?f(n,t.operation===`backup-coverage`,`selectCount may be omitted only for backup-coverage`):p(n,`selectCount`,t.selectCount,t.operation===`p-dispersion`?2:1,t.facilityCount),d(n,`demand rows`,t.demand.length,k),this.candidateCount=t.operation===`capacitated-p-median`?t.facilityCount**t.demand.length:2**t.facilityCount,d(n,`candidate count (facilityCount ** demand rows for capacitated-p-median, else 2 ** facilityCount)`,this.candidateCount,A),p(n,`maximumCandidates`,t.maximumCandidates??this.candidateCount,0,A);let r=Math.min(this.candidateCount,t.maximumCandidates??this.candidateCount),i=Math.max(1,Math.min(r,M)),a=0;for(let e=i;e>1;e=Math.ceil(e/64))a+=2;this.estimatedCommandNodeCount=2+Math.ceil(Math.max(r,1)/i)*(2+a)+Number(Object.values(t.output.status??{}).some(Boolean)),m(n,this.estimatedCommandNodeCount);for(let e of[t.costs,t.demand,t.capacities,t.threshold,t.output.objective])u(e,[`float32`],this.id);for(let e of[t.output.selectedFacilities,t.output.assignments,t.output.summary])e&&u(e,[`uint32`],this.id);f(n,t.costs.length===t.facilityCount*(t.operation===`p-dispersion`?t.facilityCount:t.demand.length),`costs must hold facilityCount times demand rows (facilityCount squared for p-dispersion)`),f(n,t.capacities.length===t.facilityCount,`capacities must hold facilityCount rows`),f(n,t.threshold.length===1,`threshold must hold one row`),f(n,t.output.selectedFacilities.length===t.facilityCount,`output.selectedFacilities must hold facilityCount rows`),f(n,t.output.assignments.length===t.demand.length,`output.assignments must hold one row per demand row`),f(n,t.output.objective.length===1,`output.objective must hold one row`),f(n,t.output.summary.length===j,`output.summary must hold ${j} rows`);let s=[t.output.selectedFacilities,t.output.assignments,t.output.objective,t.output.summary];o(n,s,t.output.status),e(this.id,F(t.output),[t.costs,t.demand,t.capacities,t.threshold])}getCommandNodes(e){let{id:a,props:o,candidateCount:u}=this,d=Math.min(u,o.maximumCandidates??u),f=o.operation===`capacitated-p-median`,p=o.operation===`p-dispersion`,m=Math.max(1,Math.min(d,M));h(e.device,a,`candidate scores`,m*16,`reduce maximumCandidates`),h(e.device,a,`facility control state`,28,`request a larger device binding limit`);for(let t of[o.costs,o.demand,o.capacities,o.threshold,...F(o.output)])h(e.device,a,`input/output aligned binding range`,r(t).size,`reduce facilityCount, demand rows or the alignment prefix`);t(e,a,{costs:o.costs.length?o.costs:void 0,demand:o.demand.length?o.demand:void 0,capacities:o.capacities,threshold:o.threshold,selectedFacilities:o.output.selectedFacilities,assignments:o.output.assignments.length?o.output.assignments:void 0,objective:o.output.objective,summary:o.output.summary,...o.output.status});let _=n(e,`${a}-${o.operation}-state`,`uint32`,N),v=n(e,`${a}-${o.operation}-scores`,`uint32`,m*4),y=`const FACILITY_COUNT: u32 = ${o.facilityCount}u;
const DEMAND_COUNT: u32 = ${o.demand.length}u;
${g}`,b=o.demand.length?o.demand:n(e,`${a}-empty-demand-input`,`float32`,1),x=o.costs.length?o.costs:b,S=o.output.assignments.length?o.output.assignments:n(e,`${a}-empty-assignments`,`uint32`,1),C=[{name:`costs`,view:x,type:`f32`,access:`read`},{name:`demand`,view:b,type:`f32`,access:`read`},{name:`capacities`,view:o.capacities,type:`f32`,access:`read`},{name:`threshold`,view:o.threshold,type:`f32`,access:`read`}],w=[i(e,{id:`${a}-initialize`,operation:`GPUNetworkFacilityOptimization`,variant:`initialize`,invocationCount:1,workgroupSize:2,bindings:[...C,{name:`state`,view:_,type:`u32`,access:`read_write`}],declarations:y,body:`state[stateOffset] = 0u;
  state[stateOffset + 1u] = 0xffffffffu;
  state[stateOffset + 2u] = 0xffffffffu;
  state[stateOffset + 3u] = 0xffffffffu;
  state[stateOffset + 4u] = 0u;
  state[stateOffset + 5u] = 0u;
  state[stateOffset + 6u] = 0u;
  for (var row = 0u; row < DEMAND_COUNT; row++) {
    if (!isFiniteValue(demand[demandOffset + row]) || demand[demandOffset + row] < 0.0) { return; }
  }
  ${f?`for (var facility = 0u; facility < FACILITY_COUNT; facility++) { if (!isFiniteValue(capacities[capacitiesOffset + facility]) || capacities[capacitiesOffset + facility] < 0.0) { return; } }`:``}
  ${!f&&!p?`if (!isFiniteValue(threshold[thresholdOffset]) || threshold[thresholdOffset] < 0.0) { return; }`:``}
  for (var entry = 0u; entry < ${o.costs.length}u; entry++) {
    let cost = costs[costsOffset + entry];
    if (cost < 0.0 || (bitcast<u32>(cost) & 0x7fffffffu) > 0x7f800000u) { return; }
  }
  ${p?`for (var first = 0u; first < FACILITY_COUNT; first++) { for (var second = 0u; second < FACILITY_COUNT; second++) { if (costs[costsOffset + first * FACILITY_COUNT + second] != costs[costsOffset + second * FACILITY_COUNT + first]) { return; } } }`:``}
  state[stateOffset + 5u] = 1u;`})];for(let t=0;t<Math.ceil(Math.max(d,1)/m);t++){w.push(i(e,{id:`${a}-score-${t}`,operation:`GPUNetworkFacilityOptimization`,variant:o.operation,invocationCount:m,bindings:[...C,{name:`state`,view:_,type:`u32`,access:`read`},{name:`scores`,view:v,type:`u32`,access:`read_write`}],declarations:y,body:`let offset = scoresOffset + index * 4u;
  scores[offset] = 0xffffffffu; scores[offset + 1u] = 0xffffffffu;
  scores[offset + 2u] = 0xffffffffu; scores[offset + 3u] = 0u;
  let candidate = state[stateOffset] + index;
  if (candidate >= ${d}u || state[stateOffset + 5u] == 0u) { return; }
  var mask = candidate; var score = 0.0; var feasible = true;
  ${f?`mask = 0u; var code = candidate; var used: array<f32, ${o.facilityCount}>;
  for (var row = 0u; row < DEMAND_COUNT; row++) {
    let facility = code % FACILITY_COUNT; code /= FACILITY_COUNT;
    let cost = costs[costsOffset + row * FACILITY_COUNT + facility]; let weight = demand[demandOffset + row];
    if (weight == 0.0) { continue; } mask |= 1u << facility;
    used[facility] += weight;
    if (!isFiniteValue(cost) || used[facility] > capacities[capacitiesOffset + facility]) { feasible = false; }
    score += weight * cost;
  }
  if (countOneBits(mask) > ${o.selectCount}u) { feasible = false; }`:`let selectedCount = countOneBits(mask);
  if (selectedCount == 0u${o.selectCount===void 0?``:` || selectedCount != ${o.selectCount}u`}) { return; }
  ${p?`var infinityBits = 0x7f800000u; score = bitcast<f32>(infinityBits);
  for (var first = 0u; first < FACILITY_COUNT; first++) { if ((mask & (1u << first)) == 0u) { continue; }
    for (var second = first + 1u; second < FACILITY_COUNT; second++) { if ((mask & (1u << second)) != 0u) { score = min(score, costs[costsOffset + first * FACILITY_COUNT + second]); } }
  }`:`for (var row = 0u; row < DEMAND_COUNT; row++) {
    var covered = 0u;
    for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
      if ((mask & (1u << facility)) != 0u && costs[costsOffset + row * FACILITY_COUNT + facility] <= threshold[thresholdOffset]) { covered++; }
    }
    if (covered == 0u) { feasible = false; }
    if (covered >= 2u) { score += demand[demandOffset + row]; }
  }`}`}
  ${p?``:`if (!isFiniteValue(score) && feasible) { scores[offset + 3u] = 1u; return; }`}
  if (!feasible) { return; }
  // Signed zero is a numeric tie, including in the maximum-dispersion objective.
  let scoreBits = select(bitcast<u32>(score), 0u, score == 0.0);
  scores[offset] = ${!f&&o.selectCount===void 0?`countOneBits(mask)`:`0u`};
  scores[offset + 1u] = ${f?`scoreBits`:`~scoreBits`};
  scores[offset + 2u] = candidate;`}));let n=c(e,{id:`${a}-${o.operation}-reduce-${t}`,operation:`GPUNetworkFacilityOptimization`,scores:v,candidateCount:m});w.push(...n.nodes,i(e,{id:`${a}-advance-${t}`,operation:`GPUNetworkFacilityOptimization`,variant:`advance`,invocationCount:1,workgroupSize:2,bindings:[{name:`state`,view:_,type:`u32`,access:`read_write`},{name:`winner`,view:n.winner,type:`u32`,access:`read`}],declarations:l,body:`let current = vec4<u32>(state[stateOffset + 1u], state[stateOffset + 2u], state[stateOffset + 3u], state[stateOffset + 4u]);
  let next = vec4<u32>(winner[winnerOffset], winner[winnerOffset + 1u], winner[winnerOffset + 2u], winner[winnerOffset + 3u]);
  let best = mergeFacilityScores(current, next);
  state[stateOffset + 1u] = best.x; state[stateOffset + 2u] = best.y;
  state[stateOffset + 3u] = best.z; state[stateOffset + 4u] = best.w;
  if (state[stateOffset + 5u] != 0u && state[stateOffset] < ${d}u) {
    state[stateOffset + 6u]++;
  }
  state[stateOffset] += ${m}u;`}))}return w.push(i(e,{id:`${a}-publish`,operation:`GPUNetworkFacilityOptimization`,variant:`publish`,invocationCount:1,workgroupSize:2,bindings:[{name:`demand`,view:b,type:`f32`,access:`read`},{name:`state`,view:_,type:`u32`,access:`read`},{name:`selected`,view:o.output.selectedFacilities,type:`u32`,access:`read_write`},{name:`assignments`,view:S,type:`u32`,access:`read_write`},{name:`objective`,view:o.output.objective,type:`f32`,access:`read_write`},{name:`summary`,view:o.output.summary,type:`u32`,access:`read_write`}],declarations:y,body:`for (var word = 0u; word < 4u; word++) { summary[summaryOffset + word] = 0u; }
  var invalidBits = 0x7fc00000u; objective[objectiveOffset] = bitcast<f32>(invalidBits);
  for (var facility = 0u; facility < FACILITY_COUNT; facility++) { selected[selectedOffset + facility] = 0u; }
  for (var row = 0u; row < DEMAND_COUNT; row++) { assignments[assignmentsOffset + row] = 0xffffffffu; }
  if (state[stateOffset + 5u] == 0u || state[stateOffset + 4u] != 0u) { return; }
  let bestCandidate = state[stateOffset + 3u];
  let haveBest = bestCandidate != 0xffffffffu;
  summary[summaryOffset] = 1u; summary[summaryOffset + 1u] = u32(haveBest);
  summary[summaryOffset + 2u] = ${d===u?`1u`:`0u`};
  summary[summaryOffset + 3u] = ${d}u;
  if (!haveBest) { return; }
  var bestMask = bestCandidate;
  ${f?`bestMask = 0u; var code = bestCandidate;
  for (var row = 0u; row < DEMAND_COUNT; row++) {
    let facility = code % FACILITY_COUNT; code /= FACILITY_COUNT;
    if (demand[demandOffset + row] > 0.0) {
      assignments[assignmentsOffset + row] = facility; bestMask |= 1u << facility;
    }
  }
  var bestCount = countOneBits(bestMask);
  for (var facility = 0u; facility < FACILITY_COUNT && bestCount < ${o.selectCount}u; facility++) {
    if ((bestMask & (1u << facility)) == 0u) { bestMask |= 1u << facility; bestCount++; }
  }`:``}
  for (var facility = 0u; facility < FACILITY_COUNT; facility++) { selected[selectedOffset + facility] = (bestMask >> facility) & 1u; }
  objective[objectiveOffset] = bitcast<f32>(${f?``:`~`}state[stateOffset + 2u]);`})),w.push(...s(e,a,`GPUNetworkFacilityOptimization`,o.output.summary,o.output.status,{invalidCount:`1u - summary[summaryOffset]`,iterationCount:`state[stateOffset + 6u]`,converged:`summary[summaryOffset] * summary[summaryOffset + 2u]`,iterationLimitReached:`summary[summaryOffset] * (1u - summary[summaryOffset + 2u])`},o.output.status?.iterationCount?[{name:`state`,view:_,type:`u32`,access:`read`}]:[])),w}};function F(e){return[e.selectedFacilities,e.assignments,e.objective,e.summary,...a(e.status)]}var I=40,L=8,R=_({parameters:D,caption:`Candidate sites chosen for backup coverage (bright) among the demand locations.`,createInputs:({random:e,world:t})=>{let n=b(e,t,L,I);return n.demandNodes.forEach((t,r)=>{let i=r%L,a=n.network.findNearestNode(n.facilityPositions[i*2]+(e()-.5)*.4,n.facilityPositions[i*2+1]+(e()-.5)*.4);n.demandNodes[r]=a,n.demandPositions[r*2]=n.network.nodes[a*2],n.demandPositions[r*2+1]=n.network.nodes[a*2+1]}),{...n,drag:v(n)}},build(e,t,n){let{network:r}=t,i=x(e,r),a=e.upload(`candidate-nodes`,`uint32`,t.facilityNodes),o=e.upload(`candidates`,`float32x2`,t.facilityPositions),s=S(e,i,t.demandNodes,r.nodeCount,a),c=e.allocate(`threshold`,`float32`,1),l=e.allocate(`coverage`,`uint32`,I),u=e.allocate(`selected`,`uint32`,L),d=e.allocate(`assignments`,`uint32`,I),f=e.allocate(`objective`,`float32`,1),p=e.allocate(`status`,`uint32`,4);return e.graph.add(new P({operation:`backup-coverage`,facilityCount:L,selectCount:Math.round(n.selectCount),costs:s.view,demand:e.upload(`demand`,`float32`,t.population).view,capacities:e.upload(`capacities`,`float32`,new Float32Array(L).fill(1e3)).view,threshold:c.view,output:{selectedFacilities:u.view,assignments:d.view,objective:f.view,summary:p.view}})),{outputs:{...E(e,r),assignments:d,selected:u,matrix:s,selectedIds:w(e,{id:`selected`,flags:u,count:L}),coverage:l,demands:e.upload(`demands`,`float32x2`,t.demandPositions),candidates:o},update:e=>{a.write(t.facilityNodes),o.write(t.facilityPositions),i.costLimit.write(Float32Array.of(4)),c.write(Float32Array.of(e.threshold))}}},rebuildOn:[`selectCount`],interaction:{kind:`drag`,hint:`drag a candidate site`,rest:({inputs:e,toUnit:t})=>t([e.drag.defaultPositions[0],e.drag.defaultPositions[1]]),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=y(t,t.drag,e.x,e.y),o=i(n.selected),s=i(n.matrix),c=[{shape:`crosshair`,at:[t.network.nodes[a.node*2],t.network.nodes[a.node*2+1]],size:14}],l=0,u=0;if(o&&s){let e=new Uint32Array(I);for(let t=0;t<L;t++)if(o[t]===1){u++;for(let n=0;n<I;n++)s[n*L+t]<=r.threshold&&e[n]++}for(let t=0;t<I;t++)e[t]=Math.min(2,e[t]),e[t]===2&&l++;n.coverage.write(e)}return{rerun:a.changed,glyphs:c,readout:u?`site ${a.facility+1} · threshold ${r.threshold.toFixed(2)} · backup ${Math.round(100*l/I)}%`:`site ${a.facility+1} · no selection covers every demand`}}},draw:({outputs:e,theme:t})=>[T(e),{layer:`dots`,positions:e.demands,values:e.coverage,ramp:`categorical`,categories:[[...t.categories[2],1],[...t.muted,1],[...t.categories[4],1]],radius:8},...C(e.candidates,t,{ids:e.selectedIds,radius:11})]});export{R as default};