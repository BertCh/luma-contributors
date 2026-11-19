import{n as e,t}from"./network-optimization-bindings-B3Zg-1cS.js";import{o as n,r,s as i,t as a}from"./kernel-C3tfQOn4.js";import{A as o,E as s,O as c,T as l,f as u,k as d,p as f,v as p,x as m}from"./gpu-contributor-utils-BIwLQON5.js";import{Q as h}from"./dist-BusamGlx.js";import{t as ee}from"./gpu-sort-xRm8l5yn.js";import{t as te}from"./gpu-scan-B4NNTfF9.js";import{t as ne}from"./gpu-gather-BJIYd708.js";import{t as g}from"./gpu-reduction-tNgOVQ_C.js";import{a as _,o as v}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as re,n as ie,o as y}from"./sorted-segment-sums-Dvmm6OdR.js";import{t as ae}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{n as b}from"./sorted-segment-offsets-ChmOtA9F.js";import{t as x}from"./binary-search-wgsl-CMmaMYIt.js";import{r as oe,u as se}from"./wgsl-common-DfyXiYB0.js";import{t as S}from"./permutation-random-SiwCiDNI.js";import{t as C}from"./philox-random-DfVxWNPn.js";import{t as w}from"./define-run-B-brkCe4.js";import{n as T,r as ce,t as E}from"./commute-flows-CmHJas7f.js";import{l as D,s as O,u as le}from"./pointer-overlay-Qh8qflcA.js";import{n as ue}from"./graphs-Bws-bfBV.js";import{ya as k}from"./entry-preview-D70TdSbK.js";function de(e,n,r){let a=[[`weights`,r.weights],[`parameters`,r.parameters],[`topology.offsets`,r.topology?.offsets],[`topology.neighbors`,r.topology?.neighbors],[`topology.weights`,r.topology?.weights],[`topology.nodePositions`,r.topology?.nodePositions],[`output.communities`,r.output.communities],[`output.objectiveTrace`,r.output.objectiveTrace],[`output.summary`,r.output.summary],...Object.entries(r.output.status??{}).map(([e,t])=>[`output.status.${e}`,t])];for(let[t,r]of a)r&&p(e.device,n,t,i(r).size,`reduce the view length or its prefix within the aligned binding`);t(e,n,Object.fromEntries(a))}function fe(e,t,r){return{...e,description:`GPU-gated Louvain primitive (public ComputePass dispatch adapter)`,resources:[...e.resources??[],{buffer:t,usage:`storage-read`}],workload:{...e.workload,commandCount:2},compile:i=>{let o=e.compile(i),s=[()=>o.destroy?.()];try{let c=i.device.createBuffer({byteLength:16,usage:h.UNIFORM|h.COPY_DST});s.push(()=>c.destroy());let l=i.device.createBuffer({byteLength:12,usage:h.STORAGE|h.INDIRECT});s.push(()=>l.destroy());let u=new a(i.device,{id:`${e.id}-louvain-gate`,source:`const stateOffset: u32 = ${t.byteOffset/4}u;
@group(0) @binding(0) var<storage, read> state: array<u32>;
@group(0) @binding(1) var<uniform> geometry: vec4<u32>;
@group(0) @binding(2) var<storage, read_write> dispatch: array<u32>;
@compute @workgroup_size(1) fn main() {
  dispatch[0] = select(0u, geometry.x, ${r});
  dispatch[1] = geometry.y; dispatch[2] = geometry.z;
}`,shaderLayout:{bindings:[{name:`state`,group:0,location:0,type:`read-only-storage`},{name:`geometry`,group:0,location:1,type:`uniform`},{name:`dispatch`,group:0,location:2,type:`storage`}]}});s.push(()=>u.destroy());let d=``;return{encode:e=>{let r=e.computePass,i,a,s=new Proxy(r,{get(o,s){if(s===`setPipeline`)return e=>{i=e};if(s===`setBindings`)return e=>{a=e};if(s===`dispatch`)return(o,s=1,f=1)=>{let p=`${o},${s},${f}`;p!==d&&(c.write(Uint32Array.of(o,s,f,0)),d=p),u.dispatch(r,{bindings:{state:n(t,e.getBuffer),geometry:c,dispatch:l},x:1}),r.setPipeline(i),r.setBindings(a),r.dispatchIndirect(l)};let f=Reflect.get(o,s,o);return typeof f==`function`?f.bind(o):f}});o.encode({...e,computePass:s})},destroy:()=>{o.destroy?.(),u.destroy(),c.destroy(),l.destroy()}}}catch(e){for(let e of s.reverse())try{e()}catch{}throw e}}}}var pe=65535,A=1073741823,j=1073741823,M=32,N=64,me=2049,he=10,P=352,ge=320,F=208,I=65536,L=`GPUNetworkLouvain`;function R(e){return P+(e.maximumLevels??8)*((e.maximumIterations??16)*ge+F)+1}function z(t,n,r){u(t,`nodeCount`,n.nodeCount,1,A);let i=o(t,n.topology,{weights:!0});u(t,`topology.neighbors.length`,i.edgeCount,0,j),l(t,i.nodeCount===n.nodeCount,`topology.offsets must hold nodeCount + 1 rows`),u(t,`maximumLevels`,n.maximumLevels??8,1,M),u(t,`maximumIterations`,n.maximumIterations??16,1,N),u(t,`seed`,n.seed??0,0,4294967295),f(t,r,I);for(let[e,r,i,a]of[[`parameters`,n.parameters,`float32`,2],[`output.communities`,n.output.communities,`uint32`,n.nodeCount],[`output.objectiveTrace`,n.output.objectiveTrace,`float32`,1+(n.maximumLevels??8)*(n.maximumIterations??16)],[`output.summary`,n.output.summary,`uint32`,4]])s(r,[i],t),l(t,r.length===a,`${e} must hold ${a} rows`);e(t,q(n.output),[...d(n.topology),n.parameters])}function B(e,t,n,i=`two-word`){let a=n.nodeCount,o=i===`packed-reference`;o&&u(t,`packed-reference nodeCount`,a,1,pe);let s=n.topology.neighbors.length,c=Math.max(2*s,1);p(e.device,t,`sparse edge scratch`,Math.max(c*2,c+1)*4,`reduce topology edge capacity`),p(e.device,t,`sparse node scratch`,Math.max(a*4,a+1)*4,`reduce nodeCount`),de(e,t,n);for(let[r,i]of[[`sparse radix histograms`,16*Math.ceil(Math.max(a,c)/256)*4],[`sparse runtime state`,40],[`sparse round budget`,(n.maximumIterations??16)*4],[`sparse level budget`,(n.maximumLevels??8)*4],[`sparse dispatch gate allowance`,ge*3*4]])p(e.device,t,r,i,`reduce the schedule or graph shape`);m(t,e,[...d(n.topology),n.parameters,...q(n.output)]);let l=[],f=(n,i)=>r(e,`${t}-${n}`,`uint32`,i),h=new Set,S=n=>{let i=r(e,`${t}-${n}`,o?`uint32`:`uint32x2`,c);return h.add(i),i},C=(n,i)=>r(e,`${t}-${n}`,`float32`,i),w=(e,t)=>({name:e,view:t,type:t.format===`float32`?`f32`:`u32`,access:`read`}),T=(e,t,n=!1)=>({name:e,view:t,type:n?`atomic<u32>`:t.format===`float32`?`f32`:`u32`,access:`read_write`}),ce=`const NODE_COUNT: u32 = ${a}u;\nconst RECORD_COUNT: u32 = ${c}u;\nconst NONE: u32 = 0xffffffffu;\nconst NO_PAIR: vec2<u32> = vec2<u32>(NODE_COUNT);\n${se}`,E=(n,r,i,a,s=``)=>v(e,{id:`${t}-${n}`,operation:L,invocationCount:r,bindings:i,declarations:`${ce}\n${i.filter(e=>h.has(e.view)).map(e=>{let t=e.name,n=t.charAt(0).toUpperCase()+t.slice(1);return`fn get${n}Pair(position: u32) -> vec2<u32> {
  ${o?`let packedKey = ${t}[${t}Offset + position];
  if (packedKey == NONE) { return NO_PAIR; }
  return vec2<u32>(packedKey / NODE_COUNT, packedKey % NODE_COUNT);`:`return vec2<u32>(${t}[${t}Offset + 2u * position + 1u], ${t}[${t}Offset + 2u * position]);`}
}
${e.access===`read_write`?`fn set${n}Pair(position: u32, key: vec2<u32>) {
  ${o?`${t}[${t}Offset + position] = select(key.x * NODE_COUNT + key.y, NONE, key.x == NODE_COUNT);`:`${t}[${t}Offset + 2u * position] = key.y;
  ${t}[${t}Offset + 2u * position + 1u] = key.x;`}
}`:``}`}).join(`
`)}\n${s}`,body:a}),D=o?`${x(`pairKeys`,`u32`,`PackedPairKeys`)}
fn lowerBoundPairKeys(first: u32, last: u32, key: vec2<u32>) -> u32 {
  return lowerBoundPackedPairKeys(first, last, key.x * NODE_COUNT + key.y);
}`:x(`pairKeys`,`u64`),O=f(`state`,he),le=f(`round-budget`,n.maximumIterations??16),ue=f(`level-budget`,n.maximumLevels??8),k=(n,r,i)=>{let a=_(e,{id:`${t}-${n}-dispatch`,operation:L,shapes:r.map(e=>[(e.workload?.maximumWorkgroupCount??1)*2,2])});l.push(E(`${n}-gate`,1,[w(`state`,O),a.binding],`if (${i}) { ${a.armWGSL} } else { ${a.disarmWGSL} }`)),r.forEach((e,n)=>{if(e.type!==`compute`)throw Error(`${t} sparse stage requires compute nodes`);if(!e.workload?.maximumWorkgroupCount){l.push(fe(e,O,i));return}let r=a.getGate(n);l.push({...e,condition:r.condition,resources:[...e.resources??[],...r.extraResources]})})},A=(n,r,i)=>new g({id:`${t}-${n}`,input:r,output:i,operation:`sum`}).getCommandNodes(e),j=f(`labels`,a),M=f(`groups`,a),N=C(`degree-out`,a),P=C(`degree-in`,a),F=C(`totals-out`,a),I=C(`totals-in`,a),R=S(`edge-keys`),z=C(`edge-weights`,c),B=f(`source-keys`,Math.max(s,1)),V=f(`destination-keys`,Math.max(s,1)),H=C(`raw-weights`,Math.max(s,1)),U=f(`invalid`,Math.max(a+1,s,1)),W=f(`invalid-count`,1),G=C(`total`,1),K=f(`changed`,a),_e=f(`any-changed`,1),J=f(`proposals`,a),Y=f(`reservations`,a),X=f(`accepted`,a),Z=S(`proposal-keys`),Q=C(`proposal-weights`,c),ve=n=>y(e,{id:`${t}-${n}`,operation:L,segmentCount:a,segmentKeys:j,reductions:[{name:`out`,contributions:N,output:F},{name:`in`,contributions:P,output:I}]}),ye=(e,t)=>{let r=C(`${e}-internal`,c),i=C(`${e}-penalty`,a),o=C(`${e}-internal-total`,1),s=C(`${e}-penalty-total`,1);return[E(`${e}-internal`,c,[w(`edgeKeys`,R),w(`edgeWeights`,z),w(`labels`,j),T(`internal`,r)],`let key = getEdgeKeysPair(index);
var value = 0.0;
if (key.x != NODE_COUNT && labels[labelsOffset + key.x] == labels[labelsOffset + key.y]) { value = 0.5 * edgeWeights[edgeWeightsOffset + index]; }
internal[internalOffset + index] = value;`),E(`${e}-penalty`,a,[w(`totalsOut`,F),w(`totalsIn`,I),w(`parameters`,n.parameters),T(`penalty`,i)],`penalty[penaltyOffset + index] = parameters[parametersOffset] * totalsOut[totalsOutOffset + index] * totalsIn[totalsInOffset + index];`),...A(`${e}-internal-total`,r,o),...A(`${e}-penalty-total`,i,s),E(`${e}-trace`,1,[w(`internalTotal`,o),w(`penaltyTotal`,s),T(`state`,O),T(`trace`,n.output.objectiveTrace)],`let value = internalTotal[internalTotalOffset] - penaltyTotal[penaltyTotalOffset];
if (!isFiniteValue(value)) { state[stateOffset] = 0u; state[stateOffset + 4u] = 0u; return; }
trace[traceOffset + ${t?`0u`:`state[stateOffset + 7u] + 1u`}] = value;`)]},$=(n,r,i)=>{let s=f(`${n}-order`,c),l=S(`${n}-sorted-keys`),u=f(`${n}-primary-keys`,c),d=f(`${n}-secondary-keys`,c),p=C(`${n}-sorted-weights`,c),m=f(`${n}-flags`,c),h=f(`${n}-ranks`,c),g=f(`${n}-offsets`,c+1),_=S(`${n}-pair-keys`),v=C(`${n}-pair-weights`,c),y=f(`${n}-sorted-order`,c),b=[],x=y;if(o)b.push(...new ee({id:`${t}-${n}-sort`,keys:r,values:s,outputKeys:l,outputValues:y,algorithm:`radix`}).getCommandNodes(e));else{let i=ae(e,{id:`${t}-${n}-sort`,operation:L,indices:s,indicesAreIdentity:!0,algorithm:`radix`,keys:[{view:u,keyBits:re(a)},{view:d,keyBits:re(a)}]});b.push(...i.nodes,...new ne({id:`${t}-${n}-gather-keys`,source:r,indices:i.sortedIndices,output:l}).getCommandNodes(e)),x=i.sortedIndices}return{commands:[E(`${n}-order`,c,[w(`keys`,r),T(`order`,s),T(`primaryKeys`,u),T(`secondaryKeys`,d)],`let key = getKeysPair(index);
order[orderOffset + index] = index;
primaryKeys[primaryKeysOffset + index] = key.x;
secondaryKeys[secondaryKeysOffset + index] = key.y;`),...b,...new ne({id:`${t}-${n}-gather`,source:i,indices:x,output:p}).getCommandNodes(e),E(`${n}-flags`,c,[w(`sortedKeys`,l),T(`flags`,m),T(`offsets`,g),T(`pairKeys`,_)],`var starts = index == 0u;
if (index > 0u) { starts = any(getSortedKeysPair(index) != getSortedKeysPair(index - 1u)); }
flags[flagsOffset + index] = select(0u, 1u, starts);
offsets[offsetsOffset + index] = RECORD_COUNT;
if (index == 0u) { offsets[offsetsOffset + RECORD_COUNT] = RECORD_COUNT; }
setPairKeysPair(index, NO_PAIR);`),...new te({id:`${t}-${n}-scan`,input:m,output:h,mode:`inclusive`}).getCommandNodes(e),E(`${n}-segments`,c,[w(`sortedKeys`,l),w(`flags`,m),w(`ranks`,h),T(`offsets`,g),T(`pairKeys`,_)],`if (flags[flagsOffset + index] != 0u) {
let segment = ranks[ranksOffset + index] - 1u;
offsets[offsetsOffset + segment] = index;
setPairKeysPair(segment, getSortedKeysPair(index));
}`),ie(e,{id:`${t}-${n}-sum`,operation:L,segmentCount:c,input:p,segmentOffsets:g,output:v})],pairKeys:_,pairWeights:v}};l.push(E(`initialize-state`,me,[T(`state`,O),T(`trace`,n.output.objectiveTrace)],`if (index < arrayLength(&state) - stateOffset) { state[stateOffset + index] = 0u; }
if (index < arrayLength(&trace) - traceOffset) { trace[traceOffset + index] = getNaN(); }`,oe)),l.push(E(`validate-offsets`,U.length,[w(`offsets`,n.topology.offsets),T(`invalid`,U)],`var failed = false;
if (index <= NODE_COUNT) {
let offset = offsets[offsetsOffset + index];
failed = offset > ${s}u || (index == 0u && offset != 0u);
if (index < NODE_COUNT) { failed = failed || offset > offsets[offsetsOffset + index + 1u]; }
}
invalid[invalidOffset + index] = select(0u, 1u, failed);`)),l.push(...new g({id:`${t}-invalid-offset-count`,input:U,output:W,operation:`sum`}).getCommandNodes(e)),l.push(E(`prepare-arcs`,H.length,[w(`offsets`,n.topology.offsets),w(`neighbors`,n.topology.neighbors),w(`weights`,n.topology.weights),w(`invalidCount`,W),T(`sourceKeys`,B),T(`targetKeys`,V),T(`rawWeights`,H),T(`invalid`,U)],`sourceKeys[sourceKeysOffset + index] = NODE_COUNT;
targetKeys[targetKeysOffset + index] = NODE_COUNT;
rawWeights[rawWeightsOffset + index] = 0.0;
invalid[invalidOffset + index] = 0u;
if (invalidCount[invalidCountOffset] != 0u || index >= ${s}u || index >= offsets[offsetsOffset + NODE_COUNT]) { return; }
let destination = neighbors[neighborsOffset + index];
if (destination >= NODE_COUNT) { return; }
let weight = weights[weightsOffset + index];
if (!isFiniteValue(weight) || weight < 0.0) { invalid[invalidOffset + index] = 1u; return; }
sourceKeys[sourceKeysOffset + index] = upperBoundOffsets(0u, NODE_COUNT + 1u, index) - 1u;
targetKeys[targetKeysOffset + index] = destination;
rawWeights[rawWeightsOffset + index] = weight;`,x(`offsets`)));let be=f(`invalid-weight-count`,1);l.push(...new g({id:`${t}-invalid-weight-count`,input:U,output:be,operation:`sum`}).getCommandNodes(e)),l.push(...A(`total-weight`,H,G));for(let[n,r,i]of[[`out`,B,N],[`in`,V,P]])l.push(...y(e,{id:`${t}-initial-${n}`,operation:L,segmentCount:a,segmentKeys:r,sumContributions:H,sums:i}));l.push(E(`initialize-vertices`,a,[T(`degreeOut`,N),T(`degreeIn`,P),w(`total`,G),T(`labels`,j),T(`groups`,M),T(`communities`,n.output.communities),T(`totalsOut`,F),T(`totalsIn`,I)],`let mass = total[totalOffset];
var outgoing = 0.0; var incoming = 0.0;
if (mass > 0.0 && isFiniteValue(mass)) { outgoing = degreeOut[degreeOutOffset + index] / mass; incoming = degreeIn[degreeInOffset + index] / mass; }
degreeOut[degreeOutOffset + index] = outgoing; degreeIn[degreeInOffset + index] = incoming;
totalsOut[totalsOutOffset + index] = outgoing; totalsIn[totalsInOffset + index] = incoming;
labels[labelsOffset + index] = index; groups[groupsOffset + index] = index; communities[communitiesOffset + index] = index;`)),l.push(E(`initialize-edges`,c,[w(`sourceKeys`,B),w(`targetKeys`,V),w(`rawWeights`,H),w(`total`,G),T(`edgeKeys`,R),T(`edgeWeights`,z)],`setEdgeKeysPair(index, NO_PAIR); edgeWeights[edgeWeightsOffset + index] = 0.0;
let slot = index / 2u;
if (slot >= ${s}u) { return; }
let source = sourceKeys[sourceKeysOffset + slot]; let destination = targetKeys[targetKeysOffset + slot];
let mass = total[totalOffset];
if (rawWeights[rawWeightsOffset + slot] == 0.0 || source >= NODE_COUNT || destination >= NODE_COUNT || !(mass > 0.0) || !isFiniteValue(mass)) { return; }
var first = source; var second = destination;
if (index % 2u == 1u) { first = destination; second = source; }
setEdgeKeysPair(index, vec2<u32>(first, second));
edgeWeights[edgeWeightsOffset + index] = rawWeights[rawWeightsOffset + slot] / mass;`));let xe=f(`symmetry-invalid-count`,1);if(n.directed)l.push(E(`symmetry-skip`,1,[T(`symmetryInvalidCount`,xe)],`symmetryInvalidCount[symmetryInvalidCountOffset] = 0u;`));else{l.push(E(`symmetry-keys`,c,[w(`sourceKeys`,B),w(`targetKeys`,V),w(`rawWeights`,H),T(`proposalKeys`,Z),T(`proposalWeights`,Q)],`setProposalKeysPair(index, NO_PAIR); proposalWeights[proposalWeightsOffset + index] = 0.0;
if (index >= ${s}u) { return; }
let source = sourceKeys[sourceKeysOffset + index]; let destination = targetKeys[targetKeysOffset + index];
if (source >= NODE_COUNT || destination >= NODE_COUNT) { return; }
setProposalKeysPair(index, vec2<u32>(source, destination));
proposalWeights[proposalWeightsOffset + index] = rawWeights[rawWeightsOffset + index];`));let n=$(`symmetry`,Z,Q),r=f(`symmetry-invalid`,c);l.push(...n.commands,E(`symmetry-validate`,c,[w(`pairKeys`,n.pairKeys),w(`pairWeights`,n.pairWeights),T(`failures`,r)],`failures[failuresOffset + index] = 0u;
let key = getPairKeysPair(index); if (key.x == NODE_COUNT) { return; }
let reverseKey = key.yx;
let position = lowerBoundPairKeys(0u, RECORD_COUNT, reverseKey);
var reverseWeight = 0.0;
if (position < RECORD_COUNT && all(getPairKeysPair(position) == reverseKey)) { reverseWeight = pairWeights[pairWeightsOffset + position]; }
let weight = pairWeights[pairWeightsOffset + index];
if (abs(weight - reverseWeight) > 0.000001 * max(abs(weight), abs(reverseWeight))) { failures[failuresOffset + index] = 1u; }`,D),...new g({id:`${t}-symmetry-invalid-count`,input:r,output:xe,operation:`sum`}).getCommandNodes(e))}l.push(E(`initialize-control`,1,[T(`state`,O),w(`parameters`,n.parameters),w(`invalidCount`,W),w(`invalidWeights`,be),w(`symmetryInvalidCount`,xe),w(`total`,G)],`let resolution = parameters[parametersOffset]; let tolerance = parameters[parametersOffset + 1u];
let valid = symmetryInvalidCount[symmetryInvalidCountOffset] == 0u && invalidCount[invalidCountOffset] == 0u && invalidWeights[invalidWeightsOffset] == 0u && isFiniteValue(total[totalOffset]) && isFiniteValue(resolution) && resolution >= 0.0 && isFiniteValue(tolerance) && tolerance >= 0.0;
state[stateOffset + 4u] = select(0u, 1u, valid);
state[stateOffset] = select(0u, 1u, valid && total[totalOffset] > 0.0);
state[stateOffset + 1u] = 1u;
state[stateOffset + 5u] = select(0u, 1u, valid && total[totalOffset] == 0.0);`)),k(`initial-objective`,ye(`initial`,!0),`state[stateOffset + 4u] != 0u`);let Se=n.maximumLevels??8,Ce=n.maximumIterations??16;for(let r=0;r<Se;r++){for(let i=0;i<Ce;i++){let o=`level-${r}-round-${i}`,s=[E(`${o}-reset`,a,[T(`proposals`,J),T(`reservations`,Y)],`proposals[proposalsOffset + index] = NONE; reservations[reservationsOffset + index] = NONE;`),E(`${o}-neighbor-communities`,c,[w(`edgeKeys`,R),w(`edgeWeights`,z),w(`labels`,j),T(`proposalKeys`,Z),T(`proposalWeights`,Q)],`let key = getEdgeKeysPair(index);
setProposalKeysPair(index, NO_PAIR); proposalWeights[proposalWeightsOffset + index] = 0.0;
if (key.x == NODE_COUNT) { return; }
let source = key.x; let destination = key.y;
setProposalKeysPair(index, vec2<u32>(source, labels[labelsOffset + destination]));
// A self-loop is invariant under a move and must not enter its connection gain.
if (source != destination) { proposalWeights[proposalWeightsOffset + index] = edgeWeights[edgeWeightsOffset + index]; }`)],u=$(`${o}-candidates`,Z,Q);s.push(...u.commands);let d=C(`${o}-gains`,c),p=f(`${o}-node-keys`,c),m=C(`${o}-best-gain`,a),h=f(`${o}-node-offsets`,a+1),ee=C(`${o}-strengths`,a*4);s.push(E(`${o}-strengths`,a,[w(`degreeOut`,N),w(`degreeIn`,P),w(`totalsOut`,F),w(`totalsIn`,I),T(`strengths`,ee)],`strengths[strengthsOffset + index * 4u] = degreeOut[degreeOutOffset + index];
strengths[strengthsOffset + index * 4u + 1u] = degreeIn[degreeInOffset + index];
strengths[strengthsOffset + index * 4u + 2u] = totalsOut[totalsOutOffset + index];
strengths[strengthsOffset + index * 4u + 3u] = totalsIn[totalsInOffset + index];`)),s.push(E(`${o}-gain`,c,[w(`pairKeys`,u.pairKeys),w(`pairWeights`,u.pairWeights),w(`labels`,j),w(`strengths`,ee),w(`parameters`,n.parameters),T(`gains`,d),T(`nodeKeys`,p)],`let key = getPairKeysPair(index);
gains[gainsOffset + index] = 0.0; nodeKeys[nodeKeysOffset + index] = NODE_COUNT;
if (key.x == NODE_COUNT) { return; }
let source = key.x; let destination = key.y; let original = labels[labelsOffset + source];
nodeKeys[nodeKeysOffset + index] = source;
if (destination == original) { return; }
let originalKey = vec2<u32>(source, original);
let position = lowerBoundPairKeys(0u, RECORD_COUNT, originalKey);
var connection = 0.0;
if (position < RECORD_COUNT && all(getPairKeysPair(position) == originalKey)) { connection = pairWeights[pairWeightsOffset + position]; }
let outgoing = strengths[strengthsOffset + source * 4u]; let incoming = strengths[strengthsOffset + source * 4u + 1u];
let penalty = outgoing * (strengths[strengthsOffset + destination * 4u + 3u] - strengths[strengthsOffset + original * 4u + 3u] + incoming) + incoming * (strengths[strengthsOffset + destination * 4u + 2u] - strengths[strengthsOffset + original * 4u + 2u] + outgoing);
let gain = pairWeights[pairWeightsOffset + index] - connection - parameters[parametersOffset] * penalty;
if (isFiniteValue(gain) && gain > parameters[parametersOffset + 1u]) { gains[gainsOffset + index] = gain; }`,D)),s.push(...b(e,{id:`${t}-${o}-nodes`,operation:L,groupCount:a,sortedGroupKeys:p,segmentOffsets:h})),s.push(ie(e,{id:`${t}-${o}-best-gain`,operation:L,segmentCount:a,input:d,segmentOffsets:h,output:m,reduction:`max`})),s.push(E(`${o}-choose`,c,[w(`pairKeys`,u.pairKeys),w(`gains`,d),w(`bestGain`,m),T(`proposals`,J,!0)],`let key = getPairKeysPair(index); if (key.x == NODE_COUNT) { return; }
let source = key.x; let gain = gains[gainsOffset + index];
if (gain > 0.0 && gain == bestGain[bestGainOffset + source]) { atomicMin(&proposals[proposalsOffset + source], key.y); }`)),s.push(E(`${o}-reserve`,a,[w(`proposals`,J),w(`labels`,j),T(`reservations`,Y,!0)],`let destination = proposals[proposalsOffset + index]; if (destination == NONE) { return; }
let priority = index ^ ${n.seed??0}u;
atomicMin(&reservations[reservationsOffset + labels[labelsOffset + index]], priority);
atomicMin(&reservations[reservationsOffset + destination], priority);`)),s.push(E(`${o}-accept`,a,[w(`proposals`,J),w(`labels`,j),w(`reservations`,Y),T(`accepted`,X),T(`changed`,K)],`let proposedCommunity = proposals[proposalsOffset + index]; let originalCommunity = labels[labelsOffset + index]; let priority = index ^ ${n.seed??0}u;
var acceptedCommunity = originalCommunity;
if (proposedCommunity != NONE) {
  if (reservations[reservationsOffset + originalCommunity] == priority && reservations[reservationsOffset + proposedCommunity] == priority) { acceptedCommunity = proposedCommunity; }
}
accepted[acceptedOffset + index] = acceptedCommunity; changed[changedOffset + index] = select(0u, 1u, acceptedCommunity != originalCommunity);`)),s.push(E(`${o}-apply`,a,[w(`accepted`,X),w(`groups`,M),T(`labels`,j),T(`communities`,n.output.communities)],`labels[labelsOffset + index] = accepted[acceptedOffset + index]; communities[communitiesOffset + index] = accepted[acceptedOffset + groups[groupsOffset + index]];`)),s.push(...new g({id:`${t}-${o}-changed`,input:K,output:_e,operation:`max`}).getCommandNodes(e)),s.push(...ve(`${o}-totals`),...ye(o,!1)),k(o,s,`state[stateOffset] != 0u && state[stateOffset + 1u] != 0u`),l.push(E(`${o}-control`,1,[T(`state`,O),w(`anyChanged`,_e),w(`roundBudget`,le),w(`levelBudget`,ue)],`if (state[stateOffset] == 0u || state[stateOffset + 1u] == 0u) { return; }
state[stateOffset + 6u] = state[stateOffset + 9u] + 1u; state[stateOffset + 7u] = state[stateOffset + 7u] + 1u;
state[stateOffset + 8u] = state[stateOffset + 8u] + 1u;
if (anyChanged[anyChangedOffset] == 0u) {
  state[stateOffset + 1u] = 0u;
  if (state[stateOffset + 2u] == 0u) { state[stateOffset + 5u] = 1u; state[stateOffset] = 0u; }
  if (state[stateOffset + 6u] >= arrayLength(&levelBudget) - levelBudgetOffset) { state[stateOffset] = 0u; }
} else {
  state[stateOffset + 2u] = 1u;
  if (state[stateOffset + 8u] >= arrayLength(&roundBudget) - roundBudgetOffset) { state[stateOffset] = 0u; }
}`))}if(r+1<Se){let e=`aggregate-${r}`,t=[E(`${e}-keys`,c,[w(`edgeKeys`,R),w(`labels`,j),T(`proposalKeys`,Z)],`let key = getEdgeKeysPair(index); var next = NO_PAIR;
if (key.x != NODE_COUNT) { next = vec2<u32>(labels[labelsOffset + key.x], labels[labelsOffset + key.y]); }
setProposalKeysPair(index, next);`)],i=$(e,Z,z);t.push(...i.commands),t.push(E(`${e}-edges`,c,[w(`pairKeys`,i.pairKeys),w(`pairWeights`,i.pairWeights),T(`edgeKeys`,R),T(`edgeWeights`,z)],`setEdgeKeysPair(index, getPairKeysPair(index)); edgeWeights[edgeWeightsOffset + index] = pairWeights[pairWeightsOffset + index];`)),t.push(E(`${e}-vertices`,a,[w(`totalsOut`,F),w(`totalsIn`,I),w(`communities`,n.output.communities),T(`degreeOut`,N),T(`degreeIn`,P),T(`groups`,M),T(`labels`,j)],`degreeOut[degreeOutOffset + index] = totalsOut[totalsOutOffset + index]; degreeIn[degreeInOffset + index] = totalsIn[totalsInOffset + index]; groups[groupsOffset + index] = communities[communitiesOffset + index]; labels[labelsOffset + index] = index;`)),k(e,t,`state[stateOffset] != 0u && state[stateOffset + 1u] == 0u`),l.push(E(`${e}-control`,1,[T(`state`,O)],`if (state[stateOffset] != 0u && state[stateOffset + 1u] == 0u) { state[stateOffset + 1u] = 1u; state[stateOffset + 2u] = 0u; state[stateOffset + 8u] = 0u; state[stateOffset + 9u] = state[stateOffset + 9u] + 1u; }`))}}return l.push(E(`publish-summary`,1,[w(`state`,O),T(`summary`,n.output.summary)],`summary[summaryOffset] = state[stateOffset + 4u]; summary[summaryOffset + 1u] = state[stateOffset + 5u]; summary[summaryOffset + 2u] = state[stateOffset + 6u]; summary[summaryOffset + 3u] = state[stateOffset + 7u];`)),l}var V=32,H=32,U=64,W=2,G=4;function K(e,t){return{declarations:`var<private> louvainVisitOrder: array<u32, ${e}> = array<u32, ${e}>(${Array.from({length:e},(e,t)=>`${t}u`).join(`, `)});`,body:`var visitStream = createPhiloxStream(vec2<u32>(${t}u, ${C.networkLouvainVisitOrder.id}u), 0u, 0u, 0u);
for (var remaining = ${e}u; remaining > 1u; remaining--) {
  let selected = nextPhiloxBelow(&visitStream, remaining);
  let temporary = louvainVisitOrder[remaining - 1u];
  louvainVisitOrder[remaining - 1u] = louvainVisitOrder[selected];
  louvainVisitOrder[selected] = temporary;
}`}}var _e=class{id;props;estimatedCommandNodeCount;constructor(t){this.id=t.id??`network-louvain`,this.props=t;let{id:n}=this;l(n,!!t.weights!=!!t.topology,`exactly one of weights or topology must be supplied`);for(let[e,r]of Object.entries(t.output.status??{}))r&&(s(r,[`uint32`],`${n} output.status.${e}`),l(n,r.length===1,`output.status.${e} must hold one row`));for(let[e,r,i]of[[`communities`,t.output.communities,`uint32`],[`objectiveTrace`,t.output.objectiveTrace,`float32`],[`summary`,t.output.summary,`uint32`]])s(r,[i],`${n} output.${e}`);let r=q(t.output),i=r.map(e=>e.buffer.defaultBuffer?c(e.buffer.defaultBuffer):e.buffer);if(l(n,new Set(i).size===r.length,`output views must use distinct buffers, including summary and status ports`),this.estimatedCommandNodeCount=t.topology?R(t):2,t.topology){z(n,t,this.estimatedCommandNodeCount);return}u(n,`nodeCount`,t.nodeCount,1,V),l(n,t.seed===void 0||Number.isInteger(t.seed)&&t.seed>=0&&t.seed<=4294967295,`seed must be an integer in [0, 4294967295]`);let a=t.maximumLevels??8,o=t.maximumIterations??16;u(n,`maximumLevels`,a,1,H),u(n,`maximumIterations`,o,1,U),s(t.weights,[`float32`],this.id),s(t.parameters,[`float32`],this.id),s(t.output.communities,[`uint32`],this.id),s(t.output.objectiveTrace,[`float32`],this.id),s(t.output.summary,[`uint32`],this.id),l(n,t.weights.length===t.nodeCount**2,`weights must hold nodeCount squared rows`),l(n,t.parameters.length===W,`parameters must hold ${W} rows: resolution and minimum improvement`),l(n,t.output.communities.length===t.nodeCount,`output.communities must hold nodeCount rows`),l(n,t.output.objectiveTrace.length===1+a*o,`output.objectiveTrace must hold 1 + maximumLevels * maximumIterations rows`),l(n,t.output.summary.length===G,`output.summary must hold ${G} rows`),e(this.id,q(t.output),[t.weights,t.parameters])}getCommandNodes(e){let{id:t,props:n}=this;if(n.topology)return[...B(e,t,n),...J(e,t,n.output)];de(e,t,n),p(e.device,t,`dense level budget`,(n.maximumLevels??8)*4,`reduce maximumLevels`),p(e.device,t,`dense round budget`,(n.maximumIterations??16)*4,`reduce maximumIterations`),m(t,e,[n.weights,n.parameters,...q(n.output)]);let i=r(e,`${t}-level-budget`,`uint32`,n.maximumLevels??8),a=r(e,`${t}-round-budget`,`uint32`,n.maximumIterations??16);return[v(e,{id:t,operation:`GPUNetworkLouvain`,variant:`bounded-coarsened-local-moves`,invocationCount:1,workgroupSize:2,bindings:[{name:`weights`,view:n.weights,type:`f32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`communities`,view:n.output.communities,type:`u32`,access:`read_write`},{name:`trace`,view:n.output.objectiveTrace,type:`f32`,access:`read_write`},{name:`summary`,view:n.output.summary,type:`u32`,access:`read_write`},{name:`levelBudget`,view:i,type:`u32`,access:`read`},{name:`roundBudget`,view:a,type:`u32`,access:`read`}],declarations:`${oe}
${n.seed===void 0?``:`${S}
${K(n.nodeCount,n.seed).declarations}`}
const NODE_COUNT: u32 = ${n.nodeCount}u;
${se}
fn objective(total: f32, resolution: f32) -> f32 {
  var internal = 0.0;
  var strengths: array<f32, ${n.nodeCount}>;
  var incoming: array<f32, ${n.nodeCount}>;
  for (var source = 0u; source < NODE_COUNT; source++) {
    let sourceCommunity = communities[communitiesOffset + source];
    for (var destination = 0u; destination < NODE_COUNT; destination++) {
      let weight = weights[weightsOffset + source * NODE_COUNT + destination];
      strengths[sourceCommunity] += weight;
      incoming[communities[communitiesOffset + destination]] += weight;
      if (sourceCommunity == communities[communitiesOffset + destination]) { internal += weight; }
    }
  }
  var penalty = 0.0;
  for (var community = 0u; community < NODE_COUNT; community++) { penalty += (strengths[community] / total) * (incoming[community] / total); }
  return internal / total - resolution * penalty;
}`,body:`for (var word = 0u; word < 4u; word++) { summary[summaryOffset + word] = 0u; }
for (var entry = 0u; entry < arrayLength(&trace) - traceOffset; entry++) { trace[traceOffset + entry] = getNaN(); }
var groups: array<u32, ${n.nodeCount}>;
${n.seed===void 0?``:K(n.nodeCount,n.seed).body}
for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { communities[communitiesOffset + vertex] = vertex; groups[vertex] = vertex; }
let resolution = parameters[parametersOffset];
let tolerance = parameters[parametersOffset + 1u];
if (!isFiniteValue(resolution) || resolution < 0.0 || !isFiniteValue(tolerance) || tolerance < 0.0) { return; }
var total = 0.0;
for (var source = 0u; source < NODE_COUNT; source++) {
  for (var destination = 0u; destination < NODE_COUNT; destination++) {
    let weight = weights[weightsOffset + source * NODE_COUNT + destination];
    if (!isFiniteValue(weight) || weight < 0.0${n.directed?``:` || weight != weights[weightsOffset + destination * NODE_COUNT + source]`}) { return; }
    total += weight;
  }
}
if (!isFiniteValue(total)) { return; }
summary[summaryOffset] = 1u;
if (total == 0.0) { trace[traceOffset] = 0.0; summary[summaryOffset + 1u] = 1u; return; }
var current = objective(total, resolution);
if (!isFiniteValue(current)) { summary[summaryOffset] = 0u; return; }
trace[traceOffset] = current;
var sweeps = 0u;
for (var level = 0u; level < arrayLength(&levelBudget) - levelBudgetOffset; level++) {
  var levelChanged = false;
  var stable = false;
  for (var round = 0u; round < arrayLength(&roundBudget) - roundBudgetOffset; round++) {
    var changed = false;
    for (var visit = 0u; visit < NODE_COUNT; visit++) {
      let group = ${n.seed===void 0?`visit`:`louvainVisitOrder[visit]`};
      var present = false;
      var original = 0u;
      for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { if (groups[vertex] == group) { present = true; original = communities[communitiesOffset + vertex]; } }
      if (!present) { continue; }
      var best = original;
      var bestObjective = current;
      for (var candidate = 0u; candidate < NODE_COUNT; candidate++) {
        // Only existing neighboring communities are Louvain local-move candidates.
        var adjacent = false;
        for (var source = 0u; source < NODE_COUNT; source++) {
          if (groups[source] != group) { continue; }
          for (var destination = 0u; destination < NODE_COUNT; destination++) {
            if (groups[destination] != group && communities[communitiesOffset + destination] == candidate && (weights[weightsOffset + source * NODE_COUNT + destination] > 0.0 || weights[weightsOffset + destination * NODE_COUNT + source] > 0.0)) { adjacent = true; }
          }
        }
        if (!adjacent || candidate == original) { continue; }
        for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { if (groups[vertex] == group) { communities[communitiesOffset + vertex] = candidate; } }
        let candidateObjective = objective(total, resolution);
        if (candidateObjective > bestObjective + tolerance) { best = candidate; bestObjective = candidateObjective; }
        for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { if (groups[vertex] == group) { communities[communitiesOffset + vertex] = original; } }
      }
      for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { if (groups[vertex] == group) { communities[communitiesOffset + vertex] = best; } }
      if (best != original) { changed = true; current = bestObjective; }
    }
    sweeps++;
    trace[traceOffset + sweeps] = current;
    levelChanged = levelChanged || changed;
    if (!changed) { stable = true; break; }
  }
  summary[summaryOffset + 2u] = level + 1u;
  summary[summaryOffset + 3u] = sweeps;
  if (!stable) { return; }
  if (!levelChanged) { summary[summaryOffset + 1u] = 1u; return; }
  // An implicit quotient graph preserves every original weighted edge exactly.
  for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { groups[vertex] = communities[communitiesOffset + vertex]; }
}`}),...J(e,t,n.output)]}};function q(e){return[e.communities,e.objectiveTrace,e.summary,...Object.values(e.status??{}).filter(e=>!!e)]}function J(e,t,n){let r=Object.entries(n.status??{}).filter(e=>!!e[1]);if(!r.length)return[];let i={converged:`summary[summaryOffset + 1u]`,iterationCount:`summary[summaryOffset + 3u]`,iterationLimitReached:`select(0u, 1u, summary[summaryOffset] != 0u && summary[summaryOffset + 1u] == 0u)`,invalidCount:`select(1u, 0u, summary[summaryOffset] != 0u)`,approximation:`1u`};return[v(e,{id:`${t}-status`,operation:`GPUNetworkLouvain`,invocationCount:1,bindings:[{name:`summary`,view:n.summary,type:`u32`,access:`read`},...r.map(([e,t])=>({name:e,view:t,type:`u32`,access:`read_write`}))],body:r.map(([e])=>`${e}[${e}Offset] = ${i[e]};`).join(`
`)})]}var Y=O(k,`resolution`),X=O(k,`minimumJobs`),Z=32,Q=w({parameters:k,caption:`Commuter communities among 32 Chicago tracts, weighted by jobs flowing between them.`,createInputs:({world:e})=>{let t=new Float64Array(T.length);for(let[e,n,r]of E)t[e]+=r,t[n]+=r;let n=Array.from(t.keys()).sort((e,n)=>t[n]-t[e]).slice(0,Z),r=new Float32Array(1024);for(let[e,t,i]of E){let a=n.indexOf(e),o=n.indexOf(t);a<0||o<0||(r[a*Z+o]+=i,r[o*Z+a]+=i)}let i=ce(e);return{pairs:r,positions:ue(Float32Array.from(n.flatMap(e=>[i[e*2],i[e*2+1]])),e),weights:new Float32Array(1024),lines:new Float32Array(4096),lineCount:{value:0}}},build(e,t){let n=e.allocate(`weights`,`float32`,1024),r=e.allocate(`parameters`,`float32`,2),i=e.allocate(`communities`,`uint32`,Z),a=e.allocate(`objective-trace`,`float32`,129),o=e.allocate(`summary`,`uint32`,4);e.graph.add(new _e({nodeCount:Z,seed:4332,weights:n.view,parameters:r.view,output:{communities:i.view,objectiveTrace:a.view,summary:o.view}}));let s=e.upload(`ties`,`float32x4`,t.lines);return{outputs:{communities:i,ties:s,positions:e.upload(`positions`,`float32x2`,t.positions)},update:e=>{let i=0;for(let n=0;n<Z;n++)for(let r=0;r<Z;r++){let a=t.pairs[n*Z+r],o=a>=e.minimumJobs?a:0;t.weights[n*Z+r]=o,o>0&&n<r&&t.lines.set([t.positions[n*2],t.positions[n*2+1],t.positions[r*2],t.positions[r*2+1]],i++*4)}t.lines.fill(-100,i*4),t.lineCount.value=i,n.write(t.weights),s.write(t.lines),r.write(Float32Array.of(e.resolution,1e-6))}}},interaction:{kind:`pad`,hint:`tune the communities`,rest:[D(Y,Y.default),1-D(X,X.default)],resolve(e,{outputs:t,peek:n}){let r=le(Y,e.unitX),i=le(X,1-e.unitY),a=n(t.communities),o=a?new Set(a).size:0;return{values:{resolution:r,minimumJobs:i},readout:`resolution ${r.toFixed(1)} · minimumJobs ${i} · ${o} communities`}}},draw:({outputs:e})=>[{layer:`lines`,context:!0,segments:e.ties,radius:1.2},{layer:`dots`,positions:e.positions,values:e.communities,radius:9}]});export{Q as default};