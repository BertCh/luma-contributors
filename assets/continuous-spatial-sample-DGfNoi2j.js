import{a as e}from"./graph-chunk-dispatch-zBt8k2-o.js";import{c as t,o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{v as r}from"./spatial-context-DwZEd4EZ.js";import{D as i,E as a,T as o,f as s,x as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{n as u,t as d}from"./philox-random-DfVxWNPn.js";import{u as f}from"./wgsl-common-DfyXiYB0.js";import{t as p}from"./define-run-B-brkCe4.js";import{u as m}from"./inputs-CJKKIRKK.js";import{$s as h}from"./entry-preview-D70TdSbK.js";var g=4096,_=4096,v=65536,y=65536,b=1048576,x=256,S=class{props;id;spatialContext;samplingIntensity;provenance;constructor(e){this.props=e;let t=e.id??`continuous-spatial-sample`;this.id=t,this.spatialContext=r(t,e.spatialContext);let{frame:n}=e;a(n.positions,[`float32x2`],t),s(t,`frame.positions rows`,n.positions.length,n.type===`area`?3:2,g),o(t,n.type!==`area`||n.positions.length%3==0,`area frame.positions must hold whole triangles (a multiple of 3 rows)`),o(t,Number.isFinite(Math.fround(e.spacing))&&Math.fround(e.spacing)>0&&Number.isFinite(Math.fround(e.spacing**2))&&Math.fround(e.spacing**2)>0,`spacing must be positive with a positive finite float32 square`);let c=e.waves??1;if(s(t,`waves`,c,1,_),s(t,`capacity`,e.capacity,1,v),s(t,`maximumCandidates`,e.maximumCandidates??y,1,b),s(t,`seed`,e.seed,0,4294967295),s(t,`replicate`,e.replicate??0,0,4294967295-c+1),n.legacySites&&(a(n.legacySites,[n.type===`line`?`float32`:`float32x2`],t),s(t,`frame.legacySites rows`,n.legacySites.length,0,x)),n.type===`area`){o(t,n.bounds.every(e=>Number.isFinite(Math.fround(e))),`frame.bounds must be finite in float32`),o(t,n.bounds[2]>n.bounds[0]&&n.bounds[3]>n.bounds[1],`frame.bounds must have a positive extent on both axes`);let r=Math.ceil((n.bounds[2]-n.bounds[0])/e.spacing)+1,i=Math.ceil((n.bounds[3]-n.bounds[1])/e.spacing)+1;o(t,Number.isSafeInteger(r*i)&&r*i<4294967295,`frame.bounds divided by spacing must give fewer than 2^32 lattice cells`)}e.parameters&&(a(e.parameters,[`uint32`],t),i(t,`parameters`,e.parameters,2));for(let[n,r]of[[e.output.positions,`float32x2`],[e.output.designWeights,`float32`],[e.output.provenance,`uint32x4`],[e.output.status,`uint32x4`]])a(n,[r],t);o(t,e.output.positions.length===c*e.capacity&&e.output.designWeights.length===c*e.capacity&&e.output.provenance.length===c*e.capacity&&e.output.status.length===c,`output views must hold waves times capacity rows (status: waves rows)`),l(t,Object.values(e.output),[n.positions,n.legacySites,e.parameters]),o(t,new Set(Object.values(e.output).map(e=>e.buffer)).size===4,`output views must use four distinct buffers`),this.samplingIntensity=1/(n.type===`line`?e.spacing:e.spacing**2),this.provenance=Object.freeze({design:`random-shift-lattice`,frame:n.type,separationMetric:n.type===`line`?`arc-length`:`euclidean`,randomPhaseBits:24,replacement:`independent-complete-waves`,generator:`Philox4x32-10`})}getCommandNodes(r){let{props:i,id:a}=this,{frame:o}=i;c(a,r,[o.positions,o.legacySites,i.parameters,...Object.values(i.output)]);let s=o.legacySites?.length??0;e(r,a,{framePositions:o.positions,legacySites:o.legacySites,parameters:i.parameters,...i.output});let l=o.type===`area`,p=l?Math.ceil((o.bounds[2]-o.bounds[0])/i.spacing)+1:0,m=l?Math.ceil((o.bounds[3]-o.bounds[1])/i.spacing)+1:0;return[n(r,{id:a,operation:`GPUContinuousSpatialSample`,variant:o.type,invocationCount:i.waves??1,bindings:[{name:`framePositions`,view:o.positions,type:`f32`,access:`read`},...o.legacySites?[{name:`legacy`,view:o.legacySites,type:`f32`,access:`read`}]:[],...i.parameters?[{name:`parameters`,view:i.parameters,type:`u32`,access:`read`}]:[],{name:`positions`,view:i.output.positions,type:`f32`,access:`read_write`},{name:`weights`,view:i.output.designWeights,type:`f32`,access:`read_write`},{name:`provenance`,view:i.output.provenance,type:`u32`,access:`read_write`},{name:`status`,view:i.output.status,type:`u32`,access:`read_write`}],declarations:`${u}
const SAMPLE_SPACING: f32 = ${t(i.spacing)};
const SAMPLE_CAPACITY: u32 = ${i.capacity}u;
${f}
fn samplePosition(row:u32)->vec2f{let offset=framePositionsOffset+2u*row;return vec2f(framePositions[offset],framePositions[offset+1u]);}
fn sampleCross(first:vec2f,second:vec2f)->f32{return first.x*second.y-first.y*second.x;}
fn samplePrimitive(point:vec2f)->u32 {
${l?`for(var triangle=0u;triangle<${o.positions.length/3}u;triangle++){
 let first=samplePosition(triangle*3u);let second=samplePosition(triangle*3u+1u);let third=samplePosition(triangle*3u+2u);
 let orientation=sign(sampleCross(second-first,third-first));
 if(sampleCross(second-first,point-first)*orientation>=0.0&&sampleCross(third-second,point-second)*orientation>=0.0&&sampleCross(first-third,point-third)*orientation>=0.0){return triangle;}
}`:``}
 return 0xffffffffu;
}
fn sampleLine(measure:f32)->vec3f {
 var consumed=0.0;
 for(var segment=0u;segment<${o.positions.length-1}u;segment++){
  let first=samplePosition(segment);let second=samplePosition(segment+1u);let extent=distance(first,second);
  if(extent>0.0&&measure<=consumed+extent){return vec3f(mix(first,second,clamp((measure-consumed)/extent,0.0,1.0)),f32(segment));}consumed+=extent;
 }
 return vec3f(samplePosition(${o.positions.length-1}u),f32(${o.positions.length-2}u));
}
${o.legacySites?`fn sampleLegacy(row:u32)->vec3f{${l?`let offset=legacyOffset+row*2u;return vec3f(legacy[offset],legacy[offset+1u],0.0);`:`return vec3f(sampleLine(legacy[legacyOffset+row]).xy,legacy[legacyOffset+row]);`}}`:``}
fn sampleWrite(wave:u32,slot:u32,point:vec2f,weight:f32,source:u32,replicate:u32,ordinal:u32,role:u32){
 if(slot>=SAMPLE_CAPACITY){return;}let row=wave*SAMPLE_CAPACITY+slot;
 positions[positionsOffset+row*2u]=point.x;positions[positionsOffset+row*2u+1u]=point.y;weights[weightsOffset+row]=weight;
 provenance[provenanceOffset+row*4u]=source;provenance[provenanceOffset+row*4u+1u]=replicate;provenance[provenanceOffset+row*4u+2u]=ordinal;provenance[provenanceOffset+row*4u+3u]=role;
}`,body:`var failure=0u;var required=0u;var randomAccepted=0u;var measure=0.0;
let seed=${i.parameters?`parameters[parametersOffset]`:`${i.seed}u`};let initialReplicate=${i.parameters?`parameters[parametersOffset+1u]`:`${i.replicate??0}u`};
if(initialReplicate>0xffffffffu-index){failure=1u;}
let replicate=initialReplicate+index;
let words=philox4x32(vec4u(replicate,0u,0u,0u),vec2u(seed,${d.continuousSpatialSampling.id}u));
let phase=vec2f(philoxUnitFloat(words.x),philoxUnitFloat(words.y))*SAMPLE_SPACING;
for(var vertex=0u;vertex<${o.positions.length}u;vertex++){
 let point=samplePosition(vertex);if(!isFiniteValue(point.x)||!isFiniteValue(point.y)||max(abs(point.x),abs(point.y))*0.00000095367431640625>SAMPLE_SPACING){failure=1u;}
 ${l?`if(any(point<vec2f(${o.bounds.slice(0,2).map(t).join(`,`)}))||any(point>vec2f(${o.bounds.slice(2).map(t).join(`,`)}))){failure=1u;}`:`if(vertex+1u<${o.positions.length}u){measure+=distance(point,samplePosition(vertex+1u));}`}
}
${l?`for(var triangle=0u;triangle<${o.positions.length/3}u;triangle++){let first=samplePosition(triangle*3u);let determinant=sampleCross(samplePosition(triangle*3u+1u)-first,samplePosition(triangle*3u+2u)-first);if(!isFiniteValue(determinant)||determinant==0.0){failure=1u;}}`:`if(!isFiniteValue(measure)||measure<=0.0){failure=1u;}`}
${o.legacySites?`for(var fixed=0u;fixed<${s}u;fixed++){
 let point=sampleLegacy(fixed);
 if(!isFiniteValue(point.x)||!isFiniteValue(point.y)||!isFiniteValue(point.z)${l?`||samplePrimitive(point.xy)==0xffffffffu`:`||point.z<0.0||point.z>measure`}){failure=1u;}
 for(var previous=0u;previous<fixed;previous++){let other=sampleLegacy(previous);if(${l?`distance(point.xy,other.xy)`:`abs(point.z-other.z)`}<SAMPLE_SPACING){failure=1u;}}
}`:``}
if(failure==0u){
 ${o.legacySites?`for(var fixed=0u;fixed<${s}u;fixed++){sampleWrite(index,required,sampleLegacy(fixed).xy,0.0,fixed,replicate,fixed,2u);required++;}`:``}
 var candidate=0u;
 loop {
  ${l?`if(candidate>=${p*m}u){break;}`:`let distanceAlong=phase.x+f32(candidate)*SAMPLE_SPACING;if(distanceAlong>=measure){break;}`}
  if(candidate>=${i.maximumCandidates??65536}u){failure|=4u;break;}
  ${l?`let point=vec2f(${o.bounds.slice(0,2).map(t).join(`,`)})+phase+vec2f(f32(candidate%${p}u),f32(candidate/${p}u))*SAMPLE_SPACING;
 let source=samplePrimitive(point);var accepted=source!=0xffffffffu&&all(point<vec2f(${o.bounds.slice(2).map(t).join(`,`)}));`:`let location=sampleLine(distanceAlong);let point=location.xy;let source=u32(location.z);var accepted=true;`}
 ${o.legacySites?`for(var fixed=0u;fixed<${s}u;fixed++){let other=sampleLegacy(fixed);if(${l?`distance(point,other.xy)`:`abs(distanceAlong-other.z)`}<SAMPLE_SPACING){accepted=false;}}`:``}
  if(accepted){sampleWrite(index,required,point,${t(l?i.spacing**2:i.spacing)},source,replicate,candidate,select(1u,0u,index==0u));required++;randomAccepted++;}
  candidate++;
 }
}
if(required>SAMPLE_CAPACITY){failure|=2u;}
let written=min(required,SAMPLE_CAPACITY);
for(var slot=written;slot<SAMPLE_CAPACITY;slot++){sampleWrite(index,slot,vec2f(0.0),0.0,0xffffffffu,replicate,0xffffffffu,0xffffffffu);}
status[statusOffset+index*4u]=written;status[statusOffset+index*4u+1u]=required;status[statusOffset+index*4u+2u]=randomAccepted;status[statusOffset+index*4u+3u]=failure;`})]}},C=4e3,w=[.03,.045,.06,.09,.12],T=p({parameters:h,caption:`Random-shift lattice sites inside the Chicago tract boundaries.`,createInputs:({world:e})=>{let t=m(()=>.5,e,6,0),n=new Float32Array(t.triangleIndices.length*2);t.triangleIndices.forEach((e,r)=>{n[r*2]=t.vertices[e*2],n[r*2+1]=t.vertices[e*2+1]});let[r,i,a,o]=[1/0,1/0,-1/0,-1/0];for(let e=0;e<n.length;e+=2)r=Math.min(r,n[e]),a=Math.max(a,n[e]),i=Math.min(i,n[e+1]),o=Math.max(o,n[e+1]);return{frame:n,outline:t.outline,bounds:[r-.01,i-.01,a+.01,o+.01]}},build(e,{frame:t,outline:n,bounds:r},i){let a=e.upload(`sites`,`float32x2`,new Float32Array(C*2).fill(NaN));return e.graph.add(new S({frame:{type:`area`,positions:e.upload(`frame`,`float32x2`,t).view,bounds:r},spacing:i.spacing,seed:i.seed,capacity:C,output:{positions:a.view,designWeights:e.allocate(`design-weights`,`float32`,C).view,provenance:e.allocate(`provenance`,`uint32x4`,C).view,status:e.allocate(`status`,`uint32x4`,1).view}})),{outputs:{sites:a,outline:e.upload(`outline`,`float32x4`,n),outlineCount:n.length/4}}},buildCache:5,rebuildOn:[`spacing`,`seed`],interaction:{kind:`reach`,hint:`choose the spacing`,rest:[.5,.5],resolve(e,{inputs:t}){let n=w[Math.min(w.length-1,Math.floor(e.unitX*w.length))],[r,i,a,o]=t.bounds,s=[(r+a)/2,(i+o)/2];return{settle:{spacing:n},glyphs:[{shape:`box`,minimum:s,maximum:[s[0]+n,s[1]+n]}],readout:`spacing ${n}`}}},draw:({outputs:e,theme:t})=>[{layer:`lines`,context:!0,segments:e.outline,count:e.outlineCount,radius:1.6,color:[...t.muted,.9]},{layer:`dots`,positions:e.sites,radius:6.5,color:[...t.ink,1]},{layer:`dots`,positions:e.sites,radius:4.5,color:[...t.categories[1],1]}]});export{T as default};