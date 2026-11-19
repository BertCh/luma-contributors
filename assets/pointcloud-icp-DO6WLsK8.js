import{t as e}from"./construction-snapshot-DdkX9Pc0.js";import{c as t,o as n,r,t as i}from"./kernel-C3tfQOn4.js";import{D as a,T as o,f as s,g as c,p as l}from"./gpu-contributor-utils-BIwLQON5.js";import{t as u}from"./symmetric-eigen-wgsl-W1CmtqPF.js";import{u as d}from"./wgsl-common-DfyXiYB0.js";import{i as f,r as p,t as m,u as h}from"./pointcloud-grid-index-Bt2ZJi74.js";import{a as g,i as _,n as v}from"./pointcloud-kernel-nodes-YcR7W84x.js";import{r as y,t as b}from"./pointcloud-local-origin-D2ecC5-G.js";import{a as ee,r as x}from"./pointcloud-chunk-kernel-nodes-0zmunsXp.js";import{a as te}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as S}from"./define-run-B-brkCe4.js";import{a as C}from"./pointer-overlay-Qh8qflcA.js";import{c as w,n as T,o as E,r as D}from"./text-DYSgVV-n.js";import{t as O}from"./convert-CaKqeDF0.js";import{u as k}from"./helpers-DPAd8TjN.js";import{r as A}from"./gpu-command-graph-BdzdmV6u.js";import{n as j,t as M}from"./gpu-dispatch-utils-CbRzFAgT.js";import{Tc as N}from"./entry-preview-D70TdSbK.js";import{o as P,t as F}from"./pointcloud-display-run-BsAozfd0.js";var I=1e-5;function L(e,t){return`${e}
var centre=vec3f(0.0);var centreWeight=0.0;
for(var point=index*256u;point<min((index+1u)*256u,${t}u);point++){
 let weight=readWeights(point);if(weight<=0.0){continue;}
 let normal=safeNormal(readTargetNormals(readMatchIds(point)));if(dot(normal,normal)<0.5){continue;}
 let position=rotation*readSource(point)+translation;centreWeight+=weight;centre+=(position-centre)*(weight/centreWeight);
}
var system:array<f32,36>;var right:array<f32,6>;var count=0u;var totalWeight=0.0;var residualSum=0.0;
for(var point=index*256u;point<min((index+1u)*256u,${t}u);point++){
 let weight=readWeights(point);if(weight<=0.0){continue;}
 let normal=safeNormal(readTargetNormals(readMatchIds(point)));if(dot(normal,normal)<0.5){continue;}
 let position=rotation*readSource(point)+translation;let difference=readMatches(point)-position;
 let rotational=cross(position-centre,normal);let residual=dot(difference,normal);
 var jacobian:array<f32,6>;jacobian[0]=rotational.x;jacobian[1]=rotational.y;jacobian[2]=rotational.z;jacobian[3]=normal.x;jacobian[4]=normal.y;jacobian[5]=normal.z;
 count++;totalWeight+=weight;residualSum+=weight*residual*residual;
 for(var row=0u;row<6u;row++){right[row]+=weight*jacobian[row]*residual;for(var column=0u;column<6u;column++){system[row*6u+column]+=weight*jacobian[row]*jacobian[column];}}
}
let base=index*48u;writeMoments(base,f32(count));writeMoments(base+1u,totalWeight);writeMoments(base+2u,residualSum);
for(var entry=0u;entry<36u;entry++){writeMoments(base+3u+entry,system[entry]);}for(var entry=0u;entry<6u;entry++){writeMoments(base+39u+entry,right[entry]);}
writeMoments(base+45u,centre.x);writeMoments(base+46u,centre.y);writeMoments(base+47u,centre.z);`}function R(e,t,n,r,i){return`${e}
for(var entry=0u;entry<17u;entry++){writeNext(entry,readState(entry));}
var system:array<f32,42>;var count=0.0;var weight=0.0;var residualSum=0.0;var centroid=vec3f(0.0);
for(var batchIndex=0u;batchIndex<${typeof t==`number`?`${t}u`:t};batchIndex++){
 let base=batchIndex*48u;let blockWeight=readMoments(base+1u);if(!(blockWeight>0.0)){continue;}
 count+=readMoments(base);weight+=blockWeight;residualSum+=readMoments(base+2u);
 centroid+=(vec3f(readMoments(base+45u),readMoments(base+46u),readMoments(base+47u))-centroid)*(blockWeight/weight);
}
for(var batchIndex=0u;batchIndex<${typeof t==`number`?`${t}u`:t};batchIndex++){
 let base=batchIndex*48u;if(!(readMoments(base+1u)>0.0)){continue;}
 let offset=vec3f(readMoments(base+45u),readMoments(base+46u),readMoments(base+47u))-centroid;
 // Row-major skew matrix of the offset: transfer * n = cross(offset, n).
 var transfer:array<f32,9>;transfer[1]=-offset.z;transfer[2]=offset.y;transfer[3]=offset.z;transfer[5]=-offset.x;transfer[6]=-offset.y;transfer[7]=offset.x;
 var blockSystem:array<f32,42>;
 for(var row=0u;row<6u;row++){for(var column=0u;column<6u;column++){blockSystem[row*7u+column]=readMoments(base+3u+row*6u+column);}blockSystem[row*7u+6u]=readMoments(base+39u+row);}
 // Rotation rows, then rotation columns: each reads only the translation rows or columns, which do not change.
 for(var row=0u;row<3u;row++){for(var column=0u;column<7u;column++){for(var term=0u;term<3u;term++){blockSystem[row*7u+column]+=transfer[row*3u+term]*blockSystem[(3u+term)*7u+column];}}}
 for(var row=0u;row<6u;row++){for(var column=0u;column<3u;column++){for(var term=0u;term<3u;term++){blockSystem[row*7u+column]+=blockSystem[row*7u+3u+term]*transfer[column*3u+term];}}}
 for(var entry=0u;entry<42u;entry++){system[entry]+=blockSystem[entry];}
}
let residual=sqrt(residualSum/max(weight,1e-30));writeHistory(${typeof n==`number`?`${n}u`:n},vec3f(count,residual,0.0));
if(readState(16u)!=1.0){return;}if(count<6.0){writeNext(16u,2.0);return;}
var scale:array<f32,6>;
for(var row=0u;row<6u;row++){let diagonal=system[row*7u+row];if(!(diagonal>0.0)){writeNext(16u,3.0);return;}scale[row]=1.0/sqrt(diagonal);}
for(var row=0u;row<6u;row++){for(var column=0u;column<6u;column++){system[row*7u+column]=system[row*7u+column]*scale[row]*scale[column];}system[row*7u+6u]*=scale[row];}
for(var column=0u;column<6u;column++){
 var pivot=column;for(var row=column+1u;row<6u;row++){if(abs(system[row*7u+column])>abs(system[pivot*7u+column])){pivot=row;}}
 if(!(abs(system[pivot*7u+column])>${I})){writeNext(16u,3.0);return;}
 for(var entry=column;entry<7u;entry++){let temporary=system[column*7u+entry];system[column*7u+entry]=system[pivot*7u+entry];system[pivot*7u+entry]=temporary;}
 let divisor=system[column*7u+column];for(var entry=column;entry<7u;entry++){system[column*7u+entry]/=divisor;}
 for(var row=0u;row<6u;row++){if(row==column){continue;}let factor=system[row*7u+column];for(var entry=column;entry<7u;entry++){system[row*7u+entry]-=factor*system[column*7u+entry];}}
}
let angular=vec3f(system[6]*scale[0],system[13]*scale[1],system[20]*scale[2]);let centroidTranslation=vec3f(system[27]*scale[3],system[34]*scale[4],system[41]*scale[5]);
if(!all(abs(angular)<=vec3f(3.402823e38))||!all(abs(centroidTranslation)<=vec3f(3.402823e38))){writeNext(16u,3.0);return;}
let angle=length(angular);let axis=angular/max(angle,1e-30);let sine=sin(angle);let halfSine=sin(0.5*angle);let versine=2.0*halfSine*halfSine;
let identity=mat3x3f(vec3f(1,0,0),vec3f(0,1,0),vec3f(0,0,1));
let outer=mat3x3f(axis*axis.x,axis*axis.y,axis*axis.z);
let skew=mat3x3f(vec3f(0,axis.z,-axis.y),vec3f(-axis.z,0,axis.x),vec3f(axis.y,-axis.x,0));
// Rodrigues rotation minus the identity, formed without the cancellation of 1 - cos(angle).
let rotationChange=(outer-identity)*versine+skew*sine;
let deltaRotation=identity+rotationChange;
// A rotation about the centroid is the same rotation about the origin plus a translation.
let deltaTranslation=centroidTranslation-rotationChange*centroid;
let updatedRotation=deltaRotation*rotation;let updatedTranslation=deltaRotation*translation+deltaTranslation;
for(var column=0u;column<3u;column++){for(var row=0u;row<3u;row++){writeNext(column*4u+row,updatedRotation[column][row]);}}
writeNext(12u,updatedTranslation.x);writeNext(13u,updatedTranslation.y);writeNext(14u,updatedTranslation.z);
let step=length(deltaTranslation);let rotationStep=max(max(length(rotationChange[0]),length(rotationChange[1])),length(rotationChange[2]));
if(step<=${typeof r==`number`?Math.fround(r):r}&&rotationStep<=${typeof i==`number`?Math.fround(i):i}){writeNext(16u,0.0);}
writeHistory(${typeof n==`number`?`${n}u`:n},vec3f(count,residual,step));`}function z(e){return`let mergeBegin = index * 256u;
let mergeEnd = min(mergeBegin + 256u, readDispatchParameter(0u));
`+(e?`var system:array<f32,42>;var count=0.0;var weight=0.0;var residualSum=0.0;var centroid=vec3f(0.0);
for(var batchIndex=mergeBegin;batchIndex<mergeEnd;batchIndex++){
 let base=batchIndex*48u;let blockWeight=readPartials(base+1u);if(!(blockWeight>0.0)){continue;}
 count+=readPartials(base);weight+=blockWeight;residualSum+=readPartials(base+2u);
 centroid+=(vec3f(readPartials(base+45u),readPartials(base+46u),readPartials(base+47u))-centroid)*(blockWeight/weight);
}
for(var batchIndex=mergeBegin;batchIndex<mergeEnd;batchIndex++){
 let base=batchIndex*48u;if(!(readPartials(base+1u)>0.0)){continue;}
 let offset=vec3f(readPartials(base+45u),readPartials(base+46u),readPartials(base+47u))-centroid;
 // Row-major skew matrix of the offset: transfer * n = cross(offset, n).
 var transfer:array<f32,9>;transfer[1]=-offset.z;transfer[2]=offset.y;transfer[3]=offset.z;transfer[5]=-offset.x;transfer[6]=-offset.y;transfer[7]=offset.x;
 var blockSystem:array<f32,42>;
 for(var row=0u;row<6u;row++){for(var column=0u;column<6u;column++){blockSystem[row*7u+column]=readPartials(base+3u+row*6u+column);}blockSystem[row*7u+6u]=readPartials(base+39u+row);}
 // Rotation rows, then rotation columns: each reads only the translation rows or columns, which do not change.
 for(var row=0u;row<3u;row++){for(var column=0u;column<7u;column++){for(var term=0u;term<3u;term++){blockSystem[row*7u+column]+=transfer[row*3u+term]*blockSystem[(3u+term)*7u+column];}}}
 for(var row=0u;row<6u;row++){for(var column=0u;column<3u;column++){for(var term=0u;term<3u;term++){blockSystem[row*7u+column]+=blockSystem[row*7u+3u+term]*transfer[column*3u+term];}}}
 for(var entry=0u;entry<42u;entry++){system[entry]+=blockSystem[entry];}
}
let outputBase = index * 48u;
writeMoments(outputBase, count); writeMoments(outputBase + 1u, weight); writeMoments(outputBase + 2u, residualSum);
for (var row = 0u; row < 6u; row++) {
  for (var column = 0u; column < 6u; column++) { writeMoments(outputBase + 3u + row * 6u + column, system[row * 7u + column]); }
  writeMoments(outputBase + 39u + row, system[row * 7u + 6u]);
}
writeMoments(outputBase + 45u, centroid.x); writeMoments(outputBase + 46u, centroid.y); writeMoments(outputBase + 47u, centroid.z);
`:`var count=0u;var weightSum=0.0;var residualSum=0.0;var sourceMean=vec3f(0.0);var targetMean=vec3f(0.0);
var covariance=mat3x3f(vec3f(0.0),vec3f(0.0),vec3f(0.0));
for(var batchIndex=mergeBegin;batchIndex<mergeEnd;batchIndex++){
 let base=batchIndex*18u;let weight=readPartials(base+1u);if(weight<=0.0){continue;}
 count+=u32(readPartials(base));residualSum+=readPartials(base+2u);
 let firstMean=vec3f(readPartials(base+3u),readPartials(base+4u),readPartials(base+5u));
 let secondMean=vec3f(readPartials(base+6u),readPartials(base+7u),readPartials(base+8u));
 let firstDelta=firstMean-sourceMean;let secondDelta=secondMean-targetMean;
 let nextWeight=weightSum+weight;
 covariance+=mat3x3f(firstDelta*secondDelta.x,firstDelta*secondDelta.y,firstDelta*secondDelta.z)*(weightSum*weight/nextWeight);
 for(var column=0u;column<3u;column++){for(var row=0u;row<3u;row++){covariance[column][row]+=readPartials(base+9u+column*3u+row);}}
 sourceMean+=firstDelta*(weight/nextWeight);targetMean+=secondDelta*(weight/nextWeight);weightSum=nextWeight;
}
let outputBase = index * 18u;
writeMoments(outputBase, f32(count)); writeMoments(outputBase + 1u, weightSum); writeMoments(outputBase + 2u, residualSum);
for (var axis = 0u; axis < 3u; axis++) { writeMoments(outputBase + 3u + axis, sourceMean[axis]); writeMoments(outputBase + 6u + axis, targetMean[axis]); }
for (var column = 0u; column < 3u; column++) { for (var row = 0u; row < 3u; row++) { writeMoments(outputBase + 9u + column * 3u + row, covariance[column][row]); } }
`)}var B=512,V=12,H=8,U=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];function W(e,t){let n=t.initialTransform??U;o(e,n.length===16&&n.every(e=>Number.isFinite(Math.fround(e))),`initialTransform must hold 16 finite numbers`),o(e,n[3]===0&&n[7]===0&&n[11]===0&&n[15]===1,`initialTransform must be a column-major rigid transform with bottom row 0, 0, 0, 1`);for(let t=0;t<3;t++)for(let r=0;r<3;r++){let i=0;for(let e=0;e<3;e++)i+=n[t*4+e]*n[r*4+e];o(e,Math.abs(i-Number(t===r))<1e-4,`initialTransform rotation must be orthonormal`)}let r=n[0]*(n[5]*n[10]-n[9]*n[6])-n[4]*(n[1]*n[10]-n[9]*n[2])+n[8]*(n[1]*n[6]-n[5]*n[2]);o(e,r>0,`initialTransform rotation must not be a reflection`),t.origin&&o(e,t.origin.every(Number.isFinite),`origin must hold finite numbers`);let i=t.origin?y(n,t.origin):n;return o(e,i.every(e=>Number.isFinite(Math.fround(e))),`initialTransform in the local frame must hold 16 finite float32 values`),i}var ne=class{props;id;iterations;estimatedCommandNodeCount;localInitialTransform;constructor(t){this.props=t,t=e(t,[`parameters`,`id`,`source`,`target`,`origin`,`initialTransform`,`maximumDistance`,`targetGrid`,`metric`,`targetNormals`,`robust`,`iterations`,`translationTolerance`,`rotationTolerance`,`output`]),this.id=t.id??`gpu-pointcloud-icp`,o(this.id,t.output!==void 0&&t.output!==null,`output must be given`),this.props=t=Object.freeze({...t,output:e(t.output,[`transform`,`history`,`inlierMask`,`status`])}),_(this.id,{source:t.source,target:t.target,targetNormals:t.metric===`point-to-plane`?t.targetNormals:void 0,parameters:t.parameters},{transform:t.output.transform,history:t.output.history,inlierMask:t.output.inlierMask,status:t.output.status},[{outputs:{transform:t.output.transform,status:t.output.status}},{inputs:{parameters:t.parameters},outputs:{history:t.output.history}},{inputs:{targetNormals:t.metric===`point-to-plane`?t.targetNormals:void 0},outputs:{inlierMask:t.output.inlierMask}}]),this.iterations=t.iterations??20,s(this.id,`iterations`,this.iterations,1,100);let n=Math.ceil(t.source.length/256),r=0;for(;n>1;)n=Math.ceil(n/256),r++;this.estimatedCommandNodeCount=4+this.iterations*(4+r*2)+(t.targetGrid?B:0),l(this.id,this.estimatedCommandNodeCount),c(this.id,`maximumDistance`,t.maximumDistance),o(this.id,t.maximumDistance>0&&Math.fround(t.maximumDistance**2)>0&&Number.isFinite(Math.fround(t.maximumDistance**2)),`maximumDistance must be positive with a finite nonzero float32 square`),c(this.id,`translationTolerance`,t.translationTolerance??1e-5,0),c(this.id,`rotationTolerance`,t.rotationTolerance??1e-5,0),a(this.id,`transform`,t.output.transform,16),a(this.id,`history`,t.output.history,this.iterations),a(this.id,`inlierMask`,t.output.inlierMask,t.source.length),a(this.id,`status`,t.output.status,1),t.metric===`point-to-plane`&&o(this.id,t.targetNormals?.length===t.target.length,`targetNormals must hold one row per target point for the point-to-plane metric`),t.robust&&(c(this.id,`robust.scale`,t.robust.scale),o(this.id,Math.fround(t.robust.scale)>0,`robust.scale must be positive`),t.robust.method===`tukey`&&o(this.id,Math.fround(t.robust.scale**2)>0&&Number.isFinite(Math.fround(t.robust.scale**2)),`robust.scale must have a finite nonzero float32 square for the tukey method`)),this.localInitialTransform=W(this.id,t),t.parameters&&a(this.id,`parameters`,t.parameters,22),t.targetGrid?f(this.id,`targetGrid`,t.targetGrid):o(this.id,t.source.length*t.target.length<=h,`source x target is ${t.source.length*t.target.length}, above the ${h} limit of a search without an index; pass targetGrid`)}getAbsoluteTransform(e){return this.props.origin?b(e,this.props.origin):Array.from(e)}getCommandNodes(e){let{source:t,target:n,parameters:i,output:{transform:a,history:o,inlierMask:s,status:c}}=this.props;te(e,this.id,{source:t,target:n,transform:a,history:o,inlierMask:s,status:c,parameters:i,targetNormals:this.props.metric===`point-to-plane`?this.props.targetNormals:void 0}),g(e,this.id,{"ICP matches":t.length*12,"ICP weights and match IDs":t.length*4,"ICP block moments":Math.ceil(t.length/256)*(this.props.metric===`point-to-plane`?48:18)*4,"ICP state":68}),this.props.targetGrid&&p(e,this.id,n,this.props.targetGrid.dimensions),ee(e,this.id,V,1);let l=[0,1].map(t=>r(e,`${this.id}-state-${t}`,`float32`,17)),f=r(e,`${this.id}-matches`,`float32x3`,t.length),h=this.props.metric===`point-to-plane`,_=h?r(e,`${this.id}-match-ids`,`uint32`,t.length):void 0,y=r(e,`${this.id}-weights`,`float32`,t.length),b=Math.ceil(t.length/256),S=r(e,`${this.id}-moments`,`float32`,b*(h?48:18)),C=[S],w=[],T=b;for(;T>1;)w.push(T),T=Math.ceil(T/256),C.push(r(e,`${this.id}-moment-level-${C.length}`,`float32`,T*(h?48:18)));let E=C.at(-1),D=this.localInitialTransform,O=v(e,{id:`${this.id}-initialize`,length:1,inputs:i?{parameters:i}:{},outputs:{state:l[0]},body:i?`for (var entry = 0u; entry < 16u; entry++) { writeState(entry, readParameters(entry + 6u)); } writeState(16u, 1.0);`:`${D.map((e,t)=>`writeState(${t}u, ${Math.fround(e)});`).join(`
`)} writeState(16u, 1.0);`}),k=`let rotation = mat3x3f(vec3f(readState(0u),readState(1u),readState(2u)), vec3f(readState(4u),readState(5u),readState(6u)), vec3f(readState(8u),readState(9u),readState(10u)));
let translation = vec3f(readState(12u),readState(13u),readState(14u));`,A=this.props.targetGrid?m(e,`${this.id}-target-index`,n,this.props.targetGrid):void 0;A&&O.push(...A.nodes);let j=`for (var candidate=0u;candidate<${n.length}u;candidate++){`,M=`}`;if(A&&this.props.targetGrid){let e=this.props.targetGrid;j=`let displacement=position-vec3f(${e.origin.map(Math.fround).join(`,`)});
let lower=vec3i(clamp(floor(getPointCloudQuotient3(displacement-vec3f(${i?`readParameters(0u)`:Math.fround(this.props.maximumDistance)}),${Math.fround(e.cellSize)}))-vec3f(1.0),vec3f(0.0),vec3f(${e.dimensions.join(`,`)})));
let upper=vec3i(clamp(floor(getPointCloudQuotient3(displacement+vec3f(${i?`readParameters(0u)`:Math.fround(this.props.maximumDistance)}),${Math.fround(e.cellSize)}))+vec3f(1.0),vec3f(-1.0),vec3f(${e.dimensions.map(e=>e-1).join(`,`)})));
for(var depth=lower.z;depth<=upper.z;depth++){for(var row=lower.y;row<=upper.y;row++){for(var column=lower.x;column<=upper.x;column++){
let cell=(u32(depth)*${e.dimensions[1]}u+u32(row))*${e.dimensions[0]}u+u32(column);
for(var slot=readOffsets(cell);slot<readOffsets(cell+1u);slot++){let candidate=readSortedIndices(slot);`,M=`}}}}`}let N=this.props.robust,P=N?N.method===`huber`?`min(1.0,getScaledQuotient(${i?`readParameters(2u)`:Math.fround(N.scale)},max(sqrt(distanceSquared),1e-30),1.0,0))`:`pow(max(0.0,1.0-getScaledQuotient(distanceSquared,${i?`readParameters(3u)`:Math.fround(N.scale**2)},1.0,0)),2.0)`:`1.0`,F=r=>v(e,{id:`${this.id}-match-${r}`,length:t.length,inputs:{source:t,...i?{parameters:i}:{},targetPositions:n,state:l[r%2],...A?{sortedIndices:A.sortedIndices,offsets:A.offsets}:{}},outputs:{weights:y,..._?{matchIds:_}:{matches:f}},body:`${k}
let position = rotation * readSource(index) + translation;
var nearest=vec3f(0.0);var nearestId=0xffffffffu;var distanceSquared:f32=${i?`readParameters(1u)`:Math.fround(this.props.maximumDistance**2)};
if(all(abs(position)<=vec3f(3.402823e38))){
 ${j}
 let point=readTargetPositions(candidate);let difference=point-position;let distance=dot(difference,difference);
 if(distance<=distanceSquared&&(distance<distanceSquared||candidate<nearestId)){nearestId=candidate;nearest=point;distanceSquared=distance;}
 ${M}
}
let weight=select(0.0,${P},nearestId!=0xffffffffu);
writeWeights(index,weight);${_?`writeMatchIds(index,nearestId);`:`writeMatches(index,nearest);`}`});for(let r=0;r<this.iterations;r++){O.push(...F(r)),O.push(...v(e,{id:`${this.id}-moments-${r}`,length:b,inputs:{source:t,...h?{targetPositions:n}:{matches:f},weights:y,state:l[r%2],..._?{matchIds:_,targetNormals:this.props.targetNormals}:{}},outputs:{moments:S},declarations:h?`fn readMatches(row: u32) -> vec3f { return readTargetPositions(readMatchIds(row)); }`:void 0,body:h?L(k,t.length):`${k}
var count=0u;var weightSum=0.0;var residualSum=0.0;var sourceMean=vec3f(0.0);var targetMean=vec3f(0.0);
var covariance=mat3x3f(vec3f(0.0),vec3f(0.0),vec3f(0.0));
for(var point=index*256u;point<min((index+1u)*256u,${t.length}u);point++){
 let weight=readWeights(point);if(weight<=0.0){continue;}count++;
 let transformed=rotation*readSource(point)+translation;let matched=readMatches(point);
 let sourceDelta=transformed-sourceMean;weightSum+=weight;sourceMean+=sourceDelta*(weight/weightSum);targetMean+=(matched-targetMean)*(weight/weightSum);
 let targetDelta=matched-targetMean;
 covariance+=mat3x3f(sourceDelta*targetDelta.x,sourceDelta*targetDelta.y,sourceDelta*targetDelta.z)*weight;
 let difference=matched-transformed;residualSum+=weight*dot(difference,difference);
}
let base=index*18u;writeMoments(base,f32(count));writeMoments(base+1u,weightSum);writeMoments(base+2u,residualSum);
for(var axis=0u;axis<3u;axis++){writeMoments(base+3u+axis,sourceMean[axis]);writeMoments(base+6u+axis,targetMean[axis]);}
for(var column=0u;column<3u;column++){for(var row=0u;row<3u;row++){writeMoments(base+9u+column*3u+row,covariance[column][row]);}}`}));for(let[t,n]of w.entries())O.push(...x(e,{id:`${this.id}-merge-${r}-${t}`,length:Math.ceil(n/256),inputs:{partials:C[t]},outputs:{moments:C[t+1]},runtimeWords:[n],body:z(h)}));O.push(...x(e,{id:`${this.id}-solve-${r}`,length:1,runtimeWords:[r,+!!b],inputs:{moments:E,state:l[r%2],...i?{parameters:i}:{}},outputs:{next:l[(r+1)%2],history:o},declarations:h?void 0:d+u(4),body:h?R(k,`readDispatchParameter(1u)`,`readDispatchParameter(0u)`,i?`readParameters(4u)`:this.props.translationTolerance??1e-5,i?`readParameters(5u)`:this.props.rotationTolerance??1e-5):`${k}
for (var entry = 0u; entry < 17u; entry++) { writeNext(entry, readState(entry)); }
var count=0u;var weightSum=0.0;var residualSum=0.0;var sourceMean=vec3f(0.0);var targetMean=vec3f(0.0);
var covariance=mat3x3f(vec3f(0.0),vec3f(0.0),vec3f(0.0));
for(var batchIndex=0u;batchIndex<readDispatchParameter(1u);batchIndex++){
 let base=batchIndex*18u;let weight=readMoments(base+1u);if(weight<=0.0){continue;}
 count+=u32(readMoments(base));residualSum+=readMoments(base+2u);
 let firstMean=vec3f(readMoments(base+3u),readMoments(base+4u),readMoments(base+5u));
 let secondMean=vec3f(readMoments(base+6u),readMoments(base+7u),readMoments(base+8u));
 let firstDelta=firstMean-sourceMean;let secondDelta=secondMean-targetMean;
 let nextWeight=weightSum+weight;
 covariance+=mat3x3f(firstDelta*secondDelta.x,firstDelta*secondDelta.y,firstDelta*secondDelta.z)*(weightSum*weight/nextWeight);
 for(var column=0u;column<3u;column++){for(var row=0u;row<3u;row++){covariance[column][row]+=readMoments(base+9u+column*3u+row);}}
 sourceMean+=firstDelta*(weight/nextWeight);targetMean+=secondDelta*(weight/nextWeight);weightSum=nextWeight;
}
let residual=sqrt(residualSum/max(weightSum,1e-30));
writeHistory(readDispatchParameter(0u),vec3f(f32(count),residual,0.0));
if(readState(16u)!=1.0){return;}if(count<3u||weightSum<=0.0){writeNext(16u,2.0);return;}
covariance*=1.0/weightSum;
let trace = covariance[0][0] + covariance[1][1] + covariance[2][2];
let difference = vec3f(covariance[2][1]-covariance[1][2], covariance[0][2]-covariance[2][0], covariance[1][0]-covariance[0][1]);
var system = mat4x4f(vec4f(trace,difference),
 vec4f(difference.x, covariance[0][0]-covariance[1][1]-covariance[2][2], covariance[1][0]+covariance[0][1], covariance[2][0]+covariance[0][2]),
 vec4f(difference.y, covariance[1][0]+covariance[0][1], -covariance[0][0]+covariance[1][1]-covariance[2][2], covariance[2][1]+covariance[1][2]),
 vec4f(difference.z, covariance[2][0]+covariance[0][2], covariance[2][1]+covariance[1][2], -covariance[0][0]-covariance[1][1]+covariance[2][2]));
var matrix: array<f32, 16>;
for (var row = 0u; row < 4u; row++) {
  for (var column = 0u; column < 4u; column++) { matrix[row * 4u + column] = system[column][row]; }
}
var vectors: array<f32, 16>;
let decomposition = getSymmetricEigen_4(&matrix, &vectors, 4u, ${H}u, 1e-6);
if (decomposition.status != 0u) { writeNext(16u,4.0); return; }
let largest = decomposition.values[3];
let runnerUp = decomposition.values[2];
if (!(largest - runnerUp > max(abs(largest) * 1e-6,1e-30))) { writeNext(16u,3.0); return; }
let quaternion = normalize(vec4f(vectors[3], vectors[7], vectors[11], vectors[15]));
let scalar = quaternion.x; let first = quaternion.y; let second = quaternion.z; let third = quaternion.w;
let deltaRotation = mat3x3f(
 vec3f(1.0-2.0*(second*second+third*third), 2.0*(first*second+scalar*third), 2.0*(first*third-scalar*second)),
 vec3f(2.0*(first*second-scalar*third), 1.0-2.0*(first*first+third*third), 2.0*(second*third+scalar*first)),
 vec3f(2.0*(first*third+scalar*second), 2.0*(second*third-scalar*first), 1.0-2.0*(first*first+second*second)));
let deltaTranslation = targetMean - deltaRotation * sourceMean;
let updatedRotation = deltaRotation * rotation; let updatedTranslation = deltaRotation * translation + deltaTranslation;
for (var column = 0u; column < 3u; column++) { for (var row = 0u; row < 3u; row++) { writeNext(column*4u+row, updatedRotation[column][row]); } }
writeNext(12u,updatedTranslation.x); writeNext(13u,updatedTranslation.y); writeNext(14u,updatedTranslation.z);
let step = length(deltaTranslation);
let rotationStep = max(max(length(deltaRotation[0]-vec3f(1,0,0)),length(deltaRotation[1]-vec3f(0,1,0))),length(deltaRotation[2]-vec3f(0,0,1)));
if (step <= ${i?`readParameters(4u)`:Math.fround(this.props.translationTolerance??1e-5)} && rotationStep <= ${i?`readParameters(5u)`:Math.fround(this.props.rotationTolerance??1e-5)}) { writeNext(16u,0.0); }
writeHistory(readDispatchParameter(0u),vec3f(f32(count),residual,step));`}))}return O.push(...F(this.iterations)),_&&O.push(...v(e,{id:`${this.id}-final-plane-mask`,length:t.length,inputs:{weights:y,matchIds:_,targetNormals:this.props.targetNormals},outputs:{inlierMask:s},body:`var valid=false;if(readWeights(index)>0.0){let normal=safeNormal(readTargetNormals(readMatchIds(index)));valid=dot(normal,normal)>0.5;}writeInlierMask(index,u32(valid));`})),_||O.push(...v(e,{id:`${this.id}-final-mask`,length:t.length,inputs:{weights:y},outputs:{inlierMask:s},body:`writeInlierMask(index, u32(readWeights(index) > 0.0));`})),O.push(...v(e,{id:`${this.id}-publish`,length:1,inputs:{state:l[this.iterations%2]},outputs:{transform:a,status:c},body:`for (var entry = 0u; entry < 16u; entry++) { writeTransform(entry,readState(entry)); } writeStatus(0u,u32(readState(16u)));`})),O}},G=64,re=class{id;points;transform;footprint;constructor(e,t,n,r){this.id=e,this.points=t,this.transform=n,this.footprint=r}getCommandNodes(e){let{id:r,points:a,transform:o,footprint:s}=this,c=M(r,s.length,G,e.device.limits.maxComputeWorkgroupsPerDimension),l=`
const pointsOffset: u32 = ${t(a)}u;
const transformOffset: u32 = ${t(o)}u;
const footprintOffset: u32 = ${t(s)}u;
@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> transform: array<f32>;
@group(0) @binding(2) var<storage, read_write> footprint: array<f32>;
@compute @workgroup_size(${G})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${j(c,G)}
  if (index >= ${s.length}u) { return; }
  let point = vec4f(
    points[pointsOffset + index * 3u],
    points[pointsOffset + index * 3u + 1u],
    points[pointsOffset + index * 3u + 2u],
    1.0
  );
  for (var component = 0u; component < 2u; component++) {
    var sum = 0.0;
    for (var column = 0u; column < 4u; column++) {
      sum += transform[transformOffset + column * 4u + component] * point[column];
    }
    footprint[footprintOffset + index * 2u + component] = sum;
  }
}`;return[A({id:r,resources:[{buffer:a,usage:`storage-read`},{buffer:o,usage:`storage-read`},{buffer:s,usage:`storage-read-write`}],workload:{operation:`TransformedPoints`,variant:`xy`,commandCount:1,maximumWorkgroupCount:c.x*c.y*c.z,maximumInvocationCount:c.x*c.y*c.z*G,readByteLength:s.length*12,writeByteLength:s.length*8},compile:({device:e})=>{let t=new i(e,{id:r,source:l,shaderLayout:{bindings:[`points`,`transform`,`footprint`].map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:r})=>{let i={points:n(a,r),transform:n(o,r),footprint:n(s,r)};t.dispatch(e,{bindings:i,...c})},destroy:()=>t.destroy()}}})]}};function K(e,t,n,r){let i=e.allocate(t,`float32x2`,n.length);return e.graph.add(new re(`${t}-transform`,n.view,r.view,i.view)),i}var ie=9e3,q=1500,J=400,ae=4,Y=[.66,.07,.97,.3],X=[.62,.66,.76],Z=[.3,.7,1],Q=[1,.6,.2],$=[.832,-.555];function oe({random:e,world:t}){let n=k(e,t.width,ie),r=new Float32Array(q*3);for(let e=0;e<q;e++)r.set(n.positions.subarray(e*6*3,e*6*3+3),e*3);return{width:t.width,target:r,rmseTop:.05}}var se=S({parameters:N,caption:`The King County scan (gray), a rotated and shifted copy of it (blue) and that copy after ICP (orange); the curve is the RMSE per iteration. Closest-point matching on this nearly flat terrain approaches the scan gradually, so a large rotation may not be fully undone within the iterations.`,createInputs:oe,build(e,t,n){let r=e.upload(`target`,`float32x3`,t.target),i=e.upload(`source`,`float32x3`,new Float32Array(q*3)),a=e.allocate(`transform`,`float32`,16),o=e.allocate(`history`,`float32x3`,n.iterations),s=e.allocate(`status`,`uint32`,1);e.graph.add(new ne({source:i.view,target:r.view,maximumDistance:n.maximumDistance,iterations:n.iterations,output:{transform:a.view,history:o.view,inlierMask:e.allocate(`inlier-mask`,`uint32`,q).view,status:s.view}}));let c=ce(e),l=K(e,`aligned-footprint`,i,a);return{outputs:{target:K(e,`target-footprint`,r,c),initial:K(e,`initial-footprint`,i,c),aligned:F(e,`converged-footprint`,l,s,q,{components:2,statusIndex:0}),status:s,rmse:O(e,`rmse`,o,n.iterations,{componentStart:1})},update:e=>{let n=e.rotation*Math.PI/180,r=t.width/2,a=0,o=Float32Array.from({length:q*3},(i,a)=>{let o=Math.floor(a/3),s=t.target[o*3]-r,c=t.target[o*3+1]-.5;return a%3==0?r+s*Math.cos(n)-c*Math.sin(n)+e.shift*$[0]:a%3==1?.5+s*Math.sin(n)+c*Math.cos(n)+e.shift*$[1]:t.target[a]});for(let e=0;e<q;e++)a+=(o[e*3]-t.target[e*3])**2+(o[e*3+1]-t.target[e*3+1])**2;t.rmseTop=Math.max(Math.sqrt(a/q),.01),i.write(o)}}},rebuildOn:[`iterations`,`maximumDistance`],async afterRun({outputs:e,upload:t,values:n}){let r=await e.status.buffer.readAsync(),i=new Uint32Array(r.buffer.slice(r.byteOffset,r.byteOffset+r.byteLength))[0],a=await e.aligned.buffer.readAsync(),o=new Float32Array(a.buffer.slice(a.byteOffset,a.byteOffset+a.byteLength)),s=await e.initial.buffer.readAsync(),c=new Float32Array(s.buffer.slice(s.byteOffset,s.byteOffset+s.byteLength)),l=[];for(let e=0;e<J;e+=ae)l.push(c[e*2],c[e*2+1],o[e*2],o[e*2+1]);return{links:t(`links`,`float32x4`,Float32Array.from(l)),iterations:n.iterations,status:i}},interaction:C(N,{kind:`pad`,hint:`rotate across, shift up`,x:`rotation`,y:`shift`,decorate:(e,t,{inputs:n,outputs:r,peek:i})=>{let a=n.width/2,o=[],s=e.rotation*Math.PI/180;for(let e=0;e<=12;e++){let t=e/12*s;o.push(a+Math.cos(t)*.12,.5+Math.sin(t)*.12)}let c=i(r.rmse),l=i(r.status)?.[0];return{glyphs:[{shape:`crosshair`,at:[a,.5],size:6},{shape:`polyline`,points:o}],readout:`${P(l)}${l===0?``:` (alignment hidden)`} · rotation ${e.rotation}° · shift ${e.shift.toFixed(3)}${c?` · RMSE ${c[0].toFixed(3)} -> ${c[c.length-1].toFixed(3)}`:``}`}}}),draw:({inputs:e,outputs:t,extra:n,theme:r,world:i})=>{let a=E(i,Y),o=T({rectangle:a,x:{range:[1,Math.max(2,n?.iterations??2)],ticks:3,title:`iteration`},y:{range:[0,e.rmseTop],ticks:2,title:`RMSE`}});return[{layer:`dots`,context:!0,positions:t.target,count:J,color:[...X,.9],radius:5.2},{layer:`dots`,context:!0,positions:t.initial,count:J,color:[...Z,.95],radius:2.6},...n?[{layer:`lines`,segments:n.links,color:[...r.ink,.6],radius:.9}]:[],{layer:`dots`,positions:t.aligned,count:J,color:[...Q,1],radius:2.6},{layer:`text`,...o.grid},{layer:`chart`,kind:`line`,values:t.rmse,range:[0,e.rmseTop],rectangle:Y,radius:2.4,color:[.3,.75,1,1]},{layer:`text`,...w(o,D({corner:`top-left`,entries:[{label:`target`,swatch:X},{label:`source before ICP`,swatch:Z},{label:`source after ICP: ${P(n?.status)}${n?.status===0?``:` (hidden)`}`,swatch:Q}]}))}]}});function ce(e){return e.upload(`identity`,`float32`,Float32Array.from({length:16},(e,t)=>+(t%5==0)))}export{se as default};