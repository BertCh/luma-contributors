// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** @internal Point-to-plane Gauss-Newton moments, one independent bounded source block. */
export function getPlaneMomentsSource(loadTransform: string, rows: number): string {
  return `${loadTransform}
var system:array<f32,36>;var right:array<f32,6>;var count=0u;var totalWeight=0.0;var residualSum=0.0;
for(var point=index*256u;point<min((index+1u)*256u,${rows}u);point++){
 let weight=readWeights(point);if(weight<=0.0){continue;}
 let normal=safeNormal(readTargetNormals(readMatchIds(point)));if(dot(normal,normal)<0.5){continue;}
 let position=rotation*readSource(point)+translation;let difference=readMatches(point)-position;
 let rotational=cross(position,normal);let residual=dot(difference,normal);
 var jacobian:array<f32,6>;jacobian[0]=rotational.x;jacobian[1]=rotational.y;jacobian[2]=rotational.z;jacobian[3]=normal.x;jacobian[4]=normal.y;jacobian[5]=normal.z;
 count++;totalWeight+=weight;residualSum+=weight*residual*residual;
 for(var row=0u;row<6u;row++){right[row]+=weight*jacobian[row]*residual;for(var column=0u;column<6u;column++){system[row*6u+column]+=weight*jacobian[row]*jacobian[column];}}
}
let base=index*45u;writeMoments(base,f32(count));writeMoments(base+1u,totalWeight);writeMoments(base+2u,residualSum);
for(var entry=0u;entry<36u;entry++){writeMoments(base+3u+entry,system[entry]);}for(var entry=0u;entry<6u;entry++){writeMoments(base+39u+entry,right[entry]);}`;
}

/** @internal Rank-revealing six-dimensional normal solve and exact Rodrigues update. */
export function getPlaneSolveSource(
  loadTransform: string,
  blocks: number,
  iteration: number,
  translationTolerance: number,
  rotationTolerance: number
): string {
  return `${loadTransform}
for(var entry=0u;entry<17u;entry++){writeNext(entry,readState(entry));}
var system:array<f32,42>;var count=0.0;var weight=0.0;var residualSum=0.0;
for(var batchIndex=0u;batchIndex<${blocks}u;batchIndex++){
 let base=batchIndex*45u;count+=readMoments(base);weight+=readMoments(base+1u);residualSum+=readMoments(base+2u);
 for(var row=0u;row<6u;row++){for(var column=0u;column<6u;column++){system[row*7u+column]+=readMoments(base+3u+row*6u+column);}system[row*7u+6u]+=readMoments(base+39u+row);}
}
let residual=sqrt(residualSum/max(weight,1e-30));writeHistory(${iteration}u,vec3f(count,residual,0.0));
if(readState(16u)!=1.0){return;}if(count<6.0){writeNext(16u,2.0);return;}
var scale=0.0;for(var row=0u;row<6u;row++){scale=max(scale,abs(system[row*7u+row]));}
for(var column=0u;column<6u;column++){
 var pivot=column;for(var row=column+1u;row<6u;row++){if(abs(system[row*7u+column])>abs(system[pivot*7u+column])){pivot=row;}}
 if(!(abs(system[pivot*7u+column])>max(scale*1e-7,1e-30))){writeNext(16u,3.0);return;}
 for(var entry=column;entry<7u;entry++){let temporary=system[column*7u+entry];system[column*7u+entry]=system[pivot*7u+entry];system[pivot*7u+entry]=temporary;}
 let divisor=system[column*7u+column];for(var entry=column;entry<7u;entry++){system[column*7u+entry]/=divisor;}
 for(var row=0u;row<6u;row++){if(row==column){continue;}let factor=system[row*7u+column];for(var entry=column;entry<7u;entry++){system[row*7u+entry]-=factor*system[column*7u+entry];}}
}
let angular=vec3f(system[6],system[13],system[20]);let deltaTranslation=vec3f(system[27],system[34],system[41]);
let angle=length(angular);let axis=angular/max(angle,1e-30);let cosine=cos(angle);let sine=sin(angle);
let identity=mat3x3f(vec3f(1,0,0),vec3f(0,1,0),vec3f(0,0,1));
let outer=mat3x3f(axis*axis.x,axis*axis.y,axis*axis.z);
let skew=mat3x3f(vec3f(0,axis.z,-axis.y),vec3f(-axis.z,0,axis.x),vec3f(axis.y,-axis.x,0));
let deltaRotation=identity*cosine+outer*(1.0-cosine)+skew*sine;
let updatedRotation=deltaRotation*rotation;let updatedTranslation=deltaRotation*translation+deltaTranslation;
for(var column=0u;column<3u;column++){for(var row=0u;row<3u;row++){writeNext(column*4u+row,updatedRotation[column][row]);}}
writeNext(12u,updatedTranslation.x);writeNext(13u,updatedTranslation.y);writeNext(14u,updatedTranslation.z);
let step=length(deltaTranslation);let rotationStep=max(max(length(deltaRotation[0]-vec3f(1,0,0)),length(deltaRotation[1]-vec3f(0,1,0))),length(deltaRotation[2]-vec3f(0,0,1)));
if(step<=${Math.fround(translationTolerance)}&&rotationStep<=${Math.fround(rotationTolerance)}){writeNext(16u,0.0);}
writeHistory(${iteration}u,vec3f(count,residual,step));`;
}
