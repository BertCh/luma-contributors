// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Number of float32 words one 256-row block writes: count, weight, weighted squared residual, the
 * 6 by 6 normal matrix, the 6 right-hand sides and the block centre.
 * @internal
 */
export const POINT_CLOUD_PLANE_ICP_BLOCK_WORDS = 48;

/**
 * Smallest pivot the point-to-plane solve accepts, on the normal matrix after it is scaled to a
 * unit diagonal. That matrix does not depend on the coordinate unit or on the distance from the
 * origin, so the limit is a pure rank test: float32 rounding leaves pivots near 1e-6 in a
 * rank-deficient system (one plane, two planes), and a well-posed scene has pivots near 1.
 */
const POINT_CLOUD_PLANE_ICP_PIVOT_TOLERANCE = 1e-5;

/**
 * @internal Point-to-plane Gauss-Newton moments, one independent bounded source block.
 *
 * The rotational Jacobian `cross(position - centre, normal)` is taken about the block's own
 * weighted centre, not about the coordinate origin: about the origin the rotation columns of a
 * cloud at distance `c` are nearly `c` times the translation columns, and float32 cannot separate
 * them. The solve moves every block to the centroid of all matched points.
 */
export function getPlaneMomentsSource(loadTransform: string, rows: number): string {
  return `${loadTransform}
var centre=vec3f(0.0);var centreWeight=0.0;
for(var point=index*256u;point<min((index+1u)*256u,${rows}u);point++){
 let weight=readWeights(point);if(weight<=0.0){continue;}
 let normal=safeNormal(readTargetNormals(readMatchIds(point)));if(dot(normal,normal)<0.5){continue;}
 let position=rotation*readSource(point)+translation;centreWeight+=weight;centre+=(position-centre)*(weight/centreWeight);
}
var system:array<f32,36>;var right:array<f32,6>;var count=0u;var totalWeight=0.0;var residualSum=0.0;
for(var point=index*256u;point<min((index+1u)*256u,${rows}u);point++){
 let weight=readWeights(point);if(weight<=0.0){continue;}
 let normal=safeNormal(readTargetNormals(readMatchIds(point)));if(dot(normal,normal)<0.5){continue;}
 let position=rotation*readSource(point)+translation;let difference=readMatches(point)-position;
 let rotational=cross(position-centre,normal);let residual=dot(difference,normal);
 var jacobian:array<f32,6>;jacobian[0]=rotational.x;jacobian[1]=rotational.y;jacobian[2]=rotational.z;jacobian[3]=normal.x;jacobian[4]=normal.y;jacobian[5]=normal.z;
 count++;totalWeight+=weight;residualSum+=weight*residual*residual;
 for(var row=0u;row<6u;row++){right[row]+=weight*jacobian[row]*residual;for(var column=0u;column<6u;column++){system[row*6u+column]+=weight*jacobian[row]*jacobian[column];}}
}
let base=index*${POINT_CLOUD_PLANE_ICP_BLOCK_WORDS}u;writeMoments(base,f32(count));writeMoments(base+1u,totalWeight);writeMoments(base+2u,residualSum);
for(var entry=0u;entry<36u;entry++){writeMoments(base+3u+entry,system[entry]);}for(var entry=0u;entry<6u;entry++){writeMoments(base+39u+entry,right[entry]);}
writeMoments(base+45u,centre.x);writeMoments(base+46u,centre.y);writeMoments(base+47u,centre.z);`;
}

/**
 * @internal Rank-revealing six-dimensional normal solve and exact Rodrigues update.
 *
 * Block moments are moved to the weighted centroid of all matched points with the exact transfer
 * `J = [a + d x n, n]` (`d` is the block centre minus the centroid), so every term has the size of
 * the scene and none has the size of its distance from the origin. The system is scaled to a unit
 * diagonal before elimination (rotation rows carry a squared length, translation rows do not), and
 * a pivot at or below {@link POINT_CLOUD_PLANE_ICP_PIVOT_TOLERANCE} reports `DEGENERATE`. The
 * solved motion is a rotation about the centroid plus a translation; it is composed back into the
 * rotation and translation about the origin that the state holds.
 */
export function getPlaneSolveSource(
  loadTransform: string,
  blocks: number,
  iteration: number,
  translationTolerance: number,
  rotationTolerance: number
): string {
  const words = POINT_CLOUD_PLANE_ICP_BLOCK_WORDS;
  return `${loadTransform}
for(var entry=0u;entry<17u;entry++){writeNext(entry,readState(entry));}
var system:array<f32,42>;var count=0.0;var weight=0.0;var residualSum=0.0;var centroid=vec3f(0.0);
for(var batchIndex=0u;batchIndex<${blocks}u;batchIndex++){
 let base=batchIndex*${words}u;let blockWeight=readMoments(base+1u);if(!(blockWeight>0.0)){continue;}
 count+=readMoments(base);weight+=blockWeight;residualSum+=readMoments(base+2u);
 centroid+=(vec3f(readMoments(base+45u),readMoments(base+46u),readMoments(base+47u))-centroid)*(blockWeight/weight);
}
for(var batchIndex=0u;batchIndex<${blocks}u;batchIndex++){
 let base=batchIndex*${words}u;if(!(readMoments(base+1u)>0.0)){continue;}
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
let residual=sqrt(residualSum/max(weight,1e-30));writeHistory(${iteration}u,vec3f(count,residual,0.0));
if(readState(16u)!=1.0){return;}if(count<6.0){writeNext(16u,2.0);return;}
var scale:array<f32,6>;
for(var row=0u;row<6u;row++){let diagonal=system[row*7u+row];if(!(diagonal>0.0)){writeNext(16u,3.0);return;}scale[row]=1.0/sqrt(diagonal);}
for(var row=0u;row<6u;row++){for(var column=0u;column<6u;column++){system[row*7u+column]=system[row*7u+column]*scale[row]*scale[column];}system[row*7u+6u]*=scale[row];}
for(var column=0u;column<6u;column++){
 var pivot=column;for(var row=column+1u;row<6u;row++){if(abs(system[row*7u+column])>abs(system[pivot*7u+column])){pivot=row;}}
 if(!(abs(system[pivot*7u+column])>${POINT_CLOUD_PLANE_ICP_PIVOT_TOLERANCE})){writeNext(16u,3.0);return;}
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
if(step<=${Math.fround(translationTolerance)}&&rotationStep<=${Math.fround(rotationTolerance)}){writeNext(16u,0.0);}
writeHistory(${iteration}u,vec3f(count,residual,step));`;
}
