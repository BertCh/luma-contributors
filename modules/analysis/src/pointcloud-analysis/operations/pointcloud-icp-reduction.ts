// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getPlaneMomentTransferSource} from './pointcloud-plane-transfer';

// A bounded merge limits each invocation to 256 partial blocks; levels form a fixed-order tree.
export const POINT_CLOUD_ICP_MERGE_BLOCK_COUNT = 256;

/** Centered sufficient-statistic merge: no invocation loops over the complete cloud or block set. */
export function getPointCloudICPMergeSource(pointToPlane: boolean): string {
  const bounds = `let mergeBegin = index * ${POINT_CLOUD_ICP_MERGE_BLOCK_COUNT}u;
let mergeEnd = min(mergeBegin + ${POINT_CLOUD_ICP_MERGE_BLOCK_COUNT}u, readDispatchParameter(0u));
`;
  return (
    bounds +
    (pointToPlane
      ? `var system:array<f32,42>;var count=0.0;var weight=0.0;var residualSum=0.0;var centroid=vec3f(0.0);
for(var batchIndex=mergeBegin;batchIndex<mergeEnd;batchIndex++){
 let base=batchIndex*48u;let blockWeight=readPartials(base+1u);if(!(blockWeight>0.0)){continue;}
 count+=readPartials(base);weight+=blockWeight;residualSum+=readPartials(base+2u);
 centroid+=(vec3f(readPartials(base+45u),readPartials(base+46u),readPartials(base+47u))-centroid)*(blockWeight/weight);
}
for(var batchIndex=mergeBegin;batchIndex<mergeEnd;batchIndex++){
 let base=batchIndex*48u;if(!(readPartials(base+1u)>0.0)){continue;}
${getPlaneMomentTransferSource('readPartials')}
}
let outputBase = index * 48u;
writeMoments(outputBase, count); writeMoments(outputBase + 1u, weight); writeMoments(outputBase + 2u, residualSum);
for (var row = 0u; row < 6u; row++) {
  for (var column = 0u; column < 6u; column++) { writeMoments(outputBase + 3u + row * 6u + column, system[row * 7u + column]); }
  writeMoments(outputBase + 39u + row, system[row * 7u + 6u]);
}
writeMoments(outputBase + 45u, centroid.x); writeMoments(outputBase + 46u, centroid.y); writeMoments(outputBase + 47u, centroid.z);
`
      : `var count=0u;var weightSum=0.0;var residualSum=0.0;var sourceMean=vec3f(0.0);var targetMean=vec3f(0.0);
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
`)
  );
}
