// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * @internal Transfer a weighted, centred plane normal system to `centroid` and add it to
 * `system`. `base` addresses the 48-word moment record. Rows are rotation then translation;
 * the augmented matrix is row-major, with the right-hand side in column six.
 *
 * For offset d = blockCentre - centroid, J = [a + cross(d, n), n]. Applying the skew
 * matrix to rotation rows and then columns forms T H Tᵀ and T b. Translation rows are
 * unchanged in the first pass, and translation columns are unchanged in the second pass.
 * Counts, weights, residuals, centroid accumulation and singular solves stay with the caller.
 */
export function getPlaneMomentTransferSource(readMoment: 'readMoments' | 'readPartials'): string {
  return ` let offset=vec3f(${readMoment}(base+45u),${readMoment}(base+46u),${readMoment}(base+47u))-centroid;
 // Row-major skew matrix of the offset: transfer * n = cross(offset, n).
 var transfer:array<f32,9>;transfer[1]=-offset.z;transfer[2]=offset.y;transfer[3]=offset.z;transfer[5]=-offset.x;transfer[6]=-offset.y;transfer[7]=offset.x;
 var blockSystem:array<f32,42>;
 for(var row=0u;row<6u;row++){for(var column=0u;column<6u;column++){blockSystem[row*7u+column]=${readMoment}(base+3u+row*6u+column);}blockSystem[row*7u+6u]=${readMoment}(base+39u+row);}
 // Rotation rows, then rotation columns: each reads only the translation rows or columns, which do not change.
 for(var row=0u;row<3u;row++){for(var column=0u;column<7u;column++){for(var term=0u;term<3u;term++){blockSystem[row*7u+column]+=transfer[row*3u+term]*blockSystem[(3u+term)*7u+column];}}}
 for(var row=0u;row<6u;row++){for(var column=0u;column<3u;column++){for(var term=0u;term<3u;term++){blockSystem[row*7u+column]+=blockSystem[row*7u+3u+term]*transfer[column*3u+term];}}}
 for(var entry=0u;entry<42u;entry++){system[entry]+=blockSystem[entry];}`;
}
