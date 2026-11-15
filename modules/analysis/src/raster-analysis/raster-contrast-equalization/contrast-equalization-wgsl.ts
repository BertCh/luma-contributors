// luma.gl
// SPDX-License-Identifier: MIT AND BSD-3-Clause
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright (c) 2009-2022 the scikit-image team

/*
 * BSD-3-Clause notice for the scikit-image adaptation:
 * Copyright (c) 2009-2022 the scikit-image team. All rights reserved.
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice, this list of
 *    conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of
 *    conditions and the following disclaimer in the documentation and/or other materials
 *    provided with the distribution.
 * 3. Neither the name of scikit-image nor the names of its contributors may be used to endorse
 *    or promote products derived from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Bounded histogram clipping, following scikit-image 0.24's clip_histogram redistribution.
 * Adapted algorithm: https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/exposure/_adapthist.py
 * The histogram is storage-backed: one invocation exclusively owns each tile's at most 256 bins.
 * @internal
 */
export const CLIP_CONTRAST_HISTOGRAM_WGSL = /* wgsl */ `
let start = histogramOffset + index * BIN_COUNT;
let sampleCount = counts[countsOffset + index];
let fraction = parameters[parametersOffset + 4u];
var limit = sampleCount;
if (fraction > 0.0) { limit = max(1u, u32(floor(fraction * f32(sampleCount)))); }
var excess = 0i;
for (var bin = 0u; bin < BIN_COUNT; bin++) {
  let value = histogram[start + bin];
  excess += i32(value - min(value, limit));
  histogram[start + bin] = min(value, limit);
}
let increment = excess / i32(BIN_COUNT);
let upper = i32(limit) - increment;
for (var bin = 0u; bin < BIN_COUNT; bin++) {
  var value = i32(histogram[start + bin]);
  if (value < upper) { value += increment; excess -= increment; }
  if (value >= upper && value < i32(limit)) { excess -= i32(limit) - value; value = i32(limit); }
  histogram[start + bin] = u32(value);
}
// Each productive sweep reduces excess by at least one; the initial excess is at most the
// bounded tile area. A skewed three-bin histogram can require more sweeps than BIN_COUNT.
for (var sweep = 0u; sweep < MAXIMUM_REDISTRIBUTION_SWEEPS && excess > 0i; sweep++) {
  let previous = excess;
  for (var first = 0u; first < BIN_COUNT && excess > 0i; first++) {
    var available = 0u;
    for (var bin = 0u; bin < BIN_COUNT; bin++) {
      available += u32(histogram[start + bin] < limit);
    }
    let stride = max(1u, available / u32(excess));
    for (var bin = first; bin < BIN_COUNT; bin += stride) {
      if (histogram[start + bin] < limit) { histogram[start + bin]++; excess--; }
    }
  }
  if (excess == previous) { break; }
}`;
