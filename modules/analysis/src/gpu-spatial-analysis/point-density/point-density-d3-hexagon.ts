// luma.gl
// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Copyright (c) 2012-2016 Mike Bostock
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/*!
 * Adapted from d3-hexbin, https://github.com/d3/d3-hexbin/blob/main/src/hexbin.js
 * Copyright Mike Bostock, 2012-2016. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the author nor the names of contributors may be used to
 *   endorse or promote products derived from this software without specific prior
 *   written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

const HORIZONTAL_SPACING = 2 * Math.sin(Math.PI / 3);

/**
 * d3-hexbin's odd-r assignment, including its normalized-distance comparison and Math.round
 * ties toward positive infinity. This differs from nearest-center cube rounding away from ties.
 */
export function getGPUPointDensityD3HexagonCell(
  x: number,
  y: number,
  originX: number,
  originY: number,
  radius: number
): [number, number] {
  const normalizedY = (y - originY) / radius / 1.5;
  let row = Math.round(normalizedY);
  const normalizedX = (x - originX) / radius / HORIZONTAL_SPACING - (row & 1) / 2;
  let column = Math.round(normalizedX);
  const deltaY = normalizedY - row;
  if (Math.abs(deltaY) * 3 > 1) {
    const deltaX = normalizedX - column;
    const alternativeColumn = column + (normalizedX < column ? -1 : 1) / 2;
    const alternativeRow = row + (normalizedY < row ? -1 : 1);
    const alternativeDeltaX = normalizedX - alternativeColumn;
    const alternativeDeltaY = normalizedY - alternativeRow;
    if (
      deltaX * deltaX + deltaY * deltaY >
      alternativeDeltaX * alternativeDeltaX + alternativeDeltaY * alternativeDeltaY
    ) {
      column = alternativeColumn + (row & 1 ? 1 : -1) / 2;
      row = alternativeRow;
    }
  }
  return [column + 0, row + 0];
}

/** Float32 GPU implementation of the d3 rule. WGSL round() would use the wrong half-tie rule. */
export const GPU_POINT_DENSITY_D3_HEXAGON_WGSL = /* wgsl */ `
fn getPointDensityD3HexagonCell(x: f32, y: f32, originX: f32, originY: f32, radius: f32) -> vec2<i32> {
  let normalizedY = (y - originY) / radius / 1.5;
  var row = floor(normalizedY + 0.5);
  let parity = f32(i32(row) & 1);
  let normalizedX = (x - originX) / radius / 1.7320508075688772 - parity * 0.5;
  var column = floor(normalizedX + 0.5);
  let deltaY = normalizedY - row;
  if (abs(deltaY) * 3.0 > 1.0) {
    let deltaX = normalizedX - column;
    let alternativeColumn = column + select(0.5, -0.5, normalizedX < column);
    let alternativeRow = row + select(1.0, -1.0, normalizedY < row);
    let alternativeDeltaX = normalizedX - alternativeColumn;
    let alternativeDeltaY = normalizedY - alternativeRow;
    if (deltaX * deltaX + deltaY * deltaY > alternativeDeltaX * alternativeDeltaX + alternativeDeltaY * alternativeDeltaY) {
      column = alternativeColumn + select(-0.5, 0.5, parity != 0.0);
      row = alternativeRow;
    }
  }
  return vec2<i32>(i32(column), i32(row));
}
`;
