// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Independent binary64 Gaussian/Abramson fixture, h=1, alpha=1/2, unit weights.
 * Derivation (Python standard library, no source imports):
 * points=[(0,0),(.5,0),(0,.5),(4,0)]
 * K(d2,h)=exp(-d2/(2*h*h))/(2*pi*h*h), zero for d2 >= 25*h*h
 * pilot[i]=sum(K((xi-xj)^2+(yi-yj)^2,1) for j)
 * g=exp(sum(log(p) for p in pilot)/4); h[i]=sqrt(g/pilot[i])
 * values[x]=sum(K((x-xi)^2+yi^2,h[i]) for i), x in [0,2,4].
 */
export const ADAPTIVE_INDEPENDENT_FIXTURE = {
  positions: [0, 0, 0.5, 0, 0, 0.5, 4, 0],
  pilot: [0.44011582224659906, 0.423906831735602, 0.42360579869333836, 0.1596036006339419],
  bandwidths: [0.8726131561624353, 0.8891397769438932, 0.8894556518881487, 1.449051780921215],
  values: [0.5543401345940513, 0.10658262721278872, 0.07589667038818469]
};

/** Epanechnikov product kernel, spatial h=2, temporal l=2; two events at (0,0), times .5,1.5.
 * Query (1,0) has Ks=2/(pi*4)*(1-1/4)=3/(8pi).
 * At t=2 or t=0, temporal sum is 3/8 * ((1-(.5/2)^2)+(1-(1.5/2)^2))
 * = 45/128 + 21/128 = 33/64. Window [0,4] contains all temporal mass at t=2,
 * and exactly half at t=0. Thus interior=99/(512pi), boundary=99/(256pi).
 */
export const SPACE_TIME_INDEPENDENT_VALUES = [0.1230964012976378, 0.0615482006488189];
