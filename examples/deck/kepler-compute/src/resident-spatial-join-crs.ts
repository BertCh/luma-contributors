// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Coordinate systems the resident Spatial Join can join in. Declarations are EPSG codes for
 * EPSG:4326 / EPSG:3857 and the UTM families (EPSG:326NN, 327NN, 258NN, 269NN) or proj4 parameter
 * strings such as '+proj=utm +zone=33 +datum=WGS84' or '+proj=tmerc +lon_0=9 +k=0.9996 +x_0=500000'
 * for +proj=longlat, merc (spherical Web Mercator) and tmerc/utm on a WGS84 or GRS80 class
 * ellipsoid. Anything else (other projections, datum shifts, +towgs84 with non-zero terms) is
 * refused rather than guessed. Ellipsoidal transverse Mercator uses the Krueger series (the same
 * series as proj4's etmerc) evaluated in Float32 on the GPU.
 */

export type ProjectionDescriptor =
  | {kind: 'longlat'; label: 'EPSG:4326'}
  | {kind: 'mercator'; label: 'EPSG:3857'}
  | {
      kind: 'tmerc';
      label: string;
      semiMajorAxis: number;
      flattening: number;
      scale: number;
      centralMeridianDegrees: number;
      latitudeOriginDegrees: number;
      falseEasting: number;
      falseNorthing: number;
    };

const ELLIPSOIDS: Record<string, [number, number]> = {
  wgs84: [6378137, 1 / 298.257223563],
  grs80: [6378137, 1 / 298.257222101],
  // GRS80-class datums used by the UTM EPSG families below.
  nad83: [6378137, 1 / 298.257222101],
  etrs89: [6378137, 1 / 298.257222101]
};

const LONGLAT_ALIASES = new Set(['4326', 'CRS:84', 'EPSG:4326', 'WGS84', 'OGC:CRS84']);
const MERCATOR_ALIASES = new Set(['3857', '900913', 'EPSG:3857', 'EPSG:900913']);

function makeTransverseMercator(
  ellipsoid: [number, number],
  scale: number,
  centralMeridianDegrees: number,
  latitudeOriginDegrees: number,
  falseEasting: number,
  falseNorthing: number
): ProjectionDescriptor {
  const [semiMajorAxis, flattening] = ellipsoid;
  // The label is canonical, so equal definitions compare equal and the label parses back.
  const label = `tmerc:a=${semiMajorAxis};f=${flattening};k=${scale};lon0=${centralMeridianDegrees};lat0=${latitudeOriginDegrees};x0=${falseEasting};y0=${falseNorthing}`;
  return {
    kind: 'tmerc',
    label,
    semiMajorAxis,
    flattening,
    scale,
    centralMeridianDegrees,
    latitudeOriginDegrees,
    falseEasting,
    falseNorthing
  };
}

function makeUtm(
  zone: number,
  isSouth: boolean,
  ellipsoid: [number, number]
): ProjectionDescriptor | undefined {
  if (!Number.isInteger(zone) || zone < 1 || zone > 60) return undefined;
  return makeTransverseMercator(
    ellipsoid,
    0.9996,
    (zone - 1) * 6 - 180 + 3,
    0,
    500000,
    isSouth ? 10000000 : 0
  );
}

function parseEpsgCode(code: number): ProjectionDescriptor | undefined {
  if (code >= 32601 && code <= 32660) return makeUtm(code - 32600, false, ELLIPSOIDS['wgs84']);
  if (code >= 32701 && code <= 32760) return makeUtm(code - 32700, true, ELLIPSOIDS['wgs84']);
  // ETRS89 / UTM zones 28N to 37N and NAD83 / UTM zones 1N to 23N.
  if (code >= 25828 && code <= 25837) return makeUtm(code - 25800, false, ELLIPSOIDS['grs80']);
  if (code >= 26901 && code <= 26923) return makeUtm(code - 26900, false, ELLIPSOIDS['grs80']);
  return undefined;
}

function parseProj4Parameters(text: string): ProjectionDescriptor | undefined {
  const parameters = new Map<string, string>();
  for (const token of text.trim().split(/\s+/)) {
    const match = /^\+([a-z_0-9]+)(?:=(.*))?$/.exec(token);
    if (!match) return undefined;
    parameters.set(match[1], match[2] ?? 'true');
  }
  const projection = parameters.get('proj');
  const number = (key: string, fallback: number): number | undefined => {
    const raw = parameters.get(key);
    if (raw === undefined) return fallback;
    const value = Number(raw);
    return Number.isFinite(value) ? value : undefined;
  };
  // Datum shifts are not applied; zero or absent towgs84 is the only accepted form.
  const towgs84 = parameters.get('towgs84');
  if (towgs84 && !towgs84.split(',').every(term => Number(term) === 0)) return undefined;
  if (parameters.has('nadgrids') || parameters.has('pm') || parameters.has('axis'))
    return undefined;
  const units = parameters.get('units');
  if (units && units !== 'm') return undefined;
  if (projection === 'longlat' || projection === 'latlong') {
    const datum = parameters.get('datum');
    if (datum && !['wgs84', 'nad83', 'etrs89'].includes(datum.toLowerCase())) return undefined;
    return {kind: 'longlat', label: 'EPSG:4326'};
  }
  if (projection === 'merc') {
    // Only the spherical Web Mercator definition is the GPU's EPSG:3857.
    if (
      parameters.get('a') === '6378137' &&
      parameters.get('b') === '6378137' &&
      Number(parameters.get('k') ?? parameters.get('k_0') ?? 1) === 1 &&
      Number(parameters.get('lon_0') ?? 0) === 0 &&
      Number(parameters.get('x_0') ?? 0) === 0 &&
      Number(parameters.get('y_0') ?? 0) === 0
    )
      return {kind: 'mercator', label: 'EPSG:3857'};
    return undefined;
  }
  let ellipsoid: [number, number] | undefined;
  const ellipsoidName = (
    parameters.get('ellps') ??
    parameters.get('datum') ??
    'wgs84'
  ).toLowerCase();
  if (parameters.has('a')) {
    const a = number('a', 6378137);
    const rf = parameters.has('rf') ? number('rf', 0) : undefined;
    const f = parameters.has('f') ? number('f', 0) : undefined;
    if (a === undefined) return undefined;
    ellipsoid = [a, rf ? 1 / rf : (f ?? 0)];
  } else {
    ellipsoid = ELLIPSOIDS[ellipsoidName];
  }
  if (!ellipsoid) return undefined;
  if (projection === 'utm') {
    const zone = Number(parameters.get('zone'));
    return makeUtm(zone, parameters.has('south'), ellipsoid);
  }
  if (projection === 'tmerc') {
    const scale = number('k', number('k_0', 1) ?? 1);
    const centralMeridian = number('lon_0', 0);
    const latitudeOrigin = number('lat_0', 0);
    const falseEasting = number('x_0', 0);
    const falseNorthing = number('y_0', 0);
    if (
      scale === undefined ||
      centralMeridian === undefined ||
      latitudeOrigin === undefined ||
      falseEasting === undefined ||
      falseNorthing === undefined
    )
      return undefined;
    return makeTransverseMercator(
      ellipsoid,
      scale,
      centralMeridian,
      latitudeOrigin,
      falseEasting,
      falseNorthing
    );
  }
  return undefined;
}

/** Parses a declared CRS (EPSG code, alias, canonical tmerc label or proj4 parameters). */
export function parseProjectionDescriptor(declared: string): ProjectionDescriptor | undefined {
  const text = declared.trim();
  const upper = text.toUpperCase();
  if (LONGLAT_ALIASES.has(upper)) return {kind: 'longlat', label: 'EPSG:4326'};
  if (MERCATOR_ALIASES.has(upper)) return {kind: 'mercator', label: 'EPSG:3857'};
  const epsg = /^(?:EPSG:?)?(\d{4,6})$/.exec(upper);
  if (epsg) return parseEpsgCode(Number(epsg[1]));
  if (text.startsWith('tmerc:')) {
    const fields = new Map(
      text
        .slice(6)
        .split(';')
        .map(part => part.split('=') as [string, string])
    );
    const read = (key: string) => Number(fields.get(key));
    const values = ['a', 'f', 'k', 'lon0', 'lat0', 'x0', 'y0'].map(read);
    if (values.some(value => !Number.isFinite(value))) return undefined;
    return makeTransverseMercator(
      [values[0], values[1]],
      values[2],
      values[3],
      values[4],
      values[5],
      values[6]
    );
  }
  if (text.startsWith('+')) return parseProj4Parameters(text);
  return undefined;
}

/** Canonical label for a declared CRS, or undefined when the GPU join cannot run in it. */
export function normalizeProjectionLabel(declared: string): string | undefined {
  return parseProjectionDescriptor(declared)?.label;
}

const DEGREES = Math.PI / 180;

type SeriesConstants = {
  n: number;
  eccentricity: number;
  rectifyingRadius: number;
  alpha: [number, number, number, number];
  beta: [number, number, number, number];
};

/** Krueger series constants to order n^4 (Karney 2011). */
export function getSeriesConstants(semiMajorAxis: number, flattening: number): SeriesConstants {
  const n = flattening / (2 - flattening);
  const n2 = n * n;
  const n3 = n2 * n;
  const n4 = n2 * n2;
  return {
    n,
    eccentricity: Math.sqrt(flattening * (2 - flattening)),
    rectifyingRadius: (semiMajorAxis / (1 + n)) * (1 + n2 / 4 + n4 / 64),
    alpha: [
      n / 2 - (2 / 3) * n2 + (5 / 16) * n3 + (41 / 180) * n4,
      (13 / 48) * n2 - (3 / 5) * n3 + (557 / 1440) * n4,
      (61 / 240) * n3 - (103 / 140) * n4,
      (49561 / 161280) * n4
    ],
    beta: [
      n / 2 - (2 / 3) * n2 + (37 / 96) * n3 - (1 / 360) * n4,
      n2 / 48 + n3 / 15 - (437 / 1440) * n4,
      (17 / 480) * n3 - (37 / 840) * n4,
      (4397 / 161280) * n4
    ]
  };
}

function forwardUnscaled(
  constants: SeriesConstants,
  latitudeRadians: number,
  longitudeRadians: number
): [number, number] {
  const {eccentricity, alpha} = constants;
  const tau = Math.tan(latitudeRadians);
  const sigma = Math.sinh(eccentricity * Math.atanh((eccentricity * tau) / Math.hypot(1, tau)));
  const tauPrime = tau * Math.hypot(1, sigma) - sigma * Math.hypot(1, tau);
  const xiPrime = Math.atan2(tauPrime, Math.cos(longitudeRadians));
  const etaPrime = Math.asinh(
    Math.sin(longitudeRadians) / Math.hypot(tauPrime, Math.cos(longitudeRadians))
  );
  let xi = xiPrime;
  let eta = etaPrime;
  for (let term = 1; term <= 4; term++) {
    xi += alpha[term - 1] * Math.sin(2 * term * xiPrime) * Math.cosh(2 * term * etaPrime);
    eta += alpha[term - 1] * Math.cos(2 * term * xiPrime) * Math.sinh(2 * term * etaPrime);
  }
  return [eta, xi];
}

/** CPU reference (double precision) of the forward transform in degrees to projected meters. */
export function projectToTransverseMercator(
  descriptor: Extract<ProjectionDescriptor, {kind: 'tmerc'}>,
  longitudeDegrees: number,
  latitudeDegrees: number
): [number, number] {
  const constants = getSeriesConstants(descriptor.semiMajorAxis, descriptor.flattening);
  const [eta, xi] = forwardUnscaled(
    constants,
    latitudeDegrees * DEGREES,
    (longitudeDegrees - descriptor.centralMeridianDegrees) * DEGREES
  );
  const origin = forwardUnscaled(constants, descriptor.latitudeOriginDegrees * DEGREES, 0)[1];
  const radius = constants.rectifyingRadius * descriptor.scale;
  return [
    descriptor.falseEasting + radius * eta,
    descriptor.falseNorthing + radius * (xi - origin)
  ];
}

/** CPU reference of the inverse transform: projected meters to degrees. */
export function unprojectFromTransverseMercator(
  descriptor: Extract<ProjectionDescriptor, {kind: 'tmerc'}>,
  easting: number,
  northing: number
): [number, number] {
  const constants = getSeriesConstants(descriptor.semiMajorAxis, descriptor.flattening);
  const {eccentricity, beta} = constants;
  const radius = constants.rectifyingRadius * descriptor.scale;
  const origin = forwardUnscaled(constants, descriptor.latitudeOriginDegrees * DEGREES, 0)[1];
  const eta = (easting - descriptor.falseEasting) / radius;
  const xi = (northing - descriptor.falseNorthing) / radius + origin;
  let xiPrime = xi;
  let etaPrime = eta;
  for (let term = 1; term <= 4; term++) {
    xiPrime -= beta[term - 1] * Math.sin(2 * term * xi) * Math.cosh(2 * term * eta);
    etaPrime -= beta[term - 1] * Math.cos(2 * term * xi) * Math.sinh(2 * term * eta);
  }
  const tauPrime = Math.sin(xiPrime) / Math.hypot(Math.sinh(etaPrime), Math.cos(xiPrime));
  const longitude = Math.atan2(Math.sinh(etaPrime), Math.cos(xiPrime));
  let tau = tauPrime;
  const oneMinusESquared = 1 - eccentricity * eccentricity;
  for (let iteration = 0; iteration < 6; iteration++) {
    const sigma = Math.sinh(eccentricity * Math.atanh((eccentricity * tau) / Math.hypot(1, tau)));
    const tauPrimeEstimate = tau * Math.hypot(1, sigma) - sigma * Math.hypot(1, tau);
    tau +=
      ((tauPrime - tauPrimeEstimate) / Math.hypot(1, tauPrimeEstimate)) *
      ((1 + oneMinusESquared * tau * tau) / (oneMinusESquared * Math.hypot(1, tau)));
  }
  return [descriptor.centralMeridianDegrees + longitude / DEGREES, Math.atan(tau) / DEGREES];
}

const EARTH_RADIUS = 6378137;
const MAXIMUM_MERCATOR_LATITUDE = 85.0511287798066;

/** CPU reference: any supported descriptor to longitude/latitude degrees. */
export function unprojectDescriptor(
  descriptor: ProjectionDescriptor,
  x: number,
  y: number
): [number, number] {
  if (descriptor.kind === 'longlat') return [x, y];
  if (descriptor.kind === 'mercator')
    return [
      x / EARTH_RADIUS / DEGREES,
      (2 * Math.atan(Math.exp(y / EARTH_RADIUS)) - Math.PI / 2) / DEGREES
    ];
  return unprojectFromTransverseMercator(descriptor, x, y);
}

/** CPU reference: longitude/latitude degrees to any supported descriptor. */
export function projectDescriptor(
  descriptor: ProjectionDescriptor,
  longitude: number,
  latitude: number
): [number, number] {
  if (descriptor.kind === 'longlat') return [longitude, latitude];
  if (descriptor.kind === 'mercator') {
    const clamped = Math.max(
      -MAXIMUM_MERCATOR_LATITUDE,
      Math.min(MAXIMUM_MERCATOR_LATITUDE, latitude)
    );
    return [
      EARTH_RADIUS * longitude * DEGREES,
      EARTH_RADIUS * Math.log(Math.tan(Math.PI / 4 + (clamped * DEGREES) / 2))
    ];
  }
  return projectToTransverseMercator(descriptor, longitude, latitude);
}

function toWgslNumber(value: number): string {
  const text = String(value);
  return /[.e]/i.test(text) ? text : `${text}.0`;
}

/** WGSL functions unprojectPoint(vec2<f32>) -> lon/lat degrees and projectPoint(vec2<f32>) for a descriptor. */
export function makeProjectionWgsl(name: string, descriptor: ProjectionDescriptor): string {
  if (descriptor.kind === 'longlat')
    return `fn ${name}Unproject(p: vec2<f32>) -> vec2<f32> { return p; }\nfn ${name}Project(p: vec2<f32>) -> vec2<f32> { return p; }`;
  if (descriptor.kind === 'mercator')
    return `
fn ${name}Unproject(p: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(p.x / ${EARTH_RADIUS}.0 * 57.29577951308232, (2.0 * atan(exp(p.y / ${EARTH_RADIUS}.0)) - 1.5707963267948966) * 57.29577951308232);
}
fn ${name}Project(p: vec2<f32>) -> vec2<f32> {
  let latitude = clamp(p.y, -${MAXIMUM_MERCATOR_LATITUDE}, ${MAXIMUM_MERCATOR_LATITUDE});
  return vec2<f32>(${EARTH_RADIUS}.0 * p.x * 0.017453292519943295, ${EARTH_RADIUS}.0 * log(tan(0.7853981633974483 + latitude * 0.008726646259971648)));
}`;
  const constants = getSeriesConstants(descriptor.semiMajorAxis, descriptor.flattening);
  const radius = constants.rectifyingRadius * descriptor.scale;
  const origin = forwardUnscaled(constants, descriptor.latitudeOriginDegrees * DEGREES, 0)[1];
  const list = (values: readonly number[]) => values.map(toWgslNumber).join(', ');
  const e = toWgslNumber(constants.eccentricity);
  return `
const ${name}Alpha = vec4<f32>(${list(constants.alpha)});
const ${name}Beta = vec4<f32>(${list(constants.beta)});
fn ${name}Project(p: vec2<f32>) -> vec2<f32> {
  let lambda = (p.x - ${toWgslNumber(descriptor.centralMeridianDegrees)}) * 0.017453292519943295;
  let phi = p.y * 0.017453292519943295;
  let tau = tan(phi);
  let sigma = sinh(${e} * atanh(${e} * tau / sqrt(1.0 + tau * tau)));
  let tauPrime = tau * sqrt(1.0 + sigma * sigma) - sigma * sqrt(1.0 + tau * tau);
  let xiPrime = atan2(tauPrime, cos(lambda));
  let etaPrime = asinh(sin(lambda) / sqrt(tauPrime * tauPrime + cos(lambda) * cos(lambda)));
  var xi = xiPrime;
  var eta = etaPrime;
  for (var term = 1u; term <= 4u; term = term + 1u) {
    let t = f32(term) * 2.0;
    let a = ${name}Alpha[term - 1u];
    xi = xi + a * sin(t * xiPrime) * cosh(t * etaPrime);
    eta = eta + a * cos(t * xiPrime) * sinh(t * etaPrime);
  }
  return vec2<f32>(${toWgslNumber(descriptor.falseEasting)} + ${toWgslNumber(radius)} * eta, ${toWgslNumber(descriptor.falseNorthing)} + ${toWgslNumber(radius)} * (xi - ${toWgslNumber(origin)}));
}
fn ${name}Unproject(p: vec2<f32>) -> vec2<f32> {
  let eta = (p.x - ${toWgslNumber(descriptor.falseEasting)}) / ${toWgslNumber(radius)};
  let xi = (p.y - ${toWgslNumber(descriptor.falseNorthing)}) / ${toWgslNumber(radius)} + ${toWgslNumber(origin)};
  var xiPrime = xi;
  var etaPrime = eta;
  for (var term = 1u; term <= 4u; term = term + 1u) {
    let t = f32(term) * 2.0;
    let b = ${name}Beta[term - 1u];
    xiPrime = xiPrime - b * sin(t * xi) * cosh(t * eta);
    etaPrime = etaPrime - b * cos(t * xi) * sinh(t * eta);
  }
  let tauPrime = sin(xiPrime) / sqrt(sinh(etaPrime) * sinh(etaPrime) + cos(xiPrime) * cos(xiPrime));
  let lambda = atan2(sinh(etaPrime), cos(xiPrime));
  var tau = tauPrime;
  for (var iteration = 0u; iteration < 5u; iteration = iteration + 1u) {
    let sigma = sinh(${e} * atanh(${e} * tau / sqrt(1.0 + tau * tau)));
    let estimate = tau * sqrt(1.0 + sigma * sigma) - sigma * sqrt(1.0 + tau * tau);
    tau = tau + (tauPrime - estimate) / sqrt(1.0 + estimate * estimate) * (1.0 + ${toWgslNumber(1 - constants.eccentricity ** 2)} * tau * tau) / (${toWgslNumber(1 - constants.eccentricity ** 2)} * sqrt(1.0 + tau * tau));
  }
  return vec2<f32>(${toWgslNumber(descriptor.centralMeridianDegrees)} + lambda * 57.29577951308232, atan(tau) * 57.29577951308232);
}`;
}
