// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUHealpixIndexScheme} from './gpu-healpix-point-to-cell';
import {HEALPIX_MAXIMUM_ORDER} from './healpix-index-wgsl';

/** OGC definition URI of the HEALPix DGGRS (OGC API - DGGS Part 1, DGGRS definition examples). */
export const HEALPIX_DGGRS_URI = 'https://www.opengis.net/def/dggrs/OGC/1.0/HEALPix';

/** Schema URI of the DGGS-JSON encoding. */
export const DGGS_JSON_SCHEMA_URI =
  'https://schemas.opengis.net/ogcapi/dggs/1.0/core/schemas/dggs-json/dggs-json.json';

/**
 * Largest relative depth of one zone-data document: `4^10` = 1,048,576 sub-zones per property, a
 * dense JSON array that is already tens of megabytes of text.
 */
export const HEALPIX_DGGS_JSON_MAXIMUM_RELATIVE_DEPTH = 10;

/** One property of the zone data: values aligned with the cell keys. */
export type HealpixDGGSJSONProperty = {
  /** Human-readable title (JSON Schema `title`). */
  title?: string;
  /** Unit of the values (`x-ogc-unit`). */
  unit?: string;
  /** One value per entry of `cells`; NaN and non-finite values become `null`. */
  values: ArrayLike<number>;
};

/** Inputs of {@link createHealpixDGGSJSON}. */
export type HealpixDGGSJSONProps = {
  /** HEALPix order of the cells. */
  order: number;
  /** Key scheme of `cells`, as written by `GPUHealpixPointToCell`. Defaults to `'nested'`. */
  scheme?: GPUHealpixIndexScheme;
  /** Order of the ancestral zone, at most `order`. Defaults to 0 (a base cell). */
  ancestorOrder?: number;
  /** Nested index of the ancestral zone at `ancestorOrder`. Defaults to 0n. */
  ancestor?: bigint;
  /** Read-back cell keys (for example `GPUHealpixCellAggregation` `output.cells`, first `count` entries). */
  cells: readonly bigint[];
  /** Named properties, each aligned with `cells`. */
  properties: Readonly<Record<string, HealpixDGGSJSONProperty>>;
  /** DGGRS URI. Defaults to {@link HEALPIX_DGGRS_URI}. */
  dggrs?: string;
};

/** A DGGS-JSON zone data document (OGC API - DGGS Part 1, requirements class zone-data JSON). */
export type HealpixDGGSJSON = {
  $schema: string;
  dggrs: string;
  zoneId: string;
  depths: number[];
  schema: {
    $schema: string;
    type: 'object';
    properties: Record<string, {type: 'number'; title?: string; 'x-ogc-unit'?: string}>;
  };
  values: Record<
    string,
    {depth: number; shape: {count: number; subZones: number}; data: (number | null)[]}[]
  >;
};

/**
 * Builds the DGGS-JSON zone data (OGC API - DGGS Part 1, approved 2025-10-24) of the cells below
 * one ancestral zone, from read-back aggregation output.
 *
 * Per the JSON schema (`dggs-json.json`: required `dggrs`, `zoneId`, `depths`, `values`; each value
 * entry has `depth`, `shape {count, subZones}` and a `data` array of numbers or null), the data of
 * a property at depth `d` is dense: one entry for each of the `4^d` descendants of the ancestor
 * at that relative depth, in ascending nested index, `null` where no cell was supplied.
 *
 * **Not OGC zone identifiers.** The OGC HEALPix DGGRS defines its own zone IDs and sub-zone order
 * (root rhombus hex digits and row-major scanlines, on the WGS84 ellipsoid), not the nested
 * index on the sphere that this library computes. `zoneId` here is the text
 * `healpix-nested:<ancestorOrder>:<ancestor>`, and `data` follows the nested order. A client that
 * expects the OGC ZIRS or sub-zone order needs a mapping that is not provided, and a geodetic
 * latitude on the WGS84 ellipsoid is not converted to the authalic latitude.
 *
 * Duplicated cells keep the last value. Cells outside the ancestor throw.
 */
export function createHealpixDGGSJSON(props: HealpixDGGSJSONProps): HealpixDGGSJSON {
  const ancestorOrder = props.ancestorOrder ?? 0;
  const ancestor = props.ancestor ?? 0n;
  const relativeDepth = props.order - ancestorOrder;
  if (
    !Number.isInteger(props.order) ||
    props.order > HEALPIX_MAXIMUM_ORDER ||
    !Number.isInteger(ancestorOrder) ||
    ancestorOrder < 0 ||
    relativeDepth < 0 ||
    relativeDepth > HEALPIX_DGGS_JSON_MAXIMUM_RELATIVE_DEPTH
  ) {
    throw new Error(
      `createHealpixDGGSJSON order - ancestorOrder must be an integer in [0, ${HEALPIX_DGGS_JSON_MAXIMUM_RELATIVE_DEPTH}] and order at most ${HEALPIX_MAXIMUM_ORDER}`
    );
  }
  if (ancestor < 0n || ancestor >= 12n * 4n ** BigInt(ancestorOrder)) {
    throw new Error('createHealpixDGGSJSON ancestor is not a nested index at ancestorOrder');
  }
  const subZones = 4 ** relativeDepth;
  const shift = 2n * BigInt(relativeDepth);
  const marker = props.scheme === 'uniq' ? 4n ** BigInt(props.order + 1) : 0n;
  const slots = props.cells.map(cell => {
    const nested = cell - marker;
    if (nested < 0n || nested >> shift !== ancestor) {
      throw new Error('createHealpixDGGSJSON a cell lies outside the ancestral zone');
    }
    return Number(nested - (ancestor << shift));
  });
  const document: HealpixDGGSJSON = {
    $schema: DGGS_JSON_SCHEMA_URI,
    dggrs: props.dggrs ?? HEALPIX_DGGRS_URI,
    zoneId: `healpix-nested:${ancestorOrder}:${ancestor}`,
    depths: [relativeDepth],
    schema: {
      $schema: 'https://json-schema.org/draft/2020-12/schema',
      type: 'object',
      properties: {}
    },
    values: {}
  };
  for (const [name, property] of Object.entries(props.properties)) {
    if (property.values.length < props.cells.length) {
      throw new Error(`createHealpixDGGSJSON property ${name} needs one value per cell`);
    }
    const data: (number | null)[] = new Array(subZones).fill(null);
    slots.forEach((slot, index) => {
      const value = property.values[index];
      data[slot] = Number.isFinite(value) ? value : null;
    });
    document.schema.properties[name] = {
      type: 'number',
      ...(property.title ? {title: property.title} : {}),
      ...(property.unit ? {'x-ogc-unit': property.unit} : {})
    };
    document.values[name] = [{depth: relativeDepth, shape: {count: subZones, subZones}, data}];
  }
  return document;
}
