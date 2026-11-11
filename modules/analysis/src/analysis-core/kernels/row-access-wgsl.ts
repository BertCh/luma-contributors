// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphVectorView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUUint32Rows} from './gpu-contributor-types';
import {getWGSLKernelBinding, type WGSLKernelBinding} from './wgsl-kernel-nodes';

/**
 * Row accessors and the mask test for WGSL kernels built with `createWGSLKernelNode` (contributor
 * contract rule 11). Both read bindings by name through the generated `${name}Offset` constant, so
 * they pair with {@link getWGSLKernelBinding}.
 */

/** Options of {@link getRowAccessorWGSL}. @internal */
export type RowAccessorOptions = {
  /** Scalar type of the binding. Defaults to `'f32'`. */
  scalarType?: 'f32' | 'u32' | 'i32';
  /** Elements between consecutive rows. Defaults to the component count (interleaved rows). */
  stride?: number;
  /** Also declare the writer. Defaults to false. */
  writable?: boolean;
  /**
   * Name stem of the generated functions. Defaults to the binding name with its first letter in
   * upper case: binding `positions` gives `readPositions(row)` and `writePositions(row, value)`.
   */
  functionStem?: string;
};

/**
 * Declares `fn read<Stem>(row: u32) -> T` for a binding of interleaved rows, where `T` is the scalar
 * type for one component and `vecN<scalar>` for 2 to 4, and with `writable` also
 * `fn write<Stem>(row: u32, value: T)`. Row `row` starts at element `${bindingName}Offset +
 * row * stride`. Replaces the local `getPosition`, `readPoint`, `loadState` lambdas.
 *
 * @param bindingName WGSL binding name.
 * @param componentCount Components per row, 1 to 4.
 * @param options Scalar type, stride, writer and function names.
 * @internal
 */
export function getRowAccessorWGSL(
  bindingName: string,
  componentCount: number,
  options: RowAccessorOptions = {}
): string {
  const scalarType = options.scalarType ?? 'f32';
  const stride = options.stride ?? componentCount;
  const stem = options.functionStem ?? bindingName.charAt(0).toUpperCase() + bindingName.slice(1);
  if (!Number.isInteger(componentCount) || componentCount < 1 || componentCount > 4) {
    throw new Error(
      `getRowAccessorWGSL componentCount must be an integer in [1, 4], got ${componentCount}`
    );
  }
  if (!Number.isInteger(stride) || stride < componentCount) {
    throw new Error(
      `getRowAccessorWGSL stride must be an integer of at least componentCount (${componentCount}), got ${stride}`
    );
  }
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(bindingName) || !/^[A-Za-z0-9_]+$/.test(stem)) {
    throw new Error(`getRowAccessorWGSL bindingName and functionStem must be WGSL identifiers`);
  }
  const type = componentCount === 1 ? scalarType : `vec${componentCount}<${scalarType}>`;
  const element = (component: number): string =>
    `${bindingName}[base${component === 0 ? '' : ` + ${component}u`}]`;
  const components = Array.from({length: componentCount}, (_, component) => component);
  const reader = `fn read${stem}(row: u32) -> ${type} {
  let base = ${bindingName}Offset + row * ${stride}u;
  return ${componentCount === 1 ? element(0) : `${type}(${components.map(element).join(', ')})`};
}
`;
  if (!options.writable) {
    return reader;
  }
  const writes = components
    .map(component =>
      componentCount === 1
        ? `${element(0)} = value;`
        : `${element(component)} = value[${component}];`
    )
    .join('\n  ');
  return `${reader}fn write${stem}(row: u32, value: ${type}) {
  let base = ${bindingName}Offset + row * ${stride}u;
  ${writes}
}
`;
}

/**
 * Resolves an optional mask (`GPUUint32Rows`, the `mask` of the spatial ports) to the packed view
 * a kernel binds: the view itself, the single chunk of a one-chunk vector view, or `undefined`
 * when there is no mask. A mask split over several chunks throws; pack it first.
 *
 * @param id Contributor ID, for the message.
 * @param mask Optional mask rows.
 * @internal
 */
export function getMaskView(
  id: string,
  mask: GPUUint32Rows | undefined
): GraphDataView<'uint32'> | undefined {
  if (!mask) {
    return undefined;
  }
  if (!(mask instanceof GraphVectorView)) {
    return mask;
  }
  if (mask.data.length !== 1) {
    throw new Error(`${id} mask must be one packed view, got ${mask.data.length} chunks`);
  }
  return mask.data[0];
}

/**
 * The storage binding list for an optional mask: one read-only `u32` binding named `bindingName`,
 * or none. Spread it into the `bindings` of `createWGSLKernelNode` next to
 * {@link getMaskTestWGSL} with the same arguments.
 *
 * @internal
 */
export function getMaskBindings(
  mask: GraphDataView<'uint32'> | undefined,
  bindingName: string = 'mask'
): WGSLKernelBinding[] {
  return mask ? [getWGSLKernelBinding(bindingName, mask)] : [];
}

/**
 * Declares `fn <functionName>(row: u32) -> bool`: true when row `row` is selected. With a mask a
 * row is selected when its word is not zero (the convention of the spatial ports and
 * `GPUScalarScale`); without one every row is selected, and the generated function reads no
 * binding, so the same kernel body serves both cases.
 *
 * @param mask Optional packed mask view (see {@link getMaskView}); only its presence matters.
 * @param bindingName WGSL binding name of the mask. Defaults to `'mask'`.
 * @param functionName Generated function name. Defaults to `'isRowSelected'`.
 * @internal
 */
export function getMaskTestWGSL(
  mask: GraphDataView<'uint32'> | undefined,
  bindingName: string = 'mask',
  functionName: string = 'isRowSelected'
): string {
  return mask
    ? `fn ${functionName}(row: u32) -> bool { return ${bindingName}[${bindingName}Offset + row] != 0u; }\n`
    : `fn ${functionName}(row: u32) -> bool { return true; }\n`;
}
