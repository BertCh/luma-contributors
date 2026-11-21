// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Shared power-of-two scaling leaves ratios unchanged and lifts tiny SSDs before squaring. @internal */
export const DETECTOR_NORMALIZATION_WGSL = /* wgsl */ `
fn detectorNormalizationShift(maximum:u32)->i32 {
  if(maximum==0u || maximum>=0x2f800000u){return 0i;}
  let exponent=(maximum>>23u)&255u;
  if(exponent!=0u){return 127i-i32(exponent);}
  return 149i-i32(31u-countLeadingZeros(maximum));
}
fn normalizeDetectorValue(value:f32,shift:i32)->f32 {
  let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;
  let exponent=magnitude>>23u;let fraction=magnitude&0x7fffffu;
  if(shift==0i || magnitude==0u || exponent==255u){return value;}
  if(exponent==0u){
    let scaled=f32(fraction)*bitcast<f32>(u32(127i+shift-149i)<<23u);
    return bitcast<f32>(bitcast<u32>(scaled)|(bits&0x80000000u));
  }
  return bitcast<f32>((bits&0x807fffffu)|(u32(i32(exponent)+shift)<<23u));
}
fn restoreDetectorSumOfSquares(value:f32,shift:i32)->f32 {
  let bits=bitcast<u32>(value);let exponent=(bits>>23u)&255u;
  if(shift==0i || exponent==255u || (bits&0x7fffffffu)==0u){return value;}
  let targetExponent=i32(exponent)-2i*shift;
  if(targetExponent>0i){return bitcast<f32>((bits&0x807fffffu)|(u32(targetExponent)<<23u));}
  if(targetExponent< -23i){return 0.0;}
  let discarded=u32(1i-targetExponent);let significand=(bits&0x7fffffu)|0x800000u;
  let quotient=significand>>discarded;
  let remainder=significand&((1u<<discarded)-1u);let halfway=1u<<(discarded-1u);
  let rounded=quotient+u32(remainder>halfway || (remainder==halfway && (quotient&1u)!=0u));
  return bitcast<f32>(rounded);
}
`;

/** Normalize all usable response rows with one shared exponent; never read rows on the host. @internal */
export function createDetectorNormalizationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  values: GraphDataView<'float32'>,
  labels: GraphDataView<'uint32'>,
  labelCapacity: number
) {
  const magnitudeRows = createTransientView(graph, `${id}-magnitude-rows`, 'uint32', values.length);
  const magnitude = createTransientView(graph, `${id}-magnitude`, 'uint32', 1);
  const normalized = createTransientView(graph, `${id}-normalized`, 'float32', values.length);
  return {
    magnitude,
    normalized,
    nodes: [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-magnitude-rows`,
        operation: 'GPUGeographicalDetector',
        invocationCount: values.length,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'labels', view: labels, type: 'u32', access: 'read'},
          {name: 'magnitudeRows', view: magnitudeRows, type: 'u32', access: 'read_write'}
        ],
        body: `let bits=bitcast<u32>(values[valuesOffset+index])&0x7fffffffu;
magnitudeRows[magnitudeRowsOffset+index]=select(0u,bits,bits<0x7f800000u && labels[labelsOffset+index]<${labelCapacity}u);`
      }),
      ...new GPUReduction({
        id: `${id}-magnitude`,
        input: magnitudeRows,
        output: magnitude,
        operation: 'max'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-normalize-values`,
        operation: 'GPUGeographicalDetector',
        invocationCount: values.length,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'magnitude', view: magnitude, type: 'u32', access: 'read'},
          {name: 'normalized', view: normalized, type: 'f32', access: 'read_write'}
        ],
        declarations: DETECTOR_NORMALIZATION_WGSL,
        body: 'normalized[normalizedOffset+index]=normalizeDetectorValue(values[valuesOffset+index],detectorNormalizationShift(magnitude[magnitudeOffset]));'
      })
    ]
  };
}
