// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {getPointCloudKernelNodes} from './pointcloud-kernel-nodes';
import {GPU_POINT_CLOUD_MAXIMUM_BRUTE_FORCE_WORK} from './pointcloud-local-origin';

export type GPUPointCloudChangeProps = {
  id?: string;
  reference: GraphDataView<'float32x3'>;
  comparison: GraphDataView<'float32x3'>;
  cores: GraphDataView<'float32x3'>;
  normals: GraphDataView<'float32x3'>;
  /** Explicit aligned-frame certification, or a column-major comparison-to-reference rigid transform (for example ICP output). */
  alignment: 'already-aligned' | {transform: GraphDataView<'float32'>};
  radius: number;
  halfDepth: number;
  minimumSupport?: number;
  /** Nonnegative registration uncertainty in coordinate units, added to the sampling standard error. */
  registrationError?: number;
  /** Optional aligned-frame translation covariance [xx,xy,xz,yy,yz,zz]; projected along each normal. */
  registrationCovariance?: readonly [number, number, number, number, number, number];
  /** Linear matches the M3C2 convention; quadrature combines independent Gaussian registration and sampling errors. */
  uncertaintyCombination?: 'linear' | 'quadrature';
  /** Two-sided normal quantile, default1.96 (95%); caller specifies their confidence convention. */
  confidenceMultiplier?: number;
  /** Optional known covariance of normal-direction measurement error: sigma² exp(-distance²/(2 length²)).
   * Uses all within-cylinder pairs exactly; variances are supplied separately from empirical roughness. */
  samplingCovariance?: {
    referenceVariance: number;
    comparisonVariance: number;
    correlationLength: number;
  };
  /** Signed comparison-minus-reference displacement, lower 95% bound, upper 95% bound. */
  changes: GraphDataView<'float32x3'>;
  /** Reference roughness, comparison roughness (sample standard deviations), 95% detection limit. */
  roughness: GraphDataView<'float32x3'>;
  /** Interleaved reference/comparison counts for each core. */
  counts: GraphDataView<'uint32'>;
  /** 0 valid, 1 invalid core/normal, 2 insufficient sample support, 3 nonfinite uncertainty. */
  status: GraphDataView<'uint32'>;
};

/**
 * M3C2-style cylindrical mean differences with explicit orientation, alignment and support.
 *
 * Precision: all positions are float32. At projected magnitudes (~1e6 m) absolute coordinates are
 * quantized to ~0.06 m, so make every cloud, core and alignment transform relative to one shared
 * float64 origin on the CPU (`getPointCloudLocalOrigin`, `makePointCloudLocalPositions`) and read
 * ICP transforms in that same local frame.
 *
 * Work: every core scans every reference and comparison point, so `cores * (reference +
 * comparison)` is limited to `GPU_POINT_CLOUD_MAXIMUM_BRUTE_FORCE_WORK`; `samplingCovariance`
 * additionally scans in-cylinder pairs and limits each cloud to 32768 points.
 */
export class GPUPointCloudChange {
  readonly id: string;
  constructor(readonly props: GPUPointCloudChangeProps) {
    this.id = props.id ?? 'gpu-pointcloud-change';
    assert(props.alignment === 'already-aligned' || props.alignment.transform.length === 16);
    assert(
      props.radius > 0 &&
        Number.isFinite(Math.fround(props.radius ** 2)) &&
        Math.fround(props.radius ** 2) > 0
    );
    assert(props.halfDepth > 0 && Number.isFinite(Math.fround(props.halfDepth)));
    assert(
      Number.isSafeInteger(props.minimumSupport ?? 3) &&
        (props.minimumSupport ?? 3) >= 2 &&
        (props.minimumSupport ?? 3) <= 0xffffffff
    );
    assert(
      (props.registrationError ?? 0) >= 0 &&
        Number.isFinite(Math.fround((props.registrationError ?? 0) ** 2))
    );
    assert(
      (props.confidenceMultiplier ?? 1.96) > 0 &&
        Number.isFinite(Math.fround(props.confidenceMultiplier ?? 1.96))
    );
    if (props.samplingCovariance) {
      const covariance = props.samplingCovariance;
      assert(
        [covariance.referenceVariance, covariance.comparisonVariance].every(
          value => value >= 0 && Number.isFinite(Math.fround(value))
        )
      );
      assert(
        covariance.correlationLength > 0 &&
          Number.isFinite(Math.fround(2 * covariance.correlationLength ** 2)) &&
          Math.fround(covariance.correlationLength ** 2) > 0
      );
    }
    if (props.registrationCovariance) {
      assert(props.registrationCovariance.every(value => Number.isFinite(Math.fround(value))));
      // Normalize before testing principal minors so validity is independent of coordinate units.
      const scale = Math.max(...props.registrationCovariance.map(Math.abs));
      const [xx, xy, xz, yy, yz, zz] = props.registrationCovariance.map(value =>
        scale > 0 ? value / scale : 0
      );
      const tolerance = 32 * Number.EPSILON;
      assert(
        xx >= 0 &&
          yy >= 0 &&
          zz >= 0 &&
          xx * yy - xy * xy >= -tolerance &&
          xx * zz - xz * xz >= -tolerance &&
          yy * zz - yz * yz >= -tolerance
      );
      assert(
        xx * yy * zz + 2 * xy * xz * yz - xx * yz * yz - yy * xz * xz - zz * xy * xy >= -tolerance
      );
    }
    // Exhaustive cylinder scan: cap cores x points per dispatch; subsample or tile the clouds for larger inputs.
    assert(
      props.cores.length * (props.reference.length + props.comparison.length) <=
        GPU_POINT_CLOUD_MAXIMUM_BRUTE_FORCE_WORK
    );
    // All-pairs sampling covariance is quadratic per core; bound each cloud so one thread stays short.
    if (props.samplingCovariance)
      assert(Math.max(props.reference.length, props.comparison.length) <= 32768);
    for (const view of [props.normals, props.changes, props.roughness, props.status])
      assert(view.length === props.cores.length);
    assert(props.counts.length === props.cores.length * 2);
  }
  getCommandNodes<Parameters>(graph: GPUCommandGraph<Parameters>) {
    const {reference, comparison, cores, normals, changes, roughness, counts, status} = this.props;
    const nodes: GPUCommandNode<Parameters>[] = [];
    let alignedComparison = comparison;
    if (this.props.alignment !== 'already-aligned') {
      alignedComparison = createTransientView(
        graph,
        `${this.id}-aligned-comparison`,
        'float32x3',
        comparison.length
      );
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-align`,
          length: comparison.length,
          inputs: {comparison, transform: this.props.alignment.transform},
          outputs: {aligned: alignedComparison},
          body: `let rotation = mat3x3f(vec3f(readTransform(0u),readTransform(1u),readTransform(2u)),vec3f(readTransform(4u),readTransform(5u),readTransform(6u)),vec3f(readTransform(8u),readTransform(9u),readTransform(10u)));
let translation = vec3f(readTransform(12u),readTransform(13u),readTransform(14u));
writeAligned(index,rotation*readComparison(index)+translation);`
        })
      );
    }
    nodes.push(
      ...getPointCloudKernelNodes(graph, {
        id: this.id,
        length: cores.length,
        inputs: {reference, comparison: alignedComparison, cores, normals},
        outputs: {changes, roughness, counts, status},
        body: `let core = readCores(index); let normal = safeNormal(readNormals(index));
writeChanges(index,vec3f(0.0)); writeRoughness(index,vec3f(0.0)); writeCounts(index*2u,0u); writeCounts(index*2u+1u,0u);
if (!all(abs(core) <= vec3f(3.402823e38)) || dot(normal,normal) < 0.5) { writeStatus(index,1u); return; }
${(['reference', 'comparison'] as const)
  .map(
    name => `
var ${name}Count = 0u; var ${name}Mean = 0.0; var ${name}Variance = 0.0;
for (var point = 0u; point < ${this.props[name].length}u; point++) {
  let difference = read${name[0].toUpperCase() + name.slice(1)}(point) - core;
  let distance = dot(difference,normal);
  let radial = difference - normal * distance;
  if (!(abs(distance) <= ${Math.fround(this.props.halfDepth)}) || !(dot(radial,radial) <= ${Math.fround(this.props.radius ** 2)})) { continue; }
  ${name}Count++; let delta = distance - ${name}Mean; ${name}Mean += delta / f32(${name}Count); ${name}Variance += delta * (distance - ${name}Mean);
}`
  )
  .join('\n')}
writeCounts(index*2u,referenceCount); writeCounts(index*2u+1u,comparisonCount);
if (referenceCount < ${this.props.minimumSupport ?? 3}u || comparisonCount < ${this.props.minimumSupport ?? 3}u) { writeStatus(index,2u); return; }
let referenceVarianceValue = max(referenceVariance / f32(referenceCount - 1u),0.0);
let comparisonVarianceValue = max(comparisonVariance / f32(comparisonCount - 1u),0.0);
${
  this.props.samplingCovariance
    ? (['reference', 'comparison'] as const)
        .map(name => {
          const read = `read${name[0].toUpperCase() + name.slice(1)}`;
          return `var ${name}CorrelationSum=0.0;
for(var first=0u;first<${this.props[name].length}u;first++){
 let firstPoint=${read}(first);let firstDelta=firstPoint-core;let firstNormal=dot(firstDelta,normal);let firstRadial=firstDelta-normal*firstNormal;
 if(!(abs(firstNormal)<=${Math.fround(this.props.halfDepth)})||!(dot(firstRadial,firstRadial)<=${Math.fround(this.props.radius ** 2)})){continue;}
 ${name}CorrelationSum+=1.0;
 for(var second=first+1u;second<${this.props[name].length}u;second++){
  let secondPoint=${read}(second);let secondDelta=secondPoint-core;let secondNormal=dot(secondDelta,normal);let secondRadial=secondDelta-normal*secondNormal;
  if(!(abs(secondNormal)<=${Math.fround(this.props.halfDepth)})||!(dot(secondRadial,secondRadial)<=${Math.fround(this.props.radius ** 2)})){continue;}
  let separation=firstPoint-secondPoint;${name}CorrelationSum+=2.0*exp(-dot(separation,separation)/${Math.fround(2 * this.props.samplingCovariance!.correlationLength ** 2)});
 }
}`;
        })
        .join('\n')
    : ''
}
let samplingVariance=${this.props.samplingCovariance ? `${Math.fround(this.props.samplingCovariance.referenceVariance)}*referenceCorrelationSum/(f32(referenceCount)*f32(referenceCount))+${Math.fround(this.props.samplingCovariance.comparisonVariance)}*comparisonCorrelationSum/(f32(comparisonCount)*f32(comparisonCount))` : 'referenceVarianceValue/f32(referenceCount)+comparisonVarianceValue/f32(comparisonCount)'};
let registrationCovariance=mat3x3f(${(() => {
          const [xx, xy, xz, yy, yz, zz] = this.props.registrationCovariance ?? [0, 0, 0, 0, 0, 0];
          return [xx, xy, xz, xy, yy, yz, xz, yz, zz].map(Math.fround).join(',');
        })()});
let registrationVariance=max(dot(normal,registrationCovariance*normal),0.0)+${Math.fround((this.props.registrationError ?? 0) ** 2)};
let detection=${Math.fround(this.props.confidenceMultiplier ?? 1.96)}*${this.props.uncertaintyCombination === 'quadrature' ? 'sqrt(samplingVariance+registrationVariance)' : '(sqrt(samplingVariance)+sqrt(registrationVariance))'};
let difference = comparisonMean - referenceMean;
// A finite input can still overflow an accumulated moment or published confidence interval.
if (any((bitcast<vec4u>(vec4f(referenceVarianceValue,comparisonVarianceValue,samplingVariance,registrationVariance)) & vec4u(0x7f800000u)) == vec4u(0x7f800000u)) ||
    any((bitcast<vec4u>(vec4f(detection,difference,difference-detection,difference+detection)) & vec4u(0x7f800000u)) == vec4u(0x7f800000u))) {
  writeStatus(index,3u); return;
}
writeChanges(index,vec3f(difference,difference-detection,difference+detection));
writeRoughness(index,vec3f(sqrt(referenceVarianceValue),sqrt(comparisonVarianceValue),detection)); writeStatus(index,0u);`
      })
    );
    return nodes;
  }
}
