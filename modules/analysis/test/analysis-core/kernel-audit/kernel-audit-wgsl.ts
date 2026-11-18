// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Node-side shader binding metadata; WGSL resource names may differ from host names. */
export type WGSLDeclaredBinding = {name: string; group?: number; location?: number};

/**
 * Static checks over generated WGSL that `wgsl_reflect` does not make, for the library-wide kernel
 * audit (quality rule 16 plus the compile traps of BRIEF rule 6). Every check here caught, or would
 * have caught, a kernel that Dawn rejected on a real GPU while `wgsl_reflect` accepted it.
 */

/** WGSL keywords (WGSL spec, "Keyword Summary"). */
export const WGSL_KEYWORDS: ReadonlySet<string> = new Set([
  'alias',
  'break',
  'case',
  'const',
  'const_assert',
  'continue',
  'continuing',
  'default',
  'diagnostic',
  'discard',
  'else',
  'enable',
  'false',
  'fn',
  'for',
  'if',
  'let',
  'loop',
  'override',
  'requires',
  'return',
  'struct',
  'switch',
  'true',
  'var',
  'while'
]);

/**
 * WGSL reserved words (WGSL spec, "Reserved Words"). Any use as an identifier is a shader-creation
 * error, wherever it appears (declaration, parameter, struct member or reference).
 */
export const WGSL_RESERVED_WORDS: ReadonlySet<string> = new Set([
  'NULL',
  'Self',
  'abstract',
  'active',
  'alignas',
  'alignof',
  'as',
  'asm',
  'asm_fragment',
  'async',
  'attribute',
  'auto',
  'await',
  'become',
  'cast',
  'catch',
  'class',
  'co_await',
  'co_return',
  'co_yield',
  'coherent',
  'column_major',
  'common',
  'compile',
  'compile_fragment',
  'concept',
  'const_cast',
  'consteval',
  'constexpr',
  'constinit',
  'crate',
  'debugger',
  'decltype',
  'delete',
  'demote',
  'demote_to_helper',
  'do',
  'dynamic_cast',
  'enum',
  'explicit',
  'export',
  'extends',
  'extern',
  'external',
  'fallthrough',
  'filter',
  'final',
  'finally',
  'friend',
  'from',
  'fxgroup',
  'get',
  'goto',
  'groupshared',
  'highp',
  'impl',
  'implements',
  'import',
  'inline',
  'instanceof',
  'interface',
  'layout',
  'lowp',
  'macro',
  'macro_rules',
  'match',
  'mediump',
  'meta',
  'mod',
  'module',
  'move',
  'mut',
  'mutable',
  'namespace',
  'new',
  'nil',
  'noexcept',
  'noinline',
  'nointerpolation',
  'non_coherent',
  'noncoherent',
  'noperspective',
  'null',
  'nullptr',
  'of',
  'operator',
  'package',
  'packoffset',
  'partition',
  'pass',
  'patch',
  'pixelfragment',
  'precise',
  'precision',
  'premerge',
  'priv',
  'protected',
  'pub',
  'public',
  'readonly',
  'ref',
  'regardless',
  'register',
  'reinterpret_cast',
  'require',
  'resource',
  'restrict',
  'self',
  'set',
  'shared',
  'sizeof',
  'smooth',
  'snorm',
  'static',
  'static_assert',
  'static_cast',
  'std',
  'subroutine',
  'super',
  'target',
  'template',
  'this',
  'thread_local',
  'throw',
  'trait',
  'try',
  'type',
  'typedef',
  'typeid',
  'typename',
  'typeof',
  'union',
  'unless',
  'unorm',
  'unsafe',
  'unsized',
  'use',
  'using',
  'varying',
  'virtual',
  'volatile',
  'wgsl',
  'where',
  'with',
  'writeonly',
  'yield'
]);

/** WGSL built-in functions (WGSL spec, "Built-in Functions", including subgroup and quad). */
const WGSL_BUILTIN_FUNCTIONS: ReadonlySet<string> = new Set([
  'bitcast',
  'all',
  'any',
  'select',
  'arrayLength',
  'abs',
  'acos',
  'acosh',
  'asin',
  'asinh',
  'atan',
  'atanh',
  'atan2',
  'ceil',
  'clamp',
  'cos',
  'cosh',
  'countLeadingZeros',
  'countOneBits',
  'countTrailingZeros',
  'cross',
  'degrees',
  'determinant',
  'distance',
  'dot',
  'dot4U8Packed',
  'dot4I8Packed',
  'exp',
  'exp2',
  'extractBits',
  'faceForward',
  'firstLeadingBit',
  'firstTrailingBit',
  'floor',
  'fma',
  'fract',
  'frexp',
  'insertBits',
  'inverseSqrt',
  'ldexp',
  'length',
  'log',
  'log2',
  'max',
  'min',
  'mix',
  'modf',
  'normalize',
  'pow',
  'quantizeToF16',
  'radians',
  'reflect',
  'refract',
  'reverseBits',
  'round',
  'saturate',
  'sign',
  'sin',
  'sinh',
  'smoothstep',
  'sqrt',
  'step',
  'tan',
  'tanh',
  'transpose',
  'trunc',
  'dpdx',
  'dpdxCoarse',
  'dpdxFine',
  'dpdy',
  'dpdyCoarse',
  'dpdyFine',
  'fwidth',
  'fwidthCoarse',
  'fwidthFine',
  'textureDimensions',
  'textureGather',
  'textureGatherCompare',
  'textureLoad',
  'textureNumLayers',
  'textureNumLevels',
  'textureNumSamples',
  'textureSample',
  'textureSampleBias',
  'textureSampleCompare',
  'textureSampleCompareLevel',
  'textureSampleGrad',
  'textureSampleLevel',
  'textureSampleBaseClampToEdge',
  'textureStore',
  'atomicLoad',
  'atomicStore',
  'atomicAdd',
  'atomicSub',
  'atomicMax',
  'atomicMin',
  'atomicAnd',
  'atomicOr',
  'atomicXor',
  'atomicExchange',
  'atomicCompareExchangeWeak',
  'pack4x8snorm',
  'pack4x8unorm',
  'pack4xI8',
  'pack4xU8',
  'pack4xI8Clamp',
  'pack4xU8Clamp',
  'pack2x16snorm',
  'pack2x16unorm',
  'pack2x16float',
  'unpack4x8snorm',
  'unpack4x8unorm',
  'unpack4xI8',
  'unpack4xU8',
  'unpack2x16snorm',
  'unpack2x16unorm',
  'unpack2x16float',
  'storageBarrier',
  'textureBarrier',
  'workgroupBarrier',
  'workgroupUniformLoad',
  'subgroupAdd',
  'subgroupExclusiveAdd',
  'subgroupInclusiveAdd',
  'subgroupAll',
  'subgroupAnd',
  'subgroupAny',
  'subgroupBallot',
  'subgroupBroadcast',
  'subgroupBroadcastFirst',
  'subgroupElect',
  'subgroupMax',
  'subgroupMin',
  'subgroupMul',
  'subgroupExclusiveMul',
  'subgroupInclusiveMul',
  'subgroupOr',
  'subgroupShuffle',
  'subgroupShuffleDown',
  'subgroupShuffleUp',
  'subgroupShuffleXor',
  'subgroupXor',
  'quadBroadcast',
  'quadSwapDiagonal',
  'quadSwapX',
  'quadSwapY'
]);

/** Predeclared type generators and scalar types that may be called as value constructors. */
const WGSL_CONSTRUCTOR_TYPES =
  /^(?:bool|i32|u32|f32|f16|array|vec[234][ifuh]?|mat[234]x[234][fh]?)$/;

/** One audit finding inside one WGSL module. */
export type WGSLFinding = {
  /** Check that failed. */
  check: string;
  /** Short description with the offending text. */
  detail: string;
};

import {
  evaluateIntegerExpression,
  getWGSLConstantValues,
  getWGSLModuleUsage,
  stripWGSLComments
} from '../../../src/testing/wgsl-memory-usage';
export {
  evaluateIntegerExpression,
  getWGSLConstantValues,
  getWGSLModuleUsage,
  stripWGSLComments,
  WGSLTypeLayouts
} from '../../../src/testing/wgsl-memory-usage';
export type {WGSLModuleUsage} from '../../../src/testing/wgsl-memory-usage';

/** Rejects reserved words used anywhere as identifiers, and keywords in declaration positions. */
export function findReservedIdentifiers(code: string): WGSLFinding[] {
  const findings = new Map<string, WGSLFinding>();
  for (const match of code.matchAll(/[A-Za-z_][A-Za-z0-9_]*/g)) {
    if (WGSL_RESERVED_WORDS.has(match[0])) {
      const context = code.slice(Math.max(0, match.index - 24), match.index + 24).trim();
      findings.set(match[0], {
        check: 'reserved word',
        detail: `\`${match[0]}\` in "${context.replace(/\s+/g, ' ')}"`
      });
    }
  }
  const declaration =
    /\b(?:let|var(?:<[^>]*>)?|const|override|fn|struct|alias)\s+([A-Za-z_]\w*)|[(,{]\s*([A-Za-z_]\w*)\s*:(?!:)/g;
  for (const match of code.matchAll(declaration)) {
    const name = match[1] ?? match[2];
    if (WGSL_KEYWORDS.has(name) && !(match[2] && name === 'default')) {
      findings.set(`keyword ${name}`, {check: 'keyword identifier', detail: `\`${name}\``});
    }
  }
  return [...findings.values()];
}

const OPERAND = String.raw`[A-Za-z_]\w*(?:\s*(?:\.\s*[A-Za-z_]\w*|\[[^\[\]]*\]))*`;
const SELF_COMPARE = new RegExp(
  String.raw`(^|[(,;{=?]|&&|\|\||\breturn\b)\s*(${OPERAND})\s*(==|!=)\s*(${OPERAND})\s*(?=$|[),;{\]]|&&|\|\|)`,
  'gm'
);

const F32_MAX = String.raw`3\.402823466e\+38`;

/**
 * Finite tests of upstream `@luma.gl/gpgpu` primitives that use the folded `v == v` NaN test
 * (`v == v && abs(v) <= F32_MAX`). The audit cannot fix upstream files (BRIEF rule 2), so these
 * exact idioms are filed as a request (see the T-AUDIT handoff) and ignored here. The analysis
 * sources themselves must not contain the idiom: `findSourceSelfCompareIdioms` checks that.
 */
export const UPSTREAM_SELF_COMPARE_IDIOMS: readonly {pattern: RegExp; source: string}[] = [
  {
    // gpu-bvh.ts:273,412, gpu-bvh-query.ts:343,428, gpu-segmented-bvh.ts:402,
    // gpu-spatial-utils.ts:19, gpu-grid-index-query.ts:420,603, gpu-point-spatial-filter.ts:322,
    // gpu-virtual-geometry-selection.ts:447, gpu-group-aggregation.ts:543, gpu-reduction.ts:419,
    // gpu-histogram.ts:438,565, gpu-vector-search/gpu-k-means.ts:403
    pattern: new RegExp(String.raw`\b(\w+) == \1 && abs\(\1\) <= ${F32_MAX}`, 'g'),
    source: 'published @luma.gl/graph finite(value) helpers and raster texture validity tests'
  },
  {
    // gpu-raster-band-math.js:216 (published @luma.gl/graph)
    pattern: /\b(\w+) == \1 && abs\(\1\) <= bitcast<f32>\(0x7f7fffffu\)/g,
    source: 'published @luma.gl/graph gpu-raster band-math finite(value) helper'
  },
  {
    pattern: new RegExp(String.raw`\btotal == total && abs\(total\) <= ${F32_MAX}`, 'g'),
    source: 'gpgpu/src/gpu-graph/gpu-graph-page-rank-internals.ts:466'
  },
  {
    pattern: new RegExp(
      String.raw`\bx == x && y == y && abs\(x\) <= ${F32_MAX} && abs\(y\) <= ${F32_MAX}`,
      'g'
    ),
    source: 'gpu-grid-index-internals.ts:271, gpu-grid-binning.ts:319, gpu-grid-aggregation.ts:376'
  },
  {
    pattern: new RegExp(String.raw`\bz == z && abs\(z\) <= ${F32_MAX}`, 'g'),
    source: 'gpu-grid-index-internals.ts:244'
  },
  {
    pattern: new RegExp(String.raw`\bweight == weight && abs\(weight\) <= ${F32_MAX}`, 'g'),
    source: 'gpu-grid-aggregation.ts:377'
  },
  {
    pattern: new RegExp(
      String.raw`(boundsValues\[BOUNDS_OFFSET \+ \d+u\]) == \1 && (boundsValues\[BOUNDS_OFFSET \+ \d+u\]) == \2 && abs\(\1\) <= ${F32_MAX}`,
      'g'
    ),
    source: 'gpu-grid-index-internals.ts:345'
  }
];

/** Rejects `x == x` / `x != x` NaN tests, which Tint and Metal fold to a constant. */
export function findSelfCompares(code: string): WGSLFinding[] {
  const findings: WGSLFinding[] = [];
  for (const {pattern} of UPSTREAM_SELF_COMPARE_IDIOMS) {
    code = code.replace(pattern, 'true');
  }
  for (const match of code.matchAll(SELF_COMPARE)) {
    const left = match[2].replace(/\s+/g, '');
    if (left === match[4].replace(/\s+/g, '')) {
      findings.push({check: 'NaN self-compare', detail: `\`${left} ${match[3]} ${left}\``});
    }
  }
  return findings;
}

function parseIntegerLiteral(text: string): number | undefined {
  const match = /^\s*(0[xX][0-9a-fA-F]+|\d+)[iu]?\s*$/.exec(text);
  return match ? Number(match[1]) : undefined;
}

/**
 * Rejects a const-expression `bitcast<f32>(<integer constant>)` whose bits are an infinity or NaN.
 * Const evaluation of a non-finite value is a shader-creation error; the value must come from a
 * runtime `var`.
 */
export function findConstantNonFiniteBitcasts(code: string): WGSLFinding[] {
  const constants = new Map<string, number>();
  for (const match of code.matchAll(/\bconst\s+(\w+)\s*(?::\s*\w+\s*)?=\s*([^;]+);/g)) {
    const value = parseIntegerLiteral(match[2]);
    if (value !== undefined) constants.set(match[1], value);
  }
  const findings: WGSLFinding[] = [];
  const bitcast = /bitcast<\s*(f32|f16|vec[234]<\s*f32\s*>|vec[234]f)\s*>\s*\(\s*([\w]+)\s*\)/g;
  for (const match of code.matchAll(bitcast)) {
    const value = parseIntegerLiteral(match[2]) ?? constants.get(match[2]);
    if (value === undefined) continue;
    const nonFinite =
      match[1] === 'f16' ? (value & 0x7c00) === 0x7c00 : (value & 0x7f800000) === 0x7f800000;
    if (nonFinite) {
      findings.push({check: 'const non-finite bitcast', detail: `\`${match[0]}\``});
    }
  }
  for (const match of code.matchAll(
    /(?<![\w.])(\d+(?:\.\d*)?(?:e[+-]?\d+)?[fh]?)\s*\/\s*(0(?:\.0*)?[fhiu]?)(?![\w.])/g
  )) {
    findings.push({check: 'const division by zero', detail: `\`${match[0]}\``});
  }
  return findings;
}

// A template list follows its name directly (`bitcast<f32>(`), so `a < vec2<f32>(0.0)` is no call of `a`.
const CALL = /(?<![\w.@])([A-Za-z_]\w*)(?:<\w[\w\s,<>]*>)?\s*\(/g;
const CALL_KEYWORDS = new Set(['if', 'for', 'while', 'switch', 'return', 'loop', 'else', 'case']);

/** Rejects calls to functions that the module neither declares nor gets from WGSL. */
export function findUndeclaredCalls(code: string): WGSLFinding[] {
  const declared = new Set<string>();
  for (const match of code.matchAll(/\b(?:fn|struct|alias)\s+([A-Za-z_]\w*)/g)) {
    declared.add(match[1]);
  }
  const findings = new Map<string, WGSLFinding>();
  for (const match of code.matchAll(CALL)) {
    const name = match[1];
    if (/\bfn\s*$/.test(code.slice(Math.max(0, match.index - 8), match.index))) continue;
    if (
      declared.has(name) ||
      CALL_KEYWORDS.has(name) ||
      WGSL_BUILTIN_FUNCTIONS.has(name) ||
      WGSL_CONSTRUCTOR_TYPES.test(name)
    ) {
      continue;
    }
    findings.set(name, {check: 'undeclared call', detail: `\`${name}(\``});
  }
  return [...findings.values()];
}

/**
 * Finds resource declarations without an identifier reference elsewhere in the module. Auto
 * pipeline layouts omit unused resources, even when the node's explicit binding record has them.
 * This is a syntactic check, not entry-point reachability or complete WGSL name resolution.
 * Comments, declaration text, longer identifiers and member names do not count as references.
 * With a node binding list, unused module-only declarations that the node never binds are valid.
 */
export function findUnusedBindings(
  source: string,
  declaredBindings?: readonly WGSLDeclaredBinding[],
  pipelineConstants: Readonly<Record<string, number>> = {}
): WGSLFinding[] {
  const code = stripWGSLComments(source);
  const identifier = String.raw`[_\p{XID_Start}][\p{XID_Continue}]*`;
  const declarations = new RegExp(
    String.raw`((?:@${identifier}\s*\([^;{}]*?\)\s*)+)var(?:\s*<[^;{}>]*>)?\s+(${identifier})\s*:[^;]*;`,
    'gu'
  );
  const bindings: {name: string; attributes: string}[] = [];
  const body = code.replace(
    declarations,
    (declaration: string, attributes: string, name: string) => {
      if (!/@binding\s*\(/.test(attributes) || !/@group\s*\(/.test(attributes)) return declaration;
      bindings.push({name, attributes: attributes.trim().replace(/\s+/g, ' ')});
      return ' ';
    }
  );
  if (bindings.length === 0) return [];
  const references = new Set<string>();
  for (const match of body.matchAll(
    new RegExp(String.raw`(?<![\p{XID_Continue}])${identifier}`, 'gu')
  )) {
    let previous = match.index - 1;
    while (previous >= 0 && /\s/.test(body[previous])) previous--;
    if (body[previous] === '.') continue;
    // A struct field or typed local declaration is not a reference to a global resource.
    let next = match.index + match[0].length;
    while (next < body.length && /\s/.test(body[next])) next++;
    if (body[next] === ':') continue;
    references.add(match[0]);
  }
  const constants = declaredBindings
    ? getWGSLConstantValues(code, pipelineConstants)
    : new Map<string, number>();
  const getAttributeValue = (attributes: string, name: string): number | undefined => {
    const attribute = new RegExp(`@${name}\\s*\\(`).exec(attributes);
    if (!attribute) return undefined;
    const first = attribute.index + attribute[0].length;
    let depth = 1;
    for (let index = first; index < attributes.length; index++) {
      if (attributes[index] === '(') depth++;
      if (attributes[index] === ')') depth--;
      if (depth === 0) return evaluateIntegerExpression(attributes.slice(first, index), constants);
    }
    return undefined;
  };
  return bindings
    .filter(({name, attributes}) => {
      if (references.has(name)) return false;
      if (!declaredBindings) return true;
      const group = getAttributeValue(attributes, 'group');
      const location = getAttributeValue(attributes, 'binding');
      return declaredBindings.some(binding =>
        binding.location !== undefined && group !== undefined && location !== undefined
          ? (binding.group ?? 0) === group && binding.location === location
          : binding.name === name
      );
    })
    .map(({name, attributes}) => ({check: 'unused binding', detail: `${name} (${attributes})`}));
}

/** Core WebGPU limits the audit checks every kernel against. */
export const CORE_SHADER_LIMITS = {
  maxStorageBuffersPerShaderStage: 8,
  maxUniformBuffersPerShaderStage: 12,
  maxStorageTexturesPerShaderStage: 4,
  maxSampledTexturesPerShaderStage: 16,
  maxSamplersPerShaderStage: 16,
  maxPrivateBytes: 8192,
  maxFunctionBytes: 8192,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSize: [256, 256, 64]
} as const;

/** Every static finding of one module: compile traps plus core-limit violations. */
export function auditWGSLModule(
  source: string,
  pipelineConstants: Readonly<Record<string, number>> = {},
  declaredBindings?: readonly WGSLDeclaredBinding[]
): WGSLFinding[] {
  const code = stripWGSLComments(source);
  const findings = [
    ...findReservedIdentifiers(code),
    ...findSelfCompares(code),
    ...findConstantNonFiniteBitcasts(code),
    ...findUndeclaredCalls(code),
    ...findUnusedBindings(code, declaredBindings, pipelineConstants)
  ];
  const usage = getWGSLModuleUsage(code, pipelineConstants);
  for (const unresolved of [...usage.unresolvedLengths, ...usage.unresolvedTypes]) {
    findings.push({check: 'unresolved memory layout', detail: unresolved});
  }
  const limits = CORE_SHADER_LIMITS;
  const over = (check: string, value: number, limit: number, detail = '') => {
    if (value > limit) findings.push({check, detail: `${value} > ${limit}${detail}`});
  };
  over('storage buffers per stage', usage.storageBuffers, limits.maxStorageBuffersPerShaderStage);
  over('uniform buffers per stage', usage.uniformBuffers, limits.maxUniformBuffersPerShaderStage);
  over(
    'storage textures per stage',
    usage.storageTextures,
    limits.maxStorageTexturesPerShaderStage
  );
  over(
    'sampled textures per stage',
    usage.sampledTextures,
    limits.maxSampledTexturesPerShaderStage
  );
  over('samplers per stage', usage.samplers, limits.maxSamplersPerShaderStage);
  over('private bytes', usage.privateBytes, limits.maxPrivateBytes);
  over(
    'function bytes',
    usage.functionBytes,
    limits.maxFunctionBytes,
    ` in fn ${usage.functionName}`
  );
  over('workgroup bytes', usage.workgroupStorageBytes, limits.maxComputeWorkgroupStorageSize);
  for (const size of usage.workgroupSizes) {
    const [x = 1, y = 1, z = 1] = size;
    if ([x, y, z].some(value => !Number.isFinite(value))) {
      findings.push({check: 'workgroup size', detail: `unresolved @workgroup_size(${size})`});
      continue;
    }
    over('workgroup invocations', x * y * z, limits.maxComputeInvocationsPerWorkgroup);
    limits.maxComputeWorkgroupSize.forEach((limit, axis) =>
      over(`workgroup size ${'xyz'[axis]}`, [x, y, z][axis], limit)
    );
  }
  return findings;
}
