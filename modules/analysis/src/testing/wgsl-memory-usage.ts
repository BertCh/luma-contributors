// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLBuiltinMembers, inferWGSLExpressionType} from './wgsl-expression-type';

/** Removes `//` line comments and (possibly nested) block comments, keeping line breaks. */
export function stripWGSLComments(source: string): string {
  let output = '';
  let depth = 0;
  for (let index = 0; index < source.length; index++) {
    const pair = source.slice(index, index + 2);
    if (pair === '/*') {
      if (depth === 0) output += ' ';
      depth++;
      index++;
    } else if (depth > 0 && pair === '*/') {
      depth--;
      index++;
    } else if (depth > 0) {
      if (source[index] === '\n') output += '\n';
    } else if (pair === '//') {
      while (index < source.length && source[index] !== '\n') index++;
      output += '\n';
    } else {
      output += source[index];
    }
  }
  return output;
}

// Memory and binding accounting.

function countWGSLStorageDeclarations(code: string): number {
  return (code.match(/\bvar\s*<\s*storage\b/g) ?? []).length;
}

/** Counts storage declarations with explicit access or WGSL's default read access. */
export function getWGSLStorageBindingCount(source: string): number {
  return countWGSLStorageDeclarations(stripWGSLComments(source));
}

type WGSLDeclaration = {
  kind: string;
  name: string;
  type?: string;
  expression?: string;
  addressSpace?: string;
  pipelineId?: string;
  scope: number;
  position: number;
  functionName?: string;
};

/** One declaration scanner for constants and all address spaces, preserving block scopes. */
function scanWGSLDeclarations(code: string) {
  const declarations: WGSLDeclaration[] = [];
  const scopes = [{parent: -1, functionName: undefined as string | undefined, loop: false}];
  const stack = [0];
  let pendingFunction: string | undefined;
  const tokens =
    /(?<![_\p{XID_Continue}])fn\s+([_\p{XID_Start}][_\p{XID_Continue}]*)\s*\(|(?<![_\p{XID_Continue}])for\s*\(|[{}]|(?<![_\p{XID_Continue}])(const|override|var|let)(?![_\p{XID_Continue}])/gu;
  for (let token = tokens.exec(code); token; token = tokens.exec(code)) {
    if (token[1]) {
      pendingFunction = token[1];
      continue;
    }
    if (/^for\b/.test(token[0])) {
      const parent = stack[stack.length - 1];
      scopes.push({parent, functionName: scopes[parent].functionName, loop: true});
      stack.push(scopes.length - 1);
      continue;
    }
    if (token[0] === '{') {
      const parent = stack[stack.length - 1];
      scopes.push({
        parent,
        functionName: pendingFunction ?? scopes[parent].functionName,
        loop: false
      });
      stack.push(scopes.length - 1);
      pendingFunction = undefined;
      continue;
    }
    if (token[0] === '}') {
      if (stack.length > 1) stack.pop();
      if (scopes[stack[stack.length - 1]].loop) stack.pop();
      continue;
    }
    const end = code.indexOf(';', tokens.lastIndex);
    if (end < 0) continue;
    const declaration =
      /^\s*(?:<\s*(\w+)(?:\s*,[^>]*)?\s*>\s*)?([_\p{XID_Start}][_\p{XID_Continue}]*)\s*([\s\S]*)$/u.exec(
        code.slice(tokens.lastIndex, end)
      );
    if (!declaration) continue;
    const [, addressSpace, name, rest] = declaration;
    const equals = rest.indexOf('=');
    const prefix = code.slice(
      Math.max(
        code.lastIndexOf(';', token.index),
        code.lastIndexOf('}', token.index),
        code.lastIndexOf('{', token.index)
      ) + 1,
      token.index
    );
    const pipelineId = /@id\s*\(\s*([^)]*)\)/.exec(prefix)?.[1];
    const scope = stack[stack.length - 1];
    declarations.push({
      kind: token[2],
      name,
      addressSpace,
      pipelineId,
      scope,
      position: token.index,
      functionName: scopes[scope].functionName,
      type: rest.startsWith(':')
        ? rest.slice(1, equals < 0 ? undefined : equals).trim()
        : undefined,
      expression: equals < 0 ? undefined : rest.slice(equals + 1).trim()
    });
    tokens.lastIndex = end + 1;
  }
  return {declarations, scopes};
}

/** Values of module `const` and `override` declarations, with numeric override IDs honored. */
export function getWGSLConstantValues(
  code: string,
  pipelineConstants: Readonly<Record<string, number>> = {}
): Map<string, number> {
  const values = new Map<string, number>();
  const declarations = scanWGSLDeclarations(stripWGSLComments(code)).declarations.filter(
    declaration =>
      declaration.scope === 0 && (declaration.kind === 'const' || declaration.kind === 'override')
  );
  // Module declarations may reference constants declared later in the source.
  for (let pass = 0; pass < declarations.length; pass++) {
    let progress = false;
    for (const {kind, name, expression, pipelineId} of declarations) {
      if (values.has(name)) continue;
      const resolvedId =
        pipelineId === undefined ? undefined : evaluateIntegerExpression(pipelineId, values);
      if (pipelineId !== undefined && resolvedId === undefined) continue;
      const identifier = resolvedId === undefined ? name : String(resolvedId);
      const value =
        kind === 'override' && Object.hasOwn(pipelineConstants, identifier)
          ? pipelineConstants[identifier]
          : expression === undefined
            ? undefined
            : evaluateIntegerExpression(expression, values);
      if (value !== undefined && Number.isSafeInteger(value)) {
        values.set(name, value);
        progress = true;
      }
    }
    if (!progress) break;
  }
  return values;
}

/** Integer-only constant expressions; division truncates at each operation, as in WGSL. */
export function evaluateIntegerExpression(
  expression: string,
  values: ReadonlyMap<string, number>
): number | undefined {
  try {
    const tokens: string[] = [];
    let position = 0;
    while (position < expression.length) {
      const match =
        /^\s*(0[xX][\da-fA-F]+[iu]?|\d+[iu]?|[_\p{XID_Start}][_\p{XID_Continue}]*|<<|>>|[()+\-*/%&|^~])/u.exec(
          expression.slice(position)
        );
      if (!match) {
        if (/^\s*$/.test(expression.slice(position))) break;
        return undefined;
      }
      tokens.push(match[1]);
      position += match[0].length;
    }
    let cursor = 0;
    const precedence: Record<string, number> = {
      '|': 1,
      '^': 2,
      '&': 3,
      '<<': 4,
      '>>': 4,
      '+': 5,
      '-': 5,
      '*': 6,
      '/': 6,
      '%': 6
    };
    const checked = (value: bigint): bigint => {
      if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER))
        throw new Error('integer out of range');
      return value;
    };
    function parsePrimary(): bigint {
      const token = tokens[cursor++];
      if (token === '+' || token === '-') {
        const value = parsePrimary();
        return token === '-' ? -value : value;
      }
      // Bitwise negation requires a concrete integer type; refuse rather than guess its width.
      if (token === '~') throw new Error('unresolved integer width');
      if (token === '(') {
        const value = parseBinary(1);
        if (tokens[cursor++] !== ')') throw new Error('missing parenthesis');
        return value;
      }
      if ((token === 'u32' || token === 'i32') && tokens[cursor] === '(') {
        cursor++;
        const value = parseBinary(1);
        if (
          tokens[cursor++] !== ')' ||
          (token === 'u32'
            ? value < 0n || value > 0xffffffffn
            : value < -0x80000000n || value > 0x7fffffffn)
        )
          throw new Error('unresolved conversion');
        return value;
      }
      if (/^(?:0[xX][\da-fA-F]+|\d+)[iu]?$/.test(token ?? ''))
        return checked(BigInt(token.replace(/[iu]$/, '')));
      const value = values.get(token);
      if (value === undefined || !Number.isSafeInteger(value))
        throw new Error('unresolved identifier');
      return BigInt(value);
    }
    function parseBinary(minimumPrecedence: number): bigint {
      let left = parsePrimary();
      while ((precedence[tokens[cursor]] ?? 0) >= minimumPrecedence) {
        const operator = tokens[cursor++];
        const right = parseBinary(precedence[operator] + 1);
        switch (operator) {
          case '+':
            left += right;
            break;
          case '-':
            left -= right;
            break;
          case '*':
            left *= right;
            break;
          case '/':
            left /= right;
            break;
          case '%':
            left %= right;
            break;
          case '<<':
          case '>>':
            if (left < 0n || right < 0n || right >= 32n) throw new Error('unresolved shift');
            left = operator === '<<' ? left << right : left >> right;
            if (left > 0xffffffffn) throw new Error('unresolved shift overflow');
            break;
          case '&':
            left &= right;
            break;
          case '|':
            left |= right;
            break;
          case '^':
            left ^= right;
            break;
        }
        checked(left);
      }
      return left;
    }
    const value = parseBinary(1);
    return cursor === tokens.length ? Number(checked(value)) : undefined;
  } catch {
    return undefined;
  }
}

type TypeLayout = {size: number; align: number};

function roundUp(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

/** Splits `a, b<c, d>, e` at top-level commas. */
function splitTopLevel(text: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let index = 0; index < text.length; index++) {
    const character = text[index];
    if (character === '<' || character === '(') depth++;
    else if (character === '>' || character === ')') depth--;
    else if (character === ',' && depth === 0) {
      parts.push(text.slice(start, index));
      start = index + 1;
    }
  }
  parts.push(text.slice(start));
  return parts.map(part => part.trim()).filter(Boolean);
}

/** WGSL host-shareable size and alignment of the types of one module. */
export class WGSLTypeLayouts {
  private readonly structs = new Map<
    string,
    {name: string; type: string; size?: number; alignment?: number}[]
  >();
  private readonly aliases = new Map<string, string>();
  private readonly layouts = new Map<string, TypeLayout>();
  /** Array lengths that could not be resolved, for diagnostics. */
  readonly unresolvedLengths = new Set<string>();
  /** Types whose storage layout is unknown to this estimator. */
  readonly unresolvedTypes = new Set<string>();

  constructor(
    code: string,
    private readonly constants: ReadonlyMap<string, number>,
    private readonly definitionLayouts?: WGSLTypeLayouts
  ) {
    for (const match of code.matchAll(
      /(?<![_\p{XID_Continue}])struct\s+([_\p{XID_Start}][_\p{XID_Continue}]*)\s*\{([^}]*)\}/gu
    )) {
      const memberTypes = splitTopLevel(match[2]).map(member => {
        const attribute = (name: string) => {
          const expression = new RegExp(`@${name}\\s*\\(([^)]*)\\)`).exec(member)?.[1];
          if (expression === undefined) return undefined;
          const value = evaluateIntegerExpression(expression, constants);
          if (value === undefined) this.unresolvedLengths.add(`@${name}(${expression})`);
          return value;
        };
        return {
          name:
            /([_\p{XID_Start}][_\p{XID_Continue}]*)\s*$/u.exec(
              member.slice(0, member.indexOf(':'))
            )?.[1] ?? '',
          type: member.slice(member.indexOf(':') + 1).trim(),
          size: attribute('size'),
          alignment: attribute('align')
        };
      });
      this.structs.set(match[1], memberTypes);
    }
    for (const match of code.matchAll(
      /(?<![_\p{XID_Continue}])alias\s+([_\p{XID_Start}][_\p{XID_Continue}]*)\s*=\s*([^;]+);/gu
    )) {
      this.aliases.set(match[1], match[2].trim());
    }
  }

  /** Returns a layout and records anything that prevents establishing a bound. */
  getLayout(type: string): TypeLayout {
    const key = type.replace(/\s+/g, '');
    const cached = this.layouts.get(key);
    if (cached) return cached;
    const layout = this.computeLayout(key);
    this.layouts.set(key, layout);
    return layout;
  }

  /** Distinguishes constructors from ordinary function calls in inferred initializers. */
  isConstructorType(type: string): boolean {
    const name = type.replace(/\s+/g, '').split('<')[0];
    return (
      this.aliases.has(name) ||
      this.structs.has(name) ||
      /^(?:array|vec[234][fiuh]?|mat[234]x[234][fh]?|u32|i32|f32|f16|bool)$/.test(name)
    );
  }

  /** Resolve aliases without evaluating the byte layout of a runtime storage array. */
  getCanonicalType(type: string, visited = new Set<string>()): string {
    const key = type.replace(/\s+/g, '');
    if (visited.has(key)) return key;
    const alias = this.aliases.get(key);
    if (alias) return this.getCanonicalType(alias, new Set(visited).add(key));
    return key.replace(
      /^(vec[234]|mat[234]x[234])([fiuh])$/,
      (_, shape: string, suffix: string) =>
        `${shape}<${({f: 'f32', i: 'i32', u: 'u32', h: 'f16'} as Record<string, string>)[suffix]}>`
    );
  }

  /** Opaque resource handles have binding limits, not private/workgroup byte layouts. */
  getHandleKind(
    type: string,
    visited = new Set<string>()
  ): 'storageTexture' | 'sampledTexture' | 'sampler' | undefined {
    const key = type.replace(/\s+/g, '');
    if (visited.has(key)) return undefined;
    const alias = this.aliases.get(key);
    if (alias) return this.getHandleKind(alias, new Set(visited).add(key));
    if (key === 'sampler' || key === 'sampler_comparison') return 'sampler';
    if (/^texture_storage_(?:1d|2d|2d_array|3d)<\w+,(?:read|write|read_write)>$/.test(key))
      return 'storageTexture';
    if (
      key === 'texture_external' ||
      /^texture_(?:1d|2d|2d_array|3d|cube|cube_array|multisampled_2d)<(?:f32|i32|u32)>$/.test(
        key
      ) ||
      /^texture_depth_(?:2d|2d_array|cube|cube_array|multisampled_2d)$/.test(key)
    )
      return 'sampledTexture';
    return undefined;
  }

  private computeLayout(type: string): TypeLayout {
    if (this.definitionLayouts && (this.aliases.has(type) || this.structs.has(type))) {
      return this.definitionLayouts.getLayout(type);
    }
    if (this.aliases.has(type)) return this.getLayout(this.aliases.get(type)!);
    const struct: {name: string; type: string; size?: number; alignment?: number}[] | undefined =
      this.structs.get(type) ?? getWGSLBuiltinMembers(type);
    if (struct) {
      let offset = 0;
      let align = 1;
      for (const member of struct) {
        const layout = this.getLayout(member.type);
        const memberAlign = Math.max(layout.align, member.alignment ?? 1);
        offset = roundUp(offset, memberAlign) + Math.max(layout.size, member.size ?? 0);
        align = Math.max(align, memberAlign);
      }
      return {size: roundUp(offset, align), align};
    }
    if (/^(?:f16)$/.test(type)) return {size: 2, align: 2};
    if (/^(?:bool|i32|u32|f32)$/.test(type) || /^atomic</.test(type)) return {size: 4, align: 4};
    const vector = /^vec([234])(?:<([_\p{XID_Start}][_\p{XID_Continue}]*)>|([ifuh]))$/u.exec(type);
    if (vector) {
      const componentType = this.getCanonicalType(
        vector[2] ?? ({i: 'i32', f: 'f32', u: 'u32', h: 'f16'} as Record<string, string>)[vector[3]]
      );
      if (!/^(?:bool|i32|u32|f32|f16)$/.test(componentType)) {
        this.unresolvedTypes.add(componentType);
        return {size: 4, align: 4};
      }
      const component = this.getLayout(componentType).size;
      const count = Number(vector[1]);
      const align = (count === 2 ? 2 : 4) * component;
      return {size: count * component, align};
    }
    const matrix = /^mat([234])x([234])(?:<([_\p{XID_Start}][_\p{XID_Continue}]*)>|([fh]))?$/u.exec(
      type
    );
    if (matrix) {
      const column = this.getLayout(
        `vec${matrix[2]}<${matrix[3] ?? (matrix[4] === 'h' ? 'f16' : 'f32')}>`
      );
      return {size: Number(matrix[1]) * roundUp(column.size, column.align), align: column.align};
    }
    const array = /^array<(.+)>$/.exec(type);
    if (array) {
      const [elementType, lengthExpression] = splitTopLevel(array[1]);
      const element = this.getLayout(elementType);
      const stride = roundUp(element.size, element.align);
      if (lengthExpression === undefined) {
        this.unresolvedLengths.add('runtime-sized array');
        return {size: stride, align: element.align};
      }
      const length = evaluateIntegerExpression(lengthExpression, this.constants);
      if (length === undefined || !Number.isSafeInteger(length) || length < 1) {
        this.unresolvedLengths.add(lengthExpression);
        return {size: stride, align: element.align};
      }
      return {size: stride * length, align: element.align};
    }
    this.unresolvedTypes.add(type);
    return {size: 4, align: 4};
  }

  /** Whether a type is an array, including aliases and structs containing arrays. */
  containsArray(type: string): boolean {
    const key = type.replace(/\s+/g, '');
    if (/^array</.test(key)) return true;
    const alias = this.aliases.get(key);
    if (alias) return this.containsArray(alias);
    return this.structs.get(key)?.some(member => this.containsArray(member.type)) ?? false;
  }

  /** Arrays and structs require aggregate storage even without an explicit local type. */
  isAggregateType(type: string): boolean {
    const key = this.getCanonicalType(type);
    return key.startsWith('array<') || this.structs.has(key) || Boolean(getWGSLBuiltinMembers(key));
  }

  /** Type of an indexed element or named member, without evaluating the selected row. */
  getSelectedType(
    type: string,
    member?: string
  ): {type: string; moduleScoped?: boolean} | undefined {
    const key = type.replace(/\s+/g, '');
    const alias = this.aliases.get(key);
    if (alias) {
      const result = this.getSelectedType(alias, member);
      return result && {...result, moduleScoped: true};
    }
    if (member) {
      const field = (this.structs.get(key) ?? getWGSLBuiltinMembers(key))?.find(
        field => field.name === member
      );
      if (field) return {type: field.type, moduleScoped: true};
    }
    const vector = /^vec([234])(?:<([^>]+)>|([fiuh]))$/.exec(key);
    if (vector) {
      const component =
        vector[2] ??
        ({f: 'f32', i: 'i32', u: 'u32', h: 'f16'} as Record<string, string>)[vector[3]];
      if (!member || /^[xyzwrgba]$/.test(member)) return {type: component};
      if (/^[xyzwrgba]{2,4}$/.test(member)) return {type: `vec${member.length}<${component}>`};
    }
    if (member) return undefined;
    const array = /^array<(.+)>$/.exec(key);
    if (array) return {type: splitTopLevel(array[1])[0]};
    const matrix = /^mat([234])x([234])(?:<([^>]+)>|([fh]))?$/.exec(key);
    if (matrix)
      return {type: `vec${matrix[2]}<${matrix[3] ?? (matrix[4] === 'h' ? 'f16' : 'f32')}>`};
    return undefined;
  }
}

/** Function return types also determine the storage of an inferred local initializer. */
function getFunctionTypes(code: string) {
  const types = new Map<string, {returnType?: string; parameters: Map<string, string>}>();
  for (const match of code.matchAll(
    /(?<![_\p{XID_Continue}])fn\s+([_\p{XID_Start}][_\p{XID_Continue}]*)\s*\(/gu
  )) {
    let depth = 1;
    const start = match.index + match[0].length;
    let position = start;
    for (; position < code.length && depth > 0; position++) {
      if (code[position] === '(') depth++;
      else if (code[position] === ')') depth--;
    }
    const result = /^\s*->\s*([^{}]+)\{/.exec(code.slice(position));
    const parameters = new Map<string, string>();
    for (const parameter of splitTopLevel(code.slice(start, position - 1))) {
      const declaration = /([_\p{XID_Start}][_\p{XID_Continue}]*)\s*:\s*([\s\S]+)$/u.exec(
        parameter
      );
      if (declaration) parameters.set(declaration[1], declaration[2].trim());
    }
    types.set(match[1], {
      returnType: result?.[1].replace(/@\w+(?:\([^)]*\))?\s*/g, '').trim(),
      parameters
    });
  }
  return types;
}

/** Byte totals of one WGSL module against the core limits. */
export type WGSLModuleUsage = {
  /** `var<private>` bytes. */
  privateBytes: number;
  /** Largest function-address-space bytes of any one function, and that function. */
  functionBytes: number;
  functionName: string;
  /** Sum over every function, conservative even when locals survive nested calls. */
  totalFunctionBytes: number;
  /** Local arrays and structs containing arrays, including inferred var/let copies and fragments. */
  localArrayBytes: number;
  /** Local arrays and structs, including structs whose member size is explicitly enlarged. */
  localAggregateBytes: number;
  /** Declared `var<workgroup>` payload bytes, before WebGPU's per-variable rounding. */
  workgroupBytes: number;
  /** Workgroup storage limit usage: each variable's store size rounded up to 16 bytes. */
  workgroupStorageBytes: number;
  /** `var<storage, ...>` bindings. */
  storageBuffers: number;
  /** `var<uniform>` bindings. */
  uniformBuffers: number;
  /** Storage-texture bindings. */
  storageTextures: number;
  /** Sampled-texture bindings. */
  sampledTextures: number;
  /** Sampler bindings. */
  samplers: number;
  /** `@workgroup_size` of each entry point, resolved. */
  workgroupSizes: number[][];
  /** Array lengths the accounting could not resolve (counted as one element). */
  unresolvedLengths: string[];
  /** Unrecognized memory types; totals are not bounds when this is nonempty. */
  unresolvedTypes: string[];
};

/** Accounts the private, function, workgroup memory and the bindings of one module. */
export function getWGSLModuleUsage(
  code: string,
  pipelineConstants: Readonly<Record<string, number>> = {}
): WGSLModuleUsage {
  code = stripWGSLComments(code);
  const constants = getWGSLConstantValues(code, pipelineConstants);
  const layouts = new WGSLTypeLayouts(code, constants);
  const {declarations, scopes} = scanWGSLDeclarations(code);
  const functionTypes = getFunctionTypes(code);
  type ResolvedType = {type?: string; context?: WGSLDeclaration; unresolved?: string};
  const resolvedTypes = new Map<WGSLDeclaration, ResolvedType>();
  const resolveType = (
    declaration: WGSLDeclaration,
    visited = new Set<WGSLDeclaration>()
  ): ResolvedType => {
    if (declaration.type) return {type: declaration.type, context: declaration};
    const cached = resolvedTypes.get(declaration);
    if (cached) return cached;
    if (visited.has(declaration)) return {unresolved: `cyclic initializer of ${declaration.name}`};
    const resolved = inferWGSLExpressionType<WGSLDeclaration>(declaration.expression ?? '', {
      context: declaration,
      canonical: type => layouts.getCanonicalType(type),
      member: (type, member) => layouts.getSelectedType(type, member),
      constructor: type => layouts.isConstructorType(type),
      returnType: name => functionTypes.get(name)?.returnType,
      identifier: identifier => {
        for (let scope = declaration.scope; scope >= 0; scope = scopes[scope].parent) {
          // Parameters shadow module names; locals shadow parameters in their block.
          if (scope === 0 && declaration.functionName) {
            const parameter = functionTypes
              .get(declaration.functionName)
              ?.parameters.get(identifier);
            if (parameter) return {type: parameter};
          }
          const target = declarations
            .filter(
              candidate =>
                candidate.name === identifier &&
                candidate.scope === scope &&
                candidate !== declaration &&
                (scope === 0 || candidate.position < declaration.position)
            )
            .at(-1);
          if (target) {
            const result = resolveType(target, new Set(visited).add(declaration));
            return result.type ? {type: result.type, context: result.context} : undefined;
          }
        }
        return undefined;
      }
    });
    const result = resolved ?? {
      unresolved: `initializer type of ${declaration.name}: ${declaration.expression ?? '<missing>'}`
    };
    resolvedTypes.set(declaration, result);
    return result;
  };
  const unresolvedLengths = new Set<string>();
  const unresolvedTypes = new Set<string>();
  const functionTotals = new Map<string, number>();
  let privateBytes = 0;
  let workgroupBytes = 0;
  let workgroupStorageBytes = 0;
  let localArrayBytes = 0;
  let localAggregateBytes = 0;
  let storageTextures = 0;
  let sampledTextures = 0;
  let samplers = 0;
  for (const declaration of declarations) {
    if (declaration.kind !== 'var' && declaration.kind !== 'let') continue;
    // Module texture/sampler variables implicitly use WGSL's handle address space. Ordinary
    // module data variables still default to private, including unknown data types we must reject.
    const handleKind =
      declaration.scope === 0 && !declaration.addressSpace && declaration.type
        ? layouts.getHandleKind(declaration.type)
        : undefined;
    if (handleKind) {
      if (handleKind === 'storageTexture') storageTextures++;
      else if (handleKind === 'sampledTexture') sampledTextures++;
      else samplers++;
      continue;
    }
    const addressSpace =
      declaration.addressSpace ?? (declaration.functionName ? 'function' : 'private');
    if (!['private', 'workgroup', 'function'].includes(addressSpace)) continue;
    const resolvedType = resolveType(declaration);
    const typeContext = resolvedType.context;
    if (resolvedType.unresolved) unresolvedTypes.add(resolvedType.unresolved);
    // Immutable arrays/structs are value copies too. Scalar/vector let bindings remain registers.
    if (
      declaration.kind === 'let' &&
      resolvedType.type &&
      !layouts.isAggregateType(resolvedType.type)
    ) {
      // Register bindings still need a recognized type before they can certify a layout.
      // Pointers and opaque handles have no data layout of their own; their pointee/value
      // copies are resolved separately when used by another declaration.
      if (
        !layouts.getCanonicalType(resolvedType.type).startsWith('ptr<') &&
        !layouts.getHandleKind(resolvedType.type)
      ) {
        layouts.getLayout(resolvedType.type);
      }
      continue;
    }
    const scopedConstants = new Map(constants);
    const ancestors: number[] = [];
    for (let scope = typeContext?.scope ?? 0; scope > 0; scope = scopes[scope].parent)
      ancestors.unshift(scope);
    for (const scope of ancestors) {
      for (const local of declarations) {
        if (
          local.scope !== scope ||
          local.kind !== 'const' ||
          local.position >= (typeContext?.position ?? declaration.position)
        )
          continue;
        // A local declaration shadows the outer name even when its value is unresolved.
        const value =
          local.expression === undefined
            ? undefined
            : evaluateIntegerExpression(local.expression, scopedConstants);
        scopedConstants.delete(local.name);
        if (value !== undefined) scopedConstants.set(local.name, value);
      }
    }
    const variableLayouts = !typeContext?.scope
      ? layouts
      : new WGSLTypeLayouts(code, scopedConstants, layouts);
    const type = resolvedType.type;
    const effectiveLayouts = variableLayouts;
    let bytes = 4;
    if (type) {
      bytes = effectiveLayouts.getLayout(type).size;
    } else if (
      addressSpace !== 'function' &&
      !/^(?:true|false|[+\-]?(?:\d|\.))/.test(declaration.expression ?? '')
    ) {
      unresolvedTypes.add(`inferred type of ${declaration.name}`);
    }
    if (addressSpace === 'private') privateBytes += bytes;
    else if (addressSpace === 'workgroup') {
      workgroupBytes += bytes;
      workgroupStorageBytes += roundUp(bytes, 16);
    } else {
      const name = declaration.functionName ?? '<function declaration fragment>';
      functionTotals.set(name, (functionTotals.get(name) ?? 0) + bytes);
      if (type && effectiveLayouts.containsArray(type)) localArrayBytes += bytes;
      if (type && effectiveLayouts.isAggregateType(type)) localAggregateBytes += bytes;
    }
    for (const unresolved of variableLayouts.unresolvedLengths) unresolvedLengths.add(unresolved);
    for (const unresolved of variableLayouts.unresolvedTypes) unresolvedTypes.add(unresolved);
  }
  let functionBytes = 0;
  let totalFunctionBytes = 0;
  let functionName = '';
  for (const [name, total] of functionTotals) {
    totalFunctionBytes += total;
    if (total > functionBytes) {
      functionBytes = total;
      functionName = name;
    }
  }
  const workgroupSizes: number[][] = [];
  for (const match of code.matchAll(/@workgroup_size\s*\(([^)]*)\)/g)) {
    workgroupSizes.push(
      splitTopLevel(match[1]).map(part => evaluateIntegerExpression(part, constants) ?? Number.NaN)
    );
  }
  const count = (pattern: RegExp): number => (code.match(pattern) ?? []).length;
  return {
    privateBytes,
    functionBytes,
    functionName,
    totalFunctionBytes,
    localArrayBytes,
    localAggregateBytes,
    workgroupBytes,
    workgroupStorageBytes,
    storageBuffers: countWGSLStorageDeclarations(code),
    uniformBuffers: count(/var\s*<\s*uniform\s*>/g),
    storageTextures,
    sampledTextures,
    samplers,
    workgroupSizes,
    unresolvedLengths: [...new Set([...layouts.unresolvedLengths, ...unresolvedLengths])],
    unresolvedTypes: [...new Set([...layouts.unresolvedTypes, ...unresolvedTypes])]
  };
}
