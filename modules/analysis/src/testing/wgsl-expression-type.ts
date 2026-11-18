// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** A type carries the declaration scope in which its array length was written. */
export type WGSLExpressionType<Context> = {type: string; context?: Context};

type TypeEnvironment<Context> = {
  context?: Context;
  identifier: (name: string) => WGSLExpressionType<Context> | undefined;
  canonical: (type: string) => string;
  member: (type: string, member?: string) => {type: string; moduleScoped?: boolean} | undefined;
  constructor: (type: string) => boolean;
  returnType: (name: string) => string | undefined;
};

/** The anonymous structs returned by WGSL builtins also have ordinary data layouts. */
export function getWGSLBuiltinMembers(type: string): {name: string; type: string}[] | undefined {
  const result = /^__(frexp|modf|atomic_compare_exchange)_result<(.+)>$/.exec(type);
  if (!result) return undefined;
  const [, kind, value] = result;
  if (kind === 'atomic_compare_exchange')
    return [
      {name: 'old_value', type: value},
      {name: 'exchanged', type: 'bool'}
    ];
  const vector = /^vec([234])/.exec(value);
  return [
    {name: 'fract', type: value},
    {
      name: kind === 'frexp' ? 'exp' : 'whole',
      type: kind === 'frexp' ? (vector ? `vec${vector[1]}<i32>` : 'i32') : value
    }
  ];
}

/**
 * Reads expression types, not values. It deliberately does not validate a WGSL program or
 * evaluate array indices. A call's declared return type is sufficient even when an argument's
 * value is dynamic. Unknown expression types stay unknown, never an assumed scalar bound.
 */
export function inferWGSLExpressionType<Context>(
  expression: string,
  environment: TypeEnvironment<Context>
): WGSLExpressionType<Context> | undefined {
  const tokens: string[] = [];
  let offset = 0;
  while (offset < expression.length) {
    const match =
      /^\s*(0[xX](?:[\da-fA-F]+(?:\.[\da-fA-F]*)?|\.[\da-fA-F]+)(?:[pP][+\-]?\d+)?[iuh]?|(?:\d+\.?\d*|\.\d+)(?:[eE][+\-]?\d+)?[iufh]?|[_\p{XID_Start}][_\p{XID_Continue}]*|&&|\|\||<<|>>|<=|>=|==|!=|[()[\].,<>+\-*/%&|^!~])/u.exec(
        expression.slice(offset)
      );
    if (!match) {
      if (/^\s*$/.test(expression.slice(offset))) break;
      return undefined;
    }
    tokens.push(match[1]);
    offset += match[0].length;
  }
  let cursor = 0;
  const canonical = (value: WGSLExpressionType<Context> | undefined) =>
    value && environment.canonical(value.type);
  const component = (type: string): string => {
    const vector = /^vec[234](?:<([^>]+)>|([fiuh]))$/.exec(type);
    const matrix = /^mat[234]x[234](?:<([^>]+)>|([fh]))$/.exec(type);
    const match = vector ?? matrix;
    return match
      ? (match[1] ?? ({f: 'f32', i: 'i32', u: 'u32', h: 'f16'} as Record<string, string>)[match[2]])
      : type;
  };
  const vectorWidth = (type: string) => /^vec([234])/.exec(type)?.[1];
  const numeric = (type: string) =>
    /^(?:bool|[uif]32|f16|vec[234](?:<[^>]+>|[fiuh])|mat[234]x[234](?:<[^>]+>|[fh]))$/.test(type);
  const pointee = (value: WGSLExpressionType<Context> | undefined) => {
    const pointer = value && /^ptr<\w+,([\s\S]+)>$/.exec(environment.canonical(value.type));
    if (!pointer) return undefined;
    // A pointer's optional access mode follows the type at the outermost comma.
    const type = pointer[1].replace(/,(?:read|write|read_write)$/, '');
    return {...value, type};
  };
  const mergeNumeric = (
    left: WGSLExpressionType<Context> | undefined,
    right: WGSLExpressionType<Context> | undefined
  ) => {
    const first = canonical(left);
    const second = canonical(right);
    if (!first || !second || !numeric(first) || !numeric(second)) return undefined;
    if (/^(?:vec|mat)/.test(first)) return left;
    if (/^(?:vec|mat)/.test(second)) return right;
    return component(first) === 'f32' ? left : right;
  };
  const boolean = (value: WGSLExpressionType<Context> | undefined) => {
    const type = canonical(value);
    if (!type || !numeric(type)) return undefined;
    const width = vectorWidth(type);
    return {type: width ? `vec${width}<bool>` : 'bool'};
  };
  const precedence: Record<string, number> = {
    '||': 1,
    '&&': 2,
    '|': 3,
    '^': 4,
    '&': 5,
    '==': 6,
    '!=': 6,
    '<': 7,
    '<=': 7,
    '>': 7,
    '>=': 7,
    '<<': 8,
    '>>': 8,
    '+': 9,
    '-': 9,
    '*': 10,
    '/': 10,
    '%': 10
  };
  function binary(
    operator: string,
    left: WGSLExpressionType<Context> | undefined,
    right: WGSLExpressionType<Context> | undefined
  ) {
    const merged = mergeNumeric(left, right);
    if (/^(?:==|!=|<|<=|>|>=|&&|\|\|)$/.test(operator)) return boolean(merged);
    if (operator === '*') {
      const first = canonical(left);
      const second = canonical(right);
      const firstMatrix = first && /^mat([234])x([234])/.exec(first);
      const secondMatrix = second && /^mat([234])x([234])/.exec(second);
      if (firstMatrix && secondMatrix)
        return {type: `mat${secondMatrix[1]}x${firstMatrix[2]}<${component(first!)}>`};
      if (firstMatrix && second && vectorWidth(second))
        return {type: `vec${firstMatrix[2]}<${component(first!)}>`};
      if (secondMatrix && first && vectorWidth(first))
        return {type: `vec${secondMatrix[1]}<${component(second!)}>`};
    }
    return merged;
  }
  function call(
    name: string,
    template: string | undefined,
    arguments_: (WGSLExpressionType<Context> | undefined)[]
  ): WGSLExpressionType<Context> | undefined {
    const declared = environment.returnType(name);
    if (declared) return {type: declared};
    const first = arguments_[0];
    const firstType = canonical(first);
    if (name === 'bitcast' && template) return {type: template};
    if (name === 'array' && !template) {
      if (!first) return undefined;
      const element = arguments_.every(argument => canonical(argument) === firstType)
        ? first
        : arguments_.reduce(mergeNumeric);
      return element && {...element, type: `array<${element.type},${arguments_.length}>`};
    }
    if (/^(?:vec[234]|mat[234]x[234])$/.test(name) && !template)
      return firstType && numeric(firstType)
        ? {type: `${name}<${component(firstType)}>`}
        : undefined;
    const constructed = template ? `${name}<${template}>` : name;
    if (environment.constructor(constructed))
      return {type: constructed, context: environment.context};
    if (name === 'arrayLength') return {type: 'u32'};
    if (/^textureNum(?:Layers|Levels|Samples)$/.test(name)) return {type: 'u32'};
    if (name === 'textureDimensions' && firstType) {
      const dimension = /(?:1d|2d|3d|cube)/.exec(firstType)?.[0];
      return dimension
        ? {type: dimension === '1d' ? 'u32' : dimension === '3d' ? 'vec3<u32>' : 'vec2<u32>'}
        : undefined;
    }
    if (name === 'all' || name === 'any') return {type: 'bool'};
    if (name === 'workgroupUniformLoad') return pointee(first);
    if (/^atomic(?:Load|Add|Sub|Max|Min|And|Or|Xor|Exchange|CompareExchangeWeak)$/.test(name)) {
      const atomic = pointee(first);
      const value = atomic && /^atomic<(.+)>$/.exec(environment.canonical(atomic.type))?.[1];
      return value
        ? {
            type:
              name === 'atomicCompareExchangeWeak'
                ? `__atomic_compare_exchange_result<${value}>`
                : value
          }
        : undefined;
    }
    if ((name === 'frexp' || name === 'modf') && firstType)
      return {type: `__${name}_result<${firstType}>`};
    if (/^(?:dot|length|distance|determinant)$/.test(name))
      return firstType && numeric(firstType) ? {type: component(firstType)} : undefined;
    if (name === 'transpose' && firstType) {
      const matrix = /^mat([234])x([234])/.exec(firstType);
      return matrix ? {type: `mat${matrix[2]}x${matrix[1]}<${component(firstType)}>`} : undefined;
    }
    if (/^(?:min|max|clamp|select|mix|step|smoothstep|pow|atan2|ldexp|fma)$/.test(name))
      return arguments_.slice(0, name === 'select' ? 2 : arguments_.length).reduce(mergeNumeric);
    if (
      /^(?:abs|acos|acosh|asin|asinh|atan|atanh|ceil|cos|cosh|degrees|exp|exp2|floor|fract|inverseSqrt|log|log2|normalize|quantizeToF16|radians|round|saturate|sign|sin|sinh|sqrt|tan|tanh|trunc|countLeadingZeros|countOneBits|countTrailingZeros|firstLeadingBit|firstTrailingBit|reverseBits|extractBits|insertBits|cross|reflect|refract|faceForward)$/.test(
        name
      )
    )
      return firstType && numeric(firstType) ? first : undefined;
    if (/^(?:isFinite|isInf|isNan|isNormal)$/.test(name)) return boolean(first);
    if (/^pack(?:4x8|2x16)/.test(name) || name === 'dot4U8Packed') return {type: 'u32'};
    if (name === 'dot4I8Packed') return {type: 'i32'};
    if (/^unpack4x8/.test(name)) return {type: 'vec4<f32>'};
    if (/^unpack2x16/.test(name)) return {type: 'vec2<f32>'};
    if (/^texture(?:Load|Sample|Gather)/.test(name) && firstType) {
      if (/^textureSampleCompare/.test(name)) return {type: 'f32'};
      if (/^texture_depth/.test(firstType))
        return {type: name.startsWith('textureGather') ? 'vec4<f32>' : 'f32'};
      const sample = /<(f32|u32|i32)>$/.exec(firstType)?.[1];
      const storage = /<([^,]+),/.exec(firstType)?.[1];
      const element =
        sample ??
        (storage?.endsWith('uint')
          ? 'u32'
          : storage?.endsWith('sint')
            ? 'i32'
            : storage
              ? 'f32'
              : undefined);
      return element ? {type: `vec4<${element}>`} : undefined;
    }
    return undefined;
  }
  function primary(): WGSLExpressionType<Context> | undefined {
    const token = tokens[cursor++];
    if (!token) throw new Error('missing operand');
    if (/^[+\-!~*&]$/.test(token)) {
      const operand = primary();
      if (token === '*') return pointee(operand);
      if (token === '&') return operand && {...operand, type: `ptr<function,${operand.type}>`};
      if (token === '!') return boolean(operand);
      return operand && numeric(environment.canonical(operand.type)) ? operand : undefined;
    }
    let result: WGSLExpressionType<Context> | undefined;
    if (token === '(') {
      result = parseBinary(1);
      if (tokens[cursor++] !== ')') throw new Error('missing parenthesis');
    } else if (/^(?:\d|\.\d)/.test(token)) {
      const hexadecimal = /^0x/i.test(token);
      result = {
        type: /u$/.test(token)
          ? 'u32'
          : /i$/.test(token)
            ? 'i32'
            : /h$/.test(token)
              ? 'f16'
              : (hexadecimal ? /[.pP]/.test(token) : /[.eEf]/.test(token))
                ? 'f32'
                : 'i32'
      };
    } else if (token === 'true' || token === 'false') result = {type: 'bool'};
    else if (/^[_\p{XID_Start}][_\p{XID_Continue}]*$/u.test(token)) {
      let template: string | undefined;
      if (tokens[cursor] === '<') {
        let depth = 1;
        let end = cursor + 1;
        for (; end < tokens.length && depth; end++) {
          if (tokens[end] === '<') depth++;
          else if (tokens[end] === '>') depth--;
          else if (tokens[end] === '>>') depth -= 2;
        }
        if (depth === 0 && tokens[end] === '(') {
          template = tokens
            .slice(cursor + 1, end)
            .join('')
            .slice(0, -1);
          cursor = end;
        }
      }
      if (tokens[cursor] === '(') {
        cursor++;
        const arguments_: (WGSLExpressionType<Context> | undefined)[] = [];
        while (tokens[cursor] !== ')') {
          arguments_.push(parseBinary(1));
          if (tokens[cursor] !== ',') break;
          cursor++;
          if (tokens[cursor] === ')') break;
        }
        if (tokens[cursor++] !== ')') throw new Error('missing call parenthesis');
        result = call(token, template, arguments_);
      } else result = environment.identifier(token);
    } else throw new Error('unknown expression');
    while (tokens[cursor] === '.' || tokens[cursor] === '[') {
      const selector = tokens[cursor++];
      let member: string | undefined;
      if (selector === '.') member = tokens[cursor++];
      else {
        parseBinary(1);
        if (tokens[cursor++] !== ']') throw new Error('missing index bracket');
      }
      const selected = result && environment.member(result.type, member);
      result = selected
        ? {type: selected.type, context: selected.moduleScoped ? undefined : result?.context}
        : undefined;
    }
    return result;
  }
  function parseBinary(minimum: number): WGSLExpressionType<Context> | undefined {
    let left = primary();
    while ((precedence[tokens[cursor]] ?? 0) >= minimum) {
      const operator = tokens[cursor++];
      left = binary(operator, left, parseBinary(precedence[operator] + 1));
    }
    return left;
  }
  try {
    const result = parseBinary(1);
    return cursor === tokens.length ? result : undefined;
  } catch {
    return undefined;
  }
}
