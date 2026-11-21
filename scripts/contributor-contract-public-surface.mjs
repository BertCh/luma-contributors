// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import typeScript from 'typescript';

/** Finds internal declarations exposed by a public props type or class signature, without typechecking. */
export function findInternalPublicMembers(declaration, resolveDeclaration, publicName) {
  const findings = [];
  const active = new Set();
  const allowAll = () => true;
  const isInternal = node =>
    typeScript.getJSDocTags(node).some(tag => tag.tagName.text === 'internal');
  const isPrivate = node =>
    node.modifiers?.some(
      modifier =>
        modifier.kind === typeScript.SyntaxKind.PrivateKeyword ||
        modifier.kind === typeScript.SyntaxKind.ProtectedKeyword
    ) ||
    (node.name && typeScript.isPrivateIdentifier(node.name));
  const report = (node, name) => findings.push({node, name});
  const bindParameters = (target, arguments_, bindings) => {
    const result = new Map(bindings);
    for (const [index, parameter] of (target.typeParameters ?? []).entries()) {
      const argument = arguments_?.[index] ?? parameter.default;
      if (argument)
        result.set(parameter, {
          node: argument,
          bindings: arguments_?.[index] ? bindings : result
        });
    }
    return result;
  };
  const getKeys = (node, bindings, visiting = new Set()) => {
    if (!node || visiting.has(node)) return new Set();
    visiting.add(node);
    if (typeScript.isLiteralTypeNode(node) && typeScript.isStringLiteral(node.literal))
      return new Set([node.literal.text]);
    if (typeScript.isUnionTypeNode(node))
      return new Set(node.types.flatMap(type => [...getKeys(type, bindings, visiting)]));
    if (typeScript.isTypeReferenceNode(node)) {
      const target = resolveDeclaration(node.typeName);
      const argument = bindings.get(target);
      if (argument) return getKeys(argument.node, argument.bindings, visiting);
      if (target && typeScript.isTypeAliasDeclaration(target))
        return getKeys(target.type, bindings, visiting);
    }
    return new Set();
  };
  // Select the indexed property before walking its value. Walking the whole object would
  // falsely report internal members on branches that the public type never exposes.
  const selectMembers = (node, bindings, keys, visiting = new Set(), instanceType = false) => {
    if (!node || visiting.has(node)) return [];
    const next = new Set([...visiting, node]);
    if (typeScript.isTypeReferenceNode(node) || typeScript.isExpressionWithTypeArguments(node)) {
      const reference = typeScript.isTypeReferenceNode(node) ? node.typeName : node.expression;
      const target = resolveDeclaration(reference);
      const argument = bindings.get(target);
      if (argument)
        return selectMembers(argument.node, argument.bindings, keys, next, instanceType);
      if (target)
        return selectMembers(
          target,
          bindParameters(target, node.typeArguments, bindings),
          keys,
          next,
          instanceType ||
            (typeScript.isTypeReferenceNode(node) && typeScript.isClassDeclaration(target))
        );
      if (typeScript.isIdentifier(reference)) {
        const [input, selection] = node.typeArguments ?? [];
        if (reference.text === 'Pick' || reference.text === 'Omit') {
          const selected = getKeys(selection, bindings);
          return selectMembers(
            input,
            bindings,
            new Set(
              [...keys].filter(key =>
                reference.text === 'Pick' ? selected.has(key) : !selected.has(key)
              )
            ),
            next
          );
        }
        if (['Partial', 'Required', 'Readonly', 'NonNullable'].includes(reference.text))
          return selectMembers(input, bindings, keys, next, instanceType);
        if (reference.text === 'Record' && [...keys].some(key => getKeys(input, bindings).has(key)))
          return [{valueType: selection, bindings}];
      }
    } else if (
      typeScript.isTypeAliasDeclaration(node) ||
      typeScript.isParenthesizedTypeNode(node)
    ) {
      return selectMembers(node.type, bindings, keys, next, instanceType);
    } else if (typeScript.isUnionTypeNode(node) || typeScript.isIntersectionTypeNode(node)) {
      return node.types.flatMap(type => selectMembers(type, bindings, keys, next, instanceType));
    } else if (typeScript.isIndexedAccessTypeNode(node)) {
      return selectMembers(
        node.objectType,
        bindings,
        getKeys(node.indexType, bindings),
        next
      ).flatMap(selected =>
        selectMembers(
          selected.valueType ?? selected.member.type,
          selected.bindings,
          keys,
          next,
          instanceType
        )
      );
    } else if (
      typeScript.isInterfaceDeclaration(node) ||
      typeScript.isClassDeclaration(node) ||
      typeScript.isTypeLiteralNode(node)
    ) {
      const members = [...node.members]
        .filter(
          member =>
            !isPrivate(member) &&
            !(
              instanceType &&
              member.modifiers?.some(
                modifier => modifier.kind === typeScript.SyntaxKind.StaticKeyword
              )
            ) &&
            member.name &&
            keys.has(member.name.getText().replace(/^['"]|['"]$/g, ''))
        )
        .map(member => ({member, bindings}));
      for (const clause of node.heritageClauses ?? []) {
        if (clause.token === typeScript.SyntaxKind.ExtendsKeyword)
          for (const type of clause.types)
            members.push(...selectMembers(type, bindings, keys, next, instanceType));
      }
      return members;
    }
    return [];
  };
  const visitSignature = (signature, bindings, name) => {
    for (const parameter of signature.parameters) {
      if (isPrivate(parameter)) continue;
      const parameterName = `${name}.${parameter.name.getText()}`;
      if (isInternal(parameter)) report(parameter, parameterName);
      visit(parameter.type, bindings, parameterName);
    }
    visit(signature.type, bindings, `${name}.return`);
  };
  const visitMembers = (members, bindings, name, accepts, instanceType) => {
    for (const member of members) {
      if (isPrivate(member)) continue;
      if (
        instanceType &&
        (typeScript.isConstructorDeclaration(member) ||
          member.modifiers?.some(modifier => modifier.kind === typeScript.SyntaxKind.StaticKeyword))
      )
        continue;
      const memberName = typeScript.isConstructorDeclaration(member)
        ? 'constructor'
        : member.name?.getText().replace(/^['"]|['"]$/g, '');
      if (memberName && !accepts(memberName)) continue;
      // Only overload declarations are visible when an implementation signature has overloads.
      if (
        member.body &&
        members.some(
          other =>
            other !== member &&
            !other.body &&
            other.kind === member.kind &&
            (typeScript.isConstructorDeclaration(member) ||
              other.name?.getText() === member.name?.getText())
        )
      )
        continue;
      const memberPath = memberName ? `${name}.${memberName}` : name;
      if (isInternal(member)) report(member, memberPath);
      if (member.parameters) visitSignature(member, bindings, memberPath);
      visit(member.type, bindings, memberPath);
      if (
        member.initializer &&
        (typeScript.isArrowFunction(member.initializer) ||
          typeScript.isFunctionExpression(member.initializer))
      ) {
        visitSignature(member.initializer, bindings, memberPath);
      }
    }
  };
  const visit = (node, bindings, name, accepts = allowAll, instanceType = false) => {
    if (!node || active.has(node)) return;
    active.add(node);
    try {
      if (typeScript.isTypeReferenceNode(node) || typeScript.isExpressionWithTypeArguments(node)) {
        const reference = typeScript.isTypeReferenceNode(node) ? node.typeName : node.expression;
        const target = resolveDeclaration(reference);
        const argument = bindings.get(target);
        if (argument) return visit(argument.node, argument.bindings, name, accepts);
        if (
          target &&
          (typeScript.isTypeAliasDeclaration(target) ||
            typeScript.isInterfaceDeclaration(target) ||
            typeScript.isClassDeclaration(target))
        ) {
          return visit(
            target,
            bindParameters(target, node.typeArguments, bindings),
            name,
            accepts,
            instanceType ||
              (typeScript.isTypeReferenceNode(node) && typeScript.isClassDeclaration(target))
          );
        }
        if (typeScript.isIdentifier(reference)) {
          const [input, keys] = node.typeArguments ?? [];
          if (reference.text === 'Omit' || reference.text === 'Pick') {
            const selected = getKeys(keys, bindings);
            return visit(
              input,
              bindings,
              name,
              key =>
                accepts(key) && (reference.text === 'Pick' ? selected.has(key) : !selected.has(key))
            );
          }
          if (['Partial', 'Required', 'Readonly', 'NonNullable'].includes(reference.text))
            return visit(input, bindings, name, accepts);
          if (reference.text === 'Array' || reference.text === 'ReadonlyArray')
            return visit(input, bindings, `${name}[]`);
          if (reference.text === 'Record')
            return visit(node.typeArguments?.[1], bindings, `${name}[]`);
        }
      } else if (
        typeScript.isTypeAliasDeclaration(node) ||
        typeScript.isParenthesizedTypeNode(node) ||
        typeScript.isOptionalTypeNode(node)
      ) {
        visit(node.type, bindings, name, accepts);
      } else if (typeScript.isUnionTypeNode(node) || typeScript.isIntersectionTypeNode(node)) {
        for (const type of node.types) visit(type, bindings, name, accepts);
      } else if (
        typeScript.isClassDeclaration(node) ||
        typeScript.isInterfaceDeclaration(node) ||
        typeScript.isTypeLiteralNode(node)
      ) {
        visitMembers(node.members, bindings, name, accepts, instanceType);
        for (const clause of node.heritageClauses ?? []) {
          if (clause.token === typeScript.SyntaxKind.ExtendsKeyword)
            for (const type of clause.types) visit(type, bindings, name, accepts, instanceType);
        }
      } else if (typeScript.isIndexedAccessTypeNode(node)) {
        for (const selected of selectMembers(
          node.objectType,
          bindings,
          getKeys(node.indexType, bindings)
        )) {
          if (selected.valueType) {
            visit(selected.valueType, selected.bindings, name, accepts);
            continue;
          }
          if (isInternal(selected.member))
            report(selected.member, `${name}.${selected.member.name.getText()}`);
          if (selected.member.parameters) visitSignature(selected.member, selected.bindings, name);
          visit(selected.member.type, selected.bindings, name, accepts);
        }
      } else if (typeScript.isArrayTypeNode(node)) {
        visit(node.elementType, bindings, `${name}[]`);
      } else if (typeScript.isTupleTypeNode(node)) {
        for (const element of node.elements) visit(element, bindings, `${name}[]`);
      } else if (typeScript.isNamedTupleMember(node)) {
        visit(node.type, bindings, name);
      } else if (typeScript.isFunctionTypeNode(node) || typeScript.isConstructorTypeNode(node)) {
        visitSignature(node, bindings, name);
      }
    } finally {
      active.delete(node);
    }
  };
  visit(declaration, bindParameters(declaration, [], new Map()), publicName);
  return findings;
}
