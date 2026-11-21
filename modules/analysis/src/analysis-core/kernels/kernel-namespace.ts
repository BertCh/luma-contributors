// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {validateWGSLIdentifier} from './wgsl-identifiers';
import {validateRequirement} from './gpu-contributor-utils';

/** Validate generated module declarations before allocating resources. @internal */
export function validateKernelNamespace(
  id: string,
  bindings: readonly {name: string}[],
  textures: readonly {name: string}[],
  reservedNames: readonly string[]
): void {
  const generatedNames = [
    ...reservedNames,
    'workgroupId',
    'localInvocationIndex',
    'workgroupIndex',
    'index'
  ];
  const names = new Map(generatedNames.map(name => [name, `generated ${name}`]));
  function claim(name: string, owner: string, role: string): void {
    validateWGSLIdentifier(id, role, name);
    const previous = names.get(name);
    validateRequirement(
      id,
      previous === undefined,
      `kernel names ${previous} and ${owner} collide at ${name}`
    );
    names.set(name, owner);
  }
  for (const binding of bindings) {
    claim(binding.name, binding.name, 'binding name');
    claim(`${binding.name}Offset`, binding.name, 'binding offset');
  }
  for (const texture of textures) claim(texture.name, texture.name, 'texture name');
}
