// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
// Read build/watch inputs without creating a TypeScript program or typechecking dependencies.
import fs from 'node:fs';
import path from 'node:path';
import typeScript from 'typescript';

export const SOURCE_EXTENSIONS = /\.(?:[cm]?[jt]s|[jt]sx)$/;

function listFallbackInputs(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    if (['node_modules', 'dist', '.git', '.typecheck'].includes(entry.name)) return [];
    const file = path.join(directory, entry.name);
    return entry.isDirectory()
      ? listFallbackInputs(file)
      : SOURCE_EXTENSIONS.test(file) || file.endsWith('.json')
        ? [file]
        : [];
  });
}

export function getTypeScriptProjectInputs(configuration, fallbackDirectories = []) {
  const configFiles = new Set();
  if (!fs.existsSync(configuration))
    return {
      files: fallbackDirectories.flatMap(listFallbackInputs).sort(),
      configFiles: [],
      options: {allowJs: true, jsx: typeScript.JsxEmit.React}
    };
  const parsed = typeScript.getParsedCommandLineOfConfigFile(
    configuration,
    {},
    {
      ...typeScript.sys,
      readFile(file) {
        const contents = typeScript.sys.readFile(file);
        if (contents !== undefined && file.endsWith('.json')) configFiles.add(path.resolve(file));
        return contents;
      },
      onUnRecoverableConfigFileDiagnostic(diagnostic) {
        throw new Error(typeScript.flattenDiagnosticMessageText(diagnostic.messageText, '\n'));
      }
    }
  );
  const errors = parsed?.errors.filter(error => error.code !== 18003) ?? [];
  if (!parsed || errors.length)
    throw new Error(
      `Invalid project ${configuration}: ${errors.map(error => typeScript.flattenDiagnosticMessageText(error.messageText, '\n')).join('\n')}`
    );
  const files = new Set(parsed.fileNames.map(file => path.resolve(file)));
  if (parsed.options.resolveJsonModule) {
    // Config expansion omits JSON dependencies, including those imported by allowJs loaders.
    // Read import specifiers without constructing a program or typechecking the project.
    for (const file of [...files]) {
      if (!SOURCE_EXTENSIONS.test(file)) continue;
      const contents = typeScript.sys.readFile(file);
      if (contents === undefined) continue;
      for (const imported of typeScript.preProcessFile(contents, true, true).importedFiles) {
        const resolved = typeScript.resolveModuleName(
          imported.fileName,
          file,
          parsed.options,
          typeScript.sys
        ).resolvedModule?.resolvedFileName;
        if (resolved?.endsWith('.json')) files.add(path.resolve(resolved));
      }
    }
  }
  return {
    files: [...files].sort(),
    configFiles: [...configFiles].sort(),
    options: parsed.options
  };
}

export function captureTypeScriptProjectInputs(configuration, fallbackDirectories = []) {
  const project = getTypeScriptProjectInputs(configuration, fallbackDirectories);
  const entries = [...new Set([...project.files, ...project.configFiles])].sort().map(file => {
    const information = fs.statSync(file);
    // Restoring any input's timestamp and size must not certify an older compiler cycle.
    const content = createHash('sha256').update(fs.readFileSync(file)).digest('hex');
    return [file, information.mtimeMs, information.size, content];
  });
  return {
    fingerprint: createHash('sha256').update(JSON.stringify(entries)).digest('hex'),
    latestEdit: Math.max(0, ...entries.map(entry => entry[1]))
  };
}
