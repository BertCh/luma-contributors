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
      options: {
        allowJs: true,
        declaration: true,
        jsx: typeScript.JsxEmit.React,
        configFilePath: path.resolve(configuration)
      }
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
    // Config expansion omits consumed JSON. Share use-site discovery with freshness, but
    // installed dependencies remain freshness-only rather than becoming compiler emit roots.
    for (const file of listTypeScriptDependencyInputs(
      {files: [...files], options: parsed.options},
      {includeExternal: false}
    ))
      if (file.endsWith('.json')) files.add(file);
  }
  return {
    files: [...files].sort(),
    configFiles: [...configFiles].sort(),
    options: parsed.options,
    projectReferences: parsed.projectReferences ?? []
  };
}

export const DECLARATION_FILE = /\.d\.[cm]?ts$/;

const DECLARATION_SUFFIX_BY_RUNTIME = new Map([
  ['.js', '.d.ts'],
  ['.cjs', '.d.cts'],
  ['.mjs', '.d.mts']
]);

/** Qualify only import types whose target declarations exist beside the compiler output. */
export function rewriteDeclarationImportTypes(source, declarationFile, extensions = ['.js']) {
  const suffixes = new Map(
    extensions.map(extension => {
      const runtime = path.posix.extname(extension) || extension;
      return [path.posix.basename(extension, runtime), runtime];
    })
  );
  const directory = path.dirname(declarationFile);
  const preferred = declarationFile.endsWith('.d.cts')
    ? '.cjs'
    : declarationFile.endsWith('.d.mts')
      ? '.mjs'
      : '.js';
  const syntax = typeScript.createSourceFile(
    declarationFile,
    source,
    typeScript.ScriptTarget.Latest
  );
  const additions = [];
  const resolvedSuffixes = new Map();

  function resolveSuffix(specifier) {
    const configured = suffixes.get(path.posix.extname(specifier));
    if (!configured) return '';
    const index = specifier.endsWith('/') ? 'index' : '/index';
    const stems =
      specifier === '.' || specifier === '..' || specifier.endsWith('/') ? [index] : ['', index];
    function findTarget(ending) {
      const declarationSuffix = DECLARATION_SUFFIX_BY_RUNTIME.get(ending);
      if (!declarationSuffix) return '';
      for (const stem of stems) {
        const target = path.resolve(directory, `${specifier}${stem}${declarationSuffix}`);
        if (typeScript.sys.fileExists(target)) return stem + ending;
      }
      return '';
    }
    // Match the configured extension and its file-before-index lookup first. If that fails,
    // prefer the declaration's native format and accept another format only if unambiguous.
    const resolved = findTarget(configured) || findTarget(preferred);
    if (resolved) return resolved;
    const alternatives = [...DECLARATION_SUFFIX_BY_RUNTIME.keys()]
      .filter(ending => ending !== configured && ending !== preferred)
      .map(findTarget)
      .filter(Boolean);
    return alternatives.length === 1 ? alternatives[0] : '';
  }

  function visit(node) {
    if (
      typeScript.isImportTypeNode(node) &&
      typeScript.isLiteralTypeNode(node.argument) &&
      typeScript.isStringLiteral(node.argument.literal)
    ) {
      const literal = node.argument.literal;
      const specifier = literal.text;
      if (
        specifier === '.' ||
        specifier === '..' ||
        specifier.startsWith('./') ||
        specifier.startsWith('../')
      ) {
        if (!resolvedSuffixes.has(specifier))
          resolvedSuffixes.set(specifier, resolveSuffix(specifier));
        const suffix = resolvedSuffixes.get(specifier);
        if (suffix) additions.push({offset: literal.getEnd() - 1, suffix});
      }
    }
    typeScript.forEachChild(node, visit);
  }
  visit(syntax);

  // Insert inside the original quotes to preserve escapes, comments and all other emitted text.
  for (const {offset, suffix} of additions.reverse())
    source = source.slice(0, offset) + suffix + source.slice(offset);
  return source;
}

/** Shared lexical dependency resolution. Configured emit roots remain in project.files.
 * A tracked host lets measurement retain package metadata reads and broader library freshness.
 */
export function listTypeScriptDependencyInputs(
  project,
  {
    host = typeScript.sys,
    includeLibraries = false,
    includeExternal = true,
    ignoredFiles = new Set(),
    ignoredPackages = new Set()
  } = {}
) {
  const moduleCache = typeScript.createModuleResolutionCache(
    process.cwd(),
    file => file,
    project.options
  );
  const queue = project.files.map(file => ({file, depth: 0}));
  const visitedFiles = new Set();
  const visitedDepths = new Map();
  const libraryDirectory = path.dirname(typeScript.getDefaultLibFilePath(project.options));
  if (includeLibraries && !project.options.noLib)
    queue.push(
      ...(project.options.lib ?? [typeScript.getDefaultLibFileName(project.options)]).map(file => ({
        file: path.join(libraryDirectory, file),
        depth: 0
      }))
    );
  function addTypeReference(reference, containingFile, containingMode, depth = 0) {
    const name = typeof reference === 'string' ? reference : reference.fileName;
    if (ignoredPackages.has(name)) return;
    const resolved = typeScript.resolveTypeReferenceDirective(
      name,
      containingFile,
      project.options,
      host,
      undefined,
      undefined,
      typeScript.getModeForFileReference(reference, containingMode)
    ).resolvedTypeReferenceDirective;
    if (resolved && (includeExternal || !resolved.isExternalLibraryImport))
      queue.push({file: resolved.resolvedFileName, depth});
  }
  const configuration = project.options.configFilePath ?? path.resolve('__types__.ts');
  const configurationMode = typeScript.getImpliedNodeFormatForFile(
    configuration,
    moduleCache.getPackageJsonInfoCache(),
    host,
    project.options
  );
  for (const name of typeScript.getAutomaticTypeDirectiveNames(project.options, host))
    addTypeReference(name, configuration, configurationMode);
  for (let index = 0; index < queue.length; index++) {
    const {depth} = queue[index];
    const file = path.resolve(queue[index].file);
    // A later, shallower route may admit imports elided through an earlier deep route.
    if ((visitedDepths.get(file) ?? Infinity) <= depth || ignoredFiles.has(file)) continue;
    const source = host.readFile(file);
    if (source === undefined) continue;
    visitedFiles.add(file);
    visitedDepths.set(file, depth);
    if (!SOURCE_EXTENSIONS.test(file)) continue;
    const format = typeScript.getImpliedNodeFormatForFileWorker(
      file,
      moduleCache.getPackageJsonInfoCache(),
      host,
      project.options
    );
    const sourceFile = typeScript.createSourceFile(
      file,
      source,
      {
        languageVersion: typeScript.ScriptTarget.Latest,
        jsDocParsingMode: typeScript.JSDocParsingMode.ParseAll,
        impliedNodeFormat: typeof format === 'object' ? format.impliedNodeFormat : format,
        setExternalModuleIndicator: typeScript.getSetExternalModuleIndicator(project.options)
      },
      true
    );
    if (typeof format === 'object') Object.assign(sourceFile, format);
    // Keep each use site: one specifier can select both import and require declarations.
    const imports = [];
    function addImport(literal) {
      if (typeScript.isStringLiteralLike(literal))
        imports.push({
          specifier: literal.text,
          mode: typeScript.getModeForUsageLocation(sourceFile, literal, project.options)
        });
    }
    function visit(node) {
      if (
        (typeScript.isImportDeclaration(node) || typeScript.isExportDeclaration(node)) &&
        node.moduleSpecifier
      )
        addImport(node.moduleSpecifier);
      else if (
        typeScript.isImportEqualsDeclaration(node) &&
        typeScript.isExternalModuleReference(node.moduleReference) &&
        node.moduleReference.expression
      )
        addImport(node.moduleReference.expression);
      else if (typeScript.isImportTypeNode(node) && typeScript.isLiteralTypeNode(node.argument))
        addImport(node.argument.literal);
      else if (typeScript.isJSDocImportTag(node) && node.moduleSpecifier)
        addImport(node.moduleSpecifier);
      else if (typeScript.isCallExpression(node) && node.arguments[0]) {
        if (node.expression.kind === typeScript.SyntaxKind.ImportKeyword)
          addImport(node.arguments[0]);
        else if (typeScript.isSourceFileJS(sourceFile) && typeScript.isRequireCall(node, true))
          addImport(node.arguments[0]);
      }
      // TypeScript collects JSDoc import types/tags only for JavaScript. Documentation
      // in typed sources and declarations must not create freshness dependencies.
      if (typeScript.isSourceFileJS(sourceFile))
        for (const documentation of node.jsDoc ?? []) visit(documentation);
      typeScript.forEachChild(node, visit);
    }
    visit(sourceFile);
    // Match the compiler's synthetic runtime import eligibility and effective emit syntax.
    // In bundler mode, a dependency's implied Node format is not its import resolution mode.
    if (
      typeScript.isSourceFileJS(sourceFile) ||
      (!sourceFile.isDeclarationFile &&
        (typeScript.getIsolatedModules(project.options) || typeScript.isExternalModule(sourceFile)))
    ) {
      const runtimes = [
        project.options.importHelpers ? 'tslib' : undefined,
        typeScript.getJSXRuntimeImport(
          typeScript.getJSXImplicitImportBase(project.options, sourceFile),
          project.options
        )
      ];
      for (const runtime of runtimes) {
        if (!runtime) continue;
        const declaration = typeScript.factory.createImportDeclaration(
          undefined,
          undefined,
          typeScript.factory.createStringLiteral(runtime),
          undefined
        );
        declaration.parent = sourceFile;
        declaration.moduleSpecifier.parent = declaration;
        addImport(declaration.moduleSpecifier);
      }
    }
    for (const {specifier, mode} of imports) {
      if ([...ignoredPackages].some(name => specifier === name || specifier.startsWith(`${name}/`)))
        continue;
      const resolved = typeScript.resolveModuleName(
        specifier,
        file,
        project.options,
        host,
        moduleCache,
        undefined,
        mode
      ).resolvedModule;
      if (
        !resolved ||
        project.options.noResolve ||
        (!includeExternal && resolved.isExternalLibraryImport)
      )
        continue;
      const isJavaScript = /\.(?:[cm]?js|jsx)$/.test(resolved.resolvedFileName);
      const nextDepth = depth + (resolved.isExternalLibraryImport ? 1 : 0);
      const isExternalJavaScript =
        resolved.isExternalLibraryImport &&
        isJavaScript &&
        (!resolved.originalPath ||
          resolved.resolvedFileName.split(path.sep).includes('node_modules'));
      if (isJavaScript && !typeScript.getAllowJSCompilerOption(project.options)) continue;
      if (isExternalJavaScript && nextDepth > (project.options.maxNodeModuleJsDepth ?? 0)) continue;
      queue.push({file: resolved.resolvedFileName, depth: nextDepth});
    }
    for (const reference of project.options.noResolve ? [] : sourceFile.referencedFiles) {
      const target = path.resolve(path.dirname(file), reference.fileName);
      const candidates = typeScript.hasExtension(target)
        ? [target]
        : [
            ...(project.options.allowNonTsExtensions ? [target] : []),
            ...typeScript
              .getSupportedExtensions(project.options)[0]
              .map(extension => target + extension)
          ];
      const selected = candidates.find(candidate => host.fileExists(candidate));
      if (selected) queue.push({file: selected, depth});
    }
    for (const reference of project.options.noResolve ? [] : sourceFile.typeReferenceDirectives)
      addTypeReference(
        reference,
        file,
        typeScript.getDefaultResolutionModeForFileWorker(sourceFile, project.options),
        depth
      );
    if (includeLibraries)
      for (const reference of sourceFile.libReferenceDirectives)
        queue.push({
          file: path.join(libraryDirectory, `lib.${reference.fileName.toLowerCase()}.d.ts`),
          depth
        });
  }
  return [...visitedFiles].sort();
}

/** Inputs of referenced projects stand for their generated declarations, including on a clean build. */
export function listTypeScriptFreshnessInputs(
  project,
  {includeLibraries = false, host = typeScript.sys} = {}
) {
  const projects = [];
  const configurations = new Set();
  function addProject(current) {
    const configuration = current.options.configFilePath;
    if (configuration && configurations.has(configuration)) return;
    if (configuration) configurations.add(configuration);
    projects.push(current);
    for (const reference of current.projectReferences ?? [])
      addProject(getTypeScriptProjectInputs(typeScript.resolveProjectReferencePath(reference)));
  }
  addProject(project);
  const ignoredFiles = new Set();
  const ignoredPackages = new Set();
  for (const current of projects) {
    if (!current.options.configFilePath) continue;
    const directory = path.dirname(current.options.configFilePath);
    const metadata = host.readFile(path.join(directory, 'package.json'));
    if (metadata) {
      const name = JSON.parse(metadata).name;
      if (name) ignoredPackages.add(name);
    }
    const outputDirectory = current.options.outDir ?? path.join(directory, 'dist');
    for (const entry of getTypeScriptOutputPlan(current, directory))
      for (const output of [...entry.compilerOutputs, ...entry.transformedOutputs])
        ignoredFiles.add(path.resolve(outputDirectory, output));
  }
  const roots = new Set(project.files);
  const inputs = new Set();
  for (const current of projects) {
    for (const file of listTypeScriptDependencyInputs(current, {
      host,
      includeLibraries,
      ignoredFiles,
      ignoredPackages
    }))
      if (!roots.has(file)) inputs.add(file);
    if (current !== project) {
      for (const file of current.configFiles ?? []) inputs.add(file);
      const metadata = path.join(path.dirname(current.options.configFilePath), 'package.json');
      if (host.fileExists(metadata)) {
        host.readFile(metadata);
        inputs.add(metadata);
      }
    }
  }
  return [...inputs].sort();
}

/** External declarations are freshness-only inputs; never plan emitted files for them. */
export function listExternalDeclarationInputs(project) {
  const roots = new Set(project.files);
  return listTypeScriptDependencyInputs(project).filter(
    file => !roots.has(file) && DECLARATION_FILE.test(file)
  );
}

export function captureTypeScriptProjectInputs(configuration, fallbackDirectories = []) {
  const project = getTypeScriptProjectInputs(configuration, fallbackDirectories);
  const entries = [
    ...new Set([
      ...project.files,
      ...project.configFiles,
      ...listTypeScriptFreshnessInputs(project)
    ])
  ]
    .sort()
    .map(file => {
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

/** Compiler outputs and transformed siblings are distinct: native .cts outputs need compiler repair. */
export function getTypeScriptOutputPlan(project, moduleDirectory) {
  const sourceDirectory = project.options.rootDir ?? path.resolve(moduleDirectory, 'src');
  const outputDirectory = project.options.outDir ?? path.resolve(moduleDirectory, 'dist');
  const options = {
    ...project.options,
    rootDir: sourceDirectory,
    outDir: outputDirectory,
    configFilePath: project.options.configFilePath ?? path.resolve(moduleDirectory, 'tsconfig.json')
  };
  const commandLine = {options, fileNames: project.files, errors: []};
  const plan = project.files.flatMap(source => {
    const relative = path.relative(sourceDirectory, source);
    if (relative.startsWith('..') || DECLARATION_FILE.test(source)) return [];
    const compilerOutputs = typeScript
      .getOutputFileNames(commandLine, source, !typeScript.sys.useCaseSensitiveFileNames)
      .map(file => path.relative(outputDirectory, file));
    return [
      {
        source,
        compilerOutputs,
        transformedOutputs: compilerOutputs.flatMap(file => {
          if (!file.endsWith('.js')) return [];
          const commonJS = file.replace(/\.js$/, '.cjs');
          return compilerOutputs.includes(`${file}.map`)
            ? [commonJS, `${commonJS}.map`]
            : [commonJS];
        })
      }
    ];
  });
  const owners = new Map();
  for (const entry of plan) {
    for (const output of [...entry.compilerOutputs, ...entry.transformedOutputs]) {
      if (owners.has(output))
        throw new Error(
          `competing output owners for ${output}: ${owners.get(output)} and ${entry.source}`
        );
      owners.set(output, entry.source);
    }
  }
  return plan;
}
