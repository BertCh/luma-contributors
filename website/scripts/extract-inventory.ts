// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Extracts the user-facing operations of @luma-contributors/analysis into
// src/catalog/generated/inventory.json, using the TypeScript compiler API.
//
// Usage (from the website directory, Node 22.6+):
//   node --experimental-strip-types scripts/extract-inventory.ts
//
// An export is user-facing when it is a non-abstract class named GPU*, a function named
// add*Recipe / add*ToGraph / create* / make*, or any class or function listed as an operator in the
// spatial-analysis ontology. Analysis-core internals (kernels, contracts, graph support,
// ring assembly), low-level helpers, WGSL builders, types and constants are skipped.

import {existsSync, readdirSync, readFileSync, statSync, writeFileSync, mkdirSync} from 'node:fs';
import {dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import ts from 'typescript';

const websiteDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repositoryRoot = resolve(websiteDirectory, '..');
const analysisSource = join(repositoryRoot, 'modules/analysis/src');
const packageName = '@luma-contributors/analysis';

type InventoryProp = {name: string; type: string; optional: boolean; jsdoc: string};
type InventoryOntology = {
  capability: string;
  domain: string;
  stage: string;
  inputs: string[];
  outputs: string[];
};
type InventorySymbol = {
  symbol: string;
  kind: 'class' | 'function' | 'recipe';
  importPath: string;
  sourceFile: string;
  directory: string;
  propsType: string | null;
  props: InventoryProp[];
  /** Resolved output fields: the class \`output\`/\`outputs\` prop type, or the function return type. */
  outputs: InventoryProp[];
  summary: string;
  docsSlug: string | null;
  lensScenarios: string[];
  ontology: InventoryOntology | null;
};

const FUNCTION_NAME_PATTERN = /^(add\w+Recipe|add\w+ToGraph|create[A-Z]\w*|make[A-Z]\w*)$/;
const LOW_LEVEL_PATTERN = /(Wgsl|WGSL|Shader|Kernel|Descriptor|Annotation|Ref$|Validator|Hash)/;
const MAX_TYPE_LENGTH = 60;
const MAX_JSDOC_LENGTH = 70;
const MAX_LENS_SCENARIOS = 3;
const MAX_CONTENT_LENS_MATCHES = 3;

function listEntryPoints(): {importPath: string; file: string}[] {
  const packageJson = JSON.parse(readFileSync(join(repositoryRoot, 'modules/analysis/package.json'), 'utf8'));
  const entryPoints: {importPath: string; file: string}[] = [];
  for (const exportKey of Object.keys(packageJson.exports)) {
    const importPath = exportKey === '.' ? packageName : `${packageName}/${exportKey.slice(2)}`;
    const directory = exportKey === '.' ? '' : exportKey.slice(2);
    const file = join(analysisSource, directory, 'index.ts');
    if (existsSync(file)) entryPoints.push({importPath, file});
  }
  return entryPoints;
}

function listFilesRecursive(directory: string, extension: string, skipDirectories: string[] = []): string[] {
  const files: string[] = [];
  for (const name of readdirSync(directory)) {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) {
      if (!skipDirectories.includes(name)) files.push(...listFilesRecursive(path, extension, skipDirectories));
    } else if (name.endsWith(extension)) {
      files.push(path);
    }
  }
  return files;
}

function summarizeText(text: string, maxLength: number): string {
  const firstSentence = text.replace(/\s+/g, ' ').trim().split(/(?<=\.)\s/)[0] ?? '';
  return firstSentence.length > maxLength ? `${firstSentence.slice(0, maxLength - 1)}…` : firstSentence;
}

function readOntology(): Map<string, InventoryOntology> {
  const ontologyDirectory = join(analysisSource, 'gpu-spatial-analysis/ontology');
  const byOperator = new Map<string, InventoryOntology>();
  const readStrings = (node: ts.Expression | undefined): string[] =>
    node && ts.isArrayLiteralExpression(node)
      ? node.elements.filter(ts.isStringLiteralLike).map(element => element.text)
      : [];
  for (const name of readdirSync(ontologyDirectory).filter(entry => entry.endsWith('.ts'))) {
    const source = ts.createSourceFile(name, readFileSync(join(ontologyDirectory, name), 'utf8'), ts.ScriptTarget.ES2022, true);
    const visit = (node: ts.Node): void => {
      if (ts.isObjectLiteralExpression(node)) {
        const fields = new Map<string, ts.Expression>();
        for (const property of node.properties) {
          if (ts.isPropertyAssignment(property) && ts.isIdentifier(property.name)) {
            fields.set(property.name.text, property.initializer);
          }
        }
        const operators = readStrings(fields.get('operators'));
        const identifier = fields.get('id');
        if (operators.length > 0 && identifier && ts.isStringLiteralLike(identifier)) {
          const readText = (key: string) => {
            const value = fields.get(key);
            return value && ts.isStringLiteralLike(value) ? value.text : '';
          };
          const ontology = {
            capability: identifier.text,
            domain: readText('domain'),
            stage: readText('stage'),
            inputs: readStrings(fields.get('inputs')),
            outputs: readStrings(fields.get('outputs'))
          };
          for (const operator of operators) if (!byOperator.has(operator)) byOperator.set(operator, ontology);
        }
      }
      ts.forEachChild(node, visit);
    };
    visit(source);
  }
  return byOperator;
}

function readDocumentationPages(): {slugs: string[]; pages: {slug: string; text: string}[]} {
  const docsRoot = join(repositoryRoot, 'docs');
  const pages = listFilesRecursive(join(docsRoot, 'api'), '.md').map(file => {
    let slug = relative(docsRoot, file).replace(/\.md$/, '');
    if (slug.endsWith('/README')) slug = slug.slice(0, -'/README'.length);
    return {slug, text: readFileSync(file, 'utf8')};
  });
  return {slugs: pages.map(page => page.slug).sort(), pages};
}

function readLensScenarios(): {ids: string[]; records: {id: string; text: string}[]} {
  const directory = join(repositoryRoot, 'docs/lens-records');
  const records = readdirSync(directory)
    .filter(name => name.endsWith('.json') && statSync(join(directory, name)).isFile())
    .map(name => ({id: name.replace(/\.json$/, ''), text: readFileSync(join(directory, name), 'utf8')}));
  return {ids: records.map(record => record.id).sort(), records};
}

function main(): void {
  const entryPoints = listEntryPoints();
  const configFile = ts.readConfigFile(join(repositoryRoot, 'tsconfig.json'), ts.sys.readFile);
  const parsed = ts.parseJsonConfigFileContent(configFile.config, ts.sys, repositoryRoot);
  const program = ts.createProgram({
    rootNames: entryPoints.map(entryPoint => entryPoint.file),
    options: {...parsed.options, noEmit: true, plugins: undefined}
  });
  const checker = program.getTypeChecker();
  const ontologyByOperator = readOntology();
  const documentation = readDocumentationPages();
  const lensScenarios = readLensScenarios();

  const formatType = (type: ts.Type): string => {
    const text = checker.typeToString(type, undefined, ts.TypeFormatFlags.NoTruncation | ts.TypeFormatFlags.UseAliasDefinedOutsideCurrentScope).replace(/\s+/g, ' ');
    return text.length > MAX_TYPE_LENGTH ? `${text.slice(0, MAX_TYPE_LENGTH - 1)}…` : text;
  };

  const resolveProps = (parameter: ts.Symbol | undefined, location: ts.Node): {propsType: string | null; props: InventoryProp[]} => {
    if (!parameter) return {propsType: null, props: []};
    const parameterType = checker.getNonNullableType(checker.getTypeOfSymbolAtLocation(parameter, location));
    const declaration = parameter.valueDeclaration;
    const typeNode = declaration && ts.isParameter(declaration) ? declaration.type : undefined;
    const propsType = typeNode ? typeNode.getText().replace(/\s+/g, ' ').slice(0, MAX_TYPE_LENGTH) : formatType(parameterType);
    const props: InventoryProp[] = [];
    // Unions (for example discriminated option sets) contribute every member's props; a prop that
    // is missing from some member is reported as optional.
    const members = parameterType.isUnion() ? parameterType.types : [parameterType];
    const merged = new Map<string, {property: ts.Symbol; count: number}>();
    for (const member of members) {
      if (!(member.flags & ts.TypeFlags.Object || member.isIntersection())) continue;
      for (const property of checker.getPropertiesOfType(member)) {
        if (property.name.startsWith('__@')) continue;
        const propertyFile = property.declarations?.[0]?.getSourceFile().fileName ?? '';
        if (propertyFile.includes('node_modules')) continue;
        const entry = merged.get(property.name);
        if (entry) entry.count += 1;
        else merged.set(property.name, {property, count: 1});
      }
    }
    for (const {property, count} of merged.values()) {
      const propertyType = checker.getTypeOfSymbolAtLocation(property, location);
      props.push({
        name: property.name,
        type: formatType(propertyType),
        optional: (property.flags & ts.SymbolFlags.Optional) !== 0 || count < members.length,
        jsdoc: summarizeText(ts.displayPartsToString(property.getDocumentationComment(checker)), MAX_JSDOC_LENGTH)
      });
    }
    return {propsType, props};
  };

  const listObjectFields = (type: ts.Type, location: ts.Node): InventoryProp[] => {
    const fields: InventoryProp[] = [];
    const members = type.isUnion() ? type.types : [type];
    const seen = new Set<string>();
    for (const member of members) {
      if (!(member.flags & ts.TypeFlags.Object || member.isIntersection())) continue;
      if (checker.getSignaturesOfType(member, ts.SignatureKind.Call).length > 0) continue;
      for (const property of checker.getPropertiesOfType(member)) {
        if (property.name.startsWith('__@') || seen.has(property.name)) continue;
        const propertyFile = property.declarations?.[0]?.getSourceFile().fileName ?? '';
        if (propertyFile.includes('node_modules') || propertyFile.includes('lib.')) continue;
        seen.add(property.name);
        fields.push({
          name: property.name,
          type: formatType(checker.getTypeOfSymbolAtLocation(property, location)),
          optional: (property.flags & ts.SymbolFlags.Optional) !== 0,
          jsdoc: summarizeText(ts.displayPartsToString(property.getDocumentationComment(checker)), MAX_JSDOC_LENGTH)
        });
      }
    }
    return fields;
  };

  // Functions whose options are not a single props object: report every parameter by name and
  // flatten object-typed parameters as dotted names such as \`options.scale\`.
  const resolveParameterProps = (signature: ts.Signature, location: ts.Node): InventoryProp[] => {
    const props: InventoryProp[] = [];
    for (const parameter of signature.getParameters()) {
      if (/^(device|graph|commandGraph)$/.test(parameter.name)) continue;
      const declaration = parameter.valueDeclaration;
      const optional = !!declaration && ts.isParameter(declaration) && (!!declaration.questionToken || !!declaration.initializer);
      const parameterType = checker.getNonNullableType(checker.getTypeOfSymbolAtLocation(parameter, location));
      props.push({name: parameter.name, type: formatType(parameterType), optional, jsdoc: ''});
      const isPlainObject = (parameterType.flags & ts.TypeFlags.Object) !== 0 && checker.getSignaturesOfType(parameterType, ts.SignatureKind.Call).length === 0 && parameterType.symbol?.name !== 'Float32Array' && !checker.isArrayLikeType(parameterType);
      if (!isPlainObject) continue;
      for (const field of resolveProps(parameter, location).props) props.push({...field, name: `${parameter.name}.${field.name}`});
    }
    return props;
  };

  const resolveOutputs = (isClass: boolean, props: InventoryProp[], parameter: ts.Symbol | undefined, signature: ts.Signature | undefined, location: ts.Node): InventoryProp[] => {
    if (isClass) {
      const outputProp = parameter && checker.getPropertyOfType(checker.getNonNullableType(checker.getTypeOfSymbolAtLocation(parameter, location)), 'outputs') ||
        parameter && checker.getPropertyOfType(checker.getNonNullableType(checker.getTypeOfSymbolAtLocation(parameter, location)), 'output');
      return outputProp ? listObjectFields(checker.getNonNullableType(checker.getTypeOfSymbolAtLocation(outputProp, location)), location) : [];
    }
    if (!signature) return [];
    let returnType = checker.getReturnTypeOfSignature(signature);
    if (returnType.symbol?.name === 'Promise') returnType = checker.getTypeArguments(returnType as ts.TypeReference)[0] ?? returnType;
    return listObjectFields(returnType, location);
  };

  const symbols = new Map<string, InventorySymbol>();
  for (const entryPoint of entryPoints) {
    const sourceFile = program.getSourceFile(entryPoint.file);
    const moduleSymbol = sourceFile && checker.getSymbolAtLocation(sourceFile);
    if (!moduleSymbol) continue;
    for (const exported of checker.getExportsOfModule(moduleSymbol)) {
      const name = exported.name;
      const target = exported.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(exported) : exported;
      const declaration = target.declarations?.find(candidate => ts.isClassDeclaration(candidate) || ts.isFunctionDeclaration(candidate));
      if (!declaration) continue;
      const declarationFile = declaration.getSourceFile().fileName;
      if (!declarationFile.startsWith(analysisSource) || declarationFile.includes('/testing/')) continue;
      const isClass = ts.isClassDeclaration(declaration);
      const isOperator = ontologyByOperator.has(name);
      const isAbstract = isClass && (ts.getCombinedModifierFlags(declaration) & ts.ModifierFlags.Abstract) !== 0;
      const included = isClass
        ? !isAbstract && (/^GPU[A-Z]/.test(name) || isOperator)
        : (FUNCTION_NAME_PATTERN.test(name) && !LOW_LEVEL_PATTERN.test(name)) || isOperator;
      const isCoreInternal = /\/analysis-core\/(kernels|graph-support|contracts|ring-assembly)\//.test(declarationFile);
      if (!included || (isCoreInternal && !isOperator)) continue;

      const signature = isClass
        ? checker.getTypeOfSymbolAtLocation(target, declaration).getConstructSignatures()[0]
        : checker.getTypeOfSymbolAtLocation(target, declaration).getCallSignatures()[0];
      // Skip leading infrastructure parameters (device, graph) to reach the options object.
      const parameter = signature?.getParameters().find(candidate => {
        const typeText = checker.typeToString(checker.getTypeOfSymbolAtLocation(candidate, declaration));
        return !/^(device|graph|commandGraph)$/.test(candidate.name) && !/^(Device|GPUCommandGraph)\b/.test(typeText);
      });
      const resolved = resolveProps(parameter, declaration);
      const propsType = resolved.propsType;
      const props = resolved.props.length > 0 || isClass || !signature ? resolved.props : resolveParameterProps(signature, declaration);
      const outputs = resolveOutputs(isClass, props, parameter, signature, declaration);

      const existing = symbols.get(name);
      const importPath = entryPoint.importPath;
      // Prefer the most specific entry point when a symbol is re-exported from the root.
      if (existing && existing.importPath.length >= importPath.length) continue;

      const sourcePath = relative(repositoryRoot, declarationFile);
      const kind: InventorySymbol['kind'] = isClass ? 'class' : /Recipe$|ToGraph$/.test(name) ? 'recipe' : 'function';
      const wordPattern = new RegExp(`\\b${name}\\b`);
      const pageScores = documentation.pages
        .map(page => ({page, count: (page.text.match(new RegExp(`\\b${name}\\b`, 'g')) ?? []).length}))
        .filter(({count}) => count > 0)
        .map(({page, count}) => ({slug: page.slug, score: count - (/catalog|cross-reference|^api$/.test(page.slug) ? 1000 : 0)}))
        .sort((left, right) => right.score - left.score || left.slug.localeCompare(right.slug));
      const kebabName = name
        .replace(/^(GPU|add|create|make)/, '')
        .replace(/(Recipe|ToGraph)$/, '')
        .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
        .replace(/([A-Z]+)([A-Z][a-z])/g, '$1-$2')
        .toLowerCase();
      const scenarioName = (id: string): string => id.slice(id.indexOf('.') + 1);
      const nameMatches = lensScenarios.records
        .filter(record => [kebabName, `${kebabName}-recipe`].includes(scenarioName(record.id)))
        .map(record => record.id);
      const nearMatches = lensScenarios.records
        .filter(record => scenarioName(record.id).includes(kebabName) && !nameMatches.includes(record.id))
        .map(record => record.id);
      // Content references are only trusted when specific: a symbol named in many records is noise.
      const contentMatches = lensScenarios.records
        .filter(record => wordPattern.test(record.text) && !nameMatches.includes(record.id) && !nearMatches.includes(record.id))
        .map(record => record.id);
      const lensMatches = [
        ...nameMatches,
        ...nearMatches,
        ...(contentMatches.length <= MAX_CONTENT_LENS_MATCHES ? contentMatches : [])
      ];
      symbols.set(name, {
        symbol: name,
        kind,
        importPath,
        sourceFile: sourcePath,
        directory: relative(analysisSource, dirname(declarationFile)).split('/').slice(0, 2).join('/'),
        propsType,
        props,
        outputs,
        summary: summarizeText(ts.displayPartsToString(target.getDocumentationComment(checker)), MAX_JSDOC_LENGTH * 2),
        docsSlug: pageScores[0]?.slug ?? null,
        lensScenarios: lensMatches.slice(0, MAX_LENS_SCENARIOS),
        ontology: ontologyByOperator.get(name) ?? null
      });
    }
  }

  const sorted = [...symbols.values()].sort((left, right) => left.symbol.localeCompare(right.symbol));
  const output = {
    entryPoints: entryPoints.map(entryPoint => entryPoint.importPath),
    docSlugs: documentation.slugs,
    lensScenarioIds: lensScenarios.ids,
    symbols: sorted
  };
  const outputFile = join(websiteDirectory, 'src/catalog/generated/inventory.json');
  mkdirSync(dirname(outputFile), {recursive: true});
  // One symbol per line keeps diffs readable and the file small.
  const lines = sorted.map(symbol => `  ${JSON.stringify(symbol)}`).join(',\n');
  const header = JSON.stringify({entryPoints: output.entryPoints, docSlugs: output.docSlugs, lensScenarioIds: output.lensScenarioIds}).slice(0, -1);
  writeFileSync(outputFile, `${header},"symbols":[\n${lines}\n]}\n`);
  console.log(`inventory: ${sorted.length} symbols -> ${relative(repositoryRoot, outputFile)}`);
}

main();
