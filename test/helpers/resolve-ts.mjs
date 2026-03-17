// resolve-ts.mjs — Node loader hook that resolves extensionless imports to .ts
// Usage: node --loader ./test/helpers/resolve-ts.mjs ...
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

export async function resolve(specifier, context, nextResolve) {
  try {
    return await nextResolve(specifier, context);
  } catch (err) {
    if (err.code === 'ERR_MODULE_NOT_FOUND' && context.parentURL
        && (specifier.startsWith('./') || specifier.startsWith('../'))) {
      const parentDir = dirname(fileURLToPath(context.parentURL));
      const tsPath = join(parentDir, specifier + '.ts');
      if (existsSync(tsPath)) {
        return nextResolve(specifier + '.ts', context);
      }
    }
    throw err;
  }
}
