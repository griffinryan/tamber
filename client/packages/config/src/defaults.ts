const DEFAULT_WORKER_URL = 'http://localhost:8000';

type EnvGetter = () => string | undefined;

const readEnv = (): EnvGetter => {
  if (typeof import.meta !== 'undefined' && (import.meta as Record<string, unknown>).env) {
    const env = (import.meta as Record<string, any>).env;
    return () => env?.VITE_WORKER_URL;
  }
  if (typeof process !== 'undefined' && process.env) {
    return () => process.env.VITE_WORKER_URL || process.env.TIMBRE_WORKER_URL;
  }
  return () => undefined;
};

export const resolveWorkerUrl = (fallback = DEFAULT_WORKER_URL): string => {
  const envUrl = readEnv()();
  return envUrl && envUrl.length > 0 ? envUrl : fallback;
};

export const configDefaults = {
  workerUrl: DEFAULT_WORKER_URL,
};
