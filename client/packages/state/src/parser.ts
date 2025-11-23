export type PromptConfig = {
  prompt: string;
  duration?: number;
  model?: string;
  cfg?: number | 'off';
  seed?: number;
  motif?: string;
  size?: 'small' | 'medium' | 'large';
};

const DURATION_MIN = 90;
const DURATION_MAX = 180;

export const parsePrompt = (input: string): PromptConfig => {
  const parts = input.trim().split(/\s+/);
  const config: PromptConfig = { prompt: '' };

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    if (part === '/duration' && parts[i + 1]) {
      const value = Number(parts[i + 1]);
      if (!Number.isNaN(value)) {
        config.duration = Math.min(Math.max(value, DURATION_MIN), DURATION_MAX);
        i++;
        continue;
      }
    }
    if (part === '/model' && parts[i + 1]) {
      config.model = parts[i + 1];
      i++;
      continue;
    }
    if (part === '/cfg' && parts[i + 1]) {
      const next = parts[i + 1];
      if (next === 'off') {
        config.cfg = 'off';
      } else {
        const value = Number(next);
        if (!Number.isNaN(value)) {
          config.cfg = value;
        }
      }
      i++;
      continue;
    }
    if (part === '/seed' && parts[i + 1]) {
      const value = Number(parts[i + 1]);
      if (!Number.isNaN(value)) {
        config.seed = value;
      }
      i++;
      continue;
    }
    if (part === '/motif' && parts[i + 1]) {
      config.motif = parts[i + 1];
      i++;
      continue;
    }
    if (['/small', '/medium', '/large'].includes(part)) {
      config.size = part.replace('/', '') as PromptConfig['size'];
      continue;
    }
    config.prompt = [config.prompt, part].filter(Boolean).join(' ');
  }

  return config;
};
