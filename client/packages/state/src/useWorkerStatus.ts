import { useEffect, useMemo, useState } from 'react';
import { createWorkerClient, WorkerStatus } from '@timbre/api';
import { resolveWorkerUrl } from '@timbre/config';

type StatusState = {
  status?: string;
  error?: string | null;
};

export const useWorkerStatus = (baseUrl?: string, intervalMs = 6000): StatusState => {
  const resolvedBaseUrl = baseUrl ?? resolveWorkerUrl();
  const client = useMemo(() => createWorkerClient({ baseUrl: resolvedBaseUrl }), [resolvedBaseUrl]);
  const [state, setState] = useState<StatusState>({});

  useEffect(() => {
    let mounted = true;
    let timer: ReturnType<typeof setInterval> | null = null;

    const tick = async () => {
      try {
        const next: WorkerStatus = await client.getStatus();
        if (mounted) {
          setState({ status: next.status, error: null });
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to reach worker';
        if (mounted) {
          setState({ status: undefined, error: message });
        }
      }
    };

    void tick();
    timer = setInterval(tick, intervalMs);

    return () => {
      mounted = false;
      if (timer) {
        clearInterval(timer);
      }
    };
  }, [client, intervalMs]);

  return state;
};
