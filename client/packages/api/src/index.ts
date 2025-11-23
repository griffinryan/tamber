export type WorkerStatus = {
  status: string;
};

export type WorkerClientOptions = {
  baseUrl?: string;
  fetchImpl?: typeof fetch;
};

export type WorkerClient = {
  getStatus: () => Promise<WorkerStatus>;
};

const defaultWorkerUrl = 'http://localhost:8000';

export const createWorkerClient = (options: WorkerClientOptions = {}): WorkerClient => {
  const fetcher = options.fetchImpl ?? globalThis.fetch;
  const baseUrl = options.baseUrl ?? defaultWorkerUrl;

  if (!fetcher) {
    throw new Error('Fetch implementation is required for WorkerClient.');
  }

  const getStatus = async (): Promise<WorkerStatus> => {
    const response = await fetcher(`${baseUrl.replace(/\/$/, '')}/status`);
    if (!response.ok) {
      throw new Error(`Worker status request failed (${response.status})`);
    }
    return (await response.json()) as WorkerStatus;
  };

  return {
    getStatus,
  };
};
