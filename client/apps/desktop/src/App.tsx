import React, { useMemo, useState } from 'react';
import { Button, H4, Paragraph, Separator, Text, Theme, XStack, YStack } from 'tamagui';
import { resolveWorkerUrl } from '@timbre/config';
import { GlassCard, InputField, Pill, StatusCard } from '@timbre/ui';
import { parsePrompt, useWorkerStatus } from '@timbre/state';
import { TamaguiProvider } from 'tamagui';
import config from '../../tamagui.config';

type StatusShape = {
  status?: string;
};

type JobCard = {
  id: string;
  prompt: string;
  status: 'pending' | 'running' | 'done' | 'failed';
};

const STORAGE_KEY = 'timbre.workerUrl';

const App: React.FC = () => {
  const defaultUrl = useMemo(() => resolveWorkerUrl(), []);
  const [workerUrl, setWorkerUrl] = useState<string>(() => {
    if (typeof window === 'undefined') return defaultUrl;
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored && stored.length > 0 ? stored : defaultUrl;
  });
  const status = useWorkerStatus(workerUrl);
  const [prompt, setPrompt] = useState<string>('/duration 120 /model musicgen-stereo-medium oceanic synth shimmer');
  const [jobs, setJobs] = useState<JobCard[]>([]);

  const parsed = useMemo(() => parsePrompt(prompt), [prompt]);

  const submit = () => {
    // Stub: local job card until API integration is wired.
    const id = `job_${Date.now()}`;
    setJobs((prev) => [
      {
        id,
        prompt: parsed.prompt || prompt,
        status: 'pending',
      },
      ...prev,
    ]);
  };

  return (
    <TamaguiProvider config={config}>
      <Theme name="dark">
        <YStack
          minHeight="100vh"
          padding="$6"
          gap="$4"
          backgroundColor="$background"
          backgroundImage={`radial-gradient(circle at 10% 20%, rgba(126, 241, 255, 0.12), transparent 24%),
            radial-gradient(circle at 80% 0%, rgba(124, 140, 255, 0.18), transparent 24%)`}
        >
          <YStack gap="$2">
            <Text color="$textMuted" letterSpacing={2} fontSize={12} textTransform="uppercase">
              React Native Web + Electron
            </Text>
            <H4 color="$text" fontFamily="$heading" fontSize={32} letterSpacing={0.6}>
              Timbre Client
            </H4>
            <Paragraph color="$textMuted" maxWidth={720}>
              Liquid-glass scaffold for the cross-platform client. Desktop today, iOS/Android next.
              Point `VITE_WORKER_URL` to your running worker.
            </Paragraph>
          </YStack>

          <XStack gap="$4" flexWrap="wrap">
            <YStack flex={1} minWidth={340} gap="$3">
              <StatusCard status={status.status} error={status.error ?? null} baseUrl={workerUrl} />
              <GlassCard>
                <Text color="$text" fontFamily="$heading" fontSize={18}>
                  Prompt composer
                </Text>
                <InputField
                  value={prompt}
                  onChangeText={setPrompt}
                  placeholder="Describe the sound and optional /flags"
                  multiline
                  rows={4}
                />
                <XStack gap="$2" flexWrap="wrap">
                  <Pill variant="info">/duration {parsed.duration ?? '90-180'}</Pill>
                  <Pill variant="info">/model {parsed.model ?? 'default'}</Pill>
                  <Pill variant="info">/cfg {parsed.cfg ?? 'off|number'}</Pill>
                  <Pill variant="info">/seed {parsed.seed ?? 'optional'}</Pill>
                  {parsed.size ? <Pill variant="success">/{parsed.size}</Pill> : null}
                </XStack>
                <Button
                  size="$5"
                  backgroundColor="$accentAlt"
                  color="$background"
                  onPress={submit}
                  iconAfter={<Text fontFamily="$heading">→</Text>}
                >
                  Submit (stub)
                </Button>
                <Paragraph color="$textMuted">
                  Flags mirror the CLI: /duration 90-180, /model, /cfg 6.5|off, /seed, /motif, and
                  /small|/medium|/large. Job submission is stubbed until API wiring lands.
                </Paragraph>
              </GlassCard>
            </YStack>
            <YStack flex={1} minWidth={340} gap="$3">
              <GlassCard>
                <Text color="$text" fontFamily="$heading" fontSize={18}>
                  Jobs
                </Text>
                {jobs.length === 0 ? (
                  <Paragraph color="$textMuted">No jobs yet. Submit one to see it here.</Paragraph>
                ) : (
                  jobs.map((job) => (
                    <YStack key={job.id} padding="$3" borderRadius="$3" backgroundColor="rgba(255,255,255,0.02)" gap="$2">
                      <XStack ai="center" justifyContent="space-between">
                        <Text color="$text" fontFamily="$heading">
                          {job.id}
                        </Text>
                        <Pill variant={job.status === 'failed' ? 'danger' : job.status === 'done' ? 'success' : 'info'}>
                          {job.status}
                        </Pill>
                      </XStack>
                      <Paragraph color="$textMuted">{job.prompt}</Paragraph>
                    </YStack>
                  ))
                )}
              </GlassCard>
              <GlassCard>
                <Text color="$text" fontFamily="$heading" fontSize={18}>
                  Palette & mood
                </Text>
                <Paragraph color="$textMuted">
                  Matte midnight base with neon edges and soft glass panels. The heading uses
                  <Text fontFamily="$heading"> Alagard </Text>
                  to echo the CLI character while staying sharp.
                </Paragraph>
                <Separator marginVertical="$2" borderColor="$border" />
                <Paragraph color="$textMuted">
                  This bundle will ship via Electron for desktop distribution, then ride the same code
                  to iOS/Android with platform shims for audio and filesystem.
                </Paragraph>
              </GlassCard>
              <GlassCard>
                <Text color="$text" fontFamily="$heading" fontSize={18}>
                  Worker settings
                </Text>
                <InputField
                  value={workerUrl}
                  onChangeText={setWorkerUrl}
                  placeholder={defaultUrl}
                  prefix="URL"
                  multiline={false}
                  rows={1}
                />
                <Button
                  size="$4"
                  backgroundColor="$accent"
                  color="$background"
                  onPress={() => {
                    if (typeof window !== 'undefined') {
                      window.localStorage.setItem(STORAGE_KEY, workerUrl);
                    }
                  }}
                >
                  Save override
                </Button>
                <Paragraph color="$textMuted">
                  Overrides VITE_WORKER_URL/TIMBRE_WORKER_URL locally. Restart Electron after changing
                  to ensure the preload layer picks up new values when we add native calls.
                </Paragraph>
              </GlassCard>
            </YStack>
          </XStack>
        </YStack>
      </Theme>
    </TamaguiProvider>
  );
};

export default App;
