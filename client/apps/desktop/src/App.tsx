import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button, H4, Paragraph, Separator, Text, Theme, XStack, YStack } from 'tamagui';
import { createWorkerClient } from '@timbre/api';
import { resolveWorkerUrl } from '@timbre/config';
import { StatusCard } from '@timbre/ui';
import { TamaguiProvider } from 'tamagui';
import config from '../../tamagui.config';

type StatusShape = {
  status?: string;
};

const App: React.FC = () => {
  const baseUrl = useMemo(() => resolveWorkerUrl(), []);
  const client = useMemo(() => createWorkerClient({ baseUrl }), [baseUrl]);
  const [status, setStatus] = useState<StatusShape>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const next = await client.getStatus();
      setStatus(next);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to reach worker';
      setError(message);
      setStatus({});
    } finally {
      setLoading(false);
    }
  }, [client]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

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
            <YStack flex={1} minWidth={320} gap="$3">
              <StatusCard status={status.status} error={error} baseUrl={baseUrl} />
              <Button
                theme="blue"
                size="$5"
                disabled={loading}
                onPress={refresh}
                backgroundColor="$accentAlt"
                color="$background"
              >
                {loading ? 'Checking…' : 'Refresh status'}
              </Button>
            </YStack>
            <YStack
              flex={1}
              minWidth={320}
              gap="$3"
              padding="$4"
              backgroundColor="$panel"
              borderRadius="$3"
              borderWidth={1}
              borderColor="$border"
              shadowColor="rgba(124, 140, 255, 0.28)"
              shadowOffset={{ width: 0, height: 16 }}
              shadowRadius={36}
            >
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
                We will wrap this bundle in Electron for desktop distribution, then aim the same code
                at iOS/Android with platform shims for audio and filesystem.
              </Paragraph>
            </YStack>
          </XStack>
        </YStack>
      </Theme>
    </TamaguiProvider>
  );
};

export default App;
