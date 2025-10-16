// components/right-rail/RightRail.tsx
import { Card, Stack, Text } from '@mantine/core';

function MiniCard({ title }: { title: string }) {
  return (
    <Card shadow="xs" padding="sm" withBorder style={{ background: 'white' }}>
      <div style={{ width: '100%', height: 56, borderRadius: 6, border: '1px solid #e9ecef', marginBottom: 8 }} />
      <Text size="xs" lineClamp={2}>{title}</Text>
    </Card>
  );
}

export default function RightRail() {
  return (
    <Stack gap="sm">
      <MiniCard title="Key concept" />
      <MiniCard title="Related example" />
      <MiniCard title="Reference link" />
      <MiniCard title="Extra note" />
    </Stack>
  );
}
