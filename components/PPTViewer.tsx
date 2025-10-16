import { Box, Button, Group, ScrollArea, Stack, Text } from '@mantine/core';
import { useState } from 'react';

type Slide = { id: number; title: string; imageUrl?: string };

export default function PPTViewer({ slides }: { slides: Slide[] }) {
  const [index, setIndex] = useState(0);
  const current = slides[index];

  const goto = (i: number) => setIndex(Math.max(0, Math.min(slides.length - 1, i)));

  return (
    <Stack gap="sm" style={{ height: '100%' }}>
      <Box flex="1" style={{ border: '1px solid #e9ecef', borderRadius: 8, background: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 320 }}>
        {/* Replace with real rendering (images or pdf canvas) */}
        <Text size="lg" fw={600}>Slide {current.id}: {current.title}</Text>
      </Box>

      <Group justify="space-between">
        <Group>
          <Button variant="light" onClick={() => goto(index - 1)} disabled={index === 0}>Prev</Button>
          <Button variant="light" onClick={() => goto(index + 1)} disabled={index === slides.length - 1}>Next</Button>
        </Group>
        <Text size="sm" c="dimmed">{index + 1} / {slides.length}</Text>
      </Group>

      <ScrollArea type="auto" style={{ borderTop: '1px solid #eee', paddingTop: 8 }}>
        <Group gap="xs" wrap="nowrap">
          {slides.map((s, i) => (
            <Box
              key={s.id}
              onClick={() => goto(i)}
              style={{
                width: 80, height: 50, borderRadius: 6, border: i === index ? '2px solid #1c7ed6' : '1px solid #e9ecef',
                background: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer'
              }}
            >
              <Text size="xs" c="dimmed">{s.id}</Text>
            </Box>
          ))}
        </Group>
      </ScrollArea>
    </Stack>
  );
}
