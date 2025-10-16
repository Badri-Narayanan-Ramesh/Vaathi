// components/chat/ChatPanel.tsx
import { useEffect, useState } from 'react';
import { Box, Button, Paper, ScrollArea, Stack, Text, Textarea } from '@mantine/core';

type Msg = { role: 'user' | 'assistant'; text: string };

export default function ChatPanel({ pageNum, numPages }: { pageNum: number; numPages: number }) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState('');

  // Update the explanation whenever slide changes
  useEffect(() => {
    const explanation = `Explanation for slide ${pageNum} (of ${numPages}). Replace this with your model output.`;
    setMessages((prev) => {
      // keep last assistant explanation at top by replacing any existing system explainer
      const withoutAuto = prev.filter(m => !(m.role === 'assistant' && m.text.startsWith('Explanation for slide')));
      return [{ role: 'assistant', text: explanation }, ...withoutAuto];
    });
  }, [pageNum, numPages]);

  const ask = async () => {
    const q = input.trim();
    if (!q) return;
    setInput('');
    setMessages(prev => [{ role: 'user', text: q }, ...prev]);
    // TODO: call your backend; below is a placeholder response
    const a = `Placeholder answer about slide ${pageNum}.`;
    setMessages(prev => [{ role: 'assistant', text: a }, ...prev]);
  };

  return (
    <Stack gap="sm" style={{ height: '100%' }}>
      <Paper withBorder p="md" radius="md" style={{ background: 'white', height: 380 }}>
        <Text fw={600} mb="xs">Explanation</Text>
        <ScrollArea style={{ height: 330 }}>
          <Stack gap="xs">
            {messages.map((m, i) => (
              <Box key={i} p="sm" style={{
                border: '1px solid #eef1f5',
                borderRadius: 8,
                background: m.role === 'assistant' ? '#fff' : '#f6f8fb'
              }}>
                <Text size="sm">{m.text}</Text>
              </Box>
            ))}
          </Stack>
        </ScrollArea>
      </Paper>

      <Textarea
        minRows={3}
        placeholder="Ask something about this slide…"
        value={input}
        onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setInput(e.currentTarget.value)}
      />
      <Button onClick={ask} aria-label="send">Send</Button>
    </Stack>
  );
}
