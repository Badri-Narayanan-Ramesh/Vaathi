import { Button, Stack, Textarea, Text, Paper } from '@mantine/core';
import { useState, useEffect } from 'react';

export default function ExplanationPanel({ slideIndex }: { slideIndex: number }) {
  const [answer, setAnswer] = useState('');
  const [question, setQuestion] = useState('');

  // Stub: fetch explanation for the current slide
  useEffect(() => {
    // TODO: replace with your model call
    setAnswer(`Auto-generated explanation for slide ${slideIndex + 1} will appear here.`);
  }, [slideIndex]);

  return (
    <Stack gap="sm" style={{ height: '100%' }}>
      <Paper withBorder p="md" radius="md" style={{ background: 'white' }}>
        <Text fw={600} mb="xs">Explanation</Text>
        <Text size="sm" c="dimmed">{answer}</Text>
      </Paper>

      <Textarea
        minRows={4}
        placeholder="Ask a question about this slide…"
        value={question}
        onChange={(e) => setQuestion(e.currentTarget.value)}
      />
      <Button onClick={() => { /* call your backend; show optimistic reply */ setQuestion(''); }}>Ask</Button>
    </Stack>
  );
}
