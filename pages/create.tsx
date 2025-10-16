import { useRouter } from 'next/router';
import { AppShell, Button, Container, FileInput, Stack, TextInput, Title } from '@mantine/core';
import { useState } from 'react';
import { IconUpload } from '@tabler/icons-react';

export default function Create() {
  const [name, setName] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const router = useRouter();

  const handleNext = async () => {
    // TODO: upload to your API; store an id in localStorage or query string
    // For now we just persist minimal state locally
    if (!file) return;
    const blobUrl = URL.createObjectURL(file);
    localStorage.setItem('pptUrl', blobUrl);
    localStorage.setItem('pptName', name || file.name);
    router.push('/viewer');
  };

  return (
    <AppShell header={{ height: 56 }} padding="md">
      <AppShell.Header style={{ borderBottom: '1px solid #eee', background: 'white' }}>
        <Title order={4} pl="md">Create New</Title>
      </AppShell.Header>

      <AppShell.Main>
        <Container size="sm">
          <Stack gap="md" mt="md">
            <TextInput
              label="Name"
              placeholder="e.g., Quantum Notes – Week 2"
              value={name}
              onChange={(e) => setName(e.currentTarget.value)}
            />
            <FileInput
              label="Upload PPTX"
              placeholder="Choose file"
              leftSection={<IconUpload size={16} />}
              accept=".ppt,.pptx,.pdf"
              value={file}
              onChange={setFile}
            />
            <Button onClick={handleNext} disabled={!file}>Next</Button>
          </Stack>
        </Container>
      </AppShell.Main>
    </AppShell>
  );
}
