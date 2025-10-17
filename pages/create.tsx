import { useRouter } from 'next/router';
import { AppShell, Button, Container, FileInput, Stack, TextInput, Title, Loader, Text } from '@mantine/core';
import { useState } from 'react';
import { IconUpload } from '@tabler/icons-react';
import { uploadDocument, listPages } from '../lib/api';

export default function Create() {
  const [name, setName] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const handleNext = async () => {
    if (!file) return;
    
    setUploading(true);
    setError(null);
    
    try {
      // Upload to backend
      const result = await uploadDocument(file, name || file.name);
      
      // Store document info in localStorage
      const blobUrl = URL.createObjectURL(file);
      localStorage.setItem('pptUrl', blobUrl);
      localStorage.setItem('pptName', result.name);
      localStorage.setItem('docId', result.doc_id);
      localStorage.setItem('pageCount', result.page_count.toString());
      
      console.log('✅ Document uploaded:', result);
      
      // Preflight: ensure backend can see the document to avoid flicker
      try {
        await listPages(result.doc_id);
      } catch (e: any) {
        console.error('⚠️ Backend not ready for doc yet:', e);
        setError('The server is still preparing your document. Please try Next again.');
        return;
      }

      // Navigate to viewer (replace to avoid back button going to create again)
      router.replace('/viewer');
    } catch (err: any) {
      console.error('❌ Upload failed:', err);
      setError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
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
              disabled={uploading}
            />
            <FileInput
              label="Upload PDF/PPTX"
              placeholder="Choose file"
              leftSection={<IconUpload size={16} />}
              accept=".ppt,.pptx,.pdf"
              value={file}
              onChange={setFile}
              disabled={uploading}
            />
            {error && (
              <Text c="red" size="sm">{error}</Text>
            )}
            <Button 
              onClick={handleNext} 
              disabled={!file || uploading}
              leftSection={uploading ? <Loader size="xs" color="white" /> : undefined}
            >
              {uploading ? 'Uploading...' : 'Next'}
            </Button>
          </Stack>
        </Container>
      </AppShell.Main>
    </AppShell>
  );
}
