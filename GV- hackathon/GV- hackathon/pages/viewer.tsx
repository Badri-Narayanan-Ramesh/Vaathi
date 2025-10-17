import { useEffect, useMemo, useRef, useState } from "react";
import { AppShell, Container, Group, Title, Text, Button, Stack, Card, ScrollArea, Textarea, ActionIcon, Grid, Divider, Badge, Loader } from "@mantine/core";
import { useRouter } from "next/router";
import dynamic from "next/dynamic";
import { IconSend, IconChevronLeft, IconChevronRight, IconFileText, IconLink } from "@tabler/icons-react";
import { explainPage, askQuestion, listPages } from "../lib/api";

const PdfSlides = dynamic(() => import("../components/PdfSlides"), { ssr: false });

type ChatMsg = { role: "user" | "assistant"; text: string; loading?: boolean };

export default function Viewer() {
  const router = useRouter();
  const [url, setUrl] = useState<string | null>(null);
  const [name, setName] = useState<string>("Untitled");
  const [docId, setDocId] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  const [page, setPage] = useState(1);
  const [numPages, setNumPages] = useState(1);

  const [draft, setDraft] = useState("");
  const [chat, setChat] = useState<ChatMsg[]>([
    { role: "assistant", text: "Ask about any slide, I'll explain or cross-reference notes & refs." },
  ]);
  
  const [loadingExplanation, setLoadingExplanation] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const init = async () => {
      const storedUrl = localStorage.getItem("pptUrl");
      const storedName = localStorage.getItem("pptName");
      const storedDocId = localStorage.getItem("docId");
      const storedPageCount = localStorage.getItem("pageCount");

      if (storedName) setName(storedName);
      if (storedUrl) setUrl(storedUrl);
      if (storedDocId) setDocId(storedDocId);
      if (storedPageCount) setNumPages(parseInt(storedPageCount, 10));

      // Preflight: verify doc exists on backend; if missing, clear stale state but don't redirect
      if (storedDocId) {
        try {
          await listPages(storedDocId);
        } catch (e) {
          // Clear stale session; viewer will show "No file loaded" with a button
          localStorage.removeItem('docId');
          localStorage.removeItem('pptUrl');
          localStorage.removeItem('pptName');
          localStorage.removeItem('pageCount');
          setDocId(null);
          setUrl(null);
          setNumPages(1);
          setName('Untitled');
        }
      }

      setReady(true);
    };
    init();
  }, []);

  // Auto-fetch explanation when page changes
  useEffect(() => {
    if (!docId || !ready) return;
    
    const fetchExplanation = async () => {
      setLoadingExplanation(true);
      try {
        const result = await explainPage(docId, page - 1); // 0-indexed in backend
        
        // Add explanation as assistant message
        setChat(prev => [
          ...prev,
          { 
            role: "assistant", 
            text: `📄 **Slide ${page} Summary:**\n\n${result.explanation}` 
          }
        ]);
        
        console.log('✅ Explanation loaded for page', page);
      } catch (error: any) {
        console.error('❌ Failed to load explanation:', error);
        const msg = String(error?.message || error);
        // If the backend restarted, in-memory doc store is cleared -> doc not found
        if (msg.includes('Document not found') || msg.includes('404')) {
          setChat(prev => [
            ...prev,
            {
              role: 'assistant',
              text: `⚠️ This session expired or the server restarted, so the document isn't available anymore. Please re-upload your file.`,
            },
          ]);
          try {
            localStorage.removeItem('docId');
            localStorage.removeItem('pptUrl');
            localStorage.removeItem('pptName');
            localStorage.removeItem('pageCount');
          } catch {}
          setDocId(null);
          setUrl(null);
          setNumPages(1);
          setName('Untitled');
        } else {
          setChat(prev => [
            ...prev,
            { 
              role: "assistant", 
              text: `⚠️ Could not load explanation for slide ${page}. Error: ${msg}` 
            }
          ]);
        }
      } finally {
        setLoadingExplanation(false);
      }
    };

    fetchExplanation();
  }, [page, docId, ready]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [chat]);

  const onSend = async () => {
    const trimmed = draft.trim();
    if (!trimmed || !docId) return;

    // Add user message
    setChat(prev => [...prev, { role: "user", text: trimmed }]);
    setDraft("");

    // Add loading message
    setChat(prev => [...prev, { role: "assistant", text: "Thinking...", loading: true }]);

    try {
      // Call backend Q&A
      const result = await askQuestion(docId, trimmed, 3);
      
      // Replace loading message with actual answer
      setChat(prev => {
        const newChat = [...prev];
        newChat[newChat.length - 1] = { 
          role: "assistant", 
          text: result.answer,
          loading: false
        };
        return newChat;
      });
      
      console.log('✅ Answer received');
    } catch (error: any) {
      console.error('❌ Q&A failed:', error);
      const msg = String(error?.message || error);

      // If doc isn't found (e.g., server restarted), clear session and redirect
      if (msg.includes('Document not found') || msg.includes('404')) {
        setChat(prev => {
          const newChat = [...prev];
          newChat[newChat.length - 1] = {
            role: 'assistant',
            text: `⚠️ This session expired or the server restarted. Please re-upload your file to continue.`,
            loading: false,
          };
          return newChat;
        });
        try {
          localStorage.removeItem('docId');
          localStorage.removeItem('pptUrl');
          localStorage.removeItem('pptName');
          localStorage.removeItem('pageCount');
        } catch {}
        setDocId(null);
        setUrl(null);
        setNumPages(1);
        setName('Untitled');
      } else {
        // Replace loading message with error
        setChat(prev => {
          const newChat = [...prev];
          newChat[newChat.length - 1] = { 
            role: "assistant", 
            text: `⚠️ Sorry, I couldn't answer that. Error: ${msg}`,
            loading: false
          };
          return newChat;
        });
      }
    }
  };

  const pageButtons = useMemo(() => {
    const arr: number[] = [];
    for (let i = 1; i <= numPages; i++) arr.push(i);
    return arr;
  }, [numPages]);

  const goPrev = () => setPage((p) => Math.max(1, p - 1));
  const goNext = () => setPage((p) => Math.min(numPages, p + 1));

  const goCreate = () => router.push("/create");

  return (
    <AppShell header={{ height: 56 }} padding="md">
      <AppShell.Header className="border-b">
        <Container size="xl" className="h-full flex items-center">
          <Group justify="space-between" w="100%">
            <Group gap="sm">
              <Badge variant="light" size="lg">Viewer</Badge>
              <Title order={4} style={{ fontWeight: 600 }}>{name}</Title>
              <Text c="dimmed" size="sm">{numPages} pages</Text>
            </Group>
            <Group>
              <Button variant="light" onClick={goCreate}>Upload another</Button>
            </Group>
          </Group>
        </Container>
      </AppShell.Header>

      <AppShell.Main>
        {!ready ? (
          <Container size="xl"><Text>Loading…</Text></Container>
        ) : !url ? (
          <Container size="xl">
            <Stack align="center" py="xl">
              <Text size="lg">No file loaded.</Text>
              <Button onClick={goCreate}>Go to Create</Button>
            </Stack>
          </Container>
        ) : (
          <Container size="xl" pt="md" pb="lg">
            <Grid gutter="md">
              {/* LEFT: Slides */}
              <Grid.Col span={{ base: 12, lg: 6 }}>
                <Card withBorder radius="lg" p="sm" style={{ height: "calc(100dvh - 140px)" }}>
                  <Stack gap="xs" style={{ height: "100%" }}>
                    <Group justify="space-between" px="sm" py={4}>
                      <Text fw={600}>Slides</Text>
                      <Group gap="xs">
                        <Button size="xs" variant="subtle" leftSection={<IconChevronLeft size={16} />} onClick={goPrev} disabled={page === 1}>Prev</Button>
                        <Button size="xs" variant="subtle" rightSection={<IconChevronRight size={16} />} onClick={goNext} disabled={page === numPages}>Next</Button>
                      </Group>
                    </Group>
                    <Divider />
                    <ScrollArea style={{ flex: 1 }} type="hover">
                      <PdfSlides
                        fileUrl={url!}
                        page={page}
                        onNumPages={setNumPages}      // make PdfSlides call this when it knows the page count
                      />
                    </ScrollArea>
                    <Divider />
                    <ScrollArea.Autosize mah={90}>
                      <Group wrap="wrap" gap={6} px="sm" py={6}>
                        {pageButtons.map((i) => (
                          <Button key={i} size="compact-sm" variant={i === page ? "filled" : "light"} onClick={() => setPage(i)}>
                            {i}
                          </Button>
                        ))}
                      </Group>
                    </ScrollArea.Autosize>
                  </Stack>
                </Card>
              </Grid.Col>

              {/* CENTER: Explanation chat */}
              <Grid.Col span={{ base: 12, lg: 4 }}>
                
                <Card withBorder radius="lg" p="sm" style={{ height: "calc(100dvh - 140px)" }}>
                  <Stack gap="xs" style={{ height: "100%" }}>
                    <Group justify="space-between" px="sm" py={4}>
                      <Text fw={600}>Explanation</Text>
                      <ActionIcon size="lg" variant="filled" onClick={onSend} aria-label="Mic">
                          <IconSend size={16} />
                        </ActionIcon>
                      <Text c="dimmed" size="sm">Slide {page}/{numPages}</Text>
                    </Group>
                    <Divider />
                    <ScrollArea style={{ flex: 1 }} type="hover">
                      <Stack px="sm" py="xs">
                        {chat.map((m, idx) => (
                          <Card
                            key={idx}
                            shadow="xs"
                            radius="md"
                            withBorder
                            padding="sm"
                            style={{ alignSelf: m.role === "user" ? "flex-end" : "flex-start", maxWidth: "90%" }}
                          >
                            <Text size="xs" c="dimmed" mb={4}>{m.role === "user" ? "You" : "AI"}</Text>
                            <Text size="sm">{m.text}</Text>
                          </Card>
                        ))}
                      </Stack>
                    </ScrollArea>
                    <Divider />
                    <Stack gap="xs">
                      <Textarea
                        placeholder="Ask a question about this slide…"
                        autosize
                        minRows={2}
                        value={draft}
                        onChange={(e) => setDraft(e.currentTarget.value)}
                      />
                      <Group justify="space-between">
                        <Text size="xs" c="dimmed">Tip: reference slide numbers (e.g., “On slide {page}, what does RoBERTa change?”)</Text>
                        <ActionIcon size="lg" variant="filled" onClick={onSend} aria-label="Send">
                          <IconSend size={16} />
                        </ActionIcon>
                      </Group>

                          <Button size="lg" variant="light" >
                      <Text size="sm">Voice Mode</Text>
                          </Button>
                    </Stack>
                  </Stack>
                </Card>
              </Grid.Col>

              {/* RIGHT: Notes + Refs */}
              <Grid.Col span={{ base: 12, lg: 2 }}>
                <Stack gap="md" style={{ height: "calc(100dvh - 140px)" }}>
                  <Card withBorder radius="lg" p="sm" style={{ flex: 1, display: "flex", flexDirection: "column" }}>
                    <Group gap={6} mb={6}><IconFileText size={16} /><Text fw={600}>Tools</Text></Group>
                    <ScrollArea style={{ flex: 1 }} type="hover">
                      <Stack gap="xs">
                      <Button size="compact-sm" variant="light" >
                      <Text size="sm">Flashcards</Text>
                          </Button>

                          <Button size="compact-sm" variant="light" >
                      <Text size="sm">Quizzes</Text>
                          </Button>

                          <Button size="compact-sm" variant="light" >
                      <Text size="sm">Notes</Text>
                          </Button>
                      
                      </Stack>
                    </ScrollArea>
                  </Card>

                  <Card withBorder radius="lg" p="sm" style={{ flex: 1, display: "flex", flexDirection: "column" }}>
                    <Group gap={6} mb={6}><IconLink size={16} /><Text fw={600}>Placeholder</Text></Group>
                    <ScrollArea style={{ flex: 1 }} type="hover">
                      <Stack gap="xs">
                        <Text size="sm">• WOOO</Text>
                        <Text size="sm">• DUMMY </Text>
                        <Text size="sm">• sdfdsf</Text>
                      </Stack>
                    </ScrollArea>
                  </Card>
                </Stack>
              </Grid.Col>
            </Grid>
          </Container>
        )}
      </AppShell.Main>
    </AppShell>
  );
}
