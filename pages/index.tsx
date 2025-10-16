import { AppShell, Container, Group, SimpleGrid, Text, Title } from '@mantine/core';
import RecentCard from '../components/RecentCard';

export default function Home() {
  return (
    <AppShell header={{ height: 56 }} padding="md">
      <AppShell.Header style={{ borderBottom: '1px solid #eee', background: 'white' }}>
        <Group h="100%" px="md" justify="space-between">
          <Title order={4}>Hi 👋</Title>
          <Text c="dimmed" size="sm">Welcome back</Text>
        </Group>
      </AppShell.Header>

      <AppShell.Main>
        <Container size="lg">
          <Group mb="sm"><Title order={5}>Recent</Title></Group>
          <SimpleGrid cols={{ base: 1, sm: 2, md: 4 }} spacing="md">
            <RecentCard title="Create New" create />
            <RecentCard title="ML Notes" subtitle="Lecture 05 – Transformers" href="/viewer" />
            <RecentCard title="Quantum Notes" subtitle="Week 3 deck" href="/viewer" />
            <RecentCard title="Algorithms" subtitle="Greedy vs DP" href="/viewer" />
          </SimpleGrid>
        </Container>
      </AppShell.Main>
    </AppShell>
  );
}
