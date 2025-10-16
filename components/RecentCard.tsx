import { Card, Group, Text, ActionIcon } from '@mantine/core';
import { IconUpload, IconChevronRight } from '@tabler/icons-react';
import Link from 'next/link';

type Props = {
  title: string;
  subtitle?: string;
  href?: string;        // for existing pages
  create?: boolean;     // if true => “Create New”
};

export default function RecentCard({ title, subtitle, href, create }: Props) {
  const content = (
    <Card shadow="xs" padding="lg" withBorder style={{ cursor: 'pointer', height: 140, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
      <div>
        <Group gap="sm">
          {create ? <IconUpload size={18} /> : null}
          <Text fw={600}>{title}</Text>
        </Group>
        {subtitle ? <Text c="dimmed" size="sm" mt="xs" lineClamp={2}>{subtitle}</Text> : null}
      </div>
      <Group justify="space-between">
        <div />
        <ActionIcon variant="subtle" aria-label="open"><IconChevronRight /></ActionIcon>
      </Group>
    </Card>
  );

  if (create) return <Link href="/create" style={{ textDecoration: 'none' }}>{content}</Link>;
  if (href) return <Link href={href} style={{ textDecoration: 'none' }}>{content}</Link>;
  return content;
}
