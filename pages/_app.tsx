import type { AppProps } from 'next/app';
import Head from 'next/head';
import { MantineProvider } from '@mantine/core';

// ✅ ADD THESE TWO LINES
import '@mantine/core/styles.css';
import '@mantine/notifications/styles.css';

import '../styles/globals.css';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <title>PPT Explainer</title>
        <meta name="viewport" content="minimum-scale=1, initial-scale=1, width=device-width" />
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet" />
      </Head>

      <MantineProvider
        defaultColorScheme="light"
        theme={{
          fontFamily: 'Roboto, system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif',
          headings: { fontFamily: 'Roboto, sans-serif' },
          defaultRadius: 'md',
        }}
      >
        <Component {...pageProps} />
      </MantineProvider>
    </>
  );
}
