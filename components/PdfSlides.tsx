import { useEffect, useRef } from "react";
import { Box } from "@mantine/core";
import * as pdfjsLib from "pdfjs-dist";

if (typeof window !== "undefined") {
  // Tell pdf.js to use an ESM worker that webpack can bundle
  (pdfjsLib as any).GlobalWorkerOptions.workerPort = new Worker(
    new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url),
    { type: "module" }
  );
}

type Props = {
  fileUrl: string;
  page: number;
  onNumPages?: (n: number) => void;
  scale?: number;
};

export default function PdfSlides({ fileUrl, page, onNumPages, scale = 1.4 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pdfRef = useRef<pdfjsLib.PDFDocumentProxy | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const task = pdfjsLib.getDocument(fileUrl);
      const doc = await task.promise;
      if (cancelled) return;
      pdfRef.current = doc;
      onNumPages?.(doc.numPages);
    })().catch(err => {
      console.error("PDF load error:", err);
      onNumPages?.(0);
    });
    return () => { cancelled = true; };
  }, [fileUrl, onNumPages]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const doc = pdfRef.current;
      const canvas = canvasRef.current;
      if (!doc || !canvas) return;
      const pageObj = await doc.getPage(page);
      if (cancelled) return;

      const dpr = window.devicePixelRatio || 1;
      const viewport = pageObj.getViewport({ scale });
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      canvas.width = Math.floor(viewport.width * dpr);
      canvas.height = Math.floor(viewport.height * dpr);
      canvas.style.width = `${viewport.width}px`;
      canvas.style.height = `${viewport.height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      await pageObj.render({ canvasContext: ctx, viewport }).promise;
    })();
    return () => { cancelled = true; };
  }, [page, scale]);

  return (
    <Box style={{ border: "1px solid #e9ecef", borderRadius: 8, background: "white" }}>
      <canvas ref={canvasRef} style={{ width: "100%", height: "auto", display: "block" }} />
    </Box>
  );
}