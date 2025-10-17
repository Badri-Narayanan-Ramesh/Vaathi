import { useEffect, useRef } from "react";
import { Box } from "@mantine/core";

type Props = {
  fileUrl: string;
  page: number;
  onNumPages?: (n: number) => void;
  scale?: number;
};

export default function PdfSlides({ fileUrl, page, onNumPages, scale = 1.4 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pdfRef = useRef<any>(null);
  const libRef = useRef<any>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (typeof window === "undefined") return;
      // Dynamically import pdf.js only on the client to avoid SSR issues
      const pdfjsLib = (await import("pdfjs-dist")).default || (await import("pdfjs-dist"));
      libRef.current = (pdfjsLib as any);
      try {
        (libRef.current as any).GlobalWorkerOptions.workerSrc =
          "https://unpkg.com/pdfjs-dist@4.10.38/build/pdf.worker.min.js";
      } catch {}

      const task = (libRef.current as any).getDocument(fileUrl);
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