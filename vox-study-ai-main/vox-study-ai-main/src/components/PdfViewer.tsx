import { useState, useEffect } from "react";
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";

interface PdfViewerProps {
  docId: string;
  pageId: number;
  totalPages: number;
  onPageChange: (pageId: number) => void;
}

export const PdfViewer = ({ docId, pageId, totalPages, onPageChange }: PdfViewerProps) => {
  const [zoom, setZoom] = useState(100);
  const [inputPage, setInputPage] = useState(pageId.toString());

  useEffect(() => {
    setInputPage(pageId.toString());
  }, [pageId]);

  const handlePrevious = () => {
    if (pageId > 1) {
      onPageChange(pageId - 1);
    }
  };

  const handleNext = () => {
    if (pageId < totalPages) {
      onPageChange(pageId + 1);
    }
  };

  const handlePageInput = (e: React.FormEvent) => {
    e.preventDefault();
    const newPage = parseInt(inputPage);
    if (newPage >= 1 && newPage <= totalPages) {
      onPageChange(newPage);
    } else {
      setInputPage(pageId.toString());
    }
  };

  const handleZoomIn = () => {
    setZoom((prev) => Math.min(prev + 10, 200));
  };

  const handleZoomOut = () => {
    setZoom((prev) => Math.max(prev - 10, 50));
  };

  return (
    <div className="flex flex-col h-full bg-muted/30">
      {/* Controls */}
      <div className="border-b bg-card p-4">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={handlePrevious}
              disabled={pageId <= 1}
            >
              <ChevronLeft className="w-4 h-4" />
            </Button>

            <form onSubmit={handlePageInput} className="flex items-center gap-2">
              <Input
                type="number"
                value={inputPage}
                onChange={(e) => setInputPage(e.target.value)}
                className="w-16 text-center"
                min={1}
                max={totalPages}
              />
              <span className="text-sm text-muted-foreground">/ {totalPages}</span>
            </form>

            <Button
              variant="outline"
              size="icon"
              onClick={handleNext}
              disabled={pageId >= totalPages}
            >
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="outline" size="icon" onClick={handleZoomOut}>
              <ZoomOut className="w-4 h-4" />
            </Button>
            <span className="text-sm text-muted-foreground w-12 text-center">
              {zoom}%
            </span>
            <Button variant="outline" size="icon" onClick={handleZoomIn}>
              <ZoomIn className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* PDF Display Area */}
      <div className="flex-1 overflow-auto p-8 flex items-center justify-center">
        <Card
          className="bg-white shadow-lg"
          style={{
            width: `${zoom}%`,
            minHeight: '600px',
          }}
        >
          <div className="aspect-[8.5/11] flex items-center justify-center p-8 text-muted-foreground">
            <div className="text-center">
              <p className="text-lg font-medium mb-2">Page {pageId}</p>
              <p className="text-sm">
                PDF rendering will be handled by your backend
              </p>
              <p className="text-xs mt-4">
                Document ID: {docId}
              </p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};
