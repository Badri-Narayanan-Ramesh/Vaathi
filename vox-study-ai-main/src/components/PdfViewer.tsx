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
  const [zoom, setZoom] = useState(50);
  const [inputPage, setInputPage] = useState((pageId + 1).toString()); // Display as 1-based
  const [imageError, setImageError] = useState(false);

  const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
  const imageUrl = `${API_BASE}/pages/${docId}/pages/${pageId}/image`;

  useEffect(() => {
    setInputPage((pageId + 1).toString()); // Display as 1-based
    setImageError(false); // Reset error when page changes
  }, [pageId]);

  const handlePrevious = () => {
    if (pageId > 0) { // 0-based indexing
      onPageChange(pageId - 1);
    }
  };

  const handleNext = () => {
    if (pageId < totalPages - 1) { // 0-based indexing
      onPageChange(pageId + 1);
    }
  };

  const handlePageInput = (e: React.FormEvent) => {
    e.preventDefault();
    const displayPage = parseInt(inputPage); // User enters 1-based
    const backendPage = displayPage - 1; // Convert to 0-based
    if (backendPage >= 0 && backendPage < totalPages) {
      onPageChange(backendPage);
    } else {
      setInputPage((pageId + 1).toString());
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
              disabled={pageId <= 0}
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
              disabled={pageId >= totalPages - 1}
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
        <div
          className="bg-white shadow-lg"
          style={{
            width: `${zoom}%`,
            maxWidth: '100%',
          }}
        >
          {imageError ? (
            <Card className="p-8 text-center text-muted-foreground">
              <p className="text-lg font-medium mb-2">Failed to load page</p>
              <p className="text-sm">Page {pageId + 1} could not be rendered</p>
              <p className="text-xs mt-4">Document ID: {docId}</p>
            </Card>
          ) : (
            <img
              src={imageUrl}
              alt={`Page ${pageId + 1}`}
              className="w-full h-auto"
              onError={() => setImageError(true)}
              onLoad={() => console.log('✅ Page image loaded:', pageId + 1)}
            />
          )}
        </div>
      </div>
    </div>
  );
};
