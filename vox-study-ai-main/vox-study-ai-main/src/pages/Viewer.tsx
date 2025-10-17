import { useEffect, useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { PdfViewer } from "@/components/PdfViewer";
import { RightRail } from "@/components/RightRail";
import { useStore } from "@/lib/store";
import { listPages } from "@/lib/api";
import { toast } from "sonner";
import { Loader2 } from "lucide-react";

const Viewer = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const docId = searchParams.get("doc_id");
  const { doc, setDocument, setPage } = useStore();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!docId) {
      toast.error("No document ID provided");
      navigate("/");
      return;
    }

    const loadDocument = async () => {
      try {
        const response = await listPages(docId);
        
        setDocument({
          docId,
          name: `Document ${docId}`,
          pages: response.pages,
          pageId: response.pages[0]?.page_id || 1,
        });
      } catch (error) {
        console.error("Failed to load document:", error);
        toast.error("Failed to load document");
        navigate("/");
      } finally {
        setLoading(false);
      }
    };

    loadDocument();
  }, [docId]);

  const handlePageChange = (pageId: number) => {
    setPage(pageId);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="w-12 h-12 animate-spin text-primary" />
      </div>
    );
  }

  if (!doc || !docId) {
    return null;
  }

  return (
    <div className="flex h-[calc(100vh-8rem)]">
      {/* Left: PDF Viewer */}
      <div className="flex-1">
        <PdfViewer
          docId={docId}
          pageId={doc.pageId}
          totalPages={doc.pages.length}
          onPageChange={handlePageChange}
        />
      </div>

      {/* Right: Tabs Panel */}
      <div className="w-[480px]">
        <RightRail
          docId={docId}
          pageId={doc.pageId}
          onJumpToPage={handlePageChange}
        />
      </div>
    </div>
  );
};

export default Viewer;
