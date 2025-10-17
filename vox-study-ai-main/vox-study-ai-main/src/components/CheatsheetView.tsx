import { useState, useEffect } from "react";
import { Download, Copy, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { getCheatsheet } from "@/lib/api";
import { toast } from "sonner";

interface CheatsheetViewProps {
  docId: string;
  pageId: number;
}

export const CheatsheetView = ({ docId, pageId }: CheatsheetViewProps) => {
  const [content, setContent] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const loadCheatsheet = async () => {
    setLoading(true);
    try {
      const response = await getCheatsheet(docId, pageId);
      setContent(response.content);
    } catch (error) {
      console.error('Cheatsheet error:', error);
      toast.error('Failed to load cheatsheet');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadCheatsheet();
  }, [docId, pageId]);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    toast.success('Copied to clipboard');
  };

  const handleDownload = () => {
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cheatsheet-page-${pageId}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success('Cheatsheet downloaded');
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!content) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">No cheatsheet available</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <Button variant="outline" size="sm" onClick={handleCopy} className="gap-2">
          <Copy className="w-4 h-4" />
          Copy
        </Button>
        <Button variant="outline" size="sm" onClick={handleDownload} className="gap-2">
          <Download className="w-4 h-4" />
          Download .md
        </Button>
      </div>

      <Card className="p-6">
        <div className="prose prose-sm max-w-none dark:prose-invert">
          <pre className="whitespace-pre-wrap font-sans text-sm text-foreground">
            {content}
          </pre>
        </div>
      </Card>
    </div>
  );
};
