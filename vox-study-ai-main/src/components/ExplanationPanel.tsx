import { useState } from "react";
import { Sparkles, Volume2, Copy, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { getExplanation, tts } from "@/lib/api";
import { useStore } from "@/lib/store";
import { toast } from "sonner";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ExplanationPanelProps {
  docId: string;
  pageId: number;
  model?: string;
}

export const ExplanationPanel = ({ docId, pageId, model }: ExplanationPanelProps) => {
  const [explanation, setExplanation] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [generatingVoice, setGeneratingVoice] = useState(false);
  const { setAudioBlob, setAudioPlaying } = useStore();

  const handleExplain = async () => {
    setLoading(true);
    try {
      const response = await getExplanation(docId, pageId, model);
      setExplanation(response.explanation);
    } catch (error) {
      console.error('Explanation error:', error);
      toast.error('Failed to generate explanation');
    } finally {
      setLoading(false);
    }
  };

  const handleVoiceOut = async () => {
    if (!explanation) return;
    
    setGeneratingVoice(true);
    try {
      const blob = await tts(explanation);
      setAudioBlob(blob);
      setAudioPlaying(true);
      toast.success('Playing explanation');
    } catch (error) {
      console.error('TTS error:', error);
      toast.error('Failed to generate voice');
    } finally {
      setGeneratingVoice(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(explanation);
    toast.success('Copied to clipboard');
  };

  return (
    <div className="space-y-4">
      <Button
        onClick={handleExplain}
        disabled={loading}
        className="w-full gap-2"
      >
        {loading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Generating...
          </>
        ) : (
          <>
            <Sparkles className="w-4 h-4" />
            Explain This Page
          </>
        )}
      </Button>

      {explanation && (
        <Card className="p-6">
          <div className="prose prose-sm max-w-none dark:prose-invert mb-4 break-words prose-pre:whitespace-pre-wrap prose-pre:break-words 
            prose-headings:font-semibold prose-headings:text-foreground
            prose-p:text-foreground prose-p:leading-relaxed prose-p:my-2
            prose-ul:my-2 prose-li:text-foreground prose-li:my-1
            prose-strong:text-foreground prose-strong:font-semibold
            [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {explanation}
            </ReactMarkdown>
          </div>

          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleVoiceOut}
              disabled={generatingVoice}
              className="gap-2"
            >
              <Volume2 className="w-4 h-4" />
              {generatingVoice ? 'Generating...' : 'Voice Out'}
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={handleCopy}
              className="gap-2"
            >
              <Copy className="w-4 h-4" />
              Copy
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
};
