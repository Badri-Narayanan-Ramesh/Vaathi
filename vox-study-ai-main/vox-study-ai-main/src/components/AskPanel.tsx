import { useState, useRef } from "react";
import { Send, Mic, MicOff, Volume2, Loader2, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { askQuestion, tts, stt } from "@/lib/api";
import { useStore } from "@/lib/store";
import { toast } from "sonner";

interface AskPanelProps {
  docId: string;
  onJumpToPage: (pageId: number) => void;
}

export const AskPanel = ({ docId, onJumpToPage }: AskPanelProps) => {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [generatingVoice, setGeneratingVoice] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const { setAudioBlob, setAudioPlaying } = useStore();

  const handleAsk = async () => {
    if (!question.trim()) {
      toast.error('Please enter a question');
      return;
    }

    setLoading(true);
    try {
      const response = await askQuestion(docId, question);
      setAnswer(response.answer);
      setCitations(response.citations.map(c => c.page_id));
    } catch (error) {
      console.error('Ask error:', error);
      toast.error('Failed to get answer');
    } finally {
      setLoading(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const file = new File([blob], 'recording.webm', { type: 'audio/webm' });
        
        try {
          const response = await stt(file);
          setQuestion(response.text);
          toast.success('Voice transcribed');
        } catch (error) {
          console.error('STT error:', error);
          toast.error('Failed to transcribe audio');
        }

        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setRecording(true);
      toast.info('Recording...');
    } catch (error) {
      console.error('Recording error:', error);
      toast.error('Failed to access microphone');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const handleVoiceOut = async () => {
    if (!answer) return;
    
    setGeneratingVoice(true);
    try {
      const blob = await tts(answer);
      setAudioBlob(blob);
      setAudioPlaying(true);
      toast.success('Playing answer');
    } catch (error) {
      console.error('TTS error:', error);
      toast.error('Failed to generate voice');
    } finally {
      setGeneratingVoice(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <Input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about this document..."
          onKeyPress={(e) => e.key === 'Enter' && handleAsk()}
          className="flex-1"
        />
        
        <Button
          variant="outline"
          size="icon"
          onClick={recording ? stopRecording : startRecording}
          className={recording ? 'bg-destructive text-destructive-foreground' : ''}
        >
          {recording ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
        </Button>

        <Button onClick={handleAsk} disabled={loading} className="gap-2">
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </Button>
      </div>

      {answer && (
        <Card className="p-6">
          <div className="prose prose-sm max-w-none dark:prose-invert mb-4">
            <p className="text-foreground leading-relaxed">{answer}</p>
          </div>

          {citations.length > 0 && (
            <div className="mb-4">
              <p className="text-sm font-medium mb-2">Citations:</p>
              <div className="flex flex-wrap gap-2">
                {citations.map((pageId) => (
                  <Button
                    key={pageId}
                    variant="outline"
                    size="sm"
                    onClick={() => onJumpToPage(pageId)}
                    className="gap-2"
                  >
                    Jump to Slide {pageId}
                    <ArrowRight className="w-3 h-3" />
                  </Button>
                ))}
              </div>
            </div>
          )}

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
        </Card>
      )}
    </div>
  );
};
