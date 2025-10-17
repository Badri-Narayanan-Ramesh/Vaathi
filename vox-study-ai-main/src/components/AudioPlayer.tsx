import { useEffect, useRef } from "react";
import { Play, Pause } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useStore } from "@/lib/store";
import { Card } from "@/components/ui/card";

export const AudioPlayer = () => {
  const { audio, setAudioRate, setAudioPlaying } = useStore();
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (!audioRef.current || !audio.blob) return;

    const url = URL.createObjectURL(audio.blob);
    audioRef.current.src = url;
    audioRef.current.playbackRate = audio.rate;

    if (audio.isPlaying) {
      audioRef.current.play();
    }

    return () => URL.revokeObjectURL(url);
  }, [audio.blob]);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.playbackRate = audio.rate;
    }
  }, [audio.rate]);

  const togglePlay = () => {
    if (!audioRef.current) return;

    if (audio.isPlaying) {
      audioRef.current.pause();
      setAudioPlaying(false);
    } else {
      audioRef.current.play();
      setAudioPlaying(true);
    }
  };

  if (!audio.blob) return null;

  return (
    <Card className="fixed bottom-6 right-6 p-4 shadow-lg z-50 w-80">
      <div className="flex items-center gap-3">
        <Button
          size="icon"
          variant="default"
          onClick={togglePlay}
          className="rounded-full shadow-md"
        >
          {audio.isPlaying ? (
            <Pause className="w-5 h-5" />
          ) : (
            <Play className="w-5 h-5" />
          )}
        </Button>

        <div className="flex-1">
          <p className="text-sm font-medium">Playing audio</p>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs text-muted-foreground">Speed:</span>
            <Select
              value={audio.rate.toString()}
              onValueChange={(value) => setAudioRate(parseFloat(value))}
            >
              <SelectTrigger className="w-20 h-7 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1.0">1.0x</SelectItem>
                <SelectItem value="1.25">1.25x</SelectItem>
                <SelectItem value="1.5">1.5x</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      <audio
        ref={audioRef}
        onEnded={() => setAudioPlaying(false)}
        onPlay={() => setAudioPlaying(true)}
        onPause={() => setAudioPlaying(false)}
      />
    </Card>
  );
};
