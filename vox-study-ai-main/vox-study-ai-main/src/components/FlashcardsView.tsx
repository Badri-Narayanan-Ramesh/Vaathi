import { useState, useEffect } from "react";
import { RotateCw, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { getFlashcards } from "@/lib/api";
import { toast } from "sonner";
import type { FlashcardItem } from "@/lib/types";

interface FlashcardsViewProps {
  docId: string;
  pageId: number;
}

export const FlashcardsView = ({ docId, pageId }: FlashcardsViewProps) => {
  const [flashcards, setFlashcards] = useState<FlashcardItem[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [loading, setLoading] = useState(false);

  const loadFlashcards = async () => {
    setLoading(true);
    try {
      const response = await getFlashcards(docId, pageId);
      setFlashcards(response.items);
      setCurrentIndex(0);
      setFlipped(false);
    } catch (error) {
      console.error('Flashcards error:', error);
      toast.error('Failed to load flashcards');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadFlashcards();
  }, [docId, pageId]);

  const currentCard = flashcards[currentIndex];

  const nextCard = () => {
    setFlipped(false);
    setCurrentIndex((prev) => (prev + 1) % flashcards.length);
  };

  const prevCard = () => {
    setFlipped(false);
    setCurrentIndex((prev) => (prev - 1 + flashcards.length) % flashcards.length);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (flashcards.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground mb-4">No flashcards generated yet</p>
        <Button onClick={loadFlashcards} variant="outline" className="gap-2">
          <RotateCw className="w-4 h-4" />
          Generate Flashcards
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <p className="text-sm text-muted-foreground">
          Card {currentIndex + 1} of {flashcards.length}
        </p>
        <Button variant="outline" size="sm" onClick={loadFlashcards} className="gap-2">
          <RotateCw className="w-4 h-4" />
          Regenerate
        </Button>
      </div>

      <div
        className="relative h-80 cursor-pointer perspective-1000"
        onClick={() => setFlipped(!flipped)}
      >
        <Card
          className={`absolute inset-0 p-8 flex items-center justify-center transition-all duration-500 transform ${
            flipped ? 'rotate-y-180' : ''
          }`}
          style={{
            backfaceVisibility: 'hidden',
            transformStyle: 'preserve-3d',
          }}
        >
          <div className="text-center">
            <p className="text-sm text-muted-foreground mb-2">Question</p>
            <p className="text-lg font-medium">{currentCard.q}</p>
          </div>
        </Card>

        <Card
          className={`absolute inset-0 p-8 flex items-center justify-center transition-all duration-500 transform bg-primary/5 ${
            flipped ? '' : 'rotate-y-180'
          }`}
          style={{
            backfaceVisibility: 'hidden',
            transformStyle: 'preserve-3d',
            transform: flipped ? 'rotateY(0deg)' : 'rotateY(180deg)',
          }}
        >
          <div className="text-center">
            <p className="text-sm text-muted-foreground mb-2">Answer</p>
            <p className="text-lg font-medium">{currentCard.a}</p>
          </div>
        </Card>
      </div>

      <div className="flex gap-2 justify-center">
        <Button variant="outline" onClick={prevCard} disabled={flashcards.length <= 1}>
          Previous
        </Button>
        <Button variant="outline" onClick={nextCard} disabled={flashcards.length <= 1}>
          Next
        </Button>
      </div>

      <p className="text-xs text-center text-muted-foreground">
        Click the card to flip it
      </p>
    </div>
  );
};
