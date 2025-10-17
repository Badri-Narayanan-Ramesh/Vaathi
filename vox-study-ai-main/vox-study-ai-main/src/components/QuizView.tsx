import { useState, useEffect } from "react";
import { RotateCw, Loader2, Check, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { getQuiz } from "@/lib/api";
import { toast } from "sonner";
import type { QuizItem } from "@/lib/types";

interface QuizViewProps {
  docId: string;
  pageId: number;
}

export const QuizView = ({ docId, pageId }: QuizViewProps) => {
  const [quiz, setQuiz] = useState<QuizItem[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<string>("");
  const [showResult, setShowResult] = useState(false);
  const [loading, setLoading] = useState(false);

  const loadQuiz = async () => {
    setLoading(true);
    try {
      const response = await getQuiz(docId, pageId);
      setQuiz(response.items);
      setCurrentIndex(0);
      setSelectedAnswer("");
      setShowResult(false);
    } catch (error) {
      console.error('Quiz error:', error);
      toast.error('Failed to load quiz');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadQuiz();
  }, [docId, pageId]);

  const currentQuestion = quiz[currentIndex];

  const handleSubmit = () => {
    if (!selectedAnswer) {
      toast.error('Please select an answer');
      return;
    }
    setShowResult(true);
  };

  const nextQuestion = () => {
    setSelectedAnswer("");
    setShowResult(false);
    setCurrentIndex((prev) => (prev + 1) % quiz.length);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (quiz.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground mb-4">No quiz generated yet</p>
        <Button onClick={loadQuiz} variant="outline" className="gap-2">
          <RotateCw className="w-4 h-4" />
          Generate Quiz
        </Button>
      </div>
    );
  }

  const isCorrect = selectedAnswer === currentQuestion.answer;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <p className="text-sm text-muted-foreground">
          Question {currentIndex + 1} of {quiz.length}
        </p>
        <Button variant="outline" size="sm" onClick={loadQuiz} className="gap-2">
          <RotateCw className="w-4 h-4" />
          Regenerate
        </Button>
      </div>

      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-6">{currentQuestion.question}</h3>

        <RadioGroup value={selectedAnswer} onValueChange={setSelectedAnswer}>
          <div className="space-y-3">
            {currentQuestion.options.map((option, index) => {
              const optionLetter = String.fromCharCode(65 + index);
              const isThisCorrect = option === currentQuestion.answer;
              const isSelected = selectedAnswer === option;

              return (
                <div
                  key={index}
                  className={`flex items-center space-x-3 p-4 rounded-lg border-2 transition-colors ${
                    showResult && isThisCorrect
                      ? 'border-success bg-success/10'
                      : showResult && isSelected && !isThisCorrect
                      ? 'border-destructive bg-destructive/10'
                      : isSelected
                      ? 'border-primary bg-primary/5'
                      : 'border-border'
                  }`}
                >
                  <RadioGroupItem
                    value={option}
                    id={`option-${index}`}
                    disabled={showResult}
                  />
                  <Label
                    htmlFor={`option-${index}`}
                    className="flex-1 cursor-pointer font-normal"
                  >
                    <span className="font-semibold mr-2">{optionLetter}.</span>
                    {option}
                  </Label>
                  {showResult && isThisCorrect && (
                    <Check className="w-5 h-5 text-success" />
                  )}
                  {showResult && isSelected && !isThisCorrect && (
                    <X className="w-5 h-5 text-destructive" />
                  )}
                </div>
              );
            })}
          </div>
        </RadioGroup>

        {showResult && (
          <div
            className={`mt-6 p-4 rounded-lg ${
              isCorrect ? 'bg-success/10 border border-success' : 'bg-destructive/10 border border-destructive'
            }`}
          >
            <p className="font-medium">
              {isCorrect ? '✓ Correct!' : '✗ Incorrect'}
            </p>
            {!isCorrect && (
              <p className="text-sm mt-1">
                The correct answer is: {currentQuestion.answer}
              </p>
            )}
          </div>
        )}

        <div className="flex gap-2 mt-6">
          {!showResult ? (
            <Button onClick={handleSubmit} className="flex-1">
              Submit Answer
            </Button>
          ) : (
            <Button onClick={nextQuestion} className="flex-1">
              Next Question
            </Button>
          )}
        </div>
      </Card>
    </div>
  );
};
