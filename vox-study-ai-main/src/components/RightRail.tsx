import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ExplanationPanel } from "./ExplanationPanel";
import { AskPanel } from "./AskPanel";
import { FlashcardsView } from "./FlashcardsView";
import { QuizView } from "./QuizView";
import { CheatsheetView } from "./CheatsheetView";
import { useStore } from "@/lib/store";
import { useEffect, useState } from "react";

interface RightRailProps {
  docId: string;
  pageId: number;
  onJumpToPage: (pageId: number) => void;
}

export const RightRail = ({ docId, pageId, onJumpToPage }: RightRailProps) => {
  const { ui } = useStore();
  const [modelKey, setModelKey] = useState(ui.selectedModel);

  useEffect(() => {
    setModelKey(ui.selectedModel + '-' + Date.now()); // force remount on model change
  }, [ui.selectedModel]);

  return (
    <div className="h-full flex flex-col bg-card border-l overflow-hidden">
      <Tabs defaultValue="explanation" className="flex-1 flex flex-col overflow-hidden" key={modelKey}>
        <TabsList className="w-full justify-start rounded-none border-b bg-muted/30 p-0 h-auto flex-shrink-0">
          <TabsTrigger value="explanation" className="rounded-none data-[state=active]:shadow-none px-6 py-3">
            Explanation
          </TabsTrigger>
          <TabsTrigger value="ask" className="rounded-none data-[state=active]:shadow-none px-6 py-3">
            Ask
          </TabsTrigger>
          <TabsTrigger value="flashcards" className="rounded-none data-[state=active]:shadow-none px-6 py-3">
            Flashcards
          </TabsTrigger>
          <TabsTrigger value="quiz" className="rounded-none data-[state=active]:shadow-none px-6 py-3">
            Quiz
          </TabsTrigger>
          <TabsTrigger value="cheatsheet" className="rounded-none data-[state=active]:shadow-none px-6 py-3">
            Cheatsheet
          </TabsTrigger>
        </TabsList>

        <div className="flex-1 overflow-y-auto p-6 min-h-0">
          <TabsContent value="explanation" className="mt-0">
            <ExplanationPanel docId={docId} pageId={pageId} model={ui.selectedModel} />
          </TabsContent>

          <TabsContent value="ask" className="mt-0">
            <AskPanel docId={docId} pageId={pageId} onJumpToPage={onJumpToPage} model={ui.selectedModel} />
          </TabsContent>

          <TabsContent value="flashcards" className="mt-0">
            <FlashcardsView docId={docId} pageId={pageId} model={ui.selectedModel} />
          </TabsContent>

          <TabsContent value="quiz" className="mt-0">
            <QuizView docId={docId} pageId={pageId} model={ui.selectedModel} />
          </TabsContent>

          <TabsContent value="cheatsheet" className="mt-0">
            <CheatsheetView docId={docId} pageId={pageId} model={ui.selectedModel} />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
};
