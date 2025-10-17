import { Moon, Sun, Plus, Brain } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useStore } from "@/lib/store";
import { useNavigate } from "react-router-dom";

export const TopBar = () => {
  const { ui, toggleTheme, setSelectedModel } = useStore();
  const navigate = useNavigate();

  return (
    <header className="border-b bg-card shadow-sm sticky top-0 z-50">
      <div className="container flex items-center justify-between h-16 px-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-primary flex items-center justify-center">
            <span className="text-xl font-bold text-primary-foreground">AT</span>
          </div>
          <h1 className="text-xl font-bold">AI Tutor</h1>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 border rounded-md px-3 py-1.5 bg-background">
            <Brain className="w-4 h-4 text-muted-foreground" />
            <select
              value={ui.selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="text-sm bg-transparent border-none outline-none cursor-pointer model-select"
              title="Select AI Model"
            >
              <option value="gemini-1.5-flash">Gemini 1.5 Flash</option>
              <option value="gemini-1.5-pro">Gemini 1.5 Pro</option>
              <option value="qwen2.5:3b-instruct">Ollama: Qwen 2.5 3B</option>
            </select>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate("/create")}
            className="gap-2"
          >
            <Plus className="w-4 h-4" />
            Create New
          </Button>

          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            aria-label="Toggle theme"
          >
            {ui.theme === "light" ? (
              <Moon className="w-5 h-5" />
            ) : (
              <Sun className="w-5 h-5" />
            )}
          </Button>
        </div>
      </div>
    </header>
  );
};
