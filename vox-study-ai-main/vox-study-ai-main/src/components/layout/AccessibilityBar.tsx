import { Type, Eye, Contrast } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useStore } from "@/lib/store";

export const AccessibilityBar = () => {
  const { ui, toggleDyslexiaFont, setTextScale, toggleHighContrast } = useStore();

  return (
    <div className="border-b bg-muted/30 px-6 py-2">
      <div className="container flex items-center gap-4 text-sm">
        <span className="text-muted-foreground font-medium">Accessibility:</span>

        <Button
          variant={ui.dyslexiaFont ? "default" : "ghost"}
          size="sm"
          onClick={toggleDyslexiaFont}
          className="gap-2"
        >
          <Type className="w-4 h-4" />
          Dyslexia Font
        </Button>

        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-muted-foreground" />
          <Select
            value={ui.textScale.toString()}
            onValueChange={(value) => setTextScale(parseInt(value) as 100 | 115 | 130)}
          >
            <SelectTrigger className="w-24 h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="100">100%</SelectItem>
              <SelectItem value="115">115%</SelectItem>
              <SelectItem value="130">130%</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Button
          variant={ui.highContrast ? "default" : "ghost"}
          size="sm"
          onClick={toggleHighContrast}
          className="gap-2"
        >
          <Contrast className="w-4 h-4" />
          High Contrast
        </Button>
      </div>
    </div>
  );
};
