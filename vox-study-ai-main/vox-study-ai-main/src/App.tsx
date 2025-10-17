import { useEffect } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useStore } from "@/lib/store";
import { TopBar } from "@/components/layout/TopBar";
import { AccessibilityBar } from "@/components/layout/AccessibilityBar";
import { AudioPlayer } from "@/components/AudioPlayer";
import Home from "./pages/Home";
import Create from "./pages/Create";
import Viewer from "./pages/Viewer";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const AppContent = () => {
  const { ui } = useStore();

  useEffect(() => {
    const root = document.documentElement;
    
    if (ui.theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }

    if (ui.dyslexiaFont) {
      root.classList.add('dyslexia-font');
    } else {
      root.classList.remove('dyslexia-font');
    }

    if (ui.highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }

    root.classList.remove('text-scale-100', 'text-scale-115', 'text-scale-130');
    root.classList.add(`text-scale-${ui.textScale}`);
  }, [ui]);

  return (
    <div className="min-h-screen bg-background">
      <TopBar />
      <AccessibilityBar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/create" element={<Create />} />
        <Route path="/viewer" element={<Viewer />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
      <AudioPlayer />
    </div>
  );
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AppContent />
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
