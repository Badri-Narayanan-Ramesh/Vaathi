import { ArrowRight, BookOpen, Brain, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { useStore } from "@/lib/store";
import { RecentCard } from "@/components/RecentCard";

const Home = () => {
  const navigate = useNavigate();
  const { recentDocs } = useStore();

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-primary py-24">
        <div className="absolute inset-0 bg-grid-white/[0.05] bg-[size:20px_20px]" />
        <div className="container relative z-10 px-6">
          <div className="max-w-3xl mx-auto text-center">
            <h1 className="text-5xl font-bold mb-6 text-primary-foreground">
              Learn Smarter with AI
            </h1>
            <p className="text-xl mb-8 text-primary-foreground/90">
              Transform your PDFs and presentations into interactive learning experiences with AI-powered explanations, flashcards, and quizzes.
            </p>
            <Button
              size="lg"
              variant="secondary"
              className="gap-2 shadow-lg"
              onClick={() => navigate("/create")}
            >
              Get Started
              <ArrowRight className="w-5 h-5" />
            </Button>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-16 bg-background">
        <div className="container px-6">
          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            <div className="text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-secondary flex items-center justify-center mx-auto mb-4 shadow-md">
                <Brain className="w-8 h-8 text-secondary-foreground" />
              </div>
              <h3 className="text-lg font-semibold mb-2">AI Explanations</h3>
              <p className="text-muted-foreground">
                Get instant, conversational explanations for any slide or concept
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-accent flex items-center justify-center mx-auto mb-4 shadow-md">
                <BookOpen className="w-8 h-8 text-accent-foreground" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Smart Flashcards</h3>
              <p className="text-muted-foreground">
                Auto-generated flashcards to help you memorize key concepts
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-primary flex items-center justify-center mx-auto mb-4 shadow-md">
                <Zap className="w-8 h-8 text-primary-foreground" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Interactive Quizzes</h3>
              <p className="text-muted-foreground">
                Test your knowledge with AI-generated multiple choice questions
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Recent Documents */}
      {recentDocs.length > 0 && (
        <section className="py-16 bg-muted/30">
          <div className="container px-6">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-3xl font-bold mb-8">Recent Documents</h2>
              <div className="grid gap-4">
                {recentDocs.map((doc) => (
                  <RecentCard key={doc.docId} {...doc} />
                ))}
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
};

export default Home;
