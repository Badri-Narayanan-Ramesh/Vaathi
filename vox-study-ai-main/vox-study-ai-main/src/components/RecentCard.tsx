import { FileText, Clock } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";

interface RecentCardProps {
  docId: string;
  name: string;
  pageCount: number;
  lastVisited: number;
}

export const RecentCard = ({ docId, name, pageCount, lastVisited }: RecentCardProps) => {
  const navigate = useNavigate();
  const formattedDate = new Date(lastVisited).toLocaleDateString();

  return (
    <Card className="p-6 hover:shadow-md transition-shadow cursor-pointer" onClick={() => navigate(`/viewer?doc_id=${docId}`)}>
      <div className="flex items-start gap-4">
        <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
          <FileText className="w-6 h-6 text-primary" />
        </div>

        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-lg mb-1 truncate">{name}</h3>
          <p className="text-sm text-muted-foreground mb-3">
            {pageCount} {pageCount === 1 ? 'page' : 'pages'}
          </p>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Clock className="w-3 h-3" />
            <span>Last visited {formattedDate}</span>
          </div>
        </div>

        <Button variant="outline" size="sm" onClick={(e) => {
          e.stopPropagation();
          navigate(`/viewer?doc_id=${docId}`);
        }}>
          Open
        </Button>
      </div>
    </Card>
  );
};
