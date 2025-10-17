import { useState } from "react";
import { Upload, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useNavigate } from "react-router-dom";
import { uploadFile } from "@/lib/api";
import { useStore } from "@/lib/store";
import { toast } from "sonner";

const Create = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const navigate = useNavigate();
  const { addRecentDoc } = useStore();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      const validTypes = ['application/pdf', 'application/vnd.ms-powerpoint', 'application/vnd.openxmlformats-officedocument.presentationml.presentation'];
      if (!validTypes.includes(selectedFile.type)) {
        toast.error('Please upload a PDF or PPTX file');
        return;
      }
      setFile(selectedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      toast.error('Please select a file');
      return;
    }

    setUploading(true);
    console.log('🔵 Starting upload for file:', file.name);

    try {
      const response = await uploadFile(file, file.name);
      console.log('✅ Upload response:', response);
      
      addRecentDoc({
        docId: response.doc_id,
        name: response.name,
        pageCount: response.page_count,
      });
      
      toast.success('Document uploaded successfully!');
      console.log('🔵 Navigating to viewer with doc_id:', response.doc_id);
      navigate(`/viewer?doc_id=${response.doc_id}`);
    } catch (error) {
      console.error('❌ Upload error:', error);
      toast.error('Failed to upload document. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <Card className="w-full max-w-2xl p-8">
        <div className="text-center mb-8">
          <div className="w-20 h-20 rounded-2xl bg-gradient-primary flex items-center justify-center mx-auto mb-4 shadow-lg">
            <FileText className="w-10 h-10 text-primary-foreground" />
          </div>
          <h1 className="text-3xl font-bold mb-2">Upload Your Document</h1>
          <p className="text-muted-foreground">
            Upload a PDF or PowerPoint presentation to get started
          </p>
        </div>

        <div className="space-y-6">
          <div>
            <Label htmlFor="file" className="text-base font-medium">
              Choose File
            </Label>
            <div className="mt-2">
              <Input
                id="file"
                type="file"
                accept=".pdf,.ppt,.pptx"
                onChange={handleFileChange}
                className="cursor-pointer"
              />
            </div>
            {file && (
              <p className="mt-2 text-sm text-muted-foreground flex items-center gap-2">
                <FileText className="w-4 h-4" />
                {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
              </p>
            )}
          </div>

          <Button
            onClick={handleUpload}
            disabled={!file || uploading}
            className="w-full gap-2"
            size="lg"
          >
            <Upload className="w-5 h-5" />
            {uploading ? 'Uploading...' : 'Upload & Continue'}
          </Button>
        </div>

        <div className="mt-8 p-4 bg-muted rounded-lg">
          <h3 className="font-medium mb-2">Supported Formats:</h3>
          <ul className="text-sm text-muted-foreground space-y-1">
            <li>• PDF Documents (.pdf)</li>
            <li>• PowerPoint Presentations (.ppt, .pptx)</li>
          </ul>
        </div>
      </Card>
    </div>
  );
};

export default Create;
