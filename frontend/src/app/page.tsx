"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Shield,
  AlertTriangle,
  CheckCircle,
  Search,
  History,
  BarChart3,
  Newspaper,
  Link,
  FileText,
  TrendingUp,
  TrendingDown,
  Zap,
  Info,
  Moon,
  Sun,
  ChevronRight,
  Sparkles,
  Target,
  Clock,
  Download,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Line,
  Area,
  AreaChart,
  Legend,
} from "recharts";
import { fetchBackend } from "@/lib/backend";

const APP_NAME = "Fake News Detector";

// Types
type PredictionLabel = "Real" | "Fake";
type BackendStatus = "checking" | "connected" | "disconnected";

interface PredictionResult {
  prediction: PredictionLabel;
  confidence: number;
  keywords: [string, number][];
  input_text?: string;
  analyzed_at?: string;
}

interface HistoryItem {
  id: number;
  input_text: string;
  prediction: PredictionLabel;
  confidence: number;
  timestamp: string;
}

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
}

interface BackendHealthResponse {
  status?: string;
  model_loaded?: boolean;
  timestamp?: string;
}

interface BackendMetricsResponse {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  model_loaded?: boolean;
}

interface BackendHistoryEntry {
  id?: number;
  input_text?: string;
  text?: string;
  prediction?: string;
  confidence?: number;
  timestamp?: string;
}

interface BackendHistoryResponse {
  items?: BackendHistoryEntry[];
}

interface TrainingStats {
  total_samples: number;
  real_samples: number;
  fake_samples: number;
  verified_samples?: number;
  unverified_predictions?: number;
  min_samples_for_retraining: number;
}

interface RetrainStatus {
  should_retrain: boolean;
  current_samples: number;
  min_samples_required: number;
  verified_samples?: number;
  unverified_predictions?: number;
  data_balance?: {
    real_ratio: number;
    fake_ratio: number;
  };
}

interface BackendKeyword {
  word?: string;
  keyword?: string;
  importance?: number;
  score?: number;
  type?: "real" | "fake";
}

interface BackendPredictionResponse {
  success?: boolean;
  prediction?: string;
  confidence?: number;
  keywords?: BackendKeyword[];
  error?: string;
  detail?: string;
  timestamp?: string;
}

const emptyMetrics: ModelMetrics = {
  accuracy: 0,
  precision: 0,
  recall: 0,
  f1_score: 0,
};

const normalizePredictionLabel = (
  prediction?: string | null,
): PredictionLabel => (prediction?.toUpperCase() === "REAL" ? "Real" : "Fake");

const isRealPrediction = (prediction?: string | null) =>
  normalizePredictionLabel(prediction) === "Real";

const formatPercentage = (value?: number | null, digits = 2) =>
  value == null ? "--" : `${(value * 100).toFixed(digits)}%`;

const isKeywordEntry = (
  keyword: [string, number] | null,
): keyword is [string, number] => keyword !== null;

const getKeywordScore = (keyword: BackendKeyword) => {
  const baseScore =
    typeof keyword.importance === "number"
      ? keyword.importance
      : typeof keyword.score === "number"
        ? Math.abs(keyword.score)
        : 0;

  if (keyword.type === "fake") {
    return -Math.abs(baseScore);
  }

  if (keyword.type === "real") {
    return Math.abs(baseScore);
  }

  return typeof keyword.score === "number" ? keyword.score : baseScore;
};

const mapPredictionKeywords = (keywords: BackendKeyword[] = []) =>
  keywords
    .map((keyword) => {
      const word = (keyword.word ?? keyword.keyword ?? "").trim();
      if (!word) {
        return null;
      }

      return [word, getKeywordScore(keyword)] as [string, number];
    })
    .filter(isKeywordEntry);

const mapHistoryItem = (
  item: BackendHistoryEntry,
  index: number,
): HistoryItem => ({
  id: typeof item.id === "number" ? item.id : Date.now() + index,
  input_text: item.input_text ?? item.text ?? "Untitled analysis",
  prediction: normalizePredictionLabel(item.prediction),
  confidence: typeof item.confidence === "number" ? item.confidence : 0,
  timestamp: item.timestamp ?? new Date().toISOString(),
});

// Animation variants
const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
};

const scaleIn = {
  initial: { opacity: 0, scale: 0.9 },
  animate: { opacity: 1, scale: 1 },
  exit: { opacity: 0, scale: 0.9 },
};

export default function FakeNewsDetector() {
  const [activeTab, setActiveTab] = useState("predict");
  const [inputMode, setInputMode] = useState<"text" | "url">("text");
  const [inputText, setInputText] = useState("");
  const [inputUrl, setInputUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [isRetraining, setIsRetraining] = useState(false);
  const [retrainStatus, setRetrainStatus] = useState<RetrainStatus | null>(null);
  const [trainingStats, setTrainingStats] = useState<TrainingStats | null>(null);

  // Apply dark mode
  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);

  const loadBackendData = useCallback(async () => {
    try {
      const healthResponse = await fetchBackend("/health");

      if (!healthResponse.ok) {
        throw new Error("Backend health check failed");
      }

      const healthData = (await healthResponse.json()) as BackendHealthResponse;

      if (healthData.status !== "healthy") {
        throw new Error("Backend is not healthy");
      }

      setBackendStatus("connected");
      setModelLoaded(Boolean(healthData.model_loaded));

      const [metricsResponse, historyResponse] = await Promise.all([
        fetchBackend("/metrics"),
        fetchBackend("/history?limit=20"),
      ]);

      if (metricsResponse.ok) {
        const metricsData =
          (await metricsResponse.json()) as BackendMetricsResponse;
        const isModelLoaded = Boolean(metricsData.model_loaded);
        setModelLoaded(isModelLoaded);
        setMetrics(
          isModelLoaded
            ? {
                accuracy: metricsData.accuracy ?? 0,
                precision: metricsData.precision ?? 0,
                recall: metricsData.recall ?? 0,
                f1_score: metricsData.f1_score ?? 0,
              }
            : null,
        );
      }

      if (historyResponse.ok) {
        const historyData =
          (await historyResponse.json()) as BackendHistoryResponse;
        setHistory((historyData.items ?? []).map(mapHistoryItem));
      }
    } catch (error) {
      console.error("Failed to load backend data:", error);
      setBackendStatus("disconnected");
      setModelLoaded(false);
      setMetrics(null);
      setHistory([]);
    }
  }, []);

  const loadTrainingStats = useCallback(async () => {
    try {
      const response = await fetchBackend("/training/stats");
      if (response.ok) {
        const data = await response.json();
        setTrainingStats(data);
      }
    } catch (error) {
      console.error("Failed to load training stats:", error);
    }
  }, []);

  const loadRetrainStatus = useCallback(async () => {
    try {
      const response = await fetchBackend("/retrain/status");
      if (response.ok) {
        const data = await response.json();
        setRetrainStatus(data);
      }
    } catch (error) {
      console.error("Failed to load retrain status:", error);
    }
  }, []);

  const handleRetrain = useCallback(async () => {
    setIsRetraining(true);
    try {
      const response = await fetchBackend("/retrain", {
        method: "POST",
      });
      const data = await response.json();

      if (response.ok && data.success) {
        // Reload data after retraining
        await loadBackendData();
        await loadTrainingStats();
        await loadRetrainStatus();
        alert("Model retrained successfully!");
      } else {
        alert(`Retraining failed: ${data.error || "Unknown error"}`);
      }
    } catch (error) {
      console.error("Retraining failed:", error);
      alert("Failed to retrain model");
    } finally {
      setIsRetraining(false);
    }
  }, [loadBackendData, loadTrainingStats, loadRetrainStatus]);

  useEffect(() => {
    void loadBackendData();
    void loadTrainingStats();
    void loadRetrainStatus();
  }, [loadBackendData, loadTrainingStats, loadRetrainStatus]);

  // Handle prediction
  const handlePredict = useCallback(async () => {
    const text = inputMode === "text" ? inputText : inputUrl;
    if (!text.trim()) return;

    setIsLoading(true);
    setResult(null);
    setPredictionError(null);

    try {
      let response;
      if (inputMode === "text") {
        response = await fetchBackend("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
      } else {
        response = await fetchBackend("/predict-url", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: text }),
        });
      }

      const data = (await response
        .json()
        .catch(() => null)) as BackendPredictionResponse | null;

      if (!response.ok) {
        const errorMessage =
          data?.error ?? data?.detail ?? "Prediction service error";

        setBackendStatus(
          /connect|unreachable|network|failed to fetch/i.test(errorMessage)
            ? "disconnected"
            : "connected",
        );

        throw new Error(errorMessage);
      }

      if (!data || data.success === false || data.error || !data.prediction) {
        throw new Error(
          data?.error ??
            data?.detail ??
            "Prediction service returned an invalid response",
        );
      }

      const prediction: PredictionResult = {
        prediction: normalizePredictionLabel(data.prediction),
        confidence: data.confidence || 0,
        keywords: mapPredictionKeywords(data.keywords),
        input_text: text,
        analyzed_at: data.timestamp ?? new Date().toISOString(),
      };

      setBackendStatus("connected");
      setResult(prediction);

      const newItem: HistoryItem = {
        id: Date.now(),
        input_text: text.slice(0, 100) + (text.length > 100 ? "..." : ""),
        prediction: prediction.prediction,
        confidence: prediction.confidence,
        timestamp: new Date().toISOString(),
      };
      setHistory((prev) => [newItem, ...prev]);
    } catch (error) {
      console.error("Prediction failed:", error);
      setPredictionError(
        error instanceof Error
          ? error.message
          : "Failed to connect to the backend prediction service",
      );
    } finally {
      setIsLoading(false);
    }
  }, [inputText, inputUrl, inputMode]);

  const handleDownloadReport = useCallback(() => {
    if (!result) {
      return;
    }

    const analyzedAt = result.analyzed_at ?? new Date().toISOString();
    const reportLines = [
      APP_NAME,
      "Analysis Report",
      "",
      `Generated: ${analyzedAt}`,
      `Prediction: ${result.prediction}`,
      `Confidence: ${result.confidence}%`,
      `Inference Mode: ${modelLoaded ? "Trained model" : "Fallback mode"}`,
      "",
      "Key Contributing Words:",
      ...(result.keywords.length > 0
        ? result.keywords.map(
            ([word, score], index) =>
              `${index + 1}. ${word} (${score > 0 ? "Real" : "Fake"}, impact ${Math.abs(score).toFixed(3)})`,
          )
        : ["No keyword explanation available"]),
      "",
      "Analyzed Content:",
      result.input_text ?? "No source text available",
      "",
    ];

    const blob = new Blob([reportLines.join("\n")], {
      type: "text/plain;charset=utf-8",
    });
    const link = document.createElement("a");
    const safeTimestamp = analyzedAt.replace(/[:.]/g, "-");

    link.href = URL.createObjectURL(blob);
    link.download = `fake-news-detector-report-${safeTimestamp}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
  }, [modelLoaded, result]);

  // Filter history
  const filteredHistory = history.filter(
    (item) =>
      item.input_text.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.prediction.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  // Chart data
  const activeMetrics = metrics ?? emptyMetrics;
  const metricsData = [
    {
      name: "Accuracy",
      value: activeMetrics.accuracy * 100,
      fill: "hsl(var(--chart-1))",
    },
    {
      name: "Precision",
      value: activeMetrics.precision * 100,
      fill: "hsl(var(--chart-2))",
    },
    {
      name: "Recall",
      value: activeMetrics.recall * 100,
      fill: "hsl(var(--chart-3))",
    },
    {
      name: "F1 Score",
      value: activeMetrics.f1_score * 100,
      fill: "hsl(var(--chart-4))",
    },
  ];

  const realPredictionCount = history.filter((item) =>
    isRealPrediction(item.prediction),
  ).length;
  const fakePredictionCount = history.length - realPredictionCount;

  const trendBuckets = new Map<
    string,
    { name: string; confidenceTotal: number; predictions: number }
  >();

  history.forEach((item) => {
    const date = new Date(item.timestamp);
    if (Number.isNaN(date.getTime())) {
      return;
    }

    const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`;
    const monthLabel = date.toLocaleString("en-US", { month: "short" });
    const bucket = trendBuckets.get(monthKey) ?? {
      name: monthLabel,
      confidenceTotal: 0,
      predictions: 0,
    };

    bucket.confidenceTotal += item.confidence;
    bucket.predictions += 1;
    trendBuckets.set(monthKey, bucket);
  });

  const trendData = Array.from(trendBuckets.entries())
    .sort(([leftKey], [rightKey]) => leftKey.localeCompare(rightKey))
    .slice(-6)
    .map(([, bucket]) => ({
      name: bucket.name,
      confidence:
        bucket.predictions > 0
          ? Number((bucket.confidenceTotal / bucket.predictions).toFixed(1))
          : 0,
      predictions: bucket.predictions,
    }));

  const pieData = [
    {
      name: "Real News",
      value: realPredictionCount,
      color: "hsl(142, 76%, 36%)",
    },
    {
      name: "Fake News",
      value: fakePredictionCount,
      color: "hsl(0, 84%, 60%)",
    },
  ].filter((item) => item.value > 0);

  const metricCards = [
    {
      label: "Accuracy",
      value: formatPercentage(metrics?.accuracy),
      icon: Target,
      accent: "bg-emerald-100 dark:bg-emerald-900/30",
      accentText: "text-emerald-600 dark:text-emerald-400",
    },
    {
      label: "Precision",
      value: formatPercentage(metrics?.precision),
      icon: Zap,
      accent: "bg-amber-100 dark:bg-amber-900/30",
      accentText: "text-amber-600 dark:text-amber-400",
    },
    {
      label: "Recall",
      value: formatPercentage(metrics?.recall),
      icon: TrendingUp,
      accent: "bg-sky-100 dark:bg-sky-900/30",
      accentText: "text-sky-600 dark:text-sky-400",
    },
    {
      label: "F1 Score",
      value: formatPercentage(metrics?.f1_score),
      icon: BarChart3,
      accent: "bg-fuchsia-100 dark:bg-fuchsia-900/30",
      accentText: "text-fuchsia-600 dark:text-fuchsia-400",
    },
  ];

  const backendBadgeConfig: Record<
    BackendStatus,
    {
      label: string;
      className: string;
      dotClassName: string;
    }
  > = {
    checking: {
      label: "Backend: checking",
      className:
        "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-300",
      dotClassName: "bg-amber-500",
    },
    connected: {
      label: "Backend: connected",
      className:
        "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-300",
      dotClassName: "bg-emerald-500",
    },
    disconnected: {
      label: "Backend: disconnected",
      className:
        "border-red-200 bg-red-50 text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300",
      dotClassName: "bg-red-500",
    },
  };

  const metricsBadgeLabel =
    modelLoaded && metrics
      ? `${formatPercentage(metrics.accuracy, 1)} Accuracy`
      : backendStatus === "connected"
        ? "Model not trained"
        : "Metrics unavailable";

  return (
    <TooltipProvider>
      <div
        className={`min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950 transition-colors duration-300`}
      >
        {/* Header */}
        <header className="sticky top-0 z-50 backdrop-blur-xl bg-white/80 dark:bg-slate-900/80 border-b border-slate-200 dark:border-slate-800">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <motion.div
                className="flex items-center gap-3"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
              >
                <div className="relative">
                  <Shield className="w-10 h-10 text-emerald-600 dark:text-emerald-400" />
                  <motion.div
                    className="absolute -top-1 -right-1 w-4 h-4 bg-amber-400 rounded-full"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                </div>
                <div>
                  <h1 className="text-xl font-bold bg-gradient-to-r from-emerald-600 to-teal-600 dark:from-emerald-400 dark:to-teal-400 bg-clip-text text-transparent">
                    {APP_NAME}
                  </h1>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Powered by TF-IDF & Machine Learning
                  </p>
                </div>
              </motion.div>

              <div className="flex items-center gap-2 sm:gap-4">
                <Badge
                  variant="outline"
                  className={`gap-2 px-3 py-1 ${backendBadgeConfig[backendStatus].className}`}
                >
                  <span
                    className={`h-2 w-2 rounded-full ${backendBadgeConfig[backendStatus].dotClassName}`}
                  />
                  <span className="text-xs font-medium">
                    {backendBadgeConfig[backendStatus].label}
                  </span>
                </Badge>
                <Badge
                  variant="outline"
                  className="hidden sm:flex gap-1.5 px-3 py-1"
                >
                  <Zap className="w-3.5 h-3.5 text-amber-500" />
                  <span className="text-xs font-medium">
                    {metricsBadgeLabel}
                  </span>
                </Badge>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setDarkMode(!darkMode)}
                  className="relative"
                >
                  {darkMode ? (
                    <Sun className="w-5 h-5" />
                  ) : (
                    <Moon className="w-5 h-5" />
                  )}
                </Button>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="space-y-6"
          >
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <TabsList className="grid w-full grid-cols-5 lg:w-auto lg:inline-grid bg-white dark:bg-slate-800 shadow-sm border border-slate-200 dark:border-slate-700">
                <TabsTrigger value="predict" className="gap-2">
                  <Search className="w-4 h-4" />
                  <span className="hidden sm:inline">Predict</span>
                </TabsTrigger>
                <TabsTrigger value="dashboard" className="gap-2">
                  <BarChart3 className="w-4 h-4" />
                  <span className="hidden sm:inline">Dashboard</span>
                </TabsTrigger>
                <TabsTrigger value="history" className="gap-2">
                  <History className="w-4 h-4" />
                  <span className="hidden sm:inline">History</span>
                </TabsTrigger>
                <TabsTrigger value="retrain" className="gap-2">
                  <Zap className="w-4 h-4" />
                  <span className="hidden sm:inline">Retrain</span>
                </TabsTrigger>
                <TabsTrigger value="about" className="gap-2">
                  <Info className="w-4 h-4" />
                  <span className="hidden sm:inline">About</span>
                </TabsTrigger>
              </TabsList>
            </motion.div>

            {/* Predict Tab */}
            <TabsContent value="predict" className="space-y-6">
              <div className="grid lg:grid-cols-2 gap-6">
                {/* Input Section */}
                <motion.div
                  variants={fadeInUp}
                  initial="initial"
                  animate="animate"
                  transition={{ delay: 0.2 }}
                >
                  <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-slate-200 dark:border-slate-700 shadow-xl">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Newspaper className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                        Analyze News Article
                      </CardTitle>
                      <CardDescription>
                        Enter news text or provide a URL to detect fake news
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {/* Mode Toggle */}
                      <div className="flex gap-2 p-1 bg-slate-100 dark:bg-slate-700 rounded-lg">
                        <Button
                          variant={inputMode === "text" ? "default" : "ghost"}
                          size="sm"
                          onClick={() => setInputMode("text")}
                          className="flex-1 gap-2"
                        >
                          <FileText className="w-4 h-4" />
                          Text Input
                        </Button>
                        <Button
                          variant={inputMode === "url" ? "default" : "ghost"}
                          size="sm"
                          onClick={() => setInputMode("url")}
                          className="flex-1 gap-2"
                        >
                          <Link className="w-4 h-4" />
                          URL Input
                        </Button>
                      </div>

                      {/* Text Input */}
                      <AnimatePresence mode="wait">
                        {inputMode === "text" ? (
                          <motion.div
                            key="text"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 20 }}
                          >
                            <Textarea
                              placeholder="Paste or type news article text here..."
                              value={inputText}
                              onChange={(e) => setInputText(e.target.value)}
                              className="min-h-[200px] resize-none bg-white dark:bg-slate-900"
                            />
                            <div className="flex justify-between mt-2 text-xs text-slate-500">
                              <span>Min 10 characters required</span>
                              <span>{inputText.length} characters</span>
                            </div>
                          </motion.div>
                        ) : (
                          <motion.div
                            key="url"
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                          >
                            <Input
                              type="url"
                              placeholder="https://example.com/news-article"
                              value={inputUrl}
                              onChange={(e) => setInputUrl(e.target.value)}
                              className="h-12 bg-white dark:bg-slate-900"
                            />
                            <p className="mt-2 text-xs text-slate-500">
                              We&apos;ll extract the article content
                              automatically
                            </p>
                          </motion.div>
                        )}
                      </AnimatePresence>

                      {/* Quick Examples */}
                      <div className="space-y-2">
                        <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                          Quick Examples:
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {[
                            {
                              label: "Real News",
                              text: "The Federal Reserve announced today that interest rates will remain unchanged following their quarterly meeting, citing stable economic indicators.",
                            },
                            {
                              label: "Fake News",
                              text: "SHOCKING: Scientists discover that drinking coffee makes you immortal! Government has been hiding this secret for decades!",
                            },
                          ].map((example, i) => (
                            <Button
                              key={i}
                              variant="outline"
                              size="sm"
                              onClick={() => {
                                setInputText(example.text);
                                setInputMode("text");
                              }}
                              className="text-xs"
                            >
                              <Sparkles className="w-3 h-3 mr-1" />
                              {example.label}
                            </Button>
                          ))}
                        </div>
                      </div>

                      {predictionError ? (
                        <Alert
                          variant="destructive"
                          className="border-red-200 dark:border-red-900/60"
                        >
                          <AlertTriangle />
                          <AlertTitle>Prediction unavailable</AlertTitle>
                          <AlertDescription>{predictionError}</AlertDescription>
                        </Alert>
                      ) : backendStatus === "disconnected" ? (
                        <Alert className="border-amber-200 bg-amber-50/80 text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-100">
                          <Info />
                          <AlertTitle>Backend disconnected</AlertTitle>
                          <AlertDescription className="text-amber-800 dark:text-amber-200">
                            Start the FastAPI backend to run live predictions,
                            fetch history, and load model metrics.
                          </AlertDescription>
                        </Alert>
                      ) : !modelLoaded ? (
                        <Alert className="border-amber-200 bg-amber-50/80 text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-100">
                          <Info />
                          <AlertTitle>Model not trained</AlertTitle>
                          <AlertDescription className="text-amber-800 dark:text-amber-200">
                            The backend is online, but predictions are running
                            in fallback mode until a trained model is restored.
                          </AlertDescription>
                        </Alert>
                      ) : null}
                    </CardContent>
                    <CardFooter>
                      <Button
                        onClick={handlePredict}
                        disabled={
                          isLoading ||
                          (inputMode === "text"
                            ? inputText.length < 10
                            : !inputUrl)
                        }
                        className="w-full h-12 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white shadow-lg"
                      >
                        {isLoading ? (
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{
                              duration: 1,
                              repeat: Infinity,
                              ease: "linear",
                            }}
                          >
                            <Shield className="w-5 h-5" />
                          </motion.div>
                        ) : (
                          <>
                            <Target className="w-5 h-5 mr-2" />
                            Analyze Article
                          </>
                        )}
                      </Button>
                    </CardFooter>
                  </Card>
                </motion.div>

                {/* Results Section */}
                <motion.div
                  variants={fadeInUp}
                  initial="initial"
                  animate="animate"
                  transition={{ delay: 0.3 }}
                >
                  <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-slate-200 dark:border-slate-700 shadow-xl h-full">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <BarChart3 className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                        Analysis Results
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <AnimatePresence mode="wait">
                        {isLoading ? (
                          <motion.div
                            key="loading"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="flex flex-col items-center justify-center py-16"
                          >
                            <motion.div
                              animate={{
                                scale: [1, 1.2, 1],
                                rotate: [0, 180, 360],
                              }}
                              transition={{ duration: 2, repeat: Infinity }}
                              className="w-16 h-16 rounded-full border-4 border-slate-200 dark:border-slate-700 border-t-emerald-500"
                            />
                            <p className="mt-4 text-slate-600 dark:text-slate-400">
                              Analyzing article...
                            </p>
                          </motion.div>
                        ) : result ? (
                          <motion.div
                            key="result"
                            variants={scaleIn}
                            initial="initial"
                            animate="animate"
                            exit="exit"
                            className="space-y-6"
                          >
                            {/* Prediction Badge */}
                            <div className="flex items-center justify-center">
                              <motion.div
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                transition={{ type: "spring", delay: 0.2 }}
                                className={`relative p-6 rounded-full ${
                                  result.prediction === "Real"
                                    ? "bg-emerald-100 dark:bg-emerald-900/30"
                                    : "bg-red-100 dark:bg-red-900/30"
                                }`}
                              >
                                {result.prediction === "Real" ? (
                                  <CheckCircle className="w-16 h-16 text-emerald-600 dark:text-emerald-400" />
                                ) : (
                                  <AlertTriangle className="w-16 h-16 text-red-600 dark:text-red-400" />
                                )}
                                <motion.div
                                  initial={{ scale: 0 }}
                                  animate={{ scale: 1 }}
                                  transition={{ delay: 0.4 }}
                                  className={`absolute -top-2 -right-2 px-3 py-1 rounded-full text-xs font-bold ${
                                    result.prediction === "Real"
                                      ? "bg-emerald-500 text-white"
                                      : "bg-red-500 text-white"
                                  }`}
                                >
                                  {result.prediction}
                                </motion.div>
                              </motion.div>
                            </div>

                            {/* Confidence Meter */}
                            <div className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span className="text-slate-600 dark:text-slate-400">
                                  Confidence
                                </span>
                                <span className="font-bold">
                                  {result.confidence}%
                                </span>
                              </div>
                              <Progress
                                value={result.confidence}
                                className={`h-3 ${
                                  result.prediction === "Real"
                                    ? "[&>div]:bg-emerald-500"
                                    : "[&>div]:bg-red-500"
                                }`}
                              />
                            </div>

                            {/* Key Words */}
                            <div className="space-y-3">
                              <h4 className="font-semibold text-sm text-slate-700 dark:text-slate-300 flex items-center gap-2">
                                <Sparkles className="w-4 h-4 text-amber-500" />
                                Key Contributing Words
                              </h4>
                              {result.keywords.length > 0 ? (
                                <div className="flex flex-wrap gap-2">
                                  {result.keywords.map(([word, score], i) => (
                                    <motion.div
                                      key={word}
                                      initial={{ opacity: 0, scale: 0 }}
                                      animate={{ opacity: 1, scale: 1 }}
                                      transition={{ delay: 0.1 * i }}
                                    >
                                      <Tooltip>
                                        <TooltipTrigger>
                                          <Badge
                                            variant="outline"
                                            className={`px-3 py-1 cursor-pointer transition-all hover:scale-105 ${
                                              score > 0
                                                ? "border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300"
                                                : "border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300"
                                            }`}
                                          >
                                            {word}
                                            {score > 0 ? (
                                              <TrendingUp className="w-3 h-3 ml-1" />
                                            ) : (
                                              <TrendingDown className="w-3 h-3 ml-1" />
                                            )}
                                          </Badge>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                          <p>
                                            Impact: {Math.abs(score).toFixed(3)}
                                          </p>
                                          <p className="text-xs">
                                            {score > 0
                                              ? "Contributes to Real"
                                              : "Contributes to Fake"}
                                          </p>
                                        </TooltipContent>
                                      </Tooltip>
                                    </motion.div>
                                  ))}
                                </div>
                              ) : (
                                <div className="rounded-xl border border-dashed border-slate-300 dark:border-slate-700 px-4 py-3 text-sm text-slate-500 dark:text-slate-400">
                                  No keyword explanation was returned for this
                                  prediction.
                                </div>
                              )}
                            </div>

                            {/* Word Impact Visualization */}
                            <div className="space-y-3">
                              <h4 className="font-semibold text-sm text-slate-700 dark:text-slate-300">
                                Word Impact Analysis
                              </h4>
                              {result.keywords.length > 0 ? (
                                <div className="h-32">
                                  <ResponsiveContainer width="100%" height="100%">
                                    <BarChart
                                      data={result.keywords.map(
                                        ([word, score]) => ({
                                          word,
                                          impact: Math.abs(score),
                                        }),
                                      )}
                                      layout="vertical"
                                    >
                                      <CartesianGrid
                                        strokeDasharray="3 3"
                                        className="stroke-slate-200 dark:stroke-slate-700"
                                      />
                                      <XAxis
                                        type="number"
                                        domain={[0, "auto"]}
                                        tick={{ fontSize: 10 }}
                                      />
                                      <YAxis
                                        dataKey="word"
                                        type="category"
                                        width={60}
                                        tick={{ fontSize: 10 }}
                                      />
                                      <Bar
                                        dataKey="impact"
                                        fill={
                                          result.prediction === "Real"
                                            ? "hsl(142, 76%, 36%)"
                                            : "hsl(0, 84%, 60%)"
                                        }
                                        radius={[0, 4, 4, 0]}
                                      />
                                    </BarChart>
                                  </ResponsiveContainer>
                                </div>
                              ) : (
                                <div className="h-32 rounded-xl border border-dashed border-slate-300 dark:border-slate-700 flex items-center justify-center text-sm text-slate-500 dark:text-slate-400">
                                  Word impact will appear once explanation data is
                                  available.
                                </div>
                              )}
                            </div>

                            <Button
                              variant="outline"
                              onClick={handleDownloadReport}
                              className="w-full gap-2"
                            >
                              <Download className="w-4 h-4" />
                              Download Report
                            </Button>
                          </motion.div>
                        ) : (
                          <motion.div
                            key="empty"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="flex flex-col items-center justify-center py-16 text-center"
                          >
                            <div className="w-20 h-20 rounded-full bg-slate-100 dark:bg-slate-700 flex items-center justify-center mb-4">
                              <Search className="w-10 h-10 text-slate-400" />
                            </div>
                            <h3 className="font-semibold text-slate-700 dark:text-slate-300">
                              {backendStatus === "disconnected"
                                ? "Backend Unavailable"
                                : "No Analysis Yet"}
                            </h3>
                            <p className="text-sm text-slate-500 mt-1">
                              {backendStatus === "disconnected"
                                ? "Reconnect the backend to run live article analysis."
                                : "Enter text or URL to begin analysis"}
                            </p>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </CardContent>
                  </Card>
                </motion.div>
              </div>
            </TabsContent>

            {/* Dashboard Tab */}
            <TabsContent value="dashboard" className="space-y-6">
              {/* Stats Overview */}
              <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {metricCards.map((stat, i) => (
                  <motion.div
                    key={stat.label}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.1 }}
                  >
                    <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-slate-200 dark:border-slate-700">
                      <CardContent className="p-6">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm text-slate-500 dark:text-slate-400">
                              {stat.label}
                            </p>
                            <p className="text-2xl font-bold mt-1">
                              {stat.value}
                            </p>
                            <Badge
                              variant="outline"
                              className={`mt-2 ${stat.accentText} border-current/30`}
                            >
                              {backendStatus === "connected"
                                ? "Live backend"
                                : backendStatus === "checking"
                                  ? "Checking"
                                  : "Unavailable"}
                            </Badge>
                          </div>
                          <div className={`p-3 rounded-xl ${stat.accent}`}>
                            <stat.icon
                              className={`w-6 h-6 ${stat.accentText}`}
                            />
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </div>

              {/* Charts */}
              <div className="grid lg:grid-cols-2 gap-6">
                {/* Metrics Bar Chart */}
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-slate-200 dark:border-slate-700">
                    <CardHeader>
                      <CardTitle className="text-lg">
                        Model Performance Metrics
                      </CardTitle>
                      <CardDescription>
                        Current model accuracy and performance indicators
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={metricsData}>
                            <CartesianGrid
                              strokeDasharray="3 3"
                              className="stroke-slate-200 dark:stroke-slate-700"
                            />
                            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                            <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
                            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                              {metricsData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.fill} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>

                {/* Predictions Distribution */}
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 }}
                >
                  <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-slate-200 dark:border-slate-700">
                    <CardHeader>
                      <CardTitle className="text-lg">
                        Prediction Distribution
                      </CardTitle>
                      <CardDescription>
                        Real vs Fake news classification breakdown
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      {pieData.length > 0 ? (
                        <div className="h-64">
                          <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                              <Pie
                                data={pieData}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                              >
                                {pieData.map((entry, index) => (
                                  <Cell
                                    key={`cell-${index}`}
                                    fill={entry.color}
                                  />
                                ))}
                              </Pie>
                              <Legend />
                            </PieChart>
                          </ResponsiveContainer>
                        </div>
                      ) : (
                        <div className="flex h-64 flex-col items-center justify-center text-center">
                          <BarChart3 className="mb-4 h-10 w-10 text-slate-300 dark:text-slate-600" />
                          <p className="font-medium text-slate-600 dark:text-slate-300">
                            No prediction data yet
                          </p>
                          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                            Run a few analyses to populate the distribution
                            chart.
                          </p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>
              </div>

              {/* Trend Chart */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
              >
                <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-slate-200 dark:border-slate-700">
                  <CardHeader>
                    <CardTitle className="text-lg">
                      Prediction Activity Trends
                    </CardTitle>
                    <CardDescription>
                      Monthly average confidence and prediction volume
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {trendData.length > 0 ? (
                      <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={trendData}>
                            <defs>
                              <linearGradient
                                id="colorConfidence"
                                x1="0"
                                y1="0"
                                x2="0"
                                y2="1"
                              >
                                <stop
                                  offset="5%"
                                  stopColor="hsl(142, 76%, 36%)"
                                  stopOpacity={0.3}
                                />
                                <stop
                                  offset="95%"
                                  stopColor="hsl(142, 76%, 36%)"
                                  stopOpacity={0}
                                />
                              </linearGradient>
                            </defs>
                            <CartesianGrid
                              strokeDasharray="3 3"
                              className="stroke-slate-200 dark:stroke-slate-700"
                            />
                            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                            <YAxis
                              yAxisId="left"
                              domain={[0, 100]}
                              tick={{ fontSize: 12 }}
                            />
                            <YAxis
                              yAxisId="right"
                              orientation="right"
                              tick={{ fontSize: 12 }}
                            />
                            <Area
                              yAxisId="left"
                              type="monotone"
                              dataKey="confidence"
                              stroke="hsl(142, 76%, 36%)"
                              fillOpacity={1}
                              fill="url(#colorConfidence)"
                            />
                            <Line
                              yAxisId="right"
                              type="monotone"
                              dataKey="predictions"
                              stroke="hsl(38, 92%, 50%)"
                              strokeWidth={2}
                              dot={{ fill: "hsl(38, 92%, 50%)" }}
                            />
                            <Legend />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div className="flex h-64 flex-col items-center justify-center text-center">
                        <Clock className="mb-4 h-10 w-10 text-slate-300 dark:text-slate-600" />
                        <p className="font-medium text-slate-600 dark:text-slate-300">
                          No trend data yet
                        </p>
                        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                          Run predictions over time to populate confidence and
                          activity trends.
                        </p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            </TabsContent>

            {/* History Tab */}
            <TabsContent value="history" className="space-y-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-slate-200 dark:border-slate-700">
                  <CardHeader>
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                      <div>
                        <CardTitle className="text-lg">
                          Prediction History
                        </CardTitle>
                        <CardDescription>
                          View your past analyses and results
                        </CardDescription>
                      </div>
                      <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                        <Input
                          placeholder="Search history..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          className="pl-10 w-full sm:w-64"
                        />
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <ScrollArea className="h-[500px]">
                      <div className="space-y-3">
                        <AnimatePresence>
                          {filteredHistory.length > 0 ? (
                            filteredHistory.map((item, i) => (
                              <motion.div
                                key={item.id}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: 20 }}
                                transition={{ delay: i * 0.05 }}
                              >
                                <div className="flex items-start gap-4 p-4 rounded-lg bg-slate-50 dark:bg-slate-700/50 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors">
                                  <div
                                    className={`p-2 rounded-lg ${
                                      item.prediction === "Real"
                                        ? "bg-emerald-100 dark:bg-emerald-900/30"
                                        : "bg-red-100 dark:bg-red-900/30"
                                    }`}
                                  >
                                    {item.prediction === "Real" ? (
                                      <CheckCircle className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                                    ) : (
                                      <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400" />
                                    )}
                                  </div>
                                  <div className="flex-1 min-w-0">
                                    <p className="text-sm font-medium text-slate-700 dark:text-slate-300 truncate">
                                      {item.input_text}
                                    </p>
                                    <div className="flex items-center gap-3 mt-2 text-xs text-slate-500 dark:text-slate-400">
                                      <Badge
                                        variant="outline"
                                        className={
                                          item.prediction === "Real"
                                            ? "border-emerald-300 dark:border-emerald-700 text-emerald-600 dark:text-emerald-400"
                                            : "border-red-300 dark:border-red-700 text-red-600 dark:text-red-400"
                                        }
                                      >
                                        {item.prediction}
                                      </Badge>
                                      <span>{item.confidence}% confidence</span>
                                      <span className="flex items-center gap-1">
                                        <Clock className="w-3 h-3" />
                                        {new Date(
                                          item.timestamp,
                                        ).toLocaleDateString()}
                                      </span>
                                    </div>
                                  </div>
                                  <ChevronRight className="w-5 h-5 text-slate-400" />
                                </div>
                              </motion.div>
                            ))
                          ) : (
                            <motion.div
                              key="empty-history"
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              className="flex flex-col items-center justify-center py-16 text-center"
                            >
                              <History className="mb-4 h-10 w-10 text-slate-300 dark:text-slate-600" />
                              <h3 className="font-semibold text-slate-700 dark:text-slate-300">
                                {history.length === 0
                                  ? "No history yet"
                                  : "No matching results"}
                              </h3>
                              <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                                {history.length === 0
                                  ? "Run a live prediction to populate backend history."
                                  : "Try a different search term."}
                              </p>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    </ScrollArea>
                  </CardContent>
                </Card>
              </motion.div>
            </TabsContent>

            {/* Retrain Tab */}
            <TabsContent value="retrain" className="space-y-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="max-w-4xl mx-auto space-y-6"
              >
                <Card className="bg-gradient-to-br from-blue-500 to-purple-600 text-white border-0 overflow-hidden">
                  <CardContent className="p-8 relative">
                    <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -translate-y-1/2 translate-x-1/2" />
                    <div className="relative z-10">
                      <Zap className="w-12 h-12 mb-4" />
                      <h2 className="text-3xl font-bold mb-2">
                        Model Retraining
                      </h2>
                      <p className="text-blue-100">
                        Improve the model&apos;s accuracy by retraining it with
                        verified historical labels while preserving a fixed
                        validation holdout.
                      </p>
                    </div>
                  </CardContent>
                </Card>

                {/* Training Stats */}
                {trainingStats && (
                  <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-slate-200 dark:border-slate-700">
                    <CardHeader>
                      <CardTitle>Training Data Statistics</CardTitle>
                      <CardDescription>
                        Current verified data available for retraining
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-600">
                            {trainingStats.total_samples || 0}
                          </div>
                          <div className="text-sm text-slate-600 dark:text-slate-400">
                            Total Samples
                          </div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-600">
                            {trainingStats.real_samples || 0}
                          </div>
                          <div className="text-sm text-slate-600 dark:text-slate-400">
                            Real News
                          </div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-red-600">
                            {trainingStats.fake_samples || 0}
                          </div>
                          <div className="text-sm text-slate-600 dark:text-slate-400">
                            Fake News
                          </div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-purple-600">
                            {trainingStats.min_samples_for_retraining || 50}
                          </div>
                          <div className="text-sm text-slate-600 dark:text-slate-400">
                            Min Required
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Retrain Status */}
                {retrainStatus && (
                  <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-slate-200 dark:border-slate-700">
                    <CardHeader>
                      <CardTitle>Retraining Status</CardTitle>
                      <CardDescription>
                        Check whether enough verified labels are available
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">
                          Should Retrain:
                        </span>
                        <Badge
                          variant={
                            retrainStatus.should_retrain
                              ? "default"
                              : "secondary"
                          }
                        >
                          {retrainStatus.should_retrain ? "Yes" : "No"}
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">
                          Current Samples:
                        </span>
                        <span className="text-sm">
                          {retrainStatus.current_samples || 0}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">
                          Minimum Required:
                        </span>
                        <span className="text-sm">
                          {retrainStatus.min_samples_required || 50}
                        </span>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Retrain Button */}
                <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-slate-200 dark:border-slate-700">
                  <CardContent className="p-6">
                    <div className="text-center space-y-4">
                      <p className="text-slate-600 dark:text-slate-400">
                        Retrain the model using verified historical labels.
                        Evaluation runs on a fixed holdout set before the new
                        model is saved.
                      </p>
                      <Button
                        onClick={handleRetrain}
                        disabled={
                          isRetraining || !retrainStatus?.should_retrain
                        }
                        className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white px-8 py-3"
                      >
                        {isRetraining ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                            Retraining...
                          </>
                        ) : (
                          <>
                            <Zap className="w-4 h-4 mr-2" />
                            Retrain Model
                          </>
                        )}
                      </Button>
                      {!retrainStatus?.should_retrain && (
                        <p className="text-sm text-amber-600 dark:text-amber-400">
                          Not enough data for retraining. Need at least{" "}
                          {retrainStatus?.min_samples_required || 50} samples.
                        </p>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </TabsContent>

            {/* About Tab */}
            <TabsContent value="about" className="space-y-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="max-w-4xl mx-auto space-y-6"
              >
                {/* Hero Section */}
                <Card className="bg-gradient-to-br from-emerald-500 to-teal-600 text-white border-0 overflow-hidden">
                  <CardContent className="p-8 relative">
                    <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -translate-y-1/2 translate-x-1/2" />
                    <div className="relative z-10">
                      <Shield className="w-12 h-12 mb-4" />
                      <h2 className="text-3xl font-bold mb-2">
                        How TF-IDF Powers This Detector
                      </h2>
                      <p className="text-emerald-100">
                        Term Frequency-Inverse Document Frequency is a powerful
                        statistical measure that evaluates how relevant a word
                        is to a document in a collection.
                      </p>
                    </div>
                  </CardContent>
                </Card>

                {/* How It Works */}
                <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-slate-200 dark:border-slate-700">
                  <CardHeader>
                    <CardTitle>How TF-IDF Works</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {[
                      {
                        step: 1,
                        title: "Term Frequency (TF)",
                        description:
                          "Measures how frequently a term appears in a document. A higher value means the term appears more often in that document.",
                        formula:
                          "TF(t, d) = (Number of times term t appears in document d) / (Total terms in d)",
                      },
                      {
                        step: 2,
                        title: "Inverse Document Frequency (IDF)",
                        description:
                          'Measures how important a term is. Common words like "the" have low IDF, while rare words have high IDF.',
                        formula:
                          "IDF(t) = log(Total documents / Documents containing term t)",
                      },
                      {
                        step: 3,
                        title: "TF-IDF Score",
                        description:
                          "The product of TF and IDF. High scores indicate terms that are frequent in a document but rare across all documents.",
                        formula: "TF-IDF(t, d) = TF(t, d) × IDF(t)",
                      },
                    ].map((item, i) => (
                      <motion.div
                        key={item.step}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="flex gap-4"
                      >
                        <div className="flex-shrink-0 w-10 h-10 rounded-full bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center">
                          <span className="font-bold text-emerald-600 dark:text-emerald-400">
                            {item.step}
                          </span>
                        </div>
                        <div className="flex-1">
                          <h4 className="font-semibold text-slate-700 dark:text-slate-300">
                            {item.title}
                          </h4>
                          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                            {item.description}
                          </p>
                          <code className="text-xs bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded mt-2 inline-block">
                            {item.formula}
                          </code>
                        </div>
                      </motion.div>
                    ))}
                  </CardContent>
                </Card>

                {/* Model Architecture */}
                <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-slate-200 dark:border-slate-700">
                  <CardHeader>
                    <CardTitle>Model Architecture</CardTitle>
                    <CardDescription>
                      Our fake news detection pipeline
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                      {[
                        { icon: FileText, label: "Input Text" },
                        { icon: Search, label: "Preprocess" },
                        { icon: BarChart3, label: "TF-IDF Vector" },
                        { icon: Target, label: "Logistic Regression" },
                        { icon: Shield, label: "Prediction" },
                      ].map((item, i) => (
                        <div key={i} className="flex items-center gap-2">
                          <motion.div
                            whileHover={{ scale: 1.1 }}
                            className="flex flex-col items-center gap-2 p-4 rounded-xl bg-slate-50 dark:bg-slate-700"
                          >
                            <item.icon className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
                            <span className="text-xs font-medium text-slate-700 dark:text-slate-300">
                              {item.label}
                            </span>
                          </motion.div>
                          {i < 4 && (
                            <ChevronRight className="hidden sm:block w-4 h-4 text-slate-400" />
                          )}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Recommended TF-IDF Parameters */}
                <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-slate-200 dark:border-slate-700">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-amber-500" />
                      Optimized TF-IDF Configuration
                    </CardTitle>
                    <CardDescription>
                      Recommended parameters for improved accuracy
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid sm:grid-cols-2 gap-4">
                      {[
                        {
                          param: "max_features",
                          value: "15,000",
                          desc: "Larger vocabulary for better coverage",
                        },
                        {
                          param: "ngram_range",
                          value: "(1, 2)",
                          desc: "Unigrams + Bigrams capture phrases",
                        },
                        {
                          param: "min_df",
                          value: "2",
                          desc: "Remove rare words (noise reduction)",
                        },
                        {
                          param: "max_df",
                          value: "0.85",
                          desc: "Remove overly common words",
                        },
                        {
                          param: "sublinear_tf",
                          value: "True",
                          desc: "Log scaling for better weights",
                        },
                        {
                          param: "stop_words",
                          value: '"english"',
                          desc: "Built-in stopword removal",
                        },
                      ].map((item, i) => (
                        <motion.div
                          key={item.param}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.05 }}
                          className="p-4 rounded-lg bg-slate-50 dark:bg-slate-700"
                        >
                          <div className="flex items-center justify-between mb-1">
                            <code className="text-sm font-medium text-emerald-600 dark:text-emerald-400">
                              {item.param}
                            </code>
                            <Badge variant="secondary">{item.value}</Badge>
                          </div>
                          <p className="text-xs text-slate-600 dark:text-slate-400">
                            {item.desc}
                          </p>
                        </motion.div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </TabsContent>
          </Tabs>
        </main>

        {/* Footer */}
        <footer className="border-t border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <Shield className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                <span className="text-sm text-slate-600 dark:text-slate-400">
                  {APP_NAME} - Powered by TF-IDF & Machine Learning
                </span>
              </div>
              <div className="flex items-center gap-4 text-sm text-slate-500 dark:text-slate-400">
                <span>Logistic Regression Model</span>
                <Separator orientation="vertical" className="h-4" />
                <span>{metricsBadgeLabel}</span>
                <Separator orientation="vertical" className="h-4" />
                <span>{history.length} Total Predictions</span>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </TooltipProvider>
  );
}
