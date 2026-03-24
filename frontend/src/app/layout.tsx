import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const APP_NAME = "Fake News Detector";

export const metadata: Metadata = {
  title: `${APP_NAME} - TF-IDF & Machine Learning`,
  description:
    "Detect fake news using TF-IDF vectorization and Logistic Regression. Upload news text or URLs to analyze credibility.",
  keywords: [
    "Fake News",
    "TF-IDF",
    "Machine Learning",
    "NLP",
    "Logistic Regression",
    "News Analysis",
  ],
  authors: [{ name: `${APP_NAME} Team` }],
  icons: {
    icon: "/logo.svg",
  },
  openGraph: {
    title: APP_NAME,
    description: "Detect fake news using TF-IDF and Machine Learning",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased bg-background text-foreground">
        {children}
        <Toaster />
      </body>
    </html>
  );
}
