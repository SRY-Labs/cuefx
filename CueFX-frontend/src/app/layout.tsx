import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CueFX Studio",
  description: "Direct, style, and preview your visual effects shots in CueFX Studio.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
