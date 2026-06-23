import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'lora-switchboard',
  description: 'Multi-tenant LoRA inference engine',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
