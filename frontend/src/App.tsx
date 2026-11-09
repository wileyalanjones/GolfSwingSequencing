import { useState } from "react"
import { EventCard } from "./EventCard"
import type { SwingEvent } from "./types"

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [events, setEvents] = useState<SwingEvent[]>([])

  async function handleAnalyze() {
    if (!file) {
      setError("Please choose a video first")
      return 
    }
    const formData = new FormData()
    formData.append("video", file)
    setLoading(true)
    setError(null)
    setEvents([])
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(typeof data.detail === "string" ? data.detail : "Something went wrong")
      }
      setEvents(data.events)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong")
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900">
      <div className="mx-auto max-w-4xl px-6 py-12">
        <h1 className="text-4xl font-bold tracking-tight text-green-700">
          Golf Swing Sequencer
        </h1>
        <p className="mt-2 text-gray-600">
          Upload a video of a golf swing to find its 8 key moments.
        </p>

        <div className="mt-8 flex flex-col items-start gap-4 rounded-xl bg-white p-6 shadow-sm ring-1 ring-gray-200 sm:flex-row sm:items-center">
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="block text-sm text-gray-600
              file:mr-4 file:cursor-pointer file:rounded-lg file:border-0
              file:bg-green-600 file:px-4 file:py-2 file:text-white
              hover:file:bg-green-700"
          />
          <button
            disabled={loading}
            onClick={handleAnalyze}
            className="rounded-lg bg-gray-900 px-5 py-2 text-sm font-medium text-white transition hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50 sm:ml-auto"
          >
            {loading ? "Analyzing..." : "Analyze"}
          </button>
        </div>

        {error && (
          <p className="mt-4 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-200">
            {error}
          </p>
        )}

        <div className="mt-8 grid grid-cols-2 gap-4 md:grid-cols-4">
          {events.map((e) => (
            <EventCard key={e.event} event={e} />
          ))}
        </div>
      </div>
    </main>
  )
}

export default App
