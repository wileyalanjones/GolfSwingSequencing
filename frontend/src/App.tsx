import { useState } from "react"
import { EventCard } from "./EventCard"
import type { SwingEvent } from "./types"
import { LightBox } from "./LightBox"

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [events, setEvents] = useState<SwingEvent[]>([])
  const [dragging, setDragging] = useState(false)
  const [selected, setSelected] = useState<SwingEvent | null>(null)
  const [open, setOpen] = useState(false)

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

  function handleDrop(e: React.DragEvent<HTMLLabelElement>) {
    e.preventDefault()
    const dropped = e.dataTransfer.files[0]
    if (!dropped) return
    if (!dropped.type.startsWith("video/")) {
      setError("Please drop a video file")
      return
    }
    setError(null)
    setFile(dropped)
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

        <div className="mt-8 flex flex-col gap-4 rounded-xl bg-white p-6 shadow-sm ring-1 ring-gray-200">
          <label 
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          className={`flex w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 text-center transition ${
            dragging
              ? "border-green-500 bg-green-50"
              : "border-gray-300 hover:border-green-500 hover:bg-green-50"
          }`}
  >
            <span>{file ? file.name : "Click to upload a swing video"}</span>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="hidden"
            />
          </label>

          <button
            disabled={loading}
            onClick={handleAnalyze}
            className="rounded-lg bg-gray-900 px-5 py-2 text-sm font-medium text-white transition hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50 "
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
            <EventCard key={e.event} event={e} onClick={() => { setSelected(e); setOpen(true) }}/>
          ))}
        </div>
        <div>
          {selected && <LightBox event={selected} open={open} onClose={() => setOpen(false)}/>}
        </div>
      </div>
    </main>
  )
}

export default App
