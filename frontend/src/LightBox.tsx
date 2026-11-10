import type { SwingEvent } from "./types";
import { useEffect } from "react";
import { MdDownload } from "react-icons/md"

type LightBoxProps = {
    event: SwingEvent,
    open: boolean,
    onClose: () => void
}

export function LightBox({event, open, onClose}: LightBoxProps) {   
    
    const filename = `${event.event.toLowerCase().replace(/\s+/g, "-")}-frame-${event.frame}.jpg`

    useEffect(() => {
        console.log("lightbox mounted")
        return () => console.log("lightbox unmounted")
    }, [])

    useEffect(() => {
        if (!open) return;
        const handleKey = (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose()
        }
        window.addEventListener("keydown", handleKey);
        return () => window.removeEventListener("keydown", handleKey)
    },[open, onClose])

    return (
        <div
            onClick={onClose}
            className={`fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 transition-opacity duration-200 ${
            open ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
        >
            <figure
            onClick={(e) => e.stopPropagation()}
            className="flex max-h-full flex-col items-center rounded-xl border border-white/30 bg-black/40 p-3"
            >
                <img
                    src={event.image}
                    alt={event.event}
                    className="max-h-[80vh] max-w-full rounded-lg object-contain"
                />
            <figcaption className="relative mt-3 w-full text-center text-white">
                <span className="font-semibold">{event.event}</span>
                <span className="text-gray-300"> (frame {event.frame})</span>
                <a
                href={event.image}
                download={filename}
                aria-label="Download image"
                className="absolute right-0 top-1/2 -translate-y-1/2 rounded-lg bg-white/10 p-2 hover:bg-white/20"
                >
                <MdDownload size={20} />
                </a>
            </figcaption>
            </figure>
        </div>
    )
  }