import type { SwingEvent } from "./types";
import { useEffect } from "react";

type LightBocProps = {
    event: SwingEvent,
    open: boolean,
    onClose: () => void
}

export function LightBox({event, open, onClose}: LightBocProps) {   
    
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
                <figcaption className="mt-3 text-white">
                    <span className="font-semibold">{event.event}</span>
                    <span className="text-gray-300"> (frame {event.frame})</span>
                </figcaption>
            </figure>
        </div>
    )
  }