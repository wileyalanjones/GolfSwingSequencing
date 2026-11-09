import type { SwingEvent } from "./types"

export function EventCard({ event }: { event: SwingEvent }) {
    return (
        <figure className="overflow-hidden rounded-xl bg-white shadow-sm ring-1 ring-gray-200">
            <img src={event.image} alt={event.event} className="w-full" />
            <figcaption className="p-3 text-sm">
                <span className="font-semibold">{event.event}</span>
                <span className="text-gray-500"> (frame {event.frame})</span>
            </figcaption>
        </figure>
    )
}
