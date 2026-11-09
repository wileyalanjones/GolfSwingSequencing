export type SwingEvent = {
    event: string;
    frame: number;
    image: string;
}

export type AnalyzeResponse = {
    events: SwingEvent[]
}