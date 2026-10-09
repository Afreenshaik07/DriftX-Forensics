def calculate_onset_lags(events):
    if not events:
        return {}

    ordered = sorted(
        events,
        key=lambda event: event["timestamp"]
    )

    first_timestamp = ordered[0]["timestamp"]

    return {
        event["stage"]: event["timestamp"] - first_timestamp
        for event in ordered
    }
