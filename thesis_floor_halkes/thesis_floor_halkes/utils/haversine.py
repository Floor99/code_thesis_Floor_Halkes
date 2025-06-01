import math


def haversine(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
    """
    Calculate great‐circle distance between two (lat, lon) pairs in kilometers.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    # convert degrees → radians
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6_371_000.0  # Earth radius in m
    return R * c
