import requests

def get_address_osm(lat, lon):
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "format": "jsonv2",
            "lat": lat,
            "lon": lon
        }
        headers = {
            "User-Agent": "AttendanceApp/1.0 (contact@example.com)"  # nên có email để tránh bị chặn
        }
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()  # bắt lỗi HTTP nếu có
        data = response.json()
        return data.get("display_name", "")
    except requests.RequestException as e:
        print(f"[OSM] Request error: {e}")
        return ""
    except ValueError:
        print("[OSM] JSON decode error")
        return ""

