import requests

def get_address_osm(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}"
        headers = {"User-Agent": "AttendanceApp"}
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        return data.get("display_name", "")
    except:
        return ""
