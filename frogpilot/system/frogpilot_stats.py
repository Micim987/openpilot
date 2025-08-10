import json
import os
import random
import requests
import socket
import subprocess
import sys

from collections import Counter
from datetime import datetime, timezone
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError, ReadTimeoutError

import openpilot.system.sentry as sentry

from openpilot.common.conversions import Conversions as CV
from openpilot.system.version import get_build_metadata

from openpilot.frogpilot.common.frogpilot_utilities import run_cmd
from openpilot.frogpilot.common.frogpilot_variables import get_frogpilot_toggles, params, params_tracking

BASE_URL = "https://nominatim.openstreetmap.org"
MINIMUM_POPULATION = 100_000

def get_county_center(latitude, longitude):
  try:
    with requests.Session() as session:
      session.headers.update({"Accept-Language": "en"})
      session.headers.update({"User-Agent": "frogpilot-county-center-checker/1.0 (https://github.com/FrogAi/FrogPilot)"})

      location_data = session.get(f"{BASE_URL}/reverse", params={"format": "jsonv2", "lat": latitude, "lon": longitude, "addressdetails": 1, "extratags": 1}, timeout=10).json()
      address = location_data.get("address", {})
      city = address.get("city") or address.get("hamlet") or address.get("municipality") or address.get("town") or address.get("village")
      county = address.get("county", "Unknown")
      country = address.get("country", "United States")
      country_code = address.get("country_code", "US").upper()
      state = address.get("state", "N/A") if country_code == "US" else "N/A"

      if city:
        try:
          city_params = {"city": city, "country": country_code, "format": "json", "limit": 1, "extratags": 1}
          if country_code == "US" and state != "N/A":
            city_params["state"] = state
          city_data = session.get(f"{BASE_URL}/search", params=city_params, timeout=10).json()

          if city_data:
            population_string = (city_data[0].get("extratags", {}) or {}).get("population")
            try:
              city_population = int(str(population_string).replace(",", "").strip()) if population_string else 0
            except Exception:
              city_population = 0

            if city_population >= MINIMUM_POPULATION:
              center_latitude = float(city_data[0]["lat"])
              center_longitude = float(city_data[0]["lon"])
              print(f"Using city center for {city}, {state}, {country} → ({center_latitude}, {center_longitude})")
              return center_latitude, center_longitude, city, state, country
        except Exception as city_error:
          if not isinstance(city_error, (ConnectTimeoutError, NewConnectionError, ReadTimeoutError, TimeoutError, socket.gaierror, socket.timeout)):
            sentry.capture_exception(city_error, crash_log=False)

      try:
        if country_code == "US" and state != "N/A":
          state_capital_data = session.get(f"{BASE_URL}/search", params={"q": f"capital city of {state}, United States", "countrycodes": country_code.lower(), "format": "json", "limit": 5, "addressdetails": 1}, timeout=10).json()

          candidate = None
          for entry in state_capital_data or []:
            if entry.get("class") == "place" and entry.get("type") in ("city", "municipality", "town"):
              address_info = entry.get("address", {}) or {}
              if (state in entry.get("display_name", "")) or (address_info.get("state") == state):
                candidate = entry
                break

          if not candidate:
            for entry in state_capital_data or []:
              if entry.get("class") in ("boundary", "place") and entry.get("type") in ("administrative",):
                if ("Capitol" not in entry.get("display_name", "")) and (state in entry.get("display_name", "")):
                  candidate = entry
                  break

          if candidate:
            capital_latitude = float(candidate["lat"])
            capital_longitude = float(candidate["lon"])
            capital_name = candidate.get("display_name", "").split(",")[0] or "Unknown"
            print(f"City < {MINIMUM_POPULATION}. Using state capital for {state} → ({capital_latitude}, {capital_longitude})")
            return capital_latitude, capital_longitude, capital_name, state, country

        country_capital_data = None
        for query in (f"capital city of {country}", f"capital of {country}", f"{country} capital city"):
          country_capital_data = session.get(f"{BASE_URL}/search", params={"q": query, "countrycodes": country_code.lower(), "format": "json", "limit": 10, "addressdetails": 1}, timeout=10).json()
          if country_capital_data:
            break

        candidate = None
        for entry in country_capital_data or []:
          if entry.get("class") in ("boundary", "place") and entry.get("type") in ("administrative", "city", "municipality", "town"):
            address_info = entry.get("address", {}) or {}
            if (address_info.get("country_code", "").upper() == country_code) or (country in entry.get("display_name", "")):
              candidate = entry
              break

        if not candidate and country_capital_data:
          for entry in country_capital_data:
            if entry.get("class") in ("boundary", "place") and entry.get("type") in ("administrative", "city", "municipality", "town"):
              candidate = entry
              break

        if not candidate and not country_capital_data:
          country_capital_data = session.get(f"{BASE_URL}/search", params={"q": f"capital city of {country}", "format": "json", "limit": 10, "addressdetails": 1}, timeout=10).json()
          for entry in country_capital_data or []:
            if entry.get("class") in ("boundary", "place") and entry.get("type") in ("administrative", "city", "municipality", "town"):
              candidate = entry
              break

        if candidate:
          capital_latitude = float(candidate["lat"])
          capital_longitude = float(candidate["lon"])
          capital_name = candidate.get("display_name", "").split(",")[0] or "Unknown"
          print(f"City < {MINIMUM_POPULATION}. Using country capital for {country} → ({capital_latitude}, {capital_longitude})")
          return capital_latitude, capital_longitude, capital_name, state, country

        sentry.capture_exception(Exception(f"Capital lookup returned no results for {state}/{country_code}"), crash_log=False)
      except Exception as capital_error:
        if not isinstance(capital_error, (ConnectTimeoutError, NewConnectionError, ReadTimeoutError, TimeoutError, socket.gaierror, socket.timeout)):
          sentry.capture_exception(capital_error, crash_log=False)

      print(f"Falling back to (0, 0) for {latitude}, {longitude}")
      return float(0.0), float(0.0), city or "Unknown", state if country_code == "US" else "N/A", country

  except (requests.RequestException, socket.gaierror, socket.timeout, TimeoutError, Exception) as error:
    print(f"Falling back due to geocoding error: {error}")
    return float(0.0), float(0.0), "Unknown", "N/A", "Unknown"

def install_influxdb_client():
  try:
    import influxdb_client
    import influxdb_client.client.write_api
  except ModuleNotFoundError:
    print("influxdb-client not found. Attempting installation...")
    stock_mount_options = subprocess.run(["findmnt", "-no", "OPTIONS", "/"], capture_output=True, text=True, check=True).stdout.strip()

    run_cmd(["sudo", "mount", "-o", "remount,rw", "/"], "Successfully remounted / as read-write", "Failed to remount / as read-write")
    run_cmd(["sudo", sys.executable, "-m", "pip", "install", "influxdb-client"], "Successfully installed influxdb-client", "Failed to install influxdb-client", report=False)
    run_cmd(["sudo", "mount", "-o", f"remount,{stock_mount_options}", "/"], "Successfully restored stock mount options", "Failed to restore stock mount options")

def send_stats():
  frogpilot_toggles = get_frogpilot_toggles()

  if frogpilot_toggles.frogs_go_moo:
    return

  if frogpilot_toggles.car_make == "mock":
    return

  install_influxdb_client()

  from influxdb_client import InfluxDBClient, Point
  from influxdb_client.client.write_api import SYNCHRONOUS

  bucket = os.environ.get("STATS_BUCKET", "")
  org_ID = os.environ.get("STATS_ORG_ID", "")
  token = os.environ.get("STATS_TOKEN", "")
  url = os.environ.get("STATS_URL", "")

  frogpilot_stats = json.loads(params.get("FrogPilotStats") or "{}")

  location = json.loads(params.get("LastGPSPosition") or "{}")
  if not (location.get("latitude") and location.get("longitude")):
    return
  original_latitude = location.get("latitude")
  original_longitude = location.get("longitude")
  latitude, longitude, city, state, country = get_county_center(original_latitude, original_longitude)

  theme_sources = [
    frogpilot_toggles.icon_pack.replace("-animated", ""),
    frogpilot_toggles.color_scheme,
    frogpilot_toggles.distance_icons.replace("-animated", ""),
    frogpilot_toggles.signal_icons.replace("-animated", ""),
    frogpilot_toggles.sound_pack
  ]

  theme_counter = Counter(theme_sources)
  most_common = theme_counter.most_common()
  max_count = most_common[0][1]

  selected_theme = random.choice([item for item, count in most_common if count == max_count]).replace("-user_created", "").replace("_", " ")

  point = (Point("user_stats")
    .field("car_make", "GM" if frogpilot_toggles.car_make == "gm" else frogpilot_toggles.car_make.title())
    .field("car_model", frogpilot_toggles.car_model)
    .field("city", city)
    .field("country", country)
    .field("driving_model", frogpilot_toggles.model_name.replace("🗺️", "").replace("📡", "").replace("👀", "").replace("(Default)", "").strip())
    .field("event", 1)
    .field("frogpilot_drives", params_tracking.get_int("FrogPilotDrives"))
    .field("frogpilot_hours", params_tracking.get_int("FrogPilotMinutes") / 60)
    .field("frogpilot_miles", params_tracking.get_int("FrogPilotKilometers") * CV.KPH_TO_MPH)
    .field("has_cc_long", frogpilot_toggles.has_cc_long)
    .field("has_openpilot_longitudinal", frogpilot_toggles.openpilot_longitudinal)
    .field("has_pedal", frogpilot_toggles.has_pedal)
    .field("has_sdsu", frogpilot_toggles.has_sdsu)
    .field("has_zss", frogpilot_toggles.has_zss)
    .field("latitude", latitude)
    .field("longitude", longitude)
    .field("state", state)
    .field("theme", selected_theme.title())
    .field("total_aol_seconds", float(frogpilot_stats.get("TotalAOLTime", 0)))
    .field("total_lateral_seconds", float(frogpilot_stats.get("TotalLateralTime", 0)))
    .field("total_longitudinal_seconds", float(frogpilot_stats.get("TotalLongitudinalTime", 0)))
    .field("total_tracked_seconds", float(frogpilot_stats.get("TotalTrackedTime", 0)))
    .field("using_stock_acc", not (frogpilot_toggles.has_cc_long or frogpilot_toggles.openpilot_longitudinal))

    .tag("branch", get_build_metadata().channel)
    .tag("dongle_id", params.get("FrogPilotDongleId", encoding="utf-8"))

    .time(datetime.now(timezone.utc))
  )

  try:
    InfluxDBClient(org=org_ID, token=token, url=url).write_api(write_options=SYNCHRONOUS).write(bucket=bucket, org=org_ID, record=point)
    print("Successfully sent FrogPilot stats!")
  except Exception as exception:
    if not isinstance(exception, (ConnectTimeoutError, NewConnectionError, ReadTimeoutError, TimeoutError, socket.gaierror, socket.timeout)):
      sentry.capture_exception(exception, crash_log=False)
    print(f"Failed to send FrogPilot stats: {exception}")
