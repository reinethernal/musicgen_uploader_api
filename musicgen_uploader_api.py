#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MusicGen → AzuraCast (через API) с «флотом» треков:
- Генерация по prompt (audiocraft / facebook/musicgen-*)
- Кодирование в MP3 (ffmpeg), запись ID3 (mutagen, фреймы TIT2/TPE1/TALB/COMM)
- Загрузка через AzuraCast API в station media path
- Добавление загруженного трека в указанный плейлист
- Поддержка «флота» (fleet-min/fleet-max)
- Ежедневное освежение N треков в заданное время
- Состояние в out/.state.json

Зависимости (pip):
  audiocraft==1.3.0
  transformers>=4.38
  requests
  mutagen
  torch / torchaudio / torchvision (совместимые с вашей CUDA/CPU сборкой)
  + ffmpeg в системе

Пример запуска см. в конце файла.
"""

from __future__ import annotations
import os
import sys
import time
import json
import uuid
import argparse
import datetime as dt
import subprocess as sp
from pathlib import Path
from typing import Any, Dict, List, Optional

# Безопасные загрузки
os.environ.setdefault("TRANSFORMERS_PREFER_SAFETENSORS", "1")
# Выключаем быстрый транспорт HF, чтобы не требовать hf_transfer
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import numpy as np
import requests
import torch

from mutagen.id3 import (
    ID3, ID3NoHeaderError,
    TIT2, TPE1, TALB, COMM,
)

from audiocraft.models import MusicGen


# ----------------------------- логгер ------------------------------------

def log(level: str, msg: str) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    log("!", msg)
    sys.exit(code)


# ----------------------------- ffmpeg ------------------------------------

def encode_pcm_to_mp3(pcm_bytes: bytes, src_sr: int = 32000, bitrate: str = "192k") -> bytes:
    """
    Вход: interleaved s16le stereo @ src_sr → MP3 44.1kHz stereo (libmp3lame).
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ac", "2", "-ar", str(src_sr), "-i", "pipe:0",
        "-vn", "-ac", "2", "-ar", "44100",
        "-c:a", "libmp3lame", "-b:a", bitrate,
        "-f", "mp3", "pipe:1",
    ]
    p = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = p.communicate(pcm_bytes)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {err.decode('utf-8', 'ignore')}")
    return out


def write_id3_tags(mp3_path: Path,
                   title: str,
                   artist: str = "MusicGen",
                   album: str = "AI Generated",
                   comment: Optional[str] = None) -> None:
    """
    Пишем ID3 через «классические» фреймы (без EasyID3), чтобы избежать
    ошибок наподобие "'comment' is not a valid key".
    """
    try:
        tags = ID3(str(mp3_path))
    except ID3NoHeaderError:
        tags = ID3()

    # Ставим/заменяем текстовые фреймы.
    tags.setall("TIT2", [TIT2(encoding=3, text=title)])
    tags.setall("TPE1", [TPE1(encoding=3, text=artist)])
    tags.setall("TALB", [TALB(encoding=3, text=album)])

    if comment:
        # lang="eng", desc="comment" — произвольно, но единообразно.
        tags.setall("COMM", [COMM(encoding=3, lang="eng", desc="comment", text=comment)])

    tags.save(str(mp3_path))


# ----------------------------- AzuraCast API -----------------------------

class AzuraAPI:
    def __init__(self, base_url: str, api_key: str, station_id: int, timeout: int = 30):
        self.base = base_url.rstrip("/")
        self.key = api_key
        self.sid = int(station_id)
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"X-API-Key": self.key})
        self.s.verify = True  # если самоподписанный, можно отключить, но лучше настроить TLS

    def _url(self, path: str) -> str:
        return f"{self.base}{path if path.startswith('/') else '/' + path}"

    def get_playlists(self) -> List[Dict[str, Any]]:
        r = self.s.get(self._url(f"/api/station/{self.sid}/playlists"), timeout=self.timeout)
        r.raise_for_status()
        return r.json() if r.content else []

    def get_playlist_id_by_name(self, name: str) -> Optional[int]:
        for p in self.get_playlists():
            if str(p.get("name", "")).strip().casefold() == name.strip().casefold():
                return int(p.get("id"))
        return None

    def upload_file(self, local_path: Path, ac_path: str) -> Dict[str, Any]:
        """
        POST /api/station/{sid}/files/upload?path=/incoming
        """
        params = {"path": ac_path}
        files = {"file": (local_path.name, open(local_path, "rb"), "audio/mpeg")}
        try:
            r = self.s.post(
                self._url(f"/api/station/{self.sid}/files/upload"),
                params=params,
                files=files,
                timeout=max(self.timeout, 120),
            )
        finally:
            files["file"][1].close()
        if r.status_code >= 400:
            raise RuntimeError(f"Azura upload HTTP {r.status_code}: {r.text[:400]}")
        try:
            return r.json()
        except Exception:
            return {"ok": True, "raw": r.text}

    def search_media_by_basename(self, basename: str) -> Optional[Dict[str, Any]]:
        """
        GET /api/station/{sid}/files?search=...
        Находим элемент, чей path/basename заканчивается на basename.
        """
        params = {"search": basename}
        r = self.s.get(self._url(f"/api/station/{self.sid}/files"), params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json() if r.content else []
        if not isinstance(data, list):
            if isinstance(data, dict) and isinstance(data.get("files"), list):
                data = data["files"]
            else:
                return None

        target = basename.strip().casefold()
        for item in data:
            p = str(item.get("path") or item.get("pathname") or item.get("basename") or "").strip()
            if p and p.strip("/").split("/")[-1].casefold() == target:
                return item
        return None

    def set_file_playlists(self, media_id: Any, playlist_ids: List[int]) -> Dict[str, Any]:
        r = self.s.put(
            self._url(f"/api/station/{self.sid}/file/{media_id}"),
            json={"playlists": playlist_ids},
            timeout=self.timeout,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Azura set playlists HTTP {r.status_code}: {r.text[:400]}")
        return r.json() if r.content else {"ok": True}

    def delete_media(self, media_id: Any) -> None:
        r = self.s.delete(self._url(f"/api/station/{self.sid}/file/{media_id}"), timeout=self.timeout)
        if r.status_code not in (200, 204):
            raise RuntimeError(f"Azura delete media HTTP {r.status_code}: {r.text[:300]}")


# ----------------------------- генерация ---------------------------------

def tensor_to_s16le_stereo_bytes(wav: torch.Tensor) -> bytes:
    """
    wav: (B,C,T) float32 [-1..1] @ 32 kHz → interleaved s16le stereo bytes.
    """
    if wav.dim() != 3 or wav.size(0) != 1:
        raise ValueError(f"Unexpected wav shape: {tuple(wav.shape)} (expected (1,C,T))")
    x = wav[0].detach().cpu().clamp(-1, 1).numpy()  # (C,T)
    C, T = x.shape
    if C == 1:
        x = np.repeat(x, 2, axis=0)
        C = 2
    elif C > 2:
        x = x[:2, :]
    i16 = (x * 32767.0).astype(np.int16)
    interleaved = np.empty((i16.shape[1] * 2,), dtype=np.int16)
    interleaved[0::2] = i16[0]
    interleaved[1::2] = i16[1]
    return interleaved.tobytes()


def load_musicgen(model_id: str, device: str) -> MusicGen:
    log("i", f"{device.upper()}; loading model: {model_id} …")
    model = MusicGen.get_pretrained(model_id)
    # MusicGen не имеет .to(..) на верхнем уровне — переносим блоки вручную
    for attr in ("lm", "compression_model", "bwe_model"):
        m = getattr(model, attr, None)
        if m is not None:
            m.to(device)
    return model


def generate_track_mp3(model: MusicGen,
                       device: str,
                       prompt: str,
                       duration: int,
                       bitrate: str) -> bytes:
    model.set_generation_params(duration=duration)
    with torch.inference_mode():
        wav = model.generate(descriptions=[prompt], progress=False)  # (B,C,T) @ 32k
    pcm = tensor_to_s16le_stereo_bytes(wav)
    mp3 = encode_pcm_to_mp3(pcm, src_sr=32000, bitrate=bitrate)
    del wav
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return mp3


# ----------------------------- состояние ---------------------------------

def state_path_for(outdir: Path) -> Path:
    return outdir / ".state.json"


def load_state(path: Path) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        log("w", f"State load failed ({path}): {e}")
        return None


def save_state(path: Path, st: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def ensure_state_defaults(st: dict | None) -> dict:
    """
    Гарантирует обязательные ключи.
    """
    if not isinstance(st, dict):
        st = {}
    st.setdefault("version", 1)
    st.setdefault("fleet", [])               # [{basename, remote_path, media_id, uploaded_at, duration}]
    st.setdefault("remote_index", {})        # reserved
    st.setdefault("playlist_id", None)
    st.setdefault("last_refresh_date", None) # 'YYYY-MM-DD'
    return st


def prune_local(outdir: Path, keep: int) -> None:
    files = sorted(outdir.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
    if keep < 0:
        return
    while len(files) > keep:
        p = files.pop(0)
        try:
            p.unlink(missing_ok=True)
            log("i", f"Local prune: removed {p.name}")
        except Exception as e:
            log("w", f"Local prune failed {p}: {e}")


# ----------------------------- main --------------------------------------

def parse_time_hhmm(s: str) -> dt.time:
    h, m = s.strip().split(":")
    return dt.time(hour=int(h), minute=int(m))


def main():
    ap = argparse.ArgumentParser("MusicGen → AzuraCast (fleet uploader)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default="facebook/musicgen-small")
    ap.add_argument("--duration", type=int, default=120)
    ap.add_argument("--bitrate", default="192k")

    ap.add_argument("--outdir", type=Path, default=Path("./out"))
    ap.add_argument("--local-keep", type=int, default=60)

    ap.add_argument("--fleet-min", type=int, default=40)
    ap.add_argument("--fleet-max", type=int, default=50)

    ap.add_argument("--daily-refresh", type=int, default=10,
                    help="Сколько треков удалить/добавить раз в сутки")
    ap.add_argument("--daily-at", default="03:30", help="Время локальное HH:MM")

    ap.add_argument("--tick-sec", type=int, default=15)

    # AzuraCast
    ap.add_argument("--ac-base-url", required=True)
    ap.add_argument("--ac-api-key", required=True)
    ap.add_argument("--ac-station-id", type=int, required=True)
    ap.add_argument("--ac-path", default="/incoming")
    ap.add_argument("--ac-playlist", required=True)

    # CUDA
    ap.add_argument("--cuda", action="store_true", help="Использовать CUDA при наличии")
    ap.add_argument("--gpu", type=int, default=0)

    args = ap.parse_args()

    # CUDA
    device = "cpu"
    if args.cuda:
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(args.gpu)
                device = "cuda"
            except Exception as e:
                die(f"CUDA requested but not usable: {e}")
        else:
            die("CUDA requested but torch.cuda.is_available()=False")

    # Директории и состояние
    args.outdir.mkdir(parents=True, exist_ok=True)
    st_path = state_path_for(args.outdir)
    st = ensure_state_defaults(load_state(st_path))
    save_state(st_path, st)

    # API
    api = AzuraAPI(args.ac_base_url, args.ac_api_key, args.ac_station_id)
    if not st.get("playlist_id"):
        pl_id = api.get_playlist_id_by_name(args.ac_playlist)
        if not pl_id:
            die(f"Playlist '{args.ac_playlist}' not found in station {args.ac_station_id}")
        st["playlist_id"] = int(pl_id)
        save_state(st_path, st)
        log("i", f"Playlist '{args.ac_playlist}' id={pl_id}")

    # Модель
    model = load_musicgen(args.model, device)

    # Вспомогательное: генерируем и заливаем один трек
    def gen_and_upload_one() -> Optional[Dict[str, Any]]:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{ts}_musicgen_{uuid.uuid4().hex[:6]}.mp3"
        local_mp3 = args.outdir / base
        title = f"MusicGen {ts}"
        comment = args.prompt

        log("i", f"Generating: «{args.prompt}» ({args.duration}s)…")
        try:
            mp3_bytes = generate_track_mp3(model, device, args.prompt, args.duration, args.bitrate)
        except Exception as e:
            log("w", f"Generate failed: {e}")
            return None

        try:
            local_mp3.write_bytes(mp3_bytes)
            write_id3_tags(local_mp3, title=title, comment=comment)
            size_mb = local_mp3.stat().st_size / (1024 * 1024)
            log("i", f"Saved: {local_mp3.name} ({size_mb:.2f} MB)")
        except Exception as e:
            log("w", f"Save/tags failed: {e}")
            return None

        # Upload
        try:
            log("i", f"AzuraCast upload → {args.ac_base_url} (station {args.ac_station_id}), path={args.ac_path}")
            resp = api.upload_file(local_mp3, args.ac_path)
            log("i", f"AzuraCast upload OK: {type(resp).__name__}")
        except Exception as e:
            log("w", f"Azura upload failed: {e}")
            return None

        # Найти media_id по basename
        media = None
        try:
            media = api.search_media_by_basename(local_mp3.name)
        except Exception as e:
            log("w", f"Azura search failed: {e}")

        if not media:
            log("w", "Uploaded file not found in media list (indexing may be delayed). Will retry later.")
            return {
                "basename": local_mp3.name,
                "remote_path": f"{args.ac_path.rstrip('/')}/{local_mp3.name}",
                "media_id": None,
                "uploaded_at": dt.datetime.now().isoformat(timespec="seconds"),
                "duration": args.duration,
            }

        media_id = media.get("id") or media.get("media_id") or media.get("unique_id")
        if not media_id:
            log("w", f"Media found but id missing: {media}")
            return None

        # Привязка к плейлисту
        try:
            api.set_file_playlists(media_id, [int(st["playlist_id"])])
            log("i", f"Added to playlist: {args.ac_playlist} (id={st['playlist_id']})")
        except Exception as e:
            log("w", f"Set playlist failed: {e}")

        return {
            "basename": local_mp3.name,
            "remote_path": f"{args.ac_path.rstrip('/')}/{local_mp3.name}",
            "media_id": media_id,
            "uploaded_at": dt.datetime.now().isoformat(timespec="seconds"),
            "duration": args.duration,
        }

    def try_fill_missing_ids():
        changed = False
        for rec in st["fleet"]:
            if not rec.get("media_id"):
                m = api.search_media_by_basename(rec["basename"])
                if m:
                    media_id = m.get("id") or m.get("media_id") or m.get("unique_id")
                    if media_id:
                        rec["media_id"] = media_id
                        changed = True
        if changed:
            save_state(st_path, st)

    def delete_oldest(n: int):
        if n <= 0 or not st["fleet"]:
            return

        def key_fn(r):
            try:
                return dt.datetime.fromisoformat(r.get("uploaded_at", "1970-01-01T00:00:00"))
            except Exception:
                return dt.datetime(1970, 1, 1)

        st["fleet"].sort(key=key_fn)
        to_del = st["fleet"][:n]
        keep = st["fleet"][n:]

        for rec in to_del:
            media_id = rec.get("media_id")
            if media_id:
                try:
                    api.delete_media(media_id)
                    log("i", f"Deleted remote media id={media_id} ({rec['basename']})")
                except Exception as e:
                    log("w", f"Delete remote failed for {media_id}: {e}")
            # локальный файл
            lp = args.outdir / rec["basename"]
            try:
                lp.unlink(missing_ok=True)
            except Exception:
                pass

        st["fleet"] = keep
        save_state(st_path, st)

    # Основной цикл
    daily_at = parse_time_hhmm(args.daily_at)

    while True:
        try:
            # 1) Дотянуть media_id, если индексатор ещё не успел
            try_fill_missing_ids()

            # 2) Додержать минимум
            while len(st["fleet"]) < args.fleet_min:
                rec = gen_and_upload_one()
                if rec:
                    st["fleet"].append(rec)
                    save_state(st_path, st)
                else:
                    time.sleep(args.tick_sec)
                    break

            # 3) Обрезать до максимума
            if len(st["fleet"]) > args.fleet_max:
                delete_oldest(len(st["fleet"]) - args.fleet_max)

            # 4) Ежедневное освежение
            now = dt.datetime.now()
            today_str = now.strftime("%Y-%m-%d")
            if st.get("last_refresh_date") != today_str and now.time() >= daily_at:
                n = min(args.daily_refresh, len(st["fleet"]))
                log("i", f"Daily refresh: removing {n} oldest …")
                delete_oldest(n)
                st["last_refresh_date"] = today_str
                save_state(st_path, st)
                # дозаполнить до минимума (не более n раз)
                while len(st["fleet"]) < args.fleet_min and n > 0:
                    rec = gen_and_upload_one()
                    if rec:
                        st["fleet"].append(rec)
                        save_state(st_path, st)
                        n -= 1
                    else:
                        break

            # 5) Локальная чистка
            prune_local(args.outdir, args.local_keep)

        except KeyboardInterrupt:
            log("i", "Stopped by user.")
            break
        except Exception as e:
            log("w", f"Loop error: {e}")

        time.sleep(args.tick_sec)


if __name__ == "__main__":
    main()
