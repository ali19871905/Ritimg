#!/usr/bin/env python3
# RITIM4K - terminal rhythm game
# Linux/macOS: pip install pygame librosa numpy soundfile
# Windows:     pip install pygame librosa numpy soundfile windows-curses

from __future__ import annotations

import curses
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pygame

try:
    import librosa
except ImportError:
    print("Eksik paket: librosa\nKur: python -m pip install librosa")
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parent
MUSIC_DIR = ROOT / "musics"
DATA_FILE = ROOT / "player.json"
DEFAULT_KEYS = ["q", "w", " ", "e", "r", "t", "y"]
LANE_COUNTS = (4, 5, 6, 7)
AUDIO_EXTENSIONS = {".mp3", ".ogg", ".wav", ".flac", ".m4a"}
APP_NAME = "RITIM4K"


def load_profile():
    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        return {
            "name": str(data.get("name", "Oyuncu"))[:16] or "Oyuncu",
            "keys": data.get("keys", DEFAULT_KEYS.copy()),
            "lanes": int(data.get("lanes", 4)),
        }
    except (OSError, ValueError, TypeError):
        return {"name": "Oyuncu", "keys": DEFAULT_KEYS.copy(), "lanes": 4}


def save_profile(profile):
    DATA_FILE.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def key_name(key):
    return "SPACE" if key == " " else key.upper()


def scan_music():
    MUSIC_DIR.mkdir(exist_ok=True)
    return sorted(
        [p for p in MUSIC_DIR.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def analyze_song(path: Path, lane_count: int):
    """Extract beat/onset times and map spectral energy to lanes."""
    print("\nŞarkı analiz ediliyor. Uzun şarkılarda biraz sürebilir...")
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    if y.size == 0:
        raise ValueError("Ses dosyası boş.")
    duration = librosa.get_duration(y=y, sr=sr)
    hop = 256
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop,
        backtrack=False, units="frames"
    )
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)

    # Onset yoksa sessizlik/ambient parçalarda enerji tepelerini kullan.
    if len(times) == 0:
        peaks = librosa.util.peak_pick(
            onset_env, pre_max=3, post_max=3, pre_avg=5, post_avg=5,
            delta=0.15, wait=8
        )
        times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)

    # Zayıf ve çok sık vuruşları filtrele; oyunun okunabilirliği artsın.
    strength = onset_env[frames] if len(frames) else np.array([])
    notes = []
    last_time = -1.0
    for i, t in enumerate(times):
        t = float(t)
        if t < 0.35 or t > duration - 0.15 or t - last_time < 0.105:
            continue
        frame = min(int(round(t * sr / hop)), max(0, onset_env.size - 1))
        # Spektral ağırlık merkezi ve chroma, aynı şarkıda lane dağılımını çeşitlendirir.
        start = max(0, int(t * sr) - 2048)
        end = min(y.size, start + 4096)
        segment = y[start:end]
        if segment.size:
            spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size)))
            bands = np.array_split(spectrum, lane_count)
            lane = int(np.argmax([float(np.mean(b)) if b.size else 0 for b in bands]))
        else:
            lane = i % lane_count
        # Ufak deterministik çeşitlilik: aynı frekans bölgesine yığılmayı azaltır.
        if notes and lane == notes[-1]["lane"] and random.random() < 0.22:
            lane = (lane + random.randint(1, lane_count - 1)) % lane_count
        notes.append({"time": t, "lane": lane, "hit": False})
        last_time = t

    if not notes:
        raise ValueError("Bu dosyada oynanabilir ritim bulunamadı.")
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    bpm_value = float(np.asarray(bpm).reshape(-1)[0]) if np.size(bpm) else 0.0
    return notes, duration, bpm_value


def safe_addstr(stdscr, y, x, text, attr=0):
    height, width = stdscr.getmaxyx()
    if 0 <= y < height and x < width:
        try:
            stdscr.addnstr(y, max(0, x), str(text), max(0, width - max(0, x) - 1), attr)
        except curses.error:
            pass


def center(stdscr, y, text, attr=0):
    _, width = stdscr.getmaxyx()
    safe_addstr(stdscr, y, max(0, (width - len(text)) // 2), text, attr)


def wait_key(stdscr, message="Devam etmek için bir tuşa bas"):
    stdscr.nodelay(False)
    safe_addstr(stdscr, curses.LINES - 2, 2, message)
    stdscr.getch()
    stdscr.nodelay(True)


def menu(stdscr, title, options, selected=0):
    stdscr.nodelay(False)
    while True:
        stdscr.erase()
        center(stdscr, 2, APP_NAME, curses.A_BOLD)
        center(stdscr, 4, title, curses.A_BOLD)
        for i, option in enumerate(options):
            attr = curses.A_REVERSE | curses.A_BOLD if i == selected else curses.A_NORMAL
            center(stdscr, 7 + i * 2, option, attr)
        center(stdscr, 7 + len(options) * 2 + 1, "↑/↓: seç   ENTER: onay   ESC: geri")
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("w")):
            selected = (selected - 1) % len(options)
        elif key in (curses.KEY_DOWN, ord("s")):
            selected = (selected + 1) % len(options)
        elif key in (10, 13, curses.KEY_ENTER):
            stdscr.nodelay(True)
            return selected
        elif key == 27:
            stdscr.nodelay(True)
            return None


def choose_song(stdscr):
    songs = scan_music()
    if not songs:
        stdscr.erase()
        center(stdscr, 4, "'musics' klasöründe şarkı bulunamadı.")
        center(stdscr, 6, "MP3, OGG, WAV, FLAC veya M4A dosyası ekle.")
        wait_key(stdscr)
        return None
    index = 0
    stdscr.nodelay(False)
    while True:
        stdscr.erase()
        center(stdscr, 2, "ŞARKI SEÇ", curses.A_BOLD)
        for i, song in enumerate(songs):
            label = song.name
            attr = curses.A_REVERSE if i == index else curses.A_NORMAL
            center(stdscr, 5 + i, label, attr)
        center(stdscr, min(curses.LINES - 2, 7 + len(songs)), "↑/↓ seç | ENTER onay | ESC geri")
        key = stdscr.getch()
        if key == curses.KEY_UP:
            index = (index - 1) % len(songs)
        elif key == curses.KEY_DOWN:
            index = (index + 1) % len(songs)
        elif key in (10, 13, curses.KEY_ENTER):
            stdscr.nodelay(True)
            return songs[index]
        elif key == 27:
            stdscr.nodelay(True)
            return None


def edit_name(stdscr, profile):
    curses.echo()
    stdscr.nodelay(False)
    stdscr.erase()
    center(stdscr, 3, "Yeni oyuncu ismini yaz (en fazla 16 karakter):")
    safe_addstr(stdscr, 6, 4, "> ")
    try:
        value = stdscr.getstr(6, 6, 16).decode("utf-8", "ignore").strip()
    except curses.error:
        value = ""
    curses.noecho()
    stdscr.nodelay(True)
    if value:
        profile["name"] = value[:16]
        save_profile(profile)


def edit_keys(stdscr, profile):
    count = profile["lanes"]
    keys = profile["keys"][:]
    while len(keys) < 7:
        keys.append(DEFAULT_KEYS[len(keys)])
    for lane in range(count):
        stdscr.erase()
        center(stdscr, 3, f"{lane + 1}. şerit için yeni tuşa bas", curses.A_BOLD)
        center(stdscr, 5, "SPACE kullanmak için boşluk tuşuna bas.")
        center(stdscr, 7, "ESC: iptal")
        stdscr.nodelay(False)
        key = stdscr.getch()
        if key == 27:
            stdscr.nodelay(True)
            return
        char = " " if key == 32 else chr(key).lower() if 0 <= key < 256 else ""
        if not char or char in [k for j, k in enumerate(keys) if j != lane] or char in ("\n", "\r"):
            center(stdscr, 10, "Bu tuş kullanılamaz veya zaten seçilmiş. Tekrar dene.")
            stdscr.getch()
            continue
        keys[lane] = char
    profile["keys"] = keys
    save_profile(profile)
    stdscr.nodelay(True)


def choose_lanes(stdscr, profile):
    options = ["4K", "5K", "6K", "7K"]
    selected = LANE_COUNTS.index(profile["lanes"]) if profile["lanes"] in LANE_COUNTS else 0
    result = menu(stdscr, "ŞERİT SAYISI", options, selected)
    if result is not None:
        profile["lanes"] = LANE_COUNTS[result]
        save_profile(profile)


def draw_game(stdscr, profile, song, notes, duration, bpm):
    lane_count = profile["lanes"]
    keys = profile["keys"][:lane_count]
    if len(set(keys)) != len(keys):
        raise ValueError("Tuş atamaları benzersiz olmalı. Menüden tuşları değiştir.")
    score = combo = max_combo = misses = 0
    health = 100.0
    judged = 0
    hit_line = min(curses.LINES - 5, max(10, curses.LINES - 7))
    top = 5
    travel_time = 1.65
    pygame.mixer.music.load(str(song))
    pygame.mixer.music.play()
    start = time.monotonic()
    stdscr.nodelay(True)
    stdscr.keypad(True)

    while True:
        now = time.monotonic() - start
        if now >= duration + 1.0 or health <= 0:
            break
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        hit_line = min(height - 5, max(10, height - 7))
        center(stdscr, 0, f"{profile['name']}  |  PUAN: {score}  |  CAN: {int(health)}%", curses.A_BOLD)
        safe_addstr(stdscr, 1, 2, f"{song.name[:max(10, width-35)]}  |  {bpm:.0f} BPM")
        safe_addstr(stdscr, 2, 2, f"COMBO: {combo}   EN İYİ: {max_combo}   KAÇAN: {misses}")
        # Sağlık çubuğu
        bar_len = min(30, max(10, width - 12))
        filled = int(bar_len * health / 100)
        safe_addstr(stdscr, 3, 2, "CAN [" + "#" * filled + "-" * (bar_len - filled) + "]")

        lane_width = max(3, min(8, (width - 4) // lane_count))
        total_width = lane_width * lane_count
        left = max(1, (width - total_width) // 2)
        for lane in range(lane_count):
            x = left + lane * lane_width
            safe_addstr(stdscr, top - 1, x, f"{key_name(keys[lane]):^{lane_width}}")
            for y in range(top, hit_line):
                safe_addstr(stdscr, y, x + lane_width - 1, "|")
            safe_addstr(stdscr, hit_line, x, "=" * lane_width, curses.A_BOLD)
            safe_addstr(stdscr, hit_line + 1, x, f"{key_name(keys[lane]):^{lane_width}}", curses.A_BOLD)

        for note in notes:
            if note["hit"]:
                continue
            dt = note["time"] - now
            if dt < -0.18:
                note["hit"] = True
                misses += 1
                combo = 0
                health = max(0, health - 9)
                judged += 1
                continue
            if 0 <= dt <= travel_time:
                y = hit_line - int((dt / travel_time) * (hit_line - top))
                x = left + note["lane"] * lane_width + lane_width // 2
                safe_addstr(stdscr, y, x, "●", curses.A_BOLD)

        pressed = []
        while True:
            key = stdscr.getch()
            if key == -1:
                break
            if key in (ord("q"), 27) and key not in [ord(k) for k in keys if k != " "]:
                if key == 27:
                    pygame.mixer.music.stop()
                    return None
            if key == 32:
                pressed.append(" ")
            elif 0 <= key < 256:
                pressed.append(chr(key).lower())

        for char in pressed:
            if char not in keys:
                continue
            lane = keys.index(char)
            candidates = [
                n for n in notes if not n["hit"] and n["lane"] == lane
                and abs(n["time"] - now) <= 0.19
            ]
            if candidates:
                note = min(candidates, key=lambda n: abs(n["time"] - now))
                delta = abs(note["time"] - now)
                note["hit"] = True
                judged += 1
                if delta <= 0.055:
                    points, label = 300, "PERFECT"
                    health = min(100, health + 1.2)
                elif delta <= 0.115:
                    points, label = 150, "GREAT"
                    health = min(100, health + 0.5)
                else:
                    points, label = 50, "GOOD"
                score += points + combo * 2
                combo += 1
                max_combo = max(max_combo, combo)
                safe_addstr(stdscr, hit_line - 2, 2, f"{label} +{points}", curses.A_BOLD)
            else:
                # Boş tuşa basmak küçük hata; oyunu tek hamlede bitirmez.
                health = max(0, health - 2)
                combo = 0
        stdscr.refresh()
        time.sleep(0.012)

    pygame.mixer.music.stop()
    stdscr.nodelay(False)
    stdscr.erase()
    center(stdscr, 2, "OYUN BİTTİ" if health > 0 else "CANIN BİTTİ!", curses.A_BOLD)
    center(stdscr, 5, f"Oyuncu: {profile['name']}")
    center(stdscr, 7, f"Şarkı: {song.name}")
    center(stdscr, 9, f"PUAN: {score}")
    center(stdscr, 10, f"En iyi combo: {max_combo}")
    center(stdscr, 11, f"Kaçan nota: {misses}")
    center(stdscr, 13, f"İsabet oranı: {100 * (judged - misses) / max(1, judged):.1f}%")
    wait_key(stdscr)
    stdscr.nodelay(True)
    return score


def main(stdscr):
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    pygame.mixer.init()
    curses.curs_set(0)
    profile = load_profile()
    selected_song = None

    while True:
        options = [
            "Oyna",
            "Şarkı seç" + (f" [{selected_song.name[:20]}]" if selected_song else ""),
            "İsim değiştir",
            "Tuşları değiştir",
            "Şerit sayısı (şu an %dK)" % profile["lanes"],
            "Çıkış",
        ]
        choice = menu(stdscr, f"Hoş geldin, {profile['name']}!", options)
        if choice is None or choice == 5:
            break
        if choice == 1:
            selected_song = choose_song(stdscr)
        elif choice == 2:
            edit_name(stdscr, profile)
        elif choice == 3:
            edit_keys(stdscr, profile)
        elif choice == 4:
            choose_lanes(stdscr, profile)
        elif choice == 0:
            if selected_song is None:
                selected_song = choose_song(stdscr)
            if selected_song is None:
                continue
            try:
                notes, duration, bpm = analyze_song(selected_song, profile["lanes"])
                draw_game(stdscr, profile, selected_song, notes, duration, bpm)
            except Exception as exc:
                stdscr.erase()
                center(stdscr, 4, "Oyun başlatılamadı:")
                center(stdscr, 6, str(exc)[:max(10, curses.COLS - 4)])
                wait_key(stdscr, "Devam etmek için bir tuşa bas")
    pygame.mixer.quit()
    pygame.quit()


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            import windows_curses  # noqa: F401
        except ImportError:
            print("Windows için: pip install windows-curses")
            raise SystemExit(1)
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pygame.mixer.music.stop()
        pygame.quit()
