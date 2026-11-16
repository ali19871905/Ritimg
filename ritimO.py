from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pygame
import librosa

ROOT = Path(__file__).resolve().parent
MUSIC_DIR = ROOT / "musics"
SOUND_DIR = ROOT / "sounds"
DATA_FILE = ROOT / "player.json"
AUDIO_EXTENSIONS = {".mp3", ".ogg", ".wav", ".flac", ".m4a"}

WIDTH, HEIGHT = 1100, 720
FPS = 120
BG = (18, 20, 31)
WHITE = (240, 243, 250)
MUTED = (155, 165, 185)
CYAN = (70, 210, 235)
PINK = (255, 105, 155)
GREEN = (105, 230, 155)
RED = (255, 85, 95)
YELLOW = (255, 220, 105)


def load_profile():
    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {"name": str(data.get("name", "Oyuncu"))[:16] or "Oyuncu"}


def save_profile(profile):
    DATA_FILE.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def scan_music():
    MUSIC_DIR.mkdir(exist_ok=True)
    return sorted(
        (p for p in MUSIC_DIR.iterdir()
         if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS),
        key=lambda p: p.name.lower()
    )


def load_sounds():
    SOUND_DIR.mkdir(exist_ok=True)
    result = {}
    for name in ("true", "false"):
        p = SOUND_DIR / f"{name}.mp3"
        try:
            result[name] = pygame.mixer.Sound(str(p)) if p.exists() else None
        except pygame.error:
            result[name] = None
    return result


def effect(sounds, name):
    snd = sounds.get(name)
    if snd:
        try:
            snd.play()
        except pygame.error:
            pass


def analyze_song(path):
    print("Şarkı analiz ediliyor...")
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    if not y.size:
        raise ValueError("Ses dosyası boş.")
    duration = librosa.get_duration(y=y, sr=sr)
    hop = 256
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(
        onset_envelope=onset, sr=sr, hop_length=hop, units="frames"
    )
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    if not len(times):
        peaks = librosa.util.peak_pick(
            onset, pre_max=3, post_max=3, pre_avg=5, post_avg=5,
            delta=0.15, wait=8
        )
        times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)

    notes = []
    last = -1.0
    for t in times:
        t = float(t)
        if t < 0.6 or t > duration - 0.2 or t - last < 0.24:
            continue
        x = random.randint(115, WIDTH - 115)
        y_pos = random.randint(145, HEIGHT - 125)
        if notes:
            prev = notes[-1]
            for _ in range(12):
                x = random.randint(115, WIDTH - 115)
                y_pos = random.randint(145, HEIGHT - 125)
                if math.hypot(x - prev["x"], y_pos - prev["y"]) > 145:
                    break
        notes.append({"time": t, "x": x, "y": y_pos, "hit": False})
        last = t

    if not notes:
        raise ValueError("Bu şarkıda oynanabilir ritim bulunamadı.")
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    bpm = float(np.asarray(bpm).reshape(-1)[0]) if np.size(bpm) else 0
    return notes, duration, bpm


def text(surface, font, value, pos, color=WHITE, center=False):
    image = font.render(str(value), True, color)
    rect = image.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surface.blit(image, rect)
    return rect


def button(screen, font, rect, label, mouse, clicked):
    hovered = rect.collidepoint(mouse)
    color = (57, 78, 103) if hovered else (36, 47, 66)
    pygame.draw.rect(screen, color, rect, border_radius=12)
    pygame.draw.rect(screen, CYAN if hovered else (75, 91, 115), rect, 2, border_radius=12)
    text(screen, font, label, rect.center, WHITE, True)
    return hovered and clicked


def draw_menu(screen, fonts, profile, selected_song, mouse, clicked):
    screen.fill(BG)
    title, normal, small = fonts
    text(screen, title, "RITIM O", (WIDTH // 2, 105), CYAN, True)
    text(screen, small, "Fareyle ritmi yakala!", (WIDTH // 2, 150), MUTED, True)
    text(screen, normal, f"Oyuncu: {profile['name']}", (WIDTH // 2, 215), WHITE, True)
    song_label = selected_song.name if selected_song else "Henüz şarkı seçilmedi"
    text(screen, small, f"Şarkı: {song_label[:55]}", (WIDTH // 2, 260), MUTED, True)

    actions = {}
    actions["play"] = button(screen, normal, pygame.Rect(390, 315, 320, 55), "Oyna", mouse, clicked)
    actions["song"] = button(screen, normal, pygame.Rect(390, 385, 320, 55), "Şarkı seç", mouse, clicked)
    actions["name"] = button(screen, normal, pygame.Rect(390, 455, 320, 55), "İsim değiştir", mouse, clicked)
    actions["quit"] = button(screen, normal, pygame.Rect(390, 525, 320, 55), "Çıkış", mouse, clicked)
    text(screen, small, "Hedef zamanı gelince daireye tıkla. ESC: menü", (WIDTH // 2, 640), MUTED, True)
    return actions


def choose_song_screen(screen, fonts, songs, mouse, clicked):
    screen.fill(BG)
    title, normal, small = fonts
    text(screen, title, "ŞARKI SEÇ", (WIDTH // 2, 75), CYAN, True)
    if not songs:
        text(screen, normal, "musics klasöründe şarkı bulunamadı.", (WIDTH // 2, 250), WHITE, True)
        text(screen, small, "MP3 / OGG / WAV / FLAC / M4A ekleyebilirsin.", (WIDTH // 2, 295), MUTED, True)
    rows = []
    for i, song in enumerate(songs[:9]):
        rect = pygame.Rect(180, 145 + i * 49, 740, 40)
        pygame.draw.rect(screen, (37, 49, 68), rect, border_radius=8)
        text(screen, small, song.name[:90], (rect.x + 14, rect.centery), WHITE)
        rows.append((rect, song))
    back = button(screen, normal, pygame.Rect(440, 630, 220, 48), "Geri", mouse, clicked)
    chosen = None
    if clicked:
        for rect, song in rows:
            if rect.collidepoint(mouse):
                chosen = song
    return chosen, back


def play_game(screen, clock, fonts, profile, song, sounds):
    title, normal, small = fonts
    notes, duration, bpm = analyze_song(song)
    pygame.mixer.music.load(str(song))
    pygame.mixer.music.play()
    start = time.monotonic()
    score = combo = best_combo = misses = 0
    health = 100.0
    feedback = ""
    feedback_until = 0
    approach = 1.14
    hit_window = 0.19
    running = True

    while running:
        now = time.monotonic() - start
        mouse = pygame.mouse.get_pos()
        clicked = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.mixer.music.stop()
                return "quit", score
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.mixer.music.stop()
                return "menu", score
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked = True

        if now > duration + 1.4 or health <= 0:
            break

        screen.fill(BG)
        pygame.draw.rect(screen, (34, 42, 59), (35, 95, WIDTH - 70, HEIGHT - 135), 2, border_radius=14)
        for x in range(85, WIDTH, 80):
            pygame.draw.line(screen, (25, 31, 46), (x, 105), (x, HEIGHT - 50), 1)

        text(screen, normal, profile["name"], (55, 28), WHITE)
        text(screen, normal, f"PUAN {score:,}", (WIDTH // 2, 30), YELLOW, True)
        text(screen, small, f"COMBO {combo}  |  EN İYİ {best_combo}  |  KAÇAN {misses}", (WIDTH - 330, 34), WHITE)
        pygame.draw.rect(screen, (50, 57, 72), (55, 65, 210, 13), border_radius=7)
        pygame.draw.rect(screen, GREEN if health > 35 else RED, (55, 65, int(210 * health / 100), 13), border_radius=7)
        text(screen, small, f"{bpm:.0f} BPM", (WIDTH - 100, 65), MUTED)

        for note in notes:
            if note["hit"]:
                continue
            delta = note["time"] - now
            if delta < -hit_window:
                note["hit"] = True
                misses += 1
                combo = 0
                health = max(0, health - 8)
                feedback = "MISS"
                feedback_until = now + 0.35
                effect(sounds, "false")
                continue
            if 0 <= delta <= approach:
                radius = 34
                outer = radius + int((delta / approach) * 105)
                pygame.draw.circle(screen, (105, 135, 165), (note["x"], note["y"]), outer, 3)
                pygame.draw.circle(screen, CYAN, (note["x"], note["y"]), radius)
                pygame.draw.circle(screen, WHITE, (note["x"], note["y"]), radius, 3)
                pygame.draw.circle(screen, (22, 40, 58), (note["x"], note["y"]), 8)

        if clicked:
            candidates = [
                n for n in notes if not n["hit"]
                and abs(n["time"] - now) <= hit_window
                and math.hypot(mouse[0] - n["x"], mouse[1] - n["y"]) <= 43
            ]
            if candidates:
                note = min(candidates, key=lambda n: abs(n["time"] - now))
                delta = abs(note["time"] - now)
                note["hit"] = True
                if delta <= 0.065:
                    points, feedback = 300, "PERFECT!"
                    health = min(100, health + 1.4)
                elif delta <= 0.125:
                    points, feedback = 150, "GREAT!"
                    health = min(100, health + 0.6)
                else:
                    points, feedback = 50, "GOOD!"
                score += points + combo * 2
                combo += 1
                best_combo = max(best_combo, combo)
                effect(sounds, "true")
            else:
                health = max(0, health - 2)
                combo = 0
                feedback = "BOŞ TIKLAMA"
                effect(sounds, "false")
            feedback_until = now + 0.4

        if now < feedback_until:
            text(screen, title, feedback, (WIDTH // 2, HEIGHT - 78), GREEN if "PERFECT" in feedback or "GREAT" in feedback or "GOOD" in feedback else RED, True)

        text(screen, small, "Hedef daireye, yaklaşma halkası küçülürken tıkla  |  ESC: menü",
             (WIDTH // 2, HEIGHT - 25), MUTED, True)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.mixer.music.stop()
    screen.fill(BG)
    text(screen, title, "OYUN BİTTİ" if health > 0 else "CANIN BİTTİ!", (WIDTH // 2, 150), CYAN, True)
    text(screen, normal, f"Oyuncu: {profile['name']}", (WIDTH // 2, 250), WHITE, True)
    text(screen, normal, f"PUAN: {score:,}", (WIDTH // 2, 315), YELLOW, True)
    text(screen, normal, f"En iyi combo: {best_combo}", (WIDTH // 2, 365), WHITE, True)
    text(screen, normal, f"Kaçan nota: {misses}", (WIDTH // 2, 410), WHITE, True)
    text(screen, small, "Bir tuşa basarak ana menüye dön.", (WIDTH // 2, 540), MUTED, True)
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit", score
            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                waiting = False
        clock.tick(30)
    return "menu", score


def main():
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    try:
        pygame.mixer.init()
    except pygame.error:
        pass
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("RITIM O")
    clock = pygame.time.Clock()
    fonts = (
        pygame.font.SysFont("DejaVu Sans", 48, bold=True),
        pygame.font.SysFont("DejaVu Sans", 25, bold=True),
        pygame.font.SysFont("DejaVu Sans", 17),
    )
    profile = load_profile()
    sounds = load_sounds()
    songs = scan_music()
    selected_song = None
    state = "menu"
    name_input = False
    name_text = profile["name"]

    while state != "quit":
        mouse = pygame.mouse.get_pos()
        clicked = False
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                state = "quit"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked = True
            elif event.type == pygame.KEYDOWN:
                if name_input:
                    if event.key == pygame.K_RETURN:
                        if name_text.strip():
                            profile["name"] = name_text.strip()[:16]
                            save_profile(profile)
                        name_input = False
                    elif event.key == pygame.K_ESCAPE:
                        name_input = False
                    elif event.key == pygame.K_BACKSPACE:
                        name_text = name_text[:-1]
                    elif event.unicode.isprintable() and len(name_text) < 16:
                        name_text += event.unicode
                elif event.key == pygame.K_ESCAPE and state != "menu":
                    state = "menu"

        if state == "menu":
            actions = draw_menu(screen, fonts, profile, selected_song, mouse, clicked)
            if actions["play"]:
                if selected_song:
                    state = "game"
                else:
                    state = "songs"
            elif actions["song"]:
                songs = scan_music()
                state = "songs"
            elif actions["name"]:
                name_text = profile["name"]
                name_input = True
            elif actions["quit"]:
                state = "quit"
            if name_input:
                pygame.draw.rect(screen, (35, 47, 66), (290, 605, 520, 55), border_radius=10)
                text(screen, fonts[2], "Yeni isim (ENTER ile kaydet): " + name_text, (310, 623), WHITE)
        elif state == "songs":
            chosen, back = choose_song_screen(screen, fonts, songs, mouse, clicked)
            if chosen:
                selected_song = chosen
                state = "menu"
            elif back:
                state = "menu"
        elif state == "game":
            try:
                state, _ = play_game(screen, clock, fonts, profile, selected_song, sounds)
            except Exception as exc:
                pygame.mixer.music.stop()
                screen.fill(BG)
                text(screen, fonts[1], "Oyun başlatılamadı:", (WIDTH // 2, 270), RED, True)
                text(screen, fonts[2], str(exc)[:100], (WIDTH // 2, 320), WHITE, True)
                text(screen, fonts[2], "Ana menüye dönmek için bir tuşa bas.", (WIDTH // 2, 390), MUTED, True)
                pygame.display.flip()
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                            waiting = False
                        elif event.type == pygame.QUIT:
                            waiting = False
                            state = "quit"
                    clock.tick(30)
                if state != "quit":
                    state = "menu"
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
