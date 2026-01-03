import pygame
import sys
import random
import cv2
import mediapipe as mp
import sounddevice as sd
import numpy as np
from statistics import mean, stdev


pygame.init()

# ======================
# Screen & game assets
# ======================
WIDTH = 1270
HEIGHT = 720
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()
FPS = 60

BACKGROUND = pygame.image.load("background.png").convert_alpha()
BIRD = pygame.image.load("bird.png").convert_alpha()
PIPE = pygame.image.load("pipe.png").convert_alpha()
ROTATED_PIPE = pygame.image.load("rotated_pipe.png").convert_alpha()

POINT_SOUND = pygame.mixer.Sound("sfx_point.wav")
HIT_SOUND = pygame.mixer.Sound("sfx_hit.wav")

pygame.display.set_caption("Flappy Bird")

# ======================
# Pipe gap settings
# ======================
PIPE_GAP = 220  # vertical distance between upper and lower pipe (tune this)
GAP_TOP_MARGIN = 150    # how far from top we allow the gap
GAP_BOTTOM_MARGIN = 150  # how far from bottom we allow the gap


# ============================
# Hand Gesture Controller
# ============================
class HandGestureController:
    """
    Simple hand-gesture flap detector using MediaPipe Hands.
    Rule: if index finger tip is above a vertical threshold -> flap once.
    """

    def __init__(self, camera_index=0, y_threshold_ratio=0.4, cooldown_frames=10):
        self.cap = cv2.VideoCapture(camera_index)
        self.y_threshold_ratio = y_threshold_ratio
        self.cooldown_frames = cooldown_frames
        self.cooldown_counter = 0

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def should_flap(self):
        """Return True if a 'flap' gesture is detected this frame."""
        try:
            if not self.cap.isOpened():
                return False

            # simple cooldown so it doesn't spam flaps continuously
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                ret, _ = self.cap.read()
                return False

            ret, frame = self.cap.read()
            if not ret:
                return False

            # Mirror the frame like a selfie camera
            debug_frame = cv2.flip(frame, 1)

            # Show webcam window so user can see their hand
            cv2.imshow("Raise your hand to flap", debug_frame)
            cv2.waitKey(1)  # non-blocking; lets the window refresh

            # Use the same flipped frame for detection
            frame = debug_frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                index_tip = hand_landmarks.landmark[
                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP
                ]

                # Normalized y: 0 = top, 1 = bottom
                if index_tip.y < self.y_threshold_ratio:
                    self.cooldown_counter = self.cooldown_frames
                    return True

            return False

        except Exception as e:
            print("Hand gesture error:", e)
            return False

    def release(self):
        self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()


# ============================
# Voice Controller
# ============================
class VoiceController:
    def __init__(self, sensitivity=3.0):
        self.sensitivity = sensitivity
        self.current_volume = 0.0

        self.stream = sd.InputStream(
            channels=1,
            samplerate=44100,
            callback=self.audio_callback
        )
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
        self.current_volume = float(volume_norm)

    def get_volume(self):
        """
        Returns a scaled volume factor, range typically 0 → ~1+
        Adjusted by 'sensitivity'.
        """
        return min(self.current_volume * self.sensitivity, 1.5)

    def release(self):
        try:
            self.stream.stop()
            self.stream.close()
        except:
            pass

class EvalTracker:
    """
    In-memory evaluation tracker (prints to terminal, no CSV).
    Logs score + survival time for each run.
    Logs flap_count only for keyboard/hand (voice is continuous control).
    """
    def __init__(self, mode: str, target_runs: int = 0, auto_stop: bool = True):
        self.mode = mode
        self.target_runs = max(0, int(target_runs))
        self.auto_stop = auto_stop
        self.results = []  # list of dicts: {"score": int, "survival_s": float, "flaps": int|None}
        self._start_ticks = None
        self._flaps = 0

    def start_episode(self):
        self._start_ticks = pygame.time.get_ticks()
        self._flaps = 0

    def on_flap(self):
        self._flaps += 1

    def end_episode(self, score: int):
        end_ticks = pygame.time.get_ticks()
        survival_s = 0.0
        if self._start_ticks is not None:
            survival_s = (end_ticks - self._start_ticks) / 1000.0

        flaps = self._flaps if self.mode in ("keyboard", "hand") else None

        self.results.append({
            "score": int(score),
            "survival_s": float(survival_s),
            "flaps": flaps
        })

        self.print_last_and_summary()

    def _safe_stdev(self, xs):
        return stdev(xs) if len(xs) >= 2 else 0.0

    def print_last_and_summary(self):
        n = len(self.results)
        last = self.results[-1]

        scores = [r["score"] for r in self.results]
        survivals = [r["survival_s"] for r in self.results]

        avg_score = mean(scores)
        sd_score = self._safe_stdev(scores)
        best_score = max(scores)

        avg_surv = mean(survivals)
        sd_surv = self._safe_stdev(survivals)

        # Print last run
        line = f"[EVAL] Run {n}"
        if self.target_runs > 0:
            line += f"/{self.target_runs}"
        line += f" | mode={self.mode} | score={last['score']} | survival={last['survival_s']:.2f}s"
        if last["flaps"] is not None:
            line += f" | flaps={last['flaps']}"
        print(line)

        # Print running summary
        print(
            f"[EVAL] Summary ({n} runs) | "
            f"score: mean={avg_score:.2f}, sd={sd_score:.2f}, best={best_score} | "
            f"survival: mean={avg_surv:.2f}s, sd={sd_surv:.2f}s"
        )

    def should_stop(self) -> bool:
        return self.auto_stop and self.target_runs > 0 and len(self.results) >= self.target_runs

# ============================
# Game
# ============================
class Game:
    def __init__(self, control_mode="keyboard", eval_runs=0, auto_stop_eval=True):
        """
        control_mode: "keyboard", "hand", or "voice"
        """
        self.control_mode = control_mode
        # ===== Evaluation (terminal output) =====
        self.eval = None
        if int(eval_runs) > 0:
            self.eval = EvalTracker(mode=self.control_mode, target_runs=int(eval_runs), auto_stop=auto_stop_eval)

        self.hand_controller = None
        self.voice_controller = None

        if self.control_mode == "hand":
            self.hand_controller = HandGestureController()
        elif self.control_mode == "voice":
            self.voice_controller = VoiceController()

        self.reset_game()
        if self.eval is not None:
            self.eval.start_episode()

    @staticmethod
    def generate_pipe_pair():
        """
        Generate one pair of pipes (upper & lower) with a constant vertical gap.
        Returns: (lower_y, upper_y)
        """
        # Choose gap's top y position
        min_gap_y = GAP_TOP_MARGIN
        max_gap_y = HEIGHT - GAP_BOTTOM_MARGIN - PIPE_GAP
        gap_y = random.randrange(min_gap_y, max_gap_y)

        upper_y = gap_y - ROTATED_PIPE.get_height()
        lower_y = gap_y + PIPE_GAP

        return lower_y, upper_y

    def reset_game(self):
        self.game_on = True
        self.bird_x = 100
        self.bird_y = 100

        self.pipes_x = [WIDTH + i * 200 for i in range(7)]
        self.lower_pipe_y = []
        self.upper_pipe_y = []

        for _ in range(7):
            low, up = self.generate_pipe_pair()
            self.lower_pipe_y.append(low)
            self.upper_pipe_y.append(up)

        self.gravity = 0
        self.pipe_velocity = 0
        self.flap = 0
        self.score = 0
        self.rotate_angle = 0
        self.is_game_over = False
        self.play_sound = True

    def moving_pipes(self):
        for i in range(7):
            self.pipes_x[i] -= self.pipe_velocity

        # When a pipe goes off screen, respawn it ahead with a new gap (same size)
        for i in range(7):
            if self.pipes_x[i] < -50:
                self.pipes_x[i] = WIDTH + 100
                low, up = self.generate_pipe_pair()
                self.lower_pipe_y[i] = low
                self.upper_pipe_y[i] = up

    def flapping(self):
        self.bird_y += self.gravity
        if not self.is_game_over:
            self.flap -= 1
            self.bird_y -= self.flap

    def is_collide(self):
        for i in range(7):
            if (
                self.bird_x >= self.pipes_x[i]
                and self.bird_x <= self.pipes_x[i] + PIPE.get_width()
                and (
                    self.bird_y + BIRD.get_height() - 15 >= self.lower_pipe_y[i]
                    or self.bird_y
                    <= self.upper_pipe_y[i] + ROTATED_PIPE.get_height() - 15
                )
            ):
                return True

            if (
                self.bird_x == self.pipes_x[i]
                and self.bird_y <= self.lower_pipe_y[i]
                and self.bird_y >= self.upper_pipe_y[i]
            ):
                if not self.is_game_over:
                    self.score += 1
                    pygame.mixer.Sound.play(POINT_SOUND)

        if self.bird_y <= 0:
            return True

        if self.bird_y + BIRD.get_height() >= HEIGHT:
            self.gravity = 0
            return True

        return False

    def game_over(self):
        # (A) Collision happens -> set game over ONCE + print eval ONCE
        if (not self.is_game_over) and self.is_collide():
            self.is_game_over = True
            self.pipe_velocity = 0
            self.flap = 0
            self.rotate_angle = -90

            # play sound once
            if self.play_sound:
                pygame.mixer.Sound.play(HIT_SOUND)
                self.play_sound = False

            # ✅ evaluation print happens here (ONLY ONCE)
            if self.eval is not None:
                self.eval.end_episode(self.score)

                # if you enabled auto-stop after N runs
                if self.eval.should_stop():
                    print("[EVAL] Target runs reached. Exiting evaluation.")
                    self.game_on = False

        # (B) While game over -> keep showing text every frame
        if self.is_game_over:
            self.display_text("Game Over!", (255, 255, 255), 450, 300, 84, "Fixedsys", bold=True)
            self.display_text("Press Enter To Play Again", (255, 255, 255), 400, 600, 48, "Fixedsys", bold=True)

    @staticmethod
    def display_text(text, color, x, y, size, style, bold=False):
        font = pygame.font.SysFont(style, size, bold=bold)
        screen_text = font.render(text, True, color)
        SCREEN.blit(screen_text, (x, y))

    def main_game(self):
        while self.game_on:
            flap_trigger = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if self.hand_controller is not None:
                        self.hand_controller.release()
                    if self.voice_controller is not None:
                        self.voice_controller.release()
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if not self.is_game_over:
                            flap_trigger = True  # keyboard
                    if event.key == pygame.K_RETURN:
                        self.reset_game()

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        self.rotate_angle = 0

            # Hand gesture input
            if self.control_mode == "hand" and not self.is_game_over:
                if self.hand_controller.should_flap():
                    flap_trigger = True

            # Voice input
            if self.control_mode == "voice" and not self.is_game_over:
                volume = self.voice_controller.get_volume()  # 0.0 → 1.5 typically

                # apply physics proportional to volume
                self.pipe_velocity = 5  # always move forward
                self.gravity = 8 * (1 - volume)  # silence -> strong fall, loud -> less fall
                self.flap = 25 * volume  # loud -> strong upward push
                self.rotate_angle = 10 * volume  # tilt based on volume

                # prevent falling too fast
                if self.gravity < -5:
                    self.gravity = -5

            # Apply flap if triggered
            if flap_trigger and not self.is_game_over:
                if self.eval is not None:
                    self.eval.on_flap()

                self.pipe_velocity = 5
                self.gravity = 10
                self.flap = 20
                self.rotate_angle = 15

            # Drawing
            SCREEN.blit(BACKGROUND, (0, 0))

            for i in range(7):
                SCREEN.blit(PIPE, (self.pipes_x[i], self.lower_pipe_y[i]))
                SCREEN.blit(ROTATED_PIPE, (self.pipes_x[i], self.upper_pipe_y[i]))

            SCREEN.blit(
                pygame.transform.rotozoom(BIRD, self.rotate_angle, 0.7),
                (self.bird_x, self.bird_y),
            )

            self.moving_pipes()
            self.flapping()
            self.game_over()
            self.display_text(
                str(self.score), (255, 255, 255), 600, 50, 68, "Fixedsys", bold=True
            )

            pygame.display.update()
            CLOCK.tick(FPS)

        # cleanup when loop exits
        if self.hand_controller is not None:
            self.hand_controller.release()
        if self.voice_controller is not None:
            self.voice_controller.release()


if __name__ == "__main__":
    mode = input("Choose mode: (k)eyboard, (h)and, or (v)oice? ").strip().lower()

    if mode == "h":
        control_mode = "hand"
    elif mode == "v":
        control_mode = "voice"
    else:
        control_mode = "keyboard"

    runs_str = input("How many evaluation runs? (0 = free play): ").strip()
    try:
        eval_runs = int(runs_str) if runs_str else 0
    except ValueError:
        eval_runs = 0

    flappy_bird = Game(control_mode=control_mode, eval_runs=eval_runs, auto_stop_eval=True)
    flappy_bird.main_game()

