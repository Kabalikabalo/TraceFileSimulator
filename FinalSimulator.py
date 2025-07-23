import re
import cv2
import numpy as np


# === Parameters ===
FRAME_INTERVAL_MS = 50
FPS = 1000 // FRAME_INTERVAL_MS
WIDTH, HEIGHT = 1280, 960

# === Map Settings ===
draw_scale = 1
map_base_size = 600
map_size = map_base_size * draw_scale
map_center = np.array([map_size // 2, map_size // 2], dtype=np.float32)
position = map_center.copy()
heading = 0.0
path_points = [position.copy()]

# === Load Steering Wheel Image ===
wheel_img = cv2.imread("steering_wheel.png", cv2.IMREAD_UNCHANGED)
if wheel_img is None:
    raise FileNotFoundError("Missing steering_wheel.png")
wheel_img = cv2.resize(wheel_img, (400, 400), interpolation=cv2.INTER_CUBIC)

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def overlay_transparent(background, overlay, x, y):
    if overlay.shape[2] < 4:
        return background
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y + overlay.shape[0], x:x + overlay.shape[1], c] = (
            alpha * overlay[:, :, c] +
            (1 - alpha) * background[y:y + overlay.shape[0], x:x + overlay.shape[1], c]
        )
    return background

def parse_trace(trace_file_path):
    pattern = re.compile(
        r'^\s*\d+\)\s+([\d.]+)\s+Rx\s+0*([0-9A-Fa-f]+)\s+8\s+((?:[0-9A-Fa-f]{2}\s+){8})'
    )
    data_by_time = {}
    with open(trace_file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            match = re.match(pattern, line)
            if not match:
                continue
            timestamp_sec = float(match.group(1))
            can_id = match.group(2).upper()
            bytes_str = match.group(3).strip().split()
            msg = [int(b, 16) for b in bytes_str]
            timestamp_ms = int(round(timestamp_sec))
            frame_time = round(timestamp_ms / FRAME_INTERVAL_MS) * FRAME_INTERVAL_MS
            if frame_time not in data_by_time:
                data_by_time[frame_time] = {}
            data_by_time[frame_time].setdefault(can_id, []).append(msg)
    return data_by_time

def draw_brake_rings(frame, brake_level):
    center_x, center_y = WIDTH - 200, 160
    radii = [20, 35, 50]
    thickness = 5
    color = (0, 0, 255)
    if brake_level == 0:
        return frame
    for i in range(brake_level):
        cv2.circle(frame, (center_x, center_y), radii[i], color, thickness, cv2.LINE_AA)
    return frame

def draw_map():
    map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255
    if len(path_points) < 2:
        return map_img

    pts = np.array(path_points)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    bbox_size = max_xy - min_xy
    padding = 40 * draw_scale
    scale_x = (map_size - 2 * padding) / bbox_size[0] if bbox_size[0] > 0 else 1
    scale_y = (map_size - 2 * padding) / bbox_size[1] if bbox_size[1] > 0 else 1
    scale = min(scale_x, scale_y)
    offset = min_xy - padding / scale

    def transform(pt):
        return ((pt - offset) * scale).astype(int)

    for i in range(1, len(path_points)):
        pt1 = transform(path_points[i - 1])
        pt2 = transform(path_points[i])
        cv2.line(map_img, tuple(pt1), tuple(pt2), (255, 0, 255), 6, cv2.LINE_AA)
    cv2.circle(map_img, tuple(transform(path_points[-1])), 8, (0, 0, 0), -1)
    return map_img

def draw_dashboard(frame, speed, direction, voltage, current, soc, odo, distance_km, brake_level, angle):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)
    line_height = 80
    x0, y0 = 40, 80
    spacing = 60

    cv2.putText(frame, f"Speed: {speed} km/h", (x0, y0), font, 1.6, color, 4)
    y0 += line_height + spacing

    cv2.rectangle(frame, (x0, y0 - 60), (x0 + 100, y0), (0, 0, 0), 3)
    cv2.putText(frame, direction, (x0 + 20, y0 - 10), font, 1.6, color, 4)
    y0 += line_height + spacing

    cv2.circle(frame, (x0 + 25, y0 - 20), 25, (0, 0, 0), 3)
    cv2.putText(frame, "V", (x0 + 12, y0 - 10), font, 1.2, color, 3)
    cv2.putText(frame, f"{voltage:.1f} V", (x0 + 60, y0), font, 1.2, color, 3)
    y0 += line_height + spacing

    cv2.circle(frame, (x0 + 25, y0 - 20), 25, (0, 0, 0), 3)
    cv2.putText(frame, "A", (x0 + 12, y0 - 10), font, 1.2, color, 3)
    cv2.putText(frame, f"{current:.1f} A", (x0 + 60, y0), font, 1.2, color, 3)
    y0 += line_height + spacing

    # SoC as battery and %
    battery_x, battery_y = x0, y0
    battery_w, battery_h = 160, 50
    level = int(battery_w * (soc / 100))
    cv2.rectangle(frame, (battery_x, battery_y), (battery_x + battery_w, battery_y + battery_h), (0, 0, 0), 3)
    cv2.rectangle(frame, (battery_x, battery_y), (battery_x + level, battery_y + battery_h), (0, 200, 0), -1)
    cv2.putText(frame, f"{soc}%", (battery_x + battery_w + 10, battery_y + 40), font, 1.2, color, 3)
    y0 += line_height + spacing

    cv2.putText(frame, f"Odometer: {odo} km", (x0, y0), font, 1.2, color, 3)
    y0 += line_height + spacing

    cv2.putText(frame, f"Distance: {distance_km:.2f} km", (x0, y0), font, 1.2, color, 3)
    y0 += line_height + spacing

    cv2.putText(frame, f"Angle: {angle} deg", (x0, y0), font, 1.2, color, 3)
    return frame
def generate_video(trace_file_path, output_path="steering_dashboard.mp4"):
    global heading, position, path_points
    data = parse_trace(trace_file_path)
    all_times = sorted(data.keys())
    if not all_times:
        print("No valid data.")
        return

    start_time, end_time = min(all_times), max(all_times)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))

    last_values = {
        "speed": 0, "angle": 0, "direction": "N", "voltage": 0,
        "current": 0, "soc": 0, "odo": 0, "brake": 0
    }
    current_angle = 0
    distance_travelled = 0

    time_range = list(range(start_time, end_time + 1, FRAME_INTERVAL_MS))
    total_frames = len(time_range)

    for i, t in enumerate(time_range):
        if i % 10 == 0 or i == total_frames - 1:
            progress = int((i + 1) / total_frames * 50)
            bar = '#' * progress + '-' * (50 - progress)
            #print(f"\rRendering: [{bar}] {i + 1}/{total_frames} frames", end='', flush=True)

        msgs = data.get(t, {})
        for can_id, frames in msgs.items():
            for msg in frames:
                if can_id == "18D":
                    last_values["speed"] = msg[4]
                    gear_bits = (msg[0] >> 1) & 0b11
                    last_values["direction"] = {0: "N", 1: "F", 2: "R", 3: "E"}.get(gear_bits, "?")
                elif can_id == "28D":
                    last_values["voltage"] = msg[2] * 0.5
                    raw = (msg[1] << 8) | msg[0]
                    last_values["current"] = raw * 0.1 - 1000
                    last_values["soc"] = msg[5]
                elif can_id == "38D":
                    odo = (msg[6] << 16) | (msg[5] << 8) | msg[4]
                    last_values["odo"] = odo
                elif can_id == "60A" and msg[7] in (0xF0, 0xF1):
                    raw = (msg[0] << 8) | msg[1]
                    last_values["angle"] = round(raw * 0.1 - 1080)
                elif can_id == "629":
                    last_values["brake"] = msg[0] & 0x03

        angle_diff = last_values["angle"] - current_angle
        current_angle += 9 if angle_diff > 9 else -9 if angle_diff < -9 else angle_diff

        real_car_angle = current_angle / 54.0
        heading += real_car_angle * 0.12
        frame_dist = last_values["speed"] * 0.05 / 3.6
        dx = np.cos(np.radians(heading)) * frame_dist * draw_scale
        dy = np.sin(np.radians(heading)) * frame_dist * draw_scale

        for _ in range(4):
            position += np.array([dx, dy]) / 4
            path_points.append(position.copy())

        distance_travelled += frame_dist / 1000

        frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

        map_img = draw_map()
        map_small = cv2.resize(map_img, (map_base_size, map_base_size), interpolation=cv2.INTER_AREA)
        map_x = (WIDTH - map_base_size) // 2
        map_y = (HEIGHT - map_base_size) // 2
        frame[map_y:map_y + map_base_size, map_x:map_x + map_base_size] = map_small

        rotated = rotate_image(wheel_img.copy(), -current_angle)
        wheel_size = 350
        wheel_img_resized = cv2.resize(rotated, (wheel_size, wheel_size), interpolation=cv2.INTER_CUBIC)
        frame = overlay_transparent(frame, wheel_img_resized, WIDTH - wheel_size - 40, HEIGHT - wheel_size - 40)

        frame = draw_dashboard(frame,
                               speed=last_values["speed"],
                               direction=last_values["direction"],
                               voltage=last_values["voltage"],
                               current=last_values["current"],
                               soc=last_values["soc"],
                               odo=last_values["odo"],
                               distance_km=distance_travelled,
                               brake_level=last_values["brake"],
                               angle=current_angle)
        frame = draw_brake_rings(frame, last_values["brake"])
        out.write(frame)

    #print("\n✅ Video rendering complete.")
    out.release()
    #print(f"✅ Saved to: {output_path}")

def simulate(trace_file_path, output_path):
    generate_video(trace_file_path, output_path)