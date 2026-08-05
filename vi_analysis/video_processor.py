import cv2
import numpy as np
import os
import json
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from sklearn.cluster import KMeans

class FootballVideoAnalyzer:
    def __init__(self, video_path, output_dir='output', sample_rate=10,
                 home_color=None, away_color=None,
                 left_goal_team='home', right_goal_team='away'):
        """
        初始化足球视频分析器
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            sample_rate: 采样率（Hz）
            home_color: 主队球衣主颜色 (BGR格式) 例如：[0,0,255]
            away_color: 客队球衣主颜色 (BGR格式) 例如：[255,0,0]
            left_goal_team: 左球门球队标签
            right_goal_team: 右球门球队标签
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.sample_interval = 1.0 / sample_rate

        self.home_color = np.array(home_color) if home_color else None
        self.away_color = np.array(away_color) if away_color else None
        self.left_goal_team = left_goal_team
        self.right_goal_team = right_goal_team

        self.field_coords = {'min_x': 0, 'max_x': 105, 'min_y': 0, 'max_y': 68}
        self.homography_matrix = None
        self.field_corners_pixel = None

        self.frames_dir = os.path.join(output_dir, 'frames')
        self.data_dir = os.path.join(output_dir, 'data')
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        self.load_yolo_model()
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0,
                                max_cosine_distance=0.3, nn_budget=None,
                                embedder="mobilenet", half=True, bgr=True, embedder_gpu=True)
        self.left_goal_id = 23
        self.right_goal_id = 24

    def load_yolo_model(self):
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.classes = [0]
            print("YOLO模型加载成功")
        except:
            print("警告：无法加载YOLO模型")
            self.model = None

    def detect_field_lines(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

        if lines is None:
            h, w = frame.shape[:2]
            return np.array([[w*0.1, h*0.1], [w*0.9, h*0.1], [w*0.9, h*0.9], [w*0.1, h*0.9]], dtype=np.float32)

        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10 or angle > 170:
                h_lines.append((y1 + y2) / 2)
            elif 80 < angle < 100:
                v_lines.append((x1 + x2) / 2)

        if len(h_lines) < 2 or len(v_lines) < 2:
            h, w = frame.shape[:2]
            return np.array([[w*0.1, h*0.1], [w*0.9, h*0.1], [w*0.9, h*0.9], [w*0.1, h*0.9]], dtype=np.float32)

        top_y, bottom_y = np.min(h_lines), np.max(h_lines)
        left_x, right_x = np.min(v_lines), np.max(v_lines)
        return np.array([[left_x, top_y], [right_x, top_y], [right_x, bottom_y], [left_x, bottom_y]], dtype=np.float32)

    def calculate_homography(self, frame):
        self.field_corners_pixel = self.detect_field_lines(frame)
        field_corners = np.array([[0,0], [105,0], [105,68], [0,68]], dtype=np.float32)
        self.homography_matrix, _ = cv2.findHomography(self.field_corners_pixel, field_corners)
        self.draw_field_boundary(frame)
        return self.homography_matrix

    def pixel_to_field_coords(self, pixel_x, pixel_y):
        if self.homography_matrix is None:
            return None, None
        pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        field_point = cv2.perspectiveTransform(pixel_point, self.homography_matrix)
        return field_point[0][0][0], field_point[0][0][1]

    def is_within_field(self, bbox):
        if self.field_corners_pixel is None:
            return True
        x1, y1, x2, y2 = bbox
        center = ((x1+x2)/2, (y1+y2)/2)
        return cv2.pointPolygonTest(self.field_corners_pixel, center, False) >= 0

    def extract_dominant_color(self, roi):
        if roi.size == 0:
            return None
        roi = cv2.resize(roi, (64, 128))
        h, w = roi.shape[:2]
        upper = roi[int(h*0.1):int(h*0.5), int(w*0.2):int(w*0.8)]
        if upper.size == 0:
            return None
        hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(pixels)
        dominant = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
        return cv2.cvtColor(np.uint8([[dominant]]), cv2.COLOR_HSV2BGR)[0][0]

    def color_distance(self, c1, c2):
        if c1 is None or c2 is None:
            return float('inf')
        c1_hsv = cv2.cvtColor(np.uint8([[c1]]), cv2.COLOR_BGR2HSV)[0][0]
        c2_hsv = cv2.cvtColor(np.uint8([[c2]]), cv2.COLOR_BGR2HSV)[0][0]
        h_diff = min(abs(c1_hsv[0]-c2_hsv[0]), 180-abs(c1_hsv[0]-c2_hsv[0])) * 2
        s_diff = abs(c1_hsv[1]-c2_hsv[1])
        v_diff = abs(c1_hsv[2]-c2_hsv[2])
        return np.sqrt(h_diff**2 + s_diff**2 + v_diff**2)

    def assign_team(self, player_color):
        if player_color is None or self.home_color is None or self.away_color is None:
            return "unknown"
        d_home = self.color_distance(player_color, self.home_color)
        d_away = self.color_distance(player_color, self.away_color)
        threshold = 50
        if d_home < threshold and d_home < d_away:
            return "home"
        elif d_away < threshold and d_away < d_home:
            return "away"
        else:
            return "referee"

    def detect_objects(self, frame):
        if self.model is None:
            return []
        results = self.model(frame)
        dets = results.xyxy[0].cpu().numpy()
        bbs, rois = [], []
        for det in dets:
            x1,y1,x2,y2,conf,cls = det
            if cls == 0 and conf > 0.5:
                if self.is_within_field([x1,y1,x2,y2]):
                    bbs.append([x1,y1,x2,y2,conf])
                    rois.append(frame[int(y1):int(y2), int(x1):int(x2)])
                else:
                    rois.append(None)
        tracks = self.tracker.update_tracks(bbs, frame=frame)
        players = []
        for i, track in enumerate(tracks):
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()
            cx, cy = (ltrb[0]+ltrb[2])/2, (ltrb[1]+ltrb[3])/2
            if i < len(rois) and rois[i] is not None:
                color = self.extract_dominant_color(rois[i])
                team = self.assign_team(color)
            else:
                team = "unknown"
            fx, fy = self.pixel_to_field_coords(cx, cy)
            players.append({
                'pixel': {'x':float(cx), 'y':float(cy)},
                'bbox': {'x1':float(ltrb[0]), 'y1':float(ltrb[1]), 'x2':float(ltrb[2]), 'y2':float(ltrb[3])},
                'track_id': int(tid),
                'team': team,
                'field': {'x':float(fx) if fx else None, 'y':float(fy) if fy else None}
            })
        return players

    def add_goals(self, players_info, frame_w, frame_h):
        if self.homography_matrix is not None:
            invH = np.linalg.inv(self.homography_matrix)
            left_pix = cv2.perspectiveTransform(np.array([[[0,34]]], dtype=np.float32), invH)[0][0]
            right_pix = cv2.perspectiveTransform(np.array([[[105,34]]], dtype=np.float32), invH)[0][0]
        else:
            left_pix = [frame_w*0.1, frame_h*0.5]
            right_pix = [frame_w*0.9, frame_h*0.5]
        players_info.append({
            'pixel':{'x':float(left_pix[0]),'y':float(left_pix[1])},
            'bbox':None, 'track_id':self.left_goal_id, 'team':self.left_goal_team,
            'field':{'x':0.0,'y':34.0}
        })
        players_info.append({
            'pixel':{'x':float(right_pix[0]),'y':float(right_pix[1])},
            'bbox':None, 'track_id':self.right_goal_id, 'team':self.right_goal_team,
            'field':{'x':105.0,'y':34.0}
        })
        return players_info

    def draw_field_boundary(self, frame):
        if self.field_corners_pixel is not None:
            pts = self.field_corners_pixel.reshape((-1,1,2)).astype(np.int32)
            cv2.polylines(frame, [pts], True, (0,255,255), 3)

    def draw_detections(self, frame, players_info):
        self.draw_field_boundary(frame)
        for p in players_info:
            if p['bbox'] is None:
                color = (0,255,255)  # yellow for goal
                label = f"Goal {p['track_id']}"
            else:
                if p['team'] == 'home':
                    color = (0,0,255)
                elif p['team'] == 'away':
                    color = (255,0,0)
                else:
                    color = (255,255,255)
                bbox = p['bbox']
                cv2.rectangle(frame, (int(bbox['x1']),int(bbox['y1'])), (int(bbox['x2']),int(bbox['y2'])), color, 2)
                label = f"ID:{p['track_id']} {p['team']}"
            cx, cy = int(p['pixel']['x']), int(p['pixel']['y'])
            cv2.circle(frame, (cx,cy), 5, color, -1)
            cv2.putText(frame, label, (cx-20,cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("无法打开视频")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"视频信息: fps={fps}, 总帧数={total_frames}, 分辨率={w}x{h}")

        ret, first = cap.read()
        if ret:
            print("检测球场边界...")
            self.calculate_homography(first)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        sample_interval = int(fps / self.sample_rate)
        all_samples = []
        frame_count = 0
        sample_count = 0
        pbar = tqdm(total=total_frames//sample_interval, desc="处理视频")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % sample_interval == 0:
                # 保存原始帧
                frame_path = os.path.join(self.frames_dir, f'frame_{sample_count:06d}.jpg')
                cv2.imwrite(frame_path, frame)

                # 检测所有球员
                all_players = self.detect_objects(frame)
                # 过滤出参赛球员
                filtered = [p for p in all_players if p['team'] in ['home','away']]
                # 添加球门
                filtered = self.add_goals(filtered, w, h)

                # 绘制并保存标注图（使用all_players显示裁判）
                self.draw_detections(frame, all_players)
                ann_path = os.path.join(self.frames_dir, f'annotated_{sample_count:06d}.jpg')
                cv2.imwrite(ann_path, frame)

                sample_data = {
                    'timestamp': frame_count / fps,
                    'frame_number': frame_count,
                    'sample_number': sample_count,
                    'frame_file': frame_path,
                    'players': filtered,
                    'num_players': len(filtered),
                    'field_corners': self.field_corners_pixel.tolist() if self.field_corners_pixel is not None else None
                }
                all_samples.append(sample_data)
                sample_count += 1
                pbar.update(1)
            frame_count += 1

        pbar.close()
        cap.release()
        self.save_data(all_samples)
        print(f"处理完成，共 {sample_count} 个采样点")
        return all_samples

    def save_data(self, all_samples):
        json_path = os.path.join(self.data_dir, 'samples.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        print(f"数据已保存至 {json_path}")