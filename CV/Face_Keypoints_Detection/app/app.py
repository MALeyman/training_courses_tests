import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading
import cv2
import torch

class DetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Детекция ключевых точек")
        self.geometry("1500x700")

        self.model_name = tk.StringVar(value="resnet50_2020-07-20")
        self.device_name = tk.StringVar(value="cpu")
        self.media_path = None
        self.cap = None
        self.stop_video = False
        self.model = None

        # интерфейс
        self.create_widgets()


    def create_widgets(self):
        # Левая панель управления
        control_frame = ttk.Frame(self, width=250)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(control_frame, text="Выбор модели:").pack(anchor=tk.W, pady=(10,0))
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_name,
                                   values=["resnet50_2020-07-20", ""], state="readonly")
        model_combo.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Устройство:").pack(anchor=tk.W, pady=(10,0))
        device_combo = ttk.Combobox(control_frame, textvariable=self.device_name,
                                    values=["cpu", "cuda"], state="readonly")
        device_combo.pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="Загрузить модель", command=self.load_model).pack(fill=tk.X, pady=10)

        ttk.Label(control_frame, text="Выберите файл (изображение или видео):").pack(anchor=tk.W, pady=(20,0))
        ttk.Button(control_frame, text="Выбрать файл", command=self.select_file).pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="Запустить детекцию", command=self.start_detection).pack(fill=tk.X, pady=20)

        # Правая панель для вывода изображения/видео
        self.display_frame = ttk.Frame(self)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.display_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)


    def load_model(self):
        import retinaface.pre_trained_models as rmodels
        device = torch.device(self.device_name.get() if torch.cuda.is_available() else "cpu")
        self.model = rmodels.get_model(self.model_name.get(), max_size=1024, device=device)
        self.model.eval()
        self.show_message(f"Модель {self.model_name.get()} загружена на {device}")


    def select_file(self):
        filetypes = [("Видео и изображения", "*.mp4 *.avi *.mov *.jpg *.jpeg *.png"), ("Все файлы", "*.*")]
        path = filedialog.askopenfilename(title="Выберите файл", filetypes=filetypes)
        if path:
            self.media_path = path
            self.show_message(f"Выбран файл: {path}")


    def start_detection(self):
        if self.model is None:
            self.show_message("Сначала загрузите модель!")
            return
        if not self.media_path:
            self.show_message("Сначала выберите файл!")
            return

        # Если видео — запускаем поток с видео
        if self.media_path.lower().endswith((".mp4", ".avi", ".mov")):
            self.stop_video = False
            threading.Thread(target=self.process_video, daemon=True).start()
        else:
            # Изображение
            self.process_image()


    def process_image(self):
        img = cv2.imread(self.media_path)
        if img is None:
            self.show_message("Не удалось загрузить изображение!")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotation = self.model.predict_jsons(img_rgb)
        from retinaface.utils import vis_annotations
        img_vis = vis_annotations(img_rgb, annotation)
        self.show_image(img_vis)


    def process_video(self):
        self.cap = cv2.VideoCapture(self.media_path)
        if not self.cap.isOpened():
            self.show_message("Не удалось открыть видео!")
            return

        while not self.stop_video:
            ret, frame = self.cap.read()
            if not ret:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotation = self.model.predict_jsons(img_rgb)
            from retinaface.utils import vis_annotations
            img_vis = vis_annotations(img_rgb, annotation)
            self.show_image(img_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()


    def show_image(self, img_rgb):
        img_pil = Image.fromarray(img_rgb)
        # Подгоняем размер под canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_pil = img_pil.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)


    def show_message(self, msg):
        print(msg)  


    def on_closing(self):
        self.stop_video = True
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = DetectionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()








