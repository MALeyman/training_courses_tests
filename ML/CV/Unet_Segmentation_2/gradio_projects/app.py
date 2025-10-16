
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import gradio as gr


if __name__ == "__main__":
    # =============   Если запускать с "app.py"
    from projects.segmentation_1 import get_segmentation_tab
    from projects.detection_1 import get_detection_tab_1
    from projects.home_tab import home_tab
    from projects.detection_2 import get_detection_tab_2
else:
    # ===============  Если запускать с ноутбука  "gradio.ipynb"
    from gradio_projects.projects.segmentation_1 import get_segmentation_tab
    from gradio_projects.projects.home_tab import home_tab
    from gradio_projects.projects.detection_1 import get_detection_tab_1
    from gradio_projects.projects.detection_2 import get_detection_tab_2

def main():
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Главная"):
                home_tab()
            with gr.TabItem("Сегментация дорожных сцен"):
                get_segmentation_tab()
            with gr.TabItem("Детекция лиц"):
                get_detection_tab_1()
            with gr.TabItem("Детекция с БПЛА"):
                get_detection_tab_2()


    demo.launch(debug=True)


if __name__ == "__main__":
    main()

