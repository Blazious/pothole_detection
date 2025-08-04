# app.py
import gradio as gr
from detect import detect_image, detect_video

def handle_image(img):
    result_path, beep_path = detect_image(img)
    return result_path, beep_path

def handle_video(video):
    result_path, beep_path = detect_video(video)
    return result_path, beep_path

with gr.Blocks() as demo:
    gr.Markdown("## 🕳️ Pothole Detector")

    with gr.Tabs():
        with gr.TabItem("Image Detection"):
            with gr.Row():
                img_input = gr.Image(type="pil", label="Upload Image")
                img_output = gr.Image(label="Detection Result")
            audio_output_img = gr.Audio(label="Detection Beep", type="filepath", visible=True)
            img_input.change(fn=handle_image, inputs=[img_input], outputs=[img_output, audio_output_img])

        with gr.TabItem("Video Detection"):
            with gr.Row():
                video_input = gr.Video(label="Upload Video")
                video_output = gr.Video(label="Detection Result")
            audio_output_vid = gr.Audio(label="Detection Beep", type="filepath", visible=True)
            video_input.change(fn=handle_video, inputs=[video_input], outputs=[video_output, audio_output_vid])

        gr.Markdown("""
                    ### 🔍 About This Tool

                    - Designed for demonstration and educational purposes  
                    - AI-generated pothole detection may not be 100% accurate  
                    - Always verify road conditions with physical inspection  

                    ---

                    ⏱️ **Processing Note:**  
                    Video analysis duration varies depending on the video's length and resolution.  
                    Typical processing can take **5–6 minutes** as the system carefully analyzes individual frames.
                    """)

demo.launch()


