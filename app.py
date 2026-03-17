import cv2
import gradio as gr
from people_counter import PeopleCounter

counter = PeopleCounter()


def process_webcam(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    processed = counter.process_frame(frame)

    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    return processed


def process_video(video):

    # Gradio sometimes sends a dict instead of a string path
    if isinstance(video, dict):
        video = video["path"]

    cap = cv2.VideoCapture(video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Fix if FPS is zero
    if fps == 0 or fps is None:
        fps = 25

    out = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"avc1"),  # better for browser playback
        fps,
        (width, height)
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        processed = counter.process_frame(frame)

        out.write(processed)

    cap.release()
    out.release()

    return "output.mp4"

    if isinstance(video, dict):
        video = video["path"]

    cap = cv2.VideoCapture(video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width,height)
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        processed = counter.process_frame(frame)

        out.write(processed)

    cap.release()
    out.release()

    return "output.mp4"


with gr.Blocks() as demo:

    gr.Markdown("# AI People Counter")

    with gr.Tab("Live Webcam"):

        webcam = gr.Image(type="numpy", streaming=True)
        webcam_output = gr.Image()

        webcam.stream(
            process_webcam,
            inputs=webcam,
            outputs=webcam_output
        )

    with gr.Tab("Upload Video"):

        video_input = gr.Video(label="Upload Video")
        video_output = gr.Video()

        btn = gr.Button("Run Detection")

        btn.click(
            process_video,
            inputs=video_input,
            outputs=video_output
        )

demo.launch()