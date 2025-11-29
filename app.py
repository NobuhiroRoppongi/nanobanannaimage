# app.py
# pip install streamlit google-genai

import os
import base64
import mimetypes
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types


def save_binary_file(file_path: Path, data: bytes):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(data)
    return file_path


def generate_image(api_key: str, prompt: str, save_dir: str, file_base_name: str):
    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash-image"

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]

    generate_content_config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"]
    )

    result = client.models.generate_content(
        model=model, contents=contents, config=generate_content_config
    )

    saved_files = []
    text_parts = []

    if not result.candidates:
        return saved_files, ""

    content = result.candidates[0].content
    if not content or not content.parts:
        return saved_files, ""

    save_dir_path = Path(save_dir)
    image_count = 0

    for part in content.parts:
        # 🔹 IMAGE output
        if getattr(part, "inline_data", None) and part.inline_data.data:
            data_buffer = part.inline_data.data

            if isinstance(data_buffer, str):
                data_buffer = base64.b64decode(data_buffer)

            extension = mimetypes.guess_extension(part.inline_data.mime_type) or ".png"

            # 🔥 Use user-defined file name
            file_name = f"{file_base_name}{'' if image_count == 0 else f'_{image_count}'}{extension}"
            image_count += 1

            save_path = save_binary_file(save_dir_path / file_name, data_buffer)
            saved_files.append(save_path)

        # 🔹 TEXT output
        if getattr(part, "text", None):
            text_parts.append(part.text)

    return saved_files, "\n".join(text_parts)


# ================= STREAMLIT UI ================= #

st.title("🖼️ Gemini Image Generator")

api_key = st.text_input("Gemini API Key", type="password")
prompt = st.text_area("Enter Prompt", height=160)

# Folder path
default_dir = str(Path.cwd() / "outputs")
save_dir = st.text_input("Output Folder Path", default_dir)

# 🔥 New: output file name input
file_base_name = st.text_input(
    "Output File Name (without extension)",
    value="generated_image"
)

if st.button("Generate"):
    if not api_key:
        st.error("API key missing.")
    elif not prompt.strip():
        st.error("Prompt missing.")
    elif not file_base_name.strip():
        st.error("File name missing.")
    else:
        with st.spinner("Generating image…"):
            try:
                files, txt = generate_image(api_key, prompt, save_dir, file_base_name)
            except Exception as e:
                st.error(f"Error: {e}")
            else:
                if txt:
                    st.subheader("Model Response Text")
                    st.write(txt)

                if files:
                    st.subheader("Generated Image Preview")
                    for f in files:
                        st.image(str(f), caption=f"Saved → {f.name}")
                        st.caption(str(f))
                else:
                    st.warning("No image returned from model.")
