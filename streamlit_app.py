import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
import time 
# Detect if GPU is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set wide layout for the app
st.set_page_config(page_title="AI Image Q&A", layout="wide")

# Custom CSS for full-width expansion
st.markdown("""
    <style>
        .main .block-container {
            max-width: 90%;
            padding: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Load BLIP model and processor
@st.cache_resource
def load_model():
    """Loads BLIP Processor and Model with GPU support if available."""
    with st.spinner("🚀 Loading AI Model... Please wait!"):
        time.sleep(2)  # Simulate loading delay
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(DEVICE)
        st.sidebar.success(f"✅ Model Loaded! Running on: **{DEVICE.upper()}**")
    return processor, model


processor, model = load_model()

def get_answer_blip(image, question):
    """Processes the image and question, then returns BLIP's answer."""
    inputs = processor(image, question, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Streamlit UI - Header
st.title("\U0001F5BC️ AI Image Question Answering with BLIP")
st.write(
    "Upload an image and ask any question about it. This app uses the **BLIP Vision-Language Model** to analyze images "
    "and generate text-based answers. \U0001F680"
)

# Sidebar - About the App
st.sidebar.header("ℹ️ About the App")
st.sidebar.write(
    "**What this app does:**\n"
    "- \U0001F4F8 Upload an image.\n"
    "- \U0001F914 Ask questions about the image.\n"
    "- \U0001F9E0 AI generates answers.\n"
)

# Sidebar - Technical Details
with st.sidebar:
    st.header("📌 App Technical Details")
    st.markdown(
    f"""
    - **Model:** [BLIP VQA Base](https://huggingface.co/Salesforce/blip-vqa-base)  
    - **Framework:** 🤖 Transformers (Hugging Face)  
    - **Backend:** PyTorch  
    - **Frontend:** Streamlit  
    - **GPU Support:** {'✅ Yes' if DEVICE == 'cuda' else '❌ No (Running on CPU)'}  
    - **Image Processing:** PIL (Pillow)  
    - **Caching:** `@st.cache_resource`
    """
    )
    
# Sidebar - Author Information
st.sidebar.markdown("---")
st.sidebar.write("👤 **Author**: Moses Sabila")
st.sidebar.write("[🔗 GitHub Repository](https://github.com/TheODDYSEY/Visual-Answering-Transformers-Model.git)")

# Layout with two columns
col1, col2 = st.columns([1, 1])  # Equal width columns

with col1:
    # Image upload section
    uploaded_file = st.file_uploader("📂 Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        # Image Metadata
        st.write("### 📝 Image Details:")
        st.write(f"📌 **Format**: {image.format}")
        st.write(f"📌 **Size**: {image.size[0]} x {image.size[1]} pixels")
        st.write(f"📌 **Color Mode**: {image.mode}")

with col2:
    if uploaded_file:
        # Predefined universal questions
        universal_questions = [
            "What is happening in the image?",
            "Describe the image in detail.",
            "How many people or objects are in the image?",
            "What colors are present in the image?",
            "What is the background of the image?",
            "Is there any text in the image? If yes, what does it say?",
            "What emotions are visible in the image?",
            "Is this image taken indoors or outdoors?",
            "What time of day does this image appear to be?",
            "Is there any movement happening in the image?"
        ]

        # Allow multiple questions selection
        selected_questions = st.multiselect("📝 Select questions:", universal_questions)
        custom_question = st.text_input("✍️ Or enter a custom question:")

        # If custom question is provided, add it to the list
        if custom_question:
            selected_questions.append(custom_question)

        # Button to get answers
        if st.button("🤖 Get Answers"):
            if selected_questions:
                st.subheader("🔍 AI Answers:")
                for question in selected_questions:
                    with st.spinner(f"🤖 Thinking about: '{question}'..."):
                        time.sleep(2)  # Simulate AI processing time
                        answer = get_answer_blip(image, question)
                        st.success(f"**Q:** {question}")
                        st.write(f"**A:** {answer}\n")
            else:
                st.warning("⚠️ Please select or enter at least one question.")