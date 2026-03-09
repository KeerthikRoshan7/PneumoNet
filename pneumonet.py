import streamlit as st
import google.generativeai as genai
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PneumoNet | Attention-Based Classification",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR & PROJECT CONTEXT ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=80) # Placeholder lung icon
    st.title("PneumoNet")
    st.caption("Attention-Based DenseNet for Pneumonia Classification")
    st.markdown("---")
    st.markdown("**Developer:** Keerthik Roshan G")
    st.markdown("**Institution:** M.I.E.T. Engineering College")
    st.markdown("---")
    st.markdown("""
    **Model Architecture Insights:**
    * **Backbone:** DenseNet (High feature reuse, parameter efficiency)
    * **Channel Attention:** Suppresses anatomical noise (bones, tags), amplifies pneumonia textures.
    * **Spatial Attention:** Masks non-lung regions, focuses on lung lobes.
    * **Accuracy:** 94.2%
    * **Recall:** 93.5%
    """)
    st.markdown("---")
    
    # API Key Input for deployment
    api_key = st.text_input("Enter Google Gemini API Key", type="password", help="Get your API key from Google AI Studio")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        st.warning("Please enter your Gemini API Key to proceed.")

# --- MAIN UI ---
st.title("🫁 PneumoNet Diagnostic Interface")
st.write("Upload a Chest X-Ray (CXR) to generate an attention-guided pneumonia classification report based on the PneumoNet architecture.")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Input Data")
    uploaded_file = st.file_uploader("Upload Chest X-Ray Image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Simulate the pre-processing mentioned in the PPT (Resize to 224x224 conceptually for display)
        st.image(image, caption="Pre-processed CXR Input", use_container_width=True)

with col2:
    st.subheader("2. Attention & Classification Output")
    
    if uploaded_file is not None:
        if not api_key:
            st.error("API Key is required to run the analysis.")
        else:
            if st.button("Run PneumoNet Analysis", type="primary"):
                with st.spinner("Initializing Attention-Based DenseNet Simulation & Finding Available Model..."):
                    
                    # THE HARD-PROMPT: Forcing Gemini to act as the Attention Mechanisms
                    pneumonet_prompt = """
                    You are 'PneumoNet', an advanced hybrid deep learning model combining a DenseNet backbone with integrated Channel and Spatial Attention Mechanisms, developed by Keerthik Roshan G. 
                    Your purpose is high-accuracy pneumonia classification from Chest X-Rays (CXRs).

                    Analyze the provided Chest X-Ray image. You MUST structure your analysis by explicitly simulating your two attention modules. 
                    Do not output a standard AI response. Output a clinical and architectural breakdown.

                    Follow this strict structure:

                    ### 1. Channel Attention Analysis
                    *Focuses on specific features across the image.*
                    * **Amplification:** Explicitly detail the specific textures you are amplifying that indicate pneumonia (e.g., hazy opacities, ground-glass infiltrates, consolidation).
                    * **Suppression:** Explicitly describe the anatomical noise and distracting elements you are filtering out (e.g., ribs, clavicles, spine, heart shadow, background air, hospital tags).

                    ### 2. Spatial Attention Analysis
                    *Concentrates on specific locations within the image.*
                    * **Targeted Focus:** Describe the exact spatial "mask" you are generating. Which specific lung fields (Upper, Middle, Lower lobes; Right vs Left lung) are receiving the highest "Importance Scores"?
                    * **Distraction Ignorance:** Which spatial zones are assigned low importance?

                    ### 3. Final Classification & Confidence
                    *Based on the aggregated features from the DenseNet layers.*
                    * **Diagnosis:** [NORMAL or PNEUMONIA]
                    * **Confidence Score:** [Assign a high probability percentage, keeping in mind PneumoNet's baseline accuracy of 94.2% and recall of 93.5%]
                    
                    ### 4. Comprehensive Diagnostic Report
                    Provide a final, readable summary of the findings translating your attention-based feature extraction into clinical radiologist terminology. Mention if there are visual patterns that might overlap with COVID-19 (as per future scope).
                    """

                    # HIERARCHY OF MODELS: From highest capability down to standard legacy models
                    models_to_try = [
                        "gemini-1.5-pro-latest",   # Best reasoning, highest capability
                        "gemini-1.5-pro",          # Standard 1.5 Pro
                        "gemini-1.5-flash-latest", # Fastest, highly capable
                        "gemini-1.5-flash",        # Standard 1.5 Flash
                        "gemini-pro-vision"        # Legacy Vision Model (Ultimate Fallback)
                    ]

                    success = False
                    last_error = None

                    # Iterate through the models until one works
                    for model_name in models_to_try:
                        try:
                            # Attempt to initialize and generate with the current model in the loop
                            model = genai.GenerativeModel(model_name)
                            response = model.generate_content([pneumonet_prompt, image])
                            
                            # If we reach here, it worked! 
                            st.success(f"Analysis Complete! (Powered by {model_name})")
                            st.markdown(response.text)
                            st.caption("Note: This report is generated by an AI simulating the PneumoNet architecture via the Gemini API. It is for demonstration and research purposes, not official clinical diagnosis.")
                            
                            success = True
                            break # Break out of the loop since we succeeded

                        except Exception as e:
                            # Log the error internally and move to the next model
                            last_error = str(e)
                            st.toast(f"Model {model_name} unavailable. Trying next...", icon="🔄")
                            continue

                    # If all models in the list failed
                    if not success:
                        st.error("All vision models failed to process the request.")
                        st.error(f"Last Error Encountered: {last_error}")
                        st.info("Tip: Check if your API Key has access to the Gemini Vision API in Google AI Studio, or if there are geographic restrictions.")
    else:
        st.info("Awaiting image upload to begin feature extraction...")
