import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Edge AI Recyclable Items Classifier",
    page_icon="♻️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .header {
        text-align: center;
        color: #2c3e50;
    }
    .upload-area {
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    .result-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="header"><h1>♻️ Edge AI Recyclable Items Classifier</h1></div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Model Information
with st.sidebar:
    st.header("🔧 Model Information")
    st.markdown("""
    **Model Type**: TensorFlow Lite CNN
    **Task**: Image Classification
    **Categories**: Recyclable Items
    
    **Deployment Options**:
    - 🟢 Streamlit Web App
    - 🟡 Raspberry Pi
    - 🔵 Android
    - 🟣 Edge Devices
    
    **Benefits**:
    - ✅ Real-time inference
    - ✅ No internet required
    - ✅ Privacy-focused
    - ✅ Low latency
    """)

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📷 Upload Image")
    
    # File upload area
    uploaded_file = st.file_uploader(
        "Choose an image of a recyclable item",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload an image for classification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name)
            temp_image_path = tmp_file.name
    else:
        st.info("Please upload an image to classify.")
        temp_image_path = None

with col2:
    st.header("🤖 Classification Results")
    
    if temp_image_path:
        try:
            # Load the TensorFlow Lite model
            try:
                interpreter = tf.lite.Interpreter(model_path="recyclable_items_model.tflite")
                interpreter.allocate_tensors()
                
                # Get input and output details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Load and preprocess image
                image = cv2.imread(temp_image_path)
                image = cv2.resize(image, (224, 224))  # Adjust size based on your model
                image = image.astype(np.float32) / 255.0  # Normalize
                image = np.expand_dims(image, axis=0)  # Add batch dimension
                
                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], image)
                
                # Run inference
                interpreter.invoke()
                
                # Get output tensor
                predictions = interpreter.get_tensor(output_details[0]['index'])
                
                # Process predictions (adjust based on your model output)
                if len(predictions.shape) > 1:
                    predictions = predictions[0]  # Take first batch item
                
                # Create probability dictionary
                class_names = ['Plastic Bottle', 'Aluminum Can', 'Paper', 'Glass']
                prediction_dict = {}
                for i, class_name in enumerate(class_names):
                    prediction_dict[class_name] = float(predictions[i])
                
                # Normalize probabilities
                total = sum(prediction_dict.values())
                if total > 0:
                    prediction_dict = {k: v/total for k, v in prediction_dict.items()}
                
            except Exception as e:
                st.error(f"Error loading model or running inference: {str(e)}")
                st.info("Make sure you have a trained TFLite model in the correct location.")
                # Use mock predictions as fallback
                predictions = {
                    'Plastic Bottle': 0.85,
                    'Aluminum Can': 0.10,
                    'Paper': 0.03,
                    'Glass': 0.02
                }
            
            # Show loading spinner
            with st.spinner("🔄 Loading model and processing image..."):
                # Small delay to show the spinner
                import time
                time.sleep(1)
            
            # Display results
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("🎯 Classification Results")
            
            # Display prediction probabilities
            for item, confidence in predictions.items():
                st.write(f"**{item}**: {confidence:.1%}")
                st.progress(confidence)
            
            # Display the top prediction
            top_prediction = max(predictions, key=predictions.get)
            st.success(f"🏆 **Most Likely**: {top_prediction} ({predictions[top_prediction]:.1%})")
            
            st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error during classification: {str(e)}")
            st.info("Make sure you have a trained TFLite model in the correct location.")
    else:
        st.info("Upload an image to see classification results.")

# Footer information
st.markdown("---")
st.markdown("### 📋 Instructions")
st.markdown("""
1. **Upload Image**: Click on the upload area to select an image of a recyclable item
2. **Processing**: The model will process your image and display classification results
3. **Results**: View the classification probabilities and the most likely item category

### 🔧 Technical Details
- **Model**: TensorFlow Lite CNN optimized for edge deployment
- **Input**: Images of recyclable items (plastic, aluminum, paper, glass)
- **Output**: Classification probabilities for each category
- **Deployment**: Can be deployed on Raspberry Pi, Android, or other edge devices

### 📁 Required Files
- `recyclable_items_model.tflite`: Trained TensorFlow Lite model
- `class_names.txt`: Class name mappings
- Model weights and configuration files

### ⚠️ Note
This is a template application. You'll need to:
1. Train your own model on recyclable items dataset
2. Convert the model to TensorFlow Lite format
3. Update the model loading and inference code
4. Add proper image preprocessing
""")

# Cleanup
if temp_image_path and os.path.exists(temp_image_path):
    os.unlink(temp_image_path)