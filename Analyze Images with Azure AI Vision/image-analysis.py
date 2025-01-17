# <----------------------------------------Module 1: Import Libraries and Setup--------------------------------------------->
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import sys
from matplotlib import pyplot as plt
from azure.core.exceptions import HttpResponseError
import requests

# Import namespaces for Azure AI Vision services
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# <----------------------------------------Main Function--------------------------------------------->
def main():
    try:
        # Load environment variables for Azure service credentials
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')
        
        # Check if credentials are loaded
        if not ai_endpoint or not ai_key:
            raise ValueError("AI_SERVICE_ENDPOINT or AI_SERVICE_KEY is not set in the environment.")

        # Set image file path (default or from command-line argument)
        image_file = 'C:\\Users\\pc\\OneDrive\\Documents\\Ai-102 Lab\\Lab02\\mslearn-ai-vision\\Labfiles\\01-analyze-images\\Python\\image-analysis\\images\\street.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Open and read image as binary data
        with open(image_file, "rb") as f:
            image_data = f.read()

        # Authenticate Azure AI Vision client
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )

        # Analyze the image and remove background
        AnalyzeImage(image_file, image_data, cv_client)
        BackgroundForeground(ai_endpoint, ai_key, image_file)

    except Exception as ex:
        print("Error:", ex)

# <----------------------------------------Image Analysis Function--------------------------------------------->
def AnalyzeImage(image_filename, image_data, cv_client):
    print('\nAnalyzing image...')

    try:
        # Perform image analysis with selected visual features
        poller = cv_client.begin_analyze_image(
            image=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE
            ]
        )
        result = poller.result()  # Retrieve analysis result

        # Display analysis results
        if result.caption:
            print(f"\nCaption: '{result.caption.text}' (confidence: {result.caption.confidence * 100:.2f}%)")

        if result.dense_captions:
            print("\nDense Captions:")
            for caption in result.dense_captions.list:
                print(f" Caption: '{caption.text}' (confidence: {caption.confidence * 100:.2f}%)")

        if result.tags:
            print("\nTags:")
            for tag in result.tags.list:
                print(f" Tag: '{tag.name}' (confidence: {tag.confidence * 100:.2f}%)")

        if result.objects:
            print("\nObjects in image:")
            display_annotated_objects(image_filename, result.objects)

    except HttpResponseError as e:
        print(f"Status code: {e.status_code}, Reason: {e.reason}, Message: {e.error.message}")

# <----------------------------------------Display Annotated Objects--------------------------------------------->
def display_annotated_objects(image_filename, objects):
    # Open and prepare image for drawing
    image = Image.open(image_filename)
    fig = plt.figure(figsize=(image.width / 100, image.height / 100))
    plt.axis('off')
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    # Draw bounding boxes around objects
    for detected_object in objects.list:
        print(f" {detected_object.tags[0].name} (confidence: {detected_object.tags[0].confidence * 100:.2f}%)")
        r = detected_object.bounding_box
        bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
        draw.rectangle(bounding_box, outline=color, width=3)
        plt.annotate(detected_object.tags[0].name, (r.x, r.y), backgroundcolor=color)

    # Save annotated image
    plt.imshow(image)
    plt.tight_layout(pad=0)
    outputfile = 'objects.jpg'
    fig.savefig(outputfile)
    print(' Results saved in', outputfile)

# <----------------------------------------Background Removal Function--------------------------------------------->
def BackgroundForeground(endpoint, key, image_file):
    # Set API endpoint and request parameters for background removal
    api_version = "2023-02-01-preview"
    mode = "backgroundRemoval"  # or "foregroundMatting"
    url = f"{endpoint}/computervision/imageanalysis:analyze?api-version={api_version}&mode={mode}"

    # Prepare headers and image URL for the API request
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"}
    image_url = f"https://github.com/MicrosoftLearning/mslearn-ai-vision/blob/main/Labfiles/01-analyze-images/Python/image-analysis/{os.path.basename(image_file)}?raw=true"
    body = {"url": image_url}

    # Send request for background removal
    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        image_data = response.content

        # Save the output image
        with open("background.png", "wb") as file:
            file.write(image_data)
        print(" Background removed and saved as 'background.png'.")

    except requests.exceptions.RequestException as e:
        print("Background removal failed:", e)

# <----------------------------------------Script Execution--------------------------------------------->
if __name__ == "__main__":
    main()
